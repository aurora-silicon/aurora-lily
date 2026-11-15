//! Quality checks of Qwen3.8-Flash-Next (either checkpoint format) against
//! reference implementations on exact token ids: per-token NLL of a
//! continuation by teacher forcing, greedy continuations, tokenization, and
//! the scoring of ds4's official continuation fixtures with ds4's own prompt
//! rendering (`gguf-tools/quality-testing`, the no-thinking chat prompt).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use clap::{Parser, Subcommand};
use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::qwen4exp::Qwen4ExpModel;
use lily::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(name = "lily-qwen38", about = "Qwen3.8-Flash-Next quality checks")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Per-token NLL of each case's continuation: for every `<case>.ids`
    /// (whitespace-separated token ids) in the directory with a
    /// `<case>.split` (the prompt's length), writes `<case>.lily` lines
    /// `i target nll argmax margin` and a `mean` line, as q38ref does.
    Nll {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        cases: PathBuf,
        /// Only cases whose name starts with this.
        #[arg(long, default_value = "")]
        only: String,
    },
    /// Greedy continuation of a token id file: N ids on one line.
    Greedy {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        ids: PathBuf,
        #[arg(long, default_value_t = 32)]
        n: usize,
    },
    /// The tokenizer's ids of each file's text (no special tokens), one
    /// line per file.
    Tok {
        #[arg(long)]
        model: PathBuf,
        files: Vec<PathBuf>,
    },
    /// ds4's official continuation fixtures (manifest.tsv, prompts/,
    /// continuations/): mean NLL and greedy agreement per case and overall.
    Score {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
        /// Raw text: the prompt with no chat template around it.
        #[arg(long)]
        raw: bool,
    },
    /// The prompt's last logits by prefill against the same prompt fed one
    /// token at a time after a one-token prefill (the GEMM against the GEMV
    /// path): max difference and both argmaxes.
    Selfcheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
    },
    /// The chat prompt ids of a conversation (a JSON file: `messages`,
    /// optional `tools`), with thinking off and on, one line each.
    ChatIds {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        conversation: PathBuf,
    },
    /// One user message through the chat template, decoded greedily.
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long)]
        think: bool,
    },
}

fn read_ids(path: &Path) -> Result<Vec<u32>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .split_whitespace()
        .map(|s| s.parse::<u32>().context("token id"))
        .collect()
}

fn log_softmax_at(row: &[f32], i: usize) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max + row.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln();
    row[i] as f64 - lse
}

fn top2(row: &[f32]) -> (usize, f32) {
    let (mut a, mut b) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > row[a] {
            a = i;
        }
    }
    for (i, &v) in row.iter().enumerate() {
        if i != a && v > b {
            b = v;
        }
    }
    (a, row[a] - b)
}

struct Runner {
    ctx: MetalContext,
    model: Qwen4ExpModel,
}

impl Runner {
    fn load(path: &Path) -> Result<Self> {
        let ctx = MetalContext::new()?;
        let t = Instant::now();
        let model = <Qwen4ExpModel as LanguageModel>::load(
            &ctx,
            path,
            &LoadOptions { mtp_drafts: 0, ..LoadOptions::default() },
        )?;
        eprintln!("load {:.1} s", t.elapsed().as_secs_f64());
        Ok(Self { ctx, model })
    }

    /// Teacher-forces `ids` from `split` on: `f(i, logits)` sees the logits
    /// that predict `ids[i]` for every `i` in `split..ids.len()`, and
    /// greedily `extra` tokens past the end (`ids.len()..`).
    fn forced(
        &self,
        ids: &[u32],
        split: usize,
        extra: usize,
        mut f: impl FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<(f64, f64)> {
        ensure!(split >= 1 && split <= ids.len(), "split {split} of {} ids", ids.len());
        let vocab = LanguageModel::vocab_size(&self.model);
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= vocab) {
            anyhow::bail!("token id {bad} is outside the vocabulary of {vocab}");
        }
        let (m, ctx) = (&self.model, &self.ctx);
        let n = ids.len() + extra;
        let mut state = LanguageModel::new_state(m, ctx, n + 1)?;
        let mut scratch = LanguageModel::new_scratch_with_capacity(m, ctx, n + 1)?;
        let greedy = SamplingParams::greedy();
        let t = Instant::now();
        LanguageModel::prefill(
            m,
            ctx,
            &mut state,
            &mut scratch,
            &ids[..split],
            Some(Draw { params: &greedy, step: 0 }),
        )?;
        let prefill = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut slot = 0usize;
        for i in split..n {
            let logits = scratch.logits().to_f32()?;
            f(i, &logits)?;
            if i + 1 == n {
                break;
            }
            let input = if i < ids.len() {
                ids[i]
            } else {
                scratch.next_token().view(slot, &[1])?.to_u32()?[0]
            };
            let step = i - split + 1;
            let encoded = LanguageModel::encode_decode_step(
                m,
                ctx,
                &state,
                &scratch,
                slot,
                1 - slot,
                Draw { params: &greedy, step },
            )?;
            LanguageModel::prepare_step_inputs(m, &mut state, &scratch, input)?;
            scratch
                .next_token()
                .view(slot, &[1])?
                .write_bytes(bytemuck::bytes_of(&input))?;
            let pending = encoded.commit()?;
            state.advance(1);
            pending.wait()?;
            slot = 1 - slot;
        }
        Ok((prefill, t.elapsed().as_secs_f64()))
    }
}

/// ds4's no-thinking chat prompt (`encode_chat_prompt`, DS4_THINK_NONE):
/// each piece tokenized on its own.
fn ds4_chat_ids(tok: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let id = |s: &str| tok.token_id(s).with_context(|| format!("no token {s}"));
    let (start, end) = (id("<|im_start|>")?, id("<|im_end|>")?);
    let (think, unthink) = (id("<think>")?, id("</think>")?);
    let mut v = vec![start];
    v.extend(tok.encode("user")?);
    v.extend(tok.encode("\n")?);
    v.extend(tok.encode(prompt)?);
    v.push(end);
    v.extend(tok.encode("\n")?);
    v.push(start);
    v.extend(tok.encode("assistant")?);
    v.extend(tok.encode("\n")?);
    v.push(think);
    v.extend(tok.encode("\n\n")?);
    v.push(unthink);
    v.extend(tok.encode("\n\n")?);
    Ok(v)
}

fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Nll { model, cases, only } => {
            let mut names: Vec<String> = std::fs::read_dir(&cases)?
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    e.file_name()
                        .to_str()
                        .and_then(|n| n.strip_suffix(".ids"))
                        .map(String::from)
                })
                .filter(|n| {
                    n.starts_with(&only) && cases.join(format!("{n}.split")).exists()
                })
                .collect();
            names.sort();
            let r = Runner::load(&model)?;
            for name in names {
                let ids = read_ids(&cases.join(format!("{name}.ids")))?;
                let split: usize =
                    std::fs::read_to_string(cases.join(format!("{name}.split")))?
                        .trim()
                        .parse()?;
                let mut out =
                    std::fs::File::create(cases.join(format!("{name}.lily")))?;
                let (mut sum, mut hits, mut k) = (0f64, 0usize, 0usize);
                let (pf, dt) = r.forced(&ids, split, 0, |i, row| {
                    let nll = -log_softmax_at(row, ids[i] as usize);
                    let (a, margin) = top2(row);
                    writeln!(out, "{i} {} {nll:.6} {a} {margin:.4}", ids[i])?;
                    sum += nll;
                    hits += usize::from(a == ids[i] as usize);
                    k += 1;
                    Ok(())
                })?;
                writeln!(out, "mean {:.6} argmax {hits}/{k}", sum / k as f64)?;
                println!(
                    "{name}: {k} tokens, mean NLL {:.6}, argmax {hits}/{k}, prefill {} in {pf:.2} s, {:.1} tok/s forced",
                    sum / k as f64,
                    split,
                    k as f64 / dt.max(1e-9)
                );
            }
        }
        Cmd::Greedy { model, ids, n } => {
            let ids = read_ids(&ids)?;
            let r = Runner::load(&model)?;
            let mut out = Vec::new();
            let len = ids.len();
            r.forced(&ids, len, n, |_, row| {
                out.push(top2(row).0 as u32);
                Ok(())
            })?;
            let tok = Tokenizer::from_model_dir(&model)?;
            println!(
                "{}",
                out.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            );
            eprintln!("{:?}", tok.decode(&out, false)?);
        }
        Cmd::Tok { model, files } => {
            let tok = Tokenizer::from_model_dir(&model)?;
            for f in files {
                let ids = tok.encode(&std::fs::read_to_string(&f)?)?;
                println!(
                    "{}",
                    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
                );
            }
        }
        Cmd::Score { model, fixture, limit, raw } => {
            let tok = Tokenizer::from_model_dir(&model)?;
            let r = Runner::load(&model)?;
            let manifest = std::fs::read_to_string(fixture.join("manifest.tsv"))?;
            let (mut nll_sum, mut n_tok, mut agree, mut first, mut cases) =
                (0f64, 0usize, 0usize, 0usize, 0usize);
            let mut lcp_sum = 0usize;
            for line in
                manifest.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            {
                if limit.is_some_and(|n| cases >= n) {
                    break;
                }
                let cols: Vec<&str> = line.split('\t').collect();
                let file = |dir: &str, col: &str| {
                    fixture
                        .join(dir)
                        .join(Path::new(col).file_name().unwrap_or_default())
                };
                let prompt = std::fs::read_to_string(file("prompts", cols[1]))?;
                let cont = std::fs::read_to_string(file("continuations", cols[2]))?;
                let mut ids = if raw {
                    tok.encode(&prompt)?
                } else {
                    ds4_chat_ids(&tok, &prompt)?
                };
                let split = ids.len();
                let c = tok.encode(&cont)?;
                ids.extend(&c);
                let (mut nll, mut hits, mut lcp, mut open, mut f1) =
                    (0f64, 0, 0, true, false);
                r.forced(&ids, split, 0, |i, row| {
                    nll -= log_softmax_at(row, ids[i] as usize);
                    let hit = top2(row).0 == ids[i] as usize;
                    hits += usize::from(hit);
                    if i == split {
                        f1 = hit;
                    }
                    open &= hit;
                    lcp += usize::from(open);
                    Ok(())
                })?;
                println!(
                    "{}  prompt {:5} cont {:4}  nll {:.4}  agree {:5.1}%  first {}  lcp {}",
                    cols[0],
                    split,
                    c.len(),
                    nll / c.len() as f64,
                    100.0 * hits as f64 / c.len() as f64,
                    f1,
                    lcp
                );
                nll_sum += nll;
                n_tok += c.len();
                agree += hits;
                first += usize::from(f1);
                lcp_sum += lcp;
                cases += 1;
            }
            println!(
                "total: {cases} cases, {n_tok} tokens, avg NLL {:.6}/token, first-token match {first}/{cases}, avg greedy prefix {:.2}, greedy agreement {:.2}%",
                nll_sum / n_tok as f64,
                lcp_sum as f64 / cases as f64,
                100.0 * agree as f64 / n_tok as f64
            );
        }
        Cmd::Selfcheck { model, prompt } => {
            let tok = Tokenizer::from_model_dir(&model)?;
            let ids = tok.encode(&prompt)?;
            let r = Runner::load(&model)?;
            let mut a = Vec::new();
            r.forced(&ids, ids.len(), 1, |_, row| {
                a = row.to_vec();
                Ok(())
            })?;
            let mut b = Vec::new();
            r.forced(&ids, 1, 1, |i, row| {
                if i == ids.len() {
                    b = row.to_vec();
                }
                Ok(())
            })?;
            let diff =
                a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let scale = a.iter().map(|x| x.abs()).fold(0f32, f32::max);
            println!(
                "{} tokens: prefill argmax {} decode argmax {}, max |diff| {diff:.4} of max |logit| {scale:.2}, finite {}",
                ids.len(),
                top2(&a).0,
                top2(&b).0,
                a.iter().chain(&b).all(|x| x.is_finite())
            );
        }
        Cmd::ChatIds { model, conversation } => {
            use lily::tokenizer::ChatRender;
            let tok = Tokenizer::from_model_dir(&model)?;
            let conv: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&conversation)?)?;
            let messages = conv["messages"].as_array().context("messages")?.clone();
            let tools = conv["tools"].as_array().cloned();
            for think in [false, true] {
                let render = ChatRender {
                    messages: &messages,
                    tools: tools.as_deref(),
                    enable_thinking: think,
                    reasoning_effort: None,
                    preserve_thinking: None,
                };
                let ids = tok.encode_chat(&render)?;
                println!(
                    "{}",
                    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
                );
            }
            println!("stops {:?}", tok.stop_tokens());
        }
        Cmd::Chat { model, prompt, n, think } => {
            use lily::tokenizer::ChatRender;
            let tok = Tokenizer::from_model_dir(&model)?;
            let messages = [serde_json::json!({"role": "user", "content": prompt})];
            let render = ChatRender {
                messages: &messages,
                tools: None,
                enable_thinking: think,
                reasoning_effort: None,
                preserve_thinking: None,
            };
            let ids = tok.encode_chat(&render)?;
            let r = Runner::load(&model)?;
            let mut out = Vec::new();
            let stops = tok.stop_tokens().to_vec();
            let len = ids.len();
            let mut done = false;
            let (pf, dt) = r.forced(&ids, len, n, |_, row| {
                let t = top2(row).0 as u32;
                if !done {
                    if stops.contains(&t) {
                        done = true;
                    } else {
                        out.push(t);
                    }
                }
                Ok(())
            })?;
            println!("{}", tok.decode(&out, false)?);
            eprintln!(
                "prompt {len} tokens, prefill {pf:.2} s ({:.1} tok/s); {n} steps in {dt:.2} s ({:.2} tok/s)",
                len as f64 / pf,
                n as f64 / dt
            );
        }
    }
    Ok(())
}
