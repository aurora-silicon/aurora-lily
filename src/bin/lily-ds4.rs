//! DeepSeek V4 Flash bring-up tool: compares the GPU forward pass with the
//! CPU reference layer by layer on real weights, and runs greedy generation.
//!
//! ```sh
//! lily-ds4 compare --model DeepSeek-V4-Flash-....gguf --tokens 0,671,6102 --layers 4
//! lily-ds4 generate --model DeepSeek-V4-Flash-....gguf --tokens ... --n 32
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use lily::deepseek4::model::{Ds4Model, LayerTrace, Logits};
use lily::deepseek4::reference::{RefModel, Trace};
use lily::metal::MetalContext;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// GPU against the CPU reference, layer by layer.
    Compare {
        #[arg(long)]
        model: PathBuf,
        /// Comma-separated token ids.
        #[arg(long, conflicts_with = "text")]
        tokens: Option<String>,
        /// A text file: BOS and its first `prompt_len` tokens.
        #[arg(long)]
        text: Option<PathBuf>,
        #[arg(long, default_value_t = 2300)]
        prompt_len: usize,
        /// Layers to compare (the CPU reference is slow).
        #[arg(long, default_value_t = 4)]
        layers: usize,
        /// The first layer compared (the CPU starts from the GPU's input).
        #[arg(long, default_value_t = 0)]
        from: usize,
        /// Rows a GPU run takes (the prompt is fed in runs of this many).
        #[arg(long, default_value_t = 256)]
        step: usize,
    },
    /// Teacher-forced scoring of ds4's official DeepSeek V4 Flash 0731
    /// continuations (`gguf-tools/quality-testing/data/flash`): NLL per
    /// token, first-token match, greedy prefix; and whether the local
    /// tokens of each continuation are the API's.
    Score {
        #[arg(long)]
        model: PathBuf,
        /// A fixture directory (manifest.tsv, prompts/, continuations/,
        /// responses/); paths in the manifest are taken by file name.
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Prefill and greedy decode speed: BOS and the first `prompt_len`
    /// tokens of a text, then `n` decoded tokens, `runs` times on fresh
    /// state.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 1024)]
        prompt_len: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
    },
    /// Speculative decoding with the DSpark drafter (support file beside the
    /// model or LILY_DS4_DSPARK) against plain greedy decoding: the same
    /// tokens, acceptance, speed.
    Spec {
        #[arg(long)]
        model: PathBuf,
        /// A raw text (its first `prompt_len` tokens after BOS), unless
        /// `--chat` gives a user message.
        #[arg(long)]
        text: Option<PathBuf>,
        #[arg(long, default_value_t = 256)]
        prompt_len: usize,
        #[arg(long)]
        chat: Option<String>,
        #[arg(long, default_value_t = 256)]
        n: usize,
        #[arg(long, default_value_t = 1)]
        runs: usize,
    },
    /// Long-context NLL: BOS and the first `prompt_len` tokens of each text
    /// as one prompt (runs of the prefill), then the next `n` tokens
    /// teacher-forced one decode step each; prints the mean NLL per text
    /// and each position's (`--per-token`) for paired comparisons.
    Ppl {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, num_args = 1..)]
        text: Vec<PathBuf>,
        #[arg(long, default_value_t = 4096)]
        prompt_len: usize,
        #[arg(long, default_value_t = 256)]
        n: usize,
        #[arg(long)]
        per_token: bool,
    },
    /// Greedy generation from token ids.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokens: String,
        #[arg(long, default_value_t = 32)]
        n: usize,
    },
}

fn parse_tokens(s: &str) -> Result<Vec<u32>> {
    s.split(',').map(|t| t.trim().parse::<u32>().context("token id")).collect()
}

/// Exits 75 when another lily instance holds the lock (`lily::instance`).
fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    lily::instance::acquire()?;
    match cli.cmd {
        Cmd::Compare { model, tokens, text, prompt_len, layers, from, step } => {
            let ids = match (tokens, text) {
                (Some(t), _) => parse_tokens(&t)?,
                (None, Some(f)) => text_tokens(&model, &f, prompt_len)?,
                (None, None) => anyhow::bail!("--tokens or --text"),
            };
            compare(&model, &ids, from, layers, step)
        }
        Cmd::Generate { model, tokens, n } => {
            generate(&model, &parse_tokens(&tokens)?, n)
        }
        Cmd::Score { model, fixture, limit } => score(&model, &fixture, limit),
        Cmd::Spec { model, text, prompt_len, chat, n, runs } => {
            spec(&model, text.as_ref(), prompt_len, chat.as_deref(), n, runs)
        }
        Cmd::Bench { model, text, prompt_len, n, runs } => {
            bench(&model, &text, prompt_len, n, runs)
        }
        Cmd::Ppl { model, text, prompt_len, n, per_token } => {
            ppl(&model, &text, prompt_len, n, per_token)
        }
    }
}

/// BOS and the first `n` tokens of a text file.
fn text_tokens(model: &PathBuf, file: &PathBuf, n: usize) -> Result<Vec<u32>> {
    let tok = lily::tokenizer::Tokenizer::from_model_dir(model)?;
    let text = std::fs::read_to_string(file)?;
    let mut ids = vec![0u32];
    ids.extend(tok.encode(&text)?.into_iter().take(n));
    Ok(ids)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

/// Relative L2 error and max abs error of `a` against `b`.
fn err(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut d2 = 0f64;
    let mut n2 = 0f64;
    let mut mx = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        d2 += d * d;
        n2 += f64::from(y) * f64::from(y);
        mx = mx.max(d.abs());
    }
    ((d2 / n2.max(1e-30)).sqrt(), mx)
}

fn compare(
    path: &PathBuf,
    tokens: &[u32],
    from: usize,
    layers: usize,
    step: usize,
) -> Result<()> {
    let ctx = MetalContext::new()?;
    let t0 = Instant::now();
    let model = Ds4Model::load(&ctx, path)?;
    eprintln!("mapped in {:.1} s", t0.elapsed().as_secs_f64());
    let cfg = model.cfg.clone();
    let mut st = model.new_state(&ctx, tokens.len() + 8)?;
    let sc = model.new_scratch(&ctx)?;
    let mut gpu: Vec<LayerTrace> = vec![LayerTrace::default(); cfg.n_layer];
    let mut last_logits = Vec::new();
    for chunk in tokens.chunks(step.max(1)) {
        let mut tr = Vec::new();
        model.forward_traced(&ctx, &mut st, &sc, chunk, &mut tr)?;
        // Keep only the layers compared (and the input of the first): a
        // long prompt's traces of every layer would not fit in memory.
        for (il, (g, t)) in gpu.iter_mut().zip(tr).enumerate() {
            if il + 1 < from || il >= from + layers {
                continue;
            }
            g.attn_in.extend(t.attn_in);
            g.q.extend(t.q);
            g.kv.extend(t.kv);
            g.heads.extend(t.heads);
            g.attn_out.extend(t.attn_out);
            g.ffn_in.extend(t.ffn_in);
            g.ffn_out.extend(t.ffn_out);
            g.sel.extend(t.sel);
            g.h.extend(t.h);
        }
        last_logits = sc.logits.to_f32()?[..cfg.vocab].to_vec();
    }
    let top = argmax(&last_logits);
    eprintln!("GPU traced forward done; top token {top} ({:.3})", last_logits[top]);
    let hash = RefModel::hash_tables(&cfg, &model.gguf)?;
    let rm = RefModel::new(cfg.clone(), &model.gguf, hash);
    let hd = cfg.hc_dim();
    let mut hs = rm.embed(tokens)?;
    if from > 0 {
        let g = &gpu[from - 1].h;
        for (i, h) in hs.iter_mut().enumerate() {
            h.copy_from_slice(&g[i * hd..(i + 1) * hd]);
        }
    }
    println!(
        "layer ratio  attn_in        q              kv             heads          attn_out       ffn_in         ffn_out        h   (rel L2 / max abs)"
    );
    for il in from..(from + layers).min(cfg.n_layer) {
        let t = Instant::now();
        let traces: Vec<Trace> = rm.layer(il, tokens, &mut hs)?;
        let g = &gpu[il];
        let cat = |f: fn(&Trace) -> &Vec<f32>| -> Vec<f32> {
            traces.iter().flat_map(|t| f(t).clone()).collect()
        };
        let h_cpu: Vec<f32> = hs.iter().flatten().copied().collect();
        let cols = [
            err(&g.attn_in, &cat(|t| &t.attn_in)),
            err(&g.q, &cat(|t| &t.q)),
            err(&g.kv, &cat(|t| &t.kv)),
            err(&g.heads, &cat(|t| &t.heads)),
            err(&g.attn_out, &cat(|t| &t.attn_out)),
            err(&g.ffn_in, &cat(|t| &t.ffn_in)),
            err(&g.ffn_out, &cat(|t| &t.ffn_out)),
            err(&g.h, &h_cpu),
        ];
        print!("{il:5} {:5}", cfg.ratio(il));
        for (r, m) in cols {
            print!("  {r:.2e}/{m:.1e}");
        }
        println!("   ({:.1} s)", t.elapsed().as_secs_f64());
        let k = cfg.n_expert_used;
        let diff: Vec<usize> = traces
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                let mut a: Vec<u32> = t.selected.iter().map(|&e| e as u32).collect();
                let mut b = g.sel[i * k..(i + 1) * k].to_vec();
                a.sort();
                b.sort();
                a != b
            })
            .map(|(i, _)| i)
            .collect();
        if !diff.is_empty() {
            println!("       experts differ at rows {diff:?}");
        }
        // Continue from the GPU's state so errors do not accumulate.
        for (i, h) in hs.iter_mut().enumerate() {
            h.copy_from_slice(&g.h[i * hd..(i + 1) * hd]);
        }
    }
    Ok(())
}

fn generate(path: &PathBuf, prompt: &[u32], n: usize) -> Result<()> {
    let ctx = MetalContext::new()?;
    let t0 = Instant::now();
    let model = Ds4Model::load(&ctx, path)?;
    eprintln!("mapped in {:.1} s", t0.elapsed().as_secs_f64());
    let mut st = model.new_state(&ctx, prompt.len() + n + 8)?;
    let sc = model.new_scratch(&ctx)?;
    let t = Instant::now();
    model.forward(&ctx, &mut st, &sc, prompt, Logits::Last)?;
    let pf = t.elapsed().as_secs_f64();
    eprintln!(
        "prefill {} tokens in {:.2} s ({:.1} tok/s)",
        prompt.len(),
        pf,
        prompt.len() as f64 / pf
    );
    let vocab = model.cfg.vocab;
    let mut out = Vec::new();
    let t = Instant::now();
    let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    for _ in 0..n {
        out.push(next);
        model.forward(&ctx, &mut st, &sc, &[next], Logits::Last)?;
        next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
    eprintln!("decode {n} tokens in {dt:.2} s ({:.2} tok/s)", n as f64 / dt);
    Ok(())
}

fn log_softmax_at(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - m).exp()).sum();
    f64::from(logits[target] - m) - sum.ln()
}

/// ds4's `score_official` with its default (no-thinking chat) prompt:
/// `BOS <｜User｜>prompt<｜Assistant｜></think>`, the continuation
/// teacher-forced after it.
fn score(path: &PathBuf, fixture: &PathBuf, limit: Option<usize>) -> Result<()> {
    use lily::deepseek4::model::MAX_ROWS;
    let tok = lily::tokenizer::Tokenizer::from_model_dir(path)?;
    let ctx = MetalContext::new()?;
    let model = Ds4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let manifest = std::fs::read_to_string(fixture.join("manifest.tsv"))?;
    let sc = model.new_scratch(&ctx)?;
    let (mut tot_nll, mut tot_n, mut tot_agree) = (0f64, 0usize, 0usize);
    let (mut first_matches, mut total_lcp, mut cases) = (0usize, 0usize, 0usize);
    let (mut prompt_count_ok, mut bounds_ok) = (0usize, 0usize);
    let mut case_nll_sum = 0f64;
    println!("case      prompt(api) cont(api)  nll/tok  agree   first lcp  prefill_s");
    for line in manifest.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        if limit.is_some_and(|n| cases >= n) {
            break;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let case = cols[0];
        let file = |dir: &str, col: &str| {
            fixture.join(dir).join(std::path::Path::new(col).file_name().unwrap())
        };
        let prompt = std::fs::read_to_string(file("prompts", cols[1]))?;
        let cont = std::fs::read_to_string(file("continuations", cols[2]))?;
        let resp: serde_json::Value = match cols.get(3) {
            Some(c) if !c.is_empty() => {
                serde_json::from_slice(&std::fs::read(file("responses", c))?)?
            }
            _ => serde_json::Value::Null,
        };
        let api_prompt = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
        let api_tokens: Vec<Vec<u8>> = resp["choices"][0]["logprobs"]["content"]
            .as_array()
            .map(|v| {
                v.iter()
                    .map(|t| {
                        t["bytes"]
                            .as_array()
                            .map(|b| {
                                b.iter()
                                    .map(|x| x.as_u64().unwrap_or(0) as u8)
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let rendered =
            format!("<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>");
        let p_ids = tok.encode(&rendered)?;
        let c_ids = tok.encode(&cont)?;
        // The continuation's local token boundaries against the API's.
        let local: Vec<Vec<u8>> = c_ids
            .iter()
            .map(|&i| Ok(tok.decode(&[i], false)?.into_bytes()))
            .collect::<Result<_>>()?;
        let same_bounds = local == api_tokens;
        bounds_ok += usize::from(same_bounds);
        prompt_count_ok += usize::from(p_ids.len() == api_prompt);
        let mut st = model.new_state(&ctx, p_ids.len() + c_ids.len() + 8)?;
        let t = Instant::now();
        model.forward(&ctx, &mut st, &sc, &p_ids[..p_ids.len() - 1], Logits::None)?;
        let pf = t.elapsed().as_secs_f64();
        let feed: Vec<u32> = std::iter::once(*p_ids.last().unwrap())
            .chain(c_ids.iter().copied())
            .collect();
        let (mut nll, mut agree, mut lcp) = (0f64, 0usize, 0usize);
        let (mut open, mut first) = (true, false);
        for (ci, chunk) in feed[..c_ids.len()].chunks(MAX_ROWS).enumerate() {
            model.forward(&ctx, &mut st, &sc, chunk, Logits::All)?;
            let logits = sc.logits.to_f32()?;
            for r in 0..chunk.len() {
                let row = &logits[r * vocab..(r + 1) * vocab];
                let at = ci * MAX_ROWS + r;
                let target = c_ids[at] as usize;
                nll -= log_softmax_at(row, target);
                let hit = argmax(row) == target;
                agree += usize::from(hit);
                if at == 0 {
                    first = hit;
                }
                open &= hit;
                lcp += usize::from(open);
            }
        }
        println!(
            "{case}  {:5}({:5}) {:4}({:4}{})  {:7.4}  {:5.1}%  {:5} {:3}  {:6.2}",
            p_ids.len(),
            api_prompt,
            c_ids.len(),
            api_tokens.len(),
            if same_bounds { "" } else { "!" },
            nll / c_ids.len() as f64,
            100.0 * agree as f64 / c_ids.len() as f64,
            first,
            lcp,
            pf
        );
        tot_nll += nll;
        tot_n += c_ids.len();
        tot_agree += agree;
        first_matches += usize::from(first);
        total_lcp += lcp;
        case_nll_sum += nll / c_ids.len() as f64;
        cases += 1;
    }
    println!(
        "total: {cases} cases, {tot_n} tokens, avg NLL {:.6}/token (case mean {:.6}), first-token match {first_matches}/{cases}, avg greedy prefix {:.2}, greedy agreement {:.2}%, prompt tokens = API's in {prompt_count_ok}/{cases}, continuation tokens = API's in {bounds_ok}/{cases}",
        tot_nll / tot_n as f64,
        case_nll_sum / cases as f64,
        total_lcp as f64 / cases as f64,
        100.0 * tot_agree as f64 / tot_n as f64
    );
    Ok(())
}

fn bench(
    path: &PathBuf,
    text: &PathBuf,
    prompt_len: usize,
    n: usize,
    runs: usize,
) -> Result<()> {
    let prompt = text_tokens(path, text, prompt_len)?;
    let ctx = MetalContext::new()?;
    let model = Ds4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let sc = model.new_scratch(&ctx)?;
    let mut st = model.new_state(&ctx, prompt.len() + n + 8)?;
    for run in 0..runs {
        st.reset();
        let _ = lily::metal::profile::take();
        let t = Instant::now();
        model.forward(&ctx, &mut st, &sc, &prompt, Logits::Last)?;
        let pf = t.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("prefill of {} tokens, per prefill:", prompt.len());
            profile_report(1);
        }
        let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
        let mut out = Vec::with_capacity(n);
        let mut steps = Vec::with_capacity(n);
        let t = Instant::now();
        for _ in 0..n {
            out.push(next);
            let s = Instant::now();
            model.forward(&ctx, &mut st, &sc, &[next], Logits::Last)?;
            next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
            steps.push(s.elapsed().as_secs_f64() * 1e3);
        }
        let dt = t.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("decode at prompt {}, per step:", prompt.len());
            profile_report(n);
        }
        steps.sort_by(f64::total_cmp);
        let digest = out.iter().fold(0xcbf29ce484222325u64, |h, &t| {
            (h ^ u64::from(t)).wrapping_mul(0x100000001b3)
        });
        println!(
            "run {run}: prompt {} prefill {pf:.2} s ({:.1} tok/s), decode {n} in {dt:.2} s ({:.2} tok/s, median {:.2} ms/step), digest {digest:016x}",
            prompt.len(),
            prompt.len() as f64 / pf,
            n as f64 / dt,
            steps[steps.len() / 2]
        );
    }
    Ok(())
}

fn ppl(
    path: &PathBuf,
    texts: &[PathBuf],
    prompt_len: usize,
    n: usize,
    per_token: bool,
) -> Result<()> {
    let ctx = MetalContext::new()?;
    let model = Ds4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let sc = model.new_scratch(&ctx)?;
    let mut all = Vec::new();
    for text in texts {
        let ids = text_tokens(path, text, prompt_len + n)?;
        let name = text.display();
        anyhow::ensure!(ids.len() == prompt_len + n + 1, "{name} is too short");
        let mut st = model.new_state(&ctx, ids.len() + 8)?;
        model.forward(&ctx, &mut st, &sc, &ids[..=prompt_len], Logits::Last)?;
        let mut nll = Vec::with_capacity(n);
        for j in 0..n {
            let at = prompt_len + 1 + j;
            let logits = &sc.logits.to_f32()?[..vocab];
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + logits.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
            nll.push(lse - logits[ids[at] as usize] as f64);
            if j + 1 < n {
                model.forward(&ctx, &mut st, &sc, &ids[at..=at], Logits::Last)?;
            }
        }
        let mean = nll.iter().sum::<f64>() / n as f64;
        println!("ppl {name}: prompt {prompt_len}, {n} tokens, mean NLL {mean:.6}");
        if per_token {
            for (j, v) in nll.iter().enumerate() {
                println!("tok {name} {j} {v:.6}");
            }
        }
        all.extend(nll);
    }
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    println!("ppl total: {} tokens, mean NLL {mean:.6}", all.len());
    Ok(())
}

/// With `LILY_KERNEL_PROFILE=1`: GPU time per kernel name per pass (each
/// dispatch timed alone, so the sum exceeds a real pass).
fn profile_report(steps: usize) {
    let passes = lily::metal::profile::take();
    let mut by: std::collections::HashMap<&str, (usize, f64)> = Default::default();
    let mut total = 0.0;
    for p in &passes {
        for k in &p.kernels {
            let e = by.entry(k.name).or_default();
            e.0 += 1;
            e.1 += k.gpu_secs;
            total += k.gpu_secs;
        }
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    let n = steps as f64;
    eprintln!("  ms/step  share  calls/step  kernel");
    for (name, (calls, secs)) in rows {
        eprintln!(
            "  {:7.3}  {:4.1}%  {:9.1}  {name}",
            secs / n * 1e3,
            100.0 * secs / total,
            calls as f64 / n
        );
    }
    eprintln!("  {:7.3}  kernels summed per step", total / n * 1e3);
}

fn spec(
    path: &PathBuf,
    text: Option<&PathBuf>,
    prompt_len: usize,
    chat: Option<&str>,
    n: usize,
    runs: usize,
) -> Result<()> {
    use lily::engine::{Draw, LanguageModel, LoadOptions};
    use lily::kernels::sample::SamplingParams;
    let tok = lily::tokenizer::Tokenizer::from_model_dir(path)?;
    let ids = match (chat, text) {
        (Some(q), _) => {
            let messages = [serde_json::json!({"role": "user", "content": q})];
            tok.encode_chat(&lily::tokenizer::ChatRender {
                messages: &messages,
                tools: None,
                enable_thinking: false,
                reasoning_effort: None,
                preserve_thinking: None,
            })?
        }
        (None, Some(f)) => text_tokens(path, f, prompt_len)?,
        (None, None) => anyhow::bail!("--chat or --text"),
    };
    let ctx = MetalContext::new()?;
    let options = LoadOptions { mtp_drafts: 5, ..Default::default() };
    let model = <Ds4Model as LanguageModel>::load(&ctx, path, &options)?;
    let k = model.max_drafts();
    let greedy = SamplingParams::greedy();
    let mut sc = LanguageModel::new_scratch_with_capacity(&model, &ctx, 0)?;
    let vocab = model.cfg.vocab;
    for run in 0..runs {
        // Plain greedy reference.
        let mut st = LanguageModel::new_state(&model, &ctx, ids.len() + n + 16)?;
        let draw = Some(Draw { params: &greedy, step: 0 });
        LanguageModel::prefill(&model, &ctx, &mut st, &mut sc, &ids, draw)?;
        let mut plain = vec![sc.next_token.to_u32()?[0]];
        let t = Instant::now();
        while plain.len() < n {
            let last = *plain.last().unwrap();
            model.forward(&ctx, &mut st, &sc, &[last], Logits::Last)?;
            plain.push(argmax(&sc.logits.to_f32()?[..vocab]) as u32);
        }
        let plain_s = t.elapsed().as_secs_f64();
        // Speculative.
        let mut st = LanguageModel::new_state(&model, &ctx, ids.len() + n + 16)?;
        let draw = Some(Draw { params: &greedy, step: 0 });
        LanguageModel::prefill(&model, &ctx, &mut st, &mut sc, &ids, draw)?;
        let mut out = vec![sc.next_token.to_u32()?[0]];
        let t = Instant::now();
        let mut proposals = model.draft_initial(
            &ctx,
            &mut st,
            &mut sc,
            out[0],
            k,
            &greedy,
            out.len(),
        )?;
        let (mut steps, mut accepted, mut proposed) = (0usize, 0usize, 0usize);
        let mut hist = vec![0usize; k + 1];
        let (mut t_verify, mut t_finish) = (0f64, 0f64);
        while out.len() < n {
            let pending = *out.last().unwrap();
            let tv = Instant::now();
            let (draws, pend) = model.verify(
                &ctx,
                &mut st,
                &mut sc,
                pending,
                &proposals,
                &greedy,
                out.len(),
                k,
            )?;
            let kept = proposals.iter().zip(&draws).take_while(|(p, d)| p == d).count();
            out.extend_from_slice(&draws[..=kept]);
            accepted += kept;
            proposed += proposals.len();
            hist[kept] += 1;
            steps += 1;
            let tf = Instant::now();
            t_verify += (tf - tv).as_secs_f64();
            proposals =
                model.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pend)?;
            t_finish += tf.elapsed().as_secs_f64();
        }
        let spec_s = t.elapsed().as_secs_f64();
        out.truncate(n);
        let same = plain.iter().zip(&out).take_while(|(a, b)| a == b).count();
        println!(
            "run {run}: plain {:.2} tok/s; speculative {:.2} tok/s, {steps} steps, {:.2} tokens a step, accepted {accepted}/{proposed} drafts ({:.0}%), kept per step {hist:?}; same first {same}/{n} tokens; per step: verify {:.2} ms, finish+draft {:.2} ms; plain step {:.2} ms",
            n as f64 / plain_s,
            n as f64 / spec_s,
            (n - 1) as f64 / steps as f64,
            100.0 * accepted as f64 / proposed.max(1) as f64,
            t_verify / steps as f64 * 1e3,
            t_finish / steps as f64 * 1e3,
            plain_s / (n - 1) as f64 * 1e3,
        );
        if run == 0 {
            println!("speculative text: {:?}", tok.decode(&out, false)?);
            if same < n {
                println!("plain text:       {:?}", tok.decode(&plain, false)?);
            }
        }
    }
    Ok(())
}
