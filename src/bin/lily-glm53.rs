//! GLM 5.3 Flash bring-up tool: compares the GPU forward pass with the CPU
//! reference layer by layer on real weights, and runs greedy generation.
//!
//! ```sh
//! lily-glm53 compare --model GLM-5.3-Flash-Q2.gguf --tokens 154822,9703,11 --layers 4
//! lily-glm53 generate --model GLM-5.3-Flash-Q2.gguf --tokens ... --n 32
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use lily::glm53::reference::{RefModel, Trace};
use lily::glm53::{Glm53Model, LayerTrace};
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
        #[arg(long)]
        tokens: String,
        /// Trunk layers to compare (the CPU reference is slow).
        #[arg(long, default_value_t = 4)]
        layers: usize,
        /// Feed the tokens one at a time instead of in runs of four.
        #[arg(long)]
        single: bool,
    },
    /// Teacher-forced scoring of ds4's official GLM 5.3 Flash continuations:
    /// mean NLL per token and greedy agreement, per case and overall.
    Score {
        #[arg(long)]
        model: PathBuf,
        /// A fixture directory (manifest.tsv, prompts/, continuations/,
        /// responses/); paths in the manifest are taken by file name.
        #[arg(long)]
        fixture: PathBuf,
        /// Reasoning effort the fixture was collected with.
        #[arg(long, default_value = "low")]
        effort: String,
        /// Score only the first N cases.
        #[arg(long)]
        limit: Option<usize>,
        /// ds4's legacy no-thinking prompt (`[gMASK]<sop><|user|>...
        /// <|assistant|><think></think>`, no effort line, no reasoning): the
        /// rendering ds4's 100-case references were scored with.
        #[arg(long)]
        legacy: bool,
        /// Raw text: the prompt after `[gMASK]<sop>` with no chat template,
        /// the continuation right after it (the fixture needs no responses).
        #[arg(long)]
        raw: bool,
        /// Write every scored token's NLL, greedy hit and top-2 margin to
        /// this file (tab-separated: case, index, target, nll, hit, margin).
        #[arg(long)]
        dump: Option<PathBuf>,
    },
    /// Chat: renders one user message with the GGUF's template and decodes
    /// greedily, printing the text.
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        n: usize,
        /// Open a reasoning block (the template's default).
        #[arg(long)]
        think: bool,
    },
    /// Speed: prefill the first `prompt_len` tokens of a text file (after
    /// `[gMASK]<sop>`), then decode `n` tokens greedily; `runs` times, each
    /// on a fresh state.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 1024)]
        prompt_len: usize,
        #[arg(long, default_value_t = 256)]
        n: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
    },
    /// Greedy speculative decoding with the MTP head through the engine
    /// interface, against plain greedy decoding of the same prompt.
    Spec {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 256)]
        prompt_len: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 2)]
        runs: usize,
        /// Use the chat template around a question instead of raw text.
        #[arg(long)]
        chat: Option<String>,
    },
    /// Where greedy speculative decoding first parts from plain decoding:
    /// replays the speculation to the verify pass that draws the first
    /// differing token and compares, layer by layer with the production
    /// kernels, that pass's row against one-row steps on the same state and
    /// against plain decoding's own state (routed experts and their top-8
    /// margin, sublayer outputs, logits, recurrent state).
    SpecDiff {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 256)]
        prompt_len: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        /// Use the chat template around a question instead of raw text.
        #[arg(long)]
        chat: Option<String>,
    },
    /// Step time by row count: after a prompt, forward passes of 1 to 4
    /// rows (every row's logits, as a verify pass), each from the prompt's
    /// state (restored); medians.
    Rows {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 256)]
        prompt_len: usize,
        #[arg(long, default_value_t = 24)]
        n: usize,
    },
    /// Greedy generation from token ids.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokens: String,
        #[arg(long, default_value_t = 16)]
        n: usize,
    },
}

fn parse_tokens(s: &str) -> Result<Vec<u32>> {
    s.split(',').map(|t| t.trim().parse::<u32>().context("token id")).collect()
}

/// Exits 75 when another lily instance holds the lock (`lily::instance`):
/// every subcommand loads the model.
fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    // Before any GPU context or mapping: one model on this machine at once.
    lily::instance::acquire()?;
    match cli.cmd {
        Cmd::Compare { model, tokens, layers, single } => {
            compare(&model, &parse_tokens(&tokens)?, layers, single)
        }
        Cmd::Generate { model, tokens, n } => {
            generate(&model, &parse_tokens(&tokens)?, n)
        }
        Cmd::Score { model, fixture, effort, limit, legacy, raw, dump } => {
            score(&model, &fixture, &effort, limit, legacy, raw, dump.as_deref())
        }
        Cmd::Chat { model, prompt, n, think } => chat(&model, &prompt, n, think),
        Cmd::Spec { model, text, prompt_len, n, runs, chat } => {
            spec(&model, &text, prompt_len, n, runs, chat.as_deref())
        }
        Cmd::SpecDiff { model, text, prompt_len, n, chat } => {
            spec_diff(&model, &text, prompt_len, n, chat.as_deref())
        }
        Cmd::Rows { model, text, prompt_len, n } => rows(&model, &text, prompt_len, n),
        Cmd::Bench { model, text, prompt_len, n, runs } => {
            bench(&model, &text, prompt_len, n, runs)
        }
    }
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

fn compare(path: &PathBuf, tokens: &[u32], layers: usize, single: bool) -> Result<()> {
    let ctx = MetalContext::new()?;
    let t0 = Instant::now();
    let model = Glm53Model::load(&ctx, path)?;
    eprintln!("mapped in {:.1} s", t0.elapsed().as_secs_f64());
    let cfg = model.cfg.clone();
    let mut st = model.new_state(&ctx, tokens.len() + 8)?;
    let mut sc = model.new_scratch(&ctx)?;
    // GPU: per layer, the rows of every step concatenated in token order.
    let mut gpu: Vec<LayerTrace> = vec![LayerTrace::default(); cfg.n_trunk];
    let step = if single { 1 } else { lily::glm53::model::MAX_ROWS };
    let mut last_logits = Vec::new();
    for chunk in tokens.chunks(step) {
        let mut tr = Vec::new();
        model.forward_traced(&ctx, &mut st, &mut sc, chunk, &mut tr)?;
        for (g, t) in gpu.iter_mut().zip(tr) {
            g.attn_in.extend(t.attn_in);
            g.attn_out.extend(t.attn_out);
            g.ffn_in.extend(t.ffn_in);
            g.ffn_out.extend(t.ffn_out);
            g.h.extend(t.h);
        }
        last_logits = sc.logits.to_f32()?[..cfg.vocab].to_vec();
    }
    eprintln!("GPU traced forward done; top token {}", argmax(&last_logits));
    // CPU, layer by layer from the GPU's own input to each layer, so every
    // layer is compared alone.
    let rm = RefModel::new(cfg.clone(), &model.gguf);
    let hd = cfg.hc_dim();
    let mut hs = rm.embed(tokens)?;
    println!(
        "layer  kind   attn_in        attn_out       ffn_in         ffn_out        h (rel L2 / max abs)"
    );
    for il in 0..layers.min(cfg.n_trunk) {
        let t = Instant::now();
        let traces: Vec<Trace> = rm.trunk_layer(il, &mut hs)?;
        let g = &gpu[il];
        let cat = |f: fn(&Trace) -> &Vec<f32>| -> Vec<f32> {
            traces.iter().flat_map(|t| f(t).clone()).collect()
        };
        let h_cpu: Vec<f32> = hs.iter().flatten().copied().collect();
        let cols = [
            err(&g.attn_in, &cat(|t| &t.attn_in)),
            err(&g.attn_out, &cat(|t| &t.attn_out)),
            err(&g.ffn_in, &cat(|t| &t.ffn_in)),
            err(&g.ffn_out, &cat(|t| &t.ffn_out)),
            err(&g.h, &h_cpu),
        ];
        print!("{il:5}  {:5?}", cfg.kinds[il]);
        for (r, m) in cols {
            print!("  {r:.2e}/{m:.1e}");
        }
        println!("   ({:.1} s)", t.elapsed().as_secs_f64());
        let sel: Vec<String> =
            traces.iter().map(|t| format!("{:?}", t.selected)).collect();
        if !traces[0].selected.is_empty() {
            println!("       experts {}", sel.join(" "));
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
    let model = Glm53Model::load(&ctx, path)?;
    eprintln!("mapped in {:.1} s", t0.elapsed().as_secs_f64());
    let mut st = model.new_state(&ctx, prompt.len() + n + 8)?;
    let mut sc = model.new_scratch(&ctx)?;
    let t = Instant::now();
    model.prefill(&ctx, &mut st, &mut sc, prompt)?;
    let pf = t.elapsed().as_secs_f64();
    eprintln!(
        "prefill {} tokens in {:.2} s ({:.1} tok/s)",
        prompt.len(),
        pf,
        prompt.len() as f64 / pf
    );
    let mut out = Vec::new();
    // Profile only the decode steps.
    let _ = lily::metal::profile::take();
    let t = Instant::now();
    let vocab = model.cfg.vocab;
    let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    for _ in 0..n {
        out.push(next);
        model.forward(&ctx, &mut st, &mut sc, &[next], true)?;
        next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    }
    let dt = t.elapsed().as_secs_f64();
    if ctx.profiling() {
        profile_report(n);
    }
    println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
    eprintln!("decode {n} tokens in {dt:.2} s ({:.2} tok/s)", n as f64 / dt);
    Ok(())
}

fn log_softmax_at(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - m).exp()).sum();
    f64::from(logits[target] - m) - sum.ln()
}

fn score(
    path: &PathBuf,
    fixture: &PathBuf,
    effort: &str,
    limit: Option<usize>,
    legacy: bool,
    raw: bool,
    dump: Option<&std::path::Path>,
) -> Result<()> {
    use lily::tokenizer::Tokenizer;
    use std::io::Write as _;
    let mut dump = dump.map(std::fs::File::create).transpose()?;
    let tok = Tokenizer::from_gguf(path)?;
    let ctx = MetalContext::new()?;
    let model = Glm53Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let manifest = std::fs::read_to_string(fixture.join("manifest.tsv"))?;
    let mut sc = model.new_scratch(&ctx)?;
    let (mut tot_nll, mut tot_n, mut tot_agree) = (0f64, 0usize, 0usize);
    let mut token_count_matches = 0;
    let (mut first_matches, mut total_lcp) = (0usize, 0usize);
    let mut case_nll_sum = 0f64;
    let mut cases = 0;
    let effort_cap = {
        let mut c = effort.chars();
        c.next()
            .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
            .unwrap_or_default()
    };
    println!("case       prompt(api) cont   nll/tok  agree   first lcp  prefill_s");
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
        let resp: serde_json::Value = if raw {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&std::fs::read(file("responses", cols[3]))?)?
        };
        let msg = &resp["choices"][0]["message"];
        let reasoning = msg["reasoning"]
            .as_str()
            .or(msg["reasoning_content"].as_str())
            .unwrap_or("");
        let api_prompt = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
        // ds4's render_glm_references.py: the official template with the
        // API's reasoning in the prefix.
        let rendered = if raw {
            format!("[gMASK]<sop>{prompt}")
        } else if legacy {
            format!("[gMASK]<sop><|user|>{prompt}<|assistant|><think></think>")
        } else {
            format!(
                "[gMASK]<sop><|system|>Reasoning Effort: {effort_cap}<|user|>{prompt}<|assistant|><think>{reasoning}</think>"
            )
        };
        let p_ids = tok.encode(&rendered)?;
        let c_ids = tok.encode(&cont)?;
        let mut st = model.new_state(&ctx, p_ids.len() + c_ids.len() + 8)?;
        let t = Instant::now();
        // All but the prompt's last token, then teacher-force from there.
        model.prefill(&ctx, &mut st, &mut sc, &p_ids[..p_ids.len() - 1])?;
        let pf = t.elapsed().as_secs_f64();
        let feed: Vec<u32> = std::iter::once(*p_ids.last().unwrap())
            .chain(c_ids.iter().copied())
            .collect();
        let (mut nll, mut agree) = (0f64, 0usize);
        let (mut lcp, mut prefix_open) = (0usize, true);
        let mut first = false;
        for (ci, chunk) in
            feed[..c_ids.len()].chunks(lily::glm53::model::MAX_ROWS).enumerate()
        {
            model.forward_all_logits(&ctx, &mut st, &mut sc, chunk)?;
            let logits = sc.logits.to_f32()?;
            for r in 0..chunk.len() {
                let row = &logits[r * vocab..(r + 1) * vocab];
                let at = ci * lily::glm53::model::MAX_ROWS + r;
                let target = c_ids[at] as usize;
                let lp = log_softmax_at(row, target);
                nll -= lp;
                let top = argmax(row);
                let hit = top == target;
                if let Some(f) = dump.as_mut() {
                    let second = row
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != top)
                        .map(|(_, &v)| v)
                        .fold(f32::NEG_INFINITY, f32::max);
                    writeln!(
                        f,
                        "{case}\t{at}\t{target}\t{:.6}\t{}\t{:.4}",
                        -lp,
                        u8::from(hit),
                        row[top] - second
                    )?;
                }
                agree += usize::from(hit);
                if ci == 0 && r == 0 {
                    first = hit;
                }
                prefix_open &= hit;
                lcp += usize::from(prefix_open);
            }
        }
        // The reasoning-free prompt of a case without reasoning has the API's
        // token count; with reasoning the API counts it as completion.
        let reasoning_tokens = if legacy || reasoning.is_empty() {
            0
        } else {
            tok.encode(reasoning)?.len() + 2
        };
        token_count_matches +=
            usize::from(p_ids.len() - reasoning_tokens == api_prompt);
        println!(
            "{case}  {:5}({:5}) {:4}  {:7.4}  {:5.1}%  {:5} {:3}  {:6.1}",
            p_ids.len(),
            api_prompt,
            c_ids.len(),
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
        "total: {cases} cases, {tot_n} tokens, avg NLL {:.6}/token (case mean {:.6}), first-token match {first_matches}/{cases}, avg greedy prefix {:.2}, greedy agreement {:.2}%, prompt token count = API's in {token_count_matches}/{cases}",
        tot_nll / tot_n as f64,
        case_nll_sum / cases as f64,
        total_lcp as f64 / cases as f64,
        100.0 * tot_agree as f64 / tot_n as f64
    );
    Ok(())
}

fn chat(path: &PathBuf, prompt: &str, n: usize, think: bool) -> Result<()> {
    use lily::tokenizer::{ChatRender, Tokenizer};
    let tok = Tokenizer::from_gguf(path)?;
    let messages = [serde_json::json!({"role": "user", "content": prompt})];
    let render = ChatRender {
        messages: &messages,
        tools: None,
        enable_thinking: think,
        reasoning_effort: None,
        preserve_thinking: None,
    };
    eprintln!("{:?}", tok.render(&render)?);
    let ids = tok.encode_chat(&render)?;
    let ctx = MetalContext::new()?;
    let model = Glm53Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let mut st = model.new_state(&ctx, ids.len() + n + 8)?;
    let mut sc = model.new_scratch(&ctx)?;
    let t = Instant::now();
    model.prefill(&ctx, &mut st, &mut sc, &ids)?;
    let pf = t.elapsed().as_secs_f64();
    let mut out = Vec::new();
    let t = Instant::now();
    let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    while out.len() < n && !tok.stop_tokens().contains(&next) {
        out.push(next);
        model.forward(&ctx, &mut st, &mut sc, &[next], true)?;
        next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("{}", tok.decode(&out, false)?);
    eprintln!(
        "prompt {} tokens, prefill {:.2} s ({:.1} tok/s); decode {} tokens in {:.2} s ({:.2} tok/s); stop {next}",
        ids.len(),
        pf,
        ids.len() as f64 / pf,
        out.len(),
        dt,
        out.len() as f64 / dt
    );
    Ok(())
}

/// With `LILY_KERNEL_PROFILE=1`: GPU time per kernel name per decode step
/// (each dispatch timed alone, so the sum exceeds a real step).
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

fn bench(
    path: &PathBuf,
    text: &PathBuf,
    prompt_len: usize,
    n: usize,
    runs: usize,
) -> Result<()> {
    let tok = lily::tokenizer::Tokenizer::from_gguf(path)?;
    let text = std::fs::read_to_string(text)?;
    let mut ids = tok.encode("[gMASK]<sop>")?;
    let body = tok.encode(&text)?;
    anyhow::ensure!(
        body.len() + ids.len() >= prompt_len,
        "text has {} tokens",
        body.len()
    );
    ids.extend_from_slice(&body[..prompt_len - ids.len()]);
    let ctx = MetalContext::new()?;
    let model = Glm53Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let mut sc = model.new_scratch(&ctx)?;
    for run in 0..runs {
        let mut st = model.new_state(&ctx, prompt_len + n + 8)?;
        let _ = lily::metal::profile::take();
        let t = Instant::now();
        model.prefill(&ctx, &mut st, &mut sc, &ids)?;
        let pf = t.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("prefill of {prompt_len} tokens, per prefill:");
            profile_report(1);
        }
        let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
        let t = Instant::now();
        let mut steps = Vec::with_capacity(n);
        let mut digest = 0xcbf29ce484222325u64;
        for _ in 0..n {
            let s = Instant::now();
            model.forward(&ctx, &mut st, &mut sc, &[next], true)?;
            steps.push(s.elapsed().as_secs_f64() * 1e3);
            next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
            digest = (digest ^ u64::from(next)).wrapping_mul(0x100000001b3);
        }
        let dt = t.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("decode at prompt {prompt_len}, per step:");
            profile_report(n);
        }
        steps.sort_by(f64::total_cmp);
        println!(
            "run {run}: prompt {prompt_len}: prefill {pf:.2} s = {:.1} tok/s; decode {n} tokens {dt:.2} s = {:.2} tok/s; step median {:.2} ms p10 {:.2} ms; digest {digest:016x}",
            prompt_len as f64 / pf,
            n as f64 / dt,
            steps[n / 2],
            steps[n / 10],
        );
    }
    Ok(())
}

fn rows(
    path: &std::path::Path,
    text: &std::path::Path,
    prompt_len: usize,
    n: usize,
) -> Result<()> {
    use lily::engine::DecodeStateApi;
    anyhow::ensure!(n > 0, "--n must be at least 1");
    let tok = lily::tokenizer::Tokenizer::from_gguf(path)?;
    let mut ids = tok.encode("[gMASK]<sop>")?;
    let body = tok.encode(&std::fs::read_to_string(text)?)?;
    anyhow::ensure!(
        prompt_len > ids.len(),
        "--prompt-len must exceed the {}-token prefix",
        ids.len()
    );
    let used = prompt_len - ids.len();
    anyhow::ensure!(
        body.len() >= used + 4,
        "the text has {} tokens, {} needed",
        body.len(),
        used + 4
    );
    ids.extend_from_slice(&body[..used]);
    let ctx = MetalContext::new()?;
    let model = Glm53Model::load(&ctx, path)?;
    let mut sc = model.new_scratch(&ctx)?;
    let mut st = model.new_state(&ctx, prompt_len + 16)?;
    model.prefill(&ctx, &mut st, &mut sc, &ids)?;
    // Every timed pass starts from the prompt's state: the recurrent parts
    // (KDA, indexer tails) restored, the caches rewritten from the position.
    let snap = st.snapshot(&ctx)?;
    let mut line = format!("rows: prompt {prompt_len}:");
    for m in [1usize, 2, 3, 4, 1] {
        let toks: Vec<u32> = body[used..used + m].to_vec();
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            st.restore(&ctx, &snap)?;
            let s = Instant::now();
            if m == 1 {
                model.forward(&ctx, &mut st, &mut sc, &toks, true)?;
            } else {
                model.forward_all_logits(&ctx, &mut st, &mut sc, &toks)?;
            }
            t.push(s.elapsed().as_secs_f64() * 1e3);
        }
        t.sort_by(f64::total_cmp);
        line += &format!(" m={m} {:.2} ms (p10 {:.2})", t[n / 2], t[n / 10]);
    }
    println!("{line}");
    Ok(())
}

/// The prompt of `spec` and `spec-diff`: a chat question, or raw text.
fn prompt_ids(
    tok: &lily::tokenizer::Tokenizer,
    text: &PathBuf,
    prompt_len: usize,
    chat: Option<&str>,
) -> Result<Vec<u32>> {
    Ok(match chat {
        Some(q) => {
            let messages = [serde_json::json!({"role": "user", "content": q})];
            tok.encode_chat(&lily::tokenizer::ChatRender {
                messages: &messages,
                tools: None,
                enable_thinking: false,
                reasoning_effort: None,
                preserve_thinking: None,
            })?
        }
        None => {
            let mut ids = tok.encode("[gMASK]<sop>")?;
            let body = tok.encode(&std::fs::read_to_string(text)?)?;
            ids.extend_from_slice(&body[..prompt_len - ids.len()]);
            ids
        }
    })
}

/// Relative L2 distance of `a` from `b`.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (f64::from(x) - f64::from(y)).powi(2);
        n += f64::from(y).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

/// The top token and its margin over the second.
fn top2(l: &[f32]) -> (usize, f32) {
    let t = argmax(l);
    let second = l
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != t)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    (t, l[t] - second)
}

fn spec_diff(
    path: &PathBuf,
    text: &PathBuf,
    prompt_len: usize,
    n: usize,
    chat: Option<&str>,
) -> Result<()> {
    use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions};
    use lily::glm53::model::{LayerState, StepLayer};
    use lily::kernels::sample::SamplingParams;
    let tok = lily::tokenizer::Tokenizer::from_gguf(path)?;
    let ids = prompt_ids(&tok, text, prompt_len, chat)?;
    let ctx = MetalContext::new()?;
    let options = LoadOptions { mtp_drafts: 1, ..Default::default() };
    let model = <Glm53Model as LanguageModel>::load(&ctx, path, &options)?;
    let cfg = model.cfg.clone();
    let (vocab, hd, k) = (cfg.vocab, cfg.hidden, cfg.n_expert_used);
    let greedy = SamplingParams::greedy();
    let mut sc = LanguageModel::new_scratch_with_capacity(&model, &ctx, 0)?;
    let cap = ids.len() + n + 8;
    let fresh =
        |sc: &mut lily::glm53::model::Scratch| -> Result<lily::glm53::Glm53State> {
            let mut st = LanguageModel::new_state(&model, &ctx, cap)?;
            LanguageModel::prefill(
                &model,
                &ctx,
                &mut st,
                sc,
                &ids,
                Some(Draw { params: &greedy, step: 0 }),
            )?;
            Ok(st)
        };
    // Routing of every fed row, by state position: per layer the sorted
    // experts, the top-8 margin and the FFN input.
    type Rec = Vec<(Vec<u32>, f32, Vec<f32>)>;
    let rec = |l: &[StepLayer], r: usize| -> Rec {
        l.iter()
            .map(|x| {
                let mut sel = if x.sel.is_empty() {
                    vec![]
                } else {
                    x.sel[r * k..(r + 1) * k].to_vec()
                };
                sel.sort();
                let m = x.margin.get(r).copied().unwrap_or(f32::NAN);
                (sel, m, x.ffn_in[r * hd..(r + 1) * hd].to_vec())
            })
            .collect()
    };
    let mut plain_rec: std::collections::BTreeMap<usize, Rec> = Default::default();
    let mut spec_rec: std::collections::BTreeMap<usize, Rec> = Default::default();
    // Plain greedy decoding (the production kernels, a pass a layer).
    let mut st = fresh(&mut sc)?;
    let mut plain = vec![sc.next_token.to_u32()?[0]];
    while plain.len() < n {
        let p = st.pos;
        let (l, lg) =
            model.step_layers(&ctx, &mut st, &mut sc, &[*plain.last().unwrap()])?;
        plain_rec.insert(p, rec(&l, 0));
        plain.push(argmax(&lg[..vocab]) as u32);
    }
    // Speculative decoding through the engine; each verify pass is also
    // replayed a layer at a time first (from a snapshot) for its routing.
    // A second run stops before the verify pass that draws the first token
    // that differs.
    let mut stop_at: Option<usize> = None;
    let (st, pending, feed, row, d) = loop {
        let mut st = fresh(&mut sc)?;
        let mut out = vec![sc.next_token.to_u32()?[0]];
        let mut proposals =
            model.draft_initial(&ctx, &mut st, &mut sc, out[0], 1, &greedy, 1)?;
        let mut found = None;
        while out.len() < n {
            let pending = *out.last().unwrap();
            let l = out.len();
            let feed: Vec<u32> =
                std::iter::once(pending).chain(proposals.iter().copied()).collect();
            if let Some(d) = stop_at
                && d < l + feed.len()
            {
                found = Some((pending, feed, d - l, d));
                break;
            }
            let pos0 = st.pos;
            let layers = if stop_at.is_none() {
                let snap = st.snapshot(&ctx)?;
                let (layers, _) = model.step_layers(&ctx, &mut st, &mut sc, &feed)?;
                st.restore(&ctx, &snap)?;
                Some(layers)
            } else {
                None
            };
            let (draws, pend) = model
                .verify(&ctx, &mut st, &mut sc, pending, &proposals, &greedy, l, 1)?;
            let kept = proposals.first().map_or(0, |&p| usize::from(draws[0] == p));
            if let Some(layers) = layers {
                for r in 0..=kept {
                    spec_rec.insert(pos0 + r, rec(&layers, r));
                }
            }
            out.extend_from_slice(&draws[..=kept]);
            proposals =
                model.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pend)?;
        }
        if let Some((pending, feed, row, d)) = found {
            break (st, pending, feed, row, d);
        }
        out.truncate(n);
        match plain.iter().zip(&out).position(|(a, b)| a != b) {
            Some(d) if stop_at.is_none() => {
                println!("speculative output parts from plain at token {d} of {n}");
                stop_at = Some(d);
            }
            _ => {
                println!("speculative output equals plain decoding for {n} tokens");
                return Ok(());
            }
        }
    };
    // Where the two runs first route a row differently.
    let mut before = 0f64;
    let mut first = None;
    'pos: for (&p, pr) in &plain_rec {
        let Some(srec) = spec_rec.get(&p) else { continue };
        for (il, ((s1, m1, x1), (s2, m2, x2))) in pr.iter().zip(srec).enumerate() {
            let r = rel_l2(x2, x1);
            if s1 != s2 {
                let gone: Vec<u32> =
                    s1.iter().filter(|e| !s2.contains(e)).copied().collect();
                let came: Vec<u32> =
                    s2.iter().filter(|e| !s1.contains(e)).copied().collect();
                println!(
                    "first routing difference: position {p} (token {}), layer {il}: plain {gone:?} -> speculative {came:?}; top-8 margin plain {m1:.3e}, speculative {m2:.3e}; FFN input relative L2 {r:.2e}; largest FFN-input difference at all earlier rows and layers {before:.2e}",
                    p - ids.len()
                );
                first = Some(p);
                break 'pos;
            }
            before = before.max(r);
        }
    }
    if first.is_none() {
        println!(
            "no routing difference before the parting; largest FFN-input difference {before:.2e}"
        );
    }
    let small: Vec<(usize, usize, f32)> = plain_rec
        .iter()
        .flat_map(|(&p, r)| {
            r.iter()
                .enumerate()
                .filter(|(_, x)| x.1 < 1e-5)
                .map(move |(il, x)| (p, il, x.1))
        })
        .collect();
    println!("plain rows with a top-8 margin below 1e-5: {small:?}");
    println!(
        "verify pass at position {} feeds {feed:?} (pending {pending}); its row {row} draws token {d}",
        st.pos
    );
    let snap = st.snapshot(&ctx)?;
    let mut st = st;
    // (a) The verify pass; (b) one-row steps on the same state.
    let (two, two_logits) = model.step_layers(&ctx, &mut st, &mut sc, &feed)?;
    st.restore(&ctx, &snap)?;
    let mut one = Vec::new();
    let mut one_logits = Vec::new();
    for &t in &feed[..=row] {
        let (l, lg) = model.step_layers(&ctx, &mut st, &mut sc, &[t])?;
        one = l;
        one_logits = lg;
    }
    st.restore(&ctx, &snap)?;
    // (c) Plain decoding's own state at the same position.
    let mut ps = fresh(&mut sc)?;
    let base = ps.pos;
    for &t in &plain[..st.pos - base] {
        model.forward(&ctx, &mut ps, &mut sc, &[t], true)?;
    }
    anyhow::ensure!(ps.pos == st.pos, "plain state at {} not {}", ps.pos, st.pos);
    // Recurrent state and caches, speculative against plain.
    let mut worst_kda = (0.0f64, 0usize);
    let mut worst_lat = (0.0f64, 0usize);
    // The trunk's layers (the MTP block's cache is the MTP head's own).
    for (il, (a, b)) in st.layers.iter().zip(&ps.layers).enumerate().take(cfg.n_trunk) {
        match (a, b) {
            (
                LayerState::Kda { s: sa, conv: ca, .. },
                LayerState::Kda { s: sb, conv: cb, .. },
            ) => {
                let r = rel_l2(&sa.to_f32()?, &sb.to_f32()?)
                    .max(rel_l2(&ca.to_f32()?, &cb.to_f32()?));
                if r > worst_kda.0 {
                    worst_kda = (r, il);
                }
            }
            (
                LayerState::Dsa { latent: la, .. },
                LayerState::Dsa { latent: lb, .. },
            ) => {
                let rows = st.pos * cfg.kv_lora;
                let r = rel_l2(&la.to_f32()?[..rows], &lb.to_f32()?[..rows]);
                if r > worst_lat.0 {
                    worst_lat = (r, il);
                }
            }
            _ => anyhow::bail!("layer kinds differ"),
        }
    }
    println!(
        "state, speculative vs plain: KDA worst relative L2 {:.2e} (layer {}), DSA latent cache worst {:.2e} (layer {})",
        worst_kda.0, worst_kda.1, worst_lat.0, worst_lat.1
    );
    let mut plain_layers = Vec::new();
    let mut plain_logits = Vec::new();
    for &t in &feed[..=row] {
        let (l, lg) = model.step_layers(&ctx, &mut ps, &mut sc, &[t])?;
        plain_layers = l;
        plain_logits = lg;
    }
    let pick = |l: &StepLayer, r: usize| -> (Vec<f32>, Vec<f32>, Vec<u32>, f32) {
        let h = |v: &Vec<f32>| v[r * hd..(r + 1) * hd].to_vec();
        let mut sel =
            if l.sel.is_empty() { vec![] } else { l.sel[r * k..(r + 1) * k].to_vec() };
        sel.sort();
        (
            h(&l.attn_out),
            h(&l.ffn_out),
            sel,
            l.margin.get(r).copied().unwrap_or(f32::NAN),
        )
    };
    println!(
        "layer  verify-vs-one-row: attn_out ffn_out experts   one-row-vs-plain: attn_out ffn_out experts   margins (verify / one-row / plain)"
    );
    for il in 0..cfg.n_trunk {
        let (a2, f2, s2, m2) = pick(&two[il], row);
        let (a1, f1, s1, m1) = pick(&one[il], 0);
        let (ap, fp, sp, mp) = pick(&plain_layers[il], 0);
        let diff = |x: &[u32], y: &[u32]| x.iter().filter(|e| !y.contains(e)).count();
        println!(
            "{il:5}  {:.2e} {:.2e} {:>2}   {:.2e} {:.2e} {:>2}   {m2:.2e} / {m1:.2e} / {mp:.2e}",
            rel_l2(&a2, &a1),
            rel_l2(&f2, &f1),
            diff(&s2, &s1),
            rel_l2(&a1, &ap),
            rel_l2(&f1, &fp),
            diff(&s1, &sp),
        );
    }
    let lr = |l: &[f32], r: usize| l[r * vocab..(r + 1) * vocab].to_vec();
    for (name, l) in [
        ("verify row", lr(&two_logits, row)),
        ("one-row, same state", lr(&one_logits, 0)),
        ("plain state", lr(&plain_logits, 0)),
    ] {
        let (t, m) = top2(&l);
        println!("{name:22} top token {t} margin {m:.4}");
    }
    println!(
        "logits relative L2: verify vs one-row {:.2e}, one-row vs plain {:.2e}",
        rel_l2(&lr(&two_logits, row), &lr(&one_logits, 0)),
        rel_l2(&lr(&one_logits, 0), &lr(&plain_logits, 0))
    );
    println!("plain token {d}: {}", plain[d]);
    Ok(())
}

fn spec(
    path: &PathBuf,
    text: &PathBuf,
    prompt_len: usize,
    n: usize,
    runs: usize,
    chat: Option<&str>,
) -> Result<()> {
    use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions};
    use lily::kernels::sample::SamplingParams;
    let tok = lily::tokenizer::Tokenizer::from_gguf(path)?;
    let ids = prompt_ids(&tok, text, prompt_len, chat)?;
    let ctx = MetalContext::new()?;
    let options = LoadOptions { mtp_drafts: 1, ..Default::default() };
    let model = <Glm53Model as LanguageModel>::load(&ctx, path, &options)?;
    let greedy = SamplingParams::greedy();
    let mut sc = LanguageModel::new_scratch_with_capacity(&model, &ctx, 0)?;
    for run in 0..runs {
        // Plain greedy reference.
        let mut st = LanguageModel::new_state(&model, &ctx, ids.len() + n + 8)?;
        LanguageModel::prefill(
            &model,
            &ctx,
            &mut st,
            &mut sc,
            &ids,
            Some(Draw { params: &greedy, step: 0 }),
        )?;
        let mut plain = vec![sc.next_token.to_u32()?[0]];
        // Plain decoding's top-1 minus top-2 logit at each token.
        let mut margins = vec![f32::NAN];
        let _ = lily::metal::profile::take();
        let t = Instant::now();
        while plain.len() < n {
            let last = *plain.last().unwrap();
            model.forward(&ctx, &mut st, &mut sc, &[last], true)?;
            let l = &sc.logits.to_f32()?[..model.cfg.vocab];
            let top = argmax(l);
            let second = l
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != top)
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max);
            margins.push(l[top] - second);
            plain.push(top as u32);
        }
        let plain_s = t.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("profile: plain decode, per step:");
            profile_report(n - 1);
        }
        // Speculative.
        let mut st = LanguageModel::new_state(&model, &ctx, ids.len() + n + 8)?;
        LanguageModel::prefill(
            &model,
            &ctx,
            &mut st,
            &mut sc,
            &ids,
            Some(Draw { params: &greedy, step: 0 }),
        )?;
        let mut out = vec![sc.next_token.to_u32()?[0]];
        let _ = lily::metal::profile::take();
        let t = Instant::now();
        let mut proposals = model.draft_initial(
            &ctx,
            &mut st,
            &mut sc,
            out[0],
            1,
            &greedy,
            out.len(),
        )?;
        let (mut steps, mut accepted, mut proposed) = (0usize, 0usize, 0usize);
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
                1,
            )?;
            // No proposal (an unlikely draft): a one-row step.
            let kept = proposals.first().map_or(0, |&d| usize::from(draws[0] == d));
            out.extend_from_slice(&draws[..=kept]);
            accepted += kept;
            proposed += proposals.len();
            steps += 1;
            let tf = Instant::now();
            t_verify += (tf - tv).as_secs_f64();
            proposals =
                model.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pend)?;
            t_finish += tf.elapsed().as_secs_f64();
        }
        let spec_s = t.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("profile: speculative, per step:");
            profile_report(steps);
        }
        out.truncate(n);
        let same = plain.iter().zip(&out).take_while(|(a, b)| a == b).count();
        if same < n {
            println!(
                "first difference at token {same}: plain's top-2 margin there {:.4}, smallest margin before it {:.4}",
                margins[same],
                margins[1..same].iter().copied().fold(f32::INFINITY, f32::min)
            );
        }
        println!(
            "run {run}: plain {:.2} tok/s; speculative {:.2} tok/s, {steps} steps, accepted {accepted}/{proposed} proposed ({:.0}%); same first {same}/{n} tokens; pos {}; per step: verify {:.2} ms, finish {:.2} ms; plain step {:.2} ms",
            n as f64 / plain_s,
            n as f64 / spec_s,
            100.0 * accepted as f64 / proposed.max(1) as f64,
            st.pos(),
            t_verify / steps as f64 * 1e3,
            t_finish / steps as f64 * 1e3,
            plain_s / (n - 1) as f64 * 1e3,
        );
        if run == 0 {
            println!("speculative text: {:?}", tok.decode(&out, false)?);
        }
    }
    Ok(())
}
