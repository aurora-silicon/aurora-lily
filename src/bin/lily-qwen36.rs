//! Qwen3.6 bring-up tool: greedy generation and teacher-forced scoring on
//! the GPU, in the same formats as the llama.cpp reference tool
//! (`q36ref`), so the two can be compared token by token.
//!
//! ```sh
//! lily-qwen36 generate --model Qwen3.6-35B-A3B-Q4_K_M.gguf --ids prompt.txt --n 32
//! lily-qwen36 nll --model Qwen3.6-35B-A3B-Q4_K_M.gguf --ids case.txt --split 900
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use lily::metal::MetalContext;
use lily::qwen36::{Logits, Qwen36Model};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Greedy continuation of the token ids in a file (whitespace-separated).
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        ids: PathBuf,
        #[arg(long, default_value_t = 32)]
        n: usize,
    },
    /// Per-token NLL of ids[split..] given ids[..split]: "i target nll
    /// argmax margin" lines and a mean line (q36ref's format). With several
    /// files, a summary line each.
    Nll {
        #[arg(long)]
        model: PathBuf,
        /// Token id files.
        #[arg(long, num_args = 1..)]
        ids: Vec<PathBuf>,
        /// Prompt length; without it each file's `<name>.split` gives it.
        #[arg(long)]
        split: Option<usize>,
        /// Write every token's line to `<file>.lily` instead of stdout.
        #[arg(long)]
        quiet: bool,
    },
    /// Token ids of text files (the GGUF's tokenizer, special tokens
    /// parsed), one line each.
    Tok {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, num_args = 1..)]
        text: Vec<PathBuf>,
    },
    /// Greedy speculative decoding with the MTP head through the engine
    /// interface, against plain greedy decoding of the same prompt (the
    /// first `prompt_len` ids of a file).
    Spec {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        ids: PathBuf,
        #[arg(long, default_value_t = 1024)]
        prompt_len: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 2)]
        runs: usize,
    },
    /// Speed: prefill the first `prompt_len` ids of a file, then decode `n`
    /// tokens greedily; `runs` times, each on a fresh state.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        ids: PathBuf,
        #[arg(long, default_value_t = 1024)]
        prompt_len: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
    },
}

fn read_ids(path: &PathBuf) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    s.split_whitespace().map(|t| t.parse::<u32>().context("token id")).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Generate { model, ids, n } => generate(&model, &read_ids(&ids)?, n),
        Cmd::Nll { model, ids, split, quiet } => nll(&model, &ids, split, quiet),
        Cmd::Tok { model, text } => {
            let tok = lily::tokenizer::Tokenizer::from_gguf(&model)?;
            for f in text {
                let ids = tok.encode(&std::fs::read_to_string(&f)?)?;
                let s: Vec<String> = ids.iter().map(u32::to_string).collect();
                println!("{}", s.join(" "));
            }
            Ok(())
        }
        Cmd::Spec { model, ids, prompt_len, n, runs } => {
            spec(&model, &read_ids(&ids)?, prompt_len, n, runs)
        }
        Cmd::Bench { model, ids, prompt_len, n, runs } => {
            bench(&model, &read_ids(&ids)?, prompt_len, n, runs)
        }
    }
}

fn generate(path: &PathBuf, prompt: &[u32], n: usize) -> Result<()> {
    let ctx = MetalContext::new()?;
    let t0 = Instant::now();
    let model = Qwen36Model::load(&ctx, path)?;
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let cap = prompt.len() + n + 8;
    let mut st = model.new_state(&ctx, cap)?;
    let mut sc = model.new_scratch(&ctx)?;
    let t0 = Instant::now();
    model.prefill(&ctx, &mut st, &mut sc, prompt)?;
    let pf = t0.elapsed().as_secs_f64();
    let mut out = Vec::with_capacity(n);
    let t0 = Instant::now();
    for i in 0..n {
        let logits = sc.logits_last.to_f32()?;
        let next = argmax(&logits) as u32;
        out.push(next);
        if i + 1 < n {
            model.forward(&ctx, &mut st, &mut sc, &[next], Logits::Last)?;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let s: Vec<String> = out.iter().map(u32::to_string).collect();
    println!("{}", s.join(" "));
    eprintln!(
        "prefill {} tokens {:.2}s ({:.0} tok/s); decode {} tokens {:.2}s ({:.1} tok/s)",
        prompt.len(),
        pf,
        prompt.len() as f64 / pf,
        n,
        dt,
        (n.max(2) - 1) as f64 / dt
    );
    Ok(())
}

fn log_softmax_at(logits: &[f32], target: usize) -> f64 {
    let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let z: f64 = logits.iter().map(|&x| (x as f64 - mx).exp()).sum();
    logits[target] as f64 - mx - z.ln()
}

fn nll(path: &PathBuf, files: &[PathBuf], split: Option<usize>, quiet: bool) -> Result<()> {
    use std::io::Write as _;
    let ctx = MetalContext::new()?;
    let model = Qwen36Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let max_len = files
        .iter()
        .map(|f| read_ids(f).map(|v| v.len()))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let mut st = model.new_state(&ctx, max_len + 8)?;
    let mut sc = model.new_scratch(&ctx)?;
    let (mut all_nll, mut all_n, mut all_hits) = (0f64, 0usize, 0usize);
    for f in files {
        let ids = read_ids(f)?;
        let split = match split {
            Some(s) => s,
            None => std::fs::read_to_string(f.with_extension("split"))?.trim().parse()?,
        };
        anyhow::ensure!(split >= 1 && split < ids.len(), "split {split} of {}", ids.len());
        st.reset();
        let t0 = Instant::now();
        model.prefill(&ctx, &mut st, &mut sc, &ids[..split])?;
        let pf = t0.elapsed().as_secs_f64();
        let mut lines = String::new();
        let (mut tot, mut hits) = (0f64, 0usize);
        // Row i's logits predict ids[split + i]; the prompt's last row is in
        // logits_last, then the continuation is fed in steps of up to four
        // rows, every row's logits.
        let mut pending: Vec<f32> = sc.logits_last.to_f32()?[..vocab].to_vec();
        let mut i = split;
        while i < ids.len() {
            let mut rows = vec![pending.clone()];
            let take = (ids.len() - i).min(lily::qwen36::model::MAX_ROWS);
            // Feed ids[i..i+take): their logits predict ids[i+1..].
            model.forward(&ctx, &mut st, &mut sc, &ids[i..i + take], Logits::All)?;
            let all = sc.logits.to_f32()?;
            for r in 0..take {
                rows.push(all[r * vocab..(r + 1) * vocab].to_vec());
            }
            for (r, l) in rows.iter().take(take).enumerate() {
                let target = ids[i + r] as usize;
                let lp = log_softmax_at(l, target);
                let am = argmax(l);
                let mut sorted: Vec<f32> = vec![l[am], f32::NEG_INFINITY];
                for (j, &x) in l.iter().enumerate() {
                    if j != am && x > sorted[1] {
                        sorted[1] = x;
                    }
                }
                tot -= lp;
                hits += usize::from(am == target);
                lines.push_str(&format!(
                    "{} {} {:.6} {} {:.5}\n",
                    i + r - split,
                    target,
                    -lp,
                    am,
                    sorted[0] - sorted[1]
                ));
            }
            pending = rows.pop().expect("rows");
            i += take;
        }
        let n = ids.len() - split;
        lines.push_str(&format!("mean {:.6} n {} hits {}\n", tot / n as f64, n, hits));
        if quiet {
            let mut out = std::fs::File::create(f.with_extension("lily"))?;
            out.write_all(lines.as_bytes())?;
        } else {
            print!("{lines}");
        }
        println!(
            "{} mean {:.6} n {} hits {} prefill {:.2}s",
            f.display(),
            tot / n as f64,
            n,
            hits,
            pf
        );
        all_nll += tot;
        all_n += n;
        all_hits += hits;
    }
    println!("total mean {:.6} n {} hits {}", all_nll / all_n as f64, all_n, all_hits);
    Ok(())
}

fn bench(path: &PathBuf, ids: &[u32], prompt_len: usize, n: usize, runs: usize) -> Result<()> {
    anyhow::ensure!(ids.len() >= prompt_len, "{} ids for a prompt of {prompt_len}", ids.len());
    let ctx = MetalContext::new()?;
    let model = Qwen36Model::load(&ctx, path)?;
    let cap = prompt_len + n + 8;
    let mut st = model.new_state(&ctx, cap)?;
    let mut sc = model.new_scratch(&ctx)?;
    for run in 0..runs {
        st.reset();
        let _ = lily::metal::profile::take();
        let t0 = Instant::now();
        model.prefill(&ctx, &mut st, &mut sc, &ids[..prompt_len])?;
        let pf = t0.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("prefill of {prompt_len} tokens:");
            profile_report(1);
        }
        let mut next = argmax(&sc.logits_last.to_f32()?) as u32;
        let mut out = vec![next];
        let t0 = Instant::now();
        for _ in 1..n {
            model.forward(&ctx, &mut st, &mut sc, &[next], Logits::Last)?;
            next = argmax(&sc.logits_last.to_f32()?) as u32;
            out.push(next);
        }
        let dt = t0.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("decode at prompt {prompt_len}, per step:");
            profile_report(n - 1);
        }
        let digest = out.iter().fold(0xcbf29ce484222325u64, |h, &t| {
            (h ^ u64::from(t)).wrapping_mul(0x100000001b3)
        });
        println!(
            "run {run}: prefill {prompt_len} in {:.3}s = {:.1} tok/s; decode {} steps {:.2} ms/step = {:.1} tok/s; digest {digest:016x}",
            pf,
            prompt_len as f64 / pf,
            n - 1,
            dt * 1e3 / (n - 1) as f64,
            (n - 1) as f64 / dt
        );
    }
    Ok(())
}

/// The kernel profile (`LILY_KERNEL_PROFILE=1`) since the last take, per
/// step of `steps`.
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

fn spec(path: &PathBuf, ids: &[u32], prompt_len: usize, n: usize, runs: usize) -> Result<()> {
    use lily::engine::{Draw, LanguageModel, LoadOptions};
    use lily::kernels::sample::SamplingParams;
    anyhow::ensure!(ids.len() >= prompt_len, "{} ids for a prompt of {prompt_len}", ids.len());
    let ctx = MetalContext::new()?;
    let opts = LoadOptions { mtp_drafts: 1, ..Default::default() };
    let model = <Qwen36Model as LanguageModel>::load(&ctx, path, &opts)?;
    anyhow::ensure!(model.mtp.is_some(), "no MTP head found");
    let cap = prompt_len + n + 16;
    let greedy = SamplingParams::greedy();
    let mut st = LanguageModel::new_state(&model, &ctx, cap)?;
    let mut sc = model.new_scratch_with_capacity(&ctx, cap)?;
    let prompt = &ids[..prompt_len];
    // Plain greedy reference through the same prefill.
    st.reset();
    LanguageModel::prefill(&model, &ctx, &mut st, &mut sc, prompt, Some(Draw { params: &greedy, step: 0 }))?;
    let mut plain = vec![sc.next_token.to_u32()?[0]];
    // Top-2 logit margin of each plain draw past the first.
    let mut margins = vec![f32::NAN];
    let t0 = Instant::now();
    while plain.len() < n {
        model.forward(&ctx, &mut st, &mut sc, &[*plain.last().expect("one")], Logits::Last)?;
        let l = sc.logits_last.to_f32()?;
        let a = argmax(&l);
        let second = l.iter().enumerate().filter(|&(i, _)| i != a).fold(f32::NEG_INFINITY, |m, (_, &x)| m.max(x));
        margins.push(l[a] - second);
        plain.push(a as u32);
    }
    let plain_dt = t0.elapsed().as_secs_f64();
    for run in 0..runs {
        st.reset();
        LanguageModel::prefill(&model, &ctx, &mut st, &mut sc, prompt, Some(Draw { params: &greedy, step: 0 }))?;
        let mut out = vec![sc.next_token.to_u32()?[0]];
        let t0 = Instant::now();
        let (mut drafted, mut accepted, mut steps) = (0usize, 0usize, 0usize);
        let mut proposals = model.draft_initial(&ctx, &mut st, &mut sc, out[0], 1, &greedy, 1)?;
        while out.len() < n {
            let pending = *out.last().expect("one");
            let (draws, pass) =
                model.verify(&ctx, &mut st, &mut sc, pending, &proposals, &greedy, out.len(), 1)?;
            let mut kept = 0;
            for (j, &t) in draws.iter().enumerate() {
                out.push(t);
                kept = j;
                if j >= proposals.len() || proposals[j] != t {
                    break;
                }
            }
            drafted += proposals.len();
            accepted += kept;
            steps += 1;
            proposals = model.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pass)?;
        }
        let dt = t0.elapsed().as_secs_f64();
        out.truncate(n);
        let same = out.iter().zip(&plain).take_while(|(a, b)| a == b).count();
        println!(
            "run {run}: spec {} tokens {:.2}s = {:.1} tok/s ({steps} steps, accepted {accepted}/{drafted} = {:.0}%); plain {:.1} tok/s; same tokens for {same}/{n}{}",
            n - 1,
            dt,
            (n - 1) as f64 / dt,
            100.0 * accepted as f64 / drafted.max(1) as f64,
            (n - 1) as f64 / plain_dt,
            if same < n {
                format!(" (plain's top-2 margin there {:.4})", margins[same])
            } else {
                String::new()
            },
        );
    }
    Ok(())
}
