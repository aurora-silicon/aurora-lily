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
        /// Token id files; `split` applies to each unless `--split-from`.
        #[arg(long, num_args = 1..)]
        ids: Vec<PathBuf>,
        #[arg(long)]
        split: usize,
        /// Write every token's line to `<file>.lily` instead of stdout.
        #[arg(long)]
        quiet: bool,
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
    }
}

fn generate(path: &PathBuf, prompt: &[u32], n: usize) -> Result<()> {
    let ctx = MetalContext::new()?;
    let t0 = Instant::now();
    let model = Qwen36Model::load(&ctx, path)?;
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let cap = prompt.len() + n + 8;
    let mut st = model.new_state(&ctx, cap)?;
    let mut sc = model.new_scratch(&ctx, cap)?;
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

fn nll(path: &PathBuf, files: &[PathBuf], split: usize, quiet: bool) -> Result<()> {
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
    let mut sc = model.new_scratch(&ctx, max_len + 8)?;
    let (mut all_nll, mut all_n, mut all_hits) = (0f64, 0usize, 0usize);
    for f in files {
        let ids = read_ids(f)?;
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
