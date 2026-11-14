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
        #[arg(long)]
        tokens: String,
        /// Layers to compare (the CPU reference is slow).
        #[arg(long, default_value_t = 4)]
        layers: usize,
        /// Rows a GPU run takes (the prompt is fed in runs of this many).
        #[arg(long, default_value_t = 256)]
        step: usize,
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
        Cmd::Compare { model, tokens, layers, step } => {
            compare(&model, &parse_tokens(&tokens)?, layers, step)
        }
        Cmd::Generate { model, tokens, n } => {
            generate(&model, &parse_tokens(&tokens)?, n)
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

fn compare(path: &PathBuf, tokens: &[u32], layers: usize, step: usize) -> Result<()> {
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
        for (g, t) in gpu.iter_mut().zip(tr) {
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
    println!(
        "layer ratio  attn_in        q              kv             heads          attn_out       ffn_in         ffn_out        h   (rel L2 / max abs)"
    );
    for il in 0..layers.min(cfg.n_layer) {
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
