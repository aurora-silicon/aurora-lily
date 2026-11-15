//! Gemma 4 bring-up tool: teacher-forced scoring of token-id cases (the
//! format of `tools/reference/gemma4/llama_score.c`, so lily and llama.cpp
//! score the same tokens with the same GGUF), and greedy generation.
//!
//! ```sh
//! lily-gemma4 score --model gemma-4-26B-A4B-it-UD-Q5_K_M.gguf --cases cases.tsv [--base llama.lp]
//! lily-gemma4 generate --model ... --tokens 2,105,2364 --n 32
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use clap::{Parser, Subcommand};
use half::f16;
use lily::gemma4::model::{MAX_ROWS, PREFILL_ROWS};
use lily::gemma4::{Gemma4Model, Logits};
use lily::metal::MetalContext;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Teacher-forced scoring: per case (`id<TAB>first<TAB>ids`) the mean
    /// NLL of tokens first.. and the greedy agreement; with --base (a
    /// llama_score --dump of the same cases) the KL divergence from it and
    /// the top-1 agreement with it; --dump writes this run's log-softmax
    /// rows in the same format.
    Score {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        cases: PathBuf,
        #[arg(long)]
        base: Option<PathBuf>,
        #[arg(long)]
        dump: Option<PathBuf>,
        /// Score only the first N cases.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Chat: renders one user message with the GGUF's template (lily's
    /// tokenizer), prints its token ids, and decodes greedily.
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
    /// State check on real weights: after a prompt of `n` tokens (from the
    /// cases file's first case, repeated), a snapshot; greedy steps; then
    /// back to the snapshot (and, on a second state, the global caches'
    /// prefix copied plus the snapshot), the same steps again. Every step's
    /// logits must be bit-identical to the first run's.
    StateCheck {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        cases: PathBuf,
        #[arg(long, default_value_t = 3000)]
        n: usize,
        #[arg(long, default_value_t = 24)]
        steps: usize,
    },
    /// GPU memory of a configuration: the loaded weights, the scratch and
    /// one session for `ctx` tokens with `checkpoints` snapshots, exactly
    /// (the backend's allocation counter).
    Mem {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        #[arg(long, default_value_t = 1)]
        checkpoints: usize,
    },
    /// Speed: a user turn holding the first tokens of a text file, cut to
    /// `prompt_len` tokens and prefilled, then `n` greedy decode steps;
    /// `runs` times on fresh states. LILY_KERNEL_PROFILE=1 adds the
    /// per-kernel tables.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 1024)]
        prompt_len: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 2)]
        runs: usize,
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

fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    lily::instance::acquire()?;
    match cli.cmd {
        Cmd::Score { model, cases, base, dump, limit } => {
            score(&model, &cases, base.as_deref(), dump.as_deref(), limit)
        }
        Cmd::Generate { model, tokens, n } => {
            generate(&model, &parse_tokens(&tokens)?, n)
        }
        Cmd::Chat { model, prompt, n, think } => chat(&model, &prompt, n, think),
        Cmd::StateCheck { model, cases, n, steps } => {
            state_check(&model, &cases, n, steps)
        }
        Cmd::Mem { model, ctx, checkpoints } => mem(&model, ctx, checkpoints),
        Cmd::Bench { model, text, prompt_len, n, runs } => {
            bench(&model, &text, prompt_len, n, runs)
        }
    }
}

fn argmax(x: &[f32]) -> usize {
    let mut b = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[b] {
            b = i;
        }
    }
    b
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = row.iter().map(|&l| f64::from(l - mx).exp()).sum();
    let ls = s.ln();
    row.iter().map(|&l| (f64::from(l - mx) - ls) as f32).collect()
}

fn score(
    path: &PathBuf,
    cases: &PathBuf,
    base: Option<&std::path::Path>,
    dump: Option<&std::path::Path>,
    limit: Option<usize>,
) -> Result<()> {
    use std::io::{Read as _, Write as _};
    let text = std::fs::read_to_string(cases)?;
    let mut list = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        ensure!(cols.len() == 3, "case line of {} columns", cols.len());
        list.push((
            cols[0].to_string(),
            cols[1].parse::<usize>()?,
            parse_tokens(cols[2])?,
        ));
    }
    if let Some(n) = limit {
        list.truncate(n);
    }
    let maxlen = list.iter().map(|c| c.2.len()).max().unwrap_or(1);
    let ctx = MetalContext::new()?;
    let model = Gemma4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let sc = model.new_scratch(&ctx, maxlen + 8)?;
    let mut base =
        base.map(std::fs::File::open).transpose()?.map(std::io::BufReader::new);
    let mut dump =
        dump.map(std::fs::File::create).transpose()?.map(std::io::BufWriter::new);
    let mut hbuf = vec![0u8; 2 * vocab];
    let (mut tn, mut tnll, mut tagree, mut tkld, mut tsame) =
        (0usize, 0f64, 0usize, 0f64, 0usize);
    println!("case\tscored\tnll\tagree\tkld\tsame_top\tprefill_s");
    for (id, first, toks) in &list {
        let (first, n) = (*first, toks.len());
        ensure!(first >= 1 && first < n, "case {id}: first {first} of {n}");
        let mut st = model.new_state(&ctx, n + 8)?;
        let t = Instant::now();
        if first > 1 {
            model.prefill(&ctx, &mut st, &sc, &toks[..first - 1])?;
        }
        let pf = t.elapsed().as_secs_f64();
        let (mut nll, mut agree, mut kld, mut same, mut scored) = (0f64, 0, 0f64, 0, 0);
        for chunk_start in (first - 1..n - 1).step_by(MAX_ROWS) {
            let end = (chunk_start + MAX_ROWS).min(n - 1);
            model.forward(&ctx, &mut st, &sc, &toks[chunk_start..end], Logits::All)?;
            let logits = sc.logits.to_f32()?;
            for (r, p) in (chunk_start..end).enumerate() {
                let lp = log_softmax(&logits[r * vocab..(r + 1) * vocab]);
                let target = toks[p + 1] as usize;
                nll -= f64::from(lp[target]);
                let top = argmax(&lp);
                agree += usize::from(top == target);
                if let Some(d) = dump.as_mut() {
                    for (i, &v) in lp.iter().enumerate() {
                        hbuf[2 * i..2 * i + 2]
                            .copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
                    }
                    d.write_all(&hbuf)?;
                }
                if let Some(b) = base.as_mut() {
                    b.read_exact(&mut hbuf)?;
                    let (mut k, mut btop, mut bmax) = (0f64, 0usize, f32::NEG_INFINITY);
                    for (i, &v) in lp.iter().enumerate() {
                        let bv = f16::from_bits(u16::from_le_bytes([
                            hbuf[2 * i],
                            hbuf[2 * i + 1],
                        ]))
                        .to_f32();
                        if bv > bmax {
                            bmax = bv;
                            btop = i;
                        }
                        if bv > -60.0 {
                            k += f64::from(bv).exp() * f64::from(bv - v);
                        }
                    }
                    kld += k;
                    same += usize::from(btop == top);
                }
                scored += 1;
            }
        }
        println!(
            "{id}\t{scored}\t{:.6}\t{:.2}%\t{:.6}\t{:.2}%\t{pf:.2}",
            nll / scored as f64,
            100.0 * agree as f64 / scored as f64,
            kld / scored as f64,
            100.0 * same as f64 / scored as f64
        );
        tn += scored;
        tnll += nll;
        tagree += agree;
        tkld += kld;
        tsame += same;
    }
    println!(
        "total\t{tn}\t{:.6}\t{:.2}%\t{:.6}\t{:.2}%",
        tnll / tn as f64,
        100.0 * tagree as f64 / tn as f64,
        tkld / tn as f64,
        100.0 * tsame as f64 / tn as f64
    );
    let _ = PREFILL_ROWS;
    Ok(())
}

fn generate(path: &PathBuf, tokens: &[u32], n: usize) -> Result<()> {
    ensure!(!tokens.is_empty(), "no tokens");
    let ctx = MetalContext::new()?;
    let model = Gemma4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let cap = tokens.len() + n + 8;
    let sc = model.new_scratch(&ctx, cap)?;
    let mut st = model.new_state(&ctx, cap)?;
    let t = Instant::now();
    model.prefill(&ctx, &mut st, &sc, tokens)?;
    let pf = t.elapsed().as_secs_f64();
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
    eprintln!(
        "prompt {} tokens in {pf:.2} s; decode {n} tokens in {dt:.2} s ({:.2} tok/s)",
        tokens.len(),
        n as f64 / dt
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
    println!(
        "ids\t{}",
        ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
    );
    let ctx = MetalContext::new()?;
    let model = Gemma4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let cap = ids.len() + n + 8;
    let sc = model.new_scratch(&ctx, cap)?;
    let mut st = model.new_state(&ctx, cap)?;
    model.prefill(&ctx, &mut st, &sc, &ids)?;
    let mut out = Vec::new();
    let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    while out.len() < n && !tok.stop_tokens().contains(&next) {
        out.push(next);
        model.forward(&ctx, &mut st, &sc, &[next], Logits::Last)?;
        next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    }
    println!("{}", tok.decode(&out, false)?);
    eprintln!("stop {next}");
    Ok(())
}

fn state_check(path: &PathBuf, cases: &PathBuf, n: usize, steps: usize) -> Result<()> {
    use lily::engine::DecodeStateApi;
    let text = std::fs::read_to_string(cases)?;
    let line = text.lines().find(|l| !l.starts_with('#')).context("no case")?;
    let ids = parse_tokens(line.split('\t').nth(2).context("case ids")?)?;
    let prompt: Vec<u32> = ids.iter().copied().cycle().take(n).collect();
    let ctx = MetalContext::new()?;
    let model = Gemma4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let cap = n + steps + 8;
    let sc = model.new_scratch(&ctx, cap)?;
    let run = |st: &mut lily::gemma4::Gemma4State| -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
        for _ in 0..steps {
            model.forward(&ctx, st, &sc, &[next], Logits::Last)?;
            let l = sc.logits.to_f32()?[..vocab].to_vec();
            next = argmax(&l) as u32;
            out.push(l);
        }
        Ok(out)
    };
    let mut st = model.new_state(&ctx, cap)?;
    model.prefill(&ctx, &mut st, &sc, &prompt)?;
    let last = sc.logits.to_f32()?[..vocab].to_vec();
    let snap = st.snapshot(&ctx)?;
    let first = run(&mut st)?;
    st.restore(&ctx, &snap)?;
    sc.logits_last.write_bytes(bytemuck::cast_slice(&last))?;
    let again = run(&mut st)?;
    let mut other = model.new_state(&ctx, 16)?;
    other.copy_prefix_from(&ctx, &st, n)?;
    other.restore(&ctx, &snap)?;
    other.ensure_capacity(&ctx, cap)?;
    sc.logits_last.write_bytes(bytemuck::cast_slice(&last))?;
    let copied = run(&mut other)?;
    let same = |a: &[Vec<f32>], b: &[Vec<f32>]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
    };
    println!(
        "prompt {n}, {steps} steps: restored {}, prefix-copied {}",
        if same(&first, &again) { "bit-identical" } else { "DIFFERENT" },
        if same(&first, &copied) { "bit-identical" } else { "DIFFERENT" }
    );
    ensure!(same(&first, &again) && same(&first, &copied), "state check failed");
    Ok(())
}

fn mem(path: &PathBuf, tokens: usize, checkpoints: usize) -> Result<()> {
    use lily::engine::DecodeStateApi;
    let ctx = MetalContext::new()?;
    let base = ctx.current_allocated();
    let model = Gemma4Model::load(&ctx, path)?;
    let w = ctx.current_allocated() - base;
    let sc = model.new_scratch(&ctx, tokens)?;
    let s = ctx.current_allocated() - base - w;
    let st = model.new_state(&ctx, tokens)?;
    let snaps: Vec<_> =
        (0..checkpoints).map(|_| st.snapshot(&ctx)).collect::<Result<_>>()?;
    let total = ctx.current_allocated() - base;
    let gb = |b: usize| b as f64 / 1e9;
    println!(
        "weights {:.3} GB, scratch {:.3} GB, session ({tokens} tokens) {:.3} GB, {checkpoints} checkpoints {:.3} GB: total {:.3} GB ({total} B)",
        gb(w),
        gb(s),
        gb(st.bytes()),
        gb(total - w - s - st.bytes()),
        gb(total)
    );
    drop((snaps, sc));
    Ok(())
}

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

fn bench(path: &PathBuf, text: &PathBuf, prompt_len: usize, n: usize, runs: usize) -> Result<()> {
    let tok = lily::tokenizer::Tokenizer::from_gguf(path)?;
    let text = std::fs::read_to_string(text)?;
    let mut ids = vec![2u32];
    ids.extend(tok.encode("<|turn>user\nContinue this text.\n\n")?);
    let body = tok.encode(&text)?;
    ensure!(body.len() + ids.len() >= prompt_len, "text has {} tokens", body.len());
    ids.extend_from_slice(&body[..prompt_len - ids.len()]);
    let ctx = MetalContext::new()?;
    let model = Gemma4Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let sc = model.new_scratch(&ctx, prompt_len + n + 8)?;
    for run in 0..runs {
        let mut st = model.new_state(&ctx, prompt_len + n + 8)?;
        let _ = lily::metal::profile::take();
        let t = Instant::now();
        model.prefill(&ctx, &mut st, &sc, &ids)?;
        let pf = t.elapsed().as_secs_f64();
        if ctx.profiling() {
            eprintln!("prefill of {prompt_len} tokens:");
            profile_report(1);
        }
        let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
        let t = Instant::now();
        let mut steps = Vec::with_capacity(n);
        let mut digest = 0xcbf29ce484222325u64;
        for _ in 0..n {
            let s = Instant::now();
            model.forward(&ctx, &mut st, &sc, &[next], Logits::Last)?;
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
