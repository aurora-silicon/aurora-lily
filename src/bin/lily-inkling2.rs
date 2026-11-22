//! Inkling 2 Mini bring-up, memory-fit, and benchmark CLI (`lily-inkling2`).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use lily::inkling2::Inkling2Model;
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::tokenizer::{ChatRender, Tokenizer};

#[derive(Parser)]
#[command(name = "lily-inkling2", about = "Inkling 2 Mini bring-up and benchmark tool")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    memory_limit_gb: Option<f64>,
    #[arg(long, default_value_t = 3)]
    mtp_drafts: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print model shape, weight size, and fitted context under `--memory-limit-gb`.
    Fit {
        #[arg(long, default_value_t = 1)]
        checkpoints: usize,
    },
    /// Generate a short greedy reply from a prompt.
    Chat {
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
    },
    /// Measure prefill and decode throughput on a synthetic prompt.
    Bench {
        #[arg(long, default_value_t = 256)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 64)]
        decode_tokens: usize,
    },
    /// Break down per-component GPU latency of one decode step.
    Profile {
        #[arg(long, default_value_t = 10)]
        iters: usize,
        /// Record per-kernel GPU timestamps for prefill, decode, verify, and DSpark.
        #[arg(long)]
        kernel_profile: bool,
        /// Context length before profiling decode/verify.
        #[arg(long, default_value_t = 32)]
        ctx_len: usize,
    },
}

fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn print_kernel_profile(label: &str) {
    let passes = lily::metal::profile::take();
    if passes.is_empty() {
        return;
    }
    let count = passes.len() as f64;
    let mut by_kernel: std::collections::HashMap<&'static str, (usize, f64)> =
        std::collections::HashMap::new();
    let (mut dispatches, mut kernel_secs, mut span_secs) = (0usize, 0.0f64, 0.0f64);
    for pass in &passes {
        span_secs += pass.span_secs;
        for sample in &pass.kernels {
            dispatches += 1;
            kernel_secs += sample.gpu_secs;
            let entry = by_kernel.entry(sample.name).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += sample.gpu_secs;
        }
    }
    let mut rows: Vec<(&str, usize, f64)> =
        by_kernel.into_iter().map(|(k, (c, s))| (k, c, s)).collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(b.0)));
    println!(
        "\n=== kernel profile [{label}]: {} passes, {:.1} dispatches/pass ===",
        passes.len(),
        dispatches as f64 / count
    );
    println!("  {:>9}  {:>6}  {:>10}  kernel", "ms/pass", "share", "calls/pass");
    for (name, calls, secs) in &rows {
        let share = if kernel_secs > 0.0 { 100.0 * secs / kernel_secs } else { 0.0 };
        println!(
            "  {:>9.3}  {:>5.1}%  {:>10.1}  {name}",
            secs / count * 1e3,
            share,
            *calls as f64 / count
        );
    }
    println!(
        "  kernels sum {:.3} ms | mean span {:.3} ms | overhead {:.3} ms",
        kernel_secs / count * 1e3,
        span_secs / count * 1e3,
        (span_secs - kernel_secs) / count * 1e3
    );
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    lily::instance::acquire()?;
    let enable_profile = matches!(cli.cmd, Cmd::Profile { kernel_profile: true, .. });
    let ctx = MetalContext::new_with_profile(enable_profile)?;
    let opts = LoadOptions {
        memory_limit: cli.memory_limit_gb.map(|g| (g * (1u64 << 30) as f64) as u64),
        mtp_drafts: cli.mtp_drafts,
        vision: lily::engine::VisionMode::Off,
        ..LoadOptions::default()
    };
    let model = <Inkling2Model as LanguageModel>::load(&ctx, &cli.model, &opts)?;

    match cli.cmd {
        Cmd::Fit { checkpoints } => {
            println!(
                "inkling2: arch={} phys={} visits={} hidden={} experts={}x{} (top {})",
                model.cfg.arch,
                model.cfg.n_phys,
                model.cfg.n_visits,
                model.cfg.hidden,
                model.cfg.n_expert,
                model.cfg.ff_expert,
                model.cfg.n_expert_used
            );
            let w_bytes: usize =
                model.weight_buffers().iter().map(|b| b.length()).sum();
            println!(
                "weights: {:.2} GB, global KV: {} B/tok, recurrent: {:.2} MB, prefill_rows: {}",
                w_bytes as f64 / 1e9,
                model.bytes_per_token(),
                model.recurrent_bytes() as f64 / 1e6,
                model.prefill_rows,
            );
            if let Some(limit) = opts.memory_limit {
                let scratch = model.new_scratch(&ctx, model.prefill_rows)?;
                let alloc = ctx.current_allocated() as u64;
                drop(scratch);
                let max_ctx = model.fit_context(limit, alloc, checkpoints)?;
                println!(
                    "fit: limit={:.2} GB -> max_context={max_ctx} ({checkpoints} snapshots)",
                    limit as f64 / 1e9
                );
            }
        }
        Cmd::Chat { prompt, max_tokens } => {
            let tok = if cli.model.is_dir() {
                Tokenizer::from_model_dir(&cli.model)?
            } else {
                Tokenizer::from_gguf(&cli.model)?
            };
            let msgs = [serde_json::json!({"role": "user", "content": prompt})];
            let ids = tok.encode_chat(&ChatRender {
                messages: &msgs,
                tools: None,
                enable_thinking: true,
                reasoning_effort: None,
                preserve_thinking: None,
            })?;
            let mut state = model.new_state(&ctx, ids.len() + max_tokens + 16)?;
            let mut scratch = model.new_scratch(&ctx, model.prefill_rows)?;
            scratch.begin_request();
            let params = SamplingParams::greedy();
            let draw = Draw { params: &params, step: 0 };
            model.prefill(&ctx, &mut state, &mut scratch, &ids, Some(draw))?;
            let mut out = Vec::with_capacity(max_tokens);
            let mut next = scratch.next_token().to_u32()?[0];
            let max_drafts = model.max_drafts();
            if max_drafts > 0 {
                let mut drafts = if model.eos_token_ids().contains(&next) {
                    Vec::new()
                } else {
                    model.draft_initial(
                        &ctx,
                        &mut state,
                        &mut scratch,
                        next,
                        max_drafts.min(max_tokens.saturating_sub(1)),
                        &params,
                        1,
                    )?
                };
                while out.len() < max_tokens {
                    if model.eos_token_ids().contains(&next) {
                        break;
                    }
                    let rem = max_tokens - out.len();
                    if drafts.is_empty() || rem <= 1 {
                        out.push(next);
                        if out.len() >= max_tokens {
                            break;
                        }
                        model.prepare_step_inputs(&mut state, &scratch, next)?;
                        let pass = model.encode_decode_step(
                            &ctx,
                            &state,
                            &scratch,
                            0,
                            0,
                            Draw { params: &params, step: out.len() },
                        )?;
                        pass.commit()?.wait()?;
                        state.advance(1);
                        next = scratch.next_token().to_u32()?[0];
                        continue;
                    }
                    let use_drafts = &drafts[..drafts.len().min(rem - 1)];
                    let (draws, pending) = model.verify(
                        &ctx,
                        &mut state,
                        &mut scratch,
                        next,
                        use_drafts,
                        &params,
                        out.len() + 1,
                        max_drafts,
                    )?;
                    let mut acc = 0usize;
                    let mut hit_eos = false;
                    out.push(next);
                    while acc < use_drafts.len() && draws[acc] == use_drafts[acc] {
                        if model.eos_token_ids().contains(&draws[acc]) {
                            hit_eos = true;
                            break;
                        }
                        out.push(draws[acc]);
                        acc += 1;
                    }
                    next = draws[acc];
                    if hit_eos || model.eos_token_ids().contains(&next) {
                        let _ = model.finish_speculation(
                            &ctx,
                            &mut state,
                            &mut scratch,
                            acc,
                            false,
                            pending,
                        )?;
                        break;
                    }
                    let more = out.len() < max_tokens;
                    drafts = model.finish_speculation(
                        &ctx,
                        &mut state,
                        &mut scratch,
                        acc,
                        more,
                        pending,
                    )?;
                }
            } else {
                for step in 0..max_tokens {
                    if model.eos_token_ids().contains(&next) {
                        break;
                    }
                    out.push(next);
                    model.prepare_step_inputs(&mut state, &scratch, next)?;
                    let pass = model.encode_decode_step(
                        &ctx,
                        &state,
                        &scratch,
                        0,
                        0,
                        Draw { params: &params, step: step + 1 },
                    )?;
                    pass.commit()?.wait()?;
                    state.advance(1);
                    next = scratch.next_token().to_u32()?[0];
                }
            }
            println!("{}", tok.decode(&out, true)?);
        }
        Cmd::Bench { prompt_tokens, decode_tokens } => {
            let ids: Vec<u32> =
                (0..prompt_tokens).map(|i| (100 + (i % 1000)) as u32).collect();
            let mut state =
                model.new_state(&ctx, prompt_tokens + decode_tokens + 16)?;
            let mut scratch = model.new_scratch(&ctx, model.prefill_rows)?;
            scratch.begin_request();
            let params = SamplingParams::greedy();
            let t0 = Instant::now();
            model.prefill(
                &ctx,
                &mut state,
                &mut scratch,
                &ids,
                Some(Draw { params: &params, step: 0 }),
            )?;
            let prefill_s = t0.elapsed().as_secs_f64();
            let mut next = scratch.next_token().to_u32()?[0];
            let max_drafts = model.max_drafts();
            let mut generated = 0usize;
            let mut accepted_total = 0usize;
            let mut proposed_total = 0usize;
            let t1 = Instant::now();
            if max_drafts > 0 {
                let mut drafts = model.draft_initial(
                    &ctx,
                    &mut state,
                    &mut scratch,
                    next,
                    max_drafts.min(decode_tokens.saturating_sub(1)),
                    &params,
                    1,
                )?;
                while generated < decode_tokens {
                    let rem = decode_tokens - generated;
                    if drafts.is_empty() || rem <= 1 {
                        scratch
                            .next_token
                            .view(0, &[1])?
                            .write_bytes(bytemuck::cast_slice(&[next]))?;
                        model.prepare_step_inputs(&mut state, &scratch, next)?;
                        let pass = model.encode_decode_step(
                            &ctx,
                            &state,
                            &scratch,
                            0,
                            0,
                            Draw { params: &params, step: generated + 1 },
                        )?;
                        pass.commit()?.wait()?;
                        state.advance(1);
                        next = scratch.next_token().to_u32()?[0];
                        generated += 1;
                        continue;
                    }
                    let use_drafts = &drafts[..drafts.len().min(rem - 1)];
                    proposed_total += use_drafts.len();
                    let (draws, pending) = model.verify(
                        &ctx,
                        &mut state,
                        &mut scratch,
                        next,
                        use_drafts,
                        &params,
                        generated + 1,
                        max_drafts,
                    )?;
                    let mut acc = 0usize;
                    while acc < use_drafts.len() && draws[acc] == use_drafts[acc] {
                        acc += 1;
                    }
                    accepted_total += acc;
                    generated += 1 + acc;
                    next = draws[acc];
                    let more = generated < decode_tokens;
                    drafts = model.finish_speculation(
                        &ctx,
                        &mut state,
                        &mut scratch,
                        acc,
                        more,
                        pending,
                    )?;
                }
            } else {
                for step in 0..decode_tokens {
                    model.prepare_step_inputs(&mut state, &scratch, next)?;
                    let pass = model.encode_decode_step(
                        &ctx,
                        &state,
                        &scratch,
                        0,
                        0,
                        Draw { params: &params, step: step + 1 },
                    )?;
                    pass.commit()?.wait()?;
                    state.advance(1);
                    next = scratch.next_token().to_u32()?[0];
                    generated += 1;
                }
            }
            let decode_s = t1.elapsed().as_secs_f64();
            if proposed_total > 0 {
                println!(
                    "prefill: {} tok in {:.3}s ({:.1} tok/s) | decode: {} tok in {:.3}s ({:.1} tok/s, drafts {}/{} accepted)",
                    prompt_tokens,
                    prefill_s,
                    prompt_tokens as f64 / prefill_s,
                    generated,
                    decode_s,
                    generated as f64 / decode_s,
                    accepted_total,
                    proposed_total,
                );
            } else {
                println!(
                    "prefill: {} tok in {:.3}s ({:.1} tok/s) | decode: {} tok in {:.3}s ({:.1} tok/s)",
                    prompt_tokens,
                    prefill_s,
                    prompt_tokens as f64 / prefill_s,
                    generated,
                    decode_s,
                    generated as f64 / decode_s,
                );
            }
        }
        Cmd::Profile { iters, kernel_profile, ctx_len } => {
            let ids: Vec<u32> = (0..ctx_len).map(|i| (100 + (i % 1000)) as u32).collect();
            let mut state = model.new_state(&ctx, ctx_len + 128)?;
            let mut scratch = model.new_scratch(&ctx, model.prefill_rows)?;
            let min_rows = if kernel_profile {
                scratch
                    .verify
                    .as_ref()
                    .map_or(4, |v| v.max_rows)
                    .min(scratch.max_rows)
            } else if scratch.max_rows >= 4
                && scratch.verify.as_ref().is_none_or(|v| v.max_rows >= 4)
            {
                4
            } else {
                1
            };
            anyhow::ensure!(
                ctx_len + min_rows <= state.capacity,
                "ctx_len={ctx_len} leaves fewer than {min_rows} KV slots below capacity {}",
                state.capacity
            );
            scratch.begin_request();
            let params = SamplingParams::greedy();
            if kernel_profile {
                // Warmup one short forward first so pipelines are compiled.
                let warm_ids: Vec<u32> = (0..model.prefill_rows.min(ctx_len))
                    .map(|i| (100 + i) as u32)
                    .collect();
                model.prefill(
                    &ctx,
                    &mut state,
                    &mut scratch,
                    &warm_ids,
                    Some(Draw { params: &params, step: 0 }),
                )?;
                state.reset()?;
                let _ = lily::metal::profile::take();

                // 1. Prefill profile.
                model.prefill(
                    &ctx,
                    &mut state,
                    &mut scratch,
                    &ids,
                    Some(Draw { params: &params, step: 0 }),
                )?;
                print_kernel_profile(&format!("prefill ctx={ctx_len}"));

                // 2. Plain decode (1 tok) profile.
                scratch.stage_tokens(&[100])?;
                model.forward_rows(&ctx, &state, &scratch, 1, false, true, None)?;
                let _ = lily::metal::profile::take();
                for _ in 0..iters.min(3) {
                    model.forward_rows(&ctx, &state, &scratch, 1, false, true, None)?;
                }
                print_kernel_profile(&format!("decode(1) ctx={ctx_len}"));

                // 3. Verify (v_rows tok) profile.
                let v_rows = scratch
                    .verify
                    .as_ref()
                    .map_or(4, |v| v.max_rows)
                    .min(scratch.max_rows)
                    .min(state.capacity.saturating_sub(state.pos));
                if v_rows >= 2 {
                    let v_drafts = v_rows - 1;
                    let v_toks: Vec<u32> = (1..=v_rows as u32).collect();
                    scratch.stage_tokens(&v_toks)?;
                    let cp = ctx.begin()?;
                    model.encode_forward_rows(
                        &ctx,
                        &cp,
                        &state,
                        &scratch,
                        v_rows,
                        None,
                        false,
                        true,
                        model.dspark.is_some(),
                        scratch.verify.is_some(),
                    )?;
                    cp.commit_wait()?;
                    let _ = lily::metal::profile::take();
                    for _ in 0..iters.min(3) {
                        let cp = ctx.begin()?;
                        model.encode_forward_rows(
                            &ctx,
                            &cp,
                            &state,
                            &scratch,
                            v_rows,
                            None,
                            false,
                            true,
                            model.dspark.is_some(),
                            scratch.verify.is_some(),
                        )?;
                        cp.commit_wait()?;
                    }
                    print_kernel_profile(&format!("verify({v_rows}) ctx={ctx_len}"));

                    // 4. DSpark (inj v_rows + prop v_drafts) profile.
                    if let (Some(ds), Some(ds_state), Some(cap_buf)) =
                        (&model.dspark, &mut state.dspark, &scratch.dspark_cap)
                    {
                        let anchor_embd =
                            scratch.logits_raw.view(0, &[model.cfg.hidden])?;
                        ds.prepare_propose(1)?;
                        let _ = lily::metal::profile::take();
                        for _ in 0..iters.min(3) {
                            let cp = ctx.begin()?;
                            ds.encode_inject(
                                &ctx,
                                &cp,
                                ds_state,
                                cap_buf,
                                scratch.max_rows,
                                v_rows,
                                ctx_len.saturating_sub(v_rows),
                            )?;
                            cp.memory_barrier()?;
                            ds.encode_propose(
                                &ctx,
                                &cp,
                                ds_state,
                                &anchor_embd,
                                model.draft_output_head(),
                                model.cfg.vocab_unpadded,
                                ctx_len,
                                v_drafts,
                            )?;
                            cp.commit_wait()?;
                        }
                        print_kernel_profile(&format!(
                            "dspark(inj{v_rows}+prop{v_drafts}) ctx={ctx_len}"
                        ));
                    }
                }
            } else {
                model.prefill(
                    &ctx,
                    &mut state,
                    &mut scratch,
                    &ids,
                    Some(Draw { params: &params, step: 0 }),
                )?;
                model.profile_decode(&ctx, &mut state, &scratch, iters)?;
            }
        }
    }
    Ok(())
}
