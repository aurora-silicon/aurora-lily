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
    },
}

fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    lily::instance::acquire()?;
    let ctx = MetalContext::new()?;
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
            let tok = Tokenizer::from_gguf(&cli.model)?;
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
        Cmd::Profile { iters } => {
            let ids: Vec<u32> = (0..32).map(|i| (100 + i) as u32).collect();
            let mut state = model.new_state(&ctx, 128)?;
            let mut scratch = model.new_scratch(&ctx, model.prefill_rows)?;
            scratch.begin_request();
            let params = SamplingParams::greedy();
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
    Ok(())
}
