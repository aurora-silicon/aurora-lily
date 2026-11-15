//! Optional checks of the Qwen3.8 Q2 GGUF's paged token embedding (the file
//! in `LILY_QWEN38_Q2`, a lily checkpoint for its n-gram table in
//! `LILY_NGRAM_TABLE`): under a memory limit the trunk embeds every token
//! from rows read from the file and the draft head from its resident narrow
//! rows, and the outputs equal the mapped table's. The load is fitted to
//! 44.7 GiB both ways, with 256-row prompt chunks so a prompt spans several,
//! and tokens outside the draft head's narrow LM head sit in the first chunk
//! and a later one. Each run loads the file (45 GB): run alone.

use std::path::Path;

use anyhow::{Context, Result};
use lily::engine::{DecodeStateApi, LanguageModel, LoadOptions};
use lily::generate::{GenerateOptions, Generator};
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::qwen4exp::Qwen4ExpModel;

fn model_path() -> Result<String> {
    std::env::var("LILY_QWEN38_Q2").context("set LILY_QWEN38_Q2 to the Q2 GGUF")
}

/// A token outside the draft head's narrow LM head (ids 98 304 to 248 043).
const OUTSIDE: u32 = 200_000;
/// Where the prompt is split for the cached-prefix run (a chunk boundary).
const PREFIX: usize = 512;

/// What one load produces: greedy tokens of a plain run (parked decode
/// steps), of a two-draft run, of a two-draft continuation after a cached
/// prefix (restored into a new state), the draft head's proposals after
/// that continuation, and the drafts accepted of the drafted runs.
#[derive(Debug, PartialEq)]
struct Runs {
    plain: Vec<u32>,
    drafted: Vec<u32>,
    continued: Vec<u32>,
    proposals: Vec<u32>,
    accepted: (usize, usize),
}

fn runs(paged: bool) -> Result<Runs> {
    let path = model_path()?;
    // SAFETY: the test is single-threaded here; the loader reads the
    // variable once per load.
    unsafe { std::env::set_var("LILY_EMBED_PAGED", if paged { "1" } else { "0" }) };
    let ctx = MetalContext::new()?;
    let limit = (44.7 * (1u64 << 30) as f64) as u64;
    let mut model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&path),
        &LoadOptions { mtp_drafts: 2, memory_limit: Some(limit), ..LoadOptions::default() },
    )?;
    assert_eq!(model.max_batch_rows(), 0, "a fitted load batches no rows");
    model.set_prefill_chunk(256)?;
    let mut generator = Generator::from_model_dir(Path::new(&path))?;
    generator.add_stop_tokens(&model.eos_token_ids());
    let text = "The library keeps a short list of rules for its readers. ".repeat(120);
    let mut prompt = generator.tokenizer().encode(&text)?;
    prompt.truncate(700);
    prompt[100] = OUTSIDE + 1;
    prompt[600] = OUTSIDE;
    let greedy = SamplingParams::greedy();
    let options = |drafts| GenerateOptions {
        max_tokens: 32,
        sampling: &greedy,
        stop_tokens: &[],
        drafts,
        thinking: None,
    };
    let capacity = prompt.len() + 64;
    let fresh = |model: &Qwen4ExpModel| -> Result<_> {
        Ok((model.new_state(&ctx, capacity)?, model.new_scratch_with_capacity(&ctx, capacity)?))
    };

    let (mut state, mut scratch) = fresh(&model)?;
    let plain = generator
        .generate(&ctx, &model, &mut state, &mut scratch, &prompt, &options(0), &mut |_| Ok(true))?
        .tokens;
    let (mut state, mut scratch) = fresh(&model)?;
    let d = generator.generate(
        &ctx,
        &model,
        &mut state,
        &mut scratch,
        &prompt,
        &options(2),
        &mut |_| Ok(true),
    )?;

    // The prefix in one state, snapshotted, restored into another; the
    // continuation feeds the rest (its chunks start past 0: the draft head's
    // catch-up reuses their staged rows).
    let (mut a, mut scratch) = fresh(&model)?;
    model.prefill(&ctx, &mut a, &mut scratch, &prompt[..PREFIX], None)?;
    let snap = a.snapshot(&ctx)?;
    let (mut b, _) = fresh(&model)?;
    b.copy_prefix_from(&ctx, &a, PREFIX)?;
    b.restore(&ctx, &snap)?;
    drop(a);
    let c = generator.generate(
        &ctx,
        &model,
        &mut b,
        &mut scratch,
        &prompt[PREFIX..],
        &options(2),
        &mut |_| Ok(true),
    )?;
    let last = *c.tokens.last().context("no tokens")?;
    let proposals =
        model.draft_initial(&ctx, &mut b, &mut scratch, last, 2, &greedy, c.tokens.len())?;
    Ok(Runs {
        plain,
        drafted: d.tokens,
        continued: c.tokens,
        proposals,
        accepted: (d.accepted, c.accepted),
    })
}

/// Paged and mapped give the same tokens, proposals and acceptance (the draft
/// head embeds the prompt's tokens and the accepted drafts exactly; the
/// draws here are inside its narrow head).
#[test]
#[ignore = "requires LILY_QWEN38_Q2"]
fn the_paged_embedding_gives_the_mapped_tables_outputs() -> Result<()> {
    let mapped = runs(false)?;
    let paged = runs(true)?;
    assert_eq!(paged, mapped);
    Ok(())
}
