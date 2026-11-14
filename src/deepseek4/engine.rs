//! DeepSeek V4 Flash behind the engine's [`LanguageModel`] interface: what
//! the server and the bench drive. Speculative decoding drafts with DSpark
//! (when its support file is loaded) under ds4's scheduler; one session at
//! a time, no batching across sessions.

use std::path::Path;

use anyhow::{Result, ensure};

use super::model::{
    CompState, Ds4Model, Ds4State, EXACT_ROWS, Enc, Logits, PREFILL_ROWS, Scratch,
};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{EncodedPass, MetalContext, PendingPass};
use crate::tensor::{DType, Tensor};

/// When to draft (ds4's DSpark scheduler, count-only): after a block with no
/// draft (the confidence head kept none) the next [`NO_DRAFT_SKIP`] steps
/// draft nothing, and when the last [`WINDOW`] blocks kept fewer than
/// [`MIN_KEPT`] tokens on average the next [`SLOW_SKIP`]. A step without a
/// block is a one-row verify pass; the drafter's support rows are seeded
/// all the same.
#[derive(Default, Clone)]
pub struct Schedule {
    skip: usize,
    kept: Vec<usize>,
}

const NO_DRAFT_SKIP: usize = 3;
const WINDOW: usize = 4;
const MIN_KEPT: f32 = 1.5;
const SLOW_SKIP: usize = 8;

impl Schedule {
    /// Whether this step drafts (counting down a pause).
    fn draft_now(&mut self) -> bool {
        if self.skip > 0 {
            self.skip -= 1;
            return false;
        }
        true
    }

    /// A block was drafted: `drafts` proposed.
    fn drafted(&mut self, drafts: usize) {
        if drafts == 0 {
            self.skip = NO_DRAFT_SKIP;
        }
    }

    /// A verify pass of a block kept `kept` of its drafts.
    fn verified(&mut self, kept: usize) {
        self.kept.push(kept);
        if self.kept.len() >= WINDOW {
            let mean = self.kept.iter().sum::<usize>() as f32 / self.kept.len() as f32;
            if mean < MIN_KEPT {
                self.skip = SLOW_SKIP;
            }
            self.kept.clear();
        }
    }
}

impl Ds4Model {
    /// A block for the next verify pass, as the schedule allows.
    fn scheduled_block(
        &self,
        ctx: &MetalContext,
        state: &mut Ds4State,
        scratch: &Scratch,
        t: u32,
        limit: usize,
    ) -> Result<Vec<u32>> {
        if limit == 0 || !state.sched.draft_now() {
            return Ok(Vec::new());
        }
        let block = self.propose(ctx, state, scratch, t, limit)?;
        state.sched.drafted(block.len());
        Ok(block)
    }
}

/// Copies `src`'s first `bytes` into `dst` through host memory (both
/// shared-storage, the GPU idle on them).
fn host_copy(src: &Tensor, dst: &Tensor, bytes: usize) {
    assert!(bytes <= src.byte_len() && bytes <= dst.byte_len());
    // SAFETY: both buffers are host-visible and hold at least `bytes`; the
    // GPU is idle on them by the callers' contracts.
    unsafe {
        core::ptr::copy_nonoverlapping(src.contents_ptr(), dst.contents_ptr(), bytes);
    }
}

fn clone_tensor(ctx: &MetalContext, t: &Tensor) -> Result<Tensor> {
    let c = Tensor::zeros(ctx, t.shape(), t.dtype())?;
    host_copy(t, &c, t.byte_len());
    Ok(c)
}

/// The part of a state that is not append-only: each layer's raw window
/// ring and compressor rings.
pub struct Ds4Snapshot {
    pos: usize,
    tensors: Vec<Tensor>,
}

impl SnapshotApi for Ds4Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.tensors.iter().map(Tensor::byte_len).sum()
    }
}

/// The tensors a snapshot holds, in a fixed order.
fn rings(st: &Ds4State) -> Vec<&Tensor> {
    st.layers
        .iter()
        .flat_map(|l| {
            std::iter::once(&l.ring).chain(
                l.comp
                    .iter()
                    .chain(l.index.iter())
                    .flat_map(|c| [&c.ring_kv, &c.ring_sc]),
            )
        })
        .collect()
}

/// The compressed caches (append-only, `rows(pos)` rows valid) with each
/// one's ratio and row width.
fn caches<'s>(
    st: &'s mut Ds4State,
    cfg: &super::Config,
) -> Vec<(&'s mut CompState, usize, usize)> {
    st.layers
        .iter_mut()
        .enumerate()
        .flat_map(|(il, l)| {
            let r = cfg.ratio(il);
            let mut v = Vec::new();
            if let Some(c) = l.comp.as_mut() {
                v.push((c, r, cfg.head_dim));
            }
            if let Some(c) = l.index.as_mut() {
                v.push((c, r, cfg.indexer_dim));
            }
            v
        })
        .collect()
}

impl DecodeStateApi for Ds4State {
    type Snapshot = Ds4Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        // A decode step seeds its row's DSpark support rows as it runs.
        if let Some(ds) = self.ds.as_mut() {
            ds.note(self.pos, n);
        }
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        Ds4State::reset(self);
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        let capacity = tokens.max(self.capacity * 2);
        let cfg = self.cfg.clone();
        let pos = self.pos;
        for (c, r, dim) in caches(self, &cfg) {
            let grown = Tensor::zeros(ctx, &[(capacity / r + 1) * dim], DType::F16)?;
            host_copy(&c.cache, &grown, (pos / r) * dim * 2);
            c.cache = grown;
        }
        self.score_cap = capacity / 4 + 1;
        self.scores = Tensor::zeros(ctx, &[PREFILL_ROWS * self.score_cap], DType::F32)?;
        let (plain, yarn) = super::model::rope_tables(ctx, &cfg, capacity)?;
        self.rope_plain = plain;
        self.rope_yarn = yarn;
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.ring.byte_len()
                    + l.comp
                        .iter()
                        .chain(l.index.iter())
                        .map(|c| {
                            c.cache.byte_len()
                                + c.ring_kv.byte_len()
                                + c.ring_sc.byte_len()
                        })
                        .sum::<usize>()
            })
            .sum::<usize>()
            + self.scores.byte_len()
            + self.rope_plain.byte_len()
            + self.rope_yarn.byte_len()
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Ds4Snapshot> {
        let tensors = rings(self)
            .into_iter()
            .map(|t| clone_tensor(ctx, t))
            .collect::<Result<_>>()?;
        Ok(Ds4Snapshot { pos: self.pos, tensors })
    }

    fn restore(&mut self, _ctx: &MetalContext, snapshot: &Ds4Snapshot) -> Result<()> {
        ensure!(snapshot.pos <= self.capacity, "snapshot past the state's capacity");
        for (dst, src) in rings(self).into_iter().zip(&snapshot.tensors) {
            host_copy(src, dst, src.byte_len());
        }
        self.pos = snapshot.pos;
        self.spec = None;
        // The drafter's support rows are not in the snapshot: reseeded as
        // the session goes on.
        if let Some(ds) = self.ds.as_mut() {
            ds.invalidate();
        }
        Ok(())
    }

    fn copy_prefix_from(
        &mut self,
        ctx: &MetalContext,
        from: &Self,
        tokens: usize,
    ) -> Result<()> {
        ensure!(tokens <= from.pos, "prefix of {tokens} from a state at {}", from.pos);
        self.ensure_capacity(ctx, tokens)?;
        let cfg = self.cfg.clone();
        let mut from_caches: Vec<&Tensor> = Vec::new();
        for l in &from.layers {
            for c in l.comp.iter().chain(l.index.iter()) {
                from_caches.push(&c.cache);
            }
        }
        for ((c, r, dim), src) in caches(self, &cfg).into_iter().zip(from_caches) {
            host_copy(src, &c.cache, (tokens / r) * dim * 2);
        }
        Ok(())
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    fn logits(&self) -> &Tensor {
        &self.logits_last
    }

    fn begin_request(&self) {
        self.sampler.reset_counts();
    }
}

impl Ds4Model {
    /// Feeds `tokens` in runs of [`PREFILL_ROWS`], drawing after the last
    /// when `draw`; before each run, stops if `stop` says so (the state then
    /// sits at a run boundary). Returns the tokens fed.
    fn prefill_chunks(
        &self,
        ctx: &MetalContext,
        state: &mut Ds4State,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        ensure!(!tokens.is_empty(), "empty prefill");
        state.ensure_capacity(ctx, state.pos + tokens.len())?;
        let chunks: Vec<&[u32]> = tokens.chunks(PREFILL_ROWS).collect();
        let mut fed = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            if stop() {
                return Ok(fed);
            }
            let last = i + 1 == chunks.len();
            let logits =
                if last && draw.is_some() { Logits::Last } else { Logits::None };
            self.run(ctx, state, scratch, chunk, logits, None)?;
            fed += chunk.len();
            if let (true, Some(d)) = (last, draw) {
                let pass = ctx.begin()?;
                let out = scratch.next_token.view(0, &[1])?;
                sample_f32(
                    ctx,
                    &pass,
                    &scratch.logits_last,
                    &scratch.sampler,
                    d.params,
                    d.step,
                    &out,
                )?;
                pass.commit_wait()?;
            }
        }
        Ok(fed)
    }
}

impl LanguageModel for Ds4Model {
    type State = Ds4State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "deepseek-v4-flash";

    fn load(ctx: &MetalContext, path: &Path, options: &LoadOptions) -> Result<Self> {
        let mut model = Ds4Model::load(ctx, path)?;
        if options.mtp_drafts > 0 {
            let support = dspark_path(path)?;
            model.load_dspark(ctx, &support)?;
            eprintln!("DSpark drafter: {}", support.display());
        }
        Ok(model)
    }

    /// DSpark's block (five drafts) when its support file is loaded.
    fn max_drafts(&self) -> usize {
        self.dspark_block()
    }

    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut Ds4State,
        scratch: &mut Scratch,
        first: u32,
        drafts: usize,
        _params: &SamplingParams,
        _step0: usize,
    ) -> Result<Vec<u32>> {
        state.ensure_capacity(ctx, state.pos + self.dspark_block() + 1)?;
        self.scheduled_block(ctx, state, scratch, first, drafts)
    }

    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Ds4State,
        scratch: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        let rows = 1 + drafts.len();
        ensure!(rows <= EXACT_ROWS, "a verify pass of {rows} rows");
        ensure!(state.spec.is_none(), "a verify pass is already pending");
        // Room for this pass and the block drafted after it.
        state.ensure_capacity(ctx, state.pos + rows + self.dspark_block() + 1)?;
        let tokens: Vec<u32> =
            std::iter::once(pending).chain(drafts.iter().copied()).collect();
        crate::glm53::model::write_words(&scratch.spec_tokens, &tokens)?;
        let vocab = self.cfg.vocab;
        let sc: &Scratch = scratch;
        let pass = ctx.begin_concurrent()?;
        // A pass of one row has nothing to roll back: it writes its rings
        // as a decode step does.
        let enc = Enc {
            logits: Logits::All,
            defer: rows > 1,
            capture: self.dspark.is_some(),
        };
        let n = self.cfg.n_layer;
        self.encode_layers(ctx, &pass, state, sc, &sc.spec_tokens, rows, 0..n, enc)?;
        for r in 0..rows {
            let logits = sc.logits.view(r * vocab, &[vocab])?;
            let out = sc.draws.view(r, &[1])?;
            sample_f32(ctx, &pass, &logits, &sc.sampler, params, step0 + r, &out)?;
            pass.memory_barrier()?;
        }
        pass.commit_wait()?;
        let draws = scratch.draws.to_u32()?[..rows].to_vec();
        state.spec = Some((state.pos, rows, draws.clone()));
        state.spec_k = next_drafts;
        state.pos += rows;
        // The rings and the next block wait for the kept count.
        Ok((draws, ctx.begin()?.commit()?))
    }

    fn finish_speculation(
        &self,
        ctx: &MetalContext,
        state: &mut Ds4State,
        scratch: &mut Scratch,
        accepted: usize,
        propose: bool,
        draft: PendingPass<'_>,
    ) -> Result<Vec<u32>> {
        draft.wait()?;
        let (pos0, rows, draws) = state
            .spec
            .take()
            .ok_or_else(|| anyhow::anyhow!("no verify pass pending"))?;
        ensure!(accepted < rows, "{accepted} accepted of {} drafts", rows - 1);
        if rows > 1 {
            self.commit_rows(ctx, state, scratch, pos0, accepted + 1)?;
        } else if let Some(ds) = state.ds.as_mut() {
            ds.note(pos0, 1);
        }
        state.pos = pos0 + accepted + 1;
        for &token in &draws[accepted + 1..] {
            scratch.sampler.uncount(token);
        }
        if rows > 1 {
            state.sched.verified(accepted);
        }
        if !propose {
            return Ok(Vec::new());
        }
        let k = state.spec_k;
        self.scheduled_block(ctx, state, scratch, draws[accepted], k)
    }

    fn max_position_embeddings(&self) -> usize {
        self.cfg.context_length
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        crate::deepseek4::tokenizer::stop_ids(&self.gguf)
            .unwrap_or_else(|_| vec![self.cfg.eos])
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn bytes_per_token(&self) -> usize {
        // The f16 compressed rows a token adds: attention rows of 512 per
        // ratio, the ratio-4 layers' indexer rows of 128 per 4.
        (0..self.cfg.n_layer)
            .map(|il| match self.cfg.ratio(il) {
                0 => 0,
                r => {
                    self.cfg.head_dim * 2 / r
                        + if self.cfg.has_indexer(il) {
                            self.cfg.indexer_dim * 2 / r
                        } else {
                            0
                        }
                }
            })
            .sum()
    }

    fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        Ds4Model::weight_buffers(self)
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Ds4State> {
        Ds4Model::new_state(self, ctx, capacity.max(4))
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        _capacity_tokens: usize,
    ) -> Result<Scratch> {
        self.new_scratch(ctx)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut Ds4State,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        self.prefill_chunks(ctx, state, scratch, tokens, draw, &|| false).map(drop)
    }

    fn prefill_chunk_tokens(&self) -> Option<usize> {
        Some(PREFILL_ROWS)
    }

    fn prefill_until(
        &self,
        ctx: &MetalContext,
        state: &mut Ds4State,
        scratch: &mut Scratch,
        tokens: &[u32],
        vision: Option<&crate::qwen4exp::VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        ensure!(vision.is_none(), "DeepSeek V4 Flash has no vision path here");
        self.prefill_chunks(ctx, state, scratch, tokens, None, stop)
    }

    fn prepare_step_inputs(
        &self,
        _state: &mut Ds4State,
        _scratch: &Scratch,
        _token: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &Ds4State,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        ensure!(state.pos < state.capacity, "decode step past the state's capacity");
        let pass = ctx.begin_concurrent()?;
        let token = scratch.next_token.view(slot_in, &[1])?;
        let enc =
            Enc { logits: Logits::Last, defer: false, capture: self.dspark.is_some() };
        self.encode_layers(
            ctx,
            &pass,
            state,
            scratch,
            &token,
            1,
            0..self.cfg.n_layer,
            enc,
        )?;
        let out = scratch.next_token.view(slot_out, &[1])?;
        sample_f32(
            ctx,
            &pass,
            &scratch.logits_last,
            &scratch.sampler,
            draw.params,
            draw.step,
            &out,
        )?;
        pass.end()
    }
}

/// The DSpark support file for the model at `model`: `LILY_DS4_DSPARK`, or
/// the one `*DSpark*.gguf` file beside it.
fn dspark_path(model: &Path) -> Result<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("LILY_DS4_DSPARK") {
        return Ok(p.into());
    }
    let dir = model.parent().unwrap_or(Path::new("."));
    let found: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "gguf")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("DSpark"))
        })
        .collect();
    ensure!(
        found.len() == 1,
        "speculation needs DeepSeek V4 Flash's DSpark support file beside the model \
         (found {}); set LILY_DS4_DSPARK",
        found.len()
    );
    Ok(found[0].clone())
}

#[cfg(test)]
#[path = "../../tests/unit/deepseek4/engine.rs"]
mod tests;
