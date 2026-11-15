//! Gemma 4 behind the engine's [`LanguageModel`] interface: what the
//! server and the bench drive. One session at a time, no draft head yet,
//! no batching across sessions.
//!
//! A state's per-token caches are the global layers' K/V (copied by
//! prefix); its "recurrent" part, what a snapshot holds, is the
//! sliding-window layers' rings, which a later position overwrites.

use std::path::Path;

use anyhow::{Result, ensure};

use super::model::{
    Gemma4Model, Gemma4State, LOOKAHEAD, LayerCache, Logits, MAX_ROWS, MAX_SEQ,
    PREFILL_ROWS, Scratch, capacity_cap, check_tokens,
};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{EncodedPass, MetalContext, PendingPass};
use crate::tensor::{DType, Tensor};

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

/// Copies `bytes` from `src` at byte `from` to `dst` at byte `to` through
/// host memory (both shared-storage, the GPU idle on them).
fn host_copy_at(src: &Tensor, from: usize, dst: &Tensor, to: usize, bytes: usize) {
    assert!(from + bytes <= src.byte_len() && to + bytes <= dst.byte_len());
    // SAFETY: both ranges lie inside host-visible buffers (asserted); the
    // GPU is idle on them by the callers' contracts.
    unsafe {
        core::ptr::copy_nonoverlapping(
            src.contents_ptr().add(from),
            dst.contents_ptr().add(to),
            bytes,
        );
    }
}

/// The sliding-window rings at one position: each ring's last `window`
/// positions (fewer at the start) in position order, in a window-sized
/// buffer; and the draft model's input there.
pub struct Gemma4Snapshot {
    pos: usize,
    tensors: Vec<Tensor>,
    h_last: Tensor,
}

impl SnapshotApi for Gemma4Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.tensors.iter().map(Tensor::byte_len).sum::<usize>()
            + self.h_last.byte_len()
    }
}

/// Each ring tensor with its slots and bytes per slot.
fn rings(st: &Gemma4State) -> Vec<(&Tensor, usize, usize)> {
    st.layers
        .iter()
        .filter(|l| l.swa)
        .flat_map(|l| {
            let per_slot = l.k.byte_len() / l.slots;
            [(&l.k, l.slots, per_slot), (&l.v, l.slots, per_slot)]
        })
        .collect()
}

/// The slot ranges (start slot, count) that hold positions `lo..pos` of a
/// ring of `slots`, in position order: one range, or two across the wrap.
fn ring_ranges(lo: usize, pos: usize, slots: usize) -> Vec<(usize, usize)> {
    let n = pos - lo;
    let start = lo % slots;
    if start + n <= slots {
        vec![(start, n)]
    } else {
        vec![(start, slots - start), (0, n - (slots - start))]
    }
}

impl DecodeStateApi for Gemma4State {
    type Snapshot = Gemma4Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        // Stale cache rows are never read: attention reads positions below
        // the state's, which a later feed writes first.
        self.pos = 0;
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        ensure!(
            tokens <= self.max_capacity,
            "a session of {tokens} tokens exceeds the {} rows this model serves",
            self.max_capacity
        );
        let capacity =
            tokens.max(self.capacity.saturating_mul(2)).min(self.max_capacity);
        for l in &mut self.layers {
            if l.swa {
                continue;
            }
            let grow = |t: &Tensor| -> Result<Tensor> {
                let per_tok = t.numel() / l.slots;
                let n = Tensor::zeros(ctx, &[capacity * per_tok], DType::F16)?;
                host_copy(t, &n, self.pos * per_tok * 2);
                Ok(n)
            };
            *l = LayerCache {
                k: grow(&l.k)?,
                v: grow(&l.v)?,
                slots: capacity,
                swa: false,
            };
        }
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        Gemma4State::bytes(self)
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Gemma4Snapshot> {
        let lo = self.pos.saturating_sub(self.window);
        let mut tensors = Vec::new();
        for (ring, slots, per_slot) in rings(self) {
            // A fixed size (the window's), whatever the position: the
            // server sizes checkpoints from the shapes.
            let t = Tensor::zeros(ctx, &[self.window * per_slot / 2], DType::F16)?;
            let mut at = 0;
            for (start, count) in ring_ranges(lo, self.pos, slots) {
                host_copy_at(ring, start * per_slot, &t, at, count * per_slot);
                at += count * per_slot;
            }
            tensors.push(t);
        }
        let h_last = Tensor::zeros(ctx, &[self.h_last.numel()], DType::F32)?;
        host_copy(&self.h_last, &h_last, h_last.byte_len());
        Ok(Gemma4Snapshot { pos: self.pos, tensors, h_last })
    }

    fn restore(
        &mut self,
        _ctx: &MetalContext,
        snapshot: &Gemma4Snapshot,
    ) -> Result<()> {
        ensure!(snapshot.pos <= self.capacity, "snapshot past the state's capacity");
        let lo = snapshot.pos.saturating_sub(self.window);
        for ((ring, slots, per_slot), src) in
            rings(self).into_iter().zip(&snapshot.tensors)
        {
            let mut at = 0;
            for (start, count) in ring_ranges(lo, snapshot.pos, slots) {
                host_copy_at(src, at, ring, start * per_slot, count * per_slot);
                at += count * per_slot;
            }
        }
        host_copy(&snapshot.h_last, &self.h_last, self.h_last.byte_len());
        self.pos = snapshot.pos;
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
        for (dst, src) in self.layers.iter().zip(&from.layers) {
            if src.swa {
                continue;
            }
            host_copy(&src.k, &dst.k, tokens * (src.k.numel() / src.slots) * 2);
            host_copy(&src.v, &dst.v, tokens * (src.v.numel() / src.slots) * 2);
        }
        Ok(())
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    // The step's last row's logits, which is what the trait means.
    #[allow(clippy::misnamed_getters)]
    fn logits(&self) -> &Tensor {
        &self.logits_last
    }

    fn begin_request(&self) {
        self.sampler().reset_counts();
    }
}

impl Gemma4Model {
    /// Feeds `tokens` in runs of `prefill_rows`, drawing after the last
    /// when `draw`; before committing each run, stops if `stop` says so
    /// (the state then sits at a run boundary). Returns the tokens fed.
    fn prefill_chunks(
        &self,
        ctx: &MetalContext,
        state: &mut Gemma4State,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        ensure!(!tokens.is_empty(), "empty prefill");
        state.ensure_capacity(ctx, state.pos + tokens.len())?;
        let chunks: Vec<&[u32]> = tokens.chunks(self.prefill_rows).collect();
        let mut fed = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            if stop() {
                return Ok(fed);
            }
            let last = i + 1 == chunks.len();
            scratch.stage_tokens(chunk)?;
            let logits =
                if last && draw.is_some() { Logits::Last } else { Logits::None };
            let sc: &Scratch = scratch;
            self.run_rows(ctx, state, sc, chunk.len(), logits, |pass| {
                if let (true, Some(d)) = (last, draw) {
                    let out = sc.next_token.view(0, &[1])?;
                    sample_f32(
                        ctx,
                        pass,
                        &sc.logits_last,
                        sc.sampler(),
                        d.params,
                        d.step,
                        &out,
                    )?;
                }
                Ok(())
            })?;
            state.pos += chunk.len();
            fed += chunk.len();
        }
        Ok(fed)
    }
}

/// What the passes the server keeps in flight allocate for their arguments
/// (the backend's recycled 1 MiB arenas, one a pass; a Gemma step's take
/// well under one), counted by the memory-limit fit.
const PASS_ARENAS: usize = 4 << 20;

/// A draft is proposed only when the draft model's probability of it is
/// at least this; the chain stops at the first that is not.
const DRAFT_MIN_PROB: f32 = 0.3;

/// Drafts a verify pass from position `pos` may carry within
/// `max_capacity` rows: it feeds the pending token and the drafts.
pub(crate) fn drafts_fit(pos: usize, max_capacity: usize, n: usize) -> usize {
    n.min(max_capacity.saturating_sub(pos + 1))
}

impl Gemma4Model {
    /// Up to `n` greedy drafts following `token` (at the state's position),
    /// from the state's last hidden (`h_last`); the proposals end before the
    /// first draft below the probability gate, and at the state's last
    /// rows. Waits.
    fn propose(
        &self,
        ctx: &MetalContext,
        state: &Gemma4State,
        scratch: &Scratch,
        token: u32,
        n: usize,
    ) -> Result<Vec<u32>> {
        let Some(draft) = &self.draft else { return Ok(Vec::new()) };
        check_tokens(&[token], self.cfg.vocab)?;
        let n = drafts_fit(state.pos, state.max_capacity, n);
        if n == 0 {
            return Ok(Vec::new());
        }
        draft.stage_token(token)?;
        let pass = self.begin(ctx)?;
        draft.encode(ctx, &pass, self, state, scratch, &state.h_last, n)?;
        pass.commit_wait()?;
        let (tokens, probs) = draft.read(n)?;
        let keep = probs.iter().take_while(|&&p| p >= self.draft_min_prob).count();
        Ok(tokens[..keep].to_vec())
    }
}

impl Gemma4Model {
    /// Bytes of the scratch's attention partials per token of context.
    /// The most GPU memory one session holds on its way to `capacity` rows
    /// with `checkpoints` snapshots: its caches at that capacity plus the
    /// one global layer's K and V that `ensure_capacity` holds twice while
    /// it replaces them (the old ones are freed layer by layer).
    pub fn session_peak(&self, capacity: usize, checkpoints: usize) -> u64 {
        super::model::session_peak(&self.cfg, capacity, checkpoints)
    }

    /// The longest context (a multiple of 256 tokens) whose session's peak
    /// ([`Self::session_peak`], with the verify lookahead and `checkpoints`
    /// snapshots) and the scratch's growth fit `limit` bytes next to
    /// `allocated` bytes (the weights and the scratch); `context_cap`
    /// becomes it, so the server caps --max-seq there, and the session
    /// store keeps that peak free for the request's session.
    pub fn fit_context(
        &mut self,
        limit: u64,
        allocated: u64,
        checkpoints: usize,
    ) -> Result<usize> {
        let room = limit.saturating_sub(allocated);
        let context = fit_tokens(&self.cfg, room, checkpoints);
        ensure!(
            context >= 4096,
            "--memory-limit-gb {:.1} leaves {:.2} GB after {:.1} GB of weights and scratch: a context \
             of {context} tokens (lily wants at least 4096)",
            limit as f64 / (1u64 << 30) as f64,
            room as f64 / 1e9,
            allocated as f64 / 1e9
        );
        self.context_cap = context;
        self.acquire_reserve =
            Some(self.session_peak(capacity_cap(context), checkpoints));
        Ok(context)
    }
}

/// GPU bytes a token of context adds to the scratch: the decode attention's
/// partials (MAX_ROWS rows, a chunk of ATTN_CHUNK keys).
pub(crate) fn partial_bytes_per_token(cfg: &super::config::Config) -> u64 {
    let d = cfg.head_dim.max(cfg.head_dim_swa);
    (4 * crate::kernels::gemma4::attn_partial_len(MAX_ROWS, cfg.n_head, d, 1)
        / crate::kernels::gemma4::ATTN_CHUNK) as u64
        + 1
}

/// The longest context (a multiple of 256 tokens, at most the model's and
/// MAX_SEQ) whose session at its peak (`session_peak` at `capacity_cap` of
/// it, with `checkpoints` snapshots) and attention partials fit `room`
/// bytes.
pub(crate) fn fit_tokens(cfg: &super::config::Config, room: u64, checkpoints: usize) -> usize {
    let peak = |c: usize| super::model::session_peak(cfg, c, checkpoints);
    let per_token = (peak(256) - peak(0)) / 256 + partial_bytes_per_token(cfg);
    let rows = room.saturating_sub(peak(0)) / per_token.max(1);
    let context = (rows as usize).saturating_sub(LOOKAHEAD) / 256 * 256;
    context.min(cfg.context_length).min(MAX_SEQ)
}

/// The prompt run a memory limit takes, from each run's fitted context
/// (`(rows, context)`, longest run first and 256 rows last; `None` where
/// less than 4096 tokens fit): the longest run that keeps a context of
/// 32K tokens, or that of the last run when that is less (a 16 GB tier
/// keeps its context and short runs). `None` when nothing fits.
pub(crate) fn choose_prompt_run(fits: &[(usize, Option<usize>)]) -> Option<usize> {
    let floor = fits.last()?.1?.min(32768);
    fits.iter().position(|f| f.1.is_some_and(|c| c >= floor))
}

impl LanguageModel for Gemma4Model {
    type State = Gemma4State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "gemma-4-26b-a4b";

    fn load(ctx: &MetalContext, path: &Path, options: &LoadOptions) -> Result<Self> {
        let mut model = Gemma4Model::load(ctx, path)?;
        model.memory_limit = options.memory_limit;
        if options.mtp_drafts > 0 {
            match super::draft::find_beside(path, model.cfg.hidden)? {
                Some(dp) => {
                    let d = super::draft::Gemma4Draft::load(ctx, &dp, &model.cfg)?;
                    eprintln!(
                        "draft model: {} ({:.2} GB)",
                        dp.display(),
                        d.bytes() as f64 / 1e9
                    );
                    model.draft = Some(d);
                    model.drafts = options.mtp_drafts.min(super::draft::MAX_DRAFTS);
                    model.draft_min_prob = DRAFT_MIN_PROB;
                }
                None => eprintln!(
                    "draft model: none beside {} (a gemma4-assistant GGUF); no speculative decoding",
                    path.display()
                ),
            }
        }
        if let Some(limit) = model.memory_limit {
            // The weights, a scratch as the server allocates it (made here
            // once to measure it; its attention partials grow with the
            // context, which fit_context counts per token) and the passes'
            // argument arenas, then the longest context whose session fits
            // the rest: the server caps --max-seq there.
            let before = ctx.current_allocated();
            let checkpoints = options.session_checkpoints.unwrap_or(
                crate::serve::CHECKPOINTS_PER_SESSION
                    + crate::serve::DECODE_CHECKPOINTS_PER_SESSION,
            );
            // The run size (choose_prompt_run): scratch grows ~0.3 MB a
            // run row, and longer runs prefill faster (PREFILL_ROWS).
            let mut fits = Vec::new();
            for rows in [PREFILL_ROWS, PREFILL_ROWS / 2, PREFILL_ROWS / 4] {
                model.prefill_rows = rows;
                let probe = model.new_scratch(ctx, 1)?;
                let scratch = ctx.current_allocated().saturating_sub(before);
                drop(probe);
                let allocated = before + scratch + PASS_ARENAS;
                let context = model.fit_context(limit, allocated as u64, checkpoints).ok();
                fits.push((rows, allocated, context));
            }
            let contexts: Vec<(usize, Option<usize>)> =
                fits.iter().map(|&(rows, _, c)| (rows, c)).collect();
            let pick = choose_prompt_run(&contexts).unwrap_or(fits.len() - 1);
            let (rows, allocated, _) = fits[pick];
            model.prefill_rows = rows;
            let context = model.fit_context(limit, allocated as u64, checkpoints)?;
            eprintln!(
                "gemma4: --memory-limit-gb {:.1} ({:.1} GB) leaves {:.2} GB after {:.1} GB of weights and \
                 scratch: contexts of up to {context} tokens (one session of at most {:.2} GB with \
                 {checkpoints} snapshots), prompt runs of {rows} rows; --max-seq is capped there",
                limit as f64 / (1u64 << 30) as f64,
                limit as f64 / 1e9,
                limit.saturating_sub(allocated as u64) as f64 / 1e9,
                allocated as f64 / 1e9,
                model.session_peak(capacity_cap(context), checkpoints) as f64 / 1e9,
            );
        }
        Ok(model)
    }

    fn max_drafts(&self) -> usize {
        if self.draft.is_some() { self.drafts } else { 0 }
    }

    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut Gemma4State,
        scratch: &mut Scratch,
        first: u32,
        drafts: usize,
        _params: &SamplingParams,
        _step0: usize,
    ) -> Result<Vec<u32>> {
        self.propose(ctx, state, scratch, first, drafts)
    }

    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Gemma4State,
        scratch: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        _next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        let rows = 1 + drafts.len();
        ensure!(rows <= MAX_ROWS, "verify of {rows} rows");
        ensure!(state.spec.is_none(), "a verify pass is already pending");
        state.ensure_capacity(ctx, state.pos + rows)?;
        check_tokens(&[pending], self.cfg.vocab)?;
        check_tokens(drafts, self.cfg.vocab)?;
        let tokens: Vec<u32> =
            std::iter::once(pending).chain(drafts.iter().copied()).collect();
        scratch.stage_tokens(&tokens)?;
        let vocab = self.cfg.vocab;
        let sc: &Scratch = scratch;
        self.run_rows(ctx, state, sc, rows, Logits::All, |pass| {
            for r in 0..rows {
                let logits = sc.logits.view(r * vocab, &[vocab])?;
                let out = sc.draws.view(r, &[1])?;
                sample_f32(ctx, pass, &logits, sc.sampler(), params, step0 + r, &out)?;
                pass.memory_barrier()?;
            }
            Ok(())
        })?;
        let draws = scratch.draws.to_u32()?[..rows].to_vec();
        state.spec = Some((state.pos, rows, draws.clone()));
        state.pos += rows;
        // The next drafts are drawn in finish_speculation, once the kept
        // count is known.
        Ok((draws, ctx.begin()?.commit()?))
    }

    fn finish_speculation(
        &self,
        ctx: &MetalContext,
        state: &mut Gemma4State,
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
        // Rows past the accepted ones wrote cache slots that later
        // positions overwrite before anything reads them.
        state.pos = pos0 + 1 + accepted;
        // The pass left its last row's hidden in h_last; the draft model's
        // input is the last kept row's (the GPU is idle on both).
        if accepted + 1 < rows {
            let h = scratch.h_row(accepted, self.cfg.hidden)?;
            host_copy(&h, &state.h_last, state.h_last.byte_len());
        }
        for &token in &draws[accepted + 1..] {
            scratch.sampler().uncount(token);
        }
        if !propose {
            return Ok(Vec::new());
        }
        self.propose(ctx, state, scratch, draws[accepted], self.drafts)
    }

    fn max_position_embeddings(&self) -> usize {
        self.context_cap
    }

    fn memory_limit(&self) -> Option<u64> {
        self.memory_limit
    }

    fn acquire_reserve(&self) -> Option<u64> {
        self.acquire_reserve
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        super::tokenizer::stop_ids(&self.gguf).unwrap_or_else(|_| vec![self.cfg.eos])
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    /// The global layers' f16 K and V per token (the sliding-window rings
    /// are a fixed size).
    fn bytes_per_token(&self) -> usize {
        (0..self.cfg.n_layer)
            .filter(|&il| !self.cfg.swa[il])
            .map(|il| 2 * (self.cfg.k_cache_width(il) + self.cfg.kv_width(il)))
            .sum()
    }

    /// One session's sliding-window rings: the tiers' budgets are well
    /// below the server's Qwen-sized floor.
    fn cache_budget_floor(&self) -> Option<usize> {
        let slots = super::model::ring_slots(&self.cfg);
        Some(
            (0..self.cfg.n_layer)
                .filter(|&il| self.cfg.swa[il])
                .map(|il| 2 * slots * (self.cfg.k_cache_width(il) + self.cfg.kv_width(il)))
                .sum(),
        )
    }

    /// The global layers' caches for `capacity` tokens, the rings, and
    /// `checkpoints` snapshots of the rings' windows.
    fn session_bytes(&self, capacity: usize, checkpoints: usize) -> Option<u64> {
        Some(super::model::session_bytes(&self.cfg, capacity, checkpoints))
    }

    fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        Gemma4Model::weight_buffers(self)
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Gemma4State> {
        Gemma4Model::new_state(self, ctx, capacity.max(1))
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        // States of a `capacity_tokens` context grow LOOKAHEAD rows past it.
        self.new_scratch(ctx, capacity_cap(capacity_tokens.max(1)))
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut Gemma4State,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        self.prefill_chunks(ctx, state, scratch, tokens, draw, &|| false).map(drop)
    }

    fn prefill_chunk_tokens(&self) -> Option<usize> {
        Some(self.prefill_rows)
    }

    fn prefill_until(
        &self,
        ctx: &MetalContext,
        state: &mut Gemma4State,
        scratch: &mut Scratch,
        tokens: &[u32],
        vision: Option<&crate::qwen4exp::VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        ensure!(vision.is_none(), "lily serves Gemma 4 text-only");
        self.prefill_chunks(ctx, state, scratch, tokens, None, stop)
    }

    fn prepare_step_inputs(
        &self,
        _state: &mut Gemma4State,
        _scratch: &Scratch,
        token: u32,
    ) -> Result<()> {
        check_tokens(&[token], self.cfg.vocab)?;
        Ok(())
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &Gemma4State,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        ensure!(state.pos < state.capacity, "decode step past the state's capacity");
        let pass = self.begin(ctx)?;
        let token = scratch.next_token.view(slot_in, &[1])?;
        self.encode_layers(
            ctx,
            &pass,
            state,
            scratch,
            &token,
            1,
            Logits::Last,
            0..self.cfg.n_layer,
        )?;
        let out = scratch.next_token.view(slot_out, &[1])?;
        sample_f32(
            ctx,
            &pass,
            &scratch.logits_last,
            scratch.sampler(),
            draw.params,
            draw.step,
            &out,
        )?;
        pass.end()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gemma4/engine.rs"]
mod tests;
