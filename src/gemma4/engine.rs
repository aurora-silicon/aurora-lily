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
    Gemma4Model, Gemma4State, LayerCache, Logits, PREFILL_ROWS, Scratch,
};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::sample::sample_f32;
use crate::metal::{EncodedPass, MetalContext};
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

fn clone_tensor(ctx: &MetalContext, t: &Tensor) -> Result<Tensor> {
    let c = Tensor::zeros(ctx, t.shape(), t.dtype())?;
    host_copy(t, &c, t.byte_len());
    Ok(c)
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
/// buffer.
pub struct Gemma4Snapshot {
    pos: usize,
    tensors: Vec<Tensor>,
}

impl SnapshotApi for Gemma4Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.tensors.iter().map(Tensor::byte_len).sum()
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
        let capacity = tokens.max(self.capacity * 2);
        for l in &mut self.layers {
            if l.swa {
                continue;
            }
            let per_tok = l.k.numel() / l.slots;
            let grow = |t: &Tensor| -> Result<Tensor> {
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
        Ok(Gemma4Snapshot { pos: self.pos, tensors })
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
            let per_tok = src.k.numel() / src.slots;
            host_copy(&src.k, &dst.k, tokens * per_tok * 2);
            host_copy(&src.v, &dst.v, tokens * per_tok * 2);
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
        self.sampler().reset_counts();
    }
}

impl Gemma4Model {
    /// Feeds `tokens` in runs of [`PREFILL_ROWS`], drawing after the last
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
        let chunks: Vec<&[u32]> = tokens.chunks(PREFILL_ROWS).collect();
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

impl LanguageModel for Gemma4Model {
    type State = Gemma4State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "gemma-4-26b-a4b";

    fn load(ctx: &MetalContext, path: &Path, _options: &LoadOptions) -> Result<Self> {
        Gemma4Model::load(ctx, path)
    }

    fn max_position_embeddings(&self) -> usize {
        self.cfg.context_length
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
            .map(|il| 4 * self.cfg.kv_width(il))
            .sum()
    }

    /// One session's sliding-window rings: the tiers' budgets are well
    /// below the server's Qwen-sized floor.
    fn cache_budget_floor(&self) -> Option<usize> {
        let slots = super::model::ring_slots(&self.cfg);
        Some(
            (0..self.cfg.n_layer)
                .filter(|&il| self.cfg.swa[il])
                .map(|il| 4 * slots * self.cfg.kv_width(il))
                .sum(),
        )
    }

    /// The global layers' caches for `capacity` tokens, the rings, and
    /// `checkpoints` snapshots of the rings' windows.
    fn session_bytes(&self, capacity: usize, checkpoints: usize) -> Option<u64> {
        let cfg = &self.cfg;
        let slots = super::model::ring_slots(cfg);
        let ring: usize = (0..cfg.n_layer)
            .filter(|&il| cfg.swa[il])
            .map(|il| 4 * cfg.kv_width(il))
            .sum();
        Some(
            (self.bytes_per_token() * capacity
                + ring * slots
                + checkpoints * ring * cfg.window) as u64,
        )
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
        self.new_scratch(ctx, capacity_tokens.max(1))
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
        Some(PREFILL_ROWS)
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
        _token: u32,
    ) -> Result<()> {
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
