//! Qwen3.6 behind the engine's [`LanguageModel`] interface: what the server
//! and the bench drive. One session at a time, no batching across sessions.

use std::path::Path;

use anyhow::{Result, ensure};

use super::model::{LayerState, Logits, PREFILL_ROWS, Qwen36Model, Qwen36State, Scratch};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::attention::KvCache;
use crate::kernels::sample::sample_f32;
use crate::metal::{EncodedPass, MetalContext};
use crate::tensor::Tensor;

/// Copies `src`'s first `bytes` into `dst` through host memory (both
/// shared-storage, the GPU idle on them).
fn host_copy(src: &Tensor, dst: &Tensor, bytes: usize) {
    host_copy_at(src, 0, dst, 0, bytes);
}

/// Copies `bytes` from byte `from` of `src` to byte `to` of `dst`.
fn host_copy_at(src: &Tensor, from: usize, dst: &Tensor, to: usize, bytes: usize) {
    assert!(from + bytes <= src.byte_len() && to + bytes <= dst.byte_len());
    // SAFETY: both buffers are host-visible and hold the ranges; the GPU is
    // idle on them by the callers' contracts.
    unsafe {
        core::ptr::copy_nonoverlapping(
            src.contents_ptr().add(from),
            dst.contents_ptr().add(to),
            bytes,
        );
    }
}

fn clone_tensor(ctx: &MetalContext, t: &Tensor) -> Result<Tensor> {
    let c = Tensor::zeros(ctx, t.shape(), t.dtype())?;
    host_copy(t, &c, t.byte_len());
    Ok(c)
}

/// Copies the first `tokens` rows of every head of a `[heads, capacity, D]`
/// cache into another of a different capacity.
fn copy_cache_rows(src: &KvCache, dst: &KvCache, tokens: usize) {
    for (s, d) in src.tensors().zip(dst.tensors()) {
        let (heads, sc, dc) = (s.shape()[0], s.shape()[1], d.shape()[1]);
        let row = s.byte_len() / (heads * sc);
        for h in 0..heads {
            host_copy_at(s, h * sc * row, d, h * dc * row, tokens * row);
        }
    }
}

/// The recurrent part of a state: the GDN conv windows and states.
pub struct Qwen36Snapshot {
    pos: usize,
    tensors: Vec<Tensor>,
}

impl SnapshotApi for Qwen36Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.tensors.iter().map(Tensor::byte_len).sum()
    }
}

/// The tensors a snapshot holds, in a fixed order.
fn recurrent(st: &Qwen36State) -> Vec<&Tensor> {
    st.layers
        .iter()
        .flat_map(|l| match l {
            LayerState::Gdn { conv, s, .. } => vec![conv, s],
            LayerState::Attn { .. } => vec![],
        })
        .collect()
}

impl DecodeStateApi for Qwen36State {
    type Snapshot = Qwen36Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        Qwen36State::reset(self);
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        let capacity = tokens.max(self.capacity * 2).next_multiple_of(4);
        for l in &mut self.layers {
            if let LayerState::Attn { k, v } = l {
                for c in [k, v] {
                    let (heads, _, d) = c.kv().shape();
                    let format = c.kv().format()?;
                    let n = KvCache::zeros(ctx, format, heads, capacity, d)?;
                    copy_cache_rows(c, &n, self.pos);
                    *c = n;
                }
            }
        }
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Gdn { conv, s, conv_alt, s_alt } => {
                    conv.byte_len() + s.byte_len() + conv_alt.byte_len() + s_alt.byte_len()
                }
                LayerState::Attn { k, v } => k.bytes() + v.bytes(),
            })
            .sum()
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Qwen36Snapshot> {
        let tensors =
            recurrent(self).into_iter().map(|t| clone_tensor(ctx, t)).collect::<Result<_>>()?;
        Ok(Qwen36Snapshot { pos: self.pos, tensors })
    }

    fn restore(&mut self, _ctx: &MetalContext, snapshot: &Qwen36Snapshot) -> Result<()> {
        ensure!(snapshot.pos <= self.capacity, "snapshot past the state's capacity");
        for (dst, src) in recurrent(self).into_iter().zip(&snapshot.tensors) {
            host_copy(src, dst, src.byte_len());
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
            if let (LayerState::Attn { k: dk, v: dv }, LayerState::Attn { k: sk, v: sv }) =
                (dst, src)
            {
                copy_cache_rows(sk, dk, tokens);
                copy_cache_rows(sv, dv, tokens);
            }
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

impl Qwen36Model {
    /// Feeds `tokens` in runs of [`PREFILL_ROWS`], drawing after the last
    /// when `draw`; before committing each run, stops if `stop` says so
    /// (the state then sits at a run boundary). Returns the tokens fed.
    fn prefill_chunks(
        &self,
        ctx: &MetalContext,
        state: &mut Qwen36State,
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
            let logits = if last && draw.is_some() { Logits::Last } else { Logits::None };
            let sc: &Scratch = scratch;
            self.run_rows(ctx, state, sc, chunk.len(), logits, None, |pass| {
                if let (true, Some(d)) = (last, draw) {
                    let out = sc.next_token.view(0, &[1])?;
                    sample_f32(ctx, pass, &sc.logits_last, sc.sampler(), d.params, d.step, &out)?;
                }
                Ok(())
            })?;
            state.pos += chunk.len();
            fed += chunk.len();
        }
        Ok(fed)
    }
}

impl LanguageModel for Qwen36Model {
    type State = Qwen36State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "qwen3.6-35b-a3b";

    fn load(ctx: &MetalContext, path: &Path, _options: &LoadOptions) -> Result<Self> {
        Qwen36Model::load(ctx, path)
    }

    fn max_position_embeddings(&self) -> usize {
        self.cfg.context_length
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        vec![self.cfg.eos, self.cfg.pad]
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn bytes_per_token(&self) -> usize {
        // K and V rows of every full-attention layer.
        2 * self.cfg.attn_layers() * self.kv_format.row_bytes(self.cfg.head_dim)
            * self.cfg.n_head_kv
    }

    fn kv_format(&self) -> crate::kernels::attention::KvFormat {
        self.kv_format
    }

    fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        Qwen36Model::weight_buffers(self)
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Qwen36State> {
        Qwen36Model::new_state(self, ctx, capacity.max(4))
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        let _ = capacity_tokens;
        self.new_scratch(ctx)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut Qwen36State,
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
        state: &mut Qwen36State,
        scratch: &mut Scratch,
        tokens: &[u32],
        vision: Option<&crate::qwen4exp::VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        ensure!(vision.is_none(), "lily serves Qwen3.6 as text only");
        self.prefill_chunks(ctx, state, scratch, tokens, None, stop)
    }

    fn prepare_step_inputs(
        &self,
        _state: &mut Qwen36State,
        _scratch: &Scratch,
        _token: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &Qwen36State,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        ensure!(state.pos < state.capacity, "decode step past the state's capacity");
        let pass = self.begin(ctx)?;
        let token = scratch.next_token.view(slot_in, &[1])?;
        let n = self.cfg.n_layer;
        self.encode_layers(ctx, &pass, state, scratch, &token, 1, Logits::Last, None, 0..n)?;
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
