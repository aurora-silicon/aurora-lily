//! Qwen3.6 behind the engine's [`LanguageModel`] interface: what the server
//! and the bench drive. One session at a time, no batching across sessions.

use std::path::Path;

use anyhow::{Result, ensure};

use super::model::{
    LayerState, Logits, PREFILL_ROWS_LIMITED, Qwen36Model, Qwen36State, Scratch,
};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::attention::KvCache;
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{EncodedPass, MetalContext, PendingPass};
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

/// The capacity a state of `capacity` rows grows to for `tokens`: doubling,
/// at most `max` (both multiples of 4; `tokens <= max`).
pub(crate) fn grown_capacity(capacity: usize, tokens: usize, max: usize) -> usize {
    tokens.max(capacity.saturating_mul(2)).next_multiple_of(4).min(max)
}

/// Whether a draft may be proposed for the token at `p`: its verify pass
/// feeds rows `p` and `p + 1`, and the head's catch-up after it writes its
/// row `p + 2`.
pub(crate) fn draft_fits(p: usize, max_capacity: usize) -> bool {
    p + 3 <= max_capacity
}

/// The MTP pairs a finished verify pass from `pos0` writes, head rows
/// `pos0 + 1 ..= pos0 + pairs`, within `capacity` rows: the kept rows', and
/// with `propose` the new pending token's, whose output is the draft.
/// Returns the pairs and whether the draft pair is among them.
pub(crate) fn mtp_pairs(
    pos0: usize,
    accepted: usize,
    propose: bool,
    capacity: usize,
) -> (usize, bool) {
    let want = accepted + usize::from(propose);
    let room = capacity.saturating_sub(pos0 + 1);
    (want.min(room), propose && want <= room)
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

/// The tensors a snapshot holds, in a fixed order: the GDN windows and
/// states, the MTP head's pending row.
fn recurrent(st: &Qwen36State) -> Vec<&Tensor> {
    st.layers
        .iter()
        .flat_map(|l| match l {
            LayerState::Gdn { conv, s, .. } => vec![conv, s],
            LayerState::Attn { .. } => vec![],
        })
        .chain(st.mtp.as_ref().map(|m| &m.pending_h))
        .collect()
}

/// The K/V caches of a state that grow with its capacity.
fn caches_mut(st: &mut Qwen36State) -> Vec<&mut KvCache> {
    let mut v: Vec<&mut KvCache> = Vec::new();
    for l in &mut st.layers {
        if let LayerState::Attn { k, v: vv } = l {
            v.push(k);
            v.push(vv);
        }
    }
    if let Some(m) = &mut st.mtp {
        v.push(&mut m.k);
        v.push(&mut m.v);
    }
    v
}

fn caches(st: &Qwen36State) -> Vec<&KvCache> {
    let mut v: Vec<&KvCache> = Vec::new();
    for l in &st.layers {
        if let LayerState::Attn { k, v: vv } = l {
            v.push(k);
            v.push(vv);
        }
    }
    if let Some(m) = &st.mtp {
        v.push(&m.k);
        v.push(&m.v);
    }
    v
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
        ensure!(
            tokens <= self.max_capacity,
            "a session of {tokens} tokens exceeds the {} rows this model serves",
            self.max_capacity
        );
        let capacity = grown_capacity(self.capacity, tokens, self.max_capacity);
        let rows =
            self.pos.max(self.mtp.as_ref().map_or(0, |m| m.pos)).min(self.capacity);
        for c in caches_mut(self) {
            let (heads, _, d) = c.kv().shape();
            let format = c.kv().format()?;
            let n = KvCache::zeros(ctx, format, heads, capacity, d)?;
            copy_cache_rows(c, &n, rows);
            *c = n;
        }
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Gdn { conv, s, conv_alt, s_alt } => {
                    conv.byte_len()
                        + s.byte_len()
                        + conv_alt.byte_len()
                        + s_alt.byte_len()
                }
                LayerState::Attn { k, v } => k.bytes() + v.bytes(),
            })
            .sum::<usize>()
            + self
                .mtp
                .as_ref()
                .map_or(0, |m| m.k.bytes() + m.v.bytes() + m.pending_h.byte_len())
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Qwen36Snapshot> {
        let tensors = recurrent(self)
            .into_iter()
            .map(|t| clone_tensor(ctx, t))
            .collect::<Result<_>>()?;
        Ok(Qwen36Snapshot { pos: self.pos, tensors })
    }

    fn restore(
        &mut self,
        _ctx: &MetalContext,
        snapshot: &Qwen36Snapshot,
    ) -> Result<()> {
        ensure!(snapshot.pos <= self.capacity, "snapshot past the state's capacity");
        for (dst, src) in recurrent(self).into_iter().zip(&snapshot.tensors) {
            host_copy(src, dst, src.byte_len());
        }
        self.pos = snapshot.pos;
        if let Some(m) = &mut self.mtp {
            m.pos = snapshot.pos;
        }
        self.spec = None;
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
        ensure!(
            self.mtp.is_some() == from.mtp.is_some(),
            "prefix copy between states with and without the MTP head"
        );
        for (dst, src) in caches(self).into_iter().zip(caches(from)) {
            copy_cache_rows(src, dst, tokens);
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
    /// Feeds `tokens` in runs of [`Qwen36Model::prefill_rows`], drawing after the last
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
        super::model::check_tokens(tokens, self.cfg.vocab)?;
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
            let mtp = self.mtp.is_some() && state.mtp.is_some();
            if let Some(m) = &state.mtp {
                // The head's pairs of this run start with the session's
                // pending row.
                Qwen36Model::hbuf(sc)
                    .view(0, &[self.cfg.hidden])?
                    .write_bytes(m.pending_h.contents())?;
            }
            let pos0 = state.pos;
            let st: &Qwen36State = state;
            self.run_rows(ctx, st, sc, chunk.len(), logits, None, |pass| {
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
                if mtp {
                    pass.memory_barrier()?;
                    self.encode_mtp_prompt(ctx, pass, st, sc, pos0, chunk.len())?;
                }
                Ok(())
            })?;
            if let Some(m) = &mut state.mtp {
                self.copy_row(Qwen36Model::hbuf(sc), chunk.len(), &m.pending_h)?;
                m.pos = pos0 + chunk.len();
            }
            state.pos += chunk.len();
            fed += chunk.len();
        }
        Ok(fed)
    }
}

/// A draft whose probability under the MTP head is below this is not
/// proposed (GLM's heuristic, `glm53::engine`): the raw MTP softmax at
/// temperature 1, not a calibrated acceptance probability.
const DRAFT_MIN_PROB: f32 = 0.3;

impl Qwen36Model {
    /// The MTP step's draft as a proposal list: empty when its probability
    /// is below [`Qwen36Model::draft_min_prob`].
    fn proposal(&self, scratch: &Scratch) -> Result<Vec<u32>> {
        let p = scratch.mtp_prob.to_f32()?[0];
        let draft = scratch.mtp_io.to_u32()?[1];
        Ok(if p >= self.draft_min_prob { vec![draft] } else { Vec::new() })
    }
}

impl LanguageModel for Qwen36Model {
    type State = Qwen36State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "qwen3.6-35b-a3b";

    fn load(ctx: &MetalContext, path: &Path, options: &LoadOptions) -> Result<Self> {
        let mut model = Qwen36Model::load(ctx, path)?;
        model.draft_min_prob = DRAFT_MIN_PROB;
        model.memory_limit = options.memory_limit;
        if model.memory_limit.is_some() {
            model.prefill_rows = PREFILL_ROWS_LIMITED;
        }
        if options.mtp_drafts > 0 {
            match model.find_mtp(path)? {
                Some(p) => {
                    model.load_mtp(ctx, &p)?;
                    eprintln!("qwen3.6: MTP head from {}", p.display());
                }
                None => eprintln!(
                    "qwen3.6: no mtp-*.gguf beside {}: decoding without drafts",
                    path.display()
                ),
            }
        }
        if let Some(limit) = model.memory_limit {
            // The weights and a scratch as the server allocates it (made
            // here once to measure it), then the longest context whose
            // session fits the rest: the server caps --max-seq there.
            let before = ctx.current_allocated();
            let probe = Qwen36Model::new_scratch(&model, ctx)?;
            let scratch = ctx.current_allocated().saturating_sub(before);
            drop(probe);
            let allocated = before + scratch;
            let checkpoints = crate::serve::CHECKPOINTS_PER_SESSION
                + crate::serve::DECODE_CHECKPOINTS_PER_SESSION;
            let context = model.fit_context(limit, allocated as u64, checkpoints)?;
            eprintln!(
                "qwen3.6: --memory-limit-gb {:.1} ({:.1} GB) leaves {:.2} GB after {:.1} GB of weights and \
                 scratch: contexts of up to {context} tokens (one session of {:.2} GB with \
                 {checkpoints} snapshots); --max-seq is capped there",
                limit as f64 / (1u64 << 30) as f64,
                limit as f64 / 1e9,
                limit.saturating_sub(allocated as u64) as f64 / 1e9,
                allocated as f64 / 1e9,
                model.session_bytes(super::model::capacity_cap(context), checkpoints)
                    as f64
                    / 1e9,
            );
        }
        Ok(model)
    }

    /// One draft a step: the MTP block predicts one token ahead.
    fn max_drafts(&self) -> usize {
        usize::from(self.mtp.is_some())
    }

    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut Qwen36State,
        scratch: &mut Scratch,
        first: u32,
        _drafts: usize,
        _params: &SamplingParams,
        _step0: usize,
    ) -> Result<Vec<u32>> {
        super::model::check_tokens(&[first], self.cfg.vocab)?;
        let p = state.pos;
        if !draft_fits(p, state.max_capacity) {
            // The context's last rows: plain steps to the end.
            return Ok(Vec::new());
        }
        state.ensure_capacity(ctx, p + 3)?;
        let m = state.mtp.as_mut().ok_or_else(|| anyhow::anyhow!("no MTP state"))?;
        // The pair (hidden at p - 1, first) at head row p; a head that fell
        // behind (rows fed without it) resumes here.
        m.pos = p;
        crate::glm53::model::write_words(&scratch.mtp_io, &[first])?;
        Qwen36Model::hbuf(scratch)
            .view(0, &[self.cfg.hidden])?
            .write_bytes(m.pending_h.contents())?;
        let sc: &Scratch = scratch;
        let st: &Qwen36State = state;
        let pass = self.begin(ctx)?;
        let token = sc.mtp_io.view(0, &[1])?;
        let hin = Qwen36Model::hbuf(sc).view(0, &[self.cfg.hidden])?;
        self.encode_mtp(ctx, &pass, st, sc, &token, &hin, p, 1, true)?;
        pass.commit_wait()?;
        if let Some(m) = &mut state.mtp {
            m.pos = p + 1;
        }
        self.proposal(scratch)
    }

    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Qwen36State,
        scratch: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        _next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        let rows = 1 + drafts.len();
        ensure!(rows <= 2, "Qwen3.6 verifies one draft a step, got {}", drafts.len());
        ensure!(state.spec.is_none(), "a verify pass is already pending");
        state.ensure_capacity(ctx, state.pos + rows)?;
        // And the head's row after them, where the context leaves one.
        let after = (state.pos + rows + 1).min(state.max_capacity);
        state.ensure_capacity(ctx, after)?;
        super::model::check_tokens(&[pending], self.cfg.vocab)?;
        super::model::check_tokens(drafts, self.cfg.vocab)?;
        let toks: Vec<u32> =
            std::iter::once(pending).chain(drafts.iter().copied()).collect();
        crate::glm53::model::write_words(&scratch.spec_tokens, &toks)?;
        crate::glm53::model::write_words(&scratch.tokens, &toks)?;
        // The GDN keeps its state after row 0 aside: the rollback if row 1's
        // draft is rejected.
        let snap = (rows > 1).then_some(0);
        let vocab = self.cfg.vocab;
        let sc: &Scratch = scratch;
        self.run_rows(ctx, state, sc, rows, Logits::All, snap, |pass| {
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
        Ok((draws, ctx.begin()?.commit()?))
    }

    fn finish_speculation(
        &self,
        ctx: &MetalContext,
        state: &mut Qwen36State,
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
        state.pos = pos0 + 1 + accepted;
        if accepted + 1 < rows {
            // Back to the state after row 0.
            state.take_gdn_snapshots();
        }
        for &token in &draws[accepted + 1..] {
            scratch.sampler().uncount(token);
        }
        // The head's pairs of the kept rows: (hidden row r, draw r) at head
        // row pos0 + 1 + r; with `propose` also the last draw's (the new
        // pending token), whose greedy next token is the draft.
        // No draft for the context's last rows ([`draft_fits`]).
        let propose = propose && draft_fits(pos0 + 1 + accepted, state.max_capacity);
        let (pairs, draft) = mtp_pairs(pos0, accepted, propose, state.capacity);
        if pairs > 0 {
            let sc: &Scratch = scratch;
            let st: &Qwen36State = state;
            let h = self.cfg.hidden;
            let pass = self.begin(ctx)?;
            let tokens = sc.draws.view(0, &[pairs])?;
            let hin = sc.hidden_norm.view(0, &[pairs * h])?;
            self.encode_mtp(ctx, &pass, st, sc, &tokens, &hin, pos0 + 1, pairs, draft)?;
            pass.commit_wait()?;
        }
        let m = state.mtp.as_mut().ok_or_else(|| anyhow::anyhow!("no MTP state"))?;
        m.pos = pos0 + 1 + pairs;
        if !draft {
            // At rest (or at the context's end): the next pair takes the
            // last kept row's hidden state.
            self.copy_row(&scratch.hidden_norm, accepted, &m.pending_h)?;
            return Ok(Vec::new());
        }
        self.proposal(scratch)
    }

    /// The file's context, or less: what the attention kernel or the
    /// memory limit takes ([`Qwen36Model::fit_context`]).
    fn max_position_embeddings(&self) -> usize {
        self.context_cap
    }

    fn session_bytes(&self, capacity: usize, checkpoints: usize) -> Option<u64> {
        Some(Qwen36Model::session_bytes(self, capacity, checkpoints))
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        vec![self.cfg.eos, self.cfg.pad]
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn bytes_per_token(&self) -> usize {
        // K and V rows of every full-attention layer and the MTP head's.
        let layers = self.cfg.attn_layers() + usize::from(self.mtp.is_some());
        2 * layers * self.kv_format.row_bytes(self.cfg.head_dim) * self.cfg.n_head_kv
    }

    fn memory_limit(&self) -> Option<u64> {
        self.memory_limit
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
        Some(self.prefill_rows)
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
        self.encode_layers(
            ctx,
            &pass,
            state,
            scratch,
            &token,
            1,
            Logits::Last,
            None,
            0..n,
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
#[path = "../../tests/unit/qwen36/engine.rs"]
mod tests;
