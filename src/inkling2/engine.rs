//! Inkling 2 Mini behind lily's [`LanguageModel`] interface: session cache,
//! snapshots, `--memory-limit-gb` fit for 16 GB / 32 GB / 64 GB / 128 GB
//! machines, DSpark speculative decoding, and `inkling2v` vision tower.

use std::path::Path;

use anyhow::{Result, ensure};

use super::model::{
    Inkling2Model, Inkling2State, MAX_SEQ, Scratch, capacity_cap, check_tokens,
};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{EncodedPass, MetalContext, PendingPass};
use crate::tensor::{DType, Tensor};

/// Public alias for server wiring.
pub type Inkling2Engine = Inkling2Model;

fn host_copy(src: &Tensor, dst: &Tensor, bytes: usize) {
    assert!(bytes <= src.byte_len() && bytes <= dst.byte_len());
    // SAFETY: both tensors are host-visible shared buffers, GPU idle.
    unsafe {
        core::ptr::copy_nonoverlapping(src.contents_ptr(), dst.contents_ptr(), bytes);
    }
}

/// Recurrent state snapshot at one position: all 40 SWA KV rings, all 48
/// short-convolution states (K, V, attn_post, ffn_post), and DSpark's recurrent
/// state (`f_rnn_last` + `pos`).
pub struct Inkling2Snapshot {
    pos: usize,
    swa_kv: Vec<(Tensor, Tensor)>,
    sconv: Vec<[Tensor; 4]>,
    dspark_f_rnn: Option<Tensor>,
    dspark_pos: usize,
}

impl SnapshotApi for Inkling2Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        let kv: usize =
            self.swa_kv.iter().map(|(k, v)| k.byte_len() + v.byte_len()).sum();
        let sc: usize =
            self.sconv.iter().flat_map(|a| a.iter()).map(Tensor::byte_len).sum();
        let ds = self.dspark_f_rnn.as_ref().map_or(0, Tensor::byte_len);
        kv + sc + ds
    }
}

impl DecodeStateApi for Inkling2State {
    type Snapshot = Inkling2Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        for s in &mut self.slots {
            for h in [&s.k_hist, &s.v_hist, &s.attn_hist, &s.ffn_hist] {
                h.zero_fill();
            }
        }
        if let Some(ds) = &mut self.dspark {
            ds.reset();
        }
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        ensure!(tokens <= MAX_SEQ, "context {tokens} exceeds {MAX_SEQ}");
        let mut new_cap = self.capacity.max(64);
        while new_cap < tokens {
            new_cap = (new_cap * 2).min(MAX_SEQ);
        }
        for s in &mut self.slots {
            if s.swa {
                continue;
            }
            let kw = s.k.shape()[1];
            let nk = Tensor::zeros(ctx, &[new_cap, kw], DType::F16)?;
            let nv = Tensor::zeros(ctx, &[new_cap, kw], DType::F16)?;
            let copy_bytes = self.pos * kw * 2;
            if copy_bytes > 0 {
                host_copy(&s.k, &nk, copy_bytes);
                host_copy(&s.v, &nv, copy_bytes);
            }
            s.k = nk;
            s.v = nv;
            s.slots = new_cap;
        }
        self.capacity = new_cap;
        if let Some(ds) = &mut self.dspark {
            ds.ensure_capacity(ctx, new_cap)?;
        }
        Ok(())
    }

    fn bytes(&self) -> usize {
        let slots_bytes: usize = self
            .slots
            .iter()
            .map(|s| {
                s.k.byte_len()
                    + s.v.byte_len()
                    + s.k_hist.byte_len()
                    + s.v_hist.byte_len()
                    + s.attn_hist.byte_len()
                    + s.ffn_hist.byte_len()
            })
            .sum();
        let ds_bytes: usize = self.dspark.as_ref().map_or(0, |ds| {
            ds.k_ctx.iter().map(Tensor::byte_len).sum::<usize>()
                + ds.v_ctx.iter().map(Tensor::byte_len).sum::<usize>()
                + ds.f_rnn_last.byte_len()
        });
        slots_bytes + ds_bytes
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Inkling2Snapshot> {
        let mut swa_kv = Vec::new();
        let mut sconv = Vec::with_capacity(self.slots.len());
        for s in &self.slots {
            if s.swa {
                let k = Tensor::zeros(ctx, s.k.shape(), DType::F16)?;
                let v = Tensor::zeros(ctx, s.v.shape(), DType::F16)?;
                host_copy(&s.k, &k, s.k.byte_len());
                host_copy(&s.v, &v, s.v.byte_len());
                swa_kv.push((k, v));
            }
            let copy = |h: &Tensor| -> Result<Tensor> {
                let c = Tensor::zeros(ctx, h.shape(), DType::F32)?;
                host_copy(h, &c, h.byte_len());
                Ok(c)
            };
            sconv.push([
                copy(&s.k_hist)?,
                copy(&s.v_hist)?,
                copy(&s.attn_hist)?,
                copy(&s.ffn_hist)?,
            ]);
        }
        let (dspark_f_rnn, dspark_pos) = match &self.dspark {
            Some(ds) => {
                let f = Tensor::zeros(ctx, ds.f_rnn_last.shape(), DType::F32)?;
                host_copy(&ds.f_rnn_last, &f, ds.f_rnn_last.byte_len());
                (Some(f), ds.pos)
            }
            None => (None, 0),
        };
        Ok(Inkling2Snapshot { pos: self.pos, swa_kv, sconv, dspark_f_rnn, dspark_pos })
    }

    fn restore(&mut self, _ctx: &MetalContext, snap: &Inkling2Snapshot) -> Result<()> {
        let mut swa_i = 0;
        for (i, s) in self.slots.iter_mut().enumerate() {
            if s.swa {
                let (sk, sv) = &snap.swa_kv[swa_i];
                host_copy(sk, &s.k, sk.byte_len().min(s.k.byte_len()));
                host_copy(sv, &s.v, sv.byte_len().min(s.v.byte_len()));
                swa_i += 1;
            }
            for (dst, src) in [&s.k_hist, &s.v_hist, &s.attn_hist, &s.ffn_hist]
                .into_iter()
                .zip(&snap.sconv[i])
            {
                host_copy(src, dst, src.byte_len());
            }
        }
        if let (Some(ds), Some(f_src)) = (&mut self.dspark, &snap.dspark_f_rnn) {
            host_copy(f_src, &ds.f_rnn_last, f_src.byte_len());
            ds.pos = snap.dspark_pos;
        }
        self.pos = snap.pos;
        Ok(())
    }

    fn copy_prefix_from(
        &mut self,
        ctx: &MetalContext,
        from: &Self,
        tokens: usize,
    ) -> Result<()> {
        ensure!(tokens <= from.pos, "copy_prefix {tokens} > {}", from.pos);
        self.ensure_capacity(ctx, tokens)?;
        for (dst, src) in self.slots.iter_mut().zip(&from.slots) {
            if dst.swa {
                continue;
            }
            let kw = dst.k.shape()[1];
            let bytes = tokens * kw * 2;
            if bytes > 0 {
                host_copy(&src.k, &dst.k, bytes);
                host_copy(&src.v, &dst.v, bytes);
            }
        }
        if let (Some(dst_ds), Some(src_ds)) = (&mut self.dspark, &from.dspark) {
            let ds_tokens = tokens.min(src_ds.pos);
            dst_ds.ensure_capacity(ctx, ds_tokens)?;
            let bytes = ds_tokens * dst_ds.kv_w * 2;
            if bytes > 0 {
                for l in 0..dst_ds.k_ctx.len() {
                    host_copy(&src_ds.k_ctx[l], &dst_ds.k_ctx[l], bytes);
                    host_copy(&src_ds.v_ctx[l], &dst_ds.v_ctx[l], bytes);
                }
            }
            host_copy(&src_ds.f_rnn_last, &dst_ds.f_rnn_last, src_ds.f_rnn_last.byte_len());
            dst_ds.pos = ds_tokens;
        }
        self.pos = tokens;
        Ok(())
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    fn logits(&self) -> &Tensor {
        &self.logits
    }

    fn begin_request(&self) {
        self.sampler.reset_counts();
    }
}

impl Inkling2Model {
    /// Fits the longest context whose session (global KV + recurrent + snapshots)
    /// stays under `limit` bytes after `base_allocated`.
    pub fn fit_context(
        &self,
        limit: u64,
        base_allocated: u64,
        checkpoints: usize,
    ) -> Result<usize> {
        let per_tok = self.bytes_per_token() as u64;
        let rec = self.recurrent_bytes() as u64;
        let fixed = base_allocated + rec * (1 + checkpoints as u64);
        ensure!(
            limit > fixed,
            "Inkling 2 Mini needs at least {:.2} GB (limit {:.2} GB)",
            fixed as f64 / 1e9,
            limit as f64 / 1e9
        );
        let room = limit - fixed;
        let max_toks = if per_tok > 0 {
            ((room / per_tok) as usize).min(MAX_SEQ).min(self.cfg.context_length)
        } else {
            MAX_SEQ.min(self.cfg.context_length)
        };
        ensure!(
            max_toks >= 512,
            "Inkling 2 Mini fits only {max_toks} tokens under {:.2} GB limit (need >= 512)",
            limit as f64 / 1e9
        );
        Ok(max_toks)
    }
}

impl LanguageModel for Inkling2Model {
    type State = Inkling2State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "inkling-2-mini";

    fn load(ctx: &MetalContext, path: &Path, options: &LoadOptions) -> Result<Self> {
        let mut model = Inkling2Model::load(ctx, path)?;
        model.memory_limit = options.memory_limit;

        // Optional DSpark speculative drafter beside the GGUF or inside the native dir.
        if options.mtp_drafts > 0
            && let Some(dp) = super::dspark::find_beside(path, &model.cfg)?
        {
            let d = super::dspark::Inkling2Dspark::load(ctx, &dp, &model.cfg)?;
            eprintln!(
                "inkling2 dspark: {} ({:.2} GB)",
                dp.display(),
                d.bytes as f64 / 1e9
            );
            model.drafts = options.mtp_drafts.min(super::dspark::MAX_DRAFTS);
            model.dspark = Some(d);
            // When the main LM head is BF16 (~926 MB), build a compact Q4_0 copy
            // (~261 MB) for DSpark drafting only; verification still uses exact BF16.
            if model.output.ty == crate::gguf::GgmlType::Bf16 {
                let dq = super::native::quantize_bf16_to_q4_0(ctx, &model.output)?;
                model.draft_output = Some(dq);
            }
        }

        // Fit prompt chunk size and max context under `--memory-limit-gb`.
        if let Some(limit) = model.memory_limit {
            let before = ctx.current_allocated();
            let checkpoints = options.session_checkpoints.unwrap_or(
                crate::serve::CHECKPOINTS_PER_SESSION
                    + crate::serve::DECODE_CHECKPOINTS_PER_SESSION,
            );
            let mut chosen = None;
            let start_rows = model.prefill_rows;
            for rows in [1024, 512, 256, 128, 64, 32] {
                if rows > start_rows {
                    continue;
                }
                model.prefill_rows = rows;
                let probe = model.new_scratch(ctx, rows)?;
                let scratch_bytes = ctx.current_allocated().saturating_sub(before);
                drop(probe);
                let allocated = (before + scratch_bytes) as u64;
                if let Ok(ctx_len) = model.fit_context(limit, allocated, checkpoints) {
                    chosen = Some((rows, allocated, ctx_len));
                    if ctx_len >= 8192 {
                        break;
                    }
                }
            }
            let (rows, allocated, ctx_len) = chosen.ok_or_else(|| {
                anyhow::anyhow!(
                    "inkling2: weights and scratch exceed --memory-limit-gb {:.1}",
                    limit as f64 / (1u64 << 30) as f64
                )
            })?;
            model.prefill_rows = rows;
            model.fitted_context = Some(ctx_len);
            eprintln!(
                "inkling2: --memory-limit-gb {:.1} ({:.2} GB) leaves {:.2} GB after {:.2} GB weights+scratch: \
                 up to {ctx_len} tokens ({checkpoints} snapshots), prompt runs of {rows} rows",
                limit as f64 / (1u64 << 30) as f64,
                limit as f64 / 1e9,
                limit.saturating_sub(allocated) as f64 / 1e9,
                allocated as f64 / 1e9,
            );
        }

        Ok(model)
    }

    fn max_position_embeddings(&self) -> usize {
        self.fitted_context.unwrap_or(self.cfg.context_length.min(MAX_SEQ))
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        vec![self.cfg.eos, self.cfg.pad]
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_unpadded
    }

    fn bytes_per_token(&self) -> usize {
        Inkling2Model::bytes_per_token(self)
    }

    fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        Inkling2Model::weight_buffers(self)
    }

    fn memory_limit(&self) -> Option<u64> {
        self.memory_limit
    }

    fn session_bytes(&self, capacity: usize, checkpoints: usize) -> Option<u64> {
        let per_tok = self.bytes_per_token() as u64;
        let rec = self.recurrent_bytes() as u64;
        Some(capacity_cap(capacity) as u64 * per_tok + rec * (1 + checkpoints as u64))
    }

    fn prefill_chunk_tokens(&self) -> Option<usize> {
        Some(self.prefill_rows)
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Inkling2State> {
        Inkling2Model::new_state(self, ctx, capacity)
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        _capacity_tokens: usize,
    ) -> Result<Scratch> {
        self.new_scratch(ctx, self.prefill_rows)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut Inkling2State,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        check_tokens(tokens, self.cfg.vocab_unpadded)?;
        state.ensure_capacity(ctx, state.pos + tokens.len())?;
        let chunk_max = scratch.max_rows;
        let mut off = 0;
        while off < tokens.len() {
            let chunk = (tokens.len() - off).min(chunk_max);
            let is_last = off + chunk == tokens.len();
            scratch.stage_tokens(&tokens[off..off + chunk])?;
            let has_ds = self.dspark.is_some();
            {
                let cp = ctx.begin_concurrent()?;
                self.encode_forward_rows(
                    ctx,
                    &cp,
                    state,
                    scratch,
                    chunk,
                    None,
                    false,
                    is_last && draw.is_some(),
                    has_ds,
                    false,
                )?;
                cp.commit_wait()?;
            }
            if let (Some(ds), Some(ds_state), Some(cap_buf)) =
                (&self.dspark, &mut state.dspark, &scratch.dspark_cap)
            {
                ds.inject_gpu(ctx, ds_state, cap_buf, scratch.max_rows, chunk, state.pos)?;
            }
            state.pos += chunk;
            off += chunk;
        }
        if let Some(d) = draw {
            let cp = ctx.begin_concurrent()?;
            let out = scratch.next_token.view(0, &[1])?;
            sample_f32(
                ctx,
                &cp,
                &scratch.logits,
                &scratch.sampler,
                d.params,
                d.step,
                &out,
            )?;
            cp.commit_wait()?;
        }
        Ok(())
    }

    fn prepare_step_inputs(
        &self,
        _state: &mut Inkling2State,
        scratch: &Scratch,
        token: u32,
    ) -> Result<()> {
        check_tokens(&[token], self.cfg.vocab_unpadded)?;
        scratch.stage_tokens(&[token])
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &Inkling2State,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        let cp = ctx.begin_concurrent()?;
        let tok_in = scratch.next_token.view(slot_in, &[1])?;
        self.encode_forward_rows(
            ctx,
            &cp,
            state,
            scratch,
            1,
            Some(&tok_in),
            false,
            true,
            false,
            false,
        )?;
        cp.memory_barrier()?;
        let out = scratch.next_token.view(slot_out, &[1])?;
        sample_f32(
            ctx,
            &cp,
            &scratch.logits,
            &scratch.sampler,
            draw.params,
            draw.step,
            &out,
        )?;
        cp.end()
    }

    fn max_drafts(&self) -> usize {
        if self.dspark.is_some() { self.drafts } else { 0 }
    }

    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut Inkling2State,
        scratch: &mut Scratch,
        first: u32,
        drafts: usize,
        _params: &SamplingParams,
        _step0: usize,
    ) -> Result<Vec<u32>> {
        let (Some(ds), Some(ds_state)) = (&self.dspark, &state.dspark) else {
            return Ok(Vec::new());
        };
        check_tokens(&[first], self.cfg.vocab_unpadded)?;
        scratch.stage_tokens(&[first])?;
        ds.prepare_propose(first)?;
        let anchor_embd = scratch.logits_raw.view(0, &[self.cfg.hidden])?;
        let cp = ctx.begin_concurrent()?;
        crate::kernels::inkling2::embed(
            ctx,
            &cp,
            &scratch.ids,
            &self.tok_embd,
            &anchor_embd,
            1,
        )?;
        cp.memory_barrier()?;
        crate::kernels::inkling2::rms_norm(
            ctx,
            &cp,
            &anchor_embd,
            Some(&self.tok_norm),
            &anchor_embd,
            1,
            self.cfg.hidden,
            self.cfg.rms_eps,
        )?;
        cp.memory_barrier()?;
        ds.encode_propose(
            ctx,
            &cp,
            ds_state,
            &anchor_embd,
            self.draft_output_weight(),
            self.cfg.vocab_unpadded,
            state.pos,
            drafts,
        )?;
        cp.commit_wait()?;
        ds.read_drafts(drafts)
    }

    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Inkling2State,
        scratch: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        let rows = 1 + drafts.len();
        state.ensure_capacity(ctx, state.pos + rows)?;
        let tokens: Vec<u32> =
            std::iter::once(pending).chain(drafts.iter().copied()).collect();
        check_tokens(&tokens, self.cfg.vocab_unpadded)?;

        let pos_before = state.pos;
        let vs = scratch
            .verify
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verify scratch not allocated"))?;
        ensure!(rows <= vs.max_rows, "verify rows {rows} > {}", vs.max_rows);

        scratch.stage_tokens(&tokens)?;
        let cp = ctx.begin_concurrent()?;
        self.encode_forward_rows(
            ctx,
            &cp,
            state,
            scratch,
            rows,
            None,
            false,
            true,
            self.dspark.is_some(),
            true,
        )?;
        cp.memory_barrier()?;
        let vocab = self.cfg.vocab_unpadded;
        for r in 0..rows {
            let row_logits = vs.logits.view(r * vocab, &[vocab])?;
            let out = vs.draws_gpu.view(r, &[1])?;
            sample_f32(ctx, &cp, &row_logits, &scratch.sampler, params, step0 + r, &out)?;
            if r + 1 < rows {
                cp.memory_barrier()?;
            }
        }
        cp.commit_wait()?;

        let draws = vs.draws_gpu.to_u32()?[..rows].to_vec();

        if let Some(vs_mut) = scratch.verify.as_mut() {
            vs_mut.pending = Some(super::model::VerifyPending {
                pos_before,
                rows,
                draws: draws.clone(),
                next_drafts,
            });
        }
        state.pos = pos_before + rows;
        Ok((draws, ctx.begin()?.commit()?))
    }

    fn finish_speculation(
        &self,
        ctx: &MetalContext,
        state: &mut Inkling2State,
        scratch: &mut Scratch,
        accepted: usize,
        propose: bool,
        draft: PendingPass<'_>,
    ) -> Result<Vec<u32>> {
        draft.wait()?;
        let pending = scratch
            .verify
            .as_mut()
            .and_then(|v| v.pending.take())
            .ok_or_else(|| anyhow::anyhow!("finish_speculation without verify"))?;
        let kept = 1 + accepted;
        ensure!(kept <= pending.rows, "finish_speculation kept {kept} > {}", pending.rows);

        // O(1) rollback of short-conv histories if any draft was rejected.
        if kept < pending.rows
            && let Some(vs) = &scratch.verify
        {
            let row_idx = kept - 1;
            let hist_len = self.cfg.sconv_l_cache - 1;
            for (slot, sc_slot) in state.slots.iter_mut().enumerate() {
                let tr = &vs.trace[slot];
                let kw = sc_slot.k_hist.shape()[0];
                let h = sc_slot.attn_hist.shape()[0];
                let k_bytes = kw * hist_len * 4;
                let h_bytes = h * hist_len * 4;
                let k_src = tr[0].view(row_idx * kw * hist_len, &[kw, hist_len])?;
                let v_src = tr[1].view(row_idx * kw * hist_len, &[kw, hist_len])?;
                let a_src = tr[2].view(row_idx * h * hist_len, &[h, hist_len])?;
                let f_src = tr[3].view(row_idx * h * hist_len, &[h, hist_len])?;
                host_copy(&k_src, &sc_slot.k_hist, k_bytes);
                host_copy(&v_src, &sc_slot.v_hist, k_bytes);
                host_copy(&a_src, &sc_slot.attn_hist, h_bytes);
                host_copy(&f_src, &sc_slot.ffn_hist, h_bytes);
            }
        }
        state.pos = pending.pos_before + kept;

        // Inject accepted target features into DSpark GPU KV + propose next drafts
        // ALL IN ONE SINGLE GPU COMMAND BUFFER (zero CPU round-trips!).
        if let (Some(ds), Some(ds_state), Some(cap_buf)) =
            (&self.dspark, &mut state.dspark, &scratch.dspark_cap)
        {
            ds_state.ensure_capacity(ctx, pending.pos_before + kept)?;
            let do_propose = propose && pending.next_drafts > 0;
            if do_propose {
                let anchor = pending.draws[accepted];
                check_tokens(&[anchor], self.cfg.vocab_unpadded)?;
                scratch.stage_tokens(&[anchor])?;
                ds.prepare_propose(anchor)?;
            }
            let cp = ctx.begin_concurrent()?;
            ds.encode_inject(
                ctx,
                &cp,
                ds_state,
                cap_buf,
                scratch.max_rows,
                kept,
                pending.pos_before,
            )?;
            if do_propose {
                cp.memory_barrier()?;
                let anchor_embd = scratch.logits_raw.view(0, &[self.cfg.hidden])?;
                crate::kernels::inkling2::embed(
                    ctx,
                    &cp,
                    &scratch.ids,
                    &self.tok_embd,
                    &anchor_embd,
                    1,
                )?;
                cp.memory_barrier()?;
                crate::kernels::inkling2::rms_norm(
                    ctx,
                    &cp,
                    &anchor_embd,
                    Some(&self.tok_norm),
                    &anchor_embd,
                    1,
                    self.cfg.hidden,
                    self.cfg.rms_eps,
                )?;
                cp.memory_barrier()?;
                ds.encode_propose(
                    ctx,
                    &cp,
                    ds_state,
                    &anchor_embd,
                    self.draft_output_weight(),
                    self.cfg.vocab_unpadded,
                    state.pos,
                    pending.next_drafts,
                )?;
            }
            cp.commit_wait()?;
            if do_propose {
                return ds.read_drafts(pending.next_drafts);
            }
        }
        Ok(Vec::new())
    }
}
