//! Speculative decoding with the checkpoint's multi-token-prediction head.
//!
//! A step verifies the pending token plus `k` drafts in one batched trunk
//! pass (`BatchMode::Verify`), which draws one token per row and records the
//! GDN state after every row. The draft pass follows it on the queue without
//! the host in between: a first dispatch compares the draws with the drafts
//! and writes the accepted count and everything derived from it (positions,
//! indexer blocks) into a control block, which the pass's later dispatches
//! read as inline arguments (`kernels::spec`, `kernels::Arg::Gpu`). The pass
//! rolls the recurrent state back to the accepted prefix (state copy plus
//! conv-window rewind, no recomputation), catches the draft head up on every
//! verified row with real trunk hiddens (rows past the accepted one are
//! position-indexed garbage the chain overwrites before anything reads them),
//! chains `k` new proposals from the head's own residual at GPU-supplied
//! positions. Attention caches are position indexed, so rejected rows are
//! simply overwritten later.
//!
//! The host meanwhile reads the draws and emits tokens, then reads the
//! proposals, stages the next verify pass's inputs and encodes and commits
//! it whole. Until 2026-10-19 the next verify pass was encoded ahead for
//! every possible accepted count and committed parked behind the draft pass,
//! waiting on the GPU for its n-gram rows; under window-server activity on
//! macOS 27 a parked pass resumed about 2 ms after its release, and every
//! dozen or so steps only after about 360 ms, which halved speculative
//! decode while the screen was in use. Committing the pass whole costs 2.5%
//! with the screen idle (`docs/performance.md`, "Waiting for the GPU").
//! When a generation ends at a row before the one the GPU accepted (a stop
//! token among accepted drafts), a small host-encoded pass rolls the state
//! back further.
//!
//! Outputs are exact: every emitted token is a draw from the trunk's own
//! distribution for its prefix; the drafts only decide how many rows a pass
//! can confirm.

use std::time::Instant;

use anyhow::{Result, ensure};

use crate::engine::DecodeStateApi;
use crate::kernels::elementwise::copy_words;
use crate::kernels::gdn::{GDN_HEAD_DIM, conv_window_rollback};
use crate::kernels::sample::{
    SamplingParams, draft_params, draft_step, sample_draft_f32, sample_draft_prob_f32,
};
use crate::kernels::spec::{
    SLOT_ACCEPTED, SLOT_KEEP, copy_row, ctrl_word, slot_block, slot_count, slot_pos,
    spec_accept,
};
use crate::kernels::{Arg, Pos, Rope};
#[cfg(target_os = "linux")]
use crate::metal::DetachedPass;
use crate::metal::{EncodedPass, MetalContext, PendingPass};
use crate::moe_ffn::{prefix_rows, project_mat};
use crate::tensor::Tensor;
use crate::weights::{GgufLinear, LinearWeights};

use super::model::{
    AttnPos, BatchMode, DecodeState, LayerState, MAX_DRAFTS, Qwen4ExpModel, Scratch,
    SpecPending,
};
use super::ngram::NgramHasher;

fn profiling() -> bool {
    std::env::var_os("LILY_PROFILE").is_some()
}

/// GPU end time of the last draft pass (`LILY_PROFILE` only), for the gap
/// to the verify pass after it.
static DRAFT_GPU_END: std::sync::Mutex<Option<f64>> = std::sync::Mutex::new(None);

/// The draft head's narrow LM head: Qwen3.8's ids below this and from
/// [`DRAFT_SPECIAL_FROM`] on (the tokenizer's added tokens: turn, tool and
/// thinking markers). On English prose, code and lily's own sources 99.8 to
/// 100% of the tokens are among them (2026-11-15); its multilingual ids
/// (151 643 up) are not, and Italian text draws 14% of its tokens there.
/// The rows read per proposal drop from 248 320 to 98 580 (Q2 file: 1.2 to
/// 0.5 ms). The trunk draws every emitted token with the whole LM head, so
/// greedy output is unchanged; under a sampler, speculative sampling still
/// draws from the trunk's distribution, but with other proposals a given
/// seed may draw other tokens.
const DRAFT_LOW_ROWS: usize = 98_304;
const DRAFT_SPECIAL_FROM: usize = 248_044;
const QWEN38_VOCAB: usize = 248_320;
/// Positions after a fed token outside the narrow head during which drafts
/// take the whole LM head: text in another script keeps drafting at full
/// acceptance.
const WIDE_WINDOW: usize = 96;

/// Whether the checkpoint's vocabulary has the narrow draft head (Qwen3.8's).
pub(super) fn has_narrow_head(cfg: &super::config::Qwen4ExpConfig) -> bool {
    cfg.vocab_size == QWEN38_VOCAB
}

/// `LILY_DRAFT_HEAD=full`: the draft head always takes the whole LM head.
pub(super) fn narrow_head_env() -> bool {
    !std::env::var("LILY_DRAFT_HEAD").is_ok_and(|v| v == "full")
}

pub(super) fn in_narrow_head(token: u32) -> bool {
    let t = token as usize;
    t < DRAFT_LOW_ROWS || t >= DRAFT_SPECIAL_FROM
}

/// Which LM head the draft head takes, from the position of the last fed
/// token outside the narrow head. A position, not a countdown, so it is the
/// same whatever pieces the tokens were fed in (the server prefills a
/// prompt up to its last token, then that token; a cached prefix feeds only
/// the suffix) and it travels with a snapshot to restores and forks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftVocab {
    last_wide: Option<usize>,
}

impl DraftVocab {
    /// `tokens` fed at positions `pos0..`.
    pub fn note(&mut self, pos0: usize, tokens: &[u32]) {
        if let Some(i) = tokens.iter().rposition(|&t| !in_narrow_head(t)) {
            self.last_wide = Some(pos0 + i);
        }
    }

    /// Whether drafts at state position `pos` take the whole LM head.
    pub fn wide(&self, pos: usize) -> bool {
        self.last_wide.is_some_and(|at| pos < at + WIDE_WINDOW)
    }

    /// Kept for a snapshot at `pos`: nothing past it.
    pub fn at(self, pos: usize) -> Self {
        Self { last_wide: self.last_wide.filter(|&at| at < pos) }
    }

    /// A state read back from the disk tier, whose layout does not carry
    /// the history: taken as if the token before `pos` were outside the
    /// narrow head (the whole head for the first drafts, never a narrow
    /// head the history would not have allowed).
    pub fn after_disk_restore(pos: usize) -> Self {
        Self { last_wide: Some(pos.saturating_sub(1)) }
    }
}

/// The draft gate: a step verifies the first proposal and the ones after
/// it while the head's confidence in them is at least this (per position);
/// the rest are dropped. The confidence is a heuristic: the largest softmax
/// probability of the head's raw logits (before a sampler's temperature or
/// truncation), not the probability the proposal was drawn with. A
/// verified row costs about a fifth of a plain step (q4-xl: 2.9 ms of 14.6,
/// the Q2 file 4.5 of 18.7), so a proposal below a third or so costs more
/// than it returns. The first is always verified: a verify pass of one row
/// takes other kernels, whose rounding would make the output differ from
/// the ungated run's. Greedy output is unchanged; under a sampler,
/// speculative sampling still draws from the trunk's distribution, but a
/// step verifying fewer rows may draw other tokens for a given seed. Chosen
/// on logged (confidence, accepted) steps of both checkpoints over prose,
/// code and Italian, 2026-11-15 (`lily-wt/night/q38.md`).
const DRAFT_MIN_PROB: [f32; MAX_DRAFTS] = [0.0, 0.4, 0.3];

/// `LILY_DRAFT_GATE=0` verifies every proposal. Linux only: the confidence
/// kernels exist only in GLSL.
pub(super) fn draft_gate_env() -> bool {
    cfg!(target_os = "linux") && std::env::var("LILY_DRAFT_GATE").map_or(true, |v| v != "0")
}

/// The proposals the gate keeps of `tokens` drawn with `probs`.
fn gate_proposals(mut tokens: Vec<u32>, probs: &[f32]) -> Vec<u32> {
    let keep = probs
        .iter()
        .zip(DRAFT_MIN_PROB)
        .take_while(|(p, min)| **p >= *min)
        .count();
    tokens.truncate(keep);
    tokens
}

/// Proposal `i`'s draw from the head's `logits` into the spec scratch, with
/// its probability when the step is `gated`.
#[allow(clippy::too_many_arguments)]
fn draft_draw(
    ctx: &MetalContext,
    pass: &crate::metal::ComputePass<'_>,
    logits: &Tensor,
    s: &Scratch,
    sp: &super::model::SpecScratch,
    params: &SamplingParams,
    step: usize,
    i: usize,
    gated: bool,
) -> Result<()> {
    let out = sp.draft_tokens.view(i, &[1])?;
    if !gated {
        let dist = sp.dists.slot(i)?;
        return sample_draft_f32(ctx, pass, logits, &s.sampler, params, step, &out, &dist);
    }
    sample_draft_prob_f32(
        ctx,
        pass,
        logits,
        &s.sampler,
        params,
        step,
        &out,
        &sp.dists.slot(i)?,
        &sp.draft_probs.view(i, &[1])?,
        &sp.prob_sums,
    )
}

/// `w`'s rows `[start, start + rows)`.
fn head_rows(w: &LinearWeights, start: usize, rows: usize) -> Result<LinearWeights> {
    Ok(match w {
        LinearWeights::Affine(q) => LinearWeights::Affine(q.view_rows(start, rows)?),
        LinearWeights::Gguf(g) => {
            ensure!(g.parts.len() == 1, "an LM head of {} parts", g.parts.len());
            let p = &g.parts[0];
            ensure!(start + rows <= p.rows, "rows past the LM head");
            let mut v = p.clone();
            v.t = p.t.view(start * p.row_bytes, &[rows * p.row_bytes])?;
            v.rows = rows;
            LinearWeights::Gguf(GgufLinear { parts: vec![v] })
        }
    })
}

/// A point of the step's host work in the backend's trace (`LILY_VK_TRACE`).
#[cfg(target_os = "linux")]
fn mark(name: &'static str) {
    crate::metal::trace::mark(name);
}

#[cfg(not(target_os = "linux"))]
fn mark(_name: &'static str) {}

/// The next verify pass, encoded while the draft pass runs (Linux): a
/// verify pass's encoding is 0.7 ms of host time at the top clock, and
/// encoded after the proposals are read it is time the GPU idles. Nothing
/// of the pass depends on the proposals themselves: it reads them, and the
/// n-gram rows staged from them, from buffers written before it is
/// committed. What it does depend on is here, and [`Qwen4ExpModel::verify`]
/// uses the pass only when all of it still holds.
#[cfg(target_os = "linux")]
pub(super) struct VerifyAhead {
    pass: DetachedPass,
    pos: usize,
    conv_slot: usize,
    m: usize,
    params: SamplingParams,
    step0: usize,
    /// [`MetalContext::passes_begun`] after the encoding: unchanged, no
    /// other pass was encoded since (none that could have regrown the state
    /// or the scratch the pass addresses).
    begun: u64,
}

/// What the Linux speculative loop keeps in the state between its calls.
#[cfg(target_os = "linux")]
#[derive(Default)]
pub(super) struct SpecAhead {
    /// The sampling and the draw index of the verify pass in flight.
    drawn: Option<(SamplingParams, usize)>,
    verify: Option<VerifyAhead>,
}

impl Qwen4ExpModel {
    /// Whether the draft gate drops proposals ([`DRAFT_MIN_PROB`]): on lily's
    /// own checkpoints, whose verify passes of two to four rows compute every
    /// row alike (the gate leaves the output as it was). Not on GGUF files,
    /// whose GEMVs are a kernel per row count with their own rounding: there
    /// it changed the tokens, and on prose cost GPU time per token
    /// (2026-11-15).
    fn draft_gated(&self) -> bool {
        self.draft_gate && self.config.gguf_down_cols.is_none()
    }

    /// Whether drafts from `state` take the draft head's whole LM head.
    fn draft_wide(&self, state: &DecodeState) -> bool {
        !self.draft_narrow || state.draft_vocab.wide(state.pos)
    }

    /// Turns the narrow draft head and the draft gate on or off (both on by
    /// default unless the environment says otherwise); greedy output does
    /// not depend on either.
    pub fn set_draft_policy(&mut self, narrow: bool, gate: bool) {
        self.draft_narrow = narrow;
        self.draft_gate = gate && cfg!(target_os = "linux");
    }

    /// Draft tokens per step, or 0 without the head.
    pub fn max_drafts(&self) -> usize {
        if self.weights.mtp.is_some() { MAX_DRAFTS } else { 0 }
    }

    /// Makes room for a verify pass of `m` rows plus the draft pass after it.
    fn ensure_room(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        m: usize,
    ) -> Result<()> {
        state.ensure_capacity(ctx, state.pos + m + MAX_DRAFTS)?;
        self.ensure_prefill_scratch(ctx, s, m)
    }

    /// See [`crate::engine::LanguageModel::verify`].
    #[allow(clippy::too_many_arguments)]
    pub fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        ensure!(state.spec.is_none(), "verify while a speculative step is pending");
        ensure!(
            drafts.len() <= MAX_DRAFTS,
            "{} drafts exceed {MAX_DRAFTS}",
            drafts.len()
        );
        ensure!(
            s.spec.is_some() && self.weights.mtp.is_some(),
            "verify without a draft head"
        );
        mark("verify");
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(pending);
        tokens.extend_from_slice(drafts);
        let m = tokens.len();
        let chain = next_drafts.min(MAX_DRAFTS);
        // Room for this pass and the draft pass behind it (its chain rows);
        // nothing runs, so the caches can grow.
        self.ensure_room(ctx, state, s, m)?;
        let s = &*s;
        let capacity = s.prefill.as_ref().expect("prefill scratch just ensured");
        // The token-dependent inputs, ids and n-gram rows, are written before
        // the pass is encoded and committed whole.
        let ps = capacity.chunk(&tokens)?;
        let ple_w = self.weights.layers.iter().find_map(|l| l.ple.as_deref());
        mark("stage");
        if let (Some(w), Some(p), Some(pst)) = (ple_w, &ps.ple, &state.ple) {
            self.stage_ngram(w, p, &tokens, pst.hist)?;
        }
        mark("staged");
        let pos_before = state.pos;
        let hist_before = state.ple.as_ref().map(|p| p.hist);
        #[cfg(target_os = "linux")]
        let ahead = Self::take_verify_ahead(ctx, state, m, params, step0)?;
        #[cfg(not(target_os = "linux"))]
        let ahead: Option<EncodedPass<'a>> = None;
        let encoded = match ahead {
            Some(encoded) => encoded,
            None => {
                self.encode_batch(ctx, state, s, &ps, BatchMode::Verify { params, step0 })?
            }
        };
        let committed = encoded.commit()?;
        // Same bookkeeping as a prefill chunk; the draft pass rolls it back.
        // Done before the draft pass is encoded: it sees the state as the
        // verify pass leaves it.
        state.pos += m;
        if let Some(pst) = &mut state.ple {
            pst.hist = NgramHasher::advance(pst.hist, &tokens);
        }
        state.conv_slot = 1 - state.conv_slot;
        // The draft pass follows on the queue; the GPU reads the accepted
        // count itself.
        let draft = self
            .encode_draft_selected(
                ctx,
                state,
                s,
                pos_before,
                m,
                chain,
                params,
                step0 + m,
            )?
            .commit()?;
        mark("committed");
        let done = committed.wait_retain_polled()?;
        let verify_gpu_end = if profiling() {
            let t = done.timing()?;
            let gap = DRAFT_GPU_END
                .lock()
                .ok()
                .and_then(|g| *g)
                .map_or(f64::NAN, |end| (t.gpu_start_secs - end) * 1e3);
            eprintln!(
                "profile verify m={m}: gpu {:.2} ms, started {gap:.3} ms after the draft pass ended",
                (t.gpu_end_secs - t.gpu_start_secs) * 1e3
            );
            Some(t.gpu_end_secs)
        } else {
            None
        };
        let sampled = s.spec.as_ref().expect("spec scratch").verify_tokens.to_u32()?
            [..m]
            .to_vec();
        mark("verified");
        state.spec = Some(SpecPending {
            pos_before,
            hist_before,
            tokens,
            sampled: sampled.clone(),
            uses_penalties: params.uses_penalties(),
            chain,
            verify_gpu_end,
        });
        Ok((sampled, draft))
    }

    /// See [`crate::engine::LanguageModel::finish_speculation`].
    pub fn finish_speculation(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        accepted: usize,
        propose: bool,
        draft: PendingPass<'_>,
    ) -> Result<Vec<u32>> {
        let entered = Instant::now();
        mark("finish");
        let pending = state.spec.take().ok_or_else(|| {
            anyhow::anyhow!("finish_speculation without a verify pass")
        })?;
        let m = pending.tokens.len();
        ensure!(accepted < m, "accepted {accepted} of {} drafts", m - 1);
        // What the GPU accepted: the leading rows whose draw equals the draft
        // after them. The engine may stop earlier (a stop token among them),
        // never later.
        let gpu_accepted = pending
            .sampled
            .iter()
            .zip(&pending.tokens[1..])
            .take_while(|(draw, draft)| draw == draft)
            .count();
        ensure!(
            accepted <= gpu_accepted,
            "accepted {accepted} rows but only {gpu_accepted} drafts were confirmed"
        );
        let chain = pending.chain;
        // The emitted draws: draw j is fed at pos_before + 1 + j. The draft
        // pass already running chose its head before they were known, so a
        // token outside the narrow head moves the step after it to the whole
        // head (one pass late).
        state.draft_vocab.note(pending.pos_before + 1, &pending.sampled[..=accepted]);

        // Host-side rollback: position, hash history, penalty counts of the
        // rejected rows' draws. (The running draft pass never touches the
        // counts: its draws are greedy.)
        state.pos = pending.pos_before + accepted + 1;
        if let (Some(pst), Some(before)) = (&mut state.ple, pending.hist_before) {
            pst.hist = NgramHasher::advance(before, &pending.tokens[..=accepted]);
        }
        if pending.uses_penalties {
            for &token in &pending.sampled[accepted + 1..] {
                s.sampler.uncount(token);
            }
        }

        let proposals = if !propose {
            // The generation ends here: the draft pass's proposals are not
            // needed, but its state changes must be complete before the state
            // is used again.
            self.collect_drafts(s, draft, 0, &pending)?;
            if accepted < gpu_accepted {
                // The draft pass rolled back to the GPU's count; roll back
                // further to the engine's.
                let s = &*s;
                let capacity = s
                    .prefill
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
                let wide = self.config.hc_width();
                let hidden = capacity.hyper.view(0, &[1, wide])?;
                let keep = capacity.hyper.view(accepted * wide, &[wide])?;
                self.encode_draft(
                    ctx,
                    state,
                    s,
                    &hidden,
                    0,
                    0,
                    pending.pos_before,
                    Some(&keep),
                    Some((accepted, m)),
                    0,
                    &SamplingParams::greedy(),
                    0,
                )?
                .commit()?
                .wait()?;
            }
            Vec::new()
        } else {
            ensure!(
                accepted == gpu_accepted,
                "the step continues past a row the GPU did not confirm"
            );
            ensure!(chain > 0, "the draft pass proposed nothing to verify next");
            #[cfg(target_os = "linux")]
            self.encode_verify_ahead(ctx, state, s, 1 + chain, accepted + 1)?;
            self.collect_drafts(s, draft, chain, &pending)?
        };
        mark("finished");
        if profiling() {
            eprintln!(
                "profile host finish: done {:.2} ms after entry",
                entered.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(proposals)
    }

    /// The verify pass [`Self::encode_verify_ahead`] left in the state, if
    /// it is the pass [`Self::verify`] is about to encode: same position,
    /// rows, sampling and draw index, and no pass encoded since. Notes the
    /// sampling and the draw index for the step's own encoding ahead.
    #[cfg(target_os = "linux")]
    fn take_verify_ahead<'a>(
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        m: usize,
        params: &SamplingParams,
        step0: usize,
    ) -> Result<Option<EncodedPass<'a>>> {
        state.ahead.drawn = Some((*params, step0));
        let Some(ahead) = state.ahead.verify.take() else { return Ok(None) };
        let same = ahead.pos == state.pos
            && ahead.conv_slot == state.conv_slot
            && ahead.m == m
            && ahead.params == *params
            && ahead.step0 == step0
            && ahead.begun == ctx.passes_begun();
        if same { ahead.pass.attach(ctx).map(Some) } else { Ok(None) }
    }

    /// Encodes the verify pass of the next step, `m` rows at the state's
    /// position (already rolled back to the accepted prefix), `fed` draws
    /// after the last one's first, and keeps it in the state for
    /// [`Self::verify`]. The draft pass may be running: nothing here writes
    /// a buffer, and the state and the scratch must already have room, or
    /// nothing is encoded. Neither under the expert cache, where encoding a
    /// pass asks for its experts, nor in the per-kernel profile.
    #[cfg(target_os = "linux")]
    fn encode_verify_ahead(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &Scratch,
        m: usize,
        fed: usize,
    ) -> Result<()> {
        mark("ahead");
        state.ahead.verify = None;
        let Some((params, step0)) = state.ahead.drawn.take() else { return Ok(()) };
        if self.weights.expert_cache.is_some() || ctx.profiling() {
            return Ok(());
        }
        let Some(capacity) = s.prefill.as_ref() else { return Ok(()) };
        if capacity.m < m || state.pos + m + MAX_DRAFTS > state.capacity() {
            return Ok(());
        }
        let step0 = step0 + fed;
        let ps = capacity.rows(m)?;
        let pass = self
            .encode_batch(ctx, state, s, &ps, BatchMode::Verify { params: &params, step0 })?
            .detach();
        state.ahead.verify = Some(VerifyAhead {
            pass,
            pos: state.pos,
            conv_slot: state.conv_slot,
            m,
            params,
            step0,
            begun: ctx.passes_begun(),
        });
        mark("aheaded");
        Ok(())
    }

    /// The draft head's logits `[vocab]` of its mixed row `x`, ordered after
    /// on return: the whole LM head into the spec scratch's first logits
    /// row, or (`wide` false, the narrow head in use) the narrow head's rows
    /// into `draft_logits`, whose other ids hold -inf.
    #[allow(clippy::too_many_arguments)]
    fn draft_head_logits(
        &self,
        ctx: &MetalContext,
        pass: &crate::metal::ComputePass<'_>,
        mtp: &super::weights::MtpWeights,
        x: &Tensor,
        sp: &super::model::SpecScratch,
        s: &Scratch,
        wide: bool,
    ) -> Result<Tensor> {
        let vocab = self.config.vocab_size;
        let head = mtp.head(&self.weights.lm_head);
        let narrow = sp.draft_logits.as_ref().filter(|_| !wide);
        let Some(out) = narrow else {
            let logits = sp.logits.view(0, &[1, vocab])?;
            project_mat(ctx, pass, x, head, &logits, &s.dequant)?;
            pass.level_barrier(&[&logits])?;
            return logits.view(0, &[vocab]);
        };
        let low = head_rows(head, 0, DRAFT_LOW_ROWS)?;
        let special = head_rows(head, DRAFT_SPECIAL_FROM, vocab - DRAFT_SPECIAL_FROM)?;
        let low_out = out.view(0, &[1, DRAFT_LOW_ROWS])?;
        let special_out =
            out.view(DRAFT_SPECIAL_FROM, &[1, vocab - DRAFT_SPECIAL_FROM])?;
        match (&low, &special) {
            // One dispatch: the GEMV takes both row ranges as sets.
            (LinearWeights::Gguf(l), LinearWeights::Gguf(h)) => {
                crate::moe_ffn::gguf_gemv_sets(
                    ctx,
                    pass,
                    x,
                    &[(l, &low_out), (h, &special_out)],
                    1,
                )?;
            }
            _ => {
                project_mat(ctx, pass, x, &special, &special_out, &s.dequant)?;
                project_mat(ctx, pass, x, &low, &low_out, &s.dequant)?;
            }
        }
        pass.level_barrier(&[out])?;
        Ok(out.clone())
    }

    /// See [`crate::engine::LanguageModel::draft_initial`].
    #[allow(clippy::too_many_arguments)]
    pub fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        first: u32,
        drafts: usize,
        params: &SamplingParams,
        step0: usize,
    ) -> Result<Vec<u32>> {
        ensure!(
            state.spec.is_none(),
            "draft_initial during a pending speculative step"
        );
        ensure!(state.pos > 0, "draft_initial before any token was fed");
        let wide = self.config.hc_width();
        let drafts = drafts.min(MAX_DRAFTS);
        state.draft_vocab.note(state.pos, &[first]);
        self.ensure_room(ctx, state, s, 1 + drafts)?;
        let hidden = state
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no draft head state"))?
            .hidden
            .view(0, &[1, wide])?;
        let encoded = self.encode_draft(
            ctx,
            state,
            s,
            &hidden,
            1,
            0,
            state.pos - 1,
            None,
            None,
            drafts,
            params,
            step0,
        )?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        sp.mtp_ids.view(0, &[1])?.write_bytes(bytemuck::cast_slice(&[first]))?;
        let done = encoded.commit()?.wait_retain_polled()?;
        if profiling() {
            let t = done.timing()?;
            eprintln!(
                "profile draft initial drafts={drafts}: gpu {:.2} ms",
                (t.gpu_end_secs - t.gpu_start_secs) * 1e3
            );
            // The first verify pass's gap is to this pass, not to another
            // generation's last draft pass.
            if let Ok(mut g) = DRAFT_GPU_END.lock() {
                *g = Some(t.gpu_end_secs);
            }
        }
        if drafts == 0 {
            return Ok(Vec::new());
        }
        let tokens = sp.draft_tokens.to_u32()?[..drafts].to_vec();
        if !self.draft_gated() {
            return Ok(tokens);
        }
        Ok(gate_proposals(tokens, &sp.draft_probs.to_f32()?[..drafts]))
    }

    /// Waits for the committed draft pass of `pending`'s step and reads its
    /// first `drafts` proposals.
    fn collect_drafts(
        &self,
        s: &Scratch,
        committed: PendingPass<'_>,
        drafts: usize,
        pending: &SpecPending,
    ) -> Result<Vec<u32>> {
        let done = committed.wait_retain_polled()?;
        if profiling() {
            let t = done.timing()?;
            if let Ok(mut g) = DRAFT_GPU_END.lock() {
                *g = Some(t.gpu_end_secs);
            }
            let gap = pending
                .verify_gpu_end
                .map_or(f64::NAN, |end| (t.gpu_start_secs - end) * 1e3);
            eprintln!(
                "profile draft m={} chain={}: gpu {:.2} ms, started {gap:.3} ms after the verify pass ended",
                pending.tokens.len(),
                pending.chain,
                (t.gpu_end_secs - t.gpu_start_secs) * 1e3
            );
        }
        if drafts == 0 {
            return Ok(Vec::new());
        }
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        let tokens = sp.draft_tokens.to_u32()?[..drafts].to_vec();
        if !self.draft_gated() {
            return Ok(tokens);
        }
        Ok(gate_proposals(tokens, &sp.draft_probs.to_f32()?[..drafts]))
    }

    /// Encodes the draft pass that follows a verify pass of `m` rows at
    /// `pos0`, without committing. Nothing about the accepted count is known
    /// on the host: the pass decides it (`spec_accept`) and its dispatches
    /// read the control words. It rolls the recurrent state back to the
    /// accepted prefix, sets the head's hidden to the accepted row's trunk
    /// hidden, catches the head up on every verified row (ids are the trunk's
    /// draws: the accepted rows' successors, garbage past them) and chains
    /// `chain` proposals into the spec scratch.
    #[allow(clippy::too_many_arguments)]
    fn encode_draft_selected<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        pos0: usize,
        m: usize,
        chain: usize,
        params: &SamplingParams,
        draw0: usize,
    ) -> Result<EncodedPass<'a>> {
        let cfg = &self.config;
        let wide = cfg.hc_width();
        let ratio = cfg.indexer.compress_ratio;
        let mtp = self
            .weights
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no draft head"))?;
        let mst =
            state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head state"))?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        let capacity = s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
        ensure!(
            (1..=MAX_DRAFTS + 1).contains(&m) && chain <= MAX_DRAFTS,
            "draft pass for {m} rows and {chain} proposals"
        );
        ensure!(
            pos0 + m + chain <= state.capacity(),
            "no room for the draft head's chain"
        );
        let ctrl = &sp.ctrl;
        let accepted = ctrl_word(ctrl, SLOT_ACCEPTED)?;
        let keep_rows = ctrl_word(ctrl, SLOT_KEEP)?;
        let chain_words: Vec<(Tensor, Tensor, Tensor)> = (0..chain)
            .map(|i| {
                Ok((
                    ctrl_word(ctrl, slot_pos(i))?,
                    ctrl_word(ctrl, slot_block(i))?,
                    ctrl_word(ctrl, slot_count(i))?,
                ))
            })
            .collect::<Result<_>>()?;

        let pass = ctx.begin_concurrent()?;
        pass.set_label("draft");
        #[cfg(target_os = "linux")]
        pass.set_short();
        let draws = sp.verify_tokens.view(0, &[m])?;
        spec_accept(
            ctx,
            &pass,
            &draws,
            &capacity.ids.view(0, &[m])?,
            ctrl,
            pos0,
            ratio,
            chain,
        )?;
        pass.level_barrier(&[ctrl])?;

        // Rollback to the accepted prefix. With every draft accepted the
        // state is already right: the mid-state copy skips (that row does
        // not exist) and the conv rewind keeping all m rows reproduces the
        // window the verify pass wrote.
        if m > 1 {
            let heads = cfg.linear_num_value_heads;
            let per_state = heads * GDN_HEAD_DIM * GDN_HEAD_DIM;
            let mut gdn_index = 0usize;
            for lstate in &state.layers {
                if let LayerState::Gdn { state: st, conv_windows } = lstate {
                    let mid = sp.mid[gdn_index].view(0, &[m - 1, per_state])?;
                    copy_row(ctx, &pass, &mid, Arg::Gpu(&accepted), st)?;
                    let x = prefix_rows(&sp.conv_in[gdn_index], m)?;
                    conv_window_rollback(
                        ctx,
                        &pass,
                        &conv_windows[1 - state.conv_slot],
                        &x,
                        &conv_windows[state.conv_slot],
                        Pos::gpu(&keep_rows, 1, m),
                    )?;
                    gdn_index += 1;
                }
            }
            if let (Some(pst), Some(x)) = (&state.ple, &sp.ple_conv_in) {
                let x = prefix_rows(x, m)?;
                conv_window_rollback(
                    ctx,
                    &pass,
                    &pst.conv_windows[1 - state.conv_slot],
                    &x,
                    &pst.conv_windows[state.conv_slot],
                    Pos::gpu(&keep_rows, 1, m),
                )?;
            }
            pass.level_barrier(&[])?;
        }

        // The head's next input: the trunk hidden of the accepted row.
        let hidden = capacity.hyper.view(0, &[m, wide])?;
        copy_row(ctx, &pass, &hidden, Arg::Gpu(&accepted), &mst.hidden)?;
        // Catch-up over every verified row; the head's caches are position
        // indexed, so the rows past the accepted one are overwritten by the
        // chain (and by the next step) before anything valid reads them.
        // Every row here is a generated token: its rotary position is the
        // sequence index plus the prompt's delta, and no image row applies.
        let rope = Rope::Delta(state.rope_delta);
        let ps = capacity.rows(m)?;
        self.mtp_block(
            ctx,
            &pass,
            mtp,
            mst,
            &hidden,
            &draws,
            AttnPos::host(pos0),
            s,
            &ps,
            rope,
            &[],
        )?;
        if chain > 0 {
            let hyper = ps
                .mtp_hyper
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("draft head without scratch"))?;
            copy_row(
                ctx,
                &pass,
                &hyper.view(0, &[m, wide])?,
                Arg::Gpu(&accepted),
                &sp.chain_in,
            )?;
            pass.level_barrier(&[&sp.chain_in])?;
            let ps1 = capacity.rows(1)?;
            let head_params = draft_params(params);
            let mut last = sp.chain_in.view(0, &[1, wide])?;
            for (i, (pos, block, count)) in chain_words.iter().enumerate() {
                // Head: mixer over the last residual row, the head's LM head
                // (the trunk's, or its 4-bit copy), argmax.
                self.hc_read_batched(ctx, &pass, &mtp.mixer, &last, s, &ps1)?;
                let logits =
                    self.draft_head_logits(
                    ctx,
                    &pass,
                    mtp,
                    &ps1.hc.mixed,
                    sp,
                    s,
                    self.draft_wide(state),
                )?;
                let out = sp.draft_tokens.view(i, &[1])?;
                draft_draw(
                    ctx,
                    &pass,
                    &logits,
                    s,
                    sp,
                    &head_params,
                    draft_step(draw0 + i),
                    i,
                    self.draft_gated(),
                )?;
                pass.level_barrier(&[&sp.draft_tokens])?;
                if i + 1 < chain {
                    // Chain: the head's own residual stands in for the trunk
                    // hidden of the token it just proposed, at position
                    // pos0 + accepted + 1 + i (the GPU knows which).
                    let at =
                        AttnPos::gpu(pos, pos0 + 1 + i, pos0 + m + i, block, count);
                    self.mtp_block(
                        ctx,
                        &pass,
                        mtp,
                        mst,
                        &last,
                        &out,
                        at,
                        s,
                        &ps1,
                        rope,
                        &[],
                    )?;
                    last = hyper.view(0, &[1, wide])?;
                }
            }
        }
        s.sync.signal_done(&pass)?;
        pass.end()
    }

    /// Encodes a host-driven draft pass without committing: optional rollback
    /// of the recurrent state to `accepted` of `m` verified rows, the head's
    /// catch-up over `rows` rows (`hidden` rows paired with the ids in the
    /// spec scratch's `mtp_ids`, head positions from `pos0`), `keep` copied
    /// into the state's hidden, and `drafts` chained proposals from the
    /// residual of catch-up row `chain_row` (chain positions follow it) into
    /// the spec scratch. Used for the
    /// first proposals of a request, for rolling back past the GPU's count,
    /// and as the reference the GPU-selected pass is tested against.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_draft<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        hidden: &Tensor,
        rows: usize,
        chain_row: usize,
        pos0: usize,
        keep: Option<&Tensor>,
        rollback: Option<(usize, usize)>,
        drafts: usize,
        params: &SamplingParams,
        draw0: usize,
    ) -> Result<EncodedPass<'a>> {
        let cfg = &self.config;
        let wide = cfg.hc_width();
        let mtp = self
            .weights
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no draft head"))?;
        let mst =
            state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head state"))?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        let capacity = s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
        let head_params = draft_params(params);
        ensure!(rows <= MAX_DRAFTS + 1, "too many catch-up rows");
        ensure!(
            drafts == 0 || chain_row < rows,
            "drafts need a catch-up row to follow"
        );

        let pass = ctx.begin_concurrent()?;
        pass.set_label("draft");
        #[cfg(target_os = "linux")]
        pass.set_short();
        if let Some((accepted, m)) = rollback {
            let heads = cfg.linear_num_value_heads;
            let per_state = heads * GDN_HEAD_DIM * GDN_HEAD_DIM;
            let mut gdn_index = 0usize;
            for lstate in &state.layers {
                if let LayerState::Gdn { state: st, conv_windows } = lstate {
                    let mid = sp.mid[gdn_index].view(
                        accepted * per_state,
                        &[heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                    )?;
                    copy_words(ctx, &pass, &mid, st)?;
                    let x = prefix_rows(&sp.conv_in[gdn_index], m)?;
                    conv_window_rollback(
                        ctx,
                        &pass,
                        &conv_windows[1 - state.conv_slot],
                        &x,
                        &conv_windows[state.conv_slot],
                        accepted + 1,
                    )?;
                    gdn_index += 1;
                }
            }
            if let (Some(pst), Some(x)) = (&state.ple, &sp.ple_conv_in) {
                let x = prefix_rows(x, m)?;
                conv_window_rollback(
                    ctx,
                    &pass,
                    &pst.conv_windows[1 - state.conv_slot],
                    &x,
                    &pst.conv_windows[state.conv_slot],
                    accepted + 1,
                )?;
            }
            pass.level_barrier(&[])?;
        }
        if let Some(keep) = keep {
            copy_words(ctx, &pass, keep, &mst.hidden)?;
        }
        if rows > 0 {
            // Generated tokens only (see `encode_draft_selected`).
            let rope = Rope::Delta(state.rope_delta);
            let ps = capacity.rows(rows)?;
            let ids = sp.mtp_ids.view(0, &[rows])?;
            self.mtp_block(
                ctx,
                &pass,
                mtp,
                mst,
                hidden,
                &ids,
                AttnPos::host(pos0),
                s,
                &ps,
                rope,
                &[],
            )?;
            if drafts > 0 {
                let hyper = ps.mtp_hyper.as_ref().expect("draft head scratch");
                let ps1 = capacity.rows(1)?;
                let mut last = hyper.view(chain_row * wide, &[1, wide])?;
                for i in 0..drafts {
                    // Head: mixer over the last residual row, the head's LM
                    // head (the trunk's, or its 4-bit copy), argmax.
                    self.hc_read_batched(ctx, &pass, &mtp.mixer, &last, s, &ps1)?;
                    let logits = self.draft_head_logits(
                        ctx,
                        &pass,
                        mtp,
                        &ps1.hc.mixed,
                        sp,
                        s,
                        self.draft_wide(state),
                    )?;
                    let out = sp.draft_tokens.view(i, &[1])?;
                    draft_draw(
                        ctx,
                        &pass,
                        &logits,
                        s,
                        sp,
                        &head_params,
                        draft_step(draw0 + i),
                        i,
                        self.draft_gated(),
                    )?;
                    pass.level_barrier(&[&sp.draft_tokens])?;
                    if i + 1 < drafts {
                        // Chain: the head's own residual stands in for the trunk
                        // hidden of the token it just proposed.
                        self.mtp_block(
                            ctx,
                            &pass,
                            mtp,
                            mst,
                            &last,
                            &out,
                            AttnPos::host(pos0 + chain_row + 1 + i),
                            s,
                            &ps1,
                            rope,
                            &[],
                        )?;
                        last = hyper.view(0, &[1, wide])?;
                    }
                }
            }
        }
        s.sync.signal_done(&pass)?;
        pass.end()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/spec.rs"]
mod tests;
