//! DSpark, DeepSeek V4 Flash's block drafter (ds4's `--dspark`, the
//! `deepseek4-dspark` support GGUF): three extra layers that propose up to
//! five tokens at once from the target's own features. A child of `model`
//! (it builds its stages from the model's layer types and kernels).
//!
//! ds4 (MIT; ds4.c `metal_graph_eval_dspark_stage_block` and around):
//! - The target features of position p are the means of the four HC streams
//!   after layers 40, 41 and 42; `main_x(p) = rmsnorm(main_proj . feat)`.
//! - Each stage keeps a support cache of target KV rows, `fp8(rope_p(
//!   rmsnorm(attn_kv . main_x(p))))` (plain RoPE), for the positions up to
//!   the last fed one, `fp`: the drafter sees the last 128 of them. ds4's
//!   block row 0 is exactly the support row of `fp`, so here it is seeded
//!   into the cache with the others and the block has only the draft rows.
//! - The block: the pending token `t` and four noise tokens at positions
//!   `fp + 1 ..`, through the stages (HC, a non-causal attention over the
//!   support rows and the whole block, the MoE FFN), the final HC collapse
//!   and norm, and the model's own output head.
//! - Draft i is the argmax of its logits plus a rank-256 Markov term of the
//!   draft before it (`w2 . w1[prev]`, `prev = t` for the first); drafting
//!   stops at the first draft whose confidence `sigmoid(conf . [hidden_i,
//!   w1[prev]])` is below the threshold (ds4's Metal default, 0.6).
//!
//! Verification is exact: a draft only changes how many tokens a verify
//! pass keeps.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};

use super::{
    Config, Ds4Model, Ds4State, EXACT_ROWS, HcW, LayerW, Mapped, Run, Scratch, d, k,
    load_layer,
};
use crate::deepseek4::schema::{DsparkMeta, validate_dspark};
use crate::gguf::Gguf;
use crate::kernels::deepseek4::{CompRows, CompressorGpu, HeadOp, RawWindow};
use crate::kernels::glm53::QWeight;
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Proposals below this confidence end the block (ds4's Metal default).
pub const CONFIDENCE: f32 = 0.6;

/// The drafter's weights.
pub(crate) struct Dspark {
    mapped: Mapped,
    stages: Vec<LayerW>,
    main_proj: QWeight,
    main_norm: Tensor,
    norm: Tensor,
    hc_head: HcW,
    w1: QWeight,
    w2: QWeight,
    conf: QWeight,
    /// Drafts a block proposes.
    pub block: usize,
    noise: u32,
    /// The target layers whose features feed it, in order.
    pub targets: Vec<usize>,
    pub threshold: f32,
}

/// A session's drafter state: each stage's support ring (f16 `[128][512]`,
/// slot = position % 128) and which positions it holds.
pub(crate) struct DsState {
    rings: Vec<Tensor>,
    /// The seeded positions are `lo..end` (`end` = `None`: none).
    lo: usize,
    end: Option<usize>,
}

impl DsState {
    /// Forgets the seeded rows (a reset or a restore: the rings no longer
    /// match the state).
    pub fn invalidate(&mut self) {
        self.lo = 0;
        self.end = None;
    }

    /// Records `rows` seeded support rows from position `p0`.
    pub fn note(&mut self, p0: usize, rows: usize) {
        match self.end {
            // Contiguous with (or rewriting the end of) what is seeded.
            Some(end) if p0 >= self.lo && p0 <= end => self.end = Some(p0 + rows),
            _ => {
                self.lo = p0;
                self.end = Some(p0 + rows);
            }
        }
    }
}

/// The drafter's per-step buffers.
pub(crate) struct DsScratch {
    /// Target features of up to 128 rows: `[rows][3][hidden]`.
    feat: Tensor,
    mainx: Tensor,
    kv: Tensor,
    /// Per stage: the support rows of a run (f16 `[128][512]`), written to
    /// the ring at once or, for a verify pass, by `commit_rows`.
    kv16: Vec<Tensor>,
    /// The block's token ids (`t` and the noise tokens).
    emb_tok: Tensor,
    /// `t`, then the drafts (U32 `[block + 1]`).
    toks: Tensor,
    fvec: Tensor,
    conf: Tensor,
    mscore: Tensor,
    scores: Tensor,
}

impl Dspark {
    /// Opens and maps the support file, checked against the model.
    pub fn load(ctx: &MetalContext, path: &Path, cfg: &Config) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        let meta = DsparkMeta::read(&gguf, cfg, EXACT_ROWS - 1)
            .with_context(|| format!("DSpark support file {}", path.display()))?;
        validate_dspark(&gguf, cfg, &meta)
            .with_context(|| format!("DSpark support file {}", path.display()))?;
        let mapped = Mapped::map(ctx, &gguf)?;
        let stages = (0..meta.stages)
            .map(|s| load_layer(&mapped, &format!("mtp.{s}"), 0, false, false))
            .collect::<Result<Vec<_>>>()?;
        let last = format!("mtp.{}", meta.stages - 1);
        let f = |n: &str| mapped.f32(n);
        let q = |n: &str| mapped.q(n);
        let ds = Self {
            main_proj: q("mtp.0.main_proj.weight")?,
            main_norm: f("mtp.0.main_norm.weight")?,
            norm: f(&format!("{last}.norm.weight"))?,
            hc_head: HcW {
                fn_w: q(&format!("{last}.hc_head_fn.weight"))?,
                scale: f(&format!("{last}.hc_head_scale.weight"))?,
                base: f(&format!("{last}.hc_head_base.weight"))?,
            },
            w1: q(&format!("{last}.markov_head.markov_w1.weight"))?,
            w2: q(&format!("{last}.markov_head.markov_w2.weight"))?,
            conf: q(&format!("{last}.confidence_head.proj.weight"))?,
            mapped,
            stages,
            block: meta.block,
            noise: meta.noise as u32,
            targets: meta.targets,
            threshold: CONFIDENCE,
        };
        Ok(ds)
    }

    pub fn new_state(&self, ctx: &MetalContext) -> Result<DsState> {
        let rings = (0..self.stages.len())
            .map(|_| Tensor::zeros(ctx, &[d::WINDOW * d::KV_DIM], DType::F16))
            .collect::<Result<_>>()?;
        Ok(DsState { rings, lo: 0, end: None })
    }

    pub fn new_scratch(&self, ctx: &MetalContext, cfg: &Config) -> Result<DsScratch> {
        let f = |n: usize| Tensor::zeros(ctx, &[n], DType::F32);
        let u = |n: usize| Tensor::zeros(ctx, &[n], DType::U32);
        let rows = d::WINDOW;
        Ok(DsScratch {
            feat: f(rows * self.targets.len() * cfg.hidden)?,
            mainx: f(rows * cfg.hidden)?,
            kv: f(rows * cfg.head_dim)?,
            kv16: (0..self.stages.len())
                .map(|_| Tensor::zeros(ctx, &[rows * cfg.head_dim], DType::F16))
                .collect::<Result<_>>()?,
            emb_tok: u(EXACT_ROWS)?,
            toks: u(EXACT_ROWS)?,
            fvec: f(self.conf.cols)?,
            conf: f(EXACT_ROWS)?,
            mscore: f(cfg.vocab)?,
            scores: f(cfg.vocab)?,
        })
    }

    pub fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        self.mapped.buffers()
    }
}

impl Ds4Model {
    /// Loads the DSpark support file: sessions made from now on keep its
    /// state, and `max_drafts` proposes blocks.
    pub fn load_dspark(&mut self, ctx: &MetalContext, path: &Path) -> Result<()> {
        self.dspark = Some(Dspark::load(ctx, path, &self.cfg)?);
        Ok(())
    }

    /// Drafts a block proposes, 0 without the drafter.
    pub fn dspark_block(&self) -> usize {
        self.dspark.as_ref().map_or(0, |ds| ds.block)
    }

    /// The streams' means of rows `row0..row0 + rows` into feature slot
    /// `slot`.
    pub(super) fn capture(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        slot: usize,
        row0: usize,
        rows: usize,
    ) -> Result<()> {
        let (Some(ds), Some(dsc)) = (&self.dspark, &sc.ds) else { return Ok(()) };
        d::hc_capture(
            ctx,
            pass,
            &sc.h,
            &dsc.feat,
            row0,
            rows,
            slot,
            ds.targets.len(),
            self.cfg.hidden,
        )
    }

    /// The support rows of the captured `rows` (positions `p0..`): written
    /// to each stage's ring, or kept for `commit_rows` when `defer`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_seeds(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Ds4State,
        sc: &Scratch,
        p0: usize,
        rows: usize,
        defer: bool,
    ) -> Result<()> {
        let (Some(ds), Some(dsc), Some(dst)) = (&self.dspark, &sc.ds, &st.ds) else {
            return Ok(());
        };
        let cfg = &self.cfg;
        self.proj(ctx, pass, &[(&ds.main_proj, &dsc.mainx)], &dsc.feat, rows)?;
        pass.memory_barrier()?;
        k::rmsnorm(
            ctx,
            pass,
            &dsc.mainx,
            Some(&ds.main_norm),
            &dsc.mainx,
            rows,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        for (s, l) in ds.stages.iter().enumerate() {
            self.proj(ctx, pass, &[(&l.kv, &dsc.kv)], &dsc.mainx, rows)?;
            pass.memory_barrier()?;
            k::rmsnorm(
                ctx,
                pass,
                &dsc.kv,
                Some(&l.kv_norm),
                &dsc.kv,
                rows,
                cfg.head_dim,
                cfg.rms_eps,
            )?;
            pass.memory_barrier()?;
            d::kv_store(ctx, pass, &dsc.kv, &dsc.kv16[s], &st.rope_plain, p0, rows)?;
            if !defer {
                pass.memory_barrier()?;
                d::ring_store(ctx, pass, &dsc.kv16[s], &dst.rings[s], p0, rows)?;
            }
        }
        pass.memory_barrier()
    }

    /// Records that a run of `m` rows at the state's position seeded its
    /// last support rows (call before advancing the position).
    pub(super) fn note_seeded(&self, st: &mut Ds4State, m: usize) {
        let rows = m.min(d::WINDOW);
        let p0 = st.pos + m - rows;
        if let Some(ds) = st.ds.as_mut() {
            ds.note(p0, rows);
        }
    }

    /// Writes the first `rows` rows of the verify pass at `pos0` to the
    /// rings it deferred: each layer's raw window and compressor rings, and
    /// the drafter's support rings. GPU idle on the scratch after.
    pub(crate) fn commit_rows(
        &self,
        ctx: &MetalContext,
        st: &mut Ds4State,
        sc: &Scratch,
        pos0: usize,
        rows: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let pass = ctx.begin_concurrent()?;
        for (il, l) in self.layers.iter().enumerate() {
            let ls = &st.layers[il];
            let vb = &sc.verify[il];
            d::ring_store(ctx, &pass, &vb.kv_run, &ls.ring, pos0, rows)?;
            if let (Some(c), Some(s)) = (&l.comp, &ls.comp) {
                let cg = CompressorGpu {
                    ape: &c.ape,
                    norm: &c.norm,
                    ring_kv: &s.ring_kv,
                    ring_sc: &s.ring_sc,
                    dim: cfg.head_dim,
                    ratio: cfg.ratio(il),
                    indexer: false,
                };
                d::comp_ring(ctx, &pass, &cg, &vb.ckv, &vb.csc, pos0, rows)?;
            }
            if let (Some(ix), Some(s)) = (&l.indexer, &ls.index) {
                let cg = CompressorGpu {
                    ape: &ix.comp.ape,
                    norm: &ix.comp.norm,
                    ring_kv: &s.ring_kv,
                    ring_sc: &s.ring_sc,
                    dim: cfg.indexer_dim,
                    ratio: cfg.ratio(il),
                    indexer: true,
                };
                d::comp_ring(ctx, &pass, &cg, &vb.ikv, &vb.isc, pos0, rows)?;
            }
        }
        if let (Some(ds), Some(dsc), Some(dst)) = (&self.dspark, &sc.ds, &st.ds) {
            for s in 0..ds.stages.len() {
                d::ring_store(ctx, &pass, &dsc.kv16[s], &dst.rings[s], pos0, rows)?;
            }
        }
        pass.commit_wait()?;
        if let Some(dst) = st.ds.as_mut() {
            dst.note(pos0, rows);
        }
        Ok(())
    }

    /// A block of up to `limit` drafts following `t`, the token after the
    /// last fed position, from the support rows seeded up to it; empty when
    /// they are not (no drafter, a session just restored, nothing fed).
    /// Waits for the GPU.
    pub(crate) fn propose(
        &self,
        ctx: &MetalContext,
        st: &Ds4State,
        sc: &Scratch,
        t: u32,
        limit: usize,
    ) -> Result<Vec<u32>> {
        let (Some(ds), Some(dsc), Some(dst)) = (&self.dspark, &sc.ds, &st.ds) else {
            return Ok(Vec::new());
        };
        if limit == 0 || st.pos == 0 || dst.end != Some(st.pos) {
            return Ok(Vec::new());
        }
        let cfg = &self.cfg;
        let n = ds.block;
        let pos0 = st.pos;
        ensure!(pos0 + n <= st.capacity, "drafting past the state's capacity");
        let mut emb = vec![ds.noise; n];
        emb[0] = t;
        crate::glm53::model::write_words(&dsc.emb_tok, &emb)?;
        crate::glm53::model::write_words(&dsc.toks, &[t])?;
        let r = Run { pos0, m: n, tokens: &dsc.emb_tok, defer: false, traced: false };
        let pass = ctx.begin_concurrent()?;
        d::embed(ctx, &pass, &self.embed, &dsc.emb_tok, &sc.h, n)?;
        pass.memory_barrier()?;
        for (s, l) in ds.stages.iter().enumerate() {
            let prev = (s > 0).then_some(&sc.ffn_out);
            self.hc_pre(ctx, &pass, sc, &l.hc[0], &l.attn_norm, n, prev)?;
            self.draft_attention(
                ctx,
                &pass,
                sc,
                l,
                &dst.rings[s],
                &st.rope_plain,
                pos0,
                n,
                dst.lo,
            )?;
            self.hc_pre(ctx, &pass, sc, &l.hc[1], &l.ffn_norm, n, Some(&sc.attn_out))?;
            self.ffn(ctx, &pass, sc, l, r)?;
        }
        k::hc_post(ctx, &pass, &sc.h, &sc.ffn_out, &sc.split, &sc.h, n, cfg.hidden)?;
        pass.memory_barrier()?;
        // The final collapse (its output, before the norm, feeds the
        // confidence head) and the model's head.
        k::rmsnorm(ctx, &pass, &sc.h, None, &sc.flat, n, cfg.hc_dim(), cfg.rms_eps)?;
        pass.memory_barrier()?;
        self.proj(ctx, &pass, &[(&ds.hc_head.fn_w, &sc.opre)], &sc.flat, n)?;
        pass.memory_barrier()?;
        d::out_collapse(
            ctx,
            &pass,
            &sc.opre,
            &ds.hc_head.scale,
            &ds.hc_head.base,
            &sc.h,
            &sc.hidden,
            n,
            cfg.hidden,
            cfg.hc_eps,
        )?;
        pass.memory_barrier()?;
        k::rmsnorm(
            ctx,
            &pass,
            &sc.hidden,
            Some(&ds.norm),
            &sc.hidden_norm,
            n,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        self.proj(ctx, &pass, &[(&self.output, &sc.logits)], &sc.hidden_norm, n)?;
        pass.memory_barrier()?;
        // Draft by draft: the Markov term of the previous token, the
        // confidence, the argmax.
        let rank = ds.w1.cols;
        let state = dsc.fvec.view(cfg.hidden, &[rank])?;
        let greedy = SamplingParams::greedy();
        for i in 0..n {
            let hid = sc.hidden.view(i * cfg.hidden, &[cfg.hidden])?;
            d::markov_prep(
                ctx, &pass, &ds.w1, &dsc.toks, i, &hid, &dsc.fvec, cfg.hidden,
            )?;
            pass.memory_barrier()?;
            let conf = dsc.conf.view(i, &[1])?;
            k::gemv(ctx, &pass, &ds.conf, &dsc.fvec, &conf, 1)?;
            k::gemv(ctx, &pass, &ds.w2, &state, &dsc.mscore, 1)?;
            pass.memory_barrier()?;
            let logits = sc.logits.view(i * cfg.vocab, &[cfg.vocab])?;
            k::add(ctx, &pass, &logits, &dsc.mscore, &dsc.scores, cfg.vocab)?;
            pass.memory_barrier()?;
            let out = dsc.toks.view(i + 1, &[1])?;
            sample_f32(ctx, &pass, &dsc.scores, &sc.sampler, &greedy, 0, &out)?;
            pass.memory_barrier()?;
        }
        pass.commit_wait()?;
        let toks = dsc.toks.to_u32()?;
        let conf = dsc.conf.to_f32()?;
        Ok((0..n.min(limit))
            .take_while(|&i| 1.0 / (1.0 + (-conf[i]).exp()) >= ds.threshold)
            .map(|i| toks[i + 1])
            .collect())
    }

    /// A stage's attention over its block (`n` rows at `pos0..`): every row
    /// sees the support rows (`lo..pos0`, the last 128) and the whole
    /// block. Plain RoPE (`table`, by position) as the model's first layers.
    #[allow(clippy::too_many_arguments)]
    fn draft_attention(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        l: &LayerW,
        ring: &Tensor,
        table: &Tensor,
        pos0: usize,
        n: usize,
        lo: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        self.proj(ctx, pass, &[(&l.q_a, &sc.qr), (&l.kv, &sc.kv)], &sc.xn, n)?;
        pass.memory_barrier()?;
        k::rmsnorm(
            ctx,
            pass,
            &sc.qr,
            Some(&l.q_a_norm),
            &sc.qr,
            n,
            cfg.q_lora,
            cfg.rms_eps,
        )?;
        k::rmsnorm(
            ctx,
            pass,
            &sc.kv,
            Some(&l.kv_norm),
            &sc.kv,
            n,
            cfg.head_dim,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        self.proj(ctx, pass, &[(&l.q_b, &sc.q)], &sc.qr, n)?;
        d::kv_store(ctx, pass, &sc.kv, &sc.kv_run, table, pos0, n)?;
        pass.memory_barrier()?;
        d::head_norm_rope(
            ctx,
            pass,
            &sc.q,
            table,
            pos0,
            n,
            cfg.n_head,
            cfg.head_dim,
            cfg.n_rot,
            cfg.rms_eps,
            HeadOp::NormRope,
        )?;
        pass.memory_barrier()?;
        let raw = RawWindow { at: Some(pos0 + n - 1), window: d::WINDOW + n, lo };
        d::attend(
            ctx,
            pass,
            &sc.q,
            &sc.kv_run,
            ring,
            CompRows::None,
            &l.sinks,
            &sc.heads,
            n,
            cfg.n_head,
            pos0,
            raw,
        )?;
        pass.memory_barrier()?;
        d::head_norm_rope(
            ctx,
            pass,
            &sc.heads,
            table,
            pos0,
            n,
            cfg.n_head,
            cfg.head_dim,
            cfg.n_rot,
            0.0,
            HeadOp::InverseRope,
        )?;
        pass.memory_barrier()?;
        self.grouped_out(ctx, pass, sc, l, n, false)
    }
}
