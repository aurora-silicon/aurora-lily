//! Gemma 4 on the GPU: weights mapped read-only from the GGUF, the
//! per-session K/V caches, and the forward pass over up to
//! [`PREFILL_ROWS`] consecutive tokens (decode is one row; verify passes
//! up to [`MAX_ROWS`]).
//!
//! The graph is llama.cpp's (`src/models/gemma4.cpp`): per layer
//! attention (sliding-window or global), then a dense GeGLU MLP and the
//! routed experts side by side, each with its own norms, summed, normed,
//! added to the residual and scaled by the layer's output scale.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};

use super::config::Config;
use crate::gguf::Gguf;
use crate::glm53::model::Mapped;
use crate::kernels::gemma4::{self as g4, AttnShape};
use crate::kernels::glm53::{self as k, QWeight};
use crate::kernels::sample::SamplerScratch;
use crate::metal::{Buffer, ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Rows one step takes on the GEMV path (decode, verify, scoring).
pub const MAX_ROWS: usize = k::GEMV_MAX_ROWS;

/// The most rows a prompt run takes ([`Gemma4Model::prefill_rows`]: fewer
/// under a memory limit, the scratch growing ~0.3 MB a row). Each routed
/// expert's weights are decoded once a run, so longer runs pay: 4K-token
/// prompts 2366 -> 3462 tok/s (UD-IQ2_M) and 2111 -> 3184 (UD-Q5_K_M) at
/// 1024 rows against 256 (2026-11-15).
pub const PREFILL_ROWS: usize = 1024;

/// The rows a prompt run's K/V store and attention take at a time (in
/// turn within each layer): the sliding-window layers' rings hold the
/// window plus this many, whatever the run.
pub const ATTN_ROWS: usize = 256;

/// The longest context lily serves Gemma at: the server's limit (positions
/// and cache offsets fit the kernels' u32 arithmetic well past it).
pub const MAX_SEQ: usize = crate::kernels::attention::MAX_SEQ;

/// Rows a state holds past its context: a verify pass at the context's last
/// position feeds the pending token and up to three drafts.
pub const LOOKAHEAD: usize = MAX_ROWS;

/// The rows a state of `context` tokens may grow to.
pub fn capacity_cap(context: usize) -> usize {
    (context + LOOKAHEAD).min(MAX_SEQ)
}

/// Fails on a token id outside the vocabulary (the embedding would read
/// past its rows).
pub fn check_tokens(tokens: &[u32], vocab: usize) -> Result<()> {
    if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
        anyhow::bail!("token id {t} is outside the vocabulary of {vocab}");
    }
    Ok(())
}

struct LayerW {
    attn_norm: Tensor,
    q: QWeight,
    k: QWeight,
    /// None: V is the K projection (`attention_k_eq_v`, the global layers).
    v: Option<QWeight>,
    o: QWeight,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attn_norm: Tensor,
    ffn_norm: Tensor,
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    post_ffw_norm_1: Tensor,
    router: QWeight,
    /// `ffn_gate_inp.scale / sqrt(hidden)`: the router input's norm weight.
    router_w: Tensor,
    pre_ffw_norm_2: Tensor,
    gate_up_exps: QWeight,
    down_exps: QWeight,
    down_scale: Tensor,
    post_ffw_norm_2: Tensor,
    post_ffw_norm: Tensor,
    out_scale: Tensor,
}

/// The model's weights on the GPU.
pub struct Gemma4Model {
    pub cfg: Config,
    pub gguf: Gguf,
    mapped: Mapped,
    layers: Vec<LayerW>,
    /// The token embedding, also the LM head (tied).
    embed: QWeight,
    output_norm: Tensor,
    rope_freqs: Tensor,
    /// The draft model (`--mtp-drafts`), when one was found and loaded.
    pub draft: Option<super::draft::Gemma4Draft>,
    /// Drafts a speculative step proposes.
    pub drafts: usize,
    /// A draft is proposed only when the draft model's softmax probability
    /// of it is at least this (the chain stops at the first that is not).
    pub draft_min_prob: f32,
    /// `--memory-limit-gb` in bytes: the footprint the server keeps to.
    pub memory_limit: Option<u64>,
    /// Rows a prompt run takes (at most [`PREFILL_ROWS`]; scratches are
    /// sized by it).
    pub prefill_rows: usize,
    /// Q5_K attention and dense weights (UD-IQ2_M), not on a G16 GPU:
    /// steps of up to MAX_ROWS rows run their projections over int8
    /// activations, which the kernels before them write.
    q8x: bool,
    /// Q5_K routed experts' gate/up (UD-Q5_K_M), not on a G16 GPU: steps of
    /// up to MAX_ROWS rows run them over the int8 form of their input, which
    /// the residual's norms write.
    q8x_experts: bool,
    /// The longest context served: the file's, the server's limit, or what
    /// fits the memory limit ([`Gemma4Model::fit_context`]).
    pub context_cap: usize,
    /// What the session store keeps free for the request's session under a
    /// fitted `--memory-limit-gb`: the session's peak at the fitted context.
    pub acquire_reserve: Option<u64>,
}

impl Gemma4Model {
    /// Opens and maps `path` (the GGUF file), after taking the instance
    /// lock (one model on this machine at a time).
    pub fn load(ctx: &MetalContext, path: &Path) -> Result<Self> {
        Self::load_locked(ctx, path, crate::instance::acquire)
    }

    pub(crate) fn load_locked(
        ctx: &MetalContext,
        path: &Path,
        lock: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        lock()?;
        let gguf = Gguf::open(path)?;
        let cfg = Config::from_gguf(&gguf)?;
        ensure!(
            gguf.tensor("output.weight").is_none(),
            "an untied output.weight (lily ties the head to token_embd as Gemma does)"
        );
        super::schema::validate_gguf(&super::schema::target(&cfg), &gguf)?;
        let mapped = Mapped::map(ctx, &gguf)?;
        let h = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{il}.{s}");
            let q = |s: &str| mapped.q(&n(s));
            // Shapes and types as the schema checked them.
            let f = |s: &str, len: usize| -> Result<Tensor> {
                let t = mapped.f32(&n(s))?;
                debug_assert_eq!(t.numel(), len, "{}", n(s));
                Ok(t)
            };
            let d = cfg.head_dim_of(il);
            let l = LayerW {
                attn_norm: f("attn_norm.weight", h)?,
                q: q("attn_q.weight")?,
                k: q("attn_k.weight")?,
                v: if gguf.tensor(&n("attn_v.weight")).is_some() {
                    Some(q("attn_v.weight")?)
                } else {
                    None
                },
                o: q("attn_output.weight")?,
                q_norm: f("attn_q_norm.weight", d)?,
                k_norm: f("attn_k_norm.weight", d)?,
                post_attn_norm: f("post_attention_norm.weight", h)?,
                ffn_norm: f("ffn_norm.weight", h)?,
                gate: q("ffn_gate.weight")?,
                up: q("ffn_up.weight")?,
                down: q("ffn_down.weight")?,
                post_ffw_norm_1: f("post_ffw_norm_1.weight", h)?,
                router: q("ffn_gate_inp.weight")?,
                router_w: {
                    let s = f("ffn_gate_inp.scale", h)?.to_f32()?;
                    let r = 1.0 / (h as f32).sqrt();
                    let w: Vec<f32> = s.iter().map(|&v| v * r).collect();
                    Tensor::from_f32(ctx, &w, &[h])?
                },
                pre_ffw_norm_2: f("pre_ffw_norm_2.weight", h)?,
                gate_up_exps: q("ffn_gate_up_exps.weight")?,
                down_exps: q("ffn_down_exps.weight")?,
                down_scale: f("ffn_down_exps.scale", cfg.n_expert)?,
                post_ffw_norm_2: f("post_ffw_norm_2.weight", h)?,
                post_ffw_norm: f("post_ffw_norm.weight", h)?,
                out_scale: f("layer_output_scale.weight", 1)?,
            };
            layers.push(l);
        }
        // A G16 GPU (an M4) keeps its own Q5_K GEMVs (`gemv_g16.comp`).
        let q8x = !ctx.g16()
            && layers.first().is_some_and(|l| l.q.ty == crate::gguf::GgmlType::Q5K);
        let q8x_experts = !ctx.g16()
            && layers
                .first()
                .is_some_and(|l| l.gate_up_exps.ty == crate::gguf::GgmlType::Q5K);
        let embed = mapped.q("token_embd.weight")?;
        let output_norm = mapped.f32("output_norm.weight")?;
        let rope_freqs = mapped.f32("rope_freqs.weight")?;
        Ok(Self {
            gguf,
            mapped,
            layers,
            embed,
            output_norm,
            rope_freqs,
            context_cap: cfg.context_length.min(MAX_SEQ),
            cfg,
            draft: None,
            drafts: 0,
            draft_min_prob: 0.0,
            memory_limit: None,
            acquire_reserve: None,
            prefill_rows: PREFILL_ROWS,
            q8x,
            q8x_experts,
        })
    }

    /// The buffers holding the mapped file.
    pub fn weight_buffers(&self) -> Vec<Buffer> {
        self.mapped.buffers()
    }

    /// The token embedding (also the head).
    pub(crate) fn embed(&self) -> &QWeight {
        &self.embed
    }

    /// Layer `il`'s K norm weight (attention reads the keys its K cache
    /// leaves out as V times it).
    pub(crate) fn k_norm(&self, il: usize) -> &Tensor {
        &self.layers[il].k_norm
    }

    /// The layer whose caches the draft model's sliding-window (or global)
    /// layers read: the target's last of that kind (llama.cpp's
    /// `n_layer - 2` and `n_layer - 1` for Gemma 4).
    pub(crate) fn shared_kv_layer(&self, swa: bool) -> Result<usize> {
        (0..self.cfg.n_layer)
            .rev()
            .find(|&il| self.cfg.swa[il] == swa)
            .context("the target has no layer of the kind the draft model reads")
    }

    pub fn new_state(
        &self,
        ctx: &MetalContext,
        capacity: usize,
    ) -> Result<Gemma4State> {
        Gemma4State::new(ctx, &self.cfg, capacity, capacity_cap(self.context_cap))
    }

    pub fn new_scratch(&self, ctx: &MetalContext, capacity: usize) -> Result<Scratch> {
        Scratch::new(ctx, &self.cfg, capacity, self.prefill_rows)
    }
}

/// Slots of a sliding-window layer's ring.
pub fn ring_slots(cfg: &Config) -> usize {
    cfg.window + ATTN_ROWS
}

/// GPU bytes of one session's caches for `capacity` tokens (the global
/// layers' f16 K and V), its sliding-window rings and the draft model's
/// input, and `checkpoints` snapshots of the rings' windows and that input.
pub fn session_bytes(cfg: &Config, capacity: usize, checkpoints: usize) -> u64 {
    let per_row = |swa: bool| -> usize {
        (0..cfg.n_layer)
            .filter(|&il| cfg.swa[il] == swa)
            .map(|il| 2 * (cfg.k_cache_width(il) + cfg.kv_width(il)))
            .sum()
    };
    // Each snapshot also keeps the draft model's input (4 B a hidden value).
    (per_row(false) * capacity
        + per_row(true) * (ring_slots(cfg) + checkpoints * cfg.window)
        + 4 * cfg.hidden * (1 + checkpoints)) as u64
}

/// The most GPU memory one session holds on its way to `capacity` rows
/// with `checkpoints` snapshots: [`session_bytes`] plus the one global
/// layer's K and V that growth holds twice while it replaces them (the old
/// ones are freed layer by layer, after the copy).
pub fn session_peak(cfg: &Config, capacity: usize, checkpoints: usize) -> u64 {
    let widest = (0..cfg.n_layer)
        .filter(|&il| !cfg.swa[il])
        .map(|il| cfg.k_cache_width(il) + cfg.kv_width(il))
        .max()
        .unwrap_or(0);
    session_bytes(cfg, capacity, checkpoints) + (2 * widest * capacity) as u64
}

/// One layer's K and V caches, f16 `[slots][kv heads][head dim]` (K: the
/// layer's `Config::k_rot` pairs a head).
pub struct LayerCache {
    pub k: Tensor,
    pub v: Tensor,
    /// Slots: the state's capacity (global layers) or the ring's.
    pub slots: usize,
    pub swa: bool,
}

/// A session: the caches at position `pos`.
pub struct Gemma4State {
    pub pos: usize,
    pub capacity: usize,
    /// The rows the global layers' caches may grow to.
    pub max_capacity: usize,
    pub layers: Vec<LayerCache>,
    /// F32 [hidden]: the post-norm hidden of the last fed position, the
    /// draft model's input (a snapshot keeps it, so a restored or forked
    /// state drafts as the state it was taken from).
    pub h_last: Tensor,
    /// A verify pass that fed rows from this position and the draws of its
    /// rows, until `finish_speculation` decides what stays.
    pub(crate) spec: Option<(usize, usize, Vec<u32>)>,
    /// The sliding window: what a snapshot keeps of each ring.
    pub window: usize,
}

impl Gemma4State {
    pub fn new(
        ctx: &MetalContext,
        cfg: &Config,
        capacity: usize,
        max_capacity: usize,
    ) -> Result<Self> {
        ensure!(
            capacity <= max_capacity && max_capacity <= capacity_cap(MAX_SEQ),
            "a session of {capacity} tokens exceeds the {max_capacity} rows this model serves"
        );
        let layers = (0..cfg.n_layer)
            .map(|il| -> Result<LayerCache> {
                let swa = cfg.swa[il];
                let slots = if swa { ring_slots(cfg) } else { capacity };
                Ok(LayerCache {
                    k: Tensor::zeros(ctx, &[slots * cfg.k_cache_width(il)], DType::F16)?,
                    v: Tensor::zeros(ctx, &[slots * cfg.kv_width(il)], DType::F16)?,
                    slots,
                    swa,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            pos: 0,
            capacity,
            max_capacity,
            layers,
            h_last: Tensor::zeros(ctx, &[cfg.hidden], DType::F32)?,
            spec: None,
            window: cfg.window,
        })
    }

    /// Bytes of the caches and the draft model's input.
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| l.k.byte_len() + l.v.byte_len()).sum::<usize>()
            + self.h_last.byte_len()
    }
}

/// Per-step intermediates for up to `rows` rows.
pub struct Scratch {
    /// The rows a run may take.
    rows: usize,
    pub tokens: Tensor,
    pub x: Tensor,
    h: Tensor,
    /// BF16: a prompt run's projection input (the bfloat-input GEMMs';
    /// matrix units only).
    xb: Tensor,
    /// A step's projection input as int8 codes and per 32 values (scale,
    /// sum of codes), rows of [`q8_width`] values.
    xq8: Tensor,
    xds: Tensor,
    q8_width: usize,
    /// The routed experts' input (n2) as int8, rows of `hidden` values.
    eq8: Tensor,
    eds: Tensor,
    q: Tensor,
    kv: Tensor,
    vraw: Tensor,
    qo: Tensor,
    heads: Tensor,
    attn: Tensor,
    r: Tensor,
    n1: Tensor,
    n2: Tensor,
    t: Tensor,
    g: Tensor,
    u: Tensor,
    mid: Tensor,
    m1: Tensor,
    m2: Tensor,
    rlogits: Tensor,
    sel: Tensor,
    rw: Tensor,
    emid: Tensor,
    moe_group: g4::MoeGroupScratch,
    partial: Tensor,
    /// The partials' capacity in key chunks.
    chunks: usize,
    pub logits: Tensor,
    /// The last (or only) row's logits: `logits[0..vocab]`.
    pub logits_last: Tensor,
    sampler: SamplerScratch,
    /// U32[2]: the ping-pong slots the in-graph sampler draws into.
    pub next_token: Tensor,
    /// U32[MAX_ROWS]: a verify pass's draws, one a row.
    pub draws: Tensor,
    vocab: usize,
}

/// The widest input a step's projections may take as int8 (a row of
/// [`Scratch`]'s int8 buffers): the layer's input and the residual's norms
/// (hidden), the attention's heads (the query width) and the dense MLP's
/// down projection's (ff).
pub(crate) fn q8_width(cfg: &Config) -> usize {
    cfg.hidden.max(cfg.max_q_width()).max(cfg.ff)
}

impl Scratch {
    /// The int8 buffers' row width, code bytes and scale floats.
    #[cfg(test)]
    pub(crate) fn q8_buffers(&self) -> (usize, usize, usize) {
        (self.q8_width, self.xq8.byte_len(), self.xds.numel())
    }

    /// Scratch for states of up to `capacity` tokens and runs of up to
    /// `rows` rows.
    pub fn new(ctx: &MetalContext, cfg: &Config, capacity: usize, rows: usize) -> Result<Self> {
        ensure!((MAX_ROWS..=PREFILL_ROWS).contains(&rows), "runs of {rows} rows");
        let m = rows;
        let f = |n: usize| Tensor::zeros(ctx, &[n], DType::F32);
        let h = cfg.hidden;
        let chunks = capacity.max(1).div_ceil(g4::ATTN_CHUNK);
        let partial = g4::attn_partial_len(
            MAX_ROWS,
            cfg.n_head,
            cfg.head_dim.max(cfg.head_dim_swa),
            chunks,
        );
        let logits = f(MAX_ROWS * cfg.vocab)?;
        // One pool for the attention's q, qo, heads and raw K/V and the
        // grouped experts' f32 outputs: the experts run after the attention
        // output projection has read heads, and the next layer's q waits
        // for the experts (each ~0.1 MB a run row).
        let (qw, kvw) = (cfg.max_q_width(), cfg.max_kv_width());
        let moe = g4::MoeGroupScratch::pool_len(m, cfg.n_expert_used, cfg.ff_expert, h);
        let pool = f((3 * m * qw + 2 * m * kvw).max(moe))?;
        // Within a layer: h (its input) is dead from the projections to
        // the FFN's end, so it holds the attention output; n1 and t are
        // dead after the dense gate/up and the router read them, so they
        // hold the dense MLP's and the experts' outputs; GeGLU writes its
        // gate in place.
        let (hb, n1, t, g) = (f(m * h)?, f(m * h)?, f(m * h)?, f(m * cfg.ff)?);
        Ok(Self {
            rows: m,
            logits_last: logits.view(0, &[cfg.vocab])?,
            tokens: Tensor::zeros(ctx, &[m], DType::U32)?,
            x: f(m * h)?,
            h: hb.clone(),
            xq8: Tensor::zeros(ctx, &[MAX_ROWS * q8_width(cfg)], DType::I8)?,
            xds: f(MAX_ROWS * q8_width(cfg).div_ceil(32) * 2)?,
            q8_width: q8_width(cfg),
            eq8: Tensor::zeros(ctx, &[MAX_ROWS * h], DType::I8)?,
            eds: f(MAX_ROWS * h.div_ceil(32) * 2)?,
            xb: Tensor::zeros(
                ctx,
                &[if ctx.matrix_units() {
                    m * cfg.max_q_width().max(cfg.ff).max(h)
                } else {
                    2
                }],
                DType::BF16,
            )?,
            q: pool.view(0, &[m * qw])?,
            kv: pool.view(3 * m * qw, &[m * kvw])?,
            vraw: pool.view(3 * m * qw + m * kvw, &[m * kvw])?,
            qo: pool.view(m * qw, &[m * qw])?,
            heads: pool.view(2 * m * qw, &[m * qw])?,
            attn: hb,
            r: f(m * h)?,
            n2: f(m * h)?,
            u: f(m * cfg.ff)?,
            mid: g.clone(),
            g,
            m1: t.clone(),
            m2: n1.clone(),
            n1,
            t,
            rlogits: f(m * cfg.n_expert)?,
            sel: Tensor::zeros(ctx, &[m * cfg.n_expert_used], DType::U32)?,
            rw: f(m * cfg.n_expert_used)?,
            emid: f(MAX_ROWS * cfg.n_expert_used * cfg.ff_expert)?,
            moe_group: g4::MoeGroupScratch::in_pool(
                ctx,
                m,
                cfg.n_expert_used,
                cfg.n_expert,
                cfg.ff_expert,
                h,
                &pool,
            )?,
            partial: f(partial)?,
            chunks,
            logits,
            sampler: SamplerScratch::new(ctx, cfg.vocab)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
            draws: Tensor::zeros(ctx, &[MAX_ROWS], DType::U32)?,
            vocab: cfg.vocab,
        })
    }

    /// Stages a run's token ids (at most the scratch's rows, each in the
    /// vocabulary).
    pub(crate) fn stage_tokens(&self, tokens: &[u32]) -> Result<()> {
        ensure!(tokens.len() <= self.rows, "a run of {} tokens", tokens.len());
        check_tokens(tokens, self.vocab)?;
        crate::glm53::model::write_words(&self.tokens, tokens)
    }

    /// The sampler's buffers.
    pub fn sampler(&self) -> &SamplerScratch {
        &self.sampler
    }

    /// The decode attention's partials (the draft model's attention uses
    /// them too).
    pub(crate) fn partial(&self) -> &Tensor {
        &self.partial
    }

    /// Row `row` of `h`: a position's post-norm hidden after a step.
    pub fn h_row(&self, row: usize, hidden: usize) -> Result<Tensor> {
        self.h.view(row * hidden, &[hidden])
    }
}

/// Which rows' logits a forward step computes into `Scratch::logits`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Logits {
    None,
    /// The last row's, at `logits[0..vocab]`.
    Last,
    /// Every row's, row `i` at `logits[i * vocab..]` (at most MAX_ROWS).
    All,
}

impl Gemma4Model {
    /// The pass a step is encoded into: concurrent, its dependency levels
    /// separated by barriers; `LILY_GEMMA4_SERIAL=1` makes every dispatch
    /// its own level.
    pub fn begin<'c>(&self, ctx: &'c MetalContext) -> Result<ComputePass<'c>> {
        static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SERIAL.get_or_init(|| std::env::var_os("LILY_GEMMA4_SERIAL").is_some()) {
            ctx.begin()
        } else {
            ctx.begin_concurrent()
        }
    }

    /// Projections of one input `x`: GEMVs (weights of one type and width
    /// share a dispatch, up to four) for up to MAX_ROWS rows, the
    /// matrix-unit GEMM beyond.
    fn proj(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        sets: &[(&QWeight, &Tensor)],
        x: &Tensor,
        m: usize,
    ) -> Result<()> {
        self.proj_q8(ctx, pass, sc, sets, x, m, false)
    }

    /// [`Self::proj`]; `pre`: the step's producer left `x` as int8 in the
    /// scratch already.
    #[allow(clippy::too_many_arguments)]
    fn proj_q8(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        sets: &[(&QWeight, &Tensor)],
        x: &Tensor,
        m: usize,
        pre: bool,
    ) -> Result<()> {
        if m > MAX_ROWS {
            // The input once as bfloat for the weights the bfloat-input
            // GEMM takes (the same products as the f32-input one, which
            // rounds X in every column tile): UD-IQ2_M's 4K prompt
            // 2115 -> 2378 tok/s, 2026-11-15.
            let cols = sets[0].0.cols;
            let bx = ctx.matrix_units()
                && sets.iter().all(|(w, _)| w.cols == cols && k::gemm_bx_takes(w));
            if bx {
                k::to_bf16(ctx, pass, x, &sc.xb, m * cols)?;
                pass.memory_barrier()?;
            }
            for &(w, y) in sets {
                if bx {
                    k::gemm_bx(ctx, pass, w, &sc.xb, y, m)?;
                } else {
                    k::gemm(ctx, pass, w, x, y, m)?;
                }
            }
            return Ok(());
        }
        // Q5_K projections over int8 activations.
        let cols = sets[0].0.cols;
        if self.q8x
            && cols.is_multiple_of(256)
            && cols >= 1024
            && cols <= sc.q8_width
            && sets
                .iter()
                .all(|(w, _)| w.ty == crate::gguf::GgmlType::Q5K && w.cols == cols)
        {
            if !pre {
                g4::quant_q8x(ctx, pass, x, &sc.xq8, &sc.xds, m, cols)?;
                pass.memory_barrier()?;
            }
            for group in sets.chunks(4) {
                g4::gemv_q5_k_q8x(ctx, pass, group, &sc.xq8, &sc.xds, m)?;
            }
            return Ok(());
        }
        let mut left: Vec<(&QWeight, &Tensor)> = sets.to_vec();
        while let Some(&(w0, _)) = left.first() {
            let (same, rest): (Vec<_>, Vec<_>) = left.iter().partition(|(w, _)| {
                w.ty == w0.ty && w.cols == w0.cols && w.row_bytes == w0.row_bytes
            });
            for group in same.chunks(4) {
                k::gemv_sets(ctx, pass, group, x, m)?;
            }
            left = rest;
        }
        Ok(())
    }

    /// Feeds `tokens` (at most the scratch's rows) at the state's position,
    /// computes the logits `logits` asks for, and advances the state.
    pub fn forward(
        &self,
        ctx: &MetalContext,
        st: &mut Gemma4State,
        sc: &Scratch,
        tokens: &[u32],
        logits: Logits,
    ) -> Result<()> {
        let m = tokens.len();
        sc.stage_tokens(tokens)?;
        self.run_rows(ctx, st, sc, m, logits, |_| Ok(()))?;
        st.pos += m;
        Ok(())
    }

    /// Runs a step of the `m` rows staged in `sc.tokens` at the state's
    /// position (without advancing it), `tail` encoded at its end, and
    /// waits: one pass for a step of up to MAX_ROWS rows, a pass a layer
    /// (two in flight) for a prompt run.
    pub fn run_rows(
        &self,
        ctx: &MetalContext,
        st: &Gemma4State,
        sc: &Scratch,
        m: usize,
        logits: Logits,
        tail: impl FnOnce(&ComputePass<'_>) -> Result<()>,
    ) -> Result<()> {
        let n = self.cfg.n_layer;
        if m <= MAX_ROWS {
            let pass = self.begin(ctx)?;
            self.encode_layers(ctx, &pass, st, sc, &sc.tokens, m, logits, 0..n)?;
            tail(&pass)?;
            return pass.commit_wait();
        }
        let mut pending: std::collections::VecDeque<crate::metal::PendingPass<'_>> =
            std::collections::VecDeque::new();
        let mut tail = Some(tail);
        for il in 0..n {
            let pass = self.begin(ctx)?;
            self.encode_layers(ctx, &pass, st, sc, &sc.tokens, m, logits, il..il + 1)?;
            while pending.len() >= 2 {
                pending.pop_front().expect("in flight").wait()?;
            }
            if il + 1 == n {
                (tail.take().expect("tail runs once"))(&pass)?;
                pass.commit_wait()?;
            } else {
                pending.push_back(pass.commit()?);
            }
        }
        for p in pending {
            p.wait()?;
        }
        Ok(())
    }

    /// Encodes layers `layers` of a step of `m` rows at the state's position
    /// (token ids in `tokens`, U32): the embedding when they start at layer
    /// 0, the head when they end at the last one.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layers(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Gemma4State,
        sc: &Scratch,
        tokens: &Tensor,
        m: usize,
        logits: Logits,
        layers: std::ops::Range<usize>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let pos0 = st.pos;
        ensure!((1..=sc.rows).contains(&m), "step of {m} rows");
        ensure!(logits != Logits::All || m <= MAX_ROWS, "logits of {m} rows");
        ensure!(
            pos0 + m <= st.capacity,
            "state capacity {} for position {}",
            st.capacity,
            pos0 + m
        );
        ensure!(layers.end <= cfg.n_layer, "layers {layers:?}");
        if layers.start == 0 {
            g4::embed(
                ctx,
                pass,
                &self.embed,
                tokens,
                &sc.x,
                m,
                (cfg.hidden as f32).sqrt(),
            )?;
            pass.memory_barrier()?;
            // The first layer's input; each layer's FFN end writes the next
            // one's (and the last layer's the head's).
            let w = &self.layers[0].attn_norm;
            let q8 = (self.q8x && m <= MAX_ROWS).then_some((&sc.xq8, &sc.xds));
            g4::rmsnorm_q8x(
                ctx,
                pass,
                &sc.x,
                w,
                &sc.h,
                m,
                cfg.hidden,
                cfg.rms_eps,
                q8,
            )?;
            pass.memory_barrier()?;
        }
        for il in layers.clone() {
            self.encode_layer(ctx, pass, st, sc, il, m)?;
        }
        if layers.end == cfg.n_layer {
            // The last row's hidden for the draft model, beside the head
            // (both read the last layer's output).
            g4::copy(ctx, pass, &sc.h_row(m - 1, cfg.hidden)?, &st.h_last, cfg.hidden)?;
            self.encode_head(ctx, pass, sc, m, logits)?;
        }
        Ok(())
    }

    /// One layer of a step, from its input `sc.h` (rmsnorm(x) attn_norm)
    /// and the residual `sc.x` to the next layer's.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Gemma4State,
        sc: &Scratch,
        il: usize,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let l = &self.layers[il];
        let c = &st.layers[il];
        let (h, eps) = (cfg.hidden, cfg.rms_eps);
        let pos0 = st.pos;
        // Attention.
        let kw = cfg.kv_width(il);
        // The producers of the Q5_K projections' inputs write them as int8
        // too (the layer's input norm, the attention's merge, the residual's
        // norms).
        let q8 = (self.q8x && m <= MAX_ROWS).then_some((&sc.xq8, &sc.xds));
        let q8e = (self.q8x_experts && m <= MAX_ROWS).then_some((&sc.eq8, &sc.eds));
        let pre = q8.is_some();
        match &l.v {
            Some(v) => self.proj_q8(
                ctx,
                pass,
                sc,
                &[(&l.q, &sc.q), (&l.k, &sc.kv), (v, &sc.vraw)],
                &sc.h,
                m,
                pre,
            )?,
            None => self.proj_q8(
                ctx,
                pass,
                sc,
                &[(&l.q, &sc.q), (&l.k, &sc.kv)],
                &sc.h,
                m,
                pre,
            )?,
        }
        pass.memory_barrier()?;
        let a = AttnShape {
            head_dim: cfg.head_dim_of(il),
            heads: cfg.n_head,
            kv_heads: cfg.n_head_kv[il],
            ring: c.slots,
            window: if c.swa { cfg.window } else { 0 },
            k_cache: &c.k,
            v_cache: &c.v,
            k_rot: cfg.k_rot[il],
            k_norm: &l.k_norm,
        };
        // A prompt run's K/V store and attention in turns of ATTN_ROWS rows
        // (a ring holds the window and one turn); its projections whole.
        let (qw, d) = (cfg.q_width(il), cfg.head_dim_of(il));
        let turn = if m > MAX_ROWS { ATTN_ROWS } else { m };
        for off in (0..m).step_by(turn) {
            let n = turn.min(m - off);
            let rows = |t: &Tensor, w: usize| t.view(off * w, &[n * w]);
            let (q, qo, heads) = (rows(&sc.q, qw)?, rows(&sc.qo, qw)?, rows(&sc.heads, qw)?);
            let k_raw = rows(&sc.kv, kw)?;
            let v_raw = if l.v.is_some() { rows(&sc.vraw, kw)? } else { k_raw.clone() };
            g4::qkv(
                ctx,
                pass,
                &a,
                &q,
                &k_raw,
                &v_raw,
                kw,
                0,
                &l.q_norm,
                &l.k_norm,
                (!c.swa).then_some(&self.rope_freqs),
                &qo,
                pos0 + off,
                n,
                cfg.rope_base_of(il),
                eps,
            )?;
            pass.memory_barrier()?;
            ensure!(
                n > MAX_ROWS || g4::attn_chunks(&a, pos0 + off, n) <= sc.chunks,
                "attention past the scratch's {} chunks",
                sc.chunks
            );
            debug_assert_eq!(qw, cfg.n_head * d);
            g4::attention_q8x(
                ctx,
                pass,
                &a,
                &qo,
                &sc.partial,
                &heads,
                pos0 + off,
                n,
                MAX_ROWS,
                q8,
            )?;
            pass.memory_barrier()?;
        }
        self.proj_q8(ctx, pass, sc, &[(&l.o, &sc.attn)], &sc.heads, m, pre)?;
        pass.memory_barrier()?;
        // The residual, and the inputs of the dense MLP, the routed
        // experts and the router.
        g4::post_attn_q8x(
            ctx,
            pass,
            &sc.attn,
            &l.post_attn_norm,
            &sc.x,
            &sc.r,
            [&l.ffn_norm, &l.pre_ffw_norm_2, &l.router_w],
            [&sc.n1, &sc.n2, &sc.t],
            m,
            h,
            eps,
            [q8, q8e],
        )?;
        pass.memory_barrier()?;
        self.proj_q8(
            ctx,
            pass,
            sc,
            &[(&l.gate, &sc.g), (&l.up, &sc.u)],
            &sc.n1,
            m,
            pre,
        )?;
        self.proj(ctx, pass, sc, &[(&l.router, &sc.rlogits)], &sc.t, m)?;
        pass.memory_barrier()?;
        g4::geglu(ctx, pass, &sc.g, &sc.u, &sc.mid, m * cfg.ff)?;
        g4::router(
            ctx,
            pass,
            &sc.rlogits,
            &l.down_scale,
            &sc.sel,
            &sc.rw,
            m,
            cfg.n_expert,
            cfg.n_expert_used,
        )?;
        pass.memory_barrier()?;
        self.proj(ctx, pass, sc, &[(&l.down, &sc.m1)], &sc.mid, m)?;
        if m > MAX_ROWS {
            // A prompt run: grouped by expert on the matrix units.
            g4::moe_grouped(
                ctx,
                pass,
                &l.gate_up_exps,
                &l.down_exps,
                cfg.ff_expert,
                &sc.n2,
                &sc.sel,
                &sc.rw,
                &sc.m2,
                &sc.moe_group,
                m,
                cfg.n_expert_used,
                cfg.n_expert,
            )?;
        } else {
            g4::moe_gate_up_q8x(
                ctx,
                pass,
                &l.gate_up_exps,
                cfg.ff_expert,
                &sc.n2,
                &sc.sel,
                &sc.emid,
                m,
                cfg.n_expert_used,
                q8e,
            )?;
            pass.memory_barrier()?;
            g4::moe_down(
                ctx,
                pass,
                &l.down_exps,
                h,
                &sc.emid,
                &sc.sel,
                &sc.rw,
                &sc.m2,
                m,
                cfg.n_expert_used,
            )?;
        }
        pass.memory_barrier()?;
        // The FFN's end, with the next layer's input norm (or the head's).
        let next = self.layers.get(il + 1).map_or(&self.output_norm, |n| &n.attn_norm);
        g4::ffn_out_q8x(
            ctx,
            pass,
            &sc.m1,
            &sc.m2,
            [&l.post_ffw_norm_1, &l.post_ffw_norm_2, &l.post_ffw_norm],
            &sc.r,
            &l.out_scale,
            &sc.x,
            Some((next, &sc.h)),
            m,
            h,
            eps,
            q8,
        )?;
        pass.memory_barrier()
    }

    fn encode_head(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        m: usize,
        logits: Logits,
    ) -> Result<()> {
        let cfg = &self.cfg;
        if logits == Logits::None {
            return Ok(());
        }
        let (first, rows) = if logits == Logits::All { (0, m) } else { (m - 1, 1) };
        // The last layer's FFN end left rmsnorm(x) output_norm in sc.h.
        let hn = sc.h.view(first * cfg.hidden, &[rows * cfg.hidden])?;
        k::gemv(ctx, pass, &self.embed, &hn, &sc.logits, rows)?;
        pass.memory_barrier()?;
        if cfg.softcap > 0.0 {
            g4::softcap(ctx, pass, &sc.logits, rows * cfg.vocab, cfg.softcap)?;
            pass.memory_barrier()?;
        }
        Ok(())
    }

    /// Feeds a whole prompt in runs of `prefill_rows`, leaving the last
    /// row's logits.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        st: &mut Gemma4State,
        sc: &Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prefill");
        let n = tokens.len().div_ceil(self.prefill_rows);
        for (i, chunk) in tokens.chunks(self.prefill_rows).enumerate() {
            let l = if i + 1 == n { Logits::Last } else { Logits::None };
            self.forward(ctx, st, sc, chunk, l)
                .with_context(|| format!("prefill run {i} at {}", st.pos))?;
        }
        Ok(())
    }
}
