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
use crate::gguf::{GgmlType, Gguf};
use crate::glm53::model::Mapped;
use crate::kernels::gemma4::{self as g4, AttnShape};
use crate::kernels::glm53::{self as k, QWeight};
use crate::kernels::sample::SamplerScratch;
use crate::metal::{Buffer, ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Rows one step takes on the GEMV path (decode, verify, scoring).
pub const MAX_ROWS: usize = k::GEMV_MAX_ROWS;

/// Rows a prompt run takes. The sliding-window layers' rings hold the
/// window plus one run, and the scratch is sized by it: 256 rows keep the
/// 16 GB tier's configuration (UD-IQ2_M, 8K context) under its 11 GB
/// budget (512 rows: 10.84 GB with one checkpoint, 256: 10.71).
pub const PREFILL_ROWS: usize = 256;

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
}

fn check_w(name: &str, w: &QWeight, cols: usize, rows: usize) -> Result<()> {
    ensure!(
        w.cols == cols && w.rows == rows,
        "{name} is {} x {}, Gemma needs {cols} x {rows}",
        w.cols,
        w.rows
    );
    Ok(())
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
        let mapped = Mapped::map(ctx, &gguf)?;
        let h = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{il}.{s}");
            let q = |s: &str| mapped.q(&n(s));
            let f = |s: &str, len: usize| -> Result<Tensor> {
                let t = mapped.f32(&n(s))?;
                ensure!(
                    t.numel() == len,
                    "{} has {} values, not {len}",
                    n(s),
                    t.numel()
                );
                Ok(t)
            };
            let (qw, kw) = (cfg.q_width(il), cfg.kv_width(il));
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
            check_w(&n("attn_q"), &l.q, h, qw)?;
            check_w(&n("attn_k"), &l.k, h, kw)?;
            if let Some(v) = &l.v {
                check_w(&n("attn_v"), v, h, kw)?;
            }
            ensure!(
                l.v.is_some() || !cfg.swa[il],
                "layer {il}: a sliding-window layer without attn_v"
            );
            check_w(&n("attn_output"), &l.o, qw, h)?;
            check_w(&n("ffn_gate"), &l.gate, h, cfg.ff)?;
            check_w(&n("ffn_up"), &l.up, h, cfg.ff)?;
            check_w(&n("ffn_down"), &l.down, cfg.ff, h)?;
            check_w(&n("ffn_gate_inp"), &l.router, h, cfg.n_expert)?;
            ensure!(l.router.ty == GgmlType::F32, "router of {:?}", l.router.ty);
            check_w(
                &n("ffn_gate_up_exps"),
                &l.gate_up_exps,
                h,
                2 * cfg.ff_expert * cfg.n_expert,
            )?;
            check_w(
                &n("ffn_down_exps"),
                &l.down_exps,
                cfg.ff_expert,
                h * cfg.n_expert,
            )?;
            layers.push(l);
        }
        let embed = mapped.q("token_embd.weight")?;
        check_w("token_embd", &embed, h, cfg.vocab)?;
        ensure!(
            gguf.tensor("output.weight").is_none(),
            "an untied output.weight (lily ties the head to token_embd as Gemma does)"
        );
        let output_norm = mapped.f32("output_norm.weight")?;
        let rope_freqs = mapped.f32("rope_freqs.weight")?;
        ensure!(
            rope_freqs.numel() == cfg.head_dim / 2,
            "rope_freqs of {}",
            rope_freqs.numel()
        );
        Ok(Self { cfg, gguf, mapped, layers, embed, output_norm, rope_freqs })
    }

    /// The buffers holding the mapped file.
    pub fn weight_buffers(&self) -> Vec<Buffer> {
        self.mapped.buffers()
    }

    pub fn new_state(
        &self,
        ctx: &MetalContext,
        capacity: usize,
    ) -> Result<Gemma4State> {
        Gemma4State::new(ctx, &self.cfg, capacity)
    }

    pub fn new_scratch(&self, ctx: &MetalContext, capacity: usize) -> Result<Scratch> {
        Scratch::new(ctx, &self.cfg, capacity)
    }
}

/// Slots of a sliding-window layer's ring.
pub fn ring_slots(cfg: &Config) -> usize {
    cfg.window + PREFILL_ROWS
}

/// One layer's K and V caches, f16 `[slots][kv heads][head dim]`.
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
    pub layers: Vec<LayerCache>,
    /// The sliding window: what a snapshot keeps of each ring.
    pub window: usize,
}

impl Gemma4State {
    pub fn new(ctx: &MetalContext, cfg: &Config, capacity: usize) -> Result<Self> {
        let layers = (0..cfg.n_layer)
            .map(|il| -> Result<LayerCache> {
                let swa = cfg.swa[il];
                let slots = if swa { ring_slots(cfg) } else { capacity };
                let n = slots * cfg.kv_width(il);
                Ok(LayerCache {
                    k: Tensor::zeros(ctx, &[n], DType::F16)?,
                    v: Tensor::zeros(ctx, &[n], DType::F16)?,
                    slots,
                    swa,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { pos: 0, capacity, layers, window: cfg.window })
    }

    /// Bytes of the caches.
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| l.k.byte_len() + l.v.byte_len()).sum()
    }
}

/// Per-step intermediates for up to [`PREFILL_ROWS`] rows.
pub struct Scratch {
    pub tokens: Tensor,
    pub x: Tensor,
    h: Tensor,
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
}

impl Scratch {
    /// Scratch for states of up to `capacity` tokens.
    pub fn new(ctx: &MetalContext, cfg: &Config, capacity: usize) -> Result<Self> {
        let m = PREFILL_ROWS;
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
        Ok(Self {
            logits_last: logits.view(0, &[cfg.vocab])?,
            tokens: Tensor::zeros(ctx, &[m], DType::U32)?,
            x: f(m * h)?,
            h: f(m * h)?,
            q: f(m * cfg.max_q_width())?,
            kv: f(m * cfg.max_kv_width())?,
            vraw: f(m * cfg.max_kv_width())?,
            qo: f(m * cfg.max_q_width())?,
            heads: f(m * cfg.max_q_width())?,
            attn: f(m * h)?,
            r: f(m * h)?,
            n1: f(m * h)?,
            n2: f(m * h)?,
            t: f(m * h)?,
            g: f(m * cfg.ff)?,
            u: f(m * cfg.ff)?,
            mid: f(m * cfg.ff)?,
            m1: f(m * h)?,
            m2: f(m * h)?,
            rlogits: f(m * cfg.n_expert)?,
            sel: Tensor::zeros(ctx, &[m * cfg.n_expert_used], DType::U32)?,
            rw: f(m * cfg.n_expert_used)?,
            emid: f(MAX_ROWS * cfg.n_expert_used * cfg.ff_expert)?,
            moe_group: g4::MoeGroupScratch::new(
                ctx,
                m,
                cfg.n_expert_used,
                cfg.n_expert,
                cfg.ff_expert,
                h,
            )?,
            partial: f(partial)?,
            chunks,
            logits,
            sampler: SamplerScratch::new(ctx, cfg.vocab)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
        })
    }

    /// Stages a run's token ids (at most PREFILL_ROWS).
    pub(crate) fn stage_tokens(&self, tokens: &[u32]) -> Result<()> {
        crate::glm53::model::write_words(&self.tokens, tokens)
    }

    /// The sampler's buffers.
    pub fn sampler(&self) -> &SamplerScratch {
        &self.sampler
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
        sets: &[(&QWeight, &Tensor)],
        x: &Tensor,
        m: usize,
    ) -> Result<()> {
        if m > MAX_ROWS {
            for &(w, y) in sets {
                k::gemm(ctx, pass, w, x, y, m)?;
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

    /// Feeds `tokens` (at most PREFILL_ROWS) at the state's position,
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
        ensure!((1..=PREFILL_ROWS).contains(&m), "step of {m} rows");
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
            g4::rmsnorm(ctx, pass, &sc.x, w, &sc.h, m, cfg.hidden, cfg.rms_eps)?;
            pass.memory_barrier()?;
        }
        for il in layers.clone() {
            self.encode_layer(ctx, pass, st, sc, il, m)?;
        }
        if layers.end == cfg.n_layer {
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
        match &l.v {
            Some(v) => self.proj(
                ctx,
                pass,
                &[(&l.q, &sc.q), (&l.k, &sc.kv), (v, &sc.vraw)],
                &sc.h,
                m,
            )?,
            None => self.proj(ctx, pass, &[(&l.q, &sc.q), (&l.k, &sc.kv)], &sc.h, m)?,
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
        };
        g4::qkv(
            ctx,
            pass,
            &a,
            &sc.q,
            &sc.kv,
            if l.v.is_some() { &sc.vraw } else { &sc.kv },
            kw,
            0,
            &l.q_norm,
            &l.k_norm,
            (!c.swa).then_some(&self.rope_freqs),
            &sc.qo,
            pos0,
            m,
            cfg.rope_base_of(il),
            eps,
        )?;
        pass.memory_barrier()?;
        ensure!(
            m > MAX_ROWS || g4::attn_chunks(&a, pos0, m) <= sc.chunks,
            "attention past the scratch's {} chunks",
            sc.chunks
        );
        g4::attention(
            ctx,
            pass,
            &a,
            &sc.qo,
            &sc.partial,
            &sc.heads,
            pos0,
            m,
            MAX_ROWS,
        )?;
        pass.memory_barrier()?;
        self.proj(ctx, pass, &[(&l.o, &sc.attn)], &sc.heads, m)?;
        pass.memory_barrier()?;
        // The residual, and the inputs of the dense MLP, the routed
        // experts and the router.
        g4::post_attn(
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
        )?;
        pass.memory_barrier()?;
        self.proj(ctx, pass, &[(&l.gate, &sc.g), (&l.up, &sc.u)], &sc.n1, m)?;
        self.proj(ctx, pass, &[(&l.router, &sc.rlogits)], &sc.t, m)?;
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
        self.proj(ctx, pass, &[(&l.down, &sc.m1)], &sc.mid, m)?;
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
            g4::moe_gate_up(
                ctx,
                pass,
                &l.gate_up_exps,
                cfg.ff_expert,
                &sc.n2,
                &sc.sel,
                &sc.emid,
                m,
                cfg.n_expert_used,
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
        g4::ffn_out(
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

    /// Feeds a whole prompt in runs of PREFILL_ROWS, leaving the last
    /// row's logits.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        st: &mut Gemma4State,
        sc: &Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prefill");
        let n = tokens.len().div_ceil(PREFILL_ROWS);
        for (i, chunk) in tokens.chunks(PREFILL_ROWS).enumerate() {
            let l = if i + 1 == n { Logits::Last } else { Logits::None };
            self.forward(ctx, st, sc, chunk, l)
                .with_context(|| format!("prefill run {i} at {}", st.pos))?;
        }
        Ok(())
    }
}
