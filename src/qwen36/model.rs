//! Qwen3.6 on the GPU: weights mapped read-only from the GGUF, the
//! per-session caches and recurrent state, and the forward pass over up to
//! [`PREFILL_ROWS`] consecutive tokens at a time (decode is one row).
//!
//! The graph is llama.cpp's `qwen35moe` (`src/models/qwen35moe.cpp`), which
//! defines what the file's tensors mean: 40 layers, three Gated DeltaNet
//! layers then one gated full-attention layer, each followed by a MoE of
//! 256 experts (top 8, softmax) plus a sigmoid-gated shared expert.

use std::path::Path;

use anyhow::{Result, ensure};

use super::config::Config;
use crate::gguf::Gguf;
use crate::glm53::model::Mapped;
use crate::kernels::attention::{self as attn, KvCache, KvFormat};
use crate::kernels::glm53::{self as g, QWeight};
use crate::kernels::qwen36::{self as k, GdnParams, GdnSnapshot, GdnState};
use crate::kernels::sample::SamplerScratch;
use crate::metal::{Buffer, ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Rows a decode/verify step takes (the GEMV's limit).
pub const MAX_ROWS: usize = g::GEMV_MAX_ROWS;

/// Rows a prompt step takes (projections on the matrix units).
pub const PREFILL_ROWS: usize = 1024;

struct GdnW {
    qkv: QWeight,
    gate: QWeight,
    alpha: QWeight,
    beta: QWeight,
    out: QWeight,
    conv: Tensor,
    ssm_a: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
}

struct AttnW {
    q: QWeight,
    k: QWeight,
    v: QWeight,
    o: QWeight,
    q_norm: Tensor,
    k_norm: Tensor,
}

enum Mixer {
    Gdn(GdnW),
    Attn(AttnW),
}

struct FfnW {
    router: QWeight,
    sh_gate_inp: QWeight,
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    sh_gate: QWeight,
    sh_up: QWeight,
    sh_down: QWeight,
}

struct LayerW {
    attn_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    ffn: FfnW,
}

/// The model's weights on the GPU.
pub struct Qwen36Model {
    pub cfg: Config,
    pub gguf: Gguf,
    mapped: Mapped,
    layers: Vec<LayerW>,
    embed: QWeight,
    output: QWeight,
    output_norm: Tensor,
    /// K/V cache element format.
    pub kv_format: KvFormat,
}

fn load_block(m: &Mapped, il: usize, recurrent: bool) -> Result<LayerW> {
    let n = |s: &str| format!("blk.{il}.{s}");
    let q = |s: &str| m.q(&n(s));
    let f = |s: &str| m.f32(&n(s));
    let mixer = if recurrent {
        Mixer::Gdn(GdnW {
            qkv: q("attn_qkv.weight")?,
            gate: q("attn_gate.weight")?,
            alpha: q("ssm_alpha.weight")?,
            beta: q("ssm_beta.weight")?,
            out: q("ssm_out.weight")?,
            conv: f("ssm_conv1d.weight")?,
            ssm_a: f("ssm_a")?,
            dt_bias: f("ssm_dt.bias")?,
            norm: f("ssm_norm.weight")?,
        })
    } else {
        Mixer::Attn(AttnW {
            q: q("attn_q.weight")?,
            k: q("attn_k.weight")?,
            v: q("attn_v.weight")?,
            o: q("attn_output.weight")?,
            q_norm: f("attn_q_norm.weight")?,
            k_norm: f("attn_k_norm.weight")?,
        })
    };
    Ok(LayerW {
        attn_norm: f("attn_norm.weight")?,
        post_norm: f("post_attention_norm.weight")?,
        mixer,
        ffn: FfnW {
            router: q("ffn_gate_inp.weight")?,
            sh_gate_inp: q("ffn_gate_inp_shexp.weight")?,
            gate: q("ffn_gate_exps.weight")?,
            up: q("ffn_up_exps.weight")?,
            down: q("ffn_down_exps.weight")?,
            sh_gate: q("ffn_gate_shexp.weight")?,
            sh_up: q("ffn_up_shexp.weight")?,
            sh_down: q("ffn_down_shexp.weight")?,
        },
    })
}

impl Qwen36Model {
    /// Opens and maps `path` (the GGUF file).
    pub fn load(ctx: &MetalContext, path: &Path) -> Result<Self> {
        Self::load_locked(ctx, path, crate::instance::acquire)
    }

    /// [`Self::load`] with the instance lock taken by `lock`, first (one
    /// model on this machine at a time, `crate::instance`).
    pub(crate) fn load_locked(
        ctx: &MetalContext,
        path: &Path,
        lock: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        lock()?;
        let gguf = Gguf::open(path)?;
        let cfg = Config::from_gguf(&gguf)?;
        super::schema::validate(&gguf, &cfg)?;
        let mapped = Mapped::map(ctx, &gguf)?;
        let layers = (0..cfg.n_layer)
            .map(|il| load_block(&mapped, il, cfg.recurrent[il]))
            .collect::<Result<_>>()?;
        let embed = mapped.q("token_embd.weight")?;
        let output = mapped.q("output.weight")?;
        let output_norm = mapped.f32("output_norm.weight")?;
        Ok(Self {
            cfg,
            gguf,
            mapped,
            layers,
            embed,
            output,
            output_norm,
            kv_format: KvFormat::Bf16,
        })
    }

    /// The buffers holding the mapped file.
    pub fn weight_buffers(&self) -> Vec<Buffer> {
        self.mapped.buffers()
    }

    pub fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Qwen36State> {
        Qwen36State::new(ctx, &self.cfg, self.kv_format, capacity)
    }

    pub fn new_scratch(&self, ctx: &MetalContext) -> Result<Scratch> {
        Scratch::new(ctx, &self.cfg)
    }
}

/// One layer's recurrent state or caches.
pub enum LayerState {
    /// `conv_alt`/`s_alt` receive the state after a verify pass's first row,
    /// which becomes current when its draft is rejected.
    Gdn { conv: Tensor, s: Tensor, conv_alt: Tensor, s_alt: Tensor },
    Attn { k: KvCache, v: KvCache },
}

/// A session: caches and recurrent state at position `pos`.
pub struct Qwen36State {
    pub pos: usize,
    pub capacity: usize,
    pub layers: Vec<LayerState>,
}

impl Qwen36State {
    pub fn new(
        ctx: &MetalContext,
        cfg: &Config,
        format: KvFormat,
        capacity: usize,
    ) -> Result<Self> {
        let capacity = capacity.next_multiple_of(4);
        let ch = cfg.gdn_qkv();
        let layers = (0..cfg.n_layer)
            .map(|il| -> Result<LayerState> {
                Ok(if cfg.recurrent[il] {
                    let conv = || Tensor::zeros(ctx, &[3 * ch], DType::F32);
                    let s = || {
                        Tensor::zeros(
                            ctx,
                            &[cfg.gdn_v_heads * cfg.gdn_dim * cfg.gdn_dim],
                            DType::F32,
                        )
                    };
                    LayerState::Gdn { conv: conv()?, s: s()?, conv_alt: conv()?, s_alt: s()? }
                } else {
                    let c = || {
                        KvCache::zeros(ctx, format, cfg.n_head_kv, capacity, cfg.head_dim)
                    };
                    LayerState::Attn { k: c()?, v: c()? }
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { pos: 0, capacity, layers })
    }

    /// Zeroes the recurrent state and rewinds (GPU idle).
    pub fn reset(&mut self) {
        for l in &self.layers {
            if let LayerState::Gdn { conv, s, .. } = l {
                conv.zero_fill();
                s.zero_fill();
            }
        }
        self.pos = 0;
    }

    /// Makes the GDN states saved after a verify pass's first row current.
    pub(crate) fn take_gdn_snapshots(&mut self) {
        for l in &mut self.layers {
            if let LayerState::Gdn { conv, s, conv_alt, s_alt } = l {
                std::mem::swap(conv, conv_alt);
                std::mem::swap(s, s_alt);
            }
        }
    }
}

/// Per-step intermediates, for up to [`PREFILL_ROWS`] rows.
pub struct Scratch {
    pub tokens: Tensor,
    /// The residual stream.
    pub x: Tensor,
    pub xn: Tensor,
    xbf: Tensor,
    /// The mixer's (GDN or attention) output projection.
    mix: Tensor,
    // GDN
    qkv: Tensor,
    z: Tensor,
    ga: Tensor,
    gb: Tensor,
    c: Tensor,
    o: Tensor,
    y: Tensor,
    // Attention
    qg: Tensor,
    ak: Tensor,
    av: Tensor,
    qb: Tensor,
    kb: Tensor,
    vb: Tensor,
    attn: Tensor,
    split_partials: Tensor,
    split_stats: Tensor,
    // FFN
    rlogits: Tensor,
    shg: Tensor,
    pub rsel: Tensor,
    rw: Tensor,
    moe_mid: Tensor,
    moe_out: Tensor,
    moe_group: g::MoeGroupScratch,
    sfg: Tensor,
    sfu: Tensor,
    sfm: Tensor,
    sout: Tensor,
    // Head
    hidden_norm: Tensor,
    pub logits: Tensor,
    /// The last (or only) row's logits: `logits[0..vocab]`.
    pub logits_last: Tensor,
    sampler: SamplerScratch,
    /// U32[2]: the ping-pong slots the in-graph sampler draws into.
    pub next_token: Tensor,
    /// U32[MAX_ROWS]: a verify pass's draws, one a row.
    pub draws: Tensor,
    /// U32[MAX_ROWS]: tokens a verify pass feeds.
    pub spec_tokens: Tensor,
}

impl Scratch {
    pub(crate) fn new(ctx: &MetalContext, cfg: &Config) -> Result<Self> {
        let r = PREFILL_ROWS;
        let f = |n: usize| Tensor::zeros(ctx, &[n], DType::F32);
        let b = |n: usize| Tensor::zeros(ctx, &[n], DType::BF16);
        let u = |n: usize| Tensor::zeros(ctx, &[n], DType::U32);
        let h = cfg.hidden;
        let ch = cfg.gdn_qkv();
        let inner = cfg.gdn_inner();
        let hq = cfg.n_head * cfg.head_dim;
        let kv = cfg.kv_width();
        let used = cfg.n_expert_used;
        // The split decode's partials, for any context (bounded: 2 MB).
        let splits = attn::sdpa_split_scratch_splits(attn::MAX_SEQ);
        let widest = [h, inner, hq, cfg.ff_shared].into_iter().max().unwrap_or(h);
        let logits = f(MAX_ROWS * cfg.vocab)?;
        Ok(Self {
            logits_last: logits.view(0, &[cfg.vocab])?,
            tokens: u(r)?,
            x: f(r * h)?,
            xn: f(r * h)?,
            xbf: b(r * widest)?,
            mix: f(r * h)?,
            qkv: f(r * ch)?,
            z: f(r * inner)?,
            ga: f(r * cfg.gdn_v_heads)?,
            gb: f(r * cfg.gdn_v_heads)?,
            c: f(r * ch)?,
            o: f(r * inner)?,
            y: f(r * inner)?,
            qg: f(r * 2 * hq)?,
            ak: f(r * kv)?,
            av: f(r * kv)?,
            qb: Tensor::zeros(ctx, &[r, cfg.n_head, cfg.head_dim], DType::BF16)?,
            kb: b(r * kv)?,
            vb: b(r * kv)?,
            attn: Tensor::zeros(ctx, &[r, cfg.n_head, cfg.head_dim], DType::BF16)?,
            split_partials: f(cfg.n_head * splits * cfg.head_dim)?,
            split_stats: f(cfg.n_head * splits * 2)?,
            rlogits: f(r * cfg.n_expert)?,
            shg: f(r)?,
            rsel: u(r * used)?,
            rw: f(r * used)?,
            moe_mid: f(r * used * cfg.ff_expert)?,
            moe_out: f(r * h)?,
            moe_group: g::MoeGroupScratch::new(ctx, r * used, cfg.n_expert, cfg.ff_expert, h)?,
            sfg: f(r * cfg.ff_shared)?,
            sfu: f(r * cfg.ff_shared)?,
            sfm: f(r * cfg.ff_shared)?,
            sout: f(r * h)?,
            hidden_norm: f(MAX_ROWS * h)?,
            logits,
            sampler: SamplerScratch::new(ctx, cfg.vocab)?,
            next_token: u(2)?,
            draws: u(MAX_ROWS)?,
            spec_tokens: u(MAX_ROWS)?,
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
    /// Every row's, row `i` at `logits[i * vocab..]` (at most MAX_ROWS rows).
    All,
}

impl Qwen36Model {
    /// The pass a step is encoded into: concurrent, its dependency levels
    /// separated by barriers; `LILY_QWEN36_SERIAL=1` makes every dispatch
    /// its own level (A/B and debugging).
    pub fn begin<'c>(&self, ctx: &'c MetalContext) -> Result<ComputePass<'c>> {
        static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SERIAL.get_or_init(|| std::env::var_os("LILY_QWEN36_SERIAL").is_some()) {
            ctx.begin()
        } else {
            ctx.begin_concurrent()
        }
    }

    /// Feeds `tokens` (at most [`PREFILL_ROWS`]) at the state's position,
    /// leaving logits as `logits` says, and advances the state.
    pub fn forward(
        &self,
        ctx: &MetalContext,
        st: &mut Qwen36State,
        sc: &mut Scratch,
        tokens: &[u32],
        logits: Logits,
    ) -> Result<()> {
        let m = tokens.len();
        ensure!((1..=PREFILL_ROWS).contains(&m), "forward of {m} rows");
        ensure!(st.pos + m <= st.capacity, "state capacity {}", st.capacity);
        sc.stage_tokens(tokens)?;
        self.run_rows(ctx, st, sc, m, logits, None, |_| Ok(()))?;
        st.pos += m;
        Ok(())
    }

    /// Feeds a whole prompt in runs of [`PREFILL_ROWS`] rows, the last
    /// row's logits at the end.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        st: &mut Qwen36State,
        sc: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        let n = tokens.len();
        for (i, chunk) in tokens.chunks(PREFILL_ROWS).enumerate() {
            let last = (i + 1) * PREFILL_ROWS >= n;
            let l = if last { Logits::Last } else { Logits::None };
            self.forward(ctx, st, sc, chunk, l)?;
        }
        Ok(())
    }

    /// Runs a step of the `m` rows staged in `sc.tokens` at the state's
    /// position (without advancing it), `tail` encoded at its end, and
    /// waits. Prompt runs take a pass per few layers (a command buffer's
    /// time is limited by the kernel driver); decode steps one pass.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows(
        &self,
        ctx: &MetalContext,
        st: &Qwen36State,
        sc: &Scratch,
        m: usize,
        logits: Logits,
        snap: Option<usize>,
        tail: impl FnOnce(&ComputePass<'_>) -> Result<()>,
    ) -> Result<()> {
        let n = self.cfg.n_layer;
        let per_pass = if m <= MAX_ROWS { n } else { 4 };
        let mut pending: std::collections::VecDeque<crate::metal::PendingPass<'_>> =
            std::collections::VecDeque::new();
        let mut start = 0;
        let mut tail = Some(tail);
        while start < n {
            let end = (start + per_pass).min(n);
            let pass = self.begin(ctx)?;
            self.encode_layers(ctx, &pass, st, sc, &sc.tokens, m, logits, snap, start..end)?;
            while pending.len() >= 2 {
                pending.pop_front().expect("in flight").wait()?;
            }
            if end == n {
                (tail.take().expect("tail runs once"))(&pass)?;
                pass.commit_wait()?;
            } else {
                pending.push_back(pass.commit()?);
            }
            start = end;
        }
        for p in pending {
            p.wait()?;
        }
        Ok(())
    }

    /// Encodes layers `layers` of a step of `m` rows at the state's
    /// position into `pass` (tokens read from `tokens` on the GPU): the
    /// embedding when they start at layer 0, the head when they end at the
    /// last layer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layers(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Qwen36State,
        sc: &Scratch,
        tokens: &Tensor,
        m: usize,
        logits: Logits,
        snap: Option<usize>,
        layers: std::ops::Range<usize>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        ensure!((1..=PREFILL_ROWS).contains(&m), "step of {m} rows");
        ensure!(logits != Logits::All || m <= MAX_ROWS, "logits of {m} rows");
        ensure!(st.pos + m <= st.capacity, "state capacity {}", st.capacity);
        ensure!(layers.end <= cfg.n_layer, "layers {layers:?}");
        if layers.start == 0 {
            k::embed(ctx, pass, &self.embed, tokens, &sc.x, m)?;
            pass.memory_barrier()?;
        }
        for il in layers.clone() {
            let l = &self.layers[il];
            g::rmsnorm(ctx, pass, &sc.x, Some(&l.attn_norm), &sc.xn, m, cfg.hidden, cfg.rms_eps)?;
            pass.memory_barrier()?;
            match (&l.mixer, &st.layers[il]) {
                (Mixer::Gdn(w), LayerState::Gdn { conv, s, conv_alt, s_alt }) => {
                    let snap = snap.map(|row| GdnSnapshot { row, conv: conv_alt, s: s_alt });
                    self.gdn(ctx, pass, sc, w, &GdnState { conv, s }, snap.as_ref(), m)?;
                }
                (Mixer::Attn(w), LayerState::Attn { k: kc, v: vc }) => {
                    self.attention(ctx, pass, sc, w, kc, vc, st.pos, m)?;
                }
                _ => anyhow::bail!("layer {il}'s state does not match its kind"),
            }
            g::add(ctx, pass, &sc.x, &sc.mix, &sc.x, m * cfg.hidden)?;
            pass.memory_barrier()?;
            g::rmsnorm(ctx, pass, &sc.x, Some(&l.post_norm), &sc.xn, m, cfg.hidden, cfg.rms_eps)?;
            pass.memory_barrier()?;
            self.ffn(ctx, pass, sc, &l.ffn, m)?;
        }
        if layers.end < cfg.n_layer {
            return Ok(());
        }
        self.encode_head(ctx, pass, sc, m, logits)
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
        let x = sc.x.view(first * cfg.hidden, &[rows * cfg.hidden])?;
        g::rmsnorm(ctx, pass, &x, Some(&self.output_norm), &sc.hidden_norm, rows, cfg.hidden, cfg.rms_eps)?;
        pass.memory_barrier()?;
        g::gemv(ctx, pass, &self.output, &sc.hidden_norm, &sc.logits, rows)?;
        pass.memory_barrier()
    }

    /// Projections of `x`'s `m` rows: one GEMV dispatch for up to four
    /// weights of a type at decode, a GEMM each for a prompt run (over `xb`,
    /// `x` as bfloat, where the GEMM takes it).
    fn proj(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sets: &[(&QWeight, &Tensor)],
        x: &Tensor,
        xb: Option<&Tensor>,
        m: usize,
    ) -> Result<()> {
        if m <= MAX_ROWS {
            return g::gemv_sets(ctx, pass, sets, x, m);
        }
        for &(w, y) in sets {
            match xb {
                Some(xb) if g::gemm_bx_takes(w) => g::gemm_bx(ctx, pass, w, xb, y, m)?,
                _ => g::gemm(ctx, pass, w, x, y, m)?,
            }
        }
        Ok(())
    }

    /// For a prompt run: `x`'s `m` rows of `width` rounded to bfloat into
    /// `sc.xbf` (the GEMMs round their input to bfloat anyway).
    fn bf16_input<'s>(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &'s Scratch,
        x: &Tensor,
        width: usize,
        m: usize,
    ) -> Result<Option<&'s Tensor>> {
        if m <= MAX_ROWS {
            return Ok(None);
        }
        g::to_bf16(ctx, pass, x, &sc.xbf, m * width)?;
        pass.memory_barrier()?;
        Ok(Some(&sc.xbf))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        w: &GdnW,
        st: &GdnState<'_>,
        snap: Option<&GdnSnapshot<'_>>,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
        self.proj(
            ctx,
            pass,
            &[(&w.qkv, &sc.qkv), (&w.gate, &sc.z), (&w.alpha, &sc.ga), (&w.beta, &sc.gb)],
            &sc.xn,
            xb,
            m,
        )?;
        pass.memory_barrier()?;
        let p = GdnParams { conv_w: &w.conv, ssm_a: &w.ssm_a, dt_bias: &w.dt_bias, norm: &w.norm };
        k::gdn(
            ctx,
            pass,
            &p,
            st,
            snap,
            &sc.qkv,
            &sc.ga,
            &sc.gb,
            &sc.z,
            &sc.c,
            &sc.o,
            &sc.y,
            m,
            cfg.gdn_k_heads,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        let xb = self.bf16_input(ctx, pass, sc, &sc.y, cfg.gdn_inner(), m)?;
        self.proj(ctx, pass, &[(&w.out, &sc.mix)], &sc.y, xb, m)?;
        pass.memory_barrier()
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        w: &AttnW,
        kc: &KvCache,
        vc: &KvCache,
        pos0: usize,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let (nq, nkv, d) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim);
        let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
        self.proj(ctx, pass, &[(&w.q, &sc.qg)], &sc.xn, xb, m)?;
        self.proj(ctx, pass, &[(&w.k, &sc.ak), (&w.v, &sc.av)], &sc.xn, xb, m)?;
        pass.memory_barrier()?;
        k::attn_prep(
            ctx,
            pass,
            &sc.qg,
            &sc.ak,
            &sc.av,
            &w.q_norm,
            &w.k_norm,
            &sc.qb,
            &sc.kb,
            &sc.vb,
            m,
            nq,
            nkv,
            pos0,
            cfg.rope_dim,
            cfg.rope_theta,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        let kb = sc.kb.view(0, &[m, nkv, d])?;
        let vb = sc.vb.view(0, &[m, nkv, d])?;
        attn::scatter_kv(ctx, pass, kc, &kb, pos0)?;
        attn::scatter_kv(ctx, pass, vc, &vb, pos0)?;
        pass.memory_barrier()?;
        let scale = 1.0 / (d as f32).sqrt();
        let q = sc.qb.view(0, &[m, nq, d])?;
        let out = sc.attn.view(0, &[m, nq, d])?;
        if m == 1 {
            attn::sdpa_decode(
                ctx,
                pass,
                &q,
                kc,
                vc,
                &out,
                pos0 + 1,
                scale,
                Some((&sc.split_partials, &sc.split_stats)),
            )?;
        } else {
            ensure!(self.kv_format == KvFormat::Bf16, "prompt attention reads bf16 caches");
            attn::sdpa_prefill(ctx, pass, &q, &kc.values, &vc.values, &out, pos0, scale)?;
        }
        pass.memory_barrier()?;
        k::attn_gate(ctx, pass, &sc.attn, &sc.qg, &sc.y, m, nq)?;
        pass.memory_barrier()?;
        let xb = self.bf16_input(ctx, pass, sc, &sc.y, nq * d, m)?;
        self.proj(ctx, pass, &[(&w.o, &sc.mix)], &sc.y, xb, m)?;
        pass.memory_barrier()
    }

    fn ffn(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        f: &FfnW,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let used = cfg.n_expert_used;
        let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
        if m <= MAX_ROWS {
            let sets = [(&f.router, &sc.rlogits), (&f.sh_gate_inp, &sc.shg)];
            g::gemv_sets(ctx, pass, &sets, &sc.xn, m)?;
        } else {
            k::router_logits(ctx, pass, &f.router, &f.sh_gate_inp, &sc.xn, &sc.rlogits, &sc.shg, m)?;
        }
        self.proj(ctx, pass, &[(&f.sh_gate, &sc.sfg), (&f.sh_up, &sc.sfu)], &sc.xn, xb, m)?;
        pass.memory_barrier()?;
        k::router(ctx, pass, &sc.rlogits, &sc.rsel, &sc.rw, m, cfg.n_expert, used)?;
        g::swiglu(ctx, pass, &sc.sfg, &sc.sfu, &sc.sfm, m * cfg.ff_shared, 0.0)?;
        pass.memory_barrier()?;
        if m > MAX_ROWS {
            // A prompt run: the gated shared expert, then the routed experts
            // grouped by expert on the matrix units add their sum to it.
            let xb = self.bf16_input(ctx, pass, sc, &sc.sfm, cfg.ff_shared, m)?;
            self.proj(ctx, pass, &[(&f.sh_down, &sc.sout)], &sc.sfm, xb, m)?;
            pass.memory_barrier()?;
            k::shared_gate(ctx, pass, &sc.sout, &sc.shg, &sc.moe_out, m, cfg.hidden)?;
            pass.memory_barrier()?;
            g::moe_grouped(
                ctx,
                pass,
                &f.gate,
                &f.up,
                &f.down,
                cfg.ff_expert,
                cfg.hidden,
                &sc.xn,
                &sc.rsel,
                &sc.rw,
                &sc.moe_out,
                &sc.moe_group,
                m,
                used,
                cfg.n_expert,
                0.0,
            )?;
            pass.memory_barrier()?;
            g::add(ctx, pass, &sc.x, &sc.moe_out, &sc.x, m * cfg.hidden)?;
            return pass.memory_barrier();
        }
        g::moe_gate_up(
            ctx,
            pass,
            &f.gate,
            &f.up,
            cfg.ff_expert,
            &sc.xn,
            &sc.rsel,
            &sc.rw,
            &sc.moe_mid,
            m,
            used,
            0.0,
        )?;
        let xb = self.bf16_input(ctx, pass, sc, &sc.sfm, cfg.ff_shared, m)?;
        self.proj(ctx, pass, &[(&f.sh_down, &sc.sout)], &sc.sfm, xb, m)?;
        pass.memory_barrier()?;
        g::moe_down(
            ctx,
            pass,
            &f.down,
            cfg.hidden,
            &sc.moe_mid,
            &sc.rsel,
            None,
            &sc.moe_out,
            m,
            used,
        )?;
        pass.memory_barrier()?;
        k::ffn_residual(ctx, pass, &sc.x, &sc.moe_out, &sc.sout, &sc.shg, m, cfg.hidden)?;
        pass.memory_barrier()
    }
}
