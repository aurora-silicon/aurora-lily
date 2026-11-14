//! DeepSeek V4 Flash on the GPU: weights mapped read-only from the GGUF, the
//! per-session caches (each layer's raw window ring, compressed rows and
//! compressor rings), and the forward pass over runs of consecutive tokens
//! (decode is one row; a prompt is fed in runs of rows).
//!
//! The graph is ds4's (`layer_forward_raw_swa_one`, ds4.c); `reference.rs`
//! holds the same graph on the CPU and `forward_traced` exposes each layer's
//! intermediates for a layer-by-layer comparison.

use std::path::Path;

use anyhow::{Result, ensure};

use super::config::Config;
use crate::gguf::{GgmlType, Gguf};
use crate::glm53::model::Mapped;
use crate::kernels::deepseek4::{self as d, CompRows, CompressorGpu, HeadOp};
use crate::kernels::glm53::{self as k, GemvLayout, QWeight};
use crate::metal::{Buffer, ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Rows a GEMV takes at once; longer runs go through it in chunks.
pub const MAX_ROWS: usize = k::GEMV_MAX_ROWS;

/// Rows a forward run takes at most.
pub const PREFILL_ROWS: usize = 2048;

/// The indexer's selection width (`indexer_top_k`).
const TOP_K: usize = 512;

struct HcW {
    fn_w: QWeight,
    scale: Tensor,
    base: Tensor,
}

/// A compressor's weights.
struct CompW {
    kv: QWeight,
    gate: QWeight,
    ape: Tensor,
    norm: Tensor,
}

struct IndexerW {
    comp: CompW,
    q_b: QWeight,
    proj: QWeight,
}

struct LayerW {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    hc: [HcW; 2],
    q_a: QWeight,
    q_a_norm: Tensor,
    q_b: QWeight,
    kv: QWeight,
    kv_norm: Tensor,
    sinks: Tensor,
    out_a: QWeight,
    out_b: QWeight,
    comp: Option<CompW>,
    indexer: Option<IndexerW>,
    router: QWeight,
    bias: Option<Tensor>,
    hash: Option<QWeight>,
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    sh_gate: QWeight,
    sh_up: QWeight,
    sh_down: QWeight,
}

/// The model's weights on the GPU.
pub struct Ds4Model {
    pub cfg: Config,
    pub gguf: Gguf,
    mapped: Mapped,
    layers: Vec<LayerW>,
    embed: QWeight,
    output: QWeight,
    output_norm: Tensor,
    out_hc: HcW,
}

/// An F16 tensor view of a mapped F16 weight (the compressors' position
/// bias).
fn f16_view(mapped: &Mapped, name: &str) -> Result<Tensor> {
    let (buf, at, t) = mapped.get(name)?;
    ensure!(t.ty == GgmlType::F16, "{name} is {:?}, expected F16", t.ty);
    ensure!(at % 2 == 0, "{name} is not 2-byte aligned in its mapping");
    let whole = Tensor::from_buffer(buf.clone(), &[buf.length() / 2], DType::F16)?;
    whole.view(at / 2, &[t.row_len() * t.rows()])
}

impl Ds4Model {
    /// Opens and maps `path` (the GGUF file).
    pub fn load(ctx: &MetalContext, path: &Path) -> Result<Self> {
        Self::load_locked(ctx, path, crate::instance::acquire)
    }

    /// [`Self::load`] with the instance lock taken by `lock` first, before
    /// the file is opened or anything mapped.
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
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{il}.{s}");
            let q = |s: &str| mapped.q(&n(s));
            let f = |s: &str| mapped.f32(&n(s));
            let side = |p: &str| -> Result<HcW> {
                Ok(HcW {
                    fn_w: q(&format!("hc_{p}_fn.weight"))?,
                    scale: f(&format!("hc_{p}_scale.weight"))?,
                    base: f(&format!("hc_{p}_base.weight"))?,
                })
            };
            let comp = |p: &str| -> Result<CompW> {
                Ok(CompW {
                    kv: q(&format!("{p}_kv.weight"))?,
                    gate: q(&format!("{p}_gate.weight"))?,
                    ape: f16_view(&mapped, &n(&format!("{p}_ape.weight")))?,
                    norm: f(&format!("{p}_norm.weight"))?,
                })
            };
            let hash = cfg.is_hash_layer(il);
            layers.push(LayerW {
                attn_norm: f("attn_norm.weight")?,
                ffn_norm: f("ffn_norm.weight")?,
                hc: [side("attn")?, side("ffn")?],
                q_a: q("attn_q_a.weight")?,
                q_a_norm: f("attn_q_a_norm.weight")?,
                q_b: q("attn_q_b.weight")?,
                kv: q("attn_kv.weight")?,
                kv_norm: f("attn_kv_a_norm.weight")?,
                sinks: f("attn_sinks.weight")?,
                out_a: q("attn_output_a.weight")?,
                out_b: q("attn_output_b.weight")?,
                comp: (cfg.ratio(il) != 0)
                    .then(|| comp("attn_compressor"))
                    .transpose()?,
                indexer: if cfg.has_indexer(il) {
                    Some(IndexerW {
                        comp: comp("indexer_compressor")?,
                        q_b: q("indexer.attn_q_b.weight")?,
                        proj: q("indexer.proj.weight")?,
                    })
                } else {
                    None
                },
                router: q("ffn_gate_inp.weight")?,
                bias: (!hash).then(|| f("exp_probs_b.bias")).transpose()?,
                hash: hash.then(|| q("ffn_gate_tid2eid.weight")).transpose()?,
                gate: q("ffn_gate_exps.weight")?,
                up: q("ffn_up_exps.weight")?,
                down: q("ffn_down_exps.weight")?,
                sh_gate: q("ffn_gate_shexp.weight")?,
                sh_up: q("ffn_up_shexp.weight")?,
                sh_down: q("ffn_down_shexp.weight")?,
            });
        }
        let embed = mapped.q("token_embd.weight")?;
        let output = mapped.q("output.weight")?;
        let output_norm = mapped.f32("output_norm.weight")?;
        let out_hc = HcW {
            fn_w: mapped.q("output_hc_fn.weight")?,
            scale: mapped.f32("output_hc_scale.weight")?,
            base: mapped.f32("output_hc_base.weight")?,
        };
        Ok(Self { cfg, gguf, mapped, layers, embed, output, output_norm, out_hc })
    }

    /// The buffers holding the mapped file.
    pub fn weight_buffers(&self) -> Vec<Buffer> {
        self.mapped.buffers()
    }

    pub fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Ds4State> {
        Ds4State::new(ctx, &self.cfg, capacity)
    }

    pub fn new_scratch(&self, ctx: &MetalContext) -> Result<Scratch> {
        Scratch::new(ctx, &self.cfg)
    }
}

/// One compressor's cache and ring.
pub struct CompState {
    /// f16 `[cap][dim]`: the compressed rows.
    pub cache: Tensor,
    pub(crate) ring_kv: Tensor,
    pub(crate) ring_sc: Tensor,
}

impl CompState {
    fn new(
        ctx: &MetalContext,
        dim: usize,
        ratio: usize,
        capacity: usize,
    ) -> Result<Self> {
        let lanes = if ratio == 4 { 2 } else { 1 };
        let ring = lanes * ratio * lanes * dim;
        Ok(Self {
            cache: Tensor::zeros(ctx, &[(capacity / ratio + 1) * dim], DType::F16)?,
            ring_kv: Tensor::zeros(ctx, &[ring], DType::F32)?,
            ring_sc: Tensor::zeros(ctx, &[ring], DType::F32)?,
        })
    }
}

/// One layer's caches.
pub struct LayerState {
    /// f16 `[128][512]`: the raw window, slot = position % 128.
    pub ring: Tensor,
    pub comp: Option<CompState>,
    pub index: Option<CompState>,
}

/// A session: caches at position `pos`.
pub struct Ds4State {
    pub(crate) cfg: Config,
    pub pos: usize,
    pub capacity: usize,
    pub layers: Vec<LayerState>,
    /// Indexer scores `[PREFILL_ROWS][score_cap]`.
    pub(crate) scores: Tensor,
    pub(crate) score_cap: usize,
    /// RoPE tables of positions `0..capacity` (`kernels::deepseek4::rope_table`):
    /// the plain layers' and the compressed layers' (YaRN). Built once, so a
    /// step encoded while the one before runs writes nothing the GPU reads.
    pub(crate) rope_plain: Tensor,
    pub(crate) rope_yarn: Tensor,
}

/// The RoPE tables of positions `0..n`.
pub(crate) fn rope_tables(
    ctx: &MetalContext,
    cfg: &Config,
    n: usize,
) -> Result<(Tensor, Tensor)> {
    let plain = d::rope_table(&cfg.rope(0), 0, n);
    let yarn = d::rope_table(&cfg.rope(2), 0, n);
    Ok((
        Tensor::from_f32(ctx, &plain, &[plain.len()])?,
        Tensor::from_f32(ctx, &yarn, &[yarn.len()])?,
    ))
}

impl Ds4State {
    pub fn new(ctx: &MetalContext, cfg: &Config, capacity: usize) -> Result<Self> {
        let layers = (0..cfg.n_layer)
            .map(|il| -> Result<LayerState> {
                let r = cfg.ratio(il);
                Ok(LayerState {
                    ring: Tensor::zeros(ctx, &[d::WINDOW * d::KV_DIM], DType::F16)?,
                    comp: (r != 0)
                        .then(|| CompState::new(ctx, cfg.head_dim, r, capacity))
                        .transpose()?,
                    index: cfg
                        .has_indexer(il)
                        .then(|| CompState::new(ctx, cfg.indexer_dim, r, capacity))
                        .transpose()?,
                })
            })
            .collect::<Result<_>>()?;
        let score_cap = capacity / 4 + 1;
        let (rope_plain, rope_yarn) = rope_tables(ctx, cfg, capacity)?;
        Ok(Self {
            cfg: cfg.clone(),
            pos: 0,
            capacity,
            layers,
            scores: Tensor::zeros(ctx, &[PREFILL_ROWS * score_cap], DType::F32)?,
            score_cap,
            rope_plain,
            rope_yarn,
        })
    }

    /// Back to position 0. The caches need no clearing: a row is read only
    /// after it is written; the compressor rings' first windows are handled
    /// by position.
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Per-run intermediates, for up to [`PREFILL_ROWS`] rows.
pub struct Scratch {
    pub tokens: Tensor,
    /// The HC streams `[m][4][hidden]`.
    pub h: Tensor,
    flat: Tensor,
    hc_partial: Tensor,
    split: Tensor,
    pub xn: Tensor,
    pub attn_out: Tensor,
    pub ffn_out: Tensor,
    qr: Tensor,
    pub q: Tensor,
    pub kv: Tensor,
    kv_run: Tensor,
    pub heads: Tensor,
    low: Tensor,
    ckv: Tensor,
    csc: Tensor,
    ikv: Tensor,
    isc: Tensor,
    iq: Tensor,
    iw: Tensor,
    isel: Tensor,
    icount: Tensor,
    fg: Tensor,
    fu: Tensor,
    fmid: Tensor,
    rlogits: Tensor,
    pub rsel: Tensor,
    rw: Tensor,
    moe_mid: Tensor,
    moe_group: k::MoeGroupScratch,
    opre: Tensor,
    pub hidden: Tensor,
    hidden_norm: Tensor,
    /// `[MAX_ROWS][vocab]`.
    pub logits: Tensor,
    /// The last (or only) row's logits: `logits[0..vocab]`.
    pub logits_last: Tensor,
    pub(crate) sampler: crate::kernels::sample::SamplerScratch,
    /// U32[2]: the ping-pong slots the in-graph sampler draws into.
    pub next_token: Tensor,
}

impl Scratch {
    fn new(ctx: &MetalContext, cfg: &Config) -> Result<Self> {
        let m = PREFILL_ROWS;
        let f = |n: usize| Tensor::zeros(ctx, &[n], DType::F32);
        let u = |n: usize| Tensor::zeros(ctx, &[n], DType::U32);
        let hd = cfg.head_dim;
        let logits = f(MAX_ROWS * cfg.vocab)?;
        Ok(Self {
            tokens: u(m)?,
            h: f(m * cfg.hc_dim())?,
            flat: f(m * cfg.hc_dim())?,
            hc_partial: f(m * k::hc_mix_groups(cfg.hidden) * 25)?,
            split: f(m * cfg.hc_mix())?,
            xn: f(m * cfg.hidden)?,
            attn_out: f(m * cfg.hidden)?,
            ffn_out: f(m * cfg.hidden)?,
            qr: f(m * cfg.q_lora)?,
            q: f(m * cfg.q_dim())?,
            kv: f(m * hd)?,
            kv_run: Tensor::zeros(ctx, &[m * hd], DType::F16)?,
            heads: f(m * cfg.q_dim())?,
            low: f(m * cfg.o_groups * cfg.o_lora)?,
            ckv: f(m * 2 * hd)?,
            csc: f(m * 2 * hd)?,
            ikv: f(m * 2 * cfg.indexer_dim)?,
            isc: f(m * 2 * cfg.indexer_dim)?,
            iq: f(m * cfg.indexer_heads * cfg.indexer_dim)?,
            iw: f(m * cfg.indexer_heads)?,
            isel: u(m * TOP_K)?,
            icount: u(m)?,
            fg: f(m * cfg.ff_expert)?,
            fu: f(m * cfg.ff_expert)?,
            fmid: f(m * cfg.ff_expert)?,
            rlogits: f(m * cfg.n_expert)?,
            rsel: u(m * cfg.n_expert_used)?,
            rw: f(m * cfg.n_expert_used)?,
            moe_mid: f(MAX_ROWS * cfg.n_expert_used * cfg.ff_expert)?,
            moe_group: k::MoeGroupScratch::new(
                ctx,
                m * cfg.n_expert_used,
                cfg.n_expert,
                cfg.ff_expert,
                cfg.hidden,
            )?,
            opre: f(m * cfg.n_hc)?,
            hidden: f(m * cfg.hidden)?,
            hidden_norm: f(m * cfg.hidden)?,
            logits_last: logits.view(0, &[cfg.vocab])?,
            logits,
            sampler: crate::kernels::sample::SamplerScratch::new(ctx, cfg.vocab)?,
            next_token: u(2)?,
        })
    }

    pub(crate) fn stage_tokens(&self, tokens: &[u32]) -> Result<()> {
        crate::glm53::model::write_words(&self.tokens, tokens)
    }
}

/// Which rows' logits a run computes into `Scratch::logits`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Logits {
    None,
    /// The last row's, at `logits[0..vocab]`.
    Last,
    /// Every row's (at most [`MAX_ROWS`] rows), row `i` at `logits[i * vocab..]`.
    All,
}

/// What a traced forward reads back after each layer (row-major over the
/// run's rows).
#[derive(Default, Clone)]
pub struct LayerTrace {
    pub attn_in: Vec<f32>,
    pub q: Vec<f32>,
    pub kv: Vec<f32>,
    pub heads: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub sel: Vec<u32>,
    /// The streams after the layer.
    pub h: Vec<f32>,
}

/// A run being encoded: its first position, rows and token ids (on the
/// GPU).
#[derive(Clone, Copy)]
struct Run<'t> {
    pos0: usize,
    m: usize,
    tokens: &'t Tensor,
}

fn read(t: &Tensor, n: usize) -> Result<Vec<f32>> {
    Ok(t.to_f32()?[..n].to_vec())
}

impl Ds4Model {
    /// Feeds `tokens` at the state's position in runs of at most
    /// [`PREFILL_ROWS`] and leaves the logits `logits` asks for (of the
    /// last run) in `sc.logits`.
    pub fn forward(
        &self,
        ctx: &MetalContext,
        st: &mut Ds4State,
        sc: &Scratch,
        tokens: &[u32],
        logits: Logits,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "forward of no tokens");
        ensure!(
            logits != Logits::All || tokens.len() <= MAX_ROWS,
            "logits of every row"
        );
        let n = tokens.len();
        for (i, chunk) in tokens.chunks(PREFILL_ROWS).enumerate() {
            let last = (i + 1) * PREFILL_ROWS >= n;
            self.run(
                ctx,
                st,
                sc,
                chunk,
                if last { logits } else { Logits::None },
                None,
            )?;
        }
        Ok(())
    }

    /// [`Self::forward`] of one run (at most [`PREFILL_ROWS`] tokens) reading
    /// each layer's intermediates back.
    pub fn forward_traced(
        &self,
        ctx: &MetalContext,
        st: &mut Ds4State,
        sc: &Scratch,
        tokens: &[u32],
        trace: &mut Vec<LayerTrace>,
    ) -> Result<()> {
        self.run(ctx, st, sc, tokens, Logits::Last, Some(trace))
    }

    pub(crate) fn run(
        &self,
        ctx: &MetalContext,
        st: &mut Ds4State,
        sc: &Scratch,
        tokens: &[u32],
        logits: Logits,
        trace: Option<&mut Vec<LayerTrace>>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let m = tokens.len();
        ensure!((1..=PREFILL_ROWS).contains(&m), "run of {m} rows");
        let pos0 = st.pos;
        ensure!(
            pos0 + m <= st.capacity,
            "state capacity {} for position {}",
            st.capacity,
            pos0 + m
        );
        for &t in tokens {
            ensure!((t as usize) < cfg.vocab, "token {t} outside the vocabulary");
        }
        sc.stage_tokens(tokens)?;
        if let Some(trace) = trace {
            self.run_traced(ctx, st, sc, m, trace)?;
            st.pos += m;
            return Ok(());
        }
        let n = cfg.n_layer;
        if m <= MAX_ROWS {
            let pass = ctx.begin_concurrent()?;
            self.encode_layers(ctx, &pass, st, sc, &sc.tokens, m, 0..n, logits)?;
            pass.commit_wait()?;
        } else {
            // A pass a layer keeps a command buffer's GPU time short: the
            // driver times one from its submission.
            for il in 0..n {
                let pass = ctx.begin_concurrent()?;
                self.encode_layers(
                    ctx,
                    &pass,
                    st,
                    sc,
                    &sc.tokens,
                    m,
                    il..il + 1,
                    logits,
                )?;
                pass.commit_wait()?;
            }
        }
        st.pos += m;
        Ok(())
    }

    /// Encodes layers `layers` of a run of `m` rows at the state's position
    /// into `pass`, the token ids read from `tokens` (U32) on the GPU,
    /// without advancing the state: the embedding when they start at layer
    /// 0, the final HC post and the head when they end at the last layer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layers(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Ds4State,
        sc: &Scratch,
        tokens: &Tensor,
        m: usize,
        layers: std::ops::Range<usize>,
        logits: Logits,
    ) -> Result<()> {
        let cfg = &self.cfg;
        ensure!((1..=PREFILL_ROWS).contains(&m), "run of {m} rows");
        ensure!(st.pos + m <= st.capacity, "state capacity {}", st.capacity);
        ensure!(layers.end <= cfg.n_layer, "layers {layers:?}");
        let r = Run { pos0: st.pos, m, tokens };
        if layers.start == 0 {
            d::embed(ctx, pass, &self.embed, tokens, &sc.h, m)?;
            pass.memory_barrier()?;
        }
        for il in layers.clone() {
            let l = &self.layers[il];
            // Each HC pre applies the HC post of the sublayer before it.
            let prev = (il > 0).then_some(&sc.ffn_out);
            self.hc_pre(ctx, pass, sc, &l.hc[0], &l.attn_norm, m, prev)?;
            self.attention(ctx, pass, st, sc, l, il, r)?;
            self.hc_pre(ctx, pass, sc, &l.hc[1], &l.ffn_norm, m, Some(&sc.attn_out))?;
            self.ffn(ctx, pass, sc, l, r)?;
        }
        if layers.end < cfg.n_layer {
            return Ok(());
        }
        k::hc_post(ctx, pass, &sc.h, &sc.ffn_out, &sc.split, &sc.h, m, cfg.hidden)?;
        pass.memory_barrier()?;
        self.head(ctx, pass, sc, m, logits)
    }

    /// A run committing after every sublayer and reading its intermediates
    /// back (the tokens staged in `sc.tokens`).
    fn run_traced(
        &self,
        ctx: &MetalContext,
        st: &Ds4State,
        sc: &Scratch,
        m: usize,
        trace: &mut Vec<LayerTrace>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let r = Run { pos0: st.pos, m, tokens: &sc.tokens };
        let mut pass = ctx.begin_concurrent()?;
        d::embed(ctx, &pass, &self.embed, &sc.tokens, &sc.h, m)?;
        pass.memory_barrier()?;
        for il in 0..cfg.n_layer {
            let l = &self.layers[il];
            self.hc_pre(ctx, &pass, sc, &l.hc[0], &l.attn_norm, m, None)?;
            pass.commit_wait()?;
            pass = ctx.begin_concurrent()?;
            let attn_in = read(&sc.xn, m * cfg.hidden)?;
            self.attention(ctx, &pass, st, sc, l, il, r)?;
            k::hc_post(
                ctx,
                &pass,
                &sc.h,
                &sc.attn_out,
                &sc.split,
                &sc.h,
                m,
                cfg.hidden,
            )?;
            pass.memory_barrier()?;
            self.hc_pre(ctx, &pass, sc, &l.hc[1], &l.ffn_norm, m, None)?;
            pass.commit_wait()?;
            pass = ctx.begin_concurrent()?;
            let ffn_in = read(&sc.xn, m * cfg.hidden)?;
            self.ffn(ctx, &pass, sc, l, r)?;
            k::hc_post(
                ctx,
                &pass,
                &sc.h,
                &sc.ffn_out,
                &sc.split,
                &sc.h,
                m,
                cfg.hidden,
            )?;
            pass.memory_barrier()?;
            pass.commit_wait()?;
            pass = ctx.begin_concurrent()?;
            trace.push(LayerTrace {
                attn_in,
                q: read(&sc.q, m * cfg.q_dim())?,
                kv: read(&sc.kv, m * cfg.head_dim)?,
                heads: read(&sc.heads, m * cfg.q_dim())?,
                attn_out: read(&sc.attn_out, m * cfg.hidden)?,
                ffn_in,
                ffn_out: read(&sc.ffn_out, m * cfg.hidden)?,
                sel: sc.rsel.to_u32()?[..m * cfg.n_expert_used].to_vec(),
                h: read(&sc.h, m * cfg.hc_dim())?,
            });
        }
        self.head(ctx, &pass, sc, m, Logits::Last)?;
        pass.commit_wait()
    }

    /// `y_s = w_s x` for `m` rows: one GEMV dispatch per set and chunk of
    /// [`MAX_ROWS`] rows (f32 throughout).
    fn proj(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sets: &[(&QWeight, &Tensor)],
        x: &Tensor,
        m: usize,
    ) -> Result<()> {
        if m > MAX_ROWS {
            // A prompt run: the matrix units (operands rounded to bfloat,
            // f32 accumulation).
            for &(w, y) in sets {
                k::gemm(ctx, pass, w, x, y, m)?;
            }
            return Ok(());
        }
        let cols = sets[0].0.cols;
        for c0 in (0..m).step_by(MAX_ROWS) {
            let rows = (m - c0).min(MAX_ROWS);
            let xv = x.view(c0 * cols, &[rows * cols])?;
            let ys: Vec<(&QWeight, Tensor)> = sets
                .iter()
                .map(|&(w, y)| Ok((w, y.view(c0 * w.rows, &[rows * w.rows])?)))
                .collect::<Result<_>>()?;
            let refs: Vec<(&QWeight, &Tensor)> =
                ys.iter().map(|(w, y)| (*w, y)).collect();
            k::gemv_sets(ctx, pass, &refs, &xv, rows)?;
        }
        Ok(())
    }

    /// HC pre of a sublayer: the control projection of the normalized
    /// streams, the Sinkhorn split into `sc.split` and the normalized
    /// sublayer input `sc.xn`.
    #[allow(clippy::too_many_arguments)]
    fn hc_pre(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        hc: &HcW,
        norm: &Tensor,
        m: usize,
        post: Option<&Tensor>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        // GLM's fused kernels: the previous sublayer's HC post (when `post`
        // gives its output; its weights are still in `sc.split`) with the
        // partial sums of the control projection over f32 streams, then
        // the Sinkhorn split and the normalized input. (The projection as
        // a bfloat GEMM at prefill cost the official-continuation score
        // 6 first-token matches, 2026-11-14.)
        let post = post.map(|block| (block, &sc.split));
        k::hc_post_mix(
            ctx,
            pass,
            &sc.h,
            post,
            &hc.fn_w,
            &sc.hc_partial,
            m,
            cfg.hidden,
        )?;
        pass.memory_barrier()?;
        k::hc_pre2(
            ctx,
            pass,
            &sc.hc_partial,
            &hc.scale,
            &hc.base,
            &sc.h,
            norm,
            &sc.split,
            &sc.xn,
            m,
            cfg.hidden,
            cfg.sinkhorn_iters,
            cfg.hc_eps,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()
    }

    /// The attention sublayer from `sc.xn` to `sc.attn_out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Ds4State,
        sc: &Scratch,
        l: &LayerW,
        il: usize,
        r: Run,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let (pos0, m) = (r.pos0, r.m);
        let ls = &st.layers[il];
        let yarn = cfg.ratio(il) != 0;
        // The state's RoPE tables are indexed by position.
        let (table, t0) =
            if yarn { (&st.rope_yarn, pos0) } else { (&st.rope_plain, pos0) };
        // Level 1: the projections of xn.
        self.proj(ctx, pass, &[(&l.q_a, &sc.qr), (&l.kv, &sc.kv)], &sc.xn, m)?;
        match (&l.comp, &l.indexer) {
            (Some(c), Some(ix)) => self.proj(
                ctx,
                pass,
                &[
                    (&c.kv, &sc.ckv),
                    (&c.gate, &sc.csc),
                    (&ix.comp.kv, &sc.ikv),
                    (&ix.comp.gate, &sc.isc),
                ],
                &sc.xn,
                m,
            )?,
            (Some(c), None) => self.proj(
                ctx,
                pass,
                &[(&c.kv, &sc.ckv), (&c.gate, &sc.csc)],
                &sc.xn,
                m,
            )?,
            _ => {}
        }
        if let Some(ix) = &l.indexer {
            self.proj(ctx, pass, &[(&ix.proj, &sc.iw)], &sc.xn, m)?;
        }
        pass.memory_barrier()?;
        // Level 2: norms, compressor pools.
        k::rmsnorm(
            ctx,
            pass,
            &sc.qr,
            Some(&l.q_a_norm),
            &sc.qr,
            m,
            cfg.q_lora,
            cfg.rms_eps,
        )?;
        k::rmsnorm(
            ctx,
            pass,
            &sc.kv,
            Some(&l.kv_norm),
            &sc.kv,
            m,
            cfg.head_dim,
            cfg.rms_eps,
        )?;
        let comp_gpu = l.comp.as_ref().map(|c| {
            let s = ls.comp.as_ref().expect("compressed layer state");
            CompressorGpu {
                ape: &c.ape,
                norm: &c.norm,
                ring_kv: &s.ring_kv,
                ring_sc: &s.ring_sc,
                dim: cfg.head_dim,
                ratio: cfg.ratio(il),
                indexer: false,
            }
        });
        let index_gpu = l.indexer.as_ref().map(|ix| {
            let s = ls.index.as_ref().expect("indexer state");
            CompressorGpu {
                ape: &ix.comp.ape,
                norm: &ix.comp.norm,
                ring_kv: &s.ring_kv,
                ring_sc: &s.ring_sc,
                dim: cfg.indexer_dim,
                ratio: cfg.ratio(il),
                indexer: true,
            }
        });
        if let (Some(cg), Some(s)) = (&comp_gpu, &ls.comp) {
            d::comp_pool(
                ctx,
                pass,
                cg,
                &sc.ckv,
                &sc.csc,
                &s.cache,
                &st.rope_yarn,
                0,
                pos0,
                m,
                cfg.rms_eps,
            )?;
        }
        if let (Some(cg), Some(s)) = (&index_gpu, &ls.index) {
            d::comp_pool(
                ctx,
                pass,
                cg,
                &sc.ikv,
                &sc.isc,
                &s.cache,
                &st.rope_yarn,
                0,
                pos0,
                m,
                cfg.rms_eps,
            )?;
        }
        pass.memory_barrier()?;
        // Level 3: queries, the KV store, compressor rings.
        self.proj(ctx, pass, &[(&l.q_b, &sc.q)], &sc.qr, m)?;
        let sparse = l.indexer.is_some() && (pos0 + m) / 4 > TOP_K;
        if let (true, Some(ix)) = (sparse, &l.indexer) {
            self.proj(ctx, pass, &[(&ix.q_b, &sc.iq)], &sc.qr, m)?;
        }
        d::kv_store(ctx, pass, &sc.kv, &sc.kv_run, table, t0, m)?;
        if let Some(cg) = &comp_gpu {
            d::comp_ring(ctx, pass, cg, &sc.ckv, &sc.csc, pos0, m)?;
        }
        if let Some(cg) = &index_gpu {
            d::comp_ring(ctx, pass, cg, &sc.ikv, &sc.isc, pos0, m)?;
        }
        pass.memory_barrier()?;
        // Level 4: query norm and rotation; indexer queries.
        d::head_norm_rope(
            ctx,
            pass,
            &sc.q,
            table,
            t0,
            m,
            cfg.n_head,
            cfg.head_dim,
            cfg.n_rot,
            cfg.rms_eps,
            HeadOp::NormRope,
        )?;
        if sparse {
            d::indexer_q(ctx, pass, &sc.iq, &st.rope_yarn, t0, m, cfg.indexer_heads)?;
        }
        pass.memory_barrier()?;
        let rows = if sparse {
            let index = ls.index.as_ref().expect("indexer state");
            let scale = 1.0 / ((cfg.indexer_dim * cfg.indexer_heads) as f32).sqrt();
            ensure!(m <= PREFILL_ROWS, "indexer scores of {m} rows");
            d::indexer_scores(
                ctx,
                pass,
                &sc.iq,
                &sc.iw,
                &index.cache,
                &st.scores,
                m,
                cfg.indexer_heads,
                st.score_cap,
                pos0,
                scale,
            )?;
            pass.memory_barrier()?;
            d::topk(
                ctx,
                pass,
                &st.scores,
                &sc.isel,
                &sc.icount,
                m,
                st.score_cap,
                pos0,
                TOP_K,
            )?;
            pass.memory_barrier()?;
            CompRows::Selected {
                cache: &ls.comp.as_ref().expect("compressed state").cache,
                sel: &sc.isel,
                count: &sc.icount,
                k: TOP_K,
            }
        } else if let Some(s) = &ls.comp {
            CompRows::All { cache: &s.cache, ratio: cfg.ratio(il) }
        } else {
            CompRows::None
        };
        d::attend(
            ctx, pass, &sc.q, &sc.kv_run, &ls.ring, rows, &l.sinks, &sc.heads, m,
            cfg.n_head, pos0,
        )?;
        pass.memory_barrier()?;
        // Rotate back; the ring takes the run's rows.
        d::head_norm_rope(
            ctx,
            pass,
            &sc.heads,
            table,
            t0,
            m,
            cfg.n_head,
            cfg.head_dim,
            cfg.n_rot,
            0.0,
            HeadOp::InverseRope,
        )?;
        d::ring_store(ctx, pass, &sc.kv_run, &ls.ring, pos0, m)?;
        pass.memory_barrier()?;
        // The grouped output projection.
        let group_in = cfg.o_group_heads() * cfg.head_dim;
        debug_assert_eq!(l.out_a.cols, group_in);
        if m > MAX_ROWS {
            // Per group: low[i][g] (1024) = out_a[g] (1024 x 4096) . heads[i][g].
            let gl = k::GemmLayout {
                ldx: cfg.q_dim(),
                ldy: l.out_a.rows,
                batches: cfg.o_groups,
                x_batch: group_in,
                w_batch: cfg.o_lora,
                y_batch: cfg.o_lora,
            };
            k::gemm_with(ctx, pass, &l.out_a, &sc.heads, &sc.low, m, cfg.o_lora, gl)?;
        }
        for c0 in (0..m).step_by(MAX_ROWS).take_while(|_| m <= MAX_ROWS) {
            let rows = (m - c0).min(MAX_ROWS);
            let x = sc.heads.view(c0 * cfg.q_dim(), &[rows * cfg.q_dim()])?;
            let y = sc.low.view(c0 * l.out_a.rows, &[rows * l.out_a.rows])?;
            k::gemv_with(
                ctx,
                pass,
                &l.out_a,
                &x,
                &y,
                rows,
                GemvLayout {
                    x_group: cfg.o_lora,
                    x_row_stride: cfg.q_dim(),
                    y_row_stride: l.out_a.rows,
                },
            )?;
        }
        pass.memory_barrier()?;
        self.proj(ctx, pass, &[(&l.out_b, &sc.attn_out)], &sc.low, m)?;
        pass.memory_barrier()
    }

    /// The FFN sublayer from `sc.xn` to `sc.ffn_out`.
    fn ffn(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        l: &LayerW,
        r: Run<'_>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let m = r.m;
        let lim = cfg.swiglu_limit;
        let used = cfg.n_expert_used;
        self.proj(ctx, pass, &[(&l.router, &sc.rlogits)], &sc.xn, m)?;
        self.proj(ctx, pass, &[(&l.sh_gate, &sc.fg), (&l.sh_up, &sc.fu)], &sc.xn, m)?;
        pass.memory_barrier()?;
        d::router(
            ctx,
            pass,
            &sc.rlogits,
            l.bias.as_ref(),
            l.hash.as_ref(),
            r.tokens,
            &sc.rsel,
            &sc.rw,
            m,
            cfg.n_expert,
            used,
            cfg.expert_weight_scale,
        )?;
        k::swiglu(ctx, pass, &sc.fg, &sc.fu, &sc.fmid, m * cfg.ff_expert, lim)?;
        pass.memory_barrier()?;
        if m > MAX_ROWS {
            // A prompt run: the shared expert, then the routed experts
            // grouped by expert on the matrix units (adding to it).
            self.proj(ctx, pass, &[(&l.sh_down, &sc.ffn_out)], &sc.fmid, m)?;
            pass.memory_barrier()?;
            k::moe_grouped(
                ctx,
                pass,
                &l.gate,
                &l.up,
                &l.down,
                cfg.ff_expert,
                cfg.hidden,
                &sc.xn,
                &sc.rsel,
                &sc.rw,
                &sc.ffn_out,
                &sc.moe_group,
                m,
                used,
                cfg.n_expert,
                lim,
            )?;
            return pass.memory_barrier();
        }
        k::moe_gate_up(
            ctx,
            pass,
            &l.gate,
            &l.up,
            cfg.ff_expert,
            &sc.xn,
            &sc.rsel,
            &sc.rw,
            &sc.moe_mid,
            m,
            used,
            lim,
        )?;
        self.proj(ctx, pass, &[(&l.sh_down, &sc.ffn_out)], &sc.fmid, m)?;
        pass.memory_barrier()?;
        k::moe_down(
            ctx,
            pass,
            &l.down,
            cfg.hidden,
            &sc.moe_mid,
            &sc.rsel,
            Some(&sc.ffn_out),
            &sc.ffn_out,
            m,
            used,
        )?;
        pass.memory_barrier()
    }

    /// The output head: HC collapse, norm, vocabulary projection of the
    /// rows `logits` asks for (the collapse of every row is kept in
    /// `sc.hidden`).
    fn head(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        m: usize,
        logits: Logits,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let (first, rows) = match logits {
            Logits::None => return Ok(()),
            Logits::Last => (m - 1, 1),
            Logits::All => (0, m),
        };
        ensure!(rows <= MAX_ROWS, "logits of {rows} rows");
        // Only the rows whose logits are asked for, through the f32 GEMVs.
        let h = sc.h.view(first * cfg.hc_dim(), &[rows * cfg.hc_dim()])?;
        k::rmsnorm(ctx, pass, &h, None, &sc.flat, rows, cfg.hc_dim(), cfg.rms_eps)?;
        pass.memory_barrier()?;
        self.proj(ctx, pass, &[(&self.out_hc.fn_w, &sc.opre)], &sc.flat, rows)?;
        pass.memory_barrier()?;
        d::out_collapse(
            ctx,
            pass,
            &sc.opre,
            &self.out_hc.scale,
            &self.out_hc.base,
            &h,
            &sc.hidden,
            rows,
            cfg.hidden,
            cfg.hc_eps,
        )?;
        pass.memory_barrier()?;
        k::rmsnorm(
            ctx,
            pass,
            &sc.hidden,
            Some(&self.output_norm),
            &sc.hidden_norm,
            rows,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        k::gemv(ctx, pass, &self.output, &sc.hidden_norm, &sc.logits, rows)?;
        pass.memory_barrier()
    }
}
