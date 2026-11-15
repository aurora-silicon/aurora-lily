//! GLM 5.3 Flash on the GPU: weights mapped read-only from the GGUF, the
//! per-session caches and recurrent state, and the forward pass over up to
//! [`MAX_ROWS`] consecutive tokens at a time (decode is one row; a prompt is
//! fed in runs of rows).
//!
//! The graph is ds4's (`glm_graph_forward_token`); `reference.rs` holds the
//! same graph on the CPU and `forward_traced` exposes each layer's
//! intermediates for a layer-by-layer comparison.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};

use super::config::{Config, LayerKind};
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::kernels::glm53::{
    self as k, GemvLayout, KdaInputs, KdaParams, PoolParams, QWeight, Rows,
};
use crate::kernels::sample::SamplerScratch;
use crate::metal::{Buffer, ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Rows one forward step takes (the GEMV's limit).
pub const MAX_ROWS: usize = k::GEMV_MAX_ROWS;

/// Rows a prefill step takes (projections on the matrix units).
pub const PREFILL_ROWS: usize = 2048;

/// Mapped regions are at most this large (single tensors larger than it get
/// a region of their own).
const REGION_BYTES: u64 = 1 << 30;

/// The tensors of a GGUF file, mapped (GLM's, DeepSeek's, Qwen3.6's and
/// Gemma 4's models).
pub(crate) struct Mapped {
    tensors: std::collections::HashMap<String, (Buffer, usize, TensorInfo)>,
    pub(crate) buffers: Vec<Buffer>,
}

impl Mapped {
    pub(crate) fn map(ctx: &MetalContext, g: &Gguf) -> Result<Self> {
        Self::map_only(ctx, g, |_| true)
    }

    /// [`Self::map`] of the tensors `keep` selects only (Qwen3.6's MTP file
    /// also carries copies of the embedding and LM head it does not need).
    pub(crate) fn map_only(
        ctx: &MetalContext,
        g: &Gguf,
        keep: impl Fn(&TensorInfo) -> bool,
    ) -> Result<Self> {
        let mut order: Vec<&TensorInfo> =
            g.tensors.iter().filter(|t| keep(t)).collect();
        order.sort_by_key(|t| t.offset);
        let mut tensors = std::collections::HashMap::new();
        let mut buffers = Vec::new();
        let mut i = 0;
        while i < order.len() {
            let start = order[i].offset;
            let mut end = start + order[i].bytes;
            let mut j = i + 1;
            while j < order.len()
                && order[j].offset + order[j].bytes - start <= REGION_BYTES
            {
                end = order[j].offset + order[j].bytes;
                j += 1;
            }
            let (buf, inner) = ctx
                .new_buffer_mapped(&g.path, start, usize::try_from(end - start)?)
                .with_context(|| format!("mapping {} bytes at {start}", end - start))?;
            for t in &order[i..j] {
                let at = inner + usize::try_from(t.offset - start)?;
                tensors.insert(t.name.clone(), (buf.clone(), at, (*t).clone()));
            }
            buffers.push(buf);
            i = j;
        }
        Ok(Self { tensors, buffers })
    }

    pub(crate) fn buffers(&self) -> Vec<Buffer> {
        self.buffers.clone()
    }

    pub(crate) fn get(&self, name: &str) -> Result<&(Buffer, usize, TensorInfo)> {
        self.tensors.get(name).with_context(|| format!("GGUF has no tensor {name}"))
    }

    pub(crate) fn q(&self, name: &str) -> Result<QWeight> {
        let (buf, at, t) = self.get(name)?;
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length()], DType::I8)?;
        let view = whole.view(*at, &[t.bytes as usize])?;
        QWeight::new(view, t.ty, t.row_len(), t.rows())
    }

    pub(crate) fn f32(&self, name: &str) -> Result<Tensor> {
        let (buf, at, t) = self.get(name)?;
        ensure!(t.ty == GgmlType::F32, "{name} is {:?}, expected F32", t.ty);
        ensure!(at % 4 == 0, "{name} is not 4-byte aligned in its mapping");
        let n = t.row_len() * t.rows();
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length() / 4], DType::F32)?;
        whole.view(at / 4, &[n])
    }

    /// Raw bytes as I8 (the BF16 `pool_ape`).
    pub(crate) fn bytes(&self, name: &str) -> Result<Tensor> {
        let (buf, at, t) = self.get(name)?;
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length()], DType::I8)?;
        whole.view(*at, &[t.bytes as usize])
    }
}

struct HcW {
    fn_w: QWeight,
    scale: Tensor,
    base: Tensor,
}

struct KdaW {
    q: QWeight,
    k: QWeight,
    v: QWeight,
    f_a: QWeight,
    f_b: QWeight,
    beta: QWeight,
    g_a: QWeight,
    g_b: QWeight,
    out: QWeight,
    q_conv: Tensor,
    k_conv: Tensor,
    v_conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    o_norm: Tensor,
}

struct IndexerW {
    q_b: QWeight,
    k: QWeight,
    proj: QWeight,
    gate: QWeight,
    norm_w: Tensor,
    norm_b: Tensor,
    ape: Tensor,
}

struct DsaW {
    q_a: QWeight,
    q_a_norm: Tensor,
    q_b: QWeight,
    kv_a: QWeight,
    kv_a_norm: Tensor,
    k_b: QWeight,
    v_b: QWeight,
    out: QWeight,
    indexer: Option<IndexerW>,
}

enum AttnW {
    Kda(KdaW),
    Dsa(DsaW),
}

enum FfnW {
    Dense {
        gate: QWeight,
        up: QWeight,
        down: QWeight,
    },
    Moe {
        router: QWeight,
        bias: Tensor,
        gate: QWeight,
        up: QWeight,
        down: QWeight,
        sh_gate: QWeight,
        sh_up: QWeight,
        sh_down: QWeight,
    },
}

struct LayerW {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    hc: Option<[HcW; 2]>,
    attn: AttnW,
    ffn: FfnW,
}

/// The MTP block's own weights besides its layer's (`blk.45.nextn.*`).
struct NextnW {
    eh_proj: QWeight,
    enorm: Tensor,
    hnorm: Tensor,
    shared_head_norm: Tensor,
}

/// The model's weights on the GPU.
pub struct Glm53Model {
    pub cfg: Config,
    pub gguf: Gguf,
    mapped: Mapped,
    layers: Vec<LayerW>,
    nextn: NextnW,
    embed: QWeight,
    output: QWeight,
    /// The output head requantized to Q4_0 (half the bytes) for the MTP
    /// draft only ([`Self::build_draft_head`]): what is emitted is decided
    /// by the full head of the verify pass.
    draft_head: Option<QWeight>,
    output_norm: Tensor,
    /// Whether the engine speculates with the MTP block (`--mtp-drafts`).
    pub mtp: bool,
    /// The engine proposes the MTP block's draft only when its softmax
    /// probability is at least this; otherwise the next verify pass is a
    /// one-row step.
    pub draft_min_prob: f32,
}

impl Glm53Model {
    /// Opens and maps `path` (the GGUF file).
    pub fn load(ctx: &MetalContext, path: &Path) -> Result<Self> {
        Self::load_locked(ctx, path, crate::instance::acquire)
    }

    /// [`Self::load`] with the instance lock taken by `lock`, first: before
    /// the file is opened or anything is mapped (one model on this machine
    /// at a time, `crate::instance`). Tests lock a file of their own.
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
            let hc = if il < cfg.n_trunk {
                let side = |p: &str| -> Result<HcW> {
                    Ok(HcW {
                        fn_w: q(&format!("hc_{p}_fn.weight"))?,
                        scale: f(&format!("hc_{p}_scale.weight"))?,
                        base: f(&format!("hc_{p}_base.weight"))?,
                    })
                };
                Some([side("attn")?, side("ffn")?])
            } else {
                None
            };
            let attn = match cfg.kinds[il] {
                LayerKind::Kda => AttnW::Kda(KdaW {
                    q: q("kda_q.weight")?,
                    k: q("kda_k.weight")?,
                    v: q("kda_v.weight")?,
                    f_a: q("kda_f_a.weight")?,
                    f_b: q("kda_f_b.weight")?,
                    beta: q("kda_beta.weight")?,
                    g_a: q("kda_g_a.weight")?,
                    g_b: q("kda_g_b.weight")?,
                    out: q("kda_output.weight")?,
                    q_conv: f("kda_q_conv.weight")?,
                    k_conv: f("kda_k_conv.weight")?,
                    v_conv: f("kda_v_conv.weight")?,
                    a_log: f("kda_a_log.weight")?,
                    dt_bias: f("kda_dt_bias.weight")?,
                    o_norm: f("kda_o_norm.weight")?,
                }),
                LayerKind::Dsa => AttnW::Dsa(DsaW {
                    q_a: q("attn_q_a.weight")?,
                    q_a_norm: f("attn_q_a_norm.weight")?,
                    q_b: q("attn_q_b.weight")?,
                    kv_a: q("attn_kv_a_mqa.weight")?,
                    kv_a_norm: f("attn_kv_a_norm.weight")?,
                    k_b: q("attn_k_b.weight")?,
                    v_b: q("attn_v_b.weight")?,
                    out: q("attn_output.weight")?,
                    indexer: if il < cfg.n_trunk {
                        Some(IndexerW {
                            q_b: q("indexer.attn_q_b.weight")?,
                            k: q("indexer.attn_k.weight")?,
                            proj: q("indexer.proj.weight")?,
                            gate: q("indexer.pool_gate.weight")?,
                            norm_w: f("indexer.k_norm.weight")?,
                            norm_b: f("indexer.k_norm.bias")?,
                            ape: mapped.bytes(&n("indexer.pool_ape.weight"))?,
                        })
                    } else {
                        None
                    },
                }),
            };
            let ffn = if cfg.is_moe(il) {
                FfnW::Moe {
                    router: q("ffn_gate_inp.weight")?,
                    bias: f("exp_probs_b.bias")?,
                    gate: q("ffn_gate_exps.weight")?,
                    up: q("ffn_up_exps.weight")?,
                    down: q("ffn_down_exps.weight")?,
                    sh_gate: q("ffn_gate_shexp.weight")?,
                    sh_up: q("ffn_up_shexp.weight")?,
                    sh_down: q("ffn_down_shexp.weight")?,
                }
            } else {
                FfnW::Dense {
                    gate: q("ffn_gate.weight")?,
                    up: q("ffn_up.weight")?,
                    down: q("ffn_down.weight")?,
                }
            };
            layers.push(LayerW {
                attn_norm: f("attn_norm.weight")?,
                ffn_norm: f("ffn_norm.weight")?,
                hc,
                attn,
                ffn,
            });
        }
        let embed = mapped.q("token_embd.weight")?;
        let output = mapped.q("output.weight")?;
        let output_norm = mapped.f32("output_norm.weight")?;
        let n = |s: &str| format!("blk.{}.nextn.{s}", cfg.mtp_layer());
        let nextn = NextnW {
            eh_proj: mapped.q(&n("eh_proj.weight"))?,
            enorm: mapped.f32(&n("enorm.weight"))?,
            hnorm: mapped.f32(&n("hnorm.weight"))?,
            shared_head_norm: mapped.f32(&n("shared_head_norm.weight"))?,
        };
        Ok(Self {
            cfg,
            gguf,
            mapped,
            layers,
            nextn,
            embed,
            output,
            draft_head: None,
            output_norm,
            mtp: false,
            draft_min_prob: 0.0,
        })
    }

    /// Builds the MTP draft's Q4_0 output head from the Q8_0 head (each
    /// block's 32 values requantized as llama.cpp's Q4_0: the scale is the
    /// largest magnitude's signed value over -8). A draft step reads 0.36
    /// instead of 0.67 GB of head. No-op for a head that is not Q8_0.
    pub fn build_draft_head(&mut self, ctx: &MetalContext) -> Result<()> {
        let w = &self.output;
        if w.ty != crate::gguf::GgmlType::Q8_0 {
            return Ok(());
        }
        ensure!(w.cols.is_multiple_of(32), "head rows of {} values", w.cols);
        let src = w.t.contents();
        let blocks = w.rows * w.cols / 32;
        ensure!(src.len() >= blocks * 34, "Q8_0 head of {} bytes", src.len());
        let mut dst = vec![0u8; blocks * 18];
        let threads =
            std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let per = blocks.div_ceil(threads);
        std::thread::scope(|scope| {
            for (t, out) in dst.chunks_mut(per * 18).enumerate() {
                let src = &src[t * per * 34..];
                scope.spawn(move || {
                    for (b, o) in out.chunks_mut(18).enumerate() {
                        q8_0_to_q4_0(&src[b * 34..b * 34 + 34], o);
                    }
                });
            }
        });
        self.draft_head = Some(QWeight::from_bytes(
            ctx,
            &dst,
            crate::gguf::GgmlType::Q4_0,
            w.cols,
            w.rows,
        )?);
        Ok(())
    }

    /// The buffers holding the mapped file.
    pub fn weight_buffers(&self) -> Vec<Buffer> {
        self.mapped.buffers.clone()
    }

    pub fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Glm53State> {
        Glm53State::new(ctx, &self.cfg, capacity)
    }

    pub fn new_scratch(&self, ctx: &MetalContext) -> Result<Scratch> {
        Scratch::new(ctx, &self.cfg)
    }
}

/// One layer's recurrent state or caches.
pub enum LayerState {
    /// `conv_alt`/`s_alt` receive the state after a verify pass's first row,
    /// which becomes current when its draft is rejected.
    Kda {
        conv: Tensor,
        s: Tensor,
        conv_alt: Tensor,
        s_alt: Tensor,
    },
    Dsa {
        latent: Tensor,
        pooled: Tensor,
        tail_k: Tensor,
        tail_g: Tensor,
    },
}

/// A session: caches and recurrent state at position `pos`.
pub struct Glm53State {
    pub pos: usize,
    pub capacity: usize,
    pub layers: Vec<LayerState>,
    /// Indexer pool scores, `[MAX_ROWS][pool_cap]` (scratch sized by the
    /// session's capacity).
    pub(crate) pool_scores: Tensor,
    pub(crate) pool_cap: usize,
    /// The first main position whose MTP cache row is written in the
    /// current run of MTP steps (the draft head attends from there).
    pub(crate) mtp_min: Option<usize>,
    /// The main position the next MTP step continues the run at.
    pub(crate) mtp_next: Option<usize>,
    /// A verify pass that fed rows from this position and the draws of its
    /// rows, until `finish_speculation` decides what stays.
    pub(crate) spec: Option<(usize, usize, Vec<u32>)>,
}

impl Glm53State {
    pub fn new(ctx: &MetalContext, cfg: &Config, capacity: usize) -> Result<Self> {
        let capacity = capacity.next_multiple_of(4);
        let p = cfg.kda_projection();
        // Every KDA layer's conv windows and states, and their alternates,
        // views of one allocation (about 140 allocations of their own cost
        // ~10 ms a session), zeroed on the GPU below: the recurrent state
        // starts at zero (the alternates are written before they are read).
        let conv_len = 3 * (cfg.kda_conv - 1) * p;
        let s_len = cfg.kda_heads * cfg.kda_dim * cfg.kda_dim;
        let n_kda = cfg.kinds.iter().filter(|k| **k == LayerKind::Kda).count();
        let kda_all =
            Tensor::uninit(ctx, &[n_kda * 2 * (conv_len + s_len)], DType::F32)?;
        let kda_at = std::cell::Cell::new(0usize);
        let take = |n: usize| -> Result<Tensor> {
            let at = kda_at.get();
            kda_at.set(at + n);
            kda_all.view(at, &[n])
        };
        let layers = (0..cfg.n_layer)
            .map(|il| -> Result<LayerState> {
                Ok(match cfg.kinds[il] {
                    LayerKind::Kda => LayerState::Kda {
                        conv: take(conv_len)?,
                        s: take(s_len)?,
                        conv_alt: take(conv_len)?,
                        s_alt: take(s_len)?,
                    },
                    LayerKind::Dsa => LayerState::Dsa {
                        latent: Tensor::zeros(
                            ctx,
                            &[capacity * cfg.kv_lora],
                            DType::F16,
                        )?,
                        pooled: Tensor::zeros(
                            ctx,
                            &[capacity / 4 * cfg.indexer_dim],
                            DType::F16,
                        )?,
                        tail_k: Tensor::zeros(ctx, &[4 * cfg.indexer_dim], DType::F32)?,
                        tail_g: Tensor::zeros(ctx, &[4 * cfg.indexer_dim], DType::F32)?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // 305 MB zeroed on the GPU in well under a millisecond, where the
        // host's memset of fresh buffers took ~20 ms (2026-11-15).
        let pass = ctx.begin()?;
        k::zero(ctx, &pass, &kda_all)?;
        pass.commit_wait()?;
        let pool_cap = capacity / 4 + 1;
        let pool_scores = Tensor::zeros(ctx, &[PREFILL_ROWS * pool_cap], DType::F32)?;
        Ok(Self {
            pos: 0,
            capacity,
            layers,
            pool_scores,
            pool_cap,
            mtp_min: None,
            mtp_next: None,
            spec: None,
        })
    }

    /// Zeroes everything (GPU idle).
    pub fn reset(&mut self) {
        for l in &self.layers {
            match l {
                LayerState::Kda { conv, s, .. } => {
                    conv.zero_fill();
                    s.zero_fill();
                }
                LayerState::Dsa { tail_k, tail_g, .. } => {
                    tail_k.zero_fill();
                    tail_g.zero_fill();
                }
            }
        }
        self.pos = 0;
        self.mtp_min = None;
        self.mtp_next = None;
        self.spec = None;
    }

    /// Makes the KDA states saved after a verify pass's first row current.
    pub(crate) fn take_kda_snapshots(&mut self) {
        for l in &mut self.layers {
            if let LayerState::Kda { conv, s, conv_alt, s_alt } = l {
                std::mem::swap(conv, conv_alt);
                std::mem::swap(s, s_alt);
            }
        }
    }
}

/// One Q8_0 block (f16 scale, 32 int8) as a Q4_0 block (f16 scale, 16
/// bytes of nibbles: value j low, j + 16 high), as llama.cpp's
/// `quantize_row_q4_0_ref` quantizes the block's values.
fn q8_0_to_q4_0(b: &[u8], out: &mut [u8]) {
    let d8 = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
    let x: [f32; 32] = std::array::from_fn(|i| d8 * f32::from(b[2 + i] as i8));
    let max =
        x.iter().copied().fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    let q = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
    for j in 0..16 {
        out[2 + j] = q(x[j]) | (q(x[j + 16]) << 4);
    }
}

/// Per-step intermediates, for up to [`MAX_ROWS`] rows.
pub struct Scratch {
    pub tokens: Tensor,
    pub h: Tensor,
    hc_partial: Tensor,
    split: Tensor,
    pub xn: Tensor,
    pub attn_out: Tensor,
    pub ffn_out: Tensor,
    // KDA
    kq: Tensor,
    kk: Tensor,
    kv: Tensor,
    kgate: Tensor,
    kbeta: Tensor,
    kog: Tensor,
    klow: Tensor,
    klow2: Tensor,
    kout: Tensor,
    /// BF16: a prompt run's GEMM input rounded once (bf16_input).
    xbf: Tensor,
    /// The conv windows before each chunk of a prompt run's KDA prepare.
    kedges: Tensor,
    // DSA
    qr: Tensor,
    q: Tensor,
    kv_raw: Tensor,
    qk_low: Tensor,
    scores: Tensor,
    denom: Tensor,
    lat: Tensor,
    /// The DSA weighted sum's partial sums of a few rows' slot chunks.
    dsa_partial: Tensor,
    heads: Tensor,
    ik: Tensor,
    ig: Tensor,
    iq: Tensor,
    iw: Tensor,
    sel: Tensor,
    sel_count: Tensor,
    // FFN
    fg: Tensor,
    fu: Tensor,
    fmid: Tensor,
    rlogits: Tensor,
    rsel: Tensor,
    rw: Tensor,
    moe_mid: Tensor,
    moe_group: k::MoeGroupScratch,
    // Head
    /// The stream mean of the last step's rows (the MTP head's input).
    pub hidden: Tensor,
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
    /// U32[2]: the MTP step's input token (0) and its draft (1).
    pub mtp_io: Tensor,
    mtp_emb: Tensor,
    mtp_cat: Tensor,
    mtp_x: Tensor,
    mtp_h1: Tensor,
    /// U32[S]: the cache rows the MTP head attends.
    mtp_sel: Tensor,
    /// F32[1]: the MTP draft's softmax probability.
    pub mtp_prob: Tensor,
}

/// Writes `words` to the start of the U32 tensor `t` and zeroes the rest:
/// staging through this cannot disagree with the tensor's size. GPU idle on
/// `t`.
pub(crate) fn write_words(t: &Tensor, words: &[u32]) -> Result<()> {
    ensure!(t.dtype() == DType::U32, "staging words into a {:?} tensor", t.dtype());
    ensure!(
        words.len() <= t.numel(),
        "{} words do not fit a tensor of {}",
        words.len(),
        t.numel()
    );
    let mut all = vec![0u32; t.numel()];
    all[..words.len()].copy_from_slice(words);
    t.write_bytes(bytemuck::cast_slice(&all))
}

impl Scratch {
    /// Stages a prefill or decode run's token ids (at most PREFILL_ROWS).
    pub(crate) fn stage_tokens(&self, tokens: &[u32]) -> Result<()> {
        write_words(&self.tokens, tokens)
    }

    /// Stages a verify pass's rows: the pending token, then the drafts.
    pub(crate) fn stage_spec_tokens(&self, pending: u32, drafts: &[u32]) -> Result<()> {
        let rows: Vec<u32> =
            std::iter::once(pending).chain(drafts.iter().copied()).collect();
        ensure!(rows.len() <= MAX_ROWS, "verify of {} rows", rows.len());
        write_words(&self.spec_tokens, &rows)
    }

    /// Stages the MTP step's input token (its draft slot cleared).
    pub(crate) fn stage_mtp_token(&self, token: u32) -> Result<()> {
        write_words(&self.mtp_io, &[token])
    }

    fn new(ctx: &MetalContext, cfg: &Config) -> Result<Self> {
        let m = PREFILL_ROWS;
        let f = |n: usize| Tensor::zeros(ctx, &[n], DType::F32);
        let u = |n: usize| Tensor::zeros(ctx, &[n], DType::U32);
        let p = cfg.kda_projection();
        let hq = cfg.n_head * cfg.key_dim;
        let s = cfg.indexer_selected_limit().next_multiple_of(8);
        // Logits of every row only for steps of up to MAX_ROWS (verify,
        // scoring): 256 rows would be 158 MB.
        let logits = f(MAX_ROWS * cfg.vocab)?;
        Ok(Self {
            logits_last: logits.view(0, &[cfg.vocab])?,
            tokens: u(m)?,
            h: f(m * cfg.hc_dim())?,
            hc_partial: f(m * k::hc_mix_groups(cfg.hidden) * 25)?,
            split: f(m * cfg.hc_mix())?,
            xn: f(m * cfg.hidden)?,
            attn_out: f(m * cfg.hidden)?,
            ffn_out: f(m * cfg.hidden)?,
            kq: f(m * p)?,
            kk: f(m * p)?,
            kv: f(m * p)?,
            kgate: f(m * p)?,
            kbeta: f(m * cfg.kda_heads)?,
            kog: f(m * p)?,
            klow: f(m * cfg.kda_dim)?,
            klow2: f(m * cfg.kda_dim)?,
            kout: f(m * p)?,
            xbf: Tensor::zeros(
                ctx,
                &[m * [
                    cfg.hidden,
                    p,
                    cfg.ff_dense,
                    cfg.ff_expert,
                    cfg.n_head * cfg.value_dim,
                    cfg.q_lora,
                ]
                .into_iter()
                .max()
                .unwrap_or(0)],
                DType::BF16,
            )?,
            kedges: f(k::kda_edges_len(m, cfg.kda_heads))?,
            qr: f(m * cfg.q_lora)?,
            q: f(m * hq)?,
            kv_raw: f(m * cfg.kv_lora)?,
            qk_low: f(m * cfg.n_head * cfg.kv_lora)?,
            scores: f(m * cfg.n_head * s)?,
            denom: f(m * cfg.n_head)?,
            lat: f(m * cfg.n_head * cfg.kv_lora)?,
            dsa_partial: f(MAX_ROWS * k::DSA_LORA_CHUNKS * cfg.n_head * cfg.kv_lora)?,
            heads: f(m * cfg.n_head * cfg.value_dim)?,
            ik: f(m * cfg.indexer_dim)?,
            ig: f(m * cfg.indexer_dim)?,
            iq: f(m * cfg.indexer_heads * cfg.indexer_dim)?,
            iw: f(m * cfg.indexer_heads)?,
            sel: u(m * s)?,
            sel_count: u(m)?,
            fg: f(m * cfg.ff_dense)?,
            fu: f(m * cfg.ff_dense)?,
            fmid: f(m * cfg.ff_dense)?,
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
            hidden: f(m * cfg.hidden)?,
            hidden_norm: f(m * cfg.hidden)?,
            logits,
            sampler: SamplerScratch::new(ctx, cfg.vocab)?,
            next_token: u(2)?,
            // A verify pass's rows, not a prefill run's.
            draws: u(MAX_ROWS)?,
            spec_tokens: u(MAX_ROWS)?,
            mtp_io: u(2)?,
            mtp_emb: f(cfg.hc_dim())?,
            mtp_cat: f(2 * cfg.hidden)?,
            mtp_x: f(cfg.hidden)?,
            mtp_h1: f(cfg.hidden)?,
            mtp_sel: u(s)?,
            mtp_prob: f(1)?,
        })
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
    /// Every row's, row `i` at `logits[i * vocab..]` (scoring).
    All,
}

/// What a traced forward reads back after each layer (row-major over the
/// step's rows).
#[derive(Default, Clone)]
pub struct LayerTrace {
    pub attn_in: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub h: Vec<f32>,
}

/// One layer of [`Glm53Model::step_layers`], row-major over the step's rows.
#[derive(Default, Clone)]
pub struct StepLayer {
    pub attn_out: Vec<f32>,
    /// The FFN's input (after the HC pre).
    pub ffn_in: Vec<f32>,
    pub ffn_out: Vec<f32>,
    /// Routed layers: each row's selected experts, as the GPU chose them.
    pub sel: Vec<u32>,
    /// Routed layers: each row's 8th-best minus 9th-best selection score
    /// (sigmoid(logit) + bias), reconstructed on the CPU (its own `exp`)
    /// from the GPU's router logits: near-tie evidence, not the GPU's own
    /// scores.
    pub margin: Vec<f32>,
}

/// The first fed position below `end`, and its layer, where two runs'
/// routing records (by state position: per layer the sorted selected
/// experts) differ; positions only one run has are skipped. `end` limits
/// the search to the rows both runs fed with the same tokens: for outputs
/// that first differ at decode token `d` after a prompt of `n` tokens it is
/// `n + d` (position `n + d - 1` consumes the last common token).
pub fn first_routing_difference(
    a: &std::collections::BTreeMap<usize, Vec<Vec<u32>>>,
    b: &std::collections::BTreeMap<usize, Vec<Vec<u32>>>,
    end: usize,
) -> Option<(usize, usize)> {
    a.range(..end).find_map(|(&p, la)| {
        let lb = b.get(&p)?;
        la.iter().zip(lb).position(|(x, y)| x != y).map(|il| (p, il))
    })
}

impl Glm53Model {
    /// Feeds `tokens` (at most [`MAX_ROWS`]) at the state's position and
    /// leaves the last row's logits in `scratch.logits` when `logits`.
    pub fn forward(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
        logits: bool,
    ) -> Result<()> {
        let l = if logits { Logits::Last } else { Logits::None };
        self.forward_inner(ctx, st, sc, tokens, l, None)
    }

    /// A decode/verify step of `tokens` (at most [`MAX_ROWS`]) through the
    /// production kernels (`run_step`'s), one pass a layer, reading each
    /// layer's intermediates back (diagnostics: `lily-glm53 spec-diff`).
    /// Advances the state; returns the layers and every row's logits.
    pub fn step_layers(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
    ) -> Result<(Vec<StepLayer>, Vec<f32>)> {
        let cfg = &self.cfg;
        let m = tokens.len();
        ensure!((1..=MAX_ROWS).contains(&m), "step of {m} rows");
        ensure!(st.pos + m <= st.capacity, "state capacity {}", st.capacity);
        sc.stage_tokens(tokens)?;
        let snap = (m > 1).then_some(0);
        let mut out = Vec::with_capacity(cfg.n_trunk);
        for il in 0..cfg.n_trunk {
            let pass = self.begin(ctx)?;
            let r = il..il + 1;
            self.encode_layers(
                ctx,
                &pass,
                st,
                sc,
                &sc.tokens,
                m,
                Logits::All,
                snap,
                r,
            )?;
            pass.commit_wait()?;
            let mut l = StepLayer {
                attn_out: read(&sc.attn_out, m * cfg.hidden)?,
                ffn_in: read(&sc.xn, m * cfg.hidden)?,
                ffn_out: read(&sc.ffn_out, m * cfg.hidden)?,
                ..Default::default()
            };
            if let FfnW::Moe { bias, .. } = &self.layers[il].ffn {
                let (e, k) = (cfg.n_expert, cfg.n_expert_used);
                l.sel = sc.rsel.to_u32()?[..m * k].to_vec();
                let logits = read(&sc.rlogits, m * e)?;
                let bias = bias.to_f32()?;
                for row in logits.chunks(e) {
                    let mut sc: Vec<f32> = row
                        .iter()
                        .zip(&bias)
                        .map(|(&x, &b)| 1.0 / (1.0 + (-x).exp()) + b)
                        .collect();
                    sc.sort_by(|a, b| b.total_cmp(a));
                    l.margin.push(sc[k - 1] - sc[k]);
                }
            }
            out.push(l);
        }
        let logits = read(&sc.logits, m * cfg.vocab)?;
        st.pos += m;
        Ok((out, logits))
    }

    /// [`Self::forward`] with the logits of every row.
    pub fn forward_all_logits(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        self.forward_inner(ctx, st, sc, tokens, Logits::All, None)
    }

    /// [`Self::forward`] committing after every trunk layer and reading its
    /// intermediates back.
    pub fn forward_traced(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
        trace: &mut Vec<LayerTrace>,
    ) -> Result<()> {
        self.forward_inner(ctx, st, sc, tokens, Logits::Last, Some(trace))
    }

    /// Feeds a whole prompt in runs of [`MAX_ROWS`] rows.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        let n = tokens.len();
        for (i, chunk) in tokens.chunks(PREFILL_ROWS).enumerate() {
            let last = (i + 1) * PREFILL_ROWS >= n;
            self.forward(ctx, st, sc, chunk, last)?;
        }
        Ok(())
    }

    fn forward_inner(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
        want_logits: Logits,
        trace: Option<&mut Vec<LayerTrace>>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let m = tokens.len();
        ensure!((1..=PREFILL_ROWS).contains(&m), "forward of {m} rows");
        let pos0 = st.pos;
        ensure!(
            pos0 + m <= st.capacity,
            "state capacity {} for position {}",
            st.capacity,
            pos0 + m
        );
        sc.stage_tokens(tokens)?;
        let Some(trace) = trace else {
            self.run_rows(ctx, st, sc, m, want_logits, |_| Ok(()))?;
            st.pos += m;
            return Ok(());
        };
        let mut pass = ctx.begin()?;
        k::embed(ctx, &pass, &self.embed, &sc.tokens, &sc.h, m)?;
        for il in 0..cfg.n_trunk {
            let l = &self.layers[il];
            let [hc_a, hc_f] = l.hc.as_ref().expect("trunk layer has HC");
            self.hc_pre(ctx, &pass, sc, hc_a, &l.attn_norm, m, None)?;
            pass.commit_wait()?;
            pass = ctx.begin()?;
            let attn_in = read(&sc.xn, m * cfg.hidden)?;
            self.attention(ctx, &pass, st, sc, &l.attn, il, pos0, m, None)?;
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
            self.hc_pre(ctx, &pass, sc, hc_f, &l.ffn_norm, m, None)?;
            pass.commit_wait()?;
            pass = ctx.begin()?;
            let ffn_in = read(&sc.xn, m * cfg.hidden)?;
            self.ffn(ctx, &pass, sc, &l.ffn, m)?;
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
            pass.commit_wait()?;
            pass = ctx.begin()?;
            trace.push(LayerTrace {
                attn_in,
                attn_out: read(&sc.attn_out, m * cfg.hidden)?,
                ffn_in,
                ffn_out: read(&sc.ffn_out, m * cfg.hidden)?,
                h: read(&sc.h, m * cfg.hc_dim())?,
            });
        }
        self.encode_head(ctx, &pass, sc, m, want_logits)?;
        pass.commit_wait()?;
        st.pos += m;
        Ok(())
    }

    /// Encodes one step of `m` rows at the state's position into `pass`,
    /// reading the token ids from `tokens` (U32, `m` words) on the GPU,
    /// without advancing the state.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Glm53State,
        sc: &Scratch,
        tokens: &Tensor,
        m: usize,
        logits: Logits,
        snap: Option<usize>,
    ) -> Result<()> {
        self.encode_layers(
            ctx,
            pass,
            st,
            sc,
            tokens,
            m,
            logits,
            snap,
            0..self.cfg.n_trunk,
        )
    }

    /// [`Self::encode`] of trunk layers `layers` only: the embedding when
    /// they start at layer 0, the final HC post and the head when they end
    /// at the last layer. Consecutive ranges in consecutive passes make one
    /// step.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_layers(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Glm53State,
        sc: &Scratch,
        tokens: &Tensor,
        m: usize,
        logits: Logits,
        snap: Option<usize>,
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
        ensure!(layers.end <= cfg.n_trunk, "layers {layers:?}");
        if layers.start == 0 {
            k::embed(ctx, pass, &self.embed, tokens, &sc.h, m)?;
            pass.memory_barrier()?;
        }
        for il in layers.clone() {
            let l = &self.layers[il];
            let [hc_a, hc_f] = l.hc.as_ref().expect("trunk layer has HC");
            // Each HC pre applies the HC post of the sublayer before it.
            let prev = (il > 0).then_some(&sc.ffn_out);
            self.hc_pre(ctx, pass, sc, hc_a, &l.attn_norm, m, prev)?;
            self.attention(ctx, pass, st, sc, &l.attn, il, pos0, m, snap)?;
            self.hc_pre(ctx, pass, sc, hc_f, &l.ffn_norm, m, Some(&sc.attn_out))?;
            self.ffn(ctx, pass, sc, &l.ffn, m)?;
        }
        if layers.end < cfg.n_trunk {
            return Ok(());
        }
        k::hc_post(ctx, pass, &sc.h, &sc.ffn_out, &sc.split, &sc.h, m, cfg.hidden)?;
        pass.memory_barrier()?;
        self.encode_head(ctx, pass, sc, m, logits)
    }

    /// Runs a step of the `m` rows staged in `sc.tokens` at the state's
    /// position (without advancing it), `tail` encoded at its end, and
    /// waits. A run of prompt rows takes a pass a layer: its dispatches run
    /// for milliseconds each, and the kernel driver times a command buffer
    /// from its submission (about two seconds), so one batched command
    /// buffer of a 1 024-row run, queued behind others or another client's
    /// work, ran out of time. At most two layers' passes are in flight.
    pub fn run_rows(
        &self,
        ctx: &MetalContext,
        st: &Glm53State,
        sc: &Scratch,
        m: usize,
        logits: Logits,
        tail: impl FnOnce(&ComputePass<'_>) -> Result<()>,
    ) -> Result<()> {
        let n = self.cfg.n_trunk;
        if m <= MAX_ROWS {
            return self.run_step(ctx, st, sc, &sc.tokens, m, logits, None, tail);
        }
        let mut pending: std::collections::VecDeque<crate::metal::PendingPass<'_>> =
            std::collections::VecDeque::new();
        let mut tail = Some(tail);
        for il in 0..n {
            let pass = self.begin(ctx)?;
            self.encode_layers(
                ctx,
                &pass,
                st,
                sc,
                &sc.tokens,
                m,
                logits,
                None,
                il..il + 1,
            )?;
            // At most two passes in flight: wait for the oldest before
            // submitting a third.
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

    /// Runs a step of `m` <= [`MAX_ROWS`] rows synchronously (the verify
    /// pass, `forward` and `run_rows`; tokens read from `tokens` on the
    /// GPU), `tail` encoded at its end, and waits. The engine's ordinary
    /// decode step (`encode_decode_step`) stays one pass: the generation
    /// loop encodes it while the step before runs. Here the step goes to
    /// the GPU in four passes, the first after two layers are encoded, so
    /// the GPU works while the host encodes the rest: in a speculative loop
    /// the host waits for every pass and its CPU slows down, and one pass a
    /// step left the GPU idle for its whole encoding. Verify pass on code
    /// (2026-11-14): one pass 39.1-39.2 ms, halves 37.7-37.9, thirds
    /// 36.9-37.6, layers 0-2 then thirds 36.86-36.90.
    ///
    /// Not transactional: when encoding a later pass (or `tail`) fails, the
    /// passes before it have run (they are waited for before the error
    /// returns), so the recurrent state and caches are partly advanced
    /// while `st.pos` is not. Discard the state or restore a snapshot taken
    /// before the step.
    #[allow(clippy::too_many_arguments)]
    pub fn run_step(
        &self,
        ctx: &MetalContext,
        st: &Glm53State,
        sc: &Scratch,
        tokens: &Tensor,
        m: usize,
        logits: Logits,
        snap: Option<usize>,
        tail: impl FnOnce(&ComputePass<'_>) -> Result<()>,
    ) -> Result<()> {
        let n = self.cfg.n_trunk;
        let cuts = [0, 2.min(n), n / 3, 2 * n / 3];
        let mut pending = Vec::with_capacity(cuts.len());
        for w in cuts.windows(2) {
            if w[1] > w[0] {
                let pass = self.begin(ctx)?;
                self.encode_layers(
                    ctx,
                    &pass,
                    st,
                    sc,
                    tokens,
                    m,
                    logits,
                    snap,
                    w[0]..w[1],
                )?;
                pending.push(pass.commit()?);
            }
        }
        let pass = self.begin(ctx)?;
        let last = cuts[cuts.len() - 1].max(cuts[1]);
        self.encode_layers(ctx, &pass, st, sc, tokens, m, logits, snap, last..n)?;
        tail(&pass)?;
        let r = pass.commit_wait();
        for p in pending {
            p.wait()?;
        }
        r
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
        // The stream mean is kept (the MTP head reads it) even without logits.
        let (first, rows) = if logits == Logits::All { (0, m) } else { (m - 1, 1) };
        let h = sc.h.view(first * cfg.hc_dim(), &[rows * cfg.hc_dim()])?;
        k::hc_mean(ctx, pass, &h, &sc.hidden, rows, cfg.hidden)?;
        pass.memory_barrier()?;
        if logits == Logits::None {
            return Ok(());
        }
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

    /// One MTP step (ds4 `glm_graph_mtp_step`) at main position `p`: the
    /// token at `p + 1` (`token`, U32 `[1]`) and the main model's stream mean at
    /// `p` (`hidden`, 4096 f32) through the nextn block, whose latent row `p`
    /// it writes; with `draft`, the shared head's greedy token for `p + 2`
    /// goes to `sc.mtp_io[1]`. The head attends its own rows from
    /// `st.mtp_min` (at most the dense window: ds4 attends the whole run).
    /// Host-writes the row list, so the GPU must be idle on the scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mtp(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Glm53State,
        sc: &Scratch,
        token: &Tensor,
        hidden: &Tensor,
        p: usize,
        draft: bool,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let il = cfg.mtp_layer();
        let l = &self.layers[il];
        let nx = &self.nextn;
        let limit = cfg.indexer_selected_limit();
        let s = limit.next_multiple_of(8);
        let first = st.mtp_min.unwrap_or(p).max((p + 1).saturating_sub(limit));
        if draft {
            let mut rows = vec![u32::MAX; s];
            for (slot, row) in (first..=p).enumerate() {
                rows[slot] = row as u32;
            }
            write_words(&sc.mtp_sel, &rows)?;
        }
        k::embed(ctx, pass, &self.embed, token, &sc.mtp_emb, 1)?;
        pass.memory_barrier()?;
        let e = sc.mtp_emb.view(0, &[cfg.hidden])?;
        let cat_e = sc.mtp_cat.view(0, &[cfg.hidden])?;
        let cat_h = sc.mtp_cat.view(cfg.hidden, &[cfg.hidden])?;
        k::rmsnorm(ctx, pass, &e, Some(&nx.enorm), &cat_e, 1, cfg.hidden, cfg.rms_eps)?;
        k::rmsnorm(
            ctx,
            pass,
            hidden,
            Some(&nx.hnorm),
            &cat_h,
            1,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        k::gemv(ctx, pass, &nx.eh_proj, &sc.mtp_cat, &sc.mtp_x, 1)?;
        pass.memory_barrier()?;
        k::rmsnorm(
            ctx,
            pass,
            &sc.mtp_x,
            Some(&l.attn_norm),
            &sc.xn,
            1,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        if !draft {
            // Catching up only publishes the latent row (ds4 does the same).
            let (AttnW::Dsa(w), LayerState::Dsa { latent, .. }) =
                (&l.attn, &st.layers[il])
            else {
                anyhow::bail!("the MTP block's attention is not DSA");
            };
            k::gemv(ctx, pass, &w.kv_a, &sc.xn, &sc.kv_raw, 1)?;
            pass.memory_barrier()?;
            k::kv_store(
                ctx,
                pass,
                &sc.kv_raw,
                &w.kv_a_norm,
                latent,
                p,
                1,
                cfg.rms_eps,
            )?;
            return pass.memory_barrier();
        }
        self.attention(ctx, pass, st, sc, &l.attn, il, p, 1, None)?;
        k::add(ctx, pass, &sc.mtp_x, &sc.attn_out, &sc.mtp_h1, cfg.hidden)?;
        pass.memory_barrier()?;
        k::rmsnorm(
            ctx,
            pass,
            &sc.mtp_h1,
            Some(&l.ffn_norm),
            &sc.xn,
            1,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        self.ffn(ctx, pass, sc, &l.ffn, 1)?;
        k::add(ctx, pass, &sc.mtp_h1, &sc.ffn_out, &sc.mtp_x, cfg.hidden)?;
        pass.memory_barrier()?;
        k::rmsnorm(
            ctx,
            pass,
            &sc.mtp_x,
            Some(&nx.shared_head_norm),
            &sc.hidden_norm,
            1,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        let head = self.draft_head.as_ref().unwrap_or(&self.output);
        k::gemv(ctx, pass, head, &sc.hidden_norm, &sc.logits, 1)?;
        pass.memory_barrier()?;
        let out = sc.mtp_io.view(1, &[1])?;
        crate::kernels::sample::sample_f32(
            ctx,
            pass,
            &sc.logits_last,
            &sc.sampler,
            &crate::kernels::sample::SamplingParams::greedy(),
            0,
            &out,
        )?;
        k::top_prob(ctx, pass, &sc.logits_last, &sc.mtp_prob, cfg.vocab)?;
        pass.memory_barrier()
    }

    /// Projections of one input: the GEMV (several weights in one dispatch)
    /// up to [`MAX_ROWS`] rows, the matrix-unit GEMM beyond.
    /// `xb`: the input's bfloat copy ([`Self::bf16_input`]) when there is
    /// one, which the GEMMs over Q8_0 and Q4_K weights take.
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
            return k::gemv_sets(ctx, pass, sets, x, m);
        }
        for &(w, y) in sets {
            match xb {
                Some(xb) if k::gemm_bx_takes(w) => k::gemm_bx(ctx, pass, w, xb, y, m)?,
                _ => k::gemm(ctx, pass, w, x, y, m)?,
            }
        }
        Ok(())
    }

    /// For a prompt run (more than [`MAX_ROWS`] rows): `x`'s `m` rows of
    /// `width` values rounded to bfloat into `sc.xbf`, its own dependency
    /// level, for the projections that read `x` next (the GEMMs round their
    /// input to bfloat anyway; once here instead of in every column tile).
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
        k::to_bf16(ctx, pass, x, &sc.xbf, m * width)?;
        pass.memory_barrier()?;
        Ok(Some(&sc.xbf))
    }

    /// The pass a step is encoded into: concurrent, its dependency levels
    /// separated by barriers; `LILY_GLM53_SERIAL=1` makes every dispatch its
    /// own level (A/B and debugging).
    pub fn begin<'c>(&self, ctx: &'c MetalContext) -> Result<ComputePass<'c>> {
        static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *SERIAL.get_or_init(|| std::env::var_os("LILY_GLM53_SERIAL").is_some()) {
            ctx.begin()
        } else {
            ctx.begin_concurrent()
        }
    }

    /// HC pre of a sublayer into `sc.split` and `sc.xn`, first applying the
    /// previous sublayer's HC post when `post` gives its output (the post
    /// weights are still in `sc.split`).
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

    /// The attention sublayer from `sc.xn` to `sc.attn_out`, in dependency
    /// levels: dispatches between two barriers are independent and may run
    /// side by side on a concurrent pass.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Glm53State,
        sc: &Scratch,
        a: &AttnW,
        il: usize,
        pos0: usize,
        m: usize,
        snap: Option<usize>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        match (a, &st.layers[il]) {
            (AttnW::Kda(w), LayerState::Kda { conv, s, conv_alt, s_alt }) => {
                let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
                self.proj(ctx, pass, &[(&w.q, &sc.kq), (&w.k, &sc.kk)], &sc.xn, xb, m)?;
                self.proj(
                    ctx,
                    pass,
                    &[
                        (&w.v, &sc.kv),
                        (&w.f_a, &sc.klow),
                        (&w.beta, &sc.kbeta),
                        (&w.g_a, &sc.klow2),
                    ],
                    &sc.xn,
                    xb,
                    m,
                )?;
                pass.memory_barrier()?;
                // f_b and g_b read different inputs: one dispatch each.
                self.proj(ctx, pass, &[(&w.f_b, &sc.kgate)], &sc.klow, None, m)?;
                self.proj(ctx, pass, &[(&w.g_b, &sc.kog)], &sc.klow2, None, m)?;
                pass.memory_barrier()?;
                let p = KdaParams {
                    q_conv: &w.q_conv,
                    k_conv: &w.k_conv,
                    v_conv: &w.v_conv,
                    a_log: &w.a_log,
                    dt_bias: &w.dt_bias,
                    o_norm: &w.o_norm,
                    heads: cfg.kda_heads,
                    lower_bound: cfg.kda_gate_lower_bound,
                    eps: cfg.rms_eps,
                };
                let x = KdaInputs {
                    q: &sc.kq,
                    k: &sc.kk,
                    v: &sc.kv,
                    gate: &sc.kgate,
                    beta: &sc.kbeta,
                    out_gate: &sc.kog,
                    edges: Some(&sc.kedges),
                };
                let snap = snap.map(|row| k::KdaSnapshot {
                    row,
                    conv: conv_alt,
                    state: s_alt,
                });
                k::kda(ctx, pass, &p, &x, conv, s, &sc.kout, m, snap)?;
                pass.memory_barrier()?;
                let pw = cfg.kda_projection();
                let xb = self.bf16_input(ctx, pass, sc, &sc.kout, pw, m)?;
                self.proj(ctx, pass, &[(&w.out, &sc.attn_out)], &sc.kout, xb, m)?;
                pass.memory_barrier()
            }
            (AttnW::Dsa(w), LayerState::Dsa { latent, pooled, tail_k, tail_g }) => {
                let kv = cfg.kv_lora;
                let hq = cfg.n_head * cfg.key_dim;
                let limit = cfg.indexer_selected_limit();
                let sparse = w.indexer.is_some() && pos0 + m > limit;
                // Level 1: everything that reads xn.
                let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
                self.proj(
                    ctx,
                    pass,
                    &[(&w.q_a, &sc.qr), (&w.kv_a, &sc.kv_raw)],
                    &sc.xn,
                    xb,
                    m,
                )?;
                if let Some(ix) = &w.indexer {
                    let mut sets = vec![(&ix.k, &sc.ik), (&ix.gate, &sc.ig)];
                    if sparse {
                        sets.push((&ix.proj, &sc.iw));
                    }
                    self.proj(ctx, pass, &sets, &sc.xn, xb, m)?;
                }
                pass.memory_barrier()?;
                // Level 2: the norms, the latent store, the pooled keys.
                k::rmsnorm(
                    ctx,
                    pass,
                    &sc.qr,
                    Some(&w.q_a_norm),
                    &sc.qr,
                    m,
                    cfg.q_lora,
                    cfg.rms_eps,
                )?;
                k::kv_store(
                    ctx,
                    pass,
                    &sc.kv_raw,
                    &w.kv_a_norm,
                    latent,
                    pos0,
                    m,
                    cfg.rms_eps,
                )?;
                if let Some(ix) = &w.indexer {
                    let pp = PoolParams {
                        norm_w: &ix.norm_w,
                        norm_b: &ix.norm_b,
                        ape: &ix.ape,
                    };
                    k::indexer_pool_update(
                        ctx, pass, &pp, &sc.ik, &sc.ig, pooled, tail_k, tail_g, pos0, m,
                    )?;
                }
                pass.memory_barrier()?;
                // Level 3: the queries.
                let xb = self.bf16_input(ctx, pass, sc, &sc.qr, cfg.q_lora, m)?;
                self.proj(ctx, pass, &[(&w.q_b, &sc.q)], &sc.qr, xb, m)?;
                if let (true, Some(ix)) = (sparse, &w.indexer) {
                    self.proj(ctx, pass, &[(&ix.q_b, &sc.iq)], &sc.qr, xb, m)?;
                }
                pass.memory_barrier()?;
                // Level 4: absorbed queries, pool scores.
                if m <= MAX_ROWS {
                    k::gemv_with(
                        ctx,
                        pass,
                        &w.k_b,
                        &sc.q,
                        &sc.qk_low,
                        m,
                        GemvLayout {
                            x_group: kv,
                            x_row_stride: hq,
                            y_row_stride: cfg.n_head * kv,
                        },
                    )?;
                } else {
                    // Per head: qk_low[i][h] (512) = k_b[h] (512 x 256) . q[i][h].
                    let l = k::GemmLayout {
                        ldx: hq,
                        ldy: cfg.n_head * kv,
                        batches: cfg.n_head,
                        x_batch: cfg.key_dim,
                        w_batch: kv,
                        y_batch: kv,
                    };
                    k::gemm_with(ctx, pass, &w.k_b, &sc.q, &sc.qk_low, m, kv, l)?;
                }
                let rows = if sparse {
                    let scale =
                        1.0 / ((cfg.indexer_dim * cfg.indexer_heads) as f32).sqrt();
                    k::indexer_scores(
                        ctx,
                        pass,
                        &sc.iq,
                        &sc.iw,
                        pooled,
                        &st.pool_scores,
                        m,
                        st.pool_cap,
                        pos0,
                        scale,
                    )?;
                    pass.memory_barrier()?;
                    let s = limit.next_multiple_of(8);
                    k::indexer_topk(
                        ctx,
                        pass,
                        &st.pool_scores,
                        &sc.sel,
                        &sc.sel_count,
                        m,
                        st.pool_cap,
                        s,
                        pos0,
                        cfg.indexer_top_k / cfg.indexer_pool,
                    )?;
                    Rows::Selected { sel: &sc.sel, count: s }
                } else if w.indexer.is_none() {
                    // The MTP head: the rows of its current run
                    // (`encode_mtp` writes them into mtp_sel).
                    Rows::Selected {
                        sel: &sc.mtp_sel,
                        count: limit.next_multiple_of(8),
                    }
                } else {
                    Rows::Causal { pos0 }
                };
                pass.memory_barrier()?;
                let s = match rows {
                    Rows::Causal { .. } => (pos0 + m).next_multiple_of(8),
                    Rows::Selected { count, .. } => count,
                };
                k::dsa_attend(
                    ctx,
                    pass,
                    &sc.qk_low,
                    latent,
                    rows,
                    &sc.scores,
                    &sc.denom,
                    &sc.lat,
                    &sc.dsa_partial,
                    m,
                    s,
                    1.0 / (cfg.key_dim as f32).sqrt(),
                )?;
                pass.memory_barrier()?;
                if m <= MAX_ROWS {
                    k::gemv_with(
                        ctx,
                        pass,
                        &w.v_b,
                        &sc.lat,
                        &sc.heads,
                        m,
                        GemvLayout {
                            x_group: cfg.value_dim,
                            x_row_stride: cfg.n_head * kv,
                            y_row_stride: cfg.n_head * cfg.value_dim,
                        },
                    )?;
                } else {
                    // Per head: heads[i][h] (256) = v_b[h] (256 x 512) . lat[i][h].
                    let l = k::GemmLayout {
                        ldx: cfg.n_head * kv,
                        ldy: cfg.n_head * cfg.value_dim,
                        batches: cfg.n_head,
                        x_batch: kv,
                        w_batch: cfg.value_dim,
                        y_batch: cfg.value_dim,
                    };
                    k::gemm_with(
                        ctx,
                        pass,
                        &w.v_b,
                        &sc.lat,
                        &sc.heads,
                        m,
                        cfg.value_dim,
                        l,
                    )?;
                }
                pass.memory_barrier()?;
                let hv = cfg.n_head * cfg.value_dim;
                let xb = self.bf16_input(ctx, pass, sc, &sc.heads, hv, m)?;
                self.proj(ctx, pass, &[(&w.out, &sc.attn_out)], &sc.heads, xb, m)?;
                pass.memory_barrier()
            }
            _ => anyhow::bail!("layer {il}: weights and state disagree"),
        }
    }

    /// The FFN sublayer from `sc.xn` to `sc.ffn_out`, in dependency levels.
    fn ffn(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        f: &FfnW,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let lim = cfg.swiglu_limit;
        match f {
            FfnW::Dense { gate, up, down } => {
                let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
                self.proj(ctx, pass, &[(gate, &sc.fg), (up, &sc.fu)], &sc.xn, xb, m)?;
                pass.memory_barrier()?;
                k::swiglu(ctx, pass, &sc.fg, &sc.fu, &sc.fmid, m * cfg.ff_dense, lim)?;
                pass.memory_barrier()?;
                let xb = self.bf16_input(ctx, pass, sc, &sc.fmid, cfg.ff_dense, m)?;
                self.proj(ctx, pass, &[(down, &sc.ffn_out)], &sc.fmid, xb, m)?;
                pass.memory_barrier()
            }
            FfnW::Moe { router, bias, gate, up, down, sh_gate, sh_up, sh_down } => {
                let used = cfg.n_expert_used;
                let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
                self.proj(ctx, pass, &[(router, &sc.rlogits)], &sc.xn, None, m)?;
                self.proj(
                    ctx,
                    pass,
                    &[(sh_gate, &sc.fg), (sh_up, &sc.fu)],
                    &sc.xn,
                    xb,
                    m,
                )?;
                pass.memory_barrier()?;
                k::router_topk(
                    ctx,
                    pass,
                    &sc.rlogits,
                    bias,
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
                    // A prompt run: shared expert first, then the routed
                    // experts grouped by expert: an expert of at most four
                    // rows (one tile) by register dot products, the others
                    // on the matrix units (`moe_grouped_dot`).
                    let xb =
                        self.bf16_input(ctx, pass, sc, &sc.fmid, cfg.ff_expert, m)?;
                    self.proj(ctx, pass, &[(sh_down, &sc.ffn_out)], &sc.fmid, xb, m)?;
                    pass.memory_barrier()?;
                    k::moe_grouped_dot(
                        ctx,
                        pass,
                        gate,
                        up,
                        down,
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
                    gate,
                    up,
                    cfg.ff_expert,
                    &sc.xn,
                    &sc.rsel,
                    &sc.rw,
                    &sc.moe_mid,
                    m,
                    used,
                    lim,
                )?;
                self.proj(ctx, pass, &[(sh_down, &sc.ffn_out)], &sc.fmid, None, m)?;
                pass.memory_barrier()?;
                // Routed sum plus the shared expert's output.
                k::moe_down(
                    ctx,
                    pass,
                    down,
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
        }
    }
}

fn read(t: &Tensor, n: usize) -> Result<Vec<f32>> {
    Ok(t.to_f32()?[..n].to_vec())
}

#[cfg(test)]
#[path = "../../tests/unit/glm53/model.rs"]
mod tests;
