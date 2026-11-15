//! Qwen3.6 on the GPU: weights mapped read-only from the GGUF, the
//! per-session caches and recurrent state, and the forward pass over up to
//! [`Qwen36Model::prefill_rows`] consecutive tokens at a time (decode is
//! one row).
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

/// Rows a prompt step takes at most (projections on the matrix units):
/// [`Qwen36Model::prefill_rows`] is this or [`PREFILL_ROWS_LIMITED`].
pub const PREFILL_ROWS: usize = 2048;

/// Rows a prompt step takes under `--memory-limit-gb`: the scratch of 2048
/// rows is ~0.4 GB more, which the 32 and 64 GB settings' contexts need.
/// 2048 rows a step: 4K-token prefill 3530-3540 -> 3780-3880 tok/s
/// (lily-bench, 2026-11-15; each expert's grouped GEMM tiles about twice
/// as full).
pub const PREFILL_ROWS_LIMITED: usize = 1024;

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
    /// The multi-token-prediction head, when loaded ([`Self::load_mtp`]).
    pub mtp: Option<MtpW>,
    /// `--memory-limit-gb`: what the server's session cache budget is
    /// capped to leave (the 32 GB tier runs with 24).
    pub memory_limit: Option<u64>,
    /// The longest request context served: the file's, the attention
    /// kernel's, or what the memory limit fits ([`Self::fit_context`]).
    pub context_cap: usize,
    /// Rows a prompt step takes (and its scratch holds).
    pub prefill_rows: usize,
}

/// Cache rows a session needs past its request's tokens: a verify pass's
/// draft row and the MTP head's row after it.
pub const LOOKAHEAD: usize = 4;

/// The most cache rows a session holds for a context of `context` tokens
/// (a multiple of 4, at most the attention kernel's limit).
pub fn capacity_cap(context: usize) -> usize {
    (context + LOOKAHEAD).min(attn::MAX_SEQ) / 4 * 4
}

/// The MTP block (`blk.<n_layer>` of the `mtp-*.gguf` beside the model): a
/// full-attention block with its MoE, fed by `eh_proj` of the next token's
/// normed embedding and the trunk's normed hidden state; its own head norm,
/// the trunk's embedding and LM head (the file's copies of both are the
/// same weights and are not mapped).
pub struct MtpW {
    layer: LayerW,
    eh_proj: QWeight,
    enorm: Tensor,
    hnorm: Tensor,
    head_norm: Tensor,
    mapped: Mapped,
    pub path: std::path::PathBuf,
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
        let context_cap = cfg.context_length.min(attn::MAX_SEQ);
        Ok(Self {
            cfg,
            gguf,
            mapped,
            layers,
            embed,
            output,
            output_norm,
            kv_format: KvFormat::Bf16,
            mtp: None,
            memory_limit: None,
            context_cap,
            prefill_rows: PREFILL_ROWS,
        })
    }

    /// The MTP file for the model at `path`: `LILY_QWEN36_MTP`, else the one
    /// `mtp-*.gguf` in its directory of the same architecture and width.
    pub fn find_mtp(&self, path: &Path) -> Result<Option<std::path::PathBuf>> {
        if let Some(p) = std::env::var_os("LILY_QWEN36_MTP") {
            return Ok(Some(p.into()));
        }
        let Some(dir) = path.parent() else { return Ok(None) };
        let mut found = Vec::new();
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !(name.starts_with("mtp-") && name.ends_with(".gguf")) {
                continue;
            }
            let g = Gguf::open(&p)?;
            let arch = super::config::ARCH;
            if g.architecture()? == arch
                && g.u64(&format!("{arch}.embedding_length"))? as usize
                    == self.cfg.hidden
            {
                found.push(p);
            }
        }
        found.sort();
        Ok(found.into_iter().next())
    }

    /// Maps the MTP block of `path` (its `blk.<n_layer>` tensors only).
    pub fn load_mtp(&mut self, ctx: &MetalContext, path: &Path) -> Result<()> {
        let g = Gguf::open(path)?;
        let arch = super::config::ARCH;
        ensure!(g.architecture()? == arch, "{} is not a {arch} GGUF", path.display());
        ensure!(
            g.u64(&format!("{arch}.nextn_predict_layers"))? == 1,
            "{} has no single MTP layer",
            path.display()
        );
        let il = self.cfg.n_layer;
        let h = self.cfg.hidden;
        let mut specs = super::schema::block(&self.cfg, il, false);
        let f32_ = &[crate::gguf::GgmlType::F32][..];
        let spec = |n: &str, d: &[u64], t| super::schema::Spec {
            name: format!("blk.{il}.nextn.{n}.weight"),
            dims: d.to_vec(),
            types: t,
        };
        specs.push(spec("eh_proj", &[2 * h as u64, h as u64], super::schema::DENSE));
        for n in ["enorm", "hnorm", "shared_head_norm"] {
            specs.push(spec(n, &[h as u64], f32_));
        }
        super::schema::validate_specs(
            &g,
            &specs,
            &super::schema::fused_groups(il, false),
        )?;
        let prefix = format!("blk.{il}.");
        let mapped = Mapped::map_only(ctx, &g, |t| t.name.starts_with(&prefix))?;
        let n = |s: &str| format!("blk.{il}.nextn.{s}.weight");
        self.mtp = Some(MtpW {
            layer: load_block(&mapped, il, false)?,
            eh_proj: mapped.q(&n("eh_proj"))?,
            enorm: mapped.f32(&n("enorm"))?,
            hnorm: mapped.f32(&n("hnorm"))?,
            head_norm: mapped.f32(&n("shared_head_norm"))?,
            mapped,
            path: path.to_path_buf(),
        });
        Ok(())
    }

    /// The buffers holding the mapped file.
    pub fn weight_buffers(&self) -> Vec<Buffer> {
        let mut b = self.mapped.buffers();
        if let Some(m) = &self.mtp {
            b.extend(m.mapped.buffers());
        }
        b
    }

    pub fn new_state(
        &self,
        ctx: &MetalContext,
        capacity: usize,
    ) -> Result<Qwen36State> {
        Qwen36State::new(
            ctx,
            &self.cfg,
            self.kv_format,
            capacity,
            capacity_cap(self.context_cap),
            self.mtp.is_some(),
        )
    }

    /// GPU bytes one session holds with caches for `capacity` tokens and
    /// `checkpoints` snapshots: the K/V rows (the trunk's attention layers'
    /// and the MTP head's), the GDN windows and states and their verify
    /// alternates, the MTP pending row; a snapshot is the GDN windows and
    /// states and the pending row. Matches `Qwen36State::bytes` and
    /// `Qwen36Snapshot::bytes` (checked against a real state at warm-up).
    pub fn session_bytes(&self, capacity: usize, checkpoints: usize) -> u64 {
        let c = &self.cfg;
        let cap = capacity.next_multiple_of(4) as u64;
        let mtp = self.mtp.is_some();
        let kv_layers = (c.attn_layers() + usize::from(mtp)) as u64;
        let row = self.kv_format.row_bytes(c.head_dim) as u64;
        let kv = 2 * kv_layers * c.n_head_kv as u64 * cap * row;
        let gdn_layers = c.recurrent.iter().filter(|r| **r).count() as u64;
        let gdn = gdn_layers
            * 4
            * (3 * c.gdn_qkv() + c.gdn_v_heads * c.gdn_dim * c.gdn_dim) as u64;
        let pending = if mtp { 4 * c.hidden as u64 } else { 0 };
        kv + 2 * gdn + pending + checkpoints as u64 * (gdn + pending)
    }

    /// Under `--memory-limit-gb`: the longest context whose session (with
    /// `checkpoints` snapshots, growth to its full capacity and the
    /// speculative lookahead) fits what the limit leaves after `allocated`
    /// bytes (the weights and the scratch); `context_cap` becomes it, so the
    /// server caps `--max-seq` there and refuses longer requests.
    pub fn fit_context(
        &mut self,
        limit: u64,
        allocated: u64,
        checkpoints: usize,
    ) -> Result<usize> {
        let room = limit.saturating_sub(allocated);
        let per_token =
            self.session_bytes(4, 0).saturating_sub(self.session_bytes(0, 0)) / 4;
        let fixed = self.session_bytes(0, checkpoints);
        let rows = room.saturating_sub(fixed) / per_token.max(1);
        let context = (rows as usize).saturating_sub(LOOKAHEAD) / 4 * 4;
        let context = context.min(self.cfg.context_length).min(attn::MAX_SEQ);
        ensure!(
            context >= 4096,
            "--memory-limit-gb {:.1} leaves {:.2} GB after {:.1} GB of weights and scratch: a context \
             of {context} tokens (lily wants at least 4096)",
            limit as f64 / (1u64 << 30) as f64,
            room as f64 / 1e9,
            allocated as f64 / 1e9
        );
        self.context_cap = context;
        Ok(context)
    }

    pub fn new_scratch(&self, ctx: &MetalContext) -> Result<Scratch> {
        Scratch::new(ctx, &self.cfg, self.prefill_rows)
    }
}

/// One layer's recurrent state or caches.
pub enum LayerState {
    /// `conv_alt`/`s_alt` receive the state after a verify pass's first row,
    /// which becomes current when its draft is rejected.
    Gdn {
        conv: Tensor,
        s: Tensor,
        conv_alt: Tensor,
        s_alt: Tensor,
    },
    Attn {
        k: KvCache,
        v: KvCache,
    },
}

/// The MTP head's per-session state: its K/V caches (row `p` holds the pair
/// of the trunk's hidden state at `p - 1` and the token at `p`, llama.cpp's
/// layout), the trunk's normed hidden row the next pair takes
/// (`pending_h`, the row at `pos - 1`), and the next row to write.
pub struct MtpState {
    pub k: KvCache,
    pub v: KvCache,
    pub pending_h: Tensor,
    pub pos: usize,
}

/// A session: caches and recurrent state at position `pos`.
pub struct Qwen36State {
    pub pos: usize,
    pub capacity: usize,
    /// The most rows the caches grow to (the model's context cap plus the
    /// lookahead, at most the attention kernel's limit).
    pub max_capacity: usize,
    pub layers: Vec<LayerState>,
    pub mtp: Option<MtpState>,
    /// A verify pass that fed rows from this position and the draws of its
    /// rows, until `finish_speculation` decides what stays.
    pub(crate) spec: Option<(usize, usize, Vec<u32>)>,
}

impl Qwen36State {
    pub fn new(
        ctx: &MetalContext,
        cfg: &Config,
        format: KvFormat,
        capacity: usize,
        max_capacity: usize,
        mtp: bool,
    ) -> Result<Self> {
        ensure!(
            max_capacity.is_multiple_of(4) && max_capacity <= attn::MAX_SEQ,
            "a cache of at most {max_capacity} rows"
        );
        ensure!(
            capacity <= max_capacity,
            "a session of {capacity} tokens exceeds the {max_capacity} rows this model serves"
        );
        let capacity = capacity.next_multiple_of(4).max(4);
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
                    LayerState::Gdn {
                        conv: conv()?,
                        s: s()?,
                        conv_alt: conv()?,
                        s_alt: s()?,
                    }
                } else {
                    let c = || {
                        KvCache::zeros(
                            ctx,
                            format,
                            cfg.n_head_kv,
                            capacity,
                            cfg.head_dim,
                        )
                    };
                    LayerState::Attn { k: c()?, v: c()? }
                })
            })
            .collect::<Result<_>>()?;
        let mtp = if mtp {
            let c =
                || KvCache::zeros(ctx, format, cfg.n_head_kv, capacity, cfg.head_dim);
            Some(MtpState {
                k: c()?,
                v: c()?,
                pending_h: Tensor::zeros(ctx, &[cfg.hidden], DType::F32)?,
                pos: 0,
            })
        } else {
            None
        };
        Ok(Self { pos: 0, capacity, max_capacity, layers, mtp, spec: None })
    }

    /// Zeroes the recurrent state and rewinds (GPU idle).
    pub fn reset(&mut self) {
        for l in &self.layers {
            if let LayerState::Gdn { conv, s, .. } = l {
                conv.zero_fill();
                s.zero_fill();
            }
        }
        if let Some(m) = &mut self.mtp {
            m.pending_h.zero_fill();
            m.pos = 0;
        }
        self.pos = 0;
        self.spec = None;
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

/// Per-step intermediates, for up to [`Qwen36Model::prefill_rows`] rows.
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
    pub hidden_norm: Tensor,
    // MTP
    /// `[rows + 1][hidden]`: a prompt run's MTP inputs, row 0 the
    /// session's pending row, row `i + 1` the run's row `i` normed.
    hbuf: Tensor,
    /// `[rows][2 hidden]`: normed embedding | normed hidden.
    cat: Tensor,
    /// The MTP head's normed last row.
    mtp_hn: Tensor,
    /// U32[2]: the MTP step's input token (0) and its greedy draft (1).
    pub mtp_io: Tensor,
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
    pub(crate) fn new(ctx: &MetalContext, cfg: &Config, rows: usize) -> Result<Self> {
        let r = rows;
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
        let widest = [2 * h, inner, hq, cfg.ff_shared].into_iter().max().unwrap_or(h);
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
            moe_group: g::MoeGroupScratch::new(
                ctx,
                r * used,
                cfg.n_expert,
                cfg.ff_expert,
                h,
            )?,
            sfg: f(r * cfg.ff_shared)?,
            sfu: f(r * cfg.ff_shared)?,
            sfm: f(r * cfg.ff_shared)?,
            sout: f(r * h)?,
            hidden_norm: f(MAX_ROWS * h)?,
            hbuf: f((r + 1) * h)?,
            cat: f(r * 2 * h)?,
            mtp_hn: f(h)?,
            mtp_io: u(2)?,
            logits,
            sampler: SamplerScratch::new(ctx, cfg.vocab)?,
            next_token: u(2)?,
            draws: u(MAX_ROWS)?,
            spec_tokens: u(MAX_ROWS)?,
        })
    }

    /// Stages a run's token ids (at most the scratch's rows; checked against the
    /// vocabulary by the callers, [`check_tokens`]).
    pub(crate) fn stage_tokens(&self, tokens: &[u32]) -> Result<()> {
        crate::glm53::model::write_words(&self.tokens, tokens)
    }

    /// The sampler's buffers.
    pub fn sampler(&self) -> &SamplerScratch {
        &self.sampler
    }
}

/// Refuses token ids outside the vocabulary before they reach the GPU (the
/// embedding kernel adds `id * row_bytes` to the table's address).
pub fn check_tokens(tokens: &[u32], vocab: usize) -> Result<()> {
    if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
        anyhow::bail!("token id {t} is outside the vocabulary of {vocab}");
    }
    Ok(())
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

    /// Feeds `tokens` (at most [`Self::prefill_rows`]) at the state's position,
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
        ensure!((1..=self.prefill_rows).contains(&m), "forward of {m} rows");
        ensure!(st.pos + m <= st.capacity, "state capacity {}", st.capacity);
        check_tokens(tokens, self.cfg.vocab)?;
        sc.stage_tokens(tokens)?;
        self.run_rows(ctx, st, sc, m, logits, None, |_| Ok(()))?;
        st.pos += m;
        Ok(())
    }

    /// Feeds a whole prompt in runs of [`Self::prefill_rows`] rows, the last
    /// row's logits at the end.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        st: &mut Qwen36State,
        sc: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        let n = tokens.len();
        for (i, chunk) in tokens.chunks(self.prefill_rows).enumerate() {
            let last = (i + 1) * self.prefill_rows >= n;
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
            self.encode_layers(
                ctx,
                &pass,
                st,
                sc,
                &sc.tokens,
                m,
                logits,
                snap,
                start..end,
            )?;
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
        ensure!((1..=self.prefill_rows).contains(&m), "step of {m} rows");
        ensure!(logits != Logits::All || m <= MAX_ROWS, "logits of {m} rows");
        ensure!(st.pos + m <= st.capacity, "state capacity {}", st.capacity);
        ensure!(layers.end <= cfg.n_layer, "layers {layers:?}");
        if layers.start == 0 {
            k::embed(ctx, pass, &self.embed, tokens, &sc.x, m)?;
            pass.memory_barrier()?;
        }
        for il in layers.clone() {
            // A block's FFN residual also norms the next block's input.
            let fuse = self.fuse_norm();
            let next =
                (fuse && il + 1 < layers.end).then(|| &self.layers[il + 1].attn_norm);
            self.encode_block(
                ctx,
                pass,
                sc,
                &self.layers[il],
                &st.layers[il],
                st.pos,
                snap,
                m,
                fuse && il > layers.start,
                next,
            )?;
        }
        if layers.end < cfg.n_layer {
            return Ok(());
        }
        self.encode_head(ctx, pass, sc, m, logits)
    }

    /// One block on `sc.x`'s `m` rows at positions `pos0..`: norm (unless
    /// `normed`: `sc.xn` holds it), mixer, residual, norm, FFN with its
    /// residual, which with `next_norm` also writes `sc.xn` = the new rows
    /// normed by it (the next block's input norm).
    #[allow(clippy::too_many_arguments)]
    fn encode_block(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        l: &LayerW,
        ls: &LayerState,
        pos0: usize,
        snap: Option<usize>,
        m: usize,
        normed: bool,
        next_norm: Option<&Tensor>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        if !normed {
            self.rmsnorm(ctx, pass, &sc.x, &l.attn_norm, &sc.xn, m)?;
            pass.memory_barrier()?;
        }
        match (&l.mixer, ls) {
            (Mixer::Gdn(w), LayerState::Gdn { conv, s, conv_alt, s_alt }) => {
                let snap =
                    snap.map(|row| GdnSnapshot { row, conv: conv_alt, s: s_alt });
                self.gdn(ctx, pass, sc, w, &GdnState { conv, s }, snap.as_ref(), m)?;
            }
            (Mixer::Attn(w), LayerState::Attn { k: kc, v: vc }) => {
                self.attention(ctx, pass, sc, w, kc, vc, pos0, m)?;
            }
            _ => anyhow::bail!("a block's state does not match its kind"),
        }
        if self.fuse_norm() {
            k::residual_norm(
                ctx,
                pass,
                &sc.x,
                &sc.mix,
                None,
                &l.post_norm,
                &sc.xn,
                m,
                cfg.hidden,
                cfg.rms_eps,
            )?;
        } else {
            g::add(ctx, pass, &sc.x, &sc.mix, &sc.x, m * cfg.hidden)?;
            pass.memory_barrier()?;
            self.rmsnorm(ctx, pass, &sc.x, &l.post_norm, &sc.xn, m)?;
        }
        pass.memory_barrier()?;
        self.ffn(ctx, pass, sc, &l.ffn, m, next_norm)
    }

    /// Whether residual adds and the RMSNorm after them are one kernel
    /// (`k::residual_norm`, rows of 2048).
    fn fuse_norm(&self) -> bool {
        self.cfg.hidden == 2048
    }

    /// `out = rmsnorm(x) * w` over `m` rows of the hidden width.
    fn rmsnorm(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        x: &Tensor,
        w: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        k::rmsnorm(ctx, pass, x, w, out, m, self.cfg.hidden, self.cfg.rms_eps)
    }

    /// The MTP head over `m` pairs (llama.cpp's `graph_mtp`): pair `r` is
    /// token `tokens[r]` with the trunk's normed hidden row `hin[r]` (the row
    /// before the token's position), written to the head's cache row
    /// `pos0 + r`. With `draft`, the last pair's greedy next token goes to
    /// `sc.mtp_io[1]`. Uses `sc.x` as its residual stream (after the trunk's
    /// head has read it).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mtp(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Qwen36State,
        sc: &Scratch,
        tokens: &Tensor,
        hin: &Tensor,
        pos0: usize,
        m: usize,
        draft: bool,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let w =
            self.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP head loaded"))?;
        let ms = st.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no MTP state"))?;
        ensure!((1..=self.prefill_rows).contains(&m), "MTP over {m} pairs");
        ensure!(pos0 + m <= st.capacity, "MTP rows past the state's capacity");
        k::embed(ctx, pass, &self.embed, tokens, &sc.xn, m)?;
        pass.memory_barrier()?;
        k::mtp_cat(
            ctx,
            pass,
            &sc.xn,
            hin,
            &w.enorm,
            &w.hnorm,
            &sc.cat,
            m,
            cfg.hidden,
            cfg.rms_eps,
        )?;
        pass.memory_barrier()?;
        let xb = self.bf16_input(ctx, pass, sc, &sc.cat, 2 * cfg.hidden, m)?;
        self.proj(ctx, pass, &[(&w.eh_proj, &sc.x)], &sc.cat, xb, m)?;
        pass.memory_barrier()?;
        let ls = LayerState::Attn { k: ms.k.clone(), v: ms.v.clone() };
        self.encode_block(ctx, pass, sc, &w.layer, &ls, pos0, None, m, false, None)?;
        if !draft {
            return Ok(());
        }
        let x = sc.x.view((m - 1) * cfg.hidden, &[cfg.hidden])?;
        self.rmsnorm(ctx, pass, &x, &w.head_norm, &sc.mtp_hn, 1)?;
        pass.memory_barrier()?;
        self.proj(ctx, pass, &[(&self.output, &sc.logits)], &sc.mtp_hn, None, 1)?;
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
        pass.memory_barrier()
    }

    /// For a prompt run with the MTP head: the run's rows normed into
    /// `sc.hbuf` rows `1..=m` (row 0, the session's pending row, staged by
    /// the caller), then the head over the run's pairs at `pos0..`.
    pub(crate) fn encode_mtp_prompt(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        st: &Qwen36State,
        sc: &Scratch,
        pos0: usize,
        m: usize,
    ) -> Result<()> {
        let h = self.cfg.hidden;
        let rows = sc.hbuf.view(h, &[m * h])?;
        self.rmsnorm(ctx, pass, &sc.x, &self.output_norm, &rows, m)?;
        pass.memory_barrier()?;
        let hin = sc.hbuf.view(0, &[m * h])?;
        self.encode_mtp(ctx, pass, st, sc, &sc.tokens, &hin, pos0, m, false)
    }

    /// Copies hidden row `row` of `src` (`[rows][hidden]`) into `dst`
    /// through host memory (GPU idle on both).
    pub(crate) fn copy_row(
        &self,
        src: &Tensor,
        row: usize,
        dst: &Tensor,
    ) -> Result<()> {
        let h = self.cfg.hidden;
        let v = src.view(row * h, &[h])?;
        dst.write_bytes(v.contents())
    }

    pub(crate) fn hbuf(sc: &Scratch) -> &Tensor {
        &sc.hbuf
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
        self.rmsnorm(ctx, pass, &x, &self.output_norm, &sc.hidden_norm, rows)?;
        pass.memory_barrier()?;
        self.proj(
            ctx,
            pass,
            &[(&self.output, &sc.logits)],
            &sc.hidden_norm,
            None,
            rows,
        )?;
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
            &[
                (&w.qkv, &sc.qkv),
                (&w.gate, &sc.z),
                (&w.alpha, &sc.ga),
                (&w.beta, &sc.gb),
            ],
            &sc.xn,
            xb,
            m,
        )?;
        pass.memory_barrier()?;
        let p = GdnParams {
            conv_w: &w.conv,
            ssm_a: &w.ssm_a,
            dt_bias: &w.dt_bias,
            norm: &w.norm,
        };
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
        if m <= MAX_ROWS {
            // Decode and verify rows through the decode kernel, row by row
            // (a verify row then attends as a decode step would: the prompt
            // kernel rounds in another order, and speculation parted from
            // plain greedy decoding at near ties).
            for r in 0..m {
                if r > 0 {
                    pass.memory_barrier()?;
                }
                let qr = sc.qb.view(r * nq * d, &[nq, d])?;
                let or = sc.attn.view(r * nq * d, &[nq, d])?;
                attn::sdpa_decode(
                    ctx,
                    pass,
                    &qr,
                    kc,
                    vc,
                    &or,
                    pos0 + r + 1,
                    scale,
                    Some((&sc.split_partials, &sc.split_stats)),
                )?;
            }
        } else {
            ensure!(
                self.kv_format == KvFormat::Bf16,
                "prompt attention reads bf16 caches"
            );
            attn::sdpa_prefill(
                ctx, pass, &q, &kc.values, &vc.values, &out, pos0, scale,
            )?;
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
        next_norm: Option<&Tensor>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let used = cfg.n_expert_used;
        let xb = self.bf16_input(ctx, pass, sc, &sc.xn, cfg.hidden, m)?;
        if m <= MAX_ROWS {
            let sets = [(&f.router, &sc.rlogits), (&f.sh_gate_inp, &sc.shg)];
            g::gemv_sets(ctx, pass, &sets, &sc.xn, m)?;
        } else {
            k::router_logits(
                ctx,
                pass,
                &f.router,
                &f.sh_gate_inp,
                &sc.xn,
                &sc.rlogits,
                &sc.shg,
                m,
            )?;
        }
        self.proj(
            ctx,
            pass,
            &[(&f.sh_gate, &sc.sfg), (&f.sh_up, &sc.sfu)],
            &sc.xn,
            xb,
            m,
        )?;
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
            match next_norm {
                Some(w) => k::residual_norm(
                    ctx,
                    pass,
                    &sc.x,
                    &sc.moe_out,
                    None,
                    w,
                    &sc.xn,
                    m,
                    cfg.hidden,
                    cfg.rms_eps,
                )?,
                None => g::add(ctx, pass, &sc.x, &sc.moe_out, &sc.x, m * cfg.hidden)?,
            }
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
        k::moe_down(
            ctx,
            pass,
            &f.down,
            cfg.hidden,
            &sc.moe_mid,
            &sc.rsel,
            &sc.moe_out,
            m,
            used,
        )?;
        pass.memory_barrier()?;
        match next_norm {
            Some(w) => k::residual_norm(
                ctx,
                pass,
                &sc.x,
                &sc.moe_out,
                Some((&sc.sout, &sc.shg)),
                w,
                &sc.xn,
                m,
                cfg.hidden,
                cfg.rms_eps,
            )?,
            None => k::ffn_residual(
                ctx,
                pass,
                &sc.x,
                &sc.moe_out,
                &sc.sout,
                &sc.shg,
                m,
                cfg.hidden,
            )?,
        }
        pass.memory_barrier()
    }
}
