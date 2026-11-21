//! Inkling 2 Mini on the GPU: zero-copy read-only weights from GGUF, 48-slot
//! KV + short-convolution state, looped forward pass, DSpark feature capture,
//! and vision embedding injection.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};

use super::config::{Config, LoopLayout};
use super::dspark::{DsparkState, Inkling2Dspark};
use super::vision::Inkling2Vision;
use crate::gguf::{GgmlType, Gguf};
use crate::glm53::model::Mapped;
use crate::kernels::glm53::{self as k, QWeight};
use crate::kernels::inkling2 as ik;
use crate::kernels::sample::SamplerScratch;
use crate::metal::{Buffer, ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Rows per chunk during prompt prefill (reduced under `--memory-limit-gb`).
pub const PREFILL_ROWS: usize = 256;

/// Longest context supported by the server.
pub const MAX_SEQ: usize = crate::kernels::attention::MAX_SEQ;

/// Extra rows reserved past context for speculative verification.
pub const LOOKAHEAD: usize = 8;

pub fn capacity_cap(context: usize) -> usize {
    (context + LOOKAHEAD).min(MAX_SEQ)
}

pub fn check_tokens(tokens: &[u32], vocab: usize) -> Result<()> {
    if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
        anyhow::bail!("token id {t} is outside vocabulary of {vocab}");
    }
    Ok(())
}

/// Rank-64 cross-pass controller weights for one of `(q, k, v, ffn)`.
struct LoopCtrlW {
    down: QWeight,   // [2*H, rank]
    gain: QWeight,   // [rank, H]
    forget: QWeight, // [rank, H]
}

/// Loop core weights for physical layers in `loop_start..loop_end`.
struct LoopCoreW {
    /// Normalized `[4, H]` F32 queries in `(q, k, v, ffn)` order.
    msg_query: Tensor,
    ctrl: [LoopCtrlW; 4],
}

struct LayerW {
    is_swa: bool,
    attn_norm: Option<Tensor>,
    attn_q_inp_norm: Option<Tensor>,
    attn_k_inp_norm: Option<Tensor>,
    attn_v_inp_norm: Option<Tensor>,
    wq: QWeight,
    wk: QWeight,
    wv: QWeight,
    wo: QWeight,
    attn_gate: QWeight,
    attn_q_clip: Tensor,
    attn_sink_proj: QWeight,
    attn_rel: QWeight,
    attn_rel_bias: QWeight, // F32 [rel_len, extent]
    attn_rel_clip: Tensor,
    attn_k_conv: Tensor,
    attn_v_conv: Tensor,
    attn_k_conv_proj: Option<QWeight>,
    attn_v_conv_proj: Option<QWeight>,
    attn_post_conv: Tensor,
    ffn_post_conv: Tensor,
    attn_res_query: Option<Tensor>, // [1, H] or [3, H]
    ffn_res_query: Tensor,
    ffn_norm: Tensor,
    router: QWeight,
    exp_probs_b: Option<Tensor>,
    gate_up_exps: QWeight,
    gate_up_scale: Option<Tensor>,
    down_exps: QWeight,
    down_scale: Option<Tensor>,
    loop_core: Option<LoopCoreW>,
}

/// Loaded Inkling 2 Mini model on GPU.
pub struct Inkling2Model {
    pub cfg: Config,
    pub(crate) mapped: Mapped,
    pub tok_embd: QWeight,
    pub tok_norm: Tensor,
    pub output: QWeight,
    pub output_norm: Tensor,
    pub output_res_query: Tensor,
    layers: Vec<LayerW>,
    schedule: Vec<(usize, usize, usize)>,
    pub prefill_rows: usize,
    pub memory_limit: Option<u64>,
    pub fitted_context: Option<usize>,
    pub dspark: Option<Inkling2Dspark>,
    pub drafts: usize,
    pub vision: Option<Inkling2Vision>,
}

/// Per-slot (48 slots total) KV cache and causal short-convolution states.
pub struct SlotCache {
    pub swa: bool,
    pub slots: usize,
    pub k: Tensor,         // F16 [slots, kv_w]
    pub v: Tensor,         // F16 [slots, kv_w]
    pub k_hist: Tensor,    // F32 [kv_w, n_conv - 1]
    pub v_hist: Tensor,    // F32 [kv_w, n_conv - 1]
    pub attn_hist: Tensor, // F32 [H, n_conv - 1]
    pub ffn_hist: Tensor,  // F32 [H, n_conv - 1]
}

/// Per-session decode state.
pub struct Inkling2State {
    pub pos: usize,
    pub capacity: usize,
    pub slots: Vec<SlotCache>,
    pub dspark: Option<DsparkState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopAction {
    None,
    SavePrelude,
    RestorePrelude,
}

#[derive(Clone, Copy, Debug)]
struct VisitPlan {
    loop_action: LoopAction,
    msg_ptrs: Option<(usize, usize)>,  // (u32 offset in ptrs_table, n_states)
    attn_ptrs: Option<(usize, usize)>, // (u32 offset in ptrs_table, n_states)
    block_save_idx: Option<usize>,     // Some(b_idx) at block boundary, None -> residual add
    ffn_ptrs: (usize, usize),          // (u32 offset in ptrs_table, n_states)
}

/// Pending state between [`LanguageModel::verify`] and [`LanguageModel::finish_speculation`].
pub struct VerifyPending {
    pub pos_before: usize,
    pub rows: usize,
    pub draws: Vec<u32>,
    pub next_drafts: usize,
}

/// GPU scratch for batched speculative verification (`rows = 1 + drafts`).
pub struct VerifyScratch {
    pub max_rows: usize,
    pub logits_raw: Tensor,       // F32 [max_rows, vocab_padded]
    pub logits: Tensor,           // F32 [max_rows, vocab_unpadded]
    pub draws_gpu: Tensor,        // U32 [max_rows]
    pub trace: Vec<[Tensor; 4]>,  // per slot: [k, v, attn, ffn] each [max_rows, C, hist_len]
    pub pending: Option<VerifyPending>,
}

/// Per-engine scratch buffers.
pub struct Scratch {
    pub max_rows: usize,
    pub ids: Tensor,        // U32 [max_rows]
    pub next_token: Tensor, // U32 [2]
    pub logits: Tensor,     // F32 [vocab_unpadded]
    pub logits_raw: Tensor, // F32 [vocab_padded]
    pub sampler: SamplerScratch,
    /// Optional GPU capture buffer for DSpark target layers `[n_targets, max_rows, H]`.
    pub dspark_cap: Option<Tensor>,
    /// Optional scratch for batched speculative verification (allocated only when DSpark is enabled).
    pub verify: Option<VerifyScratch>,
    // Working activations [max_rows, ...]
    cur: Tensor,
    inp_sa: Tensor,
    ffn_inp: Tensor,
    prelude_cur: Tensor,
    bank_out: Tensor,
    /// Block snapshots for attention-residual mix (up to 16).
    block_bufs: Vec<Tensor>,
    /// Precomputed pointer table U32 [n_mixes * 32] (16 u64s per mix) for `inkling2_mix`.
    ptrs_table: Tensor,
    /// Precomputed per-visit mix/copy plan for zero-sync command buffer encoding.
    visit_plans: Vec<VisitPlan>,
    output_ptrs: (usize, usize),
    mix_out: Tensor,       // [4, max_rows, H]
    msg_out: Tensor,       // [4, max_rows, H]
    attn_in: [Tensor; 3],  // [max_rows, H]
    ctrl_cat: Tensor,      // [max_rows, 2*H]
    ctrl_mid: Tensor,      // [max_rows, rank]
    ctrl_gain: Tensor,     // [max_rows, H]
    ctrl_forget: Tensor,   // [max_rows, H]
    q: Tensor,             // [max_rows, q_w]
    k_raw: Tensor,         // [max_rows, kv_w]
    v_raw: Tensor,         // [max_rows, kv_w]
    k_dyn: Tensor,         // [max_rows, n_head_kv * n_conv]
    v_dyn: Tensor,         // [max_rows, n_head_kv * n_conv]
    r_proj: Tensor,        // [max_rows, n_head * rel_len]
    rel_table: Tensor,     // [max_rows, n_head * extent]
    sinks: Tensor,         // [max_rows, n_head]
    attn_out: Tensor,      // [max_rows, q_w]
    attn_gate: Tensor,     // [max_rows, q_w]
    wo_out: Tensor,        // [max_rows, H]
    ffn_normed: Tensor,    // [max_rows, H]
    router_logits: Tensor, // [max_rows, E]
    exp_ids: Tensor,       // U32 [max_rows * K]
    exp_w: Tensor,         // F32 [max_rows * K]
    sorted_rk: Tensor,     // U32 [max_rows * K]
    moe_tiles: Tensor,     // U32 [max_rows * K * 4]
    moe_n_tiles: Tensor,   // U32 [1]
    situ_out: Tensor,      // [max_rows * K, ff]
    down_out: Tensor,      // [max_rows * K, H] (for rows > 1 grouped MoE)
    moe_out: Tensor,       // [max_rows, H]
}

impl Scratch {
    pub fn stage_tokens(&self, tokens: &[u32]) -> Result<()> {
        ensure!(
            tokens.len() <= self.max_rows,
            "staged {} > {}",
            tokens.len(),
            self.max_rows
        );
        self.ids
            .view(0, &[tokens.len()])?
            .write_bytes(bytemuck::cast_slice(tokens))
    }
}

fn view_qweight(mapped: &Mapped, _g: &Gguf, name: &str) -> Result<QWeight> {
    mapped.q(name)
}

fn view_f32(mapped: &Mapped, _g: &Gguf, name: &str) -> Result<Tensor> {
    let (buf, at, info) = mapped.get(name)?;
    ensure!(info.ty == GgmlType::F32, "tensor {name} expected F32, got {:?}", info.ty);
    ensure!(at.is_multiple_of(4), "tensor {name} offset {at} not 4-byte aligned");
    let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
    let whole = Tensor::from_buffer(buf.clone(), &[buf.length() / 4], DType::F32)?;
    whole.view(*at / 4, &shape)
}

fn copy_f32(ctx: &MetalContext, g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = g.tensor(name).with_context(|| format!("missing tensor {name}"))?;
    let mut file = std::fs::File::open(&g.path)?;
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(info.offset))?;
    let mut raw = vec![0u8; info.bytes as usize];
    file.read_exact(&mut raw)?;
    let mut out = vec![0.0f32; info.row_len() * info.rows()];
    crate::gguf::dequantize(info.ty, &raw, &mut out)?;
    let _ = ctx;
    Ok(out)
}

fn linear(
    ctx: &MetalContext,
    pass: &ComputePass,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    rows: usize,
) -> Result<()> {
    if rows <= k::gemv_max_rows(w.ty) {
        k::gemv(ctx, pass, w, x, y, rows)
    } else if rows > 12 && w.cols.is_multiple_of(64) {
        k::gemm(ctx, pass, w, x, y, rows)
    } else {
        // For small batches (e.g. speculative verify rows=5..8) or narrow widths,
        // chunked GEMV avoids 64x64 sgemm padding overhead.
        let step = k::gemv_max_rows(w.ty);
        for off in (0..rows).step_by(step) {
            let chunk = (rows - off).min(step);
            let xv = x.view(off * w.cols, &[chunk, w.cols])?;
            let yv = y.view(off * w.rows, &[chunk, w.rows])?;
            k::gemv(ctx, pass, w, &xv, &yv, chunk)?;
        }
        Ok(())
    }
}

/// Dispatches multiple linear projections sharing the same input `x`, automatically
/// batching up to 4 same-type projections into one `glm53::gemv_sets` dispatch during
/// decode/verify (`rows <= gemv_max_rows`).
fn linear_multi(
    ctx: &MetalContext,
    pass: &ComputePass,
    x: &Tensor,
    sets: &[(&QWeight, &Tensor)],
    rows: usize,
) -> Result<()> {
    let mut done = [false; 8];
    debug_assert!(sets.len() <= 8);
    for i in 0..sets.len() {
        if done[i] {
            continue;
        }
        let (wi, yi) = sets[i];
        if rows <= k::gemv_max_rows(wi.ty) {
            let mut batch: Vec<(&QWeight, &Tensor)> = vec![(wi, yi)];
            done[i] = true;
            for j in (i + 1)..sets.len() {
                if !done[j]
                    && sets[j].0.ty == wi.ty
                    && sets[j].0.cols == wi.cols
                    && batch.len() < 4
                {
                    batch.push(sets[j]);
                    done[j] = true;
                }
            }
            if batch.len() > 1 {
                k::gemv_sets(ctx, pass, &batch, x, rows)?;
            } else {
                k::gemv(ctx, pass, wi, x, yi, rows)?;
            }
        } else {
            done[i] = true;
            linear(ctx, pass, wi, x, yi, rows)?;
        }
    }
    Ok(())
}

impl Inkling2Model {
    pub fn load(ctx: &MetalContext, path: &Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        let cfg = Config::from_gguf(&g)?;
        super::schema::validate(&g, &cfg)?;
        let mapped = Mapped::map(ctx, &g)?;

        let tok_embd = view_qweight(&mapped, &g, "token_embd.weight")?;
        let tok_norm = view_f32(&mapped, &g, "token_embd_norm.weight")?;
        let output = view_qweight(&mapped, &g, "output.weight")?;
        let output_norm = view_f32(&mapped, &g, "output_norm.weight")?;
        let output_res_query = view_f32(&mapped, &g, "output_res_query.weight")?;

        // For Unsloth root layout, unpack global loop_res_query & loop_kv_res_query into per-layer [4, H].
        let unsloth_loop_q = if cfg.loop_layout == LoopLayout::Unsloth {
            Some((
                copy_f32(ctx, &g, "loop_res_query.weight")?,
                copy_f32(ctx, &g, "loop_kv_res_query.weight")?,
            ))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_phys);
        for i in 0..cfg.n_phys {
            let n = |s: &str| format!("blk.{i}.{s}");
            let is_swa = cfg.swa[i];
            let loop_core = if cfg.loop_count > 1
                && i >= cfg.loop_start
                && i < cfg.loop_end
            {
                let (msg_query, names) = match cfg.loop_layout {
                    LoopLayout::Mainline => (
                        view_f32(&mapped, &g, &n("loop_msg_query.weight"))?,
                        [
                            (
                                "loop_q_ctrl_down.weight",
                                "loop_q_ctrl_gain.weight",
                                "loop_q_ctrl_forget.weight",
                            ),
                            (
                                "loop_k_ctrl_down.weight",
                                "loop_k_ctrl_gain.weight",
                                "loop_k_ctrl_forget.weight",
                            ),
                            (
                                "loop_v_ctrl_down.weight",
                                "loop_v_ctrl_gain.weight",
                                "loop_v_ctrl_forget.weight",
                            ),
                            (
                                "loop_ffn_ctrl_down.weight",
                                "loop_ffn_ctrl_gain.weight",
                                "loop_ffn_ctrl_forget.weight",
                            ),
                        ],
                    ),
                    LoopLayout::Unsloth => {
                        let (res_q, kv_q) = unsloth_loop_q
                            .as_ref()
                            .context("missing Unsloth loop query tensors")?;
                        let h = cfg.hidden;
                        let c = i - cfg.loop_start;
                        let mut packed = Vec::with_capacity(4 * h);
                        // Order: q (attn), k, v, ffn (mlp)
                        packed.extend_from_slice(&res_q[(2 * c) * h..(2 * c + 1) * h]);
                        packed.extend_from_slice(&kv_q[(2 * c) * h..(2 * c + 1) * h]);
                        packed
                            .extend_from_slice(&kv_q[(2 * c + 1) * h..(2 * c + 2) * h]);
                        packed.extend_from_slice(
                            &res_q[(2 * c + 1) * h..(2 * c + 2) * h],
                        );
                        (
                            Tensor::from_f32(ctx, &packed, &[4, h])?,
                            [
                                (
                                    "loop_attn_down.weight",
                                    "loop_attn_gain.weight",
                                    "loop_attn_forget.weight",
                                ),
                                (
                                    "loop_k_down.weight",
                                    "loop_k_gain.weight",
                                    "loop_k_forget.weight",
                                ),
                                (
                                    "loop_v_down.weight",
                                    "loop_v_gain.weight",
                                    "loop_v_forget.weight",
                                ),
                                (
                                    "loop_mlp_down.weight",
                                    "loop_mlp_gain.weight",
                                    "loop_mlp_forget.weight",
                                ),
                            ],
                        )
                    }
                    LoopLayout::None => unreachable!(),
                };
                let ctrl = [0, 1, 2, 3].map(|m| {
                    let (d, gn, fg) = names[m];
                    Ok::<_, anyhow::Error>(LoopCtrlW {
                        down: view_qweight(&mapped, &g, &n(d))?,
                        gain: view_qweight(&mapped, &g, &n(gn))?,
                        forget: view_qweight(&mapped, &g, &n(fg))?,
                    })
                });
                let [c0, c1, c2, c3] = ctrl;
                Some(LoopCoreW { msg_query, ctrl: [c0?, c1?, c2?, c3?] })
            } else {
                None
            };

            layers.push(LayerW {
                is_swa,
                attn_norm: if is_swa {
                    Some(view_f32(&mapped, &g, &n("attn_norm.weight"))?)
                } else {
                    None
                },
                attn_q_inp_norm: if !is_swa {
                    Some(view_f32(&mapped, &g, &n("attn_q_inp_norm.weight"))?)
                } else {
                    None
                },
                attn_k_inp_norm: if !is_swa {
                    Some(view_f32(&mapped, &g, &n("attn_k_inp_norm.weight"))?)
                } else {
                    None
                },
                attn_v_inp_norm: if !is_swa {
                    Some(view_f32(&mapped, &g, &n("attn_v_inp_norm.weight"))?)
                } else {
                    None
                },
                wq: view_qweight(&mapped, &g, &n("attn_q.weight"))?,
                wk: view_qweight(&mapped, &g, &n("attn_k.weight"))?,
                wv: view_qweight(&mapped, &g, &n("attn_v.weight"))?,
                wo: view_qweight(&mapped, &g, &n("attn_output.weight"))?,
                attn_gate: view_qweight(&mapped, &g, &n("attn_gate.weight"))?,
                attn_q_clip: view_f32(&mapped, &g, &n("attn_q_clip.weight"))?,
                attn_sink_proj: view_qweight(&mapped, &g, &n("attn_sink_proj.weight"))?,
                attn_rel: view_qweight(&mapped, &g, &n("attn_rel.weight"))?,
                attn_rel_bias: view_qweight(&mapped, &g, &n("attn_rel_bias.weight"))?,
                attn_rel_clip: view_f32(&mapped, &g, &n("attn_rel_clip.weight"))?,
                attn_k_conv: view_f32(&mapped, &g, &n("attn_k_conv.weight"))?,
                attn_v_conv: view_f32(&mapped, &g, &n("attn_v_conv.weight"))?,
                attn_k_conv_proj: if !is_swa {
                    Some(view_qweight(&mapped, &g, &n("attn_k_conv_proj.weight"))?)
                } else {
                    None
                },
                attn_v_conv_proj: if !is_swa {
                    Some(view_qweight(&mapped, &g, &n("attn_v_conv_proj.weight"))?)
                } else {
                    None
                },
                attn_post_conv: view_f32(&mapped, &g, &n("attn_post_conv.weight"))?,
                ffn_post_conv: view_f32(&mapped, &g, &n("ffn_post_conv.weight"))?,
                attn_res_query: if i > 0 {
                    Some(view_f32(&mapped, &g, &n("attn_res_query.weight"))?)
                } else {
                    None
                },
                ffn_res_query: view_f32(&mapped, &g, &n("ffn_res_query.weight"))?,
                ffn_norm: view_f32(&mapped, &g, &n("ffn_norm.weight"))?,
                router: view_qweight(&mapped, &g, &n("ffn_gate_inp.weight"))?,
                exp_probs_b: if g.tensor(&n("exp_probs_b.bias")).is_some() {
                    Some(view_f32(&mapped, &g, &n("exp_probs_b.bias"))?)
                } else {
                    None
                },
                gate_up_exps: view_qweight(&mapped, &g, &n("ffn_gate_up_exps.weight"))?,
                gate_up_scale: if g.tensor(&n("ffn_gate_up_exps.scale")).is_some() {
                    Some(view_f32(&mapped, &g, &n("ffn_gate_up_exps.scale"))?)
                } else {
                    None
                },
                down_exps: view_qweight(&mapped, &g, &n("ffn_down_exps.weight"))?,
                down_scale: if g.tensor(&n("ffn_down_exps.scale")).is_some() {
                    Some(view_f32(&mapped, &g, &n("ffn_down_exps.scale"))?)
                } else {
                    None
                },
                loop_core,
            });
        }

        let schedule = cfg.schedule();
        Ok(Self {
            cfg,
            mapped,
            tok_embd,
            tok_norm,
            output,
            output_norm,
            output_res_query,
            layers,
            schedule,
            prefill_rows: PREFILL_ROWS,
            memory_limit: None,
            fitted_context: None,
            dspark: None,
            drafts: 0,
            vision: None,
        })
    }

    pub fn weight_buffers(&self) -> Vec<Buffer> {
        self.mapped.buffers()
    }

    /// Global-layer KV bytes per token (F16 K + F16 V across all global visit slots + DSpark KV).
    pub fn bytes_per_token(&self) -> usize {
        let target: usize = (0..self.cfg.n_visits)
            .map(|slot| {
                let p = self.cfg.phys_of_slot(slot);
                if self.cfg.swa[p] { 0 } else { 2 * 2 * self.cfg.kv_width(p) }
            })
            .sum();
        let dspark = self.dspark.as_ref().map_or(0, |ds| {
            ds.cfg.n_layer * 2 * 2 * ds.cfg.n_head_kv * ds.cfg.head_dim
        });
        target + dspark
    }

    /// Fixed recurrent bytes per session (SWA KV rings + all 48 short-conv states + DSpark f_rnn_last).
    pub fn recurrent_bytes(&self) -> usize {
        let hist_len = self.cfg.sconv_l_cache - 1;
        let swa_slots = self.cfg.window + self.prefill_rows;
        let target: usize = (0..self.cfg.n_visits)
            .map(|slot| {
                let p = self.cfg.phys_of_slot(slot);
                let kw = self.cfg.kv_width(p);
                let h = self.cfg.hidden;
                let sconv = 4 * hist_len * (2 * kw + 2 * h);
                let ring = if self.cfg.swa[p] { 2 * 2 * swa_slots * kw } else { 0 };
                sconv + ring
            })
            .sum();
        let dspark = self.dspark.as_ref().map_or(0, |ds| ds.cfg.hidden * 4);
        target + dspark
    }

    pub fn new_state(
        &self,
        ctx: &MetalContext,
        capacity: usize,
    ) -> Result<Inkling2State> {
        let cap = capacity.clamp(64, MAX_SEQ);
        let hist_len = self.cfg.sconv_l_cache - 1;
        let swa_slots = self.cfg.window + self.prefill_rows;
        let mut slots = Vec::with_capacity(self.cfg.n_visits);
        for slot in 0..self.cfg.n_visits {
            let p = self.cfg.phys_of_slot(slot);
            let swa = self.cfg.swa[p];
            let n_slots = if swa { swa_slots } else { cap };
            let kw = self.cfg.kv_width(p);
            let h = self.cfg.hidden;
            slots.push(SlotCache {
                swa,
                slots: n_slots,
                k: Tensor::zeros(ctx, &[n_slots, kw], DType::F16)?,
                v: Tensor::zeros(ctx, &[n_slots, kw], DType::F16)?,
                k_hist: Tensor::zeros(ctx, &[kw, hist_len], DType::F32)?,
                v_hist: Tensor::zeros(ctx, &[kw, hist_len], DType::F32)?,
                attn_hist: Tensor::zeros(ctx, &[h, hist_len], DType::F32)?,
                ffn_hist: Tensor::zeros(ctx, &[h, hist_len], DType::F32)?,
            });
        }
        Ok(Inkling2State {
            pos: 0,
            capacity: cap,
            slots,
            dspark: match &self.dspark {
                Some(d) => Some(DsparkState::new(ctx, &d.cfg, cap)?),
                None => None,
            },
        })
    }

    pub fn new_scratch(&self, ctx: &MetalContext, max_rows: usize) -> Result<Scratch> {
        let r = max_rows.max(1);
        let h = self.cfg.hidden;
        let max_qw =
            (0..self.cfg.n_phys).map(|p| self.cfg.q_width(p)).max().unwrap_or(h);
        let max_kw =
            (0..self.cfg.n_phys).map(|p| self.cfg.kv_width(p)).max().unwrap_or(h);
        let max_nh = self.cfg.n_head.iter().copied().max().unwrap_or(1);
        let max_nkv = self.cfg.n_head_kv.iter().copied().max().unwrap_or(1);
        let max_ext = self.cfg.window.max(self.cfg.rel_extent);
        let rank = self.cfg.loop_gate_rank.max(1);
        let k_exp = self.cfg.n_expert_used;
        let ff = self.cfg.ff_expert;

        let mut block_bufs = Vec::with_capacity(16);
        for _ in 0..16 {
            block_bufs.push(Tensor::zeros(ctx, &[r, h], DType::F32)?);
        }

        let cur = Tensor::zeros(ctx, &[r, h], DType::F32)?;
        let bank_out = Tensor::zeros(ctx, &[r, h], DType::F32)?;

        // Precompute every static pointer table for `inkling2_mix` across all visits + output.
        let mut ptrs_u64: Vec<u64> = Vec::new();
        let mut push_ptrs = |refs: &[&Tensor]| -> (usize, usize) {
            let entry_off = ptrs_u64.len() * 2; // u32 element offset (each u64 = 2 u32s)
            let mut raw = [0u64; 16];
            for (i, t) in refs.iter().enumerate() {
                raw[i] = t.buffer().address() + t.binding().1 as u64;
            }
            ptrs_u64.extend_from_slice(&raw);
            (entry_off, refs.len())
        };

        let mut visit_plans = Vec::with_capacity(self.schedule.len());
        let mut blocks: Vec<usize> = Vec::with_capacity(16);
        let mut prelude_blocks: Vec<usize> = Vec::new();
        let mut bank_indices: Vec<usize> = Vec::new();
        let mut next_buf = 0usize;

        for &(_slot, ip, pass_idx) in &self.schedule {
            let loop_action = if self.cfg.loop_count > 1 && ip == self.cfg.loop_start {
                if pass_idx == 0 {
                    prelude_blocks = blocks.clone();
                    LoopAction::SavePrelude
                } else {
                    bank_indices = blocks[prelude_blocks.len() + 1..].to_vec();
                    blocks = prelude_blocks.clone();
                    LoopAction::RestorePrelude
                }
            } else {
                LoopAction::None
            };

            let layer = &self.layers[ip];
            let has_msg = pass_idx > 0 && layer.loop_core.is_some();

            let msg_ptrs = if has_msg {
                let mut refs: Vec<&Tensor> =
                    bank_indices.iter().map(|&bi| &block_bufs[bi]).collect();
                refs.push(&bank_out);
                Some(push_ptrs(&refs))
            } else {
                None
            };

            let attn_ptrs = if layer.attn_res_query.is_some() {
                let mut refs: Vec<&Tensor> =
                    blocks.iter().map(|&bi| &block_bufs[bi]).collect();
                refs.push(&cur);
                Some(push_ptrs(&refs))
            } else {
                None
            };

            let block_save_idx = if ip.is_multiple_of(self.cfg.attn_res_block_size) {
                let b_idx = next_buf % block_bufs.len();
                next_buf += 1;
                blocks.push(b_idx);
                Some(b_idx)
            } else {
                None
            };

            let ffn_ptrs = {
                let mut refs: Vec<&Tensor> =
                    blocks.iter().map(|&bi| &block_bufs[bi]).collect();
                refs.push(&cur);
                push_ptrs(&refs)
            };

            visit_plans.push(VisitPlan {
                loop_action,
                msg_ptrs,
                attn_ptrs,
                block_save_idx,
                ffn_ptrs,
            });
        }

        let output_ptrs = {
            let mut refs: Vec<&Tensor> =
                blocks.iter().map(|&bi| &block_bufs[bi]).collect();
            refs.push(&cur);
            push_ptrs(&refs)
        };

        let ptrs_table = Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&ptrs_u64),
            &[ptrs_u64.len() * 2],
            DType::U32,
        )?;

        let dspark_cap = if let Some(ds) = &self.dspark {
            Some(Tensor::zeros(ctx, &[ds.cfg.target_layers.len(), r, h], DType::F32)?)
        } else {
            None
        };

        let verify = if self.dspark.is_some() && self.drafts > 0 {
            let mv = (1 + self.drafts).min(r);
            let hist_len = self.cfg.sconv_l_cache - 1;
            let mut trace = Vec::with_capacity(self.cfg.n_visits);
            for slot in 0..self.cfg.n_visits {
                let p = self.cfg.phys_of_slot(slot);
                let kw = self.cfg.kv_width(p);
                trace.push([
                    Tensor::zeros(ctx, &[mv, kw, hist_len], DType::F32)?,
                    Tensor::zeros(ctx, &[mv, kw, hist_len], DType::F32)?,
                    Tensor::zeros(ctx, &[mv, h, hist_len], DType::F32)?,
                    Tensor::zeros(ctx, &[mv, h, hist_len], DType::F32)?,
                ]);
            }
            Some(VerifyScratch {
                max_rows: mv,
                logits_raw: Tensor::zeros(
                    ctx,
                    &[mv, self.cfg.vocab_padded],
                    DType::F32,
                )?,
                logits: Tensor::zeros(
                    ctx,
                    &[mv, self.cfg.vocab_unpadded],
                    DType::F32,
                )?,
                draws_gpu: Tensor::zeros(ctx, &[mv], DType::U32)?,
                trace,
                pending: None,
            })
        } else {
            None
        };

        Ok(Scratch {
            max_rows: r,
            ids: Tensor::zeros(ctx, &[r], DType::U32)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
            logits: Tensor::zeros(ctx, &[self.cfg.vocab_unpadded], DType::F32)?,
            logits_raw: Tensor::zeros(ctx, &[self.cfg.vocab_padded], DType::F32)?,
            sampler: SamplerScratch::new(ctx, self.cfg.vocab_unpadded)?,
            dspark_cap,
            verify,
            cur,
            inp_sa: Tensor::zeros(ctx, &[r, h], DType::F32)?,
            ffn_inp: Tensor::zeros(ctx, &[r, h], DType::F32)?,
            prelude_cur: Tensor::zeros(ctx, &[r, h], DType::F32)?,
            bank_out,
            block_bufs,
            ptrs_table,
            visit_plans,
            output_ptrs,
            mix_out: Tensor::zeros(ctx, &[4, r, h], DType::F32)?,
            msg_out: Tensor::zeros(ctx, &[4, r, h], DType::F32)?,
            attn_in: [
                Tensor::zeros(ctx, &[r, h], DType::F32)?,
                Tensor::zeros(ctx, &[r, h], DType::F32)?,
                Tensor::zeros(ctx, &[r, h], DType::F32)?,
            ],
            ctrl_cat: Tensor::zeros(ctx, &[r, 2 * h], DType::F32)?,
            ctrl_mid: Tensor::zeros(ctx, &[3, r, rank], DType::F32)?,
            ctrl_gain: Tensor::zeros(ctx, &[r, h], DType::F32)?,
            ctrl_forget: Tensor::zeros(ctx, &[r, h], DType::F32)?,
            q: Tensor::zeros(ctx, &[r, max_qw], DType::F32)?,
            k_raw: Tensor::zeros(ctx, &[r, max_kw], DType::F32)?,
            v_raw: Tensor::zeros(ctx, &[r, max_kw], DType::F32)?,
            k_dyn: Tensor::zeros(
                ctx,
                &[r, max_nkv * self.cfg.sconv_l_cache],
                DType::F32,
            )?,
            v_dyn: Tensor::zeros(
                ctx,
                &[r, max_nkv * self.cfg.sconv_l_cache],
                DType::F32,
            )?,
            r_proj: Tensor::zeros(ctx, &[r, max_nh * self.cfg.rel_len], DType::F32)?,
            rel_table: Tensor::zeros(ctx, &[r, max_nh * max_ext], DType::F32)?,
            sinks: Tensor::zeros(ctx, &[r, max_nh], DType::F32)?,
            attn_out: Tensor::zeros(ctx, &[r, max_qw], DType::F32)?,
            attn_gate: Tensor::zeros(ctx, &[r, max_qw], DType::F32)?,
            wo_out: Tensor::zeros(ctx, &[r, h], DType::F32)?,
            ffn_normed: Tensor::zeros(ctx, &[r, h], DType::F32)?,
            router_logits: Tensor::zeros(ctx, &[r, self.cfg.n_expert], DType::F32)?,
            exp_ids: Tensor::zeros(ctx, &[r * k_exp], DType::U32)?,
            exp_w: Tensor::zeros(ctx, &[r * k_exp], DType::F32)?,
            sorted_rk: Tensor::zeros(ctx, &[r * k_exp], DType::U32)?,
            moe_tiles: Tensor::zeros(ctx, &[r * k_exp * 4], DType::U32)?,
            moe_n_tiles: Tensor::zeros(ctx, &[1], DType::U32)?,
            situ_out: Tensor::zeros(ctx, &[r * k_exp, ff], DType::F32)?,
            down_out: Tensor::zeros(ctx, &[r * k_exp, h], DType::F32)?,
            moe_out: Tensor::zeros(ctx, &[r, h], DType::F32)?,
        })
    }

    /// Encodes `rows` tokens into a single `ComputePass` with zero host-GPU syncs.
    /// When `verify_mode` is true, records per-row short-conv history traces and
    /// computes logits for all `rows` tokens into `sc.verify.logits`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_forward_rows(
        &self,
        ctx: &MetalContext,
        cp: &ComputePass<'_>,
        state: &Inkling2State,
        sc: &Scratch,
        rows: usize,
        ids_override: Option<&Tensor>,
        raw_embd: bool,
        compute_logits: bool,
        capture_dspark: bool,
        verify_mode: bool,
    ) -> Result<()> {
        ensure!(
            rows >= 1 && rows <= sc.max_rows,
            "forward_rows {rows} > {}",
            sc.max_rows
        );
        let h = self.cfg.hidden;
        let row_elems = rows * h;
        let pos0 = state.pos;

        // 1. Embedding + optional token_embd_norm (skipped for image embeddings!).
        if !raw_embd {
            let ids = ids_override.unwrap_or(&sc.ids);
            ik::embed(ctx, cp, ids, &self.tok_embd, &sc.cur, rows)?;
            cp.memory_barrier()?;
            ik::rms_norm(
                ctx,
                cp,
                &sc.cur,
                Some(&self.tok_norm),
                &sc.cur,
                rows,
                h,
                self.cfg.rms_eps,
            )?;
            cp.memory_barrier()?;
        }

        let target_layers: &[usize] =
            self.dspark.as_ref().map_or(&[], |d| d.cfg.target_layers.as_slice());

        for (visit_idx, &(slot, ip, pass_idx)) in self.schedule.iter().enumerate() {
            let plan = &sc.visit_plans[visit_idx];

            // Loop core state management (all on GPU).
            match plan.loop_action {
                LoopAction::None => {}
                LoopAction::SavePrelude => {
                    ik::copy_f32(ctx, cp, &sc.cur, &sc.prelude_cur, row_elems)?;
                    cp.memory_barrier()?;
                }
                LoopAction::RestorePrelude => {
                    ik::copy_f32(ctx, cp, &sc.cur, &sc.bank_out, row_elems)?;
                    cp.memory_barrier()?;
                    ik::copy_f32(ctx, cp, &sc.prelude_cur, &sc.cur, row_elems)?;
                    cp.memory_barrier()?;
                }
            }

            // Capture target hidden state for DSpark into GPU scratch if requested.
            if capture_dspark
                && let Some(cap_buf) = &sc.dspark_cap
                && let Some(ti) = target_layers.iter().position(|&l| l == visit_idx)
            {
                let dst = cap_buf.view(ti * sc.max_rows * h, &[rows, h])?;
                ik::copy_f32(ctx, cp, &sc.cur, &dst, row_elems)?;
            }

            let layer = &self.layers[ip];
            let slot_cache = &state.slots[slot];
            let nh = self.cfg.n_head[ip];
            let nkv = self.cfg.n_head_kv[ip];
            let ext = self.cfg.rel_extent_of(ip);
            let has_msg = pass_idx > 0 && layer.loop_core.is_some();

            // Cross-pass messages + save inpSA = cur + attention-residual mix (+ fused SWA norm when !has_msg!).
            if let Some((off, n_states)) = plan.msg_ptrs {
                let lc = layer
                    .loop_core
                    .as_ref()
                    .context("msg_ptrs present without loop_core")?;
                let ptrs = sc.ptrs_table.view(off, &[32])?;
                ik::attn_res_mix(
                    ctx,
                    cp,
                    &ptrs,
                    &lc.msg_query,
                    &sc.msg_out,
                    rows,
                    h,
                    n_states,
                    4,
                    1e-6,
                )?;
            }

            let mix_direct_global =
                !layer.is_swa && !has_msg && layer.attn_res_query.is_some();
            let mix_direct_msg = has_msg && layer.attn_res_query.is_some();
            let mut swa_norm_fused = false;
            if let (Some(q_res), Some((off, n_states))) =
                (&layer.attn_res_query, plan.attn_ptrs)
            {
                let n_q = if layer.is_swa { 1 } else { 3 };
                let ptrs = sc.ptrs_table.view(off, &[32])?;
                if layer.is_swa && !has_msg {
                    // Fuse inp_sa copy + SWA attn_norm right into attn_res_mix!
                    ik::attn_res_mix_fused(
                        ctx,
                        cp,
                        &ptrs,
                        q_res,
                        &sc.attn_in[0],
                        Some(&sc.inp_sa),
                        layer.attn_norm.as_ref(),
                        rows,
                        h,
                        n_states,
                        1,
                        1e-6,
                        self.cfg.rms_eps,
                    )?;
                    swa_norm_fused = true;
                } else {
                    ik::attn_res_mix_fused(
                        ctx,
                        cp,
                        &ptrs,
                        q_res,
                        &sc.mix_out,
                        Some(&sc.inp_sa),
                        None,
                        rows,
                        h,
                        n_states,
                        n_q,
                        1e-6,
                        self.cfg.rms_eps,
                    )?;
                    if !mix_direct_global && !mix_direct_msg {
                        cp.memory_barrier()?;
                        for j in 0..3 {
                            let src_j = if layer.is_swa { 0 } else { j };
                            let src_v = sc.mix_out.view(src_j * row_elems, &[rows, h])?;
                            ik::copy_f32(ctx, cp, &src_v, &sc.attn_in[j], row_elems)?;
                        }
                    }
                }
            } else {
                ik::copy_f32(ctx, cp, &sc.cur, &sc.inp_sa, row_elems)?;
                let n_copies = if layer.is_swa && !has_msg { 1 } else { 3 };
                for j in 0..n_copies {
                    ik::copy_f32(ctx, cp, &sc.cur, &sc.attn_in[j], row_elems)?;
                }
            }
            cp.memory_barrier()?;

            if has_msg {
                let lc = layer
                    .loop_core
                    .as_ref()
                    .context("has_msg set without loop_core")?;
                let all_f32 = (0..3).all(|j| {
                    lc.ctrl[j].down.ty == GgmlType::F32
                        && lc.ctrl[j].gain.ty == GgmlType::F32
                        && lc.ctrl[j].forget.ty == GgmlType::F32
                });
                let rank = self.cfg.loop_gate_rank;
                if all_f32 && rank <= 128 && rank.is_multiple_of(4) && h.is_multiple_of(4) {
                    let c0 = if mix_direct_msg {
                        sc.mix_out.view(0, &[rows, h])?
                    } else {
                        sc.attn_in[0].view(0, &[rows, h])?
                    };
                    let c1 = if mix_direct_msg {
                        sc.mix_out.view(if layer.is_swa { 0 } else { row_elems }, &[rows, h])?
                    } else {
                        sc.attn_in[1].view(0, &[rows, h])?
                    };
                    let c2 = if mix_direct_msg {
                        sc.mix_out.view(if layer.is_swa { 0 } else { 2 * row_elems }, &[rows, h])?
                    } else {
                        sc.attn_in[2].view(0, &[rows, h])?
                    };
                    ik::ctrl_down3_f32(
                        ctx,
                        cp,
                        [&c0, &c1, &c2],
                        &sc.msg_out,
                        [&lc.ctrl[0].down, &lc.ctrl[1].down, &lc.ctrl[2].down],
                        &sc.ctrl_mid,
                        rows,
                        h,
                        rank,
                        1e-6,
                    )?;
                    cp.memory_barrier()?;
                    ik::ctrl_blend3_f32(
                        ctx,
                        cp,
                        [&c0, &c1, &c2],
                        &sc.msg_out,
                        &sc.ctrl_mid,
                        [&lc.ctrl[0].gain, &lc.ctrl[1].gain, &lc.ctrl[2].gain],
                        [&lc.ctrl[0].forget, &lc.ctrl[1].forget, &lc.ctrl[2].forget],
                        [&sc.attn_in[0], &sc.attn_in[1], &sc.attn_in[2]],
                        rows,
                        h,
                        rank,
                    )?;
                    cp.memory_barrier()?;
                } else {
                    for j in 0..3 {
                        let cur_j = if mix_direct_msg {
                            let src_j = if layer.is_swa { 0 } else { j };
                            sc.mix_out.view(src_j * row_elems, &[rows, h])?
                        } else {
                            sc.attn_in[j].view(0, &[rows, h])?
                        };
                        let msg_j = sc.msg_out.view(j * row_elems, &[rows, h])?;
                        ik::ctrl_concat_norm(
                            ctx,
                            cp,
                            &cur_j,
                            &msg_j,
                            &sc.ctrl_cat,
                            rows,
                            h,
                            1e-6,
                        )?;
                        cp.memory_barrier()?;
                        linear(
                            ctx,
                            cp,
                            &lc.ctrl[j].down,
                            &sc.ctrl_cat,
                            &sc.ctrl_mid,
                            rows,
                        )?;
                        cp.memory_barrier()?;
                        linear_multi(
                            ctx,
                            cp,
                            &sc.ctrl_mid,
                            &[
                                (&lc.ctrl[j].gain, &sc.ctrl_gain),
                                (&lc.ctrl[j].forget, &sc.ctrl_forget),
                            ],
                            rows,
                        )?;
                        cp.memory_barrier()?;
                        ik::ctrl_blend(
                            ctx,
                            cp,
                            &cur_j,
                            &msg_j,
                            &sc.ctrl_gain,
                            &sc.ctrl_forget,
                            &sc.attn_in[j],
                            row_elems,
                        )?;
                        cp.memory_barrier()?;
                    }
                }
            }

            // Attention input norms (skip if already fused into SWA attn_res_mix!).
            if swa_norm_fused {
                // Already normalized inside `attn_res_mix_fused`!
            } else if layer.is_swa && has_msg {
                let norm = layer
                    .attn_norm
                    .as_ref()
                    .context("SWA layer missing attn_norm")?;
                ik::rms_norm3(
                    ctx,
                    cp,
                    [&sc.attn_in[0], &sc.attn_in[1], &sc.attn_in[2]],
                    [norm, norm, norm],
                    [&sc.attn_in[0], &sc.attn_in[1], &sc.attn_in[2]],
                    rows,
                    h,
                    self.cfg.rms_eps,
                )?;
            } else if layer.is_swa {
                let norm = layer
                    .attn_norm
                    .as_ref()
                    .context("SWA layer missing attn_norm")?;
                ik::rms_norm(
                    ctx,
                    cp,
                    &sc.attn_in[0],
                    Some(norm),
                    &sc.attn_in[0],
                    rows,
                    h,
                    self.cfg.rms_eps,
                )?;
            } else if mix_direct_global {
                let m0 = sc.mix_out.view(0, &[rows, h])?;
                let m1 = sc.mix_out.view(row_elems, &[rows, h])?;
                let m2 = sc.mix_out.view(2 * row_elems, &[rows, h])?;
                let qn = layer
                    .attn_q_inp_norm
                    .as_ref()
                    .context("global layer missing attn_q_inp_norm")?;
                let kn = layer
                    .attn_k_inp_norm
                    .as_ref()
                    .context("global layer missing attn_k_inp_norm")?;
                let vn = layer
                    .attn_v_inp_norm
                    .as_ref()
                    .context("global layer missing attn_v_inp_norm")?;
                ik::rms_norm3(
                    ctx,
                    cp,
                    [&m0, &m1, &m2],
                    [qn, kn, vn],
                    [&sc.attn_in[0], &sc.attn_in[1], &sc.attn_in[2]],
                    rows,
                    h,
                    self.cfg.rms_eps,
                )?;
            } else {
                let qn = layer
                    .attn_q_inp_norm
                    .as_ref()
                    .context("global layer missing attn_q_inp_norm")?;
                let kn = layer
                    .attn_k_inp_norm
                    .as_ref()
                    .context("global layer missing attn_k_inp_norm")?;
                let vn = layer
                    .attn_v_inp_norm
                    .as_ref()
                    .context("global layer missing attn_v_inp_norm")?;
                ik::rms_norm3(
                    ctx,
                    cp,
                    [&sc.attn_in[0], &sc.attn_in[1], &sc.attn_in[2]],
                    [qn, kn, vn],
                    [&sc.attn_in[0], &sc.attn_in[1], &sc.attn_in[2]],
                    rows,
                    h,
                    self.cfg.rms_eps,
                )?;
            }
            cp.memory_barrier()?;

            let inp_q = &sc.attn_in[0];
            let inp_k =
                if layer.is_swa && !has_msg { &sc.attn_in[0] } else { &sc.attn_in[1] };
            let inp_v =
                if layer.is_swa && !has_msg { &sc.attn_in[0] } else { &sc.attn_in[2] };

            // Q, K, V, R, sinks, gate, and optional dynamic conv projections (batched by type).
            if layer.is_swa && !has_msg {
                linear_multi(
                    ctx,
                    cp,
                    inp_q,
                    &[
                        (&layer.wq, &sc.q),
                        (&layer.wk, &sc.k_raw),
                        (&layer.wv, &sc.v_raw),
                        (&layer.attn_rel, &sc.r_proj),
                        (&layer.attn_sink_proj, &sc.sinks),
                        (&layer.attn_gate, &sc.attn_gate),
                    ],
                    rows,
                )?;
            } else {
                linear_multi(
                    ctx,
                    cp,
                    inp_q,
                    &[
                        (&layer.wq, &sc.q),
                        (&layer.attn_rel, &sc.r_proj),
                        (&layer.attn_sink_proj, &sc.sinks),
                        (&layer.attn_gate, &sc.attn_gate),
                    ],
                    rows,
                )?;
                if let (Some(kp), Some(vp)) =
                    (&layer.attn_k_conv_proj, &layer.attn_v_conv_proj)
                {
                    linear_multi(
                        ctx,
                        cp,
                        inp_k,
                        &[(&layer.wk, &sc.k_raw), (kp, &sc.k_dyn)],
                        rows,
                    )?;
                    linear_multi(
                        ctx,
                        cp,
                        inp_v,
                        &[(&layer.wv, &sc.v_raw), (vp, &sc.v_dyn)],
                        rows,
                    )?;
                } else {
                    linear(ctx, cp, &layer.wk, inp_k, &sc.k_raw, rows)?;
                    linear(ctx, cp, &layer.wv, inp_v, &sc.v_raw, rows)?;
                }
            }
            cp.memory_barrier()?;

            // Fused K/V short_conv + L2 clip(K) / RMSNorm(V) + f16 KV store,
            // plus R L2 clip + relative-position bias projection!
            let tr = if verify_mode {
                sc.verify.as_ref().map(|v| &v.trace[slot])
            } else {
                None
            };
            let (lf, la) = if layer.is_swa {
                (0, 0.0)
            } else {
                (self.cfg.log_scale_floor, self.cfg.log_scale_alpha)
            };
            ik::kv_prep(
                ctx,
                cp,
                &sc.k_raw,
                &sc.v_raw,
                &layer.attn_k_conv,
                &layer.attn_v_conv,
                if layer.is_swa { None } else { Some(&sc.k_dyn) },
                if layer.is_swa { None } else { Some(&sc.v_dyn) },
                &slot_cache.k_hist,
                &slot_cache.v_hist,
                &slot_cache.k,
                &slot_cache.v,
                tr.map(|t| &t[0]),
                tr.map(|t| &t[1]),
                rows,
                nkv,
                self.cfg.sconv_l_cache,
                layer.is_swa,
                pos0,
                slot_cache.slots,
                self.cfg.qk_clip,
                self.cfg.rms_eps,
            )?;
            if rows == 1 && nh <= 32 && self.cfg.rel_len <= 16 {
                // Fused R L2 clip + rel_bias in one dispatch!
                ik::rel_bias_clipped(
                    ctx,
                    cp,
                    &sc.r_proj,
                    &layer.attn_rel_bias,
                    &sc.rel_table,
                    Some(&layer.attn_rel_clip),
                    nh,
                    self.cfg.rel_clip,
                    pos0,
                    lf,
                    la,
                )?;
                cp.memory_barrier()?;
            } else {
                ik::l2_clip(
                    ctx,
                    cp,
                    &sc.r_proj,
                    Some(&layer.attn_rel_clip),
                    &sc.r_proj,
                    rows,
                    nh,
                    self.cfg.rel_len,
                    self.cfg.rel_clip,
                    pos0,
                    lf,
                    la,
                )?;
                cp.memory_barrier()?;
                ik::rel_bias(
                    ctx,
                    cp,
                    &sc.r_proj,
                    &layer.attn_rel_bias,
                    &sc.rel_table,
                    rows * nh,
                )?;
                cp.memory_barrier()?;
            }

            // Relative-position bias attention with zero sink + fused Q L2 clip + fused sigmoid output gate.
            ik::attention_fused(
                ctx,
                cp,
                &sc.q,
                &slot_cache.k,
                &slot_cache.v,
                &sc.rel_table,
                &sc.sinks,
                &sc.attn_out,
                Some(&sc.attn_gate),
                Some(&layer.attn_q_clip),
                rows,
                nh,
                nkv,
                pos0,
                slot_cache.slots,
                layer.is_swa,
                self.cfg.window,
                ext,
                self.cfg.qk_clip,
                lf,
                la,
            )?;
            cp.memory_barrier()?;

            // Output projection, post-attention short conv (+ fused residual add on non-boundary layers).
            linear(ctx, cp, &layer.wo, &sc.attn_out, &sc.wo_out, rows)?;
            cp.memory_barrier()?;
            let attn_res = if plan.block_save_idx.is_none() {
                Some(&sc.inp_sa)
            } else {
                None
            };
            ik::short_conv_fused(
                ctx,
                cp,
                &sc.wo_out,
                &layer.attn_post_conv,
                None,
                &slot_cache.attn_hist,
                &sc.cur,
                tr.map(|t| &t[2]),
                attn_res,
                rows,
                h,
                self.cfg.sconv_l_cache,
                0,
            )?;
            if let Some(b_idx) = plan.block_save_idx {
                ik::copy_f32(ctx, cp, &sc.inp_sa, &sc.block_bufs[b_idx], row_elems)?;
            }
            cp.memory_barrier()?;

            // Fused ffn_inp = cur copy + FFN attention-residual mix (+ fused ffn_norm when !has_msg!).
            let ffn_norm_fused = !has_msg;
            {
                let (off, n_states) = plan.ffn_ptrs;
                let ptrs = sc.ptrs_table.view(off, &[32])?;
                ik::attn_res_mix_fused(
                    ctx,
                    cp,
                    &ptrs,
                    &layer.ffn_res_query,
                    &sc.ffn_normed,
                    Some(&sc.ffn_inp),
                    if ffn_norm_fused { Some(&layer.ffn_norm) } else { None },
                    rows,
                    h,
                    n_states,
                    1,
                    1e-6,
                    self.cfg.rms_eps,
                )?;
            }
            cp.memory_barrier()?;

            if has_msg {
                let lc = layer
                    .loop_core
                    .as_ref()
                    .context("has_msg set without loop_core")?;
                let msg_ffn = sc.msg_out.view(3 * row_elems, &[rows, h])?;
                let rank = self.cfg.loop_gate_rank;
                if lc.ctrl[3].down.ty == GgmlType::F32
                    && lc.ctrl[3].gain.ty == GgmlType::F32
                    && lc.ctrl[3].forget.ty == GgmlType::F32
                    && rank <= 128
                    && rank.is_multiple_of(4)
                    && h.is_multiple_of(4)
                {
                    let mid_0 = sc.ctrl_mid.view(0, &[rows, rank])?;
                    ik::ctrl_down_f32(
                        ctx,
                        cp,
                        &sc.ffn_normed,
                        &msg_ffn,
                        &lc.ctrl[3].down,
                        &mid_0,
                        rows,
                        h,
                        rank,
                        1e-6,
                    )?;
                    cp.memory_barrier()?;
                    ik::ctrl_blend_f32(
                        ctx,
                        cp,
                        &sc.ffn_normed,
                        &msg_ffn,
                        &mid_0,
                        &lc.ctrl[3].gain,
                        &lc.ctrl[3].forget,
                        &sc.ffn_normed,
                        rows,
                        h,
                        rank,
                    )?;
                    cp.memory_barrier()?;
                } else {
                    ik::ctrl_concat_norm(
                        ctx,
                        cp,
                        &sc.ffn_normed,
                        &msg_ffn,
                        &sc.ctrl_cat,
                        rows,
                        h,
                        1e-6,
                    )?;
                    cp.memory_barrier()?;
                    linear(ctx, cp, &lc.ctrl[3].down, &sc.ctrl_cat, &sc.ctrl_mid, rows)?;
                    cp.memory_barrier()?;
                    linear_multi(
                        ctx,
                        cp,
                        &sc.ctrl_mid,
                        &[
                            (&lc.ctrl[3].gain, &sc.ctrl_gain),
                            (&lc.ctrl[3].forget, &sc.ctrl_forget),
                        ],
                        rows,
                    )?;
                    cp.memory_barrier()?;
                    ik::ctrl_blend(
                        ctx,
                        cp,
                        &sc.ffn_normed,
                        &msg_ffn,
                        &sc.ctrl_gain,
                        &sc.ctrl_forget,
                        &sc.ffn_normed,
                        row_elems,
                    )?;
                    cp.memory_barrier()?;
                }
            }

            if !ffn_norm_fused {
                ik::rms_norm(
                    ctx,
                    cp,
                    &sc.ffn_normed,
                    Some(&layer.ffn_norm),
                    &sc.ffn_normed,
                    rows,
                    h,
                    self.cfg.rms_eps,
                )?;
                cp.memory_barrier()?;
            }

            linear(ctx, cp, &layer.router, &sc.ffn_normed, &sc.router_logits, rows)?;
            cp.memory_barrier()?;
            ik::router(
                ctx,
                cp,
                &sc.router_logits,
                layer.exp_probs_b.as_ref(),
                &sc.exp_ids,
                &sc.exp_w,
                rows,
                self.cfg.n_expert,
                self.cfg.n_expert_used,
                self.cfg.route_scale,
            )?;
            cp.memory_barrier()?;

            let rows_k = rows * self.cfg.n_expert_used;
            if rows == 1 {
                ik::moe_gate_up_situ(
                    ctx,
                    cp,
                    &sc.ffn_normed,
                    &layer.gate_up_exps,
                    &sc.exp_ids,
                    layer.gate_up_scale.as_ref(),
                    &sc.situ_out,
                    rows_k,
                    self.cfg.ff_expert,
                    self.cfg.n_expert_used,
                    self.cfg.situ_beta,
                    self.cfg.situ_linear_beta,
                )?;
                cp.memory_barrier()?;
                ik::moe_down_sum(
                    ctx,
                    cp,
                    &sc.situ_out,
                    &layer.down_exps,
                    &sc.exp_ids,
                    &sc.exp_w,
                    layer.down_scale.as_ref(),
                    &sc.moe_out,
                    rows,
                    self.cfg.n_expert_used,
                    h,
                )?;
            } else {
                // Grouped multi-row MoE (up to 4 tokens per expert tile) for prefill & verify.
                ik::moe_bucket(
                    ctx,
                    cp,
                    &sc.exp_ids,
                    &sc.sorted_rk,
                    &sc.moe_tiles,
                    &sc.moe_n_tiles,
                    rows_k,
                    self.cfg.n_expert,
                )?;
                cp.memory_barrier()?;
                // Worst-case tile count for M_TILE=8: min(rows_k, (rows_k + 7*E) / 8).
                let max_tiles = rows_k.min((rows_k + 3 * self.cfg.n_expert) / 4);
                ik::moe_gu_m4(
                    ctx,
                    cp,
                    &sc.ffn_normed,
                    &layer.gate_up_exps,
                    &sc.sorted_rk,
                    &sc.moe_tiles,
                    &sc.moe_n_tiles,
                    layer.gate_up_scale.as_ref(),
                    &sc.situ_out,
                    max_tiles,
                    self.cfg.ff_expert,
                    self.cfg.n_expert_used,
                    self.cfg.situ_beta,
                    self.cfg.situ_linear_beta,
                )?;
                cp.memory_barrier()?;
                ik::moe_down_m4(
                    ctx,
                    cp,
                    &sc.situ_out,
                    &layer.down_exps,
                    &sc.sorted_rk,
                    &sc.moe_tiles,
                    &sc.moe_n_tiles,
                    &sc.down_out,
                    max_tiles,
                    h,
                )?;
                cp.memory_barrier()?;
                ik::moe_sum(
                    ctx,
                    cp,
                    &sc.down_out,
                    &sc.exp_ids,
                    &sc.exp_w,
                    layer.down_scale.as_ref(),
                    &sc.moe_out,
                    rows,
                    self.cfg.n_expert_used,
                    h,
                )?;
            }
            cp.memory_barrier()?;
            ik::short_conv_fused(
                ctx,
                cp,
                &sc.moe_out,
                &layer.ffn_post_conv,
                None,
                &slot_cache.ffn_hist,
                &sc.cur,
                tr.map(|t| &t[3]),
                Some(&sc.ffn_inp),
                rows,
                h,
                self.cfg.sconv_l_cache,
                0,
            )?;
            cp.memory_barrier()?;
        }

        // Output mix + final RMSNorm (needed for logits or DSpark capture of final norm).
        let need_final_norm = compute_logits
            || (capture_dspark && target_layers.contains(&self.cfg.n_visits));
        if need_final_norm {
            let (off, n_states) = sc.output_ptrs;
            let ptrs = sc.ptrs_table.view(off, &[32])?;
            ik::attn_res_mix(
                ctx,
                cp,
                &ptrs,
                &self.output_res_query,
                &sc.ffn_normed,
                rows,
                h,
                n_states,
                1,
                1e-6,
            )?;
            cp.memory_barrier()?;
            ik::rms_norm(
                ctx,
                cp,
                &sc.ffn_normed,
                Some(&self.output_norm),
                &sc.ffn_normed,
                rows,
                h,
                self.cfg.rms_eps,
            )?;
            if capture_dspark
                && let Some(cap_buf) = &sc.dspark_cap
                && let Some(ti) =
                    target_layers.iter().position(|&l| l == self.cfg.n_visits)
            {
                cp.memory_barrier()?;
                let dst = cap_buf.view(ti * sc.max_rows * h, &[rows, h])?;
                ik::copy_f32(ctx, cp, &sc.ffn_normed, &dst, row_elems)?;
            }
            if compute_logits {
                cp.memory_barrier()?;
                if verify_mode
                    && let Some(vs) = &sc.verify
                {
                    let all_rows = sc.ffn_normed.view(0, &[rows, h])?;
                    linear(ctx, cp, &self.output, &all_rows, &vs.logits_raw, rows)?;
                    cp.memory_barrier()?;
                    ik::softcap(
                        ctx,
                        cp,
                        &vs.logits_raw,
                        &vs.logits,
                        rows,
                        self.cfg.vocab_padded,
                        self.cfg.vocab_unpadded,
                        self.cfg.final_logit_softcapping,
                    )?;
                } else {
                    let last_row = sc.ffn_normed.view((rows - 1) * h, &[1, h])?;
                    linear(ctx, cp, &self.output, &last_row, &sc.logits_raw, 1)?;
                    cp.memory_barrier()?;
                    ik::softcap(
                        ctx,
                        cp,
                        &sc.logits_raw,
                        &sc.logits,
                        1,
                        self.cfg.vocab_padded,
                        self.cfg.vocab_unpadded,
                        self.cfg.final_logit_softcapping,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Runs `rows` tokens (from `sc.ids[0..rows]`, or pre-filled embeddings in `sc.cur`
    /// when `raw_embd` is true) at position `state.pos` in a single GPU submission.
    /// If `compute_logits` is true, writes the last row's softcapped logits to `sc.logits`.
    /// Optionally captures target layer hidden states for DSpark.
    pub fn forward_rows(
        &self,
        ctx: &MetalContext,
        state: &Inkling2State,
        sc: &Scratch,
        rows: usize,
        raw_embd: bool,
        compute_logits: bool,
        mut dspark_capture: Option<&mut Vec<Vec<Vec<f32>>>>,
    ) -> Result<()> {
        let capture = dspark_capture.is_some();
        let cp = ctx.begin()?;
        self.encode_forward_rows(
            ctx,
            &cp,
            state,
            sc,
            rows,
            None,
            raw_embd,
            compute_logits,
            capture,
            false,
        )?;
        cp.commit_wait()?;

        if let (Some(cap), Some(cap_buf), Some(ds)) =
            (dspark_capture.as_deref_mut(), &sc.dspark_cap, &self.dspark)
        {
            let h = self.cfg.hidden;
            let n_targets = ds.cfg.target_layers.len();
            let all = cap_buf.to_f32()?;
            cap.clear();
            cap.resize_with(rows, || vec![Vec::new(); n_targets]);
            for ti in 0..n_targets {
                let base = ti * sc.max_rows * h;
                for r in 0..rows {
                    cap[r][ti] = all[base + r * h..base + (r + 1) * h].to_vec();
                }
            }
        }

        Ok(())
    }

    /// Profiles one decode step (averaged over `iters` iterations after 2 warmup runs)
    /// broken down by major GPU component so we can optimize the exact bottleneck.
    pub fn profile_decode(
        &self,
        ctx: &MetalContext,
        state: &mut Inkling2State,
        sc: &Scratch,
        iters: usize,
    ) -> Result<()> {
        let h = self.cfg.hidden;
        let k_exp = self.cfg.n_expert_used;
        let ff = self.cfg.ff_expert;
        let mut t_moe_gu = 0.0f64;
        let mut t_moe_down = 0.0f64;
        let mut t_lm = 0.0f64;
        let mut t_dense = 0.0f64;
        let mut t_loop = 0.0f64;
        let mut t_full = 0.0f64;

        // Warmup full pass.
        for _ in 0..2 {
            self.forward_rows(ctx, state, sc, 1, false, true, None)?;
        }
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            self.forward_rows(ctx, state, sc, 1, false, true, None)?;
            t_full += t0.elapsed().as_secs_f64() * 1000.0;

            // Isolate 48 visits of MoE gate_up_situ.
            let t1 = std::time::Instant::now();
            let cp = ctx.begin()?;
            for &(_, ip, _) in &self.schedule {
                let layer = &self.layers[ip];
                ik::moe_gate_up_situ(
                    ctx,
                    &cp,
                    &sc.ffn_normed,
                    &layer.gate_up_exps,
                    &sc.exp_ids,
                    layer.gate_up_scale.as_ref(),
                    &sc.situ_out,
                    k_exp,
                    ff,
                    k_exp,
                    self.cfg.situ_beta,
                    self.cfg.situ_linear_beta,
                )?;
            }
            cp.commit_wait()?;
            t_moe_gu += t1.elapsed().as_secs_f64() * 1000.0;

            // Isolate 48 visits of MoE down_sum.
            let t2 = std::time::Instant::now();
            let cp = ctx.begin()?;
            for &(_, ip, _) in &self.schedule {
                let layer = &self.layers[ip];
                ik::moe_down_sum(
                    ctx,
                    &cp,
                    &sc.situ_out,
                    &layer.down_exps,
                    &sc.exp_ids,
                    &sc.exp_w,
                    layer.down_scale.as_ref(),
                    &sc.moe_out,
                    1,
                    k_exp,
                    h,
                )?;
            }
            cp.commit_wait()?;
            t_moe_down += t2.elapsed().as_secs_f64() * 1000.0;

            // Isolate 48 visits of dense attention + router projections.
            let td = std::time::Instant::now();
            let cp = ctx.begin()?;
            for &(_, ip, _) in &self.schedule {
                let layer = &self.layers[ip];
                linear_multi(
                    ctx,
                    &cp,
                    &sc.attn_in[0],
                    &[
                        (&layer.wq, &sc.q),
                        (&layer.wk, &sc.k_raw),
                        (&layer.wv, &sc.v_raw),
                        (&layer.attn_rel, &sc.r_proj),
                        (&layer.attn_sink_proj, &sc.sinks),
                        (&layer.attn_gate, &sc.attn_gate),
                    ],
                    1,
                )?;
                linear(ctx, &cp, &layer.wo, &sc.attn_out, &sc.wo_out, 1)?;
                linear(ctx, &cp, &layer.router, &sc.ffn_normed, &sc.router_logits, 1)?;
            }
            cp.commit_wait()?;
            t_dense += td.elapsed().as_secs_f64() * 1000.0;

            // Isolate 24 loop visits of loop controllers.
            let tl = std::time::Instant::now();
            let cp = ctx.begin()?;
            let rank = self.cfg.loop_gate_rank;
            for &(_, ip, pass_idx) in &self.schedule {
                let layer = &self.layers[ip];
                if pass_idx > 0
                    && let Some(lc) = &layer.loop_core
                {
                    for j in 0..4 {
                        let msg_j = sc.msg_out.view(j * h, &[1, h])?;
                        let mid_0 = sc.ctrl_mid.view(0, &[1, rank])?;
                        ik::ctrl_down_f32(
                            ctx,
                            &cp,
                            &sc.attn_in[0],
                            &msg_j,
                            &lc.ctrl[j].down,
                            &mid_0,
                            1,
                            h,
                            rank,
                            1e-6,
                        )?;
                        cp.memory_barrier()?;
                        ik::ctrl_blend_f32(
                            ctx,
                            &cp,
                            &sc.attn_in[0],
                            &msg_j,
                            &mid_0,
                            &lc.ctrl[j].gain,
                            &lc.ctrl[j].forget,
                            &sc.attn_in[0],
                            1,
                            h,
                            rank,
                        )?;
                    }
                }
            }
            cp.commit_wait()?;
            t_loop += tl.elapsed().as_secs_f64() * 1000.0;

            // Isolate LM head (output.weight + softcap).
            let t3 = std::time::Instant::now();
            let cp = ctx.begin()?;
            let last_row = sc.ffn_normed.view(0, &[1, h])?;
            linear(ctx, &cp, &self.output, &last_row, &sc.logits_raw, 1)?;
            cp.memory_barrier()?;
            ik::softcap(
                ctx,
                &cp,
                &sc.logits_raw,
                &sc.logits,
                1,
                self.cfg.vocab_padded,
                self.cfg.vocab_unpadded,
                self.cfg.final_logit_softcapping,
            )?;
            cp.commit_wait()?;
            t_lm += t3.elapsed().as_secs_f64() * 1000.0;
        }
        let n = iters as f64;
        let full = t_full / n;
        let gu = t_moe_gu / n;
        let dn = t_moe_down / n;
        let dense = t_dense / n;
        let lp = t_loop / n;
        let lm = t_lm / n;
        let misc = (full - gu - dn - dense - lp - lm).max(0.0);
        println!(
            "profile (1 tok, avg of {iters}): full={full:.2}ms ({:.1} tok/s) | moe_gu={gu:.2}ms ({:.0}%) | moe_down={dn:.2}ms ({:.0}%) | dense={dense:.2}ms ({:.0}%) | loop={lp:.2}ms ({:.0}%) | lm={lm:.2}ms ({:.0}%) | misc={misc:.2}ms ({:.0}%)",
            1000.0 / full,
            100.0 * gu / full,
            100.0 * dn / full,
            100.0 * dense / full,
            100.0 * lp / full,
            100.0 * lm / full,
            100.0 * misc / full,
        );

        // Also profile 4-row speculative verify + DSpark inject/propose when enabled.
        if sc.max_rows >= 4 {
            let v_iters = iters.min(6);
            let mut tv_full = 0.0;
            let mut tv_ds = 0.0;
            sc.stage_tokens(&[1, 2, 3, 4])?;
            // Warmup 1 pass
            {
                let cp = ctx.begin()?;
                self.encode_forward_rows(
                    ctx,
                    &cp,
                    state,
                    sc,
                    4,
                    None,
                    false,
                    true,
                    self.dspark.is_some(),
                    sc.verify.is_some(),
                )?;
                cp.commit_wait()?;
            }
            for _ in 0..v_iters {
                let t0 = std::time::Instant::now();
                let cp = ctx.begin()?;
                self.encode_forward_rows(
                    ctx,
                    &cp,
                    state,
                    sc,
                    4,
                    None,
                    false,
                    true,
                    self.dspark.is_some(),
                    sc.verify.is_some(),
                )?;
                cp.commit_wait()?;
                tv_full += t0.elapsed().as_secs_f64() * 1000.0;

                if let (Some(ds), Some(ds_state), Some(cap_buf)) =
                    (&self.dspark, &mut state.dspark, &sc.dspark_cap)
                {
                    let anchor_embd = sc.logits_raw.view(0, &[h])?;
                    ds.prepare_propose(1)?;
                    let td = std::time::Instant::now();
                    let cp = ctx.begin()?;
                    ds.encode_inject(ctx, &cp, ds_state, cap_buf, sc.max_rows, 4, 0)?;
                    cp.memory_barrier()?;
                    ds.encode_propose(
                        ctx,
                        &cp,
                        ds_state,
                        &anchor_embd,
                        &self.output,
                        self.cfg.vocab_unpadded,
                        4,
                        3,
                    )?;
                    cp.commit_wait()?;
                    tv_ds += td.elapsed().as_secs_f64() * 1000.0;
                }
            }
            let vn = v_iters as f64;
            let vf = tv_full / vn;
            let ds_ms = tv_ds / vn;
            let step_ms = vf + ds_ms;
            println!(
                "profile (verify 4 tok, avg of {v_iters}): verify={vf:.2}ms | dspark(inj+prop3)={ds_ms:.2}ms | step={step_ms:.2}ms (at 3.27 tok/step -> {:.1} tok/s)",
                3270.0 / step_ms,
            );
        }
        Ok(())
    }
}
