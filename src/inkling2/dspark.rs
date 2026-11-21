//! DSpark speculative drafter for Inkling 2 Mini (`dspark-Inkling-2-Mini-Q8_0.gguf`
//! or `dspark-Inkling-2-Mini-BF16.gguf`, `general.architecture = "dflash"`).
//!
//! Follows `ggml-org/llama.cpp` PR #32218 (`llama_model_dflash::graph_inkling2`):
//! - Zero-copy read-only `Mapped` weights (frees ~225 MB of anonymous CPU RAM).
//! - Persistent GPU scratch (`DsparkScratch`) — zero per-step Vulkan allocations.
//! - Fused GPU DFlare FFN (`wo + residual + ffn_norm + gate/up + silu_mul + down + residual`
//!   in 1 GPU pass per layer).
//! - Entire recurrent Markov head (`1..block` slots: `markov_w1` embed -> `joint_proj` ->
//!   `dspark_gru` -> `hidden_out` -> `add3` -> `target_output` (r2 Q4_K) -> greedy GPU argmax)
//!   runs in **one single GPU command buffer** with zero CPU round-trips between draft slots!

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use super::config::Config;
use crate::gguf::Gguf;
use crate::glm53::model::Mapped;
use crate::kernels::glm53::{self as k, QWeight};
use crate::kernels::inkling2 as ik;
use crate::kernels::sample::{SamplerScratch, SamplingParams, sample_f32};
use crate::metal::MetalContext;
use crate::tensor::{DType, Tensor};

/// Maximum drafts per block (`block_size - 1` = 7).
pub const MAX_DRAFTS: usize = 7;

#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub n_layer: usize,
    pub hidden: usize,
    pub ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub block_size: usize,
    pub markov_rank: usize,
    pub window: usize,
    pub swa: Vec<bool>,
    pub target_layers: Vec<usize>,
    pub final_logit_softcapping: f32,
}

impl DsparkConfig {
    pub fn from_gguf(g: &Gguf, target: &Config) -> Result<Self> {
        ensure!(
            g.str("general.architecture")? == "dflash",
            "DSpark GGUF general.architecture must be \"dflash\""
        );
        ensure!(
            g.tensor("dspark_hidden_out.weight").is_some(),
            "DSpark GGUF is missing dspark_hidden_out.weight (not an Inkling 2 DSpark file)"
        );
        let n_layer = g.u64("dflash.block_count")? as usize;
        let hidden = g.u64("dflash.embedding_length")? as usize;
        let ff = g.u64("dflash.feed_forward_length")? as usize;
        let n_head = g.u64("dflash.attention.head_count")? as usize;
        let n_head_kv = g.u64("dflash.attention.head_count_kv")? as usize;
        let head_dim = g.u64("dflash.attention.key_length")? as usize;
        let rope_theta = g.f32("dflash.rope.freq_base").unwrap_or(1_000_000.0);
        let rms_eps = g.f32("dflash.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let block_size = g.u64("dflash.block_size").unwrap_or(8) as usize;
        let window = g.u64("dflash.attention.sliding_window").unwrap_or(1024) as usize;
        let final_logit_softcapping = g
            .f32("dflash.final_logit_softcapping")
            .unwrap_or(target.final_logit_softcapping);

        ensure!(
            hidden == target.hidden,
            "DSpark hidden {hidden} != target {}",
            target.hidden
        );
        ensure!(block_size >= 2 && block_size <= 16, "DSpark block_size={block_size}");

        let markov_rank = g
            .tensor("markov_w1.weight")
            .context("DSpark missing markov_w1.weight")?
            .dims[0] as usize;

        let swa = match g.get("dflash.attention.sliding_window_pattern") {
            Some(crate::gguf::Value::Array(a)) => a
                .iter()
                .map(|v| match v {
                    crate::gguf::Value::Bool(b) => Ok(*b),
                    other => anyhow::bail!("invalid swa entry {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            _ => vec![false; n_layer],
        };
        ensure!(swa.len() == n_layer, "DSpark swa pattern length mismatch");

        let target_layers = match g.get("dflash.target_layers") {
            Some(crate::gguf::Value::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_u64().map(|x| x as usize).context("non-integer target_layers")
                })
                .collect::<Result<Vec<_>>>()?,
            _ => anyhow::bail!("DSpark missing dflash.target_layers"),
        };
        ensure!(
            !target_layers.is_empty()
                && target_layers.iter().all(|&l| l <= target.n_visits),
            "DSpark target_layers {target_layers:?} out of range for {} visits",
            target.n_visits
        );

        Ok(Self {
            n_layer,
            hidden,
            ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_theta,
            rms_eps,
            block_size,
            markov_rank,
            window,
            swa,
            target_layers,
            final_logit_softcapping,
        })
    }
}

/// Finds a DSpark GGUF (`dspark-Inkling-2-Mini-*.gguf` or any `dflash` GGUF with
/// `dspark_hidden_out.weight`) next to `model_path` (preferring Q8_0 over BF16).
pub fn find_beside(model_path: &Path, target: &Config) -> Result<Option<PathBuf>> {
    let Some(dir) = model_path.parent() else {
        return Ok(None);
    };
    let mut candidates = Vec::new();
    for search_dir in [dir.to_path_buf(), dir.join("dspark")] {
        let Ok(entries) = std::fs::read_dir(&search_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p == model_path || p.extension().and_then(|e| e.to_str()) != Some("gguf")
            {
                continue;
            }
            let Ok(g) = Gguf::open(&p) else {
                continue;
            };
            if DsparkConfig::from_gguf(&g, target).is_ok() {
                candidates.push((p, g.file_len));
            }
        }
    }
    candidates.sort_by_key(|(_, sz)| *sz);
    Ok(candidates.into_iter().next().map(|(p, _)| p))
}

/// One DFlare draft layer's weights (100% GPU weights + GPU norms).
pub struct DsparkLayerW {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub dflare_norm: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub k_ctx_norm: Tensor,
    pub wq: QWeight,
    pub wk: QWeight,
    pub wv: QWeight,
    pub wo: QWeight,
    pub wk_ctx: QWeight,
    pub wv_ctx: QWeight,
    pub ffn_gate: QWeight,
    pub ffn_up: QWeight,
    pub ffn_down: QWeight,
}

/// Pre-allocated GPU scratch for DSpark (zero per-step Vulkan allocations).
struct DsparkScratch {
    max_block: usize,
    cur: Tensor,         // F32 [max_block, hidden]
    x_h: Tensor,         // F32 [max_block, hidden]
    post_attn: Tensor,   // F32 [max_block, hidden]
    q: Tensor,           // F32 [max_block, q_w]
    k: Tensor,           // F32 [max_block, kv_w]
    v: Tensor,           // F32 [max_block, kv_w]
    attn_out: Tensor,    // F32 [max_block, q_w]
    wo_out: Tensor,      // F32 [max_block, hidden]
    gate: Tensor,        // F32 [max_block, ff]
    up: Tensor,          // F32 [max_block, ff]
    mid: Tensor,         // F32 [max_block, ff]
    down: Tensor,        // F32 [max_block, hidden]
    hidden_norm: Tensor, // F32 [max_block, hidden]
    rnn_state: Tensor,   // F32 [rank]
    joint_in: Tensor,    // F32 [1, 2*rank + 2*hidden]
    joint_out: Tensor,   // F32 [1, 3*rank]
    latent: Tensor,      // F32 [1, rank]
    corr: Tensor,        // F32 [1, hidden]
    combined: Tensor,    // F32 [1, hidden]
    logits_raw: Tensor,  // F32 [vocab_padded]
    best_toks: Tensor,   // U32 [max_block] (slot 0 = anchor, 1..block = drafts)
    sampler: SamplerScratch,
}

/// Loaded Inkling 2 Mini DSpark drafter (100% GPU).
pub struct Inkling2Dspark {
    pub cfg: DsparkConfig,
    pub bytes: usize,
    _mapped: Mapped,
    output_norm: Tensor,
    pub mask_embd: Tensor,
    /// Softmax-normalized fusion weights on GPU: `[n_layer + 1, n_target]` F32.
    pub fuse_weights: Tensor,
    pub dspark_norm: Tensor,
    markov_w1: QWeight,
    joint_proj: QWeight,
    joint_proj_b: Tensor,
    hidden_out: QWeight,
    hidden_out_b: Tensor,
    pub layers: Vec<DsparkLayerW>,
    scratch: DsparkScratch,
}

/// Per-session DSpark GPU KV cache (stores context KV from target features in F16).
pub struct DsparkState {
    pub k_ctx: Vec<Tensor>, // per layer: F16 [capacity, n_head_kv * head_dim]
    pub v_ctx: Vec<Tensor>,
    pub f_rnn_last: Tensor, // F32 [hidden]
    pub pos: usize,
    pub capacity: usize,
    pub kv_w: usize,
}

impl DsparkState {
    pub fn new(ctx: &MetalContext, cfg: &DsparkConfig, capacity: usize) -> Result<Self> {
        let cap = capacity.max(64);
        let kv_w = cfg.n_head_kv * cfg.head_dim;
        let mut k_ctx = Vec::with_capacity(cfg.n_layer);
        let mut v_ctx = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            k_ctx.push(Tensor::zeros(ctx, &[cap, kv_w], DType::F16)?);
            v_ctx.push(Tensor::zeros(ctx, &[cap, kv_w], DType::F16)?);
        }
        Ok(Self {
            k_ctx,
            v_ctx,
            f_rnn_last: Tensor::zeros(ctx, &[cfg.hidden], DType::F32)?,
            pos: 0,
            capacity: cap,
            kv_w,
        })
    }

    pub fn reset(&mut self) {
        self.f_rnn_last.zero_fill();
        self.pos = 0;
    }

    pub fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        let mut new_cap = self.capacity.max(64);
        while new_cap < tokens {
            new_cap *= 2;
        }
        let copy_bytes = self.pos * self.kv_w * 2;
        for l in 0..self.k_ctx.len() {
            let nk = Tensor::zeros(ctx, &[new_cap, self.kv_w], DType::F16)?;
            let nv = Tensor::zeros(ctx, &[new_cap, self.kv_w], DType::F16)?;
            if copy_bytes > 0 {
                // SAFETY: both tensors are host-visible shared buffers, GPU idle.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        self.k_ctx[l].contents_ptr(),
                        nk.contents_ptr(),
                        copy_bytes,
                    );
                    core::ptr::copy_nonoverlapping(
                        self.v_ctx[l].contents_ptr(),
                        nv.contents_ptr(),
                        copy_bytes,
                    );
                }
            }
            self.k_ctx[l] = nk;
            self.v_ctx[l] = nv;
        }
        self.capacity = new_cap;
        Ok(())
    }
}

fn read_f32_tensor(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info =
        g.tensor(name).with_context(|| format!("DSpark missing tensor {name}"))?;
    let mut file = std::fs::File::open(&g.path)?;
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(info.offset))?;
    let mut raw = vec![0u8; info.bytes as usize];
    file.read_exact(&mut raw)?;
    let n = info.row_len() * info.rows();
    let mut out = vec![0.0f32; n];
    crate::gguf::dequantize(info.ty, &raw, &mut out)?;
    Ok(out)
}

fn load_f32_gpu(ctx: &MetalContext, g: &Gguf, name: &str) -> Result<Tensor> {
    let v = read_f32_tensor(g, name)?;
    let len = v.len();
    Tensor::from_f32(ctx, &v, &[len])
}

impl Inkling2Dspark {
    pub fn load(ctx: &MetalContext, path: &Path, target: &Config) -> Result<Self> {
        let g = Gguf::open(path)?;
        let cfg = DsparkConfig::from_gguf(&g, target)?;
        let mapped = Mapped::map(ctx, &g)?;

        let output_norm = load_f32_gpu(ctx, &g, "output_norm.weight")?;
        let mask_embd = load_f32_gpu(ctx, &g, "mask_embd.weight")?;
        let fuse_raw = read_f32_tensor(&g, "dspark_fuse.weight")?;
        let dspark_norm = load_f32_gpu(ctx, &g, "dspark_norm.weight")?;
        let markov_w1 = mapped.q("markov_w1.weight")?;
        let joint_proj = mapped.q("joint_proj.weight")?;
        let joint_proj_b = load_f32_gpu(ctx, &g, "joint_proj.bias")?;
        let hidden_out = mapped.q("dspark_hidden_out.weight")?;
        let hidden_out_b = load_f32_gpu(ctx, &g, "dspark_hidden_out.bias")?;

        let nt = cfg.target_layers.len();
        let mut fuse_flat = Vec::with_capacity((cfg.n_layer + 1) * nt);
        for row in 0..=cfg.n_layer {
            let slice = &fuse_raw[row * nt..(row + 1) * nt];
            let mx = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut w: Vec<f32> = slice.iter().map(|&v| (v - mx).exp()).collect();
            let s: f32 = w.iter().sum();
            for v in &mut w {
                *v /= s;
            }
            fuse_flat.extend_from_slice(&w);
        }
        let fuse_weights = Tensor::from_f32(ctx, &fuse_flat, &[cfg.n_layer + 1, nt])?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{i}.{s}");
            layers.push(DsparkLayerW {
                attn_norm: load_f32_gpu(ctx, &g, &n("attn_norm.weight"))?,
                ffn_norm: load_f32_gpu(ctx, &g, &n("ffn_norm.weight"))?,
                dflare_norm: load_f32_gpu(ctx, &g, &n("dflare_norm.weight"))?,
                q_norm: load_f32_gpu(ctx, &g, &n("attn_q_norm.weight"))?,
                k_norm: load_f32_gpu(ctx, &g, &n("attn_k_norm.weight"))?,
                k_ctx_norm: load_f32_gpu(ctx, &g, &n("attn_k_ctx_norm.weight"))?,
                wq: mapped.q(&n("attn_q.weight"))?,
                wk: mapped.q(&n("attn_k.weight"))?,
                wv: mapped.q(&n("attn_v.weight"))?,
                wo: mapped.q(&n("attn_output.weight"))?,
                wk_ctx: mapped.q(&n("attn_k_ctx.weight"))?,
                wv_ctx: mapped.q(&n("attn_v_ctx.weight"))?,
                ffn_gate: mapped.q(&n("ffn_gate.weight"))?,
                ffn_up: mapped.q(&n("ffn_up.weight"))?,
                ffn_down: mapped.q(&n("ffn_down.weight"))?,
            });
        }

        let max_block = cfg.block_size.max(8);
        let h = cfg.hidden;
        let qw = cfg.n_head * cfg.head_dim;
        let kw = cfg.n_head_kv * cfg.head_dim;
        let rank = cfg.markov_rank;
        let scratch = DsparkScratch {
            max_block,
            cur: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
            x_h: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
            post_attn: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
            q: Tensor::zeros(ctx, &[max_block, qw], DType::F32)?,
            k: Tensor::zeros(ctx, &[max_block, kw], DType::F32)?,
            v: Tensor::zeros(ctx, &[max_block, kw], DType::F32)?,
            attn_out: Tensor::zeros(ctx, &[max_block, qw], DType::F32)?,
            wo_out: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
            gate: Tensor::zeros(ctx, &[max_block, cfg.ff], DType::F32)?,
            up: Tensor::zeros(ctx, &[max_block, cfg.ff], DType::F32)?,
            mid: Tensor::zeros(ctx, &[max_block, cfg.ff], DType::F32)?,
            down: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
            hidden_norm: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
            rnn_state: Tensor::zeros(ctx, &[rank], DType::F32)?,
            joint_in: Tensor::zeros(ctx, &[1, 2 * rank + 2 * h], DType::F32)?,
            joint_out: Tensor::zeros(ctx, &[1, 3 * rank], DType::F32)?,
            latent: Tensor::zeros(ctx, &[1, rank], DType::F32)?,
            corr: Tensor::zeros(ctx, &[1, h], DType::F32)?,
            combined: Tensor::zeros(ctx, &[1, h], DType::F32)?,
            logits_raw: Tensor::zeros(ctx, &[target.vocab_padded], DType::F32)?,
            best_toks: Tensor::zeros(ctx, &[max_block], DType::U32)?,
            sampler: SamplerScratch::new(ctx, target.vocab_unpadded)?,
        };

        Ok(Self {
            cfg,
            bytes: g.file_len as usize,
            _mapped: mapped,
            output_norm,
            mask_embd,
            fuse_weights,
            dspark_norm,
            markov_w1,
            joint_proj,
            joint_proj_b,
            hidden_out,
            hidden_out_b,
            layers,
            scratch,
        })
    }

    fn dispatch_linear(
        ctx: &MetalContext,
        pass: &crate::metal::ComputePass<'_>,
        w: &QWeight,
        x: &Tensor,
        y: &Tensor,
        rows: usize,
    ) -> Result<()> {
        if rows <= k::gemv_max_rows(w.ty) {
            k::gemv(ctx, pass, w, x, y, rows)
        } else {
            k::gemm(ctx, pass, w, x, y, rows)
        }
    }

    fn dispatch_linear2(
        ctx: &MetalContext,
        pass: &crate::metal::ComputePass<'_>,
        w0: &QWeight,
        w1: &QWeight,
        x: &Tensor,
        y0: &Tensor,
        y1: &Tensor,
        rows: usize,
    ) -> Result<()> {
        if w0.ty == w1.ty && rows <= k::gemv_max_rows(w0.ty) {
            k::gemv_sets(ctx, pass, &[(w0, y0), (w1, y1)], x, rows)
        } else {
            Self::dispatch_linear(ctx, pass, w0, x, y0, rows)?;
            Self::dispatch_linear(ctx, pass, w1, x, y1, rows)
        }
    }

    fn dispatch_linear3(
        ctx: &MetalContext,
        pass: &crate::metal::ComputePass<'_>,
        w0: &QWeight,
        w1: &QWeight,
        w2: &QWeight,
        x: &Tensor,
        y0: &Tensor,
        y1: &Tensor,
        y2: &Tensor,
        rows: usize,
    ) -> Result<()> {
        if w0.ty == w1.ty && w0.ty == w2.ty && rows <= k::gemv_max_rows(w0.ty) {
            k::gemv_sets(ctx, pass, &[(w0, y0), (w1, y1), (w2, y2)], x, rows)
        } else {
            Self::dispatch_linear(ctx, pass, w0, x, y0, rows)?;
            Self::dispatch_linear(ctx, pass, w1, x, y1, rows)?;
            Self::dispatch_linear(ctx, pass, w2, x, y2, rows)
        }
    }

    /// Encodes 100% GPU feature fusion + K_ctx/V_ctx projection + K_ctx norm/RoPE + F16 KV store
    /// + `f_rnn_last` update into `pass` for `rows <= scratch.max_block` (zero CPU round-trips!).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_inject(
        &self,
        ctx: &MetalContext,
        pass: &crate::metal::ComputePass<'_>,
        state: &mut DsparkState,
        cap: &Tensor,
        cap_stride: usize,
        rows: usize,
        pos0: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let sc = &self.scratch;
        ensure!(rows <= sc.max_block, "encode_inject rows {rows} > {}", sc.max_block);
        let h = self.cfg.hidden;
        let hd = self.cfg.head_dim;
        let nkv = self.cfg.n_head_kv;
        let kv_w = nkv * hd;
        let nt = self.cfg.target_layers.len();
        let xv = sc.x_h.view(0, &[rows, h])?;
        let kv = sc.k.view(0, &[rows, kv_w])?;
        let vv = sc.v.view(0, &[rows, kv_w])?;

        for il in 0..self.cfg.n_layer {
            let l = &self.layers[il];
            let fw = self.fuse_weights.view(il * nt, &[nt])?;
            ik::dspark_fuse_norm(
                ctx,
                pass,
                cap,
                &fw,
                &l.dflare_norm,
                &xv,
                rows,
                h,
                cap_stride,
                nt,
                self.cfg.rms_eps,
                0,
            )?;
            pass.memory_barrier()?;
            Self::dispatch_linear2(ctx, pass, &l.wk_ctx, &l.wv_ctx, &xv, &kv, &vv, rows)?;
            pass.memory_barrier()?;
            ik::dspark_qk_rope(
                ctx,
                pass,
                &kv,
                &l.k_ctx_norm,
                Some(&vv),
                Some(&state.k_ctx[il]),
                Some(&state.v_ctx[il]),
                rows,
                nkv,
                hd,
                pos0,
                self.cfg.rms_eps,
                self.cfg.rope_theta,
            )?;
            pass.memory_barrier()?;
        }

        let fw_rnn = self.fuse_weights.view(self.cfg.n_layer * nt, &[nt])?;
        ik::dspark_fuse_norm(
            ctx,
            pass,
            cap,
            &fw_rnn,
            &self.dspark_norm,
            &state.f_rnn_last,
            1,
            h,
            cap_stride,
            nt,
            self.cfg.rms_eps,
            rows - 1,
        )?;
        state.pos = pos0 + rows;
        Ok(())
    }

    /// Injects arbitrary `rows` (including large prefill chunks `> max_block`) on GPU in 1 pass.
    pub fn inject_gpu(
        &self,
        ctx: &MetalContext,
        state: &mut DsparkState,
        cap: &Tensor,
        cap_stride: usize,
        rows: usize,
        pos0: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        state.ensure_capacity(ctx, pos0 + rows)?;
        if rows <= self.scratch.max_block {
            let pass = ctx.begin()?;
            self.encode_inject(ctx, &pass, state, cap, cap_stride, rows, pos0)?;
            pass.commit_wait()?;
            return Ok(());
        }
        let h = self.cfg.hidden;
        let hd = self.cfg.head_dim;
        let nkv = self.cfg.n_head_kv;
        let kv_w = nkv * hd;
        let nt = self.cfg.target_layers.len();
        let tx = Tensor::zeros(ctx, &[rows, h], DType::F32)?;
        let tk = Tensor::zeros(ctx, &[rows, kv_w], DType::F32)?;
        let tv = Tensor::zeros(ctx, &[rows, kv_w], DType::F32)?;
        let pass = ctx.begin()?;
        for il in 0..self.cfg.n_layer {
            let l = &self.layers[il];
            let fw = self.fuse_weights.view(il * nt, &[nt])?;
            ik::dspark_fuse_norm(
                ctx,
                &pass,
                cap,
                &fw,
                &l.dflare_norm,
                &tx,
                rows,
                h,
                cap_stride,
                nt,
                self.cfg.rms_eps,
                0,
            )?;
            pass.memory_barrier()?;
            k::gemm(ctx, &pass, &l.wk_ctx, &tx, &tk, rows)?;
            k::gemm(ctx, &pass, &l.wv_ctx, &tx, &tv, rows)?;
            pass.memory_barrier()?;
            ik::dspark_qk_rope(
                ctx,
                &pass,
                &tk,
                &l.k_ctx_norm,
                Some(&tv),
                Some(&state.k_ctx[il]),
                Some(&state.v_ctx[il]),
                rows,
                nkv,
                hd,
                pos0,
                self.cfg.rms_eps,
                self.cfg.rope_theta,
            )?;
            pass.memory_barrier()?;
        }
        let fw_rnn = self.fuse_weights.view(self.cfg.n_layer * nt, &[nt])?;
        ik::dspark_fuse_norm(
            ctx,
            &pass,
            cap,
            &fw_rnn,
            &self.dspark_norm,
            &state.f_rnn_last,
            1,
            h,
            cap_stride,
            nt,
            self.cfg.rms_eps,
            rows - 1,
        )?;
        pass.commit_wait()?;
        state.pos = pos0 + rows;
        Ok(())
    }

    /// Host-side staging before [`encode_propose`] (must be called while GPU is idle).
    pub fn prepare_propose(&self, anchor_token: u32) -> Result<()> {
        let sc = &self.scratch;
        sc.rnn_state.zero_fill();
        sc.best_toks
            .view(0, &[1])?
            .write_bytes(bytemuck::cast_slice(&[anchor_token]))
    }

    /// Reads drafted tokens `[1..block]` after the command buffer containing [`encode_propose`] completes.
    pub fn read_drafts(&self, n_drafts: usize) -> Result<Vec<u32>> {
        let block = (1 + n_drafts.min(self.cfg.block_size - 1)).min(self.scratch.max_block);
        let all_toks = self.scratch.best_toks.to_u32()?;
        Ok(all_toks[1..block].to_vec())
    }

    /// Encodes ALL 3 DFlare layers (attn_norm, Q/K/V, Q/K norm+RoPE, GPU non-causal GQA,
    /// wo, residual, FFN) + `output_norm` + ALL `n_drafts` recurrent Markov head slots
    /// into a single `ComputePass` with ZERO CPU round-trips!
    #[allow(clippy::too_many_arguments)]
    pub fn encode_propose(
        &self,
        ctx: &MetalContext,
        pass: &crate::metal::ComputePass<'_>,
        state: &DsparkState,
        anchor_embd_normed: &Tensor,
        target_output: &QWeight,
        vocab_unpadded: usize,
        pos: usize,
        n_drafts: usize,
    ) -> Result<()> {
        let sc = &self.scratch;
        let block = (1 + n_drafts.min(self.cfg.block_size - 1)).min(sc.max_block);
        let h = self.cfg.hidden;
        let nh = self.cfg.n_head;
        let nkv = self.cfg.n_head_kv;
        let hd = self.cfg.head_dim;
        let qw = nh * hd;
        let kw = nkv * hd;
        let rank = self.cfg.markov_rank;

        let cur_v = sc.cur.view(0, &[block, h])?;
        let xh_v = sc.x_h.view(0, &[block, h])?;
        let post_v = sc.post_attn.view(0, &[block, h])?;
        let q_v = sc.q.view(0, &[block, qw])?;
        let k_v = sc.k.view(0, &[block, kw])?;
        let v_v = sc.v.view(0, &[block, kw])?;
        let ao_v = sc.attn_out.view(0, &[block, qw])?;
        let wo_v = sc.wo_out.view(0, &[block, h])?;
        let gate_v = sc.gate.view(0, &[block, self.cfg.ff])?;
        let up_v = sc.up.view(0, &[block, self.cfg.ff])?;
        let mid_v = sc.mid.view(0, &[block, self.cfg.ff])?;
        let down_v = sc.down.view(0, &[block, h])?;

        ik::dspark_init_cur(ctx, pass, anchor_embd_normed, &self.mask_embd, &cur_v, block, h)?;
        let ji_frnn = sc.joint_in.view(2 * rank + h, &[h])?;
        ik::copy_f32(ctx, pass, &state.f_rnn_last, &ji_frnn, h)?;
        pass.memory_barrier()?;

        for il in 0..self.cfg.n_layer {
            let l = &self.layers[il];
            ik::rms_norm(
                ctx,
                pass,
                &cur_v,
                Some(&l.attn_norm),
                &xh_v,
                block,
                h,
                self.cfg.rms_eps,
            )?;
            pass.memory_barrier()?;
            Self::dispatch_linear3(
                ctx, pass, &l.wq, &l.wk, &l.wv, &xh_v, &q_v, &k_v, &v_v, block,
            )?;
            pass.memory_barrier()?;
            ik::dspark_qk_rope(
                ctx,
                pass,
                &q_v,
                &l.q_norm,
                None,
                None,
                None,
                block,
                nh,
                hd,
                pos,
                self.cfg.rms_eps,
                self.cfg.rope_theta,
            )?;
            ik::dspark_qk_rope(
                ctx,
                pass,
                &k_v,
                &l.k_norm,
                None,
                None,
                None,
                block,
                nkv,
                hd,
                pos,
                self.cfg.rms_eps,
                self.cfg.rope_theta,
            )?;
            pass.memory_barrier()?;

            let ctx_start = if self.cfg.swa[il] && state.pos > self.cfg.window {
                state.pos - self.cfg.window
            } else {
                0
            };
            ik::dspark_attn(
                ctx,
                pass,
                &q_v,
                &state.k_ctx[il],
                &state.v_ctx[il],
                &k_v,
                &v_v,
                &ao_v,
                nh,
                nkv,
                hd,
                ctx_start,
                state.pos,
                block,
            )?;
            pass.memory_barrier()?;

            Self::dispatch_linear(ctx, pass, &l.wo, &ao_v, &wo_v, block)?;
            pass.memory_barrier()?;
            ik::add(ctx, pass, &cur_v, &wo_v, &post_v, block * h)?;
            pass.memory_barrier()?;
            ik::rms_norm(
                ctx,
                pass,
                &post_v,
                Some(&l.ffn_norm),
                &xh_v,
                block,
                h,
                self.cfg.rms_eps,
            )?;
            pass.memory_barrier()?;
            Self::dispatch_linear2(
                ctx,
                pass,
                &l.ffn_gate,
                &l.ffn_up,
                &xh_v,
                &gate_v,
                &up_v,
                block,
            )?;
            pass.memory_barrier()?;
            ik::silu_mul(ctx, pass, &gate_v, &up_v, &mid_v, block * self.cfg.ff)?;
            pass.memory_barrier()?;
            Self::dispatch_linear(ctx, pass, &l.ffn_down, &mid_v, &down_v, block)?;
            pass.memory_barrier()?;
            ik::add(ctx, pass, &post_v, &down_v, &cur_v, block * h)?;
            pass.memory_barrier()?;
        }

        let hn_v = sc.hidden_norm.view(0, &[block, h])?;
        let ji_rnn = sc.joint_in.view(0, &[rank])?;
        let ji_mk = sc.joint_in.view(rank, &[rank])?;
        let ji_h = sc.joint_in.view(2 * rank, &[h])?;
        let logits_unpadded = sc.logits_raw.view(0, &[vocab_unpadded])?;
        let greedy = SamplingParams::greedy();

        ik::rms_norm(
            ctx,
            pass,
            &cur_v,
            Some(&self.output_norm),
            &hn_v,
            block,
            h,
            self.cfg.rms_eps,
        )?;
        pass.memory_barrier()?;

        for slot in 1..block {
            let prev_tok_v = sc.best_toks.view(slot - 1, &[1])?;
            let out_tok_v = sc.best_toks.view(slot, &[1])?;
            let h_slot = sc.hidden_norm.view(slot * h, &[h])?;

            ik::copy_f32(ctx, pass, &sc.rnn_state, &ji_rnn, rank)?;
            ik::embed(ctx, pass, &prev_tok_v, &self.markov_w1, &ji_mk, 1)?;
            ik::copy_f32(ctx, pass, &h_slot, &ji_h, h)?;
            pass.memory_barrier()?;

            k::gemv(ctx, pass, &self.joint_proj, &sc.joint_in, &sc.joint_out, 1)?;
            pass.memory_barrier()?;

            ik::dspark_gru(
                ctx,
                pass,
                &sc.joint_out,
                &self.joint_proj_b,
                &sc.rnn_state,
                &sc.latent,
                rank,
            )?;
            pass.memory_barrier()?;

            k::gemv(ctx, pass, &self.hidden_out, &sc.latent, &sc.corr, 1)?;
            pass.memory_barrier()?;

            ik::add3(ctx, pass, &h_slot, &sc.corr, &self.hidden_out_b, &sc.combined, h)?;
            pass.memory_barrier()?;

            k::gemv(ctx, pass, target_output, &sc.combined, &sc.logits_raw, 1)?;
            pass.memory_barrier()?;

            sample_f32(ctx, pass, &logits_unpadded, &sc.sampler, &greedy, 0, &out_tok_v)?;
            if slot + 1 < block {
                pass.memory_barrier()?;
            }
        }
        Ok(())
    }
}
