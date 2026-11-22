//! Native `thinkingmachines/Inkling-2-Mini` HuggingFace safetensors loader.
//!
//! Loads the official ~19.5 GB mixed-precision checkpoint directory directly
//! (zero offline conversion):
//! - Dense attention, embeddings, unembed, loop controllers, router, and
//!   `mtp.safetensors` (DSpark) BF16 matrices: zero-copy read-only `mmap`
//!   backed by GPU buffers.
//! - Layers 0..20 routed experts (native NVFP4): 100% lossless parallel repack
//!   from interleaved `[E, 2*ff, H/2]` sequential-nibble + `F8_E4M3` scales into
//!   GGML `block_nvfp4` (`[gate; up]` half-split nibbles) + per-expert outer
//!   scale `amax[e] / outer_scale_denominator`.
//! - Layers 20..24 routed experts (native MXFP8 `F8_E4M3` + `UE8M0` per 32
//!   weights): parallel repack into `Q8_0` blocks (34B vs 33B per 32 weights,
//!   7-bit mantissa vs FP8's 3-bit) so they hit all existing `Q8_0` kernels.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, ensure};
use half::{bf16, f16};
use rayon::prelude::*;

use super::config::{Config, LoopLayout};
use super::dspark::{DsparkConfig, DsparkLayerW, DsparkScratch, Inkling2Dspark};
use super::model::{
    Inkling2Model, LayerW, LoopCoreW, LoopCtrlW, PREFILL_ROWS, PREFILL_ROWS_MU,
};
use crate::gguf::GgmlType;
use crate::glm53::model::Mapped;
use crate::kernels::glm53::QWeight;
use crate::kernels::inkling2 as ik;
use crate::kernels::sample::SamplerScratch;
use crate::metal::{Buffer, MetalContext};
use crate::safetensors::{Checkpoint, SafetensorsDType, TensorMeta};
use crate::tensor::{DType, Tensor};

const REGION_BYTES: u64 = 1 << 30;

/// Zero-copy read-only mmap view over selected tensors in a safetensors `Checkpoint`.
struct StMapped {
    tensors: HashMap<String, (Buffer, usize, usize)>,
    buffers: Vec<Buffer>,
}

impl StMapped {
    fn map_names(
        ctx: &MetalContext,
        ckpt: &Checkpoint,
        names: &[String],
    ) -> Result<Self> {
        let mut by_shard: HashMap<PathBuf, Vec<(String, u64, usize)>> = HashMap::new();
        for name in names {
            let meta = ckpt
                .meta(name)
                .with_context(|| format!("safetensors missing tensor {name}"))?;
            by_shard.entry(meta.shard().to_path_buf()).or_default().push((
                name.clone(),
                meta.start(),
                meta.byte_len(),
            ));
        }

        let mut tensors = HashMap::with_capacity(names.len());
        let mut buffers = Vec::new();
        for (shard, mut list) in by_shard {
            list.sort_by_key(|(_, start, _)| *start);
            let mut i = 0;
            while i < list.len() {
                let start = list[i].1;
                let mut end = start + list[i].2 as u64;
                let mut j = i + 1;
                while j < list.len()
                    && list[j].1 + list[j].2 as u64 - start <= REGION_BYTES
                {
                    end = end.max(list[j].1 + list[j].2 as u64);
                    j += 1;
                }
                let (buf, inner) = ctx
                    .new_buffer_mapped(&shard, start, usize::try_from(end - start)?)
                    .with_context(|| {
                        format!(
                            "mapping {} bytes at {start} from {}",
                            end - start,
                            shard.display()
                        )
                    })?;
                for (name, t_start, t_len) in &list[i..j] {
                    let at = inner + usize::try_from(*t_start - start)?;
                    tensors.insert(name.clone(), (buf.clone(), at, *t_len));
                }
                buffers.push(buf);
                i = j;
            }
        }
        Ok(Self { tensors, buffers })
    }

    fn bf16_qweight(&self, name: &str, cols: usize, rows: usize) -> Result<QWeight> {
        let (buf, at, len) = self
            .tensors
            .get(name)
            .with_context(|| format!("unmapped safetensors tensor {name}"))?;
        let expected = rows * GgmlType::Bf16.row_bytes(cols)?;
        ensure!(
            *len == expected,
            "tensor {name} is {len} bytes, expected {expected} ({rows}x{cols} BF16)"
        );
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length()], DType::I8)?;
        let view = whole.view(*at, &[*len])?;
        QWeight::new(view, GgmlType::Bf16, cols, rows)
    }

    fn push_buffer(&mut self, buf: Buffer) {
        self.buffers.push(buf);
    }

    fn into_mapped(self) -> Mapped {
        Mapped::from_buffers(self.buffers)
    }
}

/// Reads an F32 or BF16 tensor from `ckpt` into a `Vec<f32>`.
fn read_f32_vec(ckpt: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    let meta =
        ckpt.meta(name).with_context(|| format!("safetensors missing tensor {name}"))?;
    let raw = ckpt.read(name)?;
    match meta.dtype {
        SafetensorsDType::F32 => {
            ensure!(raw.len().is_multiple_of(4), "{name} byte len not multiple of 4");
            let mut out = Vec::with_capacity(raw.len() / 4);
            for &c in raw.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(c));
            }
            Ok(out)
        }
        SafetensorsDType::BF16 => {
            ensure!(raw.len().is_multiple_of(2), "{name} byte len not multiple of 2");
            let mut out = Vec::with_capacity(raw.len() / 2);
            for &c in raw.as_chunks::<2>().0 {
                out.push(bf16::from_le_bytes(c).to_f32());
            }
            Ok(out)
        }
        ref other => anyhow::bail!("tensor {name} has dtype {other:?}, expected F32/BF16"),
    }
}

fn load_f32_gpu(
    ctx: &MetalContext,
    ckpt: &Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<Tensor> {
    let v = read_f32_vec(ckpt, name)?;
    let expected: usize = shape.iter().product();
    ensure!(
        v.len() == expected,
        "tensor {name} has {} elements, expected {expected} ({shape:?})",
        v.len()
    );
    Tensor::from_f32(ctx, &v, shape)
}

fn expect_meta<'a>(
    ckpt: &'a Checkpoint,
    name: &str,
    dtype: SafetensorsDType,
    shape: &[usize],
) -> Result<&'a TensorMeta> {
    let meta =
        ckpt.meta(name).with_context(|| format!("safetensors missing tensor {name}"))?;
    ensure!(
        meta.dtype == dtype,
        "tensor {name} has dtype {:?}, expected {dtype:?}",
        meta.dtype
    );
    ensure!(
        meta.shape == shape,
        "tensor {name} has shape {:?}, expected {shape:?}",
        meta.shape
    );
    Ok(meta)
}

/// OCP FP8 E4M3FN byte -> f32 (exact; max finite = 448.0, no Inf).
#[inline]
pub(crate) fn fp8_e4m3fn_to_f32(b: u8) -> f32 {
    if (b & 0x7f) == 0x7f {
        return f32::NAN;
    }
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = u32::from((b >> 3) & 0x0f);
    let man = f32::from(b & 0x07);
    if exp == 0 {
        sign * man * f32::from_bits((127 - 9) << 23)
    } else {
        sign * (1.0 + man / 8.0) * f32::from_bits((127 + exp - 7) << 23)
    }
}

/// OCP UE8M0 byte -> f32 (`2^(e - 127)`).
#[inline]
pub(crate) fn ue8m0_to_f32(e: u8) -> f32 {
    match e {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        _ => f32::from_bits(u32::from(e) << 23),
    }
}

/// Losslessly repacks Thinking Machines native NVFP4 experts (`U8 [E, N, K/2]`
/// sequential nibbles + `F8_E4M3 [E, N, K/16]` scales) into GGML `block_nvfp4`
/// (`36` bytes per `64` values, half-split nibbles).
///
/// When `deinterleave_gu` is true (`w13_weight`), dim 1 is de-interleaved from
/// `(g0, u0, g1, u1, ...)` into `[gate(0..N/2); up(0..N/2)]`.
pub(crate) fn repack_nvfp4_experts(
    w_u8: &[u8],
    scale_fp8: &[u8],
    n_expert: usize,
    n_rows: usize,
    cols: usize,
    deinterleave_gu: bool,
) -> Result<Vec<u8>> {
    ensure!(cols.is_multiple_of(64), "NVFP4 cols {cols} not multiple of 64");
    ensure!(
        !deinterleave_gu || n_rows.is_multiple_of(2),
        "deinterleave_gu requires even n_rows ({n_rows})"
    );
    let w_row_bytes = cols / 2;
    let s_row_bytes = cols / 16;
    ensure!(w_u8.len() == n_expert * n_rows * w_row_bytes, "NVFP4 weight byte len mismatch");
    ensure!(
        scale_fp8.len() == n_expert * n_rows * s_row_bytes,
        "NVFP4 scale byte len mismatch"
    );

    let blocks_per_row = cols / 64;
    let out_row_bytes = blocks_per_row * 36;
    let mut out = vec![0u8; n_expert * n_rows * out_row_bytes];
    let half_rows = n_rows / 2;

    out.par_chunks_exact_mut(out_row_bytes).enumerate().for_each(|(global_row, dst_row)| {
        let e = global_row / n_rows;
        let out_r = global_row % n_rows;
        let src_r = if deinterleave_gu {
            if out_r < half_rows { 2 * out_r } else { 2 * (out_r - half_rows) + 1 }
        } else {
            out_r
        };
        let src_idx = e * n_rows + src_r;
        let w_row = &w_u8[src_idx * w_row_bytes..(src_idx + 1) * w_row_bytes];
        let s_row = &scale_fp8[src_idx * s_row_bytes..(src_idx + 1) * s_row_bytes];

        for b in 0..blocks_per_row {
            let dst = &mut dst_row[b * 36..(b + 1) * 36];
            for s in 0..4 {
                let sb = s_row[4 * b + s];
                // Normalize negative zero (0x80) to 0x00; block scales are non-negative.
                dst[s] = if sb == 0x80 { 0 } else { sb & 0x7f };
                let inp = &w_row[(4 * b + s) * 8..(4 * b + s + 1) * 8];
                let qout = &mut dst[4 + s * 8..4 + (s + 1) * 8];
                // HF sequential [even=lo, odd=hi] -> GGML half-split [0..7=lo, 8..15=hi].
                for j in 0..8 {
                    let lo = if (j & 1) == 0 { inp[j >> 1] & 0x0f } else { inp[j >> 1] >> 4 };
                    let hi = if (j & 1) == 0 {
                        inp[(j + 8) >> 1] & 0x0f
                    } else {
                        inp[(j + 8) >> 1] >> 4
                    };
                    qout[j] = lo | (hi << 4);
                }
            }
        }
    });

    Ok(out)
}

/// Quantizes raw row-major BF16 bytes `[rows, cols]` into a compact
/// `GgmlType::Q4_0` GPU `QWeight` (18 bytes per 32 values).
fn quantize_bf16_bytes_to_q4_0(
    ctx: &MetalContext,
    src: &[u8],
    cols: usize,
    rows: usize,
    src_row_bytes: usize,
) -> Result<(QWeight, Buffer)> {
    ensure!(cols.is_multiple_of(32), "Q4_0 cols {cols} not multiple of 32");
    ensure!(
        src.len() >= rows * src_row_bytes,
        "BF16 src {} < {}*{src_row_bytes}",
        src.len(),
        rows
    );
    let blocks_per_row = cols / 32;
    let out_row_bytes = blocks_per_row * 18;
    let mut out = vec![0u8; rows * out_row_bytes];
    out.par_chunks_exact_mut(out_row_bytes).enumerate().for_each(|(r, dst_row)| {
        let s_row = &src[r * src_row_bytes..(r + 1) * src_row_bytes];
        for b in 0..blocks_per_row {
            let mut vals = [0.0f32; 32];
            let mut max_abs = 0.0f32;
            let mut max_signed = 0.0f32;
            let base = b * 64;
            for j in 0..32 {
                let bits = u16::from_le_bytes([s_row[base + 2 * j], s_row[base + 2 * j + 1]]);
                let v = bf16::from_bits(bits).to_f32();
                vals[j] = v;
                let av = v.abs();
                if av > max_abs {
                    max_abs = av;
                    max_signed = v;
                }
            }
            // Match ggml quantize_row_q4_0_ref: d = max_signed / -8.0.
            let d = max_signed / -8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let dst = &mut dst_row[b * 18..(b + 1) * 18];
            let d_bytes = f16::from_f32(d).to_le_bytes();
            dst[0] = d_bytes[0];
            dst[1] = d_bytes[1];
            for j in 0..16 {
                let lo = ((vals[j] * id + 8.5) as i32).clamp(0, 15) as u8;
                let hi = ((vals[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
                dst[2 + j] = lo | (hi << 4);
            }
        }
    });
    let buf = ctx.new_buffer_with_bytes(&out)?;
    let t = Tensor::from_buffer(buf.clone(), &[out.len()], DType::I8)?;
    Ok((QWeight::new(t, GgmlType::Q4_0, cols, rows)?, buf))
}

/// Quantizes a BF16 `[rows, cols]` row-major weight matrix into a compact
/// `GgmlType::Q4_0` GPU `QWeight` (18 bytes per 32 values) for fast DSpark
/// speculative drafting. Speculative verification always runs against the
/// original exact `output` matrix, so this has zero effect on final logits.
pub(super) fn quantize_bf16_to_q4_0(
    ctx: &MetalContext,
    w: &QWeight,
) -> Result<(QWeight, Buffer)> {
    ensure!(w.ty == GgmlType::Bf16, "quantize_bf16_to_q4_0 expects Bf16, got {:?}", w.ty);
    // SAFETY: `w.t` is host-visible unified/mapped memory and the GPU is idle during load.
    let src = unsafe { std::slice::from_raw_parts(w.t.contents_ptr(), w.rows * w.row_bytes) };
    quantize_bf16_bytes_to_q4_0(ctx, src, w.cols, w.rows, w.row_bytes)
}

/// Repacks Thinking Machines native MXFP8 experts (`F8_E4M3 [E, N, K]` +
/// `U8 [E, N, K/32]` UE8M0 scales) into `GgmlType::Q8_0` (`34` bytes per `32`
/// values), de-interleaving dim 1 when `deinterleave_gu` is true.
pub(crate) fn repack_mxfp8_to_q8_0_experts(
    w_fp8: &[u8],
    scale_u8: &[u8],
    n_expert: usize,
    n_rows: usize,
    cols: usize,
    deinterleave_gu: bool,
) -> Result<Vec<u8>> {
    ensure!(cols.is_multiple_of(32), "MXFP8 cols {cols} not multiple of 32");
    ensure!(
        !deinterleave_gu || n_rows.is_multiple_of(2),
        "deinterleave_gu requires even n_rows ({n_rows})"
    );
    let s_row_bytes = cols / 32;
    ensure!(w_fp8.len() == n_expert * n_rows * cols, "MXFP8 weight byte len mismatch");
    ensure!(
        scale_u8.len() == n_expert * n_rows * s_row_bytes,
        "MXFP8 scale byte len mismatch"
    );

    let mut fp8_lut = [0.0f32; 256];
    for (i, slot) in fp8_lut.iter_mut().enumerate() {
        let v = fp8_e4m3fn_to_f32(i as u8);
        *slot = if v.is_finite() { v } else { 0.0 };
    }
    let mut e8m0_lut = [0.0f32; 256];
    for (i, slot) in e8m0_lut.iter_mut().enumerate() {
        let v = ue8m0_to_f32(i as u8);
        *slot = if v.is_finite() { v } else { 0.0 };
    }

    let blocks_per_row = cols / 32;
    let out_row_bytes = blocks_per_row * 34;
    let mut out = vec![0u8; n_expert * n_rows * out_row_bytes];
    let half_rows = n_rows / 2;

    out.par_chunks_exact_mut(out_row_bytes).enumerate().for_each(|(global_row, dst_row)| {
        let e = global_row / n_rows;
        let out_r = global_row % n_rows;
        let src_r = if deinterleave_gu {
            if out_r < half_rows { 2 * out_r } else { 2 * (out_r - half_rows) + 1 }
        } else {
            out_r
        };
        let src_idx = e * n_rows + src_r;
        let w_row = &w_fp8[src_idx * cols..(src_idx + 1) * cols];
        let s_row = &scale_u8[src_idx * s_row_bytes..(src_idx + 1) * s_row_bytes];

        let mut vals = [0.0f32; 32];
        for b in 0..blocks_per_row {
            let sc = e8m0_lut[usize::from(s_row[b])];
            let src = &w_row[b * 32..(b + 1) * 32];
            let mut amax = 0.0f32;
            for i in 0..32 {
                let v = fp8_lut[usize::from(src[i])] * sc;
                vals[i] = v;
                let av = v.abs();
                if av > amax {
                    amax = av;
                }
            }
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let dst = &mut dst_row[b * 34..(b + 1) * 34];
            let db = f16::from_f32(d).to_bits().to_le_bytes();
            dst[0] = db[0];
            dst[1] = db[1];
            for i in 0..32 {
                let q = (vals[i] * inv).round().clamp(-127.0, 127.0) as i8;
                dst[2 + i] = q as u8;
            }
        }
    });

    Ok(out)
}

fn load_expert_pair(
    ctx: &MetalContext,
    ckpt: &Checkpoint,
    prefix: &str,
    n_expert: usize,
    n_rows: usize,
    cols: usize,
    deinterleave_gu: bool,
) -> Result<(QWeight, Option<Tensor>, Buffer)> {
    let w_name = format!("{prefix}_weight");
    let s_name = format!("{prefix}_weight_scale");
    let meta = ckpt
        .meta(&w_name)
        .with_context(|| format!("safetensors missing expert tensor {w_name}"))?;

    match meta.dtype {
        SafetensorsDType::U8 => {
            // Native NVFP4 (layers 0..20).
            expect_meta(ckpt, &w_name, SafetensorsDType::U8, &[n_expert, n_rows, cols / 2])?;
            expect_meta(
                ckpt,
                &s_name,
                SafetensorsDType::F8E4M3,
                &[n_expert, n_rows, cols / 16],
            )?;
            let w_u8 = ckpt.read(&w_name)?;
            let s_fp8 = ckpt.read(&s_name)?;
            let packed =
                repack_nvfp4_experts(&w_u8, &s_fp8, n_expert, n_rows, cols, deinterleave_gu)?;
            let buf = ctx.new_buffer_with_bytes(&packed)?;
            let t = Tensor::from_buffer(buf.clone(), &[packed.len()], DType::I8)?;
            let qw = QWeight::new(t, GgmlType::Nvfp4, cols, n_expert * n_rows)?;

            let amax_name = format!("{prefix}_weight_amax");
            let denom_name = format!("{prefix}_weight_outer_scale_denominator");
            let amax = read_f32_vec(ckpt, &amax_name)?;
            let denom = read_f32_vec(ckpt, &denom_name)?;
            ensure!(amax.len() == n_expert, "{amax_name} len {} != {n_expert}", amax.len());
            ensure!(denom.len() == 1 && denom[0] != 0.0, "invalid {denom_name}");
            let inv_d = 1.0 / denom[0];
            let scales: Vec<f32> = amax.into_iter().map(|a| a * inv_d).collect();
            let scale_t = Tensor::from_f32(ctx, &scales, &[n_expert])?;
            Ok((qw, Some(scale_t), buf))
        }
        SafetensorsDType::F8E4M3 => {
            // Native MXFP8 (layers 20..24) -> Q8_0.
            expect_meta(ckpt, &w_name, SafetensorsDType::F8E4M3, &[n_expert, n_rows, cols])?;
            expect_meta(
                ckpt,
                &s_name,
                SafetensorsDType::U8,
                &[n_expert, n_rows, cols / 32],
            )?;
            let w_fp8 = ckpt.read(&w_name)?;
            let s_u8 = ckpt.read(&s_name)?;
            let packed = repack_mxfp8_to_q8_0_experts(
                &w_fp8,
                &s_u8,
                n_expert,
                n_rows,
                cols,
                deinterleave_gu,
            )?;
            let buf = ctx.new_buffer_with_bytes(&packed)?;
            let t = Tensor::from_buffer(buf.clone(), &[packed.len()], DType::I8)?;
            let qw = QWeight::new(t, GgmlType::Q8_0, cols, n_expert * n_rows)?;
            Ok((qw, None, buf))
        }
        ref other => anyhow::bail!("unsupported expert dtype {other:?} for {w_name}"),
    }
}

/// Loads `thinkingmachines/Inkling-2-Mini` directly from its HF directory.
pub(super) fn load_model(ctx: &MetalContext, dir: &Path) -> Result<Inkling2Model> {
    let cfg = Config::from_hf_dir(dir)?;
    let ckpt = Checkpoint::open(dir)?;

    // Collect all BF16 dense matrices to zero-copy mmap.
    let mut map_names = vec![
        "model.llm.embed.weight".to_string(),
        "model.llm.unembed.weight".to_string(),
    ];
    for i in 0..cfg.n_phys {
        let p = |s: &str| format!("model.llm.layers.{i}.{s}");
        map_names.extend([
            p("attn.wq_du.weight"),
            p("attn.wk_dv.weight"),
            p("attn.wv_dv.weight"),
            p("attn.wo_ud.weight"),
            p("attn.gate_attn.weight"),
            p("attn.attn_sink_proj.weight"),
            p("attn.wr_du.weight"),
            p("mlp.gate.weight"),
        ]);
        if !cfg.swa[i] {
            map_names.push(p("attn.k_sconv.weight_proj"));
            map_names.push(p("attn.v_sconv.weight_proj"));
        }
        if cfg.loop_count > 1 && i >= cfg.loop_start && i < cfg.loop_end {
            for c in ["attn", "k", "v", "mlp"] {
                map_names.push(p(&format!("controllers.{c}.down_weight")));
                map_names.push(p(&format!("controllers.{c}.gain_weight")));
                map_names.push(p(&format!("controllers.{c}.forget_weight")));
            }
        }
    }
    let mut st = StMapped::map_names(ctx, &ckpt, &map_names)?;

    let h = cfg.hidden;
    let tok_embd = st.bf16_qweight("model.llm.embed.weight", h, cfg.vocab_padded)?;
    let tok_norm = load_f32_gpu(ctx, &ckpt, "model.llm.embed_norm.weight", &[h])?;
    let output = st.bf16_qweight("model.llm.unembed.weight", h, cfg.vocab_padded)?;
    let output_norm = load_f32_gpu(ctx, &ckpt, "model.llm.norm.weight", &[h])?;
    let output_res_query =
        load_f32_gpu(ctx, &ckpt, "model.llm.attn_res.output_query", &[h])?;

    let loop_queries = if cfg.loop_count > 1 {
        ensure!(cfg.loop_layout == LoopLayout::Unsloth, "expected Unsloth loop layout");
        Some((
            read_f32_vec(&ckpt, "model.llm.middle_cycle_cross_pass_queries")?,
            read_f32_vec(&ckpt, "model.llm.middle_cycle_cross_pass_kv_queries")?,
        ))
    } else {
        None
    };

    let mut layers = Vec::with_capacity(cfg.n_phys);
    for i in 0..cfg.n_phys {
        let p = |s: &str| format!("model.llm.layers.{i}.{s}");
        let is_swa = cfg.swa[i];
        let (qw, kw) = (cfg.q_width(i), cfg.kv_width(i));
        let (nh, nkv, hd) = (cfg.n_head[i], cfg.n_head_kv[i], cfg.head_dim);
        let n_conv = cfg.sconv_l_cache;
        let extent = cfg.rel_extent_of(i);

        // Transpose `rel_logits_proj.proj` [rel_len, extent] -> [extent, rel_len] F32.
        let rel_raw = read_f32_vec(&ckpt, &p("attn.rel_logits_proj.proj"))?;
        ensure!(
            rel_raw.len() == cfg.rel_len * extent,
            "rel_logits_proj.proj len {} != {}*{}",
            rel_raw.len(),
            cfg.rel_len,
            extent
        );
        let mut rel_t = vec![0.0f32; extent * cfg.rel_len];
        for r in 0..cfg.rel_len {
            for e in 0..extent {
                rel_t[e * cfg.rel_len + r] = rel_raw[r * extent + e];
            }
        }
        let attn_rel_bias = QWeight::from_bytes(
            ctx,
            bytemuck::cast_slice(&rel_t),
            GgmlType::F32,
            cfg.rel_len,
            extent,
        )?;

        // SWA k/v sconv: [kw, 1, n_conv] -> [kw, n_conv] (no flip).
        // Global k/v sconv static_w: [n_conv, kw] -> flip(0).T -> [kw, n_conv].
        let (attn_k_conv, attn_v_conv) = if is_swa {
            (
                load_f32_gpu(ctx, &ckpt, &p("attn.k_sconv.weight"), &[kw, n_conv])?,
                load_f32_gpu(ctx, &ckpt, &p("attn.v_sconv.weight"), &[kw, n_conv])?,
            )
        } else {
            let flip_t = |name: &str| -> Result<Tensor> {
                let raw = read_f32_vec(&ckpt, name)?;
                ensure!(raw.len() == n_conv * kw, "{name} len {} != {n_conv}*{kw}", raw.len());
                let mut dst = vec![0.0f32; kw * n_conv];
                for c in 0..kw {
                    for tap in 0..n_conv {
                        dst[c * n_conv + tap] = raw[(n_conv - 1 - tap) * kw + c];
                    }
                }
                Tensor::from_f32(ctx, &dst, &[kw, n_conv])
            };
            (flip_t(&p("attn.k_sconv.static_w"))?, flip_t(&p("attn.v_sconv.static_w"))?)
        };

        let loop_core = if cfg.loop_count > 1 && i >= cfg.loop_start && i < cfg.loop_end {
            let (res_q, kv_q) =
                loop_queries.as_ref().context("missing cross-pass loop queries")?;
            let c = i - cfg.loop_start;
            let mut packed = Vec::with_capacity(4 * h);
            packed.extend_from_slice(&res_q[(2 * c) * h..(2 * c + 1) * h]);
            packed.extend_from_slice(&kv_q[(2 * c) * h..(2 * c + 1) * h]);
            packed.extend_from_slice(&kv_q[(2 * c + 1) * h..(2 * c + 2) * h]);
            packed.extend_from_slice(&res_q[(2 * c + 1) * h..(2 * c + 2) * h]);
            let msg_query = Tensor::from_f32(ctx, &packed, &[4, h])?;

            let rank = cfg.loop_gate_rank;
            let ctrl_names = ["attn", "k", "v", "mlp"];
            let ctrl = [0, 1, 2, 3].map(|m| {
                let cn = ctrl_names[m];
                Ok::<_, anyhow::Error>(LoopCtrlW {
                    down: st.bf16_qweight(&p(&format!("controllers.{cn}.down_weight")), 2 * h, rank)?,
                    gain: st.bf16_qweight(&p(&format!("controllers.{cn}.gain_weight")), rank, h)?,
                    forget: st.bf16_qweight(
                        &p(&format!("controllers.{cn}.forget_weight")),
                        rank,
                        h,
                    )?,
                })
            });
            let [c0, c1, c2, c3] = ctrl;
            Some(LoopCoreW { msg_query, ctrl: [c0?, c1?, c2?, c3?] })
        } else {
            None
        };

        let (gate_up_exps, gate_up_scale, gate_up_buf) = load_expert_pair(
            ctx,
            &ckpt,
            &p("mlp.experts.w13"),
            cfg.n_expert,
            2 * cfg.ff_expert,
            h,
            true,
        )?;
        let (down_exps, down_scale, down_buf) = load_expert_pair(
            ctx,
            &ckpt,
            &p("mlp.experts.w2"),
            cfg.n_expert,
            h,
            cfg.ff_expert,
            false,
        )?;
        st.push_buffer(gate_up_buf);
        st.push_buffer(down_buf);

        // attn_res.attn_queries.{k} is 1-offset: k = i - 1 for i >= 1.
        let attn_res_query = if i > 0 {
            let qname = format!("model.llm.attn_res.attn_queries.{}", i - 1);
            let shape: &[usize] = if is_swa { &[h] } else { &[3, h] };
            Some(load_f32_gpu(ctx, &ckpt, &qname, shape)?)
        } else {
            None
        };

        layers.push(LayerW {
            is_swa,
            attn_norm: if is_swa {
                Some(load_f32_gpu(ctx, &ckpt, &p("attn_norm.weight"), &[h])?)
            } else {
                None
            },
            attn_q_inp_norm: if !is_swa {
                Some(load_f32_gpu(ctx, &ckpt, &p("q_input_norm.weight"), &[h])?)
            } else {
                None
            },
            attn_k_inp_norm: if !is_swa {
                Some(load_f32_gpu(ctx, &ckpt, &p("k_input_norm.weight"), &[h])?)
            } else {
                None
            },
            attn_v_inp_norm: if !is_swa {
                Some(load_f32_gpu(ctx, &ckpt, &p("v_input_norm.weight"), &[h])?)
            } else {
                None
            },
            wq: st.bf16_qweight(&p("attn.wq_du.weight"), h, qw)?,
            wk: st.bf16_qweight(&p("attn.wk_dv.weight"), h, kw)?,
            wv: st.bf16_qweight(&p("attn.wv_dv.weight"), h, kw)?,
            wo: st.bf16_qweight(&p("attn.wo_ud.weight"), qw, h)?,
            attn_gate: st.bf16_qweight(&p("attn.gate_attn.weight"), h, qw)?,
            attn_q_clip: load_f32_gpu(ctx, &ckpt, &p("attn.q_l2_clip.scale"), &[nh, hd])?,
            attn_sink_proj: st.bf16_qweight(&p("attn.attn_sink_proj.weight"), h, nh)?,
            attn_rel: st.bf16_qweight(&p("attn.wr_du.weight"), h, nh * cfg.rel_len)?,
            attn_rel_bias,
            attn_rel_clip: load_f32_gpu(
                ctx,
                &ckpt,
                &p("attn.rel_l2_clip.scale"),
                &[nh, cfg.rel_len],
            )?,
            attn_k_conv,
            attn_v_conv,
            attn_k_conv_proj: if !is_swa {
                Some(st.bf16_qweight(&p("attn.k_sconv.weight_proj"), h, n_conv * nkv)?)
            } else {
                None
            },
            attn_v_conv_proj: if !is_swa {
                Some(st.bf16_qweight(&p("attn.v_sconv.weight_proj"), h, n_conv * nkv)?)
            } else {
                None
            },
            attn_post_conv: load_f32_gpu(ctx, &ckpt, &p("attn_sconv.weight"), &[h, n_conv])?,
            ffn_post_conv: load_f32_gpu(ctx, &ckpt, &p("mlp_sconv.weight"), &[h, n_conv])?,
            attn_res_query,
            ffn_res_query: load_f32_gpu(
                ctx,
                &ckpt,
                &format!("model.llm.attn_res.mlp_queries.{i}"),
                &[h],
            )?,
            ffn_norm: load_f32_gpu(ctx, &ckpt, &p("mlp_norm.weight"), &[h])?,
            router: st.bf16_qweight(&p("mlp.gate.weight"), h, cfg.n_expert)?,
            exp_probs_b: if ckpt.meta(&p("mlp.gate.bias")).is_some() {
                Some(load_f32_gpu(ctx, &ckpt, &p("mlp.gate.bias"), &[cfg.n_expert])?)
            } else {
                None
            },
            gate_up_exps,
            gate_up_scale,
            down_exps,
            down_scale,
            loop_core,
        });
    }

    let schedule = cfg.schedule();
    Ok(Inkling2Model {
        cfg,
        mapped: st.into_mapped(),
        tok_embd,
        tok_norm,
        output,
        draft_output: None,
        output_norm,
        output_res_query,
        layers,
        schedule,
        prefill_rows: if ctx.matrix_units() { PREFILL_ROWS_MU } else { PREFILL_ROWS },
        memory_limit: None,
        fitted_context: None,
        dspark: None,
        drafts: 0,
        vision: None,
    })
}

/// Loads `mtp.safetensors` (native Thinking Machines DSpark drafter) using
/// `mtp_config` from the sibling `config.json`.
pub(super) fn load_dspark(
    ctx: &MetalContext,
    mtp_path: &Path,
    target: &Config,
) -> Result<Inkling2Dspark> {
    let dir = mtp_path
        .parent()
        .with_context(|| format!("{} has no parent directory", mtp_path.display()))?;
    let cfg_bytes = std::fs::read(dir.join("config.json"))
        .with_context(|| format!("reading {}/config.json", dir.display()))?;
    let root: serde_json::Value =
        serde_json::from_slice(&cfg_bytes).context("parsing config.json for mtp_config")?;
    let mc = root.get("mtp_config").context("config.json missing mtp_config")?;

    let u = |k: &str, default: usize| -> usize {
        mc.get(k).and_then(|v| v.as_u64()).map_or(default, |v| v as usize)
    };
    let f = |k: &str, default: f32| -> f32 {
        mc.get(k).and_then(|v| v.as_f64()).map_or(default, |v| v as f32)
    };

    let n_layer = u("num_hidden_layers", 3);
    let hidden = u("hidden_size", target.hidden);
    let ff = u("intermediate_size", 9216);
    let n_head = u("num_attention_heads", 24);
    let n_head_kv = u("num_key_value_heads", 8);
    let head_dim = u("head_dim", 128);
    let rope_theta = f("rope_theta", 1_000_000.0);
    let rms_eps = f("rms_norm_eps", 1e-6);
    let block_size = u("block_size", 8);
    let markov_rank = u("markov_rank", 256);
    let window = u("sliding_window_size", 1024);

    let mut swa = vec![false; n_layer];
    if let Some(arr) = mc.get("local_layer_ids").and_then(|v| v.as_array()) {
        for id in arr {
            let idx = id.as_u64().context("mtp_config.local_layer_ids entry not u64")? as usize;
            ensure!(idx < n_layer, "mtp local_layer_ids {idx} >= {n_layer}");
            swa[idx] = true;
        }
    }
    let target_layers: Vec<usize> = match mc.get("target_layer_ids").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize).context("target_layer_ids not u64"))
            .collect::<Result<Vec<_>>>()?,
        None => vec![5, 23, 41, 48],
    };

    let cfg = DsparkConfig {
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
        final_logit_softcapping: target.final_logit_softcapping,
    };

    let ckpt = Checkpoint::open_file(mtp_path)?;
    let map_names = vec![
        "markov_head.markov_w1.weight".to_string(),
        "markov_head.rnn_proj.weight".to_string(),
        "markov_head.hidden_out.weight".to_string(),
    ];
    let mut st = StMapped::map_names(ctx, &ckpt, &map_names)?;

    let h = cfg.hidden;
    let qw = cfg.n_head * cfg.head_dim;
    let kw = cfg.n_head_kv * cfg.head_dim;
    let rank = cfg.markov_rank;

    let output_norm = load_f32_gpu(ctx, &ckpt, "norm.weight", &[h])?;
    let mask_embd = load_f32_gpu(ctx, &ckpt, "mask_embed", &[h])?;
    let fuse_raw = read_f32_vec(&ckpt, "fuse_weight")?;
    let dspark_norm = load_f32_gpu(ctx, &ckpt, "markov_head.dflare_norm.weight", &[h])?;
    let markov_w1 =
        st.bf16_qweight("markov_head.markov_w1.weight", rank, target.vocab_padded)?;
    let joint_proj =
        st.bf16_qweight("markov_head.rnn_proj.weight", 2 * rank + 2 * h, 3 * rank)?;
    let joint_proj_b = load_f32_gpu(ctx, &ckpt, "markov_head.rnn_proj.bias", &[3 * rank])?;
    let hidden_out = st.bf16_qweight("markov_head.hidden_out.weight", rank, h)?;
    let hidden_out_b = load_f32_gpu(ctx, &ckpt, "markov_head.hidden_out.bias", &[h])?;

    let nt = cfg.target_layers.len();
    ensure!(
        fuse_raw.len() == (cfg.n_layer + 1) * nt,
        "fuse_weight len {} != {}*{nt}",
        fuse_raw.len(),
        cfg.n_layer + 1
    );
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

    let mut load_dflare_w = |name: &str, cols: usize, rows: usize| -> Result<QWeight> {
        expect_meta(&ckpt, name, SafetensorsDType::BF16, &[rows, cols])?;
        let raw = ckpt.read(name)?;
        let (qw, buf) = quantize_bf16_bytes_to_q4_0(ctx, &raw, cols, rows, cols * 2)?;
        st.push_buffer(buf);
        Ok(qw)
    };

    let mut layers = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let p = |s: &str| format!("layers.{i}.{s}");
        layers.push(DsparkLayerW {
            attn_norm: load_f32_gpu(ctx, &ckpt, &p("input_layernorm.weight"), &[h])?,
            ffn_norm: load_f32_gpu(ctx, &ckpt, &p("post_attention_layernorm.weight"), &[h])?,
            dflare_norm: load_f32_gpu(ctx, &ckpt, &p("dflare_norm.weight"), &[h])?,
            q_norm: load_f32_gpu(ctx, &ckpt, &p("self_attn.q_norm.weight"), &[head_dim])?,
            k_norm: load_f32_gpu(ctx, &ckpt, &p("self_attn.k_norm.weight"), &[head_dim])?,
            k_ctx_norm: load_f32_gpu(
                ctx,
                &ckpt,
                &p("self_attn.k_norm_ctx.weight"),
                &[head_dim],
            )?,
            wq: load_dflare_w(&p("self_attn.q_proj.weight"), h, qw)?,
            wk: load_dflare_w(&p("self_attn.k_proj.weight"), h, kw)?,
            wv: load_dflare_w(&p("self_attn.v_proj.weight"), h, kw)?,
            wo: load_dflare_w(&p("self_attn.o_proj.weight"), qw, h)?,
            wk_ctx: load_dflare_w(&p("self_attn.k_proj_ctx.weight"), h, kw)?,
            wv_ctx: load_dflare_w(&p("self_attn.v_proj_ctx.weight"), h, kw)?,
            ffn_gate: load_dflare_w(&p("mlp.gate_proj.weight"), h, cfg.ff)?,
            ffn_up: load_dflare_w(&p("mlp.up_proj.weight"), h, cfg.ff)?,
            ffn_down: load_dflare_w(&p("mlp.down_proj.weight"), cfg.ff, h)?,
        });
    }

    let max_block = cfg.block_size.max(8);
    let nl = cfg.n_layer.max(1);
    let scratch = DsparkScratch {
        max_block,
        cur: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
        x_h: Tensor::zeros(ctx, &[nl * max_block, h], DType::F32)?,
        post_attn: Tensor::zeros(ctx, &[max_block, h], DType::F32)?,
        q: Tensor::zeros(ctx, &[max_block, qw], DType::F32)?,
        k: Tensor::zeros(ctx, &[nl * max_block, kw], DType::F32)?,
        v: Tensor::zeros(ctx, &[nl * max_block, kw], DType::F32)?,
        attn_partial: Tensor::zeros(
            ctx,
            &[max_block, cfg.n_head, ik::DSPARK_SPLIT_MAX_CHUNKS + 1, 130],
            DType::F32,
        )?,
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
        draft_part_val: Tensor::zeros(
            ctx,
            &[target.vocab_unpadded.div_ceil(32).max(1)],
            DType::F32,
        )?,
        draft_part_idx: Tensor::zeros(
            ctx,
            &[target.vocab_unpadded.div_ceil(32).max(1)],
            DType::U32,
        )?,
        best_toks: Tensor::zeros(ctx, &[max_block], DType::U32)?,
        sampler: SamplerScratch::new(ctx, target.vocab_unpadded)?,
    };

    let bytes = std::fs::metadata(mtp_path)?.len() as usize;
    Ok(Inkling2Dspark {
        cfg,
        bytes,
        _mapped: st.into_mapped(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::dequantize;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn nvfp4_repack_matches_hf_e2m1_dequant_bit_exact() {
        let mut rng = StdRng::seed_from_u64(4242);
        let (e, ff, cols) = (2usize, 4usize, 128usize);
        let n_rows = 2 * ff;
        let w_u8: Vec<u8> = (0..e * n_rows * (cols / 2)).map(|_| rng.r#gen()).collect();
        let scale_fp8: Vec<u8> =
            (0..e * n_rows * (cols / 16)).map(|_| rng.gen_range(1u8..0x7eu8)).collect();

        let packed =
            repack_nvfp4_experts(&w_u8, &scale_fp8, e, n_rows, cols, true).unwrap();
        let mut got = vec![0.0f32; e * n_rows * cols];
        dequantize(GgmlType::Nvfp4, &packed, &mut got).unwrap();

        const E2M1: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
            -4.0, -6.0,
        ];
        for exp in 0..e {
            for out_r in 0..n_rows {
                let src_r = if out_r < ff { 2 * out_r } else { 2 * (out_r - ff) + 1 };
                for c in 0..cols {
                    let byte = w_u8[(exp * n_rows + src_r) * (cols / 2) + (c / 2)];
                    let nib = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
                    let sb = scale_fp8[(exp * n_rows + src_r) * (cols / 16) + (c / 16)];
                    let want = E2M1[usize::from(nib)] * fp8_e4m3fn_to_f32(sb);
                    let g = got[(exp * n_rows + out_r) * cols + c];
                    assert!(
                        (g - want).abs() <= 1e-6 * want.abs().max(1.0),
                        "e={exp} r={out_r} c={c}: got={g} want={want}"
                    );
                }
            }
        }
    }

    #[test]
    fn mxfp8_repack_to_q8_0_matches_hf_dequant() {
        let mut rng = StdRng::seed_from_u64(4243);
        let (e, ff, cols) = (2usize, 4usize, 64usize);
        let n_rows = 2 * ff;
        let w_fp8: Vec<u8> = (0..e * n_rows * cols)
            .map(|_| {
                let mag = rng.gen_range(0u8..0x7eu8);
                let sgn = if rng.gen_bool(0.5) { 0x80u8 } else { 0 };
                mag | sgn
            })
            .collect();
        let scale_u8: Vec<u8> =
            (0..e * n_rows * (cols / 32)).map(|_| rng.gen_range(120u8..132u8)).collect();

        let packed =
            repack_mxfp8_to_q8_0_experts(&w_fp8, &scale_u8, e, n_rows, cols, true)
                .unwrap();
        let mut got = vec![0.0f32; e * n_rows * cols];
        dequantize(GgmlType::Q8_0, &packed, &mut got).unwrap();

        for exp in 0..e {
            for out_r in 0..n_rows {
                let src_r = if out_r < ff { 2 * out_r } else { 2 * (out_r - ff) + 1 };
                let mut block_amax = [0.0f32; 2];
                for c in 0..cols {
                    let q = w_fp8[(exp * n_rows + src_r) * cols + c];
                    let s = scale_u8[(exp * n_rows + src_r) * (cols / 32) + (c / 32)];
                    let v = fp8_e4m3fn_to_f32(q) * ue8m0_to_f32(s);
                    block_amax[c / 32] = block_amax[c / 32].max(v.abs());
                }
                for c in 0..cols {
                    let q = w_fp8[(exp * n_rows + src_r) * cols + c];
                    let s = scale_u8[(exp * n_rows + src_r) * (cols / 32) + (c / 32)];
                    let want = fp8_e4m3fn_to_f32(q) * ue8m0_to_f32(s);
                    let g = got[(exp * n_rows + out_r) * cols + c];
                    let tol = 0.01 * block_amax[c / 32].max(1e-6);
                    assert!(
                        (g - want).abs() <= tol,
                        "e={exp} r={out_r} c={c}: got={g} want={want} tol={tol}"
                    );
                }
            }
        }
    }
}
