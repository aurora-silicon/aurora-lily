//! Gemma 4 kernels (`vulkan/gemma4.comp`): the embedding, per-head norms
//! with RoPE and the K/V store, attention over f16 caches (a ring for the
//! sliding-window layers), GeGLU, the softmax router, routed experts with
//! the fused gate/up tensor, the norm chains around the sublayers and the
//! final logit softcap. Projections use the GGUF GEMV and GEMM of
//! [`crate::kernels::glm53`].

use anyhow::{Result, ensure};

use crate::gguf::GgmlType;
use crate::kernels::glm53::QWeight;
use crate::kernels::{f32_bytes, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// GLSL only (see `kernels::glm53`).
const SOURCE: &str = "";

/// Keys a threadgroup of the decode attention takes.
pub const ATTN_CHUNK: usize = 256;

fn check_f32(ts: &[&Tensor]) -> Result<()> {
    for t in ts {
        ensure!(
            t.dtype() == DType::F32,
            "Gemma kernels take f32 tensors, got {:?}",
            t.dtype()
        );
    }
    Ok(())
}

/// `out[i] = table[tokens[i]] * scale` for `m` rows.
pub fn embed(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &QWeight,
    tokens: &Tensor,
    out: &Tensor,
    m: usize,
    scale: f32,
) -> Result<()> {
    check_f32(&[out])?;
    ensure!(tokens.dtype() == DType::U32 && tokens.numel() >= m, "embedding tokens");
    ensure!(out.numel() >= m * table.cols, "embedding output of {}", out.numel());
    let name = match table.ty {
        GgmlType::Q8_0 => "gemma4_embed_q8_0",
        GgmlType::Q5K => "gemma4_embed_q5_k",
        GgmlType::Q6K => "gemma4_embed_q6_k",
        GgmlType::Q4K => "gemma4_embed_q4_k",
        other => anyhow::bail!("no Gemma embedding over {other:?}"),
    };
    let tasks = table.cols / 32;
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.t.binding(), tokens.binding(), out.binding()],
        &[&u32_bytes(table.cols), &u32_bytes(table.row_bytes), &f32_bytes(scale)],
        Grid::Threadgroups {
            groups: (tasks.div_ceil(64), m, 1),
            threadgroup: (64, 1, 1),
        },
    )
}

/// One layer's attention shape and cache.
pub struct AttnShape<'t> {
    pub head_dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// Cache slots: positions live at `pos % ring`.
    pub ring: usize,
    /// Sliding window (0: none).
    pub window: usize,
    pub k_cache: &'t Tensor,
    pub v_cache: &'t Tensor,
}

impl AttnShape<'_> {
    /// `kernel` for this shape's head width (`<kernel>_256` or `_512`).
    fn name(&self, kernel: [&'static str; 2]) -> Result<&'static str> {
        ensure!(
            self.kv_heads > 0
                && self.heads.is_multiple_of(self.kv_heads)
                && (self.heads / self.kv_heads).is_multiple_of(2),
            "{} heads over {} KV heads",
            self.heads,
            self.kv_heads
        );
        Ok(match self.head_dim {
            256 => kernel[0],
            512 => kernel[1],
            d => anyhow::bail!("head width {d}"),
        })
    }
}

/// Per-head RMSNorm and NEOX RoPE of `m` rows at positions `pos0..`:
/// queries (`q_norm`) to `q_out`, keys (`k_norm`) and values (no weight)
/// into the caches. `kv` holds each row's K at 0 and V at `v_offset`
/// (0 when V is the K projection), rows `kv_stride` apart.
#[allow(clippy::too_many_arguments)]
pub fn qkv(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &AttnShape<'_>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kv_stride: usize,
    v_offset: usize,
    q_norm: &Tensor,
    k_norm: &Tensor,
    freq_factors: Option<&Tensor>,
    q_out: &Tensor,
    pos0: usize,
    m: usize,
    base: f32,
    eps: f32,
) -> Result<()> {
    check_f32(&[q, k, v, q_norm, k_norm, q_out])?;
    let d = a.head_dim;
    ensure!(
        a.k_cache.dtype() == DType::F16 && a.v_cache.dtype() == DType::F16,
        "f16 caches"
    );
    ensure!(
        a.k_cache.numel() >= a.ring * a.kv_heads * d,
        "cache of {}",
        a.k_cache.numel()
    );
    ensure!(
        q.numel() >= m * a.heads * d && q_out.numel() >= m * a.heads * d,
        "queries"
    );
    ensure!(
        k.numel() >= (m - 1) * kv_stride + a.kv_heads * d
            && v.numel() >= (m - 1) * kv_stride + v_offset + a.kv_heads * d,
        "keys/values"
    );
    if let Some(ff) = freq_factors {
        check_f32(&[ff])?;
        ensure!(ff.numel() >= d / 2, "rope factors of {}", ff.numel());
    }
    ensure!(pos0 + m <= u32::MAX as usize, "position {pos0}");
    let slots = a.heads + 2 * a.kv_heads;
    let pipeline = ctx.pipeline(
        a.name(["gemma4_qkv_256", "gemma4_qkv_512"])?,
        SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[
            q.binding(),
            k.binding(),
            v.binding(),
            q_norm.binding(),
            k_norm.binding(),
            q_out.binding(),
            a.k_cache.binding(),
            a.v_cache.binding(),
            freq_factors.unwrap_or(q_norm).binding(),
        ],
        &[
            &u32_bytes(d),
            &u32_bytes(a.heads),
            &u32_bytes(a.kv_heads),
            &u32_bytes(pos0),
            &u32_bytes(a.ring),
            &f32_bytes(base),
            &u32_bytes(usize::from(freq_factors.is_some())),
            &f32_bytes(eps),
            &u32_bytes(kv_stride),
            &u32_bytes(v_offset),
        ],
        Grid::Threadgroups {
            groups: (slots.div_ceil(8), m, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Chunks of keys the decode attention of rows at `pos0..pos0 + m` takes.
pub fn attn_chunks(a: &AttnShape<'_>, pos0: usize, m: usize) -> usize {
    let keys = pos0 + m;
    let keys = if a.window > 0 { keys.min(a.window) } else { keys };
    keys.div_ceil(ATTN_CHUNK)
}

/// Floats of the decode attention's partial results for `rows` rows,
/// `heads` heads of width `d` and `chunks` chunks.
pub fn attn_partial_len(rows: usize, heads: usize, d: usize, chunks: usize) -> usize {
    rows * heads * chunks * (d + 2)
}

/// Attention of `m` rows at positions `pos0..` (queries `q`, normed and
/// rotated) over the caches, scale 1, into `heads` (`[m][heads][d]`):
/// split over key chunks and merged for a few rows, one pass over the keys
/// per row for a prompt run (`m > small_rows`).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &AttnShape<'_>,
    q: &Tensor,
    partial: &Tensor,
    heads: &Tensor,
    pos0: usize,
    m: usize,
    small_rows: usize,
) -> Result<()> {
    check_f32(&[q, partial, heads])?;
    let d = a.head_dim;
    ensure!(
        q.numel() >= m * a.heads * d && heads.numel() >= m * a.heads * d,
        "attention rows"
    );
    // Keys a row needs must still be in the cache.
    if a.window > 0 {
        ensure!(a.ring >= a.window + m - 1, "ring of {} for {m} rows", a.ring);
    } else {
        ensure!(a.ring >= pos0 + m, "cache of {} for position {}", a.ring, pos0 + m);
    }
    let common = [
        u32_bytes(d),
        u32_bytes(a.heads),
        u32_bytes(a.kv_heads),
        u32_bytes(pos0),
        u32_bytes(a.ring),
        u32_bytes(a.window),
    ];
    if m > small_rows {
        let pipeline = ctx.pipeline(
            a.name(["gemma4_attn_prefill_256", "gemma4_attn_prefill_512"])?,
            SOURCE,
            MslVersion::V3_1,
        )?;
        let mb = u32_bytes(m);
        let params: Vec<&[u8]> =
            common.iter().map(|b| b.as_slice()).chain([mb.as_slice()]).collect();
        return pass.dispatch_at(
            &pipeline,
            &[q.binding(), a.k_cache.binding(), a.v_cache.binding(), heads.binding()],
            &params,
            Grid::Threadgroups {
                groups: (m.div_ceil(8), a.heads / 2, 1),
                threadgroup: (256, 1, 1),
            },
        );
    }
    let chunks = attn_chunks(a, pos0, m);
    ensure!(
        partial.numel() >= attn_partial_len(m, a.heads, d, chunks),
        "attention partials of {} for {chunks} chunks",
        partial.numel()
    );
    let pipeline = ctx.pipeline(
        a.name(["gemma4_attn_decode_256", "gemma4_attn_decode_512"])?,
        SOURCE,
        MslVersion::V3_1,
    )?;
    let cb = u32_bytes(chunks);
    let params: Vec<&[u8]> =
        common.iter().map(|b| b.as_slice()).chain([cb.as_slice()]).collect();
    pass.dispatch_at(
        &pipeline,
        &[q.binding(), a.k_cache.binding(), a.v_cache.binding(), partial.binding()],
        &params,
        Grid::Threadgroups {
            groups: (chunks, a.heads / 2, m),
            threadgroup: (256, 1, 1),
        },
    )?;
    pass.memory_barrier()?;
    let combine = ctx.pipeline("gemma4_attn_combine", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &combine,
        &[partial.binding(), heads.binding()],
        &[&u32_bytes(d), &u32_bytes(a.heads), &u32_bytes(chunks)],
        Grid::Threadgroups { groups: (a.heads, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// `out = gelu(g) * u` over `n` values.
pub fn geglu(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    g: &Tensor,
    u: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    check_f32(&[g, u, out])?;
    ensure!(g.numel() >= n && u.numel() >= n && out.numel() >= n, "GeGLU of {n}");
    let pipeline = ctx.pipeline("gemma4_geglu", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[g.binding(), u.binding(), out.binding()],
        &[&u32_bytes(n)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Softmax router: per row the `k` most probable of `n_expert` experts and
/// their probabilities normalized over the `k`, times `down_scale[e]`.
#[allow(clippy::too_many_arguments)]
pub fn router(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    down_scale: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    m: usize,
    n_expert: usize,
    k: usize,
) -> Result<()> {
    check_f32(&[logits, down_scale, weights])?;
    ensure!(
        n_expert <= 256 && k <= 32 && k <= n_expert,
        "router of {n_expert}, top {k}"
    );
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * k, "selection");
    let pipeline = ctx.pipeline("gemma4_router", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), down_scale.binding(), sel.binding(), weights.binding()],
        &[&u32_bytes(n_expert), &u32_bytes(k)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Routed experts' gate and up (one tensor, each expert's `ff` gate rows
/// then its `ff` up rows) with GeGLU: `mid[i * used + s] = gelu(g) * u`.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate_up: &QWeight,
    ff: usize,
    x: &Tensor,
    sel: &Tensor,
    mid: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    check_f32(&[x, mid])?;
    ensure!(
        gate_up.rows.is_multiple_of(2 * ff),
        "gate/up rows {} of {ff}",
        gate_up.rows
    );
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * used, "selection");
    ensure!(
        mid.numel() >= m * used * ff && x.numel() >= m * gate_up.cols,
        "expert sizes"
    );
    let name = match gate_up.ty {
        GgmlType::Q8_0 => "gemma4_moe_gate_up_q8_0",
        GgmlType::Q5K => "gemma4_moe_gate_up_q5_k",
        GgmlType::Q6K => "gemma4_moe_gate_up_q6_k",
        GgmlType::Q4K => "gemma4_moe_gate_up_q4_k",
        GgmlType::Iq2Xxs => "gemma4_moe_gate_up_iq2_xxs",
        GgmlType::Iq2S => "gemma4_moe_gate_up_iq2_s",
        other => anyhow::bail!("no Gemma expert gate/up over {other:?}"),
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[gate_up.t.binding(), x.binding(), sel.binding(), mid.binding()],
        &[
            &u32_bytes(gate_up.cols),
            &u32_bytes(ff),
            &u32_bytes(gate_up.row_bytes),
            &u32_bytes(used),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(8), m * used, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Routed experts' down, weighted and summed: `out[i] = sum_s w[i][s]
/// down_{sel[i][s]} mid[i * used + s]`.
#[allow(clippy::too_many_arguments)]
pub fn moe_down(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    down: &QWeight,
    hidden: usize,
    mid: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    out: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    check_f32(&[mid, weights, out])?;
    ensure!(down.rows.is_multiple_of(hidden), "down rows {} of {hidden}", down.rows);
    ensure!(
        out.numel() >= m * hidden && mid.numel() >= m * used * down.cols,
        "expert sizes"
    );
    let name = match down.ty {
        GgmlType::Q8_0 => "gemma4_moe_down_q8_0",
        GgmlType::Iq4Nl => "gemma4_moe_down_iq4_nl",
        other => anyhow::bail!("no Gemma expert down over {other:?}"),
    };
    // IQ4_NL tasks are pairs of 32-value blocks read as 4-byte words.
    ensure!(
        (down.ty != GgmlType::Iq4Nl
            || down.cols.is_multiple_of(64) && down.row_bytes.is_multiple_of(4))
            && used <= 32,
        "expert down rows of {} values, {used} used",
        down.cols
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            down.t.binding(),
            mid.binding(),
            sel.binding(),
            weights.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(down.cols),
            &u32_bytes(hidden),
            &u32_bytes(down.row_bytes),
            &u32_bytes(used),
        ],
        Grid::Threadgroups {
            groups: (hidden.div_ceil(8), m, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Threads of a row kernel: four values a thread.
fn row_threads(n: usize) -> Result<usize> {
    ensure!(n.is_multiple_of(4) && n <= 4096, "row of {n} values");
    Ok(n / 4)
}

/// The attention sublayer's end and the FFN's inputs, `m` rows of `n`:
/// `r = rmsnorm(a) w_post + x`, `o_j = rmsnorm(r) w_j`.
#[allow(clippy::too_many_arguments)]
pub fn post_attn(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w_post: &Tensor,
    x: &Tensor,
    r: &Tensor,
    w: [&Tensor; 3],
    o: [&Tensor; 3],
    m: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    check_f32(&[a, w_post, x, r, w[0], w[1], w[2], o[0], o[1], o[2]])?;
    ensure!(
        [a, x, r, o[0], o[1], o[2]].iter().all(|t| t.numel() >= m * n),
        "post_attn of {m} x {n}"
    );
    let threads = row_threads(n)?;
    let pipeline = ctx.pipeline("gemma4_post_attn", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            a.binding(),
            w_post.binding(),
            x.binding(),
            r.binding(),
            w[0].binding(),
            w[1].binding(),
            w[2].binding(),
            o[0].binding(),
            o[1].binding(),
            o[2].binding(),
        ],
        &[&u32_bytes(n), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (threads, 1, 1) },
    )
}

/// The FFN's end, `m` rows of `n`: `x = (rmsnorm(rmsnorm(m1) w1 +
/// rmsnorm(m2) w2) w + r) * layer_scale[0]`, and with `next = (w, h)`
/// also `h = rmsnorm(x) w` (the next layer's input, or the head's).
#[allow(clippy::too_many_arguments)]
pub fn ffn_out(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    m1: &Tensor,
    m2: &Tensor,
    w: [&Tensor; 3],
    r: &Tensor,
    layer_scale: &Tensor,
    x: &Tensor,
    next: Option<(&Tensor, &Tensor)>,
    m: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    check_f32(&[m1, m2, w[0], w[1], w[2], r, layer_scale, x])?;
    ensure!([m1, m2, r, x].iter().all(|t| t.numel() >= m * n), "ffn_out of {m} x {n}");
    if let Some((wn, h)) = next {
        check_f32(&[wn, h])?;
        ensure!(h.numel() >= m * n, "ffn_out's next input");
    }
    let threads = row_threads(n)?;
    let (wn, h) = next.unwrap_or((x, x));
    let pipeline = ctx.pipeline("gemma4_ffn_out", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            m1.binding(),
            m2.binding(),
            w[0].binding(),
            w[1].binding(),
            w[2].binding(),
            r.binding(),
            layer_scale.binding(),
            x.binding(),
            wn.binding(),
            h.binding(),
        ],
        &[&u32_bytes(n), &f32_bytes(eps), &u32_bytes(usize::from(next.is_some()))],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (threads, 1, 1) },
    )
}

/// `h = rmsnorm(x) w`, `m` rows of `n`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    h: &Tensor,
    m: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    check_f32(&[x, w, h])?;
    ensure!(x.numel() >= m * n && h.numel() >= m * n, "rmsnorm of {m} x {n}");
    let threads = row_threads(n)?;
    let pipeline = ctx.pipeline("gemma4_rmsnorm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), h.binding()],
        &[&u32_bytes(n), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (threads, 1, 1) },
    )
}

/// `x = c tanh(x / c)` over `n` values in place.
pub fn softcap(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    n: usize,
    c: f32,
) -> Result<()> {
    check_f32(&[x])?;
    ensure!(x.numel() >= n, "softcap of {n}");
    let pipeline = ctx.pipeline("gemma4_softcap", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding()],
        &[&f32_bytes(c)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/gemma4.rs"]
mod tests;
