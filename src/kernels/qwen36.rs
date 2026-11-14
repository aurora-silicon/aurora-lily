//! Qwen3.6 kernels (`vulkan/qwen36.comp`): the embedding, the Gated
//! DeltaNet in f32 with the GGUF's tiled value heads, the attention's
//! norm/RoPE prep and output gate, the softmax router and the FFN's end.
//! GEMVs, GEMMs, RMSNorm and the routed experts are GLM's
//! (`kernels::glm53`); the attention itself is Qwen3.8's
//! (`kernels::attention`). The CPU reference is `crate::qwen36::reference`.

use anyhow::{Result, ensure};

use crate::gguf::GgmlType;
use crate::kernels::glm53::QWeight;
use crate::kernels::{f32_bytes, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// GLSL only (see `kernels::glm53`).
const SOURCE: &str = "";

/// The GDN head width the kernels are written for.
pub const GDN_DIM: usize = 128;
/// State columns one GDN scan threadgroup takes (4 subgroups of `COLS`).
const GDN_SCAN_COLS: usize = 16;
/// The attention head width the kernels are written for.
pub const HEAD_DIM: usize = 256;
/// `snap_row` meaning "no snapshot".
pub const NO_SNAPSHOT: u32 = u32::MAX;

fn check(ts: &[(&Tensor, DType)]) -> Result<()> {
    for (t, d) in ts {
        ensure!(t.dtype() == *d, "Qwen3.6 kernel takes {d:?}, got {:?}", t.dtype());
    }
    Ok(())
}

/// `out[i]` = row `tokens[i]` of the embedding table, for `m` tokens.
pub fn embed(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &QWeight,
    tokens: &Tensor,
    out: &Tensor,
    m: usize,
) -> Result<()> {
    check(&[(tokens, DType::U32), (out, DType::F32)])?;
    ensure!(out.numel() >= m * table.cols && tokens.numel() >= m, "embed of {m} rows");
    let name = match table.ty {
        GgmlType::Q4K => "qwen36_embed_q4_k",
        GgmlType::Q8_0 => "qwen36_embed_q8_0",
        t => anyhow::bail!("no embedding kernel for {t:?}"),
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.t.binding(), tokens.binding(), out.binding()],
        &[&u32_bytes(table.cols), &u32_bytes(table.row_bytes)],
        Grid::Threads { grid: (table.cols / 32, m, 1), threadgroup: (64, 1, 1) },
    )
}

/// One GDN layer's parameters on the GPU (f32, the GGUF's tensors).
pub struct GdnParams<'a> {
    /// `[C][4]` conv taps.
    pub conv_w: &'a Tensor,
    /// `[H]`: `-exp(A_log)`.
    pub ssm_a: &'a Tensor,
    pub dt_bias: &'a Tensor,
    /// `[128]` the output RMSNorm's weight.
    pub norm: &'a Tensor,
}

/// A GDN layer's recurrent state: the conv window and the delta-rule state.
pub struct GdnState<'a> {
    /// `[3][C]`.
    pub conv: &'a Tensor,
    /// `[H][128][128]`.
    pub s: &'a Tensor,
}

/// Where a run of rows also leaves the GDN state as it was after one of its
/// rows (speculative decoding's rollback point).
pub struct GdnSnapshot<'a> {
    pub row: usize,
    pub conv: &'a Tensor,
    pub s: &'a Tensor,
}

/// The GDN over `m` consecutive rows: the conv of the mixed projection
/// `qkv` (`[m][C]`) into `c` (silu'd), the delta rule into `o`
/// (`[m][H][128]`), the gated RMSNorm with `z` into `y` (`[m][H][128]`).
/// With `snap`, the conv window and state after row `snap.row` are also
/// written there.
#[allow(clippy::too_many_arguments)]
pub fn gdn(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &GdnParams<'_>,
    st: &GdnState<'_>,
    snap: Option<&GdnSnapshot<'_>>,
    qkv: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    z: &Tensor,
    c: &Tensor,
    o: &Tensor,
    y: &Tensor,
    m: usize,
    k_heads: usize,
    eps: f32,
) -> Result<()> {
    let h = p.ssm_a.numel();
    ensure!(k_heads > 0 && h.is_multiple_of(k_heads), "{h} value heads over {k_heads}");
    let ch = (2 * k_heads + h) * GDN_DIM;
    for t in [qkv, alpha, beta, z, c, o, y, p.conv_w, p.ssm_a, p.dt_bias, p.norm] {
        check(&[(t, DType::F32)])?;
    }
    check(&[(st.conv, DType::F32), (st.s, DType::F32)])?;
    ensure!(qkv.numel() >= m * ch && c.numel() >= m * ch, "GDN qkv of {m} rows");
    ensure!(alpha.numel() >= m * h && beta.numel() >= m * h, "GDN gates of {m} rows");
    let inner = h * GDN_DIM;
    ensure!(
        z.numel() >= m * inner && o.numel() >= m * inner && y.numel() >= m * inner,
        "GDN outputs of {m} rows"
    );
    ensure!(p.conv_w.numel() == 4 * ch && st.conv.numel() == 3 * ch, "GDN conv shapes");
    ensure!(st.s.numel() == h * GDN_DIM * GDN_DIM, "GDN state of {}", st.s.numel());
    ensure!(p.norm.numel() == GDN_DIM, "GDN norm of {}", p.norm.numel());
    let snap_row = snap.map_or(NO_SNAPSHOT, |s| s.row as u32);
    if let Some(s) = snap {
        ensure!(s.row < m, "snapshot after row {} of {m}", s.row);
        ensure!(
            s.conv.numel() == st.conv.numel() && s.s.numel() == st.s.numel(),
            "snapshot shapes"
        );
    }
    let conv = ctx.pipeline("qwen36_gdn_conv", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &conv,
        &[
            qkv.binding(),
            p.conv_w.binding(),
            st.conv.binding(),
            c.binding(),
            snap.map_or(st.conv, |s| s.conv).binding(),
        ],
        &[&u32_bytes(ch), &u32_bytes(m), &snap_row.to_ne_bytes()],
        Grid::Threads { grid: (ch, 1, 1), threadgroup: (128, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let scan = ctx.pipeline("qwen36_gdn_scan", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scan,
        &[
            c.binding(),
            alpha.binding(),
            beta.binding(),
            p.ssm_a.binding(),
            p.dt_bias.binding(),
            st.s.binding(),
            o.binding(),
            snap.map_or(st.s, |s| s.s).binding(),
        ],
        &[
            &u32_bytes(k_heads),
            &u32_bytes(h),
            &u32_bytes(m),
            &snap_row.to_ne_bytes(),
            &f32_bytes(eps),
        ],
        Grid::Threadgroups {
            groups: (h, GDN_DIM / GDN_SCAN_COLS, 1),
            threadgroup: (128, 1, 1),
        },
    )?;
    pass.memory_barrier()?;
    let out = ctx.pipeline("qwen36_gdn_out", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &out,
        &[o.binding(), z.binding(), p.norm.binding(), y.binding()],
        &[&u32_bytes(h), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (h, m, 1), threadgroup: (GDN_DIM, 1, 1) },
    )
}

/// The attention inputs of `m` rows at positions `pos0..`: q (the q halves
/// of `qg`'s `[q | gate]` heads) and k RMSNormed with their head weights and
/// rotated (NeoX, first `rot` dims), v converted, all to bfloat.
#[allow(clippy::too_many_arguments)]
pub fn attn_prep(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qg: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_norm: &Tensor,
    k_norm: &Tensor,
    q_out: &Tensor,
    k_out: &Tensor,
    v_out: &Tensor,
    m: usize,
    nq: usize,
    nkv: usize,
    pos0: usize,
    rot: usize,
    base: f32,
    eps: f32,
) -> Result<()> {
    let d = HEAD_DIM;
    for t in [qg, k, v, q_norm, k_norm] {
        check(&[(t, DType::F32)])?;
    }
    for t in [q_out, k_out, v_out] {
        check(&[(t, DType::BF16)])?;
    }
    ensure!(qg.numel() >= m * nq * 2 * d && q_out.numel() >= m * nq * d, "q of {m} rows");
    ensure!(
        k.numel() >= m * nkv * d
            && v.numel() >= m * nkv * d
            && k_out.numel() >= m * nkv * d
            && v_out.numel() >= m * nkv * d,
        "k/v of {m} rows"
    );
    ensure!(rot.is_multiple_of(2) && rot <= d, "rotary dims {rot}");
    let pipeline = ctx.pipeline("qwen36_attn_prep", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            qg.binding(),
            k.binding(),
            v.binding(),
            q_norm.binding(),
            k_norm.binding(),
            q_out.binding(),
            k_out.binding(),
            v_out.binding(),
        ],
        &[
            &u32_bytes(nq),
            &u32_bytes(nkv),
            &u32_bytes(pos0),
            &u32_bytes(rot),
            &f32_bytes(base),
            &f32_bytes(eps),
        ],
        Grid::Threadgroups { groups: (m, nq + nkv, 1), threadgroup: (d, 1, 1) },
    )
}

/// `y = attn * sigmoid(gate)` over `m` rows of `nq` heads.
pub fn attn_gate(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    attn: &Tensor,
    qg: &Tensor,
    y: &Tensor,
    m: usize,
    nq: usize,
) -> Result<()> {
    check(&[(attn, DType::BF16), (qg, DType::F32), (y, DType::F32)])?;
    let n = nq * HEAD_DIM;
    ensure!(attn.numel() >= m * n && y.numel() >= m * n && qg.numel() >= 2 * m * n, "gate");
    let pipeline = ctx.pipeline("qwen36_attn_gate", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[attn.binding(), qg.binding(), y.binding()],
        &[&u32_bytes(nq)],
        Grid::Threads { grid: (n, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// The softmax router's top `k` of `n_expert` per row, weights normalized.
#[allow(clippy::too_many_arguments)]
pub fn router(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    m: usize,
    n_expert: usize,
    k: usize,
) -> Result<()> {
    check(&[(logits, DType::F32), (sel, DType::U32), (weights, DType::F32)])?;
    ensure!(n_expert <= 256 && k <= 32 && k <= n_expert, "router of {n_expert}, top {k}");
    ensure!(
        logits.numel() >= m * n_expert && sel.numel() >= m * k && weights.numel() >= m * k,
        "router of {m} rows"
    );
    let pipeline = ctx.pipeline("qwen36_router", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), sel.binding(), weights.binding()],
        &[&u32_bytes(n_expert), &u32_bytes(k)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// `x += moe + shared * sigmoid(g)` over `m` rows of `hidden`.
#[allow(clippy::too_many_arguments)]
pub fn ffn_residual(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    moe: &Tensor,
    shared: &Tensor,
    g: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    for t in [x, moe, shared, g] {
        check(&[(t, DType::F32)])?;
    }
    ensure!(
        x.numel() >= m * hidden && moe.numel() >= m * hidden && shared.numel() >= m * hidden,
        "residual of {m} rows"
    );
    let pipeline = ctx.pipeline("qwen36_ffn_residual", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), moe.binding(), shared.binding(), g.binding()],
        &[&u32_bytes(hidden)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Router logits (`out[t][e] = router[e] . x[t]`) and the shared expert's
/// gate logit (`gout[t]`) of `m` rows in f32.
#[allow(clippy::too_many_arguments)]
pub fn router_logits(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    router: &QWeight,
    shared_gate: &QWeight,
    x: &Tensor,
    out: &Tensor,
    gout: &Tensor,
    m: usize,
) -> Result<()> {
    ensure!(
        router.ty == GgmlType::F32 && shared_gate.ty == GgmlType::F32,
        "router weights must be F32"
    );
    let (kk, e) = (router.cols, router.rows);
    ensure!(shared_gate.cols == kk && shared_gate.rows == 1, "shared expert gate shape");
    ensure!(kk.is_multiple_of(4), "router rows of {kk}");
    check(&[(x, DType::F32), (out, DType::F32), (gout, DType::F32)])?;
    ensure!(x.numel() >= m * kk && out.numel() >= m * e && gout.numel() >= m, "router rows");
    let pipeline = ctx.pipeline("qwen36_router_logits", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[router.t.binding(), shared_gate.t.binding(), x.binding(), out.binding(), gout.binding()],
        &[&u32_bytes(kk), &u32_bytes(e), &u32_bytes(m)],
        Grid::Threadgroups { groups: (m.div_ceil(4), (e + 1).div_ceil(8), 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/qwen36.rs"]
mod tests;
