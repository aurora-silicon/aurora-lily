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
    let conv_bindings = [
        qkv.binding(),
        p.conv_w.binding(),
        st.conv.binding(),
        c.binding(),
        snap.map_or(st.conv, |s| s.conv).binding(),
    ];
    let conv_params: [&[u8]; 3] =
        [&u32_bytes(ch), &u32_bytes(m), &snap_row.to_ne_bytes()];
    // Prompt runs: the conv row-parallel, its window moved beside the scan.
    let rows = m > 4;
    if rows {
        let conv = ctx.pipeline("qwen36_gdn_conv_rows", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &conv,
            &conv_bindings,
            &conv_params,
            Grid::Threads { grid: (ch, m, 1), threadgroup: (256, 1, 1) },
        )?;
    } else {
        let conv = ctx.pipeline("qwen36_gdn_conv", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &conv,
            &conv_bindings,
            &conv_params,
            Grid::Threads { grid: (ch, 1, 1), threadgroup: (128, 1, 1) },
        )?;
    }
    pass.memory_barrier()?;
    if rows {
        let window =
            ctx.pipeline("qwen36_gdn_conv_window", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &window,
            &conv_bindings,
            &conv_params,
            Grid::Threads { grid: (ch, 1, 1), threadgroup: (256, 1, 1) },
        )?;
    }
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
    ensure!(
        qg.numel() >= m * nq * 2 * d && q_out.numel() >= m * nq * d,
        "q of {m} rows"
    );
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
    ensure!(
        attn.numel() >= m * n && y.numel() >= m * n && qg.numel() >= 2 * m * n,
        "gate"
    );
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
    ensure!(
        n_expert <= 256 && k <= 32 && k <= n_expert,
        "router of {n_expert}, top {k}"
    );
    ensure!(
        logits.numel() >= m * n_expert
            && sel.numel() >= m * k
            && weights.numel() >= m * k,
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
        x.numel() >= m * hidden
            && moe.numel() >= m * hidden
            && shared.numel() >= m * hidden,
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
    ensure!(
        shared_gate.cols == kk && shared_gate.rows == 1,
        "shared expert gate shape"
    );
    ensure!(kk.is_multiple_of(4), "router rows of {kk}");
    check(&[(x, DType::F32), (out, DType::F32), (gout, DType::F32)])?;
    ensure!(
        x.numel() >= m * kk && out.numel() >= m * e && gout.numel() >= m,
        "router rows"
    );
    let pipeline = ctx.pipeline("qwen36_router_logits", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            router.t.binding(),
            shared_gate.t.binding(),
            x.binding(),
            out.binding(),
            gout.binding(),
        ],
        &[&u32_bytes(kk), &u32_bytes(e), &u32_bytes(m)],
        Grid::Threadgroups {
            groups: (m.div_ceil(4), (e + 1).div_ceil(8), 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// `y = shared * sigmoid(g)` over `m` rows of `hidden` (the prompt path's
/// shared expert, before the grouped routed sum is added to it).
pub fn shared_gate(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    shared: &Tensor,
    g: &Tensor,
    y: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    check(&[(shared, DType::F32), (g, DType::F32), (y, DType::F32)])?;
    ensure!(
        shared.numel() >= m * hidden && y.numel() >= m * hidden,
        "shared gate rows"
    );
    let pipeline = ctx.pipeline("qwen36_shared_gate", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[shared.binding(), g.binding(), y.binding()],
        &[&u32_bytes(hidden)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Routed experts' down projection summed over each row's `used` slots
/// (`glm53::moe_down` without `add`) for rows of 256 or 512 values: all
/// lanes at work. The model's 8 slots of rows of 512 take the wide kernel
/// (`wide`; a thread a slot and task, the sum regrouped), others the one
/// whose lanes walk the slots.
#[allow(clippy::too_many_arguments)]
pub fn moe_down(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    down: &QWeight,
    hidden: usize,
    mid: &Tensor,
    sel: &Tensor,
    out: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    let wide = used == 8 && down.cols == 512;
    moe_down_with(ctx, pass, down, hidden, mid, sel, out, m, used, wide)
}

/// [`moe_down`] through the wide kernel (`wide`: 8 slots of rows of 512)
/// or the lane-walk one.
#[allow(clippy::too_many_arguments)]
pub fn moe_down_with(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    down: &QWeight,
    hidden: usize,
    mid: &Tensor,
    sel: &Tensor,
    out: &Tensor,
    m: usize,
    used: usize,
    wide: bool,
) -> Result<()> {
    check(&[(mid, DType::F32), (out, DType::F32), (sel, DType::U32)])?;
    ensure!(down.cols == 256 || down.cols == 512, "down rows of {}", down.cols);
    ensure!(down.rows.is_multiple_of(hidden), "down rows {} of {hidden}", down.rows);
    ensure!(
        out.numel() >= m * hidden && mid.numel() >= m * used * down.cols,
        "down shapes"
    );
    ensure!(
        !wide || (used == 8 && down.cols == 512),
        "the wide down kernel takes 8 slots of 512"
    );
    let name = match (down.ty, wide) {
        (GgmlType::Q4K, true) => "qwen36_moe_down_wide_q4_k",
        (GgmlType::Q8_0, true) => "qwen36_moe_down_wide_q8_0",
        (GgmlType::Q4K, false) => "qwen36_moe_down_q4_k",
        (GgmlType::Q8_0, false) => "qwen36_moe_down_q8_0",
        (t, _) => anyhow::bail!("no Qwen3.6 expert down kernel for {t:?}"),
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    if wide {
        return pass.dispatch_at(
            &pipeline,
            &[down.t.binding(), mid.binding(), sel.binding(), out.binding()],
            &[&u32_bytes(down.cols), &u32_bytes(hidden), &u32_bytes(down.row_bytes)],
            Grid::Threadgroups {
                groups: (hidden.div_ceil(2), m, 1),
                threadgroup: (256, 1, 1),
            },
        );
    }
    let rows_per_tg = 8 * 32 / (down.cols / 32);
    pass.dispatch_at(
        &pipeline,
        &[down.t.binding(), mid.binding(), sel.binding(), out.binding()],
        &[
            &u32_bytes(down.cols),
            &u32_bytes(hidden),
            &u32_bytes(down.row_bytes),
            &u32_bytes(used),
        ],
        Grid::Threadgroups {
            groups: (hidden.div_ceil(rows_per_tg), m, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// The MTP head's input rows: `cat[r] = [rmsnorm(e[r]) * enorm |
/// rmsnorm(h[r]) * hnorm]` over `m` rows of `hidden`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_cat(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    e: &Tensor,
    h: &Tensor,
    enorm: &Tensor,
    hnorm: &Tensor,
    cat: &Tensor,
    m: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    for t in [e, h, enorm, hnorm, cat] {
        check(&[(t, DType::F32)])?;
    }
    ensure!(
        e.numel() >= m * hidden
            && h.numel() >= m * hidden
            && cat.numel() >= 2 * m * hidden,
        "MTP input of {m} rows"
    );
    let pipeline = ctx.pipeline("qwen36_mtp_cat", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[e.binding(), h.binding(), enorm.binding(), hnorm.binding(), cat.binding()],
        &[&u32_bytes(hidden), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (m, 2, 1), threadgroup: (256, 1, 1) },
    )
}

/// `x += a`, then `out = rmsnorm(x) * w` (`glm53::add` then
/// `glm53::rmsnorm`; the residual the same bits, the normed rows within
/// 1e-6, as [`rmsnorm`]) over `m` rows of `hidden` (2048);
/// with `shared`, `x += a + shared * sigmoid(g)` (`ffn_residual`).
#[allow(clippy::too_many_arguments)]
pub fn residual_norm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    a: &Tensor,
    shared: Option<(&Tensor, &Tensor)>,
    w: &Tensor,
    out: &Tensor,
    m: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    for t in [x, a, w, out] {
        check(&[(t, DType::F32)])?;
    }
    ensure!(hidden == 2048 && w.numel() >= hidden, "residual norm of rows of {hidden}");
    ensure!(
        x.numel() >= m * hidden && a.numel() >= m * hidden && out.numel() >= m * hidden,
        "residual norm of {m} rows"
    );
    let (s, g) = shared.unwrap_or((a, a));
    check(&[(s, DType::F32), (g, DType::F32)])?;
    let name = if shared.is_some() {
        ensure!(s.numel() >= m * hidden && g.numel() >= m, "residual norm shared rows");
        "qwen36_ffn_residual_norm"
    } else {
        "qwen36_add_norm"
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            x.binding(),
            a.binding(),
            s.binding(),
            g.binding(),
            w.binding(),
            out.binding(),
        ],
        &[&u32_bytes(hidden), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// `glm53::rmsnorm` with its weight for rows of 2048 (the model's hidden
/// width; other widths take GLM's): a thread's 8 loads in flight at once
/// where GLM's loop waits for each. The sum of squares rounds as the
/// compiler fuses it here (normed values within 1e-6 of GLM's).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    m: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if hidden != 2048 {
        return crate::kernels::glm53::rmsnorm(
            ctx,
            pass,
            x,
            Some(w),
            out,
            m,
            hidden,
            eps,
        );
    }
    for t in [x, w, out] {
        check(&[(t, DType::F32)])?;
    }
    ensure!(w.numel() >= hidden, "rmsnorm weight shorter than {hidden}");
    ensure!(
        x.numel() >= m * hidden && out.numel() >= m * hidden,
        "rmsnorm of {m} rows"
    );
    let pipeline = ctx.pipeline("qwen36_rmsnorm_h2048", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            x.binding(),
            x.binding(),
            x.binding(),
            x.binding(),
            w.binding(),
            out.binding(),
        ],
        &[&u32_bytes(hidden), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// Moves the GDN conv window past `m` rows of `qkv` (and writes the window
/// after row `snap.row` to the snapshot), `qwen36_gdn_conv_window`: what
/// [`gdn_step`] leaves undone. It must run in a level after the step (which
/// reads the window in every threadgroup) and before the next step.
pub fn gdn_conv_window(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &GdnParams<'_>,
    st: &GdnState<'_>,
    snap: Option<&GdnSnapshot<'_>>,
    qkv: &Tensor,
    m: usize,
) -> Result<()> {
    let ch = st.conv.numel() / 3;
    check(&[(qkv, DType::F32), (st.conv, DType::F32)])?;
    ensure!(
        st.conv.numel() == 3 * ch
            && p.conv_w.numel() == 4 * ch
            && qkv.numel() >= m * ch,
        "GDN conv window shapes"
    );
    let snap_row = snap.map_or(NO_SNAPSHOT, |s| s.row as u32);
    if let Some(s) = snap {
        ensure!(s.row < m && s.conv.numel() == st.conv.numel(), "window snapshot");
    }
    // The kernel's buffers are the conv's: 0 qkv, 1 conv w, 2 window, 3 c
    // (unused), 4 snapshot.
    let window = ctx.pipeline("qwen36_gdn_conv_window", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &window,
        &[
            qkv.binding(),
            p.conv_w.binding(),
            st.conv.binding(),
            qkv.binding(),
            snap.map_or(st.conv, |s| s.conv).binding(),
        ],
        &[&u32_bytes(ch), &u32_bytes(m), &snap_row.to_ne_bytes()],
        Grid::Threads { grid: (ch, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// [`gdn`] of up to 4 rows (decode and verify) in one dispatch and level
/// (`qwen36_gdn_step`: the conv, the delta rule and the gated norm of a
/// value head in one threadgroup), the same bits; no `c` or `o` written.
/// It only reads the conv window (value heads sharing a key head read the
/// same channels): [`gdn_conv_window`] moves it in a later level.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &GdnParams<'_>,
    st: &GdnState<'_>,
    snap: Option<&GdnSnapshot<'_>>,
    qkv: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    z: &Tensor,
    y: &Tensor,
    m: usize,
    k_heads: usize,
    eps: f32,
) -> Result<()> {
    let h = p.ssm_a.numel();
    ensure!((1..=4).contains(&m), "a GDN step of {m} rows");
    ensure!(k_heads > 0 && h.is_multiple_of(k_heads), "{h} value heads over {k_heads}");
    let ch = (2 * k_heads + h) * GDN_DIM;
    for t in [qkv, alpha, beta, z, y, p.conv_w, p.ssm_a, p.dt_bias, p.norm] {
        check(&[(t, DType::F32)])?;
    }
    check(&[(st.conv, DType::F32), (st.s, DType::F32)])?;
    ensure!(qkv.numel() >= m * ch, "GDN qkv of {m} rows");
    ensure!(alpha.numel() >= m * h && beta.numel() >= m * h, "GDN gates of {m} rows");
    let inner = h * GDN_DIM;
    ensure!(
        z.numel() >= m * inner && y.numel() >= m * inner,
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
    let pipeline = ctx.pipeline("qwen36_gdn_step", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            qkv.binding(),
            p.conv_w.binding(),
            st.conv.binding(),
            snap.map_or(st.conv, |s| s.conv).binding(),
            alpha.binding(),
            beta.binding(),
            p.ssm_a.binding(),
            p.dt_bias.binding(),
            st.s.binding(),
            snap.map_or(st.s, |s| s.s).binding(),
            z.binding(),
            p.norm.binding(),
            y.binding(),
        ],
        &[
            &u32_bytes(k_heads),
            &u32_bytes(h),
            &u32_bytes(m),
            &snap_row.to_ne_bytes(),
            &f32_bytes(eps),
        ],
        Grid::Threadgroups { groups: (h, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// The routed experts' gate and up at decode (`glm53::moe_gate_up` without
/// the clamp, the same bits) for rows of 2048 values (the model's hidden
/// width) over Q4_K or Q8_0 experts, in less code; other widths take GLM's
/// kernel.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    ff: usize,
    x: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    mid: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    let name = match gate.ty {
        GgmlType::Q4K => "qwen36_moe_gate_up_q4_k_k2048",
        GgmlType::Q8_0 => "qwen36_moe_gate_up_q8_0_k2048",
        _ => "",
    };
    if gate.cols != 2048 || name.is_empty() {
        return crate::kernels::glm53::moe_gate_up(
            ctx, pass, gate, up, ff, x, sel, weights, mid, m, used, 0.0,
        );
    }
    check(&[
        (x, DType::F32),
        (mid, DType::F32),
        (sel, DType::U32),
        (weights, DType::F32),
    ])?;
    ensure!(sel.numel() >= m * used && weights.numel() >= m * used, "selection");
    ensure!(
        gate.ty == up.ty && gate.cols == up.cols && gate.rows == up.rows,
        "gate and up experts differ"
    );
    ensure!(gate.rows.is_multiple_of(ff), "expert rows {} of {ff}", gate.rows);
    ensure!(
        mid.numel() >= m * used * ff && x.numel() >= m * gate.cols,
        "gate/up shapes"
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.t.binding(),
            up.t.binding(),
            x.binding(),
            sel.binding(),
            weights.binding(),
            mid.binding(),
        ],
        &[
            &u32_bytes(gate.cols),
            &u32_bytes(ff),
            &u32_bytes(gate.row_bytes),
            &u32_bytes(used),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(8), m * used, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// [`router`] for 8 of 256 experts, one subgroup a row and no threadgroup
/// barriers (`qwen36_router_sg`): the same picks and weights, bit for bit.
pub fn router_sg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    m: usize,
) -> Result<()> {
    check(&[(logits, DType::F32), (sel, DType::U32), (weights, DType::F32)])?;
    ensure!(
        logits.numel() >= m * 256 && sel.numel() >= m * 8 && weights.numel() >= m * 8,
        "router of {m} rows of 256 logits, top 8"
    );
    let pipeline = ctx.pipeline("qwen36_router_sg", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), sel.binding(), weights.binding()],
        &[&u32_bytes(256), &u32_bytes(8)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Whether [`gemv_q8_0_k4096`] takes `w`: Q8_0 rows of 4096 values.
pub fn gemv_q8_0_k4096_takes(w: &QWeight) -> bool {
    w.ty == GgmlType::Q8_0 && w.cols == 4096
}

/// `y[i] = w x[i]` for `m` rows of x (1 to 4) through Q8_0 rows of 4096
/// values, each row over four subgroups (the sum regrouped against
/// `glm53::gemv`; a row's sums are the same bits at any `m`, so a verify
/// row is a decode step's).
pub fn gemv_q8_0_k4096(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
) -> Result<()> {
    ensure!(
        gemv_q8_0_k4096_takes(w),
        "no K-split GEMV over {:?} rows of {}",
        w.ty,
        w.cols
    );
    check(&[(x, DType::F32), (y, DType::F32)])?;
    ensure!((1..=4).contains(&m), "K-split GEMV of {m} rows");
    ensure!(x.numel() >= m * w.cols && y.numel() >= m * w.rows, "K-split GEMV shapes");
    let ks = 4;
    let name = [
        "qwen36_gemv_q8_0_k4096_s4",
        "qwen36_gemv_q8_0_k4096_s4_m2",
        "qwen36_gemv_q8_0_k4096_s4_m3",
        "qwen36_gemv_q8_0_k4096_s4_m4",
    ][m - 1];
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.t.binding(), y.binding()],
        &[&u32_bytes(w.rows)],
        Grid::Threadgroups {
            groups: (w.rows.div_ceil(8 / ks), 1, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/qwen36.rs"]
mod tests;
