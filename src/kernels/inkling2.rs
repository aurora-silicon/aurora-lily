//! Inkling 2 Mini GPU kernels (`vulkan/inkling2.comp`).
//!
//! Dispatches RMSNorm, L2 clip + log-scale tau, block attention-residual mix,
//! rank-64 loop controllers, causal short convolutions (static & dynamic),
//! relative-position bias attention with zero sink, SITU MoE, and generic
//! GGML embedding gather / MoE expert projections.

use anyhow::{Result, ensure};

use crate::gguf::GgmlType;
use crate::kernels::glm53::QWeight;
use crate::kernels::{f32_bytes, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::Tensor;

const SOURCE: &str = "";

/// RMSNorm: `y[r] = rms(x[r], eps) * w` (or without `w` when `w` is `None`).
#[allow(clippy::too_many_arguments)]
pub fn rms_norm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: Option<&Tensor>,
    y: &Tensor,
    rows: usize,
    d: usize,
    eps: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_rms_norm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[x.binding(), w.unwrap_or(x).binding(), y.binding()],
        &[&u32_bytes(d), &f32_bytes(eps), &u32_bytes(usize::from(w.is_some()))],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Per-head L2 clip with optional per-channel scale and optional log-scale `tau(pos)`.
#[allow(clippy::too_many_arguments)]
pub fn l2_clip(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    scale: Option<&Tensor>,
    y: &Tensor,
    rows: usize,
    n_heads: usize,
    dim: usize,
    thresh: f32,
    pos0: usize,
    log_floor: usize,
    log_alpha: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_l2_clip", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[x.binding(), scale.unwrap_or(x).binding(), y.binding()],
        &[
            &u32_bytes(n_heads),
            &u32_bytes(dim),
            &f32_bytes(thresh),
            &u32_bytes(usize::from(scale.is_some())),
            &u32_bytes(pos0),
            &u32_bytes(log_floor),
            &f32_bytes(log_alpha),
        ],
        Grid::Threadgroups { groups: (n_heads, rows, 1), threadgroup: (32, 1, 1) },
    )
}

/// Block attention-residual mix: `states_ptrs` is a GPU `u64` array of `n_states`
/// buffer addresses (each `[rows, h]` f32), `queries` is `[n_q, h]` f32, `out` is
/// `[n_q, rows, h]` f32.
#[allow(clippy::too_many_arguments)]
pub fn attn_res_mix(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    states_ptrs: &Tensor,
    queries: &Tensor,
    out: &Tensor,
    rows: usize,
    h: usize,
    n_states: usize,
    n_q: usize,
    eps: f32,
) -> Result<()> {
    attn_res_mix_fused(
        ctx, pass, states_ptrs, queries, out, None, None, rows, h, n_states, n_q, eps,
        eps,
    )
}

/// [`attn_res_mix`] with optional fused copy of the last state (`cur -> copy_dst`)
/// and optional fused RMSNorm (`out = rms_norm(out, norm_w)` when `n_q == 1`).
#[allow(clippy::too_many_arguments)]
pub fn attn_res_mix_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    states_ptrs: &Tensor,
    queries: &Tensor,
    out: &Tensor,
    copy_dst: Option<&Tensor>,
    norm_w: Option<&Tensor>,
    rows: usize,
    h: usize,
    n_states: usize,
    n_q: usize,
    eps: f32,
    norm_eps: f32,
) -> Result<()> {
    ensure!((1..=16).contains(&n_states), "inkling2_mix n_states={n_states}");
    ensure!((1..=4).contains(&n_q), "inkling2_mix n_q={n_q}");
    let p = ctx.pipeline("inkling2_mix", SOURCE, MslVersion::V3_1)?;
    let tg = if h.is_multiple_of(4) && h <= 4096 {
        (h / 4).div_ceil(32) * 32
    } else {
        256
    };
    pass.dispatch_at(
        &p,
        &[
            states_ptrs.binding(),
            queries.binding(),
            out.binding(),
            copy_dst.unwrap_or(out).binding(),
            norm_w.unwrap_or(out).binding(),
        ],
        &[
            &u32_bytes(h),
            &u32_bytes(n_states),
            &u32_bytes(n_q),
            &f32_bytes(eps),
            &u32_bytes(usize::from(copy_dst.is_some())),
            &u32_bytes(usize::from(norm_w.is_some())),
            &f32_bytes(norm_eps),
        ],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (tg, 1, 1) },
    )
}

/// Loop controller input: `out[r] = [rms(cur[r]); rms(msg[r])]` (`[rows, 2 * h]`).
#[allow(clippy::too_many_arguments)]
pub fn ctrl_concat_norm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cur: &Tensor,
    msg: &Tensor,
    out: &Tensor,
    rows: usize,
    h: usize,
    eps: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_ctrl_concat_norm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[cur.binding(), msg.binding(), out.binding()],
        &[&u32_bytes(h), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (rows, 2, 1), threadgroup: (256, 1, 1) },
    )
}

/// Fused F32 loop controller concat-norm + down-projection (`[rows, rank]`).
#[allow(clippy::too_many_arguments)]
pub fn ctrl_down_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cur: &Tensor,
    msg: &Tensor,
    w_down: &QWeight,
    mid: &Tensor,
    rows: usize,
    h: usize,
    rank: usize,
    eps: f32,
) -> Result<()> {
    let name = match w_down.ty {
        GgmlType::F32 => "inkling2_ctrl_down_f32",
        GgmlType::Bf16 => "inkling2_ctrl_down_bf16",
        other => anyhow::bail!("ctrl_down unsupported weight type {other:?}"),
    };
    let p = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[cur.binding(), msg.binding(), w_down.t.binding(), mid.binding()],
        &[&u32_bytes(h), &u32_bytes(rank), &f32_bytes(eps)],
        Grid::Threadgroups {
            groups: (rank.div_ceil(8), rows, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Fused F32/BF16 loop controller gain+forget projection + blend (`out = (1-tanh(f))*cur + tanh(g)*msg`).
#[allow(clippy::too_many_arguments)]
pub fn ctrl_blend_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cur: &Tensor,
    msg: &Tensor,
    mid: &Tensor,
    w_gain: &QWeight,
    w_forget: &QWeight,
    out: &Tensor,
    rows: usize,
    h: usize,
    rank: usize,
) -> Result<()> {
    ensure!(
        w_gain.ty == w_forget.ty,
        "ctrl_blend mismatched types {:?} vs {:?}",
        w_gain.ty,
        w_forget.ty
    );
    let name = match w_gain.ty {
        GgmlType::F32 => "inkling2_ctrl_blend_f32",
        GgmlType::Bf16 => "inkling2_ctrl_blend_bf16",
        other => anyhow::bail!("ctrl_blend unsupported weight type {other:?}"),
    };
    let p = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            cur.binding(),
            msg.binding(),
            mid.binding(),
            w_gain.t.binding(),
            w_forget.t.binding(),
            out.binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(rank)],
        Grid::Threadgroups {
            groups: (h.div_ceil(64), rows, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Batched 3-way F32/BF16 loop controller down-projection (1 dispatch for Q, K, V).
#[allow(clippy::too_many_arguments)]
pub fn ctrl_down3_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cur: [&Tensor; 3],
    msg_out: &Tensor,
    w_down: [&QWeight; 3],
    mid_out: &Tensor,
    rows: usize,
    h: usize,
    rank: usize,
    eps: f32,
) -> Result<()> {
    let ty = w_down[0].ty;
    ensure!(
        w_down[1].ty == ty && w_down[2].ty == ty,
        "ctrl_down3 mismatched weight types"
    );
    let name = match ty {
        GgmlType::F32 => "inkling2_ctrl_down3_f32",
        GgmlType::Bf16 => "inkling2_ctrl_down3_bf16",
        other => anyhow::bail!("ctrl_down3 unsupported weight type {other:?}"),
    };
    let p = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            cur[0].binding(),
            cur[1].binding(),
            cur[2].binding(),
            msg_out.binding(),
            w_down[0].t.binding(),
            w_down[1].t.binding(),
            w_down[2].t.binding(),
            mid_out.binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(rank), &f32_bytes(eps), &u32_bytes(rows)],
        Grid::Threadgroups {
            groups: (rank.div_ceil(8), rows, 3),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Batched 3-way F32/BF16 coalesced loop controller gain+forget+blend (1 dispatch for Q, K, V).
#[allow(clippy::too_many_arguments)]
pub fn ctrl_blend3_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cur: [&Tensor; 3],
    msg_out: &Tensor,
    mid_out: &Tensor,
    w_gain: [&QWeight; 3],
    w_forget: [&QWeight; 3],
    out: [&Tensor; 3],
    rows: usize,
    h: usize,
    rank: usize,
) -> Result<()> {
    let ty = w_gain[0].ty;
    ensure!(
        w_gain[1].ty == ty
            && w_gain[2].ty == ty
            && w_forget[0].ty == ty
            && w_forget[1].ty == ty
            && w_forget[2].ty == ty,
        "ctrl_blend3 mismatched weight types"
    );
    let name = match ty {
        GgmlType::F32 => "inkling2_ctrl_blend3_f32",
        GgmlType::Bf16 => "inkling2_ctrl_blend3_bf16",
        other => anyhow::bail!("ctrl_blend3 unsupported weight type {other:?}"),
    };
    let p = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            cur[0].binding(),
            cur[1].binding(),
            cur[2].binding(),
            msg_out.binding(),
            mid_out.binding(),
            w_gain[0].t.binding(),
            w_gain[1].t.binding(),
            w_gain[2].t.binding(),
            w_forget[0].t.binding(),
            w_forget[1].t.binding(),
            w_forget[2].t.binding(),
            out[0].binding(),
            out[1].binding(),
            out[2].binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(rank), &u32_bytes(rows)],
        Grid::Threadgroups {
            groups: (h.div_ceil(64), rows, 3),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Batched 3-way RMSNorm (1 dispatch for Q, K, V).
#[allow(clippy::too_many_arguments)]
pub fn rms_norm3(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: [&Tensor; 3],
    w: [&Tensor; 3],
    y: [&Tensor; 3],
    rows: usize,
    h: usize,
    eps: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_rms_norm3", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            x[0].binding(),
            x[1].binding(),
            x[2].binding(),
            w[0].binding(),
            w[1].binding(),
            w[2].binding(),
            y[0].binding(),
            y[1].binding(),
            y[2].binding(),
        ],
        &[&u32_bytes(h), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (rows, 3, 1), threadgroup: (256, 1, 1) },
    )
}

/// Loop controller blend: `out = (1 - tanh(forget)) * cur + tanh(gain) * msg`.
#[allow(clippy::too_many_arguments)]
pub fn ctrl_blend(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cur: &Tensor,
    msg: &Tensor,
    gain: &Tensor,
    forget: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_ctrl_blend", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            cur.binding(),
            msg.binding(),
            gain.binding(),
            forget.binding(),
            out.binding(),
        ],
        &[&u32_bytes(n)],
        Grid::Threadgroups {
            groups: (n.div_ceil(256), 1, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Causal short convolution with residual, optional dynamic per-head taps,
/// and in-place history update (`[channels, n_conv - 1]`).
#[allow(clippy::too_many_arguments)]
pub fn short_conv(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    weight: &Tensor,
    dyn_w: Option<&Tensor>,
    history: &Tensor,
    y: &Tensor,
    rows: usize,
    channels: usize,
    n_conv: usize,
    n_heads: usize,
) -> Result<()> {
    short_conv_traced(
        ctx, pass, x, weight, dyn_w, history, y, None, rows, channels, n_conv, n_heads,
    )
}

/// [`short_conv`] with optional per-row post-update history trace (`[rows, channels, n_conv - 1]`)
/// and optional fused residual add (`y += residual`).
#[allow(clippy::too_many_arguments)]
pub fn short_conv_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    weight: &Tensor,
    dyn_w: Option<&Tensor>,
    history: &Tensor,
    y: &Tensor,
    trace: Option<&Tensor>,
    residual: Option<&Tensor>,
    rows: usize,
    channels: usize,
    n_conv: usize,
    n_heads: usize,
) -> Result<()> {
    ensure!((2..=8).contains(&n_conv), "short_conv n_conv={n_conv}");
    let p = ctx.pipeline("inkling2_sconv", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            x.binding(),
            weight.binding(),
            dyn_w.unwrap_or(x).binding(),
            history.binding(),
            y.binding(),
            trace.unwrap_or(history).binding(),
            residual.unwrap_or(x).binding(),
        ],
        &[
            &u32_bytes(rows),
            &u32_bytes(channels),
            &u32_bytes(n_conv),
            &u32_bytes(if dyn_w.is_some() { n_heads } else { 0 }),
            &u32_bytes(usize::from(trace.is_some())),
            &u32_bytes(usize::from(residual.is_some())),
        ],
        Grid::Threadgroups {
            groups: (channels.div_ceil(256), 1, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// [`short_conv`] with optional per-row post-update history trace.
#[allow(clippy::too_many_arguments)]
pub fn short_conv_traced(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    weight: &Tensor,
    dyn_w: Option<&Tensor>,
    history: &Tensor,
    y: &Tensor,
    trace: Option<&Tensor>,
    rows: usize,
    channels: usize,
    n_conv: usize,
    n_heads: usize,
) -> Result<()> {
    short_conv_fused(
        ctx, pass, x, weight, dyn_w, history, y, trace, None, rows, channels, n_conv, n_heads,
    )
}

/// Store f32 K and V (`[rows, kv_width]`) into f16 KV cache.
#[allow(clippy::too_many_arguments)]
pub fn kv_store(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    k_in: &Tensor,
    v_in: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    rows: usize,
    kv_width: usize,
    pos0: usize,
    slots: usize,
    is_swa: bool,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_kv_store", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[k_in.binding(), v_in.binding(), k_cache.binding(), v_cache.binding()],
        &[
            &u32_bytes(kv_width),
            &u32_bytes(pos0),
            &u32_bytes(slots),
            &u32_bytes(usize::from(is_swa)),
        ],
        Grid::Threadgroups {
            groups: (kv_width.div_ceil(256), rows, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Relative-position bias GQA attention with zero sink (`head_dim = 128`).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    rel_table: &Tensor,
    sinks: &Tensor,
    out: &Tensor,
    rows: usize,
    n_head: usize,
    n_head_kv: usize,
    pos0: usize,
    slots: usize,
    is_swa: bool,
    window: usize,
    extent: usize,
) -> Result<()> {
    attention_gated(
        ctx, pass, q, k_cache, v_cache, rel_table, sinks, out, None, rows, n_head,
        n_head_kv, pos0, slots, is_swa, window, extent,
    )
}

/// Fused K/V short_conv + L2 clip(K) / RMSNorm(V) + f16 KV cache store
/// (eliminates 4 dispatches + 2 barriers + k_conv/v_conv traffic per layer).
#[allow(clippy::too_many_arguments)]
pub fn kv_prep(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    k_raw: &Tensor,
    v_raw: &Tensor,
    k_conv_w: &Tensor,
    v_conv_w: &Tensor,
    k_dyn: Option<&Tensor>,
    v_dyn: Option<&Tensor>,
    k_hist: &Tensor,
    v_hist: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    k_trace: Option<&Tensor>,
    v_trace: Option<&Tensor>,
    rows: usize,
    n_head_kv: usize,
    n_conv: usize,
    is_swa: bool,
    pos0: usize,
    slots: usize,
    qk_clip: f32,
    rms_eps: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_kv_prep", SOURCE, MslVersion::V3_1)?;
    let has_trace = k_trace.is_some() && v_trace.is_some();
    pass.dispatch_at(
        &p,
        &[
            k_raw.binding(),
            v_raw.binding(),
            k_conv_w.binding(),
            v_conv_w.binding(),
            k_dyn.unwrap_or(k_raw).binding(),
            v_dyn.unwrap_or(v_raw).binding(),
            k_hist.binding(),
            v_hist.binding(),
            k_cache.binding(),
            v_cache.binding(),
            k_trace.unwrap_or(k_hist).binding(),
            v_trace.unwrap_or(v_hist).binding(),
        ],
        &[
            &u32_bytes(rows),
            &u32_bytes(n_head_kv),
            &u32_bytes(n_conv),
            &u32_bytes(usize::from(is_swa)),
            &u32_bytes(pos0),
            &u32_bytes(slots),
            &f32_bytes(qk_clip),
            &f32_bytes(rms_eps),
            &u32_bytes(usize::from(has_trace)),
        ],
        Grid::Threadgroups { groups: (n_head_kv, 2, 1), threadgroup: (128, 1, 1) },
    )
}

/// [`attention`] with optional fused output gate (`out = attn * sigmoid(gate)`).
#[allow(clippy::too_many_arguments)]
pub fn attention_gated(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    rel_table: &Tensor,
    sinks: &Tensor,
    out: &Tensor,
    gate: Option<&Tensor>,
    rows: usize,
    n_head: usize,
    n_head_kv: usize,
    pos0: usize,
    slots: usize,
    is_swa: bool,
    window: usize,
    extent: usize,
) -> Result<()> {
    attention_fused(
        ctx, pass, q, k_cache, v_cache, rel_table, sinks, out, gate, None, rows, n_head,
        n_head_kv, pos0, slots, is_swa, window, extent, 0.0, 0, 0.0,
    )
}

/// [`attention`] with optional fused sigmoid output gate AND optional fused Q L2 clip + log-scale tau.
#[allow(clippy::too_many_arguments)]
pub fn attention_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    rel_table: &Tensor,
    sinks: &Tensor,
    out: &Tensor,
    gate: Option<&Tensor>,
    q_clip: Option<&Tensor>,
    rows: usize,
    n_head: usize,
    n_head_kv: usize,
    pos0: usize,
    slots: usize,
    is_swa: bool,
    window: usize,
    extent: usize,
    qk_clip: f32,
    log_floor: usize,
    log_alpha: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_attn", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            rel_table.binding(),
            sinks.binding(),
            out.binding(),
            gate.unwrap_or(out).binding(),
            q_clip.unwrap_or(q).binding(),
        ],
        &[
            &u32_bytes(n_head),
            &u32_bytes(n_head_kv),
            &u32_bytes(pos0),
            &u32_bytes(slots),
            &u32_bytes(usize::from(is_swa)),
            &u32_bytes(window),
            &u32_bytes(extent),
            &u32_bytes(usize::from(gate.is_some())),
            &u32_bytes(usize::from(q_clip.is_some())),
            &f32_bytes(qk_clip),
            &u32_bytes(log_floor),
            &f32_bytes(log_alpha),
        ],
        Grid::Threadgroups { groups: (n_head, rows, 1), threadgroup: (32, 1, 1) },
    )
}

/// `out[i] = silu(gate[i]) * up[i]`.
pub fn silu_mul(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &Tensor,
    up: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_silu_mul", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[gate.binding(), up.binding(), out.binding()],
        &[&u32_bytes(n)],
        Grid::Threadgroups { groups: (n.div_ceil(256), 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// DSpark Markov GRU update: updates `rnn_state` in place and writes `latent`.
pub fn dspark_gru(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    joint_out: &Tensor,
    bias: &Tensor,
    rnn_state: &Tensor,
    latent: &Tensor,
    rank: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_dspark_gru", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[joint_out.binding(), bias.binding(), rnn_state.binding(), latent.binding()],
        &[&u32_bytes(rank)],
        Grid::Threadgroups { groups: (rank.div_ceil(256), 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// DSpark GPU feature fusion + RMSNorm from `cap[n_targets, cap_stride, H]`.
#[allow(clippy::too_many_arguments)]
pub fn dspark_fuse_norm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cap: &Tensor,
    weights: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    rows: usize,
    h: usize,
    cap_stride: usize,
    n_targets: usize,
    eps: f32,
    row_off: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_dspark_fuse_norm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[cap.binding(), weights.binding(), norm_w.binding(), out.binding()],
        &[
            &u32_bytes(h),
            &u32_bytes(cap_stride),
            &u32_bytes(n_targets),
            &f32_bytes(eps),
            &u32_bytes(row_off),
        ],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// DSpark per-head RMSNorm + Neox RoPE (in-place F32, or F16 store into `k_ctx`/`v_ctx`).
#[allow(clippy::too_many_arguments)]
pub fn dspark_qk_rope(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    norm_w: &Tensor,
    v_in: Option<&Tensor>,
    k_ctx: Option<&Tensor>,
    v_ctx: Option<&Tensor>,
    rows: usize,
    n_head: usize,
    hd: usize,
    pos0: usize,
    eps: f32,
    rope_theta: f32,
) -> Result<()> {
    let store_ctx = v_in.is_some() && k_ctx.is_some() && v_ctx.is_some();
    let p = ctx.pipeline("inkling2_dspark_qk_rope", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            x.binding(),
            norm_w.binding(),
            v_in.unwrap_or(x).binding(),
            k_ctx.unwrap_or(x).binding(),
            v_ctx.unwrap_or(x).binding(),
        ],
        &[
            &u32_bytes(n_head),
            &u32_bytes(hd),
            &u32_bytes(pos0),
            &f32_bytes(eps),
            &f32_bytes(rope_theta),
            &u32_bytes(usize::from(store_ctx)),
        ],
        Grid::Threadgroups { groups: (n_head, rows, 1), threadgroup: (128, 1, 1) },
    )
}

/// DSpark non-causal GQA over `[ctx_kv(F16) ; block_kv(F32)]`.
#[allow(clippy::too_many_arguments)]
pub fn dspark_attn(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_ctx: &Tensor,
    v_ctx: &Tensor,
    k_blk: &Tensor,
    v_blk: &Tensor,
    out: &Tensor,
    n_head: usize,
    n_head_kv: usize,
    hd: usize,
    ctx_start: usize,
    ctx_pos: usize,
    block: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_dspark_attn", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            q.binding(),
            k_ctx.binding(),
            v_ctx.binding(),
            k_blk.binding(),
            v_blk.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(n_head),
            &u32_bytes(n_head_kv),
            &u32_bytes(hd),
            &u32_bytes(ctx_start),
            &u32_bytes(ctx_pos),
            &u32_bytes(block),
        ],
        Grid::Threadgroups { groups: (n_head, block, 1), threadgroup: (32, 1, 1) },
    )
}

/// Stages `[anchor_embd ; mask_embd ; ...]` into `cur[block, H]`.
pub fn dspark_init_cur(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    anchor_embd: &Tensor,
    mask_embd: &Tensor,
    cur: &Tensor,
    block: usize,
    h: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_dspark_init_cur", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[anchor_embd.binding(), mask_embd.binding(), cur.binding()],
        &[&u32_bytes(h)],
        Grid::Threadgroups {
            groups: (h.div_ceil(256), block, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// `out[i] = a[i] + b[i] + c[i]`.
pub fn add3(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_add3", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[a.binding(), b.binding(), c.binding(), out.binding()],
        &[&u32_bytes(n)],
        Grid::Threadgroups { groups: (n.div_ceil(256), 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// `out[i] = a[i] * sigmoid(g[i])`.
pub fn gate_mul(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    g: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_gate_mul", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[a.binding(), g.binding(), out.binding()],
        &[&u32_bytes(n)],
        Grid::Threadgroups {
            groups: (n.div_ceil(256), 1, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// `out[i] = a[i] + b[i]`.
pub fn add(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_add", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[a.binding(), b.binding(), out.binding()],
        &[&u32_bytes(n)],
        Grid::Threadgroups {
            groups: (n.div_ceil(256), 1, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Final logit softcap + unpad (`[rows, cols_in]` -> `[rows, cols_out]`).
#[allow(clippy::too_many_arguments)]
pub fn softcap(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    out: &Tensor,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    cap: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_softcap", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[x.binding(), out.binding()],
        &[&u32_bytes(cols_out), &u32_bytes(cols_in), &f32_bytes(cap)],
        Grid::Threadgroups {
            groups: (cols_out.div_ceil(256), rows, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Maximum split-K chunks for decode/verify attention (`inkling2_attn_split`).
pub const ATTN_SPLIT_MAX_CHUNKS: usize = 128;

/// Maximum split-K context chunks for DSpark attention (`inkling2_dspark_attn_split`).
pub const DSPARK_SPLIT_MAX_CHUNKS: usize = 64;

/// Maximum rows an expert's single tile computes via the register dot path in [`moe_grouped_mu`].
pub const MOE_MU_DOT_ROWS: usize = 4;

/// Prompt-run FlashAttention on the matrix units (`inkling2_attn_mu`, `head_dim = 128`).
/// Requires `ctx.matrix_units()` and `(!is_swa || slots >= 128)`.
#[allow(clippy::too_many_arguments)]
pub fn attention_mu(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    rel_table: &Tensor,
    sinks: &Tensor,
    out: &Tensor,
    gate: Option<&Tensor>,
    q_clip: Option<&Tensor>,
    rows: usize,
    n_head: usize,
    n_head_kv: usize,
    pos0: usize,
    slots: usize,
    is_swa: bool,
    window: usize,
    extent: usize,
    qk_clip: f32,
    log_floor: usize,
    log_alpha: f32,
) -> Result<()> {
    ensure!(ctx.matrix_units(), "inkling2_attn_mu requires matrix units");
    ensure!(!is_swa || slots >= 128, "inkling2_attn_mu requires slots >= 128 for SWA");
    ensure!(n_head.is_multiple_of(n_head_kv), "inkling2_attn_mu H={n_head} kvh={n_head_kv}");
    let g = n_head / n_head_kv;
    let p = ctx.pipeline("inkling2_attn_mu", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            rel_table.binding(),
            sinks.binding(),
            out.binding(),
            gate.unwrap_or(out).binding(),
            q_clip.unwrap_or(q).binding(),
        ],
        &[
            &u32_bytes(n_head),
            &u32_bytes(n_head_kv),
            &u32_bytes(pos0),
            &u32_bytes(slots),
            &u32_bytes(usize::from(is_swa)),
            &u32_bytes(window),
            &u32_bytes(extent),
            &u32_bytes(rows),
            &u32_bytes(usize::from(gate.is_some())),
            &u32_bytes(usize::from(q_clip.is_some())),
            &f32_bytes(qk_clip),
            &u32_bytes(log_floor),
            &f32_bytes(log_alpha),
        ],
        Grid::Threadgroups {
            groups: ((rows * g).div_ceil(16), n_head_kv, 1),
            threadgroup: (128, 1, 1),
        },
    )
}

fn moe_gemm_gu_name(ty: GgmlType, r128: bool) -> Result<&'static str> {
    Ok(match (ty, r128) {
        (GgmlType::Q8_0, false) => "inkling2_gemm_moe_gu_situ_q8_0",
        (GgmlType::Q8_0, true) => "inkling2_gemm_moe_gu_situ_q8_0_r128",
        (GgmlType::Q4K, false) => "inkling2_gemm_moe_gu_situ_q4_k",
        (GgmlType::Q4K, true) => "inkling2_gemm_moe_gu_situ_q4_k_r128",
        (GgmlType::Q2K, false) => "inkling2_gemm_moe_gu_situ_q2_k",
        (GgmlType::Q2K, true) => "inkling2_gemm_moe_gu_situ_q2_k_r128",
        (GgmlType::Iq2Xxs, false) => "inkling2_gemm_moe_gu_situ_iq2_xxs",
        (GgmlType::Iq2Xxs, true) => "inkling2_gemm_moe_gu_situ_iq2_xxs_r128",
        (GgmlType::Bf16, false) => "inkling2_gemm_moe_gu_situ_bf16",
        (GgmlType::Bf16, true) => "inkling2_gemm_moe_gu_situ_bf16_r128",
        (GgmlType::F32, false) => "inkling2_gemm_moe_gu_situ_f32",
        (GgmlType::F32, true) => "inkling2_gemm_moe_gu_situ_f32_r128",
        (GgmlType::F16, false) => "inkling2_gemm_moe_gu_situ_f16",
        (GgmlType::F16, true) => "inkling2_gemm_moe_gu_situ_f16_r128",
        (GgmlType::Q4_0, false) => "inkling2_gemm_moe_gu_situ_q4_0",
        (GgmlType::Q4_0, true) => "inkling2_gemm_moe_gu_situ_q4_0_r128",
        (GgmlType::Q5K, false) => "inkling2_gemm_moe_gu_situ_q5_k",
        (GgmlType::Q5K, true) => "inkling2_gemm_moe_gu_situ_q5_k_r128",
        (GgmlType::Q6K, false) => "inkling2_gemm_moe_gu_situ_q6_k",
        (GgmlType::Q6K, true) => "inkling2_gemm_moe_gu_situ_q6_k_r128",
        (GgmlType::Iq4Nl, false) => "inkling2_gemm_moe_gu_situ_iq4_nl",
        (GgmlType::Iq4Nl, true) => "inkling2_gemm_moe_gu_situ_iq4_nl_r128",
        (GgmlType::Iq2S, false) => "inkling2_gemm_moe_gu_situ_iq2_s",
        (GgmlType::Iq2S, true) => "inkling2_gemm_moe_gu_situ_iq2_s_r128",
        (GgmlType::Mxfp4, false) => "inkling2_gemm_moe_gu_situ_mxfp4",
        (GgmlType::Mxfp4, true) => "inkling2_gemm_moe_gu_situ_mxfp4_r128",
        (GgmlType::Nvfp4, false) => "inkling2_gemm_moe_gu_situ_nvfp4",
        (GgmlType::Nvfp4, true) => "inkling2_gemm_moe_gu_situ_nvfp4_r128",
        (GgmlType::Q2_0, false) => "inkling2_gemm_moe_gu_situ_q2_0",
        (GgmlType::Q2_0, true) => "inkling2_gemm_moe_gu_situ_q2_0_r128",
        (GgmlType::Iq2Xs, false) => "inkling2_gemm_moe_gu_situ_iq2_xs",
        (GgmlType::Iq2Xs, true) => "inkling2_gemm_moe_gu_situ_iq2_xs_r128",
        (GgmlType::Iq3Xxs, false) => "inkling2_gemm_moe_gu_situ_iq3_xxs",
        (GgmlType::Iq3Xxs, true) => "inkling2_gemm_moe_gu_situ_iq3_xxs_r128",
        (GgmlType::Iq1S, false) => "inkling2_gemm_moe_gu_situ_iq1_s",
        (GgmlType::Iq1S, true) => "inkling2_gemm_moe_gu_situ_iq1_s_r128",
        (GgmlType::Iq4Xs, false) => "inkling2_gemm_moe_gu_situ_iq4_xs",
        (GgmlType::Iq4Xs, true) => "inkling2_gemm_moe_gu_situ_iq4_xs_r128",
        (GgmlType::I32, _) => anyhow::bail!("unsupported GGML type I32"),
    })
}

fn moe_gemm_down_name(ty: GgmlType, r128: bool) -> Result<&'static str> {
    Ok(match (ty, r128) {
        (GgmlType::Q8_0, false) => "inkling2_gemm_moe_down_q8_0",
        (GgmlType::Q8_0, true) => "inkling2_gemm_moe_down_q8_0_r128",
        (GgmlType::Q4K, false) => "inkling2_gemm_moe_down_q4_k",
        (GgmlType::Q4K, true) => "inkling2_gemm_moe_down_q4_k_r128",
        (GgmlType::Q2K, false) => "inkling2_gemm_moe_down_q2_k",
        (GgmlType::Q2K, true) => "inkling2_gemm_moe_down_q2_k_r128",
        (GgmlType::Iq2Xxs, false) => "inkling2_gemm_moe_down_iq2_xxs",
        (GgmlType::Iq2Xxs, true) => "inkling2_gemm_moe_down_iq2_xxs_r128",
        (GgmlType::Bf16, false) => "inkling2_gemm_moe_down_bf16",
        (GgmlType::Bf16, true) => "inkling2_gemm_moe_down_bf16_r128",
        (GgmlType::F32, false) => "inkling2_gemm_moe_down_f32",
        (GgmlType::F32, true) => "inkling2_gemm_moe_down_f32_r128",
        (GgmlType::F16, false) => "inkling2_gemm_moe_down_f16",
        (GgmlType::F16, true) => "inkling2_gemm_moe_down_f16_r128",
        (GgmlType::Q4_0, false) => "inkling2_gemm_moe_down_q4_0",
        (GgmlType::Q4_0, true) => "inkling2_gemm_moe_down_q4_0_r128",
        (GgmlType::Q5K, false) => "inkling2_gemm_moe_down_q5_k",
        (GgmlType::Q5K, true) => "inkling2_gemm_moe_down_q5_k_r128",
        (GgmlType::Q6K, false) => "inkling2_gemm_moe_down_q6_k",
        (GgmlType::Q6K, true) => "inkling2_gemm_moe_down_q6_k_r128",
        (GgmlType::Iq4Nl, false) => "inkling2_gemm_moe_down_iq4_nl",
        (GgmlType::Iq4Nl, true) => "inkling2_gemm_moe_down_iq4_nl_r128",
        (GgmlType::Iq2S, false) => "inkling2_gemm_moe_down_iq2_s",
        (GgmlType::Iq2S, true) => "inkling2_gemm_moe_down_iq2_s_r128",
        (GgmlType::Mxfp4, false) => "inkling2_gemm_moe_down_mxfp4",
        (GgmlType::Mxfp4, true) => "inkling2_gemm_moe_down_mxfp4_r128",
        (GgmlType::Nvfp4, false) => "inkling2_gemm_moe_down_nvfp4",
        (GgmlType::Nvfp4, true) => "inkling2_gemm_moe_down_nvfp4_r128",
        (GgmlType::Q2_0, false) => "inkling2_gemm_moe_down_q2_0",
        (GgmlType::Q2_0, true) => "inkling2_gemm_moe_down_q2_0_r128",
        (GgmlType::Iq2Xs, false) => "inkling2_gemm_moe_down_iq2_xs",
        (GgmlType::Iq2Xs, true) => "inkling2_gemm_moe_down_iq2_xs_r128",
        (GgmlType::Iq3Xxs, false) => "inkling2_gemm_moe_down_iq3_xxs",
        (GgmlType::Iq3Xxs, true) => "inkling2_gemm_moe_down_iq3_xxs_r128",
        (GgmlType::Iq1S, false) => "inkling2_gemm_moe_down_iq1_s",
        (GgmlType::Iq1S, true) => "inkling2_gemm_moe_down_iq1_s_r128",
        (GgmlType::Iq4Xs, false) => "inkling2_gemm_moe_down_iq4_xs",
        (GgmlType::Iq4Xs, true) => "inkling2_gemm_moe_down_iq4_xs_r128",
        (GgmlType::I32, _) => anyhow::bail!("unsupported GGML type I32"),
    })
}

/// Matrix-unit grouped MoE for prompt prefill (`inkling2_gemm.comp`):
/// `glm53_moe_group` + `glm53_to_bf16` + fused gate/up+SITU GEMM (BF16 `mid`)
/// + down GEMM (F32 `down_out`) + `inkling2_moe_sum`.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_mu(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    xb: &Tensor,
    gate_up: &QWeight,
    gate_up_scale: Option<&Tensor>,
    down: &QWeight,
    down_scale: Option<&Tensor>,
    exp_ids: &Tensor,
    exp_w: &Tensor,
    pairs: &Tensor,
    tiles: &Tensor,
    tile_count: &Tensor,
    mid_bf16: &Tensor,
    down_out: &Tensor,
    moe_out: &Tensor,
    rows: usize,
    h: usize,
    ff: usize,
    n_expert: usize,
    k_exp: usize,
    beta: f32,
    linear_beta: f32,
    force_r128: Option<bool>,
) -> Result<()> {
    ensure!(ctx.matrix_units(), "moe_grouped_mu requires matrix units");
    ensure!(
        h.is_multiple_of(64) && ff.is_multiple_of(64),
        "moe_grouped_mu requires H={h} and ff={ff} multiples of 64"
    );
    let total_pairs = rows * k_exp;
    let r128 = force_r128.unwrap_or(total_pairs >= 48 * n_expert);
    let tile_rows = if r128 { 128 } else { 64 };
    let max_tiles = total_pairs.div_ceil(tile_rows) + n_expert;

    let group = ctx.pipeline("glm53_moe_group", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &group,
        &[exp_ids.binding(), pairs.binding(), tiles.binding(), tile_count.binding()],
        &[&u32_bytes(total_pairs), &u32_bytes(n_expert), &u32_bytes(tile_rows)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (1024, 1, 1) },
    )?;
    crate::kernels::glm53::to_bf16(ctx, pass, x, xb, rows * h)?;
    pass.memory_barrier()?;

    let x_shift = if k_exp.is_power_of_two() {
        k_exp.trailing_zeros() as usize
    } else {
        0x8000_0000 | k_exp
    };
    let p_gu = ctx.pipeline(moe_gemm_gu_name(gate_up.ty, r128)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p_gu,
        &[
            xb.binding(),
            gate_up.t.binding(),
            mid_bf16.binding(),
            pairs.binding(),
            tiles.binding(),
            tile_count.binding(),
            gate_up_scale.unwrap_or(x).binding(),
        ],
        &[
            &u32_bytes(ff),
            &u32_bytes(h),
            &u32_bytes(gate_up.row_bytes),
            &u32_bytes(x_shift),
            &u32_bytes(usize::from(gate_up_scale.is_some())),
            &f32_bytes(beta),
            &f32_bytes(linear_beta),
            &u32_bytes(MOE_MU_DOT_ROWS),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(32), max_tiles, 1),
            threadgroup: (128, 1, 1),
        },
    )?;
    pass.memory_barrier()?;

    let p_dn = ctx.pipeline(moe_gemm_down_name(down.ty, r128)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p_dn,
        &[
            mid_bf16.binding(),
            down.t.binding(),
            down_out.binding(),
            pairs.binding(),
            tiles.binding(),
            tile_count.binding(),
        ],
        &[
            &u32_bytes(h),
            &u32_bytes(ff),
            &u32_bytes(down.row_bytes),
            &u32_bytes(0),
            &u32_bytes(MOE_MU_DOT_ROWS),
        ],
        Grid::Threadgroups {
            groups: (h.div_ceil(64), max_tiles, 1),
            threadgroup: (128, 1, 1),
        },
    )?;
    pass.memory_barrier()?;

    moe_sum(ctx, pass, down_out, exp_ids, exp_w, down_scale, moe_out, rows, k_exp, h)
}

fn choose_g_step(n_head: usize, n_head_kv: usize) -> usize {
    if n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
        return 1;
    }
    let g = n_head / n_head_kv;
    if g <= 4 {
        g
    } else if g.is_multiple_of(4) {
        4
    } else if g.is_multiple_of(3) {
        3
    } else if g.is_multiple_of(2) {
        2
    } else {
        1
    }
}

/// Parallel-across-rows prefill short conv (does NOT mutate `history`; call
/// [`short_conv_hist_update`] after a memory barrier).
#[allow(clippy::too_many_arguments)]
pub fn short_conv_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    weight: &Tensor,
    dyn_w: Option<&Tensor>,
    history: &Tensor,
    y: &Tensor,
    residual: Option<&Tensor>,
    rows: usize,
    channels: usize,
    n_conv: usize,
    n_heads: usize,
) -> Result<()> {
    ensure!((2..=8).contains(&n_conv), "short_conv_prefill n_conv={n_conv}");
    let p = ctx.pipeline("inkling2_sconv_prefill", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            x.binding(),
            weight.binding(),
            dyn_w.unwrap_or(x).binding(),
            history.binding(),
            y.binding(),
            residual.unwrap_or(x).binding(),
        ],
        &[
            &u32_bytes(rows),
            &u32_bytes(channels),
            &u32_bytes(n_conv),
            &u32_bytes(if dyn_w.is_some() { n_heads } else { 0 }),
            &u32_bytes(usize::from(residual.is_some())),
        ],
        Grid::Threadgroups {
            groups: (channels.div_ceil(256), rows, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Updates `history[C, hist_len]` from the last `hist_len` rows of `x[rows, C]`
/// (requires `rows >= hist_len`).
pub fn short_conv_hist_update(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    history: &Tensor,
    rows: usize,
    channels: usize,
    hist_len: usize,
) -> Result<()> {
    ensure!(rows >= hist_len, "short_conv_hist_update rows {rows} < hist_len {hist_len}");
    let p = ctx.pipeline("inkling2_sconv_hist_update", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[x.binding(), history.binding()],
        &[&u32_bytes(rows), &u32_bytes(channels), &u32_bytes(hist_len)],
        Grid::Threadgroups {
            groups: (channels.div_ceil(256), 1, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Parallel-across-rows prefill K/V short_conv + L2 clip(K) / RMSNorm(V) + f16 KV store
/// (does NOT mutate `k_hist`/`v_hist`; call [`short_conv_hist_update`] after a barrier).
#[allow(clippy::too_many_arguments)]
pub fn kv_prep_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    k_raw: &Tensor,
    v_raw: &Tensor,
    k_conv_w: &Tensor,
    v_conv_w: &Tensor,
    k_dyn: Option<&Tensor>,
    v_dyn: Option<&Tensor>,
    k_hist: &Tensor,
    v_hist: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    rows: usize,
    n_head_kv: usize,
    n_conv: usize,
    is_swa: bool,
    pos0: usize,
    slots: usize,
    qk_clip: f32,
    rms_eps: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_kv_prep_prefill", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            k_raw.binding(),
            v_raw.binding(),
            k_conv_w.binding(),
            v_conv_w.binding(),
            k_dyn.unwrap_or(k_raw).binding(),
            v_dyn.unwrap_or(v_raw).binding(),
            k_hist.binding(),
            v_hist.binding(),
            k_cache.binding(),
            v_cache.binding(),
        ],
        &[
            &u32_bytes(rows),
            &u32_bytes(n_head_kv),
            &u32_bytes(n_conv),
            &u32_bytes(usize::from(is_swa)),
            &u32_bytes(pos0),
            &u32_bytes(slots),
            &f32_bytes(qk_clip),
            &f32_bytes(rms_eps),
        ],
        Grid::Threadgroups {
            groups: (n_head_kv, 2, rows),
            threadgroup: (128, 1, 1),
        },
    )
}

/// Split-K + GQA-shared relative-position bias attention with implicit zero sink
/// (`head_dim = 128`). `partial` must hold at least `rows * n_head * ATTN_SPLIT_MAX_CHUNKS * 130` f32s.
#[allow(clippy::too_many_arguments)]
pub fn attention_split_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    rel_table: &Tensor,
    sinks: &Tensor,
    partial: &Tensor,
    out: &Tensor,
    gate: Option<&Tensor>,
    q_clip: Option<&Tensor>,
    rows: usize,
    n_head: usize,
    n_head_kv: usize,
    pos0: usize,
    slots: usize,
    is_swa: bool,
    window: usize,
    extent: usize,
    qk_clip: f32,
    log_floor: usize,
    log_alpha: f32,
) -> Result<()> {
    let g_step = choose_g_step(n_head, n_head_kv);
    let max_keys = if is_swa { (pos0 + rows).min(window) } else { pos0 + rows }.max(1);
    let min_chunk = if max_keys <= 128 {
        16
    } else if max_keys <= 512 {
        32
    } else if max_keys <= 2048 {
        64
    } else {
        128
    };
    let chunk_keys =
        max_keys.div_ceil(ATTN_SPLIT_MAX_CHUNKS).max(min_chunk).next_multiple_of(16);
    let chunks = max_keys.div_ceil(chunk_keys).clamp(1, ATTN_SPLIT_MAX_CHUNKS);

    let p_split = ctx.pipeline("inkling2_attn_split", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p_split,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            rel_table.binding(),
            sinks.binding(),
            partial.binding(),
            q_clip.unwrap_or(q).binding(),
        ],
        &[
            &u32_bytes(n_head),
            &u32_bytes(n_head_kv),
            &u32_bytes(g_step),
            &u32_bytes(pos0),
            &u32_bytes(slots),
            &u32_bytes(usize::from(is_swa)),
            &u32_bytes(window),
            &u32_bytes(extent),
            &u32_bytes(usize::from(q_clip.is_some())),
            &f32_bytes(qk_clip),
            &u32_bytes(log_floor),
            &f32_bytes(log_alpha),
            &u32_bytes(chunks),
            &u32_bytes(chunk_keys),
        ],
        Grid::Threadgroups {
            groups: (chunks, n_head / g_step, rows),
            threadgroup: (128, 1, 1),
        },
    )?;
    pass.level_barrier(&[partial])?;

    let p_comb = ctx.pipeline("inkling2_attn_combine", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p_comb,
        &[partial.binding(), out.binding(), gate.unwrap_or(out).binding()],
        &[
            &u32_bytes(n_head),
            &u32_bytes(chunks),
            &u32_bytes(usize::from(gate.is_some())),
        ],
        Grid::Threadgroups { groups: (n_head, rows, 1), threadgroup: (128, 1, 1) },
    )
}

/// Split-K + GQA-shared DSpark non-causal GQA over `[ctx_kv(F16) ; block_kv(F32)]` (`hd = 128`).
#[allow(clippy::too_many_arguments)]
pub fn dspark_attn_split(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_ctx: &Tensor,
    v_ctx: &Tensor,
    k_blk: &Tensor,
    v_blk: &Tensor,
    partial: &Tensor,
    out: &Tensor,
    n_head: usize,
    n_head_kv: usize,
    hd: usize,
    ctx_start: usize,
    ctx_pos: usize,
    block: usize,
) -> Result<()> {
    if hd != 128 {
        return dspark_attn(
            ctx, pass, q, k_ctx, v_ctx, k_blk, v_blk, out, n_head, n_head_kv, hd,
            ctx_start, ctx_pos, block,
        );
    }
    let g_step = choose_g_step(n_head, n_head_kv);
    let ctx_len = ctx_pos.saturating_sub(ctx_start);
    let (ctx_chunks, chunk_keys) = if ctx_len == 0 {
        (0, 16)
    } else {
        let min_chunk = if ctx_len <= 256 {
            32
        } else if ctx_len <= 1024 {
            64
        } else {
            128
        };
        let ck = ctx_len
            .div_ceil(DSPARK_SPLIT_MAX_CHUNKS)
            .max(min_chunk)
            .next_multiple_of(16);
        let cc = ctx_len.div_ceil(ck).clamp(1, DSPARK_SPLIT_MAX_CHUNKS);
        (cc, ck)
    };
    let total_chunks = ctx_chunks + 1;

    let p_split = ctx.pipeline("inkling2_dspark_attn_split", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p_split,
        &[
            q.binding(),
            k_ctx.binding(),
            v_ctx.binding(),
            k_blk.binding(),
            v_blk.binding(),
            partial.binding(),
        ],
        &[
            &u32_bytes(n_head),
            &u32_bytes(n_head_kv),
            &u32_bytes(g_step),
            &u32_bytes(ctx_start),
            &u32_bytes(ctx_pos),
            &u32_bytes(block),
            &u32_bytes(ctx_chunks),
            &u32_bytes(chunk_keys),
        ],
        Grid::Threadgroups {
            groups: (total_chunks, n_head / g_step, block),
            threadgroup: (128, 1, 1),
        },
    )?;
    pass.level_barrier(&[partial])?;

    let p_comb = ctx.pipeline("inkling2_dspark_attn_combine", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p_comb,
        &[partial.binding(), out.binding()],
        &[&u32_bytes(n_head), &u32_bytes(total_chunks)],
        Grid::Threadgroups { groups: (n_head, block, 1), threadgroup: (128, 1, 1) },
    )
}

/// Relative-position bias projection: `[n_rows, rel_len] @ [rel_len, ext]^T -> [n_rows, ext]`.
#[allow(clippy::too_many_arguments)]
pub fn rel_bias(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    r_proj: &Tensor,
    bias: &QWeight,
    out: &Tensor,
    n_rows: usize,
) -> Result<()> {
    rel_bias_clipped(ctx, pass, r_proj, bias, out, None, n_rows, 1, 0.0, 0, 0, 0.0)
}

/// [`rel_bias`] with optional fused L2 clip + log-scale tau (valid for any `n_rows = rows * nh` when `D <= 16`).
#[allow(clippy::too_many_arguments)]
pub fn rel_bias_clipped(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    r_proj: &Tensor,
    bias: &QWeight,
    out: &Tensor,
    r_clip: Option<&Tensor>,
    n_rows: usize,
    nh: usize,
    rel_clip: f32,
    pos0: usize,
    log_floor: usize,
    log_alpha: f32,
) -> Result<()> {
    let ty_code: usize = match bias.ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Bf16 => 2,
        other => anyhow::bail!("rel_bias unsupported type {other:?}"),
    };
    let p = ctx.pipeline("inkling2_rel_bias", SOURCE, MslVersion::V3_1)?;
    let (groups_x, groups_y) = if bias.cols <= 16 {
        (bias.rows.div_ceil(32), n_rows.div_ceil(32))
    } else {
        (bias.rows.div_ceil(128), n_rows)
    };
    pass.dispatch_at(
        &p,
        &[
            r_proj.binding(),
            bias.t.binding(),
            out.binding(),
            r_clip.unwrap_or(r_proj).binding(),
        ],
        &[
            &u32_bytes(bias.cols),
            &u32_bytes(bias.rows),
            &u32_bytes(ty_code),
            &u32_bytes(n_rows),
            &u32_bytes(usize::from(r_clip.is_some())),
            &f32_bytes(rel_clip),
            &u32_bytes(pos0),
            &u32_bytes(log_floor),
            &f32_bytes(log_alpha),
            &u32_bytes(nh.max(1)),
        ],
        Grid::Threadgroups {
            groups: (groups_x, groups_y, 1),
            threadgroup: (128, 1, 1),
        },
    )
}

/// Device-side f32 copy (`dst[0..n] = src[0..n]`).
pub fn copy_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    src: &Tensor,
    dst: &Tensor,
    n: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_copy_f32", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[src.binding(), dst.binding()],
        &[&u32_bytes(n)],
        Grid::Threadgroups { groups: (n.div_ceil(256), 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Sigmoid top-k router with optional bias and `route_scale`.
#[allow(clippy::too_many_arguments)]
pub fn router(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    bias: Option<&Tensor>,
    out_ids: &Tensor,
    out_w: &Tensor,
    rows: usize,
    n_expert: usize,
    top_k: usize,
    route_scale: f32,
) -> Result<()> {
    ensure!(n_expert <= 256 && top_k <= 16, "router E={n_expert} K={top_k}");
    let p = ctx.pipeline("inkling2_router", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            logits.binding(),
            bias.unwrap_or(logits).binding(),
            out_ids.binding(),
            out_w.binding(),
        ],
        &[
            &u32_bytes(n_expert),
            &u32_bytes(top_k),
            &u32_bytes(usize::from(bias.is_some())),
            &f32_bytes(route_scale),
        ],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// SITU activation on `[rows * k, 2 * ff]` -> `[rows * k, ff]`, with optional per-expert scale.
#[allow(clippy::too_many_arguments)]
pub fn situ(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate_up: &Tensor,
    exp_ids: &Tensor,
    exp_scale: Option<&Tensor>,
    out: &Tensor,
    rows_k: usize,
    ff: usize,
    beta: f32,
    linear_beta: f32,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_situ", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            gate_up.binding(),
            exp_ids.binding(),
            exp_scale.unwrap_or(gate_up).binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(ff),
            &f32_bytes(beta),
            &f32_bytes(linear_beta),
            &u32_bytes(usize::from(exp_scale.is_some())),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(256), rows_k, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Weighted MoE expert sum `[rows, k, h]` -> `[rows, h]`, with optional per-expert down_scale.
#[allow(clippy::too_many_arguments)]
pub fn moe_sum(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    in_buf: &Tensor,
    exp_ids: &Tensor,
    weights: &Tensor,
    down_scale: Option<&Tensor>,
    out: &Tensor,
    rows: usize,
    k: usize,
    h: usize,
) -> Result<()> {
    let p = ctx.pipeline("inkling2_moe_sum", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            in_buf.binding(),
            exp_ids.binding(),
            weights.binding(),
            down_scale.unwrap_or(in_buf).binding(),
            out.binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(k), &u32_bytes(usize::from(down_scale.is_some()))],
        Grid::Threadgroups {
            groups: (h.div_ceil(256), rows, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

fn embed_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "inkling2_embed_q8_0",
        GgmlType::Q4K => "inkling2_embed_q4_k",
        GgmlType::Q2K => "inkling2_embed_q2_k",
        GgmlType::Iq2Xxs => "inkling2_embed_iq2_xxs",
        GgmlType::Bf16 => "inkling2_embed_bf16",
        GgmlType::F32 => "inkling2_embed_f32",
        GgmlType::F16 => "inkling2_embed_f16",
        GgmlType::Q4_0 => "inkling2_embed_q4_0",
        GgmlType::Q5K => "inkling2_embed_q5_k",
        GgmlType::Q6K => "inkling2_embed_q6_k",
        GgmlType::Iq4Nl => "inkling2_embed_iq4_nl",
        GgmlType::Iq2S => "inkling2_embed_iq2_s",
        GgmlType::Mxfp4 => "inkling2_embed_mxfp4",
        GgmlType::Nvfp4 => "inkling2_embed_nvfp4",
        GgmlType::Q2_0 => "inkling2_embed_q2_0",
        GgmlType::Iq2Xs => "inkling2_embed_iq2_xs",
        GgmlType::Iq3Xxs => "inkling2_embed_iq3_xxs",
        GgmlType::Iq1S => "inkling2_embed_iq1_s",
        GgmlType::Iq4Xs => "inkling2_embed_iq4_xs",
        GgmlType::I32 => anyhow::bail!("unsupported GGML type I32"),
    })
}

fn moe_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "inkling2_moe_q8_0",
        GgmlType::Q4K => "inkling2_moe_q4_k",
        GgmlType::Q2K => "inkling2_moe_q2_k",
        GgmlType::Iq2Xxs => "inkling2_moe_iq2_xxs",
        GgmlType::Bf16 => "inkling2_moe_bf16",
        GgmlType::F32 => "inkling2_moe_f32",
        GgmlType::F16 => "inkling2_moe_f16",
        GgmlType::Q4_0 => "inkling2_moe_q4_0",
        GgmlType::Q5K => "inkling2_moe_q5_k",
        GgmlType::Q6K => "inkling2_moe_q6_k",
        GgmlType::Iq4Nl => "inkling2_moe_iq4_nl",
        GgmlType::Iq2S => "inkling2_moe_iq2_s",
        GgmlType::Mxfp4 => "inkling2_moe_mxfp4",
        GgmlType::Nvfp4 => "inkling2_moe_nvfp4",
        GgmlType::Q2_0 => "inkling2_moe_q2_0",
        GgmlType::Iq2Xs => "inkling2_moe_iq2_xs",
        GgmlType::Iq3Xxs => "inkling2_moe_iq3_xxs",
        GgmlType::Iq1S => "inkling2_moe_iq1_s",
        GgmlType::Iq4Xs => "inkling2_moe_iq4_xs",
        GgmlType::I32 => anyhow::bail!("unsupported GGML type I32"),
    })
}

/// Embedding gather from any GGML weight type (`ids: [rows] u32` -> `out: [rows, h] f32`).
pub fn embed(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    ids: &Tensor,
    w: &QWeight,
    out: &Tensor,
    rows: usize,
) -> Result<()> {
    ensure!(w.cols.is_multiple_of(32), "embed cols {} not a multiple of 32", w.cols);
    let p = ctx.pipeline(embed_name(w.ty)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[ids.binding(), w.t.binding(), out.binding()],
        &[&u32_bytes(w.cols), &u32_bytes(w.row_bytes), &u32_bytes(w.rows)],
        Grid::Threadgroups { groups: (w.cols / 32, rows, 1), threadgroup: (32, 1, 1) },
    )
}

fn moe_gu_situ_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "inkling2_moe_gu_situ_q8_0",
        GgmlType::Q4K => "inkling2_moe_gu_situ_q4_k",
        GgmlType::Q2K => "inkling2_moe_gu_situ_q2_k",
        GgmlType::Iq2Xxs => "inkling2_moe_gu_situ_iq2_xxs",
        GgmlType::Bf16 => "inkling2_moe_gu_situ_bf16",
        GgmlType::F32 => "inkling2_moe_gu_situ_f32",
        GgmlType::F16 => "inkling2_moe_gu_situ_f16",
        GgmlType::Q4_0 => "inkling2_moe_gu_situ_q4_0",
        GgmlType::Q5K => "inkling2_moe_gu_situ_q5_k",
        GgmlType::Q6K => "inkling2_moe_gu_situ_q6_k",
        GgmlType::Iq4Nl => "inkling2_moe_gu_situ_iq4_nl",
        GgmlType::Iq2S => "inkling2_moe_gu_situ_iq2_s",
        GgmlType::Mxfp4 => "inkling2_moe_gu_situ_mxfp4",
        GgmlType::Nvfp4 => "inkling2_moe_gu_situ_nvfp4",
        GgmlType::Q2_0 => "inkling2_moe_gu_situ_q2_0",
        GgmlType::Iq2Xs => "inkling2_moe_gu_situ_iq2_xs",
        GgmlType::Iq3Xxs => "inkling2_moe_gu_situ_iq3_xxs",
        GgmlType::Iq1S => "inkling2_moe_gu_situ_iq1_s",
        GgmlType::Iq4Xs => "inkling2_moe_gu_situ_iq4_xs",
        GgmlType::I32 => anyhow::bail!("unsupported GGML type I32"),
    })
}

fn moe_down_sum_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "inkling2_moe_down_sum_q8_0",
        GgmlType::Q4K => "inkling2_moe_down_sum_q4_k",
        GgmlType::Q2K => "inkling2_moe_down_sum_q2_k",
        GgmlType::Iq2Xxs => "inkling2_moe_down_sum_iq2_xxs",
        GgmlType::Bf16 => "inkling2_moe_down_sum_bf16",
        GgmlType::F32 => "inkling2_moe_down_sum_f32",
        GgmlType::F16 => "inkling2_moe_down_sum_f16",
        GgmlType::Q4_0 => "inkling2_moe_down_sum_q4_0",
        GgmlType::Q5K => "inkling2_moe_down_sum_q5_k",
        GgmlType::Q6K => "inkling2_moe_down_sum_q6_k",
        GgmlType::Iq4Nl => "inkling2_moe_down_sum_iq4_nl",
        GgmlType::Iq2S => "inkling2_moe_down_sum_iq2_s",
        GgmlType::Mxfp4 => "inkling2_moe_down_sum_mxfp4",
        GgmlType::Nvfp4 => "inkling2_moe_down_sum_nvfp4",
        GgmlType::Q2_0 => "inkling2_moe_down_sum_q2_0",
        GgmlType::Iq2Xs => "inkling2_moe_down_sum_iq2_xs",
        GgmlType::Iq3Xxs => "inkling2_moe_down_sum_iq3_xxs",
        GgmlType::Iq1S => "inkling2_moe_down_sum_iq1_s",
        GgmlType::Iq4Xs => "inkling2_moe_down_sum_iq4_xs",
        GgmlType::I32 => anyhow::bail!("unsupported GGML type I32"),
    })
}

/// Fused MoE gate+up projection + SITU activation + optional per-expert scale.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_situ(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &QWeight,
    exp_ids: &Tensor,
    exp_scale: Option<&Tensor>,
    y: &Tensor,
    rows_k: usize,
    ff: usize,
    k_exp: usize,
    beta: f32,
    linear_beta: f32,
) -> Result<()> {
    let p = ctx.pipeline(moe_gu_situ_name(w.ty)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            x.binding(),
            w.t.binding(),
            exp_ids.binding(),
            exp_scale.unwrap_or(x).binding(),
            y.binding(),
        ],
        &[
            &u32_bytes(ff),
            &u32_bytes(w.cols),
            &u32_bytes(w.row_bytes),
            &u32_bytes(k_exp),
            &u32_bytes(usize::from(exp_scale.is_some())),
            &f32_bytes(beta),
            &f32_bytes(linear_beta),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(64), rows_k, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Fused MoE down projection + weighted expert sum + optional per-expert scale.
#[allow(clippy::too_many_arguments)]
pub fn moe_down_sum(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    situ_out: &Tensor,
    w: &QWeight,
    exp_ids: &Tensor,
    exp_w: &Tensor,
    exp_scale: Option<&Tensor>,
    moe_out: &Tensor,
    rows: usize,
    k_exp: usize,
    h: usize,
) -> Result<()> {
    let p = ctx.pipeline(moe_down_sum_name(w.ty)?, SOURCE, MslVersion::V3_1)?;
    let tg = if matches!(w.ty, GgmlType::Q8_0 | GgmlType::Nvfp4) { 128 } else { 256 };
    pass.dispatch_at(
        &p,
        &[
            situ_out.binding(),
            w.t.binding(),
            exp_ids.binding(),
            exp_w.binding(),
            exp_scale.unwrap_or(situ_out).binding(),
            moe_out.binding(),
        ],
        &[
            &u32_bytes(h),
            &u32_bytes(w.cols),
            &u32_bytes(w.row_bytes),
            &u32_bytes(k_exp),
            &u32_bytes(usize::from(exp_scale.is_some())),
        ],
        Grid::Threadgroups {
            groups: (h.div_ceil(8), rows, 1),
            threadgroup: (tg, 1, 1),
        },
    )
}

fn moe_gu_m4_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "inkling2_moe_gu_m4_q8_0",
        GgmlType::Q4K => "inkling2_moe_gu_m4_q4_k",
        GgmlType::Q2K => "inkling2_moe_gu_m4_q2_k",
        GgmlType::Iq2Xxs => "inkling2_moe_gu_m4_iq2_xxs",
        GgmlType::Bf16 => "inkling2_moe_gu_m4_bf16",
        GgmlType::F32 => "inkling2_moe_gu_m4_f32",
        GgmlType::F16 => "inkling2_moe_gu_m4_f16",
        GgmlType::Q4_0 => "inkling2_moe_gu_m4_q4_0",
        GgmlType::Q5K => "inkling2_moe_gu_m4_q5_k",
        GgmlType::Q6K => "inkling2_moe_gu_m4_q6_k",
        GgmlType::Iq4Nl => "inkling2_moe_gu_m4_iq4_nl",
        GgmlType::Iq2S => "inkling2_moe_gu_m4_iq2_s",
        GgmlType::Mxfp4 => "inkling2_moe_gu_m4_mxfp4",
        GgmlType::Nvfp4 => "inkling2_moe_gu_m4_nvfp4",
        GgmlType::Q2_0 => "inkling2_moe_gu_m4_q2_0",
        GgmlType::Iq2Xs => "inkling2_moe_gu_m4_iq2_xs",
        GgmlType::Iq3Xxs => "inkling2_moe_gu_m4_iq3_xxs",
        GgmlType::Iq1S => "inkling2_moe_gu_m4_iq1_s",
        GgmlType::Iq4Xs => "inkling2_moe_gu_m4_iq4_xs",
        GgmlType::I32 => anyhow::bail!("unsupported GGML type I32"),
    })
}

fn moe_down_m4_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "inkling2_moe_down_m4_q8_0",
        GgmlType::Q4K => "inkling2_moe_down_m4_q4_k",
        GgmlType::Q2K => "inkling2_moe_down_m4_q2_k",
        GgmlType::Iq2Xxs => "inkling2_moe_down_m4_iq2_xxs",
        GgmlType::Bf16 => "inkling2_moe_down_m4_bf16",
        GgmlType::F32 => "inkling2_moe_down_m4_f32",
        GgmlType::F16 => "inkling2_moe_down_m4_f16",
        GgmlType::Q4_0 => "inkling2_moe_down_m4_q4_0",
        GgmlType::Q5K => "inkling2_moe_down_m4_q5_k",
        GgmlType::Q6K => "inkling2_moe_down_m4_q6_k",
        GgmlType::Iq4Nl => "inkling2_moe_down_m4_iq4_nl",
        GgmlType::Iq2S => "inkling2_moe_down_m4_iq2_s",
        GgmlType::Mxfp4 => "inkling2_moe_down_m4_mxfp4",
        GgmlType::Nvfp4 => "inkling2_moe_down_m4_nvfp4",
        GgmlType::Q2_0 => "inkling2_moe_down_m4_q2_0",
        GgmlType::Iq2Xs => "inkling2_moe_down_m4_iq2_xs",
        GgmlType::Iq3Xxs => "inkling2_moe_down_m4_iq3_xxs",
        GgmlType::Iq1S => "inkling2_moe_down_m4_iq1_s",
        GgmlType::Iq4Xs => "inkling2_moe_down_m4_iq4_xs",
        GgmlType::I32 => anyhow::bail!("unsupported GGML type I32"),
    })
}

/// Groups `rows_k` expert assignments by expert into up-to-4-token tiles on GPU.
#[allow(clippy::too_many_arguments)]
pub fn moe_bucket(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    exp_ids: &Tensor,
    sorted_rk: &Tensor,
    tiles: &Tensor,
    n_tiles: &Tensor,
    rows_k: usize,
    n_expert: usize,
) -> Result<()> {
    ensure!(n_expert <= 256, "moe_bucket E={n_expert}");
    let p = ctx.pipeline("inkling2_moe_bucket", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[exp_ids.binding(), sorted_rk.binding(), tiles.binding(), n_tiles.binding()],
        &[&u32_bytes(rows_k), &u32_bytes(n_expert)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Grouped multi-row (up to 4 tokens/tile) MoE gate+up + SITU + optional scale.
#[allow(clippy::too_many_arguments)]
pub fn moe_gu_m4(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &QWeight,
    sorted_rk: &Tensor,
    tiles: &Tensor,
    n_tiles: &Tensor,
    exp_scale: Option<&Tensor>,
    situ_out: &Tensor,
    max_tiles: usize,
    ff: usize,
    k_exp: usize,
    beta: f32,
    linear_beta: f32,
) -> Result<()> {
    let p = ctx.pipeline(moe_gu_m4_name(w.ty)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            x.binding(),
            w.t.binding(),
            sorted_rk.binding(),
            tiles.binding(),
            n_tiles.binding(),
            exp_scale.unwrap_or(x).binding(),
            situ_out.binding(),
        ],
        &[
            &u32_bytes(ff),
            &u32_bytes(w.cols),
            &u32_bytes(w.row_bytes),
            &u32_bytes(k_exp),
            &u32_bytes(usize::from(exp_scale.is_some())),
            &f32_bytes(beta),
            &f32_bytes(linear_beta),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(32), max_tiles, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Grouped multi-row (up to 4 tokens/tile) MoE down projection.
#[allow(clippy::too_many_arguments)]
pub fn moe_down_m4(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    situ_out: &Tensor,
    w: &QWeight,
    sorted_rk: &Tensor,
    tiles: &Tensor,
    n_tiles: &Tensor,
    down_out: &Tensor,
    max_tiles: usize,
    h: usize,
) -> Result<()> {
    let p = ctx.pipeline(moe_down_m4_name(w.ty)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[
            situ_out.binding(),
            w.t.binding(),
            sorted_rk.binding(),
            tiles.binding(),
            n_tiles.binding(),
            down_out.binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(w.cols), &u32_bytes(w.row_bytes)],
        Grid::Threadgroups {
            groups: (h.div_ceil(64), max_tiles, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Routed MoE expert projection for any GGML type:
/// `w` is `[n_expert * n_out, k_in]`, `exp_ids` is `[rows * k_exp] u32`,
/// `k_exp_for_x = k_exp` when `x` is `[rows, k_in]` (gate_up), or `0` when `x` is
/// `[rows * k_exp, k_in]` (down).
#[allow(clippy::too_many_arguments)]
pub fn moe_proj(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &QWeight,
    exp_ids: &Tensor,
    y: &Tensor,
    rows_k: usize,
    n_out: usize,
    k_exp_for_x: usize,
) -> Result<()> {
    ensure!(w.cols.is_multiple_of(32), "moe_proj cols {} not a multiple of 32", w.cols);
    let p = ctx.pipeline(moe_name(w.ty)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &p,
        &[x.binding(), w.t.binding(), exp_ids.binding(), y.binding()],
        &[
            &u32_bytes(n_out),
            &u32_bytes(w.cols),
            &u32_bytes(w.row_bytes),
            &u32_bytes(k_exp_for_x),
        ],
        Grid::Threadgroups {
            groups: (n_out.div_ceil(8), rows_k, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/inkling2.rs"]
mod tests;
