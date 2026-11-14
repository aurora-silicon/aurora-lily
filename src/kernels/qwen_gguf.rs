//! Qwen3.8-Flash-Next over GGUF weights (`vulkan/qwen_gguf.comp`): the
//! projections, routed-expert gathers and embedding the Qwen graph
//! (`crate::qwen4exp`) runs when its weights are GGUF blocks read in place
//! (ds4's Q2 file), with lily's bfloat16 activations. The block decoders are
//! GLM's; the CPU reference is `crate::gguf::dequantize`.

use anyhow::{Result, ensure};

use crate::gguf::GgmlType;
use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

pub use crate::kernels::glm53::QWeight;

/// GLSL only (Linux): the Metal versions do not exist.
const SOURCE: &str = "";

/// Activation rows a GEMV takes at once.
pub const GEMV_MAX_ROWS: usize = 4;

fn gemv_name(ty: GgmlType, m: usize) -> Result<&'static str> {
    macro_rules! rows {
        ($t:literal) => {
            [
                concat!("qg_gemv_", $t, "_m1"),
                concat!("qg_gemv_", $t, "_m2"),
                concat!("qg_gemv_", $t, "_m3"),
                concat!("qg_gemv_", $t, "_m4"),
            ]
        };
    }
    const NAMES: [[&str; GEMV_MAX_ROWS]; 5] =
        [rows!("q8_0"), rows!("f16"), rows!("f32"), rows!("bf16"), rows!("q4_k")];
    ensure!((1..=GEMV_MAX_ROWS).contains(&m), "GEMV of {m} rows");
    let t = match ty {
        GgmlType::Q8_0 => 0,
        GgmlType::F16 => 1,
        GgmlType::F32 => 2,
        GgmlType::Bf16 => 3,
        GgmlType::Q4K => 4,
        other => anyhow::bail!("no Qwen GGUF GEMV over {other:?}"),
    };
    Ok(NAMES[t][m - 1])
}

/// One dispatch of up to four GEMVs of one weight type and width over the
/// same bfloat16 input: `y_s[i] = w_s x[i]` for the `m` (<= 4) rows of `x`,
/// `x_row_stride` elements apart. Each output is bfloat16 or f32 (its
/// tensor's dtype), its rows `y_row_stride` apart (`None`: the set's rows).
/// Dispatches of one dependency level run one after another on this
/// driver, so a level's projections of one type are best one dispatch.
pub fn gemv_sets(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    sets: &[(&QWeight, &Tensor)],
    m: usize,
    x_row_stride: usize,
    y_row_stride: Option<usize>,
) -> Result<()> {
    ensure!((1..=4).contains(&sets.len()), "GEMV of {} sets", sets.len());
    ensure!((1..=GEMV_MAX_ROWS).contains(&m), "GEMV of {m} rows");
    ensure!(x.dtype() == DType::BF16, "GGUF GEMV input is {:?}", x.dtype());
    let w0 = sets[0].0;
    let k = w0.cols;
    ensure!(k.is_multiple_of(32) && k > 0, "GEMV row of {k} values");
    ensure!(
        x.numel() >= (m - 1) * x_row_stride + k && x_row_stride >= k,
        "GEMV input of {} values for {m} rows of {k} ({x_row_stride} apart)",
        x.numel()
    );
    let tasks = k / 32;
    let mut width = 32;
    while width > tasks {
        width /= 2;
    }
    // Few long rows (fewer than 128 threadgroups' worth): the threadgroup's
    // subgroups split their tasks (see the kernel).
    let total_rows: usize = sets.iter().map(|(w, _)| w.rows).sum();
    let split = if total_rows < 1024 * width / 32 && tasks >= 128 { 8 } else { 1 };
    let rows_per_tg = if split > 1 { 32 / width } else { 8 * (32 / width) };
    let mut groups = 0;
    let mut bf_mask = 0usize;
    for (s, &(w, y)) in sets.iter().enumerate() {
        ensure!(w.ty == w0.ty && w.cols == k, "GEMV sets of different types or widths");
        let stride = y_row_stride.unwrap_or(w.rows);
        ensure!(y.numel() >= (m - 1) * stride + w.rows, "GEMV output of {}", y.numel());
        match y.dtype() {
            DType::BF16 => bf_mask |= 1 << s,
            DType::F32 => {}
            other => anyhow::bail!("GEMV output of {other:?}"),
        }
        groups += w.rows.div_ceil(rows_per_tg);
    }
    let mut bindings = vec![x.binding()];
    for s in 0..4 {
        bindings.push(sets.get(s).map_or(x.binding(), |(w, _)| w.t.binding()));
    }
    for s in 0..4 {
        bindings.push(sets.get(s).map_or(x.binding(), |(_, y)| y.binding()));
    }
    let row_bytes = |s: usize| u32_bytes(sets.get(s).map_or(0, |(w, _)| w.row_bytes));
    let rows = |s: usize| u32_bytes(sets.get(s).map_or(0, |(w, _)| w.rows));
    let pipeline = ctx.pipeline(gemv_name(w0.ty, m)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &bindings,
        &[
            &u32_bytes(k),
            &row_bytes(0),
            &row_bytes(1),
            &row_bytes(2),
            &row_bytes(3),
            &rows(0),
            &rows(1),
            &rows(2),
            &rows(3),
            &u32_bytes(x_row_stride),
            &u32_bytes(y_row_stride.unwrap_or(0)),
            &u32_bytes(bf_mask),
            &u32_bytes(split),
        ],
        Grid::Threadgroups { groups: (groups, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// `y[i] = w x[i]` for the `m` (<= 4) rows of `x` (rows of `w.cols`), `y`
/// rows of `w.rows`, bfloat16 or f32.
pub fn gemv(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
) -> Result<()> {
    gemv_sets(ctx, pass, x, &[(w, y)], m, w.cols, None)
}

/// `out` = the weight as bfloat16 `[rows, cols]` (each value rounded to
/// nearest even): the staged operand of a prefill GEMM.
pub fn dequant_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    out: &Tensor,
) -> Result<()> {
    ensure!(out.dtype() == DType::BF16, "dequant output is {:?}", out.dtype());
    ensure!(out.numel() >= w.rows * w.cols, "dequant output of {}", out.numel());
    ensure!(w.cols.is_multiple_of(32), "dequant of rows of {}", w.cols);
    let name = match w.ty {
        GgmlType::Q8_0 => "qg_dequant_q8_0",
        GgmlType::F16 => "qg_dequant_f16",
        GgmlType::F32 => "qg_dequant_f32",
        GgmlType::Bf16 => "qg_dequant_bf16",
        other => anyhow::bail!("no GGUF dequant of {other:?}"),
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[w.t.binding(), out.binding()],
        &[&u32_bytes(w.cols), &u32_bytes(w.row_bytes), &u32_bytes(w.rows)],
        Grid::Threads {
            grid: (w.cols / 32, w.rows, 1),
            threadgroup: (32.min(w.cols / 32), 256 / 32.min(w.cols / 32), 1),
        },
    )
}

/// `out[r] = table[ids[r]]` for `m` ids: rows of a BF16 embedding.
pub fn embed_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &QWeight,
    ids: &Tensor,
    out: &Tensor,
    m: usize,
) -> Result<()> {
    ensure!(table.ty == GgmlType::Bf16, "embedding of {:?}", table.ty);
    ensure!(ids.dtype() == DType::U32 && ids.numel() >= m, "embedding ids");
    ensure!(
        out.dtype() == DType::BF16 && out.numel() >= m * table.cols,
        "embedding output"
    );
    let pipeline = ctx.pipeline("qg_embed_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.t.binding(), ids.binding(), out.binding()],
        &[&u32_bytes(table.cols)],
        Grid::Threads { grid: (table.cols, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Routed experts' gate and up with the SwiGLU: `act[i * used + s] =
/// silu(gate_e x[i]) * up_e x[i]` (both rounded to bfloat16 first, the
/// product too) for the `used` experts `sel[i]` of each of the `m` rows;
/// the stacks hold `ff` rows per expert.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    ff: usize,
    x: &Tensor,
    sel: &Tensor,
    act: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    ensure!(
        gate.ty == up.ty && gate.cols == up.cols && gate.rows == up.rows,
        "gate and up experts differ"
    );
    ensure!(gate.rows.is_multiple_of(ff), "expert rows {} of {ff}", gate.rows);
    ensure!(gate.cols.is_multiple_of(32), "expert rows of {}", gate.cols);
    ensure!(x.dtype() == DType::BF16 && x.numel() >= m * gate.cols, "expert input");
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * used, "selection");
    ensure!(
        act.dtype() == DType::BF16 && act.numel() >= m * used * ff,
        "expert activations"
    );
    let name = match gate.ty {
        GgmlType::Iq2Xxs => "qg_moe_gate_up_iq2_xxs",
        GgmlType::Q4K => "qg_moe_gate_up_q4_k",
        other => anyhow::bail!("no Qwen expert gate/up over {other:?}"),
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[gate.t.binding(), up.t.binding(), x.binding(), sel.binding(), act.binding()],
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

/// The shared expert a routed combine adds: its output `[m, hidden]`
/// (bfloat16) and its gate logit per row (f32 or bfloat16).
pub struct SharedAdd<'a> {
    pub out: &'a Tensor,
    pub gate: &'a Tensor,
}

/// Routed experts' down projections combined: `out[i] = sum_s scores[i *
/// used + s] * bf16(down_{sel[i][s]} act[i * used + s]) (+ sigmoid(gate[i])
/// * shared[i])`, rounded to bfloat16. Each slot has `ff` activations; the
/// down rows may be longer (ds4 pads its Q2_K rows from 640 to 768 values
/// with zeros): only their first `ff` values are read.
#[allow(clippy::too_many_arguments)]
pub fn moe_down(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    down: &QWeight,
    ff: usize,
    hidden: usize,
    act: &Tensor,
    sel: &Tensor,
    scores: &Tensor,
    shared: Option<SharedAdd<'_>>,
    out: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    ensure!(down.rows.is_multiple_of(hidden), "down rows {} of {hidden}", down.rows);
    ensure!(
        ff.is_multiple_of(32) && ff <= down.cols,
        "down rows of {} for {ff} activations",
        down.cols
    );
    ensure!(act.dtype() == DType::BF16 && act.numel() >= m * used * ff, "down input");
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * used, "selection");
    ensure!(scores.dtype() == DType::F32 && scores.numel() >= m * used, "scores");
    ensure!(out.dtype() == DType::BF16 && out.numel() >= m * hidden, "down output");
    let name = match down.ty {
        GgmlType::Q2K => "qg_moe_down_q2_k",
        GgmlType::Mxfp4 => "qg_moe_down_mxfp4",
        GgmlType::Q8_0 => "qg_moe_down_q8_0",
        other => anyhow::bail!("no Qwen expert down over {other:?}"),
    };
    let mut flags = 0usize;
    let (sh_out, sh_gate) = match &shared {
        Some(s) => {
            ensure!(
                s.out.dtype() == DType::BF16 && s.out.numel() >= m * hidden,
                "shared expert output"
            );
            ensure!(s.gate.numel() >= m, "shared expert gate");
            flags |= 1;
            match s.gate.dtype() {
                DType::BF16 => flags |= 2,
                DType::F32 => {}
                other => anyhow::bail!("shared expert gate of {other:?}"),
            }
            (s.out, s.gate)
        }
        None => (out, scores),
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            down.t.binding(),
            act.binding(),
            sel.binding(),
            scores.binding(),
            sh_out.binding(),
            sh_gate.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(ff),
            &u32_bytes(hidden),
            &u32_bytes(down.row_bytes),
            &u32_bytes(used),
            &u32_bytes(flags),
        ],
        Grid::Threadgroups {
            groups: (hidden.div_ceil(8), m, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Scratch of [`moe_grouped`] for runs of up to `pairs` (row, slot) pairs.
#[derive(Clone)]
pub struct MoeGroupScratch {
    pub pairs: Tensor,
    pub tiles: Tensor,
    pub tile_count: Tensor,
    /// `[pairs][down_cols]` bfloat activations; the columns past the
    /// experts' `ff` stay zero (the padding the down rows' tail multiplies).
    pub act: Tensor,
    /// `[pairs][hidden]` bfloat down outputs.
    pub down: Tensor,
    capacity: usize,
}

impl MoeGroupScratch {
    pub fn new(
        ctx: &MetalContext,
        pairs: usize,
        experts: usize,
        down_cols: usize,
        hidden: usize,
    ) -> Result<Self> {
        let max_tiles = pairs.div_ceil(64) + experts;
        Ok(Self {
            pairs: Tensor::zeros(ctx, &[pairs], DType::U32)?,
            tiles: Tensor::zeros(ctx, &[3 * max_tiles], DType::U32)?,
            tile_count: Tensor::zeros(ctx, &[1], DType::U32)?,
            act: Tensor::zeros(ctx, &[pairs * down_cols], DType::BF16)?,
            down: Tensor::zeros(ctx, &[pairs * hidden], DType::BF16)?,
            capacity: pairs,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Routed experts over a run of `m` rows grouped by expert, on the matrix
/// units (`vulkan/qwen_gguf_gemm.comp`, GLM's grouped GEMMs): the same
/// output as [`moe_gate_up`] then [`moe_down`] with the shared expert,
/// up to the operands' rounding to bfloat. IQ2_XXS gate/up and Q2_K down
/// (the draft head's types take the gathers).
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    down: &QWeight,
    ff: usize,
    hidden: usize,
    x: &Tensor,
    sel: &Tensor,
    scores: &Tensor,
    shared: SharedAdd<'_>,
    out: &Tensor,
    sc: &MoeGroupScratch,
    m: usize,
    used: usize,
    experts: usize,
) -> Result<()> {
    ensure!(
        gate.ty == GgmlType::Iq2Xxs
            && up.ty == GgmlType::Iq2Xxs
            && down.ty == GgmlType::Q2K,
        "grouped Qwen experts are IQ2_XXS gate/up and Q2_K down"
    );
    let pairs = m * used;
    ensure!(pairs <= sc.capacity, "grouped MoE scratch for {} pairs", sc.capacity);
    ensure!(experts <= 512, "grouping of {experts} experts");
    ensure!(
        gate.cols.is_multiple_of(256)
            && down.cols.is_multiple_of(256)
            && ff.is_multiple_of(32)
            && hidden.is_multiple_of(64)
            && ff <= down.cols,
        "grouped expert GEMMs of {ff} x {} and {hidden} x {}",
        gate.cols,
        down.cols
    );
    ensure!(
        x.dtype() == DType::BF16 && x.numel() >= m * gate.cols,
        "grouped MoE input"
    );
    ensure!(sc.act.numel() >= pairs * down.cols, "grouped MoE activations");
    ensure!(
        out.dtype() == DType::BF16 && out.numel() >= m * hidden,
        "grouped MoE output"
    );
    // Tiles of up to 128 rows once an expert averages 48 pairs or more
    // (GLM's rule; the sums are the same either way).
    let r128 = pairs >= 48 * experts;
    let tile_rows = if r128 { 128 } else { 64 };
    let group = ctx.pipeline("glm53_moe_group", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &group,
        &[
            sel.binding(),
            sc.pairs.binding(),
            sc.tiles.binding(),
            sc.tile_count.binding(),
        ],
        &[&u32_bytes(pairs), &u32_bytes(experts), &u32_bytes(tile_rows)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (1024, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let max_tiles = pairs.div_ceil(tile_rows) + experts;
    let gate_up = ctx.pipeline(
        if r128 {
            "qg_gemm_moe_gate_up_iq2_xxs_r128"
        } else {
            "qg_gemm_moe_gate_up_iq2_xxs"
        },
        SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &gate_up,
        &[
            x.binding(),
            gate.t.binding(),
            sc.act.binding(),
            sc.pairs.binding(),
            sc.tiles.binding(),
            sc.tile_count.binding(),
            up.t.binding(),
        ],
        &[
            &u32_bytes(ff),
            &u32_bytes(gate.cols),
            &u32_bytes(gate.row_bytes),
            &u32_bytes(used),
            &u32_bytes(gate.cols),
            &u32_bytes(down.cols),
        ],
        Grid::Threadgroups {
            groups: (ff / 32, max_tiles, 1),
            threadgroup: (128, 1, 1),
        },
    )?;
    pass.memory_barrier()?;
    let down_k = ctx.pipeline(
        if r128 { "qg_gemm_moe_down_q2_k_r128" } else { "qg_gemm_moe_down_q2_k" },
        SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &down_k,
        &[
            sc.act.binding(),
            down.t.binding(),
            sc.down.binding(),
            sc.pairs.binding(),
            sc.tiles.binding(),
            sc.tile_count.binding(),
        ],
        &[
            &u32_bytes(hidden),
            &u32_bytes(down.cols),
            &u32_bytes(down.row_bytes),
            &u32_bytes(1),
            &u32_bytes(down.cols),
            &u32_bytes(hidden),
        ],
        Grid::Threadgroups {
            groups: (hidden / 64, max_tiles, 1),
            threadgroup: (128, 1, 1),
        },
    )?;
    pass.memory_barrier()?;
    let gate_bf16 = match shared.gate.dtype() {
        DType::BF16 => 1,
        DType::F32 => 0,
        other => anyhow::bail!("shared expert gate of {other:?}"),
    };
    let combine = ctx.pipeline("qg_moe_combine", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &combine,
        &[
            sc.down.binding(),
            scores.binding(),
            shared.out.binding(),
            shared.gate.binding(),
            out.binding(),
        ],
        &[&u32_bytes(hidden), &u32_bytes(used), &u32_bytes(gate_bf16)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// The hyper-connection read of one row over F16 mixers, fused in two
/// kernels (the decode step's): `down` (`[R]`), `inj` (`[NI]`) and
/// `inv_rms` (`[G]`) from the normalized stream, then `mixed` (`[H]`) from
/// the up logits. The weights: down `[R][G*H]`, inject `[NI][G*H]`, up
/// `[G*H][R]`, all F16.
#[allow(clippy::too_many_arguments)]
pub fn hc_read_f16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    hyper: &Tensor,
    norm: &Tensor,
    down: &QWeight,
    inject: Option<&QWeight>,
    up: &QWeight,
    out_down: &Tensor,
    out_inj: &Tensor,
    inv_rms: &Tensor,
    mixed: &Tensor,
    h: usize,
    g: usize,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    let wide = g * h;
    let (r, ni) = (down.rows, inject.map_or(0, |w| w.rows));
    ensure!(
        g <= 4 && h.is_multiple_of(8) && r.is_multiple_of(8),
        "hc read of {g} x {h}, rank {r}"
    );
    for w in [Some(down), inject, Some(up)].into_iter().flatten() {
        ensure!(w.ty == GgmlType::F16, "fused hc read over {:?}", w.ty);
    }
    ensure!(down.cols == wide && up.rows == wide && up.cols == r, "hc read shapes");
    ensure!(inject.is_none_or(|w| w.cols == wide), "hc inject shape");
    for t in [hyper, norm, out_down, mixed] {
        ensure!(t.dtype() == DType::BF16, "hc read tensors are bfloat");
    }
    let inj = inject.unwrap_or(down);
    let pipeline = ctx.pipeline("qg_hc_read_down_f16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            hyper.binding(),
            norm.binding(),
            down.t.binding(),
            inj.t.binding(),
            out_down.binding(),
            out_inj.binding(),
            inv_rms.binding(),
        ],
        &[
            &u32_bytes(h),
            &u32_bytes(g),
            &u32_bytes(r),
            &u32_bytes(ni),
            &eps.to_ne_bytes(),
            &w_bias.to_ne_bytes(),
            &u32_bytes(down.row_bytes),
            &u32_bytes(inj.row_bytes),
        ],
        Grid::Threadgroups { groups: (r + ni, 1, 1), threadgroup: (32 * g, 1, 1) },
    )?;
    pass.level_barrier(&[out_down, out_inj, inv_rms])?;
    let pipeline = ctx.pipeline("qg_hc_read_up_mix_f16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            up.t.binding(),
            out_down.binding(),
            hyper.binding(),
            norm.binding(),
            inv_rms.binding(),
            mixed.binding(),
        ],
        &[
            &u32_bytes(h),
            &u32_bytes(g),
            &u32_bytes(r),
            &u32_bytes(up.row_bytes),
            &w_bias.to_ne_bytes(),
        ],
        Grid::Threadgroups { groups: (h.div_ceil(8), 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/qwen_gguf.rs"]
mod tests;
