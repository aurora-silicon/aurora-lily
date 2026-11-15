//! GLM 5.3 Flash kernels (`vulkan/glm53.comp`): GEMV over GGUF blocks with
//! f32 activations, routed experts, the router, hyper-connections, Kimi delta
//! attention, the DSA latent cache and attention, the pooled indexer.
//! The CPU reference is `crate::glm53::reference`.

use anyhow::{Result, ensure};

use crate::gguf::GgmlType;
use crate::kernels::{f32_bytes, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// The kernels are GLSL only so far: on macOS the Metal versions remain to
/// be written, and `ctx.pipeline` with this empty source fails there.
const SOURCE: &str = "";

/// Rows of activations a GEMV takes at once.
pub const GEMV_MAX_ROWS: usize = 4;

/// Rows a Q8_0 or F16 GEMV takes at once (DeepSeek V4's verify passes and
/// draft blocks); other types take [`GEMV_MAX_ROWS`].
pub const GEMV_MAX_ROWS_WIDE: usize = 8;

/// Rows a GEMV over weights of `ty` takes at once.
pub fn gemv_max_rows(ty: GgmlType) -> usize {
    if matches!(ty, GgmlType::Q8_0 | GgmlType::F16) {
        GEMV_MAX_ROWS_WIDE
    } else {
        GEMV_MAX_ROWS
    }
}

/// A GGUF weight on the GPU: `rows` rows of `cols` values of `ty`, read in
/// place (`t` is a byte view, I8, over the mapped file or a test upload).
#[derive(Clone)]
pub struct QWeight {
    pub t: Tensor,
    pub ty: GgmlType,
    pub cols: usize,
    pub rows: usize,
    pub row_bytes: usize,
}

impl QWeight {
    pub fn new(t: Tensor, ty: GgmlType, cols: usize, rows: usize) -> Result<Self> {
        let row_bytes = ty.row_bytes(cols)?;
        ensure!(
            t.byte_len() >= rows * row_bytes,
            "{} bytes hold no {rows} rows of {cols} {ty:?}",
            t.byte_len()
        );
        Ok(Self { t, ty, cols, rows, row_bytes })
    }

    /// Uploads raw blocks (tests).
    pub fn from_bytes(
        ctx: &MetalContext,
        bytes: &[u8],
        ty: GgmlType,
        cols: usize,
        rows: usize,
    ) -> Result<Self> {
        Self::new(
            Tensor::from_bytes(ctx, bytes, &[bytes.len()], DType::I8)?,
            ty,
            cols,
            rows,
        )
    }
}

/// The two-weight-rows Q8_0 GEMV of `m` (1 to 8) activation rows.
fn gemv_r2_name(m: usize) -> Result<&'static str> {
    const NAMES: [&str; 8] = [
        "glm53_gemv_q8_0_r2_m1",
        "glm53_gemv_q8_0_r2_m2",
        "glm53_gemv_q8_0_r2_m3",
        "glm53_gemv_q8_0_r2_m4",
        "glm53_gemv_q8_0_r2_m5",
        "glm53_gemv_q8_0_r2_m6",
        "glm53_gemv_q8_0_r2_m7",
        "glm53_gemv_q8_0_r2_m8",
    ];
    ensure!((1..=8).contains(&m), "two-row GEMV of {m} rows");
    Ok(NAMES[m - 1])
}

/// The GEMV kernel of a type and row count (`m` in 1..=4: the count is a
/// compile-time constant of each kernel).
fn gemv_name(ty: GgmlType, m: usize) -> Result<&'static str> {
    macro_rules! rows {
        ($t:literal) => {
            [
                concat!("glm53_gemv_", $t, "_m1"),
                concat!("glm53_gemv_", $t, "_m2"),
                concat!("glm53_gemv_", $t, "_m3"),
                concat!("glm53_gemv_", $t, "_m4"),
            ]
        };
    }
    const NAMES: [[&str; GEMV_MAX_ROWS]; 12] = [
        rows!("q8_0"),
        rows!("q4_k"),
        rows!("q2_k"),
        rows!("iq2_xxs"),
        rows!("bf16"),
        rows!("f32"),
        rows!("f16"),
        rows!("q4_0"),
        rows!("q5_k"),
        rows!("q6_k"),
        rows!("iq4_nl"),
        rows!("iq2_s"),
    ];
    if m > GEMV_MAX_ROWS {
        const WIDE: [[&str; 4]; 2] = [
            [
                "glm53_gemv_q8_0_m5",
                "glm53_gemv_q8_0_m6",
                "glm53_gemv_q8_0_m7",
                "glm53_gemv_q8_0_m8",
            ],
            [
                "glm53_gemv_f16_m5",
                "glm53_gemv_f16_m6",
                "glm53_gemv_f16_m7",
                "glm53_gemv_f16_m8",
            ],
        ];
        ensure!(m <= gemv_max_rows(ty), "GEMV of {m} rows over {ty:?}");
        let t = usize::from(ty == GgmlType::F16);
        return Ok(WIDE[t][m - GEMV_MAX_ROWS - 1]);
    }
    ensure!((1..=GEMV_MAX_ROWS).contains(&m), "GEMV of {m} rows");
    let t = match ty {
        GgmlType::Q8_0 => 0,
        GgmlType::Q4K => 1,
        GgmlType::Q2K => 2,
        GgmlType::Iq2Xxs => 3,
        GgmlType::Bf16 => 4,
        GgmlType::F32 => 5,
        GgmlType::F16 => 6,
        GgmlType::I32 | GgmlType::Mxfp4 => anyhow::bail!("no GEMV over {ty:?}"),
        GgmlType::Q4_0 => 7,
        GgmlType::Q5K => 8,
        GgmlType::Q6K => 9,
        GgmlType::Iq4Nl => 10,
        GgmlType::Iq2S => 11,
    };
    Ok(NAMES[t][m - 1])
}

/// On a G16 GPU (an M4), the `gemv_g16.comp` kernel of a type, row count,
/// width and activation layout, and the rows a subgroup of it takes; `None`
/// keeps the glm53_gemv kernel.
fn g16_gemv_name(
    ctx: &MetalContext,
    ty: GgmlType,
    cols: usize,
    m: usize,
    x_group: usize,
    x_row_stride: usize,
) -> Option<(&'static str, usize)> {
    if !ctx.g16() {
        return None;
    }
    let (names, rows) = match ty {
        GgmlType::Q5K if cols.is_multiple_of(256) => (
            [
                "g16_gemv_q5_k_m1",
                "g16_gemv_q5_k_m2",
                "g16_gemv_q5_k_m3",
                "g16_gemv_q5_k_m4",
            ],
            if m == 1 { 4 } else { 2 },
        ),
        _ => return None,
    };
    // Every row count's kernel or none: a verify pass's rows must sum as
    // plain decode's row does. The kernel steps between activation rows in
    // vec4s (x_row_stride / 4): a layout of another stride takes glm53_gemv,
    // at m = 1 too.
    (m <= names.len() && x_group.is_multiple_of(4) && x_row_stride.is_multiple_of(4))
        .then(|| (names[m - 1], rows))
}

fn moe_name(gate_up: bool, ty: GgmlType) -> Result<&'static str> {
    Ok(match (gate_up, ty) {
        (true, GgmlType::Iq2Xxs) => "glm53_moe_gate_up_iq2_xxs",
        (true, GgmlType::Q4K) => "glm53_moe_gate_up_q4_k",
        (true, GgmlType::Q2K) => "glm53_moe_gate_up_q2_k",
        (true, GgmlType::Q8_0) => "glm53_moe_gate_up_q8_0",
        (false, GgmlType::Q8_0) => "glm53_moe_down_q8_0",
        (false, GgmlType::Q2K) => "glm53_moe_down_q2_k",
        (false, GgmlType::Q4K) => "glm53_moe_down_q4_k",
        (false, GgmlType::Iq2Xxs) => "glm53_moe_down_iq2_xxs",
        _ => anyhow::bail!("no GLM expert kernel for {ty:?}"),
    })
}

fn check_f32(ts: &[&Tensor]) -> Result<()> {
    for t in ts {
        ensure!(
            t.dtype() == DType::F32,
            "GLM kernels take f32 tensors, got {:?}",
            t.dtype()
        );
    }
    Ok(())
}

/// The GEMV's layout of its activations and outputs beyond the plain
/// `y[i] = W x[i]`.
#[derive(Clone, Copy)]
pub struct GemvLayout {
    /// Output rows that share one input vector (per-head projections); the
    /// input of row `n` is vector `n / x_group` of the activation row.
    pub x_group: usize,
    /// f32 elements between activation rows.
    pub x_row_stride: usize,
    /// f32 elements between output rows.
    pub y_row_stride: usize,
}

/// `y[i][n] = w[n] . x[i]` for `i < m` (`m <= 4`), `x` rows of `w.cols`.
pub fn gemv(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
) -> Result<()> {
    let layout =
        GemvLayout { x_group: w.rows, x_row_stride: w.cols, y_row_stride: w.rows };
    gemv_with(ctx, pass, w, x, y, m, layout)
}

/// [`gemv`] with an explicit activation/output layout.
pub fn gemv_with(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
    l: GemvLayout,
) -> Result<()> {
    gemv_dispatch(ctx, pass, &[(w, y)], x, m, l, false)
}

/// One dispatch of up to four GEMVs of one weight type and width over the
/// same input `x` (`m` rows): `y_s[i] = w_s x[i]`. On this driver
/// dispatches of one dependency level run one after another, so a level's
/// projections are best one dispatch.
pub fn gemv_sets(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    sets: &[(&QWeight, &Tensor)],
    x: &Tensor,
    m: usize,
) -> Result<()> {
    let rows = sets.iter().map(|(w, _)| w.rows).max().unwrap_or(0);
    let cols = sets.first().map_or(0, |(w, _)| w.cols);
    let l = GemvLayout { x_group: rows.max(1), x_row_stride: cols, y_row_stride: 0 };
    gemv_dispatch(ctx, pass, sets, x, m, l, false)
}

/// [`gemv_sets`], taking for Q8_0 weights of rows of 1 024 values the
/// kernel with two weight rows a subgroup (each
/// activation load serves both; DeepSeek V4's verify and draft rows, and
/// its decode rows so that a row computes the same in any run).
pub fn gemv_sets_r2(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    sets: &[(&QWeight, &Tensor)],
    x: &Tensor,
    m: usize,
) -> Result<()> {
    let rows = sets.iter().map(|(w, _)| w.rows).max().unwrap_or(0);
    let cols = sets.first().map_or(0, |(w, _)| w.cols);
    let l = GemvLayout { x_group: rows.max(1), x_row_stride: cols, y_row_stride: 0 };
    gemv_dispatch(ctx, pass, sets, x, m, l, true)
}

/// [`gemv_with`] with [`gemv_sets_r2`]'s kernel choice.
pub fn gemv_with_r2(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
    l: GemvLayout,
) -> Result<()> {
    gemv_dispatch(ctx, pass, &[(w, y)], x, m, l, true)
}

fn gemv_dispatch(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    sets: &[(&QWeight, &Tensor)],
    x: &Tensor,
    m: usize,
    l: GemvLayout,
    r2: bool,
) -> Result<()> {
    ensure!((1..=4).contains(&sets.len()), "GEMV of {} sets", sets.len());
    let w0 = sets[0].0;
    ensure!((1..=gemv_max_rows(w0.ty)).contains(&m), "GEMV of {m} rows");
    check_f32(&[x])?;
    let tasks = w0.cols / 32;
    ensure!(
        w0.cols.is_multiple_of(32) && (tasks >= 32 || tasks.is_power_of_two()),
        "GEMV row of {} values",
        w0.cols
    );
    let max_rows = sets.iter().map(|(w, _)| w.rows).max().unwrap_or(0);
    let g16 = g16_gemv_name(ctx, w0.ty, w0.cols, m, l.x_group, l.x_row_stride);
    // DeepSeek V4's two-weight-rows Q8_0 kernel (`r2`: gemv_sets_r2,
    // gemv_with_r2), for any activation row count so that a row computes
    // the same in any run: rows of 1 024 values, a lane one task, sixteen
    // rows a threadgroup (gemv_timing m = 6, 2026-11-15: 32768 x 1024
    // 125 -> 110 us). Longer rows lost: the two rows' registers across the
    // loop (4096 x 8192 163 -> 204 us), and with each pair's tasks split
    // over two subgroups (155 us at six rows) plain decoding paid 1.5%.
    // Pairs of weight rows lie inside one input group.
    let r2 = r2
        && g16.is_none()
        && w0.ty == GgmlType::Q8_0
        && tasks == 32
        && l.x_group.is_multiple_of(2);
    // Two rows a subgroup share one input row, so grouped inputs (GLM's
    // per-head attn_k_b) take one; the G16 kernels set their own rows.
    let r = if !r2 && g16.is_none() && l.x_group >= max_rows {
        gemv_rows_per_subgroup(w0.ty, m, max_rows)
    } else {
        1
    };
    let per_sg = match g16 {
        Some((_, rows)) => rows,
        None => 32 / tasks.min(32),
    };
    let rows_per_tg = if r2 { 16 } else { 8 * per_sg * r };
    let mut groups = 0;
    for &(w, y) in sets {
        ensure!(
            w.ty == w0.ty && w.cols == w0.cols && w.row_bytes == w0.row_bytes,
            "GEMV sets of different types or widths"
        );
        check_f32(&[y])?;
        let stride = if l.y_row_stride == 0 { w.rows } else { l.y_row_stride };
        ensure!(y.numel() >= (m - 1) * stride + w.rows, "GEMV output of {}", y.numel());
        let x_groups = w.rows.div_ceil(l.x_group);
        ensure!(
            x.numel() >= (m - 1) * l.x_row_stride + x_groups * w.cols,
            "GEMV input of {} values",
            x.numel()
        );
        groups += w.rows.div_ceil(rows_per_tg);
    }
    let mut bindings = vec![x.binding()];
    for s in 0..4 {
        bindings.push(sets.get(s).map_or(x.binding(), |(w, _)| w.t.binding()));
    }
    for s in 0..4 {
        bindings.push(sets.get(s).map_or(x.binding(), |(_, y)| y.binding()));
    }
    let rows = |s: usize| u32_bytes(sets.get(s).map_or(0, |(w, _)| w.rows));
    let name = match g16 {
        Some((name, _)) => name,
        None if r2 => gemv_r2_name(m)?,
        None if r == 2 => "glm53_gemv_q5_k_m1_r2",
        None => gemv_name(w0.ty, m)?,
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &bindings,
        &[
            &u32_bytes(w0.cols),
            &u32_bytes(m),
            &u32_bytes(w0.row_bytes),
            &rows(0),
            &rows(1),
            &rows(2),
            &rows(3),
            &u32_bytes(l.x_group),
            &u32_bytes(l.x_row_stride),
            &u32_bytes(l.y_row_stride),
        ],
        Grid::Threadgroups { groups: (groups, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Weights of more rows than this take two rows a subgroup where a kernel
/// for it exists (Q5_K at one row: Gemma's LM head).
const GEMV_R2_MIN_ROWS: usize = 32768;

/// Weight rows a subgroup of the GEMV of `ty` takes for each of its lane
/// groups at `m` rows over weights of up to `rows` rows: two for a wide
/// Q5_K at one row (they share the x loads), else one.
fn gemv_rows_per_subgroup(ty: GgmlType, m: usize, rows: usize) -> usize {
    if ty == GgmlType::Q5K && m == 1 && rows >= GEMV_R2_MIN_ROWS { 2 } else { 1 }
}

/// Rows (of activations) at which [`gemm`] takes over from the GEMV.
pub const GEMM_MIN_ROWS: usize = GEMV_MAX_ROWS + 1;

/// The GEMM's operand layout beyond `y[i][n] = w[n] . x[i]`, `x` and `y`
/// rows `ldx` and `ldy` elements apart; batch `b` (grid z) reads `x` from
/// `b * x_batch` elements on, weight rows from `b * w_batch` on, and writes
/// `y` from `b * y_batch` on.
#[derive(Clone, Copy)]
pub struct GemmLayout {
    pub ldx: usize,
    pub ldy: usize,
    pub batches: usize,
    pub x_batch: usize,
    pub w_batch: usize,
    pub y_batch: usize,
}

fn gemm_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "glm53_gemm_q8_0",
        GgmlType::Q4K => "glm53_gemm_q4_k",
        GgmlType::Bf16 => "glm53_gemm_bf16",
        GgmlType::F32 => "glm53_gemm_f32",
        GgmlType::F16 => "glm53_gemm_f16",
        GgmlType::Q4_0 => "glm53_gemm_q4_0",
        GgmlType::Q5K => "glm53_gemm_q5_k",
        GgmlType::Q6K => "glm53_gemm_q6_k",
        GgmlType::Iq4Nl => "glm53_gemm_iq4_nl",
        GgmlType::Iq2S => "glm53_gemm_iq2_s",
        other => anyhow::bail!("no GLM GEMM over {other:?}"),
    })
}

/// `y[i][n] = w[n] . x[i]` for `m` rows on the matrix units (operands
/// rounded to bfloat16, f32 accumulation): the prefill's projections.
/// `n` is the weight rows a batch uses (all of `w`'s without batches).
#[allow(clippy::too_many_arguments)]
pub fn gemm_with(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
    n: usize,
    l: GemmLayout,
) -> Result<()> {
    check_f32(&[x])?;
    if !ctx.matrix_units() {
        return gemm_any(ctx, pass, w, x, y, m, n, l, sgemm_name(w.ty)?, (64, 128), 256);
    }
    gemm_any(ctx, pass, w, x, y, m, n, l, gemm_name(w.ty)?, (64, 64), 128)
}

/// The GEMM without matrix units (`sgemm.comp`: f32 operands, tiles of 64
/// activation rows x 128 weight rows, 256 threads, K in 32-deep steps).
fn sgemm_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "sgemm_q8_0",
        GgmlType::Q4_0 => "sgemm_q4_0",
        GgmlType::Q4K => "sgemm_q4_k",
        GgmlType::Q5K => "sgemm_q5_k",
        GgmlType::Q6K => "sgemm_q6_k",
        GgmlType::Iq4Nl => "sgemm_iq4_nl",
        GgmlType::Iq2S => "sgemm_iq2_s",
        GgmlType::Iq2Xxs => "sgemm_iq2_xxs",
        GgmlType::F16 => "sgemm_f16",
        GgmlType::F32 => "sgemm_f32",
        other => anyhow::bail!("no GEMM without matrix units over {other:?}"),
    })
}

/// [`gemm`] over `xb`, the input's rows as bfloat ([`to_bf16`]'s rounding:
/// the results are [`gemm`]'s bit for bit) for Q8_0, Q4_K, Q5_K and Q6_K
/// weights: the
/// f32 kernel rounds every element in every column tile (about a third of
/// a 1 024-row call; 2026-11-14).
pub fn gemm_bx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    xb: &Tensor,
    y: &Tensor,
    m: usize,
) -> Result<()> {
    let l = GemmLayout {
        ldx: w.cols,
        ldy: w.rows,
        batches: 1,
        x_batch: 0,
        w_batch: 0,
        y_batch: 0,
    };
    gemm_bx_with(ctx, pass, w, xb, y, m, w.rows, l)
}

/// [`gemm_bx`] with [`gemm_with`]'s layout (`ldx` and `x_batch` multiples
/// of 8); F16 weights too.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bx_with(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    xb: &Tensor,
    y: &Tensor,
    m: usize,
    n: usize,
    l: GemmLayout,
) -> Result<()> {
    ensure!(xb.dtype() == DType::BF16, "bfloat GEMM input is {:?}", xb.dtype());
    ensure!(w.cols.is_multiple_of(8), "bfloat GEMM rows of {} values", w.cols);
    ensure!(
        l.ldx.is_multiple_of(8) && l.x_batch.is_multiple_of(8),
        "bfloat GEMM input layout"
    );
    // Four subgroups a threadgroup for an unbatched short run over at least
    // 8 192 Q8_0 or Q4_K weight rows: gemm_small_timing (eight calls
    // overlapping, as a level's do; 2026-11-15), kda_v Q8_0 8192 x 4096 at
    // 8 / 24 / 64 rows 270 / 276 / 307 -> 186 / 195 / 206 us, kda_q Q4_K
    // 205 / 193 / 279 -> 182 / 185 / 200; narrower weights (4096 x 8192,
    // 2048 x 4096) were 10-20% slower. The same products in the same order.
    let sg4 = m <= 128 && n >= 8192 && l.batches == 1;
    let (name, rows, threads) = match (w.ty, sg4) {
        (GgmlType::Q8_0, false) => ("glm53_gemm_q8_0_bx", GEMM_BX_ROWS, GEMM_BX_THREADS),
        (GgmlType::Q4K, false) => ("glm53_gemm_q4_k_bx", GEMM_BX_ROWS, GEMM_BX_THREADS),
        (GgmlType::Q8_0, true) => ("glm53_gemm_q8_0_bx4", 128, 128),
        (GgmlType::Q4K, true) => ("glm53_gemm_q4_k_bx4", 128, 128),
        (GgmlType::F16, _) => ("glm53_gemm_f16_bx", GEMM_BX_ROWS, GEMM_BX_THREADS),
        (GgmlType::Q5K, _) => ("glm53_gemm_q5_k_bx", GEMM_BX_ROWS, GEMM_BX_THREADS),
        (GgmlType::Q6K, _) => ("glm53_gemm_q6_k_bx", GEMM_BX_ROWS, GEMM_BX_THREADS),
        (other, _) => anyhow::bail!("no bfloat-input GLM GEMM over {other:?}"),
    };
    gemm_any(ctx, pass, w, xb, y, m, n, l, name, (rows, 64), threads)
}

/// Rows of X a threadgroup of the bfloat-input GEMM takes (the kernel's
/// GM_BX_SG * GM_BX_SUB: 8 subgroups of 32).
const GEMM_BX_ROWS: usize = 256;
/// Threads a threadgroup of it (32 a subgroup of the kernel's GM_BX_SG).
const GEMM_BX_THREADS: usize = 256;

/// Whether [`gemm_bx`] takes weights of this type.
pub fn gemm_bx_takes(w: &QWeight) -> bool {
    matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K)
        && w.cols.is_multiple_of(8)
}

/// Whether [`gemm_bx_with`] takes weights of this type (F16 too).
pub fn gemm_bx_with_takes(w: &QWeight) -> bool {
    matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::F16)
        && w.cols.is_multiple_of(8)
}

/// `out` = the first `n` values of `x` as bfloat (rounded to nearest even),
/// `n` even.
pub fn to_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    check_f32(&[x])?;
    ensure!(out.dtype() == DType::BF16, "bfloat copy into {:?}", out.dtype());
    ensure!(
        n.is_multiple_of(2) && x.numel() >= n && out.numel() >= n,
        "bfloat copy of {n} values"
    );
    let to_bf16 = ctx.pipeline("glm53_to_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &to_bf16,
        &[x.binding(), out.binding()],
        &[],
        Grid::Threads { grid: (n / 2, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
fn gemm_any(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
    n: usize,
    l: GemmLayout,
    name: &'static str,
    (tile_rows, tile_cols): (usize, usize),
    threads: usize,
) -> Result<()> {
    check_f32(&[y])?;
    // The kernel walks K in 64-deep steps with no tail.
    ensure!(
        w.cols.is_multiple_of(64),
        "GEMM row of {} values (a multiple of 64)",
        w.cols
    );
    ensure!(
        l.ldx % 4 == 0 && l.x_batch % 4 == 0,
        "GEMM activations must be 16-byte aligned"
    );
    ensure!((l.batches - 1) * l.w_batch + n <= w.rows, "GEMM weight rows");
    ensure!(
        x.numel() >= (l.batches - 1) * l.x_batch + (m - 1) * l.ldx + w.cols,
        "GEMM input of {}",
        x.numel()
    );
    ensure!(
        y.numel() >= (l.batches - 1) * l.y_batch + (m - 1) * l.ldy + n,
        "GEMM output"
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.t.binding(), y.binding()],
        &[
            &u32_bytes(m),
            &u32_bytes(n),
            &u32_bytes(w.cols),
            &u32_bytes(w.row_bytes),
            &u32_bytes(l.ldx),
            &u32_bytes(l.ldy),
            &u32_bytes(l.x_batch),
            &u32_bytes(l.w_batch),
            &u32_bytes(l.y_batch),
        ],
        Grid::Threadgroups {
            groups: (n.div_ceil(tile_cols), m.div_ceil(tile_rows), l.batches),
            threadgroup: (threads, 1, 1),
        },
    )
}

/// [`gemm_with`] over plain packed rows.
pub fn gemm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
) -> Result<()> {
    let l = GemmLayout {
        ldx: w.cols,
        ldy: w.rows,
        batches: 1,
        x_batch: 0,
        w_batch: 0,
        y_batch: 0,
    };
    gemm_with(ctx, pass, w, x, y, m, w.rows, l)
}

/// Scratch for [`moe_grouped`], sized for `pairs` routed pairs.
pub struct MoeGroupScratch {
    pub pairs: Tensor,
    pub tiles: Tensor,
    pub tile_count: Tensor,
    /// The run's input rows as bfloat (the gate/up GEMMs' operand).
    pub xb: Tensor,
    pub g: Tensor,
    pub u: Tensor,
    /// BF16: the down GEMM's operand.
    pub mid: Tensor,
    pub down: Tensor,
}

impl MoeGroupScratch {
    pub fn new(
        ctx: &MetalContext,
        pairs: usize,
        experts: usize,
        ff: usize,
        hidden: usize,
    ) -> Result<Self> {
        let max_tiles = pairs.div_ceil(64) + experts;
        Ok(Self {
            pairs: Tensor::zeros(ctx, &[pairs], DType::U32)?,
            tiles: Tensor::zeros(ctx, &[3 * max_tiles], DType::U32)?,
            tile_count: Tensor::zeros(ctx, &[1], DType::U32)?,
            xb: Tensor::zeros(ctx, &[pairs * hidden], DType::BF16)?,
            g: Tensor::zeros(ctx, &[pairs * ff], DType::F32)?,
            u: Tensor::zeros(ctx, &[pairs * ff], DType::F32)?,
            mid: Tensor::zeros(ctx, &[pairs * ff], DType::BF16)?,
            down: Tensor::zeros(ctx, &[pairs * hidden], DType::F32)?,
        })
    }

    /// For runs of up to `rows` rows of `used` pairs whose IQ2_XXS gate and
    /// up take the fused GEMM (DeepSeek V4's): no separate gate and up
    /// outputs, the bfloat input `rows` rows long.
    pub fn fused(
        ctx: &MetalContext,
        rows: usize,
        used: usize,
        experts: usize,
        ff: usize,
        hidden: usize,
    ) -> Result<Self> {
        let pairs = rows * used;
        let max_tiles = pairs.div_ceil(64) + experts;
        Ok(Self {
            pairs: Tensor::zeros(ctx, &[pairs], DType::U32)?,
            tiles: Tensor::zeros(ctx, &[3 * max_tiles], DType::U32)?,
            tile_count: Tensor::zeros(ctx, &[1], DType::U32)?,
            xb: Tensor::zeros(ctx, &[rows * hidden], DType::BF16)?,
            g: Tensor::zeros(ctx, &[1], DType::F32)?,
            u: Tensor::zeros(ctx, &[1], DType::F32)?,
            mid: Tensor::zeros(ctx, &[pairs * ff], DType::BF16)?,
            down: Tensor::zeros(ctx, &[pairs * hidden], DType::F32)?,
        })
    }
}

/// Routed experts over a run of `m` rows grouped by expert (prefill): each
/// expert's rows are one GEMM on the matrix units instead of one GEMV per
/// pair. `out[i] = sum_s w[i][s] * down(swiglu(gate x[i], up x[i])) +
/// out[i]` (`out` holds the shared expert's output on entry).
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
    weights: &Tensor,
    out: &Tensor,
    sc: &MoeGroupScratch,
    m: usize,
    used: usize,
    experts: usize,
    limit: f32,
) -> Result<()> {
    // Tiles of up to 128 rows once an expert averages 48 pairs or more.
    let tiles = if m * used >= 48 * experts { MoeTiles::R128 } else { MoeTiles::R64 };
    moe_grouped_with(
        ctx, pass, gate, up, down, ff, hidden, x, sel, weights, out, sc, m, used,
        experts, limit, true, tiles, 0,
    )
}

/// How [`moe_grouped`]'s expert GEMMs tile an expert's rows (the results
/// are the same bit for bit).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeTiles {
    /// Up to 64 rows, four subgroups of 16.
    R64,
    /// Up to 128 rows, four subgroups of 32.
    R128,
}

/// [`moe_grouped`] with the tiling chosen by the caller.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_tiles(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    down: &QWeight,
    ff: usize,
    hidden: usize,
    x: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    out: &Tensor,
    sc: &MoeGroupScratch,
    m: usize,
    used: usize,
    experts: usize,
    limit: f32,
    tiles: MoeTiles,
) -> Result<()> {
    moe_grouped_with(
        ctx, pass, gate, up, down, ff, hidden, x, sel, weights, out, sc, m, used,
        experts, limit, true, tiles, 0,
    )
}

/// Most rows an expert's tile computes off the matrix units in
/// [`moe_grouped_dot`] (the kernels' GM_DOT_ROWS).
pub const MOE_DOT_ROWS: usize = 4;

/// [`moe_grouped`] where an expert with at most [`MOE_DOT_ROWS`] rows is
/// computed by register dot products of its decoded f32 weights and the
/// rows' bfloat values instead of a tile on the matrix units: a short
/// prompt's experts have one or two rows each, and their tiles cost almost
/// as much as full ones. moe_grouped_timing (2026-11-15), a layer: 24-token
/// density 2.46 -> 1.56 ms, 64-token 2.77 -> 1.93, from 256 unchanged; a
/// cap of 8 rows lost at 256 (3.43 -> 4.68). The values are the same
/// products up to the weights' bfloat rounding, summed in another order.
/// GLM 5.3's prefill.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_dot(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    down: &QWeight,
    ff: usize,
    hidden: usize,
    x: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    out: &Tensor,
    sc: &MoeGroupScratch,
    m: usize,
    used: usize,
    experts: usize,
    limit: f32,
) -> Result<()> {
    let tiles = if m * used >= 48 * experts { MoeTiles::R128 } else { MoeTiles::R64 };
    moe_grouped_with(
        ctx, pass, gate, up, down, ff, hidden, x, sel, weights, out, sc, m, used,
        experts, limit, true, tiles, MOE_DOT_ROWS,
    )
}

/// [`moe_grouped`]; `fuse_gate_up` false takes IQ2_XXS experts through the
/// separate gate and up GEMMs and SwiGLU too (the tests' reference);
/// `tiles` caps a tile's pairs (the results are the same either way).
#[allow(clippy::too_many_arguments)]
fn moe_grouped_with(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    down: &QWeight,
    ff: usize,
    hidden: usize,
    x: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    out: &Tensor,
    sc: &MoeGroupScratch,
    m: usize,
    used: usize,
    experts: usize,
    limit: f32,
    fuse_gate_up: bool,
    tiles: MoeTiles,
    dot_rows: usize,
) -> Result<()> {
    ensure!(dot_rows <= MOE_DOT_ROWS, "register path for {dot_rows} rows");
    let tile_rows = if tiles == MoeTiles::R64 { 64 } else { 128 };
    let r128 = tiles == MoeTiles::R128;
    let pairs = m * used;
    ensure!(
        sc.pairs.numel() >= pairs,
        "grouped MoE scratch for {} pairs",
        sc.pairs.numel()
    );
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
    ensure!(
        x.numel() >= m * hidden && sc.xb.numel() >= m * hidden,
        "grouped MoE input"
    );
    to_bf16(ctx, pass, x, &sc.xb, m * hidden)?;
    pass.memory_barrier()?;
    let max_tiles = pairs.div_ceil(tile_rows) + experts;
    let gemm_moe =
        |w: &QWeight, x: &Tensor, y: &Tensor, n: usize, x_shift: usize| -> Result<()> {
            let name = match (w.ty, r128) {
                (GgmlType::Iq2Xxs, false) => "glm53_gemm_moe_iq2_xxs",
                (GgmlType::Iq2Xxs, true) => "glm53_gemm_moe_iq2_xxs_r128",
                (GgmlType::Q2K, false) => "glm53_gemm_moe_q2_k",
                (GgmlType::Q2K, true) => "glm53_gemm_moe_q2_k_r128",
                (GgmlType::Q4K, false) => "glm53_gemm_moe_q4_k",
                (GgmlType::Q4K, true) => "glm53_gemm_moe_q4_k_r128",
                (GgmlType::Q8_0, false) => "glm53_gemm_moe_q8_0",
                (GgmlType::Q8_0, true) => "glm53_gemm_moe_q8_0_r128",
                (other, _) => anyhow::bail!("no grouped expert GEMM over {other:?}"),
            };
            ensure!(
                w.cols.is_multiple_of(256) && n.is_multiple_of(64),
                "grouped expert GEMM of {n} x {}",
                w.cols
            );
            let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
            pass.dispatch_at(
                &pipeline,
                &[
                    x.binding(),
                    w.t.binding(),
                    y.binding(),
                    sc.pairs.binding(),
                    sc.tiles.binding(),
                    sc.tile_count.binding(),
                ],
                &[
                    &u32_bytes(n),
                    &u32_bytes(w.cols),
                    &u32_bytes(w.row_bytes),
                    &u32_bytes(x_shift),
                    &u32_bytes(dot_rows),
                ],
                Grid::Threadgroups {
                    groups: (n.div_ceil(64), max_tiles, 1),
                    threadgroup: (128, 1, 1),
                },
            )
        };
    ensure!((1..1 << 31).contains(&used), "experts used {used}");
    // The pair's row: a shift for a power of two, else a division (the
    // kernels' x_shift with bit 31 set).
    let shift = if used.is_power_of_two() {
        used.trailing_zeros() as usize
    } else {
        0x8000_0000 | used
    };
    if gate.ty == GgmlType::Iq2Xxs && fuse_gate_up {
        // Gate, up and the SwiGLU in one GEMM: a tile is 32 columns of
        // each, written as the down GEMM's bfloat operand.
        ensure!(
            gate.cols.is_multiple_of(256) && ff.is_multiple_of(32),
            "grouped expert GEMM of {ff} x {}",
            gate.cols
        );
        let name = if r128 {
            "glm53_gemm_moe_gate_up_iq2_xxs_r128"
        } else {
            "glm53_gemm_moe_gate_up_iq2_xxs"
        };
        let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &pipeline,
            &[
                sc.xb.binding(),
                gate.t.binding(),
                sc.mid.binding(),
                sc.pairs.binding(),
                sc.tiles.binding(),
                sc.tile_count.binding(),
                up.t.binding(),
                weights.binding(),
            ],
            &[
                &u32_bytes(ff),
                &u32_bytes(gate.cols),
                &u32_bytes(gate.row_bytes),
                &u32_bytes(shift),
                &f32_bytes(limit),
                &u32_bytes(dot_rows),
            ],
            Grid::Threadgroups {
                groups: (ff / 32, max_tiles, 1),
                threadgroup: (128, 1, 1),
            },
        )?;
    } else {
        ensure!(
            sc.g.numel() >= pairs * ff && sc.u.numel() >= pairs * ff,
            "grouped MoE scratch without separate gate and up outputs"
        );
        gemm_moe(gate, &sc.xb, &sc.g, ff, shift)?;
        gemm_moe(up, &sc.xb, &sc.u, ff, shift)?;
        pass.memory_barrier()?;
        let swiglu = ctx.pipeline("glm53_moe_swiglu", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &swiglu,
            &[sc.g.binding(), sc.u.binding(), weights.binding(), sc.mid.binding()],
            &[&u32_bytes(ff), &f32_bytes(limit)],
            Grid::Threads { grid: (ff, pairs, 1), threadgroup: (256, 1, 1) },
        )?;
    }
    pass.memory_barrier()?;
    gemm_moe(down, &sc.mid, &sc.down, hidden, 0)?;
    pass.memory_barrier()?;
    let combine = ctx.pipeline("glm53_moe_combine", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &combine,
        &[sc.down.binding(), out.binding(), out.binding()],
        &[&u32_bytes(hidden), &u32_bytes(used)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Routed experts' gate and up with the clamped SwiGLU, weighted:
/// `mid[i * used + s] = w[i][s] * swiglu(gate_e x[i], up_e x[i])` for the
/// `used` experts `sel[i]` of each of the `m` rows.
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
    limit: f32,
) -> Result<()> {
    check_f32(&[x, weights, mid])?;
    ensure!(
        gate.ty == up.ty && gate.cols == up.cols && gate.rows == up.rows,
        "gate and up experts differ"
    );
    ensure!(gate.rows.is_multiple_of(ff), "expert rows {} of {ff}", gate.rows);
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * used, "selection");
    ensure!(mid.numel() >= m * used * ff, "mid of {}", mid.numel());
    // Two IQ2_XXS rows: an expert both rows picked is decoded once.
    let name = if m == 2 && gate.ty == GgmlType::Iq2Xxs {
        "glm53_moe_gate_up_iq2_xxs_m2"
    } else {
        moe_name(true, gate.ty)?
    };
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
            &f32_bytes(limit),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(8), m * used, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// [`moe_gate_up`] for IQ2_XXS experts over `xh`, the activations as f16
/// in the dot's pairing (DeepSeek V4's `x_f16_iq2`): half the activation
/// bytes a task.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_xh(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    ff: usize,
    xh: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    mid: &Tensor,
    m: usize,
    used: usize,
    limit: f32,
) -> Result<()> {
    check_f32(&[weights, mid])?;
    ensure!(xh.dtype() == DType::F16, "f16 expert activations");
    ensure!(
        gate.ty == GgmlType::Iq2Xxs && up.ty == gate.ty && gate.cols == up.cols && gate.rows == up.rows,
        "f16-activation gate/up of {:?} experts",
        gate.ty
    );
    ensure!(gate.rows.is_multiple_of(ff) && gate.cols.is_multiple_of(32), "expert shape");
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * used, "selection");
    ensure!(xh.numel() >= m * gate.cols && mid.numel() >= m * used * ff, "expert buffers");
    let pipeline = ctx.pipeline("glm53_moe_gate_up_iq2_xxs_xh", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.t.binding(),
            up.t.binding(),
            xh.binding(),
            sel.binding(),
            weights.binding(),
            mid.binding(),
        ],
        &[
            &u32_bytes(gate.cols),
            &u32_bytes(ff),
            &u32_bytes(gate.row_bytes),
            &u32_bytes(used),
            &f32_bytes(limit),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(8), m * used, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Routed experts' down projection summed over each row's slots, plus
/// `add` when given: `out[i] = sum_s down_{sel[i][s]} mid[i * used + s]
/// (+ add[i])`; `out` may be `add`.
#[allow(clippy::too_many_arguments)]
pub fn moe_down(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    down: &QWeight,
    hidden: usize,
    mid: &Tensor,
    sel: &Tensor,
    add: Option<&Tensor>,
    out: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    check_f32(&[mid, out])?;
    ensure!(down.rows.is_multiple_of(hidden), "down rows {} of {hidden}", down.rows);
    ensure!(out.numel() >= m * hidden, "out of {}", out.numel());
    let name = moe_name(false, down.ty)?;
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            down.t.binding(),
            mid.binding(),
            sel.binding(),
            out.binding(),
            add.unwrap_or(out).binding(),
        ],
        &[
            &u32_bytes(down.cols),
            &u32_bytes(hidden),
            &u32_bytes(down.row_bytes),
            &u32_bytes(used),
            &u32_bytes(usize::from(add.is_some())),
        ],
        Grid::Threadgroups {
            groups: (hidden.div_ceil(8), m, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// The router: top `k` of `sigmoid(logits) + bias` per row, weights the
/// selected probabilities normalized times `scale`.
#[allow(clippy::too_many_arguments)]
pub fn router_topk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    bias: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    m: usize,
    n_expert: usize,
    k: usize,
    scale: f32,
) -> Result<()> {
    check_f32(&[logits, bias, weights])?;
    ensure!(n_expert <= 384 && k <= 32, "router of {n_expert} experts, top {k}");
    let pipeline = ctx.pipeline("glm53_router_topk", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), bias.binding(), sel.binding(), weights.binding()],
        &[&u32_bytes(n_expert), &u32_bytes(k), &f32_bytes(scale)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// `out[0]` = the largest softmax probability of the `n` logits.
pub fn top_prob(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    check_f32(&[logits, out])?;
    ensure!(logits.numel() >= n && n > 0, "top probability of {n} logits");
    ensure!(out.numel() >= 1, "top probability into an empty tensor");
    let pipeline = ctx.pipeline("glm53_top_prob", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), out.binding()],
        &[&u32_bytes(n)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// RMSNorm of `m` rows of `n`, times `w` when given; `out` may be `x`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: Option<&Tensor>,
    out: &Tensor,
    m: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    check_f32(&[x, out])?;
    ensure!(x.numel() >= m * n && out.numel() >= m * n, "rmsnorm of {m} x {n}");
    let pipeline = ctx.pipeline("glm53_rmsnorm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.unwrap_or(x).binding(), out.binding()],
        &[&u32_bytes(n), &f32_bytes(eps), &u32_bytes(usize::from(w.is_some()))],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// HC pre after the control projection: Sinkhorn split of `mix` into
/// `split` (pre, post, comb) and `xn = rmsnorm(sum_h pre[h] * h_streams[h],
/// norm)`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    mix: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    streams: &Tensor,
    norm: &Tensor,
    split: &Tensor,
    xn: &Tensor,
    m: usize,
    hidden: usize,
    iters: usize,
    hc_eps: f32,
    rms_eps: f32,
) -> Result<()> {
    check_f32(&[mix, scale, base, streams, norm, split, xn])?;
    ensure!(
        streams.numel() >= m * 4 * hidden && mix.numel() >= m * 24,
        "hc_pre shapes"
    );
    let pipeline = ctx.pipeline("glm53_hc_pre", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            mix.binding(),
            scale.binding(),
            base.binding(),
            streams.binding(),
            norm.binding(),
            split.binding(),
            xn.binding(),
        ],
        &[
            &u32_bytes(hidden),
            &u32_bytes(iters),
            &f32_bytes(hc_eps),
            &f32_bytes(rms_eps),
        ],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// HC post: `out[dst] = block * post[dst] + sum_src comb[dst + 4 src] *
/// streams[src]`; `out` may be `streams`.
#[allow(clippy::too_many_arguments)]
pub fn hc_post(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    streams: &Tensor,
    block: &Tensor,
    split: &Tensor,
    out: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    check_f32(&[streams, block, split, out])?;
    let pipeline = ctx.pipeline("glm53_hc_post", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[streams.binding(), block.binding(), split.binding(), out.binding()],
        &[&u32_bytes(hidden)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Threadgroups (and partial sums per row) of [`hc_post_mix`].
pub fn hc_mix_groups(hidden: usize) -> usize {
    hidden / 64
}

/// HC post of the previous sublayer (when `post` gives its output and the
/// split holding its post/comb weights; in place on `streams`) fused with
/// the partial sums of the next control projection `fn_w . streams` and of
/// `sum(streams^2)`, `[m][groups][25]` into `partial`.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_mix(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    streams: &Tensor,
    post: Option<(&Tensor, &Tensor)>,
    fn_w: &QWeight,
    partial: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    check_f32(&[streams, partial])?;
    ensure!(
        matches!(fn_w.ty, GgmlType::Bf16 | GgmlType::F16)
            && fn_w.rows == 24
            && fn_w.cols == 4 * hidden,
        "HC fn shape"
    );
    ensure!(hidden.is_multiple_of(64), "hidden {hidden}");
    ensure!(partial.numel() >= m * hc_mix_groups(hidden) * 25, "partials");
    let (block, split) = post.unwrap_or((streams, streams));
    let name = if fn_w.ty == GgmlType::F16 {
        "glm53_hc_post_mix_f16"
    } else {
        "glm53_hc_post_mix"
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            streams.binding(),
            block.binding(),
            split.binding(),
            fn_w.t.binding(),
            partial.binding(),
        ],
        &[&u32_bytes(hidden), &u32_bytes(usize::from(post.is_some()))],
        Grid::Threadgroups {
            groups: (hc_mix_groups(hidden), m, 1),
            threadgroup: (128, 1, 1),
        },
    )
}

/// HC pre from [`hc_post_mix`]'s partial sums: the split into `split`
/// (pre, post, comb) and `xn = rmsnorm(sum_h pre[h] * streams[h], norm)`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre2(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    partial: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    streams: &Tensor,
    norm: &Tensor,
    split: &Tensor,
    xn: &Tensor,
    m: usize,
    hidden: usize,
    iters: usize,
    hc_eps: f32,
    rms_eps: f32,
) -> Result<()> {
    check_f32(&[partial, scale, base, streams, norm, split, xn])?;
    ensure!(hidden == 4096, "hc_pre2 is written for hidden 4096, not {hidden}");
    let pipeline = ctx.pipeline("glm53_hc_pre2", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            partial.binding(),
            scale.binding(),
            base.binding(),
            streams.binding(),
            norm.binding(),
            split.binding(),
            xn.binding(),
        ],
        &[
            &u32_bytes(hidden),
            &u32_bytes(iters),
            &f32_bytes(hc_eps),
            &f32_bytes(rms_eps),
            &u32_bytes(hc_mix_groups(hidden)),
        ],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// The mean of the four streams.
pub fn hc_mean(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    streams: &Tensor,
    out: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    check_f32(&[streams, out])?;
    let pipeline = ctx.pipeline("glm53_hc_mean", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[streams.binding(), out.binding()],
        &[&u32_bytes(hidden)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Embeds `m` tokens (U32) from a Q8_0 table into four streams each.
pub fn embed(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &QWeight,
    tokens: &Tensor,
    streams: &Tensor,
    m: usize,
) -> Result<()> {
    ensure!(table.ty == GgmlType::Q8_0, "embedding table of {:?}", table.ty);
    check_f32(&[streams])?;
    let pipeline = ctx.pipeline("glm53_embed_q8_0", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.t.binding(), tokens.binding(), streams.binding()],
        &[&u32_bytes(table.cols), &u32_bytes(table.row_bytes)],
        Grid::Threads { grid: (table.cols, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// `out = swiglu(g, u)` over `n` values with GLM's clamp.
pub fn swiglu(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    g: &Tensor,
    u: &Tensor,
    out: &Tensor,
    n: usize,
    limit: f32,
) -> Result<()> {
    check_f32(&[g, u, out])?;
    let pipeline = ctx.pipeline("glm53_swiglu", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[g.binding(), u.binding(), out.binding()],
        &[&f32_bytes(limit)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// `out = a + b` over `n` values.
pub fn add(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    check_f32(&[a, b, out])?;
    let pipeline = ctx.pipeline("glm53_add", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[a.binding(), b.binding(), out.binding()],
        &[],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Zeroes `t` on the GPU (its bytes a multiple of 16).
pub fn zero(ctx: &MetalContext, pass: &ComputePass<'_>, t: &Tensor) -> Result<()> {
    let n = t.byte_len();
    ensure!(n.is_multiple_of(16), "GPU zero fill of {n} bytes");
    let pipeline = ctx.pipeline("glm53_zero", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[t.binding()],
        &[&u32_bytes(n / 16)],
        Grid::Threads { grid: (n / 16, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// One layer's KDA parameters on the GPU (f32).
pub struct KdaParams<'a> {
    pub q_conv: &'a Tensor,
    pub k_conv: &'a Tensor,
    pub v_conv: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub o_norm: &'a Tensor,
    pub heads: usize,
    pub lower_bound: f32,
    pub eps: f32,
}

/// The raw projections KDA consumes, `[rows][projection]` (`beta`
/// `[rows][heads]`); the prefill path overwrites q, k, v and gate in place.
pub struct KdaInputs<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub gate: &'a Tensor,
    pub beta: &'a Tensor,
    pub out_gate: &'a Tensor,
    /// Scratch for runs of more than [`KDA_CHUNK`] rows: the raw rows
    /// before each chunk, [`kda_edges_len`] f32.
    pub edges: Option<&'a Tensor>,
}

/// Tokens a threadgroup of the KDA prefill's prepare stage walks.
pub const KDA_CHUNK: usize = 32;

/// The f32 values of the edge scratch for runs of up to `rows` rows.
pub fn kda_edges_len(rows: usize, heads: usize) -> usize {
    rows.div_ceil(KDA_CHUNK) * 9 * heads * 128
}

/// Where a run of KDA tokens also leaves the conv window and state as they
/// were after one of its rows (speculative decoding's rollback point).
pub struct KdaSnapshot<'a> {
    pub row: usize,
    pub conv: &'a Tensor,
    pub state: &'a Tensor,
}

/// The decay's log is above `lower_bound` a token; the chunked form's
/// factors reach e^(-16 lower_bound), so it takes bounds down to this one
/// (e^80; f32 ends at e^88).
pub const KDA_WY_MIN_BOUND: f32 = -5.0;

/// Whether [`kda`] takes the chunked form: runs of more than [`KDA_CHUNK`]
/// rows without a snapshot, and a finite gate bound in
/// [[`KDA_WY_MIN_BOUND`], 0] (decays at most 1: a positive bound would grow
/// the chunk's cumulative products past f32).
pub(crate) fn kda_takes_wy(lower_bound: f32, rows: usize, snapshot: bool) -> bool {
    !snapshot && rows > KDA_CHUNK && (KDA_WY_MIN_BOUND..=0.0).contains(&lower_bound)
}

/// KDA for `rows` consecutive tokens of one sequence: the one-token kernel
/// for a single row without a snapshot; prepare, then the chunked form
/// (glm53_kda_chunk_wy and _scan) for runs of more than [`KDA_CHUNK`] rows
/// without a snapshot, else the token-serial recurrence; then output.
/// The chunked form sums in another order than the recurrence (1 024 rows
/// a layer: 4.4 ms recurrence, see kda_decode_timing; 2026-11-14).
#[allow(clippy::too_many_arguments)]
pub fn kda(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &KdaParams<'_>,
    x: &KdaInputs<'_>,
    conv_state: &Tensor,
    state: &Tensor,
    out: &Tensor,
    rows: usize,
    snap: Option<KdaSnapshot<'_>>,
) -> Result<()> {
    if kda_takes_wy(p.lower_bound, rows, snap.is_some()) {
        return kda_wy(ctx, pass, p, x, conv_state, state, out, rows);
    }
    kda_chunked(ctx, pass, p, x, conv_state, state, out, rows, snap, KDA_CHUNK)
}

/// The chunked form of [`kda`] (runs of more than [`KDA_CHUNK`] rows, no
/// snapshot). Its scratch is the edges tensor, which the prepare stage has
/// read by then: [`KDA_CHUNK`]^2 + 128 f32 a chunk and head.
#[allow(clippy::too_many_arguments)]
fn kda_wy(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &KdaParams<'_>,
    x: &KdaInputs<'_>,
    conv_state: &Tensor,
    state: &Tensor,
    out: &Tensor,
    rows: usize,
) -> Result<()> {
    check_f32(&[x.q, x.k, x.v, x.gate, x.beta, x.out_gate, conv_state, state, out])?;
    let proj = p.heads * 128;
    ensure!(
        state.numel() >= p.heads * 128 * 128 && conv_state.numel() >= 9 * proj,
        "KDA state"
    );
    let chunks = rows.div_ceil(KDA_CHUNK);
    let Some(scratch) = x.edges else {
        anyhow::bail!("KDA run of {rows} rows without edges");
    };
    ensure!(
        scratch.numel() >= chunks * p.heads * (KDA_CHUNK * KDA_CHUNK + 128)
            && scratch.numel() >= chunks * 9 * proj,
        "KDA edges"
    );
    kda_prepare(ctx, pass, p, x, conv_state, rows, None, KDA_CHUNK, true)?;
    pass.memory_barrier()?;
    let wy = ctx.pipeline("glm53_kda_chunk_wy", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &wy,
        &[
            x.q.binding(),
            x.k.binding(),
            x.v.binding(),
            x.gate.binding(),
            x.beta.binding(),
            scratch.binding(),
        ],
        &[&u32_bytes(p.heads), &u32_bytes(rows)],
        Grid::Threadgroups { groups: (chunks, p.heads, 1), threadgroup: (128, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let scan = ctx.pipeline("glm53_kda_chunk_scan", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scan,
        &[
            x.q.binding(),
            x.k.binding(),
            x.v.binding(),
            x.gate.binding(),
            scratch.binding(),
            state.binding(),
            out.binding(),
        ],
        &[&u32_bytes(p.heads), &u32_bytes(rows)],
        Grid::Threadgroups { groups: (4, p.heads, 1), threadgroup: (128, 1, 1) },
    )?;
    pass.memory_barrier()?;
    kda_output(ctx, pass, p, x, out, rows)
}

/// The prepare stage of a prompt run (and the edge copy before it):
/// `cumulative` leaves the chunk's cumulative log decay in place of the
/// decay (the chunked form's input; `chunk` must then be KDA_CHUNK).
#[allow(clippy::too_many_arguments)]
fn kda_prepare(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &KdaParams<'_>,
    x: &KdaInputs<'_>,
    conv_state: &Tensor,
    rows: usize,
    snap: Option<(usize, &Tensor)>,
    chunk: usize,
    cumulative: bool,
) -> Result<()> {
    let proj = p.heads * 128;
    let (snap_row, snap_conv) = snap.unwrap_or((u32::MAX as usize, conv_state));
    let chunks = rows.div_ceil(chunk);
    let edges = match x.edges {
        Some(e) if chunks > 1 => {
            ensure!(e.numel() >= chunks * 9 * proj, "KDA edges");
            let k =
                ctx.pipeline("glm53_kda_prefill_edges", SOURCE, MslVersion::V3_1)?;
            pass.dispatch_at(
                &k,
                &[
                    x.q.binding(),
                    x.k.binding(),
                    x.v.binding(),
                    conv_state.binding(),
                    e.binding(),
                ],
                &[&u32_bytes(proj), &u32_bytes(chunk)],
                Grid::Threadgroups {
                    groups: ((9 * proj).div_ceil(256), chunks, 1),
                    threadgroup: (256, 1, 1),
                },
            )?;
            pass.memory_barrier()?;
            e
        }
        None if chunks > 1 => anyhow::bail!("KDA run of {rows} rows without edges"),
        _ => conv_state,
    };
    let name = if cumulative {
        "glm53_kda_chunk_prepare"
    } else {
        "glm53_kda_prefill_prepare"
    };
    let prepare = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &prepare,
        &[
            x.q.binding(),
            x.k.binding(),
            x.v.binding(),
            x.gate.binding(),
            p.q_conv.binding(),
            p.k_conv.binding(),
            p.v_conv.binding(),
            p.a_log.binding(),
            p.dt_bias.binding(),
            conv_state.binding(),
            snap_conv.binding(),
            edges.binding(),
        ],
        &[
            &u32_bytes(p.heads),
            &u32_bytes(rows),
            &f32_bytes(p.lower_bound),
            &u32_bytes(snap_row),
            &u32_bytes(chunk),
            &u32_bytes(usize::from(chunks > 1)),
        ],
        Grid::Threadgroups { groups: (p.heads, chunks, 1), threadgroup: (128, 1, 1) },
    )
}

/// The output stage: normalizes and gates `out` in place.
fn kda_output(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &KdaParams<'_>,
    x: &KdaInputs<'_>,
    out: &Tensor,
    rows: usize,
) -> Result<()> {
    let output = ctx.pipeline("glm53_kda_prefill_output", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &output,
        &[out.binding(), x.out_gate.binding(), p.o_norm.binding()],
        &[&u32_bytes(p.heads), &f32_bytes(p.eps)],
        Grid::Threadgroups { groups: (rows, p.heads, 1), threadgroup: (128, 1, 1) },
    )
}

/// [`kda`] with the token-serial recurrence, prompt runs' prepare stage
/// walking `chunk` tokens a threadgroup (tests compare chunkings bit for
/// bit).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kda_chunked(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &KdaParams<'_>,
    x: &KdaInputs<'_>,
    conv_state: &Tensor,
    state: &Tensor,
    out: &Tensor,
    rows: usize,
    snap: Option<KdaSnapshot<'_>>,
    chunk: usize,
) -> Result<()> {
    ensure!(chunk >= 3, "KDA prepare chunks of {chunk} tokens");
    let (snap_row, snap_state) = match &snap {
        Some(s) => {
            ensure!(s.row < rows, "KDA snapshot after row {} of {rows}", s.row);
            (s.row, s.state)
        }
        None => (u32::MAX as usize, state),
    };
    check_f32(&[x.q, x.k, x.v, x.gate, x.beta, x.out_gate, conv_state, state, out])?;
    let proj = p.heads * 128;
    ensure!(
        state.numel() >= p.heads * 128 * 128 && conv_state.numel() >= 9 * proj,
        "KDA state"
    );
    if rows == 1 && snap.is_none() {
        let pipeline = ctx.pipeline("glm53_kda_decode", SOURCE, MslVersion::V3_1)?;
        return pass.dispatch_at(
            &pipeline,
            &[
                x.q.binding(),
                x.k.binding(),
                x.v.binding(),
                x.gate.binding(),
                x.beta.binding(),
                x.out_gate.binding(),
                p.q_conv.binding(),
                p.k_conv.binding(),
                p.v_conv.binding(),
                p.a_log.binding(),
                p.dt_bias.binding(),
                p.o_norm.binding(),
                conv_state.binding(),
                state.binding(),
                out.binding(),
            ],
            &[&u32_bytes(p.heads), &f32_bytes(p.lower_bound), &f32_bytes(p.eps)],
            Grid::Threadgroups { groups: (1, p.heads, 1), threadgroup: (128, 1, 1) },
        );
    }
    let snap_prep = snap.as_ref().map(|s| (s.row, s.conv));
    kda_prepare(ctx, pass, p, x, conv_state, rows, snap_prep, chunk, false)?;
    pass.memory_barrier()?;
    let recurrence =
        ctx.pipeline("glm53_kda_prefill_recurrence", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &recurrence,
        &[
            x.q.binding(),
            x.k.binding(),
            x.v.binding(),
            x.gate.binding(),
            x.beta.binding(),
            state.binding(),
            out.binding(),
            snap_state.binding(),
        ],
        &[&u32_bytes(p.heads), &u32_bytes(rows), &u32_bytes(snap_row)],
        Grid::Threadgroups { groups: (p.heads, 32, 1), threadgroup: (128, 1, 1) },
    )?;
    pass.memory_barrier()?;
    kda_output(ctx, pass, p, x, out, rows)
}

/// Stores `rows` normalized latent rows (`rmsnorm(kv_raw, w)`, 512 wide) as
/// f16 at cache rows `pos0..`.
#[allow(clippy::too_many_arguments)]
pub fn kv_store(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    kv_raw: &Tensor,
    w: &Tensor,
    cache: &Tensor,
    pos0: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
    check_f32(&[kv_raw, w])?;
    ensure!(
        cache.dtype() == DType::F16 && cache.numel() >= (pos0 + rows) * 512,
        "latent cache"
    );
    let pipeline = ctx.pipeline("glm53_kv_store", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[kv_raw.binding(), w.binding(), cache.binding()],
        &[&u32_bytes(pos0), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (128, 1, 1) },
    )
}

/// The widest score row the attention kernels take (the latent mix keeps
/// its row list in threadgroup memory): the dense window, 2 051 rows,
/// rounded up; the sparse selection is as wide.
pub const DSA_MAX_SLOTS: usize = 2056;

/// The most chunks the latent mix splits its slots into; its
/// partial sums take `m * DSA_LORA_CHUNKS * 64 * 512` f32 (rows past that
/// size take one chunk).
pub const DSA_LORA_CHUNKS: usize = 8;

/// Which cache rows a DSA query attends.
#[derive(Clone, Copy)]
pub enum Rows<'t> {
    /// Causal and dense: row `i` of the batch sees `0..=pos0 + i`.
    Causal { pos0: usize },
    /// The first `count` entries of each row of `sel` (U32 `[m][s]`).
    Selected { sel: &'t Tensor, count: usize },
}

/// The scores stage of [`dsa_attend`] (`scores[i][h][s]`, -inf past each
/// row's count): on the matrix route when `mu` (its widths a nonzero
/// multiple of 8; a tile whose queries leave f16's range computes in f32),
/// else on the SIMT kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsa_scores(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qk_low: &Tensor,
    cache: &Tensor,
    rows: Rows<'_>,
    scores: &Tensor,
    m: usize,
    s: usize,
    scale: f32,
    mu: bool,
) -> Result<()> {
    check_f32(&[qk_low, scores])?;
    ensure!(scores.numel() >= m * 64 * s, "scores of {}", scores.numel());
    let (pos0, sel, has_sel, n_sel) = match rows {
        Rows::Causal { pos0 } => {
            ensure!(pos0 + m <= s, "causal rows past the scores' width {s}");
            (pos0, qk_low, 0usize, 0usize)
        }
        Rows::Selected { sel, count } => {
            ensure!(count <= s && sel.numel() >= m * s, "selection");
            (0, sel, 1, count)
        }
    };
    ensure!(s <= DSA_MAX_SLOTS, "attention over {s} slots");
    // The matrix route's weight loads are four wide: rows padded to 8, and
    // its first loads run before its slot loop (no empty rows).
    ensure!(
        !mu || (s >= 8 && s.is_multiple_of(8)),
        "prompt attention over {s} slots (a nonzero multiple of 8)"
    );
    let (name, groups, tg) = if mu {
        ("glm53_dsa_scores_mu", s.div_ceil(64), 128)
    } else {
        ("glm53_dsa_scores", s.div_ceil(32), 256)
    };
    let scores_k = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scores_k,
        &[qk_low.binding(), cache.binding(), sel.binding(), scores.binding()],
        &[
            &u32_bytes(s),
            &u32_bytes(pos0),
            &u32_bytes(has_sel),
            &u32_bytes(n_sel),
            &f32_bytes(scale),
        ],
        Grid::Threadgroups { groups: (groups, m, 1), threadgroup: (tg, 1, 1) },
    )
}

/// Absorbed DSA attention core over the latent cache: from `qk_low`
/// (`[m][64][512]`) to the mixed latent `lat` (`[m][64][512]`), with scratch
/// `scores` (`[m][64][s]`) and `denom` (`[m][64]`); `s` bounds every row's
/// count. More than [`GEMV_MAX_ROWS`] rows take the matrix route, which
/// needs `s` a nonzero multiple of 8.
#[allow(clippy::too_many_arguments)]
pub fn dsa_attend(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qk_low: &Tensor,
    cache: &Tensor,
    rows: Rows<'_>,
    scores: &Tensor,
    denom: &Tensor,
    lat: &Tensor,
    partial: &Tensor,
    m: usize,
    s: usize,
    scale: f32,
) -> Result<()> {
    check_f32(&[qk_low, scores, denom, lat, partial])?;
    ensure!(scores.numel() >= m * 64 * s, "scores of {}", scores.numel());
    let (pos0, sel, has_sel, n_sel) = match rows {
        Rows::Causal { pos0 } => {
            ensure!(pos0 + m <= s, "causal rows past the scores' width {s}");
            (pos0, qk_low, 0usize, 0usize)
        }
        Rows::Selected { sel, count } => {
            ensure!(count <= s && sel.numel() >= m * s, "selection");
            (0, sel, 1, count)
        }
    };
    ensure!(s <= DSA_MAX_SLOTS, "attention over {s} slots");
    // Prompt runs on the matrix units (glm53_dsa.comp); decode rows on the
    // SIMT kernels.
    // LILY_GLM53_DSA_F32=1 keeps prompt runs on the SIMT kernels (A/B of
    // the matrix route's precision).
    static F32_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let f32_only =
        *F32_ONLY.get_or_init(|| std::env::var_os("LILY_GLM53_DSA_F32").is_some());
    let mu = m > GEMV_MAX_ROWS && !f32_only;
    dsa_scores(ctx, pass, qk_low, cache, rows, scores, m, s, scale, mu)?;
    pass.memory_barrier()?;
    let softmax = ctx.pipeline("glm53_dsa_softmax", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &softmax,
        &[scores.binding(), denom.binding()],
        &[&u32_bytes(s)],
        Grid::Threadgroups { groups: (64, m, 1), threadgroup: (256, 1, 1) },
    )?;
    pass.memory_barrier()?;
    if mu {
        let lora = ctx.pipeline("glm53_dsa_lora_mu", SOURCE, MslVersion::V3_1)?;
        return pass.dispatch_at(
            &lora,
            &[
                scores.binding(),
                denom.binding(),
                cache.binding(),
                sel.binding(),
                lat.binding(),
            ],
            &[&u32_bytes(s), &u32_bytes(pos0), &u32_bytes(has_sel), &u32_bytes(n_sel)],
            Grid::Threadgroups { groups: (8, m, 1), threadgroup: (128, 1, 1) },
        );
    }
    // Slots split into chunks of at least 8 steps of 32 (at most
    // DSA_LORA_CHUNKS) when the rows are few enough for the partials.
    let steps = s.div_ceil(32);
    let chunks = if m * DSA_LORA_CHUNKS * 64 * 512 <= partial.numel() {
        (steps / 8).clamp(1, DSA_LORA_CHUNKS)
    } else {
        1
    };
    let chunk = steps.div_ceil(chunks) * 32;
    let lora = ctx.pipeline("glm53_dsa_lora", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &lora,
        &[
            scores.binding(),
            denom.binding(),
            cache.binding(),
            sel.binding(),
            lat.binding(),
            partial.binding(),
        ],
        &[
            &u32_bytes(s),
            &u32_bytes(pos0),
            &u32_bytes(has_sel),
            &u32_bytes(n_sel),
            &u32_bytes(chunks),
            &u32_bytes(chunk),
        ],
        Grid::Threadgroups { groups: (4, 8, m * chunks), threadgroup: (256, 1, 1) },
    )?;
    if chunks == 1 {
        return Ok(());
    }
    pass.memory_barrier()?;
    let reduce = ctx.pipeline("glm53_dsa_lora_reduce", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &reduce,
        &[partial.binding(), denom.binding(), lat.binding()],
        &[&u32_bytes(chunks)],
        Grid::Threadgroups {
            groups: (64 * 512 / 4 / 256, m, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// The pooled indexer's tensors of one layer.
pub struct PoolParams<'a> {
    pub norm_w: &'a Tensor,
    pub norm_b: &'a Tensor,
    /// BF16 `[4][128]`, as a byte view.
    pub ape: &'a Tensor,
}

/// Feeds `n_tokens` raw indexer key/gate rows at positions `pos0..` into
/// the pooled key cache (f16 `[pools][128]`) and the tail buffers.
#[allow(clippy::too_many_arguments)]
pub fn indexer_pool_update(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &PoolParams<'_>,
    raw_k: &Tensor,
    gate: &Tensor,
    pooled: &Tensor,
    tail_k: &Tensor,
    tail_g: &Tensor,
    pos0: usize,
    n_tokens: usize,
) -> Result<()> {
    check_f32(&[raw_k, gate, tail_k, tail_g])?;
    ensure!(pooled.dtype() == DType::F16, "pooled cache dtype");
    ensure!(pooled.numel() >= (pos0 + n_tokens) / 4 * 128, "pooled cache too small");
    let pipeline =
        ctx.pipeline("glm53_indexer_pool_update", SOURCE, MslVersion::V3_1)?;
    // As ds4 does: a leading partial pool, which reads the tail buffers, is
    // its own dispatch before the rest, which may write them (the pass
    // orders its dispatches).
    let lead = if pos0 % 4 == 0 { 0 } else { (4 - pos0 % 4).min(n_tokens) };
    let mut done = 0;
    for chunk in [lead, n_tokens - lead] {
        if chunk == 0 {
            continue;
        }
        let at = pos0 + done;
        let k = raw_k.view(done * 128, &[chunk * 128])?;
        let g = gate.view(done * 128, &[chunk * 128])?;
        pass.dispatch_at(
            &pipeline,
            &[
                k.binding(),
                g.binding(),
                p.norm_w.binding(),
                p.norm_b.binding(),
                p.ape.binding(),
                pooled.binding(),
                tail_k.binding(),
                tail_g.binding(),
            ],
            &[&u32_bytes(at), &u32_bytes(chunk), &f32_bytes(1e-6)],
            Grid::Threadgroups {
                groups: (chunk.div_ceil(4), 1, 1),
                threadgroup: (128, 1, 1),
            },
        )?;
        done += chunk;
        pass.memory_barrier()?;
    }
    Ok(())
}

/// Indexer scores of the complete pools for `m` query rows at positions
/// `pos0..`: `scores[i][p]` (`[m][pools_cap]`, -inf past the row's pools)
/// from `iq` (`[m][32][128]`) and the head weights `w` (`[m][32]`).
#[allow(clippy::too_many_arguments)]
pub fn indexer_scores(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    iq: &Tensor,
    w: &Tensor,
    pooled: &Tensor,
    scores: &Tensor,
    m: usize,
    pools_cap: usize,
    pos0: usize,
    scale: f32,
) -> Result<()> {
    check_f32(&[iq, w, scores])?;
    ensure!((pos0 + m) / 4 <= pools_cap, "pool scores of {pools_cap} pools");
    ensure!(scores.numel() >= m * pools_cap, "scores of {}", scores.numel());
    let pipeline = ctx.pipeline("glm53_indexer_scores", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[iq.binding(), w.binding(), pooled.binding(), scores.binding()],
        &[&u32_bytes(pools_cap), &u32_bytes(pos0), &f32_bytes(scale)],
        Grid::Threadgroups {
            groups: (pools_cap.div_ceil(8), m, 1),
            threadgroup: (32, 1, 1),
        },
    )
}

/// The rows a DSA query attends past the dense limit: the token rows of its
/// `k` best pools (ascending) and of its incomplete pool, the rest of the
/// `s` slots marked unused; `count[i]` gets the row count.
#[allow(clippy::too_many_arguments)]
pub fn indexer_topk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    scores: &Tensor,
    sel: &Tensor,
    count: &Tensor,
    m: usize,
    pools_cap: usize,
    s: usize,
    pos0: usize,
    k: usize,
) -> Result<()> {
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * s, "selection");
    ensure!(4 * k + 3 <= s, "selection rows {s} for top {k} pools");
    let pipeline = ctx.pipeline("glm53_indexer_topk", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[scores.binding(), sel.binding(), count.binding()],
        &[&u32_bytes(pools_cap), &u32_bytes(s), &u32_bytes(pos0), &u32_bytes(k)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/glm53.rs"]
mod tests;
