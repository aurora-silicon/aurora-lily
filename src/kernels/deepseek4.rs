//! DeepSeek V4 Flash kernels (`vulkan/deepseek4.comp`): the operations the
//! model adds to GLM 5.3's kernels (`kernels::glm53`: GEMV/GEMM over GGUF
//! blocks, routed experts, HC, norms). The CPU reference is
//! `crate::deepseek4::reference`.

use anyhow::{Result, ensure};

use crate::deepseek4::config::Rope;
use crate::kernels::glm53::QWeight;
use crate::kernels::{f32_bytes, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// GLSL only so far, as GLM's kernels.
const SOURCE: &str = "";

/// Raw window rows every layer attends.
pub const WINDOW: usize = 128;
/// Width of a KV row (one KV head).
pub const KV_DIM: usize = 512;

fn check(ts: &[&Tensor], dt: DType) -> Result<()> {
    for t in ts {
        ensure!(t.dtype() == dt, "DeepSeek kernel wants {dt:?}, got {:?}", t.dtype());
    }
    Ok(())
}

/// The RoPE table of positions `lo..lo + n` (row `p - lo`: the `n_rot / 2`
/// pairs (cos, sin), magnitude included), computed as the CPU reference
/// rotates (`reference::rope_tail`), so the GPU's rotation is the
/// reference's.
pub fn rope_table(rope: &Rope, lo: usize, n: usize) -> Vec<f32> {
    let half = rope.n_rot / 2;
    let mut t = Vec::with_capacity(n * half * 2);
    // Rotating the unit pairs (1, 0) gives (c, s) of each pair exactly as
    // the reference's arithmetic forms them.
    let mut x = vec![0f32; rope.n_rot];
    for p in lo..lo + n {
        for k in 0..half {
            x[2 * k] = 1.0;
            x[2 * k + 1] = 0.0;
        }
        crate::deepseek4::reference::rope_tail(&mut x, rope.n_rot, rope, p, false);
        t.extend_from_slice(&x);
    }
    t
}

/// Embeds `m` tokens (U32) from the F16 table into four streams each.
pub fn embed(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &QWeight,
    tokens: &Tensor,
    streams: &Tensor,
    m: usize,
) -> Result<()> {
    ensure!(
        table.ty == crate::gguf::GgmlType::F16,
        "embedding table of {:?}",
        table.ty
    );
    check(&[streams], DType::F32)?;
    let pipeline = ctx.pipeline("ds4_embed_f16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.t.binding(), tokens.binding(), streams.binding()],
        &[&u32_bytes(table.cols)],
        Grid::Threads { grid: (table.cols, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// What [`head_norm_rope`] does besides the rotation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HeadOp {
    /// RMSNorm without weight, then RoPE (the queries).
    NormRope,
    /// RoPE only.
    Rope,
    /// The inverse rotation (the attention output).
    InverseRope,
}

/// Per head of `x` (`[m][heads][dim]`, `dim <= 512`), in place: `op`, the
/// rotation from `table` rows `t0..t0 + m`.
#[allow(clippy::too_many_arguments)]
pub fn head_norm_rope(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    table: &Tensor,
    t0: usize,
    m: usize,
    heads: usize,
    dim: usize,
    n_rot: usize,
    eps: f32,
    op: HeadOp,
) -> Result<()> {
    check(&[x, table], DType::F32)?;
    ensure!(dim <= 512 && dim.is_multiple_of(64) && n_rot <= dim, "heads of {dim}");
    ensure!(x.numel() >= m * heads * dim, "head tensor of {}", x.numel());
    ensure!(table.numel() >= (t0 + m) * n_rot, "RoPE table of {}", table.numel());
    let pipeline = ctx.pipeline("ds4_head_norm_rope", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), table.binding()],
        &[
            &u32_bytes(dim),
            &u32_bytes(n_rot),
            &f32_bytes(eps),
            &u32_bytes(usize::from(op == HeadOp::NormRope)),
            &u32_bytes(usize::from(op == HeadOp::InverseRope)),
            &u32_bytes(t0),
        ],
        Grid::Threadgroups { groups: (heads, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// The raw KV rows of a step (`kv`, `[m][512]`, normalized): RoPE of the
/// tail from `table` rows `t0..`, FP8 round trip of the rest (in place),
/// and the f16 copy into `kv_run`.
pub fn kv_store(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    kv: &Tensor,
    kv_run: &Tensor,
    table: &Tensor,
    t0: usize,
    m: usize,
) -> Result<()> {
    check(&[kv, table], DType::F32)?;
    check(&[kv_run], DType::F16)?;
    ensure!(kv.numel() >= m * KV_DIM && kv_run.numel() >= m * KV_DIM, "kv rows");
    ensure!(table.numel() >= (t0 + m) * 64, "RoPE table");
    let pipeline = ctx.pipeline("ds4_kv_store", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[kv.binding(), kv_run.binding(), table.binding()],
        &[&u32_bytes(t0)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// The step's last (at most [`WINDOW`]) raw rows into the layer's ring
/// (slot = position % 128), after the attention has read the ring.
pub fn ring_store(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    kv_run: &Tensor,
    ring: &Tensor,
    pos0: usize,
    m: usize,
) -> Result<()> {
    check(&[kv_run, ring], DType::F16)?;
    ensure!(ring.numel() >= WINDOW * KV_DIM, "ring of {}", ring.numel());
    let j0 = m.saturating_sub(WINDOW);
    let pipeline = ctx.pipeline("ds4_ring_store", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[kv_run.binding(), ring.binding()],
        &[&u32_bytes(pos0), &u32_bytes(j0)],
        Grid::Threads { grid: (KV_DIM, m - j0, 1), threadgroup: (256, 1, 1) },
    )
}

/// One compressor of a layer on the GPU: its weights' views and its ring of
/// the last `lanes * ratio` positions' projections.
pub struct CompressorGpu<'a> {
    pub ape: &'a Tensor,
    pub norm: &'a Tensor,
    pub ring_kv: &'a Tensor,
    pub ring_sc: &'a Tensor,
    pub dim: usize,
    pub ratio: usize,
    /// Whether its rows get the indexer's simulation (else FP8).
    pub indexer: bool,
}

impl CompressorGpu<'_> {
    pub fn lanes(&self) -> usize {
        if self.ratio == 4 { 2 } else { 1 }
    }

    pub fn width(&self) -> usize {
        self.lanes() * self.dim
    }
}

/// The window ends in positions `pos0..pos0 + m` of a compressor of ratio
/// `r`: the first and the count.
pub fn window_ends(r: usize, pos0: usize, m: usize) -> (usize, usize) {
    let first = (pos0 + 1).next_multiple_of(r) - 1;
    let n = if first < pos0 + m { (pos0 + m - 1 - first) / r + 1 } else { 0 };
    (first, n)
}

/// The compressed rows of the windows ending in this step, into `cache`
/// (row `(e + 1) / ratio - 1` for window end `e`), from the step's
/// projections `kv`, `sc` (`[m][width]`, scores without the position bias)
/// and the ring; then the ring updated with the step's last positions.
/// `table` holds the layer's (YaRN) RoPE of positions from `table_lo`.
/// Two dependency levels: [`comp_pool`], then [`comp_ring`].
#[allow(clippy::too_many_arguments)]
pub fn compress(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    c: &CompressorGpu<'_>,
    kv: &Tensor,
    sc: &Tensor,
    cache: &Tensor,
    table: &Tensor,
    table_lo: usize,
    pos0: usize,
    m: usize,
    eps: f32,
) -> Result<()> {
    comp_pool(ctx, pass, c, kv, sc, cache, table, table_lo, pos0, m, eps)?;
    pass.memory_barrier()?;
    comp_ring(ctx, pass, c, kv, sc, pos0, m)
}

/// The pool of [`compress`]: the windows ending in this step into `cache`
/// (nothing when none ends).
#[allow(clippy::too_many_arguments)]
pub fn comp_pool(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    c: &CompressorGpu<'_>,
    kv: &Tensor,
    sc: &Tensor,
    cache: &Tensor,
    table: &Tensor,
    table_lo: usize,
    pos0: usize,
    m: usize,
    eps: f32,
) -> Result<()> {
    check(&[kv, sc, c.ring_kv, c.ring_sc, c.norm, table], DType::F32)?;
    check(&[cache], DType::F16)?;
    ensure!(c.dim == 512 || (c.dim == 128 && c.indexer), "compressor of {}", c.dim);
    let w = c.width();
    let ring = c.lanes() * c.ratio;
    ensure!(kv.numel() >= m * w && sc.numel() >= m * w, "compressor projections");
    ensure!(c.ring_kv.numel() >= ring * w && c.ring_sc.numel() >= ring * w, "ring");
    let (e0, n) = window_ends(c.ratio, pos0, m);
    if n == 0 {
        return Ok(());
    }
    ensure!(
        cache.numel() >= (e0 + 1 + (n - 1) * c.ratio) / c.ratio * c.dim,
        "compressed cache of {} values",
        cache.numel()
    );
    ensure!(e0 + 1 >= c.ratio + table_lo, "RoPE table starts after the window");
    ensure!(
        table.numel() >= (e0 + (n - 1) * c.ratio + 2 - c.ratio - table_lo) * 64,
        "RoPE table"
    );
    let pipeline = ctx.pipeline("ds4_comp_pool", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            kv.binding(),
            sc.binding(),
            c.ape.binding(),
            c.ring_kv.binding(),
            c.ring_sc.binding(),
            c.norm.binding(),
            table.binding(),
            cache.binding(),
        ],
        &[
            &u32_bytes(c.dim),
            &u32_bytes(c.ratio),
            &u32_bytes(pos0),
            &u32_bytes(e0),
            &u32_bytes(table_lo),
            &u32_bytes(usize::from(c.indexer)),
            &f32_bytes(eps),
        ],
        Grid::Threadgroups { groups: (n, 1, 1), threadgroup: (c.dim, 1, 1) },
    )
}

/// The ring of [`compress`]: the step's last `lanes * ratio` positions'
/// projections (raw scores: the pool adds the position bias), after the
/// pool has read the ring.
pub fn comp_ring(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    c: &CompressorGpu<'_>,
    kv: &Tensor,
    sc: &Tensor,
    pos0: usize,
    m: usize,
) -> Result<()> {
    let w = c.width();
    let ring = c.lanes() * c.ratio;
    let j0 = m.saturating_sub(ring);
    let pipeline = ctx.pipeline("ds4_comp_ring", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[kv.binding(), sc.binding(), c.ring_kv.binding(), c.ring_sc.binding()],
        &[&u32_bytes(w), &u32_bytes(ring), &u32_bytes(pos0), &u32_bytes(j0)],
        Grid::Threads { grid: (w, m - j0, 1), threadgroup: (256, 1, 1) },
    )
}

/// Indexer queries (`[m][heads][128]`): RoPE of each head's tail from
/// `table` rows `t0..`, then the Hadamard + FP4 simulation, in place.
#[allow(clippy::too_many_arguments)]
pub fn indexer_q(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    table: &Tensor,
    t0: usize,
    m: usize,
    heads: usize,
) -> Result<()> {
    check(&[q, table], DType::F32)?;
    ensure!(q.numel() >= m * heads * 128, "indexer queries");
    let pipeline = ctx.pipeline("ds4_indexer_q", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[q.binding(), table.binding()],
        &[&u32_bytes(t0)],
        Grid::Threadgroups { groups: (heads, m, 1), threadgroup: (128, 1, 1) },
    )
}

/// Indexer scores of each step row over its compressed rows (`[m][cap]`).
#[allow(clippy::too_many_arguments)]
pub fn indexer_scores(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    w: &Tensor,
    keys: &Tensor,
    scores: &Tensor,
    m: usize,
    heads: usize,
    cap: usize,
    pos0: usize,
    scale: f32,
) -> Result<()> {
    check(&[q, w, scores], DType::F32)?;
    check(&[keys], DType::F16)?;
    let n = (pos0 + m) / 4;
    ensure!(n <= cap && scores.numel() >= m * cap, "indexer scores of {cap}");
    ensure!(keys.numel() >= n * 128, "indexer keys");
    let pipeline = ctx.pipeline("ds4_indexer_scores", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[q.binding(), w.binding(), keys.binding(), scores.binding()],
        &[&u32_bytes(cap), &u32_bytes(pos0), &f32_bytes(scale), &u32_bytes(heads)],
        Grid::Threadgroups { groups: (n.div_ceil(8), m, 1), threadgroup: (256, 1, 1) },
    )
}

/// The `k` best compressed rows of each step row (ascending), all when it
/// has no more than `k`; `count[i]` the number.
#[allow(clippy::too_many_arguments)]
pub fn topk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    scores: &Tensor,
    sel: &Tensor,
    count: &Tensor,
    m: usize,
    cap: usize,
    pos0: usize,
    k: usize,
) -> Result<()> {
    check(&[scores], DType::F32)?;
    check(&[sel, count], DType::U32)?;
    ensure!(sel.numel() >= m * k && count.numel() >= m, "selection");
    let pipeline = ctx.pipeline("ds4_topk", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[scores.binding(), sel.binding(), count.binding()],
        &[&u32_bytes(cap), &u32_bytes(pos0), &u32_bytes(k)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// The compressed rows a step's attention reads.
pub enum CompRows<'t> {
    /// A layer without compression.
    None,
    /// Every row complete at the row's position (`(p + 1) / ratio`).
    All { cache: &'t Tensor, ratio: usize },
    /// The indexer's selection: `count[i]` rows `sel[i][..]` (`k` a row).
    Selected { cache: &'t Tensor, sel: &'t Tensor, count: &'t Tensor, k: usize },
}

/// Sink attention of every head of every step row over its raw window
/// (`kv_run` for the step's positions, `ring` before) and compressed rows.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    kv_run: &Tensor,
    ring: &Tensor,
    comp: CompRows<'_>,
    sinks: &Tensor,
    out: &Tensor,
    m: usize,
    heads: usize,
    pos0: usize,
) -> Result<()> {
    check(&[q, sinks, out], DType::F32)?;
    check(&[kv_run, ring], DType::F16)?;
    ensure!(
        q.numel() >= m * heads * KV_DIM && out.numel() >= m * heads * KV_DIM,
        "heads"
    );
    let (cache, sel, count, ratio, use_sel, s) = match comp {
        CompRows::None => (ring, None, None, 0, false, 0),
        CompRows::All { cache, ratio } => {
            ensure!(cache.numel() >= (pos0 + m) / ratio * KV_DIM, "compressed rows");
            (cache, None, None, ratio, false, 0)
        }
        CompRows::Selected { cache, sel, count, k } => {
            check(&[sel, count], DType::U32)?;
            (cache, Some(sel), Some(count), 4, true, k)
        }
    };
    check(&[cache], DType::F16)?;
    let pipeline = ctx.pipeline("ds4_attend", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            q.binding(),
            kv_run.binding(),
            ring.binding(),
            cache.binding(),
            sel.unwrap_or(q).binding(),
            count.unwrap_or(q).binding(),
            sinks.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(pos0),
            &u32_bytes(ratio),
            &u32_bytes(usize::from(use_sel)),
            &u32_bytes(s),
            &f32_bytes(1.0 / (KV_DIM as f32).sqrt()),
        ],
        Grid::Threadgroups { groups: (heads, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// The router: hash layers take each token's `table` row (`hash`: the I32
/// table), the others the top `k` of `sqrt(softplus(logit)) + bias`;
/// weights the selected scores normalized times `scale`.
#[allow(clippy::too_many_arguments)]
pub fn router(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    bias: Option<&Tensor>,
    hash: Option<&QWeight>,
    tokens: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    m: usize,
    n_expert: usize,
    k: usize,
    scale: f32,
) -> Result<()> {
    check(&[logits, weights], DType::F32)?;
    check(&[sel], DType::U32)?;
    ensure!(n_expert <= 256 && k <= 32, "router of {n_expert} experts, top {k}");
    ensure!(bias.is_some() != hash.is_some(), "router takes a bias or a hash table");
    if let Some(h) = hash {
        ensure!(h.ty == crate::gguf::GgmlType::I32 && h.cols == k, "hash table");
    }
    let pipeline = ctx.pipeline("ds4_router", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            logits.binding(),
            bias.unwrap_or(logits).binding(),
            hash.map_or(logits.binding(), |h| h.t.binding()),
            tokens.binding(),
            sel.binding(),
            weights.binding(),
        ],
        &[
            &u32_bytes(n_expert),
            &u32_bytes(k),
            &f32_bytes(scale),
            &u32_bytes(usize::from(hash.is_some())),
        ],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// The output head's HC collapse from the control projection `pre`
/// (`[m][4]`).
#[allow(clippy::too_many_arguments)]
pub fn out_collapse(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    pre: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    streams: &Tensor,
    out: &Tensor,
    m: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    check(&[pre, scale, base, streams, out], DType::F32)?;
    let pipeline = ctx.pipeline("ds4_out_collapse", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            pre.binding(),
            scale.binding(),
            base.binding(),
            streams.binding(),
            out.binding(),
        ],
        &[&u32_bytes(hidden), &f32_bytes(eps)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/deepseek4.rs"]
mod tests;
