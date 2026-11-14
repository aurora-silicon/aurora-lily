use half::{bf16, f16};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::gguf::dequantize;

fn ctx() -> MetalContext {
    MetalContext::new().expect("metal context")
}

fn put_f16(b: &mut [u8], at: usize, v: f32) {
    b[at..at + 2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
}

/// Random blocks of `ty` with sane scales (dot products well inside f32).
fn random_blocks(rng: &mut StdRng, ty: GgmlType, values: usize) -> Vec<u8> {
    let nb = values / ty.block_values();
    let bb = ty.block_bytes();
    let mut b: Vec<u8> = (0..nb * bb).map(|_| rng.r#gen()).collect();
    for blk in b.chunks_exact_mut(bb) {
        match ty {
            GgmlType::Q8_0 => put_f16(blk, 0, rng.gen_range(0.001..0.02)),
            GgmlType::Q4K => {
                put_f16(blk, 0, rng.gen_range(0.001..0.01));
                put_f16(blk, 2, rng.gen_range(0.001..0.01));
            }
            GgmlType::Q2K => {
                put_f16(blk, 80, rng.gen_range(0.001..0.02));
                put_f16(blk, 82, rng.gen_range(0.001..0.02));
            }
            GgmlType::Iq2Xxs => put_f16(blk, 0, rng.gen_range(0.001..0.01)),
            // Exponents around 2^-6.
            GgmlType::Mxfp4 => blk[0] = rng.gen_range(118..124),
            GgmlType::Bf16 => {
                let v = bf16::from_f32(rng.gen_range(-0.1..0.1));
                blk.copy_from_slice(&v.to_bits().to_le_bytes());
            }
            GgmlType::F16 => put_f16(blk, 0, rng.gen_range(-0.1..0.1)),
            GgmlType::F32 => {
                blk.copy_from_slice(&rng.gen_range(-0.1f32..0.1).to_le_bytes())
            }
            GgmlType::I32 => unreachable!(),
        }
    }
    b
}

fn decode(ty: GgmlType, raw: &[u8]) -> Vec<f32> {
    let mut w = vec![0f32; raw.len() / ty.block_bytes() * ty.block_values()];
    dequantize(ty, raw, &mut w).unwrap();
    w
}

/// Bfloat16 activations (as f32 values and as a GPU tensor).
fn bf16_input(ctx: &MetalContext, rng: &mut StdRng, n: usize) -> (Vec<f32>, Tensor) {
    let v: Vec<f32> =
        (0..n).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0)).to_f32()).collect();
    let t = Tensor::from_f32_as_bf16(ctx, &v, &[n]).unwrap();
    (v, t)
}

fn run(ctx: &MetalContext, f: impl FnOnce(&ComputePass<'_>)) {
    let pass = ctx.begin().expect("pass");
    f(&pass);
    pass.commit_wait().expect("commit");
}

fn dot(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() as f32
}

fn round_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

#[test]
fn gemv_sets_match_cpu_for_every_type_width_and_row_count() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(38);
    for ty in [GgmlType::Q8_0, GgmlType::F16, GgmlType::F32, GgmlType::Bf16, GgmlType::Q4K]
    {
        // The model's widths: hidden 2560, hc up 320, shared down 640, GDN
        // out 6144, eh_proj halves 2560 of 5120.
        for (cols, rows) in [(2560usize, 72usize), (320, 40), (640, 24), (6144, 9)] {
            if !cols.is_multiple_of(ty.block_values()) {
                continue;
            }
            let raw_a = random_blocks(&mut rng, ty, rows * cols);
            let raw_b = random_blocks(&mut rng, ty, 3 * cols);
            let wa = QWeight::from_bytes(&ctx, &raw_a, ty, cols, rows).unwrap();
            let wb = QWeight::from_bytes(&ctx, &raw_b, ty, cols, 3).unwrap();
            let (ca, cb) = (decode(ty, &raw_a), decode(ty, &raw_b));
            for m in 1..=GEMV_MAX_ROWS {
                let (x, tx) = bf16_input(&ctx, &mut rng, m * cols);
                let ya = Tensor::zeros(&ctx, &[m * rows], DType::BF16).unwrap();
                let yb = Tensor::zeros(&ctx, &[m * 3], DType::F32).unwrap();
                run(&ctx, |p| {
                    gemv_sets(&ctx, p, &tx, &[(&wa, &ya), (&wb, &yb)], m, cols, None)
                        .unwrap()
                });
                let expect = |c: &[f32], n_rows: usize| -> Vec<f32> {
                    let mut e = Vec::with_capacity(m * n_rows);
                    for i in 0..m {
                        let x = &x[i * cols..(i + 1) * cols];
                        for n in 0..n_rows {
                            e.push(dot(&c[n * cols..(n + 1) * cols], x));
                        }
                    }
                    e
                };
                let (ea, eb) = (expect(&ca, rows), expect(&cb, 3));
                // bfloat16 output: half an ulp of rounding plus f32 sums.
                assert_close(&ya.to_f32().unwrap(), &ea, 1e-3, 8e-3);
                assert_close(&yb.to_f32().unwrap(), &eb, 1e-4, 1e-4);
            }
        }
    }
}

/// A view of every row's second half (eh_proj's `fc_hidden` columns): rows
/// `row_bytes` of the full width apart.
#[test]
fn gemv_reads_a_column_view_of_wider_rows() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(39);
    let (cols, rows) = (512usize, 33usize);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, rows * 2 * cols);
    let full = decode(GgmlType::Q8_0, &raw);
    let t = Tensor::from_bytes(&ctx, &raw, &[raw.len()], DType::I8).unwrap();
    let half_bytes = GgmlType::Q8_0.row_bytes(cols).unwrap();
    let w = QWeight {
        t: t.view(half_bytes, &[raw.len() - half_bytes]).unwrap(),
        ty: GgmlType::Q8_0,
        cols,
        rows,
        row_bytes: 2 * half_bytes,
    };
    let (x, tx) = bf16_input(&ctx, &mut rng, cols);
    let y = Tensor::zeros(&ctx, &[rows], DType::F32).unwrap();
    run(&ctx, |p| gemv(&ctx, p, &w, &tx, &y, 1).unwrap());
    let e: Vec<f32> =
        (0..rows).map(|n| dot(&full[n * 2 * cols + cols..(n + 1) * 2 * cols], &x)).collect();
    assert_close(&y.to_f32().unwrap(), &e, 1e-4, 1e-4);
}

#[test]
fn dequant_rounds_every_type_to_bf16() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(40);
    for ty in [GgmlType::Q8_0, GgmlType::F16, GgmlType::F32, GgmlType::Bf16] {
        let (cols, rows) = (320usize, 17usize);
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let out = Tensor::zeros(&ctx, &[rows, cols], DType::BF16).unwrap();
        run(&ctx, |p| dequant_bf16(&ctx, p, &w, &out).unwrap());
        let e: Vec<f32> = decode(ty, &raw).into_iter().map(round_bf16).collect();
        assert_eq!(out.to_f32().unwrap(), e, "{ty:?}");
    }
}

#[test]
fn embed_gathers_bf16_rows() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(41);
    let (cols, rows) = (2560usize, 50usize);
    let raw = random_blocks(&mut rng, GgmlType::Bf16, rows * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Bf16, cols, rows).unwrap();
    let ids: Vec<u32> = vec![7, 0, 49, 7];
    let tid = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[ids.len()], DType::U32).unwrap();
    let out = Tensor::zeros(&ctx, &[ids.len(), cols], DType::BF16).unwrap();
    run(&ctx, |p| embed_bf16(&ctx, p, &w, &tid, &out, ids.len()).unwrap());
    let table = decode(GgmlType::Bf16, &raw);
    let e: Vec<f32> = ids
        .iter()
        .flat_map(|&i| table[i as usize * cols..(i as usize + 1) * cols].to_vec())
        .collect();
    assert_eq!(out.to_f32().unwrap(), e);
}

/// The CPU routed experts of `m` rows: gate/up/SwiGLU then the combine, in
/// the kernels' rounding (bfloat16 g, u, act, per-slot down sums, output).
#[allow(clippy::too_many_arguments)]
fn cpu_moe(
    g: &[f32],
    u: &[f32],
    d: &[f32],
    d_cols: usize,
    x: &[f32],
    sel: &[u32],
    scores: &[f32],
    shared: Option<(&[f32], &[f32])>,
    (m, used, ff, h): (usize, usize, usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    let mut act = vec![0f32; m * used * ff];
    for i in 0..m {
        let xi = &x[i * h..(i + 1) * h];
        for s in 0..used {
            let e = sel[i * used + s] as usize;
            for r in 0..ff {
                let row = (e * ff + r) * h;
                let gv = round_bf16(dot(&g[row..row + h], xi));
                let uv = round_bf16(dot(&u[row..row + h], xi));
                act[(i * used + s) * ff + r] = round_bf16(gv / (1.0 + (-gv).exp()) * uv);
            }
        }
    }
    let mut out = vec![0f32; m * h];
    for i in 0..m {
        for r in 0..h {
            let mut c = 0f32;
            for s in 0..used {
                let slot = i * used + s;
                let e = sel[slot] as usize;
                let row = (e * h + r) * d_cols;
                let v = dot(&d[row..row + ff], &act[slot * ff..(slot + 1) * ff]);
                c += scores[slot] * round_bf16(v);
            }
            if let Some((so, sg)) = shared {
                c += 1.0 / (1.0 + (-sg[i]).exp()) * so[i * h + r];
            }
            out[i * h + r] = round_bf16(c);
        }
    }
    (act, out)
}

#[test]
fn routed_experts_match_cpu_with_padded_down_rows() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(42);
    // Qwen's shapes at fewer experts: hidden 2560, FF 640, down rows padded
    // to 768 (Q2_K) as ds4's file has them; the draft head's Q4_K / MXFP4.
    let (h, ff, experts) = (2560usize, 640usize, 6usize);
    for (gty, dty, d_cols) in
        [(GgmlType::Iq2Xxs, GgmlType::Q2K, 768usize), (GgmlType::Q4K, GgmlType::Mxfp4, 640)]
    {
        let raw_g = random_blocks(&mut rng, gty, experts * ff * h);
        let raw_u = random_blocks(&mut rng, gty, experts * ff * h);
        let mut raw_d = random_blocks(&mut rng, dty, experts * h * d_cols);
        if d_cols > ff {
            // The padding of ds4's rows holds zeros; the kernel must not read
            // it either way: garbage there proves it.
            let _ = &mut raw_d;
        }
        let wg = QWeight::from_bytes(&ctx, &raw_g, gty, h, experts * ff).unwrap();
        let wu = QWeight::from_bytes(&ctx, &raw_u, gty, h, experts * ff).unwrap();
        let wd = QWeight::from_bytes(&ctx, &raw_d, dty, d_cols, experts * h).unwrap();
        let (cg, cu, cd) = (decode(gty, &raw_g), decode(gty, &raw_u), decode(dty, &raw_d));
        for (m, used) in [(1usize, 4usize), (3, 4), (2, 5)] {
            let (x, tx) = bf16_input(&ctx, &mut rng, m * h);
            let sel: Vec<u32> =
                (0..m * used).map(|_| rng.gen_range(0..experts as u32)).collect();
            let scores: Vec<f32> = (0..m * used).map(|_| rng.gen_range(0.0..0.5)).collect();
            let (so, tso) = bf16_input(&ctx, &mut rng, m * h);
            let sg: Vec<f32> = (0..m).map(|_| rng.gen_range(-2.0..2.0)).collect();
            let tsel = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32).unwrap();
            let tsc = Tensor::from_f32(&ctx, &scores, &[scores.len()]).unwrap();
            let tsg = Tensor::from_f32(&ctx, &sg, &[m]).unwrap();
            let act = Tensor::zeros(&ctx, &[m * used * ff], DType::BF16).unwrap();
            let out = Tensor::zeros(&ctx, &[m * h], DType::BF16).unwrap();
            for with_shared in [false, true] {
                run(&ctx, |p| {
                    moe_gate_up(&ctx, p, &wg, &wu, ff, &tx, &tsel, &act, m, used).unwrap();
                    p.memory_barrier().unwrap();
                    let shared = with_shared.then(|| SharedAdd { out: &tso, gate: &tsg });
                    moe_down(
                        &ctx, p, &wd, ff, h, &act, &tsel, &tsc, shared, &out, m, used,
                    )
                    .unwrap();
                });
                let (eact, eout) = cpu_moe(
                    &cg,
                    &cu,
                    &cd,
                    d_cols,
                    &x,
                    &sel,
                    &scores,
                    with_shared.then_some((so.as_slice(), sg.as_slice())),
                    (m, used, ff, h),
                );
                assert_close(&act.to_f32().unwrap(), &eact, 2e-3, 1.6e-2);
                assert_close(&out.to_f32().unwrap(), &eout, 2e-3, 1.6e-2);
            }
        }
    }
}
