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
    for ty in
        [GgmlType::Q8_0, GgmlType::F16, GgmlType::F32, GgmlType::Bf16, GgmlType::Q4K]
    {
        // The model's widths: hidden 2560, hc up 320, shared down 640, GDN
        // out 6144, eh_proj halves 2560 of 5120.
        for (cols, rows) in
            [(2560usize, 72usize), (320, 40), (640, 24), (6144, 9), (10240, 37)]
        {
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
    let e: Vec<f32> = (0..rows)
        .map(|n| dot(&full[n * 2 * cols + cols..(n + 1) * 2 * cols], &x))
        .collect();
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
    let tid =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[ids.len()], DType::U32)
            .unwrap();
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
                act[(i * used + s) * ff + r] =
                    round_bf16(gv / (1.0 + (-gv).exp()) * uv);
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
    for (gty, dty, d_cols) in [
        (GgmlType::Iq2Xxs, GgmlType::Q2K, 768usize),
        (GgmlType::Q4K, GgmlType::Mxfp4, 640),
    ] {
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
        let (cg, cu, cd) =
            (decode(gty, &raw_g), decode(gty, &raw_u), decode(dty, &raw_d));
        for (m, used) in [(1usize, 4usize), (3, 4), (2, 5)] {
            let (x, tx) = bf16_input(&ctx, &mut rng, m * h);
            let sel: Vec<u32> =
                (0..m * used).map(|_| rng.gen_range(0..experts as u32)).collect();
            let scores: Vec<f32> =
                (0..m * used).map(|_| rng.gen_range(0.0..0.5)).collect();
            let (so, tso) = bf16_input(&ctx, &mut rng, m * h);
            let sg: Vec<f32> = (0..m).map(|_| rng.gen_range(-2.0..2.0)).collect();
            let tsel = Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&sel),
                &[sel.len()],
                DType::U32,
            )
            .unwrap();
            let tsc = Tensor::from_f32(&ctx, &scores, &[scores.len()]).unwrap();
            let tsg = Tensor::from_f32(&ctx, &sg, &[m]).unwrap();
            let act = Tensor::zeros(&ctx, &[m * used * ff], DType::BF16).unwrap();
            let out = Tensor::zeros(&ctx, &[m * h], DType::BF16).unwrap();
            for with_shared in [false, true] {
                run(&ctx, |p| {
                    moe_gate_up(&ctx, p, &wg, &wu, ff, &tx, &tsel, &act, m, used)
                        .unwrap();
                    p.memory_barrier().unwrap();
                    let shared =
                        with_shared.then_some(SharedAdd { out: &tso, gate: &tsg });
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

/// The grouped GEMMs give the gathers' output up to their bfloat operands,
/// with the padded down rows (garbage in the padding proves it is unread).
#[test]
fn grouped_experts_match_the_gathers() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(43);
    let (h, ff, d_cols, experts, used) =
        (2560usize, 640usize, 768usize, 12usize, 10usize);
    let raw_g = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * h);
    let raw_u = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * h);
    let raw_d = random_blocks(&mut rng, GgmlType::Q2K, experts * h * d_cols);
    let wg =
        QWeight::from_bytes(&ctx, &raw_g, GgmlType::Iq2Xxs, h, experts * ff).unwrap();
    let wu =
        QWeight::from_bytes(&ctx, &raw_u, GgmlType::Iq2Xxs, h, experts * ff).unwrap();
    let wd =
        QWeight::from_bytes(&ctx, &raw_d, GgmlType::Q2K, d_cols, experts * h).unwrap();
    for m in [5usize, 40, 130] {
        let (_, tx) = bf16_input(&ctx, &mut rng, m * h);
        // Distinct experts per row, as the router picks them.
        let sel: Vec<u32> = (0..m)
            .flat_map(|_| {
                let mut e: Vec<u32> = (0..experts as u32).collect();
                for i in 0..used {
                    let j = rng.gen_range(i..experts);
                    e.swap(i, j);
                }
                e.truncate(used);
                e
            })
            .collect();
        let scores: Vec<f32> = (0..m * used).map(|_| rng.gen_range(0.0..0.2)).collect();
        let (_, tso) = bf16_input(&ctx, &mut rng, m * h);
        let sg: Vec<f32> = (0..m).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let tsel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[sel.len()],
            DType::U32,
        )
        .unwrap();
        let tsc = Tensor::from_f32(&ctx, &scores, &[scores.len()]).unwrap();
        let tsg = Tensor::from_f32_as_bf16(&ctx, &sg, &[m]).unwrap();
        let act = Tensor::zeros(&ctx, &[m * used * ff], DType::BF16).unwrap();
        let want = Tensor::zeros(&ctx, &[m * h], DType::BF16).unwrap();
        let got = Tensor::zeros(&ctx, &[m * h], DType::BF16).unwrap();
        let sc = MoeGroupScratch::new(&ctx, m * used, experts, d_cols, h).unwrap();
        run(&ctx, |p| {
            moe_gate_up(&ctx, p, &wg, &wu, ff, &tx, &tsel, &act, m, used).unwrap();
            p.memory_barrier().unwrap();
            moe_down(
                &ctx,
                p,
                &wd,
                ff,
                h,
                &act,
                &tsel,
                &tsc,
                Some(SharedAdd { out: &tso, gate: &tsg }),
                &want,
                m,
                used,
            )
            .unwrap();
            moe_grouped(
                &ctx,
                p,
                &wg,
                &wu,
                &wd,
                ff,
                h,
                &tx,
                &tsel,
                &tsc,
                SharedAdd { out: &tso, gate: &tsg },
                &got,
                &sc,
                m,
                used,
                experts,
            )
            .unwrap();
        });
        let (w, g) = (want.to_f32().unwrap(), got.to_f32().unwrap());
        let scale = w.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let err = w.iter().zip(&g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let rms =
            (w.iter().zip(&g).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>()
                / w.len() as f64)
                .sqrt();
        eprintln!("m {m}: max |diff| {err} of max |value| {scale}, rms {rms:.3e}");
        assert!(err <= 0.02 * scale, "m {m}: {err} of {scale}");
    }
}

/// The fused F16 hyper-connection read against its arithmetic on the CPU, at
/// the model's shape (4 streams of 2560, rank 320, 4 inject rows), one row
/// (decode) and three (a verify pass).
#[test]
fn fused_hc_read_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(44);
    let (h, g, r, ni) = (2560usize, 4usize, 320usize, 4usize);
    let wide = g * h;
    let nw: Vec<f32> =
        (0..wide).map(|_| round_bf16(rng.gen_range(-0.3..0.3))).collect();
    let tn = Tensor::from_f32_as_bf16(&ctx, &nw, &[wide]).unwrap();
    let raw_d = random_blocks(&mut rng, GgmlType::F16, r * wide);
    let raw_i = random_blocks(&mut rng, GgmlType::F16, ni * wide);
    let raw_u = random_blocks(&mut rng, GgmlType::F16, wide * r);
    let wd = QWeight::from_bytes(&ctx, &raw_d, GgmlType::F16, wide, r).unwrap();
    let wi = QWeight::from_bytes(&ctx, &raw_i, GgmlType::F16, wide, ni).unwrap();
    let wu = QWeight::from_bytes(&ctx, &raw_u, GgmlType::F16, r, wide).unwrap();
    let (cd, ci, cu) = (
        decode(GgmlType::F16, &raw_d),
        decode(GgmlType::F16, &raw_i),
        decode(GgmlType::F16, &raw_u),
    );
    let eps = 1e-6f32;
    for m in [1usize, 3] {
        let (x, tx) = bf16_input(&ctx, &mut rng, m * wide);
        let od = Tensor::zeros(&ctx, &[m * r], DType::BF16).unwrap();
        let oi = Tensor::zeros(&ctx, &[m * ni], DType::BF16).unwrap();
        let inv = Tensor::zeros(&ctx, &[m * g], DType::F32).unwrap();
        let mixed = Tensor::zeros(&ctx, &[m * h], DType::BF16).unwrap();
        run(&ctx, |p| {
            hc_read_f16(
                &ctx,
                p,
                &tx,
                &tn,
                &wd,
                Some(&wi),
                &wu,
                &od,
                &oi,
                &inv,
                &mixed,
                h,
                g,
                eps,
                1.0,
                m,
            )
            .unwrap()
        });
        let (mut e_inv, mut e_down, mut e_inj, mut e_mix) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for row in 0..m {
            let x = &x[row * wide..(row + 1) * wide];
            let inv_c: Vec<f32> = (0..g)
                .map(|s| {
                    let ms = x[s * h..(s + 1) * h].iter().map(|v| v * v).sum::<f32>()
                        / h as f32;
                    1.0 / (ms + eps).sqrt()
                })
                .collect();
            let xn: Vec<f32> =
                (0..wide).map(|i| x[i] * inv_c[i / h] * (1.0 + nw[i])).collect();
            let down: Vec<f32> = (0..r)
                .map(|j| round_bf16(dot(&cd[j * wide..(j + 1) * wide], &xn)))
                .collect();
            let act: Vec<f32> = down
                .iter()
                .map(|&d| {
                    let d = d / g as f32;
                    round_bf16(d / (1.0 + (-d).exp()))
                })
                .collect();
            for c in 0..h {
                let mut mm = 0f32;
                for s in 0..g {
                    let at = s * h + c;
                    let up = round_bf16(dot(&cu[at * r..(at + 1) * r], &act));
                    mm += 1.0 / (1.0 + (-up).exp()) * round_bf16(xn[at]);
                }
                e_mix.push(round_bf16(mm / g as f32));
            }
            e_inj.extend(
                (0..ni).map(|j| round_bf16(dot(&ci[j * wide..(j + 1) * wide], &xn))),
            );
            e_down.extend(down);
            e_inv.extend(inv_c);
        }
        assert_close(&inv.to_f32().unwrap(), &e_inv, 1e-6, 1e-4);
        assert_close(&od.to_f32().unwrap(), &e_down, 1e-3, 1.6e-2);
        assert_close(&oi.to_f32().unwrap(), &e_inj, 1e-3, 1.6e-2);
        assert_close(&mixed.to_f32().unwrap(), &e_mix, 1e-3, 1.6e-2);
    }
}

/// Ids past their tables stay inside them: an embedding id past the
/// vocabulary reads the last row, and the router's sentinel expert id (a
/// row of NaN probabilities) reads the last expert, in the gathers and the
/// grouped GEMMs alike (no out-of-bounds address, no out-of-bounds count).
#[test]
fn out_of_range_ids_stay_inside_their_tables() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(45);
    let (cols, rows) = (2560usize, 8usize);
    let raw = random_blocks(&mut rng, GgmlType::Bf16, rows * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Bf16, cols, rows).unwrap();
    let ids = [u32::MAX, 8, 7];
    let tid =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[3], DType::U32).unwrap();
    let out = Tensor::zeros(&ctx, &[3, cols], DType::BF16).unwrap();
    run(&ctx, |p| embed_bf16(&ctx, p, &w, &tid, &out, 3).unwrap());
    let last = decode(GgmlType::Bf16, &raw)[7 * cols..].to_vec();
    let got = out.to_f32().unwrap();
    for r in 0..3 {
        assert_eq!(got[r * cols..(r + 1) * cols], last[..], "row {r}");
    }

    let (h, ff, d_cols, experts, used) =
        (2560usize, 640usize, 768usize, 4usize, 10usize);
    let raw_g = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * h);
    let raw_u = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * h);
    let raw_d = random_blocks(&mut rng, GgmlType::Q2K, experts * h * d_cols);
    let wg =
        QWeight::from_bytes(&ctx, &raw_g, GgmlType::Iq2Xxs, h, experts * ff).unwrap();
    let wu =
        QWeight::from_bytes(&ctx, &raw_u, GgmlType::Iq2Xxs, h, experts * ff).unwrap();
    let wd =
        QWeight::from_bytes(&ctx, &raw_d, GgmlType::Q2K, d_cols, experts * h).unwrap();
    let m = 20usize;
    let (_, tx) = bf16_input(&ctx, &mut rng, m * h);
    let (_, tso) = bf16_input(&ctx, &mut rng, m * h);
    let tsg = Tensor::from_f32_as_bf16(&ctx, &vec![0.5; m], &[m]).unwrap();
    let scores = vec![0.1f32; m * used];
    let tsc = Tensor::from_f32(&ctx, &scores, &[m * used]).unwrap();
    let outputs = |sel: &[u32]| -> (Vec<f32>, Vec<f32>) {
        let tsel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(sel),
            &[sel.len()],
            DType::U32,
        )
        .unwrap();
        let act = Tensor::zeros(&ctx, &[m * used * ff], DType::BF16).unwrap();
        let gathered = Tensor::zeros(&ctx, &[m * h], DType::BF16).unwrap();
        let grouped = Tensor::zeros(&ctx, &[m * h], DType::BF16).unwrap();
        let sc = MoeGroupScratch::new(&ctx, m * used, experts, d_cols, h).unwrap();
        run(&ctx, |p| {
            moe_gate_up(&ctx, p, &wg, &wu, ff, &tx, &tsel, &act, m, used).unwrap();
            p.memory_barrier().unwrap();
            let shared = Some(SharedAdd { out: &tso, gate: &tsg });
            moe_down(
                &ctx, p, &wd, ff, h, &act, &tsel, &tsc, shared, &gathered, m, used,
            )
            .unwrap();
            let shared = SharedAdd { out: &tso, gate: &tsg };
            moe_grouped(
                &ctx, p, &wg, &wu, &wd, ff, h, &tx, &tsel, &tsc, shared, &grouped, &sc,
                m, used, experts,
            )
            .unwrap();
        });
        (gathered.to_f32().unwrap(), grouped.to_f32().unwrap())
    };
    // Row 0 entirely the sentinel; the others valid.
    let sel: Vec<u32> = (0..m * used)
        .map(|i| if i < used { u32::MAX } else { (i % experts) as u32 })
        .collect();
    let clamped: Vec<u32> = sel.iter().map(|&e| e.min(experts as u32 - 1)).collect();
    let (a, b) = outputs(&sel);
    let (ca, cb) = outputs(&clamped);
    assert_eq!(a, ca, "gathers");
    assert_eq!(b, cb, "grouped");
}
