use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::gguf::dequantize;
use crate::glm53::reference as r;

fn ctx() -> MetalContext {
    MetalContext::new().expect("metal context")
}

fn f32s(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn put_f16(b: &mut [u8], at: usize, v: f32) {
    b[at..at + 2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
}

/// Random blocks of `ty` with sane scales: random code bytes, f16 scales
/// small enough that the dot products stay well inside f32.
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
            GgmlType::Bf16 => {
                let v = half::bf16::from_f32(rng.gen_range(-0.1..0.1));
                blk.copy_from_slice(&v.to_bits().to_le_bytes());
            }
            GgmlType::F32 => {
                blk.copy_from_slice(&rng.gen_range(-0.1f32..0.1).to_le_bytes())
            }
            GgmlType::F16 => {
                let v = half::f16::from_f32(rng.gen_range(-0.1..0.1));
                blk.copy_from_slice(&v.to_bits().to_le_bytes());
            }
            GgmlType::I32 | GgmlType::Mxfp4 => unreachable!(),
            GgmlType::Q4_0 | GgmlType::Iq4Nl | GgmlType::Iq2S => {
                put_f16(blk, 0, rng.gen_range(0.001..0.01))
            }
            GgmlType::Q5K => {
                put_f16(blk, 0, rng.gen_range(0.001..0.01));
                put_f16(blk, 2, rng.gen_range(0.001..0.01));
            }
            GgmlType::Q6K => put_f16(blk, 208, rng.gen_range(0.0001..0.001)),
        }
    }
    b
}

/// Gemma 4's block types through the GEMV (every row count) and the
/// matrix-unit GEMM, at Gemma's widths (hidden 2816, shared FF 2112, local
/// heads 4096).
#[test]
fn gemv_and_gemm_match_cpu_for_gemma4_types() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(44);
    for ty in
        [GgmlType::Q4_0, GgmlType::Q5K, GgmlType::Q6K, GgmlType::Iq4Nl, GgmlType::Iq2S]
    {
        for (cols, rows) in [(2816usize, 72usize), (2112, 40), (4096, 24)] {
            if !cols.is_multiple_of(ty.block_values()) {
                continue;
            }
            let raw = random_blocks(&mut rng, ty, rows * cols);
            let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
            let cpu = mat(ty, &raw, rows, cols);
            for m in 1..=GEMV_MAX_ROWS {
                let x = f32s(&mut rng, m * cols, -1.0, 1.0);
                let tx = tensor(&ctx, &x);
                let y = zeros(&ctx, m * rows);
                run(&ctx, |p| gemv(&ctx, p, &w, &tx, &y, m).unwrap());
                let expected: Vec<f32> =
                    (0..m).flat_map(|i| cpu.mv(&x[i * cols..(i + 1) * cols])).collect();
                assert_close(&y.to_f32().unwrap(), &expected, 1e-4, 1e-4);
            }
            if cols.is_multiple_of(64) {
                // The GEMM rounds both operands to bfloat: so does the
                // reference.
                let m = 37;
                let x = f32s(&mut rng, m * cols, -1.0, 1.0);
                let tx = tensor(&ctx, &x);
                let y = zeros(&ctx, m * rows);
                run(&ctx, |p| gemm(&ctx, p, &w, &tx, &y, m).unwrap());
                // Weights to nearest with ties away from zero (gm_round_w),
                // activations to nearest even.
                let bf = |v: f32| half::bf16::from_f32(v).to_f32();
                let bfw = |v: f32| f32::from_bits((v.to_bits() + 0x8000) & 0xffff_0000);
                let wb = r::Mat {
                    rows,
                    cols,
                    w: cpu.w.iter().map(|&v| bfw(v)).collect(),
                };
                let xb: Vec<f32> = x.iter().map(|&v| bf(v)).collect();
                let expected: Vec<f32> =
                    (0..m).flat_map(|i| wb.mv(&xb[i * cols..(i + 1) * cols])).collect();
                assert_close(&y.to_f32().unwrap(), &expected, 1e-3, 1e-3);
            }
        }
    }
}

fn decoded(ty: GgmlType, raw: &[u8], values: usize) -> r::Mat {
    let _ = values;
    let n = raw.len() / ty.block_bytes() * ty.block_values();
    let mut w = vec![0f32; n];
    dequantize(ty, raw, &mut w).unwrap();
    r::Mat { rows: 0, cols: 0, w }
}

fn mat(ty: GgmlType, raw: &[u8], rows: usize, cols: usize) -> r::Mat {
    let mut m = decoded(ty, raw, rows * cols);
    m.rows = rows;
    m.cols = cols;
    m
}

fn run(ctx: &MetalContext, f: impl FnOnce(&ComputePass<'_>)) {
    let pass = ctx.begin().expect("pass");
    f(&pass);
    pass.commit_wait().expect("commit");
}

fn tensor(ctx: &MetalContext, v: &[f32]) -> Tensor {
    Tensor::from_f32(ctx, v, &[v.len()]).expect("tensor")
}

fn zeros(ctx: &MetalContext, n: usize) -> Tensor {
    Tensor::zeros(ctx, &[n], DType::F32).expect("zeros")
}

#[test]
fn gemv_matches_cpu_for_every_type_and_row_count() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(53);
    for ty in [
        GgmlType::Q8_0,
        GgmlType::Q4K,
        GgmlType::Q2K,
        GgmlType::Iq2Xxs,
        GgmlType::Bf16,
        GgmlType::F32,
    ] {
        for (cols, rows) in [(4096usize, 72usize), (2048, 40), (1536, 24), (256, 100)] {
            if !cols.is_multiple_of(ty.block_values()) {
                continue;
            }
            let raw = random_blocks(&mut rng, ty, rows * cols);
            let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
            let cpu = mat(ty, &raw, rows, cols);
            for m in 1..=GEMV_MAX_ROWS {
                let x = f32s(&mut rng, m * cols, -1.0, 1.0);
                let tx = tensor(&ctx, &x);
                let ty_out = zeros(&ctx, m * rows);
                run(&ctx, |p| gemv(&ctx, p, &w, &tx, &ty_out, m).unwrap());
                let expected: Vec<f32> =
                    (0..m).flat_map(|i| cpu.mv(&x[i * cols..(i + 1) * cols])).collect();
                assert_close(&ty_out.to_f32().unwrap(), &expected, 1e-4, 1e-4);
            }
        }
    }
}

#[test]
fn gemv_grouped_matches_cpu() {
    // attn_k_b's shape: 64 heads x 512 rows of 256, input per head; Q5_K
    // at all 64 heads is past the two-rows-a-subgroup threshold (32768
    // rows), which grouped inputs must not take.
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(54);
    let (per, cols) = (512, 256);
    for (ty, heads) in [(GgmlType::Q8_0, 8), (GgmlType::Q5K, 64)] {
        let raw = random_blocks(&mut rng, ty, heads * per * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, heads * per).unwrap();
        let cpu = mat(ty, &raw, heads * per, cols);
        for m in [1, 3] {
            let x = f32s(&mut rng, m * heads * cols, -1.0, 1.0);
            let tx = tensor(&ctx, &x);
            let out = zeros(&ctx, m * heads * per);
            let layout = GemvLayout {
                x_group: per,
                x_row_stride: heads * cols,
                y_row_stride: heads * per,
            };
            run(&ctx, |p| gemv_with(&ctx, p, &w, &tx, &out, m, layout).unwrap());
            let expected: Vec<f32> = (0..m)
                .flat_map(|i| {
                    cpu.mv_grouped(&x[i * heads * cols..(i + 1) * heads * cols], per)
                })
                .collect();
            assert_close(&out.to_f32().unwrap(), &expected, 1e-4, 1e-4);
        }
    }
}

/// The routed gate/up over f16 activations (decode rows of DeepSeek V4)
/// against the f32-activation kernel: six rows of six slots, within f16's
/// rounding of the activations.
#[test]
fn moe_xh_matches_f32_kernel() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(81);
    let (experts, hidden, ff, used, m) = (12, 512, 256, 6, 6);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let gw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let uw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -2.0, 2.0));
    let sel: Vec<u32> = (0..m * used).map(|s| ((s * 7 + s / used) % experts) as u32).collect();
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
    let (mid, mid_h) = (zeros(&ctx, m * used * ff), zeros(&ctx, m * used * ff));
    let xh = Tensor::zeros(&ctx, &[m * hidden], DType::F16).unwrap();
    run(&ctx, |p| {
        moe_gate_up(&ctx, p, &gw, &uw, ff, &x, &tsel, &tw, &mid, m, used, 10.0).unwrap();
        crate::kernels::deepseek4::x_f16_iq2(&ctx, p, &x, &xh, m, hidden).unwrap();
        p.memory_barrier().unwrap();
        moe_gate_up_xh(&ctx, p, &gw, &uw, ff, &xh, &tsel, &tw, &mid_h, m, used, 10.0)
            .unwrap();
    });
    let (a, b) = (mid_h.to_f32().unwrap(), mid.to_f32().unwrap());
    let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        // f16 activations: a half-ulp (2^-12) per product, with cancellation.
        assert!((x - y).abs() <= 5e-3 * scale + 1e-3 * y.abs(), "mid {k}: {x} vs {y}");
    }
}

#[test]
fn moe_gate_up_and_down_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(55);
    let (experts, hidden, ff, used) = (12, 512, 256, 8);
    let g_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw = QWeight::from_bytes(&ctx, &g_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let uw = QWeight::from_bytes(&ctx, &u_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    let gc = mat(GgmlType::Iq2Xxs, &g_raw, experts * ff, hidden);
    let uc = mat(GgmlType::Iq2Xxs, &u_raw, experts * ff, hidden);
    let dc = mat(GgmlType::Q2K, &d_raw, experts * hidden, ff);
    let m = 2;
    let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
    let sel: Vec<u32> =
        (0..m * used).map(|s| ((s * 5 + s / used) % experts) as u32).collect();
    let wts = f32s(&mut rng, m * used, 0.1, 0.6);
    let tx = tensor(&ctx, &x);
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &wts);
    let mid = zeros(&ctx, m * used * ff);
    let out = zeros(&ctx, m * hidden);
    let limit = 0.05;
    run(&ctx, |p| {
        moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, limit)
            .unwrap();
        moe_down(&ctx, p, &dw, hidden, &mid, &tsel, None, &out, m, used).unwrap();
    });
    let mut emid = vec![0f32; m * used * ff];
    let mut eout = vec![0f32; m * hidden];
    for i in 0..m {
        let xi = &x[i * hidden..(i + 1) * hidden];
        for s in 0..used {
            let e = sel[i * used + s] as usize;
            for row in 0..ff {
                let g = r::dot(gc.row(e * ff + row), xi);
                let u = r::dot(uc.row(e * ff + row), xi);
                emid[(i * used + s) * ff + row] =
                    r::swiglu(g, u, limit) * wts[i * used + s];
            }
        }
        for row in 0..hidden {
            eout[i * hidden + row] = (0..used)
                .map(|s| {
                    let e = sel[i * used + s] as usize;
                    r::dot(
                        dc.row(e * hidden + row),
                        &emid[(i * used + s) * ff..(i * used + s + 1) * ff],
                    )
                })
                .sum();
        }
    }
    assert_close(&mid.to_f32().unwrap(), &emid, 1e-5, 1e-4);
    assert_close(&out.to_f32().unwrap(), &eout, 1e-5, 1e-3);
    // With the shared expert's output added in place: exactly routed + add.
    let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let both = tensor(&ctx, &shared);
    run(&ctx, |p| {
        moe_down(&ctx, p, &dw, hidden, &mid, &tsel, Some(&both), &both, m, used)
            .unwrap()
    });
    let routed = out.to_f32().unwrap();
    let expect: Vec<f32> = routed.iter().zip(&shared).map(|(a, b)| a + b).collect();
    assert_eq!(both.to_f32().unwrap(), expect);
}

/// The two-row expert kernels (gate/up: an expert both rows picked decoded once)
/// give each row exactly what the one-row kernels give it, for selections
/// sharing 0, 1, 3 and 8 experts in any order, with and without the added
/// shared-expert output.
#[test]
fn moe_two_rows_match_one_row_exactly() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(56);
    let (experts, hidden, ff, used) = (20, 512, 256, 8);
    let g_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw = QWeight::from_bytes(&ctx, &g_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let uw = QWeight::from_bytes(&ctx, &u_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    let x = f32s(&mut rng, 2 * hidden, -2.0, 2.0);
    let add = f32s(&mut rng, 2 * hidden, -1.0, 1.0);
    let wts = f32s(&mut rng, 2 * used, 0.1, 0.6);
    let row0: Vec<u32> = vec![3, 7, 0, 12, 19, 5, 9, 14];
    let cases: [Vec<u32>; 5] = [
        vec![1, 2, 4, 6, 8, 10, 11, 13],
        vec![1, 2, 4, 6, 8, 10, 11, 3],
        vec![14, 2, 12, 6, 8, 3, 11, 13],
        vec![14, 9, 5, 19, 12, 0, 7, 3],
        vec![3, 7, 0, 12, 19, 5, 9, 14],
    ];
    for (c, row1) in cases.iter().enumerate() {
        let sel: Vec<u32> = row0.iter().chain(row1).copied().collect();
        let u32t = |v: &[u32]| {
            Tensor::from_bytes(&ctx, bytemuck::cast_slice(v), &[v.len()], DType::U32)
                .unwrap()
        };
        let tsel = u32t(&sel);
        let tx = tensor(&ctx, &x);
        let tw = tensor(&ctx, &wts);
        for with_add in [false, true] {
            // Both rows at once.
            let mid = zeros(&ctx, 2 * used * ff);
            let out = tensor(&ctx, &add);
            run(&ctx, |p| {
                moe_gate_up(
                    &ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, 2, used, 0.05,
                )
                .unwrap();
                p.memory_barrier().unwrap();
                let a = with_add.then_some(&out);
                moe_down(&ctx, p, &dw, hidden, &mid, &tsel, a, &out, 2, used).unwrap();
            });
            // Each row alone.
            let mid_all = mid.to_f32().unwrap();
            let out_all = out.to_f32().unwrap();
            for i in 0..2 {
                let xi = tensor(&ctx, &x[i * hidden..(i + 1) * hidden]);
                let si = u32t(&sel[i * used..(i + 1) * used]);
                let wi = tensor(&ctx, &wts[i * used..(i + 1) * used]);
                let mi = zeros(&ctx, used * ff);
                let oi = tensor(&ctx, &add[i * hidden..(i + 1) * hidden]);
                run(&ctx, |p| {
                    moe_gate_up(
                        &ctx, p, &gw, &uw, ff, &xi, &si, &wi, &mi, 1, used, 0.05,
                    )
                    .unwrap();
                    p.memory_barrier().unwrap();
                    let a = with_add.then_some(&oi);
                    moe_down(&ctx, p, &dw, hidden, &mi, &si, a, &oi, 1, used).unwrap();
                });
                assert_eq!(
                    mi.to_f32().unwrap(),
                    &mid_all[i * used * ff..(i + 1) * used * ff],
                    "case {c} row {i}: mid"
                );
                assert_eq!(
                    oi.to_f32().unwrap(),
                    &out_all[i * hidden..(i + 1) * hidden],
                    "case {c} row {i} add {with_add}: out"
                );
            }
        }
    }
}

/// glm53_top_prob against the CPU softmax maximum.
#[test]
fn top_prob_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(58);
    for (n, spread) in [(154_880usize, 12.0f32), (1000, 1.0), (5, 40.0)] {
        let l = f32s(&mut rng, n, -spread, spread);
        let tl = tensor(&ctx, &l);
        let out = zeros(&ctx, 1);
        run(&ctx, |p| top_prob(&ctx, p, &tl, &out, n).unwrap());
        let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = l.iter().map(|&v| f64::from(v - mx).exp()).sum();
        let got = out.to_f32().unwrap()[0];
        assert!(
            (f64::from(got) - 1.0 / z).abs() <= 1e-5 / z,
            "n={n}: {got} vs {}",
            1.0 / z
        );
    }
}

#[test]
fn router_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(56);
    let (m, e, k) = (3, 288, 8);
    let logits = f32s(&mut rng, m * e, -4.0, 4.0);
    let bias = f32s(&mut rng, e, -0.05, 0.05);
    let (tl, tb) = (tensor(&ctx, &logits), tensor(&ctx, &bias));
    let sel = Tensor::zeros(&ctx, &[m * k], DType::U32).unwrap();
    let w = zeros(&ctx, m * k);
    run(&ctx, |p| router_topk(&ctx, p, &tl, &tb, &sel, &w, m, e, k, 2.5).unwrap());
    let gsel = sel.to_u32().unwrap();
    let gw = w.to_f32().unwrap();
    for i in 0..m {
        let (esel, ew) = r::route(&logits[i * e..(i + 1) * e], &bias, k, 2.5);
        assert_eq!(
            gsel[i * k..(i + 1) * k],
            esel.iter().map(|&v| v as u32).collect::<Vec<_>>()[..]
        );
        assert_close(&gw[i * k..(i + 1) * k], &ew, 1e-6, 1e-5);
    }
}

#[test]
fn rmsnorm_hc_pre_post_mean_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(57);
    let cfg = crate::glm53::Config::glm53_flash();
    let (m, hidden) = (2, cfg.hidden);
    let h = f32s(&mut rng, m * 4 * hidden, -3.0, 3.0);
    let fn_raw = random_blocks(&mut rng, GgmlType::Bf16, 24 * 4 * hidden);
    let fn_w =
        QWeight::from_bytes(&ctx, &fn_raw, GgmlType::Bf16, 4 * hidden, 24).unwrap();
    let fn_cpu = mat(GgmlType::Bf16, &fn_raw, 24, 4 * hidden);
    let scale = [0.8f32, 1.2, 1.7];
    let base = f32s(&mut rng, 24, -0.5, 0.5);
    let norm = f32s(&mut rng, hidden, 0.5, 1.5);
    let block = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let th = tensor(&ctx, &h);
    let flat = zeros(&ctx, m * 4 * hidden);
    let mix = zeros(&ctx, m * 24);
    let (tsc, tba, tno) =
        (tensor(&ctx, &scale), tensor(&ctx, &base), tensor(&ctx, &norm));
    let split = zeros(&ctx, m * 24);
    let xn = zeros(&ctx, m * hidden);
    let tblock = tensor(&ctx, &block);
    let post_out = zeros(&ctx, m * 4 * hidden);
    let mean = zeros(&ctx, m * hidden);
    run(&ctx, |p| {
        rmsnorm(&ctx, p, &th, None, &flat, m, 4 * hidden, cfg.rms_eps).unwrap();
        gemv(&ctx, p, &fn_w, &flat, &mix, m).unwrap();
        hc_pre(
            &ctx,
            p,
            &mix,
            &tsc,
            &tba,
            &th,
            &tno,
            &split,
            &xn,
            m,
            hidden,
            20,
            1e-6,
            cfg.rms_eps,
        )
        .unwrap();
        hc_post(&ctx, p, &th, &tblock, &split, &post_out, m, hidden).unwrap();
        hc_mean(&ctx, p, &th, &mean, m, hidden).unwrap();
    });
    let gxn = xn.to_f32().unwrap();
    let gsplit = split.to_f32().unwrap();
    let gpost = post_out.to_f32().unwrap();
    let gmean = mean.to_f32().unwrap();
    for i in 0..m {
        let hi = &h[i * 4 * hidden..(i + 1) * 4 * hidden];
        let (exn, epost, ecomb) = r::hc_pre(&cfg, hi, &fn_cpu, &scale, &base, &norm);
        assert_close(&gxn[i * hidden..(i + 1) * hidden], &exn, 1e-4, 1e-3);
        assert_close(&gsplit[i * 24 + 4..i * 24 + 8], &epost, 1e-5, 1e-4);
        assert_close(&gsplit[i * 24 + 8..i * 24 + 24], &ecomb, 1e-5, 1e-4);
        // Post from the GPU's own split, so it is checked on its own.
        let gp = &gsplit[i * 24 + 4..i * 24 + 8];
        let gc = &gsplit[i * 24 + 8..i * 24 + 24];
        let ep = r::hc_post(hi, &block[i * hidden..(i + 1) * hidden], gp, gc, 4);
        assert_close(&gpost[i * 4 * hidden..(i + 1) * 4 * hidden], &ep, 1e-5, 1e-5);
        let em: Vec<f32> = (0..hidden)
            .map(|d| (0..4).map(|s| hi[s * hidden + d] * 0.25).sum())
            .collect();
        assert_close(&gmean[i * hidden..(i + 1) * hidden], &em, 1e-6, 1e-6);
    }
}

#[test]
fn embed_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(58);
    let (vocab, hidden) = (50, 4096);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, vocab * hidden);
    let table = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, hidden, vocab).unwrap();
    let cpu = mat(GgmlType::Q8_0, &raw, vocab, hidden);
    let toks = [7u32, 0, 49];
    let tt = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&toks), &[3], DType::U32)
        .unwrap();
    let out = zeros(&ctx, 3 * 4 * hidden);
    run(&ctx, |p| embed(&ctx, p, &table, &tt, &out, 3).unwrap());
    let expected: Vec<f32> =
        toks.iter().flat_map(|&t| cpu.row(t as usize).repeat(4)).collect();
    assert_eq!(out.to_f32().unwrap(), expected);
}

struct KdaCase {
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
}

fn kda_case(rng: &mut StdRng, heads: usize) -> KdaCase {
    let p = heads * 128;
    KdaCase {
        q_conv: f32s(rng, p * 4, -0.6, 0.6),
        k_conv: f32s(rng, p * 4, -0.6, 0.6),
        v_conv: f32s(rng, p * 4, -0.6, 0.6),
        a_log: f32s(rng, heads, -1.0, 1.0),
        dt_bias: f32s(rng, p, -1.0, 1.0),
        o_norm: f32s(rng, 128, 0.5, 1.5),
    }
}

/// Runs `tokens` KDA tokens on the GPU in chunks of `chunk` rows and on the
/// CPU one by one.
fn kda_compare(tokens: usize, chunk: usize, seed: u64) {
    kda_compare_gates(tokens, chunk, seed, -3.0, 3.0);
}

/// [`kda_compare`] with the raw gates drawn from `gate_lo..gate_hi`.
fn kda_compare_gates(
    tokens: usize,
    chunk: usize,
    seed: u64,
    gate_lo: f32,
    gate_hi: f32,
) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cfg = crate::glm53::Config::glm53_flash();
    let heads = 4;
    cfg.kda_heads = heads;
    let p = heads * 128;
    let c = kda_case(&mut rng, heads);
    let t = |v: &Vec<f32>| tensor(&ctx, v);
    let (qc, kc, vc, al, dt, on) = (
        t(&c.q_conv),
        t(&c.k_conv),
        t(&c.v_conv),
        t(&c.a_log),
        t(&c.dt_bias),
        t(&c.o_norm),
    );
    let params = KdaParams {
        q_conv: &qc,
        k_conv: &kc,
        v_conv: &vc,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let conv_state = zeros(&ctx, 9 * p);
    let state = zeros(&ctx, heads * 128 * 128);
    let mut st = r::KdaState::new(&cfg);
    let cp = r::KdaParams {
        q_conv: &c.q_conv,
        k_conv: &c.k_conv,
        v_conv: &c.v_conv,
        a_log: &c.a_log,
        dt_bias: &c.dt_bias,
        o_norm: &c.o_norm,
    };
    let mut done = 0;
    while done < tokens {
        let n = chunk.min(tokens - done);
        let q = f32s(&mut rng, n * p, -2.0, 2.0);
        let k = f32s(&mut rng, n * p, -2.0, 2.0);
        let v = f32s(&mut rng, n * p, -2.0, 2.0);
        let g = f32s(&mut rng, n * p, gate_lo, gate_hi);
        let b = f32s(&mut rng, n * heads, -2.0, 2.0);
        let og = f32s(&mut rng, n * p, -2.0, 2.0);
        let (tq, tk, tv, tg, tb, tog) = (t(&q), t(&k), t(&v), t(&g), t(&b), t(&og));
        let out = zeros(&ctx, n * p);
        let edges = zeros(&ctx, kda_edges_len(n, heads));
        let inputs = KdaInputs {
            q: &tq,
            k: &tk,
            v: &tv,
            gate: &tg,
            beta: &tb,
            out_gate: &tog,
            edges: Some(&edges),
        };
        run(&ctx, |pass| {
            kda(&ctx, pass, &params, &inputs, &conv_state, &state, &out, n, None)
                .unwrap()
        });
        let got = out.to_f32().unwrap();
        for j in 0..n {
            let s = j * p..(j + 1) * p;
            let e = r::kda_step(
                &cfg,
                &mut st,
                &cp,
                &q[s.clone()],
                &k[s.clone()],
                &v[s.clone()],
                &g[s.clone()],
                &b[j * heads..(j + 1) * heads],
                &og[s.clone()],
            );
            assert_close(&got[s], &e, 2e-4, 2e-3);
        }
        done += n;
    }
    assert_close(&state.to_f32().unwrap(), &st.s, 1e-4, 1e-3);
}

/// Prompt runs of more than KDA_CHUNK rows take the chunked form: whole
/// and partial chunks, a run continuing another's state, and gates at the
/// decay's extremes (the strongest decay is where the chunked form's
/// factors are largest).
#[test]
fn kda_chunked_form_matches_cpu() {
    kda_compare_gates(130, 65, 61, -3.0, 3.0);
    kda_compare_gates(100, 100, 62, -3.0, 3.0);
    kda_compare_gates(128, 64, 63, 20.0, 30.0);
    kda_compare_gates(66, 33, 64, -30.0, -20.0);
}

/// KDA from a nonzero state and conv window through runs of `runs` rows
/// (`Some(r)`: with a snapshot after row r, as a verify pass), each against
/// the CPU recurrence: every output row, the snapshot, and the state and
/// conv window after each run. Gates vary by channel (a third saturate to
/// the strongest decay, a third to none, a third random) and beta by row
/// and head (near 1, near 0, random).
fn kda_handoff_case(runs: &[(usize, Option<usize>)], lower_bound: f32, seed: u64) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cfg = crate::glm53::Config::glm53_flash();
    let heads = 4;
    cfg.kda_heads = heads;
    cfg.kda_gate_lower_bound = lower_bound;
    let p = heads * 128;
    let c = kda_case(&mut rng, heads);
    let t = |v: &Vec<f32>| tensor(&ctx, v);
    let (qc, kc, vc, al, dt, on) = (
        t(&c.q_conv),
        t(&c.k_conv),
        t(&c.v_conv),
        t(&c.a_log),
        t(&c.dt_bias),
        t(&c.o_norm),
    );
    let params = KdaParams {
        q_conv: &qc,
        k_conv: &kc,
        v_conv: &vc,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound,
        eps: 1e-5,
    };
    let cp = r::KdaParams {
        q_conv: &c.q_conv,
        k_conv: &c.k_conv,
        v_conv: &c.v_conv,
        a_log: &c.a_log,
        dt_bias: &c.dt_bias,
        o_norm: &c.o_norm,
    };
    let mut st = r::KdaState::new(&cfg);
    st.conv = f32s(&mut rng, 9 * p, -1.0, 1.0);
    st.s = f32s(&mut rng, heads * 128 * 128, -0.2, 0.2);
    let conv_state = t(&st.conv);
    let state = t(&st.s);
    for (ri, &(n, snap_row)) in runs.iter().enumerate() {
        let q = f32s(&mut rng, n * p, -2.0, 2.0);
        let k = f32s(&mut rng, n * p, -2.0, 2.0);
        let v = f32s(&mut rng, n * p, -2.0, 2.0);
        let g: Vec<f32> = (0..n * p)
            .map(|i| match (i % p) % 3 {
                0 => 30.0,
                1 => -30.0,
                _ => rng.gen_range(-3.0..3.0),
            })
            .collect();
        let b: Vec<f32> = (0..n * heads)
            .map(|i| match i % 3 {
                0 => 12.0,
                1 => -12.0,
                _ => rng.gen_range(-2.0..2.0),
            })
            .collect();
        let og = f32s(&mut rng, n * p, -2.0, 2.0);
        let (tq, tk, tv, tg, tb, tog) = (t(&q), t(&k), t(&v), t(&g), t(&b), t(&og));
        let out = zeros(&ctx, n * p);
        let edges = zeros(&ctx, kda_edges_len(n, heads));
        let inputs = KdaInputs {
            q: &tq,
            k: &tk,
            v: &tv,
            gate: &tg,
            beta: &tb,
            out_gate: &tog,
            edges: Some(&edges),
        };
        let (snap_conv, snap_state) =
            (zeros(&ctx, 9 * p), zeros(&ctx, heads * 128 * 128));
        let snap = snap_row.map(|row| KdaSnapshot {
            row,
            conv: &snap_conv,
            state: &snap_state,
        });
        run(&ctx, |pass| {
            kda(&ctx, pass, &params, &inputs, &conv_state, &state, &out, n, snap)
                .unwrap()
        });
        let got = out.to_f32().unwrap();
        for j in 0..n {
            let s = j * p..(j + 1) * p;
            let e = r::kda_step(
                &cfg,
                &mut st,
                &cp,
                &q[s.clone()],
                &k[s.clone()],
                &v[s.clone()],
                &g[s.clone()],
                &b[j * heads..(j + 1) * heads],
                &og[s.clone()],
            );
            assert_close(&got[s], &e, 2e-4, 2e-3);
            if snap_row == Some(j) {
                assert_close(&snap_state.to_f32().unwrap(), &st.s, 1e-4, 1e-3);
                assert_close(&snap_conv.to_f32().unwrap(), &st.conv, 0.0, 0.0);
            }
        }
        let what = format!("run {ri} of {runs:?}, bound {lower_bound}");
        let gs = state.to_f32().unwrap();
        assert!(gs.iter().all(|v| v.is_finite()), "{what}: state not finite");
        assert_close(&gs, &st.s, 1e-4, 1e-3);
        assert_close(&conv_state.to_f32().unwrap(), &st.conv, 0.0, 0.0);
    }
}

/// A prompt run in the chunked form hands its state and conv window to a
/// decode step and a verify pass with a snapshot (the recurrence's kernels):
/// runs of 33, 63, 64, 65 and 1 024 rows.
#[test]
fn kda_chunked_form_hands_off_to_decode_and_verify() {
    for (i, n) in [33usize, 63, 64, 65, 1024].into_iter().enumerate() {
        assert!(kda_takes_wy(-5.0, n, false));
        kda_handoff_case(&[(n, None), (1, None), (2, Some(0))], -5.0, 90 + i as u64);
    }
}

/// The chunked form only for a finite gate bound in [-5, 0]: the bounds it
/// takes (-5, -2, 0) and one it leaves to the recurrence (-6) against the
/// CPU; positive and non-finite bounds are not taken.
#[test]
fn kda_chunked_form_gate_bounds() {
    for b in [-5.0f32, -2.0, -0.0, 0.0] {
        assert!(kda_takes_wy(b, 33, false), "{b}");
    }
    for b in [-5.0001f32, -6.0, 1e-6, 3.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(!kda_takes_wy(b, 1024, false), "{b}");
    }
    assert!(!kda_takes_wy(-5.0, 32, false));
    assert!(!kda_takes_wy(-5.0, 1024, true));
    for (i, b) in [-6.0f32, -5.0, -2.0, 0.0].into_iter().enumerate() {
        kda_handoff_case(&[(97, None), (40, None)], b, 100 + i as u64);
    }
}

#[test]
fn kda_decode_matches_cpu() {
    kda_compare(6, 1, 59);
}

#[test]
fn kda_prefill_matches_cpu() {
    kda_compare(37, 16, 60);
}

/// A prompt run's prepare stage in KDA_CHUNK-token chunks gives the
/// single-chunk run's results bit for bit: output, prepared q/k/v/decay,
/// recurrent state, conv window and both snapshots, from a nonzero starting
/// state, at row counts around the chunk boundaries and with snapshots on
/// both sides of one, for four heads and one.
#[test]
fn kda_prefill_chunks_are_bit_exact() {
    // The model's head count's kind (even) and an odd one (the edge copy's
    // grid ends mid-threadgroup).
    for heads in [4, 1] {
        kda_chunks_case(heads);
    }
}

fn kda_chunks_case(heads: usize) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(76);
    let p = heads * 128;
    let c = kda_case(&mut rng, heads);
    let t = |v: &Vec<f32>| tensor(&ctx, v);
    let (qc, kc, vc, al, dt, on) = (
        t(&c.q_conv),
        t(&c.k_conv),
        t(&c.v_conv),
        t(&c.a_log),
        t(&c.dt_bias),
        t(&c.o_norm),
    );
    let params = KdaParams {
        q_conv: &qc,
        k_conv: &kc,
        v_conv: &vc,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let conv0 = f32s(&mut rng, 9 * p, -1.0, 1.0);
    let state0 = f32s(&mut rng, heads * 128 * 128, -0.2, 0.2);
    let bits = |x: &Tensor| -> Vec<u32> {
        x.to_f32().unwrap().iter().map(|v| v.to_bits()).collect()
    };
    let cases = [
        (31, Some(30)),
        (32, Some(31)),
        (33, Some(32)),
        (63, None),
        (64, Some(33)),
        (65, Some(64)),
        (65, Some(30)),
        (65, Some(31)),
    ];
    for (n, snap_row) in cases {
        let raw: Vec<Vec<f32>> = [p, p, p, p, heads, p]
            .iter()
            .map(|&w| f32s(&mut rng, n * w, -2.0, 2.0))
            .collect();
        let mut got = Vec::new();
        for chunk in [KDA_CHUNK, n] {
            let x: Vec<Tensor> = raw.iter().map(t).collect();
            let (conv, st) = (t(&conv0), t(&state0));
            let (out, edges) = (zeros(&ctx, n * p), zeros(&ctx, n.div_ceil(3) * 9 * p));
            let (sconv, sst) = (zeros(&ctx, 9 * p), zeros(&ctx, heads * 128 * 128));
            let inputs = KdaInputs {
                q: &x[0],
                k: &x[1],
                v: &x[2],
                gate: &x[3],
                beta: &x[4],
                out_gate: &x[5],
                edges: Some(&edges),
            };
            run(&ctx, |pass| {
                let snap =
                    snap_row.map(|row| KdaSnapshot { row, conv: &sconv, state: &sst });
                kda_chunked(
                    &ctx, pass, &params, &inputs, &conv, &st, &out, n, snap, chunk,
                )
                .unwrap()
            });
            let mut all: Vec<Vec<u32>> = x[..4].iter().map(bits).collect();
            all.extend([bits(&out), bits(&st), bits(&conv), bits(&sconv), bits(&sst)]);
            got.push(all);
        }
        for (k, name) in
            ["q", "k", "v", "decay", "out", "state", "conv", "snap conv", "snap state"]
                .iter()
                .enumerate()
        {
            assert!(
                got[0][k] == got[1][k],
                "{name} differs at {n} rows, {heads} heads"
            );
        }
    }
}

/// The indexer's top-k at the kept-count boundary (511, 512, 513 and 1 025
/// complete pools; the top 512 kept) with random, all-equal and few-valued
/// scores: the k highest, ties to the lower pool, in pool order, then the
/// incomplete pool's tokens, padding past the count.
#[test]
fn indexer_topk_boundaries_and_ties() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(77);
    let (k, s) = (512usize, 2056usize);
    for pools in [511usize, 512, 513, 1025] {
        for pattern in 0..3 {
            let cap = pools + 1;
            let sc: Vec<f32> = match pattern {
                0 => f32s(&mut rng, cap, -1.0, 1.0),
                1 => vec![0.25; cap],
                _ => (0..cap).map(|_| (rng.gen_range(0..4u32) as f32) * 0.5).collect(),
            };
            let pos0 = 4 * pools + 1; // visible 4 pools + 2: a tail of two
            let scores = tensor(&ctx, &sc);
            let sel = Tensor::zeros(&ctx, &[s], DType::U32).unwrap();
            let count = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
            run(&ctx, |p| {
                indexer_topk(&ctx, p, &scores, &sel, &count, 1, cap, s, pos0, k)
                    .unwrap()
            });
            let mut order: Vec<usize> = (0..pools).collect();
            order.sort_by(|&a, &b| sc[b].total_cmp(&sc[a]).then(a.cmp(&b)));
            let mut kept: Vec<usize> = order[..k.min(pools)].to_vec();
            kept.sort();
            let mut expect: Vec<u32> =
                kept.iter().flat_map(|&p| (4 * p as u32)..(4 * p as u32 + 4)).collect();
            expect.extend([4 * pools as u32, 4 * pools as u32 + 1]);
            let got = sel.to_u32().unwrap();
            assert_eq!(
                count.to_u32().unwrap()[0] as usize,
                expect.len(),
                "{pools} {pattern}"
            );
            assert_eq!(
                &got[..expect.len()],
                &expect[..],
                "{pools} pools, pattern {pattern}"
            );
            assert!(got[expect.len()..].iter().all(|&v| v == u32::MAX));
        }
    }
}

/// Runs of more than one prepare chunk (KDA_CHUNK rows): 75-row runs of
/// three chunks each, the second run continuing the first's state.
#[test]
fn kda_prefill_long_runs_match_cpu() {
    kda_compare(150, 75, 75);
}

#[test]
fn dsa_store_and_attend_match_cpu() {
    // 42 cached rows then 3 rows of queries (SIMT kernels); 25 then a run
    // of 20 (matrix units); runs from row 0 in partial tiles of the
    // padded widths 8 and 16.
    for (n, m, s) in [(45, 3, 64), (45, 20, 64), (5, 5, 8), (13, 13, 16)] {
        dsa_causal_case(n, m, s, false);
    }
}

/// Queries at and past f16's range on both routes (a tile with one goes
/// through f32 on the matrix route): finite, and as the CPU computes.
#[test]
fn dsa_attend_large_queries_stay_finite() {
    for m in [3, 20] {
        dsa_causal_case(45, m, 64, true);
    }
}

/// The matrix route's widths are padded to 8.
#[test]
fn dsa_attend_refuses_unpadded_prompt_widths() {
    let ctx = ctx();
    let (m, s) = (5, 5);
    let cache = Tensor::zeros(&ctx, &[8 * 512], DType::F16).unwrap();
    let qk = zeros(&ctx, m * 64 * 512);
    let (scores, denom) = (zeros(&ctx, m * 64 * 8), zeros(&ctx, m * 64));
    let (lat, partial) = (zeros(&ctx, m * 64 * 512), zeros(&ctx, 4 * 8 * 64 * 512));
    let pass = ctx.begin().unwrap();
    let r = dsa_attend(
        &ctx,
        &pass,
        &qk,
        &cache,
        Rows::Causal { pos0: 0 },
        &scores,
        &denom,
        &lat,
        &partial,
        m,
        s,
        1.0,
    );
    assert!(r.is_err());
    // An empty selected width on the matrix route (its first weight loads
    // run before its slot loop): refused too.
    let one = zeros(&ctx, 1);
    let sel = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
    let r = dsa_attend(
        &ctx,
        &pass,
        &qk,
        &cache,
        Rows::Selected { sel: &sel, count: 0 },
        &one,
        &denom,
        &lat,
        &partial,
        m,
        0,
        1.0,
    );
    assert!(r.is_err());
    pass.commit_wait().unwrap();
}

/// The matrix route's scores at its block edges, from an output buffer full
/// of NaN: causal rows whose counts are 63, 64, 65 and 66 with S = 192
/// (blocks wholly past a row's count only write -inf), and selected rows
/// with counts 63, 64 and 65 whose selections reach far past their slot
/// index (cache row 5 999). Valid slots against f64; every other slot -inf.
#[test]
fn dsa_scores_mu_block_boundaries() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(81);
    let n = 6000usize;
    let s = 192usize;
    let lat: Vec<f32> =
        f32s(&mut rng, n * 512, -1.5, 1.5).into_iter().map(r::round_f16).collect();
    let bytes: Vec<u8> =
        lat.iter().flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes()).collect();
    let cache = Tensor::from_bytes(&ctx, &bytes, &[n * 512], DType::F16).unwrap();
    let scale = 1.0 / 16.0;
    let check = |qk: &[f32],
                 g: &[f32],
                 m: usize,
                 count: &dyn Fn(usize) -> usize,
                 row: &dyn Fn(usize, usize) -> usize,
                 what: &str| {
        for i in 0..m {
            for h in 0..64 {
                let q = &qk[(i * 64 + h) * 512..(i * 64 + h + 1) * 512];
                for slot in 0..s {
                    let got = g[(i * 64 + h) * s + slot];
                    if slot >= count(i) {
                        assert_eq!(
                            got,
                            f32::NEG_INFINITY,
                            "{what} row {i} head {h} slot {slot}"
                        );
                        continue;
                    }
                    let k = &lat[row(i, slot) * 512..(row(i, slot) + 1) * 512];
                    let terms = q.iter().zip(k).map(|(&a, &b)| a as f64 * b as f64);
                    let want = terms.clone().sum::<f64>() * scale as f64;
                    let mag: f64 = terms.map(f64::abs).sum();
                    let tol = scale as f64 * (3e-6 * mag + 512.0 * 6e-8 * 2.0);
                    assert!(
                        got.is_finite() && (got as f64 - want).abs() <= tol,
                        "{what} row {i} head {h} slot {slot}: {got} against {want}"
                    );
                }
            }
        }
    };
    // Causal: counts pos0 + i + 1 = 63, 64, 65, 66.
    let (m, pos0) = (4usize, 62usize);
    let qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
    let tqk = tensor(&ctx, &qk);
    let scores = tensor(&ctx, &vec![f32::NAN; m * 64 * s]);
    run(&ctx, |p| {
        dsa_scores(
            &ctx,
            p,
            &tqk,
            &cache,
            Rows::Causal { pos0 },
            &scores,
            m,
            s,
            scale,
            true,
        )
        .unwrap()
    });
    check(
        &qk,
        &scores.to_f32().unwrap(),
        m,
        &|i| pos0 + i + 1,
        &|_, slot| slot,
        "causal",
    );
    // Selected: per dispatch one count; row slot -> a row far in the cache,
    // the last slots' rows the cache's last.
    let m = 2usize;
    let sel: Vec<u32> = (0..m * s)
        .map(|k| {
            let slot = k % s;
            if slot % 7 == 6 {
                (n - 1) as u32
            } else {
                (5000 + 3 * slot + k / s) as u32
            }
        })
        .collect();
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    for count in [63usize, 64, 65] {
        let qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
        let tqk = tensor(&ctx, &qk);
        let scores = tensor(&ctx, &vec![f32::NAN; m * 64 * s]);
        run(&ctx, |p| {
            dsa_scores(
                &ctx,
                p,
                &tqk,
                &cache,
                Rows::Selected { sel: &tsel, count },
                &scores,
                m,
                s,
                scale,
                true,
            )
            .unwrap()
        });
        check(
            &qk,
            &scores.to_f32().unwrap(),
            m,
            &|_| count,
            &|i, slot| sel[i * s + slot] as usize,
            &format!("selected count {count}"),
        );
    }
}

/// The scores stage on both routes against an f64 reference over 200 slots
/// (four matrix tiles): f16's largest value alone stays on the matrix
/// route, the next value past it (either sign) sends its query's tiles to
/// f32, a NaN or infinite query value makes only its own head's scores
/// non-finite, and slots past each row's count are -inf.
#[test]
fn dsa_scores_at_and_past_f16_range() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(78);
    let (n, m) = (200usize, 6usize);
    let s = n;
    let pos0 = n - m;
    let lat: Vec<f32> =
        f32s(&mut rng, n * 512, -1.5, 1.5).into_iter().map(r::round_f16).collect();
    let bytes: Vec<u8> =
        lat.iter().flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes()).collect();
    let cache = Tensor::from_bytes(&ctx, &bytes, &[n * 512], DType::F16).unwrap();
    let mut qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
    let at = |i: usize, h: usize, j: usize| (i * 64 + h) * 512 + j;
    let past = f32::from_bits(65504f32.to_bits() + 1);
    qk[at(0, 0, 0)] = 65504.0;
    qk[at(0, 1, 3)] = -65504.0;
    qk[at(1, 2, 1)] = past;
    qk[at(2, 0, 2)] = -past;
    qk[at(3, 3, 4)] = f32::NAN;
    qk[at(4, 5, 6)] = f32::INFINITY;
    let tqk = tensor(&ctx, &qk);
    let scale = 1.0 / 16.0;
    for mu in [true, false] {
        let scores = zeros(&ctx, m * 64 * s);
        run(&ctx, |p| {
            dsa_scores(
                &ctx,
                p,
                &tqk,
                &cache,
                Rows::Causal { pos0 },
                &scores,
                m,
                s,
                scale,
                mu,
            )
            .unwrap()
        });
        let g = scores.to_f32().unwrap();
        for i in 0..m {
            let count = pos0 + i + 1;
            for h in 0..64 {
                let q = &qk[at(i, h, 0)..at(i, h, 0) + 512];
                for slot in 0..s {
                    let got = g[(i * 64 + h) * s + slot];
                    if slot >= count {
                        assert_eq!(
                            got,
                            f32::NEG_INFINITY,
                            "mu {mu} row {i} head {h} slot {slot}"
                        );
                        continue;
                    }
                    if (i, h) == (3, 3) {
                        assert!(got.is_nan(), "mu {mu}: NaN query gave {got}");
                        continue;
                    }
                    if (i, h) == (4, 5) {
                        assert!(!got.is_finite(), "mu {mu}: infinite query gave {got}");
                        continue;
                    }
                    let k = &lat[slot * 512..(slot + 1) * 512];
                    let terms = q.iter().zip(k).map(|(&a, &b)| a as f64 * b as f64);
                    let want = terms.clone().sum::<f64>() * scale as f64;
                    let mag: f64 = terms.map(f64::abs).sum();
                    let tol = scale as f64 * (3e-6 * mag + 512.0 * 6e-8 * 2.0);
                    assert!(
                        got.is_finite() && (got as f64 - want).abs() <= tol,
                        "mu {mu} row {i} head {h} slot {slot}: {got} against {want} (tol {tol})"
                    );
                }
            }
        }
    }
}

fn dsa_causal_case(n: usize, m: usize, s: usize, big: bool) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(61);
    let cfg = crate::glm53::Config::glm53_flash();
    let kv_raw = f32s(&mut rng, n * 512, -2.0, 2.0);
    let w = f32s(&mut rng, 512, 0.5, 1.5);
    let cache = Tensor::zeros(&ctx, &[64 * 512], DType::F16).unwrap();
    let (tkv, tw) = (tensor(&ctx, &kv_raw), tensor(&ctx, &w));
    run(&ctx, |p| kv_store(&ctx, p, &tkv, &tw, &cache, 0, n, cfg.rms_eps).unwrap());
    let latent: Vec<f32> = (0..n)
        .flat_map(|t| {
            r::rmsnorm(&kv_raw[t * 512..(t + 1) * 512], Some(&w), cfg.rms_eps)
        })
        .map(r::round_f16)
        .collect();
    let gl: Vec<f32> = cache.contents()[..n * 1024]
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect();
    let diffs = gl.iter().zip(&latent).filter(|(a, b)| a != b).count();
    assert!(diffs * 1000 < latent.len(), "{diffs} f16 latent values differ");
    let mut qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
    if big {
        // f16's largest value, the first values past it, far past it, and
        // large values of both signs whose products cancel in part.
        let at = |i: usize, h: usize, j: usize| (i * 64 + h) * 512 + j;
        qk[at(0, 0, 0)] = 65536.0;
        qk[at(0, 1, 5)] = -65536.0;
        qk[at(1, 2, 7)] = 65520.0;
        qk[at(m - 1, 3, 0)] = 1.0e6;
        qk[at(m - 1, 3, 1)] = -1.0e6;
        for j in 0..512 {
            qk[at(1, 4, j)] = if j % 2 == 0 { 65504.0 } else { -65504.0 };
            qk[at(2, 5, j)] = 3.0e4;
        }
    }
    let tqk = tensor(&ctx, &qk);
    let scores = zeros(&ctx, m * 64 * s);
    let denom = zeros(&ctx, m * 64);
    let lat = zeros(&ctx, m * 64 * 512);
    let partial = zeros(&ctx, 4 * 8 * 64 * 512);
    let pos0 = n - m;
    run(&ctx, |p| {
        dsa_attend(
            &ctx,
            p,
            &tqk,
            &cache,
            Rows::Causal { pos0 },
            &scores,
            &denom,
            &lat,
            &partial,
            m,
            s,
            1.0 / 16.0,
        )
        .unwrap()
    });
    let glat = lat.to_f32().unwrap();
    assert!(
        glat[..m * 64 * 512].iter().all(|v| v.is_finite()),
        "non-finite latent mix"
    );
    for i in 0..m {
        let rows: Vec<usize> = (0..=pos0 + i).collect();
        for h in 0..64 {
            let ql = &qk[(i * 64 + h) * 512..(i * 64 + h + 1) * 512];
            let sc: Vec<f32> = rows
                .iter()
                .map(|&t| r::dot(ql, &gl[t * 512..(t + 1) * 512]) / 16.0)
                .collect();
            let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = sc.iter().map(|&x| (x - mx).exp()).collect();
            let d: f32 = e.iter().sum();
            let exp: Vec<f32> = (0..512)
                .map(|j| {
                    rows.iter()
                        .zip(&e)
                        .map(|(&t, &wt)| wt * gl[t * 512 + j])
                        .sum::<f32>()
                        / d
                })
                .collect();
            assert_close(
                &glat[(i * 64 + h) * 512..(i * 64 + h + 1) * 512],
                &exp,
                1e-5,
                1e-4,
            );
        }
    }
}

/// The attention over selected rows (the sparse and MTP paths): per-row
/// lists with unused slots, against the CPU.
#[test]
fn dsa_attend_selected_matches_cpu() {
    // Decode rows (SIMT kernels, the slots in chunks) and a prompt run
    // (matrix units); a short selection and a full one (its last chunk full).
    for (m, count) in [(2, 344), (6, 344), (2, 2056), (6, 2056)] {
        dsa_selected_case(m, count);
    }
}

fn dsa_selected_case(m: usize, count: usize) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(70);
    let (cached, s) = (700usize, 2056usize);
    let lat_rows: Vec<f32> =
        f32s(&mut rng, cached * 512, -1.5, 1.5).into_iter().map(r::round_f16).collect();
    let bytes: Vec<u8> = lat_rows
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let cache = Tensor::from_bytes(&ctx, &bytes, &[cached * 512], DType::F16).unwrap();
    let mut sel = vec![u32::MAX; m * s];
    let mut lists = Vec::new();
    for i in 0..m {
        let mut l = Vec::new();
        for slot in 0..count {
            // Every seventh slot unused; rows spread over the cache.
            if slot % 7 != 3 {
                let row = (slot * 131 + i * 17) % cached;
                sel[i * s + slot] = row as u32;
                l.push(row);
            }
        }
        lists.push(l);
    }
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[m * s], DType::U32)
            .unwrap();
    let qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
    let tqk = tensor(&ctx, &qk);
    let scores = zeros(&ctx, m * 64 * s);
    let denom = zeros(&ctx, m * 64);
    let lat = zeros(&ctx, m * 64 * 512);
    // Bound here: the pass reads it until run's commit_wait returns.
    let partial = zeros(&ctx, 4 * 8 * 64 * 512);
    run(&ctx, |p| {
        dsa_attend(
            &ctx,
            p,
            &tqk,
            &cache,
            Rows::Selected { sel: &tsel, count },
            &scores,
            &denom,
            &lat,
            &partial,
            m,
            s,
            1.0 / 16.0,
        )
        .unwrap()
    });
    let glat = lat.to_f32().unwrap();
    for (i, rows) in lists.iter().enumerate() {
        for h in 0..64 {
            let ql = &qk[(i * 64 + h) * 512..(i * 64 + h + 1) * 512];
            let sc: Vec<f32> = rows
                .iter()
                .map(|&t| r::dot(ql, &lat_rows[t * 512..(t + 1) * 512]) / 16.0)
                .collect();
            let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = sc.iter().map(|&x| (x - mx).exp()).collect();
            let d: f32 = e.iter().sum();
            let exp: Vec<f32> = (0..512)
                .map(|j| {
                    rows.iter()
                        .zip(&e)
                        .map(|(&t, &wt)| wt * lat_rows[t * 512 + j])
                        .sum::<f32>()
                        / d
                })
                .collect();
            assert_close(
                &glat[(i * 64 + h) * 512..(i * 64 + h + 1) * 512],
                &exp,
                1e-5,
                1e-4,
            );
        }
    }
}

/// DSA decode attention (scores, softmax, weighted latent sum) at the
/// model's shapes: a dense run at 264 rows and the sparse 2 056-slot
/// selection, `LILY_GLM53_ROWS` query rows (default 1), best of five serial
/// passes of 11 layers' calls over 11 caches.
#[test]
#[ignore = "timing only"]
fn dsa_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(71);
    let m: usize =
        std::env::var("LILY_GLM53_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let cached = 4112 + m;
    let caches: Vec<Tensor> = (0..11)
        .map(|_| {
            let bytes: Vec<u8> = f32s(&mut rng, cached * 512, -1.5, 1.5)
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect();
            Tensor::from_bytes(&ctx, &bytes, &[cached * 512], DType::F16).unwrap()
        })
        .collect();
    let s_max = 2056;
    let sel: Vec<u32> =
        (0..m * s_max).map(|k| ((k % s_max) * 2 + k / s_max) as u32).collect();
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[m * s_max], DType::U32)
            .unwrap();
    let qk = tensor(&ctx, &f32s(&mut rng, m * 64 * 512, -0.2, 0.2));
    let scores = zeros(&ctx, m * 64 * s_max);
    let denom = zeros(&ctx, m * 64);
    let lat = zeros(&ctx, m * 64 * 512);
    let partial = zeros(&ctx, 4 * 8 * 64 * 512);
    // Prompt runs: two layers (a 1 024-row call takes milliseconds), the
    // dense case causal over the run itself.
    let caches = if m > 4 { &caches[..2] } else { &caches[..] };
    for (label, sparse) in [("dense 264 rows", false), ("selected 2056", true)] {
        let s = if sparse { s_max } else { 264.max(m).next_multiple_of(8) };
        let rows = if sparse {
            Rows::Selected { sel: &tsel, count: s_max }
        } else {
            Rows::Causal { pos0: 264usize.saturating_sub(m) }
        };
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for c in caches {
                dsa_attend(
                    &ctx, &pass, &qk, c, rows, &scores, &denom, &lat, &partial, m, s,
                    0.0625,
                )
                .unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "dsa {label:16} m={m} {:8.1} us a layer",
            best / caches.len() as f64 * 1e6
        );
        // With LILY_KERNEL_PROFILE: each kernel alone, per layer.
        let mut by: Vec<(&str, f64)> = Vec::new();
        for p in crate::metal::profile::take() {
            for k in &p.kernels {
                match by.iter_mut().find(|(n, _)| *n == k.name) {
                    Some(e) => e.1 += k.gpu_secs,
                    None => by.push((k.name, k.gpu_secs)),
                }
            }
        }
        for (name, secs) in by {
            println!(
                "dsa   {name:28} {:8.1} us a layer",
                secs / (5 * caches.len()) as f64 * 1e6
            );
        }
    }
}

/// The indexer's pool scores and top-k at decode (one row): 1 024 and
/// 8 192 complete pools, the top 512 kept; best of five passes of 11 calls.
#[test]
#[ignore = "timing only"]
fn indexer_topk_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(72);
    for pools in [1024usize, 8192] {
        let cap = pools + 1;
        let scores = tensor(&ctx, &f32s(&mut rng, cap, -1.0, 1.0));
        let s = 2056;
        let sel = Tensor::zeros(&ctx, &[s], DType::U32).unwrap();
        let count = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for _ in 0..11 {
                indexer_topk(
                    &ctx,
                    &pass,
                    &scores,
                    &sel,
                    &count,
                    1,
                    cap,
                    s,
                    4 * pools,
                    512,
                )
                .unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("indexer_topk {pools:5} pools {:8.1} us", best / 11.0 * 1e6);
        let iq = tensor(&ctx, &f32s(&mut rng, 32 * 128, -1.0, 1.0));
        let w = tensor(&ctx, &f32s(&mut rng, 32, -1.0, 1.0));
        let bytes: Vec<u8> = f32s(&mut rng, pools * 128, -1.0, 1.0)
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let pooled =
            Tensor::from_bytes(&ctx, &bytes, &[pools * 128], DType::F16).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for _ in 0..11 {
                indexer_scores(
                    &ctx,
                    &pass,
                    &iq,
                    &w,
                    &pooled,
                    &scores,
                    1,
                    cap,
                    4 * pools,
                    0.01,
                )
                .unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("indexer_scores {pools:5} pools {:8.1} us", best / 11.0 * 1e6);
    }
}

#[test]
fn indexer_pool_update_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(62);
    let cfg = crate::glm53::Config::glm53_flash();
    let norm_w = f32s(&mut rng, 128, 0.5, 1.5);
    let norm_b = f32s(&mut rng, 128, -0.2, 0.2);
    let ape_f: Vec<f32> = f32s(&mut rng, 4 * 128, -1.0, 1.0)
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect();
    let ape_b: Vec<u8> = ape_f
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let (tw, tb) = (tensor(&ctx, &norm_w), tensor(&ctx, &norm_b));
    let tape = Tensor::from_bytes(&ctx, &ape_b, &[ape_b.len()], DType::I8).unwrap();
    let pp = PoolParams { norm_w: &tw, norm_b: &tb, ape: &tape };
    let pooled = Tensor::zeros(&ctx, &[16 * 128], DType::F16).unwrap();
    let tail_k = zeros(&ctx, 4 * 128);
    let tail_g = zeros(&ctx, 4 * 128);
    let mut st = r::DsaState::default();
    let mut pos = 0;
    // Feeds that start and end mid-pool, one token, and several pools.
    for n in [3, 1, 6, 1, 1, 9, 2] {
        let k = f32s(&mut rng, n * 128, -2.0, 2.0);
        let g = f32s(&mut rng, n * 128, -2.0, 2.0);
        let (tk, tg) = (tensor(&ctx, &k), tensor(&ctx, &g));
        run(&ctx, |p| {
            indexer_pool_update(
                &ctx, p, &pp, &tk, &tg, &pooled, &tail_k, &tail_g, pos, n,
            )
            .unwrap()
        });
        for t in 0..n {
            r::indexer_pool_push(
                &cfg,
                &mut st,
                &k[t * 128..(t + 1) * 128],
                &g[t * 128..(t + 1) * 128],
                &norm_w,
                &norm_b,
                &ape_f,
            );
        }
        pos += n;
    }
    let pools = pos / 4;
    let got: Vec<f32> = pooled.contents()[..pools * 256]
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect();
    assert_close(&got, &st.pooled, 2e-3, 2e-3);
}

#[test]
fn indexer_scores_and_topk_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(63);
    let cfg = crate::glm53::Config::glm53_flash();
    // Rows past the dense limit with 700 to 702 complete pools; scores
    // quantized so ties occur.
    let (m, pos0) = (3, 2803);
    let pools_cap = (pos0 + m) / 4;
    let pooled_f: Vec<f32> = f32s(&mut rng, pools_cap * 128, -1.0, 1.0)
        .iter()
        .map(|&v| r::round_f16((v * 8.0).round() / 8.0))
        .collect();
    let pooled_b: Vec<u8> = pooled_f
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let pooled =
        Tensor::from_bytes(&ctx, &pooled_b, &[pooled_f.len()], DType::F16).unwrap();
    let iq: Vec<f32> = f32s(&mut rng, m * 32 * 128, -1.0, 1.0)
        .iter()
        .map(|&v| (v * 4.0).round() / 4.0)
        .collect();
    let w = f32s(&mut rng, m * 32, 0.0, 1.0)
        .iter()
        .map(|&v| (v * 4.0).round() / 4.0)
        .collect::<Vec<_>>();
    let (tiq, tw) = (tensor(&ctx, &iq), tensor(&ctx, &w));
    let scores = zeros(&ctx, m * pools_cap);
    let s = cfg.indexer_selected_limit();
    let sel = Tensor::zeros(&ctx, &[m * s], DType::U32).unwrap();
    let count = Tensor::zeros(&ctx, &[m], DType::U32).unwrap();
    let scale = 1.0 / ((128 * 32) as f32).sqrt();
    run(&ctx, |p| {
        indexer_scores(&ctx, p, &tiq, &tw, &pooled, &scores, m, pools_cap, pos0, scale)
            .unwrap();
        indexer_topk(&ctx, p, &scores, &sel, &count, m, pools_cap, s, pos0, 512)
            .unwrap();
    });
    let gsel = sel.to_u32().unwrap();
    let gcount = count.to_u32().unwrap();
    let gscores = scores.to_f32().unwrap();
    let st = r::DsaState { pooled: pooled_f.clone(), ..Default::default() };
    for i in 0..m {
        let pos = pos0 + i;
        let pools = (pos + 1) / 4;
        let iq_i = &iq[i * 4096..(i + 1) * 4096];
        let w_i = &w[i * 32..(i + 1) * 32];
        let escores: Vec<f32> = (0..pools)
            .map(|p| {
                (0..32)
                    .map(|h| {
                        (r::dot(
                            &iq_i[h * 128..(h + 1) * 128],
                            &pooled_f[p * 128..(p + 1) * 128],
                        ) * scale)
                            .max(0.0)
                            * w_i[h]
                    })
                    .sum()
            })
            .collect();
        assert_close(
            &gscores[i * pools_cap..i * pools_cap + pools],
            &escores,
            1e-5,
            1e-5,
        );
        // The selection from the GPU's own scores (the order of equal sums
        // can differ in the last bit), compared as sets.
        let mut order: Vec<usize> = (0..pools).collect();
        let gs = &gscores[i * pools_cap..];
        order.sort_by(|&a, &b| gs[b].total_cmp(&gs[a]).then(a.cmp(&b)));
        let mut expect: Vec<u32> = order[..512]
            .iter()
            .flat_map(|&p| (4 * p as u32)..(4 * p as u32 + 4))
            .collect();
        expect.sort();
        let tail = (pos + 1) % 4;
        expect.extend(((pos + 1 - tail) as u32)..((pos + 1) as u32));
        assert_eq!(gcount[i] as usize, expect.len());
        assert_eq!(&gsel[i * s..i * s + expect.len()], &expect[..], "row {i}");
        assert!(gsel[i * s + expect.len()..(i + 1) * s].iter().all(|&v| v == u32::MAX));
        // And the CPU reference picks the same set from its own scores.
        let mut cpu = r::indexer_select(&cfg, &st, pos, iq_i, w_i);
        let ctail = cpu.split_off(2048);
        cpu.sort();
        cpu.extend(ctail);
        let same = cpu.iter().zip(&expect).filter(|(a, b)| **a as u32 == **b).count();
        assert!(
            same + 8 >= expect.len(),
            "CPU selection differs in {} rows",
            expect.len() - same
        );
    }
}

#[test]
fn hc_post_mix_and_pre2_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(64);
    let cfg = crate::glm53::Config::glm53_flash();
    let (m, hidden) = (2, cfg.hidden);
    let h = f32s(&mut rng, m * 4 * hidden, -3.0, 3.0);
    let block = f32s(&mut rng, m * hidden, -1.0, 1.0);
    // A previous split: post weights and a comb near a doubly stochastic one.
    let mut prev = Vec::new();
    for _ in 0..m {
        prev.extend(f32s(&mut rng, 4, 0.0, 1.0));
        prev.extend(f32s(&mut rng, 4, 0.0, 2.0));
        prev.extend(f32s(&mut rng, 16, 0.0, 0.5));
    }
    let fn_raw = random_blocks(&mut rng, GgmlType::Bf16, 24 * 4 * hidden);
    let fn_w =
        QWeight::from_bytes(&ctx, &fn_raw, GgmlType::Bf16, 4 * hidden, 24).unwrap();
    let fn_cpu = mat(GgmlType::Bf16, &fn_raw, 24, 4 * hidden);
    let scale = [0.8f32, 1.2, 1.7];
    let base = f32s(&mut rng, 24, -0.5, 0.5);
    let norm = f32s(&mut rng, hidden, 0.5, 1.5);
    let th = tensor(&ctx, &h);
    let (tblock, tprev) = (tensor(&ctx, &block), tensor(&ctx, &prev));
    let partial = zeros(&ctx, m * hc_mix_groups(hidden) * 25);
    let (tsc, tba, tno) =
        (tensor(&ctx, &scale), tensor(&ctx, &base), tensor(&ctx, &norm));
    let split = zeros(&ctx, m * 24);
    let xn = zeros(&ctx, m * hidden);
    run(&ctx, |p| {
        hc_post_mix(&ctx, p, &th, Some((&tblock, &tprev)), &fn_w, &partial, m, hidden)
            .unwrap();
        hc_pre2(
            &ctx,
            p,
            &partial,
            &tsc,
            &tba,
            &th,
            &tno,
            &split,
            &xn,
            m,
            hidden,
            20,
            1e-6,
            cfg.rms_eps,
        )
        .unwrap();
    });
    let gh = th.to_f32().unwrap();
    let gxn = xn.to_f32().unwrap();
    let gsplit = split.to_f32().unwrap();
    for i in 0..m {
        let hi = &h[i * 4 * hidden..(i + 1) * 4 * hidden];
        let pi = &prev[i * 24..(i + 1) * 24];
        let eh = r::hc_post(
            hi,
            &block[i * hidden..(i + 1) * hidden],
            &pi[4..8],
            &pi[8..24],
            4,
        );
        assert_close(&gh[i * 4 * hidden..(i + 1) * 4 * hidden], &eh, 1e-5, 1e-5);
        let (exn, epost, ecomb) = r::hc_pre(&cfg, &eh, &fn_cpu, &scale, &base, &norm);
        assert_close(&gsplit[i * 24 + 4..i * 24 + 8], &epost, 1e-5, 1e-4);
        assert_close(&gsplit[i * 24 + 8..i * 24 + 24], &ecomb, 1e-5, 1e-4);
        assert_close(&gxn[i * hidden..(i + 1) * hidden], &exn, 1e-4, 1e-3);
    }
}

/// The HC pair of a sublayer boundary (`hc_post_mix` then `hc_pre2`, each
/// its own level as in the step) at the model's shapes: best of five
/// serial passes of `LILY_GLM53_REPEAT` (default 64) pairs over 128 copies
/// of the control weights (100 MB: not cache resident, as in a step).
/// `LILY_GLM53_ROWS` rows (default 1).
#[test]
#[ignore = "timing only"]
fn hc_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(69);
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let (repeat, m) = (env("LILY_GLM53_REPEAT", 64), env("LILY_GLM53_ROWS", 1));
    let cfg = crate::glm53::Config::glm53_flash();
    let hidden = cfg.hidden;
    let fn_raw = random_blocks(&mut rng, GgmlType::Bf16, 24 * 4 * hidden);
    let fns: Vec<QWeight> = (0..128)
        .map(|_| {
            QWeight::from_bytes(&ctx, &fn_raw, GgmlType::Bf16, 4 * hidden, 24).unwrap()
        })
        .collect();
    let h = tensor(&ctx, &f32s(&mut rng, m * 4 * hidden, -0.01, 0.01));
    let block = tensor(&ctx, &f32s(&mut rng, m * hidden, -1e-3, 1e-3));
    let partial = zeros(&ctx, m * hc_mix_groups(hidden) * 25);
    let scale = tensor(&ctx, &[0.8f32, 1.2, 1.7]);
    let base = tensor(&ctx, &f32s(&mut rng, 24, -0.5, 0.5));
    let norm = tensor(&ctx, &f32s(&mut rng, hidden, 0.5, 1.5));
    let split = zeros(&ctx, m * 24);
    let xn = zeros(&ctx, m * hidden);
    for (label, part) in [("post_mix + pre2", 0), ("post_mix", 1), ("pre2", 2)] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for r in 0..repeat {
                if part != 2 {
                    let post = (r > 0).then_some((&block, &split));
                    hc_post_mix(
                        &ctx,
                        &pass,
                        &h,
                        post,
                        &fns[r % 128],
                        &partial,
                        m,
                        hidden,
                    )
                    .unwrap();
                }
                if part != 1 {
                    hc_pre2(
                        &ctx,
                        &pass,
                        &partial,
                        &scale,
                        &base,
                        &h,
                        &norm,
                        &split,
                        &xn,
                        m,
                        hidden,
                        20,
                        1e-6,
                        cfg.rms_eps,
                    )
                    .unwrap();
                }
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("hc {label:20} m={m} {:8.1} us", best / repeat as f64 * 1e6);
    }
}

#[test]
fn gemv_sets_match_single_gemvs() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(65);
    let cols = 4096;
    for ty in [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Bf16] {
        let rows = [72usize, 8, 130, 64];
        let raws: Vec<Vec<u8>> =
            rows.iter().map(|&r| random_blocks(&mut rng, ty, r * cols)).collect();
        let ws: Vec<QWeight> = raws
            .iter()
            .zip(rows)
            .map(|(raw, r)| QWeight::from_bytes(&ctx, raw, ty, cols, r).unwrap())
            .collect();
        for m in [1, 3] {
            let x = f32s(&mut rng, m * cols, -1.0, 1.0);
            let tx = tensor(&ctx, &x);
            let ys: Vec<Tensor> = rows.iter().map(|&r| zeros(&ctx, m * r)).collect();
            let singles: Vec<Tensor> =
                rows.iter().map(|&r| zeros(&ctx, m * r)).collect();
            let sets: Vec<(&QWeight, &Tensor)> = ws.iter().zip(&ys).collect();
            run(&ctx, |p| {
                gemv_sets(&ctx, p, &sets, &tx, m).unwrap();
                for (w, y) in ws.iter().zip(&singles) {
                    gemv(&ctx, p, w, &tx, y, m).unwrap();
                }
            });
            for (a, b) in ys.iter().zip(&singles) {
                assert_eq!(a.to_f32().unwrap(), b.to_f32().unwrap(), "{ty:?} m={m}");
            }
        }
    }
}

/// Bandwidth of the GEMVs at the model's shapes: `LILY_GLM53_REPEAT`
/// (default 16) dispatches over four weight copies (so the cache does not
/// hold them) in one pass, best of five passes.
#[test]
#[ignore = "timing only"]
fn gemv_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(66);
    let repeat: usize = std::env::var("LILY_GLM53_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let shapes = [
        ("kda_v Q8_0 8192x4096", GgmlType::Q8_0, 4096, 8192, 1),
        ("kda_out Q8_0 4096x8192", GgmlType::Q8_0, 8192, 4096, 1),
        ("kda_q Q4_K 8192x4096", GgmlType::Q4K, 4096, 8192, 1),
        ("shexp Q8_0 2048x4096", GgmlType::Q8_0, 4096, 2048, 1),
        ("head Q8_0 154880x4096", GgmlType::Q8_0, 4096, 154880, 1),
        ("kda_v Q8_0 m=4", GgmlType::Q8_0, 4096, 8192, 4),
        ("router F32 288x4096", GgmlType::F32, 4096, 288, 1),
        ("kv_a Q8_0 512x4096", GgmlType::Q8_0, 4096, 512, 1),
        ("q_a Q8_0 1536x4096", GgmlType::Q8_0, 4096, 1536, 1),
        ("f_b Q8_0 8192x128", GgmlType::Q8_0, 128, 8192, 1),
        ("attn_out Q8_0 4096x16384", GgmlType::Q8_0, 16384, 4096, 1),
    ];
    for (label, ty, cols, rows, m) in shapes {
        let copies = if rows > 100_000 { 1 } else { 4 };
        let ws: Vec<QWeight> = (0..copies)
            .map(|_| {
                let raw = random_blocks(&mut rng, ty, rows * cols);
                QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
            })
            .collect();
        let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
        let y = zeros(&ctx, m * rows);
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for r in 0..repeat {
                gemv(&ctx, &pass, &ws[r % copies], &x, &y, m).unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let bytes = ws[0].row_bytes * rows;
        let per = best / repeat as f64;
        println!(
            "{label:28} {:8.1} us  {:6.1} GB/s",
            per * 1e6,
            bytes as f64 / per / 1e9
        );
    }
}

/// The routed experts' kernels at the model's shapes (8 of 288 experts,
/// gate/up IQ2_XXS 2048 x 4096, down Q2_K 4096 x 2048), one row, best of
/// five passes of `LILY_GLM53_REPEAT` layers each over four weight copies.
#[test]
#[ignore = "timing only"]
fn moe_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(67);
    let repeat: usize = std::env::var("LILY_GLM53_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let (experts, hidden, ff, used) = (16, 4096, 2048, 8);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let layers: Vec<(QWeight, QWeight, QWeight)> = (0..4)
        .map(|_| {
            (
                mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff),
                mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff),
                mk(&mut rng, GgmlType::Q2K, ff, experts * hidden),
            )
        })
        .collect();
    let x = tensor(&ctx, &f32s(&mut rng, hidden, -1.0, 1.0));
    let sel: Vec<u32> = (0..used as u32).map(|i| i * 2).collect();
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[used], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &vec![0.3f32; used]);
    let mid = zeros(&ctx, used * ff);
    let out = zeros(&ctx, hidden);
    let xh = Tensor::zeros(&ctx, &[hidden], DType::F16).unwrap();
    run(&ctx, |p| crate::kernels::deepseek4::x_f16_iq2(&ctx, p, &x, &xh, 1, hidden).unwrap());
    for (label, part) in [("gate_up", 0), ("down", 1), ("gate_up f16 x", 2)] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for r in 0..repeat {
                let (g, u, d) = &layers[r % 4];
                match part {
                    0 => moe_gate_up(
                        &ctx, &pass, g, u, ff, &x, &tsel, &tw, &mid, 1, used, 10.0,
                    )
                    .unwrap(),
                    1 => moe_down(&ctx, &pass, d, hidden, &mid, &tsel, None, &out, 1, used)
                        .unwrap(),
                    _ => moe_gate_up_xh(
                        &ctx, &pass, g, u, ff, &xh, &tsel, &tw, &mid, 1, used, 10.0,
                    )
                    .unwrap(),
                }
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let bytes = if part % 2 == 0 { 2 * used * ff * 1056 } else { used * hidden * 672 };
        let per = best / repeat as f64;
        println!(
            "moe {label:24} {:8.1} us  {:6.1} GB/s",
            per * 1e6,
            bytes as f64 / per / 1e9
        );
    }
}

/// One-token KDA at the model's shape (64 heads), best of five serial
/// passes of 34 layers' calls over 34 states (146 MB: as in a step); then a
/// 1 024-row prompt run, best of three passes of three layers. (The prompt
/// run's q, k, v and gate are prepared in place: later passes see their
/// silu'd values, the same work.)
#[test]
#[ignore = "timing only"]
fn kda_decode_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(73);
    let heads = 64;
    let p = heads * 128;
    let mk = |rng: &mut StdRng, n: usize, lo: f32, hi: f32| {
        tensor(&ctx, &f32s(rng, n, lo, hi))
    };
    let (q, k, v) = (
        mk(&mut rng, p, -1.0, 1.0),
        mk(&mut rng, p, -1.0, 1.0),
        mk(&mut rng, p, -1.0, 1.0),
    );
    let (g, beta, og) = (
        mk(&mut rng, p, -1.0, 1.0),
        mk(&mut rng, heads, -1.0, 1.0),
        mk(&mut rng, p, -1.0, 1.0),
    );
    let convs: Vec<Tensor> = (0..3).map(|_| mk(&mut rng, 4 * p, -0.5, 0.5)).collect();
    let (a_log, dt, on) = (
        mk(&mut rng, heads, -1.0, 0.0),
        mk(&mut rng, p, -0.5, 0.5),
        mk(&mut rng, 128, 0.5, 1.5),
    );
    let layers: Vec<(Tensor, Tensor)> = (0..34)
        .map(|_| (zeros(&ctx, 9 * p), mk(&mut rng, heads * 128 * 128, -0.1, 0.1)))
        .collect();
    let out = zeros(&ctx, p);
    let edges = zeros(&ctx, kda_edges_len(1024, heads));
    let params = KdaParams {
        q_conv: &convs[0],
        k_conv: &convs[1],
        v_conv: &convs[2],
        a_log: &a_log,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let x = KdaInputs {
        q: &q,
        k: &k,
        v: &v,
        gate: &g,
        beta: &beta,
        out_gate: &og,
        edges: Some(&edges),
    };
    let mut best = f64::MAX;
    for _ in 0..5 {
        let pass = ctx.begin().unwrap();
        for (conv, st) in &layers {
            kda(&ctx, &pass, &params, &x, conv, st, &out, 1, None).unwrap();
        }
        let t = std::time::Instant::now();
        pass.commit_wait().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("kda_decode {:8.1} us a layer", best / 34.0 * 1e6);
    // A prompt run of 1 024 rows (prepare, recurrence, output), three layers.
    let n = 1024;
    let big = |rng: &mut StdRng, w: usize| mk(rng, n * w, -1.0, 1.0);
    let (q, k, v) = (big(&mut rng, p), big(&mut rng, p), big(&mut rng, p));
    let (g, beta, og) = (big(&mut rng, p), big(&mut rng, heads), big(&mut rng, p));
    let out = zeros(&ctx, n * p);
    let x = KdaInputs {
        q: &q,
        k: &k,
        v: &v,
        gate: &g,
        beta: &beta,
        out_gate: &og,
        edges: Some(&edges),
    };
    let mut best = f64::MAX;
    for _ in 0..3 {
        let pass = ctx.begin().unwrap();
        for (conv, st) in &layers[..3] {
            kda(&ctx, &pass, &params, &x, conv, st, &out, n, None).unwrap();
        }
        let t = std::time::Instant::now();
        pass.commit_wait().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("kda_prefill 1024 rows {:8.1} us a layer", best / 3.0 * 1e6);
    // With LILY_KERNEL_PROFILE: each kernel alone, per layer of the run.
    let mut by: Vec<(&str, f64)> = Vec::new();
    for p in crate::metal::profile::take() {
        for k in &p.kernels {
            match by.iter_mut().find(|(n, _)| *n == k.name) {
                Some(e) => e.1 += k.gpu_secs,
                None => by.push((k.name, k.gpu_secs)),
            }
        }
    }
    for (name, secs) in by {
        if name != "glm53_kda_decode" {
            println!("kda_   {name:28} {:8.1} us a layer", secs / 9.0 * 1e6);
        }
    }
}

#[test]
fn kda_snapshot_is_the_state_after_its_row() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(68);
    let mut cfg = crate::glm53::Config::glm53_flash();
    let heads = 2;
    cfg.kda_heads = heads;
    let p = heads * 128;
    let c = kda_case(&mut rng, heads);
    let t = |v: &Vec<f32>| tensor(&ctx, v);
    let (qc, kc, vc, al, dt, on) = (
        t(&c.q_conv),
        t(&c.k_conv),
        t(&c.v_conv),
        t(&c.a_log),
        t(&c.dt_bias),
        t(&c.o_norm),
    );
    let params = KdaParams {
        q_conv: &qc,
        k_conv: &kc,
        v_conv: &vc,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let cp = r::KdaParams {
        q_conv: &c.q_conv,
        k_conv: &c.k_conv,
        v_conv: &c.v_conv,
        a_log: &c.a_log,
        dt_bias: &c.dt_bias,
        o_norm: &c.o_norm,
    };
    // Two tokens of history first, so the conv window is not all zero.
    let mut st = r::KdaState::new(&cfg);
    let conv_state = zeros(&ctx, 9 * p);
    let state = zeros(&ctx, heads * 128 * 128);
    for n in [2usize, 3] {
        let q = f32s(&mut rng, n * p, -2.0, 2.0);
        let k = f32s(&mut rng, n * p, -2.0, 2.0);
        let v = f32s(&mut rng, n * p, -2.0, 2.0);
        let g = f32s(&mut rng, n * p, -3.0, 3.0);
        let b = f32s(&mut rng, n * heads, -2.0, 2.0);
        let og = f32s(&mut rng, n * p, -2.0, 2.0);
        let (tq, tk, tv, tg, tb, tog) = (t(&q), t(&k), t(&v), t(&g), t(&b), t(&og));
        let out = zeros(&ctx, n * p);
        let edges = zeros(&ctx, kda_edges_len(n, heads));
        let inputs = KdaInputs {
            q: &tq,
            k: &tk,
            v: &tv,
            gate: &tg,
            beta: &tb,
            out_gate: &tog,
            edges: Some(&edges),
        };
        let snap_conv = zeros(&ctx, 9 * p);
        let snap_state = zeros(&ctx, heads * 128 * 128);
        let snap = (n == 3).then_some(KdaSnapshot {
            row: 0,
            conv: &snap_conv,
            state: &snap_state,
        });
        run(&ctx, |pass| {
            kda(&ctx, pass, &params, &inputs, &conv_state, &state, &out, n, snap)
                .unwrap()
        });
        for j in 0..n {
            let s = j * p..(j + 1) * p;
            r::kda_step(
                &cfg,
                &mut st,
                &cp,
                &q[s.clone()],
                &k[s.clone()],
                &v[s.clone()],
                &g[s.clone()],
                &b[j * heads..(j + 1) * heads],
                &og[s.clone()],
            );
            if n == 3 && j == 0 {
                assert_close(&snap_state.to_f32().unwrap(), &st.s, 1e-4, 1e-3);
                assert_close(&snap_conv.to_f32().unwrap(), &st.conv, 0.0, 0.0);
            }
        }
    }
    assert_close(&state.to_f32().unwrap(), &st.s, 1e-4, 1e-3);
}

#[test]
fn gemm_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(69);
    for (ty, cols, rows, m) in [
        (GgmlType::Q8_0, 4096usize, 200usize, 70usize),
        (GgmlType::Q4K, 4096, 130, 5),
        (GgmlType::Bf16, 1536, 64, 129),
        (GgmlType::F32, 4096, 288, 33),
        (GgmlType::Q8_0, 128, 8192, 64),
    ] {
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let cpu = mat(ty, &raw, rows, cols);
        let x = f32s(&mut rng, m * cols, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let y = zeros(&ctx, m * rows);
        run(&ctx, |p| gemm(&ctx, p, &w, &tx, &y, m).unwrap());
        let g = y.to_f32().unwrap();
        // bf16 operands: compare against the exact product with a tolerance
        // scaled by the row's magnitude.
        for i in 0..m {
            let e = cpu.mv(&x[i * cols..(i + 1) * cols]);
            let scale: f32 =
                (e.iter().map(|v| v * v).sum::<f32>() / rows as f32).sqrt();
            for (n, (a, b)) in g[i * rows..(i + 1) * rows].iter().zip(&e).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-2 * scale + 1e-6,
                    "{ty:?} row {i} col {n}: {a} vs {b}"
                );
            }
        }
    }
}

/// The dense GEMM walks K in 64-deep steps: widths of 64 are computed,
/// widths that are not a multiple of 64 are refused before anything is
/// encoded.
#[test]
fn gemm_k_widths() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(75);
    let (rows, m) = (40, 5);
    for cols in [32usize, 64, 96] {
        let raw = random_blocks(&mut rng, GgmlType::Q8_0, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, rows).unwrap();
        let x = f32s(&mut rng, m * cols, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let y = zeros(&ctx, m * rows);
        let pass = ctx.begin().unwrap();
        let r = gemm(&ctx, &pass, &w, &tx, &y, m);
        pass.commit_wait().unwrap();
        if cols % 64 != 0 {
            assert!(r.is_err(), "K = {cols} accepted");
            continue;
        }
        r.unwrap();
        let cpu = mat(GgmlType::Q8_0, &raw, rows, cols);
        let g = y.to_f32().unwrap();
        for i in 0..m {
            let e = cpu.mv(&x[i * cols..(i + 1) * cols]);
            let scale: f32 =
                (e.iter().map(|v| v * v).sum::<f32>() / rows as f32).sqrt();
            for (a, b) in g[i * rows..(i + 1) * rows].iter().zip(&e) {
                assert!((a - b).abs() <= 2e-2 * scale + 1e-6, "K = {cols}: {a} vs {b}");
            }
        }
    }
}

/// The GEMM over the input's bfloat copy gives the f32-input GEMM's
/// results bit for bit (both round the input to bfloat, to nearest even,
/// and the weights the same way): Q8_0 and Q4_K, row counts within, at
/// and across the 256-row tiles, widths with partial column tiles, and
/// input values at bfloat rounding ties (the ties are the input's).
#[test]
fn gemm_bx_matches_gemm() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(78);
    // Rows within one 256-row tile, around its edge (255, 256, 257) and
    // across several (513, 1 024); widths with partial 64-column tiles.
    for (ty, cols, rows, m) in [
        (GgmlType::Q8_0, 256usize, 96usize, 70usize),
        (GgmlType::Q8_0, 512, 40, 130),
        (GgmlType::Q4K, 512, 72, 65),
        (GgmlType::Q4K, 256, 64, 33),
        (GgmlType::Q8_0, 256, 72, 255),
        (GgmlType::Q8_0, 256, 100, 256),
        (GgmlType::Q8_0, 512, 40, 257),
        (GgmlType::Q8_0, 256, 72, 513),
        (GgmlType::Q8_0, 256, 136, 1024),
        (GgmlType::Q4K, 256, 72, 255),
        (GgmlType::Q4K, 512, 100, 256),
        (GgmlType::Q4K, 256, 40, 257),
        (GgmlType::Q4K, 256, 72, 513),
        (GgmlType::Q4K, 512, 136, 1024),
        (GgmlType::F16, 256, 72, 255),
        (GgmlType::F16, 512, 100, 513),
        // Short runs over 8 192 weight rows (four-subgroup tiles), and
        // their edge at 128 and 129 rows.
        (GgmlType::Q8_0, 256, 8192, 24),
        (GgmlType::Q4K, 256, 8200, 128),
        (GgmlType::Q8_0, 256, 8192, 129),
    ] {
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let mut x = f32s(&mut rng, m * cols, -3.0, 3.0);
        // Every fifth value exactly halfway between two bfloats.
        for v in x.iter_mut().step_by(5) {
            *v = f32::from_bits((v.to_bits() & 0xffff_0000) | 0x8000);
        }
        let tx = tensor(&ctx, &x);
        let xb = Tensor::zeros(&ctx, &[m * cols], DType::BF16).unwrap();
        let (y, yb) = (zeros(&ctx, m * rows), zeros(&ctx, m * rows));
        run(&ctx, |p| {
            gemm(&ctx, p, &w, &tx, &y, m).unwrap();
            to_bf16(&ctx, p, &tx, &xb, m * cols).unwrap();
            p.memory_barrier().unwrap();
            gemm_bx(&ctx, p, &w, &xb, &yb, m).unwrap();
        });
        let (a, b) = (y.to_f32().unwrap(), yb.to_f32().unwrap());
        assert!(
            a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "{ty:?} {rows} x {cols}, {m} rows"
        );
    }
}

/// The batched layout over a bfloat copy (DeepSeek V4's grouped output
/// projection) gives the f32-input batched GEMM's results bit for bit.
#[test]
fn gemm_bx_batched_matches_gemm_with() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(79);
    let (heads, per, cols, m) = (4, 128, 256, 300);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, heads * per * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, heads * per).unwrap();
    let x = f32s(&mut rng, m * heads * cols, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let xb = Tensor::zeros(&ctx, &[m * heads * cols], DType::BF16).unwrap();
    let (y, yb) = (zeros(&ctx, m * heads * per), zeros(&ctx, m * heads * per));
    let l = GemmLayout {
        ldx: heads * cols,
        ldy: heads * per,
        batches: heads,
        x_batch: cols,
        w_batch: per,
        y_batch: per,
    };
    run(&ctx, |p| {
        gemm_with(&ctx, p, &w, &tx, &y, m, per, l).unwrap();
        to_bf16(&ctx, p, &tx, &xb, m * heads * cols).unwrap();
        p.memory_barrier().unwrap();
        gemm_bx_with(&ctx, p, &w, &xb, &yb, m, per, l).unwrap();
    });
    let (a, b) = (y.to_f32().unwrap(), yb.to_f32().unwrap());
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}

/// DeepSeek V4's grouped output projection as it runs (a dispatch a group
/// over offset views of the weights, input and output) gives the batched
/// grid's results bit for bit: eight groups, 300 rows (a partial 256-row
/// tile), the last group's last row at the tensors' ends.
#[test]
fn gemm_bx_split_groups_match_batched() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(83);
    let (groups, per, cols, m) = (8, 128, 256, 300);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, groups * per * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, groups * per).unwrap();
    let x = tensor(&ctx, &f32s(&mut rng, m * groups * cols, -1.0, 1.0));
    let xb = Tensor::zeros(&ctx, &[m * groups * cols], DType::BF16).unwrap();
    let (y, ys) = (zeros(&ctx, m * groups * per), zeros(&ctx, m * groups * per));
    let l = GemmLayout {
        ldx: groups * cols,
        ldy: groups * per,
        batches: groups,
        x_batch: cols,
        w_batch: per,
        y_batch: per,
    };
    let one = GemmLayout { batches: 1, x_batch: 0, w_batch: 0, y_batch: 0, ..l };
    run(&ctx, |p| {
        to_bf16(&ctx, p, &x, &xb, m * groups * cols).unwrap();
        p.memory_barrier().unwrap();
        gemm_bx_with(&ctx, p, &w, &xb, &y, m, per, l).unwrap();
        for g in 0..groups {
            let t = w.t.view(g * per * w.row_bytes, &[per * w.row_bytes]).unwrap();
            let wg = QWeight::new(t, GgmlType::Q8_0, cols, per).unwrap();
            let xv = xb.view(g * cols, &[xb.numel() - g * cols]).unwrap();
            let yv = ys.view(g * per, &[ys.numel() - g * per]).unwrap();
            gemm_bx_with(&ctx, p, &wg, &xv, &yv, m, per, one).unwrap();
        }
    });
    let (a, b) = (y.to_f32().unwrap(), ys.to_f32().unwrap());
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}

#[test]
fn gemm_batched_matches_grouped_gemv() {
    // attn_k_b's shape for 4 heads, 70 rows: per head a 512 x 256 block.
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(70);
    let (heads, per, cols, m) = (4, 512, 256, 70);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, heads * per * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, heads * per).unwrap();
    let cpu = mat(GgmlType::Q8_0, &raw, heads * per, cols);
    let x = f32s(&mut rng, m * heads * cols, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let y = zeros(&ctx, m * heads * per);
    let l = GemmLayout {
        ldx: heads * cols,
        ldy: heads * per,
        batches: heads,
        x_batch: cols,
        w_batch: per,
        y_batch: per,
    };
    run(&ctx, |p| gemm_with(&ctx, p, &w, &tx, &y, m, per, l).unwrap());
    let g = y.to_f32().unwrap();
    for i in 0..m {
        let e = cpu.mv_grouped(&x[i * heads * cols..(i + 1) * heads * cols], per);
        let scale: f32 = (e.iter().map(|v| v * v).sum::<f32>() / e.len() as f32).sqrt();
        for (a, b) in g[i * heads * per..(i + 1) * heads * per].iter().zip(&e) {
            assert!((a - b).abs() <= 2e-2 * scale, "{a} vs {b}");
        }
    }
}

#[test]
fn moe_grouped_matches_per_pair_kernels() {
    moe_grouped_case(8);
}

/// DeepSeek V4's six slots a row: the pair's row is a division, not a
/// shift.
#[test]
fn moe_grouped_six_slots_matches_per_pair_kernels() {
    moe_grouped_case(6);
}

fn moe_grouped_case(used: usize) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(71);
    let (experts, hidden, ff, m) = (12, 512, 256, 70);
    let g_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw = QWeight::from_bytes(&ctx, &g_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let uw = QWeight::from_bytes(&ctx, &u_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
    // Distinct experts per row, skewed so some experts get more than 64 pairs.
    let sel: Vec<u32> = (0..m)
        .flat_map(|i| (0..used).map(move |s| ((i * (s + 1) + s * s) % experts) as u32))
        .collect();
    let wts = f32s(&mut rng, m * used, 0.1, 0.6);
    let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &wts);
    let grouped = tensor(&ctx, &shared);
    let reference = tensor(&ctx, &shared);
    let mid = zeros(&ctx, m * used * ff);
    let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
    run(&ctx, |p| {
        moe_grouped(
            &ctx, p, &gw, &uw, &dw, ff, hidden, &tx, &tsel, &tw, &grouped, &gs, m,
            used, experts, 0.05,
        )
        .unwrap();
        moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, 0.05)
            .unwrap();
        moe_down(
            &ctx,
            p,
            &dw,
            hidden,
            &mid,
            &tsel,
            Some(&reference),
            &reference,
            m,
            used,
        )
        .unwrap();
    });
    let a = grouped.to_f32().unwrap();
    let b = reference.to_f32().unwrap();
    let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 3e-2 * scale, "element {k}: {x} vs {y}");
    }
}

/// The register path of short runs' experts ([`moe_grouped_dot`]): a sparse
/// selection (40 experts, 6 and 9 rows of 8 slots: experts with one, two
/// and more rows, so both paths) against the per-pair f32 kernels, at least
/// as close as the bfloat tiles ([`moe_grouped`]), for 8 and 6 slots.
#[test]
fn moe_grouped_dot_matches_per_pair_kernels() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(80);
    let (experts, hidden, ff) = (40, 512, 256);
    let g_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw = QWeight::from_bytes(&ctx, &g_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let uw = QWeight::from_bytes(&ctx, &u_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    for (m, used) in [(6usize, 8usize), (9, 8), (7, 6)] {
        let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
        // Distinct experts a row; rows 0 and 1 share three, so some experts
        // have two rows and a few three.
        let sel: Vec<u32> = (0..m)
            .flat_map(|i| (0..used).map(move |s| ((i * 5 + s * 7) % experts) as u32))
            .collect();
        for i in 0..m {
            let row = &sel[i * used..(i + 1) * used];
            for a in 0..used {
                assert!(!row[a + 1..].contains(&row[a]), "row {i} repeats an expert");
            }
        }
        let wts = f32s(&mut rng, m * used, 0.1, 0.6);
        let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let tsel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[sel.len()],
            DType::U32,
        )
        .unwrap();
        let tw = tensor(&ctx, &wts);
        let (tiles, dot, reference) =
            (tensor(&ctx, &shared), tensor(&ctx, &shared), tensor(&ctx, &shared));
        let mid = zeros(&ctx, m * used * ff);
        let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
        run(&ctx, |p| {
            moe_grouped(
                &ctx, p, &gw, &uw, &dw, ff, hidden, &tx, &tsel, &tw, &tiles, &gs, m,
                used, experts, 0.05,
            )
            .unwrap();
        });
        run(&ctx, |p| {
            moe_grouped_dot(
                &ctx, p, &gw, &uw, &dw, ff, hidden, &tx, &tsel, &tw, &dot, &gs, m,
                used, experts, 0.05,
            )
            .unwrap();
            moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, 0.05)
                .unwrap();
            moe_down(
                &ctx,
                p,
                &dw,
                hidden,
                &mid,
                &tsel,
                Some(&reference),
                &reference,
                m,
                used,
            )
            .unwrap();
        });
        let b = reference.to_f32().unwrap();
        let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
        let mut err = [0f64; 2];
        for (route, t) in [&tiles, &dot].into_iter().enumerate() {
            let a = t.to_f32().unwrap();
            for (k, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (x - y).abs() <= 3e-2 * scale,
                    "m {m} route {route} element {k}: {x} vs {y}"
                );
                err[route] += f64::from(x - y).powi(2);
            }
        }
        assert!(
            err[1] <= err[0],
            "m {m}: register path error {} above tiles' {}",
            err[1],
            err[0]
        );
    }
}

/// The prefill's grouped routed experts at the model's shapes (gate/up
/// IQ2_XXS 2048 x 4096, down Q2_K 4096 x 2048) over 64 experts, for runs
/// whose pairs per expert match a 256-token and a 4K-token prompt of the
/// model (288 experts, 8 used), and a 24- and 64-token prompt: best of five
/// passes of four calls.
#[test]
#[ignore = "timing only"]
fn moe_grouped_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(72);
    let (experts, hidden, ff, used) = (64, 4096, 2048, 8);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let gw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let uw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let dw = mk(&mut rng, GgmlType::Q2K, ff, experts * hidden);
    // Rows such that pairs per expert = n * 8 / 288 for prompts of 256, 1K
    // and 4K tokens; DeepSeek V4's 2K (6 of 256: 48 a expert). A 24-token
    // prompt picks ~140 of 288 experts, 1.4 pairs each; 64 tokens ~240, 2.1
    // each.
    for (label, m) in [
        ("24-token density", 11),
        ("64-token density", 17),
        ("256-token density", 57),
        ("1K-token density", 228),
        ("4K-token density", 910),
        ("DS4 2K density", 384),
    ] {
        let sel: Vec<u32> = (0..m)
            .flat_map(|i| {
                let start = (i * 37) % experts;
                (0..used).map(move |s| ((start + s * 7) % experts) as u32)
            })
            .collect();
        let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -1.0, 1.0));
        let tsel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[sel.len()],
            DType::U32,
        )
        .unwrap();
        let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
        let out = zeros(&ctx, m * hidden);
        let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
        for (tile_rows, dot) in
            [(MoeTiles::R64, 0), (MoeTiles::R64, MOE_DOT_ROWS), (MoeTiles::R128, 0)]
        {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let pass = ctx.begin().unwrap();
                for _ in 0..4 {
                    moe_grouped_with(
                        &ctx, &pass, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &out,
                        &gs, m, used, experts, 10.0, true, tile_rows, dot,
                    )
                    .unwrap();
                    pass.memory_barrier().unwrap();
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64() / 4.0);
            }
            println!(
                "moe grouped {label:20} m {m:4} tiles {tile_rows:?}{}: {:8.1} us",
                if dot > 0 { " dot" } else { "    " },
                best * 1e6
            );
        }
    }
}

/// DeepSeek V4's prompt-run Q8_0 GEMMs over bfloat input at a run of 2 048
/// rows; the grouped output projection's eight groups as one batched grid
/// and as a dispatch a group (the model's): best of three passes of four
/// calls.
#[test]
#[ignore = "timing only"]
fn gemm_bx_timing_ds4() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(80);
    let m = 2048;
    for (label, cols, rows, batches, split) in [
        ("q_a 1024x4096", 4096usize, 1024usize, 1usize, false),
        ("kv 512x4096", 4096, 512, 1, false),
        ("q_b 32768x1024", 1024, 32768, 1, false),
        ("out_a 8 x 1024x4096 batched", 4096, 8192, 8, false),
        ("out_a 8 dispatches", 4096, 8192, 8, true),
        ("out_b 4096x8192", 8192, 4096, 1, false),
        ("shexp gate 2048x4096", 4096, 2048, 1, false),
        ("shexp down 4096x2048", 2048, 4096, 1, false),
    ] {
        let raw = random_blocks(&mut rng, GgmlType::Q8_0, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, rows).unwrap();
        let xcols = cols * batches;
        let x = tensor(&ctx, &f32s(&mut rng, m * xcols, -1.0, 1.0));
        let n = rows / batches;
        let y = zeros(&ctx, m * rows);
        let xb = Tensor::zeros(&ctx, &[m * xcols], DType::BF16).unwrap();
        run(&ctx, |p| to_bf16(&ctx, p, &x, &xb, m * xcols).unwrap());
        let l = GemmLayout {
            ldx: xcols,
            ldy: rows,
            batches,
            x_batch: cols,
            w_batch: n,
            y_batch: n,
        };
        let mut best = f64::MAX;
        for _ in 0..3 {
            let pass = ctx.begin().unwrap();
            for _ in 0..4 {
                if !split {
                    gemm_bx_with(&ctx, &pass, &w, &xb, &y, m, n, l).unwrap();
                    continue;
                }
                for b in 0..batches {
                    let one = GemmLayout { batches: 1, ..l };
                    let xv = xb.view(b * cols, &[m * xcols - b * cols]).unwrap();
                    let yv = y.view(b * n, &[m * rows - b * n]).unwrap();
                    let t = w.t.view(b * n * w.row_bytes, &[n * w.row_bytes]).unwrap();
                    let wv = QWeight::new(t, GgmlType::Q8_0, cols, n).unwrap();
                    gemm_bx_with(&ctx, &pass, &wv, &xv, &yv, m, n, one).unwrap();
                }
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64() / 4.0);
        }
        println!(
            "gemm_bx {label:28} m {m}: {:8.1} us  {:5.1} TFLOP/s",
            best * 1e6,
            2.0 * (m * rows * cols) as f64 / best / 1e12
        );
    }
}

/// The prefill's dense GEMMs at the model's shapes for a run of 1 024 rows:
/// best of three passes of four calls.
#[test]
#[ignore = "timing only"]
fn gemm_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(73);
    let m = 1024;
    for (label, ty, cols, rows) in [
        ("kda_v Q8_0 8192x4096", GgmlType::Q8_0, 4096usize, 8192usize),
        ("kda_out Q8_0 4096x8192", GgmlType::Q8_0, 8192, 4096),
        ("kda_q Q4_K 8192x4096", GgmlType::Q4K, 4096, 8192),
        ("f_b Q8_0 8192x128", GgmlType::Q8_0, 128, 8192),
        ("shexp Q8_0 2048x4096", GgmlType::Q8_0, 4096, 2048),
    ] {
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
        let y = zeros(&ctx, m * rows);
        let xb = Tensor::zeros(&ctx, &[m * cols], DType::BF16).unwrap();
        run(&ctx, |p| to_bf16(&ctx, p, &x, &xb, m * cols).unwrap());
        for bx in [false, true] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let pass = ctx.begin().unwrap();
                for _ in 0..4 {
                    if bx {
                        gemm_bx(&ctx, &pass, &w, &xb, &y, m).unwrap();
                    } else {
                        gemm(&ctx, &pass, &w, &x, &y, m).unwrap();
                    }
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64() / 4.0);
            }
            println!(
                "gemm {label:24} m {m}{}: {:8.1} us  {:5.1} TFLOP/s",
                if bx { " bx" } else { "" },
                best * 1e6,
                2.0 * (m * rows * cols) as f64 / best / 1e12
            );
        }
    }
}

/// The fused gate/up/SwiGLU expert GEMM gives the separate GEMMs' and
/// SwiGLU's values bit for bit, with tiles of 64 rows and of 128 (the
/// same again): experts with 16, 32, 48, 64, 65, 100 and 129 pairs (one
/// to four working subgroups, a second tile), small ones, unused ones;
/// the clamp off, tight and loose.
#[test]
fn moe_grouped_fused_matches_separate() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(74);
    let (experts, hidden, ff, used, m) = (12, 1024, 512, 8, 60);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let gw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let uw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let dw = mk(&mut rng, GgmlType::Q2K, ff, experts * hidden);
    // 480 pairs in a scrambled order.
    let counts = [16usize, 32, 48, 64, 65, 0, 129, 100, 16, 10, 0, 0];
    assert_eq!(counts.iter().sum::<usize>(), m * used);
    let mut sel: Vec<u32> = counts
        .iter()
        .enumerate()
        .flat_map(|(e, &c)| std::iter::repeat_n(e as u32, c))
        .collect();
    for i in (1..sel.len()).rev() {
        sel.swap(i, rng.gen_range(0..=i));
    }
    let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -2.0, 2.0));
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
    let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
    for limit in [0.0f32, 1e-3, 10.0] {
        let mut got = Vec::new();
        for (fused, tile_rows) in [
            (true, MoeTiles::R64),
            (false, MoeTiles::R64),
            (true, MoeTiles::R128),
            (false, MoeTiles::R128),
        ] {
            let out = zeros(&ctx, m * hidden);
            run(&ctx, |p| {
                moe_grouped_with(
                    &ctx, p, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &out, &gs, m,
                    used, experts, limit, fused, tile_rows, 0,
                )
                .unwrap()
            });
            got.push((gs.mid.to_f32().unwrap(), out.to_f32().unwrap()));
        }
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        for k in 1..got.len() {
            assert_eq!(bits(&got[0].0), bits(&got[k].0), "mid, limit {limit}, {k}");
            assert_eq!(bits(&got[0].1), bits(&got[k].1), "out, limit {limit}, {k}");
        }
    }
}

/// DeepSeek V4's prompt scratch for Q2_K gate/up experts (which its schema
/// accepts): the model picks the separate route's scratch, and the grouped
/// experts over it give the per-pair kernels' result; the compact fused
/// scratch is refused for that route instead of written past.
#[test]
fn deepseek4_q2_k_gate_up_scratch() {
    use crate::deepseek4::model::{moe_group_scratch, separate_gate_up};
    assert!(separate_gate_up([GgmlType::Iq2Xxs, GgmlType::Q2K]));
    assert!(!separate_gate_up([GgmlType::Iq2Xxs, GgmlType::Iq2Xxs]));
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(82);
    let (experts, hidden, ff, used, m) = (12, 512, 256, 6, 40);
    let mk = |rng: &mut StdRng, cols, rows| {
        let raw = random_blocks(rng, GgmlType::Q2K, rows * cols);
        QWeight::from_bytes(&ctx, &raw, GgmlType::Q2K, cols, rows).unwrap()
    };
    let gw = mk(&mut rng, hidden, experts * ff);
    let uw = mk(&mut rng, hidden, experts * ff);
    let dw = mk(&mut rng, ff, experts * hidden);
    let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -2.0, 2.0));
    let sel: Vec<u32> = (0..m * used).map(|s| ((s * 5 + s / used) % experts) as u32).collect();
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
    let dims = (used, experts, ff, hidden);
    let gs = moe_group_scratch(&ctx, m, dims, separate_gate_up([GgmlType::Q2K])).unwrap();
    let compact = moe_group_scratch(&ctx, m, dims, false).unwrap();
    let (out, refused) = (zeros(&ctx, m * hidden), zeros(&ctx, m * hidden));
    let (mid, want) = (zeros(&ctx, m * used * ff), zeros(&ctx, m * hidden));
    let mut err = None;
    run(&ctx, |p| {
        moe_grouped_tiles(
            &ctx, p, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &out, &gs, m, used,
            experts, 10.0, MoeTiles::R64,
        )
        .unwrap();
        err = moe_grouped_tiles(
            &ctx, p, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &refused, &compact, m,
            used, experts, 10.0, MoeTiles::R64,
        )
        .err();
        moe_gate_up(&ctx, p, &gw, &uw, ff, &x, &tsel, &tw, &mid, m, used, 10.0).unwrap();
        p.memory_barrier().unwrap();
        moe_down(&ctx, p, &dw, hidden, &mid, &tsel, None, &want, m, used).unwrap();
    });
    assert!(err.is_some(), "the compact scratch must be refused for Q2_K gate/up");
    let (a, b) = (out.to_f32().unwrap(), want.to_f32().unwrap());
    let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 3e-2 * scale, "element {k}: {x} vs {y}");
    }
}

/// The grouped route of Q2_K gate/up experts (separate GEMMs and SwiGLU)
/// against the per-pair kernels.
#[test]
fn moe_grouped_q2_k_gate_up_matches_per_pair_kernels() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(76);
    let (experts, hidden, ff, used, m) = (12, 512, 256, 8, 70);
    let g_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw =
        QWeight::from_bytes(&ctx, &g_raw, GgmlType::Q2K, hidden, experts * ff).unwrap();
    let uw =
        QWeight::from_bytes(&ctx, &u_raw, GgmlType::Q2K, hidden, experts * ff).unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
    let sel: Vec<u32> = (0..m)
        .flat_map(|i| (0..used).map(move |s| ((i * (s + 1) + s * s) % experts) as u32))
        .collect();
    let wts = f32s(&mut rng, m * used, 0.1, 0.6);
    let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &wts);
    let grouped = tensor(&ctx, &shared);
    let reference = tensor(&ctx, &shared);
    let mid = zeros(&ctx, m * used * ff);
    let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
    run(&ctx, |p| {
        moe_grouped(
            &ctx, p, &gw, &uw, &dw, ff, hidden, &tx, &tsel, &tw, &grouped, &gs, m,
            used, experts, 0.05,
        )
        .unwrap();
        moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, 0.05)
            .unwrap();
        moe_down(
            &ctx,
            p,
            &dw,
            hidden,
            &mid,
            &tsel,
            Some(&reference),
            &reference,
            m,
            used,
        )
        .unwrap();
    });
    let a = grouped.to_f32().unwrap();
    let b = reference.to_f32().unwrap();
    let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 3e-2 * scale, "element {k}: {x} vs {y}");
    }
}

/// The decode GEMVs at one and more rows (the verify pass's m = 2): the
/// step's main shapes, best of five passes of `LILY_GLM53_REPEAT` calls over
/// four weight copies; rows from `LILY_GLM53_ROWS` (default "1,2").
#[test]
#[ignore = "timing only"]
fn gemv_rows_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(68);
    let repeat: usize = std::env::var("LILY_GLM53_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let rows_list: Vec<usize> = std::env::var("LILY_GLM53_ROWS")
        .unwrap_or_else(|_| "1,2".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let shapes = [
        ("kda_v Q8_0 8192x4096", GgmlType::Q8_0, 4096, 8192),
        ("kda_out Q8_0 4096x8192", GgmlType::Q8_0, 8192, 4096),
        ("shexp Q8_0 2048x4096", GgmlType::Q8_0, 4096, 2048),
        ("q_a Q8_0 1536x4096", GgmlType::Q8_0, 4096, 1536),
        ("f_b Q8_0 8192x128", GgmlType::Q8_0, 128, 8192),
        ("kda_q Q4_K 8192x4096", GgmlType::Q4K, 4096, 8192),
        ("router F32 288x4096", GgmlType::F32, 4096, 288),
    ];
    for (label, ty, cols, rows) in shapes {
        let ws: Vec<QWeight> = (0..4)
            .map(|_| {
                let raw = random_blocks(&mut rng, ty, rows * cols);
                QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
            })
            .collect();
        let mut line = format!("{label:24}");
        for &m in &rows_list {
            let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
            let y = zeros(&ctx, m * rows);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let pass = ctx.begin().unwrap();
                for r in 0..repeat {
                    gemv(&ctx, &pass, &ws[r % 4], &x, &y, m).unwrap();
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            line += &format!("  m={m} {:6.1} us", best / repeat as f64 * 1e6);
        }
        println!("{line}");
    }
}

/// The routed experts' kernels for two rows (a verify pass) sharing
/// `shared` of their 8 experts, against one row: model shapes as
/// [`moe_timing`]. Prints us a layer for gate/up and down.
#[test]
#[ignore = "timing only"]
fn moe_rows_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(69);
    let repeat: usize = std::env::var("LILY_GLM53_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let (experts, hidden, ff, used) = (32, 4096, 2048, 8);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let layers: Vec<(QWeight, QWeight, QWeight)> = (0..4)
        .map(|_| {
            (
                mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff),
                mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff),
                mk(&mut rng, GgmlType::Q2K, ff, experts * hidden),
            )
        })
        .collect();
    let x = tensor(&ctx, &f32s(&mut rng, 2 * hidden, -1.0, 1.0));
    let tw = tensor(&ctx, &vec![0.3f32; 2 * used]);
    let mid = zeros(&ctx, 2 * used * ff);
    let out = zeros(&ctx, 2 * hidden);
    for (m, shared) in [(1usize, 0usize), (2, 0), (2, 2), (2, 4), (2, 8)] {
        // Row 0 takes experts 0, 3, 6, ..; row 1 the first `shared` of
        // those (in another order) and then experts of its own.
        let mut sel: Vec<u32> = (0..used as u32).map(|i| i * 3).collect();
        let mut second: Vec<u32> = (0..shared as u32).rev().map(|i| i * 3).collect();
        second.extend((0..(used - shared) as u32).map(|i| i * 3 + 1));
        sel.extend(second);
        let tsel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[sel.len()],
            DType::U32,
        )
        .unwrap();
        let mut line = format!("moe m={m} shared {shared}:");
        for part in 0..2 {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let pass = ctx.begin().unwrap();
                for r in 0..repeat {
                    let (g, u, d) = &layers[r % 4];
                    if part == 0 {
                        moe_gate_up(
                            &ctx, &pass, g, u, ff, &x, &tsel, &tw, &mid, m, used, 10.0,
                        )
                        .unwrap();
                    } else {
                        moe_down(
                            &ctx, &pass, d, hidden, &mid, &tsel, None, &out, m, used,
                        )
                        .unwrap();
                    }
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            let name = if part == 0 { "gate_up" } else { "down" };
            line += &format!("  {name} {:6.1} us", best / repeat as f64 * 1e6);
        }
        println!("{line}");
    }
}

/// The dense bfloat-input GEMM at a short prompt's rows (`LILY_GLM53_ROWS`,
/// default 8,24,64): us a call, eight calls a pass (independent: they
/// overlap as a level's do).
#[test]
#[ignore = "timing only"]
fn gemm_small_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(74);
    let rows_list: Vec<usize> = std::env::var("LILY_GLM53_ROWS")
        .unwrap_or_else(|_| "8,24,64".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for (label, ty, cols, rows) in [
        ("kda_v Q8_0 8192x4096", GgmlType::Q8_0, 4096usize, 8192usize),
        ("kda_out Q8_0 4096x8192", GgmlType::Q8_0, 8192, 4096),
        ("kda_q Q4_K 8192x4096", GgmlType::Q4K, 4096, 8192),
        ("shexp Q8_0 2048x4096", GgmlType::Q8_0, 4096, 2048),
    ] {
        let ws: Vec<QWeight> = (0..4)
            .map(|_| {
                let raw = random_blocks(&mut rng, ty, rows * cols);
                QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
            })
            .collect();
        let bytes = ws[0].t.byte_len() as f64;
        for &m in &rows_list {
            let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
            let xb = Tensor::zeros(&ctx, &[m * cols], DType::BF16).unwrap();
            run(&ctx, |p| to_bf16(&ctx, p, &x, &xb, m * cols).unwrap());
            let ys: Vec<Tensor> = (0..8).map(|_| zeros(&ctx, m * rows)).collect();
            let mut best = f64::MAX;
            for _ in 0..4 {
                let pass = ctx.begin().unwrap();
                for (r, y) in ys.iter().enumerate() {
                    gemm_bx(&ctx, &pass, &ws[r % 4], &xb, y, m).unwrap();
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64() / 8.0);
            }
            println!(
                "gemm small {label:24} m {m:3}: {:7.1} us {:4.0} GB/s",
                best * 1e6,
                bytes / best / 1e9
            );
        }
    }
}
