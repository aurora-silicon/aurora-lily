//! Qwen3.6 kernels at small shapes against f64 CPU versions of llama.cpp's
//! arithmetic (ggml's CPU ops for qwen35moe); the GDN reference keeps its
//! whole recurrence in f64, the rest takes `crate::refops`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::gguf::dequantize;
use crate::kernels::glm53::QWeight;
use crate::refops;

fn ctx() -> MetalContext {
    MetalContext::new().expect("metal context")
}

fn f32s(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn tensor(ctx: &MetalContext, v: &[f32]) -> Tensor {
    Tensor::from_f32(ctx, v, &[v.len()]).expect("tensor")
}

fn zeros(ctx: &MetalContext, n: usize, d: DType) -> Tensor {
    Tensor::zeros(ctx, &[n], d).expect("zeros")
}

fn u32s(ctx: &MetalContext, v: &[u32]) -> Tensor {
    Tensor::from_bytes(ctx, bytemuck::cast_slice(v), &[v.len()], DType::U32)
        .expect("u32")
}

fn run(ctx: &MetalContext, f: impl FnOnce(&ComputePass<'_>)) {
    let pass = ctx.begin().expect("pass");
    f(&pass);
    pass.commit_wait().expect("commit");
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The GDN of `m` rows on the CPU: conv window `conv` ([3][C]) and state
/// `s` ([H][j][i]) updated in place; returns o ([m][H][128]) and y.
#[allow(clippy::too_many_arguments)]
fn gdn_ref(
    qkv: &[f32],
    alpha: &[f32],
    beta: &[f32],
    z: &[f32],
    w: &[f32],
    ssm_a: &[f32],
    dt: &[f32],
    norm: &[f32],
    conv: &mut [f32],
    s: &mut [f64],
    m: usize,
    hk: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    const D: usize = 128;
    let h = ssm_a.len();
    let ch = (2 * hk + h) * D;
    let mut o = vec![0f64; m * h * D];
    let mut y = vec![0f64; m * h * D];
    for t in 0..m {
        let mut c = vec![0f64; ch];
        for x in 0..ch {
            let win = [conv[x], conv[ch + x], conv[2 * ch + x], qkv[t * ch + x]];
            let acc: f64 = (0..4).map(|k| win[k] as f64 * w[x * 4 + k] as f64).sum();
            c[x] = silu(acc);
            conv[x] = conv[ch + x];
            conv[ch + x] = conv[2 * ch + x];
            conv[2 * ch + x] = qkv[t * ch + x];
        }
        for vh in 0..h {
            let kh = vh % hk;
            let l2 = |v: &[f64]| {
                let n: f64 = v.iter().map(|x| x * x).sum();
                v.iter().map(|x| x / (n + eps).sqrt()).collect::<Vec<_>>()
            };
            let q = l2(&c[kh * D..(kh + 1) * D]);
            let k = l2(&c[(hk + kh) * D..(hk + kh + 1) * D]);
            let v = &c[(2 * hk + vh) * D..(2 * hk + vh + 1) * D];
            let x = alpha[t * h + vh] as f64 + dt[vh] as f64;
            let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
            let decay = (sp * ssm_a[vh] as f64).exp();
            let b = sigmoid(beta[t * h + vh] as f64);
            let st = &mut s[vh * D * D..(vh + 1) * D * D];
            for j in 0..D {
                let col = &mut st[j * D..(j + 1) * D];
                col.iter_mut().for_each(|e| *e *= decay);
                let kv: f64 = col.iter().zip(&k).map(|(a, b)| a * b).sum();
                let d = (v[j] - kv) * b;
                col.iter_mut().zip(&k).for_each(|(e, kk)| *e += kk * d);
                o[(t * h + vh) * D + j] =
                    col.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>()
                        / (D as f64).sqrt();
            }
            let oh = &o[(t * h + vh) * D..(t * h + vh + 1) * D];
            let ms = oh.iter().map(|x| x * x).sum::<f64>() / D as f64;
            for j in 0..D {
                let i = (t * h + vh) * D + j;
                y[i] = oh[j] / (ms + eps).sqrt() * norm[j] as f64 * silu(z[i] as f64);
            }
        }
    }
    (o, y)
}

#[test]
fn gdn_matches_cpu_with_snapshot() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(36);
    let (hk, h, m, eps) = (2, 4, 5, 1e-6);
    let ch = (2 * hk + h) * GDN_DIM;
    let inner = h * GDN_DIM;
    let qkv = f32s(&mut rng, m * ch, -1.0, 1.0);
    let alpha = f32s(&mut rng, m * h, -2.0, 2.0);
    let beta = f32s(&mut rng, m * h, -2.0, 2.0);
    let z = f32s(&mut rng, m * inner, -2.0, 2.0);
    let w = f32s(&mut rng, 4 * ch, -0.5, 0.5);
    let ssm_a = f32s(&mut rng, h, -2.0, -0.1);
    let dt = f32s(&mut rng, h, -1.0, 1.0);
    let norm = f32s(&mut rng, GDN_DIM, 0.5, 1.5);
    let conv0 = f32s(&mut rng, 3 * ch, -1.0, 1.0);
    let s0 = f32s(&mut rng, h * GDN_DIM * GDN_DIM, -0.1, 0.1);

    let (mut conv_r, mut s_r) =
        (conv0.clone(), s0.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let (o_r, y_r) = gdn_ref(
        &qkv,
        &alpha,
        &beta,
        &z,
        &w,
        &ssm_a,
        &dt,
        &norm,
        &mut conv_r,
        &mut s_r,
        m,
        hk,
        eps,
    );
    // The state after row 1 (the snapshot's).
    let (mut conv_2, mut s_2) =
        (conv0.clone(), s0.iter().map(|&x| x as f64).collect::<Vec<_>>());
    gdn_ref(
        &qkv[..2 * ch],
        &alpha[..2 * h],
        &beta[..2 * h],
        &z[..2 * inner],
        &w,
        &ssm_a,
        &dt,
        &norm,
        &mut conv_2,
        &mut s_2,
        2,
        hk,
        eps,
    );

    let t = |v: &[f32]| tensor(&ctx, v);
    let (wt, at, dtt, nt) = (t(&w), t(&ssm_a), t(&dt), t(&norm));
    let p = GdnParams { conv_w: &wt, ssm_a: &at, dt_bias: &dtt, norm: &nt };
    let (conv, s) = (t(&conv0), t(&s0));
    let st = GdnState { conv: &conv, s: &s };
    let (sconv, ss) =
        (zeros(&ctx, 3 * ch, DType::F32), zeros(&ctx, s0.len(), DType::F32));
    let snap = GdnSnapshot { row: 1, conv: &sconv, s: &ss };
    let (qkv_t, a_t, b_t, z_t) = (t(&qkv), t(&alpha), t(&beta), t(&z));
    let c = zeros(&ctx, m * ch, DType::F32);
    let o = zeros(&ctx, m * inner, DType::F32);
    let y = zeros(&ctx, m * inner, DType::F32);
    run(&ctx, |pass| {
        gdn(
            &ctx,
            pass,
            &p,
            &st,
            Some(&snap),
            &qkv_t,
            &a_t,
            &b_t,
            &z_t,
            &c,
            &o,
            &y,
            m,
            hk,
            eps as f32,
        )
        .unwrap()
    });
    let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    assert_close(&o.to_f32().unwrap(), &f(&o_r), 2e-5, 2e-4);
    assert_close(&y.to_f32().unwrap(), &f(&y_r), 2e-4, 2e-3);
    assert_close(&s.to_f32().unwrap(), &f(&s_r), 2e-5, 2e-4);
    assert_close(&conv.to_f32().unwrap(), &conv_r, 0.0, 0.0);
    assert_close(&ss.to_f32().unwrap(), &f(&s_2), 2e-5, 2e-4);
    assert_close(&sconv.to_f32().unwrap(), &conv_2, 0.0, 0.0);
}

#[test]
fn attn_prep_and_gate_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(37);
    let (m, nq, nkv, d, rot, base, eps, pos0) =
        (3, 4, 2, HEAD_DIM, 64, 1e7f32, 1e-6, 1000);
    let qg = f32s(&mut rng, m * nq * 2 * d, -2.0, 2.0);
    let kx = f32s(&mut rng, m * nkv * d, -2.0, 2.0);
    let vx = f32s(&mut rng, m * nkv * d, -2.0, 2.0);
    let qn = f32s(&mut rng, d, 0.5, 1.5);
    let kn = f32s(&mut rng, d, 0.5, 1.5);
    // CPU: RMSNorm, NeoX RoPE with ggml's cumulative angle.
    let prep = |x: &[f32], w: &[f32], pos: usize| -> Vec<f32> {
        let ms = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / d as f64;
        let n: Vec<f64> = x
            .iter()
            .zip(w)
            .map(|(&v, &w)| v as f64 / (ms + eps).sqrt() * w as f64)
            .collect();
        let mut r = n.clone();
        let scale = (base as f64).powf(-2.0 / rot as f64);
        let mut theta = pos as f64;
        for i in 0..rot / 2 {
            let (c, s) = (theta.cos(), theta.sin());
            r[i] = n[i] * c - n[i + rot / 2] * s;
            r[i + rot / 2] = n[i] * s + n[i + rot / 2] * c;
            theta *= scale;
        }
        r.iter().map(|&v| v as f32).collect()
    };
    let mut q_r = Vec::new();
    let mut k_r = Vec::new();
    for t in 0..m {
        for hh in 0..nq {
            let at = (t * nq + hh) * 2 * d;
            q_r.extend(prep(&qg[at..at + d], &qn, pos0 + t));
        }
        for hh in 0..nkv {
            let at = (t * nkv + hh) * d;
            k_r.extend(prep(&kx[at..at + d], &kn, pos0 + t));
        }
    }
    let t = |v: &[f32]| tensor(&ctx, v);
    let (qg_t, k_t, v_t, qn_t, kn_t) = (t(&qg), t(&kx), t(&vx), t(&qn), t(&kn));
    let qo = zeros(&ctx, m * nq * d, DType::BF16);
    let ko = zeros(&ctx, m * nkv * d, DType::BF16);
    let vo = zeros(&ctx, m * nkv * d, DType::BF16);
    let y = zeros(&ctx, m * nq * d, DType::F32);
    run(&ctx, |pass| {
        attn_prep(
            &ctx, pass, &qg_t, &k_t, &v_t, &qn_t, &kn_t, &qo, &ko, &vo, m, nq, nkv,
            pos0, rot, base, eps as f32,
        )
        .unwrap();
        pass.memory_barrier().unwrap();
        // The gate over the bfloat q as if it were the attention's output.
        attn_gate(&ctx, pass, &qo, &qg_t, &y, m, nq).unwrap();
    });
    // bfloat outputs: relative 2^-8.
    assert_close(&qo.to_f32().unwrap(), &q_r, 1e-3, 8e-3);
    assert_close(&ko.to_f32().unwrap(), &k_r, 1e-3, 8e-3);
    assert_close(&vo.to_f32().unwrap(), &vx, 1e-3, 8e-3);
    let qb = qo.to_f32().unwrap();
    let y_r: Vec<f32> = (0..m * nq * d)
        .map(|i| {
            let (tt, hh, dd) = (i / (nq * d), i / d % nq, i % d);
            qb[i] * sigmoid(qg[(tt * nq + hh) * 2 * d + d + dd] as f64) as f32
        })
        .collect();
    assert_close(&y.to_f32().unwrap(), &y_r, 1e-6, 1e-5);
}

#[test]
fn router_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(38);
    let (m, e, k, kk) = (6, 256, 8, 512);
    let w = f32s(&mut rng, e * kk, -0.1, 0.1);
    let wg = f32s(&mut rng, kk, -0.1, 0.1);
    let x = f32s(&mut rng, m * kk, -1.0, 1.0);
    let row = |v: &[f32], r: usize| v[r * kk..(r + 1) * kk].to_vec();
    let mut logits: Vec<f32> =
        (0..m * e).map(|i| refops::dot(&row(&w, i % e), &row(&x, i / e))).collect();
    let gl: Vec<f32> = (0..m).map(|t| refops::dot(&wg, &row(&x, t))).collect();
    let rw = QWeight::from_bytes(&ctx, bytemuck::cast_slice(&w), GgmlType::F32, kk, e)
        .unwrap();
    let rg = QWeight::from_bytes(&ctx, bytemuck::cast_slice(&wg), GgmlType::F32, kk, 1)
        .unwrap();
    let xt = tensor(&ctx, &x);
    let lt = zeros(&ctx, m * e, DType::F32);
    let gt = zeros(&ctx, m, DType::F32);
    run(&ctx, |pass| router_logits(&ctx, pass, &rw, &rg, &xt, &lt, &gt, m).unwrap());
    assert_close(&lt.to_f32().unwrap(), &logits, 1e-5, 1e-4);
    assert_close(&gt.to_f32().unwrap(), &gl, 1e-5, 1e-4);

    // A tie in row 0 (lower index wins), then the selection.
    logits[3] = logits[1] + 50.0;
    logits[7] = logits[3];
    let lt = tensor(&ctx, &logits);
    let sel = zeros(&ctx, m * k, DType::U32);
    let wts = zeros(&ctx, m * k, DType::F32);
    run(&ctx, |pass| router(&ctx, pass, &lt, &sel, &wts, m, e, k).unwrap());
    let (sel, wts) = (sel.to_u32().unwrap(), wts.to_f32().unwrap());
    for t in 0..m {
        let row = &logits[t * e..(t + 1) * e];
        let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let z: f64 = row.iter().map(|&l| (l as f64 - mx).exp()).sum();
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        let top = &idx[..k];
        let ps: Vec<f64> =
            top.iter().map(|&i| (row[i] as f64 - mx).exp() / z).collect();
        let sum: f64 = ps.iter().sum();
        assert_eq!(
            &sel[t * k..(t + 1) * k],
            &top.iter().map(|&i| i as u32).collect::<Vec<_>>()[..]
        );
        let want: Vec<f32> = ps.iter().map(|p| (p / sum) as f32).collect();
        assert_close(&wts[t * k..(t + 1) * k], &want, 1e-6, 1e-5);
    }
    assert_eq!(&sel[..2], &[3, 7]);

    // A non-finite row (NaN logits) still selects experts below E, with
    // zero weights; the rows beside it are unaffected.
    let mut bad = logits.clone();
    bad[e..2 * e].iter_mut().for_each(|l| *l = f32::NAN);
    bad[2 * e + 5] = f32::INFINITY;
    let lt = tensor(&ctx, &bad);
    let sel2 = zeros(&ctx, m * k, DType::U32);
    let wts2 = zeros(&ctx, m * k, DType::F32);
    run(&ctx, |pass| router(&ctx, pass, &lt, &sel2, &wts2, m, e, k).unwrap());
    let (sel2, wts2) = (sel2.to_u32().unwrap(), wts2.to_f32().unwrap());
    for t in 1..3 {
        let row = &sel2[t * k..(t + 1) * k];
        assert!(row.iter().all(|&x| (x as usize) < e), "row {t}: {row:?}");
        let mut uniq = row.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), k, "row {t}: {row:?}");
        assert!(wts2[t * k..(t + 1) * k].iter().all(|&w| w == 0.0), "row {t}");
    }
    assert_eq!(&sel2[..k], &sel[..k]);
    assert_eq!(&sel2[3 * k..], &sel[3 * k..]);
}

#[test]
fn embed_rows_match_dequant() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(39);
    let (cols, rows) = (512, 10);
    for ty in [GgmlType::Q8_0, GgmlType::Q4K] {
        let nb = cols * rows / ty.block_values();
        let mut raw: Vec<u8> =
            (0..nb * ty.block_bytes()).map(|_| rng.r#gen()).collect();
        for blk in raw.chunks_exact_mut(ty.block_bytes()) {
            blk[0..2]
                .copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
            if ty == GgmlType::Q4K {
                blk[2..4].copy_from_slice(
                    &half::f16::from_f32(0.005).to_bits().to_le_bytes(),
                );
            }
        }
        let mut all = vec![0f32; cols * rows];
        dequantize(ty, &raw, &mut all).unwrap();
        let table = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let toks = [7u32, 0, 9, 7];
        let tt = u32s(&ctx, &toks);
        let out = zeros(&ctx, toks.len() * cols, DType::F32);
        run(&ctx, |pass| embed(&ctx, pass, &table, &tt, &out, toks.len()).unwrap());
        let want: Vec<f32> = toks
            .iter()
            .flat_map(|&t| all[t as usize * cols..(t as usize + 1) * cols].to_vec())
            .collect();
        assert_close(&out.to_f32().unwrap(), &want, 0.0, 0.0);
    }
}

#[test]
fn ffn_residual_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(40);
    let (m, h) = (3, 512);
    let x = f32s(&mut rng, m * h, -1.0, 1.0);
    let moe = f32s(&mut rng, m * h, -1.0, 1.0);
    let sh = f32s(&mut rng, m * h, -1.0, 1.0);
    let g = f32s(&mut rng, m, -3.0, 3.0);
    let want: Vec<f32> = (0..m * h)
        .map(|i| {
            let s = sh[i] * refops::sigmoid(g[i / h]);
            (moe[i] + s) + x[i]
        })
        .collect();
    let xt = tensor(&ctx, &x);
    let (mt, st, gt) = (tensor(&ctx, &moe), tensor(&ctx, &sh), tensor(&ctx, &g));
    run(&ctx, |pass| ffn_residual(&ctx, pass, &xt, &mt, &st, &gt, m, h).unwrap());
    assert_close(&xt.to_f32().unwrap(), &want, 1e-6, 1e-6);
}

/// Random Q8_0 or Q4_K blocks with small f16 scales.
fn random_blocks(rng: &mut StdRng, ty: GgmlType, values: usize) -> Vec<u8> {
    let nb = values / ty.block_values();
    let mut b: Vec<u8> = (0..nb * ty.block_bytes()).map(|_| rng.r#gen()).collect();
    let f16 = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
    for blk in b.chunks_exact_mut(ty.block_bytes()) {
        blk[0..2].copy_from_slice(&f16(rng.gen_range(0.001..0.01)));
        if ty == GgmlType::Q4K {
            blk[2..4].copy_from_slice(&f16(rng.gen_range(0.001..0.01)));
        }
    }
    b
}

#[test]
fn moe_grouped_q4_k_q8_0_match_per_pair_kernels() {
    use crate::kernels::glm53::{MoeGroupScratch, moe_down, moe_gate_up, moe_grouped};
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(41);
    let (experts, hidden, ff, used) = (12, 512, 512, 8);
    // 70 rows: 560 pairs, the 64-row tiles; 80 rows (640 pairs, past 48 an
    // expert) with every row's slot 0 on expert 0: the 128-row tiles, one
    // of them holding 80 pairs.
    for (ty, m) in [
        (GgmlType::Q4K, 70),
        (GgmlType::Q8_0, 70),
        (GgmlType::Q4K, 80),
        (GgmlType::Q8_0, 80),
    ] {
        let q = |rng: &mut StdRng, cols: usize, rows: usize| {
            let raw = random_blocks(rng, ty, cols * rows);
            QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
        };
        let gw = q(&mut rng, hidden, experts * ff);
        let uw = q(&mut rng, hidden, experts * ff);
        let dw = q(&mut rng, ff, experts * hidden);
        let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
        let sel: Vec<u32> = (0..m)
            .flat_map(|i| {
                (0..used).map(move |s| {
                    if m == 80 && s == 0 {
                        0
                    } else {
                        ((i * (s + 1) + s * s) % experts) as u32
                    }
                })
            })
            .collect();
        let wts = f32s(&mut rng, m * used, 0.1, 0.6);
        let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
        let (tx, tsel, tw) = (tensor(&ctx, &x), u32s(&ctx, &sel), tensor(&ctx, &wts));
        let grouped = tensor(&ctx, &shared);
        let reference = tensor(&ctx, &shared);
        let mid = zeros(&ctx, m * used * ff, DType::F32);
        let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
        run(&ctx, |p| {
            moe_grouped(
                &ctx, p, &gw, &uw, &dw, ff, hidden, &tx, &tsel, &tw, &grouped, &gs, m,
                used, experts, 0.0,
            )
            .unwrap();
            moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, 0.0)
                .unwrap();
            p.memory_barrier().unwrap();
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
        // bfloat operands in the grouped GEMMs.
        let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).abs() <= 3e-2 * scale, "{ty:?} element {k}: {x} vs {y}");
        }
    }
}

#[test]
fn moe_down_matches_glm53() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(43);
    let (experts, hidden, used, m) = (10, 512, 8, 3);
    for (ty, ff) in [(GgmlType::Q4K, 512), (GgmlType::Q8_0, 512), (GgmlType::Q4K, 256)]
    {
        let raw = random_blocks(&mut rng, ty, experts * hidden * ff);
        let dw = QWeight::from_bytes(&ctx, &raw, ty, ff, experts * hidden).unwrap();
        let mid = tensor(&ctx, &f32s(&mut rng, m * used * ff, -1.0, 1.0));
        let sel: Vec<u32> =
            (0..m * used).map(|p| ((p * 7 + 3) % experts) as u32).collect();
        let tsel = u32s(&ctx, &sel);
        let (a, b) =
            (zeros(&ctx, m * hidden, DType::F32), zeros(&ctx, m * hidden, DType::F32));
        run(&ctx, |p| {
            moe_down(&ctx, p, &dw, hidden, &mid, &tsel, &a, m, used).unwrap();
            crate::kernels::glm53::moe_down(
                &ctx, p, &dw, hidden, &mid, &tsel, None, &b, m, used,
            )
            .unwrap();
        });
        assert_close(&a.to_f32().unwrap(), &b.to_f32().unwrap(), 1e-5, 1e-4);
        // The wide kernel (8 slots of 512) against the lane-walk one: the
        // same sums regrouped.
        if ff == 512 {
            let c = zeros(&ctx, m * hidden, DType::F32);
            run(&ctx, |p| {
                moe_down_with(&ctx, p, &dw, hidden, &mid, &tsel, &c, m, used, false)
                    .unwrap()
            });
            assert_close(&a.to_f32().unwrap(), &c.to_f32().unwrap(), 1e-5, 1e-5);
        }
    }
}

/// The fused residual + RMSNorm kernels against glm53::add / ffn_residual
/// then glm53::rmsnorm: the same bits in x and the normed rows.
#[test]
fn residual_norm_matches_add_then_rmsnorm() {
    use crate::kernels::glm53::{add, rmsnorm};
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(79);
    for (m, h) in [(1, 2048), (3, 2048), (2, 2048)] {
        let x = f32s(&mut rng, m * h, -3.0, 3.0);
        let a = f32s(&mut rng, m * h, -1.0, 1.0);
        let sh = f32s(&mut rng, m * h, -1.0, 1.0);
        let g = f32s(&mut rng, m, -2.0, 2.0);
        let w = tensor(&ctx, &f32s(&mut rng, h, 0.5, 1.5));
        for fused_shared in [false, true] {
            let (xa, xb) = (tensor(&ctx, &x), tensor(&ctx, &x));
            let (at, st, gt) = (tensor(&ctx, &a), tensor(&ctx, &sh), tensor(&ctx, &g));
            let (oa, ob) =
                (zeros(&ctx, m * h, DType::F32), zeros(&ctx, m * h, DType::F32));
            run(&ctx, |p| {
                if fused_shared {
                    ffn_residual(&ctx, p, &xa, &at, &st, &gt, m, h).unwrap();
                } else {
                    add(&ctx, p, &xa, &at, &xa, m * h).unwrap();
                }
                p.memory_barrier().unwrap();
                rmsnorm(&ctx, p, &xa, Some(&w), &oa, m, h, 1e-6).unwrap();
                let shared = fused_shared.then_some((&st, &gt));
                residual_norm(&ctx, p, &xb, &at, shared, &w, &ob, m, h, 1e-6).unwrap();
            });
            // And the norm alone, over the new rows.
            let oc = zeros(&ctx, m * h, DType::F32);
            run(&ctx, |p| super::rmsnorm(&ctx, p, &xa, &w, &oc, m, h, 1e-6).unwrap());
            // The residual bit for bit; the normed rows within 1e-6 (the
            // sum of squares of glm53_rmsnorm's runtime loop and of these
            // unrolled ones round differently: a row's scale moves by ulps).
            let (u, v) = (xa.to_f32().unwrap(), xb.to_f32().unwrap());
            for (k, (p, q)) in u.iter().zip(&v).enumerate() {
                assert_eq!(p.to_bits(), q.to_bits(), "x m {m} {fused_shared} at {k}");
            }
            let want = oa.to_f32().unwrap();
            assert_close(&ob.to_f32().unwrap(), &want, 1e-6, 1e-6);
            assert_close(&oc.to_f32().unwrap(), &want, 1e-6, 1e-6);
        }
    }
}

/// The one-level GDN step (then its conv window moved in a later level)
/// against the conv, scan and out kernels, with value heads sharing key
/// heads (h and h + HK read the same q and k conv channels): chains
/// of steps of 1 to 4 rows carrying the window and state from step to step,
/// with snapshots at row 0 and later rows, the whole chain repeated. Every
/// step's y, window, state and snapshots the same bits. A step that moved
/// the shared window while a sharing head still read it would part here.
#[test]
fn gdn_step_matches_gdn_bitwise() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(83);
    // The model's 16 key and 32 value heads, and 160 value heads over 16 key
    // heads: more threadgroups of 1024 than the GPU holds at once, so later
    // ones start after earlier ones finished (a step that moved the shared
    // window would then feed later heads a moved window).
    for (hk, h) in [(16, 32), (16, 160)] {
        let eps = 1e-6f32;
        let ch = (2 * hk + h) * GDN_DIM;
        let inner = h * GDN_DIM;
        let t = |v: &[f32]| tensor(&ctx, v);
        let w = t(&f32s(&mut rng, 4 * ch, -0.5, 0.5));
        let ssm_a = t(&f32s(&mut rng, h, -2.0, -0.1));
        let dt = t(&f32s(&mut rng, h, -1.0, 1.0));
        let norm = t(&f32s(&mut rng, GDN_DIM, 0.5, 1.5));
        let p = GdnParams { conv_w: &w, ssm_a: &ssm_a, dt_bias: &dt, norm: &norm };
        let conv0 = f32s(&mut rng, 3 * ch, -1.0, 1.0);
        let s0 = f32s(&mut rng, h * GDN_DIM * GDN_DIM, -0.1, 0.1);
        let bits = |a: &Tensor, b: &Tensor, what: &str| {
            let (a, b) = (a.to_f32().unwrap(), b.to_f32().unwrap());
            for (k, (x, y)) in a.iter().zip(&b).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{what} at {k}: {x} vs {y}");
            }
        };
        // (rows, snapshot row) of each step of a chain.
        let chain = [(1, 0), (2, 0), (1, 0), (4, 0), (3, 1), (2, 1), (1, 0), (4, 2)];
        for round in 0..3 {
            let (c1, s1, c2, s2) = (t(&conv0), t(&s0), t(&conv0), t(&s0));
            for (step, &(m, row)) in chain.iter().enumerate() {
                let qkv = t(&f32s(&mut rng, m * ch, -1.0, 1.0));
                let a = t(&f32s(&mut rng, m * h, -2.0, 2.0));
                let b = t(&f32s(&mut rng, m * h, -2.0, 2.0));
                let z = t(&f32s(&mut rng, m * inner, -2.0, 2.0));
                let snaps = || {
                    (zeros(&ctx, 3 * ch, DType::F32), zeros(&ctx, s0.len(), DType::F32))
                };
                let ((sc1, ss1), (sc2, ss2)) = (snaps(), snaps());
                let (y1, y2) = (
                    zeros(&ctx, m * inner, DType::F32),
                    zeros(&ctx, m * inner, DType::F32),
                );
                let c = zeros(&ctx, m * ch, DType::F32);
                let o = zeros(&ctx, m * inner, DType::F32);
                run(&ctx, |pass| {
                    let st = GdnState { conv: &c1, s: &s1 };
                    let snap = GdnSnapshot { row, conv: &sc1, s: &ss1 };
                    gdn(
                        &ctx,
                        pass,
                        &p,
                        &st,
                        Some(&snap),
                        &qkv,
                        &a,
                        &b,
                        &z,
                        &c,
                        &o,
                        &y1,
                        m,
                        hk,
                        eps,
                    )
                    .unwrap();
                    let st = GdnState { conv: &c2, s: &s2 };
                    let snap = GdnSnapshot { row, conv: &sc2, s: &ss2 };
                    gdn_step(
                        &ctx,
                        pass,
                        &p,
                        &st,
                        Some(&snap),
                        &qkv,
                        &a,
                        &b,
                        &z,
                        &y2,
                        m,
                        hk,
                        eps,
                    )
                    .unwrap();
                    pass.memory_barrier().unwrap();
                    gdn_conv_window(&ctx, pass, &p, &st, Some(&snap), &qkv, m).unwrap();
                });
                let at =
                    format!("round {round} step {step} (m {m}, snapshot row {row})");
                bits(&y1, &y2, &format!("{at} y"));
                bits(&c1, &c2, &format!("{at} window"));
                bits(&s1, &s2, &format!("{at} state"));
                bits(&sc1, &sc2, &format!("{at} snapshot window"));
                bits(&ss1, &ss2, &format!("{at} snapshot state"));
            }
        }
    }
}

/// The K = 2048 gate/up kernels against GLM's: the same bits.
#[test]
fn moe_gate_up_matches_glm53_bitwise() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(80);
    let (experts, hidden, ff, used) = (12, 2048, 512, 8);
    for (ty, m) in [(GgmlType::Q4K, 1), (GgmlType::Q4K, 3), (GgmlType::Q8_0, 2)] {
        let q = |rng: &mut StdRng| {
            let raw = random_blocks(rng, ty, hidden * experts * ff);
            QWeight::from_bytes(&ctx, &raw, ty, hidden, experts * ff).unwrap()
        };
        let (gw, uw) = (q(&mut rng), q(&mut rng));
        let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -2.0, 2.0));
        let sel: Vec<u32> =
            (0..m * used).map(|p| ((p * 5 + 1) % experts) as u32).collect();
        let tsel = u32s(&ctx, &sel);
        let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
        let a = zeros(&ctx, m * used * ff, DType::F32);
        let b = zeros(&ctx, m * used * ff, DType::F32);
        run(&ctx, |p| {
            moe_gate_up(&ctx, p, &gw, &uw, ff, &x, &tsel, &tw, &a, m, used).unwrap();
            crate::kernels::glm53::moe_gate_up(
                &ctx, p, &gw, &uw, ff, &x, &tsel, &tw, &b, m, used, 0.0,
            )
            .unwrap();
        });
        let (a, b) = (a.to_f32().unwrap(), b.to_f32().unwrap());
        for (k, (u, v)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                u.to_bits(),
                v.to_bits(),
                "{ty:?} m {m} element {k}: {u} vs {v}"
            );
        }
    }
}

/// The one-subgroup router against the 256-thread one: the same picks and
/// weights, bit for bit, ties and a non-finite row included.
#[test]
fn router_sg_matches_router_bitwise() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(84);
    let (m, e, k) = (5, 256, 8);
    let mut l = f32s(&mut rng, m * e, -6.0, 6.0);
    l[10] = l[20];
    l[256 + 5] = 30.0;
    l[256 + 6] = 30.0;
    l[2 * 256 + 7] = f32::NAN;
    let lt = tensor(&ctx, &l);
    let (sa, sb) = (zeros(&ctx, m * k, DType::U32), zeros(&ctx, m * k, DType::U32));
    let (wa, wb) = (zeros(&ctx, m * k, DType::F32), zeros(&ctx, m * k, DType::F32));
    run(&ctx, |p| {
        router(&ctx, p, &lt, &sa, &wa, m, e, k).unwrap();
        router_sg(&ctx, p, &lt, &sb, &wb, m).unwrap();
    });
    assert_eq!(sa.to_u32().unwrap(), sb.to_u32().unwrap());
    let (wa, wb) = (wa.to_f32().unwrap(), wb.to_f32().unwrap());
    for (i, (a, b)) in wa.iter().zip(&wb).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "weight {i}: {a} vs {b}");
    }
}

/// The K-split Q8_0 GEMV of rows of 4096 against GLM's (sums regrouped).
#[test]
fn gemv_q8_0_k4096_matches_glm53() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(87);
    let (cols, rows) = (4096, 2048);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, rows * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, rows).unwrap();
    let x = tensor(&ctx, &f32s(&mut rng, cols, -2.0, 2.0));
    let (a, b) = (zeros(&ctx, rows, DType::F32), zeros(&ctx, rows, DType::F32));
    run(&ctx, |p| {
        gemv_q8_0_k4096(&ctx, p, &w, &x, &a).unwrap();
        crate::kernels::glm53::gemv(&ctx, p, &w, &x, &b, 1).unwrap();
    });
    assert_close(&a.to_f32().unwrap(), &b.to_f32().unwrap(), 1e-5, 1e-5);
}

/// Attention decode in splits of 64 keys against the default route: at the
/// split boundaries (64, 65, 128, 129 keys), longer contexts, and with the
/// production scratch (128 splits) at 8192 keys (the last it takes) and
/// 8193 (where it falls back), cache rows past the context poisoned with
/// NaN (a read past the context would show).
#[test]
fn sdpa_decode_split64_matches_sdpa_decode() {
    use crate::kernels::attention::{
        KvCache, KvFormat, MAX_SEQ, scatter_kv, sdpa_decode, sdpa_decode_split64,
        sdpa_split_scratch_splits,
    };
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(88);
    let (nq, kvh, d) = (16, 2, 256);
    // The model's scratch: splits for MAX_SEQ.
    let splits = sdpa_split_scratch_splits(MAX_SEQ);
    let partials = zeros(&ctx, nq * splits * d, DType::F32);
    let stats = zeros(&ctx, nq * splits * 2, DType::F32);
    for (cap, lens) in
        [(2048, vec![64, 65, 128, 129, 300, 1100, 2000]), (8196, vec![8192, 8193])]
    {
        let kc = KvCache::zeros(&ctx, KvFormat::Bf16, kvh, cap, d).unwrap();
        let vc = KvCache::zeros(&ctx, KvFormat::Bf16, kvh, cap, d).unwrap();
        let nan = Tensor::from_f32_as_bf16(
            &ctx,
            &vec![f32::NAN; cap * kvh * d],
            &[cap, kvh, d],
        )
        .unwrap();
        for len in lens {
            let kr = Tensor::from_f32_as_bf16(
                &ctx,
                &f32s(&mut rng, len * kvh * d, -1.0, 1.0),
                &[len, kvh, d],
            )
            .unwrap();
            let vr = Tensor::from_f32_as_bf16(
                &ctx,
                &f32s(&mut rng, len * kvh * d, -1.0, 1.0),
                &[len, kvh, d],
            )
            .unwrap();
            let q = Tensor::from_f32_as_bf16(
                &ctx,
                &f32s(&mut rng, nq * d, -1.0, 1.0),
                &[nq, d],
            )
            .unwrap();
            let a = zeros(&ctx, nq * d, DType::BF16);
            let b = zeros(&ctx, nq * d, DType::BF16);
            run(&ctx, |p| {
                scatter_kv(&ctx, p, &kc, &nan, 0).unwrap();
                scatter_kv(&ctx, p, &vc, &nan, 0).unwrap();
                p.memory_barrier().unwrap();
                scatter_kv(&ctx, p, &kc, &kr, 0).unwrap();
                scatter_kv(&ctx, p, &vc, &vr, 0).unwrap();
                p.memory_barrier().unwrap();
                sdpa_decode_split64(
                    &ctx,
                    p,
                    &q,
                    &kc,
                    &vc,
                    &a,
                    len,
                    0.0625,
                    (&partials, &stats),
                )
                .unwrap();
                p.memory_barrier().unwrap();
                sdpa_decode(
                    &ctx,
                    p,
                    &q,
                    &kc,
                    &vc,
                    &b,
                    len,
                    0.0625,
                    Some((&partials, &stats)),
                )
                .unwrap();
            });
            let (a, b) = (a.to_f32().unwrap(), b.to_f32().unwrap());
            assert!(a.iter().all(|v| v.is_finite()), "{len} keys: a non-finite output");
            // bfloat outputs.
            assert_close(&a, &b, 1e-2, 1e-2);
        }
    }
}

/// The one-subgroup router refuses tensors of the wrong type or too short
/// for its rows (it reads 256 logits and writes 8 picks a row).
#[test]
fn router_sg_rejects_malformed_tensors() {
    let ctx = ctx();
    let m = 3;
    let logits = zeros(&ctx, m * 256, DType::F32);
    let sel = zeros(&ctx, m * 8, DType::U32);
    let w = zeros(&ctx, m * 8, DType::F32);
    let pass = ctx.begin().unwrap();
    assert!(router_sg(&ctx, &pass, &logits, &sel, &w, m).is_ok());
    let short_logits = zeros(&ctx, m * 256 - 1, DType::F32);
    let short_sel = zeros(&ctx, m * 8 - 1, DType::U32);
    let short_w = zeros(&ctx, m * 8 - 1, DType::F32);
    let bf_logits = zeros(&ctx, m * 256, DType::BF16);
    let f_sel = zeros(&ctx, m * 8, DType::F32);
    let u_w = zeros(&ctx, m * 8, DType::U32);
    for (name, l, s, ww) in [
        ("short logits", &short_logits, &sel, &w),
        ("short selection", &logits, &short_sel, &w),
        ("short weights", &logits, &sel, &short_w),
        ("bf16 logits", &bf_logits, &sel, &w),
        ("f32 selection", &logits, &f_sel, &w),
        ("u32 weights", &logits, &sel, &u_w),
    ] {
        assert!(router_sg(&ctx, &pass, l, s, ww, m).is_err(), "{name} accepted");
    }
    pass.commit_wait().unwrap();
}
