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

/// Fill past a kernel's logical output: it must survive the dispatch.
const SENTINEL: f32 = -7.25;

/// An F32 tensor of `n` outputs and 8 sentinels after them.
fn guarded(ctx: &MetalContext, n: usize) -> Tensor {
    tensor(ctx, &vec![SENTINEL; n + 8])
}

/// The `n` logical outputs of a [`guarded`] tensor, its sentinels checked.
fn unguard(t: &Tensor, n: usize, what: &str) -> Vec<f32> {
    let mut v = t.to_f32().unwrap();
    for (k, x) in v[n..].iter().enumerate() {
        assert_eq!(x.to_bits(), SENTINEL.to_bits(), "{what}: sentinel {k} written");
    }
    v.truncate(n);
    v
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
            None,
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

/// On a device with matrix units, attn_prep and attn_gate's `_bits` kernels
/// against their bfloat kernels: the same outputs bit for bit, with
/// Agnes 3.0 Flash's head counts (24 query heads over 4 key heads).
#[cfg(target_os = "linux")]
#[test]
fn attn_prep_and_gate_bits_are_bit_identical() {
    use crate::kernels::attention::FORCE_BITS;
    let ctx = ctx();
    if !ctx.matrix_units() {
        eprintln!("no matrix units: the bfloat kernels do not run here");
        return;
    }
    let mut rng = StdRng::seed_from_u64(39);
    let (m, nq, nkv, d) = (5, 24, 4, HEAD_DIM);
    let qg = tensor(&ctx, &f32s(&mut rng, m * nq * 2 * d, -2.0, 2.0));
    let kx = tensor(&ctx, &f32s(&mut rng, m * nkv * d, -2.0, 2.0));
    let vx = tensor(&ctx, &f32s(&mut rng, m * nkv * d, -2.0, 2.0));
    let qn = tensor(&ctx, &f32s(&mut rng, d, 0.5, 1.5));
    let kn = tensor(&ctx, &f32s(&mut rng, d, 0.5, 1.5));
    let outputs = |forced: bool| -> Vec<Vec<u8>> {
        FORCE_BITS.with(|f| f.set(forced));
        let qo = zeros(&ctx, m * nq * d, DType::BF16);
        let ko = zeros(&ctx, m * nkv * d, DType::BF16);
        let vo = zeros(&ctx, m * nkv * d, DType::BF16);
        let y = zeros(&ctx, m * nq * d, DType::F32);
        run(&ctx, |pass| {
            attn_prep(
                &ctx, pass, &qg, &kx, &vx, &qn, &kn, &qo, &ko, &vo, m, nq, nkv, 77, 64,
                1e7, 1e-6,
            )
            .unwrap();
            pass.memory_barrier().unwrap();
            attn_gate(&ctx, pass, &qo, &qg, &y, m, nq).unwrap();
        });
        FORCE_BITS.with(|f| f.set(false));
        [&qo, &ko, &vo, &y].iter().map(|t| t.contents().to_vec()).collect()
    };
    assert!(outputs(false) == outputs(true));
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
    for (m, h) in
        [(1, 2048), (3, 2048), (2, 2048), (1, 2560), (3, 2560), (1, 5120), (3, 5120)]
    {
        let x = f32s(&mut rng, m * h, -3.0, 3.0);
        let a = f32s(&mut rng, m * h, -1.0, 1.0);
        let sh = f32s(&mut rng, m * h, -1.0, 1.0);
        let g = f32s(&mut rng, m, -2.0, 2.0);
        let w = tensor(&ctx, &f32s(&mut rng, h, 0.5, 1.5));
        for &fused_shared in if h == 2048 { &[false, true][..] } else { &[false][..] } {
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
            let cpu = crate::cpu_ref::rmsnorm(&u, &w.to_f32().unwrap(), h, 1e-6);
            assert_close(&ob.to_f32().unwrap(), &cpu, 1e-5, 1e-5);
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
                        None,
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

/// Routed gate/up at rows of 2048 against GLM's kernel: each row over two
/// subgroups regroups its sums, so both are compared with an f64 reference
/// (an empirical comparison on these seeded inputs: the new kernel's
/// largest absolute error at most twice GLM's, plus 1e-6; every output and
/// reference finite). ff of 512 and 510 (a partial last threadgroup), with
/// sentinels after the outputs. Each activation row of a multi-row run is
/// the same bits as that row alone (a verify row is a decode step's).
#[test]
fn moe_gate_up_matches_glm53() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(80);
    let (experts, hidden, used) = (12, 2048, 8);
    for (ty, m, ff) in [
        (GgmlType::Q4K, 1, 512),
        (GgmlType::Q4K, 3, 510),
        (GgmlType::Q8_0, 2, 510),
        (GgmlType::Q8_0, 4, 512),
    ] {
        let mut deq = Vec::new();
        let mut q = |rng: &mut StdRng| {
            let raw = random_blocks(rng, ty, hidden * experts * ff);
            let mut d = vec![0f32; hidden * experts * ff];
            dequantize(ty, &raw, &mut d).unwrap();
            deq.push(d);
            QWeight::from_bytes(&ctx, &raw, ty, hidden, experts * ff).unwrap()
        };
        let (gw, uw) = (q(&mut rng), q(&mut rng));
        let xs = f32s(&mut rng, m * hidden, -2.0, 2.0);
        let x = tensor(&ctx, &xs);
        let sel: Vec<u32> =
            (0..m * used).map(|p| ((p * 5 + 1) % experts) as u32).collect();
        let tsel = u32s(&ctx, &sel);
        let ws = f32s(&mut rng, m * used, 0.1, 0.6);
        let tw = tensor(&ctx, &ws);
        let n = m * used * ff;
        let (a, b) = (guarded(&ctx, n), guarded(&ctx, n));
        run(&ctx, |p| {
            moe_gate_up(&ctx, p, &gw, &uw, ff, &x, &tsel, &tw, &a, m, used).unwrap();
            crate::kernels::glm53::moe_gate_up(
                &ctx, p, &gw, &uw, ff, &x, &tsel, &tw, &b, m, used, 0.0,
            )
            .unwrap();
        });
        let at = format!("{ty:?} m {m} ff {ff}");
        let (a, b) = (unguard(&a, n, &at), unguard(&b, n, &at));
        let (mut err_a, mut err_b) = (0f64, 0f64);
        for slot in 0..m * used {
            let i = slot / used;
            let xi = &xs[i * hidden..(i + 1) * hidden];
            let base = sel[slot] as usize * ff;
            for r in 0..ff {
                let dot = |w: &[f32]| -> f64 {
                    let row = &w[(base + r) * hidden..(base + r + 1) * hidden];
                    row.iter().zip(xi).map(|(&w, &x)| w as f64 * x as f64).sum()
                };
                let (g, u) = (dot(&deq[0]), dot(&deq[1]));
                let want = ws[slot] as f64 * g * u / (1.0 + (-g).exp());
                let (ea, eb) = (
                    (a[slot * ff + r] as f64 - want).abs(),
                    (b[slot * ff + r] as f64 - want).abs(),
                );
                assert!(
                    want.is_finite() && ea.is_finite() && eb.is_finite(),
                    "{at} slot {slot} row {r}: {} / {} vs {want}",
                    a[slot * ff + r],
                    b[slot * ff + r]
                );
                err_a = err_a.max(ea);
                err_b = err_b.max(eb);
            }
        }
        assert!(err_a <= 2.0 * err_b + 1e-6, "{at}: error {err_a} vs GLM's {err_b}");
        for i in 0..m {
            let xi = tensor(&ctx, &xs[i * hidden..(i + 1) * hidden]);
            let si = u32s(&ctx, &sel[i * used..(i + 1) * used]);
            let wi = tensor(&ctx, &ws[i * used..(i + 1) * used]);
            let one = guarded(&ctx, used * ff);
            run(&ctx, |p| {
                moe_gate_up(&ctx, p, &gw, &uw, ff, &xi, &si, &wi, &one, 1, used)
                    .unwrap()
            });
            let one = unguard(&one, used * ff, &at);
            let rows = &a[i * used * ff..(i + 1) * used * ff];
            for (k, (u, v)) in rows.iter().zip(&one).enumerate() {
                assert_eq!(u.to_bits(), v.to_bits(), "{at} row {i} element {k}");
            }
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

/// The K-split Q8_0 GEMV of rows of 4096 against GLM's (sums regrouped) for
/// 1 to 4 activation rows over 2051 output rows (two a threadgroup: a
/// partial last one), sentinels after the outputs, and each row of a
/// multi-row dispatch the same bits as that row alone (a verify row is a
/// decode step's).
#[test]
fn gemv_q8_0_k4096_matches_glm53() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(87);
    let (cols, rows) = (4096, 2051);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, rows * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, rows).unwrap();
    let xs = f32s(&mut rng, 4 * cols, -2.0, 2.0);
    let one: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let x = tensor(&ctx, &xs[i * cols..(i + 1) * cols]);
            let y = guarded(&ctx, rows);
            run(&ctx, |p| gemv_q8_0_k4096(&ctx, p, &w, &x, &y, 1).unwrap());
            unguard(&y, rows, "m 1")
        })
        .collect();
    for m in 1..=4 {
        let x = tensor(&ctx, &xs[..m * cols]);
        let (a, b) = (guarded(&ctx, m * rows), zeros(&ctx, m * rows, DType::F32));
        run(&ctx, |p| {
            gemv_q8_0_k4096(&ctx, p, &w, &x, &a, m).unwrap();
            crate::kernels::glm53::gemv(&ctx, p, &w, &x, &b, m).unwrap();
        });
        let (a, b) = (unguard(&a, m * rows, &format!("m {m}")), b.to_f32().unwrap());
        assert!(a.iter().chain(&b).all(|v| v.is_finite()), "m {m}: non-finite output");
        assert_close(&a, &b, 1e-5, 1e-5);
        for (i, row) in one.iter().enumerate().take(m) {
            let got = &a[i * rows..(i + 1) * rows];
            for (k, (u, v)) in got.iter().zip(row).enumerate() {
                assert_eq!(u.to_bits(), v.to_bits(), "m {m} row {i} output {k}");
            }
        }
    }
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

/// The K = 2048 GEMV against `glm53::gemv_sets`: each row over two
/// subgroups regroups its sums, so both are compared with an f64 reference
/// (an empirical comparison on these seeded inputs: each set's largest
/// absolute error at most twice GLM's, plus 1e-6; every output and
/// reference finite). F32 and Q8_0 sets in one dispatch; two, three and
/// four sets; partial last threadgroups (509, 1 rows), sentinels after the
/// outputs. Each activation row of a multi-row run is the same bits as that
/// row alone.
#[test]
fn gemv_k2048_matches_gemv_sets() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(89);
    let cols = 2048;
    let (f, q) = (GgmlType::F32, GgmlType::Q8_0);
    let shapes: [&[(GgmlType, usize)]; 4] = [
        &[(f, 256), (f, 1), (q, 509), (q, 512)],
        &[(q, 8192), (q, 4096), (q, 32), (q, 32)],
        &[(q, 8192), (q, 509)],
        &[(f, 255), (q, 512), (q, 1)],
    ];
    for shape in shapes {
        let mut deq = Vec::new();
        let ws: Vec<_> = shape
            .iter()
            .map(|&(ty, rows)| {
                let raw: Vec<u8> = if ty == f {
                    f32s(&mut rng, rows * cols, -0.1, 0.1)
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect()
                } else {
                    random_blocks(&mut rng, ty, rows * cols)
                };
                let mut d = vec![0f32; rows * cols];
                dequantize(ty, &raw, &mut d).unwrap();
                deq.push(d);
                QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
            })
            .collect();
        let xs = f32s(&mut rng, 4 * cols, -2.0, 2.0);
        let one: Vec<Vec<Vec<f32>>> = (0..4)
            .map(|i| {
                let x = tensor(&ctx, &xs[i * cols..(i + 1) * cols]);
                let ys: Vec<_> = ws.iter().map(|w| guarded(&ctx, w.rows)).collect();
                run(&ctx, |p| {
                    let sets: Vec<_> = ws.iter().zip(&ys).collect();
                    gemv_k2048(&ctx, p, &sets, &x, 1).unwrap();
                });
                ys.iter().zip(&ws).map(|(y, w)| unguard(y, w.rows, "m 1")).collect()
            })
            .collect();
        for m in 1..=4 {
            let x = tensor(&ctx, &xs[..m * cols]);
            let ys: Vec<_> = ws.iter().map(|w| guarded(&ctx, m * w.rows)).collect();
            let refs: Vec<_> =
                ws.iter().map(|w| zeros(&ctx, m * w.rows, DType::F32)).collect();
            run(&ctx, |p| {
                let sets: Vec<_> = ws.iter().zip(&ys).collect();
                gemv_k2048(&ctx, p, &sets, &x, m).unwrap();
                for (w, r) in ws.iter().zip(&refs) {
                    crate::kernels::glm53::gemv_sets(&ctx, p, &[(w, r)], &x, m)
                        .unwrap();
                }
            });
            for (s, ((y, r), d)) in ys.iter().zip(&refs).zip(&deq).enumerate() {
                let rows = ws[s].rows;
                let at = format!("{shape:?} m {m} set {s}");
                let (y, r) = (unguard(y, m * rows, &at), r.to_f32().unwrap());
                let (mut err_y, mut err_r) = (0f64, 0f64);
                for i in 0..m {
                    let xi = &xs[i * cols..(i + 1) * cols];
                    for n in 0..rows {
                        let want: f64 = d[n * cols..(n + 1) * cols]
                            .iter()
                            .zip(xi)
                            .map(|(&w, &x)| w as f64 * x as f64)
                            .sum();
                        let (ey, er) = (
                            (y[i * rows + n] as f64 - want).abs(),
                            (r[i * rows + n] as f64 - want).abs(),
                        );
                        assert!(
                            want.is_finite() && ey.is_finite() && er.is_finite(),
                            "{at} row {i} output {n}: {} / {} vs {want}",
                            y[i * rows + n],
                            r[i * rows + n]
                        );
                        err_y = err_y.max(ey);
                        err_r = err_r.max(er);
                    }
                    let got = &y[i * rows..(i + 1) * rows];
                    for (k, (u, v)) in got.iter().zip(&one[i][s]).enumerate() {
                        assert_eq!(u.to_bits(), v.to_bits(), "{at} row {i} output {k}");
                    }
                }
                assert!(
                    err_y <= 2.0 * err_r + 1e-6,
                    "{at}: error {err_y} vs GLM's {err_r}"
                );
            }
        }
    }
}

/// The K = 2048 GEMV refuses other types and widths, short or non-F32
/// inputs and outputs, and row or set counts outside 1-4.
#[test]
fn gemv_k2048_rejects_malformed() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(90);
    let cols = 2048;
    let q = |rng: &mut StdRng, ty: GgmlType, cols: usize| {
        QWeight::from_bytes(&ctx, &random_blocks(rng, ty, 8 * cols), ty, cols, 8)
            .unwrap()
    };
    let q8 = q(&mut rng, GgmlType::Q8_0, cols);
    let q4 = q(&mut rng, GgmlType::Q4K, cols);
    let wide = q(&mut rng, GgmlType::Q8_0, 4096);
    assert!(
        gemv_k2048_takes(&q8) && !gemv_k2048_takes(&q4) && !gemv_k2048_takes(&wide)
    );
    let x = zeros(&ctx, 4 * 4096, DType::F32);
    let y = zeros(&ctx, 4 * 8, DType::F32);
    let short = zeros(&ctx, 2 * 8 - 1, DType::F32);
    let shortx = zeros(&ctx, 2 * cols - 1, DType::F32);
    let ux = zeros(&ctx, 4 * 4096, DType::U32);
    let uy = zeros(&ctx, 4 * 8, DType::U32);
    run(&ctx, |p| {
        assert!(gemv_k2048(&ctx, p, &[(&q8, &y)], &x, 2).is_ok());
        assert!(gemv_k2048(&ctx, p, &[(&q4, &y)], &x, 2).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&wide, &y)], &x, 2).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&q8, &short)], &x, 2).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&q8, &y)], &shortx, 2).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&q8, &y)], &x, 5).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&q8, &y)], &x, 0).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&q8, &y)], &ux, 2).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&q8, &y), (&q8, &uy)], &x, 2).is_err());
        assert!(gemv_k2048(&ctx, p, &[], &x, 1).is_err());
        assert!(gemv_k2048(&ctx, p, &[(&q8, &y); 5], &x, 1).is_err());
    });
}

/// The chunked delta rule of a prompt run (chunks of 64 rows on the matrix
/// units, bf16 operands, a ragged tail token by token) against the CPU
/// reference and the token-serial scan: outputs and state within bfloat
/// tolerances of the f64 reference, every value finite, the conv window
/// the same bits. 128 rows (two whole chunks) and 200 (three and a tail of
/// 8) over 2 key and 4 value heads; 128 rows of the model's 16 and 32.
#[test]
fn gdn_chunked_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(95);
    let eps = 1e-6;
    for (hk, h, m) in [(2, 4, 128), (2, 4, 200), (16, 32, 128)] {
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
        let t = |v: &[f32]| tensor(&ctx, v);
        let (wt, at, dtt, nt) = (t(&w), t(&ssm_a), t(&dt), t(&norm));
        let p = GdnParams { conv_w: &wt, ssm_a: &at, dt_bias: &dtt, norm: &nt };
        let (qkv_t, a_t, b_t, z_t) = (t(&qkv), t(&alpha), t(&beta), t(&z));
        let shapes = gdn_chunk_shapes(m, hk, h);
        let staging: Vec<_> =
            shapes.iter().map(|&(n, ty)| zeros(&ctx, n, ty)).collect();
        let chunk = GdnChunk {
            qkv: &staging[0],
            w: &staging[1],
            u: &staging[2],
            p: &staging[3],
            g: &staging[4],
        };
        let mut outs = Vec::new();
        for chunked in [false, true] {
            let (conv, s) = (t(&conv0), t(&s0));
            let st = GdnState { conv: &conv, s: &s };
            let c = zeros(&ctx, m * ch, DType::F32);
            let o = zeros(&ctx, m * inner, DType::F32);
            let y = zeros(&ctx, m * inner, DType::F32);
            run(&ctx, |pass| {
                gdn(
                    &ctx,
                    pass,
                    &p,
                    &st,
                    None,
                    chunked.then_some(&chunk),
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
            outs.push((
                o.to_f32().unwrap(),
                y.to_f32().unwrap(),
                s.to_f32().unwrap(),
                conv.to_f32().unwrap(),
            ));
        }
        let at = format!("{hk}/{h} heads, {m} rows");
        let (serial, chunked) = (&outs[0], &outs[1]);
        for v in [&chunked.0, &chunked.1, &chunked.2] {
            assert!(v.iter().all(|x| x.is_finite()), "{at}: a non-finite value");
        }
        assert_eq!(chunked.3, serial.3, "{at}: conv window");
        assert_eq!(chunked.3, conv_r, "{at}: conv window against the reference");
        // Relative (L2) error against the reference over each block of 32
        // rows: the chunked path's bfloat operands and tiles (~2^-8),
        // single near-zero outputs aside; the serial scan's for scale.
        let rel = |a: &[f32], b: &[f64]| {
            let d: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y).powi(2)).sum();
            let n: f64 = b.iter().map(|y| y * y).sum();
            (d / n).sqrt()
        };
        for c0 in (0..m).step_by(32) {
            let r = c0 * inner..(c0 + 32).min(m) * inner;
            let (eo, ey) = (
                rel(&chunked.0[r.clone()], &o_r[r.clone()]),
                rel(&chunked.1[r.clone()], &y_r[r.clone()]),
            );
            assert!(
                eo < 1e-2 && ey < 1e-2,
                "{at} rows {c0}..: o error {eo:.2e}, y {ey:.2e}"
            );
            assert!(
                rel(&serial.0[r.clone()], &o_r[r]) < 1e-5,
                "{at} rows {c0}..: serial o"
            );
        }
        let es = rel(&chunked.2, &s_r);
        assert!(es < 1e-2, "{at}: state error {es:.2e}");
    }
}

/// Random GDN inputs of `m` rows for `hk` key and `h` value heads; log-decay
/// per token softplus(alpha + dt) ssm_a with ssm_a in `a_range`.
struct GdnInputs {
    qkv: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    z: Vec<f32>,
}

fn gdn_inputs(rng: &mut StdRng, m: usize, hk: usize, h: usize) -> GdnInputs {
    let ch = (2 * hk + h) * GDN_DIM;
    GdnInputs {
        qkv: f32s(rng, m * ch, -1.0, 1.0),
        alpha: f32s(rng, m * h, -2.0, 2.0),
        beta: f32s(rng, m * h, -2.0, 2.0),
        z: f32s(rng, m * h * GDN_DIM, -2.0, 2.0),
    }
}

/// Relative L2 error of `a` against `b`.
fn rel_err(a: &[f32], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y).powi(2)).sum();
    let n: f64 = b.iter().map(|y| y * y).sum();
    (d / n).sqrt()
}

/// GDN prompt steps (`steps`: rows of each `gdn` call, chunked where it
/// takes them), then decode and verify steps through `gdn_step` (`decode`:
/// rows of each, the window moved by `gdn_conv_window` after each), all
/// continuing one state and conv window, against the f64 recurrence
/// continued over the same rows: per block of 128 rows (or each step) the
/// relative error of o (prompt) and y below 1e-2, every value finite, the
/// conv window the same bits, the final state within 1e-2. Returns the
/// blocks' y errors and the final state's, in order.
#[allow(clippy::too_many_arguments)]
fn gdn_chunked_run(
    ctx: &MetalContext,
    rng: &mut StdRng,
    hk: usize,
    h: usize,
    steps: &[usize],
    decode: &[usize],
    a_range: (f32, f32),
    s0_scale: f32,
) -> (Vec<f64>, f64) {
    let eps = 1e-6;
    let ch = (2 * hk + h) * GDN_DIM;
    let inner = h * GDN_DIM;
    let w = f32s(rng, 4 * ch, -0.5, 0.5);
    let ssm_a = f32s(rng, h, a_range.0, a_range.1);
    let dt = f32s(rng, h, -1.0, 1.0);
    let norm = f32s(rng, GDN_DIM, 0.5, 1.5);
    let conv0 = f32s(rng, 3 * ch, -1.0, 1.0);
    let s0 = f32s(rng, h * GDN_DIM * GDN_DIM, -s0_scale, s0_scale);
    let t = |v: &[f32]| tensor(ctx, v);
    let (wt, at, dtt, nt) = (t(&w), t(&ssm_a), t(&dt), t(&norm));
    let p = GdnParams { conv_w: &wt, ssm_a: &at, dt_bias: &dtt, norm: &nt };
    let (conv, s) = (t(&conv0), t(&s0));
    let st = GdnState { conv: &conv, s: &s };
    let (mut conv_r, mut s_r) =
        (conv0.clone(), s0.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let most = steps.iter().copied().max().unwrap_or(0);
    let staging: Vec<_> = gdn_chunk_shapes(most, hk, h)
        .iter()
        .map(|&(n, ty)| zeros(ctx, n, ty))
        .collect();
    let chunk = GdnChunk {
        qkv: &staging[0],
        w: &staging[1],
        u: &staging[2],
        p: &staging[3],
        g: &staging[4],
    };
    let mut errs = Vec::new();
    for (k, &m) in steps.iter().chain(decode).enumerate() {
        let prompt = k < steps.len();
        let x = gdn_inputs(rng, m, hk, h);
        let (o_r, y_r) = gdn_ref(
            &x.qkv,
            &x.alpha,
            &x.beta,
            &x.z,
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
        let (qkv_t, a_t, b_t, z_t) = (t(&x.qkv), t(&x.alpha), t(&x.beta), t(&x.z));
        let c = zeros(ctx, m * ch, DType::F32);
        let o = zeros(ctx, m * inner, DType::F32);
        let y = zeros(ctx, m * inner, DType::F32);
        run(ctx, |pass| {
            if prompt {
                gdn(
                    ctx,
                    pass,
                    &p,
                    &st,
                    None,
                    Some(&chunk),
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
                .unwrap();
            } else {
                gdn_step(
                    ctx, pass, &p, &st, None, &qkv_t, &a_t, &b_t, &z_t, &y, m, hk,
                    eps as f32,
                )
                .unwrap();
                pass.memory_barrier().unwrap();
                gdn_conv_window(ctx, pass, &p, &st, None, &qkv_t, m).unwrap();
            }
        });
        let at = format!("{hk}/{h} heads, step {k} ({m} rows)");
        let (o, y) = (o.to_f32().unwrap(), y.to_f32().unwrap());
        assert!(y.iter().all(|v| v.is_finite()), "{at}: a non-finite output");
        for c0 in (0..m).step_by(128) {
            let r = c0 * inner..(c0 + 128).min(m) * inner;
            let ey = rel_err(&y[r.clone()], &y_r[r.clone()]);
            if prompt {
                let eo = rel_err(&o[r.clone()], &o_r[r]);
                assert!(eo < 1e-2, "{at} rows {c0}..: o error {eo:.2e}");
            }
            assert!(ey < 1e-2, "{at} rows {c0}..: y error {ey:.2e}");
            errs.push(ey);
        }
        assert_eq!(conv.to_f32().unwrap(), conv_r, "{at}: conv window");
    }
    let s = s.to_f32().unwrap();
    assert!(s.iter().all(|v| v.is_finite()), "a non-finite state");
    let es = rel_err(&s, &s_r);
    assert!(es < 1e-2, "{hk}/{h} heads: final state error {es:.2e}");
    (errs, es)
}

/// Decode and verify steps (`gdn_step`) after a prompt run of whole chunks
/// continue its state as the f64 recurrence does: the model's 16 and 32
/// heads, 128 prompt rows then steps of 1, 3 and 1 rows.
#[test]
fn gdn_steps_after_chunked_prefill_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(96);
    let (errs, es) =
        gdn_chunked_run(&ctx, &mut rng, 16, 32, &[128], &[1, 3, 1], (-2.0, -0.1), 0.1);
    let errs: Vec<_> = errs.iter().map(|e| format!("{e:.2e}")).collect();
    eprintln!("steps after prefill: y errors {}, final state {es:.2e}", errs.join(" "));
}

/// Long prompts under weak decay (per-token log-decay -0.0002 to -0.02:
/// the incoming state is carried far) with a nonzero incoming state: 2048
/// rows in one run and as two successive prompt steps of 1024, then a
/// decode step; error per block of 128 rows and of the final state against
/// the f64 recurrence, which must not grow past 1e-2 with length.
#[test]
fn gdn_chunked_long_weak_decay_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(97);
    for steps in [&[2048][..], &[1024, 1024][..]] {
        let (errs, es) =
            gdn_chunked_run(&ctx, &mut rng, 2, 4, steps, &[1], (-0.01, -0.001), 0.5);
        eprintln!(
            "prompt steps {steps:?}: y errors by block {}, final state {es:.2e}",
            errs.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" ")
        );
    }
}

/// Qwen3.6's grouped experts with gate, up and the SwiGLU in one GEMM
/// (`glm53::moe_grouped_fused`, over its own compact scratch whose `mid`
/// starts as sentinels) against the separate GEMMs and SwiGLU
/// (`glm53::moe_grouped`): the same bits, mid and out, for Q4_K and Q8_0
/// experts, at two densities: 480 pairs over 12 experts (tiles of 64 rows)
/// and 640 (at least 48 an expert: tiles of 128 rows). Experts with 16 to
/// 129 pairs (partial tiles, a second tile), small ones, unused ones.
#[test]
fn moe_grouped_fused_matches_separate_q4_k_q8_0() {
    use crate::kernels::glm53::{MoeGroupScratch, moe_grouped, moe_grouped_fused};
    let ctx = ctx();
    if !ctx.matrix_units() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(98);
    let (experts, hidden, ff, used) = (12, 2048, 512, 8);
    for counts in [
        [16usize, 32, 48, 64, 65, 0, 129, 100, 16, 10, 0, 0],
        [16, 64, 65, 0, 129, 100, 16, 10, 0, 96, 128, 16],
    ] {
        let pairs: usize = counts.iter().sum();
        let m = pairs / used;
        assert_eq!(m * used, pairs);
        let r128 = pairs >= 48 * experts;
        let mut sel: Vec<u32> = counts
            .iter()
            .enumerate()
            .flat_map(|(e, &c)| std::iter::repeat_n(e as u32, c))
            .collect();
        for i in (1..sel.len()).rev() {
            sel.swap(i, rng.gen_range(0..=i));
        }
        let tsel = u32s(&ctx, &sel);
        let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
        let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -2.0, 2.0));
        for ty in [GgmlType::Q4K, GgmlType::Q8_0] {
            let mk = |rng: &mut StdRng, cols: usize, rows: usize| {
                QWeight::from_bytes(
                    &ctx,
                    &random_blocks(rng, ty, rows * cols),
                    ty,
                    cols,
                    rows,
                )
                .unwrap()
            };
            let (gw, uw, dw) = (
                mk(&mut rng, hidden, experts * ff),
                mk(&mut rng, hidden, experts * ff),
                mk(&mut rng, ff, experts * hidden),
            );
            let at = format!("{ty:?}, {pairs} pairs (128-row tiles: {r128})");
            // The separate route's scratch, and the fused route's own compact
            // one (no gate and up outputs) with its mid set to sentinels.
            let sep = MoeGroupScratch::new(&ctx, pairs, experts, ff, hidden).unwrap();
            let fused =
                MoeGroupScratch::fused(&ctx, m, used, experts, ff, hidden).unwrap();
            fused.mid.write_bytes(&vec![0x7fu8; fused.mid.byte_len()]).unwrap();
            let (out_s, out_f) = (
                zeros(&ctx, m * hidden, DType::F32),
                zeros(&ctx, m * hidden, DType::F32),
            );
            run(&ctx, |p| {
                moe_grouped(
                    &ctx, p, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &out_s, &sep,
                    m, used, experts, 0.0,
                )
                .unwrap();
                moe_grouped_fused(
                    &ctx, p, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &out_f, &fused,
                    m, used, experts, 0.0,
                )
                .unwrap();
            });
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            let (mid_s, mid_f) =
                (sep.mid.to_f32().unwrap(), fused.mid.to_f32().unwrap());
            assert_eq!(
                bits(&mid_s[..pairs * ff]),
                bits(&mid_f[..pairs * ff]),
                "{at}: mid"
            );
            let (out_s, out_f) = (out_s.to_f32().unwrap(), out_f.to_f32().unwrap());
            assert_eq!(bits(&out_s), bits(&out_f), "{at}: out");
            assert!(out_f.iter().all(|v| v.is_finite()), "{at}: a non-finite output");
        }
    }
}
