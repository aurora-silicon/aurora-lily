use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::gguf::dequantize;
use crate::refops::{dot, gelu_tanh as gelu, rmsnorm as cpu_rmsnorm, round_f16 as h16};

fn ctx() -> MetalContext {
    MetalContext::new().expect("metal context")
}

fn f32s(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn tensor(ctx: &MetalContext, v: &[f32]) -> Tensor {
    Tensor::from_f32(ctx, v, &[v.len()]).expect("tensor")
}

fn zeros(ctx: &MetalContext, n: usize) -> Tensor {
    Tensor::zeros(ctx, &[n], DType::F32).expect("zeros")
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

/// Random blocks with small scales (Q8_0, IQ4_NL, Q4_K, Q5_K, Q6_K,
/// IQ2_XXS, IQ2_S; F32 values in -0.1..0.1).
fn random_blocks(rng: &mut StdRng, ty: GgmlType, values: usize) -> Vec<u8> {
    if ty == GgmlType::F32 {
        return (0..values)
            .flat_map(|_| rng.gen_range(-0.1f32..0.1).to_le_bytes())
            .collect();
    }
    let nb = values / ty.block_values();
    let bb = ty.block_bytes();
    let mut b: Vec<u8> = (0..nb * bb).map(|_| rng.r#gen()).collect();
    let put = |blk: &mut [u8], at: usize, v: f32| {
        blk[at..at + 2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes())
    };
    for blk in b.chunks_exact_mut(bb) {
        match ty {
            GgmlType::Q8_0 | GgmlType::Iq4Nl | GgmlType::Iq2Xxs | GgmlType::Iq2S => {
                put(blk, 0, rng.gen_range(0.001..0.01))
            }
            GgmlType::Q6K => put(blk, 208, rng.gen_range(0.001..0.01)),
            GgmlType::Q4K | GgmlType::Q5K => {
                put(blk, 0, rng.gen_range(0.001..0.01));
                put(blk, 2, rng.gen_range(0.001..0.01));
            }
            GgmlType::Q6K => put(blk, 208, rng.gen_range(0.0001..0.001)),
            other => panic!("no random {other:?} blocks"),
        }
    }
    b
}

fn decode(ty: GgmlType, raw: &[u8]) -> Vec<f32> {
    let mut w = vec![0f32; raw.len() / ty.block_bytes() * ty.block_values()];
    dequantize(ty, raw, &mut w).unwrap();
    w
}

fn rope(v: &mut [f32], pos: usize, base: f32, ff: Option<&[f32]>) {
    let d = v.len();
    for i in 0..d / 2 {
        let f = ff.map_or(1.0, |f| f[i]);
        if f > 1e20 {
            continue;
        }
        let inv = base.powf(-2.0 * i as f32 / d as f32) / f;
        let (s, c) = (pos as f32 * inv).sin_cos();
        let (lo, hi) = (v[i], v[i + d / 2]);
        v[i] = lo * c - hi * s;
        v[i + d / 2] = hi * c + lo * s;
    }
}

/// qkv then attention (decode or prompt path) against the CPU, rows at
/// positions pos0.. with the earlier positions filled by a first call.
fn attention_case(
    d: usize,
    kvh: usize,
    window: usize,
    pos0: usize,
    m: usize,
    global: bool,
) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64((d * 7 + m * 13 + pos0) as u64);
    let heads = 16;
    let eps = 1e-6;
    let base = if global { 1e6 } else { 1e4 };
    let total = pos0 + m;
    let ring = if window > 0 { window + m + 16 } else { total + 3 };
    let kw = kvh * d;
    // Norm weights that keep the scores in a few units, as a model's are.
    let qn = f32s(&mut rng, d, 0.05, 0.15);
    let kn = f32s(&mut rng, d, 0.5, 1.5);
    let ff: Vec<f32> = (0..d / 2).map(|i| if i < d / 8 { 1.0 } else { 1e30 }).collect();
    let q = f32s(&mut rng, total * heads * d, -1.0, 1.0);
    let k = f32s(&mut rng, total * kw, -1.0, 1.0);
    // A global layer's V is its K projection (attention_k_eq_v), and its K
    // cache keeps only the turning pairs (d / 8 of them here).
    let v = if global { k.clone() } else { f32s(&mut rng, total * kw, -1.0, 1.0) };
    let k_rot = if global { d / 8 } else { d / 2 };
    let kc = Tensor::zeros(&ctx, &[ring * kvh * 2 * k_rot], DType::F16).unwrap();
    let vc = Tensor::zeros(&ctx, &[ring * kw], DType::F16).unwrap();
    let (tqn, tkn, tff) = (tensor(&ctx, &qn), tensor(&ctx, &kn), tensor(&ctx, &ff));
    let a = AttnShape {
        head_dim: d,
        heads,
        kv_heads: kvh,
        ring,
        window,
        k_cache: &kc,
        v_cache: &vc,
        k_rot,
        k_norm: &tkn,
    };
    let chunks = attn_chunks(&a, pos0, m).max(1);
    let partial = zeros(&ctx, attn_partial_len(m, heads, d, chunks));
    let qo = zeros(&ctx, total * heads * d);
    let out = zeros(&ctx, m * heads * d);
    // Positions before pos0 in runs of at most 16 (the cache fill: a run
    // must not wrap onto the ring's slots it still needs), then the rows
    // under test.
    let mut p = 0;
    while p < total {
        let n = if p < pos0 { (pos0 - p).min(16) } else { m };
        let tq = tensor(&ctx, &q[p * heads * d..(p + n) * heads * d]);
        let tk = tensor(&ctx, &k[p * kw..(p + n) * kw]);
        let tv = tensor(&ctx, &v[p * kw..(p + n) * kw]);
        run(&ctx, |pass| {
            qkv(
                &ctx,
                pass,
                &a,
                &tq,
                &tk,
                &tv,
                kw,
                0,
                &tqn,
                &tkn,
                global.then_some(&tff),
                &qo,
                p,
                n,
                base,
                eps,
            )
            .unwrap();
            if p == pos0 {
                pass.memory_barrier().unwrap();
                attention(&ctx, pass, &a, &qo, &partial, &out, pos0, m, 4).unwrap();
            }
        });
        p += n;
    }
    // CPU: normed, rotated, f16-rounded keys and values.
    let norm = |x: &[f32], w: Option<&[f32]>| cpu_rmsnorm(x, w, eps);
    let ffo = global.then_some(ff.as_slice());
    let mut kk = vec![0f32; total * kw];
    let mut vv = vec![0f32; total * kw];
    for p in 0..total {
        for h in 0..kvh {
            let r = p * kw + h * d..p * kw + (h + 1) * d;
            let mut kr = norm(&k[r.clone()], Some(&kn));
            rope(&mut kr, p, base, ffo);
            let vr = norm(&v[r.clone()], None);
            for i in 0..d {
                kk[r.start + i] = h16(kr[i]);
                vv[r.start + i] = h16(vr[i]);
            }
        }
    }
    let mut expected = vec![0f32; m * heads * d];
    for i in 0..m {
        let pos = pos0 + i;
        let lo = if window > 0 { (pos + 1).saturating_sub(window) } else { 0 };
        for h in 0..heads {
            let kv = h / (heads / kvh);
            let qr = (pos * heads + h) * d;
            let mut qv = norm(&q[qr..qr + d], Some(&qn));
            rope(&mut qv, pos, base, ffo);
            let scores: Vec<f32> = (lo..=pos)
                .map(|s| dot(&qv, &kk[s * kw + kv * d..s * kw + (kv + 1) * d]))
                .collect();
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            for c in 0..d {
                let mut acc = 0.0;
                for (j, s) in (lo..=pos).enumerate() {
                    acc += e[j] * vv[s * kw + kv * d + c];
                }
                expected[(i * heads + h) * d + c] = acc / sum;
            }
        }
    }
    let got = out.to_f32().unwrap();
    let bad: Vec<(usize, usize)> = (0..m * heads)
        .filter(|&r| {
            (0..d).any(|c| {
                let (a, e) = (got[r * d + c], expected[r * d + c]);
                (a - e).abs() > 2e-3 + 2e-3 * e.abs()
            })
        })
        .map(|r| (r / heads, r % heads))
        .collect();
    assert!(
        bad.is_empty(),
        "d {d} window {window} pos0 {pos0} m {m}: bad (row, head) {:?}",
        &bad[..bad.len().min(20)]
    );
}

#[test]
fn attention_decode_matches_cpu() {
    attention_case(256, 8, 0, 300, 1, false);
    attention_case(256, 8, 40, 300, 3, false);
    attention_case(512, 2, 0, 600, 4, true);
    attention_case(512, 2, 0, 0, 1, true);
}

#[test]
fn attention_prefill_matches_cpu() {
    attention_case(256, 8, 40, 100, 5, false);
    attention_case(256, 8, 0, 10, 9, false);
    attention_case(256, 8, 40, 100, 50, false);
    attention_case(512, 2, 0, 70, 37, true);
    attention_case(256, 8, 1024, 0, 20, false);
    // Whole prompt runs: a ring that wraps, a global layer deep in its cache.
    attention_case(256, 8, 1024, 1100, 256, false);
    attention_case(512, 2, 0, 500, 200, true);
}

#[test]
fn router_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(7);
    let (m, e, kk) = (3, 128, 8);
    let logits = f32s(&mut rng, m * e, -4.0, 4.0);
    let scale = f32s(&mut rng, e, 0.9, 1.1);
    let (tl, ts) = (tensor(&ctx, &logits), tensor(&ctx, &scale));
    let sel = Tensor::zeros(&ctx, &[m * kk], DType::U32).unwrap();
    let w = zeros(&ctx, m * kk);
    run(&ctx, |p| router(&ctx, p, &tl, &ts, &sel, &w, m, e, kk).unwrap());
    let (gs, gw) = (sel.to_u32().unwrap(), w.to_f32().unwrap());
    for i in 0..m {
        let row = &logits[i * e..(i + 1) * e];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let p: Vec<f32> = row.iter().map(|l| (l - mx).exp()).collect();
        let s: f32 = p.iter().sum();
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        let top = &idx[..kk];
        let tsum: f32 = top.iter().map(|&j| p[j] / s).sum();
        for (r, &j) in top.iter().enumerate() {
            assert_eq!(gs[i * kk + r] as usize, j, "row {i} rank {r}");
            let want = p[j] / s / tsum * scale[j];
            assert!(
                (gw[i * kk + r] - want).abs() < 1e-5,
                "row {i}: {} vs {want}",
                gw[i * kk + r]
            );
        }
    }
}

/// Rows with NaN or infinite logits still pick `k` distinct experts below
/// E (the lowest-numbered remaining ones where the probabilities are NaN),
/// with finite weights; E is not a multiple of the subgroup.
#[test]
fn router_non_finite_rows_pick_experts_below_e() {
    let ctx = ctx();
    let (e, kk) = (100, 8);
    let mut rows = [vec![f32::NAN; e], vec![0.5; e], vec![0.0; e]];
    rows[1][7] = f32::INFINITY;
    rows[1][40] = f32::INFINITY;
    rows[2][3] = f32::NAN;
    let m = rows.len();
    let logits: Vec<f32> = rows.concat();
    let scale = vec![1.0; e];
    let (tl, ts) = (tensor(&ctx, &logits), tensor(&ctx, &scale));
    let sel = Tensor::zeros(&ctx, &[m * kk], DType::U32).unwrap();
    let w = zeros(&ctx, m * kk);
    run(&ctx, |p| router(&ctx, p, &tl, &ts, &sel, &w, m, e, kk).unwrap());
    let (gs, gw) = (sel.to_u32().unwrap(), w.to_f32().unwrap());
    for i in 0..m {
        let picks = &gs[i * kk..(i + 1) * kk];
        let mut sorted = picks.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), kk, "row {i}: {picks:?}");
        assert!(picks.iter().all(|&x| (x as usize) < e), "row {i}: {picks:?}");
        assert!(gw[i * kk..(i + 1) * kk].iter().all(|w| w.is_finite()), "row {i}");
    }
    // An all-NaN row takes experts 0..k at weight 0.
    assert_eq!(&gs[..kk], &(0..kk as u32).collect::<Vec<_>>()[..]);
    assert!(gw[..kk].iter().all(|&w| w == 0.0));
}

#[test]
fn moe_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(9);
    let (hid, ff, ne, used, m) = (512, 64, 6, 3, 2);
    for (gt, dt) in [
        (GgmlType::Q8_0, GgmlType::Q8_0),
        (GgmlType::Q5K, GgmlType::Iq4Nl),
        (GgmlType::Iq2Xxs, GgmlType::Q8_0),
    ] {
        let gu_raw = random_blocks(&mut rng, gt, ne * 2 * ff * hid);
        let dn_raw = random_blocks(&mut rng, dt, ne * hid * ff);
        let gu = QWeight::from_bytes(&ctx, &gu_raw, gt, hid, ne * 2 * ff).unwrap();
        let dn = QWeight::from_bytes(&ctx, &dn_raw, dt, ff, ne * hid).unwrap();
        let (guw, dnw) = (decode(gt, &gu_raw), decode(dt, &dn_raw));
        let x = f32s(&mut rng, m * hid, -1.0, 1.0);
        let sel: Vec<u32> = vec![0, 4, 2, 5, 4, 1];
        let wts = f32s(&mut rng, m * used, 0.1, 0.5);
        let (tx, ts, tw) = (tensor(&ctx, &x), u32s(&ctx, &sel), tensor(&ctx, &wts));
        let mid = zeros(&ctx, m * used * ff);
        let out = zeros(&ctx, m * hid);
        run(&ctx, |p| {
            moe_gate_up(&ctx, p, &gu, ff, &tx, &ts, &mid, m, used).unwrap();
            p.memory_barrier().unwrap();
            moe_down(&ctx, p, &dn, hid, &mid, &ts, &tw, &out, m, used).unwrap();
        });
        let mut want_mid = vec![0f32; m * used * ff];
        for i in 0..m {
            let xi = &x[i * hid..(i + 1) * hid];
            for s in 0..used {
                let e = sel[i * used + s] as usize;
                for r in 0..ff {
                    let g = dot(&guw[(e * 2 * ff + r) * hid..][..hid], xi);
                    let u = dot(&guw[(e * 2 * ff + ff + r) * hid..][..hid], xi);
                    want_mid[(i * used + s) * ff + r] = gelu(g) * u;
                }
            }
        }
        // Q5_K's dot factors the sub-block scale and min out of its sums
        // (as ggml's does): with these random blocks' large scales and
        // mins the two sums cancel to about 1e-4 of their size.
        let tol = if gt == GgmlType::Q5K { 1e-3 } else { 1e-4 };
        let dtol = if gt == GgmlType::Q5K { 1e-3 } else { 1e-4 };
        assert_close(&mid.to_f32().unwrap(), &want_mid, tol, tol);
        // The down from the GPU's own mid (its products are large here).
        let gm = mid.to_f32().unwrap();
        let mut want = vec![0f32; m * hid];
        for i in 0..m {
            for s in 0..used {
                let e = sel[i * used + s] as usize;
                let mrow = &gm[(i * used + s) * ff..][..ff];
                for r in 0..hid {
                    want[i * hid + r] +=
                        wts[i * used + s] * dot(&dnw[(e * hid + r) * ff..][..ff], mrow);
                }
            }
        }
        assert_close(&out.to_f32().unwrap(), &want, dtol, dtol);
    }
}

#[test]
fn norms_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(11);
    let (m, n, eps) = (3, 2816, 1e-6);
    let rows: Vec<Vec<f32>> =
        (0..5).map(|_| f32s(&mut rng, m * n, -2.0, 2.0)).collect();
    let ws: Vec<Vec<f32>> = (0..5).map(|_| f32s(&mut rng, n, 0.1, 3.0)).collect();
    let t: Vec<Tensor> = rows.iter().map(|r| tensor(&ctx, r)).collect();
    let tw: Vec<Tensor> = ws.iter().map(|w| tensor(&ctx, w)).collect();
    let ls = tensor(&ctx, &[0.7]);
    let o: Vec<Tensor> = (0..7).map(|_| zeros(&ctx, m * n)).collect();
    run(&ctx, |p| {
        // r = rmsnorm(a) w0 + x; o_j = rmsnorm(r) w_{j+1}.
        post_attn(
            &ctx,
            p,
            &t[0],
            &tw[0],
            &t[1],
            &o[0],
            [&tw[1], &tw[2], &tw[3]],
            [&o[1], &o[2], &o[3]],
            m,
            n,
            eps,
        )
        .unwrap();
        ffn_out(
            &ctx,
            p,
            &t[2],
            &t[3],
            [&tw[0], &tw[1], &tw[2]],
            &t[4],
            &ls,
            &o[4],
            Some((&tw[3], &o[5])),
            m,
            n,
            eps,
        )
        .unwrap();
        rmsnorm(&ctx, p, &t[0], &tw[4], &o[6], m, n, eps).unwrap();
    });
    let nrm = |x: &[f32], w: &[f32]| cpu_rmsnorm(x, Some(w), eps);
    let mut want: Vec<Vec<f32>> = vec![Vec::new(); 7];
    for i in 0..m {
        let r = |k: usize| &rows[k][i * n..(i + 1) * n];
        let res: Vec<f32> =
            nrm(r(0), &ws[0]).iter().zip(r(1)).map(|(a, b)| a + b).collect();
        for j in 0..3 {
            want[j + 1].extend(nrm(&res, &ws[j + 1]));
        }
        want[0].extend(res);
        let s: Vec<f32> = nrm(r(2), &ws[0])
            .iter()
            .zip(nrm(r(3), &ws[1]))
            .map(|(a, b)| a + b)
            .collect();
        let x: Vec<f32> =
            nrm(&s, &ws[2]).iter().zip(r(4)).map(|(a, b)| (a + b) * 0.7).collect();
        want[5].extend(nrm(&x, &ws[3]));
        want[4].extend(x);
        want[6].extend(nrm(r(0), &ws[4]));
    }
    for j in 0..7 {
        assert_close(&o[j].to_f32().unwrap(), &want[j], 1e-5, 1e-5);
    }
}

#[test]
fn embed_geglu_softcap_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(13);
    let (vocab, n) = (40, 2816);
    for ty in [GgmlType::Q8_0, GgmlType::Q5K] {
        let raw = random_blocks(&mut rng, ty, vocab * n);
        let table = QWeight::from_bytes(&ctx, &raw, ty, n, vocab).unwrap();
        let w = decode(ty, &raw);
        let toks = [3u32, 39, 0];
        let out = zeros(&ctx, 3 * n);
        let tt = u32s(&ctx, &toks);
        run(&ctx, |p| embed(&ctx, p, &table, &tt, &out, 3, 53.0).unwrap());
        let want: Vec<f32> = toks
            .iter()
            .flat_map(|&t| w[t as usize * n..][..n].iter().map(|v| v * 53.0))
            .collect();
        assert_eq!(out.to_f32().unwrap(), want);
    }
    let g = f32s(&mut rng, 1000, -12.0, 12.0);
    let u = f32s(&mut rng, 1000, -2.0, 2.0);
    let (tg, tu, to) = (tensor(&ctx, &g), tensor(&ctx, &u), zeros(&ctx, 1000));
    let x: Vec<f32> = f32s(&mut rng, 1000, -200.0, 200.0);
    let tx = tensor(&ctx, &x);
    run(&ctx, |p| {
        geglu(&ctx, p, &tg, &tu, &to, 1000).unwrap();
        softcap(&ctx, p, &tx, 1000, 30.0).unwrap();
    });
    let want: Vec<f32> = g.iter().zip(&u).map(|(g, u)| gelu(*g) * u).collect();
    assert_close(&to.to_f32().unwrap(), &want, 1e-5, 1e-5);
    let want: Vec<f32> = x.iter().map(|v| 30.0 * (v / 30.0).tanh()).collect();
    assert_close(&tx.to_f32().unwrap(), &want, 1e-5, 1e-5);
}

/// Per-dispatch time of the routed-expert kernels at Gemma 4's shapes
/// (hidden 2816, expert FF 704, top 8): one layer's gate/up and down, the
/// dispatches of a serial pass over four layers' weights in turn, best of
/// eight passes after one that wakes the GPU's clocks.
#[test]
#[ignore = "timing only"]
fn moe_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(70);
    // Passes as long as a decode step's (32 dispatches measured 30% apart).
    let repeat = 256;
    let (experts, hidden, ff, used) = (24, 2816, 704, 8);
    for (gt, dt) in
        [
            (GgmlType::Iq2Xxs, GgmlType::Iq4Nl),
            (GgmlType::Iq2S, GgmlType::Iq4Nl),
            (GgmlType::Q5K, GgmlType::Q8_0),
        ]
    {
        let layers: Vec<(QWeight, QWeight)> = (0..4)
            .map(|_| {
                let gu = random_blocks(&mut rng, gt, experts * 2 * ff * hidden);
                let dn = random_blocks(&mut rng, dt, experts * hidden * ff);
                (
                    QWeight::from_bytes(&ctx, &gu, gt, hidden, experts * 2 * ff)
                        .unwrap(),
                    QWeight::from_bytes(&ctx, &dn, dt, ff, experts * hidden).unwrap(),
                )
            })
            .collect();
        let x = tensor(&ctx, &f32s(&mut rng, hidden, -1.0, 1.0));
        let sel: Vec<u32> = (0..used as u32).map(|i| i * 3).collect();
        let ts = u32s(&ctx, &sel);
        let tw = tensor(&ctx, &vec![0.2f32; used]);
        let mid = zeros(&ctx, used * ff);
        let out = zeros(&ctx, hidden);
        // Q5_K gate/up also over int8 x (part 2).
        let codes = Tensor::zeros(&ctx, &[hidden], DType::I8).unwrap();
        let ds = zeros(&ctx, hidden / 16);
        run(&ctx, |p| quant_q8x(&ctx, p, &x, &codes, &ds, 1, hidden).unwrap());
        // IQ2_XXS / IQ4_NL also over f16 activations (parts 3 and 4).
        let xh = Tensor::zeros(&ctx, &[hidden], DType::F16).unwrap();
        let mid_h = Tensor::zeros(&ctx, &[used * ff], DType::F16).unwrap();
        run(&ctx, |p| {
            crate::kernels::deepseek4::x_f16_iq2(&ctx, p, &x, &xh, 1, hidden).unwrap()
        });
        let parts: &[usize] = match gt {
            GgmlType::Q5K => &[0, 1, 2],
            GgmlType::Iq2Xxs => &[0, 1, 3, 4],
            _ => &[0, 1],
        };
        let mut line = format!("moe {gt:?}/{dt:?}:");
        for &part in parts {
            let mut best = f64::MAX;
            for pass_no in 0..9 {
                let pass = ctx.begin().unwrap();
                for r in 0..repeat {
                    let (gu, dn) = &layers[r % 4];
                    if part == 0 {
                        moe_gate_up(&ctx, &pass, gu, ff, &x, &ts, &mid, 1, used)
                            .unwrap();
                    } else if part == 2 {
                        let q8 = Some((&codes, &ds));
                        moe_gate_up_q8x(
                            &ctx, &pass, gu, ff, &x, &ts, &mid, 1, used, q8,
                        )
                        .unwrap();
                    } else if part == 3 {
                        moe_gate_up_xh(
                            &ctx, &pass, gu, ff, &xh, &ts, &mid, &mid_h, 1, used,
                        )
                        .unwrap();
                    } else if part == 4 {
                        moe_down_xh(
                            &ctx, &pass, dn, hidden, &mid_h, &ts, &tw, &out, 1, used,
                        )
                        .unwrap();
                    } else {
                        moe_down(
                            &ctx, &pass, dn, hidden, &mid, &ts, &tw, &out, 1, used,
                        )
                        .unwrap();
                    }
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                if pass_no > 0 {
                    best = best.min(t.elapsed().as_secs_f64());
                }
            }
            line += &format!(
                " {} {:.1} us",
                ["gate/up", "down", "gate/up int8", "gate/up f16", "down f16"][part],
                best / repeat as f64 * 1e6
            );
        }
        eprintln!("{line}");
    }
}

/// Without matrix units (an M4) a prompt run's GEMMs are sgemm.comp's, f32
/// throughout: the dense GEMM over the block types of Gemma's files and the
/// grouped experts against f32 CPU sums, at shapes with partial tiles in
/// both directions. Each output within 2^-14 of the magnitudes it sums
/// (the summation order). On a device with matrix units these paths are
/// not taken and the test returns.
#[test]
fn gemms_without_matrix_units_match_cpu() {
    let ctx = ctx();
    if ctx.matrix_units() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(31);
    let tol = 2f32.powi(-14);
    for (ty, cols, rows, m) in [
        (GgmlType::Q8_0, 2112usize, 200usize, 70usize),
        (GgmlType::Q5K, 2816, 130, 129),
        (GgmlType::Q6K, 2816, 72, 64),
        (GgmlType::Q4K, 512, 64, 5),
        (GgmlType::F32, 2816, 128, 256),
        (GgmlType::Iq4Nl, 704, 96, 65),
        (GgmlType::Iq2Xxs, 512, 66, 33),
        (GgmlType::Iq2S, 512, 64, 100),
    ] {
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let wf = decode(ty, &raw);
        let x = f32s(&mut rng, m * cols, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let y = zeros(&ctx, m * rows);
        run(&ctx, |p| crate::kernels::glm53::gemm(&ctx, p, &w, &tx, &y, m).unwrap());
        let got = y.to_f32().unwrap();
        for i in 0..m {
            let xi = &x[i * cols..(i + 1) * cols];
            for n in 0..rows {
                let wn = &wf[n * cols..(n + 1) * cols];
                let want = dot(wn, xi);
                let bound: f32 = wn.iter().zip(xi).map(|(a, b)| (a * b).abs()).sum();
                let err = (got[i * rows + n] - want).abs();
                assert!(
                    err <= bound * tol + 1e-6,
                    "{ty:?} row {i} col {n}: {} vs {want}",
                    got[i * rows + n]
                );
            }
        }
    }
    let (hid, ff, ne, used, m) = (512, 128, 12, 8, 140);
    let sel: Vec<u32> =
        (0..m * used).map(|i| ((i * 7 + i / used) % ne) as u32).collect();
    for (gt, dt) in [
        (GgmlType::Iq2Xxs, GgmlType::Iq4Nl),
        (GgmlType::Iq2S, GgmlType::Iq4Nl),
        (GgmlType::Q5K, GgmlType::Q8_0),
    ] {
        grouped_f32_matches_cpu(&ctx, &mut rng, gt, dt, (hid, ff, ne, used, m), &sel);
    }
}

/// [`moe_grouped`] without matrix units over random `gt` gate/up and `dt`
/// down weights of `ne` experts (hidden `hid`, FF `ff`), `m` rows routed
/// to `sel` (`used` a row), against f32 CPU sums: each output within 2^-14
/// of the magnitudes it sums.
fn grouped_f32_matches_cpu(
    ctx: &MetalContext,
    rng: &mut StdRng,
    gt: GgmlType,
    dt: GgmlType,
    (hid, ff, ne, used, m): (usize, usize, usize, usize, usize),
    sel: &[u32],
) {
    let tol = 2f32.powi(-14);
    let gu_raw = random_blocks(rng, gt, ne * 2 * ff * hid);
    let dn_raw = random_blocks(rng, dt, ne * hid * ff);
    let gu = QWeight::from_bytes(ctx, &gu_raw, gt, hid, ne * 2 * ff).unwrap();
    let dn = QWeight::from_bytes(ctx, &dn_raw, dt, ff, ne * hid).unwrap();
    let (guw, dnw) = (decode(gt, &gu_raw), decode(dt, &dn_raw));
    let x = f32s(rng, m * hid, -1.0, 1.0);
    let wts = f32s(rng, m * used, 0.05, 0.2);
    let (tx, ts, tw) = (tensor(ctx, &x), u32s(ctx, sel), tensor(ctx, &wts));
    let grouped = zeros(ctx, m * hid);
    let sc = MoeGroupScratch::new(ctx, m, used, ne, ff, hid).unwrap();
    run(ctx, |p| {
        moe_grouped(ctx, p, &gu, &dn, ff, &tx, &ts, &tw, &grouped, &sc, m, used, ne)
            .unwrap()
    });
    let got = grouped.to_f32().unwrap();
    for i in 0..m {
        let xi = &x[i * hid..(i + 1) * hid];
        let (mut want, mut bound) = (vec![0f32; hid], vec![0f32; hid]);
        for s in 0..used {
            let e = sel[i * used + s] as usize;
            let w = &guw[e * 2 * ff * hid..(e + 1) * 2 * ff * hid];
            // The GeGLU's inputs to within their own sums' order: the
            // bound carries the gate and up magnitudes.
            let v: Vec<(f32, f32)> = (0..ff)
                .map(|j| {
                    let (gw, uw) =
                        (&w[j * hid..(j + 1) * hid], &w[(ff + j) * hid..][..hid]);
                    let g = dot(gw, xi);
                    let u = dot(uw, xi);
                    let mag = |r: &[f32]| {
                        r.iter().zip(xi).map(|(a, b)| (a * b).abs()).sum::<f32>()
                    };
                    let wt = wts[i * used + s];
                    (wt * (gelu(g) * u), wt * (mag(gw) + g.abs()) * (mag(uw) + u.abs()))
                })
                .collect();
            for r in 0..hid {
                let d = &dnw[(e * hid + r) * ff..(e * hid + r + 1) * ff];
                want[r] += d.iter().zip(&v).map(|(a, b)| a * b.0).sum::<f32>();
                bound[r] += d
                    .iter()
                    .zip(&v)
                    .map(|(a, b)| (a * b.0).abs() + a.abs() * b.1)
                    .sum::<f32>();
            }
        }
        for r in 0..hid {
            let err = (got[i * hid + r] - want[r]).abs();
            assert!(
                err <= bound[r] * tol + 1e-6,
                "{gt:?}/{dt:?} row {i} col {r}: {} vs {}",
                got[i * hid + r],
                want[r]
            );
        }
    }
}

/// Without matrix units a grouped-expert tile of up to 32 pairs is 32 rows
/// x 256 weight rows and a larger one 64 x 128 (sgemm.comp): experts of
/// exactly 32, 33, 64 and 65 pairs (65: a full tile and one of one pair),
/// at gate/up widths (2 x 192) and hidden sizes (320) that end inside a
/// tile, against f32 CPU sums.
#[test]
fn moe_grouped_tile_shapes_without_matrix_units_match_cpu() {
    let ctx = ctx();
    if ctx.matrix_units() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(32);
    let (ff, ne, used, m) = (192, 12, 8, 70);
    // Row i: experts 0 .. 3 while i < 32, 33, 64, 65, the rest of its
    // slots from experts 4 .. 11.
    let firsts = [32, 33, 64, 65];
    let mut sel = Vec::with_capacity(m * used);
    for i in 0..m {
        let row: Vec<u32> =
            (0..4).filter(|&e| i < firsts[e]).map(|e| e as u32).collect();
        let rest = (0..used - row.len()).map(|j| (4 + (i + j) % 8) as u32);
        sel.extend(row.iter().copied().chain(rest));
    }
    for (e, &n) in firsts.iter().enumerate() {
        assert_eq!(sel.iter().filter(|&&s| s as usize == e).count(), n);
    }
    for (gt, dt, hid) in [
        (GgmlType::Iq2Xxs, GgmlType::Iq4Nl, 512),
        (GgmlType::Q8_0, GgmlType::Q8_0, 320),
    ] {
        grouped_f32_matches_cpu(&ctx, &mut rng, gt, dt, (hid, ff, ne, used, m), &sel);
    }
}

/// The routed experts' decode kernels at output sizes that end inside a
/// subgroup's rows (on a G16 GPU, an M4, IQ2_XXS gate/up takes 4 rows a
/// subgroup, g16_moe_gate_up_iq2_xxs): expert FF 37, 41 and 42, every row
/// count, against the CPU and nothing written past the output.
#[test]
fn moe_gate_up_partial_rows_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(33);
    let (ty, hid, ne, used) = (GgmlType::Iq2Xxs, 256, 10, 8);
    const UNWRITTEN: f32 = 1234.5;
    for ff in [37usize, 41, 42] {
        let raw = random_blocks(&mut rng, ty, ne * 2 * ff * hid);
        let gu = QWeight::from_bytes(&ctx, &raw, ty, hid, ne * 2 * ff).unwrap();
        let guw = decode(ty, &raw);
        for m in 1..=4 {
            let x = f32s(&mut rng, m * hid, -1.0, 1.0);
            let sel: Vec<u32> = (0..m * used)
                .map(|k| ((3 * (k / used) + k % used) % ne) as u32)
                .collect();
            let mid = tensor(&ctx, &vec![UNWRITTEN; m * used * ff + 64]);
            run(&ctx, |p| {
                moe_gate_up(
                    &ctx,
                    p,
                    &gu,
                    ff,
                    &tensor(&ctx, &x),
                    &u32s(&ctx, &sel),
                    &mid,
                    m,
                    used,
                )
                .unwrap()
            });
            let mut want = vec![UNWRITTEN; m * used * ff + 64];
            for (k, &e) in sel.iter().enumerate() {
                let (e, xi) = (e as usize, &x[k / used * hid..][..hid]);
                for r in 0..ff {
                    let g = dot(&guw[(e * 2 * ff + r) * hid..][..hid], xi);
                    let u = dot(&guw[(e * 2 * ff + ff + r) * hid..][..hid], xi);
                    want[k * ff + r] = gelu(g) * u;
                }
            }
            assert_close(&mid.to_f32().unwrap(), &want, 1e-4, 1e-4);
        }
    }
}

/// [`moe_gate_up_partial_rows_match_cpu`]'s down: IQ4_NL (on a G16 GPU 2
/// rows a subgroup, g16_moe_down_iq4_nl) at hidden sizes 37 and 41.
#[test]
fn moe_down_partial_rows_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(34);
    let (ty, ff, ne, used) = (GgmlType::Iq4Nl, 128, 10, 8);
    const UNWRITTEN: f32 = 1234.5;
    for hid in [37usize, 41] {
        let raw = random_blocks(&mut rng, ty, ne * hid * ff);
        let dn = QWeight::from_bytes(&ctx, &raw, ty, ff, ne * hid).unwrap();
        let dnw = decode(ty, &raw);
        for m in 1..=4 {
            let mid = f32s(&mut rng, m * used * ff, -1.0, 1.0);
            let sel: Vec<u32> = (0..m * used)
                .map(|k| ((3 * (k / used) + k % used) % ne) as u32)
                .collect();
            let wts = f32s(&mut rng, m * used, 0.1, 0.5);
            let out = tensor(&ctx, &vec![UNWRITTEN; m * hid + 64]);
            run(&ctx, |p| {
                moe_down(
                    &ctx,
                    p,
                    &dn,
                    hid,
                    &tensor(&ctx, &mid),
                    &u32s(&ctx, &sel),
                    &tensor(&ctx, &wts),
                    &out,
                    m,
                    used,
                )
                .unwrap()
            });
            let mut want = vec![UNWRITTEN; m * hid + 64];
            for i in 0..m {
                for r in 0..hid {
                    want[i * hid + r] = (0..used)
                        .map(|s| {
                            let (k, e) = (i * used + s, sel[i * used + s] as usize);
                            wts[k]
                                * dot(
                                    &dnw[(e * hid + r) * ff..][..ff],
                                    &mid[k * ff..][..ff],
                                )
                        })
                        .sum();
                }
            }
            assert_close(&out.to_f32().unwrap(), &want, 1e-4, 1e-4);
        }
    }
}

/// The grouped experts of a prompt run against a CPU reference that rounds
/// as the kernels do: weights to bfloat to nearest with ties away from zero,
/// the input and the weighted GeGLU output to nearest even, f32 sums. Each
/// output is within one bfloat ulp of every down-projection input it sums
/// (a rounding the two summation orders can tip) plus the f32 sums' order.
/// 140 rows over 12 experts: each expert's pairs fill more than one 64-row
/// tile.
#[test]
fn moe_grouped_matches_a_bf16_reference() {
    let ctx = ctx();
    // Without matrix units the operands stay f32
    // (gemms_without_matrix_units_match_cpu).
    if !ctx.matrix_units() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(21);
    let (hid, ff, ne, used, m) = (512, 128, 12, 8, 140);
    let bfw = |v: f32| f32::from_bits((v.to_bits() + 0x8000) & 0xffff_0000);
    let bf = |v: f32| half::bf16::from_f32(v).to_f32();
    for (gt, dt) in
        [
            (GgmlType::Iq2Xxs, GgmlType::Iq4Nl),
            (GgmlType::Iq2S, GgmlType::Iq4Nl),
            (GgmlType::Q5K, GgmlType::Q8_0),
        ]
    {
        let gu_raw = random_blocks(&mut rng, gt, ne * 2 * ff * hid);
        let dn_raw = random_blocks(&mut rng, dt, ne * hid * ff);
        let gu = QWeight::from_bytes(&ctx, &gu_raw, gt, hid, ne * 2 * ff).unwrap();
        let dn = QWeight::from_bytes(&ctx, &dn_raw, dt, ff, ne * hid).unwrap();
        let guw: Vec<f32> = decode(gt, &gu_raw).into_iter().map(bfw).collect();
        let dnw: Vec<f32> = decode(dt, &dn_raw).into_iter().map(bfw).collect();
        let x = f32s(&mut rng, m * hid, -1.0, 1.0);
        let sel: Vec<u32> =
            (0..m * used).map(|i| ((i * 7 + i / used) % ne) as u32).collect();
        let wts = f32s(&mut rng, m * used, 0.05, 0.2);
        let (tx, ts, tw) = (tensor(&ctx, &x), u32s(&ctx, &sel), tensor(&ctx, &wts));
        let grouped = zeros(&ctx, m * hid);
        let sc = MoeGroupScratch::new(&ctx, m, used, ne, ff, hid).unwrap();
        run(&ctx, |p| {
            moe_grouped(
                &ctx, p, &gu, &dn, ff, &tx, &ts, &tw, &grouped, &sc, m, used, ne,
            )
            .unwrap()
        });
        let got = grouped.to_f32().unwrap();
        let mut counts = vec![0usize; ne];
        sel.iter().for_each(|&e| counts[e as usize] += 1);
        assert!(counts.iter().all(|&c| c > 64), "{counts:?}");
        let (mut worst, mut mean) = (0f64, 0f64);
        for i in 0..m {
            let xb: Vec<f32> =
                x[i * hid..(i + 1) * hid].iter().map(|&v| bf(v)).collect();
            let (mut want, mut bound) = (vec![0f32; hid], vec![0f32; hid]);
            for s in 0..used {
                let e = sel[i * used + s] as usize;
                let w = &guw[e * 2 * ff * hid..(e + 1) * 2 * ff * hid];
                let v: Vec<f32> = (0..ff)
                    .map(|j| {
                        let g = dot(&w[j * hid..(j + 1) * hid], &xb);
                        let u = dot(&w[(ff + j) * hid..(ff + j + 1) * hid], &xb);
                        bf(wts[i * used + s] * (gelu(g) * u))
                    })
                    .collect();
                for r in 0..hid {
                    let d = &dnw[(e * hid + r) * ff..(e * hid + r + 1) * ff];
                    want[r] += dot(d, &v);
                    bound[r] +=
                        d.iter().zip(&v).map(|(a, b)| (a * b).abs()).sum::<f32>();
                }
            }
            for r in 0..hid {
                let err = (got[i * hid + r] - want[r]).abs();
                let tol = bound[r] * (2f32.powi(-7) + 1e-5) + 1e-7;
                assert!(
                    err <= tol,
                    "{gt:?}/{dt:?} row {i} col {r}: {} vs {} (bound {tol})",
                    got[i * hid + r],
                    want[r]
                );
                worst = worst.max((err / tol) as f64);
                mean += err as f64 / bound[r].max(1e-30) as f64;
            }
        }
        eprintln!(
            "{gt:?}/{dt:?}: worst error {worst:.3} of the bound, mean {:.2e} of the inputs' magnitude",
            mean / (m * hid) as f64
        );
    }
}

/// Each kernel's least GPU time in one pass among the profiled passes
/// since the last call.
fn best_kernel_secs() -> std::collections::BTreeMap<&'static str, f64> {
    let mut best = std::collections::BTreeMap::new();
    for p in crate::metal::profile::take() {
        let mut by: std::collections::BTreeMap<&'static str, f64> = Default::default();
        for k in &p.kernels {
            *by.entry(k.name).or_default() += k.gpu_secs;
        }
        for (k, s) in by {
            let e = best.entry(k).or_insert(f64::MAX);
            *e = s.min(*e);
        }
    }
    best
}

/// Prompt-run GEMM speed at Gemma 4 26B-A4B's shapes for a run of
/// `LILY_PROBE_M` rows (default 1024): the dense projections and the grouped
/// experts (routing skewed over 128 experts), best of 8 per kernel from the
/// per-kernel profile, ms and TFLOPS. Run with `--ignored --nocapture`.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn prefill_gemm_probe() {
    let ctx = MetalContext::new_with_profile(true).expect("metal context");
    // The GPU's clock varies (a fanless Mac's heat, and it rises only under
    // steady load): before each case one pass of 24 FMA peak kernels (~0.3
    // s) brings it up, and the kernel's rate after it is the yardstick.
    #[cfg(target_os = "linux")]
    let clock = |when: &str| {
        let fma = ctx.pipeline("pk_fma_f32", "", MslVersion::V3_1).expect("fma");
        let out = zeros(&ctx, 1024);
        let grid =
            Grid::Threadgroups { groups: (1024, 1, 1), threadgroup: (256, 1, 1) };
        let rounds = u32_bytes(2048);
        run(&ctx, |p| {
            for _ in 0..24 {
                p.dispatch_at(&fma, &[out.binding()], &[&rounds], grid).unwrap();
            }
        });
        crate::metal::profile::take();
        for _ in 0..4 {
            run(&ctx, |p| {
                p.dispatch_at(&fma, &[out.binding()], &[&rounds], grid).unwrap()
            });
        }
        let s = best_kernel_secs()["pk_fma_f32"];
        let flops = 2.0 * (1024.0 * 256.0 * 2048.0 * 32.0);
        eprintln!("clock {when}: f32 FMA {:.3} TFLOPS", flops / s / 1e12);
    };
    let mut rng = StdRng::seed_from_u64(5);
    let m =
        std::env::var("LILY_PROBE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    for (ty, n, k) in [
        (GgmlType::Q5K, 4096usize, 2816usize),
        (GgmlType::Q5K, 2816, 4096),
        (GgmlType::Q8_0, 2816, 2112),
    ] {
        let raw = random_blocks(&mut rng, ty, n * k);
        let w = QWeight::from_bytes(&ctx, &raw, ty, k, n).unwrap();
        let x = tensor(&ctx, &f32s(&mut rng, m * k, -1.0, 1.0));
        let y = zeros(&ctx, m * n);
        #[cfg(target_os = "linux")]
        clock("before");
        for _ in 0..8 {
            run(&ctx, |p| crate::kernels::glm53::gemm(&ctx, p, &w, &x, &y, m).unwrap());
        }
        for (name, s) in best_kernel_secs() {
            let flops = 2.0 * (m * n * k) as f64;
            eprintln!(
                "{ty:?} {m}x{n}x{k}: {name} {:.3} ms {:.3} TFLOPS",
                s * 1e3,
                flops / s / 1e12
            );
        }
    }
    let (hid, ff, ne, used) = (2816usize, 704usize, 128usize, 8usize);
    // Expert e drawn with weight 1 / (e + 8): the busiest expert several
    // times the quietest, as a prompt's routing is uneven.
    let pw: Vec<f64> = (0..ne).map(|e| 1.0 / (e as f64 + 8.0)).collect();
    let total: f64 = pw.iter().sum();
    let mut sel = Vec::with_capacity(m * used);
    for _ in 0..m {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < used {
            let mut t = rng.gen_range(0.0..total);
            let mut e = 0;
            while t >= pw[e] {
                t -= pw[e];
                e += 1;
            }
            if !row.contains(&(e as u32)) {
                row.push(e as u32);
            }
        }
        sel.extend(row);
    }
    let gu_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, ne * 2 * ff * hid);
    let gu =
        QWeight::from_bytes(&ctx, &gu_raw, GgmlType::Iq2Xxs, hid, ne * 2 * ff).unwrap();
    drop(gu_raw);
    let dn_raw = random_blocks(&mut rng, GgmlType::Iq4Nl, ne * hid * ff);
    let dn = QWeight::from_bytes(&ctx, &dn_raw, GgmlType::Iq4Nl, ff, ne * hid).unwrap();
    drop(dn_raw);
    let x = tensor(&ctx, &f32s(&mut rng, m * hid, -1.0, 1.0));
    let ts = u32s(&ctx, &sel);
    let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.05, 0.2));
    let out = zeros(&ctx, m * hid);
    let sc = MoeGroupScratch::new(&ctx, m, used, ne, ff, hid).unwrap();
    #[cfg(target_os = "linux")]
    clock("before");
    for _ in 0..8 {
        run(&ctx, |p| {
            moe_grouped(&ctx, p, &gu, &dn, ff, &x, &ts, &tw, &out, &sc, m, used, ne)
                .unwrap()
        });
    }
    let gemms = best_kernel_secs().into_iter().filter(|(n, _)| n.contains("gemm"));
    for (name, s) in gemms {
        // Gate/up: 2 ff x hid a pair; down: hid x ff.
        let macs = if name.contains("iq2") { 2 * ff * hid } else { hid * ff };
        let flops = 2.0 * (m * used * macs) as f64;
        eprintln!("grouped {name} {:.3} ms {:.3} TFLOPS", s * 1e3, flops / s / 1e12);
    }
    #[cfg(target_os = "linux")]
    clock("after");
}

/// Per-dispatch time of the decode GEMVs at Gemma 4's shapes (one row, or
/// `LILY_GEMV_M` rows):
/// attention and dense FFN projections as the model groups them, the tied
/// LM head and the dense down, best of eight passes of 128 dispatches (16
/// of the head) over weight copies in turn (so the cache does not hold
/// them), after a pass that wakes the GPU's clocks.
#[test]
#[ignore = "timing only"]
fn gemv_rows_timing() {
    use crate::kernels::glm53::gemv_sets;
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(71);
    let m: usize =
        std::env::var("LILY_GEMV_M").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let shapes: [(&str, GgmlType, usize, &[usize]); 7] = [
        ("local q/k/v Q5_K", GgmlType::Q5K, 2816, &[4096, 2048, 2048]),
        ("local o Q5_K", GgmlType::Q5K, 4096, &[2816]),
        ("gate/up Q5_K", GgmlType::Q5K, 2816, &[2112, 2112]),
        ("global q/k Q5_K", GgmlType::Q5K, 2816, &[8192, 1024]),
        ("global o Q5_K", GgmlType::Q5K, 8192, &[2816]),
        ("head Q5_K", GgmlType::Q5K, 2816, &[262144]),
        ("down Q8_0", GgmlType::Q8_0, 2112, &[2816]),
    ];
    for (label, ty, cols, rows) in shapes {
        let total: usize = rows.iter().sum();
        let copies = if total > 100_000 { 1 } else { 4 };
        let ws: Vec<Vec<QWeight>> = (0..copies)
            .map(|_| {
                rows.iter()
                    .map(|&r| {
                        let raw = random_blocks(&mut rng, ty, r * cols);
                        QWeight::from_bytes(&ctx, &raw, ty, cols, r).unwrap()
                    })
                    .collect()
            })
            .collect();
        let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
        let ys: Vec<Tensor> = rows.iter().map(|&r| zeros(&ctx, m * r)).collect();
        let repeat = if total > 100_000 { 16 } else { 128 };
        let mut best = f64::MAX;
        for pass_no in 0..9 {
            let pass = ctx.begin().unwrap();
            for r in 0..repeat {
                let sets: Vec<(&QWeight, &Tensor)> =
                    ws[r % copies].iter().zip(&ys).collect();
                gemv_sets(&ctx, &pass, &sets, &x, m).unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            if pass_no > 0 {
                best = best.min(t.elapsed().as_secs_f64());
            }
        }
        let bytes: usize = ws[0].iter().map(|w| w.row_bytes * w.rows).sum();
        let per = best / repeat as f64;
        eprintln!(
            "gemv {label:20} {total:6}x{cols:<5} {:8.1} us {:6.1} GB/s",
            per * 1e6,
            bytes as f64 / per / 1e9
        );
    }
}

/// The decode step's dense GEMVs at Gemma's shapes, one row, as the model
/// groups them (q/k/v one dispatch, gate/up one), Q5_K (UD-IQ2_M) and Q8_0
/// (UD-Q5_K_M); best of five passes of 256 dispatches over four weight
/// copies (the head: one copy, 16 dispatches): passes as long as a decode
/// step's, shorter ones measured 30% apart run to run.
#[test]
#[ignore = "timing only"]
fn gemv_timing() {
    use crate::kernels::glm53::gemv_sets;
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(71);
    let h = 2816;
    for ty in [GgmlType::Q5K, GgmlType::Q8_0] {
        let shapes: [(&str, usize, &[usize]); 5] = [
            ("q/k/v", h, &[4096, 2048, 2048]),
            ("o", 4096, &[h]),
            ("gate/up", h, &[2112, 2112]),
            ("down", 2112, &[h]),
            ("head", h, &[262144]),
        ];
        for (label, cols, rows) in shapes {
            let ty = if label == "down" { GgmlType::Q8_0 } else { ty };
            if label == "down" && ty == GgmlType::Q8_0 && cols != 2112 {
                continue;
            }
            let copies = if label == "head" { 1 } else { 4 };
            let repeat = if label == "head" { 16 } else { 256 };
            let ws: Vec<Vec<QWeight>> = (0..copies)
                .map(|_| {
                    rows.iter()
                        .map(|&r| {
                            let raw = random_blocks(&mut rng, ty, r * cols);
                            QWeight::from_bytes(&ctx, &raw, ty, cols, r).unwrap()
                        })
                        .collect()
                })
                .collect();
            let x = tensor(&ctx, &f32s(&mut rng, cols, -1.0, 1.0));
            let ys: Vec<Tensor> = rows.iter().map(|&r| zeros(&ctx, r)).collect();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let pass = ctx.begin().unwrap();
                for r in 0..repeat {
                    let sets: Vec<(&QWeight, &Tensor)> =
                        ws[r % copies].iter().zip(ys.iter()).collect();
                    gemv_sets(&ctx, &pass, &sets, &x, 1).unwrap();
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            let bytes: usize = ws[0].iter().map(|w| w.row_bytes * w.rows).sum();
            let per = best / repeat as f64;
            eprintln!(
                "gemv {ty:?} {label:8} {:8.1} us {:6.1} GB/s",
                per * 1e6,
                bytes as f64 / per / 1e9
            );
        }
    }
}

/// The two-rows-a-subgroup Q5_K GEMV (weights past GEMV_R2_MIN_ROWS rows,
/// the LM head) against the CPU, an odd row count, rows of fewer and of
/// more than 32 tasks.
#[test]
fn gemv_wide_q5_k_matches_cpu() {
    use crate::kernels::glm53::gemv;
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(72);
    for (cols, rows) in [(512, 32771), (2816, 32769)] {
        let raw = random_blocks(&mut rng, GgmlType::Q5K, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q5K, cols, rows).unwrap();
        let xv = f32s(&mut rng, cols, -1.0, 1.0);
        let x = tensor(&ctx, &xv);
        let y = zeros(&ctx, rows);
        run(&ctx, |pass| gemv(&ctx, pass, &w, &x, &y, 1).unwrap());
        let wf = decode(GgmlType::Q5K, &raw);
        let expected: Vec<f32> =
            (0..rows).map(|r| dot(&wf[r * cols..(r + 1) * cols], &xv)).collect();
        assert_close(&y.to_f32().unwrap(), &expected, 1e-4, 1e-4);
    }
}

/// The bfloat-input GEMM over Gemma's Q5_K and Q6_K weights against the
/// f32-input one, bit for bit: rows inside and across 256-row tiles,
/// partial column tiles, inputs at bfloat rounding ties.
#[test]
fn gemm_bx_q5_k_q6_k_match_gemm() {
    use crate::kernels::glm53::{gemm, gemm_bx, to_bf16};
    let ctx = ctx();
    // The bfloat-input GEMM uses matrix units.
    if !ctx.matrix_units() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(73);
    for (ty, cols, rows, m) in [
        (GgmlType::Q5K, 512usize, 72usize, 65usize),
        (GgmlType::Q5K, 256, 100, 256),
        (GgmlType::Q5K, 512, 40, 513),
        (GgmlType::Q6K, 256, 72, 70),
        (GgmlType::Q6K, 512, 136, 257),
    ] {
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let mut x = f32s(&mut rng, m * cols, -3.0, 3.0);
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

/// The device's read bandwidth (tests/vulkan/g16_probe_test.comp) over a
/// 512 MB buffer at several grid sizes: the ceiling the GEMVs aim at.
#[test]
#[ignore = "timing only"]
fn read_bandwidth_probe() {
    let ctx = ctx();
    let bytes = 512usize << 20;
    let src = Tensor::zeros(&ctx, &[bytes / 4], DType::F32).unwrap();
    let out = zeros(&ctx, 1);
    let pipeline = ctx.pipeline("g16_read_bw", SOURCE, MslVersion::V3_1).unwrap();
    for groups in [64usize, 128, 256, 512, 1024, 2048, 4096] {
        let mut best = f64::MAX;
        for pass_no in 0..6 {
            let pass = ctx.begin().unwrap();
            for _ in 0..4 {
                pass.dispatch_at(
                    &pipeline,
                    &[src.binding(), out.binding()],
                    &[&u32_bytes(bytes / 16)],
                    Grid::Threadgroups {
                        groups: (groups, 1, 1),
                        threadgroup: (256, 1, 1),
                    },
                )
                .unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            if pass_no > 0 {
                best = best.min(t.elapsed().as_secs_f64() / 4.0);
            }
        }
        eprintln!("read {groups:5} groups: {:7.1} GB/s", bytes as f64 / best / 1e9);
    }
}

/// Decode attention (one row) at a sliding-window layer's shape (256-wide
/// heads, 8 KV heads, window 1024) and a global one's (512, 2 KV heads)
/// at 1K and 4K positions: best of five passes of 128 layers (two cache
/// copies alternating).
#[test]
#[ignore = "timing only"]
fn attention_decode_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(74);
    for (label, d, kvh, window, pos) in [
        ("local 1K", 256, 8, 1024, 1000),
        ("local 4K", 256, 8, 1024, 4000),
        ("global 1K", 512, 2, 0, 1000),
        ("global 4K", 512, 2, 0, 4000),
        ("global 16K", 512, 2, 0, 16000),
        ("global 16K full K", 512, 2, 0, 16001),
    ] {
        let heads = 16;
        let ring = if window > 0 { window + 256 } else { pos + 8 };
        let caches: Vec<(Tensor, Tensor)> = (0..2)
            .map(|_| {
                let k: Vec<f32> = f32s(&mut rng, ring * kvh * d, -0.1, 0.1);
                let v: Vec<f32> = f32s(&mut rng, ring * kvh * d, -1.0, 1.0);
                let h = |x: &[f32]| {
                    let o = Tensor::zeros(&ctx, &[x.len()], DType::F16).unwrap();
                    let b: Vec<u8> =
                        x.iter().flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes()).collect();
                    o.write_bytes(&b).unwrap();
                    o
                };
                (h(&k), h(&v))
            })
            .collect();
        let q = tensor(&ctx, &f32s(&mut rng, heads * d, -1.0, 1.0));
        let kn = tensor(&ctx, &f32s(&mut rng, d, 0.5, 1.5));
        let out = zeros(&ctx, heads * d);
        let chunks = (pos + 1).min(if window > 0 { window } else { usize::MAX }).div_ceil(ATTN_CHUNK);
        let partial = zeros(&ctx, attn_partial_len(1, heads, d, chunks));
        let repeat = 128;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin_concurrent().unwrap();
            for r in 0..repeat {
                let (kc, vc) = &caches[r % 2];
                let a = AttnShape {
                    head_dim: d,
                    heads,
                    kv_heads: kvh,
                    ring,
                    window,
                    k_cache: kc,
                    v_cache: vc,
                    k_rot: if window > 0 || pos == 16001 { d / 2 } else { d / 8 },
                    k_norm: &kn,
                };
                attention(&ctx, &pass, &a, &q, &partial, &out, pos, 1, 4).unwrap();
                pass.memory_barrier().unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!("attn decode {label:10} {:7.1} us", best / repeat as f64 * 1e6);
    }
}

/// GeGLU written over its gate (the scratch's `mid` is `g`) equals it
/// written apart, bit for bit, at decode and prompt-run lengths.
#[test]
fn geglu_in_place_matches_geglu_apart() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(77);
    for n in [2112, 4 * 2112, 1000 * 2112 + 7] {
        let g = f32s(&mut rng, n, -6.0, 6.0);
        let u = f32s(&mut rng, n, -3.0, 3.0);
        let (tg, tu, apart) = (tensor(&ctx, &g), tensor(&ctx, &u), zeros(&ctx, n));
        let inplace = tensor(&ctx, &g);
        run(&ctx, |p| {
            geglu(&ctx, p, &tg, &tu, &apart, n).unwrap();
            geglu(&ctx, p, &inplace, &tu, &inplace, n).unwrap();
        });
        let (a, b) = (apart.to_f32().unwrap(), inplace.to_f32().unwrap());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "n {n}");
    }
}

/// The grouped experts with their outputs in a pool supplied from outside
/// (a view inside a larger buffer, as the model's scratch shares it) equal
/// them in their own scratch, bit for bit, and write nothing outside it.
#[test]
fn moe_grouped_in_an_external_pool_stays_in_it() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(78);
    let (hid, ff, ne, used) = (512, 128, 12, 8);
    let (gt, dt) = (GgmlType::Iq2Xxs, GgmlType::Iq4Nl);
    let gu_raw = random_blocks(&mut rng, gt, ne * 2 * ff * hid);
    let dn_raw = random_blocks(&mut rng, dt, ne * hid * ff);
    let gu = QWeight::from_bytes(&ctx, &gu_raw, gt, hid, ne * 2 * ff).unwrap();
    let dn = QWeight::from_bytes(&ctx, &dn_raw, dt, ff, ne * hid).unwrap();
    // Scratch for 140 rows; runs of fewer use the pool's front.
    let rows = 140;
    let len = MoeGroupScratch::pool_len(rows, used, ff, hid);
    let pad = 4096;
    let sentinel = -12345.5f32;
    let big = tensor(&ctx, &vec![sentinel; pad + len + pad]);
    let pool = big.view(pad, &[len]).unwrap();
    let own = MoeGroupScratch::new(&ctx, rows, used, ne, ff, hid).unwrap();
    let shared = MoeGroupScratch::in_pool(&ctx, rows, used, ne, ff, hid, &pool).unwrap();
    for m in [140, 37, 5] {
        let x = tensor(&ctx, &f32s(&mut rng, m * hid, -1.0, 1.0));
        let sel: Vec<u32> = (0..m * used).map(|i| ((i * 7 + i / used) % ne) as u32).collect();
        let ts = u32s(&ctx, &sel);
        let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.05, 0.2));
        let (a, b) = (zeros(&ctx, m * hid), zeros(&ctx, m * hid));
        run(&ctx, |p| {
            moe_grouped(&ctx, p, &gu, &dn, ff, &x, &ts, &tw, &a, &own, m, used, ne).unwrap();
            p.memory_barrier().unwrap();
            moe_grouped(&ctx, p, &gu, &dn, ff, &x, &ts, &tw, &b, &shared, m, used, ne).unwrap();
        });
        let (a, b) = (a.to_f32().unwrap(), b.to_f32().unwrap());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "m {m}");
        let all = big.to_f32().unwrap();
        let outside = all[..pad].iter().chain(&all[pad + len..]);
        assert!(outside.into_iter().all(|&v| v == sentinel), "m {m}: wrote outside the pool");
    }
}

/// The Q5_K GEMV over int8 activations ([`gemv_q5_k_q8x`] after
/// [`quant_q8x`]) against glm53_gemv_q5_k_m1 at Gemma's shapes: time
/// (passes of 256 dispatches over four copies) and error against an f32 CPU
/// sum.
#[test]
#[ignore = "timing only"]
fn int8_gemv_probe() {
    use crate::kernels::glm53::gemv;
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(79);
    for (label, rows, cols) in
        [("q/k/v", 8192, 2816), ("o", 2816, 4096), ("gate/up", 4224, 2816)]
    {
        let raws: Vec<Vec<u8>> = (0..4)
            .map(|_| random_blocks(&mut rng, GgmlType::Q5K, rows * cols))
            .collect();
        let ws: Vec<QWeight> = raws
            .iter()
            .map(|r| QWeight::from_bytes(&ctx, r, GgmlType::Q5K, cols, rows).unwrap())
            .collect();
        let xv = f32s(&mut rng, cols, -1.0, 1.0);
        let x = tensor(&ctx, &xv);
        let codes = Tensor::zeros(&ctx, &[cols], DType::I8).unwrap();
        let ds = zeros(&ctx, cols / 16);
        let (y, y8) = (zeros(&ctx, rows), zeros(&ctx, rows));
        let q =
            |p: &ComputePass<'_>| quant_q8x(&ctx, p, &x, &codes, &ds, 1, cols).unwrap();
        let g8 = |p: &ComputePass<'_>, w: &QWeight| {
            gemv_q5_k_q8x(&ctx, p, &[(w, &y8)], &codes, &ds, 1).unwrap()
        };
        run(&ctx, |p| {
            q(p);
            p.memory_barrier().unwrap();
            g8(p, &ws[0]);
            gemv(&ctx, p, &ws[0], &x, &y, 1).unwrap();
        });
        let wf = decode(GgmlType::Q5K, &raws[0]);
        let cpu: Vec<f32> =
            (0..rows).map(|r| dot(&wf[r * cols..(r + 1) * cols], &xv)).collect();
        let rms = (cpu.iter().map(|v| v * v).sum::<f32>() / rows as f32).sqrt();
        let err = |g: &[f32]| {
            g.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / rms
        };
        let (e32, e8) = (err(&y.to_f32().unwrap()), err(&y8.to_f32().unwrap()));
        let time = |f: &dyn Fn(&ComputePass<'_>, usize)| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let pass = ctx.begin().unwrap();
                for r in 0..256 {
                    f(&pass, r);
                }
                let t = std::time::Instant::now();
                pass.commit_wait().unwrap();
                best = best.min(t.elapsed().as_secs_f64() / 256.0);
            }
            best * 1e6
        };
        let t32 = time(&|p, r| gemv(&ctx, p, &ws[r % 4], &x, &y, 1).unwrap());
        let t8 = time(&|p, r| g8(p, &ws[r % 4]));
        let tq = time(&|p, _| q(p));
        eprintln!(
            "int8 {label:8} f32-x {t32:6.1} us (err {e32:.2e} of rms) | int8-x {t8:6.1} us + quant {tq:5.1} us (err {e8:.2e})"
        );
    }
}

/// The int8-activation Q5_K GEMV's rows of 2 to 4-row kernels
/// equal the one-row kernel's, bit for bit (a verify pass decodes as a
/// plain step), and it is within int8's rounding of the f32-input GEMV.
#[test]
fn int8_gemv_rows_match_one_row() {
    use crate::kernels::glm53::gemv;
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(80);
    let (rows, cols) = (520, 2816);
    let raw = random_blocks(&mut rng, GgmlType::Q5K, rows * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q5K, cols, rows).unwrap();
    let xv = f32s(&mut rng, 4 * cols, -1.0, 1.0);
    let x = tensor(&ctx, &xv);
    let codes = Tensor::zeros(&ctx, &[4 * cols], DType::I8).unwrap();
    let ds = zeros(&ctx, 4 * cols / 16);
    let mut one = Vec::new();
    for i in 0..4 {
        let xi = x.view(i * cols, &[cols]).unwrap();
        let (c1, d1, y) = (
            Tensor::zeros(&ctx, &[cols], DType::I8).unwrap(),
            zeros(&ctx, cols / 16),
            zeros(&ctx, rows),
        );
        run(&ctx, |p| {
            quant_q8x(&ctx, p, &xi, &c1, &d1, 1, cols).unwrap();
            p.memory_barrier().unwrap();
            gemv_q5_k_q8x(&ctx, p, &[(&w, &y)], &c1, &d1, 1).unwrap();
        });
        one.push(y.to_f32().unwrap());
    }
    for m in 2..=4 {
        let y = zeros(&ctx, m * rows);
        run(&ctx, |p| {
            quant_q8x(&ctx, p, &x, &codes, &ds, m, cols).unwrap();
            p.memory_barrier().unwrap();
            gemv_q5_k_q8x(&ctx, p, &[(&w, &y)], &codes, &ds, m).unwrap();
        });
        let got = y.to_f32().unwrap();
        for i in 0..m {
            assert!(
                got[i * rows..(i + 1) * rows]
                    .iter()
                    .zip(&one[i])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "m {m} row {i}"
            );
        }
    }
    let yf = zeros(&ctx, rows);
    run(&ctx, |p| gemv(&ctx, p, &w, &x.view(0, &[cols]).unwrap(), &yf, 1).unwrap());
    let f = yf.to_f32().unwrap();
    let rms = (f.iter().map(|v| v * v).sum::<f32>() / rows as f32).sqrt();
    let err =
        one[0].iter().zip(&f).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / rms;
    assert!(err < 0.05, "int8 against f32 activations: {err} of rms");
}

/// rmsnorm's, ffn_out's and the attention merge's int8 outputs equal
/// quant_q8x of their f32 outputs, bit for bit, edge blocks included
/// (zero, subnormal, NaN, infinities), and the f32 outputs are the plain
/// kernels'.
#[test]
fn producers_write_the_int8_quant_q8x_writes() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(81);
    let (m, n, eps) = (3, 2816, 1e-6);
    let t: Vec<Tensor> =
        (0..5).map(|_| tensor(&ctx, &f32s(&mut rng, m * n, -2.0, 2.0))).collect();
    let tw: Vec<Tensor> =
        (0..5).map(|_| tensor(&ctx, &f32s(&mut rng, n, 0.1, 3.0))).collect();
    // The weight of each producer's int8 output, with edge blocks: zero,
    // subnormal products, a NaN, overflow to infinities, and tiny normal
    // values.
    let mut swv = f32s(&mut rng, n, 0.1, 3.0);
    for e in 0..32 {
        swv[e] = 0.0;
        swv[32 + e] = 1e-40;
        swv[96 + e] = f32::MAX;
        swv[128 + e] = f32::INFINITY;
        swv[160 + e] = 1e-36;
    }
    swv[64 + 7] = f32::NAN;
    let sw = tensor(&ctx, &swv);
    let ls = tensor(&ctx, &[0.7]);
    let q8 = || {
        (
            Tensor::zeros(&ctx, &[m * 8192], DType::I8).unwrap(),
            zeros(&ctx, m * 8192 / 16),
        )
    };
    let same = |a: &Tensor, b: &Tensor| {
        let (a, b) = (a.to_f32().unwrap(), b.to_f32().unwrap());
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    let same_codes = |a: &Tensor, b: &Tensor, len: usize| {
        a.to_f32().unwrap()[..len] == b.to_f32().unwrap()[..len]
    };
    // The f32 output of each producer, plain and with int8, then
    // quant_q8x of the plain one (post_attn: its own test below).
    for name in ["rmsnorm", "ffn_out"] {
        let k = n;
        let (plain, with) = (zeros(&ctx, m * n), zeros(&ctx, m * n));
        let ((c1, d1), (c2, d2)) = (q8(), q8());
        let o: Vec<Tensor> = (0..6).map(|_| zeros(&ctx, m * n)).collect();
        run(&ctx, |p| {
            match name {
                "rmsnorm" => {
                    rmsnorm(&ctx, p, &t[0], &sw, &plain, m, n, eps).unwrap();
                    rmsnorm_q8x(
                        &ctx,
                        p,
                        &t[0],
                        &sw,
                        &with,
                        m,
                        n,
                        eps,
                        Some((&c1, &d1)),
                    )
                    .unwrap();
                }
                _ => {
                    ffn_out(
                        &ctx,
                        p,
                        &t[2],
                        &t[3],
                        [&tw[0], &tw[1], &tw[2]],
                        &t[4],
                        &ls,
                        &o[0],
                        Some((&sw, &plain)),
                        m,
                        n,
                        eps,
                    )
                    .unwrap();
                    ffn_out_q8x(
                        &ctx,
                        p,
                        &t[2],
                        &t[3],
                        [&tw[0], &tw[1], &tw[2]],
                        &t[4],
                        &ls,
                        &o[1],
                        Some((&sw, &with)),
                        m,
                        n,
                        eps,
                        Some((&c1, &d1)),
                    )
                    .unwrap();
                }
            }
            p.memory_barrier().unwrap();
            quant_q8x(&ctx, p, &plain, &c2, &d2, m, k).unwrap();
        });
        assert!(same(&plain, &with), "{name}: f32 output");
        assert!(same_codes(&c1, &c2, m * k), "{name}: codes");
        assert!(
            same(
                &d1.view(0, &[m * k / 16]).unwrap(),
                &d2.view(0, &[m * k / 16]).unwrap()
            ),
            "{name}: scales"
        );
        let ds = d1.to_f32().unwrap();
        let scales: Vec<f32> = ds[..m * k / 16].iter().step_by(2).copied().collect();
        assert!(scales.iter().any(|d| d.is_nan()), "{name}: no nonfinite block");
        assert!(scales.iter().any(|&d| d == 0.0), "{name}: no zero block");
    }
    // The attention merge: heads of 16 x 256 a row.
    let (d, kvh, heads, pos) = (256, 8, 16, 700);
    let ring = 1024 + 256;
    // f16 caches, V with edge dimensions: zero, a NaN, infinities of
    // both signs.
    let f16s = |v: &[f32]| {
        let b: Vec<u8> =
            v.iter().flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes()).collect();
        Tensor::from_bytes(&ctx, &b, &[v.len()], DType::F16).unwrap()
    };
    let kc = f16s(&f32s(&mut rng, ring * kvh * d, -0.1, 0.1));
    let mut vcv = f32s(&mut rng, ring * kvh * d, -0.1, 0.1);
    for (i, v) in vcv.iter_mut().enumerate() {
        match i % d {
            0..32 => *v = 0.0,
            40 => *v = f32::NAN,
            64..96 => *v = f32::INFINITY,
            96..128 => *v = f32::NEG_INFINITY,
            _ => {}
        }
    }
    let vc = f16s(&vcv);
    let kn = tensor(&ctx, &f32s(&mut rng, d, 0.5, 1.5));
    let a = AttnShape {
        head_dim: d,
        heads,
        kv_heads: kvh,
        ring,
        window: 1024,
        k_cache: &kc,
        v_cache: &vc,
        k_rot: d / 2,
        k_norm: &kn,
    };
    let q = tensor(&ctx, &f32s(&mut rng, m * heads * d, -1.0, 1.0));
    let partial = zeros(&ctx, attn_partial_len(m, heads, d, attn_chunks(&a, pos, m)));
    let (plain, with) = (zeros(&ctx, m * heads * d), zeros(&ctx, m * heads * d));
    let ((c1, d1), (c2, d2)) = (q8(), q8());
    run(&ctx, |p| {
        attention(&ctx, p, &a, &q, &partial, &plain, pos, m, 4).unwrap();
        p.memory_barrier().unwrap();
        attention_q8x(&ctx, p, &a, &q, &partial, &with, pos, m, 4, Some((&c1, &d1)))
            .unwrap();
        p.memory_barrier().unwrap();
        quant_q8x(&ctx, p, &plain, &c2, &d2, m, heads * d).unwrap();
    });
    let k = heads * d;
    assert!(same(&plain, &with), "merge: f32 output");
    assert!(same_codes(&c1, &c2, m * k), "merge: codes");
    assert!(
        same(&d1.view(0, &[m * k / 16]).unwrap(), &d2.view(0, &[m * k / 16]).unwrap()),
        "merge: scales"
    );
    let ds = d1.to_f32().unwrap();
    let scales: Vec<f32> = ds[..m * k / 16].iter().step_by(2).copied().collect();
    assert!(scales.iter().any(|d| d.is_nan()), "merge: no nonfinite block");
    assert!(scales.iter().any(|&d| d == 0.0), "merge: no zero block");
}

/// quant_q8x on edge blocks: codes in [-127, 127] with the sign of their
/// value (a subnormal maximum would round to 128), the stored sum the
/// codes', a zero block zero, extrema at +-127, codes within half a step
/// of x / d, and a NaN or an infinity flagging its block with a NaN scale
/// and zero codes.
#[test]
fn quant_q8x_edge_blocks() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(82);
    let (m, k) = (2, 512);
    let mut x = f32s(&mut rng, m * k, -1.0, 1.0);
    let tiny = |n: u32| f32::from_bits(n); // n times the smallest subnormal
    let blk = |b: usize| b * 32..(b + 1) * 32;
    x[blk(0)].fill(0.0);
    // Subnormal maxima: 2^-142 (x / d = 128) and 190 x 2^-149.
    x[blk(1)].fill(0.0);
    x[32..36].copy_from_slice(&[tiny(1 << 7), -tiny(1 << 7), tiny(1 << 6), -tiny(3)]);
    x[blk(2)].fill(0.0);
    x[64..67].copy_from_slice(&[-tiny(190), tiny(190), tiny(64)]);
    // Signed extrema.
    x[96] = f32::MAX;
    x[97] = -f32::MAX;
    x[128] = -3.0;
    x[129] = 1.5;
    // Rounding boundaries around a scale of 1.
    let halves =
        [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 2.5, 126.5, -126.5, 63.49, 63.51];
    x[160..160 + halves.len()].copy_from_slice(&halves);
    x[192 + 7] = f32::NAN;
    x[224 + 31] = f32::INFINITY;
    x[256] = f32::NEG_INFINITY;
    x[k + 288] = f32::NAN; // the second row's block 9
    let xt = tensor(&ctx, &x);
    let codes = Tensor::zeros(&ctx, &[m * k], DType::I8).unwrap();
    let ds = zeros(&ctx, m * k / 16);
    run(&ctx, |p| quant_q8x(&ctx, p, &xt, &codes, &ds, m, k).unwrap());
    let (c, ds) = (codes.to_f32().unwrap(), ds.to_f32().unwrap());
    for b in 0..m * k / 32 {
        let (xs, cs) = (&x[b * 32..(b + 1) * 32], &c[b * 32..(b + 1) * 32]);
        let (d, sum) = (ds[2 * b], ds[2 * b + 1]);
        if xs.iter().any(|v| !v.is_finite()) {
            assert!(
                d.is_nan() && sum == 0.0 && cs.iter().all(|&q| q == 0.0),
                "block {b}"
            );
            continue;
        }
        assert!(cs.iter().all(|q| (-127.0..=127.0).contains(q)), "block {b}: range");
        assert!(
            xs.iter().zip(cs).all(|(v, q)| *q == 0.0 || q.signum() == v.signum()),
            "block {b}: sign"
        );
        assert_eq!(sum, cs.iter().sum::<f32>(), "block {b}: sum");
        let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
        if amax == 0.0 {
            assert!(d == 0.0 && cs.iter().all(|&q| q == 0.0), "block {b}: zero");
            continue;
        }
        if amax / 127.0 < f32::MIN_POSITIVE {
            // The GPU may flush subnormals: then the block is zero.
            assert!(
                d == 0.0 || cs.iter().any(|q| q.abs() == 127.0),
                "block {b}: subnormal"
            );
            continue;
        }
        assert!(
            (d - amax / 127.0).abs() <= 2.0 * f32::EPSILON * d,
            "block {b}: scale {d}"
        );
        for (v, q) in xs.iter().zip(cs) {
            let r = (v / d).clamp(-127.0, 127.0);
            assert!((q - r).abs() <= 0.5 + 1e-4, "block {b}: {v} / {d} -> {q}");
            if v.abs() == amax {
                assert_eq!(q.abs(), 127.0, "block {b}: extremum {v}");
            }
        }
    }
    // The four blocks with a NaN or an infinity, and no other.
    assert_eq!(ds.iter().step_by(2).filter(|d| d.is_nan()).count(), 4);
}

/// Q5_K weights decoded from the block layout (d, dmin, 12 bytes of 6-bit
/// scales and mins, 32 bytes of high bits, 128 of low nibbles), apart from
/// the crate's dequantizer.
fn q5_k_values(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / 176 * 256);
    for b in raw.chunks_exact(176) {
        let h =
            |i: usize| f16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32();
        let (d, dmin) = (h(0), h(2));
        let sc = &b[4..16];
        let scale_min = |j: usize| {
            if j < 4 {
                (sc[j] & 63, sc[j + 4] & 63)
            } else {
                (
                    (sc[j + 4] & 0xf) | ((sc[j - 4] >> 6) << 4),
                    (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
                )
            }
        };
        let (qh, qs) = (&b[16..48], &b[48..176]);
        for j in 0..8 {
            let (s, mn) = scale_min(j);
            let (s, mn) = (d * s as f32, dmin * mn as f32);
            for l in 0..32 {
                let lo = if j % 2 == 0 {
                    qs[j / 2 * 32 + l] & 0xf
                } else {
                    qs[j / 2 * 32 + l] >> 4
                };
                let hi = (qh[l] >> j) & 1;
                out.push(s * (lo + 16 * hi) as f32 - mn);
            }
        }
    }
    out
}

/// The int8 GEMV with two to four sets of distinct row counts (each ending
/// in a partial group of eight rows) equals one dispatch per set, bit for
/// bit, and the products of the codes with independently decoded Q5_K
/// weights (nonzero scales and mins).
#[test]
fn int8_gemv_sets_match_one_set_and_the_block_layout() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(83);
    let cols = 1024;
    let rows = [523, 77, 300, 9];
    let raws: Vec<Vec<u8>> = rows
        .iter()
        .map(|&r| random_blocks(&mut rng, GgmlType::Q5K, r * cols))
        .collect();
    let ws: Vec<QWeight> = raws
        .iter()
        .zip(rows)
        .map(|(raw, r)| QWeight::from_bytes(&ctx, raw, GgmlType::Q5K, cols, r).unwrap())
        .collect();
    for m in [1, 3] {
        let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
        let codes = Tensor::zeros(&ctx, &[m * cols], DType::I8).unwrap();
        let ds = zeros(&ctx, m * cols / 16);
        run(&ctx, |p| quant_q8x(&ctx, p, &x, &codes, &ds, m, cols).unwrap());
        let one: Vec<Tensor> = rows.iter().map(|&r| zeros(&ctx, m * r)).collect();
        run(&ctx, |p| {
            for (w, y) in ws.iter().zip(&one) {
                gemv_q5_k_q8x(&ctx, p, &[(w, y)], &codes, &ds, m).unwrap();
            }
        });
        let one: Vec<Vec<f32>> = one.iter().map(|y| y.to_f32().unwrap()).collect();
        for sets in 2..=4 {
            let ys: Vec<Tensor> =
                rows[..sets].iter().map(|&r| zeros(&ctx, m * r)).collect();
            let pairs: Vec<(&QWeight, &Tensor)> = ws.iter().zip(&ys).collect();
            run(&ctx, |p| gemv_q5_k_q8x(&ctx, p, &pairs, &codes, &ds, m).unwrap());
            for (s, y) in ys.iter().enumerate() {
                let y = y.to_f32().unwrap();
                assert!(
                    y.iter().zip(&one[s]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "m {m}, {sets} sets: set {s}"
                );
            }
        }
        let (c, ds) = (codes.to_f32().unwrap(), ds.to_f32().unwrap());
        for (s, raw) in raws.iter().enumerate() {
            let w = q5_k_values(raw);
            let r = rows[s];
            let want: Vec<f32> = (0..m * r)
                .map(|o| {
                    let (i, n) = (o / r, o % r);
                    (0..cols)
                        .map(|e| {
                            let b = (i * cols + e) / 32;
                            w[n * cols + e] as f64
                                * ds[2 * b] as f64
                                * c[i * cols + e] as f64
                        })
                        .sum::<f64>() as f32
                })
                .collect();
            assert_close(&one[s], &want, 1e-4, 1e-4);
        }
    }
}

/// The routed experts' Q5_K gate/up over int8 x, one to four rows at
/// Gemma's width: each row equals its one-row dispatch bit for bit (a
/// verify pass decodes as a plain step), and gelu(g) u follows the products
/// of the codes with independently decoded weights.
#[test]
fn moe_gate_up_q8x_matches_int8_products() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(84);
    let (hid, ff, ne, used) = (2816, 72, 6, 3);
    let raw = random_blocks(&mut rng, GgmlType::Q5K, ne * 2 * ff * hid);
    let gu = QWeight::from_bytes(&ctx, &raw, GgmlType::Q5K, hid, ne * 2 * ff).unwrap();
    let w = q5_k_values(&raw);
    for m in 1..=4 {
        let x = tensor(&ctx, &f32s(&mut rng, m * hid, -1.0, 1.0));
        let sel: Vec<u32> =
            (0..m * used).map(|_| rng.gen_range(0..ne as u32)).collect();
        let ts = u32s(&ctx, &sel);
        let codes = Tensor::zeros(&ctx, &[m * hid], DType::I8).unwrap();
        let ds = zeros(&ctx, m * hid / 16);
        let mid = zeros(&ctx, m * used * ff);
        run(&ctx, |p| {
            quant_q8x(&ctx, p, &x, &codes, &ds, m, hid).unwrap();
            p.memory_barrier().unwrap();
            moe_gate_up_q8x(
                &ctx,
                p,
                &gu,
                ff,
                &x,
                &ts,
                &mid,
                m,
                used,
                Some((&codes, &ds)),
            )
            .unwrap();
    });
    let got = mid.to_f32().unwrap();
    for i in 0..m {
        let one = zeros(&ctx, used * ff);
        let (xi, ci) =
            (x.view(i * hid, &[hid]).unwrap(), codes.view(i * hid, &[hid]).unwrap());
        let (di, si) = (
            ds.view(i * hid / 16, &[hid / 16]).unwrap(),
            u32s(&ctx, &sel[i * used..][..used]),
        );
        run(&ctx, |p| {
            moe_gate_up_q8x(&ctx, p, &gu, ff, &xi, &si, &one, 1, used, Some((&ci, &di)))
                .unwrap()
        });
        let one = one.to_f32().unwrap();
        let row = &got[i * used * ff..][..used * ff];
        assert!(
            row.iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()),
            "row {i}"
        );
    }
    let (c, d) = (codes.to_f32().unwrap(), ds.to_f32().unwrap());
    let prod = |i: usize, wr: usize| -> f32 {
        (0..hid)
            .map(|k| {
                let b = (i * hid + k) / 32;
                w[wr * hid + k] as f64 * d[2 * b] as f64 * c[i * hid + k] as f64
            })
            .sum::<f64>() as f32
    };
    let mut want = vec![0f32; m * used * ff];
    for i in 0..m {
        for s in 0..used {
            let e = sel[i * used + s] as usize;
            for r in 0..ff {
                let (g, u) = (prod(i, e * 2 * ff + r), prod(i, e * 2 * ff + ff + r));
                want[(i * used + s) * ff + r] = gelu(g) * u;
            }
        }
    }
    // Q5_K's sums of scale and min terms cancel to ~1e-4 of their size with
    // these random blocks (as in moe_matches_cpu).
    assert_close(&got, &want, 1e-3, 1e-3);
    }
}

/// post_attn with every combination of its extra outputs (n1 int8, n2
/// int8, n2 f16): the residual and all three f32 outputs equal the plain
/// kernel's bit for bit; each int8 pair equals quant_q8x of its own f32
/// output, and the f16 output is n2 clamped to +-65504 in the IQ2_XXS
/// pairing. n1 and n2 take different weights with edge blocks (zero,
/// subnormal, NaN, overflow, infinite).
#[test]
fn post_attn_writes_its_int8_and_f16_outputs() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(85);
    let (m, n, eps) = (3, 2816, 1e-6);
    let a = tensor(&ctx, &f32s(&mut rng, m * n, -2.0, 2.0));
    let x = tensor(&ctx, &f32s(&mut rng, m * n, -2.0, 2.0));
    let w_post = tensor(&ctx, &f32s(&mut rng, n, 0.1, 3.0));
    let edges = |rng: &mut StdRng, at: usize| {
        let mut w = f32s(rng, n, 0.1, 3.0);
        for e in 0..32 {
            w[at + e] = 0.0;
            w[at + 32 + e] = 1e-40;
            w[at + 96 + e] = f32::MAX;
            w[at + 128 + e] = f32::INFINITY;
        }
        w[at + 64 + 7] = f32::NAN;
        tensor(&ctx, &w)
    };
    let (w1, w2) = (edges(&mut rng, 0), edges(&mut rng, 640));
    let w3 = tensor(&ctx, &f32s(&mut rng, n, 0.1, 3.0));
    let bits = |t: &Tensor| {
        t.to_f32().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    };
    let q8 =
        || (Tensor::zeros(&ctx, &[m * n], DType::I8).unwrap(), zeros(&ctx, m * n / 16));
    let (rp, op) = (zeros(&ctx, m * n), [0; 3].map(|_| zeros(&ctx, m * n)));
    run(&ctx, |p| {
        post_attn(
            &ctx,
            p,
            &a,
            &w_post,
            &x,
            &rp,
            [&w1, &w2, &w3],
            [&op[0], &op[1], &op[2]],
            m,
            n,
            eps,
        )
        .unwrap()
    });
    // quant_q8x of the plain n1 and n2.
    let want: Vec<(Tensor, Tensor)> = (0..2).map(|_| q8()).collect();
    run(&ctx, |p| {
        for j in 0..2 {
            quant_q8x(&ctx, p, &op[j], &want[j].0, &want[j].1, m, n).unwrap();
        }
    });
    // The f16 pairing of the plain n2, clamped.
    let n2 = op[1].to_f32().unwrap();
    let mut want_h = vec![0u16; m * n];
    for g in 0..m * n / 8 {
        for p in 0..4 {
            for (half, v) in [n2[8 * g + p], n2[8 * g + p + 4]].into_iter().enumerate()
            {
                want_h[8 * g + 2 * p + half] =
                    f16::from_f32(v.clamp(-65504.0, 65504.0)).to_bits();
            }
        }
    }
    for mask in 1..8u32 {
        let pairs = [q8(), q8()];
        let xh = Tensor::zeros(&ctx, &[m * n], DType::F16).unwrap();
        let (rq, oq) = (zeros(&ctx, m * n), [0; 3].map(|_| zeros(&ctx, m * n)));
        let q8s = [0, 1]
            .map(|j| (mask & (1 << j) != 0).then_some((&pairs[j].0, &pairs[j].1)));
        run(&ctx, |p| {
            post_attn_q8x(
                &ctx,
                p,
                &a,
                &w_post,
                &x,
                &rq,
                [&w1, &w2, &w3],
                [&oq[0], &oq[1], &oq[2]],
                m,
                n,
                eps,
                q8s,
                (mask & 4 != 0).then_some(&xh),
            )
            .unwrap()
        });
        assert_eq!(bits(&rq), bits(&rp), "mask {mask}: residual");
        for j in 0..3 {
            assert_eq!(bits(&oq[j]), bits(&op[j]), "mask {mask}: output {j}");
        }
        for j in 0..2 {
            if mask & (1 << j) != 0 {
                assert_eq!(
                    pairs[j].0.to_f32().unwrap(),
                    want[j].0.to_f32().unwrap(),
                    "mask {mask}: codes {j}"
                );
                assert_eq!(
                    bits(&pairs[j].1),
                    bits(&want[j].1),
                    "mask {mask}: scales {j}"
                );
            }
        }
        if mask & 4 != 0 {
            let got: Vec<u16> = bytemuck::cast_slice::<u8, u16>(xh.contents()).to_vec();
            let bad = got.iter().zip(&want_h).position(|(g, w)| {
                g != w && !(f16::from_bits(*g).is_nan() && f16::from_bits(*w).is_nan())
            });
            if let Some(i) = bad {
                panic!("mask {mask}: f16 n2 [{i}] {:#06x}, want {:#06x}", got[i], want_h[i]);
            }
        }
    }
}

/// The f16-activation experts: the IQ2_XXS gate/up over f16 x equals the
/// f32 kernel over x rounded to f16, bit for bit, and its f16 mid is its f32
/// mid clamped to +-65504 in the IQ4_NL pairing; the IQ4_NL down over that
/// equals the f32 down over the f16 mid widened, bit for bit. At Gemma's
/// width, one to four rows.
#[test]
fn f16_experts_match_the_f32_kernels_on_f16_values() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(86);
    let (hid, ff, ne, used) = (2816, 704, 10, 8);
    let gu_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, ne * 2 * ff * hid);
    let dn_raw = random_blocks(&mut rng, GgmlType::Iq4Nl, ne * hid * ff);
    let gu =
        QWeight::from_bytes(&ctx, &gu_raw, GgmlType::Iq2Xxs, hid, ne * 2 * ff).unwrap();
    let dn = QWeight::from_bytes(&ctx, &dn_raw, GgmlType::Iq4Nl, ff, ne * hid).unwrap();
    let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for m in 1..=4 {
        let xv: Vec<f32> = f32s(&mut rng, m * hid, -1.0, 1.0)
            .iter()
            .map(|v| f16::from_f32(*v).to_f32())
            .collect();
        let x = tensor(&ctx, &xv);
        let xh = Tensor::zeros(&ctx, &[m * hid], DType::F16).unwrap();
        let sel: Vec<u32> =
            (0..m * used).map(|_| rng.gen_range(0..ne as u32)).collect();
        let (ts, tw) =
            (u32s(&ctx, &sel), tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.5)));
        let (mid, mid32) = (zeros(&ctx, m * used * ff), zeros(&ctx, m * used * ff));
        let mid_h = Tensor::zeros(&ctx, &[m * used * ff], DType::F16).unwrap();
        run(&ctx, |p| {
            crate::kernels::deepseek4::x_f16_iq2(&ctx, p, &x, &xh, m, hid).unwrap();
            moe_gate_up(&ctx, p, &gu, ff, &x, &ts, &mid32, m, used).unwrap();
            p.memory_barrier().unwrap();
            moe_gate_up_xh(&ctx, p, &gu, ff, &xh, &ts, &mid, &mid_h, m, used).unwrap();
        });
        let mv = mid.to_f32().unwrap();
        assert_eq!(bits(&mv), bits(&mid32.to_f32().unwrap()), "m {m}: gate/up");
        // mid_h: of each 32 values, word j holds values j and j + 16.
        let got: Vec<u16> = bytemuck::cast_slice::<u8, u16>(mid_h.contents()).to_vec();
        let mut wide = vec![0f32; m * used * ff];
        for (b, blk) in mv.chunks(32).enumerate() {
            for j in 0..16 {
                for (half, v) in [blk[j], blk[j + 16]].into_iter().enumerate() {
                    let h = got[32 * b + 2 * j + half];
                    assert_eq!(
                        h,
                        f16::from_f32(v.clamp(-65504.0, 65504.0)).to_bits(),
                        "m {m}: mid_h"
                    );
                    wide[32 * b + j + 16 * half] = f16::from_bits(h).to_f32();
                }
            }
        }
        let (out, out32) = (zeros(&ctx, m * hid), zeros(&ctx, m * hid));
        let midw = tensor(&ctx, &wide);
        run(&ctx, |p| {
            moe_down(&ctx, p, &dn, hid, &midw, &ts, &tw, &out32, m, used).unwrap();
            moe_down_xh(&ctx, p, &dn, hid, &mid_h, &ts, &tw, &out, m, used).unwrap();
        });
        assert_eq!(
            bits(&out.to_f32().unwrap()),
            bits(&out32.to_f32().unwrap()),
            "m {m}: down"
        );
    }
}
