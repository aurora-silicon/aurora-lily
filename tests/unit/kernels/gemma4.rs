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
        return (0..values).flat_map(|_| rng.gen_range(-0.1f32..0.1).to_le_bytes()).collect();
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
        let mut line = format!("moe {gt:?}/{dt:?}:");
        for part in 0..2 {
            let mut best = f64::MAX;
            for pass_no in 0..9 {
                let pass = ctx.begin().unwrap();
                for r in 0..repeat {
                    let (gu, dn) = &layers[r % 4];
                    if part == 0 {
                        moe_gate_up(&ctx, &pass, gu, ff, &x, &ts, &mid, 1, used)
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
                if part == 0 { "gate/up" } else { "down" },
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
    let m: usize = std::env::var("LILY_GEMV_M").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
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
                    Grid::Threadgroups { groups: (groups, 1, 1), threadgroup: (256, 1, 1) },
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
