use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::gguf::dequantize;

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

/// Random blocks with small scales (Q8_0, IQ4_NL, Q5_K, IQ2_XXS).
fn random_blocks(rng: &mut StdRng, ty: GgmlType, values: usize) -> Vec<u8> {
    let nb = values / ty.block_values();
    let bb = ty.block_bytes();
    let mut b: Vec<u8> = (0..nb * bb).map(|_| rng.r#gen()).collect();
    let put = |blk: &mut [u8], at: usize, v: f32| {
        blk[at..at + 2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes())
    };
    for blk in b.chunks_exact_mut(bb) {
        match ty {
            GgmlType::Q8_0 | GgmlType::Iq4Nl | GgmlType::Iq2Xxs => {
                put(blk, 0, rng.gen_range(0.001..0.01))
            }
            GgmlType::Q5K => {
                put(blk, 0, rng.gen_range(0.001..0.01));
                put(blk, 2, rng.gen_range(0.001..0.01));
            }
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn rms(x: &[f32], eps: f32) -> f32 {
    1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
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

fn h16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
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
    let v = f32s(&mut rng, total * kw, -1.0, 1.0);
    let kc = Tensor::zeros(&ctx, &[ring * kw], DType::F16).unwrap();
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
    let norm = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
        let s = rms(x, eps);
        x.iter().enumerate().map(|(i, v)| v * s * w.map_or(1.0, |w| w[i])).collect()
    };
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
    let nrm = |x: &[f32], w: &[f32]| -> Vec<f32> {
        let s = rms(x, eps);
        x.iter().zip(w).map(|(v, w)| v * s * w).collect()
    };
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
/// dispatches of a serial pass over four layers' weights in turn.
#[test]
#[ignore = "timing only"]
fn moe_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(70);
    let repeat = 32;
    let (experts, hidden, ff, used) = (24, 2816, 704, 8);
    for (gt, dt) in
        [(GgmlType::Iq2Xxs, GgmlType::Iq4Nl), (GgmlType::Q5K, GgmlType::Q8_0)]
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
            for _ in 0..5 {
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
                best = best.min(t.elapsed().as_secs_f64());
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

/// The grouped experts of a prompt run against the per-row kernels: the
/// same sums with both operands rounded to bfloat.
#[test]
fn moe_grouped_matches_the_row_kernels() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(21);
    let (hid, ff, ne, used, m) = (512, 128, 12, 8, 37);
    for (gt, dt) in
        [(GgmlType::Iq2Xxs, GgmlType::Iq4Nl), (GgmlType::Q5K, GgmlType::Q8_0)]
    {
        let gu_raw = random_blocks(&mut rng, gt, ne * 2 * ff * hid);
        let dn_raw = random_blocks(&mut rng, dt, ne * hid * ff);
        let gu = QWeight::from_bytes(&ctx, &gu_raw, gt, hid, ne * 2 * ff).unwrap();
        let dn = QWeight::from_bytes(&ctx, &dn_raw, dt, ff, ne * hid).unwrap();
        // Small activations keep the bfloat rounding's share small.
        let x = f32s(&mut rng, m * hid, -0.1, 0.1);
        let sel: Vec<u32> =
            (0..m * used).map(|i| ((i * 7 + i / used) % ne) as u32).collect();
        let wts = f32s(&mut rng, m * used, 0.05, 0.2);
        let (tx, ts, tw) = (tensor(&ctx, &x), u32s(&ctx, &sel), tensor(&ctx, &wts));
        let mid = zeros(&ctx, m * used * ff);
        let rows = zeros(&ctx, m * hid);
        let grouped = zeros(&ctx, m * hid);
        let sc = MoeGroupScratch::new(&ctx, m, used, ne, ff, hid).unwrap();
        run(&ctx, |p| {
            moe_gate_up(&ctx, p, &gu, ff, &tx, &ts, &mid, m, used).unwrap();
            p.memory_barrier().unwrap();
            moe_down(&ctx, p, &dn, hid, &mid, &ts, &tw, &rows, m, used).unwrap();
            p.memory_barrier().unwrap();
            moe_grouped(
                &ctx, p, &gu, &dn, ff, &tx, &ts, &tw, &grouped, &sc, m, used, ne,
            )
            .unwrap();
        });
        let (a, b) = (grouped.to_f32().unwrap(), rows.to_f32().unwrap());
        let scale = b.iter().fold(0f32, |s, v| s.max(v.abs()));
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst <= 0.02 * scale, "{gt:?}/{dt:?}: worst {worst} of {scale}");
    }
}
