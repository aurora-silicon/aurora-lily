use super::*;
use crate::deepseek4::config::Yarn;

/// Records `u32 n, f32[n]` of `tests/goldens/deepseek4_ops.bin`, written by
/// `tools/reference/deepseek4/ops_golden.c` from ds4's own functions.
fn golden() -> Vec<Vec<f32>> {
    let b = include_bytes!("../../goldens/deepseek4_ops.bin");
    let mut v = Vec::new();
    let mut at = 0;
    while at < b.len() {
        let n = u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        v.push(
            b[at..at + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        );
        at += 4 * n;
    }
    v
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.to_bits() == w.to_bits(), "{what}[{i}]: {g:e} != ds4's {w:e}");
    }
}

/// Within `ulps` units in the last place (libm's cos/sin/exp may differ by
/// one between platforms).
fn assert_ulps(got: &[f32], want: &[f32], ulps: i64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (i64::from(g.to_bits() as i32) - i64::from(w.to_bits() as i32)).abs();
        let tiny = (g - w).abs() <= 1e-30;
        assert!(d <= ulps || tiny, "{what}[{i}]: {g:e} vs ds4's {w:e} ({d} ulps)");
    }
}

fn rope_of(yarn: bool) -> Rope {
    if yarn {
        Rope {
            n_rot: 64,
            base: 160000.0,
            freq_scale: 1.0 / 16.0,
            yarn: Some(Yarn { orig_ctx: 65536, beta_fast: 32.0, beta_slow: 1.0 }),
        }
    } else {
        Rope { n_rot: 64, base: 10000.0, freq_scale: 1.0, yarn: None }
    }
}

#[test]
fn ops_match_ds4_goldens() {
    let g = golden();
    let mut it = g.iter();
    let mut next = || it.next().expect("golden record").clone();
    for r in 0..6 {
        let mut x = next();
        fp8_kv_qat(&mut x, 64);
        assert_bits(&x, &next(), &format!("fp8 row {r}"));
    }
    for r in 0..6 {
        let mut x = next();
        indexer_qat(&mut x);
        assert_bits(&x, &next(), &format!("indexer qat row {r}"));
    }
    let x = next();
    let e4: Vec<f32> = x.iter().map(|&v| e4m3fn_round(v)).collect();
    assert_bits(&e4, &next(), "e4m3");
    let x = next();
    let e2: Vec<f32> = x.iter().map(|&v| e2m1fn_round(v)).collect();
    assert_bits(&e2, &next(), "e2m1");
    for yarn in [false, true] {
        let rope = rope_of(yarn);
        for pos in [0usize, 1, 7, 127, 128, 1000, 65535, 300001] {
            for inverse in [false, true] {
                for head_dim in [512usize, 128] {
                    let mut x = next();
                    rope_tail(&mut x, head_dim, &rope, pos, inverse);
                    assert_ulps(
                        &x,
                        &next(),
                        2,
                        &format!(
                            "rope yarn {yarn} pos {pos} inv {inverse} dim {head_dim}"
                        ),
                    );
                }
            }
        }
    }
    for ratio in [4usize, 128] {
        for case in 0..2 {
            let mut c = Compressor::new(16, ratio);
            c.kv = next();
            c.score = next();
            assert_ulps(
                &c.pool(),
                &next(),
                2,
                &format!("pool ratio {ratio} case {case}"),
            );
        }
    }
    assert!(it.next().is_none(), "golden has unread records");
}

/// The layer RoPE parameters follow ds4: plain layers base 1e4, compressed
/// layers the long base with YaRN; the inverse undoes the rotation.
#[test]
fn rope_params_and_inverse() {
    let c = Config::deepseek4_flash();
    assert_eq!(c.rope(0), rope_of(false));
    assert_eq!(c.rope(1), rope_of(false));
    assert_eq!(c.rope(2), rope_of(true));
    assert_eq!(c.rope(3), rope_of(true));
    // YaRN on the 64 rotated dims: the ramp runs over dim pairs 15..25
    // (64 ln(65536 / (2 pi beta)) / (2 ln 160000) for beta 32 and 1, floor
    // and ceil).
    let corr = yarn_corr_dims(64, 65536, 160000.0, 32.0, 1.0);
    assert_eq!(corr, [15.0, 25.0], "{corr:?}");
    let x0: Vec<f32> =
        (0..1024).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
    for yarn in [false, true] {
        let mut x = x0.clone();
        rope_tail(&mut x, 512, &rope_of(yarn), 4242, false);
        assert_eq!(x[..448], x0[..448], "the non-RoPE dims are untouched");
        assert_ne!(x[448..512], x0[448..512]);
        rope_tail(&mut x, 512, &rope_of(yarn), 4242, true);
        for (a, b) in x.iter().zip(&x0) {
            assert!((a - b).abs() < 1e-5, "inverse: {a} vs {b}");
        }
    }
}

/// A ratio-4 compressor emits a row every fourth position, pooling the
/// previous window (lane 0) with the current one (lane 1); a ratio-128 one
/// only its own window.
#[test]
fn compressor_windows() {
    let dim = 4;
    let mut c = Compressor::new(dim, 4);
    let mut emitted = Vec::new();
    for pos in 0..12 {
        // kv = pos in every entry, score 0: the pool is the mean of the
        // window's kv over the rows that hold something.
        let kv = vec![pos as f32; 2 * dim];
        let sc = vec![0f32; 2 * dim];
        if let Some(r) = c.push(&kv, &sc, pos) {
            emitted.push((pos, r));
        }
    }
    assert_eq!(emitted.iter().map(|e| e.0).collect::<Vec<_>>(), [3, 7, 11]);
    // First window: only lane 1 of rows 4..8 is set: mean of 0..4.
    assert_eq!(emitted[0].1, vec![1.5; dim]);
    // Later windows: previous window (lane 0) and current (lane 1).
    assert_eq!(emitted[1].1, vec![3.5; dim]);
    assert_eq!(emitted[2].1, vec![7.5; dim]);

    let mut c = Compressor::new(dim, 128);
    let mut n = 0;
    for pos in 0..256 {
        let kv = vec![(pos % 128) as f32; dim];
        if let Some(r) = c.push(&kv, &vec![0f32; dim], pos) {
            n += 1;
            assert_eq!(pos % 128, 127);
            assert_eq!(r, vec![63.5; dim]);
        }
    }
    assert_eq!(n, 2);
}

#[test]
fn router_scores_and_selection() {
    for l in [-30f32, -5.0, 0.0, 0.7, 19.0, 25.0] {
        let sp = (1.0 + f64::from(l).exp()).ln();
        assert!((f64::from(router_score(l)) - sp.sqrt()).abs() < 1e-6, "{l}");
    }
    let logits = [0.5f32, 2.0, 2.0, -1.0, 3.0, 0.0];
    let bias = [0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
    let (sel, w) = route(&logits, Some(&bias), None, 3, 1.5);
    // Expert 3 by its bias, then 4, then the tie 1/2 goes to 1.
    assert_eq!(sel, [3, 4, 1]);
    let p: Vec<f32> = [3, 4, 1].iter().map(|&e| router_score(logits[e])).collect();
    let s: f32 = p.iter().sum();
    for (wi, pi) in w.iter().zip(&p) {
        assert!((wi - pi / s * 1.5).abs() < 1e-6);
    }
    // Hash layers take the table's experts, weighted the same way.
    let (sel, w2) = route(&logits, None, Some(&[5, 0]), 2, 1.5);
    assert_eq!(sel, [5, 0]);
    let p5 = router_score(0.0);
    let p0 = router_score(0.5);
    assert!((w2[0] - p5 / (p5 + p0) * 1.5).abs() < 1e-6);
}

#[test]
fn indexer_selects_the_best_rows() {
    let dim = 4;
    // Two heads; head 1's weight is negative so its matches count against.
    let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let w = [1.0f32, -0.5];
    let keys = [
        2.0f32, 0.0, 0.0, 0.0, // 2
        2.0, 2.0, 0.0, 0.0, // 2 - 1 = 1
        -3.0, 0.0, 0.0, 0.0, // relu: 0
        3.0, 0.0, 0.0, 0.0, // 3
        2.0, 0.0, 9.0, 0.0, // 2 (tie with row 0)
    ];
    assert!(indexer_select(&q, &w, &keys, dim, 5).is_none());
    let a = indexer_select(&q, &w, &keys, dim, 2).unwrap();
    // Scores 2, 1, 0, 3, 2: row 3, then the tie of rows 0 and 4 goes to 0.
    assert_eq!(a, vec![true, false, false, true, false]);
    let a = indexer_select(&q, &w, &keys, dim, 3).unwrap();
    assert_eq!(a, vec![true, false, false, true, true]);
}

/// Sink attention against the softmax written out for two heads and three
/// rows.
#[test]
fn sink_attention_by_hand() {
    let dim = 2;
    let q = [1.0f32, 0.0, 0.0, 2.0];
    let rows: Vec<Vec<f32>> = vec![vec![1.0, 1.0], vec![2.0, -1.0], vec![0.0, 0.5]];
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let sinks = [0.3f32, -1.0];
    let out = sink_attention(&q, &refs, &sinks, dim);
    let s = 1.0 / 2f64.sqrt();
    for h in 0..2 {
        let qh = &q[h * 2..h * 2 + 2];
        let sc: Vec<f64> = rows
            .iter()
            .map(|r| {
                (f64::from(qh[0]) * f64::from(r[0])
                    + f64::from(qh[1]) * f64::from(r[1]))
                    * s
            })
            .collect();
        let den: f64 =
            f64::from(sinks[h]).exp() + sc.iter().map(|v| v.exp()).sum::<f64>();
        for d in 0..2 {
            let num: f64 =
                rows.iter().zip(&sc).map(|(r, v)| v.exp() * f64::from(r[d])).sum();
            assert!((f64::from(out[h * 2 + d]) - num / den).abs() < 1e-6);
        }
    }
}

/// The output collapse weights each stream by sigmoid(pre * scale + base)
/// + eps.
#[test]
fn output_collapse_by_hand() {
    let hidden = 3;
    let h: Vec<f32> = (0..4 * hidden).map(|i| i as f32 * 0.1 - 0.4).collect();
    let fn_w = Mat {
        rows: 4,
        cols: 4 * hidden,
        w: (0..48).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect(),
    };
    let base = [0.1f32, -0.2, 0.0, 0.3];
    let out = output_collapse(&h, &fn_w, 0.5, &base, 1e-6, 1e-6);
    let pre = fn_w.mv(&rmsnorm(&h, None, 1e-6));
    for d in 0..hidden {
        let want: f32 = (0..4)
            .map(|s| h[s * hidden + d] * (sigmoid(pre[s] * 0.5 + base[s]) + 1e-6))
            .sum();
        assert!((out[d] - want).abs() < 1e-6);
    }
}
