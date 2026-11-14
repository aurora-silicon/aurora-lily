use super::*;
use crate::deepseek4::model::Logits;
use crate::kernels::deepseek4 as d;

/// Loads the published file (without DSpark) for the real-weight tests.
fn real_model(ctx: &MetalContext) -> Ds4Model {
    let path = std::env::var("LILY_DS4_GGUF").expect("LILY_DS4_GGUF");
    <Ds4Model as LanguageModel>::load(ctx, Path::new(&path), &LoadOptions::default())
        .expect("load")
}

/// A prompt of `n` ordinary text tokens.
fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7919) % 50_000).collect()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b }) as u32
}

/// Relative L2 distance of `a` from `b` (0 when both are 0).
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (f64::from(x) - f64::from(y)).powi(2);
        n += f64::from(y).powi(2);
    }
    if n == 0.0 { d.sqrt() } else { (d / n).sqrt() }
}

/// A state forked from `base` at its snapshot `snap`, as the session cache
/// forks a prefix.
fn fork(
    m: &Ds4Model,
    ctx: &MetalContext,
    base: &Ds4State,
    snap: &Ds4Snapshot,
) -> Ds4State {
    let mut st = LanguageModel::new_state(m, ctx, base.capacity()).unwrap();
    st.copy_prefix_from(ctx, base, snap.pos()).unwrap();
    st.restore(ctx, snap).unwrap();
    st
}

fn last_logits(sc: &Scratch) -> Vec<f32> {
    sc.logits_last.to_f32().unwrap()
}

/// Compares two states at the same position: every layer's raw window
/// (the slots of `untouched` positions bit for bit, as neither state may
/// have written them since the fork), its compressor rings, and the valid
/// rows of its compressed caches, within `tol` relative L2 (one-row and
/// multi-row passes round differently in the last bits, and an FP8 code
/// can flip with them).
fn compare_states(
    cfg: &super::super::Config,
    a: &mut Ds4State,
    b: &mut Ds4State,
    untouched: std::ops::Range<usize>,
    tol: f64,
    case: &str,
) {
    assert_eq!(a.pos(), b.pos(), "{case}: position");
    let pos = a.pos();
    let row = cfg.head_dim;
    for (il, (la, lb)) in a.layers.iter().zip(&b.layers).enumerate() {
        let (ra, rb) = (la.ring.to_f32().unwrap(), lb.ring.to_f32().unwrap());
        for p in untouched.clone() {
            let s = (p % d::WINDOW) * row;
            let same = ra[s..s + row]
                .iter()
                .zip(&rb[s..s + row])
                .all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(
                same,
                "{case}: layer {il} raw slot of rejected position {p} written"
            );
        }
        let r = rel(&rb, &ra);
        assert!(r < tol, "{case}: layer {il} raw window differs ({r:.2e})");
        for (ca, cb) in la
            .comp
            .iter()
            .chain(la.index.iter())
            .zip(lb.comp.iter().chain(lb.index.iter()))
        {
            for (what, x, y) in [
                ("kv ring", &ca.ring_kv, &cb.ring_kv),
                ("score ring", &ca.ring_sc, &cb.ring_sc),
            ] {
                let r = rel(&y.to_f32().unwrap(), &x.to_f32().unwrap());
                assert!(
                    r < tol,
                    "{case}: layer {il} compressor {what} differs ({r:.2e})"
                );
            }
        }
    }
    let cache_rows = |st: &mut Ds4State| -> Vec<Vec<f32>> {
        caches(st, cfg)
            .into_iter()
            .map(|(c, r, dim)| c.cache.to_f32().unwrap()[..(pos / r) * dim].to_vec())
            .collect()
    };
    for (i, (x, y)) in cache_rows(a).iter().zip(&cache_rows(b)).enumerate() {
        let r = rel(y, x);
        assert!(r < tol, "{case}: compressed cache {i} differs ({r:.2e})");
    }
}

/// A verify pass of six rows whose first `kept + 1` are committed leaves the
/// state plain committed steps of the same tokens leave, for every kept
/// count: the rejected rows' raw window slots untouched, the compressor
/// rings and compressed caches as plain decoding has them, and the next
/// step's logits the same. The passes start before a ratio-128 boundary
/// (row 1 completes a 128-row pool and a 4-row pool) and before the
/// indexer goes sparse (rows cross position 2048). The states are forks
/// of one prefilled base (snapshot restore and prefix copy).
#[test]
#[ignore = "requires LILY_DS4_GGUF"]
fn partial_commits_match_plain_steps() {
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    let cfg = m.cfg.clone();
    let greedy = SamplingParams::greedy();
    for n in [126usize, 2046] {
        let p = prompt(n);
        let mut base = LanguageModel::new_state(&m, &ctx, n + 16).unwrap();
        let mut sc = m.new_scratch(&ctx).unwrap();
        m.forward(&ctx, &mut base, &sc, &p, Logits::Last).unwrap();
        let snap = base.snapshot(&ctx).unwrap();
        // The greedy continuation, from a fork; the base's own first step
        // checks the fork.
        let mut toks = vec![argmax(&last_logits(&sc))];
        let mut st = fork(&m, &ctx, &base, &snap);
        let mut first = None;
        while toks.len() < 7 {
            let t = *toks.last().unwrap();
            m.forward(&ctx, &mut st, &sc, &[t], Logits::Last).unwrap();
            let l = last_logits(&sc);
            first.get_or_insert_with(|| l.clone());
            toks.push(argmax(&l));
        }
        m.forward(&ctx, &mut base, &sc, &[toks[0]], Logits::Last).unwrap();
        let r = rel(&last_logits(&sc), first.as_ref().unwrap());
        assert_eq!(r, 0.0, "n={n}: a fork's step differs from the base's");
        for kept in 0..=5usize {
            let case = format!("n={n} kept={kept}");
            let mut plain = fork(&m, &ctx, &base, &snap);
            for &t in &toks[..=kept] {
                m.forward(&ctx, &mut plain, &sc, &[t], Logits::Last).unwrap();
            }
            let mut spec = fork(&m, &ctx, &base, &snap);
            // The true continuation up to the kept rows, wrong tokens after.
            let drafts: Vec<u32> = (1..=5)
                .map(|i| if i <= kept { toks[i] } else { (toks[i] + 1) % 1000 + 5 })
                .collect();
            let (_, pending) = m
                .verify(&ctx, &mut spec, &mut sc, toks[0], &drafts, &greedy, 1, 0)
                .unwrap();
            m.finish_speculation(&ctx, &mut spec, &mut sc, kept, false, pending)
                .unwrap();
            assert_eq!(spec.pos(), n + kept + 1, "{case}");
            compare_states(
                &cfg,
                &mut plain,
                &mut spec,
                n + kept + 1..n + 6,
                2e-2,
                &case,
            );
            let next = toks[kept + 1];
            m.forward(&ctx, &mut plain, &sc, &[next], Logits::Last).unwrap();
            let want = last_logits(&sc);
            m.forward(&ctx, &mut spec, &sc, &[next], Logits::Last).unwrap();
            let got = last_logits(&sc);
            let r = rel(&got, &want);
            eprintln!("{case}: next step relative L2 {r:.2e}");
            assert!(r < 5e-3, "{case}: next step logits differ ({r:.2e})");
            assert_eq!(argmax(&got), argmax(&want), "{case}: next top token");
        }
    }
}
