use super::*;

/// A held instance lock refuses the GLM load before the file is opened or
/// anything mapped: the refusal, not "no such file", for a path that does
/// not exist. (`Glm53Model::load` and the engine's `LanguageModel::load`,
/// which calls it, take the lock through `load_locked`.)
#[test]
fn a_held_instance_lock_refuses_the_load_before_mapping() {
    use crate::instance::{AlreadyRunning, InstanceLock};
    let dir =
        std::env::temp_dir().join(format!("lily-glm53-lock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lock = dir.join("instance.lock");
    let ctx = MetalContext::new().expect("context");
    let missing = dir.join("no-such-model.gguf");
    let held = InstanceLock::acquire_at(&lock).unwrap();
    let e = Glm53Model::load_locked(&ctx, &missing, || {
        InstanceLock::acquire_at(&lock).map(drop)
    })
    .err()
    .expect("refused");
    assert!(e.chain().any(|c| c.is::<AlreadyRunning>()), "{e:#}");
    assert_eq!(crate::instance::exit_status(&e), crate::instance::EXIT_ALREADY_RUNNING);
    drop(held);
    // With the lock free, the same load gets as far as opening the file.
    let e = Glm53Model::load_locked(&ctx, &missing, || {
        InstanceLock::acquire_at(&lock).map(drop)
    })
    .err()
    .expect("no file");
    assert!(!e.chain().any(|c| c.is::<AlreadyRunning>()), "{e:#}");
    assert!(format!("{e:#}").contains("no-such-model"), "{e:#}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A session's recurrent state without the model: a fresh state's KDA
/// tensors read zero although they come from memory a dropped state filled;
/// a snapshot of the recurrent tensors (packed into one allocation) restores
/// every byte of each to its own tensor, stays independent of the live state,
/// forks into another state, and follows a swapped KDA pair (the scratch's
/// alternates current after a rejected draft, the former current ones the
/// scratch's).
#[test]
fn recurrent_state_zeroes_and_round_trips_without_the_model() {
    use crate::engine::{DecodeStateApi, SnapshotApi};
    let ctx = MetalContext::new().expect("context");
    let cfg = Config::glm53_flash();
    let mut sc = Scratch::new(&ctx, &cfg).expect("scratch");
    // Every recurrent tensor (KDA conv and state, DSA incomplete-pool
    // tails), in the snapshot's order, and the scratch's KDA alternates.
    fn current(st: &Glm53State) -> Vec<&Tensor> {
        st.layers
            .iter()
            .flat_map(|l| match l {
                LayerState::Kda { conv, s, .. } => vec![conv, s],
                LayerState::Dsa { tail_k, tail_g, .. } => vec![tail_k, tail_g],
            })
            .collect()
    }
    fn alternates(sc: &Scratch) -> Vec<&Tensor> {
        sc.kda_alt.iter().flatten().flat_map(|(c, s)| [c, s]).collect()
    }
    // Tensor t's pattern under seed k: distinct for every tensor and value.
    let fill = |ts: &[&Tensor], k: u32| {
        for (t, x) in ts.iter().enumerate() {
            let v: Vec<f32> = (0..x.numel())
                .map(|i| {
                    f32::from_bits(
                        0x3f80_0000
                            ^ ((k << 26) | ((t as u32) << 18) | (i as u32 & 0x3ffff)),
                    )
                })
                .collect();
            x.write_bytes(bytemuck::cast_slice(&v)).unwrap();
        }
    };
    let check = |ts: &[&Tensor], k: u32, what: &str| {
        for (t, x) in ts.iter().enumerate() {
            let got = x.to_f32().unwrap();
            for (i, g) in got.iter().enumerate() {
                let want = 0x3f80_0000
                    ^ ((k << 26) | ((t as u32) << 18) | (i as u32 & 0x3ffff));
                assert_eq!(g.to_bits(), want, "{what}: tensor {t} value {i}");
            }
        }
    };
    {
        let st = Glm53State::new(&ctx, &cfg, 64).unwrap();
        fill(&current(&st), 1);
    }
    let mut st = Glm53State::new(&ctx, &cfg, 64).unwrap();
    for (il, l) in st.layers.iter().enumerate() {
        if let LayerState::Kda { conv, s, .. } = l {
            assert!(
                conv.to_f32().unwrap().iter().all(|&v| v.to_bits() == 0),
                "layer {il} conv"
            );
            assert!(
                s.to_f32().unwrap().iter().all(|&v| v.to_bits() == 0),
                "layer {il} state"
            );
        }
    }
    fill(&current(&st), 2);
    fill(&alternates(&sc), 3);
    st.pos = 17;
    let snap = st.snapshot(&ctx).unwrap();
    assert_eq!(snap.pos(), 17);
    assert_eq!(snap.bytes(), current(&st).iter().map(|t| t.byte_len()).sum::<usize>());
    // Mutated live state: the restore brings back every byte; twice, so
    // the snapshot does not share the live tensors.
    for _ in 0..2 {
        fill(&current(&st), 4);
        st.pos = 30;
        st.restore(&ctx, &snap).unwrap();
        assert_eq!(st.pos, 17);
        check(&current(&st), 2, "restored");
        check(&alternates(&sc), 3, "alternates untouched");
    }
    // A fork: another state restored from it.
    let mut other = Glm53State::new(&ctx, &cfg, 64).unwrap();
    other.restore(&ctx, &snap).unwrap();
    check(&current(&other), 2, "forked");
    // A swapped pair: the alternates become current, and a snapshot then
    // holds (and restores into) them.
    st.take_kda_snapshots(&mut sc);
    let swapped = st.snapshot(&ctx).unwrap();
    let before: Vec<Vec<f32>> =
        alternates(&sc).iter().map(|t| t.to_f32().unwrap()).collect();
    fill(&current(&st), 5);
    st.restore(&ctx, &swapped).unwrap();
    // The KDA tensors current now are the former alternates, back to their
    // pattern (seed 3, their order among the alternates); the former
    // current ones, alternates now, are not touched by the restore.
    let kda_now: Vec<&Tensor> = st
        .layers
        .iter()
        .flat_map(|l| match l {
            LayerState::Kda { conv, s, .. } => vec![conv, s],
            LayerState::Dsa { .. } => vec![],
        })
        .collect();
    check(&kda_now, 3, "swapped pair restored");
    for (t, (x, b)) in alternates(&sc).iter().zip(&before).enumerate() {
        let got = x.to_f32().unwrap();
        assert!(
            got.iter().zip(b).all(|(g, w)| g.to_bits() == w.to_bits()),
            "alternate {t} changed"
        );
    }
}

/// Two sessions taking turns through one scratch's KDA alternates (A
/// rejects, B rejects, A rejects; each verify finished before the other
/// session runs): a verify's writes to the alternates never change the idle
/// session or a saved snapshot; snapshot, restore and fork after several
/// swaps; state sizes unchanged by swaps; one session reset and dropped
/// while the other lives on. A "verify" here writes the alternates as the
/// verify pass's first row would, a "reject" is `take_kda_snapshots`.
#[test]
fn sessions_share_the_scratch_alternates_safely() {
    use crate::engine::DecodeStateApi;
    let ctx = MetalContext::new().expect("context");
    let cfg = Config::glm53_flash();
    let mut sc = Scratch::new(&ctx, &cfg).expect("scratch");
    fn kda(st: &Glm53State) -> Vec<&Tensor> {
        st.layers
            .iter()
            .flat_map(|l| match l {
                LayerState::Kda { conv, s } => vec![conv, s],
                LayerState::Dsa { .. } => vec![],
            })
            .collect()
    }
    fn alternates(sc: &Scratch) -> Vec<&Tensor> {
        sc.kda_alt.iter().flatten().flat_map(|(c, s)| [c, s]).collect()
    }
    let word = |k: u32, t: usize, i: usize| {
        0x3f80_0000 ^ ((k << 26) | ((t as u32) << 18) | (i as u32 & 0x3ffff))
    };
    let fill = |ts: &[&Tensor], k: u32| {
        for (t, x) in ts.iter().enumerate() {
            let v: Vec<f32> =
                (0..x.numel()).map(|i| f32::from_bits(word(k, t, i))).collect();
            x.write_bytes(bytemuck::cast_slice(&v)).unwrap();
        }
    };
    let check = |ts: &[&Tensor], k: u32, what: &str| {
        for (t, x) in ts.iter().enumerate() {
            for (i, g) in x.to_f32().unwrap().iter().enumerate() {
                assert_eq!(g.to_bits(), word(k, t, i), "{what}: tensor {t} value {i}");
            }
        }
    };
    let mut a = Glm53State::new(&ctx, &cfg, 64).unwrap();
    let mut b = Glm53State::new(&ctx, &cfg, 64).unwrap();
    fill(&kda(&a), 1);
    fill(&kda(&b), 2);
    let (bytes_a, bytes_b) = (a.bytes(), b.bytes());
    let snap_a = a.snapshot(&ctx).unwrap();
    // A: verify, reject.
    fill(&alternates(&sc), 3);
    a.take_kda_snapshots(&mut sc);
    check(&kda(&a), 3, "A after its reject");
    check(&kda(&b), 2, "B idle");
    // B: verify (into A's former tensors), reject.
    fill(&alternates(&sc), 4);
    check(&kda(&a), 3, "A idle through B's verify");
    b.take_kda_snapshots(&mut sc);
    check(&kda(&b), 4, "B after its reject");
    // A again: verify (into B's former tensors), reject.
    fill(&alternates(&sc), 5);
    check(&kda(&b), 4, "B idle through A's verify");
    a.take_kda_snapshots(&mut sc);
    check(&kda(&a), 5, "A after its second reject");
    assert_eq!((a.bytes(), b.bytes()), (bytes_a, bytes_b), "state sizes");
    // The snapshot taken before the swaps still holds A's first values.
    let mut c = Glm53State::new(&ctx, &cfg, 64).unwrap();
    c.restore(&ctx, &snap_a).unwrap();
    check(&kda(&c), 1, "snapshot from before the swaps");
    // Snapshot, mutate, restore and fork after the swaps.
    let snap = a.snapshot(&ctx).unwrap();
    fill(&kda(&a), 6);
    a.restore(&ctx, &snap).unwrap();
    check(&kda(&a), 5, "restored after swaps");
    c.restore(&ctx, &snap).unwrap();
    check(&kda(&c), 5, "forked after swaps");
    // B reset and dropped; A and the scratch carry on.
    b.reset();
    assert!(
        kda(&b).iter().all(|t| t.to_f32().unwrap().iter().all(|v| v.to_bits() == 0))
    );
    check(&kda(&a), 5, "A through B's reset");
    drop(b);
    fill(&alternates(&sc), 7);
    check(&kda(&a), 5, "A through a verify after B's drop");
    a.take_kda_snapshots(&mut sc);
    check(&kda(&a), 7, "A after a reject with B gone");
    check(&kda(&c), 5, "the fork untouched");
}

/// The MTP draft head's requantization: a Q8_0 block as Q4_0 reads back
/// (the GGUF reader's Q4_0) within half a Q4_0 step of the Q8_0 values (a
/// whole step for values the format clamps); an all-zero block stays zero.
#[test]
fn q8_0_blocks_requantize_to_q4_0() {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(83);
    let mut blocks: Vec<[u8; 34]> = Vec::new();
    for b in 0..200 {
        let mut blk = [0u8; 34];
        let d = half::f16::from_f32(rng.gen_range(1e-4f32..2e-2));
        blk[..2].copy_from_slice(&d.to_le_bytes());
        for i in 0..32 {
            let q: i8 = match b % 4 {
                0 => rng.gen_range(-127..=127),
                1 => rng.gen_range(-10..=10),
                2 => {
                    if i == 7 {
                        -127
                    } else {
                        rng.gen_range(-60..=60)
                    }
                }
                _ => {
                    if i == 31 {
                        127
                    } else {
                        rng.gen_range(-127..=127)
                    }
                }
            };
            blk[2 + i] = q as u8;
        }
        blocks.push(blk);
    }
    blocks.push([0u8; 34]);
    for blk in &blocks {
        let mut x = [0f32; 32];
        crate::gguf::dequantize(crate::gguf::GgmlType::Q8_0, blk, &mut x).unwrap();
        let mut q4 = [0u8; 18];
        q8_0_to_q4_0(blk, &mut q4);
        let mut y = [0f32; 32];
        crate::gguf::dequantize(crate::gguf::GgmlType::Q4_0, &q4, &mut y).unwrap();
        let max =
            x.iter().copied().fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let step = max.abs() / 8.0;
        for (i, (a, b)) in x.iter().zip(&y).enumerate() {
            // Q4_0 holds -8 .. 7 steps with the largest magnitude at -8:
            // half a step, a whole one for values of the other sign past
            // 7.5 steps (clamped), plus the f16 rounding of the scale.
            let clamped = a * max < 0.0 && a.abs() > 7.5 * step;
            let tol = if clamped { 1.0 } else { 0.5 } * step + 8.0 * step * 1e-3 + 1e-9;
            assert!((a - b).abs() <= tol, "value {i}: {a} vs {b}");
        }
        if max == 0.0 {
            assert!(y.iter().all(|&v| v == 0.0));
        }
    }
}

/// The weight inventory (`weight_buffers`): the mapped buffers alone
/// without a draft head; with one, its buffer once more, not twice, and
/// not again when it lies in a mapped buffer.
#[test]
fn weight_inventory_lists_the_draft_head_once() {
    let ctx = MetalContext::new().expect("context");
    let mapped: Vec<Buffer> = (0..3).map(|_| ctx.new_buffer(64).unwrap()).collect();
    let same = |a: &Buffer, b: &Buffer| std::ptr::eq::<GpuBuffer>(&**a, &**b);
    let none = weight_inventory(&mapped, None);
    assert_eq!(none.len(), 3);
    assert!(none.iter().zip(&mapped).all(|(a, b)| same(a, b)));
    let head = Tensor::zeros(&ctx, &[16], crate::tensor::DType::F32).unwrap();
    let with = weight_inventory(&mapped, Some(&head));
    assert_eq!(with.len(), 4);
    assert_eq!(with.iter().filter(|b| same(b, &head.shared_buffer())).count(), 1);
    // A view into a mapped buffer adds nothing.
    let inside =
        Tensor::from_buffer(mapped[1].clone(), &[16], crate::tensor::DType::F32)
            .unwrap()
            .view(4, &[8])
            .unwrap();
    assert_eq!(weight_inventory(&mapped, Some(&inside)).len(), 3);
}

/// Known Q4_0 bytes for hand-made Q8_0 blocks: the first of two
/// equal-magnitude extrema sets the scale's sign (both orders), values
/// half a step from a rounding boundary round up, a -128 code, a zero
/// scale (the scale then -0.0, as llama.cpp's max / -8 gives), a negative
/// Q8_0 scale.
#[test]
fn q8_0_to_q4_0_known_blocks() {
    // (Q8_0 scale bits, (index, code) pairs (others 0), Q4_0 scale bits,
    // (index, nibble) pairs (others 8)).
    #[allow(clippy::type_complexity)]
    let cases: &[(u16, &[(usize, i8)], u16, &[(usize, u8)])] = &[
        // x = 0.5 q: -8 first, then 8; d = 1.
        (
            0x3800,
            &[(0, -16), (1, 1), (2, -1), (3, 3), (4, 16)],
            0x3c00,
            &[(0, 0), (1, 9), (2, 8), (3, 10), (4, 15)],
        ),
        // 8 first, then -8; d = -1.
        (
            0x3800,
            &[(0, 16), (1, 1), (2, -1), (3, 3), (4, -16)],
            0xbc00,
            &[(0, 0), (1, 8), (2, 9), (3, 7), (4, 15)],
        ),
        // x = q / 128: -128 sets d = 0.125; 127 clamps; 64 -> 12.
        (
            0x2000,
            &[(0, -128), (1, 127), (17, 64)],
            0x3000,
            &[(0, 0), (1, 15), (17, 12)],
        ),
        // A zero scale: every value 0.
        (0x0000, &[(0, 100), (5, -7)], 0x8000, &[]),
        // A negative Q8_0 scale: x = -0.5 q.
        (0xb800, &[(0, 16), (1, -1)], 0x3c00, &[(0, 0), (1, 9)]),
    ];
    for (n, &(d8, codes, d4, nibbles)) in cases.iter().enumerate() {
        let mut blk = [0u8; 34];
        blk[..2].copy_from_slice(&d8.to_le_bytes());
        for &(i, q) in codes {
            blk[2 + i] = q as u8;
        }
        let mut want = [0x88u8; 18];
        want[..2].copy_from_slice(&d4.to_le_bytes());
        for &(i, q) in nibbles {
            let byte = &mut want[2 + i % 16];
            *byte = if i < 16 { (*byte & 0xf0) | q } else { (*byte & 0x0f) | (q << 4) };
        }
        let mut got = [0u8; 18];
        q8_0_to_q4_0(&blk, &mut got);
        assert_eq!(got, want, "case {n}");
    }
}

/// Every staging path the engine uses fits the scratch it writes, at every
/// row count it is called with: a verify pass (1 or 2 rows, up to
/// MAX_ROWS), a decode step, a prefill run up to PREFILL_ROWS, the MTP
/// step. (The verify pass once wrote MAX_ROWS words into buffers sized for
/// a prefill run and failed every speculative request.)
#[test]
fn scratch_staging_fits_every_caller() {
    let ctx = MetalContext::new().expect("context");
    let cfg = Config::glm53_flash();
    let sc = Scratch::new(&ctx, &cfg).expect("scratch");
    for drafts in 0..MAX_ROWS {
        let d: Vec<u32> = (0..drafts as u32).map(|i| 100 + i).collect();
        sc.stage_spec_tokens(7, &d).unwrap();
        let got = sc.spec_tokens.to_u32().unwrap();
        assert_eq!(got[0], 7);
        assert_eq!(&got[1..=drafts], &d[..]);
        assert!(got[drafts + 1..].iter().all(|&v| v == 0));
        // Every row's draw has a slot.
        assert!(sc.draws.numel() > drafts);
    }
    assert!(sc.stage_spec_tokens(7, &[1; MAX_ROWS]).is_err());
    for n in [1, MAX_ROWS, MAX_ROWS + 1, PREFILL_ROWS] {
        let t: Vec<u32> = (0..n as u32).collect();
        sc.stage_tokens(&t).unwrap();
        assert_eq!(&sc.tokens.to_u32().unwrap()[..n], &t[..]);
    }
    assert!(sc.stage_tokens(&vec![0; PREFILL_ROWS + 1]).is_err());
    sc.stage_mtp_token(42).unwrap();
    assert_eq!(sc.mtp_io.to_u32().unwrap(), [42, 0]);
    assert!(sc.next_token.numel() >= 2);
}

/// The engine's whole speculative cycle on the published file, after a
/// prompt longer than one prefill run: `LILY_GLM53_GGUF=<path> cargo test
/// --release --lib glm53::model -- --ignored --test-threads=1`. The verify
/// pass's staging once failed here on every request.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn speculative_cycle_after_a_long_prefill() {
    use crate::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions};
    use crate::kernels::sample::SamplingParams;
    let path = std::env::var("LILY_GLM53_GGUF").expect("LILY_GLM53_GGUF");
    let ctx = MetalContext::new().expect("context");
    let options = LoadOptions { mtp_drafts: 1, ..Default::default() };
    let mut model = <Glm53Model as LanguageModel>::load(
        &ctx,
        std::path::Path::new(&path),
        &options,
    )
    .expect("load");
    // Every draft proposed (unlikely_drafts_are_not_proposed covers the
    // threshold).
    model.draft_min_prob = 0.0;
    let mut st = LanguageModel::new_state(&model, &ctx, 64).expect("state");
    let mut sc =
        LanguageModel::new_scratch_with_capacity(&model, &ctx, 0).expect("scratch");
    let prompt: Vec<u32> =
        (0..PREFILL_ROWS as u32 + 37).map(|i| 1000 + i % 5000).collect();
    // Sampling, not greedy (greedy() sets top_k 1), with penalties.
    let params = SamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        presence_penalty: 0.3,
        frequency_penalty: 0.2,
        repetition_penalty: 1.05,
        seed: 1234,
        ..SamplingParams::greedy()
    };
    assert!(!params.is_greedy());
    LanguageModel::prefill(
        &model,
        &ctx,
        &mut st,
        &mut sc,
        &prompt,
        Some(Draw { params: &params, step: 0 }),
    )
    .expect("prefill");
    assert_eq!(st.pos(), prompt.len());
    let mut last = sc.next_token.to_u32().unwrap()[0];
    let mut proposals = model
        .draft_initial(&ctx, &mut st, &mut sc, last, 1, &params, 1)
        .expect("draft");
    for step in 0..2 {
        let pos = st.pos();
        let (draws, pending) = model
            .verify(&ctx, &mut st, &mut sc, last, &proposals, &params, 2 + 2 * step, 1)
            .expect("verify");
        assert_eq!(draws.len(), 2);
        assert!(draws.iter().all(|&t| (t as usize) < model.cfg.vocab));
        let kept = usize::from(draws[0] == proposals[0]);
        proposals = model
            .finish_speculation(&ctx, &mut st, &mut sc, kept, true, pending)
            .expect("finish");
        assert_eq!(st.pos(), pos + 1 + kept);
        assert_eq!(proposals.len(), 1);
        last = draws[kept];
    }
}

/// The routing search of `lily-glm53 spec-diff` looks only at the rows fed
/// the same tokens (positions below `end`), skips rows one run lacks, and
/// finds nothing when only later rows differ.
#[test]
fn first_routing_difference_stays_in_the_common_prefix() {
    use std::collections::BTreeMap;
    let rec = |rows: &[(usize, [u32; 2])]| -> BTreeMap<usize, Vec<Vec<u32>>> {
        rows.iter().map(|&(p, l)| (p, vec![vec![l[0]], vec![l[1]]])).collect()
    };
    let a = rec(&[(10, [1, 2]), (11, [1, 2]), (12, [1, 2]), (13, [5, 6])]);
    let b = rec(&[(10, [1, 2]), (12, [1, 9]), (13, [7, 6])]);
    // Position 11 is only in `a`; 12 differs at layer 1; 13 at layer 0.
    assert_eq!(first_routing_difference(&a, &b, 14), Some((12, 1)));
    assert_eq!(first_routing_difference(&a, &b, 13), Some((12, 1)));
    // Rows from 12 on were fed different tokens: unresolved.
    assert_eq!(first_routing_difference(&a, &b, 12), None);
    assert_eq!(first_routing_difference(&a, &a, 100), None);
}

/// Loads the published file for the real-weight tests.
/// Loads without and with speculation: the weight inventory is the mapped
/// file's buffers, plus the Q4_0 draft head's buffer exactly once when
/// speculation is on (the head is Q8_0 in this checkpoint).
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn weight_buffers_include_the_draft_head_only_with_speculation() {
    use crate::engine::{LanguageModel, LoadOptions};
    let path = std::env::var("LILY_GLM53_GGUF").expect("LILY_GLM53_GGUF");
    let ctx = MetalContext::new().expect("context");
    let load = |drafts: usize| {
        let options = LoadOptions { mtp_drafts: drafts, ..Default::default() };
        <Glm53Model as LanguageModel>::load(&ctx, std::path::Path::new(&path), &options)
            .expect("load")
    };
    let mapped = {
        let model = load(0);
        assert!(model.draft_head.is_none());
        model.weight_buffers().len()
    };
    let model = load(1);
    let head = model.draft_head.as_ref().expect("a draft head").t.shared_buffer();
    let all = model.weight_buffers();
    assert_eq!(all.len(), mapped + 1);
    let same = |b: &Buffer| std::ptr::eq::<GpuBuffer>(&**b, &*head);
    assert_eq!(all.iter().filter(|b| same(b)).count(), 1);
}

fn real_model(ctx: &MetalContext) -> Glm53Model {
    use crate::engine::{LanguageModel, LoadOptions};
    let path = std::env::var("LILY_GLM53_GGUF").expect("LILY_GLM53_GGUF");
    let options = LoadOptions { mtp_drafts: 1, ..Default::default() };
    <Glm53Model as LanguageModel>::load(ctx, std::path::Path::new(&path), &options)
        .expect("load")
}

/// The engine proposes no draft below `draft_min_prob`: the verify pass is
/// then one row, whose draw is ordinary decoding's; at 0 every step
/// proposes its draft.
#[test]
#[ignore = "needs the GLM 5.3 GGUF (LILY_GLM53_GGUF)"]
fn unlikely_drafts_are_not_proposed() {
    use crate::engine::{DecodeStateApi, LanguageModel};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let mut m = real_model(&ctx);
    let greedy = SamplingParams::greedy();
    let p = prompt(64);
    let n = 8;
    let mut want = Vec::new();
    {
        let mut st = m.new_state(&ctx, p.len() + n + 8).unwrap();
        let mut sc = m.new_scratch(&ctx).unwrap();
        m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
        want.push(argmax(&sc.logits_last.to_f32().unwrap()));
        while want.len() < n {
            let last = *want.last().unwrap();
            m.forward(&ctx, &mut st, &mut sc, &[last], true).unwrap();
            want.push(argmax(&sc.logits_last.to_f32().unwrap()));
        }
    }
    for (gate, every) in [(1.5f32, false), (0.0, true)] {
        m.draft_min_prob = gate;
        let mut st = LanguageModel::new_state(&m, &ctx, p.len() + n + 8).unwrap();
        let mut sc = m.new_scratch(&ctx).unwrap();
        m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
        let mut out = vec![want[0]];
        let mut proposals =
            m.draft_initial(&ctx, &mut st, &mut sc, want[0], 1, &greedy, 1).unwrap();
        while out.len() < n {
            assert_eq!(proposals.len(), usize::from(every), "gate {gate}");
            let pos = st.pos();
            let pending = *out.last().unwrap();
            let (draws, pend) = m
                .verify(
                    &ctx,
                    &mut st,
                    &mut sc,
                    pending,
                    &proposals,
                    &greedy,
                    out.len(),
                    1,
                )
                .unwrap();
            assert_eq!(draws.len(), 1 + proposals.len(), "gate {gate}");
            let kept = proposals.first().map_or(0, |&d| usize::from(draws[0] == d));
            out.extend_from_slice(&draws[..=kept]);
            proposals =
                m.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pend).unwrap();
            assert_eq!(st.pos(), pos + 1 + kept, "gate {gate}");
        }
        if !every {
            // One-row steps are ordinary decoding's steps.
            assert_eq!(&out[..n], &want[..], "gate {gate}");
        }
    }
}

/// A prompt of `n` ordinary text tokens.
fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7919) % 50_000).collect()
}

/// Relative L2 distance of two logit vectors.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (f64::from(x) - f64::from(y)).powi(2);
        n += f64::from(y).powi(2);
    }
    (d / n).sqrt()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b }) as u32
}

/// Ordinary decoding: the logits after each of `tokens`, fed one at a time
/// after `prompt`.
fn ordinary(
    m: &Glm53Model,
    ctx: &MetalContext,
    prompt: &[u32],
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    let mut st = m.new_state(ctx, prompt.len() + tokens.len() + 8).unwrap();
    let mut sc = m.new_scratch(ctx).unwrap();
    m.prefill(ctx, &mut st, &mut sc, prompt).unwrap();
    tokens
        .iter()
        .map(|&t| {
            m.forward(ctx, &mut st, &mut sc, &[t], true).unwrap();
            sc.logits_last.to_f32().unwrap()
        })
        .collect()
}

/// Continues `st` ordinarily with `tokens` and checks each step's logits
/// against `want`: the same top token and a relative L2 distance within
/// 5e-3 (the verify pass runs KDA's multi-row kernels where ordinary
/// decoding runs the one-row kernel, so the sums differ in the last bits).
fn continue_and_compare(
    m: &Glm53Model,
    ctx: &MetalContext,
    st: &mut Glm53State,
    sc: &mut Scratch,
    tokens: &[u32],
    want: &[Vec<f32>],
    case: &str,
) {
    for (k, (&t, w)) in tokens.iter().zip(want).enumerate() {
        m.forward(ctx, st, sc, &[t], true).unwrap();
        let got = sc.logits_last.to_f32().unwrap();
        let r = rel(&got, w);
        eprintln!("{case}: step {k} relative L2 {r:.2e}");
        assert!(r < 5e-3, "{case}: step {k} logits differ (relative L2 {r:.2e})");
        assert_eq!(argmax(&got), argmax(w), "{case}: step {k} top token");
    }
}

/// Speculative steps leave the state ordinary decoding would have, for an
/// accepted draft, a rejected one, a confirmed draft cut at the first row
/// (the generation ends there), and a rejected row that completes an
/// indexer pool past the dense window; and a snapshot taken before a step
/// restores, and forks, to that state.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn speculative_steps_match_ordinary_decoding() {
    use crate::engine::{DecodeStateApi, LanguageModel};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    let greedy = SamplingParams::greedy();
    // 2102 = 2 mod 4 and past 2051: the verify pass's row 1 (position 2103)
    // completes a pool and the attention is sparse.
    for (n, cases) in
        [(37usize, &["accept", "reject", "first-row"][..]), (2102, &["reject"][..])]
    {
        let p = prompt(n);
        // The greedy continuation; t0 is what the prompt's logits pick.
        let mut st = m.new_state(&ctx, n + 16).unwrap();
        let mut sc = m.new_scratch(&ctx).unwrap();
        m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
        let mut toks = vec![argmax(&sc.logits_last.to_f32().unwrap())];
        for _ in 0..6 {
            let last = *toks.last().unwrap();
            m.forward(&ctx, &mut st, &mut sc, &[last], true).unwrap();
            toks.push(argmax(&sc.logits_last.to_f32().unwrap()));
        }
        drop((st, sc));
        let want = ordinary(&m, &ctx, &p, &toks);
        for &case in cases {
            let mut st = LanguageModel::new_state(&m, &ctx, n + 16).unwrap();
            let mut sc = m.new_scratch(&ctx).unwrap();
            m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
            m.draft_initial(&ctx, &mut st, &mut sc, toks[0], 1, &greedy, 1).unwrap();
            // The draft: the true next token, or a wrong one.
            let draft =
                if case == "reject" { (toks[1] + 1) % 1000 + 5 } else { toks[1] };
            assert_eq!(draft == toks[1], case != "reject");
            let (draws, pending) = m
                .verify(&ctx, &mut st, &mut sc, toks[0], &[draft], &greedy, 1, 1)
                .unwrap();
            assert_eq!(draws[0], toks[1], "{case} n={n}: row 0 draw");
            if case == "accept" {
                assert_eq!(draws[1], toks[2], "{case} n={n}: row 1 draw");
            }
            let kept = usize::from(case == "accept");
            m.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pending).unwrap();
            assert_eq!(st.pos(), n + 1 + kept);
            let label = format!("{case} n={n}");
            continue_and_compare(
                &m,
                &ctx,
                &mut st,
                &mut sc,
                &toks[1 + kept..],
                &want[1 + kept..],
                &label,
            );
        }
        if n == 37 {
            let mut st = LanguageModel::new_state(&m, &ctx, n + 16).unwrap();
            let mut sc = m.new_scratch(&ctx).unwrap();
            m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
            let snap = st.snapshot(&ctx).unwrap();
            m.draft_initial(&ctx, &mut st, &mut sc, toks[0], 1, &greedy, 1).unwrap();
            let (_, pending) = m
                .verify(&ctx, &mut st, &mut sc, toks[0], &[toks[1]], &greedy, 1, 1)
                .unwrap();
            m.finish_speculation(&ctx, &mut st, &mut sc, 1, true, pending).unwrap();
            let mut fork = LanguageModel::new_state(&m, &ctx, n + 16).unwrap();
            fork.copy_prefix_from(&ctx, &st, n).unwrap();
            fork.restore(&ctx, &snap).unwrap();
            st.restore(&ctx, &snap).unwrap();
            assert_eq!((st.pos(), fork.pos()), (n, n));
            continue_and_compare(&m, &ctx, &mut st, &mut sc, &toks, &want, "restored");
            continue_and_compare(&m, &ctx, &mut fork, &mut sc, &toks, &want, "forked");
        }
    }
}

/// One ordinary sampled decode step: feeds `input`, draws with `params` at
/// request step `step`, returns the draw.
fn sampled_step(
    m: &Glm53Model,
    ctx: &MetalContext,
    st: &mut Glm53State,
    sc: &Scratch,
    input: u32,
    params: &crate::kernels::sample::SamplingParams,
    step: usize,
) -> u32 {
    use crate::engine::{DecodeStateApi, Draw, LanguageModel};
    sc.next_token
        .view(0, &[1])
        .unwrap()
        .write_bytes(bytemuck::bytes_of(&input))
        .unwrap();
    let encoded =
        m.encode_decode_step(ctx, st, sc, 0, 1, Draw { params, step }).unwrap();
    m.prepare_step_inputs(st, sc, input).unwrap();
    let pending = encoded.commit().unwrap();
    st.advance(1);
    pending.wait().unwrap();
    sc.next_token.to_u32().unwrap()[1]
}

/// Sampled speculation emits what ordinary sampled decoding emits, with
/// the same penalty counts: the same seed, steps and penalties; a forced
/// rejection (its row-1 draw is discarded and uncounted), then a forced
/// acceptance (both rows' draws kept), then ordinary steps.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn sampled_speculation_matches_ordinary_sampling() {
    use crate::engine::{Draw, LanguageModel, ScratchApi};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 40,
        top_p: 0.95,
        presence_penalty: 0.5,
        frequency_penalty: 0.5,
        repetition_penalty: 1.1,
        seed: 4242,
        ..SamplingParams::greedy()
    };
    assert!(!params.is_greedy());
    // The patterned prompt: after a pseudo-random one, two of layer 7's
    // experts tied to 1e-6 at the top-8 boundary, and the verify pass's
    // rounding (any two-row pass's) picked the other one.
    let p = prompt(37);
    let steps = 7;
    // Ordinary: t0 from the prompt, then one draw a step, and the counts
    // after each.
    let mut st = LanguageModel::new_state(&m, &ctx, p.len() + 16).unwrap();
    let mut sc = m.new_scratch(&ctx).unwrap();
    sc.begin_request();
    let draw = Some(Draw { params: &params, step: 0 });
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p, draw).unwrap();
    let mut toks = vec![sc.next_token.to_u32().unwrap()[0]];
    let mut counts = vec![sc.sampler().counts().unwrap()];
    let mut logits = vec![sc.logits_last.to_f32().unwrap()];
    let mut off_argmax =
        usize::from(toks[0] != argmax(&sc.logits_last.to_f32().unwrap()));
    for k in 1..steps {
        let t = sampled_step(&m, &ctx, &mut st, &sc, toks[k - 1], &params, k);
        off_argmax += usize::from(t != argmax(&sc.logits_last.to_f32().unwrap()));
        toks.push(t);
        counts.push(sc.sampler().counts().unwrap());
        logits.push(sc.logits_last.to_f32().unwrap());
    }
    eprintln!("sampled: {toks:?}, {off_argmax} draws off the argmax");
    assert!(off_argmax > 0, "the run never left the argmax");
    assert!(counts[steps - 1].iter().sum::<u32>() > 0, "penalties count draws");
    drop((st, sc));
    // Speculative.
    let mut st = LanguageModel::new_state(&m, &ctx, p.len() + 16).unwrap();
    let mut sc = m.new_scratch(&ctx).unwrap();
    sc.begin_request();
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p, draw).unwrap();
    assert_eq!(sc.next_token.to_u32().unwrap()[0], toks[0]);
    m.draft_initial(&ctx, &mut st, &mut sc, toks[0], 1, &params, 1).unwrap();
    // Rejected: a draft that is not t1.
    let wrong = if toks[1] == 1000 { 1001 } else { 1000 };
    let (draws, pending) =
        m.verify(&ctx, &mut st, &mut sc, toks[0], &[wrong], &params, 1, 1).unwrap();
    assert_eq!(draws[0], toks[1], "rejected step: row 0 draw");
    m.finish_speculation(&ctx, &mut st, &mut sc, 0, true, pending).unwrap();
    assert_eq!(sc.sampler().counts().unwrap(), counts[1], "counts after a rejection");
    // Accepted: the draft is t2, so rows 0 and 1 draw t2 and t3.
    let (draws, pending) =
        m.verify(&ctx, &mut st, &mut sc, toks[1], &[toks[2]], &params, 2, 1).unwrap();
    assert_eq!(draws, [toks[2], toks[3]], "accepted step: both rows' draws");
    m.finish_speculation(&ctx, &mut st, &mut sc, 1, false, pending).unwrap();
    assert_eq!(sc.sampler().counts().unwrap(), counts[3], "counts after an acceptance");
    // Ordinary steps after both.
    for k in 4..steps {
        let t = sampled_step(&m, &ctx, &mut st, &sc, toks[k - 1], &params, k);
        let got = sc.logits_last.to_f32().unwrap();
        let r = rel(&got, &logits[k]);
        eprintln!("step {k} after speculation: relative L2 {r:.2e}");
        assert!(r < 5e-3, "step {k} logits differ (relative L2 {r:.2e})");
        assert_eq!(t, toks[k], "step {k} after speculation");
        assert_eq!(sc.sampler().counts().unwrap(), counts[k], "step {k} counts");
    }
}

/// Sampled speculation that follows the engine's own proposal lists, both
/// empty (the draft below `draft_min_prob`: a one-row step) and not (the
/// threshold alternates between 1.5 and 0 from step to step), emits what
/// ordinary sampled decoding emits: the same draws at the same request
/// steps, the same penalty counts after every step, positions advancing by
/// the kept rows, and ordinary steps after it giving ordinary decoding's
/// logits. (A routing tie that the two paths' rounding decides differently
/// can part the two; this prompt and seed have none in these steps.)
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn sampled_speculation_follows_its_proposals() {
    use crate::engine::{DecodeStateApi, Draw, LanguageModel, ScratchApi};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let mut m = real_model(&ctx);
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 40,
        top_p: 0.95,
        presence_penalty: 0.5,
        frequency_penalty: 0.5,
        repetition_penalty: 1.1,
        seed: 777,
        ..SamplingParams::greedy()
    };
    let (steps, after) = (24, 3);
    // Text: drafts are accepted at times and draws leave the argmax.
    let path = std::env::var("LILY_GLM53_GGUF").expect("LILY_GLM53_GGUF");
    let tok =
        crate::tokenizer::Tokenizer::from_gguf(std::path::Path::new(&path)).unwrap();
    let mut p = tok.encode("[gMASK]<sop>").unwrap();
    p.extend(
        tok.encode(
            "The river rises in the hills north of the town, runs past the old mill \
             and the market square, and reaches the sea after a journey of",
        )
        .unwrap(),
    );
    let total = steps + 2 + after;
    // Ordinary: the draws and the counts after each.
    let mut st = LanguageModel::new_state(&m, &ctx, p.len() + total + 8).unwrap();
    let mut sc = m.new_scratch(&ctx).unwrap();
    sc.begin_request();
    let draw = Some(Draw { params: &params, step: 0 });
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p, draw).unwrap();
    let mut toks = vec![sc.next_token.to_u32().unwrap()[0]];
    let mut counts = vec![sc.sampler().counts().unwrap()];
    let mut logits = vec![sc.logits_last.to_f32().unwrap()];
    let mut off_argmax = 0;
    for k in 1..total {
        let t = sampled_step(&m, &ctx, &mut st, &sc, toks[k - 1], &params, k);
        off_argmax +=
            usize::from(k < steps && t != argmax(&sc.logits_last.to_f32().unwrap()));
        toks.push(t);
        counts.push(sc.sampler().counts().unwrap());
        logits.push(sc.logits_last.to_f32().unwrap());
    }
    eprintln!("ordinary: {toks:?}, {off_argmax} draws off the argmax");
    assert!(off_argmax > 0, "the run never left the argmax");
    drop((st, sc));
    // Speculative, as the engine's loop runs it.
    let mut st = LanguageModel::new_state(&m, &ctx, p.len() + total + 8).unwrap();
    let mut sc = m.new_scratch(&ctx).unwrap();
    sc.begin_request();
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p, draw).unwrap();
    let mut out = vec![sc.next_token.to_u32().unwrap()[0]];
    m.draft_min_prob = 0.0;
    let mut proposals =
        m.draft_initial(&ctx, &mut st, &mut sc, out[0], 1, &params, 1).unwrap();
    let (mut empty, mut proposed, mut accepted) = (0, 0, 0);
    let mut i = 0;
    while out.len() < steps {
        if proposals.is_empty() {
            empty += 1;
        } else {
            proposed += 1;
        }
        let (step0, pos) = (out.len(), st.pos());
        let pending = *out.last().unwrap();
        let (draws, pend) = m
            .verify(&ctx, &mut st, &mut sc, pending, &proposals, &params, step0, 1)
            .unwrap();
        assert_eq!(draws.len(), 1 + proposals.len(), "step {i}: rows");
        let kept = proposals.first().map_or(0, |&d| usize::from(draws[0] == d));
        accepted += kept;
        out.extend_from_slice(&draws[..=kept]);
        m.draft_min_prob = if i % 2 == 0 { 1.5 } else { 0.0 };
        proposals =
            m.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pend).unwrap();
        assert_eq!(proposals.len(), i % 2, "step {i}: proposals after the threshold");
        assert_eq!(st.pos(), pos + 1 + kept, "step {i}: position");
        assert_eq!(&out[..], &toks[..out.len()], "step {i}: draws");
        let c = &counts[out.len() - 1];
        assert_eq!(&sc.sampler().counts().unwrap(), c, "step {i}: penalty counts");
        i += 1;
    }
    eprintln!(
        "speculative: {empty} one-row steps, {proposed} drafts, {accepted} accepted"
    );
    assert!(empty > 0 && proposed > 0, "both kinds of step");
    assert!(0 < accepted && accepted < proposed, "drafts both accepted and rejected");
    // Ordinary steps continue from the speculative state.
    for k in out.len()..out.len() + after {
        let t = sampled_step(&m, &ctx, &mut st, &sc, toks[k - 1], &params, k);
        let r = rel(&sc.logits_last.to_f32().unwrap(), &logits[k]);
        eprintln!("step {k} after speculation: relative L2 {r:.2e}");
        assert!(r < 5e-3, "step {k} logits differ (relative L2 {r:.2e})");
        assert_eq!(t, toks[k], "step {k} after speculation");
        assert_eq!(sc.sampler().counts().unwrap(), counts[k], "step {k} counts");
    }
}

/// A synchronous step whose last pass fails to encode has run its earlier
/// passes (and waited for them): the recurrent state is partly advanced
/// while the position is not. Restoring a snapshot taken before the step
/// recovers it: the step then gives exactly the logits it gives from the
/// snapshot without the failure.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn a_failed_step_is_recovered_from_a_snapshot() {
    use crate::engine::DecodeStateApi;
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    let p = prompt(40);
    let mut st = m.new_state(&ctx, p.len() + 8).unwrap();
    let mut sc = m.new_scratch(&ctx).unwrap();
    m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
    let snap = st.snapshot(&ctx).unwrap();
    let token = 2025;
    m.forward(&ctx, &mut st, &mut sc, &[token], true).unwrap();
    let want = sc.logits_last.to_f32().unwrap();
    st.restore(&ctx, &snap).unwrap();
    let kda0 = |st: &Glm53State| {
        st.layers
            .iter()
            .find_map(|l| match l {
                LayerState::Kda { s, .. } => Some(s.to_f32().unwrap()),
                LayerState::Dsa { .. } => None,
            })
            .unwrap()
    };
    // The last KDA layer: its pass is the one that fails.
    let kda_last = |st: &Glm53State| {
        st.layers[..m.cfg.n_trunk]
            .iter()
            .rev()
            .find_map(|l| match l {
                LayerState::Kda { s, .. } => Some(s.to_f32().unwrap()),
                LayerState::Dsa { .. } => None,
            })
            .unwrap()
    };
    let (before, last_before) = (kda0(&st), kda_last(&st));
    sc.stage_tokens(&[token]).unwrap();
    let ran = std::cell::Cell::new(false);
    let r = m.run_step(&ctx, &st, &sc, &sc.tokens, 1, Logits::Last, None, |_| {
        ran.set(true);
        anyhow::bail!("injected failure")
    });
    assert!(ran.get(), "the tail ran");
    let e = r.expect_err("the step reports the failure");
    assert_eq!(
        e.to_string(),
        "injected failure",
        "the injected error is the one returned"
    );
    assert_eq!(st.pos(), p.len(), "the position is unchanged");
    assert_ne!(kda0(&st), before, "the first passes ran: the first KDA layer moved");
    assert_eq!(kda_last(&st), last_before, "the failed last pass never ran");
    st.restore(&ctx, &snap).unwrap();
    assert_eq!(kda0(&st), before);
    m.forward(&ctx, &mut st, &mut sc, &[token], true).unwrap();
    assert_eq!(sc.logits_last.to_f32().unwrap(), want, "the step after the restore");
}

/// prefill_until stops at a run boundary before committing it and reports
/// what it fed; the rest fed afterwards gives exactly what one prefill of
/// the whole prompt gives.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn prefill_until_stops_at_run_boundaries_and_resumes_exactly() {
    use crate::engine::{DecodeStateApi, Draw, LanguageModel};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    assert_eq!(m.prefill_chunk_tokens(), Some(PREFILL_ROWS));
    let p = prompt(2 * PREFILL_ROWS + 100);
    let greedy = SamplingParams::greedy();
    let draw = || Some(Draw { params: &greedy, step: 0 });
    let mut sc = m.new_scratch(&ctx).unwrap();
    let mut st = LanguageModel::new_state(&m, &ctx, p.len()).unwrap();
    assert_eq!(m.prefill_until(&ctx, &mut st, &mut sc, &p, None, &|| true).unwrap(), 0);
    assert_eq!(st.pos(), 0);
    let asked = std::cell::Cell::new(0);
    let stop = || {
        asked.set(asked.get() + 1);
        asked.get() > 1
    };
    let fed = m.prefill_until(&ctx, &mut st, &mut sc, &p, None, &stop).unwrap();
    assert_eq!((fed, st.pos()), (PREFILL_ROWS, PREFILL_ROWS));
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p[fed..], draw()).unwrap();
    let resumed = sc.logits_last.to_f32().unwrap();
    let resumed_draw = sc.next_token.to_u32().unwrap()[0];
    let mut whole = LanguageModel::new_state(&m, &ctx, p.len()).unwrap();
    LanguageModel::prefill(&m, &ctx, &mut whole, &mut sc, &p, draw()).unwrap();
    assert_eq!(sc.logits_last.to_f32().unwrap(), resumed, "same runs, same logits");
    assert_eq!(sc.next_token.to_u32().unwrap()[0], resumed_draw);
}
