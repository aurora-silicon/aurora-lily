//! Qwen3.6's session state against the attention kernel's bounds, and the
//! speculative cycle's paths at rest (finish without proposing, snapshots,
//! prefix forks, the context's last rows) against plain greedy decoding.

use super::*;
use crate::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, SnapshotApi};
use crate::kernels::attention::MAX_SEQ;
use crate::kernels::attention::{KvCache, KvFormat};
use crate::kernels::sample::SamplingParams;
use crate::metal::MetalContext;
use crate::qwen36::Config;
use crate::qwen36::model::{
    LayerState, Logits, Qwen36Model, Qwen36State, Scratch, capacity_cap,
};

#[test]
fn growth_doubles_up_to_the_cap() {
    // Past half the kernel's limit the next growth is the limit, not twice.
    assert_eq!(grown_capacity(150_004, 150_005, MAX_SEQ), MAX_SEQ);
    assert_eq!(grown_capacity(1000, 1001, MAX_SEQ), 2000);
    // A non-power-of-two capacity, and a request past the doubling.
    assert_eq!(grown_capacity(6, 9, 1000), 12);
    assert_eq!(grown_capacity(6, 21, 1000), 24);
    assert_eq!(grown_capacity(600, 601, 1000), 1000);
    assert_eq!(capacity_cap(MAX_SEQ), MAX_SEQ);
    assert_eq!(capacity_cap(32768), 32772);
    assert_eq!(capacity_cap(32766), 32768);
}

#[test]
fn drafts_stop_short_of_the_last_rows() {
    let max = 1000;
    assert!(draft_fits(996, max));
    assert!(draft_fits(997, max));
    assert!(!draft_fits(998, max));
    assert!(!draft_fits(999, max));
    // A verify pass from 997 with a kept draft: the draft pair (head row
    // 999) still fits; from 998 (no draft, one row) the head writes 999.
    assert_eq!(mtp_pairs(997, 1, true, 1000), (2, true));
    assert_eq!(mtp_pairs(998, 0, true, 1000), (1, true));
    assert_eq!(mtp_pairs(999, 0, true, 1000), (0, false));
    assert_eq!(mtp_pairs(998, 1, true, 1000), (1, false));
    assert_eq!(mtp_pairs(10, 1, false, 1000), (1, false));
    assert_eq!(mtp_pairs(10, 0, false, 1000), (0, false));
}

/// A small Qwen3.6 shape (one attention layer, small GDN) for state tests.
fn small() -> Config {
    let mut c = Config::qwen36_35b_a3b();
    c.n_layer = 4;
    c.recurrent = vec![true, true, true, false];
    c.gdn_k_heads = 2;
    c.gdn_v_heads = 2;
    c
}

/// Row `row` of head 0 of a cache, as bytes.
fn cache_row(c: &KvCache, row: usize) -> Vec<u8> {
    let (_, _, d) = c.kv().shape();
    let at = row * d;
    c.values.view(at, &[d]).unwrap().contents().to_vec()
}

fn mark_row(c: &KvCache, row: usize, v: u8) {
    let (_, _, d) = c.kv().shape();
    c.values.view(row * d, &[d]).unwrap().write_bytes(&vec![v; 2 * d]).unwrap();
}

#[test]
fn caches_grow_to_the_kernel_limit_and_keep_their_prefix() {
    let ctx = MetalContext::new().unwrap();
    let cfg = small();
    // A capacity past half the limit and not a power of two.
    let mut st =
        Qwen36State::new(&ctx, &cfg, KvFormat::Bf16, 150_001, MAX_SEQ, true).unwrap();
    assert_eq!(st.capacity, 150_004);
    st.pos = 150_000;
    st.mtp.as_mut().unwrap().pos = 150_001;
    let LayerState::Attn { k, .. } = &st.layers[3] else { panic!("layer 3") };
    mark_row(k, 149_999, 7);
    mark_row(&st.mtp.as_ref().unwrap().k, 150_000, 9);
    st.ensure_capacity(&ctx, 150_005).unwrap();
    assert_eq!(st.capacity, MAX_SEQ);
    let LayerState::Attn { k, .. } = &st.layers[3] else { panic!("layer 3") };
    assert!(cache_row(k, 149_999).iter().all(|&b| b == 7));
    // The head's row ahead of the trunk survives too.
    assert!(cache_row(&st.mtp.as_ref().unwrap().k, 150_000).iter().all(|&b| b == 9));
    // Nothing past the limit, and no state bigger than its cap.
    assert!(st.ensure_capacity(&ctx, MAX_SEQ + 1).is_err());
    assert_eq!(st.capacity, MAX_SEQ);
    assert!(Qwen36State::new(&ctx, &cfg, KvFormat::Bf16, 101, 100, true).is_err());
    assert!(
        Qwen36State::new(&ctx, &cfg, KvFormat::Bf16, 10, MAX_SEQ + 4, true).is_err()
    );
}

#[test]
fn a_capped_state_grows_to_its_cap_only() {
    let ctx = MetalContext::new().unwrap();
    let cfg = small();
    let mut st = Qwen36State::new(&ctx, &cfg, KvFormat::Bf16, 7, 40, false).unwrap();
    assert_eq!(st.capacity, 8);
    st.ensure_capacity(&ctx, 9).unwrap();
    assert_eq!(st.capacity, 16);
    st.ensure_capacity(&ctx, 33).unwrap();
    assert_eq!(st.capacity, 36);
    // Doubling would be 72: the cap.
    st.ensure_capacity(&ctx, 37).unwrap();
    assert_eq!(st.capacity, 40);
    assert!(st.ensure_capacity(&ctx, 41).is_err());
}

// ---------------------------------------------------------------------------
// The published file: `LILY_QWEN36_GGUF=<Q4_K_M or Q8_0 path>` (the MTP file
// beside it) `cargo test --release --lib qwen36::engine -- --ignored
// --test-threads=1`.
// ---------------------------------------------------------------------------

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b }) as u32
}

struct Run {
    tokens: Vec<u32>,
    /// The proposal each verify pass took, by the position of its row 0.
    proposals: Vec<(usize, Vec<u32>)>,
}

/// Greedy speculation from a state at rest whose last draw `first` is not
/// fed, until at least `n` tokens (counting `first`; the last is never
/// fed) or the state's cap; with
/// `rest_every`, every so many steps it finishes without proposing (as at
/// a checkpoint) and restarts with `draft_initial`.
fn speculate(
    model: &Qwen36Model,
    ctx: &MetalContext,
    st: &mut Qwen36State,
    sc: &mut Scratch,
    first: u32,
    n: usize,
    rest_every: Option<usize>,
) -> Run {
    let depth = model.max_drafts();
    let greedy = SamplingParams::greedy();
    let mut out = vec![first];
    let mut log = Vec::new();
    let mut proposals =
        model.draft_initial(ctx, st, sc, first, depth, &greedy, 1).unwrap();
    let mut step = 0;
    while out.len() < n && st.pos < st.max_capacity {
        let pending = *out.last().unwrap();
        log.push((st.pos, proposals.clone()));
        let (draws, pass) = model
            .verify(ctx, st, sc, pending, &proposals, &greedy, out.len(), depth)
            .unwrap();
        let mut kept = 0;
        for (j, &t) in draws.iter().enumerate() {
            out.push(t);
            kept = j;
            if j >= proposals.len() || proposals[j] != t {
                break;
            }
        }
        step += 1;
        let rest = rest_every.is_some_and(|e| step % e == 0);
        proposals = model.finish_speculation(ctx, st, sc, kept, !rest, pass).unwrap();
        if rest {
            let last = *out.last().unwrap();
            proposals = model
                .draft_initial(ctx, st, sc, last, depth, &greedy, out.len())
                .unwrap();
        }
    }
    Run { tokens: out, proposals: log }
}

#[test]
#[ignore = "requires LILY_QWEN36_GGUF"]
fn speculation_at_rest_restores_and_forks_like_plain_decoding() {
    let path = std::env::var("LILY_QWEN36_GGUF").expect("LILY_QWEN36_GGUF");
    let ctx = MetalContext::new().unwrap();
    let options = LoadOptions { mtp_drafts: 3, ..Default::default() };
    let model = <Qwen36Model as LanguageModel>::load(
        &ctx,
        std::path::Path::new(&path),
        &options,
    )
    .unwrap();
    assert!(model.mtp.is_some(), "no MTP file beside {path}");
    let greedy = SamplingParams::greedy();
    let prompt: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    let n = 48;
    let mut sc = LanguageModel::new_scratch_with_capacity(&model, &ctx, 0).unwrap();
    let prefill = |st: &mut Qwen36State, sc: &mut Scratch| -> u32 {
        st.reset();
        let d = Draw { params: &greedy, step: 0 };
        LanguageModel::prefill(&model, &ctx, st, sc, &prompt, Some(d)).unwrap();
        sc.next_token.to_u32().unwrap()[0]
    };

    // Plain greedy decoding.
    let mut st = LanguageModel::new_state(&model, &ctx, 512).unwrap();
    let first = prefill(&mut st, &mut sc);
    let mut plain = vec![first];
    while plain.len() < n {
        model
            .forward(&ctx, &mut st, &mut sc, &[*plain.last().unwrap()], Logits::Last)
            .unwrap();
        plain.push(argmax(&sc.logits_last.to_f32().unwrap()));
    }

    // Speculation, and speculation that comes to rest every third step.
    let mut st = LanguageModel::new_state(&model, &ctx, 512).unwrap();
    let first = prefill(&mut st, &mut sc);
    let a = speculate(&model, &ctx, &mut st, &mut sc, first, n, None);
    let agree = |t: &[u32], what: &str| {
        let k = t.len().min(plain.len());
        assert_eq!(&t[..k], &plain[..k], "{what}");
    };
    agree(&a.tokens, "speculation");
    let first = prefill(&mut st, &mut sc);
    let b = speculate(&model, &ctx, &mut st, &mut sc, first, n, Some(3));
    agree(&b.tokens, "speculation with rests");

    // At rest after 20 tokens: snapshot, go on, restore, go on again; and a
    // fork (prefix copy + snapshot) into a fresh state. The proposals after
    // the restore and in the fork are the first run's (the head's cache
    // and pending row came back with the state).
    let first = prefill(&mut st, &mut sc);
    let head = speculate(&model, &ctx, &mut st, &mut sc, first, 20, Some(1));
    let at_rest = st.pos;
    let last = *head.tokens.last().unwrap();
    let snap = st.snapshot(&ctx).unwrap();
    assert_eq!(snap.pos(), at_rest);
    let rest = |r: Run| [&head.tokens[..head.tokens.len() - 1], &r.tokens].concat();
    let go = |st: &mut Qwen36State, sc: &mut Scratch| {
        speculate(&model, &ctx, st, sc, last, n - head.tokens.len() + 1, None)
    };
    let c1 = go(&mut st, &mut sc);
    st.restore(&ctx, &snap).unwrap();
    let c2 = go(&mut st, &mut sc);
    assert_eq!(c1.proposals, c2.proposals, "proposals after the restore");
    let mut fork = LanguageModel::new_state(&model, &ctx, 64).unwrap();
    fork.copy_prefix_from(&ctx, &st, at_rest).unwrap();
    fork.restore(&ctx, &snap).unwrap();
    let c3 = go(&mut fork, &mut sc);
    assert_eq!(c1.proposals, c3.proposals, "proposals in the fork");
    agree(&rest(c1), "after the snapshot");
    agree(&rest(c2), "after the restore");
    agree(&rest(c3), "in the fork");

    // The context's last rows: a state capped 30 rows past the prompt
    // speculates to its end, drafts stopping short of it, and agrees.
    let cap = (prompt.len() + 30).next_multiple_of(4);
    let mut st =
        Qwen36State::new(&ctx, &model.cfg, model.kv_format, prompt.len(), cap, true)
            .unwrap();
    let first = prefill(&mut st, &mut sc);
    let e = speculate(&model, &ctx, &mut st, &mut sc, first, n, None);
    assert_eq!(st.pos, cap, "fed to the cap");
    agree(&e.tokens, "to the end of the context");
    assert!(e.proposals.iter().all(|(p, d)| d.is_empty() || draft_fits(*p, cap)));
}

/// What a forced speculative step does with the head's draft.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Force {
    /// Verify the head's proposal as it is.
    Keep,
    /// Verify a wrong draft in its place (the next token id): rejected.
    Reject,
    /// Verify no draft (as the gate does).
    Suppress,
}

#[derive(Default, Debug)]
struct Forced {
    tokens: Vec<u32>,
    proposals: Vec<(usize, Vec<u32>)>,
    accepted: usize,
    rejected: usize,
    suppressed: usize,
    /// The last step's outcome.
    last: Option<Force>,
}

/// The head's state after a finished step: one row past the trunk while it
/// drafts, at the trunk at rest with the kept row's hidden state pending.
fn check_head(st: &Qwen36State, sc: &Scratch, kept: usize, rest: bool, hidden: usize) {
    let m = st.mtp.as_ref().expect("MTP state");
    if rest {
        assert_eq!(m.pos, st.pos, "the head at rest stands at the trunk");
        let row = sc.hidden_norm.view(kept * hidden, &[hidden]).unwrap();
        assert_eq!(
            m.pending_h.to_f32().unwrap(),
            row.to_f32().unwrap(),
            "the pending row at rest is the kept row's hidden state"
        );
    } else {
        assert_eq!(m.pos, st.pos + 1, "the drafting head is a row past the trunk");
    }
}

/// [`speculate`] with each step's draft forced by `plan(step)` and sampling
/// `params`; it comes to rest after the steps in `rests` and restarts with
/// `draft_initial` (or, with `stop_after_rest`, returns at the first rest).
/// Checks the head's state after every step.
#[allow(clippy::too_many_arguments)]
fn speculate_forced(
    model: &Qwen36Model,
    ctx: &MetalContext,
    st: &mut Qwen36State,
    sc: &mut Scratch,
    first: u32,
    n: usize,
    params: &SamplingParams,
    plan: &dyn Fn(usize) -> Force,
    rests: &[usize],
    stop_after_rest: bool,
) -> Forced {
    let vocab = model.cfg.vocab as u32;
    let mut r = Forced { tokens: vec![first], ..Default::default() };
    let mut proposals = model.draft_initial(ctx, st, sc, first, 1, params, 1).unwrap();
    let mut step = 0;
    while r.tokens.len() < n && st.pos < st.max_capacity {
        let pending = *r.tokens.last().unwrap();
        let force = plan(step);
        let drafts: Vec<u32> = match force {
            Force::Keep => proposals.clone(),
            Force::Reject => proposals.iter().map(|&d| (d + 1) % vocab).collect(),
            Force::Suppress => Vec::new(),
        };
        r.proposals.push((st.pos, proposals.clone()));
        let (draws, pass) = model
            .verify(ctx, st, sc, pending, &drafts, params, r.tokens.len(), 1)
            .unwrap();
        let mut kept = 0;
        for (j, &t) in draws.iter().enumerate() {
            r.tokens.push(t);
            kept = j;
            if j >= drafts.len() || drafts[j] != t {
                break;
            }
        }
        match (drafts.is_empty(), kept) {
            (true, _) => r.suppressed += 1,
            (false, 1) => r.accepted += 1,
            (false, _) => r.rejected += 1,
        }
        let rest = rests.contains(&step);
        proposals = model.finish_speculation(ctx, st, sc, kept, !rest, pass).unwrap();
        check_head(st, sc, kept, rest, model.cfg.hidden);
        r.last = Some(match (drafts.is_empty(), kept) {
            (true, _) => Force::Suppress,
            (false, 1) => Force::Keep,
            (false, _) => Force::Reject,
        });
        if rest {
            assert!(proposals.is_empty(), "no proposal at rest");
            if stop_after_rest {
                break;
            }
            let last = *r.tokens.last().unwrap();
            proposals = model
                .draft_initial(ctx, st, sc, last, 1, params, r.tokens.len())
                .unwrap();
        }
        step += 1;
    }
    r
}

/// Plain decoding through the engine's decode step (the in-graph sampler,
/// its emission counts and penalties), `n` tokens after the prefill's.
fn plain_sampled(
    model: &Qwen36Model,
    ctx: &MetalContext,
    st: &mut Qwen36State,
    sc: &mut Scratch,
    first: u32,
    n: usize,
    params: &SamplingParams,
) -> Vec<u32> {
    use crate::engine::ScratchApi;
    let mut out = vec![first];
    let mut slot = 0;
    while out.len() < n {
        let pass = LanguageModel::encode_decode_step(
            model,
            ctx,
            st,
            sc,
            slot,
            1 - slot,
            Draw { params, step: out.len() },
        )
        .unwrap();
        pass.commit().unwrap().wait().unwrap();
        st.advance(1);
        slot = 1 - slot;
        out.push(sc.next_token().to_u32().unwrap()[slot]);
    }
    out
}

#[test]
#[ignore = "requires LILY_QWEN36_GGUF"]
fn forced_draft_outcomes_follow_plain_decoding() {
    use crate::engine::ScratchApi;
    let path = std::env::var("LILY_QWEN36_GGUF").expect("LILY_QWEN36_GGUF");
    let ctx = MetalContext::new().unwrap();
    let options = LoadOptions { mtp_drafts: 1, ..Default::default() };
    let mut model = <Qwen36Model as LanguageModel>::load(
        &ctx,
        std::path::Path::new(&path),
        &options,
    )
    .unwrap();
    assert!(model.mtp.is_some(), "no MTP file beside {path}");
    // Every draft proposed; the plans force the outcomes.
    model.draft_min_prob = 0.0;
    let greedy = SamplingParams::greedy();
    let prompt: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    let n = 40;
    let mut sc = LanguageModel::new_scratch_with_capacity(&model, &ctx, 0).unwrap();
    let prefill = |model: &Qwen36Model,
                   st: &mut Qwen36State,
                   sc: &mut Scratch,
                   p: &SamplingParams|
     -> u32 {
        st.reset();
        sc.begin_request();
        let d = Draw { params: p, step: 0 };
        LanguageModel::prefill(model, &ctx, st, sc, &prompt, Some(d)).unwrap();
        sc.next_token.to_u32().unwrap()[0]
    };
    // Keep, reject, suppress, keep, ...; a rejection at step 5, then rest.
    let plan =
        |s: usize| [Force::Keep, Force::Reject, Force::Suppress, Force::Keep][s % 4];

    // Greedy: plain, then the forced cycle with rests after a rejection
    // (step 5) and after a suppression (step 10).
    let mut st = LanguageModel::new_state(&model, &ctx, 512).unwrap();
    let first = prefill(&model, &mut st, &mut sc, &greedy);
    let plain = plain_sampled(&model, &ctx, &mut st, &mut sc, first, n, &greedy);
    let first = prefill(&model, &mut st, &mut sc, &greedy);
    let f = speculate_forced(
        &model,
        &ctx,
        &mut st,
        &mut sc,
        first,
        n,
        &greedy,
        &plan,
        &[5, 10],
        false,
    );
    let k = f.tokens.len().min(plain.len());
    assert_eq!(&f.tokens[..k], &plain[..k], "forced greedy speculation");
    assert!(
        f.accepted >= 2 && f.rejected >= 2 && f.suppressed >= 2,
        "every outcome forced: {} accepted, {} rejected, {} suppressed",
        f.accepted,
        f.rejected,
        f.suppressed
    );

    // A rejection, then rest: snapshot, go on, restore, go on; a fork. The
    // proposals after the restore and in the fork are the first run's.
    let first = prefill(&model, &mut st, &mut sc, &greedy);
    let head = speculate_forced(
        &model,
        &ctx,
        &mut st,
        &mut sc,
        first,
        n,
        &greedy,
        &|s| if s == 2 { Force::Reject } else { Force::Keep },
        &[2],
        true,
    );
    assert_eq!(head.last, Some(Force::Reject), "{head:?}");
    let snap = st.snapshot(&ctx).unwrap();
    let at_rest = st.pos;
    let last = *head.tokens.last().unwrap();
    let go = |st: &mut Qwen36State, sc: &mut Scratch| {
        let keep = |_: usize| Force::Keep;
        speculate_forced(&model, &ctx, st, sc, last, 16, &greedy, &keep, &[], false)
    };
    let c1 = go(&mut st, &mut sc);
    st.restore(&ctx, &snap).unwrap();
    let c2 = go(&mut st, &mut sc);
    assert_eq!(c1.proposals, c2.proposals, "proposals after the restore");
    assert_eq!(c1.tokens, c2.tokens, "tokens after the restore");
    let mut fork = LanguageModel::new_state(&model, &ctx, 128).unwrap();
    fork.copy_prefix_from(&ctx, &st, at_rest).unwrap();
    fork.restore(&ctx, &snap).unwrap();
    let c3 = go(&mut fork, &mut sc);
    assert_eq!(c1.proposals, c3.proposals, "proposals in the fork");
    assert_eq!(c1.tokens, c3.tokens, "tokens in the fork");

    // The gate at its threshold: the head's probability p of a draft
    // proposes it at a threshold of p and below, not just above.
    let first = prefill(&model, &mut st, &mut sc, &greedy);
    let snap = st.snapshot(&ctx).unwrap();
    let d0 = model.draft_initial(&ctx, &mut st, &mut sc, first, 1, &greedy, 1).unwrap();
    assert_eq!(d0.len(), 1);
    let p = sc.mtp_prob.to_f32().unwrap()[0];
    for (threshold, proposed) in
        [(p * 0.5, true), (p, true), (f32::from_bits(p.to_bits() + 1), false)]
    {
        st.restore(&ctx, &snap).unwrap();
        model.draft_min_prob = threshold;
        let d =
            model.draft_initial(&ctx, &mut st, &mut sc, first, 1, &greedy, 1).unwrap();
        assert_eq!(d.is_empty(), !proposed, "threshold {threshold} against p {p}");
        if proposed {
            assert_eq!(d, d0, "the same draft at threshold {threshold}");
        }
    }
    model.draft_min_prob = 0.0;

    // Sampled with penalties: the forced cycle draws what plain sampling
    // with the same seed draws (discarded draws leave no counts behind).
    let sampled = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.0,
        presence_penalty: 0.3,
        frequency_penalty: 0.2,
        repetition_penalty: 1.1,
        seed: 7,
    };
    let first = prefill(&model, &mut st, &mut sc, &sampled);
    let plain = plain_sampled(&model, &ctx, &mut st, &mut sc, first, n, &sampled);
    let first2 = prefill(&model, &mut st, &mut sc, &sampled);
    assert_eq!(first, first2, "the prefill's sampled draw repeats");
    let f = speculate_forced(
        &model,
        &ctx,
        &mut st,
        &mut sc,
        first,
        n,
        &sampled,
        &plan,
        &[6],
        false,
    );
    let k = f.tokens.len().min(plain.len());
    assert_eq!(&f.tokens[..k], &plain[..k], "forced sampled speculation");
    assert!(f.rejected >= 1 && f.suppressed >= 1);
}

/// Stop after every possible prefix of a four-row verify, including the two
/// partial accepts which must replay only the accepted GDN projections.
#[test]
#[ignore = "requires LILY_AGNES_GGUF with its MTP head"]
fn dense_wide_verify_restores_each_accepted_prefix() {
    let path = std::env::var("LILY_AGNES_GGUF").expect("LILY_AGNES_GGUF");
    let ctx = MetalContext::new().unwrap();
    let options = LoadOptions { mtp_drafts: 3, ..Default::default() };
    let model =
        <Qwen36Model as LanguageModel>::load(&ctx, path.as_ref(), &options).unwrap();
    assert_eq!(model.max_drafts(), 3);
    let mut scratch = model.new_scratch(&ctx).unwrap();
    let mut state = model.new_state(&ctx, 32).unwrap();
    let prompt = [100, 111, 120, 103, 113, 110, 101, 112];
    let greedy = SamplingParams::greedy();
    let prefill = |state: &mut Qwen36State, sc: &mut Scratch| {
        state.reset();
        let draw = Draw { params: &greedy, step: 0 };
        LanguageModel::prefill(&model, &ctx, state, sc, &prompt, Some(draw)).unwrap();
        sc.next_token.to_u32().unwrap()[0]
    };
    let mut plain = vec![prefill(&mut state, &mut scratch)];
    for _ in 0..9 {
        model
            .forward(
                &ctx,
                &mut state,
                &mut scratch,
                &[*plain.last().unwrap()],
                Logits::Last,
            )
            .unwrap();
        plain.push(argmax(&scratch.logits_last.to_f32().unwrap()));
    }
    for accepted in 0..=3 {
        let pending = prefill(&mut state, &mut scratch);
        model
            .draft_initial(&ctx, &mut state, &mut scratch, pending, 3, &greedy, 1)
            .unwrap();
        let mut drafts = plain[1..4].to_vec();
        if accepted < 3 {
            drafts[accepted] = (drafts[accepted] + 1) % model.cfg.vocab as u32;
        }
        let (draws, pass) = model
            .verify(&ctx, &mut state, &mut scratch, pending, &drafts, &greedy, 1, 3)
            .unwrap();
        assert_eq!(&draws[..=accepted], &plain[1..=accepted + 1]);
        model
            .finish_speculation(&ctx, &mut state, &mut scratch, accepted, false, pass)
            .unwrap();
        assert_eq!(state.pos, prompt.len() + accepted + 1);
        for pair in plain[accepted + 1..].windows(2) {
            model
                .forward(&ctx, &mut state, &mut scratch, &pair[..1], Logits::Last)
                .unwrap();
            assert_eq!(
                argmax(&scratch.logits_last.to_f32().unwrap()),
                pair[1],
                "after keeping {accepted} drafts"
            );
        }
    }
}
