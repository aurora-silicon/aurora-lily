//! Gemma 4's session state against its bounds: capped cache growth, the
//! drafts a verify pass may carry at the context's end, token ids checked
//! against the vocabulary.

use super::*;
use crate::engine::DecodeStateApi;
use crate::gemma4::Config;
use crate::metal::MetalContext;

#[test]
fn the_cap_leaves_room_for_a_verify_pass() {
    assert_eq!(capacity_cap(8192), 8192 + LOOKAHEAD);
    assert_eq!(capacity_cap(MAX_SEQ), MAX_SEQ);
    assert_eq!(capacity_cap(MAX_SEQ - 2), MAX_SEQ);
}

#[test]
fn drafts_stop_short_of_the_last_rows() {
    // A verify pass from `pos` feeds rows pos ..= pos + drafts.
    assert_eq!(drafts_fit(10, 1000, 3), 3);
    assert_eq!(drafts_fit(996, 1000, 3), 3);
    assert_eq!(drafts_fit(997, 1000, 3), 2);
    assert_eq!(drafts_fit(998, 1000, 3), 1);
    assert_eq!(drafts_fit(999, 1000, 3), 0);
    assert_eq!(drafts_fit(1000, 1000, 3), 0);
}

#[test]
fn token_ids_past_the_vocabulary_are_refused() {
    assert!(check_tokens(&[0, 5, 99], 100).is_ok());
    let e = check_tokens(&[0, 100], 100).unwrap_err().to_string();
    assert!(e.contains("token id 100"), "{e}");
    assert!(check_tokens(&[u32::MAX], 262144).is_err());
}

/// A small Gemma shape (two layers, one global) for state tests.
fn small() -> Config {
    let mut c = Config::gemma4_26b_a4b();
    c.n_layer = 2;
    c.swa = vec![true, false];
    c.n_head_kv = vec![8, 2];
    c.k_rot = vec![128, 64];
    c.window = 64;
    c
}

#[test]
fn caches_grow_by_doubling_up_to_the_cap_and_no_further() {
    let ctx = MetalContext::new().unwrap();
    let cfg = small();
    let mut st = Gemma4State::new(&ctx, &cfg, 16, 100).unwrap();
    st.ensure_capacity(&ctx, 17).unwrap();
    assert_eq!(st.capacity, 32);
    st.ensure_capacity(&ctx, 80).unwrap();
    // Twice 32 is short of 80; the request itself, then the cap.
    assert_eq!(st.capacity, 80);
    st.ensure_capacity(&ctx, 81).unwrap();
    assert_eq!(st.capacity, 100);
    assert_eq!(st.layers[1].slots, 100);
    let e = st.ensure_capacity(&ctx, 101).unwrap_err().to_string();
    assert!(e.contains("exceeds the 100 rows"), "{e}");
    // A state may not start past its cap.
    assert!(Gemma4State::new(&ctx, &cfg, 101, 100).is_err());
}

/// The shape-based session size is what a state and its snapshots
/// allocate, and the growth peak bounds what growing to the cap holds.
#[test]
fn session_bytes_are_what_a_session_allocates() {
    use crate::gemma4::model::{session_bytes, session_peak};
    let ctx = MetalContext::new().unwrap();
    let cfg = small();
    let (cap, checkpoints) = (1000, 3);
    let before = ctx.current_allocated();
    let mut st = Gemma4State::new(&ctx, &cfg, 16, cap).unwrap();
    for tokens in [17, 600, cap] {
        st.ensure_capacity(&ctx, tokens).unwrap();
    }
    let snaps: Vec<_> = (0..checkpoints).map(|_| st.snapshot(&ctx).unwrap()).collect();
    let used = (ctx.current_allocated() - before) as u64;
    assert_eq!(used, session_bytes(&cfg, cap, checkpoints));
    assert_eq!(st.bytes() as u64, session_bytes(&cfg, cap, 0));
    let widest = 2 * (cfg.k_cache_width(1) + cfg.kv_width(1)) * cap;
    assert_eq!(session_peak(&cfg, cap, checkpoints), used + widest as u64);
    drop(snaps);
}

#[test]
fn staged_tokens_are_checked_against_the_vocabulary() {
    let ctx = MetalContext::new().unwrap();
    let mut cfg = small();
    cfg.vocab = 1000;
    cfg.n_expert = 8;
    let sc = Scratch::new(&ctx, &cfg, 64, 256).unwrap();
    assert!(sc.stage_tokens(&[1, 2, 999]).is_ok());
    assert!(sc.stage_tokens(&[1, 1000]).is_err());
    assert!(sc.stage_tokens(&vec![1; PREFILL_ROWS + 1]).is_err());
}

// ---------------------------------------------------------------------------
// The published file: `LILY_GEMMA4_GGUF=<UD-Q5_K_M or UD-IQ2_M path>` (the
// draft model beside it) `cargo test --release --lib gemma4::engine --
// --ignored --test-threads=1`.
// ---------------------------------------------------------------------------

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b }) as u32
}

/// Greedy speculation from a state at rest whose last draw `first` is not
/// fed, until at least `n` tokens (counting `first`) or the state's cap;
/// with `rest_every`, every so many steps it finishes without proposing (as
/// at a checkpoint) and restarts with `draft_initial`. Returns the tokens
/// and the proposals by the position of their verify pass.
fn speculate(
    model: &Gemma4Model,
    ctx: &MetalContext,
    st: &mut Gemma4State,
    sc: &mut Scratch,
    first: u32,
    n: usize,
    rest_every: Option<usize>,
) -> (Vec<u32>, Vec<(usize, Vec<u32>)>) {
    use crate::engine::LanguageModel;
    use crate::kernels::sample::SamplingParams;
    let greedy = SamplingParams::greedy();
    let drafts = model.drafts;
    let mut out = vec![first];
    let mut log = Vec::new();
    let mut proposals =
        model.draft_initial(ctx, st, sc, first, drafts, &greedy, 1).unwrap();
    let mut step = 0;
    while out.len() < n && st.pos < st.max_capacity {
        let pending = *out.last().unwrap();
        log.push((st.pos, proposals.clone()));
        let (draws, pass) = model
            .verify(ctx, st, sc, pending, &proposals, &greedy, out.len(), drafts)
            .unwrap();
        let kept = proposals.iter().zip(&draws).take_while(|(p, d)| p == d).count();
        out.extend_from_slice(&draws[..=kept]);
        step += 1;
        let rest = rest_every.is_some_and(|e| step % e == 0);
        proposals = model.finish_speculation(ctx, st, sc, kept, !rest, pass).unwrap();
        if rest {
            let last = *out.last().unwrap();
            proposals = model
                .draft_initial(ctx, st, sc, last, drafts, &greedy, out.len())
                .unwrap();
        }
    }
    (out, log)
}

#[test]
#[ignore = "requires LILY_GEMMA4_GGUF"]
fn speculation_at_rest_restores_and_forks_like_plain_decoding() {
    use crate::engine::{Draw, LanguageModel, LoadOptions, SnapshotApi};
    use crate::kernels::sample::SamplingParams;
    let path = std::env::var("LILY_GEMMA4_GGUF").expect("LILY_GEMMA4_GGUF");
    let ctx = MetalContext::new().unwrap();
    let options = LoadOptions { mtp_drafts: 3, ..Default::default() };
    let mut model = <Gemma4Model as LanguageModel>::load(
        &ctx,
        std::path::Path::new(&path),
        &options,
    )
    .unwrap();
    assert!(model.draft.is_some(), "no draft model beside {path}");
    // Every step proposes: the prompt is arbitrary tokens, whose drafts
    // rarely reach the served gate's probability, and the checks below
    // compare proposals.
    model.draft_min_prob = 0.0;
    let model = model;
    let greedy = SamplingParams::greedy();
    // A prompt past the sliding window, so the rings wrap.
    let prompt: Vec<u32> = (0..1500u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    let n = 64;
    let mut sc = LanguageModel::new_scratch_with_capacity(&model, &ctx, 2048).unwrap();
    let prefill = |st: &mut Gemma4State, sc: &mut Scratch| -> u32 {
        st.reset().unwrap();
        let d = Draw { params: &greedy, step: 0 };
        LanguageModel::prefill(&model, &ctx, st, sc, &prompt, Some(d)).unwrap();
        sc.next_token.to_u32().unwrap()[0]
    };

    // Plain greedy decoding.
    let mut st = LanguageModel::new_state(&model, &ctx, 2048).unwrap();
    let first = prefill(&mut st, &mut sc);
    let mut plain = vec![first];
    while plain.len() < n {
        let last = *plain.last().unwrap();
        model.forward(&ctx, &mut st, &sc, &[last], Logits::Last).unwrap();
        plain.push(argmax(&sc.logits_last.to_f32().unwrap()));
    }
    let agree = |t: &[u32], what: &str| {
        let k = t.len().min(plain.len());
        assert_eq!(&t[..k], &plain[..k], "{what}");
    };

    // Speculation, and speculation that comes to rest every third step.
    let first = prefill(&mut st, &mut sc);
    let (a, _) = speculate(&model, &ctx, &mut st, &mut sc, first, n, None);
    agree(&a, "speculation");
    let first = prefill(&mut st, &mut sc);
    let (b, _) = speculate(&model, &ctx, &mut st, &mut sc, first, n, Some(3));
    agree(&b, "speculation with rests");

    // At rest after 20 tokens: snapshot, go on, restore, go on again; and a
    // fork (prefix copy + snapshot) into a fresh state with a scratch of
    // its own. The draft model's input comes back with the snapshot, so
    // the proposals after the restore and in the fork are the first run's
    // (a stale input would change them, not the verified tokens).
    let first = prefill(&mut st, &mut sc);
    let (head, _) = speculate(&model, &ctx, &mut st, &mut sc, first, 20, Some(1));
    let at_rest = st.pos;
    let last = *head.last().unwrap();
    let snap = st.snapshot(&ctx).unwrap();
    assert_eq!(snap.pos(), at_rest);
    let rest = |r: Vec<u32>| [&head[..head.len() - 1], &r].concat();
    let go = |st: &mut Gemma4State, sc: &mut Scratch| {
        speculate(&model, &ctx, st, sc, last, n - head.len() + 1, None)
    };
    let (c1, p1) = go(&mut st, &mut sc);
    st.restore(&ctx, &snap).unwrap();
    let (c2, p2) = go(&mut st, &mut sc);
    let mut fork = LanguageModel::new_state(&model, &ctx, 64).unwrap();
    fork.copy_prefix_from(&ctx, &st, at_rest).unwrap();
    fork.restore(&ctx, &snap).unwrap();
    let mut sc2 = LanguageModel::new_scratch_with_capacity(&model, &ctx, 2048).unwrap();
    let (c3, p3) = go(&mut fork, &mut sc2);
    assert!(p1.iter().any(|(_, d)| !d.is_empty()), "no proposals: {p1:?}");
    assert_eq!(p1, p2, "proposals after the restore");
    assert_eq!(p1, p3, "proposals in the fork");
    agree(&rest(c1), "after the snapshot");
    agree(&rest(c2), "after the restore");
    agree(&rest(c3), "in the fork");

    // The context's last rows: a state capped 30 rows past the prompt
    // speculates to its end, drafts stopping short of it, and agrees.
    let cap = prompt.len() + 30;
    let mut st = Gemma4State::new(&ctx, &model.cfg, prompt.len(), cap).unwrap();
    let first = prefill(&mut st, &mut sc);
    let (e, log) = speculate(&model, &ctx, &mut st, &mut sc, first, n, None);
    assert_eq!(st.pos, cap, "fed to the cap");
    agree(&e, "to the end of the context");
    assert!(log.iter().all(|(p, d)| p + 1 + d.len() <= cap), "{log:?}");
    assert!(st.ensure_capacity(&ctx, cap + 1).is_err());
}
