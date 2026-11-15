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

#[test]
fn staged_tokens_are_checked_against_the_vocabulary() {
    let ctx = MetalContext::new().unwrap();
    let mut cfg = small();
    cfg.vocab = 1000;
    cfg.n_expert = 8;
    let sc = Scratch::new(&ctx, &cfg, 64).unwrap();
    assert!(sc.stage_tokens(&[1, 2, 999]).is_ok());
    assert!(sc.stage_tokens(&[1, 1000]).is_err());
    assert!(sc.stage_tokens(&vec![1; PREFILL_ROWS + 1]).is_err());
}
