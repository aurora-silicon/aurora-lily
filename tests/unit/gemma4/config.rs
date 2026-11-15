//! Gemma 4's metadata parsing and shape validation: the published header,
//! and malformed or unsupported variants of it, which must be errors (never
//! panics or allocations sized from them).

use std::collections::BTreeMap;

use super::*;

const VOCAB: u64 = 262144;

/// The metadata of Gemma 4 26B-A4B's GGUFs (the keys lily reads).
fn published() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        m.insert(k.to_string(), v);
    };
    put("general.architecture", Value::Str("gemma4".into()));
    for (k, v) in [
        ("block_count", 30u32),
        ("context_length", 262144),
        ("embedding_length", 2816),
        ("feed_forward_length", 2112),
        ("attention.head_count", 16),
        ("expert_count", 128),
        ("expert_used_count", 8),
        ("attention.key_length", 512),
        ("attention.value_length", 512),
        ("attention.sliding_window", 1024),
        ("attention.shared_kv_layers", 0),
        ("embedding_length_per_layer_input", 0),
        ("attention.key_length_swa", 256),
        ("attention.value_length_swa", 256),
        ("expert_feed_forward_length", 704),
        ("rope.dimension_count", 512),
        ("rope.dimension_count_swa", 256),
    ] {
        put(&format!("gemma4.{k}"), Value::U32(v));
    }
    put("gemma4.rope.freq_base", Value::F32(1e6));
    put("gemma4.rope.freq_base_swa", Value::F32(1e4));
    put("gemma4.attention.layer_norm_rms_epsilon", Value::F32(1e-6));
    put("gemma4.final_logit_softcapping", Value::F32(30.0));
    put(
        "gemma4.attention.head_count_kv",
        Value::Array(
            (0..30).map(|il| Value::U32(if il % 6 == 5 { 2 } else { 8 })).collect(),
        ),
    );
    put(
        "gemma4.attention.sliding_window_pattern",
        Value::Array((0..30).map(|il| Value::Bool(il % 6 != 5)).collect()),
    );
    put("tokenizer.ggml.bos_token_id", Value::U32(2));
    put("tokenizer.ggml.eos_token_id", Value::U32(106));
    m
}

fn err(m: &BTreeMap<String, Value>, vocab: u64) -> String {
    match Config::from_metadata(m, vocab) {
        Ok(_) => panic!("accepted"),
        Err(e) => format!("{e:#}"),
    }
}

fn with(k: &str, v: Value) -> BTreeMap<String, Value> {
    let mut m = published();
    m.insert(k.to_string(), v);
    m
}

fn without(k: &str) -> BTreeMap<String, Value> {
    let mut m = published();
    m.remove(k);
    m
}

#[test]
fn the_published_header_parses() {
    let c = Config::from_metadata(&published(), VOCAB).unwrap();
    let want = Config::gemma4_26b_a4b();
    assert_eq!(format!("{c:?}"), format!("{want:?}"));
    want.validate().unwrap();
    // One KV head count for every layer is the other form llama.cpp reads.
    let m = with("gemma4.attention.head_count_kv", Value::U32(8));
    assert_eq!(Config::from_metadata(&m, VOCAB).unwrap().n_head_kv, vec![8; 30]);
    // The softcap is optional.
    let m = without("gemma4.final_logit_softcapping");
    assert_eq!(Config::from_metadata(&m, VOCAB).unwrap().softcap, 0.0);
}

#[test]
fn layer_counts_are_bounded_before_anything_is_sized() {
    let mut m = published();
    m.insert("gemma4.block_count".into(), Value::U64(u64::MAX));
    assert!(err(&m, VOCAB).contains("block_count"), "{}", err(&m, VOCAB));
    m.insert("gemma4.block_count".into(), Value::U64(1 << 20));
    assert!(err(&m, VOCAB).contains("at most"));
    m.insert("gemma4.block_count".into(), Value::U32(0));
    assert!(err(&m, VOCAB).contains("block_count is 0"));
    // Per-layer lists of another length.
    let m = with(
        "gemma4.attention.sliding_window_pattern",
        Value::Array(vec![Value::Bool(true); 29]),
    );
    assert!(err(&m, VOCAB).contains("29 entries"));
    let m =
        with("gemma4.attention.head_count_kv", Value::Array(vec![Value::U32(8); 31]));
    assert!(err(&m, VOCAB).contains("31 entries"));
    let m = with(
        "gemma4.attention.head_count_kv",
        Value::Array(vec![Value::U64(u64::MAX); 30]),
    );
    assert!(err(&m, VOCAB).contains("at most"));
    let m = with("gemma4.attention.sliding_window", Value::U64(1 << 40));
    assert!(err(&m, VOCAB).contains("at most"));
}

#[test]
fn wrong_typed_metadata_is_an_error() {
    let m = with("gemma4.attention.sliding_window_pattern", Value::Str("all".into()));
    assert!(err(&m, VOCAB).contains("not an array"));
    let m = with(
        "gemma4.attention.sliding_window_pattern",
        Value::Array(vec![Value::U32(1); 30]),
    );
    assert!(err(&m, VOCAB).contains("sliding_window_pattern holds"));
    let m = with("gemma4.block_count", Value::Str("30".into()));
    assert!(err(&m, VOCAB).contains("not a non-negative integer"));
    let m = with("gemma4.expert_count", Value::I32(-1));
    assert!(err(&m, VOCAB).contains("expert_count"));
    let m = with("gemma4.rope.freq_base", Value::U32(10));
    assert!(err(&m, VOCAB).contains("not a float"));
    let m = with("gemma4.final_logit_softcapping", Value::Str("30".into()));
    assert!(err(&m, VOCAB).contains("not a float"));
    let m = with("tokenizer.ggml.eos_token_id", Value::U64(1 << 40));
    assert!(err(&m, VOCAB).contains("not a token id"));
    assert!(
        err(&without("gemma4.attention.key_length_swa"), VOCAB).contains("missing")
    );
    let m =
        with("tokenizer.ggml.tokens", Value::Array(vec![Value::Str("a".into()); 3]));
    assert!(err(&m, VOCAB).contains("tokenizer tokens"));
}

#[test]
fn shapes_the_kernels_do_not_take_are_refused() {
    let cases: [(&str, Value, &str); 13] = [
        ("gemma4.feed_forward_length", Value::U32(2144), "multiples of 64"),
        ("gemma4.embedding_length", Value::U32(2800), "embedding_length"),
        ("gemma4.embedding_length", Value::U32(8192), "at most"),
        ("gemma4.expert_feed_forward_length", Value::U32(736), "multiples of 64"),
        ("gemma4.feed_forward_length", Value::U32(2100), "feed_forward_length"),
        ("gemma4.expert_used_count", Value::U32(6), "used"),
        ("gemma4.expert_used_count", Value::U32(0), "used"),
        ("gemma4.expert_count", Value::U32(512), "used"),
        ("gemma4.expert_count", Value::U32(4), "used"),
        ("gemma4.attention.head_count", Value::U32(12), "KV heads"),
        ("gemma4.attention.head_count_kv", Value::U32(0), "KV heads"),
        ("gemma4.rope.freq_base", Value::F32(f32::NAN), "freq_base"),
        ("gemma4.attention.layer_norm_rms_epsilon", Value::F32(0.0), "epsilon"),
    ];
    for (k, v, want) in cases {
        let e = err(&with(k, v), VOCAB);
        assert!(e.contains(want), "{k}: {e}");
    }
    let e = err(
        &with(
            "gemma4.attention.sliding_window_pattern",
            Value::Array(vec![Value::Bool(true); 30]),
        ),
        VOCAB,
    );
    assert!(e.contains("no global layer"), "{e}");
    assert!(err(&published(), MAX_VOCAB as u64 + 1).contains("vocabulary"));
    assert!(err(&published(), 100).contains("outside the vocabulary"));
    let mut m = published();
    for k in ["attention.key_length", "attention.value_length", "rope.dimension_count"]
    {
        m.insert(format!("gemma4.{k}"), Value::U32(128));
    }
    let e = err(&m, VOCAB);
    assert!(e.contains("head widths"), "{e}");
    let e = err(&with("gemma4.attention.value_length", Value::U32(256)), VOCAB);
    assert!(e.contains("widths differ"), "{e}");
    let e = err(&with("gemma4.rope.dimension_count_swa", Value::U32(128)), VOCAB);
    assert!(e.contains("partial RoPE"), "{e}");
}

#[test]
fn semantics_the_graph_lacks_are_refused() {
    for (k, v) in [
        ("gemma4.rope.scaling.type", Value::Str("yarn".into())),
        ("gemma4.rope.scaling.factor", Value::F32(4.0)),
        ("gemma4.attention.scale", Value::F32(0.1)),
        ("gemma4.embedding_length_per_layer_input", Value::U32(256)),
        ("gemma4.attention.shared_kv_layers", Value::U32(10)),
        ("gemma4.expert_weights_scale", Value::F32(2.5)),
        ("gemma4.expert_gating_func", Value::U32(2)),
    ] {
        let e = err(&with(k, v), VOCAB);
        assert!(e.contains(k.trim_start_matches("gemma4.")), "{k}: {e}");
    }
    // Their neutral values pass.
    for (k, v) in [
        ("gemma4.rope.scaling.type", Value::Str("none".into())),
        ("gemma4.rope.scaling.factor", Value::F32(1.0)),
        ("gemma4.attention.scale", Value::F32(1.0)),
        ("gemma4.expert_weights_scale", Value::F32(1.0)),
        ("gemma4.expert_gating_func", Value::U32(1)),
    ] {
        assert!(Config::from_metadata(&with(k, v), VOCAB).is_ok(), "{k}");
    }
}
