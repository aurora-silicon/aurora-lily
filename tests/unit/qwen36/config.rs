//! Qwen3.6's metadata parsing and shape validation: the published header,
//! and malformed or unsupported variants of it, which must be errors (never
//! panics or allocations sized from them).

use std::collections::BTreeMap;

use super::*;

const VOCAB: u64 = 248320;

/// The metadata of ggml-org's Qwen3.6-35B-A3B GGUFs (the keys lily reads).
fn published() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        m.insert(k.to_string(), v);
    };
    put("general.architecture", Value::Str("qwen35moe".into()));
    for (k, v) in [
        ("block_count", 40u32),
        ("context_length", 262144),
        ("embedding_length", 2048),
        ("attention.head_count", 16),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("rope.dimension_count", 64),
        ("expert_count", 256),
        ("expert_used_count", 8),
        ("expert_feed_forward_length", 512),
        ("expert_shared_feed_forward_length", 512),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 128),
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 32),
        ("ssm.inner_size", 4096),
        ("full_attention_interval", 4),
    ] {
        put(&format!("qwen35moe.{k}"), Value::U32(v));
    }
    put("qwen35moe.rope.freq_base", Value::F32(1e7));
    put("qwen35moe.attention.layer_norm_rms_epsilon", Value::F32(1e-6));
    put(
        "qwen35moe.attention.recurrent_layers",
        Value::Array((0..40).map(|il| Value::Bool(il % 4 != 3)).collect()),
    );
    put("tokenizer.ggml.eos_token_id", Value::U32(248046));
    put("tokenizer.ggml.padding_token_id", Value::U32(248044));
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
    let want = Config::qwen36_35b_a3b();
    assert_eq!(format!("{c:?}"), format!("{want:?}"));
    // Without the layer list, the interval gives the same layers.
    let c =
        Config::from_metadata(&without("qwen35moe.attention.recurrent_layers"), VOCAB)
            .unwrap();
    assert_eq!(c.recurrent, want.recurrent);
    // The MTP file's list has one more (non-recurrent) entry: taken.
    let mut l: Vec<Value> = (0..40).map(|il| Value::Bool(il % 4 != 3)).collect();
    l.push(Value::Bool(false));
    let m = with("qwen35moe.attention.recurrent_layers", Value::Array(l));
    assert!(Config::from_metadata(&m, VOCAB).is_ok());
}

#[test]
fn layer_counts_are_bounded_before_anything_is_sized() {
    let mut m = without("qwen35moe.attention.recurrent_layers");
    m.insert("qwen35moe.block_count".into(), Value::U64(u64::MAX));
    assert!(err(&m, VOCAB).contains("block_count"), "{}", err(&m, VOCAB));
    m.insert("qwen35moe.block_count".into(), Value::U64(1 << 20));
    assert!(err(&m, VOCAB).contains("at most"));
    m.insert("qwen35moe.block_count".into(), Value::U32(0));
    assert!(err(&m, VOCAB).contains("block_count is 0"));
    // A list shorter than the layers.
    let m = with(
        "qwen35moe.attention.recurrent_layers",
        Value::Array(vec![Value::Bool(true); 39]),
    );
    assert!(err(&m, VOCAB).contains("39 entries"));
}

#[test]
fn wrong_typed_metadata_is_an_error() {
    let m = with("qwen35moe.attention.recurrent_layers", Value::Str("all".into()));
    assert!(err(&m, VOCAB).contains("recurrent_layers"));
    let m = with(
        "qwen35moe.attention.recurrent_layers",
        Value::Array(vec![Value::U32(1); 40]),
    );
    assert!(err(&m, VOCAB).contains("entry"));
    let m = with("qwen35moe.block_count", Value::Str("40".into()));
    assert!(err(&m, VOCAB).contains("not a non-negative integer"));
    let m = with("qwen35moe.expert_count", Value::I32(-1));
    assert!(err(&m, VOCAB).contains("expert_count"));
    let m = with("qwen35moe.rope.freq_base", Value::U32(10));
    assert!(err(&m, VOCAB).contains("not a float"));
    assert!(err(&without("qwen35moe.ssm.state_size"), VOCAB).contains("missing"));
}

#[test]
fn shapes_the_kernels_do_not_take_are_refused() {
    let cases: [(&str, Value, &str); 9] = [
        ("qwen35moe.expert_feed_forward_length", Value::U32(768), "256 or 512"),
        ("qwen35moe.expert_count", Value::U32(4), "used"),
        ("qwen35moe.expert_used_count", Value::U32(3), "used"),
        ("qwen35moe.expert_count", Value::U32(512), "used"),
        ("qwen35moe.attention.head_count_kv", Value::U32(0), "KV heads"),
        ("qwen35moe.attention.head_count", Value::U32(15), "KV heads"),
        ("qwen35moe.expert_shared_feed_forward_length", Value::U32(768), "shared"),
        ("qwen35moe.rope.freq_base", Value::F32(f32::NAN), "freq_base"),
        ("qwen35moe.attention.layer_norm_rms_epsilon", Value::F32(0.0), "epsilon"),
    ];
    for (k, v, want) in cases {
        let e = err(&with(k, v), VOCAB);
        assert!(e.contains(want), "{k}: {e}");
    }
    assert!(err(&published(), MAX_VOCAB as u64 + 1).contains("vocabulary"));
    assert!(err(&published(), 1000).contains("outside the vocabulary"));
}

#[test]
fn semantics_the_graph_lacks_are_refused() {
    let e = err(&with("qwen35moe.rope.scaling.type", Value::Str("yarn".into())), VOCAB);
    assert!(e.contains("rope.scaling.type"), "{e}");
    let e = err(&with("qwen35moe.rope.scaling.factor", Value::F32(4.0)), VOCAB);
    assert!(e.contains("rope.scaling.factor"), "{e}");
    let e = err(&with("qwen35moe.attention.scale", Value::F32(0.1)), VOCAB);
    assert!(e.contains("attention.scale"), "{e}");
    let e = err(&with("qwen35moe.expert_weights_scale", Value::F32(2.5)), VOCAB);
    assert!(e.contains("expert_weights_scale"), "{e}");
    let e = err(&with("qwen35moe.expert_gating_func", Value::U32(2)), VOCAB);
    assert!(e.contains("softmax"), "{e}");
    // Their neutral values pass.
    for (k, v) in [
        ("qwen35moe.rope.scaling.type", Value::Str("none".into())),
        ("qwen35moe.rope.scaling.factor", Value::F32(1.0)),
        ("qwen35moe.attention.scale", Value::F32(0.0)),
        ("qwen35moe.expert_weights_scale", Value::F32(1.0)),
        ("qwen35moe.expert_gating_func", Value::U32(1)),
    ] {
        assert!(Config::from_metadata(&with(k, v), VOCAB).is_ok(), "{k}");
    }
}

/// The metadata `tools/convert/convert_agnes3_flash.py` writes for Agnes 3.0
/// Flash (`qwen35`, the dense architecture; the keys lily reads).
fn dense() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        m.insert(k.to_string(), v);
    };
    put("general.architecture", Value::Str("qwen35".into()));
    for (k, v) in [
        ("block_count", 64u32),
        ("context_length", 262144),
        ("embedding_length", 5120),
        ("feed_forward_length", 17408),
        ("attention.head_count", 24),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("rope.dimension_count", 64),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 128),
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 48),
        ("ssm.inner_size", 6144),
        ("full_attention_interval", 4),
    ] {
        put(&format!("qwen35.{k}"), Value::U32(v));
    }
    put("qwen35.rope.freq_base", Value::F32(1e7));
    put("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6));
    put(
        "qwen35.attention.recurrent_layers",
        Value::Array((0..64).map(|il| Value::Bool(il % 4 != 3)).collect()),
    );
    put("tokenizer.ggml.eos_token_id", Value::U32(248046));
    put("tokenizer.ggml.padding_token_id", Value::U32(248044));
    m
}

#[test]
fn the_dense_header_parses() {
    let c = Config::from_metadata(&dense(), VOCAB).unwrap();
    assert_eq!(format!("{c:?}"), format!("{:?}", Config::agnes3_flash()));
    assert!(!c.is_moe() && Config::qwen36_35b_a3b().is_moe());
    assert!(is_arch("qwen35") && is_arch("qwen35moe") && !is_arch("qwen3"));
}

#[test]
fn dense_shapes_and_mixtures_are_refused() {
    let dense_with = |k: &str, v: Value| {
        let mut m = dense();
        m.insert(k.to_string(), v);
        m
    };
    // The MLP width: whole Q4_K blocks, 1024 and up.
    for ff in [1000u32, 512] {
        let e = err(&dense_with("qwen35.feed_forward_length", Value::U32(ff)), VOCAB);
        assert!(e.contains("feed_forward_length"), "{ff}: {e}");
    }
    let mut m = dense();
    m.remove("qwen35.feed_forward_length");
    assert!(err(&m, VOCAB).contains("feed_forward_length"));
    // Expert metadata in a dense file is a mixture lily does not guess at.
    let e = err(&dense_with("qwen35.expert_count", Value::U32(8)), VOCAB);
    assert!(e.contains("expert_count"), "{e}");
    let e = err(&dense_with("general.architecture", Value::Str("qwen3".into())), VOCAB);
    assert!(e.contains("architecture"), "{e}");
}
