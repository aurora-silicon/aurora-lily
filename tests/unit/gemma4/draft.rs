//! The draft model's metadata: the published header parses; widths,
//! rotary dimensions and attention kinds lily's graph does not run, and
//! wrongly typed keys, are refused before anything is mapped.

use std::collections::BTreeMap;

use super::*;
use crate::gguf::Value;

/// The metadata of Gemma 4 26B-A4B's draft model (`gemma4-assistant`).
fn published() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        m.insert(k.to_string(), v);
    };
    put("general.architecture", Value::Str(ARCH.into()));
    for (k, v) in [
        ("block_count", 4u32),
        ("embedding_length", 1024),
        ("embedding_length_out", 2816),
        ("feed_forward_length", 8192),
        ("attention.head_count", 16),
        ("attention.key_length", 512),
        ("attention.value_length", 512),
        ("attention.key_length_swa", 256),
        ("attention.value_length_swa", 256),
        ("attention.sliding_window", 1024),
        ("attention.shared_kv_layers", 4),
        ("rope.dimension_count", 512),
        ("rope.dimension_count_swa", 256),
    ] {
        put(&format!("{ARCH}.{k}"), Value::U32(v));
    }
    put(&format!("{ARCH}.rope.freq_base"), Value::F32(1e6));
    put(&format!("{ARCH}.rope.freq_base_swa"), Value::F32(1e4));
    put(&format!("{ARCH}.attention.layer_norm_rms_epsilon"), Value::F32(1e-6));
    put(
        &format!("{ARCH}.attention.sliding_window_pattern"),
        Value::Array([true, true, true, false].into_iter().map(Value::Bool).collect()),
    );
    m
}

fn with(k: &str, v: Value) -> BTreeMap<String, Value> {
    let mut m = published();
    m.insert(format!("{ARCH}.{k}"), v);
    m
}

fn err(m: &BTreeMap<String, Value>, target: &TargetConfig) -> String {
    match DraftConfig::from_metadata(m, target) {
        Ok(_) => panic!("accepted"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn the_published_draft_header_parses() {
    let target = TargetConfig::gemma4_26b_a4b();
    let c = DraftConfig::from_metadata(&published(), &target).unwrap();
    assert_eq!((c.n_layer, c.hidden, c.ff, c.heads), (4, 1024, 8192, 16));
    assert_eq!((c.head_dim, c.head_dim_swa), (512, 256));
    assert_eq!(c.swa, vec![true, true, true, false]);
    // The sliding-window rotary count is optional, as for the target.
    let mut m = published();
    m.remove(&format!("{ARCH}.rope.dimension_count_swa"));
    DraftConfig::from_metadata(&m, &target).unwrap();
}

#[test]
fn draft_metadata_lilys_graph_does_not_run_is_refused() {
    let target = TargetConfig::gemma4_26b_a4b();
    let cases: [(&str, Value, &str); 10] = [
        ("feed_forward_length", Value::U32(8224), "multiples of 64"),
        ("rope.dimension_count", Value::U32(256), "partial RoPE"),
        ("rope.dimension_count_swa", Value::U32(128), "partial RoPE"),
        ("attention.value_length", Value::U32(256), "value widths"),
        ("attention.value_length_swa", Value::U32(512), "value widths"),
        ("attention.value_length_swa", Value::Str("256".into()), "not a non-negative"),
        ("rope.dimension_count", Value::F32(512.0), "not a non-negative"),
        ("attention.key_length", Value::U32(256), "head widths"),
        ("attention.head_count", Value::U32(8), "draft heads"),
        ("rope.scaling.factor", Value::F32(8.0), "rope.scaling.factor"),
    ];
    for (k, v, want) in cases {
        let e = err(&with(k, v), &target);
        assert!(e.contains(want), "{k}: {e}");
    }
    let mut m = published();
    m.remove(&format!("{ARCH}.attention.value_length"));
    assert!(err(&m, &target).contains("missing"));
    let m = with("block_count", Value::U64(1 << 20));
    assert!(err(&m, &target).contains("at most"));
    // A target without the sliding-window layers the draft reads.
    let mut global = TargetConfig::gemma4_26b_a4b();
    global.swa = vec![false; global.n_layer];
    let e = err(&published(), &global);
    assert!(e.contains("sliding-window layer the target lacks"), "{e}");
    // The other architecture's header.
    let mut m = published();
    m.insert("general.architecture".into(), Value::Str("gemma4".into()));
    assert!(err(&m, &target).contains("architecture"));
}
