//! Qwen3.8-Flash-Next's GGUF metadata (ds4's Q2 file): the published
//! header parses to the supported shape, and malformed or unsupported
//! variants of it are errors, never panics or allocations sized from them.

use std::collections::BTreeMap;

use super::*;

/// The keys lily reads from ds4's `Qwen3.8-Flash-Next-Q2.gguf`, with its
/// values.
fn published() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert("general.architecture".to_string(), Value::Str(ARCH.into()));
    for (k, v) in [
        ("block_count", 49u64),
        ("nextn_predict_layers", 1),
        ("context_length", 262144),
        ("embedding_length", 2560),
        ("vocab_size", 248320),
        ("attention.head_count", 24),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("attention.indexer.head_count", 4),
        ("attention.indexer.key_length", 128),
        ("attention.indexer.top_k", 2048),
        ("expert_count", 512),
        ("expert_used_count", 10),
        ("expert_feed_forward_length", 640),
        ("expert_shared_feed_forward_length", 640),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 128),
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 48),
        ("ssm.inner_size", 6144),
        ("full_attention_interval", 4),
        ("hyper_connection.count", 4),
        ("hyper_connection.low_rank", 320),
        ("rope.dimension_count", 64),
        ("ple.ngram_size", 3),
        ("ple.heads_per_ngram", 8),
        ("ple.conv_kernel", 4),
        ("ple.eos_token_id", 248044),
        ("embedding_length_per_layer_input", 160),
        ("ple.row_count", 320001536),
    ] {
        m.insert(format!("{ARCH}.{k}"), Value::U32(v as u32));
    }
    m.insert(format!("{ARCH}.rope.freq_base"), Value::F32(1e7));
    m.insert(format!("{ARCH}.attention.layer_norm_rms_epsilon"), Value::F32(1e-6));
    let arr = |v: &[u64]| Value::Array(v.iter().map(|&x| Value::U64(x)).collect());
    let ratios: Vec<u64> =
        (0..49).map(|i| if i == 48 || i % 4 == 3 { 4 } else { 0 }).collect();
    m.insert(format!("{ARCH}.attention.compress_ratios"), arr(&ratios));
    m.insert(format!("{ARCH}.ple.layers"), arr(&[1]));
    m.insert(
        format!("{ARCH}.ple.layer_multipliers"),
        arr(&[23703573157769, 20109073645365, 8052911324071]),
    );
    let sizes = [
        20000003u64,
        20000023,
        20000033,
        20000047,
        20000059,
        20000063,
        20000069,
        20000077,
        20000081,
        20000093,
        20000107,
        20000147,
        20000153,
        20000159,
        20000161,
        20000171,
    ];
    let mut offsets = vec![0u64];
    for s in &sizes[..15] {
        offsets.push(offsets.last().unwrap() + s);
    }
    m.insert(format!("{ARCH}.ple.head_vocab_sizes"), arr(&sizes));
    m.insert(format!("{ARCH}.ple.head_offsets"), arr(&offsets));
    m.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U32(248046));
    m.insert("tokenizer.ggml.padding_token_id".to_string(), Value::U32(248044));
    m
}

fn with(k: &str, v: Value) -> BTreeMap<String, Value> {
    let mut m = published();
    m.insert(k.to_string(), v);
    m
}

fn err(m: &BTreeMap<String, Value>) -> String {
    match std::panic::catch_unwind(|| config_from_metadata(m, 768)) {
        Ok(Ok(_)) => panic!("accepted"),
        Ok(Err(e)) => format!("{e:#}"),
        Err(_) => panic!("panicked"),
    }
}

fn k(name: &str) -> String {
    format!("{ARCH}.{name}")
}

#[test]
fn the_published_header_parses() {
    let c = config_from_metadata(&published(), 768).unwrap();
    assert_eq!(c.num_hidden_layers, 48);
    assert_eq!(
        c.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count(),
        12
    );
    assert_eq!(c.mtp.as_ref().map(|m| m.layers), Some(1));
    assert_eq!(c.ple.as_ref().map(|p| (p.layer, p.embed_dim)), Some((1, 2560)));
    assert_eq!(c.eos_token_id.as_vec(), [248046, 248044]);
    assert_eq!(c.gguf_down_cols, Some(768));
    assert_eq!(c.rotary_dim(), 64);
}

#[test]
fn counts_are_bounded_before_anything_is_sized() {
    let e = err(&with(&k("block_count"), Value::U64(u64::MAX)));
    assert!(e.contains("block_count"), "{e}");
    assert!(
        err(&with(&k("full_attention_interval"), Value::U32(0))).contains("interval")
    );
    assert!(err(&with(&k("nextn_predict_layers"), Value::U32(49))).contains("nextn"));
    assert!(
        err(&with(&k("context_length"), Value::U64(1 << 40)))
            .contains("context_length")
    );
    assert!(err(&with(&k("vocab_size"), Value::U64(u64::MAX))).contains("vocab_size"));
    assert!(err(&with(&k("ple.ngram_size"), Value::U32(1))).contains("n-gram"));
    assert!(
        err(&with(&k("ple.heads_per_ngram"), Value::U32(1 << 30)))
            .contains("heads_per")
    );
}

#[test]
fn wrong_types_and_bad_values_are_errors() {
    assert!(err(&with(&k("block_count"), Value::F32(49.0))).contains("block_count"));
    assert!(
        err(&with(&k("nextn_predict_layers"), Value::Str("1".into())))
            .contains("nextn")
    );
    assert!(
        err(&with(&k("rope.freq_base"), Value::F32(f32::NAN))).contains("freq_base")
    );
    assert!(
        err(&with(&k("attention.layer_norm_rms_epsilon"), Value::F32(-1e-6)))
            .contains("epsilon")
    );
    assert!(
        err(&with("tokenizer.ggml.eos_token_id", Value::U32(248320)))
            .contains("vocabulary")
    );
    assert!(
        err(&with(&k("ple.eos_token_id"), Value::U32(1 << 19))).contains("vocabulary")
    );
    assert!(
        err(&with(&k("ple.layers"), Value::Array(vec![Value::U64(60)])))
            .contains("PLE layer")
    );
    let short = Value::Array(vec![Value::U64(1), Value::U64(2)]);
    assert!(err(&with(&k("ple.layer_multipliers"), short)).contains("hashing"));
    assert!(
        err(&with("general.architecture", Value::Str("qwen35moe".into())))
            .contains("not a")
    );
}

#[test]
fn shapes_the_kernels_do_not_take_are_refused() {
    for (key, v) in [
        ("expert_used_count", 8u32),
        ("expert_count", 256),
        ("embedding_length", 4096),
        ("expert_feed_forward_length", 512),
        ("hyper_connection.low_rank", 256),
    ] {
        assert!(
            err(&with(&k(key), Value::U32(v)))
                .contains("supported Qwen3.8-Flash-Next shape"),
            "{key}"
        );
    }
}

/// Under `--memory-limit-gb` the context the fit picks is the longest whose
/// reserve fits, and the reserve covers the most a session holds on its way
/// there: every growth step (the layer being replaced lives twice, old and
/// new caches) with all the server's checkpoints.
#[test]
fn the_fit_covers_a_session_growing_to_the_fitted_context() {
    use crate::kernels::attention::{KvFormat, MAX_SEQ};
    use crate::qwen4exp::config::LayerType;
    use crate::qwen4exp::model::{
        CAPACITY_STEP, LOOKAHEAD, attn_layer_bytes, longest_context, session_bytes,
        session_peak,
    };
    let c = config_from_metadata(&published(), 768).unwrap();
    let checkpoints = crate::serve::CHECKPOINTS_PER_SESSION
        + crate::serve::DECODE_CHECKPOINTS_PER_SESSION;
    let layers =
        c.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count() + 1;
    for kv in [KvFormat::Bf16, KvFormat::Q8] {
        let layer = |capacity: usize| attn_layer_bytes(&c, kv, capacity) as u64;
        for room_gb in [0.6, 1.7, 4.2, 12.0, 40.0] {
            let room = (room_gb * 1e9) as u64;
            let context =
                longest_context(&c, true, kv, MAX_SEQ - LOOKAHEAD, room, checkpoints)
                    .unwrap();
            if context == 4096 {
                // Nothing longer fits: the load refuses unless this does.
                assert!(room_gb < 1.0, "{room_gb} GB");
                continue;
            }
            let capacity = context + LOOKAHEAD;
            assert_eq!(capacity % CAPACITY_STEP, 0, "{room_gb} GB: {context}");
            let reserve = session_peak(&c, true, kv, capacity, checkpoints).unwrap();
            assert!(reserve <= room, "{room_gb} GB: {reserve} > {room}");
            if capacity + CAPACITY_STEP <= MAX_SEQ {
                let next =
                    session_peak(&c, true, kv, capacity + CAPACITY_STEP, checkpoints)
                        .unwrap();
                assert!(next > room, "{room_gb} GB: {context} is not the longest");
            }
            // The recurrent state and the snapshots do not grow.
            let full = session_bytes(&c, true, kv, capacity, checkpoints).unwrap();
            let fixed = full - layers as u64 * layer(capacity);
            assert_eq!(
                fixed,
                session_bytes(&c, true, kv, CAPACITY_STEP, checkpoints).unwrap()
                    - layers as u64 * layer(CAPACITY_STEP)
            );
            let mut peak = full;
            for from in (CAPACITY_STEP..capacity).step_by(CAPACITY_STEP) {
                for i in 0..layers {
                    let grown = i as u64 * layer(capacity);
                    let old = (layers - i) as u64 * layer(from);
                    peak = peak.max(fixed + grown + layer(capacity) + old);
                }
            }
            assert!(
                peak <= reserve,
                "{kv:?} {room_gb} GB: growth peaks at {peak} > {reserve}"
            );
        }
    }
}

/// A GGUF file of one F32 tensor named `name` of `n` values (`fill` from
/// value 0), no metadata.
fn one_tensor_gguf(path: &Path, name: u8, n: u64, fill: impl Fn(u64) -> f32) {
    let mut f = Vec::from(&b"GGUF"[..]);
    f.extend(3u32.to_le_bytes());
    f.extend(1u64.to_le_bytes());
    f.extend(0u64.to_le_bytes());
    f.extend(1u64.to_le_bytes());
    f.push(name);
    f.extend(1u32.to_le_bytes());
    f.extend(n.to_le_bytes());
    f.extend(0u32.to_le_bytes());
    f.extend(0u64.to_le_bytes());
    f.resize(f.len().next_multiple_of(32), 0);
    for i in 0..n {
        f.extend(fill(i).to_le_bytes());
    }
    std::fs::write(path, f).unwrap();
}

/// The persisted sessions' key: the same file and n-gram checkpoint give
/// the same identity wherever they are; another header, sampled payload or
/// n-gram checkpoint gives another.
#[test]
fn the_gguf_identity_follows_the_file_and_the_ngram_checkpoint() {
    let dir = std::env::temp_dir().join(format!("lily-gguf-id-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let n = 1 << 18;
    let id = |name: &str, ngram: Option<&Path>| {
        gguf_identity(&Gguf::open(&dir.join(name)).unwrap(), ngram).unwrap()
    };
    one_tensor_gguf(&dir.join("a.gguf"), b't', n, |i| i as f32);
    std::fs::copy(dir.join("a.gguf"), dir.join("copy.gguf")).unwrap();
    one_tensor_gguf(&dir.join("name.gguf"), b'u', n, |i| i as f32);
    one_tensor_gguf(&dir.join("first.gguf"), b't', n, |i| {
        if i == 0 { 1.0 } else { i as f32 }
    });
    one_tensor_gguf(&dir.join("last.gguf"), b't', n, |i| {
        if i == n - 1 { 0.5 } else { i as f32 }
    });
    let a = id("a.gguf", None);
    assert_eq!(a.len(), crate::qwen4exp::config::CHECKPOINT_ID_HEX);
    assert_eq!(a, id("a.gguf", None));
    assert_eq!(a, id("copy.gguf", None));
    for other in ["name.gguf", "first.gguf", "last.gguf"] {
        assert_ne!(a, id(other, None), "{other}");
    }
    // The n-gram checkpoint: lily's own identity of its directory.
    for (sub, config) in
        [("ng1", "{\"a\":1}"), ("ng2", "{\"a\":2}"), ("ng1b", "{\"a\":1}")]
    {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
        std::fs::write(dir.join(sub).join("config.json"), config).unwrap();
        std::fs::write(dir.join(sub).join("model.safetensors.index.json"), "{}")
            .unwrap();
    }
    let ng1 = id("a.gguf", Some(&dir.join("ng1")));
    assert_ne!(ng1, a);
    assert_eq!(ng1, id("a.gguf", Some(&dir.join("ng1b"))));
    assert_ne!(ng1, id("a.gguf", Some(&dir.join("ng2"))));
    std::fs::remove_dir_all(&dir).ok();
}
