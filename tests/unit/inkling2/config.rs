use std::collections::BTreeMap;

use super::*;
use crate::gguf::Value;

fn base_meta(arch: &str, n_arr: usize) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert("general.architecture".into(), Value::Str(arch.into()));
    let mut put = |k: &str, v: Value| {
        m.insert(format!("{arch}.{k}"), v);
    };
    put("block_count", Value::U32(n_arr as u32));
    put("context_length", Value::U32(1_048_576));
    put("embedding_length", Value::U32(2304));
    put("vocab_size", Value::U32(201_024));
    put("vocab_size_unpadded", Value::U32(200_058));
    put("attention.head_count", Value::Array(vec![Value::U32(24); n_arr]));
    put("attention.head_count_kv", Value::Array(vec![Value::U32(8); n_arr]));
    put("attention.key_length", Value::U32(128));
    put("attention.value_length", Value::U32(128));
    put("attention.layer_norm_rms_epsilon", Value::F32(1e-6));
    put("attention.sliding_window", Value::U32(1024));
    put(
        "attention.sliding_window_pattern",
        Value::Array((0..n_arr).map(|i| Value::Bool(i % 6 != 5)).collect()),
    );
    put("attention.relative_length", Value::U32(16));
    put("attention.relative_extent", Value::U32(1024));
    put("attention.qk_clip", Value::F32(22.627_417));
    put("attention.relative_clip", Value::F32(8.0));
    put("attention.log_scale_floor", Value::U32(8192));
    put("attention.log_scale_alpha", Value::F32(0.1));
    put("shortconv.l_cache", Value::U32(4));
    put("attn_res.block_size", Value::U32(6));
    put("expert_count", Value::U32(128));
    put("expert_used_count", Value::U32(8));
    put("expert_feed_forward_length", Value::U32(1152));
    put("expert_weights_scale", Value::F32(8.0));
    put("activation.situ_beta", Value::F32(4.0));
    put("activation.situ_linear_beta", Value::F32(25.0));
    put("final_logit_softcapping", Value::F32(30.0));
    m
}

#[test]
fn mainline_loop_metadata_parses_and_schedules_48_visits() {
    let mut m = base_meta(ARCH_LOOP, 24);
    m.insert(format!("{ARCH_LOOP}.loop.start"), Value::U32(6));
    m.insert(format!("{ARCH_LOOP}.loop.end"), Value::U32(18));
    m.insert(format!("{ARCH_LOOP}.loop.count"), Value::U32(3));
    m.insert(format!("{ARCH_LOOP}.loop.gate_rank"), Value::U32(64));
    m.insert(format!("{ARCH_LOOP}.loop.varying"), Value::Bool(false));

    let cfg = Config::from_metadata(&m).unwrap();
    assert_eq!(cfg.loop_layout, LoopLayout::Mainline);
    assert_eq!(cfg.n_phys, 24);
    assert_eq!(cfg.n_visits, 48);
    let sched = cfg.schedule();
    assert_eq!(sched.len(), 48);
    // Prelude: 0..6
    assert_eq!(sched[0], (0, 0, 0));
    assert_eq!(sched[5], (5, 5, 0));
    // Core pass 0: 6..18 -> slots 6..18
    assert_eq!(sched[6], (6, 6, 0));
    assert_eq!(sched[17], (17, 17, 0));
    // Core pass 1: 6..18 -> slots 24..36
    assert_eq!(sched[18], (24, 6, 1));
    assert_eq!(sched[29], (35, 17, 1));
    // Core pass 2: 6..18 -> slots 36..48
    assert_eq!(sched[30], (36, 6, 2));
    assert_eq!(sched[41], (47, 17, 2));
    // Coda: 18..24 -> slots 18..24
    assert_eq!(sched[42], (18, 18, 0));
    assert_eq!(sched[47], (23, 23, 0));
    for &(slot, phys, _) in &sched {
        assert_eq!(cfg.phys_of_slot(slot), phys);
    }
}

#[test]
fn unsloth_root_loop_metadata_parses_identically() {
    let mut m = base_meta(ARCH_PLAIN, 48);
    m.insert(format!("{ARCH_PLAIN}.loop.core_start"), Value::U32(6));
    m.insert(format!("{ARCH_PLAIN}.loop.core_end"), Value::U32(18));
    m.insert(format!("{ARCH_PLAIN}.num_loops"), Value::U32(3));
    m.insert(format!("{ARCH_PLAIN}.loop.gate_rank"), Value::U32(64));

    let cfg = Config::from_metadata(&m).unwrap();
    assert_eq!(cfg.loop_layout, LoopLayout::Unsloth);
    assert_eq!(cfg.n_phys, 24);
    assert_eq!(cfg.n_visits, 48);
    assert_eq!(cfg.schedule().len(), 48);
}

#[test]
fn varying_loop_is_refused() {
    let mut m = base_meta(ARCH_LOOP, 24);
    m.insert(format!("{ARCH_LOOP}.loop.start"), Value::U32(6));
    m.insert(format!("{ARCH_LOOP}.loop.end"), Value::U32(18));
    m.insert(format!("{ARCH_LOOP}.loop.count"), Value::U32(3));
    m.insert(format!("{ARCH_LOOP}.loop.gate_rank"), Value::U32(64));
    m.insert(format!("{ARCH_LOOP}.loop.varying"), Value::Bool(true));
    assert!(Config::from_metadata(&m).is_err());
}

fn hf_config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "inkling2_loop",
        "num_hidden_layers": 24,
        "hidden_size": 2304,
        "vocab_size": 201024,
        "unpadded_vocab_size": 200058,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "intermediate_size": 1152,
        "loop_config": {
            "core_start": 6,
            "core_end": 18,
            "num_passes": 3,
            "gate_rank": 64,
            "is_varloop": false
        }
    })
}

fn load_hf(val: &serde_json::Value, tag: &str) -> Result<Config> {
    let dir = std::env::temp_dir()
        .join(format!("lily-inkling2-hf-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("config.json"), val.to_string()).expect("write");
    let res = Config::from_hf_dir(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    res
}

#[test]
fn hf_dir_enforces_shared_config_invariants() {
    let base = hf_config_json();
    let cfg = load_hf(&base, "valid").unwrap();
    assert_eq!((cfg.n_phys, cfg.n_visits), (24, 48));

    let bad_cases: [(&str, Box<dyn Fn(&mut serde_json::Value)>); 8] = [
        (
            "zero-kv-heads",
            Box::new(|v| {
                v["num_key_value_heads"] = 0.into();
            }),
        ),
        (
            "unaligned-hidden",
            Box::new(|v| {
                v["hidden_size"] = 2300.into();
            }),
        ),
        (
            "unaligned-head-dim",
            Box::new(|v| {
                v["head_dim"] = 100.into();
                v["swa_head_dim"] = 100.into();
            }),
        ),
        (
            "zero-experts-used",
            Box::new(|v| {
                v["num_experts_per_tok"] = 0.into();
            }),
        ),
        (
            "zero-rms-eps",
            Box::new(|v| {
                v["rms_norm_eps"] = 0.0.into();
            }),
        ),
        (
            "huge-loop-passes",
            Box::new(|v| {
                v["loop_config"]["num_passes"] = 1_000_000.into();
            }),
        ),
        (
            "huge-gate-rank",
            Box::new(|v| {
                v["loop_config"]["gate_rank"] = 4096.into();
            }),
        ),
        (
            "misaligned-loop",
            Box::new(|v| {
                v["loop_config"]["core_start"] = 5.into();
            }),
        ),
    ];
    for (tag, mutate) in bad_cases {
        let mut v = base.clone();
        mutate(&mut v);
        assert!(load_hf(&v, tag).is_err(), "expected error for {tag}");
    }
}

#[test]
fn dspark_mtp_json_enforces_validation() {
    use super::super::dspark::DsparkConfig;

    let target = load_hf(&hf_config_json(), "dspark-target").unwrap();
    let base = serde_json::json!({
        "num_hidden_layers": 3,
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "block_size": 8,
        "markov_rank": 256,
        "target_layer_ids": [5, 23, 41, 48]
    });
    assert!(DsparkConfig::from_mtp_json(&base, &target).is_ok());

    let bad_cases: [(&str, Box<dyn Fn(&mut serde_json::Value)>); 7] = [
        (
            "wrong-hidden",
            Box::new(|v| {
                v["hidden_size"] = 1024.into();
            }),
        ),
        (
            "bad-block-size",
            Box::new(|v| {
                v["block_size"] = 1.into();
            }),
        ),
        (
            "empty-targets",
            Box::new(|v| {
                v["target_layer_ids"] = serde_json::json!([]);
            }),
        ),
        (
            "oob-target",
            Box::new(|v| {
                v["target_layer_ids"] = serde_json::json!([49]);
            }),
        ),
        (
            "zero-kv-heads",
            Box::new(|v| {
                v["num_key_value_heads"] = 0.into();
            }),
        ),
        (
            "huge-n-layer",
            Box::new(|v| {
                v["num_hidden_layers"] = 1_000_000_000_000u64.into();
            }),
        ),
        (
            "bad-head-dim",
            Box::new(|v| {
                v["head_dim"] = 64.into();
            }),
        ),
    ];
    for (tag, mutate) in bad_cases {
        let mut v = base.clone();
        mutate(&mut v);
        assert!(
            DsparkConfig::from_mtp_json(&v, &target).is_err(),
            "expected DsparkConfig error for {tag}"
        );
    }
}
