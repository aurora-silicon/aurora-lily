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
