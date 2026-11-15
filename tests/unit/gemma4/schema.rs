//! The tensor schema against the published files' tables: the published
//! layout passes; swapped expert axes, a flattened rank, a type the
//! kernels do not read and a missing tensor are refused.

use std::collections::BTreeMap;

use super::*;

/// A tensor table: each name's dims and block type.
type Table = BTreeMap<String, (Vec<u64>, GgmlType)>;

/// The tensor table of unsloth's UD-Q5_K_M file (dims and types as its
/// header lists them).
fn published(cfg: &Config) -> Table {
    let mut m = BTreeMap::new();
    for x in target(cfg) {
        if x.optional {
            continue;
        }
        let ty = if x.name.ends_with("ffn_gate_up_exps.weight") {
            GgmlType::Q5K
        } else {
            x.types[0]
        };
        m.insert(x.name, (x.dims, ty));
    }
    m
}

fn check(m: &Table, cfg: &Config) -> Result<()> {
    validate(&target(cfg), |n| m.get(n).cloned())
}

#[test]
fn the_published_layout_passes() {
    let cfg = Config::gemma4_26b_a4b();
    let m = published(&cfg);
    check(&m, &cfg).unwrap();
    // What the header lists for layer 0 and the global layer 5.
    assert_eq!(m["blk.0.ffn_gate_up_exps.weight"].0, vec![2816, 1408, 128]);
    assert_eq!(m["blk.0.ffn_down_exps.weight"].0, vec![704, 2816, 128]);
    assert_eq!(m["blk.0.attn_k.weight"].0, vec![2816, 2048]);
    assert_eq!(m["blk.5.attn_k.weight"].0, vec![2816, 1024]);
    assert_eq!(m["blk.5.attn_q_norm.weight"].0, vec![512]);
    assert!(!m.contains_key("blk.5.attn_v.weight"));
    assert_eq!(m["rope_freqs.weight"].0, vec![256]);
}

#[test]
fn tensors_of_another_arrangement_or_type_are_refused() {
    let cfg = Config::gemma4_26b_a4b();
    let err = |f: &dyn Fn(&mut Table)| {
        let mut m = published(&cfg);
        f(&mut m);
        format!("{:#}", check(&m, &cfg).unwrap_err())
    };
    // The same element count with the expert axes swapped, or flattened.
    let e = err(&|m| {
        m.get_mut("blk.3.ffn_gate_up_exps.weight").unwrap().0 = vec![2816, 128, 1408]
    });
    assert!(e.contains("blk.3.ffn_gate_up_exps.weight is [2816, 128, 1408]"), "{e}");
    let e = err(&|m| {
        m.get_mut("blk.0.ffn_down_exps.weight").unwrap().0 = vec![704, 2816 * 128]
    });
    assert!(e.contains("ffn_down_exps"), "{e}");
    let e = err(&|m| m.get_mut("blk.0.attn_q.weight").unwrap().0 = vec![4096, 2816]);
    assert!(e.contains("attn_q.weight"), "{e}");
    let e = err(&|m| m.get_mut("blk.1.attn_norm.weight").unwrap().0 = vec![2816, 1]);
    assert!(e.contains("attn_norm"), "{e}");
    // Types the kernels of the role do not read.
    let e =
        err(&|m| m.get_mut("blk.0.ffn_down_exps.weight").unwrap().1 = GgmlType::Q5K);
    assert!(e.contains("Q5K"), "{e}");
    let e =
        err(&|m| m.get_mut("blk.0.ffn_gate_up_exps.weight").unwrap().1 = GgmlType::Q6K);
    assert!(e.contains("Q6K"), "{e}");
    let e =
        err(&|m| m.get_mut("blk.0.ffn_gate_inp.weight").unwrap().1 = GgmlType::Q8_0);
    assert!(e.contains("ffn_gate_inp"), "{e}");
    let e = err(&|m| m.get_mut("token_embd.weight").unwrap().1 = GgmlType::Iq2Xxs);
    assert!(e.contains("token_embd"), "{e}");
    // A sliding-window layer must have its V projection; a global one may.
    let e = err(&|m| {
        m.remove("blk.0.attn_v.weight");
    });
    assert!(e.contains("no tensor blk.0.attn_v.weight"), "{e}");
    let mut m = published(&cfg);
    m.insert("blk.5.attn_v.weight".into(), (vec![2816, 1024], GgmlType::Q8_0));
    check(&m, &cfg).unwrap();
}

#[test]
fn the_draft_models_tensors_are_checked_too() {
    let cfg = Config::gemma4_26b_a4b();
    let swa = [true, true, true, false];
    let expect = draft(&cfg, 1024, 8192, &swa);
    let mut m: Table =
        expect.iter().map(|x| (x.name.clone(), (x.dims.clone(), x.types[0]))).collect();
    assert_eq!(m["nextn.pre_projection.weight"].0, vec![5632, 1024]);
    assert_eq!(m["blk.3.attn_q.weight"].0, vec![1024, 8192]);
    validate(&expect, |n| m.get(n).cloned()).unwrap();
    m.get_mut("nextn.post_projection.weight").unwrap().0 = vec![2816, 1024];
    assert!(validate(&expect, |n| m.get(n).cloned()).is_err());
}
