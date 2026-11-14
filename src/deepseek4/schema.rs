//! The tensors DeepSeek V4 Flash's kernels read, with the shape and the
//! block types each may have: [`validate`] checks a file's tensor table
//! against it before anything is mapped. The kernels index weights by raw
//! GPU address from the configured shape, so a tensor that is shorter than
//! its role, or of a type a kernel does not decode, must be refused here.

use anyhow::{Result, ensure};

use super::config::Config;
use crate::gguf::{GgmlType, Gguf};

/// The quantized projections (attention, shared experts, output head).
const Q8_0: &[GgmlType] = &[GgmlType::Q8_0];
/// The small projections the published file keeps in F16 (HC controls,
/// compressors, indexer, router, embedding).
const F16: &[GgmlType] = &[GgmlType::F16];
/// Types the routed-expert kernels (per-pair and grouped) decode.
const EXPERT: &[GgmlType] = &[GgmlType::Iq2Xxs, GgmlType::Q2K];
const F32: &[GgmlType] = &[GgmlType::F32];
const I32: &[GgmlType] = &[GgmlType::I32];

/// Tensor offsets the kernels need: 16-byte vector loads of rows.
pub const ALIGNMENT: u64 = 32;

/// One required tensor.
#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    /// ggml order.
    pub dims: Vec<u64>,
    pub types: &'static [GgmlType],
}

fn spec(name: String, dims: &[usize], types: &'static [GgmlType]) -> Spec {
    Spec { name, dims: dims.iter().map(|&d| d as u64).collect(), types }
}

/// Every tensor the model reads.
pub fn expected(c: &Config) -> Vec<Spec> {
    let h = c.hidden;
    let mut v = vec![
        spec("token_embd.weight".into(), &[h, c.vocab], F16),
        spec("output.weight".into(), &[h, c.vocab], Q8_0),
        spec("output_norm.weight".into(), &[h], F32),
        spec("output_hc_fn.weight".into(), &[c.hc_dim(), c.n_hc], F16),
        spec("output_hc_scale.weight".into(), &[1], F32),
        spec("output_hc_base.weight".into(), &[c.n_hc], F32),
    ];
    let group_in = c.o_group_heads() * c.head_dim;
    for il in 0..c.n_layer {
        let n = |s: &str| format!("blk.{il}.{s}");
        v.push(spec(n("attn_norm.weight"), &[h], F32));
        v.push(spec(n("ffn_norm.weight"), &[h], F32));
        for p in ["attn", "ffn"] {
            v.push(spec(
                n(&format!("hc_{p}_fn.weight")),
                &[c.hc_dim(), c.hc_mix()],
                F16,
            ));
            v.push(spec(n(&format!("hc_{p}_scale.weight")), &[3], F32));
            v.push(spec(n(&format!("hc_{p}_base.weight")), &[c.hc_mix()], F32));
        }
        v.push(spec(n("attn_q_a.weight"), &[h, c.q_lora], Q8_0));
        v.push(spec(n("attn_q_a_norm.weight"), &[c.q_lora], F32));
        v.push(spec(n("attn_q_b.weight"), &[c.q_lora, c.q_dim()], Q8_0));
        v.push(spec(n("attn_kv.weight"), &[h, c.head_dim], Q8_0));
        v.push(spec(n("attn_kv_a_norm.weight"), &[c.head_dim], F32));
        v.push(spec(n("attn_sinks.weight"), &[c.n_head], F32));
        v.push(spec(
            n("attn_output_a.weight"),
            &[group_in, c.o_groups * c.o_lora],
            Q8_0,
        ));
        v.push(spec(n("attn_output_b.weight"), &[c.o_groups * c.o_lora, h], Q8_0));
        let r = c.ratio(il);
        if r != 0 {
            let w = c.compress_lanes(il) * c.head_dim;
            v.push(spec(n("attn_compressor_kv.weight"), &[h, w], F16));
            v.push(spec(n("attn_compressor_gate.weight"), &[h, w], F16));
            v.push(spec(n("attn_compressor_ape.weight"), &[w, r], F16));
            v.push(spec(n("attn_compressor_norm.weight"), &[c.head_dim], F32));
        }
        if c.has_indexer(il) {
            let w = c.compress_lanes(il) * c.indexer_dim;
            v.push(spec(n("indexer_compressor_kv.weight"), &[h, w], F16));
            v.push(spec(n("indexer_compressor_gate.weight"), &[h, w], F16));
            v.push(spec(n("indexer_compressor_ape.weight"), &[w, r], F16));
            v.push(spec(n("indexer_compressor_norm.weight"), &[c.indexer_dim], F32));
            v.push(spec(
                n("indexer.attn_q_b.weight"),
                &[c.q_lora, c.indexer_heads * c.indexer_dim],
                F16,
            ));
            v.push(spec(n("indexer.proj.weight"), &[h, c.indexer_heads], F16));
        }
        v.push(spec(n("ffn_gate_inp.weight"), &[h, c.n_expert], F16));
        if c.is_hash_layer(il) {
            v.push(spec(
                n("ffn_gate_tid2eid.weight"),
                &[c.n_expert_used, c.vocab],
                I32,
            ));
        } else {
            v.push(spec(n("exp_probs_b.bias"), &[c.n_expert], F32));
        }
        for t in ["ffn_gate_exps", "ffn_up_exps"] {
            v.push(spec(
                n(&format!("{t}.weight")),
                &[h, c.ff_expert, c.n_expert],
                EXPERT,
            ));
        }
        v.push(spec(n("ffn_down_exps.weight"), &[c.ff_expert, h, c.n_expert], EXPERT));
        v.push(spec(n("ffn_gate_shexp.weight"), &[h, c.ff_expert], Q8_0));
        v.push(spec(n("ffn_up_shexp.weight"), &[h, c.ff_expert], Q8_0));
        v.push(spec(n("ffn_down_shexp.weight"), &[c.ff_expert, h], Q8_0));
    }
    v
}

/// The tensors each layer reads in one fused dispatch take one block type
/// for all of its weights.
pub fn fused_groups(c: &Config) -> Vec<Vec<String>> {
    (0..c.n_layer)
        .map(|il| {
            ["ffn_gate_exps", "ffn_up_exps"]
                .iter()
                .map(|s| format!("blk.{il}.{s}.weight"))
                .collect()
        })
        .collect()
}

/// Checks every tensor the model reads: present, of the expected rank and
/// dimensions, of a type its kernels decode, at an aligned offset; that the
/// tensors of each fused group share a type; and that the file has no
/// tensor the model would ignore (an unknown tensor means a different graph).
pub fn validate(g: &Gguf, c: &Config) -> Result<()> {
    let specs = expected(c);
    for s in &specs {
        let t = g.require(&s.name)?;
        ensure!(
            t.dims == s.dims,
            "tensor {} has dimensions {:?}, DeepSeek V4 Flash needs {:?}",
            s.name,
            t.dims,
            s.dims
        );
        ensure!(
            s.types.contains(&t.ty),
            "tensor {} is {}, lily's kernels read it as one of {:?}",
            s.name,
            t.ty.name(),
            s.types.iter().map(|t| t.name()).collect::<Vec<_>>()
        );
        ensure!(
            t.offset.is_multiple_of(ALIGNMENT),
            "tensor {} is at offset {}, not a multiple of {ALIGNMENT}",
            s.name,
            t.offset
        );
    }
    ensure!(
        g.tensors.len() == specs.len(),
        "the file has {} tensors, DeepSeek V4 Flash's graph reads {}",
        g.tensors.len(),
        specs.len()
    );
    for names in fused_groups(c) {
        let ty = g.require(&names[0])?.ty;
        for n in &names[1..] {
            ensure!(
                g.require(n)?.ty == ty,
                "tensors {} are read together but of different types",
                names.join(", ")
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/deepseek4/schema.rs"]
mod tests;
