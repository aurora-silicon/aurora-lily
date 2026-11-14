//! The tensors Qwen3.6's kernels read, with the shape and the block types
//! each may have: [`validate`] checks a file's tensor table against it
//! before anything is mapped (the kernels index weights by raw GPU address
//! from the configured shape).

use anyhow::{Result, ensure};

use super::config::Config;
use crate::gguf::{GgmlType, Gguf};

/// Types the projection paths decode: the decode GEMV and the prompt GEMM.
pub(crate) const DENSE: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::Q4K];
/// Types the routed-expert kernels and the embedding decode.
const BLOCKS: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::Q4K];
const F32: &[GgmlType] = &[GgmlType::F32];

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

/// The tensors of trunk-style block `il` (also the MTP block's, `blk.40`):
/// attention of the kind `recurrent` says, the MoE.
pub fn block(c: &Config, il: usize, recurrent: bool) -> Vec<Spec> {
    let h = c.hidden;
    let n = |s: &str| format!("blk.{il}.{s}");
    let mut v = vec![
        spec(n("attn_norm.weight"), &[h], F32),
        spec(n("post_attention_norm.weight"), &[h], F32),
    ];
    if recurrent {
        let qkv = c.gdn_qkv();
        let inner = c.gdn_inner();
        v.extend([
            spec(n("attn_qkv.weight"), &[h, qkv], DENSE),
            spec(n("attn_gate.weight"), &[h, inner], DENSE),
            spec(n("ssm_alpha.weight"), &[h, c.gdn_v_heads], DENSE),
            spec(n("ssm_beta.weight"), &[h, c.gdn_v_heads], DENSE),
            spec(n("ssm_a"), &[c.gdn_v_heads], F32),
            spec(n("ssm_dt.bias"), &[c.gdn_v_heads], F32),
            spec(n("ssm_conv1d.weight"), &[c.gdn_conv, qkv], F32),
            spec(n("ssm_norm.weight"), &[c.gdn_dim], F32),
            spec(n("ssm_out.weight"), &[inner, h], DENSE),
        ]);
    } else {
        v.extend([
            spec(n("attn_q.weight"), &[h, c.q_gate()], DENSE),
            spec(n("attn_k.weight"), &[h, c.kv_width()], DENSE),
            spec(n("attn_v.weight"), &[h, c.kv_width()], DENSE),
            spec(n("attn_q_norm.weight"), &[c.head_dim], F32),
            spec(n("attn_k_norm.weight"), &[c.head_dim], F32),
            spec(n("attn_output.weight"), &[c.n_head * c.head_dim, h], DENSE),
        ]);
    }
    v.extend([
        spec(n("ffn_gate_inp.weight"), &[h, c.n_expert], F32),
        spec(n("ffn_gate_exps.weight"), &[h, c.ff_expert, c.n_expert], BLOCKS),
        spec(n("ffn_up_exps.weight"), &[h, c.ff_expert, c.n_expert], BLOCKS),
        spec(n("ffn_down_exps.weight"), &[c.ff_expert, h, c.n_expert], BLOCKS),
        spec(n("ffn_gate_inp_shexp.weight"), &[h], F32),
        spec(n("ffn_gate_shexp.weight"), &[h, c.ff_shared], DENSE),
        spec(n("ffn_up_shexp.weight"), &[h, c.ff_shared], DENSE),
        spec(n("ffn_down_shexp.weight"), &[c.ff_shared, h], DENSE),
    ]);
    v
}

/// Every tensor the trunk reads.
pub fn expected(c: &Config) -> Vec<Spec> {
    let h = c.hidden;
    let mut v = vec![
        spec("token_embd.weight".into(), &[h, c.vocab], BLOCKS),
        spec("output.weight".into(), &[h, c.vocab], DENSE),
        spec("output_norm.weight".into(), &[h], F32),
    ];
    for il in 0..c.n_layer {
        v.extend(block(c, il, c.recurrent[il]));
    }
    v
}

/// Groups of tensors read in one fused dispatch (one block type each).
pub fn fused_groups(il: usize, recurrent: bool) -> Vec<Vec<String>> {
    let group = |names: &[&str]| -> Vec<String> {
        names.iter().map(|s| format!("blk.{il}.{s}.weight")).collect()
    };
    let mut v = vec![
        group(&["ffn_gate_exps", "ffn_up_exps"]),
        group(&["ffn_gate_shexp", "ffn_up_shexp"]),
    ];
    if recurrent {
        v.push(group(&["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"]));
    } else {
        v.push(group(&["attn_k", "attn_v"]));
    }
    v
}

/// Checks `specs` and the fused groups against the file: present, of the
/// expected dimensions, of a type its kernels decode, at an aligned offset.
pub fn validate_specs(g: &Gguf, specs: &[Spec], groups: &[Vec<String>]) -> Result<()> {
    for s in specs {
        let t = g.require(&s.name)?;
        ensure!(
            t.dims == s.dims,
            "tensor {} has dimensions {:?}, Qwen3.6 needs {:?}",
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
    for names in groups {
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

/// [`validate_specs`] for the trunk.
pub fn validate(g: &Gguf, c: &Config) -> Result<()> {
    let groups: Vec<Vec<String>> =
        (0..c.n_layer).flat_map(|il| fused_groups(il, c.recurrent[il])).collect();
    validate_specs(g, &expected(c), &groups)
}
