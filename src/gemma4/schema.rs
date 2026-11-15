//! The tensors a Gemma 4 GGUF (and its draft model's) must hold: each one's
//! exact dimensions, in ggml order (`dims[0]` the row length), and the
//! block types the kernels of its role read. Checked against the file's
//! tensor table before anything is mapped, so a file whose tensors have
//! the right element counts in another arrangement (an expert tensor with
//! its axes swapped) is refused instead of read wrongly.

use anyhow::{Result, ensure};

use super::config::Config;
use crate::gguf::GgmlType;

/// Block types of the dense projections: those with both a GEMV (decode,
/// verify) and a matrix-unit GEMM (prompt runs).
pub const DENSE_TYPES: &[GgmlType] = &[
    GgmlType::Q8_0,
    GgmlType::Q4K,
    GgmlType::Bf16,
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_0,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Iq4Nl,
    GgmlType::Iq2S,
];
/// The embedding (also the head, through the GEMVs).
pub const EMBED_TYPES: &[GgmlType] =
    &[GgmlType::Q8_0, GgmlType::Q5K, GgmlType::Q6K, GgmlType::Q4K];
/// The routed experts' gate/up: the decode kernel's types that the grouped
/// prompt GEMM also takes.
pub const GATE_UP_TYPES: &[GgmlType] =
    &[GgmlType::Q8_0, GgmlType::Q5K, GgmlType::Iq2Xxs, GgmlType::Iq2S];
/// The routed experts' down projection, likewise.
pub const DOWN_TYPES: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::Iq4Nl];
/// Norm weights, scales, the router and the RoPE factors.
pub const F32: &[GgmlType] = &[GgmlType::F32];

/// One tensor the model reads.
#[derive(Clone, Debug)]
pub struct Expect {
    pub name: String,
    pub dims: Vec<u64>,
    pub types: &'static [GgmlType],
    /// Absent is allowed (the global layers' `attn_v`: V is K there).
    pub optional: bool,
}

fn t(name: String, dims: &[usize], types: &'static [GgmlType]) -> Expect {
    Expect {
        name,
        dims: dims.iter().map(|&d| d as u64).collect(),
        types,
        optional: false,
    }
}

/// The target model's tensors.
pub fn target(cfg: &Config) -> Vec<Expect> {
    let (h, e) = (cfg.hidden, cfg.n_expert);
    let mut out = vec![
        t("token_embd.weight".into(), &[h, cfg.vocab], EMBED_TYPES),
        t("output_norm.weight".into(), &[h], F32),
        t("rope_freqs.weight".into(), &[cfg.head_dim / 2], F32),
    ];
    for il in 0..cfg.n_layer {
        let n = |s: &str| format!("blk.{il}.{s}");
        let (qw, kw, d) = (cfg.q_width(il), cfg.kv_width(il), cfg.head_dim_of(il));
        let mut v = t(n("attn_v.weight"), &[h, kw], DENSE_TYPES);
        v.optional = !cfg.swa[il];
        out.extend([
            t(n("attn_norm.weight"), &[h], F32),
            t(n("attn_q.weight"), &[h, qw], DENSE_TYPES),
            t(n("attn_k.weight"), &[h, kw], DENSE_TYPES),
            v,
            t(n("attn_output.weight"), &[qw, h], DENSE_TYPES),
            t(n("attn_q_norm.weight"), &[d], F32),
            t(n("attn_k_norm.weight"), &[d], F32),
            t(n("post_attention_norm.weight"), &[h], F32),
            t(n("ffn_norm.weight"), &[h], F32),
            t(n("ffn_gate.weight"), &[h, cfg.ff], DENSE_TYPES),
            t(n("ffn_up.weight"), &[h, cfg.ff], DENSE_TYPES),
            t(n("ffn_down.weight"), &[cfg.ff, h], DENSE_TYPES),
            t(n("post_ffw_norm_1.weight"), &[h], F32),
            t(n("ffn_gate_inp.weight"), &[h, e], F32),
            t(n("ffn_gate_inp.scale"), &[h], F32),
            t(n("pre_ffw_norm_2.weight"), &[h], F32),
            // Per expert its ff_expert gate rows, then its ff_expert up rows.
            t(n("ffn_gate_up_exps.weight"), &[h, 2 * cfg.ff_expert, e], GATE_UP_TYPES),
            t(n("ffn_down_exps.weight"), &[cfg.ff_expert, h, e], DOWN_TYPES),
            t(n("ffn_down_exps.scale"), &[e], F32),
            t(n("post_ffw_norm_2.weight"), &[h], F32),
            t(n("post_ffw_norm.weight"), &[h], F32),
            t(n("layer_output_scale.weight"), &[1], F32),
        ]);
    }
    out
}

/// The draft model's tensors (`swa`: its layers' kinds; its head widths
/// are the target's).
pub fn draft(target: &Config, hidden: usize, ff: usize, swa: &[bool]) -> Vec<Expect> {
    let heads = target.n_head;
    let mut out = vec![
        t("token_embd.weight".into(), &[hidden, target.vocab], DENSE_TYPES),
        t("output_norm.weight".into(), &[hidden], F32),
        t("rope_freqs.weight".into(), &[target.head_dim / 2], F32),
        t(
            "nextn.pre_projection.weight".into(),
            &[2 * target.hidden, hidden],
            DENSE_TYPES,
        ),
        t("nextn.post_projection.weight".into(), &[hidden, target.hidden], DENSE_TYPES),
    ];
    for (il, &s) in swa.iter().enumerate() {
        let n = |x: &str| format!("blk.{il}.{x}");
        let d = if s { target.head_dim_swa } else { target.head_dim };
        out.extend([
            t(n("attn_norm.weight"), &[hidden], F32),
            t(n("attn_q.weight"), &[hidden, heads * d], DENSE_TYPES),
            t(n("attn_q_norm.weight"), &[d], F32),
            t(n("attn_output.weight"), &[heads * d, hidden], DENSE_TYPES),
            t(n("post_attention_norm.weight"), &[hidden], F32),
            t(n("ffn_norm.weight"), &[hidden], F32),
            t(n("ffn_gate.weight"), &[hidden, ff], DENSE_TYPES),
            t(n("ffn_up.weight"), &[hidden, ff], DENSE_TYPES),
            t(n("ffn_down.weight"), &[ff, hidden], DENSE_TYPES),
            t(n("post_ffw_norm.weight"), &[hidden], F32),
            t(n("layer_output_scale.weight"), &[1], F32),
        ]);
    }
    out
}

/// Checks every expected tensor against `lookup` (a tensor's dims and
/// type, `None` when the file lacks it): present unless optional, of
/// exactly the expected rank and dimensions, of a type its kernels read.
pub fn validate(
    expect: &[Expect],
    lookup: impl Fn(&str) -> Option<(Vec<u64>, GgmlType)>,
) -> Result<()> {
    for x in expect {
        let Some((dims, ty)) = lookup(&x.name) else {
            ensure!(x.optional, "GGUF has no tensor {}", x.name);
            continue;
        };
        ensure!(
            dims == x.dims,
            "{} is {dims:?}, lily's Gemma graph reads {:?} (ggml order)",
            x.name,
            x.dims
        );
        ensure!(
            x.types.contains(&ty),
            "{} is {ty:?}; lily's Gemma kernels take {:?} there",
            x.name,
            x.types
        );
    }
    Ok(())
}

/// [`validate`] over a GGUF's tensor table.
pub fn validate_gguf(expect: &[Expect], g: &crate::gguf::Gguf) -> Result<()> {
    validate(expect, |name| g.tensor(name).map(|t| (t.dims.clone(), t.ty)))
}

#[cfg(test)]
#[path = "../../tests/unit/gemma4/schema.rs"]
mod tests;
