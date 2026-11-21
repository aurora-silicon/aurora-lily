//! Tensor schema validation for Inkling 2 / Inkling 2 Loop GGUFs.
//!
//! Every tensor's exact dimensions (in ggml order, `dims[0]` contiguous) and
//! block type are checked against the file's tensor table before anything is
//! mapped. Supports both mainline `inkling2_loop` (`ggml-org` and
//! `unsloth/Mainline_Support`) and `unsloth`'s root `inkling2` looped layout.

use anyhow::{Result, ensure};

use super::config::{Config, LoopLayout};
use crate::gguf::{GgmlType, Gguf};

/// Block types supported for dense projections (GEMV + GEMM / sgemm).
pub const DENSE_TYPES: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q2K,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Iq2Xxs,
    GgmlType::Iq2Xs,
    GgmlType::Iq3Xxs,
    GgmlType::Iq1S,
    GgmlType::Iq4Nl,
    GgmlType::Iq2S,
    GgmlType::Iq4Xs,
    GgmlType::Mxfp4,
    GgmlType::Nvfp4,
    GgmlType::Q2_0,
];

/// Block types supported for embeddings and LM head.
pub const EMBED_TYPES: &[GgmlType] = DENSE_TYPES;

/// Block types supported for routed expert weights.
pub const EXPERT_TYPES: &[GgmlType] = DENSE_TYPES;

/// Unquantized F32 tensors (norms, clips, conv weights, queries, router, biases, scales).
pub const F32_ONLY: &[GgmlType] = &[GgmlType::F32];

/// F32 or BF16/F16 for small control/query tensors.
pub const FLOAT_TYPES: &[GgmlType] = &[GgmlType::F32, GgmlType::Bf16, GgmlType::F16];

#[derive(Clone, Debug)]
pub struct Expect {
    pub name: String,
    pub dims: Vec<u64>,
    pub types: &'static [GgmlType],
    pub optional: bool,
}

fn req(name: String, dims: &[usize], types: &'static [GgmlType]) -> Expect {
    Expect {
        name,
        dims: dims.iter().map(|&d| d as u64).collect(),
        types,
        optional: false,
    }
}

fn opt(name: String, dims: &[usize], types: &'static [GgmlType]) -> Expect {
    Expect {
        name,
        dims: dims.iter().map(|&d| d as u64).collect(),
        types,
        optional: true,
    }
}

/// Expected tensors for an Inkling 2 / Inkling 2 Loop GGUF.
pub fn target(cfg: &Config) -> Vec<Expect> {
    let (h, e, ff, n_conv) =
        (cfg.hidden, cfg.n_expert, cfg.ff_expert, cfg.sconv_l_cache);
    let mut out = vec![
        req("token_embd.weight".into(), &[h, cfg.vocab_padded], EMBED_TYPES),
        req("token_embd_norm.weight".into(), &[h], F32_ONLY),
        req("output.weight".into(), &[h, cfg.vocab_padded], EMBED_TYPES),
        req("output_norm.weight".into(), &[h], F32_ONLY),
        req("output_res_query.weight".into(), &[h], F32_ONLY),
    ];

    if cfg.loop_layout == LoopLayout::Unsloth {
        let n_q = 2 * (cfg.loop_end - cfg.loop_start);
        out.push(req("loop_res_query.weight".into(), &[h, n_q], F32_ONLY));
        out.push(req("loop_kv_res_query.weight".into(), &[h, n_q], F32_ONLY));
    }

    for i in 0..cfg.n_phys {
        let n = |s: &str| format!("blk.{i}.{s}");
        let (qw, kw) = (cfg.q_width(i), cfg.kv_width(i));
        let (nh, nkv, hd) = (cfg.n_head[i], cfg.n_head_kv[i], cfg.head_dim);
        let extent = cfg.rel_extent_of(i);
        let swa = cfg.swa[i];

        out.extend([
            req(n("attn_q.weight"), &[h, qw], DENSE_TYPES),
            req(n("attn_k.weight"), &[h, kw], DENSE_TYPES),
            req(n("attn_v.weight"), &[h, kw], DENSE_TYPES),
            req(n("attn_output.weight"), &[qw, h], DENSE_TYPES),
            req(n("attn_gate.weight"), &[h, qw], DENSE_TYPES),
            req(n("attn_q_clip.weight"), &[hd, nh], F32_ONLY),
            req(n("attn_sink_proj.weight"), &[h, nh], DENSE_TYPES),
            req(n("attn_rel.weight"), &[h, cfg.rel_len * nh], DENSE_TYPES),
            req(n("attn_rel_bias.weight"), &[cfg.rel_len, extent], F32_ONLY),
            req(n("attn_rel_clip.weight"), &[cfg.rel_len, nh], F32_ONLY),
            req(n("attn_k_conv.weight"), &[n_conv, kw], F32_ONLY),
            req(n("attn_v_conv.weight"), &[n_conv, kw], F32_ONLY),
            req(n("attn_post_conv.weight"), &[n_conv, h], F32_ONLY),
            req(n("ffn_post_conv.weight"), &[n_conv, h], F32_ONLY),
        ]);

        if swa {
            out.push(req(n("attn_norm.weight"), &[h], F32_ONLY));
        } else {
            out.extend([
                req(n("attn_q_inp_norm.weight"), &[h], F32_ONLY),
                req(n("attn_k_inp_norm.weight"), &[h], F32_ONLY),
                req(n("attn_v_inp_norm.weight"), &[h], F32_ONLY),
                req(n("attn_k_conv_proj.weight"), &[h, n_conv * nkv], DENSE_TYPES),
                req(n("attn_v_conv_proj.weight"), &[h, n_conv * nkv], DENSE_TYPES),
            ]);
        }

        if i > 0 {
            if swa {
                // Mainline / ggml-org stores [h] (1D) or [h, 1] (2D). Handled in validate.
                out.push(req(n("attn_res_query.weight"), &[h], F32_ONLY));
            } else {
                out.push(req(n("attn_res_query.weight"), &[h, 3], F32_ONLY));
            }
        }
        out.extend([
            req(n("ffn_res_query.weight"), &[h], F32_ONLY),
            req(n("ffn_norm.weight"), &[h], F32_ONLY),
            req(n("ffn_gate_inp.weight"), &[h, e], FLOAT_TYPES),
            opt(n("exp_probs_b.bias"), &[e], F32_ONLY),
            req(n("ffn_gate_up_exps.weight"), &[h, 2 * ff, e], EXPERT_TYPES),
            opt(n("ffn_gate_up_exps.scale"), &[e], F32_ONLY),
            req(n("ffn_down_exps.weight"), &[ff, h, e], EXPERT_TYPES),
            opt(n("ffn_down_exps.scale"), &[e], F32_ONLY),
        ]);

        if cfg.loop_count > 1 && i >= cfg.loop_start && i < cfg.loop_end {
            let r = cfg.loop_gate_rank;
            match cfg.loop_layout {
                LoopLayout::Mainline => {
                    out.push(req(n("loop_msg_query.weight"), &[h, 4], F32_ONLY));
                    for c in ["q", "k", "v", "ffn"] {
                        out.push(req(
                            n(&format!("loop_{c}_ctrl_down.weight")),
                            &[2 * h, r],
                            DENSE_TYPES,
                        ));
                        out.push(req(
                            n(&format!("loop_{c}_ctrl_gain.weight")),
                            &[r, h],
                            DENSE_TYPES,
                        ));
                        out.push(req(
                            n(&format!("loop_{c}_ctrl_forget.weight")),
                            &[r, h],
                            DENSE_TYPES,
                        ));
                    }
                }
                LoopLayout::Unsloth => {
                    for c in ["attn", "k", "v", "mlp"] {
                        out.push(req(
                            n(&format!("loop_{c}_down.weight")),
                            &[2 * h, r],
                            DENSE_TYPES,
                        ));
                        out.push(req(
                            n(&format!("loop_{c}_gain.weight")),
                            &[r, h],
                            DENSE_TYPES,
                        ));
                        out.push(req(
                            n(&format!("loop_{c}_forget.weight")),
                            &[r, h],
                            DENSE_TYPES,
                        ));
                    }
                }
                LoopLayout::None => {}
            }
        }
    }

    out
}

/// Checks every expected tensor in `gguf` against `target(cfg)`.
pub fn validate(gguf: &Gguf, cfg: &Config) -> Result<()> {
    for exp in target(cfg) {
        let Some(info) = gguf.tensor(&exp.name) else {
            ensure!(exp.optional, "missing required tensor {}", exp.name);
            continue;
        };
        // Allow trailing singleton dim (e.g. [h, 1] vs [h] for SWA attn_res_query).
        let dims_match = info.dims == exp.dims
            || (info.dims.len() == exp.dims.len() + 1
                && info.dims[..exp.dims.len()] == exp.dims[..]
                && *info.dims.last().unwrap_or(&0) == 1);
        ensure!(
            dims_match,
            "tensor {}: expected dims {:?}, got {:?}",
            exp.name,
            exp.dims,
            info.dims
        );
        ensure!(
            exp.types.contains(&info.ty),
            "tensor {}: unsupported type {:?}",
            exp.name,
            info.ty
        );
    }
    Ok(())
}
