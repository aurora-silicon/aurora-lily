//! DeepSeek V4 Flash's shape, read from the GGUF metadata (`deepseek4.*`)
//! and checked against the numbers the kernels are written for (ds4's
//! `DS4_SHAPE_FLASH`, `config_validate_deepseek4_model`,
//! `validate_compress_ratio_metadata` and `validate_swiglu_clamp_metadata`).

use anyhow::{Context, Result, bail, ensure};

use crate::gguf::{Gguf, Value};

/// The GGUF architecture name.
pub const ARCH: &str = "deepseek4";

/// `expert_gating_func` of DeepSeek V4: router scores `sqrt(softplus(l))`.
pub const GATING_SQRT_SOFTPLUS: u64 = 4;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub n_head: usize,
    /// Query/key/value width of a head (one shared KV head).
    pub head_dim: usize,
    /// RoPE dims, the last `n_rot` of each head.
    pub n_rot: usize,
    pub q_lora: usize,
    /// Rank of each output group's low projection.
    pub o_lora: usize,
    pub o_groups: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_expert: usize,
    /// Leading layers routed by token id (`ffn_gate_tid2eid`).
    pub n_hash_layer: usize,
    pub expert_weight_scale: f32,
    pub swiglu_limit: f32,
    /// Raw sliding-window rows every layer attends.
    pub window: usize,
    pub indexer_heads: usize,
    pub indexer_dim: usize,
    pub indexer_top_k: usize,
    pub n_hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub rope_orig_ctx: usize,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub compress_rope_base: f32,
    /// Per layer: 0 (raw window only), 4 (compressed + indexer) or 128.
    pub compress_ratios: Vec<usize>,
    pub context_length: usize,
    pub bos: u32,
    pub eos: u32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.architecture()?;
        ensure!(arch == ARCH, "GGUF architecture {arch:?}, expected {ARCH:?}");
        ensure!(
            g.get(&format!("{ARCH}.checkpoint_variant")).is_none(),
            "DeepSeek V4 checkpoint variants (Vision-Exp) are not supported"
        );
        let u = |k: &str| -> Result<usize> {
            let v = g.u64(&format!("{ARCH}.{k}"))?;
            usize::try_from(v).with_context(|| format!("{ARCH}.{k} {v}"))
        };
        let f = |k: &str| g.f32(&format!("{ARCH}.{k}"));
        let n_layer = u("block_count")?;
        let array = |k: &str| -> Result<&Vec<Value>> {
            match g.get(&format!("{ARCH}.{k}")) {
                Some(Value::Array(v)) => Ok(v),
                _ => bail!("{ARCH}.{k} missing"),
            }
        };
        let ratios = array("attention.compress_ratios")?;
        ensure!(
            ratios.len() >= n_layer,
            "compress_ratios has {} entries for {n_layer} layers",
            ratios.len()
        );
        let compress_ratios = ratios[..n_layer]
            .iter()
            .map(|v| {
                let r = v.as_u64().context("compress ratio is not an integer")?;
                usize::try_from(r).context("compress ratio")
            })
            .collect::<Result<Vec<_>>>()?;
        let clamps = array("swiglu_clamp_exp")?;
        ensure!(clamps.len() >= n_layer, "swiglu_clamp_exp is shorter than the layers");
        let swiglu_limit =
            clamps[0].as_f32().context("swiglu_clamp_exp is not a float")?;
        for (il, v) in clamps[..n_layer].iter().enumerate() {
            ensure!(
                v.as_f32() == Some(swiglu_limit),
                "layer {il} SwiGLU clamp {v:?} differs from layer 0's {swiglu_limit}"
            );
        }
        ensure!(
            u("expert_gating_func")? as u64 == GATING_SQRT_SOFTPLUS,
            "expert_gating_func is not sqrt-softplus"
        );
        ensure!(
            matches!(
                g.get(&format!("{ARCH}.expert_weights_norm")),
                Some(Value::Bool(true))
            ),
            "expert_weights_norm must be true"
        );
        for k in ["expert_group_count", "expert_group_used_count"] {
            if g.get(&format!("{ARCH}.{k}")).is_some() {
                ensure!(u(k)? == 0, "{k} must be 0 (no grouped routing)");
            }
        }
        ensure!(u("attention.head_count_kv")? == 1, "head_count_kv must be 1");
        ensure!(u("expert_shared_count")? == 1, "expert_shared_count must be 1");
        ensure!(
            u("attention.key_length")? == u("attention.value_length")?,
            "key_length and value_length differ"
        );
        ensure!(
            g.str(&format!("{ARCH}.rope.scaling.type"))? == "yarn",
            "rope scaling is not yarn"
        );
        let tok = |k: &str| -> Result<u32> {
            let v = g.u64(&format!("tokenizer.ggml.{k}"))?;
            u32::try_from(v).with_context(|| format!("{k} {v} is not a token id"))
        };
        let c = Self {
            n_layer,
            hidden: u("embedding_length")?,
            vocab: u("vocab_size")?,
            n_head: u("attention.head_count")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            q_lora: u("attention.q_lora_rank")?,
            o_lora: u("attention.output_lora_rank")?,
            o_groups: u("attention.output_group_count")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            ff_expert: u("expert_feed_forward_length")?,
            n_hash_layer: u("hash_layer_count")?,
            expert_weight_scale: f("expert_weights_scale")?,
            swiglu_limit,
            window: u("attention.sliding_window")?,
            indexer_heads: u("attention.indexer.head_count")?,
            indexer_dim: u("attention.indexer.key_length")?,
            indexer_top_k: u("attention.indexer.top_k")?,
            n_hc: u("hyper_connection.count")?,
            sinkhorn_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            rope_base: f("rope.freq_base")?,
            rope_scale: f("rope.scaling.factor")?,
            rope_orig_ctx: u("rope.scaling.original_context_length")?,
            yarn_beta_fast: f("rope.scaling.yarn_beta_fast")?,
            yarn_beta_slow: f("rope.scaling.yarn_beta_slow")?,
            compress_rope_base: f("attention.compress_rope_freq_base")?,
            compress_ratios,
            context_length: u("context_length")?,
            bos: tok("bos_token_id")?,
            eos: tok("eos_token_id")?,
        };
        c.validate()?;
        Ok(c)
    }

    /// DeepSeek V4 Flash's shape as the published GGUF declares it (tests
    /// build small variants from it).
    pub fn deepseek4_flash() -> Self {
        Self {
            n_layer: 43,
            hidden: 4096,
            vocab: 129280,
            n_head: 64,
            head_dim: 512,
            n_rot: 64,
            q_lora: 1024,
            o_lora: 1024,
            o_groups: 8,
            n_expert: 256,
            n_expert_used: 6,
            ff_expert: 2048,
            n_hash_layer: 3,
            expert_weight_scale: 1.5,
            swiglu_limit: 10.0,
            window: 128,
            indexer_heads: 64,
            indexer_dim: 128,
            indexer_top_k: 512,
            n_hc: 4,
            sinkhorn_iters: 20,
            hc_eps: 1e-6,
            rms_eps: 1e-6,
            rope_base: 10000.0,
            rope_scale: 16.0,
            rope_orig_ctx: 65536,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            compress_rope_base: 160000.0,
            compress_ratios: (0..43)
                .map(|il| match il {
                    0 | 1 => 0,
                    _ if il % 2 == 0 => 4,
                    _ => 128,
                })
                .collect(),
            context_length: 1 << 20,
            bos: 0,
            eos: 1,
        }
    }

    /// The shape DeepSeek V4 Flash has; the kernels assume these numbers.
    pub fn validate(&self) -> Result<()> {
        let want: [(&str, usize, usize); 17] = [
            ("block_count", self.n_layer, 43),
            ("embedding_length", self.hidden, 4096),
            ("vocab_size", self.vocab, 129280),
            ("attention.head_count", self.n_head, 64),
            ("attention.key_length", self.head_dim, 512),
            ("rope.dimension_count", self.n_rot, 64),
            ("attention.q_lora_rank", self.q_lora, 1024),
            ("attention.output_lora_rank", self.o_lora, 1024),
            ("attention.output_group_count", self.o_groups, 8),
            ("expert_count", self.n_expert, 256),
            ("expert_used_count", self.n_expert_used, 6),
            ("expert_feed_forward_length", self.ff_expert, 2048),
            ("hash_layer_count", self.n_hash_layer, 3),
            ("attention.sliding_window", self.window, 128),
            ("attention.indexer.head_count", self.indexer_heads, 64),
            ("attention.indexer.key_length", self.indexer_dim, 128),
            ("hyper_connection.count", self.n_hc, 4),
        ];
        for (name, got, expected) in want {
            ensure!(
                got == expected,
                "DeepSeek V4 Flash {name} is {got}, lily's kernels need {expected}"
            );
        }
        ensure!(
            self.indexer_top_k >= 1,
            "indexer top_k {} must be positive",
            self.indexer_top_k
        );
        ensure!(
            self.sinkhorn_iters >= 1,
            "Sinkhorn iterations {}",
            self.sinkhorn_iters
        );
        ensure!(self.compress_ratios.len() == self.n_layer, "compress_ratios length");
        for (il, &r) in self.compress_ratios.iter().enumerate() {
            let expected = match il {
                0 | 1 => 0,
                _ if il % 2 == 0 => 4,
                _ => 128,
            };
            ensure!(
                r == expected,
                "layer {il} compress ratio {r}, expected {expected}"
            );
        }
        let close =
            |got: f32, want: f32| (got - want).abs() <= want.abs().max(1.0) * 1e-6;
        for (name, got, want) in [
            ("expert_weights_scale", self.expert_weight_scale, 1.5),
            ("rope.freq_base", self.rope_base, 10000.0),
            ("rope.scaling.factor", self.rope_scale, 16.0),
            ("rope.scaling.yarn_beta_fast", self.yarn_beta_fast, 32.0),
            ("rope.scaling.yarn_beta_slow", self.yarn_beta_slow, 1.0),
            ("attention.compress_rope_freq_base", self.compress_rope_base, 160000.0),
            ("hyper_connection.epsilon", self.hc_eps, 1e-6),
            ("swiglu_clamp_exp", self.swiglu_limit, 10.0),
        ] {
            ensure!(close(got, want), "{name} is {got}, DeepSeek V4 Flash has {want}");
        }
        ensure!(
            (self.rms_eps - 1e-6).abs() <= 1e-11,
            "rms epsilon {} is not 1e-6",
            self.rms_eps
        );
        ensure!(
            self.rope_orig_ctx == 65536,
            "rope original context {}",
            self.rope_orig_ctx
        );
        for (name, id) in [("bos", self.bos), ("eos", self.eos)] {
            ensure!(
                (id as usize) < self.vocab,
                "{name} token {id} is outside the vocabulary of {}",
                self.vocab
            );
        }
        Ok(())
    }

    pub fn hc_dim(&self) -> usize {
        self.n_hc * self.hidden
    }

    /// `pre[n_hc]`, `post[n_hc]`, `comb[n_hc * n_hc]`.
    pub fn hc_mix(&self) -> usize {
        self.n_hc * (self.n_hc + 2)
    }

    pub fn ratio(&self, il: usize) -> usize {
        self.compress_ratios[il]
    }

    /// Whether layer `il` has the indexer (its ratio is 4).
    pub fn has_indexer(&self, il: usize) -> bool {
        self.compress_ratios[il] == 4
    }

    /// Lanes of the compressor state: ratio-4 layers pool overlapping
    /// windows (the previous window's rows in lane 0, the current in lane 1).
    pub fn compress_lanes(&self, il: usize) -> usize {
        if self.compress_ratios[il] == 4 { 2 } else { 1 }
    }

    pub fn is_hash_layer(&self, il: usize) -> bool {
        il < self.n_hash_layer
    }

    /// The heads one output group covers, and its input width.
    pub fn o_group_heads(&self) -> usize {
        self.n_head / self.o_groups
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// The RoPE parameters of layer `il` (ds4 `rope_tail_layer_inplace`):
    /// compressed layers use the long-context base and YaRN.
    pub fn rope(&self, il: usize) -> Rope {
        if self.compress_ratios[il] == 0 {
            Rope {
                n_rot: self.n_rot,
                base: self.rope_base,
                freq_scale: 1.0,
                yarn: None,
            }
        } else {
            Rope {
                n_rot: self.n_rot,
                base: self.compress_rope_base,
                freq_scale: 1.0 / self.rope_scale,
                yarn: Some(Yarn {
                    orig_ctx: self.rope_orig_ctx,
                    beta_fast: self.yarn_beta_fast,
                    beta_slow: self.yarn_beta_slow,
                }),
            }
        }
    }
}

/// YaRN interpolation of the low frequencies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub orig_ctx: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// RoPE of the last `n_rot` dims of a head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub n_rot: usize,
    pub base: f32,
    pub freq_scale: f32,
    pub yarn: Option<Yarn>,
}

#[cfg(test)]
#[path = "../../tests/unit/deepseek4/config.rs"]
mod tests;
