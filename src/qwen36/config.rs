//! Qwen3.6-35B-A3B's shape, read from the GGUF metadata (`qwen35moe.*`, the
//! architecture llama.cpp's converter writes for Qwen3.5/3.6 MoE models)
//! and checked against what lily's kernels are written for.

use anyhow::{Context, Result, ensure};

use crate::gguf::{Gguf, Value};

/// The GGUF architecture name.
pub const ARCH: &str = "qwen35moe";

#[derive(Clone, Debug)]
pub struct Config {
    /// Trunk layers (the MTP block, in a separate file, is not counted).
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    /// Full attention: query heads, KV heads, head width, rotary dims.
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_expert: usize,
    pub ff_shared: usize,
    /// Gated DeltaNet: key heads (q and k), value heads, head width (keys
    /// and values alike), causal conv taps.
    pub gdn_k_heads: usize,
    pub gdn_v_heads: usize,
    pub gdn_dim: usize,
    pub gdn_conv: usize,
    /// Per trunk layer: Gated DeltaNet (true) or full attention.
    pub recurrent: Vec<bool>,
    pub context_length: usize,
    /// `<|im_end|>`, the end of a turn.
    pub eos: u32,
    /// `<|endoftext|>`.
    pub pad: u32,
}

/// Trunk layers lily takes at most (the published models have 40 to 60):
/// the layer table is sized from `block_count` before any tensor is read.
pub const MAX_LAYERS: usize = 256;
/// The largest vocabulary taken (the embedding's row count).
pub const MAX_VOCAB: usize = 1 << 21;

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let vocab = g
            .require("token_embd.weight")?
            .dims
            .get(1)
            .copied()
            .context("token_embd.weight is not a matrix")?;
        Self::from_metadata(&g.metadata, vocab)
    }

    /// The shape from a GGUF's metadata and the embedding's row count. Every
    /// number is bounded before anything is sized from it; a key of the
    /// wrong type is an error (not absent), and metadata lily's graph does
    /// not implement (RoPE or attention scaling, another expert gating) is
    /// refused rather than ignored.
    pub fn from_metadata(
        meta: &std::collections::BTreeMap<String, Value>,
        vocab: u64,
    ) -> Result<Self> {
        let get = |k: &str| meta.get(k);
        let arch = get("general.architecture").and_then(Value::as_str);
        ensure!(arch == Some(ARCH), "GGUF architecture {arch:?}, expected {ARCH:?}");
        let key = |k: &str| format!("{ARCH}.{k}");
        let int = |k: &str| -> Result<u64> {
            get(k)
                .with_context(|| format!("GGUF metadata {k} missing"))?
                .as_u64()
                .with_context(|| {
                    format!("GGUF metadata {k} is not a non-negative integer")
                })
        };
        // Bounded to `max` before it sizes anything.
        let u = |k: &str, max: usize| -> Result<usize> {
            let v = int(&key(k))?;
            ensure!(v <= max as u64, "{ARCH}.{k} is {v}, lily takes at most {max}");
            Ok(v as usize)
        };
        let f = |k: &str| -> Result<f32> {
            let k = key(k);
            let v = get(&k)
                .with_context(|| format!("GGUF metadata {k} missing"))?
                .as_f32()
                .with_context(|| format!("GGUF metadata {k} is not a float"))?;
            ensure!(v.is_finite(), "{k} is {v}");
            Ok(v)
        };
        let n_layer = u("block_count", MAX_LAYERS)?;
        ensure!(n_layer > 0, "{ARCH}.block_count is 0");
        let recurrent = match get(&key("attention.recurrent_layers")) {
            Some(Value::Array(v)) => {
                ensure!(
                    v.len() >= n_layer,
                    "recurrent_layers has {} entries for {n_layer} layers",
                    v.len()
                );
                // Entries past the trunk are the MTP blocks' (a separate file).
                v[..n_layer]
                    .iter()
                    .map(|t| match t {
                        Value::Bool(b) => Ok(*b),
                        other => anyhow::bail!("recurrent_layers entry {other:?}"),
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            Some(other) => {
                anyhow::bail!("{ARCH}.attention.recurrent_layers is {other:?}")
            }
            // llama.cpp's default: every `full_attention_interval`-th layer
            // is full attention.
            None => {
                let every = u("full_attention_interval", MAX_LAYERS)?;
                ensure!(every > 0, "full_attention_interval 0");
                (0..n_layer).map(|il| (il + 1) % every != 0).collect()
            }
        };
        ensure!(vocab <= MAX_VOCAB as u64, "a vocabulary of {vocab} tokens");
        let tok = |k: &str| -> Result<u32> {
            let k = format!("tokenizer.ggml.{k}");
            let v = int(&k)?;
            u32::try_from(v).with_context(|| format!("{k} {v} is not a token id"))
        };
        // Semantics the graph does not implement.
        if let Some(v) = get(&key("rope.scaling.type")) {
            ensure!(
                v.as_str() == Some("none"),
                "{ARCH}.rope.scaling.type {v:?} is not supported"
            );
        }
        for k in ["rope.scaling.factor", "attention.scale"] {
            if get(&key(k)).is_some() {
                let v = f(k)?;
                let neutral = if k == "attention.scale" {
                    v == 0.0
                } else {
                    v == 1.0 || v == 0.0
                };
                ensure!(neutral, "{ARCH}.{k} {v} is not supported");
            }
        }
        if get(&key("expert_weights_scale")).is_some() {
            let v = f("expert_weights_scale")?;
            ensure!(
                v == 0.0 || v == 1.0,
                "{ARCH}.expert_weights_scale {v} is not supported"
            );
        }
        if get(&key("expert_gating_func")).is_some() {
            // LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX.
            let v = int(&key("expert_gating_func"))?;
            ensure!(v == 1, "{ARCH}.expert_gating_func {v} is not softmax");
        }
        let v_heads = u("ssm.time_step_rank", 1024)?;
        let inner = u("ssm.inner_size", 1 << 20)?;
        ensure!(v_heads > 0 && inner % v_heads == 0, "ssm.inner_size {inner}");
        let c = Self {
            n_layer,
            hidden: u("embedding_length", 1 << 16)?,
            vocab: vocab as usize,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            n_head: u("attention.head_count", 1024)?,
            n_head_kv: u("attention.head_count_kv", 1024)?,
            head_dim: u("attention.key_length", 1 << 12)?,
            rope_dim: u("rope.dimension_count", 1 << 12)?,
            rope_theta: f("rope.freq_base")?,
            n_expert: u("expert_count", 1 << 12)?,
            n_expert_used: u("expert_used_count", 1 << 12)?,
            ff_expert: u("expert_feed_forward_length", 1 << 16)?,
            ff_shared: u("expert_shared_feed_forward_length", 1 << 16)?,
            gdn_k_heads: u("ssm.group_count", 1024)?,
            gdn_v_heads: v_heads,
            gdn_dim: u("ssm.state_size", 1 << 12)?,
            gdn_conv: u("ssm.conv_kernel", 64)?,
            recurrent,
            context_length: u("context_length", 1 << 30)?,
            eos: tok("eos_token_id")?,
            pad: tok("padding_token_id")?,
        };
        ensure!(
            u("attention.value_length", 1 << 12)? == c.head_dim,
            "attention key and value widths differ"
        );
        ensure!(
            inner / v_heads == c.gdn_dim,
            "GDN value width {} != key width",
            inner / v_heads
        );
        c.validate()?;
        Ok(c)
    }

    /// Qwen3.6-35B-A3B's shape as ggml-org's GGUFs declare it (tests build
    /// small variants from it).
    pub fn qwen36_35b_a3b() -> Self {
        Self {
            n_layer: 40,
            hidden: 2048,
            vocab: 248320,
            rms_eps: 1e-6,
            n_head: 16,
            n_head_kv: 2,
            head_dim: 256,
            rope_dim: 64,
            rope_theta: 1e7,
            n_expert: 256,
            n_expert_used: 8,
            ff_expert: 512,
            ff_shared: 512,
            gdn_k_heads: 16,
            gdn_v_heads: 32,
            gdn_dim: 128,
            gdn_conv: 4,
            recurrent: (0..40).map(|il| il % 4 != 3).collect(),
            context_length: 262144,
            eos: 248046,
            pad: 248044,
        }
    }

    /// The shapes the kernels take, checked before anything is mapped.
    pub fn validate(&self) -> Result<()> {
        let fixed: [(&str, usize, usize); 3] = [
            ("attention.key_length", self.head_dim, 256),
            ("ssm.state_size", self.gdn_dim, 128),
            ("ssm.conv_kernel", self.gdn_conv, 4),
        ];
        for (name, got, expected) in fixed {
            ensure!(got == expected, "{name} is {got}, lily's kernels need {expected}");
        }
        ensure!(
            (1..=MAX_LAYERS).contains(&self.n_layer)
                && self.recurrent.len() == self.n_layer,
            "{} layers, {} recurrent_layers entries",
            self.n_layer,
            self.recurrent.len()
        );
        ensure!(
            (256..=1 << 16).contains(&self.hidden) && self.hidden.is_multiple_of(256),
            "embedding_length {} (lily takes multiples of 256)",
            self.hidden
        );
        ensure!(
            (1..=MAX_VOCAB).contains(&self.vocab),
            "a vocabulary of {}",
            self.vocab
        );
        ensure!(
            self.n_head > 0
                && self.n_head_kv > 0
                && self.n_head.is_multiple_of(self.n_head_kv),
            "{} query heads over {} KV heads",
            self.n_head,
            self.n_head_kv
        );
        ensure!(
            self.gdn_k_heads > 0 && self.gdn_v_heads.is_multiple_of(self.gdn_k_heads),
            "{} GDN value heads over {} key heads",
            self.gdn_v_heads,
            self.gdn_k_heads
        );
        ensure!(
            self.rope_dim > 0
                && self.rope_dim.is_multiple_of(2)
                && self.rope_dim <= self.head_dim,
            "rotary dims {}",
            self.rope_dim
        );
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 1.0,
            "rope.freq_base {}",
            self.rope_theta
        );
        ensure!(
            self.rms_eps.is_finite() && self.rms_eps > 0.0 && self.rms_eps < 1.0,
            "rms epsilon {}",
            self.rms_eps
        );
        // The router takes at most 256 experts; the grouped prompt GEMMs a
        // power of two used; the decode down kernel rows of 256 or 512.
        ensure!(
            (1..=256).contains(&self.n_expert)
                && self.n_expert_used.is_power_of_two()
                && self.n_expert_used <= 8
                && self.n_expert_used <= self.n_expert,
            "{} experts, {} used (lily takes 1, 2, 4 or 8 of at most 256)",
            self.n_expert,
            self.n_expert_used
        );
        ensure!(
            self.ff_expert == 256 || self.ff_expert == 512,
            "expert_feed_forward_length {} (lily's kernels take 256 or 512)",
            self.ff_expert
        );
        // The shared expert's down GEMV: rows of 8 or 16 tasks, or 32 and up.
        ensure!(
            self.ff_shared == 256
                || self.ff_shared == 512
                || (self.ff_shared >= 1024 && self.ff_shared.is_multiple_of(256)),
            "expert_shared_feed_forward_length {}",
            self.ff_shared
        );
        ensure!(self.context_length > 0, "context_length 0");
        for (name, id) in [("eos", self.eos), ("pad", self.pad)] {
            ensure!(
                (id as usize) < self.vocab,
                "{name} token {id} is outside the vocabulary of {}",
                self.vocab
            );
        }
        Ok(())
    }

    /// Width of the GDN's mixed q|k|v projection (the conv's channels).
    pub fn gdn_qkv(&self) -> usize {
        (2 * self.gdn_k_heads + self.gdn_v_heads) * self.gdn_dim
    }

    /// Width of the GDN's value heads (z, the output).
    pub fn gdn_inner(&self) -> usize {
        self.gdn_v_heads * self.gdn_dim
    }

    /// Width of the attention's query+gate projection.
    pub fn q_gate(&self) -> usize {
        2 * self.n_head * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    pub fn attn_layers(&self) -> usize {
        self.recurrent.iter().filter(|r| !**r).count()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen36/config.rs"]
mod tests;
