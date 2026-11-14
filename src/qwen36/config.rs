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

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.architecture()?;
        ensure!(arch == ARCH, "GGUF architecture {arch:?}, expected {ARCH:?}");
        let u = |k: &str| -> Result<usize> {
            let v = g.u64(&format!("{ARCH}.{k}"))?;
            usize::try_from(v).with_context(|| format!("{ARCH}.{k} {v}"))
        };
        let f = |k: &str| g.f32(&format!("{ARCH}.{k}"));
        let n_layer = u("block_count")?;
        let recurrent = match g.get(&format!("{ARCH}.attention.recurrent_layers")) {
            Some(Value::Array(v)) => v
                .iter()
                .map(|t| match t {
                    Value::Bool(b) => Ok(*b),
                    other => anyhow::bail!("recurrent_layers entry {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            // llama.cpp's default: every `full_attention_interval`-th layer
            // is full attention.
            _ => {
                let every = u("full_attention_interval")?;
                ensure!(every > 0, "full_attention_interval 0");
                (0..n_layer).map(|il| (il + 1) % every != 0).collect()
            }
        };
        let vocab = g
            .require("token_embd.weight")?
            .dims
            .get(1)
            .copied()
            .context("token_embd.weight is not a matrix")?;
        let tok = |k: &str| -> Result<u32> {
            let v = g.u64(&format!("tokenizer.ggml.{k}"))?;
            u32::try_from(v).with_context(|| format!("{k} {v} is not a token id"))
        };
        let v_heads = u("ssm.time_step_rank")?;
        let inner = u("ssm.inner_size")?;
        ensure!(v_heads > 0 && inner % v_heads == 0, "ssm.inner_size {inner}");
        let c = Self {
            n_layer,
            hidden: u("embedding_length")?,
            vocab: usize::try_from(vocab)?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rope_dim: u("rope.dimension_count")?,
            rope_theta: f("rope.freq_base")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            ff_expert: u("expert_feed_forward_length")?,
            ff_shared: u("expert_shared_feed_forward_length")?,
            gdn_k_heads: u("ssm.group_count")?,
            gdn_v_heads: v_heads,
            gdn_dim: u("ssm.state_size")?,
            gdn_conv: u("ssm.conv_kernel")?,
            recurrent,
            context_length: u("context_length")?,
            eos: tok("eos_token_id")?,
            pad: tok("padding_token_id")?,
        };
        ensure!(
            u("attention.value_length")? == c.head_dim,
            "attention key and value widths differ"
        );
        ensure!(inner / v_heads == c.gdn_dim, "GDN value width {} != key width", inner / v_heads);
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

    /// The numbers the kernels are written for; the rest is free.
    pub fn validate(&self) -> Result<()> {
        let fixed: [(&str, usize, usize); 5] = [
            ("attention.key_length", self.head_dim, 256),
            ("ssm.state_size", self.gdn_dim, 128),
            ("ssm.conv_kernel", self.gdn_conv, 4),
            ("expert_used_count (at most)", self.n_expert_used.max(8), 8),
            ("expert_count (at most)", self.n_expert.max(256), 256),
        ];
        for (name, got, expected) in fixed {
            ensure!(got == expected, "{name} is {got}, lily's kernels need {expected}");
        }
        ensure!(
            self.recurrent.len() == self.n_layer,
            "recurrent_layers has {} entries for {} layers",
            self.recurrent.len(),
            self.n_layer
        );
        ensure!(
            self.n_head_kv > 0 && self.n_head.is_multiple_of(self.n_head_kv),
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
            self.rope_dim.is_multiple_of(2) && self.rope_dim <= self.head_dim,
            "rotary dims {}",
            self.rope_dim
        );
        ensure!(
            self.hidden.is_multiple_of(256)
                && self.ff_expert.is_multiple_of(256)
                && self.ff_shared.is_multiple_of(256),
            "widths must be whole 256-value blocks"
        );
        ensure!(self.n_expert_used > 0, "no experts used");
        for (name, id) in [("eos", self.eos), ("pad", self.pad)] {
            ensure!(
                (id as usize) < self.vocab,
                "{name} token {id} is outside the vocabulary of {}",
                self.vocab
            );
        }
        ensure!(self.rms_eps > 0.0, "rms epsilon must be positive");
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
