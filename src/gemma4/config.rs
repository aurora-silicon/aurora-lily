//! Gemma 4's shape, read from the GGUF metadata (`gemma4.*`) and checked
//! against what the kernels are written for (llama.cpp's
//! `llama_model_gemma4::load_arch_hparams` reads the same keys).

use anyhow::{Context, Result, ensure};

use crate::gguf::{Gguf, Value};

/// The GGUF architecture name.
pub const ARCH: &str = "gemma4";

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    /// The dense (shared) MLP's width.
    pub ff: usize,
    pub ff_expert: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_head: usize,
    /// Per layer: sliding-window attention (else global).
    pub swa: Vec<bool>,
    /// Per layer: KV heads.
    pub n_head_kv: Vec<usize>,
    /// Head width of the global layers and of the sliding-window ones.
    pub head_dim: usize,
    pub head_dim_swa: usize,
    pub window: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub rms_eps: f32,
    pub softcap: f32,
    pub context_length: usize,
    pub bos: u32,
    pub eos: u32,
}

fn bools(g: &Gguf, key: &str) -> Result<Vec<bool>> {
    match g.get(key) {
        Some(Value::Array(v)) => v
            .iter()
            .map(|b| match b {
                Value::Bool(b) => Ok(*b),
                _ => anyhow::bail!("{key} holds a non-boolean"),
            })
            .collect(),
        _ => anyhow::bail!("GGUF metadata {key} missing or not an array"),
    }
}

/// A per-layer integer: an array of `n` or one value for every layer.
fn per_layer(g: &Gguf, key: &str, n: usize) -> Result<Vec<usize>> {
    match g.get(key) {
        Some(Value::Array(v)) => {
            ensure!(v.len() == n, "{key} has {} entries for {n} layers", v.len());
            v.iter()
                .map(|x| {
                    x.as_u64()
                        .and_then(|x| usize::try_from(x).ok())
                        .with_context(|| format!("{key} holds a non-integer"))
                })
                .collect()
        }
        Some(x) => {
            let x = x.as_u64().with_context(|| format!("{key} is not an integer"))?;
            Ok(vec![usize::try_from(x)?; n])
        }
        None => anyhow::bail!("GGUF metadata {key} missing"),
    }
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.architecture()?;
        ensure!(arch == ARCH, "GGUF architecture {arch:?}, expected {ARCH:?}");
        let key = |k: &str| format!("{ARCH}.{k}");
        let u = |k: &str| -> Result<usize> {
            let v = g.u64(&key(k))?;
            usize::try_from(v).with_context(|| format!("{ARCH}.{k} {v}"))
        };
        let f = |k: &str| g.f32(&key(k));
        let n_layer = u("block_count")?;
        let swa = bools(g, &key("attention.sliding_window_pattern"))?;
        ensure!(swa.len() == n_layer, "sliding_window_pattern of {} layers", swa.len());
        let n_head_kv = per_layer(g, &key("attention.head_count_kv"), n_layer)?;
        let vocab = match g.get("tokenizer.ggml.tokens") {
            Some(Value::Array(v)) => v.len(),
            _ => anyhow::bail!("tokenizer.ggml.tokens missing"),
        };
        let c = Self {
            n_layer,
            hidden: u("embedding_length")?,
            vocab,
            ff: u("feed_forward_length")?,
            ff_expert: u("expert_feed_forward_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_head: u("attention.head_count")?,
            swa,
            n_head_kv,
            head_dim: u("attention.key_length")?,
            head_dim_swa: u("attention.key_length_swa")?,
            window: u("attention.sliding_window")?,
            rope_base: f("rope.freq_base")?,
            rope_base_swa: f("rope.freq_base_swa")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
            context_length: u("context_length")?,
            bos: g.u64("tokenizer.ggml.bos_token_id")? as u32,
            eos: g.u64("tokenizer.ggml.eos_token_id")? as u32,
        };
        // What the kernels are written for (Gemma 4 26B-A4B's numbers; the
        // other Gemma 4 sizes have per-layer embeddings or KV sharing).
        ensure!(
            u("attention.value_length")? == c.head_dim
                && u("attention.value_length_swa")? == c.head_dim_swa,
            "key and value widths differ"
        );
        ensure!(
            [256, 512].contains(&c.head_dim) && [256, 512].contains(&c.head_dim_swa),
            "head widths {} / {} (the kernels take 256 and 512)",
            c.head_dim,
            c.head_dim_swa
        );
        ensure!(
            u("rope.dimension_count")? == c.head_dim
                && u("rope.dimension_count_swa").unwrap_or(c.head_dim_swa)
                    == c.head_dim_swa,
            "partial RoPE dimension counts (lily rotates whole heads, the \
             global layers' partial rotation through rope_freqs)"
        );
        ensure!(
            g.u64(&key("embedding_length_per_layer_input")).unwrap_or(0) == 0,
            "per-layer embeddings (Gemma 4 E2B/E4B) are not supported"
        );
        ensure!(
            g.u64(&key("attention.shared_kv_layers")).unwrap_or(0) == 0,
            "shared KV layers are not supported"
        );
        ensure!(
            c.hidden.is_multiple_of(256) && c.hidden <= 4096,
            "hidden {}",
            c.hidden
        );
        ensure!(
            c.ff_expert.is_multiple_of(32) && c.ff.is_multiple_of(32),
            "FF widths {} / {}",
            c.ff,
            c.ff_expert
        );
        ensure!(
            c.n_expert <= 256 && c.n_expert_used <= 32,
            "{} experts, {} used",
            c.n_expert,
            c.n_expert_used
        );
        for il in 0..n_layer {
            let kv = c.n_head_kv[il];
            ensure!(
                kv > 0 && c.n_head.is_multiple_of(kv),
                "layer {il}: {} heads over {kv} KV heads",
                c.n_head
            );
        }
        ensure!(c.window > 0, "sliding window of 0");
        Ok(c)
    }

    /// Head width of layer `il`.
    pub fn head_dim_of(&self, il: usize) -> usize {
        if self.swa[il] { self.head_dim_swa } else { self.head_dim }
    }

    /// The RoPE base of layer `il`.
    pub fn rope_base_of(&self, il: usize) -> f32 {
        if self.swa[il] { self.rope_base_swa } else { self.rope_base }
    }

    /// Query width (all heads) of layer `il`.
    pub fn q_width(&self, il: usize) -> usize {
        self.n_head * self.head_dim_of(il)
    }

    /// K (or V) width of layer `il`.
    pub fn kv_width(&self, il: usize) -> usize {
        self.n_head_kv[il] * self.head_dim_of(il)
    }

    /// The widest query projection.
    pub fn max_q_width(&self) -> usize {
        (0..self.n_layer).map(|il| self.q_width(il)).max().unwrap_or(0)
    }

    /// The widest K projection.
    pub fn max_kv_width(&self) -> usize {
        (0..self.n_layer).map(|il| self.kv_width(il)).max().unwrap_or(0)
    }
}
