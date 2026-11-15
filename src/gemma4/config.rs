//! Gemma 4's shape, read from the GGUF metadata (`gemma4.*`) and checked
//! against what the kernels are written for (llama.cpp's
//! `llama_model_gemma4::load_arch_hparams` reads the same keys).

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};

use crate::gguf::{Gguf, Value};

/// The GGUF architecture name.
pub const ARCH: &str = "gemma4";

/// Layers lily takes at most (Gemma 4 26B-A4B has 30): the layer tables
/// are sized from `block_count` before any tensor is read.
pub const MAX_LAYERS: usize = 256;
/// The largest vocabulary taken (the embedding's row count).
pub const MAX_VOCAB: usize = 1 << 21;
/// The widest hidden state the row kernels take.
pub const MAX_HIDDEN: usize = 4096;

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
    /// Rows a prompt run takes: not in the file, the device's
    /// ([`super::model::prefill_rows`], set when a model is loaded; else
    /// [`super::model::PREFILL_ROWS`]).
    pub prefill_rows: usize,
}

/// A metadata reader over `meta` for the keys of architecture `arch`:
/// every number is bounded before anything is sized from it, and a key of
/// the wrong type is an error (not absent).
pub(crate) struct Meta<'a> {
    pub meta: &'a BTreeMap<String, Value>,
    pub arch: &'a str,
}

impl Meta<'_> {
    pub fn get(&self, k: &str) -> Option<&Value> {
        self.meta.get(&format!("{}.{k}", self.arch))
    }

    fn int_at(&self, k: &str, v: Option<&Value>) -> Result<u64> {
        v.with_context(|| format!("GGUF metadata {k} missing"))?
            .as_u64()
            .with_context(|| format!("GGUF metadata {k} is not a non-negative integer"))
    }

    /// `arch.k`, at most `max`.
    pub fn u(&self, k: &str, max: usize) -> Result<usize> {
        let name = format!("{}.{k}", self.arch);
        let v = self.int_at(&name, self.get(k))?;
        ensure!(v <= max as u64, "{name} is {v}, lily takes at most {max}");
        Ok(v as usize)
    }

    /// `arch.k` when present (at most `max`), else `default`.
    pub fn u_or(&self, k: &str, max: usize, default: usize) -> Result<usize> {
        if self.get(k).is_some() { self.u(k, max) } else { Ok(default) }
    }

    /// `arch.k`, a finite float.
    pub fn f(&self, k: &str) -> Result<f32> {
        let name = format!("{}.{k}", self.arch);
        let v = self
            .get(k)
            .with_context(|| format!("GGUF metadata {name} missing"))?
            .as_f32()
            .with_context(|| format!("GGUF metadata {name} is not a float"))?;
        ensure!(v.is_finite(), "{name} is {v}");
        Ok(v)
    }

    /// `arch.k`: `n` booleans.
    pub fn bools(&self, k: &str, n: usize) -> Result<Vec<bool>> {
        let name = format!("{}.{k}", self.arch);
        match self.get(k) {
            Some(Value::Array(v)) => {
                ensure!(v.len() == n, "{name} has {} entries for {n} layers", v.len());
                v.iter()
                    .map(|b| match b {
                        Value::Bool(b) => Ok(*b),
                        other => anyhow::bail!("{name} holds {other:?}"),
                    })
                    .collect()
            }
            Some(other) => anyhow::bail!("{name} is {other:?}, not an array"),
            None => anyhow::bail!("GGUF metadata {name} missing"),
        }
    }

    /// `arch.k`: one integer (at most `max`) a layer, as an array of `n` or
    /// one value for every layer.
    pub fn per_layer(&self, k: &str, n: usize, max: usize) -> Result<Vec<usize>> {
        let name = format!("{}.{k}", self.arch);
        let bounded = |x: &Value| -> Result<usize> {
            let v = self.int_at(&name, Some(x))?;
            ensure!(v <= max as u64, "{name} holds {v}, lily takes at most {max}");
            Ok(v as usize)
        };
        match self.get(k) {
            Some(Value::Array(v)) => {
                ensure!(v.len() == n, "{name} has {} entries for {n} layers", v.len());
                v.iter().map(bounded).collect()
            }
            Some(x) => Ok(vec![bounded(x)?; n]),
            None => anyhow::bail!("GGUF metadata {name} missing"),
        }
    }

    /// Refuses `arch.k` unless absent or `neutral` (semantics lily's graph
    /// does not implement).
    pub fn refuse_float(&self, k: &str, neutral: &[f32]) -> Result<()> {
        if self.get(k).is_some() {
            let v = self.f(k)?;
            ensure!(neutral.contains(&v), "{}.{k} {v} is not supported", self.arch);
        }
        Ok(())
    }

    /// Refuses `arch.k` unless absent or `neutral`.
    pub fn refuse_int(&self, k: &str, neutral: u64) -> Result<()> {
        if let Some(v) = self.get(k) {
            let name = format!("{}.{k}", self.arch);
            let v = self.int_at(&name, Some(v))?;
            ensure!(v == neutral, "{name} {v} is not supported");
        }
        Ok(())
    }
}

/// A token id from `tokenizer.ggml.{k}`.
pub(crate) fn token_id(meta: &BTreeMap<String, Value>, k: &str) -> Result<u32> {
    let k = format!("tokenizer.ggml.{k}");
    let v = meta
        .get(&k)
        .with_context(|| format!("GGUF metadata {k} missing"))?
        .as_u64()
        .with_context(|| format!("GGUF metadata {k} is not a non-negative integer"))?;
    u32::try_from(v).with_context(|| format!("{k} {v} is not a token id"))
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let embd = g.require("token_embd.weight")?;
        ensure!(embd.dims.len() == 2, "token_embd.weight is not a matrix");
        Self::from_metadata(&g.metadata, embd.dims[1])
    }

    /// The shape from a GGUF's metadata and the embedding's row count (the
    /// vocabulary; the tokenizer's list must have as many entries).
    /// Metadata lily's graph does not implement (RoPE scaling, attention
    /// scaling, per-layer embeddings, shared KV layers) is refused rather
    /// than ignored.
    pub fn from_metadata(meta: &BTreeMap<String, Value>, vocab: u64) -> Result<Self> {
        let arch = meta.get("general.architecture").and_then(Value::as_str);
        ensure!(arch == Some(ARCH), "GGUF architecture {arch:?}, expected {ARCH:?}");
        let m = Meta { meta, arch: ARCH };
        ensure!(vocab <= MAX_VOCAB as u64, "a vocabulary of {vocab} tokens");
        if let Some(t) = meta.get("tokenizer.ggml.tokens") {
            let Value::Array(t) = t else {
                anyhow::bail!("tokenizer.ggml.tokens is not an array")
            };
            ensure!(
                t.len() as u64 == vocab,
                "{} tokenizer tokens for an embedding of {vocab} rows",
                t.len()
            );
        }
        let n_layer = m.u("block_count", MAX_LAYERS)?;
        ensure!(n_layer > 0, "{ARCH}.block_count is 0");
        let swa = m.bools("attention.sliding_window_pattern", n_layer)?;
        let n_head_kv = m.per_layer("attention.head_count_kv", n_layer, 1024)?;
        // Semantics the graph does not implement.
        if let Some(v) = m.get("rope.scaling.type") {
            ensure!(
                v.as_str() == Some("none"),
                "{ARCH}.rope.scaling.type {v:?} is not supported"
            );
        }
        m.refuse_float("rope.scaling.factor", &[0.0, 1.0])?;
        m.refuse_float("attention.scale", &[0.0, 1.0])?;
        m.refuse_int("embedding_length_per_layer_input", 0)?;
        m.refuse_int("attention.shared_kv_layers", 0)?;
        m.refuse_float("expert_weights_scale", &[0.0, 1.0])?;
        // LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX.
        m.refuse_int("expert_gating_func", 1)?;
        let c = Self {
            n_layer,
            hidden: m.u("embedding_length", MAX_HIDDEN)?,
            vocab: vocab as usize,
            ff: m.u("feed_forward_length", 1 << 16)?,
            ff_expert: m.u("expert_feed_forward_length", 1 << 16)?,
            n_expert: m.u("expert_count", 1 << 12)?,
            n_expert_used: m.u("expert_used_count", 1 << 12)?,
            n_head: m.u("attention.head_count", 1024)?,
            swa,
            n_head_kv,
            head_dim: m.u("attention.key_length", 1 << 12)?,
            head_dim_swa: m.u("attention.key_length_swa", 1 << 12)?,
            window: m.u("attention.sliding_window", 1 << 20)?,
            rope_base: m.f("rope.freq_base")?,
            rope_base_swa: m.f("rope.freq_base_swa")?,
            rms_eps: m.f("attention.layer_norm_rms_epsilon")?,
            softcap: if m.get("final_logit_softcapping").is_some() {
                m.f("final_logit_softcapping")?
            } else {
                0.0
            },
            context_length: m.u("context_length", 1 << 30)?,
            bos: token_id(meta, "bos_token_id")?,
            eos: token_id(meta, "eos_token_id")?,
            prefill_rows: super::model::PREFILL_ROWS,
        };
        ensure!(
            m.u("attention.value_length", 1 << 12)? == c.head_dim
                && m.u("attention.value_length_swa", 1 << 12)? == c.head_dim_swa,
            "key and value widths differ"
        );
        ensure!(
            m.u("rope.dimension_count", 1 << 12)? == c.head_dim
                && m.u_or("rope.dimension_count_swa", 1 << 12, c.head_dim_swa)?
                    == c.head_dim_swa,
            "partial RoPE dimension counts (lily rotates whole heads, the \
             global layers' partial rotation through rope_freqs)"
        );
        c.validate()?;
        Ok(c)
    }

    /// Gemma 4 26B-A4B's shape as its GGUFs declare it (tests build
    /// variants from it).
    pub fn gemma4_26b_a4b() -> Self {
        let swa: Vec<bool> = (0..30).map(|il| il % 6 != 5).collect();
        Self {
            n_layer: 30,
            hidden: 2816,
            vocab: 262144,
            ff: 2112,
            ff_expert: 704,
            n_expert: 128,
            n_expert_used: 8,
            n_head: 16,
            n_head_kv: swa.iter().map(|&s| if s { 8 } else { 2 }).collect(),
            swa,
            head_dim: 512,
            head_dim_swa: 256,
            window: 1024,
            rope_base: 1e6,
            rope_base_swa: 1e4,
            rms_eps: 1e-6,
            softcap: 30.0,
            context_length: 262144,
            bos: 2,
            eos: 106,
            prefill_rows: super::model::PREFILL_ROWS,
        }
    }

    /// The shapes the kernels take, checked before anything is mapped.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_LAYERS).contains(&self.n_layer)
                && self.swa.len() == self.n_layer
                && self.n_head_kv.len() == self.n_layer,
            "{} layers, {} sliding-window flags, {} KV head counts",
            self.n_layer,
            self.swa.len(),
            self.n_head_kv.len()
        );
        ensure!(
            (1..=MAX_VOCAB).contains(&self.vocab),
            "a vocabulary of {}",
            self.vocab
        );
        // Row kernels: a thread per 4 values, at most 4096; the GEMVs and
        // GEMMs take whole 256-value blocks; the grouped experts 64-row
        // tiles.
        ensure!(
            (256..=MAX_HIDDEN).contains(&self.hidden)
                && self.hidden.is_multiple_of(256),
            "embedding_length {} (lily takes multiples of 256 up to {MAX_HIDDEN})",
            self.hidden
        );
        ensure!(
            self.ff > 0 && self.ff.is_multiple_of(64),
            "feed_forward_length {} (the prompt GEMM's down projection takes multiples of 64)",
            self.ff
        );
        // The decode down kernel reads IQ4_NL in pairs of 32-value blocks,
        // the grouped GEMMs 64-row tiles.
        ensure!(
            self.ff_expert > 0 && self.ff_expert.is_multiple_of(64),
            "expert_feed_forward_length {} (lily takes multiples of 64)",
            self.ff_expert
        );
        // The router: 8 experts a lane of one subgroup, at most 32 picks;
        // the grouped experts take a power of two used.
        ensure!(
            (1..=256).contains(&self.n_expert)
                && self.n_expert_used.is_power_of_two()
                && self.n_expert_used <= 32
                && self.n_expert_used <= self.n_expert,
            "{} experts, {} used (lily takes a power of two up to 32 of at most 256)",
            self.n_expert,
            self.n_expert_used
        );
        ensure!(
            [256, 512].contains(&self.head_dim)
                && [256, 512].contains(&self.head_dim_swa),
            "head widths {} / {} (the kernels take 256 and 512)",
            self.head_dim,
            self.head_dim_swa
        );
        // Attention: two query heads a subgroup, of one KV head.
        ensure!(self.n_head > 0 && self.n_head <= 128, "{} query heads", self.n_head);
        for il in 0..self.n_layer {
            let kv = self.n_head_kv[il];
            ensure!(
                kv > 0
                    && self.n_head.is_multiple_of(kv)
                    && (self.n_head / kv).is_multiple_of(2),
                "layer {il}: {} heads over {kv} KV heads (lily takes an even group)",
                self.n_head
            );
        }
        ensure!(
            self.swa.iter().any(|&s| !s),
            "no global layer (the draft model and the head read the last one)"
        );
        ensure!(
            (1..=1 << 16).contains(&self.window),
            "sliding window of {}",
            self.window
        );
        for (name, base) in [
            ("rope.freq_base", self.rope_base),
            ("rope.freq_base_swa", self.rope_base_swa),
        ] {
            ensure!(base.is_finite() && base > 1.0, "{name} {base}");
        }
        ensure!(
            self.rms_eps.is_finite() && self.rms_eps > 0.0 && self.rms_eps < 1.0,
            "rms epsilon {}",
            self.rms_eps
        );
        ensure!(
            self.softcap.is_finite() && self.softcap >= 0.0,
            "final_logit_softcapping {}",
            self.softcap
        );
        ensure!(self.context_length > 0, "context_length 0");
        for (name, id) in [("bos", self.bos), ("eos", self.eos)] {
            ensure!(
                (id as usize) < self.vocab,
                "{name} token {id} is outside the vocabulary of {}",
                self.vocab
            );
        }
        Ok(())
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

#[cfg(test)]
#[path = "../../tests/unit/gemma4/config.rs"]
mod tests;
