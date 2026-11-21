//! Inkling 2 Mini's shape, read from GGUF metadata (`inkling2_loop.*` for
//! mainline `llama.cpp` / `ggml-org` / `unsloth/Mainline_Support`, or
//! `inkling2.*` for `unsloth`'s root looped layout) and checked against what
//! the kernels implement.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};

use crate::gguf::{Gguf, Value};

/// Mainline `llama.cpp` GGUF architecture name for Inkling 2 Mini.
pub const ARCH_LOOP: &str = "inkling2_loop";
/// Non-looped or Unsloth-root GGUF architecture name.
pub const ARCH_PLAIN: &str = "inkling2";

/// Maximum logical layer visits accepted before sizing tables.
pub const MAX_LAYERS: usize = 256;
/// Maximum padded vocabulary accepted.
pub const MAX_VOCAB: usize = 1 << 21;
/// Maximum hidden width supported by the row kernels.
pub const MAX_HIDDEN: usize = 4096;

/// Which GGUF loop tensor naming convention the file uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopLayout {
    /// Non-looped `inkling2`.
    None,
    /// Mainline `inkling2_loop` (`blk.N.loop_msg_query`, `blk.N.loop_{q,k,v,ffn}_ctrl_*`).
    Mainline,
    /// Unsloth root `inkling2` (`loop_res_query`, `loop_kv_res_query`, `blk.N.loop_{attn,k,v,mlp}_*`).
    Unsloth,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub loop_layout: LoopLayout,
    /// Number of physical layers (24 for Inkling 2 Mini).
    pub n_phys: usize,
    /// Number of logical layer visits (48 for Inkling 2 Mini).
    pub n_visits: usize,
    /// Loop core start physical layer (6), end exclusive (18), passes (3), gate rank (64).
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    pub loop_gate_rank: usize,
    /// Hidden dimension (2304).
    pub hidden: usize,
    /// Padded vocabulary size (201024; rows in `token_embd` and `output`).
    pub vocab_padded: usize,
    /// Unpadded vocabulary size (200058; tokenizer size and output logits).
    pub vocab_unpadded: usize,
    /// Per physical layer: attention query head count (24).
    pub n_head: Vec<usize>,
    /// Per physical layer: attention KV head count (8).
    pub n_head_kv: Vec<usize>,
    /// Head dimension for global and SWA layers (128).
    pub head_dim: usize,
    pub head_dim_swa: usize,
    /// Per physical layer: true for sliding-window attention, false for global.
    pub swa: Vec<bool>,
    /// Sliding window size (1024).
    pub window: usize,
    /// Relative-position bias projection width (16).
    pub rel_len: usize,
    /// Relative-position bias table extent for global layers (1024).
    pub rel_extent: usize,
    /// L2 clip thresholds for Q/K (22.627417) and R (8.0).
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// Log-scale attention floor (8192) and alpha (0.1) on global layers.
    pub log_scale_floor: usize,
    pub log_scale_alpha: f32,
    /// Causal short-convolution kernel width (4; state holds `sconv_l_cache - 1` = 3).
    pub sconv_l_cache: usize,
    /// Block attention-residual period (6 layers per block).
    pub attn_res_block_size: usize,
    /// MoE parameters: 128 experts, top 8, width 1152, route_scale 8.0.
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_expert: usize,
    pub route_scale: f32,
    /// SITU activation softcaps: `situ_beta` (4.0) and `situ_linear_beta` (25.0).
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    /// Final logit softcapping (30.0; 0.0 disables).
    pub final_logit_softcapping: f32,
    /// RMSNorm epsilon (1e-6).
    pub rms_eps: f32,
    /// Maximum context length advertised by metadata (1048576).
    pub context_length: usize,
    pub eos: u32,
    pub pad: u32,
}

impl Config {
    /// Reads and validates an Inkling 2 / Inkling 2 Loop configuration from GGUF metadata.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        Self::from_metadata(&gguf.metadata)
    }

    pub fn from_metadata(meta: &BTreeMap<String, Value>) -> Result<Self> {
        let arch = meta
            .get("general.architecture")
            .and_then(Value::as_str)
            .context("GGUF metadata general.architecture missing")?;
        ensure!(
            arch == ARCH_LOOP || arch == ARCH_PLAIN,
            "expected general.architecture = {ARCH_LOOP:?} or {ARCH_PLAIN:?}, got {arch:?}"
        );
        let m = Meta { meta, arch };

        let block_count = m.u("block_count", MAX_LAYERS)?;
        ensure!(block_count > 0, "{arch}.block_count must be nonzero");

        let (
            loop_layout,
            n_phys,
            n_visits,
            loop_start,
            loop_end,
            loop_count,
            loop_gate_rank,
        ) = if arch == ARCH_LOOP {
            let start = m.u("loop.start", block_count)?;
            let end = m.u("loop.end", block_count)?;
            let count = m.u("loop.count", 16)?;
            let rank = m.u("loop.gate_rank", 1024)?;
            let varying = m.bool_or("loop.varying", false)?;
            ensure!(!varying, "inkling2_loop.loop.varying=true is not supported");
            ensure!(
                count >= 2
                    && start > 0
                    && start < end
                    && end <= block_count
                    && rank > 0,
                "invalid inkling2_loop parameters: start={start} end={end} count={count} rank={rank}"
            );
            let n_visits = block_count + (count - 1) * (end - start);
            ensure!(
                n_visits <= MAX_LAYERS,
                "loop visits {n_visits} exceed {MAX_LAYERS}"
            );
            (LoopLayout::Mainline, block_count, n_visits, start, end, count, rank)
        } else if m.get("loop.core_end").is_some() {
            // Unsloth root looped layout: block_count is total visits.
            let start = m.u("loop.core_start", block_count)?;
            let end = m.u("loop.core_end", block_count)?;
            let count = m.u("num_loops", 16)?;
            let rank = m.u("loop.gate_rank", 1024)?;
            ensure!(
                count >= 2 && start > 0 && start < end && rank > 0,
                "invalid inkling2 loop parameters: start={start} end={end} count={count} rank={rank}"
            );
            let extra = (count - 1) * (end - start);
            ensure!(
                block_count > extra && end <= block_count - extra,
                "inkling2 block_count={block_count} too small for loop [{start},{end}) x {count}"
            );
            let n_phys = block_count - extra;
            (LoopLayout::Unsloth, n_phys, block_count, start, end, count, rank)
        } else {
            (LoopLayout::None, block_count, block_count, 0, 0, 1, 0)
        };

        let hidden = m.u("embedding_length", MAX_HIDDEN)?;
        ensure!(
            hidden > 0 && hidden.is_multiple_of(64),
            "embedding_length {hidden} must be a positive multiple of 64"
        );

        let vocab_padded = m.u("vocab_size", MAX_VOCAB)?;
        let vocab_unpadded = m.u("vocab_size_unpadded", vocab_padded)?;
        ensure!(
            vocab_unpadded > 0 && vocab_unpadded <= vocab_padded,
            "invalid vocab sizes {vocab_unpadded}/{vocab_padded}"
        );

        let head_dim = m.u("attention.key_length", 512)?;
        let val_dim = m.u("attention.value_length", 512)?;
        ensure!(
            head_dim == val_dim && head_dim > 0 && head_dim.is_multiple_of(32),
            "unsupported head_dim {head_dim}/{val_dim}"
        );
        let head_dim_swa = m.u_or("attention.key_length_swa", 512, head_dim)?;
        let val_dim_swa = m.u_or("attention.value_length_swa", 512, head_dim_swa)?;
        ensure!(
            head_dim_swa == val_dim_swa && head_dim_swa == head_dim,
            "differing SWA head_dim {head_dim_swa} vs {head_dim} not supported"
        );

        let n_arr = if loop_layout == LoopLayout::Unsloth { n_visits } else { n_phys };
        let n_head_raw = m.per_layer("attention.head_count", n_arr, 256)?;
        let n_head_kv_raw = m.per_layer("attention.head_count_kv", n_arr, 256)?;
        let swa_raw = m.bools("attention.sliding_window_pattern", n_arr)?;

        // Map to physical layers.
        let mut n_head = vec![0usize; n_phys];
        let mut n_head_kv = vec![0usize; n_phys];
        let mut swa = vec![false; n_phys];
        if loop_layout == LoopLayout::Unsloth {
            for p in 0..n_phys {
                let v = if p < loop_end {
                    p
                } else {
                    p + (loop_count - 1) * (loop_end - loop_start)
                };
                n_head[p] = n_head_raw[v];
                n_head_kv[p] = n_head_kv_raw[v];
                swa[p] = swa_raw[v];
            }
        } else {
            n_head.copy_from_slice(&n_head_raw);
            n_head_kv.copy_from_slice(&n_head_kv_raw);
            swa.copy_from_slice(&swa_raw);
        }

        for p in 0..n_phys {
            ensure!(
                n_head[p] > 0
                    && n_head_kv[p] > 0
                    && n_head[p].is_multiple_of(n_head_kv[p]),
                "layer {p}: invalid head counts {}/{}",
                n_head[p],
                n_head_kv[p]
            );
        }

        let window = m.u("attention.sliding_window", 1 << 20)?;
        let rel_len = m.u("attention.relative_length", 128)?;
        let rel_extent = m.u("attention.relative_extent", 1 << 20)?;
        let qk_clip = m.f("attention.qk_clip")?;
        let rel_clip = m.f("attention.relative_clip")?;
        let log_scale_floor = m.u("attention.log_scale_floor", 1 << 24)?;
        let log_scale_alpha = m.f("attention.log_scale_alpha")?;
        let sconv_l_cache = m.u("shortconv.l_cache", 64)?;
        let attn_res_block_size = m.u("attn_res.block_size", MAX_LAYERS)?;
        let rms_eps = m.f("attention.layer_norm_rms_epsilon")?;

        ensure!(
            window > 0
                && rel_len > 0
                && rel_extent > 0
                && log_scale_floor > 0
                && sconv_l_cache >= 2
                && attn_res_block_size > 0
                && qk_clip > 0.0
                && rel_clip > 0.0
                && rms_eps > 0.0
                && log_scale_alpha >= 0.0,
            "invalid Inkling 2 attention/conv parameters"
        );
        if loop_count > 1 {
            ensure!(
                loop_start.is_multiple_of(attn_res_block_size)
                    && (loop_end - loop_start).is_multiple_of(attn_res_block_size),
                "loop core [{loop_start}, {loop_end}) must align with attn_res_block_size={attn_res_block_size}"
            );
        }

        let n_expert = m.u("expert_count", 1024)?;
        let n_expert_used = m.u("expert_used_count", n_expert)?;
        let ff_expert = m.u_or_first("expert_feed_forward_length", 65536)?;
        let route_scale = m.f_or("expert_weights_scale", 1.0)?;
        let situ_beta = m.f("activation.situ_beta")?;
        let situ_linear_beta = m.f("activation.situ_linear_beta")?;
        let final_logit_softcapping = m.f_or("final_logit_softcapping", 0.0)?;
        let context_length = m.u("context_length", 1 << 24)?;

        ensure!(
            n_expert > 0
                && n_expert_used > 0
                && n_expert_used <= n_expert
                && ff_expert > 0
                && situ_beta > 0.0
                && situ_linear_beta > 0.0,
            "invalid Inkling 2 MoE/SITU parameters"
        );

        let eos = meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(200_006) as u32;
        let pad = meta
            .get("tokenizer.ggml.padding_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(199_999) as u32;

        Ok(Self {
            arch: arch.to_owned(),
            loop_layout,
            n_phys,
            n_visits,
            loop_start,
            loop_end,
            loop_count,
            loop_gate_rank,
            hidden,
            vocab_padded,
            vocab_unpadded,
            n_head,
            n_head_kv,
            head_dim,
            head_dim_swa,
            swa,
            window,
            rel_len,
            rel_extent,
            qk_clip,
            rel_clip,
            log_scale_floor,
            log_scale_alpha,
            sconv_l_cache,
            attn_res_block_size,
            n_expert,
            n_expert_used,
            ff_expert,
            route_scale,
            situ_beta,
            situ_linear_beta,
            final_logit_softcapping,
            rms_eps,
            context_length,
            eos,
            pad,
        })
    }

    /// Returns the execution schedule: `(cache_slot, physical_layer, pass)` for
    /// each of the `n_visits` logical layer visits in forward-pass order.
    ///
    /// Matches `llama_model_inkling2_loop::graph`:
    /// - Prelude `0..loop_start`: slot = p, pass = 0
    /// - Loop core `pass in 0..loop_count`, `p in loop_start..loop_end`:
    ///   slot = if pass == 0 { p } else { n_phys + (pass - 1) * core_len + (p - loop_start) }
    /// - Coda `loop_end..n_phys`: slot = p, pass = 0
    pub fn schedule(&self) -> Vec<(usize, usize, usize)> {
        if self.loop_count <= 1 {
            return (0..self.n_phys).map(|p| (p, p, 0)).collect();
        }
        let core_len = self.loop_end - self.loop_start;
        let mut out = Vec::with_capacity(self.n_visits);
        for p in 0..self.loop_start {
            out.push((p, p, 0));
        }
        for pass in 0..self.loop_count {
            for p in self.loop_start..self.loop_end {
                let slot = if pass == 0 {
                    p
                } else {
                    self.n_phys + (pass - 1) * core_len + (p - self.loop_start)
                };
                out.push((slot, p, pass));
            }
        }
        for p in self.loop_end..self.n_phys {
            out.push((p, p, 0));
        }
        out
    }

    /// Physical layer index owning cache slot `slot` (`0..n_visits`).
    pub fn phys_of_slot(&self, slot: usize) -> usize {
        if slot < self.n_phys || self.loop_count <= 1 {
            slot
        } else {
            self.loop_start + (slot - self.n_phys) % (self.loop_end - self.loop_start)
        }
    }

    pub fn q_width(&self, p: usize) -> usize {
        self.n_head[p] * self.head_dim
    }

    pub fn kv_width(&self, p: usize) -> usize {
        self.n_head_kv[p] * self.head_dim
    }

    pub fn rel_extent_of(&self, p: usize) -> usize {
        if self.swa[p] { self.window } else { self.rel_extent }
    }

    /// Total short-convolution channels stored per slot: K, V, attn_post, ffn_post.
    pub fn sconv_channels(&self, p: usize) -> usize {
        2 * self.kv_width(p) + 2 * self.hidden
    }
}

struct Meta<'a> {
    meta: &'a BTreeMap<String, Value>,
    arch: &'a str,
}

impl Meta<'_> {
    fn get(&self, k: &str) -> Option<&Value> {
        self.meta.get(&format!("{}.{k}", self.arch))
    }

    fn u(&self, k: &str, max: usize) -> Result<usize> {
        let name = format!("{}.{k}", self.arch);
        let v = self
            .get(k)
            .with_context(|| format!("GGUF metadata {name} missing"))?
            .as_u64()
            .with_context(|| format!("GGUF metadata {name} is not an integer"))?;
        ensure!(v <= max as u64, "{name} is {v}, max {max}");
        Ok(v as usize)
    }

    fn u_or(&self, k: &str, max: usize, default: usize) -> Result<usize> {
        if self.get(k).is_some() { self.u(k, max) } else { Ok(default) }
    }

    fn u_or_first(&self, k: &str, max: usize) -> Result<usize> {
        let name = format!("{}.{k}", self.arch);
        match self.get(k) {
            Some(Value::Array(a)) => {
                let first = a
                    .first()
                    .and_then(Value::as_u64)
                    .with_context(|| format!("{name} array empty or non-integer"))?;
                ensure!(first <= max as u64, "{name} is {first}, max {max}");
                for v in a {
                    ensure!(v.as_u64() == Some(first), "{name} varies across layers");
                }
                Ok(first as usize)
            }
            Some(_) => self.u(k, max),
            None => bail!("GGUF metadata {name} missing"),
        }
    }

    fn f(&self, k: &str) -> Result<f32> {
        let name = format!("{}.{k}", self.arch);
        let v = self
            .get(k)
            .with_context(|| format!("GGUF metadata {name} missing"))?
            .as_f32()
            .with_context(|| format!("GGUF metadata {name} is not a float"))?;
        ensure!(v.is_finite(), "{name} is {v}");
        Ok(v)
    }

    fn f_or(&self, k: &str, default: f32) -> Result<f32> {
        if self.get(k).is_some() { self.f(k) } else { Ok(default) }
    }

    fn bool_or(&self, k: &str, default: bool) -> Result<bool> {
        let name = format!("{}.{k}", self.arch);
        match self.get(k) {
            Some(Value::Bool(b)) => Ok(*b),
            Some(other) => bail!("{name} is {other:?}, expected bool"),
            None => Ok(default),
        }
    }

    fn bools(&self, k: &str, n: usize) -> Result<Vec<bool>> {
        let name = format!("{}.{k}", self.arch);
        match self.get(k) {
            Some(Value::Array(a)) => {
                ensure!(a.len() == n, "{name} has {} entries, expected {n}", a.len());
                a.iter()
                    .map(|v| match v {
                        Value::Bool(b) => Ok(*b),
                        other => bail!("{name} holds {other:?}"),
                    })
                    .collect()
            }
            Some(other) => bail!("{name} is {other:?}, expected bool array"),
            None => bail!("GGUF metadata {name} missing"),
        }
    }

    fn per_layer(&self, k: &str, n: usize, max: usize) -> Result<Vec<usize>> {
        let name = format!("{}.{k}", self.arch);
        match self.get(k) {
            Some(Value::Array(a)) => {
                ensure!(a.len() == n, "{name} has {} entries, expected {n}", a.len());
                a.iter()
                    .map(|v| {
                        let x = v.as_u64().with_context(|| {
                            format!("{name} entry {v:?} is not an integer")
                        })?;
                        ensure!(x <= max as u64, "{name} entry {x} exceeds {max}");
                        Ok(x as usize)
                    })
                    .collect()
            }
            Some(_) => Ok(vec![self.u(k, max)?; n]),
            None => bail!("GGUF metadata {name} missing"),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/inkling2/config.rs"]
mod tests;
