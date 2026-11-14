//! Qwen3.8-Flash-Next from a GGUF file in llama.cpp's `qwen4exp` schema
//! (antirez's ds4 publishes it at Q2: IQ2_XXS gate/up and Q2_K down
//! experts, Q8_0 projections, F16 hyper-connection mixers, F32 norms and
//! routers). The blocks are read in place (`kernels::qwen_gguf`), except
//! where the file's layout differs from what lily's graph reads, which the
//! loader undoes with exact moves of whole rows or blocks, or with values
//! it checks round-trip:
//!
//! - The converter (llama.cpp's `conversion/qwen.py`,
//!   `_LinearAttentionVReorderBase`) reorders the GDN value heads from
//!   grouped by key head (v head `h` reads key head `h / 3`, lily's
//!   kernels) to tiled (key head `h % 16`): the v rows of `attn_qkv`,
//!   `attn_gate`, `ssm_alpha`/`ssm_beta`, `ssm_a`, `ssm_dt.bias`, the v
//!   channels of `ssm_conv1d` and the input columns of `ssm_out`. The loader
//!   puts them back in grouped order.
//! - Zero-centred norm weights are stored folded, `1 + w`; lily keeps `w`
//!   in bfloat16 and adds the 1 in its kernels. `w` is exact in bfloat16
//!   (checked per value), so the gains are the file's.
//! - `ssm_a` is `-exp(A_log)`; lily's kernels take `A_log`.
//! - Conv weights are channel-major; lily's are tap-major.
//! - The draft head's `fc_embedding | fc_hidden` are one `eh_proj` with
//!   `[e ; h]` input columns: two column views of it.
//!
//! The 95 GB BF16 n-gram table at the end of the file is not mapped. Its
//! rows are read on demand from lily's own checkpoint of the model (the
//! same table at 4 bits, `LILY_NGRAM_TABLE`: a q4-xl directory), whose
//! hashing constants must equal the file's.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context as _, Result, bail, ensure};
use half::bf16;

use super::config::{
    EosTokenIds, GateAct, IndexerConfig, LayerType, MtpConfig, PleConfig,
    QuantizationConfig, Qwen4ExpConfig, RopeParameters, StoragePolicy,
};
use super::ngram::{self, NgramTable, PagedTable};
use super::weights::{
    AttnWeights, GdnWeights, HcWeights, IndexerWeights, LayerWeights, Mixer,
    ModelWeights, MtpWeights, PleWeights,
};
use crate::gguf::{GgmlType, Gguf, TensorInfo, Value};
use crate::kernels::qwen_gguf::QWeight;
use crate::metal::{Buffer, MetalContext};
use crate::safetensors::Checkpoint;
use crate::tensor::{DType, Tensor};
use crate::weights::{GgufLinear, LinearWeights, MlpWeights, MoeWeights};

/// The GGUF architecture name.
pub const ARCH: &str = "qwen4exp";

/// The n-gram table, which is never mapped.
const NGRAM_TABLE: &str = "per_layer_token_embd.weight";

/// Tensors read and rearranged into owned buffers instead of mapped.
const COPIED: &[&str] = &[
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_out.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_inp_shexp.weight",
];

/// Mapped regions of at most this many bytes (a tensor larger than it gets
/// a region of its own).
const REGION_BYTES: u64 = 1 << 30;

/// Whether `path` is a GGUF of this architecture.
pub fn is_qwen_gguf(path: &Path) -> bool {
    crate::gguf::is_gguf(path)
        && Gguf::open(path).is_ok_and(|g| g.architecture().is_ok_and(|a| a == ARCH))
}

fn u64s(g: &Gguf, key: &str) -> Result<Vec<u64>> {
    match g.get(key) {
        Some(Value::Array(v)) => v
            .iter()
            .map(|x| x.as_u64().context("non-integer entry"))
            .collect::<Result<_>>()
            .with_context(|| format!("reading {key}")),
        _ => bail!("GGUF metadata {key} missing"),
    }
}

fn key(g: &Gguf, k: &str) -> Result<usize> {
    Ok(usize::try_from(g.u64(&format!("{ARCH}.{k}"))?)?)
}

/// The model's configuration from the file's metadata. Qwen3.8-Flash-Next
/// facts the schema has no key for (the M-RoPE sections, the sigmoid GDN
/// gate, the renormalized top-k) are its HF config's.
pub fn config_from_gguf(g: &Gguf) -> Result<Qwen4ExpConfig> {
    ensure!(g.architecture()? == ARCH, "not a {ARCH} GGUF");
    let blocks = key(g, "block_count")?;
    let nextn = key(g, "nextn_predict_layers").unwrap_or(0);
    let mut layers = blocks - nextn;
    let mut nextn = nextn;
    // Diagnostic: the first N trunk layers only, without the draft head
    // (a partly downloaded file, a quick run).
    if let Some(n) = std::env::var("LILY_QWEN_LAYERS").ok().and_then(|v| v.parse().ok())
    {
        layers = layers.min(n);
        nextn = 0;
    }
    let interval = key(g, "full_attention_interval")?;
    let layer_types = (0..layers)
        .map(|i| {
            if (i + 1) % interval == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    let head_dim = key(g, "attention.key_length")?;
    ensure!(head_dim == key(g, "attention.value_length")?, "k and v head dims differ");
    let rope_dim = key(g, "rope.dimension_count")?;
    let theta = g.f32(&format!("{ARCH}.rope.freq_base"))?;
    let n_k = key(g, "ssm.group_count")?;
    let n_v = key(g, "ssm.time_step_rank")?;
    let d_state = key(g, "ssm.state_size")?;
    ensure!(key(g, "ssm.inner_size")? == n_v * d_state, "GDN inner size");
    let ratios = u64s(g, &format!("{ARCH}.attention.compress_ratios"))?;
    let ratio = ratios.iter().copied().find(|&r| r > 0).context("no QSA layer")?;
    let tok_eos = usize::try_from(g.u64("tokenizer.ggml.eos_token_id")?)? as u32;
    let ple_layers = u64s(g, &format!("{ARCH}.ple.layers"))?;
    let ple = match ple_layers.as_slice() {
        [] => None,
        &[layer] => {
            let heads_per = key(g, "ple.heads_per_ngram")?;
            let ngram = key(g, "ple.ngram_size")?;
            let dim = key(g, "embedding_length_per_layer_input")?;
            Some(PleConfig {
                layer: usize::try_from(layer)?,
                embed_dim: dim * (ngram - 1) * heads_per,
                conv_kernel_size: key(g, "ple.conv_kernel")?,
                ngram_size: ngram,
                heads_per_ngram: heads_per,
                eos_token_id: key(g, "ple.eos_token_id")? as u32,
                layer_multipliers: u64s(g, &format!("{ARCH}.ple.layer_multipliers"))?,
                head_vocab_sizes: u64s(g, &format!("{ARCH}.ple.head_vocab_sizes"))?,
                head_offsets: u64s(g, &format!("{ARCH}.ple.head_offsets"))?,
                padded_vocab_size: key(g, "ple.row_count")?,
                // The table this loader pages: lily's q4-xl one (checked
                // against its own config at load).
                table_shards: 0,
                quantization: QuantizationConfig { group_size: 32, bits: 4 },
            })
        }
        more => bail!("only one PLE layer is supported, got {more:?}"),
    };
    let pad = g
        .get("tokenizer.ggml.padding_token_id")
        .and_then(Value::as_u64)
        .map(|v| v as u32);
    let mut eos = vec![tok_eos];
    if let Some(p) = &ple
        && p.eos_token_id != tok_eos
    {
        eos.push(p.eos_token_id);
    }
    let _ = pad;
    let config = Qwen4ExpConfig {
        hidden_size: key(g, "embedding_length")?,
        num_hidden_layers: layers,
        layer_types,
        vocab_size: key(g, "vocab_size")?,
        rms_norm_eps: g.f32(&format!("{ARCH}.attention.layer_norm_rms_epsilon"))?,
        max_position_embeddings: key(g, "context_length")?,
        eos_token_id: EosTokenIds::Many(eos),
        num_experts: key(g, "expert_count")?,
        num_experts_per_tok: key(g, "expert_used_count")?,
        moe_intermediate_size: key(g, "expert_feed_forward_length")?,
        shared_expert_intermediate_size: key(g, "expert_shared_feed_forward_length")?,
        norm_topk_prob: true,
        num_attention_heads: key(g, "attention.head_count")?,
        num_key_value_heads: key(g, "attention.head_count_kv")?,
        head_dim,
        rope_parameters: RopeParameters {
            rope_theta: theta,
            partial_rotary_factor: rope_dim as f32 / head_dim as f32,
            mrope_section: Some(super::config::MROPE_SECTION.to_vec()),
            mrope_interleaved: Some(true),
        },
        linear_num_key_heads: n_k,
        linear_num_value_heads: n_v,
        linear_key_head_dim: d_state,
        linear_value_head_dim: d_state,
        linear_conv_kernel_dim: key(g, "ssm.conv_kernel")?,
        gdn_gate: GateAct::Sigmoid,
        hc_count: key(g, "hyper_connection.count")?,
        hc_lowrank: key(g, "hyper_connection.low_rank")?,
        ple,
        indexer: IndexerConfig {
            n_heads: key(g, "attention.indexer.head_count")?,
            head_dim: key(g, "attention.indexer.key_length")?,
            budget: key(g, "attention.indexer.top_k")?,
            compress_ratio: usize::try_from(ratio)?,
        },
        quantization: QuantizationConfig { group_size: 64, bits: 4 },
        storage: StoragePolicy::Q4,
        mtp: (nextn > 0).then_some(MtpConfig { layers: nextn, rope_theta: theta }),
        vision: None,
        gguf_down_cols: Some(g.require("blk.0.ffn_down_exps.weight")?.row_len()),
    };
    config.validate_flash_next()?;
    Ok(config)
}

/// The file's tensors in mapped regions (all but the n-gram table and the
/// tensors the loader rearranges into buffers of its own).
struct Mapped {
    tensors: std::collections::HashMap<String, (Buffer, usize, TensorInfo)>,
}

impl Mapped {
    /// Maps every tensor but the n-gram table, the copied ones, and the
    /// blocks from `first_unused` on (the draft head when it is not loaded,
    /// or the layers `LILY_QWEN_LAYERS` leaves out).
    fn map(
        ctx: &MetalContext,
        g: &Gguf,
        path: &Path,
        first_unused: usize,
    ) -> Result<Self> {
        let unused = |name: &str| {
            name.strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|il| il >= first_unused)
        };
        let skip = |t: &TensorInfo| {
            t.name == NGRAM_TABLE
                || COPIED.iter().any(|c| t.name.ends_with(c))
                || unused(&t.name)
        };
        let mut order: Vec<&TensorInfo> = g.tensors.iter().collect();
        order.sort_by_key(|t| t.offset);
        let mut tensors = std::collections::HashMap::new();
        let mut i = 0;
        while i < order.len() {
            if skip(order[i]) {
                i += 1;
                continue;
            }
            let start = order[i].offset;
            let mut end = start + order[i].bytes;
            let mut j = i + 1;
            while j < order.len()
                && !skip(order[j])
                && order[j].offset + order[j].bytes - start <= REGION_BYTES
            {
                end = order[j].offset + order[j].bytes;
                j += 1;
            }
            let (buf, inner) = ctx
                .new_buffer_mapped(path, start, usize::try_from(end - start)?)
                .with_context(|| format!("mapping {} bytes at {start}", end - start))?;
            for t in &order[i..j] {
                let at = inner + usize::try_from(t.offset - start)?;
                tensors.insert(t.name.clone(), (buf.clone(), at, (*t).clone()));
            }
            i = j;
        }
        Ok(Self { tensors })
    }

    fn get(&self, name: &str) -> Result<&(Buffer, usize, TensorInfo)> {
        self.tensors.get(name).with_context(|| format!("GGUF has no tensor {name}"))
    }

    fn q(&self, name: &str) -> Result<QWeight> {
        let (buf, at, t) = self.get(name)?;
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length()], DType::I8)?;
        let view = whole.view(*at, &[t.bytes as usize])?;
        QWeight::new(view, t.ty, t.row_len(), t.rows())
    }

    fn lin(&self, names: &[&str]) -> Result<LinearWeights> {
        Ok(LinearWeights::Gguf(GgufLinear {
            parts: names.iter().map(|n| self.q(n)).collect::<Result<_>>()?,
        }))
    }
}

/// The f32 values of a small F32 tensor.
fn f32s(g: &Gguf, name: &str) -> Result<(Vec<f32>, TensorInfo)> {
    let t = g.require(name)?.clone();
    ensure!(t.ty == GgmlType::F32, "{name} is {:?}, expected F32", t.ty);
    let raw = g.read_bytes(t.offset, t.bytes as usize)?;
    Ok((
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        t,
    ))
}

/// `v` as bfloat16, which must hold every value exactly.
fn exact_bf16(
    ctx: &MetalContext,
    v: &[f32],
    shape: &[usize],
    name: &str,
) -> Result<Tensor> {
    let b: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    if let Some(i) = b.iter().zip(v).position(|(b, &x)| b.to_f32() != x) {
        bail!("{name}[{i}] = {} is not a bfloat16 value", v[i]);
    }
    Tensor::from_bytes(ctx, bytemuck::cast_slice(&b), shape, DType::BF16)
}

/// F32 weight values as a BF16 GGUF weight (half the bytes a GEMV
/// streams) when every one is a bfloat16 value, as the checkpoint's routers
/// and GDN alpha/beta are (bf16 originals stored as F32); else as F32.
fn exact_bf16_weight(
    ctx: &MetalContext,
    v: &[f32],
    cols: usize,
    rows: usize,
) -> Result<QWeight> {
    let b: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    if b.iter().zip(v).all(|(b, &x)| b.to_f32() == x) {
        QWeight::from_bytes(ctx, bytemuck::cast_slice(&b), GgmlType::Bf16, cols, rows)
    } else {
        QWeight::from_bytes(ctx, bytemuck::cast_slice(v), GgmlType::F32, cols, rows)
    }
}

/// A small F32 GGUF weight, as [`exact_bf16_weight`] gives it.
fn small_weight(ctx: &MetalContext, g: &Gguf, name: &str) -> Result<LinearWeights> {
    let (v, t) = f32s(g, name)?;
    Ok(LinearWeights::Gguf(GgufLinear {
        parts: vec![exact_bf16_weight(ctx, &v, t.row_len(), t.rows())?],
    }))
}

/// A folded norm `1 + w` as lily's bfloat16 `w`.
fn norm(ctx: &MetalContext, g: &Gguf, name: &str) -> Result<Tensor> {
    let (v, _) = f32s(g, name)?;
    let w: Vec<f32> = v.iter().map(|&x| x - 1.0).collect();
    // `1 + w` was rounded to f32; `w` must come back exactly.
    if let Some(i) = w.iter().zip(&v).position(|(&w, &x)| w + 1.0 != x) {
        bail!("{name}[{i}] does not unfold exactly");
    }
    exact_bf16(ctx, &w, &[w.len()], name)
}

/// The grouped-order v head of tiled index... the tiled index of grouped
/// head `h` (`n_k` key heads, `r` value heads each).
fn tiled_of_grouped(h: usize, n_k: usize, r: usize) -> usize {
    (h % r) * n_k + h / r
}

/// `count` units of `unit` bytes: grouped unit `h` takes tiled unit
/// `tiled_of_grouped(h)`.
fn regroup(src: &[u8], unit: usize, n_k: usize, r: usize, dst: &mut [u8]) {
    for h in 0..n_k * r {
        let s = tiled_of_grouped(h, n_k, r);
        dst[h * unit..(h + 1) * unit].copy_from_slice(&src[s * unit..(s + 1) * unit]);
    }
}

/// A GDN projection read into a buffer of its own with its value heads in
/// grouped order: rows from `first_v_row` on are `heads` heads of `head_rows`
/// rows (`attn_qkv`, `attn_gate`), or, with `columns`, every row's columns
/// are heads of `head_rows` values (`ssm_out`).
fn regrouped(
    ctx: &MetalContext,
    g: &Gguf,
    name: &str,
    first_v_row: usize,
    head_rows: usize,
    n_k: usize,
    r: usize,
    columns: bool,
) -> Result<QWeight> {
    let t = g.require(name)?.clone();
    let raw = g.read_bytes(t.offset, t.bytes as usize)?;
    let mut out = raw.clone();
    let rb = t.row_bytes();
    if columns {
        let unit = t.ty.row_bytes(head_rows)?;
        ensure!(unit * n_k * r == rb, "{name}: rows are not {} heads", n_k * r);
        for (s, d) in raw.chunks_exact(rb).zip(out.chunks_exact_mut(rb)) {
            regroup(s, unit, n_k, r, d);
        }
    } else {
        let at = first_v_row * rb;
        ensure!(
            t.rows() == first_v_row + n_k * r * head_rows,
            "{name}: {} rows are not {first_v_row} + {} heads",
            t.rows(),
            n_k * r
        );
        regroup(&raw[at..], head_rows * rb, n_k, r, &mut out[at..]);
    }
    let buf = Tensor::from_bytes(ctx, &out, &[out.len()], DType::I8)?;
    QWeight::new(buf, t.ty, t.row_len(), t.rows())
}

/// A small F32 GDN weight of `n_k * r` heads of `head` values, regrouped.
fn regrouped_f32(
    g: &Gguf,
    name: &str,
    head: usize,
    n_k: usize,
    r: usize,
) -> Result<Vec<f32>> {
    let (v, _) = f32s(g, name)?;
    ensure!(v.len() == n_k * r * head, "{name} has {} values", v.len());
    let mut out = v.clone();
    regroup(
        bytemuck::cast_slice(&v),
        4 * head,
        n_k,
        r,
        bytemuck::cast_slice_mut(&mut out),
    );
    Ok(out)
}

/// A channel-major conv weight `[channels][taps]` as lily's tap-major
/// bfloat16 `[taps][channels]`.
fn conv_tap_major(
    ctx: &MetalContext,
    v: &[f32],
    channels: usize,
    taps: usize,
    name: &str,
) -> Result<Tensor> {
    ensure!(v.len() == channels * taps, "{name} has {} values", v.len());
    let mut t = vec![0f32; v.len()];
    for c in 0..channels {
        for k in 0..taps {
            t[k * channels + c] = v[c * taps + k];
        }
    }
    exact_bf16(ctx, &t, &[taps, channels], name)
}

struct Load<'a> {
    ctx: &'a MetalContext,
    g: &'a Gguf,
    m: &'a Mapped,
    cfg: &'a Qwen4ExpConfig,
}

impl Load<'_> {
    fn hc(&self, prefix: &str, side: &str, inject: bool) -> Result<HcWeights> {
        Ok(HcWeights {
            norm: norm(self.ctx, self.g, &format!("{prefix}{side}_norm.weight"))?,
            down: self.m.lin(&[&format!("{prefix}{side}_down.weight")])?,
            up: self.m.lin(&[&format!("{prefix}{side}_up.weight")])?,
            inject: if inject {
                Some(self.m.lin(&[&format!("{prefix}{side}_inject.weight")])?)
            } else {
                None
            },
        })
    }

    fn gdn(&self, b: &str) -> Result<GdnWeights> {
        let c = self.cfg;
        let (n_k, n_v) = (c.linear_num_key_heads, c.linear_num_value_heads);
        let r = n_v / n_k;
        let dv = c.linear_value_head_dim;
        let qk = 2 * n_k * c.linear_key_head_dim;
        let qkv = regrouped(
            self.ctx,
            self.g,
            &format!("{b}attn_qkv.weight"),
            qk,
            dv,
            n_k,
            r,
            false,
        )?;
        let z = regrouped(
            self.ctx,
            self.g,
            &format!("{b}attn_gate.weight"),
            0,
            dv,
            n_k,
            r,
            false,
        )?;
        let out = regrouped(
            self.ctx,
            self.g,
            &format!("{b}ssm_out.weight"),
            0,
            dv,
            n_k,
            r,
            true,
        )?;
        let ab = |name: &str| -> Result<QWeight> {
            let name = format!("{b}{name}.weight");
            let cols = self.g.require(&name)?.row_len();
            let w = regrouped_f32(self.g, &name, cols, n_k, r)?;
            exact_bf16_weight(self.ctx, &w, cols, n_v)
        };
        let in_proj = LinearWeights::Gguf(GgufLinear {
            parts: vec![qkv, z, ab("ssm_alpha")?, ab("ssm_beta")?],
        });
        let conv_c = c.gdn_conv_channels();
        let (conv, _) = f32s(self.g, &format!("{b}ssm_conv1d.weight"))?;
        let taps = c.linear_conv_kernel_dim;
        let mut conv_g = conv.clone();
        regroup(
            bytemuck::cast_slice(&conv[qk * taps..]),
            4 * dv * taps,
            n_k,
            r,
            bytemuck::cast_slice_mut(&mut conv_g[qk * taps..]),
        );
        let a = regrouped_f32(self.g, &format!("{b}ssm_a"), 1, n_k, r)?;
        let a_log: Vec<f32> = a
            .iter()
            .map(|&a| {
                ensure!(a < 0.0, "{b}ssm_a holds {a}, not -exp(A_log)");
                Ok((-f64::from(a)).ln() as f32)
            })
            .collect::<Result<_>>()?;
        let dt = regrouped_f32(self.g, &format!("{b}ssm_dt.bias"), 1, n_k, r)?;
        let (norm_w, _) = f32s(self.g, &format!("{b}ssm_norm.weight"))?;
        Ok(GdnWeights {
            in_proj_qkv: in_proj.view_rows(0, conv_c)?,
            in_proj_z: in_proj.view_rows(conv_c, n_v * dv)?,
            in_proj_a: in_proj.view_rows(conv_c + n_v * dv, n_v)?,
            in_proj_b: in_proj.view_rows(conv_c + n_v * dv + n_v, n_v)?,
            in_proj,
            conv_w: conv_tap_major(self.ctx, &conv_g, conv_c, taps, "ssm_conv1d")?,
            a_log: Tensor::from_f32(self.ctx, &a_log, &[n_v])?,
            dt_bias: exact_bf16(self.ctx, &dt, &[n_v], "ssm_dt.bias")?,
            norm_w: Tensor::from_f32(self.ctx, &norm_w, &[norm_w.len()])?,
            out_proj: LinearWeights::Gguf(GgufLinear { parts: vec![out] }),
        })
    }

    fn attn(&self, b: &str) -> Result<AttnWeights> {
        let n = |s: &str| format!("{b}{s}.weight");
        let qkv_proj = self.m.lin(&[&n("attn_q"), &n("attn_k"), &n("attn_v")])?;
        let c = self.cfg;
        let q_rows = 2 * c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        Ok(AttnWeights {
            q_proj: qkv_proj.view_rows(0, q_rows)?,
            k_proj: qkv_proj.view_rows(q_rows, kv)?,
            v_proj: qkv_proj.view_rows(q_rows + kv, kv)?,
            qkv_proj,
            q_norm: norm(self.ctx, self.g, &n("attn_q_norm"))?,
            k_norm: norm(self.ctx, self.g, &n("attn_k_norm"))?,
            o_proj: self.m.lin(&[&n("attn_output")])?,
            indexer: IndexerWeights {
                qk_proj: self.m.lin(&[&n("indexer.q_proj"), &n("indexer.k_proj")])?,
                q_norm: norm(self.ctx, self.g, &n("indexer.q_norm"))?,
                k_norm: norm(self.ctx, self.g, &n("indexer.k_norm"))?,
            },
        })
    }

    fn ffn(&self, b: &str) -> Result<Box<MoeWeights>> {
        let n = |s: &str| format!("{b}{s}.weight");
        let gate_up_proj = self.m.lin(&[&n("ffn_gate_shexp"), &n("ffn_up_shexp")])?;
        let i = self.cfg.shared_expert_intermediate_size;
        let w = Box::new(MoeWeights {
            gate: small_weight(self.ctx, self.g, &n("ffn_gate_inp"))?,
            expert_gate: self.m.lin(&[&n("ffn_gate_exps")])?,
            expert_up: self.m.lin(&[&n("ffn_up_exps")])?,
            expert_down: self.m.lin(&[&n("ffn_down_exps")])?,
            slot_of: None,
            cache: None,
            shared: MlpWeights {
                gate_proj: gate_up_proj.view_rows(0, i)?,
                up_proj: gate_up_proj.view_rows(i, i)?,
                gate_up_proj,
                down_proj: self.m.lin(&[&n("ffn_down_shexp")])?,
            },
            shared_gate: small_weight(self.ctx, self.g, &n("ffn_gate_inp_shexp"))?,
        });
        let (e, f, h) = (
            self.cfg.num_experts,
            self.cfg.moe_intermediate_size,
            self.cfg.hidden_size,
        );
        w.gate.expect_features(e, h, "router")?;
        w.expert_gate.expect_features(e * f, h, "expert gate")?;
        w.expert_up.expect_features(e * f, h, "expert up")?;
        ensure!(
            w.expert_down.out_features() == e * h && w.expert_down.in_features() >= f,
            "expert down is [{}, {}]",
            w.expert_down.out_features(),
            w.expert_down.in_features()
        );
        w.shared_gate.expect_features(1, h, "shared expert gate")?;
        Ok(w)
    }

    fn ple(&self, b: &str, table: NgramTable) -> Result<PleWeights> {
        let n = |s: &str| format!("{b}{s}.weight");
        let (conv, _) = f32s(self.g, &n("ple_conv1d"))?;
        let ple = self.cfg.ple.as_ref().context("PLE layer without PLE config")?;
        Ok(PleWeights {
            table,
            key_proj: self.m.lin(&[&n("ple_key")])?,
            value_proj: self.m.lin(&[&n("ple_value")])?,
            norm_key: norm(self.ctx, self.g, &n("ple_norm_key"))?,
            norm_query: norm(self.ctx, self.g, &n("ple_norm_query"))?,
            norm_conv: norm(self.ctx, self.g, &n("ple_norm_conv"))?,
            conv_w: conv_tap_major(
                self.ctx,
                &conv,
                self.cfg.hc_width(),
                ple.conv_kernel_size,
                "ple_conv1d",
            )?,
        })
    }

    fn layer(
        &self,
        il: usize,
        kind: LayerType,
        ple: Option<NgramTable>,
    ) -> Result<LayerWeights> {
        let b = format!("blk.{il}.");
        Ok(LayerWeights {
            ple: ple.map(|t| self.ple(&b, t).map(Box::new)).transpose()?,
            attn_hc: self.hc(&b, "hc_attn", true)?,
            mixer: match kind {
                LayerType::LinearAttention => Mixer::Gdn(Box::new(self.gdn(&b)?)),
                LayerType::FullAttention => Mixer::Attn(Box::new(self.attn(&b)?)),
            },
            mlp_hc: self.hc(&b, "hc_ffn", true)?,
            ffn: self.ffn(&b)?,
        })
    }

    fn mtp(&self, il: usize) -> Result<MtpWeights> {
        let b = format!("blk.{il}.");
        let h = self.cfg.hidden_size;
        let eh = self.m.q(&format!("{b}nextn.eh_proj.weight"))?;
        ensure!(
            eh.cols == 2 * h && eh.rows == h,
            "eh_proj is [{}, {}]",
            eh.rows,
            eh.cols
        );
        let half = eh.ty.row_bytes(h)?;
        let column_view = |at: usize| -> Result<LinearWeights> {
            Ok(LinearWeights::Gguf(GgufLinear {
                parts: vec![QWeight {
                    t: eh.t.view(at, &[eh.t.numel() - at])?,
                    ty: eh.ty,
                    cols: h,
                    rows: h,
                    row_bytes: eh.row_bytes,
                }],
            }))
        };
        Ok(MtpWeights {
            norm_embedding: norm(self.ctx, self.g, &format!("{b}nextn.enorm.weight"))?,
            norm_hidden: norm(self.ctx, self.g, &format!("{b}nextn.hnorm.weight"))?,
            fc_embedding: column_view(0)?,
            fc_hidden: column_view(half)?,
            layer: self.layer(il, LayerType::FullAttention, None)?,
            mixer: self.hc(&b, "nextn.hc_head", false)?,
            lm_head: None,
        })
    }
}

/// The paged n-gram table from lily's q4-xl checkpoint at `dir`, which must
/// hash as the GGUF's config says.
fn ngram_table(dir: &Path, ple: &PleConfig) -> Result<NgramTable> {
    let theirs = Qwen4ExpConfig::from_model_dir(dir)
        .with_context(|| format!("n-gram table checkpoint {}", dir.display()))?;
    let t = theirs.ple.context("the n-gram table checkpoint has no PLE")?;
    ensure!(
        t.layer_multipliers == ple.layer_multipliers
            && t.head_vocab_sizes == ple.head_vocab_sizes
            && t.head_offsets == ple.head_offsets
            && t.padded_vocab_size == ple.padded_vocab_size
            && t.embed_dim == ple.embed_dim
            && t.heads_per_ngram == ple.heads_per_ngram,
        "the n-gram table at {} hashes differently from the GGUF",
        dir.display()
    );
    ensure!(
        t.quantization.bits == PagedTable::BITS,
        "the n-gram table at {} is not {}-bit",
        dir.display(),
        PagedTable::BITS
    );
    let ckpt = Checkpoint::open(dir)?;
    let bases = ngram::shard_bases(
        &format!("model.language_model.layers.{}.ple.", t.layer),
        t.table_shards,
    );
    let table = PagedTable::open(&ckpt, &bases, t.quantization.group_size)?;
    ensure!(
        table.rows() == ple.padded_vocab_size && table.width() == ple.head_dim(),
        "the n-gram table is [{}, {}]",
        table.rows(),
        table.width()
    );
    Ok(NgramTable::Paged(Arc::new(table)))
}

/// Where the n-gram table comes from: `given`, else `LILY_NGRAM_TABLE` (a
/// q4-xl checkpoint directory).
pub fn ngram_table_dir(given: Option<&Path>) -> Result<PathBuf> {
    given.map(Path::to_path_buf).or_else(|| std::env::var_os("LILY_NGRAM_TABLE").map(PathBuf::from)).context(
        "the GGUF's BF16 n-gram table is not read yet: pass --ngram-checkpoint (or set \
         LILY_NGRAM_TABLE) to a lily Qwen3.8-Flash-Next checkpoint directory, whose \
         4-bit table is paged instead",
    )
}

/// Loads the model's weights from the GGUF at `path`, the draft head when
/// `with_mtp`. Takes the machine's instance lock first.
pub fn load(
    ctx: &MetalContext,
    path: &Path,
    config: &Qwen4ExpConfig,
    with_mtp: bool,
    ngram_checkpoint: Option<&Path>,
    memory_limit: Option<u64>,
) -> Result<ModelWeights> {
    crate::instance::acquire()?;
    let recording = ctx.record_buffers()?;
    let g = Gguf::open(path)?;
    let loads_mtp = with_mtp && config.mtp.is_some();
    let first_unused = config.num_hidden_layers + usize::from(loads_mtp);
    let mapped = Mapped::map(ctx, &g, path, first_unused)?;
    let l = Load { ctx, g: &g, m: &mapped, cfg: config };
    let table = match &config.ple {
        Some(p) => Some(ngram_table(&ngram_table_dir(ngram_checkpoint)?, p)?),
        None => None,
    };
    let mut table = table;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for (il, kind) in config.layer_types.iter().enumerate() {
        let ple = match &config.ple {
            Some(p) if p.layer == il => table.take(),
            _ => None,
        };
        layers.push(l.layer(il, *kind, ple)?);
    }
    let mtp = match (&config.mtp, with_mtp) {
        (Some(_), true) => Some(Box::new(l.mtp(config.num_hidden_layers)?)),
        _ => None,
    };
    let embed_tokens = mapped.lin(&["token_embd.weight"])?;
    let lm_head = mapped.lin(&["output.weight"])?;
    embed_tokens.expect_features(
        config.vocab_size,
        config.hidden_size,
        "token_embd",
    )?;
    lm_head.expect_features(config.vocab_size, config.hidden_size, "output")?;
    let final_mixer = l.hc("", "output_hc", false)?;
    drop(mapped);
    let buffers = recording.finish();
    Ok(ModelWeights {
        embed_tokens,
        lm_head,
        final_mixer,
        layers,
        mtp,
        vision: None,
        expert_cache: None,
        buffers,
        planned_memory: super::weights::physical_memory(),
        // `--memory-limit-gb` as is: the GGUF has no expert cache to plan,
        // so the limit caps the server's session cache (allocated + cache).
        memory_limit,
        session_reserve: None,
    })
}
