//! The tensors DeepSeek V4 Flash's kernels read, with the shape and the
//! block types each may have: [`validate`] checks a file's tensor table
//! against it before anything is mapped. The kernels index weights by raw
//! GPU address from the configured shape, so a tensor that is shorter than
//! its role, or of a type a kernel does not decode, must be refused here.

use anyhow::{Context as _, Result, bail, ensure};

use super::config::Config;
use crate::gguf::{GgmlType, Gguf, Value};

/// The quantized projections (attention, shared experts, output head).
const Q8_0: &[GgmlType] = &[GgmlType::Q8_0];
/// The small projections the published file keeps in F16 (HC controls,
/// compressors, indexer, router, embedding).
const F16: &[GgmlType] = &[GgmlType::F16];
/// Types the routed-expert kernels (per-pair and grouped) decode.
const EXPERT: &[GgmlType] = &[GgmlType::Iq2Xxs, GgmlType::Q2K];
const F32: &[GgmlType] = &[GgmlType::F32];
const I32: &[GgmlType] = &[GgmlType::I32];

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

/// Every tensor the model reads.
pub fn expected(c: &Config) -> Vec<Spec> {
    let h = c.hidden;
    let mut v = vec![
        spec("token_embd.weight".into(), &[h, c.vocab], F16),
        spec("output.weight".into(), &[h, c.vocab], Q8_0),
        spec("output_norm.weight".into(), &[h], F32),
        spec("output_hc_fn.weight".into(), &[c.hc_dim(), c.n_hc], F16),
        spec("output_hc_scale.weight".into(), &[1], F32),
        spec("output_hc_base.weight".into(), &[c.n_hc], F32),
    ];
    for il in 0..c.n_layer {
        let router = Router { types: F16, hash: c.is_hash_layer(il) };
        layer(&mut v, c, &format!("blk.{il}"), c.ratio(il), c.has_indexer(il), router);
    }
    v
}

/// How a layer routes: the gate's types, and whether a hash table or the
/// router's top-k with a bias picks its experts.
struct Router {
    types: &'static [GgmlType],
    hash: bool,
}

/// The tensors of one layer (trunk or DSpark stage) named `{prefix}.*`.
fn layer(
    v: &mut Vec<Spec>,
    c: &Config,
    prefix: &str,
    r: usize,
    indexer: bool,
    router: Router,
) {
    let h = c.hidden;
    let group_in = c.o_group_heads() * c.head_dim;
    let n = |s: &str| format!("{prefix}.{s}");
    // Ratio 4 compresses two overlapping lanes (Config::compress_lanes).
    let lanes = if r == 4 { 2 } else { 1 };
    v.push(spec(n("attn_norm.weight"), &[h], F32));
    v.push(spec(n("ffn_norm.weight"), &[h], F32));
    for p in ["attn", "ffn"] {
        v.push(spec(n(&format!("hc_{p}_fn.weight")), &[c.hc_dim(), c.hc_mix()], F16));
        v.push(spec(n(&format!("hc_{p}_scale.weight")), &[3], F32));
        v.push(spec(n(&format!("hc_{p}_base.weight")), &[c.hc_mix()], F32));
    }
    v.push(spec(n("attn_q_a.weight"), &[h, c.q_lora], Q8_0));
    v.push(spec(n("attn_q_a_norm.weight"), &[c.q_lora], F32));
    v.push(spec(n("attn_q_b.weight"), &[c.q_lora, c.q_dim()], Q8_0));
    v.push(spec(n("attn_kv.weight"), &[h, c.head_dim], Q8_0));
    v.push(spec(n("attn_kv_a_norm.weight"), &[c.head_dim], F32));
    v.push(spec(n("attn_sinks.weight"), &[c.n_head], F32));
    v.push(spec(n("attn_output_a.weight"), &[group_in, c.o_groups * c.o_lora], Q8_0));
    v.push(spec(n("attn_output_b.weight"), &[c.o_groups * c.o_lora, h], Q8_0));
    if r != 0 {
        let w = lanes * c.head_dim;
        v.push(spec(n("attn_compressor_kv.weight"), &[h, w], F16));
        v.push(spec(n("attn_compressor_gate.weight"), &[h, w], F16));
        v.push(spec(n("attn_compressor_ape.weight"), &[w, r], F16));
        v.push(spec(n("attn_compressor_norm.weight"), &[c.head_dim], F32));
    }
    if indexer {
        let w = lanes * c.indexer_dim;
        v.push(spec(n("indexer_compressor_kv.weight"), &[h, w], F16));
        v.push(spec(n("indexer_compressor_gate.weight"), &[h, w], F16));
        v.push(spec(n("indexer_compressor_ape.weight"), &[w, r], F16));
        v.push(spec(n("indexer_compressor_norm.weight"), &[c.indexer_dim], F32));
        v.push(spec(
            n("indexer.attn_q_b.weight"),
            &[c.q_lora, c.indexer_heads * c.indexer_dim],
            F16,
        ));
        v.push(spec(n("indexer.proj.weight"), &[h, c.indexer_heads], F16));
    }
    v.push(spec(n("ffn_gate_inp.weight"), &[h, c.n_expert], router.types));
    if router.hash {
        v.push(spec(n("ffn_gate_tid2eid.weight"), &[c.n_expert_used, c.vocab], I32));
    } else {
        v.push(spec(n("exp_probs_b.bias"), &[c.n_expert], F32));
    }
    for t in ["ffn_gate_exps", "ffn_up_exps"] {
        v.push(spec(n(&format!("{t}.weight")), &[h, c.ff_expert, c.n_expert], EXPERT));
    }
    v.push(spec(n("ffn_down_exps.weight"), &[c.ff_expert, h, c.n_expert], EXPERT));
    v.push(spec(n("ffn_gate_shexp.weight"), &[h, c.ff_expert], Q8_0));
    v.push(spec(n("ffn_up_shexp.weight"), &[h, c.ff_expert], Q8_0));
    v.push(spec(n("ffn_down_shexp.weight"), &[c.ff_expert, h], Q8_0));
}

/// The tensors each layer reads in one fused dispatch take one block type
/// for all of its weights.
pub fn fused_groups(c: &Config) -> Vec<Vec<String>> {
    (0..c.n_layer)
        .map(|il| {
            ["ffn_gate_exps", "ffn_up_exps"]
                .iter()
                .map(|s| format!("blk.{il}.{s}.weight"))
                .collect()
        })
        .collect()
}

/// Checks every tensor the model reads: present, of the expected rank and
/// dimensions, of a type its kernels decode, at an aligned offset; that the
/// tensors of each fused group share a type; and that the file has no
/// tensor the model would ignore (an unknown tensor means a different graph).
pub fn validate(g: &Gguf, c: &Config) -> Result<()> {
    let specs = expected(c);
    check_specs(g, &specs, &fused_groups(c))?;
    ensure!(
        g.tensors.len() == specs.len(),
        "the file has {} tensors, DeepSeek V4 Flash's graph reads {}",
        g.tensors.len(),
        specs.len()
    );
    Ok(())
}

/// Each spec present, of its rank, dimensions and types, aligned; each
/// fused group of one type.
fn check_specs(g: &Gguf, specs: &[Spec], fused: &[Vec<String>]) -> Result<()> {
    for s in specs {
        let t = g.require(&s.name)?;
        ensure!(
            t.dims == s.dims,
            "tensor {} has dimensions {:?}, DeepSeek V4 Flash needs {:?}",
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
    for names in fused {
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

/// The GGUF architecture of DSpark's support file.
pub const DSPARK_ARCH: &str = "deepseek4-dspark";

/// At most this many DSpark stages (the published file has 3).
pub const DSPARK_MAX_STAGES: usize = 8;

/// The largest Markov rank accepted (the published file's is 256).
pub const DSPARK_MAX_RANK: usize = 4096;

/// DSpark's metadata, bounded to what lily runs (ds4's
/// `dspark_weights_validate_metadata`).
#[derive(Clone, Debug)]
pub struct DsparkMeta {
    pub stages: usize,
    /// Drafts a block proposes.
    pub block: usize,
    pub markov_rank: usize,
    pub noise: usize,
    /// The target layers whose features feed the drafter, in order.
    pub targets: Vec<usize>,
}

impl DsparkMeta {
    /// Reads and bounds the support file's metadata: `max_block` drafts fit
    /// a verify pass.
    pub fn read(g: &Gguf, c: &Config, max_block: usize) -> Result<Self> {
        let arch = g.architecture()?;
        ensure!(
            arch == DSPARK_ARCH,
            "GGUF architecture {arch:?}, expected {DSPARK_ARCH:?}"
        );
        let u = |k: &str| -> Result<usize> {
            let v = g.u64(&format!("dspark.{k}"))?;
            usize::try_from(v).with_context(|| format!("dspark.{k} {v}"))
        };
        let stages = u("stage_count")?;
        ensure!(
            (1..=DSPARK_MAX_STAGES).contains(&stages) && u("n_layers")? == stages,
            "DSpark stage count {stages} (n_layers must match, at most {DSPARK_MAX_STAGES})"
        );
        let block = u("block_size")?;
        ensure!(
            (1..=max_block).contains(&block),
            "DSpark block of {block}, lily verifies at most {max_block} drafts"
        );
        let markov_rank = u("markov_rank")?;
        ensure!(
            (32..=DSPARK_MAX_RANK).contains(&markov_rank) && markov_rank % 32 == 0,
            "Markov rank {markov_rank}: a multiple of 32 up to {DSPARK_MAX_RANK}"
        );
        let noise = u("noise_token_id")?;
        ensure!(noise < c.vocab, "noise token {noise} outside the vocabulary");
        let targets = match g.get("dspark.target_layer_ids") {
            Some(Value::Array(v)) => v
                .iter()
                .map(|x| {
                    let t = x.as_u64().context("target layer is not an integer")?;
                    usize::try_from(t).context("target layer")
                })
                .collect::<Result<Vec<_>>>()?,
            _ => bail!("dspark.target_layer_ids missing"),
        };
        // The features are captured as the following layer's HC pre applies
        // the post, the last after the final post: consecutive layers ending
        // at the model's last.
        let n = targets.len();
        ensure!(
            (1..=c.n_layer).contains(&n)
                && targets
                    .iter()
                    .enumerate()
                    .all(|(i, &t)| t.checked_add(n - i) == Some(c.n_layer)),
            "DSpark target layers {targets:?}: consecutive, ending at layer {}",
            c.n_layer - 1
        );
        Ok(Self { stages, block, markov_rank, noise, targets })
    }
}

/// Every tensor of the support file (ds4's `dspark_weights_validate_layout`
/// and `_block_layout`): each stage is a layer without compression, routed
/// by a Q8_0 gate with a bias; the first stage projects the target
/// features, the last holds the HC head, the norm, and the Markov and
/// confidence heads.
pub fn dspark_expected(c: &Config, m: &DsparkMeta) -> Result<Vec<Spec>> {
    let h = c.hidden;
    let feat = m.targets.len().checked_mul(h).context("DSpark feature width")?;
    let conf = h.checked_add(m.markov_rank).context("DSpark confidence width")?;
    let mut v = Vec::new();
    for s in 0..m.stages {
        let n = |x: &str| format!("mtp.{s}.{x}");
        layer(
            &mut v,
            c,
            &format!("mtp.{s}"),
            0,
            false,
            Router { types: Q8_0, hash: false },
        );
        if s == 0 {
            v.push(spec(n("main_proj.weight"), &[feat, h], Q8_0));
            v.push(spec(n("main_norm.weight"), &[h], F32));
        }
        if s + 1 == m.stages {
            v.push(spec(n("norm.weight"), &[h], F32));
            v.push(spec(n("hc_head_fn.weight"), &[c.hc_dim(), c.n_hc], F16));
            v.push(spec(n("hc_head_scale.weight"), &[1], F32));
            v.push(spec(n("hc_head_base.weight"), &[c.n_hc], F32));
            for w in ["markov_w1", "markov_w2"] {
                v.push(spec(
                    n(&format!("markov_head.{w}.weight")),
                    &[m.markov_rank, c.vocab],
                    Q8_0,
                ));
            }
            v.push(spec(n("confidence_head.proj.weight"), &[conf, 1], Q8_0));
        }
    }
    Ok(v)
}

/// Checks the support file's tensor table as [`validate`] checks the
/// model's: every tensor present, exact, of a type the kernels read,
/// aligned, the fused gate/up pairs of one type, nothing else.
pub fn validate_dspark(g: &Gguf, c: &Config, m: &DsparkMeta) -> Result<()> {
    let specs = dspark_expected(c, m)?;
    let fused: Vec<Vec<String>> = (0..m.stages)
        .map(|s| {
            ["ffn_gate_exps", "ffn_up_exps"]
                .iter()
                .map(|t| format!("mtp.{s}.{t}.weight"))
                .collect()
        })
        .collect();
    check_specs(g, &specs, &fused)?;
    ensure!(
        g.tensors.len() == specs.len(),
        "the DSpark file has {} tensors, its graph reads {}",
        g.tensors.len(),
        specs.len()
    );
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/deepseek4/schema.rs"]
mod tests;
