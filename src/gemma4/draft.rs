//! Gemma 4's draft model (`gemma4-assistant` GGUF, "MTP" in the published
//! files): four small layers (hidden 1024) that read the target's caches
//! instead of keeping their own. The graph is llama.cpp's
//! (`src/models/gemma4-assistant.cpp`, MIT) and its drafting loop
//! (`common/speculative.cpp`, shared memory): a draft from token `t` at
//! position `p` takes `concat(embed_target(t) * sqrt(2816), h)`, `h` the
//! target's post-norm hidden of position `p - 1`, through
//! `pre_projection`; each layer's attention queries (rotated to `p`) attend
//! the target's last sliding-window layer's or last global layer's K/V at
//! positions below `p`; the head is the draft model's own tied embedding,
//! and `post_projection` gives the `h` of the next draft, which is drawn at
//! the same position `p`.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};

use super::config::Config as TargetConfig;
use super::model::{Gemma4Model, Gemma4State, MAX_ROWS, Scratch};
use crate::gguf::{Gguf, Value};
use crate::glm53::model::Mapped;
use crate::kernels::gemma4::{self as g4, AttnShape};
use crate::kernels::glm53::{self as k, QWeight};
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// The GGUF architecture name.
pub const ARCH: &str = "gemma4-assistant";

/// Drafts a pass proposes at most (a verify pass takes MAX_ROWS rows).
pub const MAX_DRAFTS: usize = MAX_ROWS - 1;

struct LayerW {
    attn_norm: Tensor,
    q: QWeight,
    q_norm: Tensor,
    o: QWeight,
    post_attn_norm: Tensor,
    ffn_norm: Tensor,
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    post_ffw_norm: Tensor,
    out_scale: Tensor,
    swa: bool,
    head_dim: usize,
}

/// The draft model's weights, and its scratch (one row).
pub struct Gemma4Draft {
    pub gguf: Gguf,
    mapped: Mapped,
    hidden: usize,
    ff: usize,
    heads: usize,
    eps: f32,
    rope_base: f32,
    rope_base_swa: f32,
    layers: Vec<LayerW>,
    pre_proj: QWeight,
    post_proj: QWeight,
    embed: QWeight,
    output_norm: Tensor,
    rope_freqs: Tensor,
    sc: DraftScratch,
}

struct DraftScratch {
    /// The pre-projection's input: target embedding then h.
    xh: Tensor,
    x: Tensor,
    hn: Tensor,
    q: Tensor,
    qo: Tensor,
    heads: Tensor,
    attn: Tensor,
    r: Tensor,
    n: Tensor,
    g: Tensor,
    u: Tensor,
    mid: Tensor,
    f: Tensor,
    /// The next draft's h (post_projection).
    h_next: Tensor,
    logits: Tensor,
    /// U32 [MAX_DRAFTS + 1]: the input token, then each draft.
    pub io: Tensor,
    /// F32 [MAX_DRAFTS]: each draft's softmax probability.
    pub probs: Tensor,
}

fn arr_bools(g: &Gguf, key: &str) -> Result<Vec<bool>> {
    match g.get(key) {
        Some(Value::Array(v)) => v
            .iter()
            .map(|b| match b {
                Value::Bool(b) => Ok(*b),
                _ => anyhow::bail!("{key} holds a non-boolean"),
            })
            .collect(),
        _ => anyhow::bail!("GGUF metadata {key} missing"),
    }
}

/// The draft model beside `target_path`: a `gemma4-assistant` GGUF in the
/// same directory whose output width is the target's hidden size.
pub fn find_beside(
    target_path: &Path,
    hidden: usize,
) -> Result<Option<std::path::PathBuf>> {
    let dir = target_path.parent().unwrap_or(Path::new("."));
    let mut found = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))?
    {
        let path = entry?.path();
        if path == target_path
            || path.extension().and_then(|e| e.to_str()) != Some("gguf")
        {
            continue;
        }
        let Ok(g) = Gguf::open(&path) else { continue };
        if g.architecture().ok() == Some(ARCH)
            && g.u64(&format!("{ARCH}.embedding_length_out")).ok()
                == Some(hidden as u64)
        {
            found.push(path);
        }
    }
    found.sort();
    Ok(found.into_iter().next())
}

impl Gemma4Draft {
    /// Maps the draft model at `path` for `target`.
    pub fn load(
        ctx: &MetalContext,
        path: &Path,
        target: &TargetConfig,
    ) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        ensure!(
            gguf.architecture()? == ARCH,
            "{} is not a {ARCH} GGUF",
            path.display()
        );
        let u = |k: &str| -> Result<usize> {
            Ok(gguf.u64(&format!("{ARCH}.{k}"))? as usize)
        };
        let f = |k: &str| gguf.f32(&format!("{ARCH}.{k}"));
        let n_layer = u("block_count")?;
        let hidden = u("embedding_length")?;
        let ff = u("feed_forward_length")?;
        let heads = u("attention.head_count")?;
        ensure!(
            u("embedding_length_out")? == target.hidden,
            "draft model for another target"
        );
        ensure!(
            u("attention.shared_kv_layers")? == n_layer,
            "the draft model keeps K/V of its own (lily reads the target's)"
        );
        ensure!(
            heads == target.n_head,
            "draft heads {heads}, target {}",
            target.n_head
        );
        let swa =
            arr_bools(&gguf, &format!("{ARCH}.attention.sliding_window_pattern"))?;
        ensure!(swa.len() == n_layer, "sliding pattern of {} layers", swa.len());
        let (dh, dh_swa) = (u("attention.key_length")?, u("attention.key_length_swa")?);
        ensure!(
            dh == target.head_dim && dh_swa == target.head_dim_swa,
            "draft head widths {dh}/{dh_swa} differ from the target's"
        );
        ensure!(u("attention.sliding_window")? == target.window, "draft window");
        let mapped = Mapped::map(ctx, &gguf)?;
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let n = |s: &str| format!("blk.{il}.{s}");
            let d = if swa[il] { dh_swa } else { dh };
            layers.push(LayerW {
                attn_norm: mapped.f32(&n("attn_norm.weight"))?,
                q: mapped.q(&n("attn_q.weight"))?,
                q_norm: mapped.f32(&n("attn_q_norm.weight"))?,
                o: mapped.q(&n("attn_output.weight"))?,
                post_attn_norm: mapped.f32(&n("post_attention_norm.weight"))?,
                ffn_norm: mapped.f32(&n("ffn_norm.weight"))?,
                gate: mapped.q(&n("ffn_gate.weight"))?,
                up: mapped.q(&n("ffn_up.weight"))?,
                down: mapped.q(&n("ffn_down.weight"))?,
                post_ffw_norm: mapped.f32(&n("post_ffw_norm.weight"))?,
                out_scale: mapped.f32(&n("layer_output_scale.weight"))?,
                swa: swa[il],
                head_dim: d,
            });
            let l = &layers[il];
            ensure!(
                l.q.cols == hidden && l.q.rows == heads * d && l.o.rows == hidden,
                "draft layer {il} attention shapes"
            );
            ensure!(
                l.gate.rows == ff && l.down.cols == ff,
                "draft layer {il} FFN shapes"
            );
        }
        let pre_proj = mapped.q("nextn.pre_projection.weight")?;
        let post_proj = mapped.q("nextn.post_projection.weight")?;
        ensure!(
            pre_proj.cols == 2 * target.hidden && pre_proj.rows == hidden,
            "pre_projection {} x {}",
            pre_proj.cols,
            pre_proj.rows
        );
        ensure!(
            post_proj.cols == hidden && post_proj.rows == target.hidden,
            "post_projection"
        );
        let embed = mapped.q("token_embd.weight")?;
        ensure!(embed.cols == hidden && embed.rows == target.vocab, "draft embedding");
        let output_norm = mapped.f32("output_norm.weight")?;
        let rope_freqs = mapped.f32("rope_freqs.weight")?;
        let z = |n: usize| Tensor::zeros(ctx, &[n], DType::F32);
        let qw = heads * dh.max(dh_swa);
        let sc = DraftScratch {
            xh: z(2 * target.hidden)?,
            x: z(hidden)?,
            hn: z(hidden)?,
            q: z(qw)?,
            qo: z(qw)?,
            heads: z(qw)?,
            attn: z(hidden)?,
            r: z(hidden)?,
            n: z(hidden)?,
            g: z(ff)?,
            u: z(ff)?,
            mid: z(ff)?,
            f: z(hidden)?,
            h_next: z(target.hidden)?,
            logits: z(target.vocab)?,
            io: Tensor::zeros(ctx, &[MAX_DRAFTS + 1], DType::U32)?,
            probs: z(MAX_DRAFTS)?,
        };
        Ok(Self {
            eps: f("attention.layer_norm_rms_epsilon")?,
            rope_base: f("rope.freq_base")?,
            rope_base_swa: f("rope.freq_base_swa")?,
            gguf,
            mapped,
            hidden,
            ff,
            heads,
            layers,
            pre_proj,
            post_proj,
            embed,
            output_norm,
            rope_freqs,
            sc,
        })
    }

    /// GPU bytes of the mapped file.
    pub fn bytes(&self) -> usize {
        self.mapped.buffers.iter().map(|b| b.length()).sum()
    }

    /// Encodes `n` chained greedy drafts at the state's position `p`: the
    /// first from the token in `io[0]` (staged by the caller) and the
    /// target hidden `h` (row of `sc.h`), each next one from the draft
    /// before it; draft `k` goes to `io[k + 1]`, its probability to
    /// `probs[k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        target: &Gemma4Model,
        st: &Gemma4State,
        tsc: &Scratch,
        h: &Tensor,
        n: usize,
    ) -> Result<()> {
        ensure!((1..=MAX_DRAFTS).contains(&n), "{n} drafts");
        ensure!(st.pos > 0, "drafting before any token was fed");
        let tc = &target.cfg;
        let th = tc.hidden;
        let sc = &self.sc;
        let p = st.pos;
        let e_part = sc.xh.view(0, &[th])?;
        let h_part = sc.xh.view(th, &[th])?;
        g4::copy(ctx, pass, h, &h_part, th)?;
        let greedy = SamplingParams::greedy();
        for d in 0..n {
            let token = sc.io.view(d, &[1])?;
            g4::embed(
                ctx,
                pass,
                target.embed(),
                &token,
                &e_part,
                1,
                (th as f32).sqrt(),
            )?;
            pass.memory_barrier()?;
            k::gemv(ctx, pass, &self.pre_proj, &sc.xh, &sc.x, 1)?;
            pass.memory_barrier()?;
            for (il, l) in self.layers.iter().enumerate() {
                self.encode_layer(ctx, pass, target, st, tsc, l, il, p)?;
            }
            // Head and next h from rmsnorm(x) output_norm (in sc.hn).
            k::gemv(ctx, pass, &self.embed, &sc.hn, &sc.logits, 1)?;
            k::gemv(ctx, pass, &self.post_proj, &sc.hn, &sc.h_next, 1)?;
            pass.memory_barrier()?;
            let out = sc.io.view(d + 1, &[1])?;
            sample_f32(ctx, pass, &sc.logits, tsc.sampler(), &greedy, 0, &out)?;
            k::top_prob(ctx, pass, &sc.logits, &sc.probs.view(d, &[1])?, tc.vocab)?;
            g4::copy(ctx, pass, &sc.h_next, &h_part, th)?;
            pass.memory_barrier()?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        target: &Gemma4Model,
        st: &Gemma4State,
        tsc: &Scratch,
        l: &LayerW,
        il: usize,
        p: usize,
    ) -> Result<()> {
        let sc = &self.sc;
        let (hd, eps) = (self.hidden, self.eps);
        g4::rmsnorm(ctx, pass, &sc.x, &l.attn_norm, &sc.hn, 1, hd, eps)?;
        pass.memory_barrier()?;
        k::gemv(ctx, pass, &l.q, &sc.hn, &sc.q, 1)?;
        pass.memory_barrier()?;
        // The target layer whose caches this layer reads: its last
        // sliding-window layer or its last global one.
        let tl = target.shared_kv_layer(l.swa);
        let c = &st.layers[tl];
        let tc = &target.cfg;
        // Queries only (no K/V slots), rotated to p.
        let qshape = AttnShape {
            head_dim: l.head_dim,
            heads: self.heads,
            kv_heads: 0,
            ring: c.slots,
            window: 0,
            k_cache: &c.k,
            v_cache: &c.v,
        };
        g4::qkv_queries(
            ctx,
            pass,
            &qshape,
            &sc.q,
            &l.q_norm,
            (!l.swa).then_some(&self.rope_freqs),
            &sc.qo,
            p,
            if l.swa { self.rope_base_swa } else { self.rope_base },
            eps,
        )?;
        pass.memory_barrier()?;
        // Keys below p: a query at p - 1 for the mask, the window one less
        // (llama.cpp: key pos k attended when p - k < window).
        let a = AttnShape {
            head_dim: l.head_dim,
            heads: self.heads,
            kv_heads: tc.n_head_kv[tl],
            ring: c.slots,
            window: if c.swa { tc.window - 1 } else { 0 },
            k_cache: &c.k,
            v_cache: &c.v,
        };
        g4::attention(
            ctx,
            pass,
            &a,
            &sc.qo,
            tsc.partial(),
            &sc.heads,
            p - 1,
            1,
            MAX_ROWS,
        )?;
        pass.memory_barrier()?;
        k::gemv(ctx, pass, &l.o, &sc.heads, &sc.attn, 1)?;
        pass.memory_barrier()?;
        g4::post_attn(
            ctx,
            pass,
            &sc.attn,
            &l.post_attn_norm,
            &sc.x,
            &sc.r,
            [&l.ffn_norm, &l.ffn_norm, &l.ffn_norm],
            [&sc.n, &sc.n, &sc.n],
            1,
            hd,
            eps,
        )?;
        pass.memory_barrier()?;
        k::gemv_sets(ctx, pass, &[(&l.gate, &sc.g), (&l.up, &sc.u)], &sc.n, 1)?;
        pass.memory_barrier()?;
        g4::geglu(ctx, pass, &sc.g, &sc.u, &sc.mid, self.ff)?;
        pass.memory_barrier()?;
        k::gemv(ctx, pass, &l.down, &sc.mid, &sc.f, 1)?;
        pass.memory_barrier()?;
        let next = self.layers.get(il + 1).map_or(&self.output_norm, |n| &n.attn_norm);
        g4::ffn_out1(
            ctx,
            pass,
            &sc.f,
            &l.post_ffw_norm,
            &sc.r,
            &l.out_scale,
            &sc.x,
            Some((next, &sc.hn)),
            1,
            hd,
            eps,
        )?;
        pass.memory_barrier()
    }

    /// Stages the first draft's input token.
    pub fn stage_token(&self, token: u32) -> Result<()> {
        crate::glm53::model::write_words(&self.sc.io, &[token])
    }

    /// The drafts of the last pass and their probabilities. GPU idle.
    pub fn read(&self, n: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        Ok((
            self.sc.io.to_u32()?[1..=n].to_vec(),
            self.sc.probs.to_f32()?[..n].to_vec(),
        ))
    }
}
