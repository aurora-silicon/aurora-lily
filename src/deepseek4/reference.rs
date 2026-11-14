//! The CPU reference of DeepSeek V4 Flash: every operation of the graph in
//! plain f32 (dot products accumulate in f64), written for clarity, not
//! speed. The GPU kernels are tested against these functions, and
//! [`RefModel`] runs the model layer by layer over a prompt on real weights
//! so the GPU path can be compared with it layer by layer.
//!
//! The semantics are ds4's CPU reference path (antirez, MIT; ds4.c):
//! `layer_forward_raw_swa_one`, `rope_tail_ext_inplace` and
//! `rope_tail_layer_inplace`, `dsv4_fp8_kv_quantize_row_inplace_cpu`,
//! `dsv4_indexer_qat_row_inplace_cpu`, `compressor_decode_one`,
//! `compressor_pool_decode_state`, `indexer_allowed_decode_one`,
//! `layer_attention_mixed_one`, `layer_grouped_out_one`, the router
//! (`layer_router_probs_one`, `layer_hash_selected_experts`,
//! `layer_topk_selected_experts_from_probs`), `layer_routed_moe_one` and
//! `output_hc_head_one`. ds4 rounds every cached row (raw window, compressed
//! rows, indexer rows) to f16; the reference does too.

use anyhow::{Context as _, Result, ensure};
use rayon::prelude::*;

use super::config::{Config, Rope};
use crate::refops::{Mat, WeightSource, dot, hc_collapse, rmsnorm, round_f16, swiglu};

pub use crate::refops::{hc_post, sigmoid, silu};

/// ds4's "minus infinity" of the compressor state and masks.
pub const NEG_INF: f32 = -1.0e30;

/// `rmsnorm` without weight of each `head_dim` slice of `x`, in place
/// (ds4 `head_rms_norm_inplace`).
pub fn head_rmsnorm(x: &mut [f32], head_dim: usize, eps: f32) {
    for h in x.chunks_exact_mut(head_dim) {
        let n = rmsnorm(h, None, eps);
        h.copy_from_slice(&n);
    }
}

fn yarn_corr_dim(n_dims: usize, orig_ctx: usize, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (orig_ctx as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// The YaRN ramp's start and end dims (ds4 `rope_yarn_corr_dims`).
pub fn yarn_corr_dims(
    n_dims: usize,
    orig_ctx: usize,
    base: f32,
    fast: f32,
    slow: f32,
) -> [f32; 2] {
    let start = yarn_corr_dim(n_dims, orig_ctx, fast, base).floor();
    let end = yarn_corr_dim(n_dims, orig_ctx, slow, base).ceil();
    [start.max(0.0), end.min((n_dims - 1) as f32)]
}

fn yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = ((i0 / 2) as f32 - low) / 0.001f32.max(high - low);
    1.0 - y.clamp(0.0, 1.0)
}

/// RoPE of the last `rope.n_rot` dims of each `head_dim` slice of `x` at
/// position `pos`, rotating adjacent pairs; `inverse` rotates back (the
/// attention output before the output projection). ds4
/// `rope_tail_layer_inplace`: YaRN layers interpolate without YaRN's
/// magnitude change (the helper's mscale is cancelled by `attn_factor`).
pub fn rope_tail(
    x: &mut [f32],
    head_dim: usize,
    rope: &Rope,
    pos: usize,
    inverse: bool,
) {
    let n_rot = rope.n_rot;
    let n_nope = head_dim - n_rot;
    let theta_scale = rope.base.powf(-2.0 / n_rot as f32);
    let sin_sign = if inverse { -1.0f32 } else { 1.0 };
    let (ext, corr, attn_factor) = match rope.yarn {
        Some(y) => {
            let corr =
                yarn_corr_dims(n_rot, y.orig_ctx, rope.base, y.beta_fast, y.beta_slow);
            let mut attn_factor = 1.0f32;
            if rope.freq_scale > 0.0 {
                attn_factor /= 1.0 + 0.1 * (1.0 / rope.freq_scale).ln();
            }
            (1.0f32, corr, attn_factor)
        }
        None => (0.0, [0.0, 0.0], 1.0),
    };
    for head in x.chunks_exact_mut(head_dim) {
        let tail = &mut head[n_nope..];
        let mut theta_extrap = pos as f32;
        for i in (0..n_rot).step_by(2) {
            let theta_interp = rope.freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;
            if ext != 0.0 {
                let ramp_mix = yarn_ramp(corr[0], corr[1], i) * ext;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / rope.freq_scale).ln();
            }
            let c = theta.cos() * mscale;
            let s = sin_sign * theta.sin() * mscale;
            let (x0, x1) = (tail[i], tail[i + 1]);
            tail[i] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
            theta_extrap *= theta_scale;
        }
    }
}

/// The value of E4M3FN code `i` (sign excluded), ds4 `dsv4_e4m3fn_value_cpu`.
pub fn e4m3fn_value(i: u32) -> f32 {
    const EXP_SCALE: [f32; 16] = [
        0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
        32.0, 64.0, 128.0, 256.0,
    ];
    let exp = ((i >> 3) & 0xf) as usize;
    let mant = (i & 7) as f32;
    if exp == 0 { mant * 0.001953125 } else { (1.0 + mant * 0.125) * EXP_SCALE[exp] }
}

/// `x` rounded to the nearest E4M3FN value (ties to the even code), clamped
/// to +-448 (ds4 `dsv4_e4m3fn_dequant_cpu`).
pub fn e4m3fn_round(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(448.0);
    let (mut lo, mut hi) = (0u32, 126u32);
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3fn_value(mid) <= ax {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let mut best = lo;
    if best < 126 {
        let bd = (ax - e4m3fn_value(best)).abs();
        let nd = (ax - e4m3fn_value(best + 1)).abs();
        if nd < bd || (nd == bd && (best + 1) & 1 == 0 && best & 1 != 0) {
            best += 1;
        }
    }
    sign * e4m3fn_value(best)
}

/// `x` rounded to the nearest E2M1 value (ties to the even code), clamped to
/// +-6 (ds4 `dsv4_e2m1fn_dequant_cpu`).
pub fn e2m1fn_round(x: f32) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(6.0);
    let mut best = 0usize;
    let mut bd = (ax - V[0]).abs();
    for (i, &v) in V.iter().enumerate().skip(1) {
        let d = (ax - v).abs();
        if d < bd || (d == bd && i & 1 == 0 && best & 1 != 0) {
            best = i;
            bd = d;
        }
    }
    sign * V[best]
}

/// The FP8 KV round trip of the first `head_dim - n_rot` dims of a row, in
/// 64-wide blocks with a power-of-two scale (ds4
/// `dsv4_fp8_kv_quantize_row_inplace_cpu`). The RoPE dims are untouched.
pub fn fp8_kv_qat(x: &mut [f32], n_rot: usize) {
    let n_nope = x.len() - n_rot;
    for blk in x[..n_nope].chunks_exact_mut(64) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0e-4);
        let scale = 2f32.powi((amax / 448.0).log2().ceil() as i32);
        for v in blk.iter_mut() {
            *v = e4m3fn_round((*v / scale).clamp(-448.0, 448.0)) * scale;
        }
    }
}

/// The 128-point Walsh-Hadamard transform scaled by 1/sqrt(128) (ds4
/// `dsv4_hadamard128_inplace_cpu`).
pub fn hadamard128(x: &mut [f32]) {
    assert_eq!(x.len(), 128);
    let mut stride = 1;
    while stride < 128 {
        for base in (0..128).step_by(2 * stride) {
            for i in 0..stride {
                let (a, b) = (x[base + i], x[base + stride + i]);
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
        stride <<= 1;
    }
    for v in x.iter_mut() {
        *v *= 0.088_388_35;
    }
}

/// The FP4 activation round trip in 32-wide blocks (ds4
/// `dsv4_fp4_act_quantize_row_inplace_cpu`).
pub fn fp4_act_qat(x: &mut [f32]) {
    for blk in x.chunks_exact_mut(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(7.052_966e-38);
        let scale = 2f32.powi((amax / 6.0).log2().ceil() as i32);
        for v in blk.iter_mut() {
            *v = e2m1fn_round((*v / scale).clamp(-6.0, 6.0)) * scale;
        }
    }
}

/// The indexer's activation simulation of a 128-wide row: Hadamard, then
/// FP4 (ds4 `dsv4_indexer_qat_row_inplace_cpu`); applied to indexer queries
/// and to the indexer's compressed rows.
pub fn indexer_qat(x: &mut [f32]) {
    hadamard128(x);
    fp4_act_qat(x);
}

/// One compressor's rolling state (ds4 `compressor_decode_one`): `lanes *
/// ratio` rows of `lanes * dim` projected values and scores. Ratio-4 layers
/// pool overlapping windows: the previous window's rows (0..4, lane 0) and
/// the current one's (4..8, lane 1).
#[derive(Clone)]
pub struct Compressor {
    pub dim: usize,
    pub ratio: usize,
    pub lanes: usize,
    pub kv: Vec<f32>,
    pub score: Vec<f32>,
}

impl Compressor {
    pub fn new(dim: usize, ratio: usize) -> Self {
        let lanes = if ratio == 4 { 2 } else { 1 };
        let n = lanes * ratio * lanes * dim;
        Self { dim, ratio, lanes, kv: vec![0.0; n], score: vec![NEG_INF; n] }
    }

    fn width(&self) -> usize {
        self.lanes * self.dim
    }

    /// Stores the projections of position `pos` (`kv_cur`, `sc_cur` with
    /// the position bias already added); at the end of a window returns the
    /// pooled row (before its norm).
    pub fn push(
        &mut self,
        kv_cur: &[f32],
        sc_cur: &[f32],
        pos: usize,
    ) -> Option<Vec<f32>> {
        let (w, r) = (self.width(), self.ratio);
        let pm = pos % r;
        let row = if self.lanes == 2 { r + pm } else { pm };
        self.kv[row * w..(row + 1) * w].copy_from_slice(kv_cur);
        self.score[row * w..(row + 1) * w].copy_from_slice(sc_cur);
        if (pos + 1) % r != 0 {
            return None;
        }
        let out = self.pool();
        if self.lanes == 2 {
            let (prev, cur) = self.kv.split_at_mut(r * w);
            prev.copy_from_slice(cur);
            let (prev, cur) = self.score.split_at_mut(r * w);
            prev.copy_from_slice(cur);
        }
        Some(out)
    }

    /// The softmax pool of the current window, per output dim over the
    /// window's rows (ds4 `compressor_pool_decode_state`); 0 where every row
    /// is empty.
    pub fn pool(&self) -> Vec<f32> {
        let (w, r, d) = (self.width(), self.ratio, self.dim);
        let mut out = vec![0f32; d];
        for (j, o) in out.iter_mut().enumerate() {
            // The pooled entries of dim j: ratio-4 sums lane 0 of the
            // previous rows with lane 1 of the current rows, row by row.
            let pairs: Vec<(usize, Option<usize>)> = if self.lanes == 2 {
                (0..r).map(|i| (i * w + j, Some((r + i) * w + d + j))).collect()
            } else {
                (0..r).map(|i| (i * w + j, None)).collect()
            };
            let mut max = NEG_INF;
            for &(a, b) in &pairs {
                max = max.max(self.score[a]);
                if let Some(b) = b {
                    max = max.max(self.score[b]);
                }
            }
            if max <= NEG_INF * 0.5 {
                continue;
            }
            let (mut den, mut sum) = (0f32, 0f32);
            for &(a, b) in &pairs {
                let wa = (self.score[a] - max).exp();
                match b {
                    Some(b) => {
                        let wb = (self.score[b] - max).exp();
                        den += wa + wb;
                        sum += wa * self.kv[a];
                        sum += wb * self.kv[b];
                    }
                    None => {
                        den += wa;
                        sum += wa * self.kv[a];
                    }
                }
            }
            *o = if den > 0.0 { sum / den } else { 0.0 };
        }
        out
    }
}

/// A compressed row: the pooled row normalized, rotated at the window's
/// first position, then put through its cache's activation simulation and
/// rounded to f16 as ds4 stores it.
pub fn finish_compressed(
    pooled: &[f32],
    norm: &[f32],
    eps: f32,
    rope: &Rope,
    comp_pos: usize,
    indexer: bool,
) -> Vec<f32> {
    let mut r = rmsnorm(pooled, Some(norm), eps);
    let dim = r.len();
    rope_tail(&mut r, dim, rope, comp_pos, false);
    if indexer {
        indexer_qat(&mut r);
    } else {
        fp8_kv_qat(&mut r, rope.n_rot);
    }
    r.iter().map(|&v| round_f16(v)).collect()
}

/// The indexer's selection over `n_comp` compressed rows (ds4
/// `indexer_allowed_decode_one`): `None` when every row is visible (no more
/// than `top_k`), else a mask of the `top_k` best rows. `q` is per head
/// (already rotated and simulated), `w` the per-head weights already scaled
/// by `1/sqrt(dim * heads)`. Ties go to the lower row.
pub fn indexer_select(
    q: &[f32],
    w: &[f32],
    keys: &[f32],
    dim: usize,
    top_k: usize,
) -> Option<Vec<bool>> {
    let n_comp = keys.len() / dim;
    if n_comp <= top_k {
        return None;
    }
    let scores = indexer_scores(q, w, keys, dim);
    let mut order: Vec<usize> = (0..n_comp).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut allowed = vec![false; n_comp];
    for &c in &order[..top_k] {
        allowed[c] = true;
    }
    Some(allowed)
}

/// `score(c) = sum_h max(q_h . k_c, 0) * w_h`.
pub fn indexer_scores(q: &[f32], w: &[f32], keys: &[f32], dim: usize) -> Vec<f32> {
    keys.par_chunks_exact(dim)
        .map(|k| {
            q.chunks_exact(dim)
                .zip(w)
                .map(|(qh, &wh)| f64::from(dot(qh, k).max(0.0)) * f64::from(wh))
                .sum::<f64>() as f32
        })
        .collect()
}

/// Sink attention of every head of `q` over `rows` (the raw window rows and
/// the visible compressed rows, keys equal values): softmax of `q.k /
/// sqrt(dim)` with the head's sink logit in the denominator (ds4
/// `layer_attention_mixed_one`).
pub fn sink_attention(
    q: &[f32],
    rows: &[&[f32]],
    sinks: &[f32],
    dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    q.par_chunks_exact(dim)
        .zip(sinks)
        .flat_map_iter(|(qh, &sink)| {
            let s: Vec<f32> = rows.iter().map(|k| dot(qh, k) * scale).collect();
            let max = s.iter().fold(sink, |m, &v| m.max(v));
            let mut den = f64::from((sink - max).exp());
            let mut o = vec![0f64; dim];
            for (k, &sv) in rows.iter().zip(&s) {
                let p = f64::from((sv - max).exp());
                den += p;
                for (oi, &ki) in o.iter_mut().zip(k.iter()) {
                    *oi += p * f64::from(ki);
                }
            }
            o.into_iter().map(move |v| (v / den) as f32)
        })
        .collect()
}

/// `sqrt(softplus(l))`, ds4's router score (`softplus_stable`).
pub fn router_score(l: f32) -> f32 {
    let sp = if l > 20.0 {
        l
    } else if l < -20.0 {
        l.exp()
    } else {
        l.exp().ln_1p()
    };
    sp.sqrt()
}

/// Expert weights of `sel`: their scores normalized, times `scale`.
pub fn route_weights(probs: &[f32], sel: &[usize], scale: f32) -> Vec<f32> {
    let sum: f32 = sel.iter().map(|&e| probs[e]).sum::<f32>().max(6.103_515_6e-5);
    sel.iter().map(|&e| probs[e] / sum * scale).collect()
}

/// The router of one token: hash layers take `hash` (the token's table
/// row), the others the top-k of score + bias. Ties go to the lower id.
pub fn route(
    logits: &[f32],
    bias: Option<&[f32]>,
    hash: Option<&[usize]>,
    k: usize,
    scale: f32,
) -> (Vec<usize>, Vec<f32>) {
    let probs: Vec<f32> = logits.iter().map(|&l| router_score(l)).collect();
    let sel: Vec<usize> = match hash {
        Some(h) => h.to_vec(),
        None => {
            let b = |e: usize| bias.map_or(0.0, |b| b[e]);
            let mut order: Vec<usize> = (0..probs.len()).collect();
            order.sort_by(|&x, &y| {
                (probs[y] + b(y)).total_cmp(&(probs[x] + b(x))).then(x.cmp(&y))
            });
            order[..k].to_vec()
        }
    };
    let w = route_weights(&probs, &sel, scale);
    (sel, w)
}

/// The output head's HC collapse: per-stream weights `sigmoid(pre * scale +
/// base) + eps` of the control projection of the normalized streams (ds4
/// `output_hc_head_one`).
pub fn output_collapse(
    h: &[f32],
    fn_w: &Mat,
    scale: f32,
    base: &[f32],
    hc_eps: f32,
    rms_eps: f32,
) -> Vec<f32> {
    let n = base.len();
    let hidden = h.len() / n;
    let pre = fn_w.mv(&rmsnorm(h, None, rms_eps));
    let w: Vec<f32> =
        (0..n).map(|i| sigmoid(pre[i] * scale + base[i]) + hc_eps).collect();
    (0..hidden).map(|d| (0..n).map(|s| h[s * hidden + d] * w[s]).sum()).collect()
}

/// The dense weights of one layer, decoded.
pub struct RefLayer {
    pub il: usize,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc: [(Mat, Vec<f32>, Vec<f32>); 2],
    pub mats: std::collections::BTreeMap<&'static str, Mat>,
    pub vecs: std::collections::BTreeMap<&'static str, Vec<f32>>,
}

impl RefLayer {
    pub fn load(cfg: &Config, src: &dyn WeightSource, il: usize) -> Result<Self> {
        let name = |s: &str| format!("blk.{il}.{s}");
        let vec = |s: &str| -> Result<Vec<f32>> { Ok(Mat::load(src, &name(s))?.w) };
        let side = |p: &str| -> Result<(Mat, Vec<f32>, Vec<f32>)> {
            Ok((
                Mat::load(src, &name(&format!("hc_{p}_fn.weight")))?,
                vec(&format!("hc_{p}_scale.weight"))?,
                vec(&format!("hc_{p}_base.weight"))?,
            ))
        };
        let mut mat_names: Vec<&'static str> = vec![
            "attn_q_a.weight",
            "attn_q_b.weight",
            "attn_kv.weight",
            "attn_output_a.weight",
            "attn_output_b.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ];
        let mut vec_names: Vec<&'static str> =
            vec!["attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_sinks.weight"];
        if cfg.ratio(il) != 0 {
            mat_names
                .extend(["attn_compressor_kv.weight", "attn_compressor_gate.weight"]);
            vec_names
                .extend(["attn_compressor_ape.weight", "attn_compressor_norm.weight"]);
        }
        if cfg.has_indexer(il) {
            mat_names.extend([
                "indexer_compressor_kv.weight",
                "indexer_compressor_gate.weight",
                "indexer.attn_q_b.weight",
                "indexer.proj.weight",
            ]);
            vec_names.extend([
                "indexer_compressor_ape.weight",
                "indexer_compressor_norm.weight",
            ]);
        }
        if !cfg.is_hash_layer(il) {
            vec_names.push("exp_probs_b.bias");
        }
        let mut mats = std::collections::BTreeMap::new();
        let mut vecs = std::collections::BTreeMap::new();
        for m in mat_names {
            mats.insert(m, Mat::load(src, &name(m))?);
        }
        for v in vec_names {
            vecs.insert(v, vec(v)?);
        }
        Ok(Self {
            il,
            attn_norm: vec("attn_norm.weight")?,
            ffn_norm: vec("ffn_norm.weight")?,
            hc: [side("attn")?, side("ffn")?],
            mats,
            vecs,
        })
    }

    pub fn m(&self, name: &str) -> &Mat {
        &self.mats[name]
    }

    pub fn v(&self, name: &str) -> &[f32] {
        &self.vecs[name]
    }
}

/// One layer's attention state: the raw window, the compressors and their
/// rows.
#[derive(Clone)]
pub struct RefLayerState {
    /// Raw rows, oldest first, at most `window`.
    pub raw: Vec<Vec<f32>>,
    pub comp: Option<Compressor>,
    pub comp_rows: Vec<Vec<f32>>,
    pub index: Option<Compressor>,
    pub index_rows: Vec<Vec<f32>>,
}

impl RefLayerState {
    pub fn new(cfg: &Config, il: usize) -> Self {
        let r = cfg.ratio(il);
        Self {
            raw: Vec::new(),
            comp: (r != 0).then(|| Compressor::new(cfg.head_dim, r)),
            comp_rows: Vec::new(),
            index: cfg.has_indexer(il).then(|| Compressor::new(cfg.indexer_dim, r)),
            index_rows: Vec::new(),
        }
    }
}

/// Intermediates of one token through one layer, for layer-by-layer
/// comparison with the GPU.
#[derive(Clone, Default)]
pub struct Trace {
    pub attn_in: Vec<f32>,
    pub q: Vec<f32>,
    pub kv: Vec<f32>,
    pub heads: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub selected: Vec<usize>,
    pub expert_weights: Vec<f32>,
    /// Compressed rows visible (after the indexer) to this token.
    pub visible_comp: usize,
}

/// A compressor step of one position: the projections of `xn`, the
/// position's bias, the push, and the finished row if the window ended.
#[allow(clippy::too_many_arguments)]
fn compress_step(
    cfg: &Config,
    l: &RefLayer,
    c: &mut Compressor,
    prefix: &str,
    xn: &[f32],
    pos: usize,
    indexer: bool,
) -> Option<Vec<f32>> {
    let kv = l.m(&format!("{prefix}_kv.weight")).mv(xn);
    let mut sc = l.m(&format!("{prefix}_gate.weight")).mv(xn);
    let ape = l.v(&format!("{prefix}_ape.weight"));
    let w = sc.len();
    let pm = pos % c.ratio;
    for (j, s) in sc.iter_mut().enumerate() {
        *s += ape[pm * w + j];
    }
    let pooled = c.push(&kv, &sc, pos)?;
    Some(finish_compressed(
        &pooled,
        l.v(&format!("{prefix}_norm.weight")),
        cfg.rms_eps,
        &cfg.rope(l.il),
        pos + 1 - c.ratio,
        indexer,
    ))
}

/// The attention sublayer of one token at `pos`: input normalized, output
/// before HC.
pub fn attention(
    cfg: &Config,
    l: &RefLayer,
    st: &mut RefLayerState,
    xn: &[f32],
    pos: usize,
    trace: &mut Trace,
) -> Vec<f32> {
    let d = cfg.head_dim;
    let rope = cfg.rope(l.il);
    let qr = rmsnorm(
        &l.m("attn_q_a.weight").mv(xn),
        Some(l.v("attn_q_a_norm.weight")),
        cfg.rms_eps,
    );
    let mut q = l.m("attn_q_b.weight").mv(&qr);
    head_rmsnorm(&mut q, d, cfg.rms_eps);
    let mut kv = rmsnorm(
        &l.m("attn_kv.weight").mv(xn),
        Some(l.v("attn_kv_a_norm.weight")),
        cfg.rms_eps,
    );
    rope_tail(&mut q, d, &rope, pos, false);
    rope_tail(&mut kv, d, &rope, pos, false);
    fp8_kv_qat(&mut kv, cfg.n_rot);
    if st.raw.len() == cfg.window {
        st.raw.remove(0);
    }
    st.raw.push(kv.iter().map(|&v| round_f16(v)).collect());
    let mut allowed = None;
    if let Some(c) = st.comp.as_mut()
        && let Some(row) = compress_step(cfg, l, c, "attn_compressor", xn, pos, false)
    {
        st.comp_rows.push(row);
    }
    if let Some(c) = st.index.as_mut() {
        if let Some(row) = compress_step(cfg, l, c, "indexer_compressor", xn, pos, true)
        {
            st.index_rows.push(row);
        }
        if st.index_rows.len() > cfg.indexer_top_k {
            let mut iq = l.m("indexer.attn_q_b.weight").mv(&qr);
            rope_tail(&mut iq, cfg.indexer_dim, &rope, pos, false);
            for h in iq.chunks_exact_mut(cfg.indexer_dim) {
                indexer_qat(h);
            }
            let s = 1.0 / ((cfg.indexer_dim * cfg.indexer_heads) as f32).sqrt();
            let w: Vec<f32> =
                l.m("indexer.proj.weight").mv(xn).iter().map(|&v| v * s).collect();
            let keys: Vec<f32> = st.index_rows.concat();
            allowed =
                indexer_select(&iq, &w, &keys, cfg.indexer_dim, cfg.indexer_top_k);
        }
    }
    let mut rows: Vec<&[f32]> = st.raw.iter().map(|r| r.as_slice()).collect();
    for (c, r) in st.comp_rows.iter().enumerate() {
        if allowed.as_ref().is_none_or(|a| a[c]) {
            rows.push(r);
        }
    }
    trace.visible_comp = rows.len() - st.raw.len();
    let mut heads = sink_attention(&q, &rows, l.v("attn_sinks.weight"), d);
    rope_tail(&mut heads, d, &rope, pos, true);
    let low = l.m("attn_output_a.weight").mv_grouped(&heads, cfg.o_lora);
    let out = l.m("attn_output_b.weight").mv(&low);
    trace.q = q;
    trace.kv = kv;
    trace.heads = heads;
    out
}

/// The FFN sublayer of one token (input normalized). Routed experts are
/// read from `src` per token; `hash` is the token's hash-routing row on the
/// hash layers.
pub fn ffn(
    cfg: &Config,
    l: &RefLayer,
    src: &dyn WeightSource,
    xn: &[f32],
    hash: Option<&[usize]>,
    trace: &mut Trace,
) -> Result<Vec<f32>> {
    let lim = cfg.swiglu_limit;
    let logits = l.m("ffn_gate_inp.weight").mv(xn);
    let bias = (!cfg.is_hash_layer(l.il)).then(|| l.v("exp_probs_b.bias"));
    let (sel, w) =
        route(&logits, bias, hash, cfg.n_expert_used, cfg.expert_weight_scale);
    let gs = l.m("ffn_gate_shexp.weight").mv(xn);
    let us = l.m("ffn_up_shexp.weight").mv(xn);
    let mid: Vec<f32> = gs.iter().zip(&us).map(|(&a, &b)| swiglu(a, b, lim)).collect();
    let mut out = l.m("ffn_down_shexp.weight").mv(&mid);
    let (ff, hid, il) = (cfg.ff_expert, cfg.hidden, l.il);
    let mut moe = vec![0f32; hid];
    for (&e, &we) in sel.iter().zip(&w) {
        ensure!(e < cfg.n_expert, "expert {e} of layer {il}");
        let g =
            src.rows(&format!("blk.{il}.ffn_gate_exps.weight"), e * ff..(e + 1) * ff)?;
        let u =
            src.rows(&format!("blk.{il}.ffn_up_exps.weight"), e * ff..(e + 1) * ff)?;
        let mid: Vec<f32> = (0..ff)
            .into_par_iter()
            .map(|r| {
                let gv = dot(&g[r * hid..(r + 1) * hid], xn);
                let uv = dot(&u[r * hid..(r + 1) * hid], xn);
                swiglu(gv, uv, lim) * we
            })
            .collect();
        let dn = src
            .rows(&format!("blk.{il}.ffn_down_exps.weight"), e * hid..(e + 1) * hid)?;
        let part: Vec<f32> = (0..hid)
            .into_par_iter()
            .map(|r| dot(&dn[r * ff..(r + 1) * ff], &mid))
            .collect();
        for (m, p) in moe.iter_mut().zip(part) {
            *m += p;
        }
    }
    for (o, m) in out.iter_mut().zip(&moe) {
        *o += m;
    }
    trace.selected = sel;
    trace.expert_weights = w;
    Ok(out)
}

/// The reference model: runs a prompt layer by layer.
pub struct RefModel<'s> {
    pub cfg: Config,
    pub src: &'s dyn WeightSource,
    /// `hash[il][token]`: the hash layers' expert rows, read on demand.
    hash_tables: Vec<Vec<usize>>,
}

impl<'s> RefModel<'s> {
    /// `hash_tables[il]` is layer il's `ffn_gate_tid2eid` flattened
    /// (`n_expert_used` ids per token) for the hash layers.
    pub fn new(
        cfg: Config,
        src: &'s dyn WeightSource,
        hash_tables: Vec<Vec<usize>>,
    ) -> Self {
        Self { cfg, src, hash_tables }
    }

    /// Reads the hash tables from a GGUF.
    pub fn hash_tables(cfg: &Config, g: &crate::gguf::Gguf) -> Result<Vec<Vec<usize>>> {
        (0..cfg.n_hash_layer)
            .map(|il| {
                let t = g.require(&format!("blk.{il}.ffn_gate_tid2eid.weight"))?;
                g.read_rows_i32(t, 0..t.rows())?
                    .into_iter()
                    .map(|v| usize::try_from(v).context("negative expert id"))
                    .collect()
            })
            .collect()
    }

    /// Embeds `tokens` into HC streams `[token][n_hc * hidden]`.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        tokens
            .iter()
            .map(|&t| {
                let row =
                    self.src.rows("token_embd.weight", t as usize..t as usize + 1)?;
                Ok(row.repeat(self.cfg.n_hc))
            })
            .collect()
    }

    /// Runs layer `il` over every token's HC state in order (positions
    /// `0..`), with fresh state, and returns the traces.
    pub fn layer(
        &self,
        il: usize,
        tokens: &[u32],
        hs: &mut [Vec<f32>],
    ) -> Result<Vec<Trace>> {
        let cfg = &self.cfg;
        ensure!(il < cfg.n_layer, "layer {il}");
        ensure!(tokens.len() == hs.len(), "tokens and states differ in length");
        let l = RefLayer::load(cfg, self.src, il)?;
        let mut st = RefLayerState::new(cfg, il);
        let [(afn, ascale, abase), (ffn_w, fscale, fbase)] = &l.hc;
        let k = cfg.n_expert_used;
        let mut traces = Vec::with_capacity(hs.len());
        for (pos, (h, &tok)) in hs.iter_mut().zip(tokens).enumerate() {
            let mut tr = Trace::default();
            let collapse = |h: &[f32], f: &Mat, s: &[f32], b: &[f32], norm: &[f32]| {
                let (x, post, comb) = hc_collapse(
                    h,
                    f,
                    s,
                    b,
                    cfg.n_hc,
                    cfg.sinkhorn_iters,
                    cfg.hc_eps,
                    cfg.rms_eps,
                );
                (rmsnorm(&x, Some(norm), cfg.rms_eps), post, comb)
            };
            let (xn, post, comb) = collapse(h, afn, ascale, abase, &l.attn_norm);
            let a = attention(cfg, &l, &mut st, &xn, pos, &mut tr);
            let h1 = hc_post(h, &a, &post, &comb, cfg.n_hc);
            let (fx, fpost, fcomb) = collapse(&h1, ffn_w, fscale, fbase, &l.ffn_norm);
            let hash = match self.hash_tables.get(il) {
                Some(tab) => {
                    let t = tok as usize;
                    ensure!(
                        (t + 1) * k <= tab.len(),
                        "token {tok} outside the hash table"
                    );
                    Some(&tab[t * k..(t + 1) * k])
                }
                None => None,
            };
            let f = ffn(cfg, &l, self.src, &fx, hash, &mut tr)?;
            *h = hc_post(&h1, &f, &fpost, &fcomb, cfg.n_hc);
            tr.attn_in = xn;
            tr.attn_out = a;
            tr.ffn_in = fx;
            tr.ffn_out = f;
            traces.push(tr);
        }
        Ok(traces)
    }

    /// The final hidden of one HC state (normalized for the head).
    pub fn final_hidden(&self, h: &[f32], out_hc: &OutputHc, norm: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let x = output_collapse(
            h,
            &out_hc.fn_w,
            out_hc.scale,
            &out_hc.base,
            c.hc_eps,
            c.rms_eps,
        );
        rmsnorm(&x, Some(norm), c.rms_eps)
    }
}

/// The output head's HC weights.
pub struct OutputHc {
    pub fn_w: Mat,
    pub scale: f32,
    pub base: Vec<f32>,
}

impl OutputHc {
    pub fn load(src: &dyn WeightSource) -> Result<Self> {
        Ok(Self {
            fn_w: Mat::load(src, "output_hc_fn.weight")?,
            scale: Mat::load(src, "output_hc_scale.weight")?.w[0],
            base: Mat::load(src, "output_hc_base.weight")?.w,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/deepseek4/reference.rs"]
mod tests;
