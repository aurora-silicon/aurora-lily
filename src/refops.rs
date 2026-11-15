//! CPU reference operations shared by the GGUF models (GLM 5.3 Flash,
//! DeepSeek V4 Flash): weights decoded to f32 rows, f64-accumulated dot
//! products, RMSNorm, the hyper-connection Sinkhorn split and post mix, the
//! clamped SwiGLU, and the tanh GELU (Gemma 4's, ggml's `ggml_gelu_f32`). The semantics are antirez's ds4 (MIT): `rms_norm_*`,
//! `hc_split_sinkhorn_one`, `hc_weighted_sum_one`, `hc_post_one`, `swiglu`
//! (ds4.c).

use std::ops::Range;

use anyhow::{Context as _, Result};
use half::f16;
use rayon::prelude::*;

use crate::gguf::Gguf;

/// Where the reference reads weights from: decoded rows of a named tensor.
pub trait WeightSource {
    /// `(row length, rows)` of tensor `name`.
    fn shape(&self, name: &str) -> Result<(usize, usize)>;
    /// Rows `rows` of `name`, decoded to f32, row-major.
    fn rows(&self, name: &str, rows: Range<usize>) -> Result<Vec<f32>>;
}

impl WeightSource for Gguf {
    fn shape(&self, name: &str) -> Result<(usize, usize)> {
        let t = self.require(name)?;
        Ok((t.row_len(), t.rows()))
    }

    fn rows(&self, name: &str, rows: Range<usize>) -> Result<Vec<f32>> {
        let t = self.require(name)?;
        self.read_rows_f32(t, rows)
    }
}

/// A dense matrix, `rows x cols`, row-major (a GGUF tensor's rows).
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
}

impl Mat {
    pub fn load(src: &dyn WeightSource, name: &str) -> Result<Self> {
        let (cols, rows) = src.shape(name)?;
        let w = src.rows(name, 0..rows).with_context(|| format!("reading {name}"))?;
        Ok(Self { rows, cols, w })
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.w[r * self.cols..(r + 1) * self.cols]
    }

    /// `y = W x`.
    pub fn mv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols, "matvec width");
        (0..self.rows).into_par_iter().map(|r| dot(self.row(r), x)).collect()
    }

    /// `y[r] = W[r] . x[r / rows_per_x]`: per-head projections whose input
    /// differs per group of rows (`attn_k_b`, `attn_v_b`).
    pub fn mv_grouped(&self, x: &[f32], rows_per_x: usize) -> Vec<f32> {
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let g = r / rows_per_x;
                dot(self.row(r), &x[g * self.cols..(g + 1) * self.cols])
            })
            .collect()
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum::<f64>() as f32
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn round_f16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// GELU, tanh approximation (`gelu_pytorch_tanh`, ggml's `ggml_gelu_f32`).
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
}

/// `x / sqrt(mean(x^2) + eps)`, times `w` when given.
pub fn rmsnorm(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let scale = 1.0 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    match w {
        Some(w) => x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect(),
        None => x.iter().map(|&v| v * scale).collect(),
    }
}

/// The hyper-connection control split (ds4 `hc_split_sinkhorn_one`):
/// `mix[2n + n*n]` -> `pre[n]`, `post[n]`, `comb[n*n]` (`comb[dst + n*src]`
/// is read as the weight of stream `src` in output stream `dst`, ds4's
/// `[dst_hc, src_hc]` addressing).
pub fn hc_split_sinkhorn(
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    n: usize,
    iters: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pre: Vec<f32> =
        (0..n).map(|i| sigmoid(mix[i] * scale[0] + base[i]) + eps).collect();
    let post: Vec<f32> = (0..n)
        .map(|i| 2.0 / (1.0 + (-(mix[n + i] * scale[1] + base[n + i])).exp()))
        .collect();
    let mut c = vec![0f32; n * n];
    // Row softmax over src for each dst, plus eps.
    for dst in 0..n {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n {
            let idx = src + dst * n;
            c[idx] = mix[2 * n + idx] * scale[2] + base[2 * n + idx];
            row_max = row_max.max(c[idx]);
        }
        let mut sum = 0f32;
        for src in 0..n {
            let idx = src + dst * n;
            c[idx] = (c[idx] - row_max).exp();
            sum += c[idx];
        }
        let inv = 1.0 / sum;
        for src in 0..n {
            c[src + dst * n] = c[src + dst * n] * inv + eps;
        }
    }
    let col_norm = |c: &mut [f32]| {
        for src in 0..n {
            let sum: f32 = (0..n).map(|dst| c[src + dst * n]).sum();
            let inv = 1.0 / (sum + eps);
            for dst in 0..n {
                c[src + dst * n] *= inv;
            }
        }
    };
    let row_norm = |c: &mut [f32]| {
        for dst in 0..n {
            let sum: f32 = (0..n).map(|src| c[src + dst * n]).sum();
            let inv = 1.0 / (sum + eps);
            for src in 0..n {
                c[src + dst * n] *= inv;
            }
        }
    };
    col_norm(&mut c);
    for _ in 1..iters {
        row_norm(&mut c);
        col_norm(&mut c);
    }
    (pre, post, c)
}

/// HC collapse: `pre`-weighted sum of the streams `h[n * hidden]` after the
/// control projection of their RMS-normalized concatenation and the Sinkhorn
/// split. Returns `(x, post, comb)`; the caller normalizes `x`.
#[allow(clippy::too_many_arguments)]
pub fn hc_collapse(
    h: &[f32],
    fn_w: &Mat,
    scale: &[f32],
    base: &[f32],
    n: usize,
    iters: usize,
    hc_eps: f32,
    rms_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = h.len() / n;
    let flat = rmsnorm(h, None, rms_eps);
    let mix = fn_w.mv(&flat);
    let (pre, post, comb) = hc_split_sinkhorn(&mix, scale, base, n, iters, hc_eps);
    let x: Vec<f32> =
        (0..hidden).map(|d| (0..n).map(|s| h[s * hidden + d] * pre[s]).sum()).collect();
    (x, post, comb)
}

/// HC post: `out[dst][d] = block[d] * post[dst] + sum_src comb[dst + n*src]
/// * h[src][d]`.
pub fn hc_post(
    h: &[f32],
    block: &[f32],
    post: &[f32],
    comb: &[f32],
    n: usize,
) -> Vec<f32> {
    let hidden = block.len();
    let mut out = vec![0f32; n * hidden];
    for dst in 0..n {
        for d in 0..hidden {
            let mut acc = block[d] * post[dst];
            for src in 0..n {
                acc += comb[dst + src * n] * h[src * hidden + d];
            }
            out[dst * hidden + d] = acc;
        }
    }
    out
}

/// SwiGLU with GLM's clamp: `silu(min(g, limit)) * clamp(u, -limit, limit)`.
pub fn swiglu(g: f32, u: f32, limit: f32) -> f32 {
    let (g, u) =
        if limit > 1e-6 { (g.min(limit), u.clamp(-limit, limit)) } else { (g, u) };
    silu(g) * u
}
