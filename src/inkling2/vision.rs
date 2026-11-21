//! Vision tower for Inkling 2 Mini (`mmproj-F16.gguf`, `mmproj-BF16.gguf`,
//! `mmproj-Inkling-2-Mini-Q8_0.gguf`, `clip.vision.projector_type = "inkling2v"`).
//!
//! Follows `ggml-org/llama.cpp`'s `tools/mtmd/models/inkling2v.cpp` (MIT):
//! - 23-layer ViT (`hidden = 1024`, `n_head = 16`, `head_dim = 64`, `ff = 4096`)
//! - Patch size 20x20, spatial merge 2x2 (effective patch stride 40x40),
//!   image normalization mean = 0.5, std = 0.5.
//! - Bilinearly resized 2D learned position embeddings (`posemb_grid_size = 32`),
//!   prepended CLS token (`v.class_embd + v.class_position_embd`), `v.pre_ln`,
//!   2D RoPE on Q/K, LayerNorm (`eps = 1e-5`), LayerScale (`ls1`, `ls2`),
//!   GELU MLP.
//! - Drops CLS token, 2x2 spatial patch merge (`1024 -> 4096`), 2-layer
//!   GELU-ERF MLP projector (`4096 -> 4096 -> 2304`).
//! - Note: Inkling 2 Mini does NOT apply `token_embd_norm` to encoded image
//!   embeddings (`inkling2.cpp` line 265).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::gguf::Gguf;
use crate::glm53::model::Mapped;
use crate::kernels::glm53::QWeight;
use crate::metal::MetalContext;
use crate::tensor::{DType, Tensor};

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub n_layer: usize,
    pub hidden: usize,
    pub ff: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub image_size: usize,
    pub proj_dim: usize,
    pub ln_eps: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rope_theta: f32,
}

impl VisionConfig {
    pub fn from_gguf(g: &Gguf, text_hidden: usize) -> Result<Self> {
        ensure!(
            g.str("general.architecture")? == "clip"
                && g.str("clip.vision.projector_type")? == "inkling2v",
            "not an inkling2v mmproj GGUF"
        );
        let n_layer = g.u64("clip.vision.block_count")? as usize;
        let hidden = g.u64("clip.vision.embedding_length")? as usize;
        let ff = g.u64("clip.vision.feed_forward_length")? as usize;
        let n_head = g.u64("clip.vision.attention.head_count")? as usize;
        let patch_size = g.u64("clip.vision.patch_size")? as usize;
        let merge_size = g.u64("clip.vision.spatial_merge_size").unwrap_or(2) as usize;
        let image_size = g.u64("clip.vision.image_size").unwrap_or(640) as usize;
        let proj_dim = g.u64("clip.vision.projection_dim")? as usize;
        let ln_eps = g.f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-5);
        ensure!(
            proj_dim == text_hidden,
            "mmproj projection_dim {proj_dim} != {text_hidden}"
        );
        ensure!(
            hidden.is_multiple_of(n_head),
            "vision hidden {hidden} not divisible by {n_head}"
        );
        Ok(Self {
            n_layer,
            hidden,
            ff,
            n_head,
            head_dim: hidden / n_head,
            patch_size,
            merge_size,
            image_size,
            proj_dim,
            ln_eps,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            rope_theta: 10_000.0,
        })
    }
}

/// Finds an `inkling2v` mmproj GGUF next to `model_path`.
pub fn find_beside(model_path: &Path, text_hidden: usize) -> Result<Option<PathBuf>> {
    let Some(dir) = model_path.parent() else {
        return Ok(None);
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(None);
    };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if p == model_path || p.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        let Ok(g) = Gguf::open(&p) else {
            continue;
        };
        if VisionConfig::from_gguf(&g, text_hidden).is_ok() {
            found.push((p, g.file_len));
        }
    }
    found.sort_by_key(|(_, sz)| *sz);
    Ok(found.into_iter().next().map(|(p, _)| p))
}

struct VitBlockW {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    ls1: Vec<f32>,
    ls2: Vec<f32>,
    wq: QWeight,
    bq: Vec<f32>,
    wk: QWeight,
    bk: Vec<f32>,
    wv: QWeight,
    bv: Vec<f32>,
    wo: QWeight,
    bo: Vec<f32>,
    ff_up: QWeight,
    ff_up_b: Vec<f32>,
    ff_down: QWeight,
    ff_down_b: Vec<f32>,
}

pub struct Inkling2Vision {
    pub cfg: VisionConfig,
    pub bytes: usize,
    _mapped: Mapped,
    patch_embd: QWeight,
    pos_embd: Vec<f32>,       // [grid * grid, hidden]
    class_embd: Vec<f32>,     // [hidden]
    class_pos_embd: Vec<f32>, // [hidden]
    pre_ln_w: Vec<f32>,
    pre_ln_b: Vec<f32>,
    blocks: Vec<VitBlockW>,
    mm1: QWeight, // [4096, 4096]
    mm2: QWeight, // [4096, 2304]
}

fn read_f32(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = g.tensor(name).with_context(|| format!("mmproj missing {name}"))?;
    let mut file = std::fs::File::open(&g.path)?;
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(info.offset))?;
    let mut raw = vec![0u8; info.bytes as usize];
    file.read_exact(&mut raw)?;
    let mut out = vec![0.0f32; info.row_len() * info.rows()];
    crate::gguf::dequantize(info.ty, &raw, &mut out)?;
    Ok(out)
}

fn view_qw(mapped: &Mapped, name: &str) -> Result<QWeight> {
    let (buf, at, info) = mapped.get(name)?;
    // patch_embd is 4D [20, 20, 3, 1024] -> flatten to [1200, 1024]
    let total = info.dims.iter().product::<u64>() as usize;
    let rows = *info.dims.last().unwrap_or(&1) as usize;
    let cols = total / rows;
    let whole = Tensor::from_buffer(buf.clone(), &[buf.length()], DType::I8)?;
    let t = whole.view(*at, &[info.bytes as usize])?;
    QWeight::new(t, info.ty, cols, rows)
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().enumerate().map(|(i, &v)| (v - mean) * inv * w[i] + b[i]).collect()
}

fn gelu_erf(x: f32) -> f32 {
    // Abramowitz & Stegun erf approximation (max error < 1.5e-7).
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign * (1.0 - poly * (-a * a).exp());
    0.5 * x * (1.0 + erf)
}

impl Inkling2Vision {
    pub fn load(ctx: &MetalContext, path: &Path, text_hidden: usize) -> Result<Self> {
        let g = Gguf::open(path)?;
        let cfg = VisionConfig::from_gguf(&g, text_hidden)?;
        let mapped = Mapped::map(ctx, &g)?;
        let patch_embd = view_qw(&mapped, "v.patch_embd.weight")?;
        let pos_embd = read_f32(&g, "v.position_embd.weight")?;
        let class_embd = read_f32(&g, "v.class_embd")?;
        let class_pos_embd = read_f32(&g, "v.class_position_embd.weight")?;
        let pre_ln_w = read_f32(&g, "v.pre_ln.weight")?;
        let pre_ln_b = read_f32(&g, "v.pre_ln.bias")?;
        let mm1 = view_qw(&mapped, "mm.1.weight")
            .or_else(|_| view_qw(&mapped, "mm.0.weight"))?;
        let mm2 = view_qw(&mapped, "mm.2.weight")?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let n = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(VitBlockW {
                ln1_w: read_f32(&g, &n("ln1.weight"))?,
                ln1_b: read_f32(&g, &n("ln1.bias"))?,
                ln2_w: read_f32(&g, &n("ln2.weight"))?,
                ln2_b: read_f32(&g, &n("ln2.bias"))?,
                ls1: read_f32(&g, &n("ls1.weight"))?,
                ls2: read_f32(&g, &n("ls2.weight"))?,
                wq: view_qw(&mapped, &n("attn_q.weight"))?,
                bq: read_f32(&g, &n("attn_q.bias"))?,
                wk: view_qw(&mapped, &n("attn_k.weight"))?,
                bk: read_f32(&g, &n("attn_k.bias"))?,
                wv: view_qw(&mapped, &n("attn_v.weight"))?,
                bv: read_f32(&g, &n("attn_v.bias"))?,
                wo: view_qw(&mapped, &n("attn_out.weight"))?,
                bo: read_f32(&g, &n("attn_out.bias"))?,
                ff_up: view_qw(&mapped, &n("ffn_up.weight"))?,
                ff_up_b: read_f32(&g, &n("ffn_up.bias"))?,
                ff_down: view_qw(&mapped, &n("ffn_down.weight"))?,
                ff_down_b: read_f32(&g, &n("ffn_down.bias"))?,
            });
        }

        Ok(Self {
            cfg,
            bytes: g.file_len as usize,
            _mapped: mapped,
            patch_embd,
            pos_embd,
            class_embd,
            class_pos_embd,
            pre_ln_w,
            pre_ln_b,
            blocks,
            mm1,
            mm2,
        })
    }

    fn gpu_linear(
        ctx: &MetalContext,
        w: &QWeight,
        x: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
    ) -> Result<Vec<f32>> {
        let tx = Tensor::from_f32(ctx, x, &[rows, w.cols])?;
        let ty = Tensor::zeros(ctx, &[rows, w.rows], DType::F32)?;
        let pass = ctx.begin()?;
        if rows <= crate::kernels::glm53::gemv_max_rows(w.ty) {
            crate::kernels::glm53::gemv(ctx, &pass, w, &tx, &ty, rows)?;
        } else {
            crate::kernels::glm53::gemm(ctx, &pass, w, &tx, &ty, rows)?;
        }
        pass.commit_wait()?;
        let mut out = ty.to_f32()?;
        if let Some(b) = bias {
            for r in 0..rows {
                for c in 0..w.rows {
                    out[r * w.rows + c] += b[c];
                }
            }
        }
        Ok(out)
    }

    /// Bilinearly resizes `[src_g, src_g, hidden]` position embeddings to `[grid_h, grid_w, hidden]`.
    fn resize_pos_embd(&self, grid_h: usize, grid_w: usize) -> Vec<f32> {
        let h = self.cfg.hidden;
        let src_g = self.cfg.image_size / self.cfg.patch_size;
        let mut out = vec![0.0f32; grid_h * grid_w * h];
        for y in 0..grid_h {
            let fy = ((y as f32 + 0.5) * (src_g as f32) / (grid_h as f32) - 0.5)
                .clamp(0.0, (src_g - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(src_g - 1);
            let wy = fy - y0 as f32;
            for x in 0..grid_w {
                let fx = ((x as f32 + 0.5) * (src_g as f32) / (grid_w as f32) - 0.5)
                    .clamp(0.0, (src_g - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(src_g - 1);
                let wx = fx - x0 as f32;
                let dst = (y * grid_w + x) * h;
                let p00 = (y0 * src_g + x0) * h;
                let p01 = (y0 * src_g + x1) * h;
                let p10 = (y1 * src_g + x0) * h;
                let p11 = (y1 * src_g + x1) * h;
                for c in 0..h {
                    let top = self.pos_embd[p00 + c] * (1.0 - wx)
                        + self.pos_embd[p01 + c] * wx;
                    let bot = self.pos_embd[p10 + c] * (1.0 - wx)
                        + self.pos_embd[p11 + c] * wx;
                    out[dst + c] = top * (1.0 - wy) + bot * wy;
                }
            }
        }
        out
    }

    /// Encodes preprocessed patches `[grid_h * grid_w, patch_dim]` (row-major) into
    /// `[grid_h * grid_w / 4, text_hidden]` f32 image embeddings.
    pub fn encode(
        &self,
        ctx: &MetalContext,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let m = self.cfg.merge_size;
        ensure!(
            grid_h.is_multiple_of(m) && grid_w.is_multiple_of(m),
            "vision grid {grid_h}x{grid_w} must be multiples of merge_size={m}"
        );
        let n_patches = grid_h * grid_w;
        let h = self.cfg.hidden;
        let patch_out =
            Self::gpu_linear(ctx, &self.patch_embd, patches, None, n_patches)?;
        let pos_emb = self.resize_pos_embd(grid_h, grid_w);

        let n_pos = n_patches + 1;
        let mut cur = vec![0.0f32; n_pos * h];
        // Prepend CLS token.
        for c in 0..h {
            cur[c] = self.class_embd[c] + self.class_pos_embd[c];
        }
        for i in 0..n_patches * h {
            cur[h + i] = patch_out[i] + pos_emb[i];
        }
        // Pre-LN.
        for p in 0..n_pos {
            let nr = layer_norm(
                &cur[p * h..(p + 1) * h],
                &self.pre_ln_w,
                &self.pre_ln_b,
                self.cfg.ln_eps,
            );
            cur[p * h..(p + 1) * h].copy_from_slice(&nr);
        }

        // 2D RoPE coordinates (CLS at (0, 0), patch (y, x) at (y + 1, x + 1)).
        let mut pos_h = Vec::with_capacity(n_pos);
        let mut pos_w = Vec::with_capacity(n_pos);
        pos_h.push(0usize);
        pos_w.push(0usize);
        for y in 0..grid_h {
            for x in 0..grid_w {
                pos_h.push(y + 1);
                pos_w.push(x + 1);
            }
        }

        let nh = self.cfg.n_head;
        let hd = self.cfg.head_dim;
        let quarter = hd / 4;
        let scale = 1.0 / (hd as f32).sqrt();
        let apply_rope_2d = |x: &mut [f32]| {
            for p in 0..n_pos {
                let ph = pos_h[p] as f32;
                let pw = pos_w[p] as f32;
                for head in 0..nh {
                    let base = (p * nh + head) * hd;
                    for i in 0..quarter {
                        let freq =
                            1.0 / self.cfg.rope_theta.powf((4 * i) as f32 / hd as f32);
                        let (sh, ch) = (ph * freq).sin_cos();
                        let (sw, cw) = (pw * freq).sin_cos();
                        let x0 = x[base + i];
                        let x1 = x[base + quarter + i];
                        x[base + i] = x0 * ch - x1 * sh;
                        x[base + quarter + i] = x0 * sh + x1 * ch;
                        let y0 = x[base + 2 * quarter + i];
                        let y1 = x[base + 3 * quarter + i];
                        x[base + 2 * quarter + i] = y0 * cw - y1 * sw;
                        x[base + 3 * quarter + i] = y0 * sw + y1 * cw;
                    }
                }
            }
        };

        for blk in &self.blocks {
            let mut n1 = vec![0.0f32; n_pos * h];
            for p in 0..n_pos {
                let nr = layer_norm(
                    &cur[p * h..(p + 1) * h],
                    &blk.ln1_w,
                    &blk.ln1_b,
                    self.cfg.ln_eps,
                );
                n1[p * h..(p + 1) * h].copy_from_slice(&nr);
            }
            let mut q = Self::gpu_linear(ctx, &blk.wq, &n1, Some(&blk.bq), n_pos)?;
            let mut k = Self::gpu_linear(ctx, &blk.wk, &n1, Some(&blk.bk), n_pos)?;
            let v = Self::gpu_linear(ctx, &blk.wv, &n1, Some(&blk.bv), n_pos)?;
            apply_rope_2d(&mut q);
            apply_rope_2d(&mut k);

            let mut attn = vec![0.0f32; n_pos * h];
            for p in 0..n_pos {
                for head in 0..nh {
                    let qv = &q[(p * nh + head) * hd..(p * nh + head + 1) * hd];
                    let mut scores = vec![0.0f32; n_pos];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..n_pos {
                        let kv = &k[(j * nh + head) * hd..(j * nh + head + 1) * hd];
                        let s = qv.iter().zip(kv).map(|(&a, &b)| a * b).sum::<f32>()
                            * scale;
                        scores[j] = s;
                        mx = mx.max(s);
                    }
                    let mut denom = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - mx).exp();
                        denom += *s;
                    }
                    let dst = &mut attn[(p * nh + head) * hd..(p * nh + head + 1) * hd];
                    for (j, &sc) in scores.iter().enumerate() {
                        let w = sc / denom;
                        let vv = &v[(j * nh + head) * hd..(j * nh + head + 1) * hd];
                        for (o, &vi) in dst.iter_mut().zip(vv) {
                            *o += w * vi;
                        }
                    }
                }
            }
            let out_proj = Self::gpu_linear(ctx, &blk.wo, &attn, Some(&blk.bo), n_pos)?;
            for p in 0..n_pos {
                for c in 0..h {
                    cur[p * h + c] += out_proj[p * h + c] * blk.ls1[c];
                }
            }

            let mut n2 = vec![0.0f32; n_pos * h];
            for p in 0..n_pos {
                let nr = layer_norm(
                    &cur[p * h..(p + 1) * h],
                    &blk.ln2_w,
                    &blk.ln2_b,
                    self.cfg.ln_eps,
                );
                n2[p * h..(p + 1) * h].copy_from_slice(&nr);
            }
            let mut up =
                Self::gpu_linear(ctx, &blk.ff_up, &n2, Some(&blk.ff_up_b), n_pos)?;
            for v in &mut up {
                *v = gelu_erf(*v);
            }
            let down =
                Self::gpu_linear(ctx, &blk.ff_down, &up, Some(&blk.ff_down_b), n_pos)?;
            for p in 0..n_pos {
                for c in 0..h {
                    cur[p * h + c] += down[p * h + c] * blk.ls2[c];
                }
            }
        }

        // Drop CLS token and perform 2x2 spatial patch merge.
        let mh = grid_h / m;
        let mw = grid_w / m;
        let n_merged = mh * mw;
        let merged_dim = h * m * m;
        let mut merged = vec![0.0f32; n_merged * merged_dim];
        for by in 0..mh {
            for bx in 0..mw {
                let dst_base = (by * mw + bx) * merged_dim;
                let mut k = 0;
                for dy in 0..m {
                    for dx in 0..m {
                        let src_idx = 1 + (by * m + dy) * grid_w + (bx * m + dx);
                        merged[dst_base + k * h..dst_base + (k + 1) * h]
                            .copy_from_slice(&cur[src_idx * h..(src_idx + 1) * h]);
                        k += 1;
                    }
                }
            }
        }

        let mut h1 = Self::gpu_linear(ctx, &self.mm1, &merged, None, n_merged)?;
        for v in &mut h1 {
            *v = gelu_erf(*v);
        }
        let proj = Self::gpu_linear(ctx, &self.mm2, &h1, None, n_merged)?;
        Tensor::from_f32(ctx, &proj, &[n_merged, self.cfg.proj_dim])
    }
}
