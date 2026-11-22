//! Exact f32 CPU reference of every Inkling 2 operation (following
//! `llama.cpp/src/models/inkling2.cpp` and `inkling2-loop.cpp`, MIT).
//!
//! Used by `tests/unit/kernels/inkling2.rs` to verify every GPU kernel
//! against an independent CPU implementation.

/// RMSNorm without learned weight (`eps = 1e-6` for attention-residual mix
/// and loop controller, or `cfg.rms_eps` for value norm).
pub fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|&v| v * inv).collect()
}

/// RMSNorm with learned weight.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(&v, &g)| v * inv * g).collect()
}

/// L2 clip with optional per-channel scale:
/// `y = x * scale` (if `scale` is `Some`), then `y *= min(1, threshold / (||y||_2 + 1e-6))`.
pub fn l2_clip(x: &[f32], scale: Option<&[f32]>, threshold: f32) -> Vec<f32> {
    let mut y: Vec<f32> = match scale {
        Some(s) => x.iter().zip(s).map(|(&a, &b)| a * b).collect(),
        None => x.to_vec(),
    };
    let norm = y.iter().map(|&v| v * v).sum::<f32>().sqrt() + 1e-6;
    let factor = (threshold / norm).clamp(0.0, 1.0);
    for v in &mut y {
        *v *= factor;
    }
    y
}

/// Log-scale attention multiplier `tau(pos) = 1 + alpha * log(max(1, (pos + 1) / floor))`.
pub fn log_scale_tau(pos: usize, floor: usize, alpha: f32) -> f32 {
    let ratio = ((pos as f32) + 1.0) / (floor as f32);
    1.0 + alpha * ratio.max(1.0).ln()
}

/// Block attention residual mix (`llama_model_inkling2::graph::mix`):
/// Given `states` = `[blocks..., cur]` (each length `h`) and `queries` (`n_q` rows of `h`),
/// RMSNorm each state (no weight, eps = 1e-6), compute `scores = softmax((states @ q) / sqrt(h))`,
/// and return `scores @ states` (note: using the RMS-normalized states, matching `attn_res_normalize_values = true`).
pub fn attn_res_mix(states: &[&[f32]], queries: &[&[f32]], eps: f32) -> Vec<Vec<f32>> {
    let h = states[0].len();
    let normed: Vec<Vec<f32>> =
        states.iter().map(|s| rms_norm_no_weight(s, eps)).collect();
    let scale = 1.0 / (h as f32).sqrt();
    queries
        .iter()
        .map(|q| {
            let mut logits: Vec<f32> = normed
                .iter()
                .map(|s| s.iter().zip(*q).map(|(&a, &b)| a * b).sum::<f32>() * scale)
                .collect();
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in &mut logits {
                *v = (*v - mx).exp();
                sum += *v;
            }
            for v in &mut logits {
                *v /= sum;
            }
            let mut out = vec![0.0f32; h];
            for (w, s) in logits.iter().zip(&normed) {
                for (o, &sv) in out.iter_mut().zip(s) {
                    *o += w * sv;
                }
            }
            out
        })
        .collect()
}

/// Cross-pass loop controller (`llama_model_inkling2::graph::control`):
/// `gate = down @ [rms(cur); rms(msg)]`,
/// `out = (1 - tanh(forget @ gate)) * cur + tanh(gain @ gate) * msg`.
pub fn loop_control(
    cur: &[f32],
    msg: &[f32],
    down: &[f32],   // [rank, 2*h] row-major
    gain: &[f32],   // [h, rank] row-major
    forget: &[f32], // [h, rank] row-major
    rank: usize,
) -> Vec<f32> {
    let h = cur.len();
    let mut cat = rms_norm_no_weight(cur, 1e-6);
    cat.extend_from_slice(&rms_norm_no_weight(msg, 1e-6));
    let mut gate = vec![0.0f32; rank];
    for r in 0..rank {
        let row = &down[r * (2 * h)..(r + 1) * (2 * h)];
        gate[r] = row.iter().zip(&cat).map(|(&a, &b)| a * b).sum();
    }
    let mut out = vec![0.0f32; h];
    for i in 0..h {
        let g_row = &gain[i * rank..(i + 1) * rank];
        let f_row = &forget[i * rank..(i + 1) * rank];
        let g = g_row.iter().zip(&gate).map(|(&a, &b)| a * b).sum::<f32>().tanh();
        let f = f_row.iter().zip(&gate).map(|(&a, &b)| a * b).sum::<f32>().tanh();
        out[i] = (1.0 - f) * cur[i] + g * msg[i];
    }
    out
}

/// Causal short convolution with residual and optional dynamic per-head taps:
/// `x` has `n_tokens` rows of `n_channels`; `history` has `[n_channels, n_conv - 1]`
/// (oldest to newest, updated in-place to the last `n_conv - 1` inputs).
/// `weight` has `[n_channels, n_conv]` (oldest to newest, matching `ggml_ssm_conv`).
/// `dynamic` (if `Some`) has `[n_tokens, n_heads, n_conv]` where `lag = 0` is current
/// token (`t`), `lag = 1` is `t - 1`, ..., `lag = n_conv - 1` is `t - (n_conv - 1)`.
pub fn short_conv(
    x: &[f32],
    weight: &[f32],
    dynamic: Option<&[f32]>,
    history: &mut [f32],
    n_tokens: usize,
    n_channels: usize,
    n_conv: usize,
    n_heads: usize,
) -> Vec<f32> {
    let hist_len = n_conv - 1;
    let mut padded = vec![0.0f32; n_channels * (hist_len + n_tokens)];
    for c in 0..n_channels {
        for j in 0..hist_len {
            padded[c * (hist_len + n_tokens) + j] = history[j * n_channels + c];
        }
        for t in 0..n_tokens {
            padded[c * (hist_len + n_tokens) + hist_len + t] = x[t * n_channels + c];
        }
        // Update history `[hist_len, n_channels]` with the last `hist_len` values.
        for j in 0..hist_len {
            history[j * n_channels + c] =
                padded[c * (hist_len + n_tokens) + n_tokens + j];
        }
    }
    let head_dim = if n_heads > 0 { n_channels / n_heads } else { 0 };
    let scale =
        1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * n_conv + 1) as f32).sqrt();
    let mut out = vec![0.0f32; n_tokens * n_channels];
    for t in 0..n_tokens {
        for c in 0..n_channels {
            let mut acc = 0.0f32;
            let p_base = c * (hist_len + n_tokens) + t;
            let w_base = c * n_conv;
            for k in 0..n_conv {
                acc += padded[p_base + k] * weight[w_base + k];
            }
            if let Some(dyn_w) = dynamic {
                let head = c / head_dim;
                let d_base = (t * n_heads + head) * n_conv;
                for lag in 0..n_conv {
                    // lag=0 is current token (p_base + n_conv - 1), lag=n_conv-1 is oldest (p_base).
                    acc += padded[p_base + (n_conv - 1 - lag)] * dyn_w[d_base + lag];
                }
            }
            out[t * n_channels + c] = (x[t * n_channels + c] + acc) * scale;
        }
    }
    out
}

/// SITU activation: `beta * tanh(gate / beta) * sigmoid(gate) * (lb * tanh(up / lb))`.
pub fn situ(gate: f32, up: f32, beta: f32, linear_beta: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-gate).exp());
    let act = beta * (gate / beta).tanh() * sig;
    let lin =
        if linear_beta > 0.0 { linear_beta * (up / linear_beta).tanh() } else { up };
    act * lin
}

/// Sigmoid top-k MoE router with optional bias, L1 normalization (`eps = 1e-20`),
/// and `route_scale`.
pub fn moe_route(
    logits: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    route_scale: f32,
) -> (Vec<u32>, Vec<f32>) {
    let probs: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let mut scored: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p + bias.map_or(0.0, |b| b[i])))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let ids: Vec<u32> = scored[..top_k].iter().map(|&(i, _)| i as u32).collect();
    let raw_w: Vec<f32> = ids.iter().map(|&i| probs[i as usize]).collect();
    let sum: f32 = raw_w.iter().sum::<f32>().max(1e-20);
    let weights: Vec<f32> = raw_w.iter().map(|&w| (w / sum) * route_scale).collect();
    (ids, weights)
}

/// Relative-position bias attention with zero attention sink (no RoPE):
/// For a single query head `q` (`[head_dim]`) at position `pos_q` attending to
/// `n_kv` keys/values at positions `pos_k[0..n_kv]`:
/// `score[j] = (q . k[j]) / sqrt(head_dim) + rel_table[dist] - sink`
/// where `dist = pos_q - pos_k[j]` if `0 <= dist < extent` else `extent` (zero bias),
/// and softmax includes an implicit zero-logit sink (`exp(0) = 1` in denominator).
pub fn rel_sink_attention(
    q: &[f32],
    k: &[f32],         // [n_kv, head_dim]
    v: &[f32],         // [n_kv, head_dim]
    rel_table: &[f32], // [extent] (R . attn_rel_bias; index `extent` is 0.0)
    sink: f32,
    pos_q: usize,
    pos_k: &[usize],
    window: Option<usize>,
) -> Vec<f32> {
    let hd = q.len();
    let n_kv = pos_k.len();
    let extent = rel_table.len();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut scores = vec![f32::NEG_INFINITY; n_kv];
    let mut max_s = 0.0f32; // zero sink logit!
    for j in 0..n_kv {
        if pos_k[j] > pos_q {
            continue;
        }
        let dist = pos_q - pos_k[j];
        if let Some(w) = window
            && dist >= w
        {
            continue;
        }
        let dot: f32 =
            q.iter().zip(&k[j * hd..(j + 1) * hd]).map(|(&a, &b)| a * b).sum();
        let rb = if dist < extent { rel_table[dist] } else { 0.0 };
        let s = dot * scale + rb - sink;
        scores[j] = s;
        if s > max_s {
            max_s = s;
        }
    }
    let mut denom = (-max_s).exp(); // zero sink
    for s in &mut scores {
        if s.is_finite() {
            *s = (*s - max_s).exp();
            denom += *s;
        } else {
            *s = 0.0;
        }
    }
    let mut out = vec![0.0f32; hd];
    for (j, &p) in scores.iter().enumerate() {
        if p == 0.0 {
            continue;
        }
        let w = p / denom;
        for (o, &vj) in out.iter_mut().zip(&v[j * hd..(j + 1) * hd]) {
            *o += w * vj;
        }
    }
    out
}

/// Final logit softcapping: `cap * tanh(x / cap)`.
pub fn softcap(x: f32, cap: f32) -> f32 {
    if cap > 0.0 { cap * (x / cap).tanh() } else { x }
}
