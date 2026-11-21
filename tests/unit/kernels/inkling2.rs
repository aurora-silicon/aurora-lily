use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::gguf::{GgmlType, dequantize};
use crate::inkling2::reference as cref;
use crate::metal::MetalContext;
use crate::tensor::{DType, Tensor};

fn ctx() -> MetalContext {
    MetalContext::new().expect("Vulkan/Metal GPU context")
}

fn f32s(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn run(ctx: &MetalContext, f: impl FnOnce(&ComputePass<'_>)) {
    let pass = ctx.begin().unwrap();
    f(&pass);
    pass.commit_wait().unwrap();
}

#[test]
fn rms_norm_and_l2_clip_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(101);
    let (rows, h) = (5, 256);
    let x = f32s(&mut rng, rows * h, -2.0, 2.0);
    let w = f32s(&mut rng, h, 0.5, 1.5);
    let tx = Tensor::from_f32(&ctx, &x, &[rows, h]).unwrap();
    let tw = Tensor::from_f32(&ctx, &w, &[h]).unwrap();
    let ty = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
    run(&ctx, |p| rms_norm(&ctx, p, &tx, Some(&tw), &ty, rows, h, 1e-6).unwrap());
    let expected: Vec<f32> = (0..rows)
        .flat_map(|r| cref::rms_norm(&x[r * h..(r + 1) * h], &w, 1e-6))
        .collect();
    assert_close(&ty.to_f32().unwrap(), &expected, 1e-5, 1e-5);

    // L2 clip with per-channel scale and log-scale tau.
    let (nh, dim) = (4, 128);
    let q = f32s(&mut rng, rows * nh * dim, -3.0, 3.0);
    let sc = f32s(&mut rng, nh * dim, 0.5, 1.5);
    let tq = Tensor::from_f32(&ctx, &q, &[rows, nh, dim]).unwrap();
    let tsc = Tensor::from_f32(&ctx, &sc, &[nh, dim]).unwrap();
    let tq_out = Tensor::zeros(&ctx, &[rows, nh, dim], DType::F32).unwrap();
    let (pos0, floor, alpha, thresh) = (10_000, 8192, 0.1, 22.627_417);
    run(&ctx, |p| {
        l2_clip(
            &ctx,
            p,
            &tq,
            Some(&tsc),
            &tq_out,
            rows,
            nh,
            dim,
            thresh,
            pos0,
            floor,
            alpha,
        )
        .unwrap()
    });
    let mut exp_q = Vec::with_capacity(q.len());
    for r in 0..rows {
        let tau = cref::log_scale_tau(pos0 + r, floor, alpha);
        for head in 0..nh {
            let s = (r * nh + head) * dim;
            let clipped = cref::l2_clip(
                &q[s..s + dim],
                Some(&sc[head * dim..(head + 1) * dim]),
                thresh,
            );
            exp_q.extend(clipped.into_iter().map(|v| v * tau));
        }
    }
    assert_close(&tq_out.to_f32().unwrap(), &exp_q, 1e-4, 1e-4);
}

#[test]
fn attn_res_mix_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(102);
    let (rows, h, n_states, n_q) = (4, 256, 3, 4);
    let states_cpu: Vec<Vec<f32>> =
        (0..n_states).map(|_| f32s(&mut rng, rows * h, -1.5, 1.5)).collect();
    let states_gpu: Vec<Tensor> = states_cpu
        .iter()
        .map(|v| Tensor::from_f32(&ctx, v, &[rows, h]).unwrap())
        .collect();
    let queries_cpu = f32s(&mut rng, n_q * h, -0.5, 0.5);
    let queries_gpu = Tensor::from_f32(&ctx, &queries_cpu, &[n_q, h]).unwrap();
    let out_gpu = Tensor::zeros(&ctx, &[n_q, rows, h], DType::F32).unwrap();
    let ptrs = Tensor::zeros(&ctx, &[32], DType::U32).unwrap();
    let mut raw = [0u64; 16];
    for (i, t) in states_gpu.iter().enumerate() {
        raw[i] = t.buffer().address() + t.binding().1 as u64;
    }
    ptrs.write_bytes(bytemuck::cast_slice(&raw)).unwrap();

    run(&ctx, |p| {
        attn_res_mix(
            &ctx,
            p,
            &ptrs,
            &queries_gpu,
            &out_gpu,
            rows,
            h,
            n_states,
            n_q,
            1e-6,
        )
        .unwrap()
    });

    let got = out_gpu.to_f32().unwrap();
    let q_slices: Vec<&[f32]> =
        (0..n_q).map(|q| &queries_cpu[q * h..(q + 1) * h]).collect();
    let mut exp = vec![0.0f32; n_q * rows * h];
    for r in 0..rows {
        let s_slices: Vec<&[f32]> =
            states_cpu.iter().map(|s| &s[r * h..(r + 1) * h]).collect();
        let mixed = cref::attn_res_mix(&s_slices, &q_slices, 1e-6);
        for q in 0..n_q {
            exp[(q * rows + r) * h..(q * rows + r + 1) * h].copy_from_slice(&mixed[q]);
        }
    }
    assert_close(&got, &exp, 1e-4, 1e-4);
}

#[test]
fn short_conv_static_and_dynamic_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(103);
    let (rows, n_heads, hd, n_conv) = (7, 4, 32, 4);
    let c = n_heads * hd;
    let x = f32s(&mut rng, rows * c, -1.0, 1.0);
    let w = f32s(&mut rng, c * n_conv, -0.5, 0.5);
    let dyn_w = f32s(&mut rng, rows * n_heads * n_conv, -0.5, 0.5);
    let mut hist_cpu = f32s(&mut rng, c * (n_conv - 1), -0.5, 0.5);

    let tx = Tensor::from_f32(&ctx, &x, &[rows, c]).unwrap();
    let tw = Tensor::from_f32(&ctx, &w, &[c, n_conv]).unwrap();
    let td = Tensor::from_f32(&ctx, &dyn_w, &[rows, n_heads, n_conv]).unwrap();
    let th = Tensor::from_f32(&ctx, &hist_cpu, &[c, n_conv - 1]).unwrap();
    let ty = Tensor::zeros(&ctx, &[rows, c], DType::F32).unwrap();

    run(&ctx, |p| {
        short_conv(&ctx, p, &tx, &tw, Some(&td), &th, &ty, rows, c, n_conv, n_heads)
            .unwrap()
    });
    let exp =
        cref::short_conv(&x, &w, Some(&dyn_w), &mut hist_cpu, rows, c, n_conv, n_heads);
    assert_close(&ty.to_f32().unwrap(), &exp, 1e-5, 1e-5);
    assert_close(&th.to_f32().unwrap(), &hist_cpu, 1e-6, 1e-6);
}

#[test]
fn rel_sink_attention_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(104);
    let (nh, nkv, hd, ext, window, slots) = (6, 2, 128, 32, 16, 32);
    let (n_prefill, n_step) = (12, 3);
    let total = n_prefill + n_step;
    let k_all = f32s(&mut rng, total * nkv * hd, -1.0, 1.0);
    let v_all = f32s(&mut rng, total * nkv * hd, -1.0, 1.0);
    let q_step = f32s(&mut rng, n_step * nh * hd, -1.0, 1.0);
    let rel_tab = f32s(&mut rng, n_step * nh * ext, -0.5, 0.5);
    let sinks = f32s(&mut rng, n_step * nh, -0.5, 0.5);

    let tk_cache = Tensor::zeros(&ctx, &[slots, nkv * hd], DType::F16).unwrap();
    let tv_cache = Tensor::zeros(&ctx, &[slots, nkv * hd], DType::F16).unwrap();
    let tk_in = Tensor::from_f32(&ctx, &k_all, &[total, nkv * hd]).unwrap();
    let tv_in = Tensor::from_f32(&ctx, &v_all, &[total, nkv * hd]).unwrap();
    let tq = Tensor::from_f32(&ctx, &q_step, &[n_step, nh * hd]).unwrap();
    let trel = Tensor::from_f32(&ctx, &rel_tab, &[n_step, nh * ext]).unwrap();
    let tsink = Tensor::from_f32(&ctx, &sinks, &[n_step, nh]).unwrap();
    let tout = Tensor::zeros(&ctx, &[n_step, nh * hd], DType::F32).unwrap();

    run(&ctx, |p| {
        kv_store(
            &ctx,
            p,
            &tk_in,
            &tv_in,
            &tk_cache,
            &tv_cache,
            total,
            nkv * hd,
            0,
            slots,
            true,
        )
        .unwrap();
        p.memory_barrier().unwrap();
        attention(
            &ctx, p, &tq, &tk_cache, &tv_cache, &trel, &tsink, &tout, n_step, nh, nkv,
            n_prefill, slots, true, window, ext,
        )
        .unwrap();
    });

    // Round K/V through f16 as the cache does.
    let k16: Vec<f32> =
        k_all.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let v16: Vec<f32> =
        v_all.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let mut exp = vec![0.0f32; n_step * nh * hd];
    let pos_k: Vec<usize> = (0..total).collect();
    for r in 0..n_step {
        let pos_q = n_prefill + r;
        for h in 0..nh {
            let kh = h / (nh / nkv);
            let k_head: Vec<f32> = (0..total)
                .flat_map(|t| {
                    k16[(t * nkv + kh) * hd..(t * nkv + kh + 1) * hd].iter().copied()
                })
                .collect();
            let v_head: Vec<f32> = (0..total)
                .flat_map(|t| {
                    v16[(t * nkv + kh) * hd..(t * nkv + kh + 1) * hd].iter().copied()
                })
                .collect();
            let qv = &q_step[(r * nh + h) * hd..(r * nh + h + 1) * hd];
            let rt = &rel_tab[(r * nh + h) * ext..(r * nh + h + 1) * ext];
            let o = cref::rel_sink_attention(
                qv,
                &k_head,
                &v_head,
                rt,
                sinks[r * nh + h],
                pos_q,
                &pos_k,
                Some(window),
            );
            exp[(r * nh + h) * hd..(r * nh + h + 1) * hd].copy_from_slice(&o);
        }
    }
    assert_close(&tout.to_f32().unwrap(), &exp, 1e-3, 1e-3);
}

#[test]
fn router_situ_and_moe_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(105);
    let (rows, n_exp, top_k, ff, h) = (3, 16, 4, 64, 64);
    let logits = f32s(&mut rng, rows * n_exp, -2.0, 2.0);
    let bias = f32s(&mut rng, n_exp, -0.3, 0.3);
    let tl = Tensor::from_f32(&ctx, &logits, &[rows, n_exp]).unwrap();
    let tb = Tensor::from_f32(&ctx, &bias, &[n_exp]).unwrap();
    let tids = Tensor::zeros(&ctx, &[rows * top_k], DType::U32).unwrap();
    let tw = Tensor::zeros(&ctx, &[rows * top_k], DType::F32).unwrap();
    run(&ctx, |p| {
        router(&ctx, p, &tl, Some(&tb), &tids, &tw, rows, n_exp, top_k, 8.0).unwrap()
    });

    let got_ids = tids.to_u32().unwrap();
    let got_w = tw.to_f32().unwrap();
    for r in 0..rows {
        let (e_ids, e_w) = cref::moe_route(
            &logits[r * n_exp..(r + 1) * n_exp],
            Some(&bias),
            top_k,
            8.0,
        );
        assert_eq!(&got_ids[r * top_k..(r + 1) * top_k], &e_ids[..]);
        assert_close(&got_w[r * top_k..(r + 1) * top_k], &e_w, 1e-5, 1e-5);
    }

    // SITU + per-expert scale.
    let gu = f32s(&mut rng, rows * top_k * 2 * ff, -2.0, 2.0);
    let esc = f32s(&mut rng, n_exp, 0.5, 1.5);
    let tgu = Tensor::from_f32(&ctx, &gu, &[rows * top_k, 2 * ff]).unwrap();
    let tesc = Tensor::from_f32(&ctx, &esc, &[n_exp]).unwrap();
    let tsitu = Tensor::zeros(&ctx, &[rows * top_k, ff], DType::F32).unwrap();
    run(&ctx, |p| {
        situ(&ctx, p, &tgu, &tids, Some(&tesc), &tsitu, rows * top_k, ff, 4.0, 25.0)
            .unwrap()
    });
    let mut exp_s = vec![0.0f32; rows * top_k * ff];
    for rk in 0..rows * top_k {
        let sc = esc[got_ids[rk] as usize];
        for i in 0..ff {
            exp_s[rk * ff + i] = cref::situ(
                gu[rk * 2 * ff + i] * sc,
                gu[rk * 2 * ff + ff + i] * sc,
                4.0,
                25.0,
            );
        }
    }
    assert_close(&tsitu.to_f32().unwrap(), &exp_s, 1e-4, 1e-4);

    // MoE projection over NVFP4 weights.
    let ty = GgmlType::Nvfp4;
    let total_vals = n_exp * h * ff;
    let mut raw: Vec<u8> = (0..total_vals / ty.block_values() * ty.block_bytes())
        .map(|_| rng.r#gen())
        .collect();
    for blk in raw.chunks_exact_mut(ty.block_bytes()) {
        for b in &mut blk[..4] {
            *b = rng.gen_range(16..64);
        }
    }
    let qw = QWeight::from_bytes(&ctx, &raw, ty, ff, n_exp * h).unwrap();
    let tdown = Tensor::zeros(&ctx, &[rows * top_k, h], DType::F32).unwrap();
    run(&ctx, |p| {
        moe_proj(&ctx, p, &tsitu, &qw, &tids, &tdown, rows * top_k, h, 0).unwrap()
    });

    let mut w_f32 = vec![0.0f32; total_vals];
    dequantize(ty, &raw, &mut w_f32).unwrap();
    let mut exp_down = vec![0.0f32; rows * top_k * h];
    for rk in 0..rows * top_k {
        let e = got_ids[rk] as usize;
        for o in 0..h {
            let wr = &w_f32[(e * h + o) * ff..(e * h + o + 1) * ff];
            let xr = &exp_s[rk * ff..(rk + 1) * ff];
            exp_down[rk * h + o] = wr.iter().zip(xr).map(|(&a, &b)| a * b).sum();
        }
    }
    assert_close(&tdown.to_f32().unwrap(), &exp_down, 1e-3, 1e-3);
}

#[test]
fn fused_moe_and_grouped_m4_match_unfused() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(106);
    let (rows, n_exp, top_k, ff, h) = (7, 8, 4, 64, 64);
    let rows_k = rows * top_k;
    let x = f32s(&mut rng, rows * h, -1.0, 1.0);
    let gu_w = f32s(&mut rng, n_exp * 2 * ff * h, -0.5, 0.5);
    let down_w = f32s(&mut rng, n_exp * h * ff, -0.5, 0.5);
    let gu_sc = f32s(&mut rng, n_exp, 0.6, 1.4);
    let down_sc = f32s(&mut rng, n_exp, 0.6, 1.4);
    let mut ids = Vec::with_capacity(rows_k);
    let mut weights = Vec::with_capacity(rows_k);
    for _ in 0..rows {
        let mut chosen = Vec::new();
        while chosen.len() < top_k {
            let e = rng.gen_range(0..n_exp) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        ids.extend_from_slice(&chosen);
        for _ in 0..top_k {
            weights.push(rng.gen_range(0.1f32..2.0));
        }
    }

    let tx = Tensor::from_f32(&ctx, &x, &[rows, h]).unwrap();
    let q_gu = QWeight::from_bytes(
        &ctx,
        bytemuck::cast_slice(&gu_w),
        GgmlType::F32,
        h,
        n_exp * 2 * ff,
    )
    .unwrap();
    let q_down = QWeight::from_bytes(
        &ctx,
        bytemuck::cast_slice(&down_w),
        GgmlType::F32,
        ff,
        n_exp * h,
    )
    .unwrap();
    let tids = Tensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        &[rows_k],
        DType::U32,
    )
    .unwrap();
    let tw = Tensor::from_f32(&ctx, &weights, &[rows_k]).unwrap();
    let tgu_sc = Tensor::from_f32(&ctx, &gu_sc, &[n_exp]).unwrap();
    let tdown_sc = Tensor::from_f32(&ctx, &down_sc, &[n_exp]).unwrap();

    // Unfused reference on GPU.
    let ref_gu = Tensor::zeros(&ctx, &[rows_k, 2 * ff], DType::F32).unwrap();
    let ref_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
    let ref_down = Tensor::zeros(&ctx, &[rows_k, h], DType::F32).unwrap();
    let ref_out = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
    run(&ctx, |p| {
        moe_proj(&ctx, p, &tx, &q_gu, &tids, &ref_gu, rows_k, 2 * ff, top_k).unwrap();
        p.memory_barrier().unwrap();
        situ(&ctx, p, &ref_gu, &tids, Some(&tgu_sc), &ref_situ, rows_k, ff, 4.0, 25.0)
            .unwrap();
        p.memory_barrier().unwrap();
        moe_proj(&ctx, p, &ref_situ, &q_down, &tids, &ref_down, rows_k, h, 0).unwrap();
        p.memory_barrier().unwrap();
        moe_sum(&ctx, p, &ref_down, &tids, &tw, Some(&tdown_sc), &ref_out, rows, top_k, h)
            .unwrap();
    });
    let expected = ref_out.to_f32().unwrap();

    // 1. Fused decode kernels (`moe_gate_up_situ` + `moe_down_sum`).
    let f_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
    let f_out = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
    run(&ctx, |p| {
        moe_gate_up_situ(
            &ctx, p, &tx, &q_gu, &tids, Some(&tgu_sc), &f_situ, rows_k, ff, top_k, 4.0,
            25.0,
        )
        .unwrap();
        p.memory_barrier().unwrap();
        moe_down_sum(
            &ctx, p, &f_situ, &q_down, &tids, &tw, Some(&tdown_sc), &f_out, rows, top_k, h,
        )
        .unwrap();
    });
    assert_close(&f_out.to_f32().unwrap(), &expected, 1e-4, 1e-4);

    // 2. Grouped m4 kernels (`moe_bucket` + `moe_gu_m4` + `moe_down_m4` + `moe_sum`).
    let sorted_rk = Tensor::zeros(&ctx, &[rows_k], DType::U32).unwrap();
    let tiles = Tensor::zeros(&ctx, &[rows_k * 4], DType::U32).unwrap();
    let n_tiles = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
    let g_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
    let g_down = Tensor::zeros(&ctx, &[rows_k, h], DType::F32).unwrap();
    let g_out = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
    let max_tiles = rows_k.min((rows_k + 3 * n_exp) / 4);
    run(&ctx, |p| {
        moe_bucket(&ctx, p, &tids, &sorted_rk, &tiles, &n_tiles, rows_k, n_exp).unwrap();
        p.memory_barrier().unwrap();
        moe_gu_m4(
            &ctx,
            p,
            &tx,
            &q_gu,
            &sorted_rk,
            &tiles,
            &n_tiles,
            Some(&tgu_sc),
            &g_situ,
            max_tiles,
            ff,
            top_k,
            4.0,
            25.0,
        )
        .unwrap();
        p.memory_barrier().unwrap();
        moe_down_m4(
            &ctx, p, &g_situ, &q_down, &sorted_rk, &tiles, &n_tiles, &g_down, max_tiles,
            h,
        )
        .unwrap();
        p.memory_barrier().unwrap();
        moe_sum(&ctx, p, &g_down, &tids, &tw, Some(&tdown_sc), &g_out, rows, top_k, h)
            .unwrap();
    });
    assert_close(&g_out.to_f32().unwrap(), &expected, 1e-4, 1e-4);
}
