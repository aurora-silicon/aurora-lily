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

#[test]
fn attention_split_matches_cpu_swa_and_global() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(201);
    let (nh, nkv, hd, ext, window, slots) = (6, 2, 128, 64, 160, 200);
    for (is_swa, n_prefill, n_step) in [
        (true, 175, 4),
        (true, 175, 3),
        (false, 310, 4),
        (false, 120, 5),
        (false, 0, 1),
    ] {
        let total = n_prefill + n_step;
        let cap = if is_swa { slots } else { total.max(64) };
        let k_all = f32s(&mut rng, total * nkv * hd, -1.0, 1.0);
        let v_all = f32s(&mut rng, total * nkv * hd, -1.0, 1.0);
        let q_step = f32s(&mut rng, n_step * nh * hd, -2.0, 2.0);
        let q_clip = f32s(&mut rng, nh * hd, 0.5, 1.5);
        let gate = f32s(&mut rng, n_step * nh * hd, -2.0, 2.0);
        let rel_tab = f32s(&mut rng, n_step * nh * ext, -0.5, 0.5);
        let sinks = f32s(&mut rng, n_step * nh, -0.5, 0.5);

        let tk_cache = Tensor::zeros(&ctx, &[cap, nkv * hd], DType::F16).unwrap();
        let tv_cache = Tensor::zeros(&ctx, &[cap, nkv * hd], DType::F16).unwrap();
        let tk_in = Tensor::from_f32(&ctx, &k_all, &[total, nkv * hd]).unwrap();
        let tv_in = Tensor::from_f32(&ctx, &v_all, &[total, nkv * hd]).unwrap();
        let tq = Tensor::from_f32(&ctx, &q_step, &[n_step, nh * hd]).unwrap();
        let tqc = Tensor::from_f32(&ctx, &q_clip, &[nh, hd]).unwrap();
        let tgate = Tensor::from_f32(&ctx, &gate, &[n_step, nh * hd]).unwrap();
        let trel = Tensor::from_f32(&ctx, &rel_tab, &[n_step, nh * ext]).unwrap();
        let tsink = Tensor::from_f32(&ctx, &sinks, &[n_step, nh]).unwrap();
        let tpart = Tensor::zeros(&ctx, &[n_step, nh, ATTN_SPLIT_MAX_CHUNKS, 130], DType::F32)
            .unwrap();
        let tout = Tensor::zeros(&ctx, &[n_step, nh * hd], DType::F32).unwrap();

        let (qk_clip, lf, la) = (22.627_417f32, if is_swa { 0 } else { 128 }, if is_swa { 0.0 } else { 0.1 });
        let pass = ctx.begin_concurrent().unwrap();
        kv_store(&ctx, &pass, &tk_in, &tv_in, &tk_cache, &tv_cache, total, nkv * hd, 0, cap, is_swa)
            .unwrap();
        pass.memory_barrier().unwrap();
        attention_split_fused(
            &ctx,
            &pass,
            &tq,
            &tk_cache,
            &tv_cache,
            &trel,
            &tsink,
            &tpart,
            &tout,
            Some(&tgate),
            Some(&tqc),
            n_step,
            nh,
            nkv,
            n_prefill,
            cap,
            is_swa,
            window,
            ext,
            qk_clip,
            lf,
            la,
        )
        .unwrap();
        pass.commit_wait().unwrap();

        let k16: Vec<f32> = k_all.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let v16: Vec<f32> = v_all.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let mut exp = vec![0.0f32; n_step * nh * hd];
        let pos_k: Vec<usize> = (0..total).collect();
        for r in 0..n_step {
            let pos_q = n_prefill + r;
            let tau = if lf > 0 { cref::log_scale_tau(pos_q, lf, la) } else { 1.0 };
            for h in 0..nh {
                let kh = h / (nh / nkv);
                let k_head: Vec<f32> = (0..total)
                    .flat_map(|t| k16[(t * nkv + kh) * hd..(t * nkv + kh + 1) * hd].iter().copied())
                    .collect();
                let v_head: Vec<f32> = (0..total)
                    .flat_map(|t| v16[(t * nkv + kh) * hd..(t * nkv + kh + 1) * hd].iter().copied())
                    .collect();
                let q_raw = &q_step[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                let qv: Vec<f32> = cref::l2_clip(q_raw, Some(&q_clip[h * hd..(h + 1) * hd]), qk_clip)
                    .into_iter()
                    .map(|v| v * tau)
                    .collect();
                let rt = &rel_tab[(r * nh + h) * ext..(r * nh + h + 1) * ext];
                let mut o = cref::rel_sink_attention(
                    &qv,
                    &k_head,
                    &v_head,
                    rt,
                    sinks[r * nh + h],
                    pos_q,
                    &pos_k,
                    if is_swa { Some(window) } else { None },
                );
                let g_slice = &gate[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                for (oi, &gi) in o.iter_mut().zip(g_slice) {
                    *oi /= 1.0 + (-gi).exp();
                }
                exp[(r * nh + h) * hd..(r * nh + h + 1) * hd].copy_from_slice(&o);
            }
        }
        assert_close(&tout.to_f32().unwrap(), &exp, 1e-3, 1e-3);
    }
}

#[test]
fn dspark_attn_split_matches_baseline() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(202);
    let (nh, nkv, hd, block) = (12, 4, 128, 4);
    for (ctx_start, ctx_pos) in [(0, 0), (0, 27), (45, 310)] {
        let cap = ctx_pos.max(64);
        let q = f32s(&mut rng, block * nh * hd, -1.0, 1.0);
        let k_ctx_f32 = f32s(&mut rng, cap * nkv * hd, -1.0, 1.0);
        let v_ctx_f32 = f32s(&mut rng, cap * nkv * hd, -1.0, 1.0);
        let k_blk = f32s(&mut rng, block * nkv * hd, -1.0, 1.0);
        let v_blk = f32s(&mut rng, block * nkv * hd, -1.0, 1.0);

        let k_ctx_bits: Vec<u16> = k_ctx_f32.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let v_ctx_bits: Vec<u16> = v_ctx_f32.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let tq = Tensor::from_f32(&ctx, &q, &[block, nh * hd]).unwrap();
        let tkc = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&k_ctx_bits), &[cap, nkv * hd], DType::F16).unwrap();
        let tvc = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&v_ctx_bits), &[cap, nkv * hd], DType::F16).unwrap();
        let tkb = Tensor::from_f32(&ctx, &k_blk, &[block, nkv * hd]).unwrap();
        let tvb = Tensor::from_f32(&ctx, &v_blk, &[block, nkv * hd]).unwrap();
        let tpart = Tensor::zeros(&ctx, &[block, nh, DSPARK_SPLIT_MAX_CHUNKS + 1, 130], DType::F32).unwrap();
        let tout_ref = Tensor::zeros(&ctx, &[block, nh * hd], DType::F32).unwrap();
        let tout_sp = Tensor::zeros(&ctx, &[block, nh * hd], DType::F32).unwrap();

        let pass = ctx.begin_concurrent().unwrap();
        dspark_attn(&ctx, &pass, &tq, &tkc, &tvc, &tkb, &tvb, &tout_ref, nh, nkv, hd, ctx_start, ctx_pos, block).unwrap();
        dspark_attn_split(&ctx, &pass, &tq, &tkc, &tvc, &tkb, &tvb, &tpart, &tout_sp, nh, nkv, hd, ctx_start, ctx_pos, block).unwrap();
        pass.commit_wait().unwrap();
        assert_close(&tout_sp.to_f32().unwrap(), &tout_ref.to_f32().unwrap(), 1e-4, 1e-4);
    }
}

#[test]
fn rel_bias_clipped_tiled_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(203);
    let (rows, nh, d, ext) = (5, 24, 16, 1024);
    let r_proj = f32s(&mut rng, rows * nh * d, -2.0, 2.0);
    let r_clip = f32s(&mut rng, nh * d, 0.5, 1.5);
    let bias = f32s(&mut rng, ext * d, -0.5, 0.5);

    let tr = Tensor::from_f32(&ctx, &r_proj, &[rows * nh, d]).unwrap();
    let trc = Tensor::from_f32(&ctx, &r_clip, &[nh, d]).unwrap();
    let qb = QWeight::from_bytes(&ctx, bytemuck::cast_slice(&bias), GgmlType::F32, d, ext).unwrap();
    let tout = Tensor::zeros(&ctx, &[rows * nh, ext], DType::F32).unwrap();
    let (thresh, pos0, lf, la) = (8.0f32, 9000, 8192, 0.1f32);
    run(&ctx, |p| {
        rel_bias_clipped(&ctx, p, &tr, &qb, &tout, Some(&trc), rows * nh, nh, thresh, pos0, lf, la)
            .unwrap()
    });

    let mut exp = vec![0.0f32; rows * nh * ext];
    for r in 0..rows {
        let tau = cref::log_scale_tau(pos0 + r, lf, la);
        for h in 0..nh {
            let idx = r * nh + h;
            let clipped: Vec<f32> = cref::l2_clip(
                &r_proj[idx * d..(idx + 1) * d],
                Some(&r_clip[h * d..(h + 1) * d]),
                thresh,
            )
            .into_iter()
            .map(|v| v * tau)
            .collect();
            for e in 0..ext {
                exp[idx * ext + e] = clipped
                    .iter()
                    .zip(&bias[e * d..(e + 1) * d])
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
        }
    }
    assert_close(&tout.to_f32().unwrap(), &exp, 1e-4, 1e-4);
}

#[test]
fn short_conv_prefill_and_kv_prep_prefill_match_sequential() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(204);
    let (rows, nkv, hd, n_conv) = (17, 4, 128, 4);
    let c = nkv * hd;
    let x = f32s(&mut rng, rows * c, -1.0, 1.0);
    let res = f32s(&mut rng, rows * c, -1.0, 1.0);
    let w = f32s(&mut rng, c * n_conv, -0.5, 0.5);
    let dyn_w = f32s(&mut rng, rows * nkv * n_conv, -0.5, 0.5);
    let hist0 = f32s(&mut rng, c * (n_conv - 1), -0.5, 0.5);

    let tx = Tensor::from_f32(&ctx, &x, &[rows, c]).unwrap();
    let tres = Tensor::from_f32(&ctx, &res, &[rows, c]).unwrap();
    let tw = Tensor::from_f32(&ctx, &w, &[c, n_conv]).unwrap();
    let td = Tensor::from_f32(&ctx, &dyn_w, &[rows, nkv, n_conv]).unwrap();
    let th = Tensor::from_f32(&ctx, &hist0, &[c, n_conv - 1]).unwrap();
    let ty = Tensor::zeros(&ctx, &[rows, c], DType::F32).unwrap();

    let pass = ctx.begin_concurrent().unwrap();
    short_conv_prefill(&ctx, &pass, &tx, &tw, Some(&td), &th, &ty, Some(&tres), rows, c, n_conv, nkv)
        .unwrap();
    pass.memory_barrier().unwrap();
    short_conv_hist_update(&ctx, &pass, &tx, &th, rows, c, n_conv - 1).unwrap();
    pass.commit_wait().unwrap();

    let mut hist_cpu = hist0.clone();
    let mut exp = cref::short_conv(&x, &w, Some(&dyn_w), &mut hist_cpu, rows, c, n_conv, nkv);
    for (o, &r) in exp.iter_mut().zip(&res) {
        *o += r;
    }
    assert_close(&ty.to_f32().unwrap(), &exp, 1e-5, 1e-5);
    assert_close(&th.to_f32().unwrap(), &hist_cpu, 1e-6, 1e-6);

    // kv_prep_prefill vs kv_prep
    let v_raw = f32s(&mut rng, rows * c, -1.0, 1.0);
    let vw = f32s(&mut rng, c * n_conv, -0.5, 0.5);
    let vdyn = f32s(&mut rng, rows * nkv * n_conv, -0.5, 0.5);
    let vhist0 = f32s(&mut rng, c * (n_conv - 1), -0.5, 0.5);
    let tv = Tensor::from_f32(&ctx, &v_raw, &[rows, c]).unwrap();
    let tvw = Tensor::from_f32(&ctx, &vw, &[c, n_conv]).unwrap();
    let tvd = Tensor::from_f32(&ctx, &vdyn, &[rows, nkv, n_conv]).unwrap();

    let kh_seq = Tensor::from_f32(&ctx, &hist0, &[c, n_conv - 1]).unwrap();
    let vh_seq = Tensor::from_f32(&ctx, &vhist0, &[c, n_conv - 1]).unwrap();
    let kc_seq = Tensor::zeros(&ctx, &[32, c], DType::F16).unwrap();
    let vc_seq = Tensor::zeros(&ctx, &[32, c], DType::F16).unwrap();

    let kh_pre = Tensor::from_f32(&ctx, &hist0, &[c, n_conv - 1]).unwrap();
    let vh_pre = Tensor::from_f32(&ctx, &vhist0, &[c, n_conv - 1]).unwrap();
    let kc_pre = Tensor::zeros(&ctx, &[32, c], DType::F16).unwrap();
    let vc_pre = Tensor::zeros(&ctx, &[32, c], DType::F16).unwrap();

    let pass = ctx.begin_concurrent().unwrap();
    kv_prep(
        &ctx, &pass, &tx, &tv, &tw, &tvw, Some(&td), Some(&tvd), &kh_seq, &vh_seq,
        &kc_seq, &vc_seq, None, None, rows, nkv, n_conv, false, 3, 32, 22.627_417, 1e-6,
    )
    .unwrap();
    kv_prep_prefill(
        &ctx, &pass, &tx, &tv, &tw, &tvw, Some(&td), Some(&tvd), &kh_pre, &vh_pre,
        &kc_pre, &vc_pre, rows, nkv, n_conv, false, 3, 32, 22.627_417, 1e-6,
    )
    .unwrap();
    pass.memory_barrier().unwrap();
    short_conv_hist_update(&ctx, &pass, &tx, &kh_pre, rows, c, n_conv - 1).unwrap();
    short_conv_hist_update(&ctx, &pass, &tv, &vh_pre, rows, c, n_conv - 1).unwrap();
    pass.commit_wait().unwrap();

    assert_eq!(kc_pre.contents(), kc_seq.contents());
    assert_eq!(vc_pre.contents(), vc_seq.contents());
    assert_close(&kh_pre.to_f32().unwrap(), &kh_seq.to_f32().unwrap(), 1e-6, 1e-6);
    assert_close(&vh_pre.to_f32().unwrap(), &vh_seq.to_f32().unwrap(), 1e-6, 1e-6);
}

#[test]
fn mix_fused_2304_router_128_and_q8_moe_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(205);

    // 1. attn_res_mix_fused at exact Inkling 2 Mini H = 2304 with copy + fused norm.
    let (rows, h, n_states) = (2, 2304, 6);
    let states_cpu: Vec<Vec<f32>> =
        (0..n_states).map(|_| f32s(&mut rng, rows * h, -1.0, 1.0)).collect();
    let states_gpu: Vec<Tensor> =
        states_cpu.iter().map(|v| Tensor::from_f32(&ctx, v, &[rows, h]).unwrap()).collect();
    let query = f32s(&mut rng, h, -0.5, 0.5);
    let norm_w = f32s(&mut rng, h, 0.5, 1.5);
    let tq = Tensor::from_f32(&ctx, &query, &[1, h]).unwrap();
    let tnw = Tensor::from_f32(&ctx, &norm_w, &[h]).unwrap();
    let tout = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
    let tcopy = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
    let ptrs = Tensor::zeros(&ctx, &[32], DType::U32).unwrap();
    let mut raw_p = [0u64; 16];
    for (i, t) in states_gpu.iter().enumerate() {
        raw_p[i] = t.buffer().address() + t.binding().1 as u64;
    }
    ptrs.write_bytes(bytemuck::cast_slice(&raw_p)).unwrap();
    run(&ctx, |p| {
        attn_res_mix_fused(
            &ctx, p, &ptrs, &tq, &tout, Some(&tcopy), Some(&tnw), rows, h, n_states, 1, 1e-6, 1e-6,
        )
        .unwrap()
    });
    let mut exp_out = Vec::with_capacity(rows * h);
    for r in 0..rows {
        let s_slices: Vec<&[f32]> =
            states_cpu.iter().map(|s| &s[r * h..(r + 1) * h]).collect();
        let mixed = cref::attn_res_mix(&s_slices, &[&query], 1e-6);
        exp_out.extend(cref::rms_norm(&mixed[0], &norm_w, 1e-6));
    }
    assert_close(&tout.to_f32().unwrap(), &exp_out, 1e-4, 1e-4);
    assert_close(&tcopy.to_f32().unwrap(), &states_cpu[n_states - 1], 1e-6, 1e-6);

    // 2. Router at exact Inkling 2 Mini shape E = 128, K = 8 (including an exact tie).
    let (r_rows, n_exp, top_k) = (3, 128, 8);
    let mut logits = f32s(&mut rng, r_rows * n_exp, -3.0, 3.0);
    let bias = f32s(&mut rng, n_exp, -0.3, 0.3);
    // Force an exact tie on row 0 between experts 5 and 77.
    logits[77] = logits[5];
    let mut bias_tied = bias.clone();
    bias_tied[77] = bias_tied[5];
    let tl = Tensor::from_f32(&ctx, &logits, &[r_rows, n_exp]).unwrap();
    let tb = Tensor::from_f32(&ctx, &bias_tied, &[n_exp]).unwrap();
    let tids = Tensor::zeros(&ctx, &[r_rows * top_k], DType::U32).unwrap();
    let tw = Tensor::zeros(&ctx, &[r_rows * top_k], DType::F32).unwrap();
    run(&ctx, |p| router(&ctx, p, &tl, Some(&tb), &tids, &tw, r_rows, n_exp, top_k, 8.0).unwrap());
    let got_ids = tids.to_u32().unwrap();
    let got_w = tw.to_f32().unwrap();
    for r in 0..r_rows {
        let (e_ids, e_w) = cref::moe_route(&logits[r * n_exp..(r + 1) * n_exp], Some(&bias_tied), top_k, 8.0);
        assert_eq!(&got_ids[r * top_k..(r + 1) * top_k], &e_ids[..]);
        assert_close(&got_w[r * top_k..(r + 1) * top_k], &e_w, 1e-5, 1e-5);
    }

    // 3. Q8_0 MoE decode (`moe_gate_up_situ` + `moe_down_sum`) and grouped m4 (`cnt = 1, 2, 3, 4`).
    let (m_rows, m_exp, m_k, ff, mh) = (9, 6, 4, 64, 64);
    let rows_k = m_rows * m_k;
    let x = f32s(&mut rng, m_rows * mh, -1.0, 1.0);
    let quant_q8 = |rng: &mut StdRng, n_vals: usize| -> Vec<u8> {
        let blocks = n_vals / 32;
        let mut raw = vec![0u8; blocks * 34];
        for b in 0..blocks {
            let d = half::f16::from_f32(rng.gen_range(0.004f32..0.015)).to_bits();
            raw[b * 34] = (d & 0xff) as u8;
            raw[b * 34 + 1] = (d >> 8) as u8;
            for q in 0..32 {
                raw[b * 34 + 2 + q] = rng.gen_range(-100i8..100i8) as u8;
            }
        }
        raw
    };
    let raw_gu = quant_q8(&mut rng, m_exp * 2 * ff * mh);
    let raw_dn = quant_q8(&mut rng, m_exp * mh * ff);
    let q_gu = QWeight::from_bytes(&ctx, &raw_gu, GgmlType::Q8_0, mh, m_exp * 2 * ff).unwrap();
    let q_dn = QWeight::from_bytes(&ctx, &raw_dn, GgmlType::Q8_0, ff, m_exp * mh).unwrap();
    // Construct assignments that hit cnt=1, 2, 3, and 4 tiles!
    let mut ids = Vec::with_capacity(rows_k);
    let mut weights = Vec::with_capacity(rows_k);
    for r in 0..m_rows {
        let row_e = match r {
            0 => [0u32, 1, 2, 3],
            1 => [0, 1, 2, 4],
            2 => [0, 1, 5, 3],
            3 => [0, 2, 4, 5],
            _ => [(r % m_exp) as u32, ((r + 1) % m_exp) as u32, ((r + 2) % m_exp) as u32, ((r + 3) % m_exp) as u32],
        };
        ids.extend_from_slice(&row_e);
        for _ in 0..m_k {
            weights.push(rng.gen_range(0.1f32..1.5));
        }
    }
    let tx = Tensor::from_f32(&ctx, &x, &[m_rows, mh]).unwrap();
    let tids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[rows_k], DType::U32).unwrap();
    let tw = Tensor::from_f32(&ctx, &weights, &[rows_k]).unwrap();

    let ref_gu = Tensor::zeros(&ctx, &[rows_k, 2 * ff], DType::F32).unwrap();
    let ref_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
    let ref_down = Tensor::zeros(&ctx, &[rows_k, mh], DType::F32).unwrap();
    let ref_out = Tensor::zeros(&ctx, &[m_rows, mh], DType::F32).unwrap();
    let f_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
    let f_out = Tensor::zeros(&ctx, &[m_rows, mh], DType::F32).unwrap();
    let sorted_rk = Tensor::zeros(&ctx, &[rows_k], DType::U32).unwrap();
    let tiles = Tensor::zeros(&ctx, &[rows_k * 4], DType::U32).unwrap();
    let n_tiles = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
    let g_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
    let g_down = Tensor::zeros(&ctx, &[rows_k, mh], DType::F32).unwrap();
    let g_out = Tensor::zeros(&ctx, &[m_rows, mh], DType::F32).unwrap();
    let max_tiles = rows_k.min((rows_k + 3 * m_exp) / 4);

    let pass = ctx.begin_concurrent().unwrap();
    moe_proj(&ctx, &pass, &tx, &q_gu, &tids, &ref_gu, rows_k, 2 * ff, m_k).unwrap();
    pass.memory_barrier().unwrap();
    situ(&ctx, &pass, &ref_gu, &tids, None, &ref_situ, rows_k, ff, 4.0, 25.0).unwrap();
    pass.memory_barrier().unwrap();
    moe_proj(&ctx, &pass, &ref_situ, &q_dn, &tids, &ref_down, rows_k, mh, 0).unwrap();
    pass.memory_barrier().unwrap();
    moe_sum(&ctx, &pass, &ref_down, &tids, &tw, None, &ref_out, m_rows, m_k, mh).unwrap();

    moe_gate_up_situ(&ctx, &pass, &tx, &q_gu, &tids, None, &f_situ, rows_k, ff, m_k, 4.0, 25.0).unwrap();
    pass.memory_barrier().unwrap();
    moe_down_sum(&ctx, &pass, &f_situ, &q_dn, &tids, &tw, None, &f_out, m_rows, m_k, mh).unwrap();

    moe_bucket(&ctx, &pass, &tids, &sorted_rk, &tiles, &n_tiles, rows_k, m_exp).unwrap();
    pass.memory_barrier().unwrap();
    moe_gu_m4(&ctx, &pass, &tx, &q_gu, &sorted_rk, &tiles, &n_tiles, None, &g_situ, max_tiles, ff, m_k, 4.0, 25.0).unwrap();
    pass.memory_barrier().unwrap();
    moe_down_m4(&ctx, &pass, &g_situ, &q_dn, &sorted_rk, &tiles, &n_tiles, &g_down, max_tiles, mh).unwrap();
    pass.memory_barrier().unwrap();
    moe_sum(&ctx, &pass, &g_down, &tids, &tw, None, &g_out, m_rows, m_k, mh).unwrap();
    pass.commit_wait().unwrap();

    let expected = ref_out.to_f32().unwrap();
    assert_close(&f_out.to_f32().unwrap(), &expected, 1e-3, 1e-3);
    assert_close(&g_out.to_f32().unwrap(), &expected, 1e-3, 1e-3);
}

#[test]
fn attention_mu_matches_cpu_swa_and_global() {
    let ctx = ctx();
    if !ctx.matrix_units() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(301);
    let (nh, nkv, hd, ext, window, slots) = (6, 2, 128, 64, 160, 256);
    for (is_swa, n_prefill, n_step) in [(true, 170, 19), (false, 140, 21)] {
        let total = n_prefill + n_step;
        let cap = if is_swa { slots } else { total.max(256) };
        let k_all = f32s(&mut rng, total * nkv * hd, -1.0, 1.0);
        let v_all = f32s(&mut rng, total * nkv * hd, -1.0, 1.0);
        let q_step = f32s(&mut rng, n_step * nh * hd, -2.0, 2.0);
        let q_clip = f32s(&mut rng, nh * hd, 0.5, 1.5);
        let gate = f32s(&mut rng, n_step * nh * hd, -2.0, 2.0);
        let rel_tab = f32s(&mut rng, n_step * nh * ext, -0.5, 0.5);
        let sinks = f32s(&mut rng, n_step * nh, -0.5, 0.5);

        let tk_cache = Tensor::zeros(&ctx, &[cap, nkv * hd], DType::F16).unwrap();
        let tv_cache = Tensor::zeros(&ctx, &[cap, nkv * hd], DType::F16).unwrap();
        let tk_in = Tensor::from_f32(&ctx, &k_all, &[total, nkv * hd]).unwrap();
        let tv_in = Tensor::from_f32(&ctx, &v_all, &[total, nkv * hd]).unwrap();
        let tq = Tensor::from_f32(&ctx, &q_step, &[n_step, nh * hd]).unwrap();
        let tqc = Tensor::from_f32(&ctx, &q_clip, &[nh, hd]).unwrap();
        let tgate = Tensor::from_f32(&ctx, &gate, &[n_step, nh * hd]).unwrap();
        let trel = Tensor::from_f32(&ctx, &rel_tab, &[n_step, nh * ext]).unwrap();
        let tsink = Tensor::from_f32(&ctx, &sinks, &[n_step, nh]).unwrap();
        let tout = Tensor::zeros(&ctx, &[n_step, nh * hd], DType::F32).unwrap();

        let (qk_clip, lf, la) = (
            22.627_417f32,
            if is_swa { 0 } else { 128 },
            if is_swa { 0.0 } else { 0.1 },
        );
        let pass = ctx.begin_concurrent().unwrap();
        kv_store(&ctx, &pass, &tk_in, &tv_in, &tk_cache, &tv_cache, total, nkv * hd, 0, cap, is_swa)
            .unwrap();
        pass.memory_barrier().unwrap();
        attention_mu(
            &ctx,
            &pass,
            &tq,
            &tk_cache,
            &tv_cache,
            &trel,
            &tsink,
            &tout,
            Some(&tgate),
            Some(&tqc),
            n_step,
            nh,
            nkv,
            n_prefill,
            cap,
            is_swa,
            window,
            ext,
            qk_clip,
            lf,
            la,
        )
        .unwrap();
        pass.commit_wait().unwrap();

        let k16: Vec<f32> = k_all.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let v16: Vec<f32> = v_all.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let mut exp = vec![0.0f32; n_step * nh * hd];
        let pos_k: Vec<usize> = (0..total).collect();
        for r in 0..n_step {
            let pos_q = n_prefill + r;
            let tau = if lf > 0 { cref::log_scale_tau(pos_q, lf, la) } else { 1.0 };
            for h in 0..nh {
                let kh = h / (nh / nkv);
                let k_head: Vec<f32> = (0..total)
                    .flat_map(|t| k16[(t * nkv + kh) * hd..(t * nkv + kh + 1) * hd].iter().copied())
                    .collect();
                let v_head: Vec<f32> = (0..total)
                    .flat_map(|t| v16[(t * nkv + kh) * hd..(t * nkv + kh + 1) * hd].iter().copied())
                    .collect();
                let q_raw = &q_step[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                let qv: Vec<f32> = cref::l2_clip(q_raw, Some(&q_clip[h * hd..(h + 1) * hd]), qk_clip)
                    .into_iter()
                    .map(|v| v * tau)
                    .collect();
                let rt = &rel_tab[(r * nh + h) * ext..(r * nh + h + 1) * ext];
                let mut o = cref::rel_sink_attention(
                    &qv,
                    &k_head,
                    &v_head,
                    rt,
                    sinks[r * nh + h],
                    pos_q,
                    &pos_k,
                    if is_swa { Some(window) } else { None },
                );
                let g_slice = &gate[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                for (oi, &gi) in o.iter_mut().zip(g_slice) {
                    *oi /= 1.0 + (-gi).exp();
                }
                exp[(r * nh + h) * hd..(r * nh + h + 1) * hd].copy_from_slice(&o);
            }
        }
        // f16 Q/P operands in matrix-unit FlashAttention match f32 CPU reference within ~3e-3.
        assert_close(&tout.to_f32().unwrap(), &exp, 5e-3, 5e-3);
    }
}

#[test]
fn moe_grouped_mu_matches_cpu_q8_nvfp4_and_f32() {
    let ctx = ctx();
    if !ctx.matrix_units() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(302);
    // Test multiple experts: some with <= 4 rows (register-dot path) and some with > 4 rows (matrix-unit path),
    // across both r64 and r128 tile modes and Q8_0 / NVFP4 / F32 weights.
    let (rows, n_exp, k_exp, ff, h) = (36, 8, 4, 128, 128);
    let rows_k = rows * k_exp;
    let x = f32s(&mut rng, rows * h, -0.5, 0.5);
    let gu_sc = f32s(&mut rng, n_exp, 0.7, 1.3);
    let dn_sc = f32s(&mut rng, n_exp, 0.7, 1.3);
    let mut ids = Vec::with_capacity(rows_k);
    let mut weights = Vec::with_capacity(rows_k);
    for r in 0..rows {
        // Skew routing so experts 0..2 get many rows (>20, matrix-unit path) and experts 5..7 get 1..3 rows (dot path)!
        let row_e = if r < 3 {
            [0u32, 1, 2, (5 + r) as u32]
        } else {
            [0u32, 1, 2, (3 + (r % 2)) as u32]
        };
        ids.extend_from_slice(&row_e);
        for _ in 0..k_exp {
            weights.push(rng.gen_range(0.1f32..0.6));
        }
    }

    let tx = Tensor::from_f32(&ctx, &x, &[rows, h]).unwrap();
    let tids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[rows_k], DType::U32).unwrap();
    let tw = Tensor::from_f32(&ctx, &weights, &[rows_k]).unwrap();
    let tgu_sc = Tensor::from_f32(&ctx, &gu_sc, &[n_exp]).unwrap();
    let tdn_sc = Tensor::from_f32(&ctx, &dn_sc, &[n_exp]).unwrap();

    for ty in [GgmlType::Q8_0, GgmlType::Nvfp4, GgmlType::F32] {
        let make_raw = |rng: &mut StdRng, n_vals: usize| -> Vec<u8> {
            match ty {
                GgmlType::F32 => {
                    let v = f32s(rng, n_vals, -0.25, 0.25);
                    bytemuck::cast_slice(&v).to_vec()
                }
                GgmlType::Q8_0 => {
                    let blocks = n_vals / 32;
                    let mut raw = vec![0u8; blocks * 34];
                    for b in 0..blocks {
                        let d = half::f16::from_f32(rng.gen_range(0.002f32..0.006)).to_bits();
                        raw[b * 34] = (d & 0xff) as u8;
                        raw[b * 34 + 1] = (d >> 8) as u8;
                        for q in 0..32 {
                            raw[b * 34 + 2 + q] = rng.gen_range(-80i8..80i8) as u8;
                        }
                    }
                    raw
                }
                GgmlType::Nvfp4 => {
                    let blocks = n_vals / 64;
                    let mut raw: Vec<u8> = (0..blocks * 36).map(|_| rng.r#gen()).collect();
                    for blk in raw.chunks_exact_mut(36) {
                        for b in &mut blk[..4] {
                            *b = rng.gen_range(8..24);
                        }
                    }
                    raw
                }
                _ => unreachable!(),
            }
        };
        let raw_gu = make_raw(&mut rng, n_exp * 2 * ff * h);
        let raw_dn = make_raw(&mut rng, n_exp * h * ff);
        let q_gu = QWeight::from_bytes(&ctx, &raw_gu, ty, h, n_exp * 2 * ff).unwrap();
        let q_dn = QWeight::from_bytes(&ctx, &raw_dn, ty, ff, n_exp * h).unwrap();

        // CPU f32 reference from dequantized weights.
        let mut gu_f32 = vec![0.0f32; n_exp * 2 * ff * h];
        let mut dn_f32 = vec![0.0f32; n_exp * h * ff];
        dequantize(ty, &raw_gu, &mut gu_f32).unwrap();
        dequantize(ty, &raw_dn, &mut dn_f32).unwrap();
        let mut expected = vec![0.0f32; rows * h];
        for r in 0..rows {
            let xr = &x[r * h..(r + 1) * h];
            for k in 0..k_exp {
                let rk = r * k_exp + k;
                let e = ids[rk] as usize;
                let w_eff = weights[rk] * dn_sc[e];
                let mut mid = vec![0.0f32; ff];
                for j in 0..ff {
                    let g_row = &gu_f32[(e * 2 * ff + j) * h..(e * 2 * ff + j + 1) * h];
                    let u_row = &gu_f32[(e * 2 * ff + ff + j) * h..(e * 2 * ff + ff + j + 1) * h];
                    let g: f32 = g_row.iter().zip(xr).map(|(&a, &b)| a * b).sum::<f32>() * gu_sc[e];
                    let u: f32 = u_row.iter().zip(xr).map(|(&a, &b)| a * b).sum::<f32>() * gu_sc[e];
                    mid[j] = cref::situ(g, u, 4.0, 25.0);
                }
                for o in 0..h {
                    let d_row = &dn_f32[(e * h + o) * ff..(e * h + o + 1) * ff];
                    let v: f32 = d_row.iter().zip(&mid).map(|(&a, &b)| a * b).sum();
                    expected[r * h + o] += w_eff * v;
                }
            }
        }

        for r128 in [false, true] {
            let max_tiles = rows_k.div_ceil(64) + n_exp + 1;
            let xb = Tensor::zeros(&ctx, &[rows, h], DType::BF16).unwrap();
            let pairs = Tensor::zeros(&ctx, &[rows_k], DType::U32).unwrap();
            let tiles = Tensor::zeros(&ctx, &[3 * max_tiles], DType::U32).unwrap();
            let tile_count = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
            let mid_bf16 = Tensor::zeros(&ctx, &[rows_k, ff], DType::BF16).unwrap();
            let down_out = Tensor::zeros(&ctx, &[rows_k, h], DType::F32).unwrap();
            let out = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();

            let pass = ctx.begin_concurrent().unwrap();
            moe_grouped_mu(
                &ctx,
                &pass,
                &tx,
                &xb,
                &q_gu,
                Some(&tgu_sc),
                &q_dn,
                Some(&tdn_sc),
                &tids,
                &tw,
                &pairs,
                &tiles,
                &tile_count,
                &mid_bf16,
                &down_out,
                &out,
                rows,
                h,
                ff,
                n_exp,
                k_exp,
                4.0,
                25.0,
                Some(r128),
            )
            .unwrap();
            pass.commit_wait().unwrap();

            // Matrix-unit path rounds operands to BF16 (7-bit mantissa -> ~1e-2 relative).
            assert_close(&out.to_f32().unwrap(), &expected, 2e-2, 2e-2);
        }
        if ty == GgmlType::Nvfp4 {
            // Also verify fused NVFP4 decode (`moe_gate_up_situ` + `moe_down_sum`)
            // and grouped m4 (`moe_gu_m4` + `moe_down_m4`) against the same CPU ref.
            let f_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
            let f_out = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
            let sorted_rk = Tensor::zeros(&ctx, &[rows_k], DType::U32).unwrap();
            let tiles = Tensor::zeros(&ctx, &[rows_k * 4], DType::U32).unwrap();
            let n_tiles = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
            let g_situ = Tensor::zeros(&ctx, &[rows_k, ff], DType::F32).unwrap();
            let g_down = Tensor::zeros(&ctx, &[rows_k, h], DType::F32).unwrap();
            let g_out = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();
            let max_tiles = rows_k.min((rows_k + 3 * n_exp) / 4);
            let pass = ctx.begin_concurrent().unwrap();
            moe_gate_up_situ(
                &ctx, &pass, &tx, &q_gu, &tids, Some(&tgu_sc), &f_situ, rows_k, ff, k_exp,
                4.0, 25.0,
            )
            .unwrap();
            pass.memory_barrier().unwrap();
            moe_down_sum(
                &ctx, &pass, &f_situ, &q_dn, &tids, &tw, Some(&tdn_sc), &f_out, rows,
                k_exp, h,
            )
            .unwrap();
            moe_bucket(&ctx, &pass, &tids, &sorted_rk, &tiles, &n_tiles, rows_k, n_exp)
                .unwrap();
            pass.memory_barrier().unwrap();
            moe_gu_m4(
                &ctx, &pass, &tx, &q_gu, &sorted_rk, &tiles, &n_tiles, Some(&tgu_sc),
                &g_situ, max_tiles, ff, k_exp, 4.0, 25.0,
            )
            .unwrap();
            pass.memory_barrier().unwrap();
            moe_down_m4(
                &ctx, &pass, &g_situ, &q_dn, &sorted_rk, &tiles, &n_tiles, &g_down,
                max_tiles, h,
            )
            .unwrap();
            pass.memory_barrier().unwrap();
            moe_sum(
                &ctx, &pass, &g_down, &tids, &tw, Some(&tdn_sc), &g_out, rows, k_exp, h,
            )
            .unwrap();
            pass.commit_wait().unwrap();
            assert_close(&f_out.to_f32().unwrap(), &expected, 1e-3, 1e-3);
            assert_close(&g_out.to_f32().unwrap(), &expected, 1e-3, 1e-3);
        }
    }
}

#[test]
fn ctrl_bf16_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(303);
    let (rows, h, rank) = (4, 128, 32);
    let cur = f32s(&mut rng, rows * h, -1.0, 1.0);
    let msg = f32s(&mut rng, rows * h, -1.0, 1.0);
    let to_bf = |v: &[f32]| -> (Vec<u8>, Vec<f32>) {
        let mut bytes = Vec::with_capacity(v.len() * 2);
        let mut rounded = Vec::with_capacity(v.len());
        for &x in v {
            let b = half::bf16::from_f32(x);
            bytes.extend_from_slice(&b.to_bits().to_le_bytes());
            rounded.push(b.to_f32());
        }
        (bytes, rounded)
    };
    let (dw_b, dw_f) = to_bf(&f32s(&mut rng, rank * 2 * h, -0.3, 0.3));
    let (gw_b, gw_f) = to_bf(&f32s(&mut rng, h * rank, -0.3, 0.3));
    let (fw_b, fw_f) = to_bf(&f32s(&mut rng, h * rank, -0.3, 0.3));

    let tcur = Tensor::from_f32(&ctx, &cur, &[rows, h]).unwrap();
    let tmsg = Tensor::from_f32(&ctx, &msg, &[rows, h]).unwrap();
    let qd = QWeight::from_bytes(&ctx, &dw_b, GgmlType::Bf16, 2 * h, rank).unwrap();
    let qg = QWeight::from_bytes(&ctx, &gw_b, GgmlType::Bf16, rank, h).unwrap();
    let qf = QWeight::from_bytes(&ctx, &fw_b, GgmlType::Bf16, rank, h).unwrap();
    let tmid = Tensor::zeros(&ctx, &[rows, rank], DType::F32).unwrap();
    let tout = Tensor::zeros(&ctx, &[rows, h], DType::F32).unwrap();

    let pass = ctx.begin_concurrent().unwrap();
    ctrl_down_f32(&ctx, &pass, &tcur, &tmsg, &qd, &tmid, rows, h, rank, 1e-6).unwrap();
    pass.memory_barrier().unwrap();
    ctrl_blend_f32(&ctx, &pass, &tcur, &tmsg, &tmid, &qg, &qf, &tout, rows, h, rank)
        .unwrap();
    pass.commit_wait().unwrap();

    let mut expected = Vec::with_capacity(rows * h);
    for r in 0..rows {
        let cr = &cur[r * h..(r + 1) * h];
        let mr = &msg[r * h..(r + 1) * h];
        let mut cn = cref::rms_norm_no_weight(cr, 1e-6);
        cn.extend(cref::rms_norm_no_weight(mr, 1e-6));
        let mut mid = vec![0.0f32; rank];
        for o in 0..rank {
            mid[o] = dw_f[o * 2 * h..(o + 1) * 2 * h]
                .iter()
                .zip(&cn)
                .map(|(&a, &b)| a * b)
                .sum();
        }
        for i in 0..h {
            let g: f32 = gw_f[i * rank..(i + 1) * rank]
                .iter()
                .zip(&mid)
                .map(|(&a, &b)| a * b)
                .sum();
            let f: f32 = fw_f[i * rank..(i + 1) * rank]
                .iter()
                .zip(&mid)
                .map(|(&a, &b)| a * b)
                .sum();
            expected.push((1.0 - f.tanh()) * cr[i] + g.tanh() * mr[i]);
        }
    }
    assert_close(&tout.to_f32().unwrap(), &expected, 1e-4, 1e-4);
}

#[test]
fn mix_many_states_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(4212);
    let (rows, h, n_states, n_q) = (3usize, 2304usize, 12usize, 3usize);
    let states: Vec<Vec<f32>> =
        (0..n_states).map(|_| f32s(&mut rng, rows * h, -1.0, 1.0)).collect();
    let queries = f32s(&mut rng, n_q * h, -0.5, 0.5);
    let tstates: Vec<Tensor> =
        states.iter().map(|s| Tensor::from_f32(&ctx, s, &[rows, h]).unwrap()).collect();
    let mut ptrs = vec![0u64; 16];
    for (i, t) in tstates.iter().enumerate() {
        ptrs[i] = t.buffer().address() + t.binding().1 as u64;
    }
    let ptrs_u32: Vec<u32> = ptrs
        .iter()
        .flat_map(|&p| [p as u32, (p >> 32) as u32])
        .collect();
    let tptrs =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ptrs_u32), &[32], DType::U32)
            .unwrap();
    let tq = Tensor::from_f32(&ctx, &queries, &[n_q, h]).unwrap();
    let tout = Tensor::zeros(&ctx, &[n_q, rows, h], DType::F32).unwrap();

    let pass = ctx.begin_concurrent().unwrap();
    attn_res_mix(&ctx, &pass, &tptrs, &tq, &tout, rows, h, n_states, n_q, 1e-6).unwrap();
    pass.commit_wait().unwrap();

    let got = tout.to_f32().unwrap();
    let mut expected = vec![0.0f32; n_q * rows * h];
    let q_refs: Vec<&[f32]> = (0..n_q).map(|q| &queries[q * h..(q + 1) * h]).collect();
    for r in 0..rows {
        let s_refs: Vec<&[f32]> =
            states.iter().map(|s| &s[r * h..(r + 1) * h]).collect();
        let mixed = cref::attn_res_mix(&s_refs, &q_refs, 1e-6);
        for (q, row) in mixed.iter().enumerate() {
            expected[(q * rows + r) * h..(q * rows + r + 1) * h].copy_from_slice(row);
        }
    }
    assert_close(&got, &expected, 1e-4, 1e-4);
}
