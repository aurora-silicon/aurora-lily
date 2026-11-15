//! The DeepSeek V4 kernels against the CPU reference (`deepseek4::reference`).

use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::deepseek4::config::Config;
use crate::deepseek4::reference as r;
use crate::gguf::GgmlType;
use crate::kernels::glm53::{self as g, GemvLayout};
use crate::refops::{Mat, round_f16};

fn ctx() -> MetalContext {
    MetalContext::new().expect("metal context")
}

fn f32s(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn run(ctx: &MetalContext, f: impl FnOnce(&ComputePass<'_>)) {
    let pass = ctx.begin().expect("pass");
    f(&pass);
    pass.commit_wait().expect("commit");
}

fn tensor(ctx: &MetalContext, v: &[f32]) -> Tensor {
    Tensor::from_f32(ctx, v, &[v.len()]).expect("tensor")
}

fn zeros(ctx: &MetalContext, n: usize, dt: DType) -> Tensor {
    Tensor::zeros(ctx, &[n], dt).expect("zeros")
}

fn u32s(ctx: &MetalContext, v: &[u32]) -> Tensor {
    Tensor::from_bytes(ctx, bytemuck::cast_slice(v), &[v.len()], DType::U32).unwrap()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes()).collect()
}

fn f16_tensor(ctx: &MetalContext, v: &[f32]) -> Tensor {
    Tensor::from_bytes(ctx, &f16_bytes(v), &[v.len()], DType::F16).unwrap()
}

fn f16_weight(ctx: &MetalContext, v: &[f32], cols: usize) -> (QWeight, Mat) {
    let rounded: Vec<f32> = v.iter().map(|&x| round_f16(x)).collect();
    let w =
        QWeight::from_bytes(ctx, &f16_bytes(v), GgmlType::F16, cols, v.len() / cols)
            .unwrap();
    (w, Mat { rows: v.len() / cols, cols, w: rounded })
}

fn bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(a.to_bits() == b.to_bits(), "{what}[{i}]: GPU {a:e}, CPU {b:e}");
    }
}

/// The F16 GEMV (DeepSeek's HC controls, compressors, indexer, router) for
/// every row count, plain and as sets.
#[test]
fn gemv_f16_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(401);
    for (cols, rows) in [(4096usize, 256usize), (16384, 24), (1024, 520), (4096, 64)] {
        let (w, cpu) = f16_weight(&ctx, &f32s(&mut rng, rows * cols, -0.1, 0.1), cols);
        for m in 1..=g::GEMV_MAX_ROWS_WIDE {
            let x = f32s(&mut rng, m * cols, -1.0, 1.0);
            let tx = tensor(&ctx, &x);
            let out = zeros(&ctx, m * rows, DType::F32);
            run(&ctx, |p| g::gemv(&ctx, p, &w, &tx, &out, m).unwrap());
            let want: Vec<f32> =
                (0..m).flat_map(|i| cpu.mv(&x[i * cols..(i + 1) * cols])).collect();
            assert_close(&out.to_f32().unwrap(), &want, 1e-4, 1e-4);
        }
    }
}

/// The prefill GEMM over F16 weights (operands rounded to bfloat).
#[test]
fn gemm_f16_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(402);
    let (cols, rows, m) = (1024usize, 256usize, 70usize);
    let (w, cpu) = f16_weight(&ctx, &f32s(&mut rng, rows * cols, -0.1, 0.1), cols);
    let x = f32s(&mut rng, m * cols, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let out = zeros(&ctx, m * rows, DType::F32);
    run(&ctx, |p| g::gemm(&ctx, p, &w, &tx, &out, m).unwrap());
    let want: Vec<f32> =
        (0..m).flat_map(|i| cpu.mv(&x[i * cols..(i + 1) * cols])).collect();
    // bfloat operands: ~2^-8 relative per product.
    assert_close(&out.to_f32().unwrap(), &want, 2e-2, 2e-2);
}

/// The grouped output projection is the GEMV with one input vector per
/// group of 1 024 rows.
#[test]
fn grouped_output_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(403);
    let (groups, rank, cols) = (8usize, 1024usize, 4096usize);
    let raw: Vec<u8> = {
        let mut b: Vec<u8> =
            (0..groups * rank * cols / 32 * 34).map(|_| rng.r#gen()).collect();
        for blk in b.chunks_exact_mut(34) {
            blk[..2].copy_from_slice(
                &f16::from_f32(rng.gen_range(0.001..0.01)).to_bits().to_le_bytes(),
            );
        }
        b
    };
    let w =
        QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, groups * rank).unwrap();
    let mut wf = vec![0f32; groups * rank * cols];
    crate::gguf::dequantize(GgmlType::Q8_0, &raw, &mut wf).unwrap();
    let cpu = Mat { rows: groups * rank, cols, w: wf };
    for m in [1usize, 4, 6, 8] {
        let x = f32s(&mut rng, m * groups * cols, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let out = zeros(&ctx, m * groups * rank, DType::F32);
        let l = GemvLayout {
            x_group: rank,
            x_row_stride: groups * cols,
            y_row_stride: groups * rank,
        };
        run(&ctx, |p| g::gemv_with(&ctx, p, &w, &tx, &out, m, l).unwrap());
        let want: Vec<f32> = (0..m)
            .flat_map(|i| {
                cpu.mv_grouped(&x[i * groups * cols..(i + 1) * groups * cols], rank)
            })
            .collect();
        assert_close(&out.to_f32().unwrap(), &want, 1e-3, 1e-4);
    }
}

#[test]
fn embed_f16_reads_the_table() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(404);
    let (vocab, hidden) = (50usize, 4096usize);
    let table = f32s(&mut rng, vocab * hidden, -1.0, 1.0);
    let (w, cpu) = f16_weight(&ctx, &table, hidden);
    let toks = [7u32, 0, 49];
    let out = zeros(&ctx, toks.len() * 4 * hidden, DType::F32);
    run(&ctx, |p| embed(&ctx, p, &w, &u32s(&ctx, &toks), &out, toks.len()).unwrap());
    let want: Vec<f32> =
        toks.iter().flat_map(|&t| cpu.row(t as usize).repeat(4)).collect();
    bits_eq(&out.to_f32().unwrap(), &want, "embedding");
}

fn tables(c: &Config, lo: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    (rope_table(&c.rope(0), lo, n), rope_table(&c.rope(2), lo, n))
}

/// Query norm + RoPE, the inverse rotation, and the KV store, at positions
/// past the YaRN context too.
#[test]
fn head_ops_and_kv_store_match_cpu() {
    let ctx = ctx();
    let c = Config::deepseek4_flash();
    let mut rng = StdRng::seed_from_u64(405);
    for pos0 in [0usize, 5, 70_000] {
        let m = 3;
        let (plain, yarn) = tables(&c, pos0, m);
        for (il, table) in [(0usize, &plain), (2, &yarn)] {
            let rope = c.rope(il);
            let tt = tensor(&ctx, table);
            let heads = 8;
            let x = f32s(&mut rng, m * heads * 512, -2.0, 2.0);
            let tx = tensor(&ctx, &x);
            run(&ctx, |p| {
                head_norm_rope(
                    &ctx,
                    p,
                    &tx,
                    &tt,
                    0,
                    m,
                    heads,
                    512,
                    64,
                    1e-6,
                    HeadOp::NormRope,
                )
                .unwrap()
            });
            let mut want = x.clone();
            for i in 0..m {
                let row = &mut want[i * heads * 512..(i + 1) * heads * 512];
                r::head_rmsnorm(row, 512, 1e-6);
                r::rope_tail(row, 512, &rope, pos0 + i, false);
            }
            assert_close(&tx.to_f32().unwrap(), &want, 1e-5, 1e-5);
            run(&ctx, |p| {
                head_norm_rope(
                    &ctx,
                    p,
                    &tx,
                    &tt,
                    0,
                    m,
                    heads,
                    512,
                    64,
                    0.0,
                    HeadOp::InverseRope,
                )
                .unwrap()
            });
            for i in 0..m {
                let row = &mut want[i * heads * 512..(i + 1) * heads * 512];
                r::rope_tail(row, 512, &rope, pos0 + i, true);
            }
            assert_close(&tx.to_f32().unwrap(), &want, 1e-5, 1e-5);

            // KV: magnitudes over a few decades, as rmsnorm'd rows times
            // their norm weight are.
            let kv: Vec<f32> = (0..m * 512)
                .map(|_| {
                    rng.gen_range(-1.0f32..1.0) * 10f32.powf(rng.gen_range(-2.0..1.5))
                })
                .collect();
            let tkv = tensor(&ctx, &kv);
            let run_t = zeros(&ctx, m * 512, DType::F16);
            run(&ctx, |p| kv_store(&ctx, p, &tkv, &run_t, &tt, 0, m).unwrap());
            let mut want = kv.clone();
            for i in 0..m {
                let row = &mut want[i * 512..(i + 1) * 512];
                r::rope_tail(row, 512, &rope, pos0 + i, false);
                r::fp8_kv_qat(row, 64);
            }
            bits_eq(&tkv.to_f32().unwrap(), &want, "kv after FP8");
            let w16: Vec<f32> = want.iter().map(|&v| round_f16(v)).collect();
            bits_eq(&run_t.to_f32().unwrap(), &w16, "kv f16 copy");
        }
    }
}

/// A compressor fed in steps of mixed sizes (decode rows, verify rows, a
/// prompt run longer than the ring) emits the reference's rows, for both
/// ratios and both caches.
#[test]
fn compressor_matches_cpu_across_steps() {
    let ctx = ctx();
    let c = Config::deepseek4_flash();
    let mut rng = StdRng::seed_from_u64(406);
    for (ratio, dim, indexer) in
        [(4usize, 512usize, false), (4, 128, true), (128, 512, false)]
    {
        let lanes = if ratio == 4 { 2 } else { 1 };
        let w = lanes * dim;
        let total = 3 * ratio + 300;
        let kv = f32s(&mut rng, total * w, -1.0, 1.0);
        let sc = f32s(&mut rng, total * w, -3.0, 3.0);
        let ape = f32s(&mut rng, ratio * w, -1.0, 1.0);
        let ape16: Vec<f32> = ape.iter().map(|&v| round_f16(v)).collect();
        let norm = f32s(&mut rng, dim, 0.5, 1.5);
        let rope = c.rope(2);
        // CPU.
        let mut cpu = r::Compressor::new(dim, ratio);
        let mut rows = Vec::new();
        for p in 0..total {
            let mut s: Vec<f32> = sc[p * w..(p + 1) * w].to_vec();
            for (j, v) in s.iter_mut().enumerate() {
                *v += ape16[(p % ratio) * w + j];
            }
            if let Some(pooled) = cpu.push(&kv[p * w..(p + 1) * w], &s, p) {
                rows.push(r::finish_compressed(
                    &pooled,
                    &norm,
                    1e-6,
                    &rope,
                    p + 1 - ratio,
                    indexer,
                ));
            }
        }
        // GPU, in steps.
        let tape = f16_tensor(&ctx, &ape);
        let tnorm = tensor(&ctx, &norm);
        let ring_kv = zeros(&ctx, lanes * ratio * w, DType::F32);
        let ring_sc = zeros(&ctx, lanes * ratio * w, DType::F32);
        let cache = zeros(&ctx, (total / ratio + 1) * dim, DType::F16);
        let cg = CompressorGpu {
            ape: &tape,
            norm: &tnorm,
            ring_kv: &ring_kv,
            ring_sc: &ring_sc,
            dim,
            ratio,
            indexer,
        };
        let mut pos = 0;
        for step in [1usize, 2, 1, 3, 4, 1, 200, 1, 1, 4].iter().cycle() {
            if pos >= total {
                break;
            }
            let m = (*step).min(total - pos);
            let lo = (pos + 1).saturating_sub(ratio);
            let table = tensor(&ctx, &rope_table(&rope, lo, pos + m - lo));
            let tkv = tensor(&ctx, &kv[pos * w..(pos + m) * w]);
            let tsc = tensor(&ctx, &sc[pos * w..(pos + m) * w]);
            run(&ctx, |p| {
                compress(&ctx, p, &cg, &tkv, &tsc, &cache, &table, lo, pos, m, 1e-6)
                    .unwrap()
            });
            pos += m;
        }
        let got = cache.to_f32().unwrap();
        for (k, want) in rows.iter().enumerate() {
            let g = &got[k * dim..(k + 1) * dim];
            // The pool's exp and the norm differ by ulps; the FP8/FP4 grid
            // can then put a value one step away, rarely.
            let off = g
                .iter()
                .zip(want)
                .filter(|(a, b)| (*a - *b).abs() > 1e-3 * (1.0 + b.abs()))
                .count();
            assert!(
                off <= dim / 64,
                "ratio {ratio} dim {dim} row {k}: {off} values differ"
            );
        }
    }
}

/// Indexer queries, scores and the top-k against the reference's
/// selection.
#[test]
fn indexer_matches_cpu() {
    let ctx = ctx();
    let c = Config::deepseek4_flash();
    let mut rng = StdRng::seed_from_u64(407);
    let heads = 64;
    let (pos0, m) = (2400usize, 3usize);
    let rope = c.rope(2);
    let table = tensor(&ctx, &rope_table(&rope, pos0, m));
    let q = f32s(&mut rng, m * heads * 128, -1.0, 1.0);
    let tq = tensor(&ctx, &q);
    run(&ctx, |p| indexer_q(&ctx, p, &tq, &table, 0, m, heads).unwrap());
    let mut qw = q.clone();
    for i in 0..m {
        let row = &mut qw[i * heads * 128..(i + 1) * heads * 128];
        r::rope_tail(row, 128, &rope, pos0 + i, false);
        for h in row.chunks_exact_mut(128) {
            r::indexer_qat(h);
        }
    }
    let gq = tq.to_f32().unwrap();
    let close = gq
        .iter()
        .zip(&qw)
        .filter(|(a, b)| (*a - *b).abs() <= 1e-5 * (1.0 + b.abs()))
        .count();
    assert!(
        close >= gq.len() * 999 / 1000,
        "indexer queries: {close} of {} close",
        gq.len()
    );
    // Score with the GPU's queries so the selection compares like with like.
    let n_max = (pos0 + m) / 4;
    let cap = n_max + 16;
    let keys: Vec<f32> =
        f32s(&mut rng, cap * 128, -1.0, 1.0).iter().map(|&v| round_f16(v)).collect();
    let tkeys = f16_tensor(&ctx, &keys);
    let w = f32s(&mut rng, m * heads, -1.0, 1.0);
    let tw = tensor(&ctx, &w);
    let scores = zeros(&ctx, m * cap, DType::F32);
    let sel = zeros(&ctx, m * 512, DType::U32);
    let count = zeros(&ctx, m, DType::U32);
    let scale = 1.0 / ((128 * heads) as f32).sqrt();
    run(&ctx, |p| {
        indexer_scores(&ctx, p, &tq, &tw, &tkeys, &scores, m, heads, cap, pos0, scale)
            .unwrap();
        p.memory_barrier().unwrap();
        topk(&ctx, p, &scores, &sel, &count, m, cap, pos0, 512).unwrap();
    });
    let gs = scores.to_f32().unwrap();
    let gsel = sel.to_u32().unwrap();
    let gcount = count.to_u32().unwrap();
    for i in 0..m {
        let n = (pos0 + i + 1) / 4;
        let ws: Vec<f32> =
            w[i * heads..(i + 1) * heads].iter().map(|&v| v * scale).collect();
        let want = r::indexer_scores(
            &gq[i * heads * 128..(i + 1) * heads * 128],
            &ws,
            &keys[..n * 128],
            128,
        );
        assert_close(&gs[i * cap..i * cap + n], &want, 1e-4, 1e-4);
        // The selection of the GPU's own scores: the 512 best, ascending.
        let row = &gs[i * cap..i * cap + n];
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        let mut best: Vec<u32> = order[..512].iter().map(|&x| x as u32).collect();
        best.sort();
        assert_eq!(gcount[i], 512);
        assert_eq!(&gsel[i * 512..(i + 1) * 512], &best[..], "row {i}");
    }
}

/// Sink attention over the raw window (ring + the step's rows) and
/// compressed rows, all or selected, against the reference.
#[test]
fn attention_matches_cpu() {
    attention_case(
        408,
        false,
        &[
            (0, 1, 0, false),
            (5, 3, 4, false),
            (300, 4, 128, false),
            (130, 70, 4, false),
            (2100, 2, 4, true),
        ],
        (2e-5, 1e-4),
    );
}

/// The matrix-unit attention of prompt runs (f16 operands): runs within
/// the first window, across it, ratio 128, selected rows, and partial key
/// steps.
#[test]
fn attention_mu_matches_cpu() {
    attention_case(
        409,
        true,
        &[
            (0, 40, 0, false),
            (130, 70, 4, false),
            (300, 33, 128, false),
            (2100, 20, 4, true),
            (7, 200, 4, false),
        ],
        (3e-4, 1e-3),
    );
}

fn attention_case(
    seed: u64,
    matrix: bool,
    cases: &[(usize, usize, usize, bool)],
    tol: (f32, f32),
) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(seed);
    let heads = 64;
    let sinks = f32s(&mut rng, heads, -1.0, 2.0);
    let tsinks = tensor(&ctx, &sinks);
    for &(pos0, m, ratio, select) in cases {
        let total = pos0 + m;
        // Every position's KV row (f16 values).
        let rows: Vec<Vec<f32>> = (0..total)
            .map(|_| {
                f32s(&mut rng, 512, -1.0, 1.0).iter().map(|&v| round_f16(v)).collect()
            })
            .collect();
        let ring: Vec<f32> = (0..128)
            .flat_map(|slot| {
                // The newest position before pos0 with this slot.
                (0..pos0)
                    .rev()
                    .find(|p| p % 128 == slot)
                    .map_or(vec![0.0; 512], |p| rows[p].clone())
            })
            .collect();
        let run_rows: Vec<f32> = rows[pos0..].concat();
        let n_comp = if ratio == 0 { 0 } else { total / ratio };
        let comp: Vec<Vec<f32>> = (0..n_comp.max(1))
            .map(|_| {
                f32s(&mut rng, 512, -1.0, 1.0).iter().map(|&v| round_f16(v)).collect()
            })
            .collect();
        let q = f32s(&mut rng, m * heads * 512, -0.2, 0.2);
        let tq = tensor(&ctx, &q);
        let tring = f16_tensor(&ctx, &ring);
        let trun = f16_tensor(&ctx, &run_rows);
        let tcomp = f16_tensor(&ctx, &comp.concat());
        let out = zeros(&ctx, m * heads * 512, DType::F32);
        // Selected: every third row in the first 512 slots.
        let k = 512;
        let mut sel = vec![0u32; m * k];
        let mut counts = vec![0u32; m];
        for i in 0..m {
            let n = (pos0 + i + 1) / 4;
            let picks: Vec<u32> =
                (0..n as u32).filter(|c| c % 3 != 1).take(k).collect();
            counts[i] = picks.len() as u32;
            sel[i * k..i * k + picks.len()].copy_from_slice(&picks);
        }
        let tsel = u32s(&ctx, &sel);
        let tcount = u32s(&ctx, &counts);
        let rows_arg = match (ratio, select) {
            (0, _) => CompRows::None,
            (_, false) => CompRows::All { cache: &tcomp, ratio },
            (_, true) => {
                CompRows::Selected { cache: &tcomp, sel: &tsel, count: &tcount, k }
            }
        };
        run(&ctx, |p| {
            attend_with(
                &ctx,
                p,
                &tq,
                &trun,
                &tring,
                rows_arg,
                &tsinks,
                &out,
                m,
                heads,
                pos0,
                RawWindow::MODEL,
                matrix,
            )
            .unwrap()
        });
        let got = out.to_f32().unwrap();
        for i in 0..m {
            let p = pos0 + i;
            let mut vis: Vec<&[f32]> =
                (p + 1 - (p + 1).min(128)..=p).map(|x| rows[x].as_slice()).collect();
            if ratio != 0 {
                if select {
                    for &cidx in &sel[i * k..i * k + counts[i] as usize] {
                        vis.push(&comp[cidx as usize]);
                    }
                } else {
                    for cc in comp.iter().take((p + 1) / ratio) {
                        vis.push(cc);
                    }
                }
            }
            let want = r::sink_attention(
                &q[i * heads * 512..(i + 1) * heads * 512],
                &vis,
                &sinks,
                512,
            );
            let got = &got[i * heads * 512..(i + 1) * heads * 512];
            assert_close(got, &want, tol.0, tol.1);
        }
        // The ring after the step holds the newest position of each slot.
        let tring2 = f16_tensor(&ctx, &ring);
        run(&ctx, |p| ring_store(&ctx, p, &trun, &tring2, pos0, m).unwrap());
        let g = tring2.to_f32().unwrap();
        for slot in 0..128 {
            if let Some(pp) = (0..total).rev().find(|x| x % 128 == slot) {
                assert_eq!(
                    &g[slot * 512..(slot + 1) * 512],
                    &rows[pp][..],
                    "slot {slot}"
                );
            }
        }
    }
}

#[test]
fn router_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(409);
    let (e, k, m) = (256usize, 6usize, 4usize);
    let logits = f32s(&mut rng, m * e, -4.0, 4.0);
    let bias = f32s(&mut rng, e, -0.2, 0.2);
    let tl = tensor(&ctx, &logits);
    let tb = tensor(&ctx, &bias);
    let vocab = 20;
    let table: Vec<i32> = (0..vocab * k).map(|_| rng.gen_range(0..e as i32)).collect();
    let hash = QWeight::from_bytes(
        &ctx,
        bytemuck::cast_slice(&table),
        GgmlType::I32,
        k,
        vocab,
    )
    .unwrap();
    let toks = [3u32, 19, 0, 7];
    let tt = u32s(&ctx, &toks);
    for use_hash in [false, true] {
        let sel = zeros(&ctx, m * k, DType::U32);
        let wts = zeros(&ctx, m * k, DType::F32);
        run(&ctx, |p| {
            router(
                &ctx,
                p,
                &tl,
                (!use_hash).then_some(&tb),
                use_hash.then_some(&hash),
                &tt,
                &sel,
                &wts,
                m,
                e,
                k,
                1.5,
            )
            .unwrap()
        });
        let gs = sel.to_u32().unwrap();
        let gw = wts.to_f32().unwrap();
        for i in 0..m {
            let hrow: Vec<usize> = table
                [toks[i] as usize * k..(toks[i] as usize + 1) * k]
                .iter()
                .map(|&x| x as usize)
                .collect();
            let (want_s, want_w) = r::route(
                &logits[i * e..(i + 1) * e],
                (!use_hash).then_some(&bias[..]),
                use_hash.then_some(&hrow[..]),
                k,
                1.5,
            );
            let got_s: Vec<usize> =
                gs[i * k..(i + 1) * k].iter().map(|&x| x as usize).collect();
            assert_eq!(got_s, want_s, "row {i} hash {use_hash}");
            assert_close(&gw[i * k..(i + 1) * k], &want_w, 1e-6, 1e-5);
        }
    }
}

#[test]
fn out_collapse_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(410);
    let (hidden, m) = (4096usize, 2usize);
    let h = f32s(&mut rng, m * 4 * hidden, -1.0, 1.0);
    let (fw, cpu_fn) =
        f16_weight(&ctx, &f32s(&mut rng, 4 * 4 * hidden, -0.05, 0.05), 4 * hidden);
    let base = f32s(&mut rng, 4, -0.5, 0.5);
    let scale = [0.7f32];
    let th = tensor(&ctx, &h);
    let flat = zeros(&ctx, m * 4 * hidden, DType::F32);
    let pre = zeros(&ctx, m * 4, DType::F32);
    let out = zeros(&ctx, m * hidden, DType::F32);
    let (ts, tb) = (tensor(&ctx, &scale), tensor(&ctx, &base));
    run(&ctx, |p| {
        g::rmsnorm(&ctx, p, &th, None, &flat, m, 4 * hidden, 1e-6).unwrap();
        p.memory_barrier().unwrap();
        g::gemv(&ctx, p, &fw, &flat, &pre, m).unwrap();
        p.memory_barrier().unwrap();
        out_collapse(&ctx, p, &pre, &ts, &tb, &th, &out, m, hidden, 1e-6).unwrap();
    });
    let want: Vec<f32> = (0..m)
        .flat_map(|i| {
            r::output_collapse(
                &h[i * 4 * hidden..(i + 1) * 4 * hidden],
                &cpu_fn,
                0.7,
                &base,
                1e-6,
                1e-6,
            )
        })
        .collect();
    assert_close(&out.to_f32().unwrap(), &want, 1e-5, 1e-4);
}

/// The fused HC post + control projection with F16 controls (GLM's
/// kernels, f16 weights) against the CPU HC collapse, with and without the
/// previous sublayer's post.
#[test]
fn hc_post_mix_f16_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(411);
    let (hidden, m, n) = (4096usize, 3usize, 4usize);
    let h0 = f32s(&mut rng, m * n * hidden, -1.0, 1.0);
    let block = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let (fw, cpu_fn) =
        f16_weight(&ctx, &f32s(&mut rng, 24 * n * hidden, -0.02, 0.02), n * hidden);
    let scale = [0.3f32, 0.4, 0.5];
    let base = f32s(&mut rng, 24, -0.5, 0.5);
    let norm = f32s(&mut rng, hidden, 0.5, 1.5);
    let prev_split: Vec<f32> =
        (0..m).flat_map(|_| f32s(&mut rng, 24, 0.0, 1.0)).collect();
    for with_post in [false, true] {
        let th = tensor(&ctx, &h0);
        let tb = tensor(&ctx, &block);
        let split = tensor(&ctx, &prev_split);
        let part = zeros(&ctx, m * g::hc_mix_groups(hidden) * 25, DType::F32);
        let xn = zeros(&ctx, m * hidden, DType::F32);
        let (ts, tbase, tn) =
            (tensor(&ctx, &scale), tensor(&ctx, &base), tensor(&ctx, &norm));
        run(&ctx, |p| {
            g::hc_post_mix(
                &ctx,
                p,
                &th,
                with_post.then_some((&tb, &split)),
                &fw,
                &part,
                m,
                hidden,
            )
            .unwrap();
            p.memory_barrier().unwrap();
            g::hc_pre2(
                &ctx, p, &part, &ts, &tbase, &th, &tn, &split, &xn, m, hidden, 20,
                1e-6, 1e-6,
            )
            .unwrap();
        });
        let gx = xn.to_f32().unwrap();
        let gs = split.to_f32().unwrap();
        for i in 0..m {
            let mut h = h0[i * n * hidden..(i + 1) * n * hidden].to_vec();
            if with_post {
                let sp = &prev_split[i * 24..(i + 1) * 24];
                h = r::hc_post(
                    &h,
                    &block[i * hidden..(i + 1) * hidden],
                    &sp[4..8],
                    &sp[8..24],
                    n,
                );
            }
            let (x, post, comb) = crate::refops::hc_collapse(
                &h, &cpu_fn, &scale, &base, n, 20, 1e-6, 1e-6,
            );
            let want = crate::refops::rmsnorm(&x, Some(&norm), 1e-6);
            assert_close(&gx[i * hidden..(i + 1) * hidden], &want, 1e-4, 1e-3);
            assert_close(&gs[i * 24 + 4..i * 24 + 8], &post, 1e-5, 1e-4);
            assert_close(&gs[i * 24 + 8..i * 24 + 24], &comb, 1e-5, 1e-4);
        }
    }
}

/// DSpark's block attention: every draft row sees the support rows (ring,
/// valid from `lo`) and the whole block, against the reference.
#[test]
fn noncausal_block_attention_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(412);
    let heads = 64;
    let sinks = f32s(&mut rng, heads, -1.0, 2.0);
    let tsinks = tensor(&ctx, &sinks);
    for (fp, lo) in [(3usize, 0usize), (200, 0), (200, 150), (1000, 990)] {
        let pos0 = fp + 1;
        let m = 5;
        let rows: Vec<Vec<f32>> = (0..pos0 + m)
            .map(|_| {
                f32s(&mut rng, 512, -1.0, 1.0).iter().map(|&v| round_f16(v)).collect()
            })
            .collect();
        let ring: Vec<f32> = (0..128)
            .flat_map(|slot| {
                (0..pos0)
                    .rev()
                    .find(|p| p % 128 == slot)
                    .map_or(vec![0.0; 512], |p| rows[p].clone())
            })
            .collect();
        let q = f32s(&mut rng, m * heads * 512, -0.2, 0.2);
        let (tq, tring, trun) = (
            tensor(&ctx, &q),
            f16_tensor(&ctx, &ring),
            f16_tensor(&ctx, &rows[pos0..].concat()),
        );
        let out = zeros(&ctx, m * heads * 512, DType::F32);
        let at = fp + m;
        let raw = RawWindow { at: Some(at), window: 128 + m, lo };
        run(&ctx, |p| {
            attend(
                &ctx,
                p,
                &tq,
                &trun,
                &tring,
                CompRows::None,
                &tsinks,
                &out,
                m,
                heads,
                pos0,
                raw,
            )
            .unwrap()
        });
        let first = lo.max((at + 1).saturating_sub(128 + m));
        let vis: Vec<&[f32]> = (first..=at).map(|x| rows[x].as_slice()).collect();
        let got = out.to_f32().unwrap();
        for i in 0..m {
            let want = r::sink_attention(
                &q[i * heads * 512..(i + 1) * heads * 512],
                &vis,
                &sinks,
                512,
            );
            assert_close(
                &got[i * heads * 512..(i + 1) * heads * 512],
                &want,
                2e-5,
                1e-4,
            );
        }
    }
}

#[test]
fn feature_capture_and_markov_prep() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(413);
    let (hidden, m) = (4096usize, 3usize);
    let h = f32s(&mut rng, (m + 2) * 4 * hidden, -1.0, 1.0);
    let th = tensor(&ctx, &h);
    let out = zeros(&ctx, m * 3 * hidden, DType::F32);
    run(&ctx, |p| {
        for slot in 0..3 {
            hc_capture(&ctx, p, &th, &out, 2, m, slot, 3, hidden).unwrap();
        }
    });
    let got = out.to_f32().unwrap();
    for i in 0..m {
        for slot in 0..3 {
            for d in (0..hidden).step_by(97) {
                let s = &h[(2 + i) * 4 * hidden..];
                let want: f32 = (0..4).map(|k| s[k * hidden + d] * 0.25).sum();
                assert_eq!(got[(i * 3 + slot) * hidden + d], want);
            }
        }
    }
    // Markov: w1 rows of 256 (Q8_0), the previous token from a buffer.
    let (vocab, rank) = (50usize, 256usize);
    let mut raw: Vec<u8> = (0..vocab * rank / 32 * 34).map(|_| rng.r#gen()).collect();
    for blk in raw.chunks_exact_mut(34) {
        blk[..2].copy_from_slice(
            &f16::from_f32(rng.gen_range(0.001..0.02)).to_bits().to_le_bytes(),
        );
    }
    let w1 = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, rank, vocab).unwrap();
    let mut wf = vec![0f32; vocab * rank];
    crate::gguf::dequantize(GgmlType::Q8_0, &raw, &mut wf).unwrap();
    let hid = f32s(&mut rng, hidden, -1.0, 1.0);
    let toks = u32s(&ctx, &[7, 41, 3]);
    let feat = zeros(&ctx, hidden + rank, DType::F32);
    run(&ctx, |p| {
        markov_prep(&ctx, p, &w1, &toks, 1, &tensor(&ctx, &hid), &feat, hidden).unwrap()
    });
    let f = feat.to_f32().unwrap();
    bits_eq(&f[..hidden], &hid, "features: hidden");
    bits_eq(&f[hidden..], &wf[41 * rank..42 * rank], "features: w1 row");
}
