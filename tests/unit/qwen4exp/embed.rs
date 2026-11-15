use half::bf16;

use std::path::Path;

use super::*;
use crate::metal::MetalContext;

/// Rows staged from the file are the table's rows, bit for bit, in token
/// order (repeats too), and an id past the table is refused.
#[test]
fn paged_rows_are_the_files_rows() {
    let ctx = MetalContext::new().expect("metal context");
    let (rows, cols, offset) = (37usize, 24usize, 100u64);
    let values: Vec<u16> =
        (0..rows * cols).map(|i| bf16::from_f32(i as f32 * 0.25 - 50.0).to_bits()).collect();
    let mut bytes = vec![0xabu8; offset as usize];
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    bytes.extend([0xcdu8; 64]);
    let path = std::env::temp_dir().join(format!("lily-paged-embed-{}.bin", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let table = PagedEmbed::open(&path, offset, rows, cols).unwrap();
    let tokens = [5u32, 0, 36, 5, 17];
    let out = Tensor::zeros(&ctx, &[8 * cols], DType::BF16).unwrap();
    table.fill(&tokens, &out).unwrap();
    let got: Vec<u16> = out.to_f32().unwrap().iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
    for (i, &t) in tokens.iter().enumerate() {
        let t = t as usize;
        assert_eq!(got[i * cols..(i + 1) * cols], values[t * cols..(t + 1) * cols], "token {t}");
    }
    assert!(got[tokens.len() * cols..].iter().all(|&v| v == 0), "past the staged rows");
    assert!(table.fill(&[37], &out).is_err(), "id past the table");
    assert_eq!(table.read_rows(3, 2).unwrap(), bytes[offset as usize + 3 * cols * 2..][..2 * cols * 2]);
    std::fs::remove_file(&path).unwrap();
}

/// With the Q2 file (`LILY_QWEN38_Q2`, n-gram table in `LILY_NGRAM_TABLE`):
/// the draft head's catch-up over a prompt chunk embeds the chunk's tokens
/// exactly from the staged rows (shifted by one in the first chunk), so
/// after a prompt fed in 256-row chunks, with a token outside the narrow LM
/// head in the first chunk and one in a later chunk, the head's attention
/// cache holds the mapped table's bits. Its narrow rows would embed those
/// tokens as zeros.
#[test]
#[ignore = "requires LILY_QWEN38_Q2"]
fn paged_catch_up_matches_the_mapped_table() {
    use crate::engine::{LanguageModel, LoadOptions};
    use crate::qwen4exp::model::{LayerState, Qwen4ExpModel};
    let path = std::env::var("LILY_QWEN38_Q2").expect("LILY_QWEN38_Q2");
    let limit = (44.7 * (1u64 << 30) as f64) as u64;
    let cache = |paged: bool| -> (Vec<u8>, Vec<u8>) {
        // SAFETY: single-threaded here; the loader reads it once a load.
        unsafe { std::env::set_var("LILY_EMBED_PAGED", if paged { "1" } else { "0" }) };
        let ctx = MetalContext::new().unwrap();
        let mut model = <Qwen4ExpModel as LanguageModel>::load(
            &ctx,
            Path::new(&path),
            &LoadOptions { mtp_drafts: 2, memory_limit: Some(limit), ..LoadOptions::default() },
        )
        .unwrap();
        model.set_prefill_chunk(256).unwrap();
        // In-range ids (the low ones), outside the narrow head at 100 and 600.
        let mut prompt: Vec<u32> = (0..700u32).map(|i| 1000 + (i * 37) % 20000).collect();
        prompt[100] = 200_001;
        prompt[600] = 200_000;
        let mut state = model.new_state(&ctx, 800).unwrap();
        let mut scratch = model.new_scratch_with_capacity(&ctx, 800).unwrap();
        model.prefill(&ctx, &mut state, &mut scratch, &prompt[..256], None).unwrap();
        model.prefill(&ctx, &mut state, &mut scratch, &prompt[256..], None).unwrap();
        let mtp = state.mtp.as_ref().expect("draft head state");
        let LayerState::Attn { k_cache, v_cache, .. } = &mtp.layer else { panic!("attention") };
        (k_cache.values.raw_bytes().to_vec(), v_cache.values.raw_bytes().to_vec())
    };
    let mapped = cache(false);
    let paged = cache(true);
    assert!(paged.0 == mapped.0, "draft head K cache");
    assert!(paged.1 == mapped.1, "draft head V cache");
}

/// With the Q2 file: the draft pass embeds an accepted draft from the verify
/// pass's staged row of it, and the next draft pass rewrites the head's row
/// of a new draw outside the narrow LM head (embedded as zeros) from its
/// staged row as the next pending token. After verify passes over a repeated
/// Chinese and Italian sentence, with accepted drafts and new draws outside
/// the narrow head, the head's attention cache holds the mapped table's bits
/// up to the last pass's position except those rewritten rows, which are
/// within a one-row call's rounding of the mapped rows (zeros are not). The
/// drafts are what followed the pending token's last earlier occurrence (not
/// the head's proposals), so the trunk alone decides what is accepted.
#[test]
#[ignore = "requires LILY_QWEN38_Q2"]
fn paged_draft_passes_match_the_mapped_table() {
    use crate::engine::{DecodeStateApi, LanguageModel, LoadOptions};
    use crate::kernels::sample::SamplingParams;
    use crate::qwen4exp::model::{LayerState, Qwen4ExpModel};
    let path = std::env::var("LILY_QWEN38_Q2").expect("LILY_QWEN38_Q2");
    let limit = (44.7 * (1u64 << 30) as f64) as u64;
    let text = "图书馆为读者保留了一份简短的规则清单。La biblioteca è aperta ogni giorno. "
        .repeat(30);
    let generator = crate::generate::Generator::from_model_dir(Path::new(&path)).unwrap();
    let prompt = generator.tokenizer().encode(&text).unwrap();
    let outside = |t: u32| (98_304..248_044).contains(&t);
    // Per head and position below the last pass's: the raw bytes of values
    // and scales, and the dequantized row.
    type Rows = Vec<(Vec<u8>, Vec<f32>)>;
    let run = |paged: bool| -> (Vec<(Vec<u32>, u32)>, Vec<usize>, usize, Rows, Rows) {
        // SAFETY: single-threaded here; the loader reads it once a load.
        unsafe { std::env::set_var("LILY_EMBED_PAGED", if paged { "1" } else { "0" }) };
        let ctx = MetalContext::new().unwrap();
        let model = <Qwen4ExpModel as LanguageModel>::load(
            &ctx,
            Path::new(&path),
            &LoadOptions { mtp_drafts: 2, memory_limit: Some(limit), ..LoadOptions::default() },
        )
        .unwrap();
        let greedy = SamplingParams::greedy();
        let n = prompt.len();
        let mut state = model.new_state(&ctx, n + 128).unwrap();
        let mut scratch = model.new_scratch_with_capacity(&ctx, n + 128).unwrap();
        model.prefill(&ctx, &mut state, &mut scratch, &prompt[..n - 1], None).unwrap();
        let mut seen = prompt.clone();
        let mut pending = prompt[n - 1];
        model.draft_initial(&ctx, &mut state, &mut scratch, pending, 2, &greedy, 1).unwrap();
        let (mut steps, mut rewritten, mut last_pos) = (Vec::new(), Vec::new(), 0);
        for step in 0..16 {
            let last = seen.len() - 1;
            let from = (0..last - 2).rev().find(|&i| seen[i] == pending).unwrap_or(0);
            let drafts = [seen[from + 1], seen[from + 2]];
            last_pos = state.pos();
            if outside(pending) {
                rewritten.push(last_pos - 1);
            }
            let (sampled, draft) = model
                .verify(&ctx, &mut state, &mut scratch, pending, &drafts, &greedy, 1 + 3 * step, 2)
                .unwrap();
            let kept = sampled.iter().zip(&drafts).take_while(|(s, d)| s == d).count();
            model.finish_speculation(&ctx, &mut state, &mut scratch, kept, true, draft).unwrap();
            steps.push((drafts[..kept].to_vec(), sampled[kept]));
            seen.extend_from_slice(&sampled[..=kept]);
            pending = sampled[kept];
        }
        let mtp = state.mtp.as_ref().expect("draft head state");
        let LayerState::Attn { k_cache, v_cache, .. } = &mtp.layer else { panic!("attention") };
        let rows = |c: &crate::kernels::attention::KvCache| -> Rows {
            let (heads, cap, d) = (c.values.shape()[0], c.values.shape()[1], c.values.shape()[2]);
            let scales = c.scales.as_ref().expect("a q8 draft head cache");
            let groups = scales.shape()[2];
            let values = c.values.raw_bytes();
            let sb = scales.raw_bytes();
            let sf: Vec<f32> = scales.to_f32().unwrap();
            let mut out = Vec::new();
            for hd in 0..heads {
                for p in 0..last_pos {
                    let r = hd * cap + p;
                    let v = &values[r * d..(r + 1) * d];
                    let raw = [v, &sb[r * groups * 2..(r + 1) * groups * 2]].concat();
                    let deq = v
                        .iter()
                        .enumerate()
                        .map(|(i, &b)| b as i8 as f32 * sf[r * groups + i / (d / groups)])
                        .collect();
                    out.push((raw, deq));
                }
            }
            out
        };
        (steps, rewritten, last_pos, rows(k_cache), rows(v_cache))
    };
    let mapped = run(false);
    let paged = run(true);
    assert_eq!(paged.0, mapped.0, "the trunk's draws");
    assert!(
        mapped.0.iter().any(|(acc, _)| acc.iter().any(|&t| outside(t))),
        "no accepted draft outside the narrow head: {:?}",
        mapped.0
    );
    assert!(!mapped.1.is_empty(), "no new draw outside the narrow head: {:?}", mapped.0);
    let last_pos = mapped.2;
    for (name, p, m) in [("K", &paged.3, &mapped.3), ("V", &paged.4, &mapped.4)] {
        for (i, (a, b)) in p.iter().zip(m).enumerate() {
            let pos = i % last_pos;
            if !mapped.1.contains(&pos) {
                assert!(a.0 == b.0, "{name} cache at position {pos} (head {})", i / last_pos);
                continue;
            }
            let err: f32 = a.1.iter().zip(&b.1).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
            let norm: f32 = b.1.iter().map(|y| y * y).sum::<f32>().sqrt();
            eprintln!("{name} rewritten position {pos}: relative error {:.4}", err / norm);
            assert!(err <= 0.02 * norm, "{name} rewritten row at {pos}: error {err} of {norm}");
        }
    }
}
