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
/// the draft head's catch-up over a prompt chunk past position 0 embeds the
/// chunk's tokens exactly from the staged rows, so after a prompt fed in
/// 256-row chunks, with a token outside the narrow LM head in a later chunk
/// (the first chunk all inside it), the head's attention cache holds the
/// mapped table's bits. Its narrow rows would embed that token as zeros.
#[test]
#[ignore = "requires LILY_QWEN38_Q2"]
fn paged_catch_up_past_the_first_chunk_matches_the_mapped_table() {
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
        // In-range ids (the low ones), one outside the narrow head at 600.
        let mut prompt: Vec<u32> = (0..700u32).map(|i| 1000 + (i * 37) % 20000).collect();
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
