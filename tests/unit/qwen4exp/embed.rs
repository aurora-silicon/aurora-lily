use half::bf16;

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
