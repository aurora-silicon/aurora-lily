use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn temp_file(tag: &str, bytes: &[u8]) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir()
        .join(format!("lily-st-{tag}-{}-{id}.safetensors", std::process::id()));
    std::fs::write(&path, bytes).expect("write temp safetensors");
    path
}

fn build_shard(header_json: &str, payload: &[u8]) -> Vec<u8> {
    let hb = header_json.as_bytes();
    let mut out = Vec::with_capacity(8 + hb.len() + payload.len());
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(hb);
    out.extend_from_slice(payload);
    out
}

#[test]
fn parses_valid_shard_and_reads_tensor() {
    let payload: Vec<u8> =
        [1.5f32, -2.25f32].into_iter().flat_map(f32::to_le_bytes).collect();
    let bytes = build_shard(
        r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &payload,
    );
    let path = temp_file("valid", &bytes);
    let ckpt = Checkpoint::open_file(&path).unwrap();
    assert_eq!(ckpt.read("w").unwrap(), payload);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn rejects_oversized_and_truncated_headers_without_allocating() {
    // 256 MiB claimed header in an 8-byte file.
    let huge = (256u64 << 20).to_le_bytes();
    let path = temp_file("huge-hdr", &huge);
    assert!(Checkpoint::open_file(&path).is_err());
    let _ = std::fs::remove_file(&path);

    // Claimed header within cap but larger than file length.
    let trunc = (1024u64).to_le_bytes();
    let path = temp_file("trunc-hdr", &trunc);
    assert!(Checkpoint::open_file(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn rejects_offset_overflow_out_of_bounds_and_reversed_offsets() {
    let cases = [
        (
            "overflow",
            r#"{"w":{"dtype":"U8","shape":[1],"data_offsets":[18446744073709551610,18446744073709551615]}}"#,
            vec![0u8; 8],
        ),
        (
            "oob",
            r#"{"w":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}}"#,
            vec![0u8; 8],
        ),
        (
            "reversed",
            r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[8,4]}}"#,
            vec![0u8; 8],
        ),
    ];
    for (tag, hdr, payload) in cases {
        let bytes = build_shard(hdr, &payload);
        let path = temp_file(tag, &bytes);
        assert!(Checkpoint::open_file(&path).is_err(), "expected err for {tag}");
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn rejects_shape_dtype_byte_length_mismatch() {
    // Shape [2, 2] of F32 needs 16 bytes, not 8.
    let bytes = build_shard(
        r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    );
    let path = temp_file("bad-len", &bytes);
    assert!(Checkpoint::open_file(&path).is_err());
    let _ = std::fs::remove_file(&path);

    // Unsupported dtypes (Other) still open without failing unrelated tensors.
    let bytes = build_shard(
        r#"{"x":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    );
    let path = temp_file("other-dtype", &bytes);
    let ckpt = Checkpoint::open_file(&path).unwrap();
    assert!(ckpt.read("x").is_err());
    let _ = std::fs::remove_file(&path);

    // 0-D scalar (shape: []) and zero-element (shape: [0]) tensors are valid.
    let scalar_payload = 42.0f32.to_le_bytes();
    let bytes = build_shard(
        r#"{"s":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"z":{"dtype":"F32","shape":[0,4],"data_offsets":[4,4]}}"#,
        &scalar_payload,
    );
    let path = temp_file("scalar-zero", &bytes);
    let ckpt = Checkpoint::open_file(&path).unwrap();
    assert_eq!(ckpt.read("s").unwrap(), scalar_payload);
    assert!(ckpt.read("z").unwrap().is_empty());
    let _ = std::fs::remove_file(&path);
}
