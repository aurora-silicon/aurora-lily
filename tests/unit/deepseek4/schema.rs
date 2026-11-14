use super::*;

/// The type a test file gives a role: the published file's.
fn first_type(s: &Spec) -> GgmlType {
    if s.name.ends_with("ffn_down_exps.weight") { GgmlType::Q2K } else { s.types[0] }
}

/// Writes a GGUF holding only a tensor table at 32-byte-aligned offsets; the
/// data is a hole of the file (sparse), so the ~87 GB table costs nothing.
fn write_table(
    name: &str,
    tensors: &[(String, Vec<u64>, GgmlType)],
) -> std::path::PathBuf {
    let mut f = Vec::from(&b"GGUF"[..]);
    f.extend(3u32.to_le_bytes());
    f.extend((tensors.len() as u64).to_le_bytes());
    f.extend(0u64.to_le_bytes());
    let mut offset = 0u64;
    for (n, dims, ty) in tensors {
        f.extend((n.len() as u64).to_le_bytes());
        f.extend(n.as_bytes());
        f.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            f.extend(d.to_le_bytes());
        }
        f.extend((*ty as u32).to_le_bytes());
        f.extend(offset.to_le_bytes());
        let rows: u64 = dims[1..].iter().product();
        let bytes = ty.row_bytes(dims[0] as usize).unwrap() as u64 * rows;
        offset = (offset + bytes).next_multiple_of(32);
    }
    let data = (f.len() as u64).next_multiple_of(32);
    let dir =
        std::env::temp_dir().join(format!("lily-ds4-schema-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.gguf"));
    let file = std::fs::File::create(&path).unwrap();
    use std::io::Write;
    (&file).write_all(&f).unwrap();
    file.set_len(data + offset).unwrap();
    path
}

fn table(c: &Config) -> Vec<(String, Vec<u64>, GgmlType)> {
    expected(c)
        .iter()
        .map(|s| (s.name.clone(), s.dims.clone(), first_type(s)))
        .collect()
}

fn check(name: &str, t: &[(String, Vec<u64>, GgmlType)]) -> Result<()> {
    let path = write_table(name, t);
    let r = Gguf::open(&path).and_then(|g| validate(&g, &Config::deepseek4_flash()));
    std::fs::remove_file(&path).ok();
    r
}

/// The schema has the published file's 1 328 tensors, 86.7 GB of them.
#[test]
fn the_published_layout_validates() {
    let c = Config::deepseek4_flash();
    let t = table(&c);
    assert_eq!(t.len(), 1328);
    let bytes: u64 = t
        .iter()
        .map(|(_, d, ty)| {
            ty.row_bytes(d[0] as usize).unwrap() as u64 * d[1..].iter().product::<u64>()
        })
        .sum();
    assert_eq!(bytes, 86_714_775_900);
    check("good", &t).unwrap();
}

#[test]
fn wrong_tables_are_refused() {
    let c = Config::deepseek4_flash();
    let t = table(&c);
    let edit = |name: &str, f: &dyn Fn(&mut (String, Vec<u64>, GgmlType))| {
        let mut t = t.clone();
        f(t.iter_mut().find(|e| e.0 == name).unwrap());
        t
    };
    let short = edit("blk.7.attn_q_b.weight", &|e| e.1[1] -= 1);
    assert!(
        format!("{:#}", check("short", &short).unwrap_err()).contains("dimensions")
    );
    let ty = edit("blk.2.indexer.proj.weight", &|e| e.2 = GgmlType::Bf16);
    assert!(format!("{:#}", check("type", &ty).unwrap_err()).contains("BF16"));
    let mixed = edit("blk.9.ffn_up_exps.weight", &|e| e.2 = GgmlType::Q2K);
    assert!(
        format!("{:#}", check("mixed", &mixed).unwrap_err())
            .contains("different types")
    );
    let mut missing = t.clone();
    missing.retain(|e| e.0 != "blk.4.indexer_compressor_ape.weight");
    assert!(check("missing", &missing).is_err());
    let mut extra = t.clone();
    extra.push((
        "blk.43.nextn.eh_proj.weight".into(),
        vec![8192, 4096],
        GgmlType::Q8_0,
    ));
    assert!(
        format!("{:#}", check("extra", &extra).unwrap_err()).contains("1329 tensors")
    );
}

/// The published support file's metadata.
fn dspark_meta() -> DsparkMeta {
    DsparkMeta {
        stages: 3,
        block: 5,
        markov_rank: 256,
        noise: 128799,
        targets: vec![40, 41, 42],
    }
}

fn dspark_table(c: &Config, m: &DsparkMeta) -> Vec<(String, Vec<u64>, GgmlType)> {
    dspark_expected(c, m)
        .unwrap()
        .iter()
        .map(|s| (s.name.clone(), s.dims.clone(), first_type(s)))
        .collect()
}

fn check_dspark(name: &str, t: &[(String, Vec<u64>, GgmlType)]) -> Result<()> {
    let path = write_table(name, t);
    let r = Gguf::open(&path)
        .and_then(|g| validate_dspark(&g, &Config::deepseek4_flash(), &dspark_meta()));
    std::fs::remove_file(&path).ok();
    r
}

/// The schema has the published support file's 81 tensors, 5.99 GB.
#[test]
fn the_published_dspark_layout_validates() {
    let c = Config::deepseek4_flash();
    let t = dspark_table(&c, &dspark_meta());
    assert_eq!(t.len(), 81);
    let bytes: u64 = t
        .iter()
        .map(|(_, d, ty)| {
            ty.row_bytes(d[0] as usize).unwrap() as u64 * d[1..].iter().product::<u64>()
        })
        .sum();
    assert_eq!(bytes, 5_989_108_652);
    check_dspark("dspark-good", &t).unwrap();
}

/// Short vectors, a wrong stage or expert layout, mixed fused types and
/// stray tensors are refused before anything is mapped.
#[test]
fn wrong_dspark_tables_are_refused() {
    let c = Config::deepseek4_flash();
    let t = dspark_table(&c, &dspark_meta());
    let edit = |name: &str, f: &dyn Fn(&mut (String, Vec<u64>, GgmlType))| {
        let mut t = t.clone();
        f(t.iter_mut().find(|e| e.0 == name).expect(name));
        t
    };
    let cases = [
        ("norm", edit("mtp.0.main_norm.weight", &|e| e.1 = vec![1]), "dimensions"),
        ("sinks", edit("mtp.1.attn_sinks.weight", &|e| e.1 = vec![1]), "dimensions"),
        (
            "hc base",
            edit("mtp.2.hc_attn_base.weight", &|e| e.1 = vec![23]),
            "dimensions",
        ),
        (
            "hc head",
            edit("mtp.2.hc_head_base.weight", &|e| e.1 = vec![1]),
            "dimensions",
        ),
        (
            "experts",
            edit("mtp.1.ffn_gate_exps.weight", &|e| e.1[2] = 255),
            "dimensions",
        ),
        (
            "markov",
            edit("mtp.2.markov_head.markov_w1.weight", &|e| e.1[1] -= 1),
            "dimensions",
        ),
        (
            "rank",
            edit("mtp.0.attn_q_a_norm.weight", &|e| e.1 = vec![1024, 1]),
            "dimensions",
        ),
        ("router", edit("mtp.0.ffn_gate_inp.weight", &|e| e.2 = GgmlType::F16), "F16"),
        (
            "vector type",
            edit("mtp.0.attn_norm.weight", &|e| e.2 = GgmlType::F16),
            "F16",
        ),
        (
            "fused",
            edit("mtp.2.ffn_up_exps.weight", &|e| e.2 = GgmlType::Q2K),
            "different types",
        ),
    ];
    for (what, t, msg) in cases {
        let e = check_dspark(what, &t).unwrap_err();
        assert!(format!("{e:#}").contains(msg), "{what}: {e:#}");
    }
    // A stage's tensors under another stage's name, a missing head, a
    // stray tensor.
    let mut moved = t.clone();
    moved.iter_mut().find(|e| e.0 == "mtp.1.attn_kv.weight").unwrap().0 =
        "mtp.3.attn_kv.weight".into();
    assert!(check_dspark("moved", &moved).is_err());
    let mut missing = t.clone();
    missing.retain(|e| e.0 != "mtp.2.confidence_head.proj.weight");
    assert!(check_dspark("missing", &missing).is_err());
    let mut extra = t.clone();
    extra.push(("mtp.0.attn_compressor_norm.weight".into(), vec![512], GgmlType::F32));
    assert!(
        format!("{:#}", check_dspark("extra", &extra).unwrap_err())
            .contains("82 tensors")
    );
}

/// A support-file header holding only metadata.
fn dspark_header(meta: &[(&str, Vec<u32>)], arch: &str) -> Result<DsparkMeta> {
    let mut f = Vec::from(&b"GGUF"[..]);
    f.extend(3u32.to_le_bytes());
    f.extend(0u64.to_le_bytes());
    f.extend((meta.len() as u64 + 1).to_le_bytes());
    let s = |f: &mut Vec<u8>, v: &str| {
        f.extend((v.len() as u64).to_le_bytes());
        f.extend(v.as_bytes());
    };
    s(&mut f, "general.architecture");
    f.extend(8u32.to_le_bytes());
    s(&mut f, arch);
    for (k, v) in meta {
        s(&mut f, k);
        if k.ends_with("target_layer_ids") {
            f.extend(9u32.to_le_bytes());
            f.extend(4u32.to_le_bytes());
            f.extend((v.len() as u64).to_le_bytes());
        } else {
            f.extend(4u32.to_le_bytes());
        }
        for x in v {
            f.extend(x.to_le_bytes());
        }
    }
    let dir =
        std::env::temp_dir().join(format!("lily-ds4-dspark-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{}.gguf", meta.iter().map(|m| m.1.len()).sum::<usize>()));
    std::fs::write(&p, f).unwrap();
    let r = Gguf::open(&p)
        .and_then(|g| DsparkMeta::read(&g, &Config::deepseek4_flash(), 7));
    std::fs::remove_file(&p).ok();
    r
}

/// Metadata outside what lily runs is refused, with checked arithmetic.
#[test]
fn dspark_metadata_is_bounded() {
    let good: Vec<(&str, Vec<u32>)> = vec![
        ("dspark.block_size", vec![5]),
        ("dspark.markov_rank", vec![256]),
        ("dspark.noise_token_id", vec![128799]),
        ("dspark.target_layer_ids", vec![40, 41, 42]),
        ("dspark.stage_count", vec![3]),
        ("dspark.n_layers", vec![3]),
    ];
    let m = dspark_header(&good, DSPARK_ARCH).unwrap();
    assert_eq!(format!("{m:?}"), format!("{:?}", dspark_meta()));
    let set = |key: &str, v: Vec<u32>| {
        let mut m = good.clone();
        m.iter_mut().find(|e| e.0 == key).unwrap().1 = v;
        m
    };
    let cases = [
        set("dspark.block_size", vec![0]),
        set("dspark.block_size", vec![8]),
        set("dspark.stage_count", vec![0]),
        set("dspark.stage_count", vec![9]),
        set("dspark.n_layers", vec![2]),
        set("dspark.markov_rank", vec![0]),
        set("dspark.markov_rank", vec![100]),
        set("dspark.markov_rank", vec![8192]),
        set("dspark.noise_token_id", vec![129280]),
        set("dspark.target_layer_ids", vec![]),
        set("dspark.target_layer_ids", vec![39, 41, 42]),
        set("dspark.target_layer_ids", vec![41, 42, 43]),
        set("dspark.target_layer_ids", vec![u32::MAX, 42]),
    ];
    for m in &cases {
        assert!(dspark_header(m, DSPARK_ARCH).is_err(), "{m:?}");
    }
    let e = dspark_header(&good, "deepseek4").unwrap_err();
    assert!(format!("{e:#}").contains("deepseek4-dspark"), "{e:#}");
}
