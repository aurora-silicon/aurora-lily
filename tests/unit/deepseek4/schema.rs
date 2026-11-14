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
