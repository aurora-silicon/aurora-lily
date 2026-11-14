use super::*;

/// The published Q2 file's header: `LILY_DS4_GGUF=<path> cargo test --lib
/// deepseek4::config -- --ignored` (reads the header only).
#[test]
#[ignore = "requires LILY_DS4_GGUF"]
fn real_header_validates() {
    let path = std::env::var("LILY_DS4_GGUF").expect("LILY_DS4_GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).expect("open");
    let c = Config::from_gguf(&g).expect("config");
    assert_eq!(c.compress_ratios, Config::deepseek4_flash().compress_ratios);
    assert_eq!((c.bos, c.eos), (0, 1));
    assert_eq!(c.rms_eps, 1e-6);
    assert_eq!(c.expert_weight_scale, 1.5);
    assert_eq!(g.tensors.len(), 1328);
    crate::deepseek4::schema::validate(&g, &c)
        .expect("the published file matches the schema");
}

/// One metadata entry of a synthetic header.
enum Meta {
    U(u64),
    F(f32),
    S(&'static str),
    B(bool),
    Us(Vec<u64>),
    Fs(Vec<f32>),
}

/// The published file's metadata (the keys the config reads).
fn metadata() -> Vec<(String, Meta)> {
    let c = Config::deepseek4_flash();
    let mut ratios: Vec<u64> = c.compress_ratios.iter().map(|&r| r as u64).collect();
    ratios.push(0); // the MTP entry
    let ints: [(&str, usize); 21] = [
        ("block_count", 43),
        ("embedding_length", 4096),
        ("vocab_size", 129280),
        ("attention.head_count", 64),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 512),
        ("attention.value_length", 512),
        ("rope.dimension_count", 64),
        ("attention.q_lora_rank", 1024),
        ("attention.output_lora_rank", 1024),
        ("attention.output_group_count", 8),
        ("expert_count", 256),
        ("expert_used_count", 6),
        ("expert_shared_count", 1),
        ("expert_feed_forward_length", 2048),
        ("hash_layer_count", 3),
        ("attention.sliding_window", 128),
        ("attention.indexer.head_count", 64),
        ("attention.indexer.key_length", 128),
        ("attention.indexer.top_k", 512),
        ("hyper_connection.count", 4),
    ];
    let mut v: Vec<(String, Meta)> =
        ints.iter().map(|(k, n)| (format!("{ARCH}.{k}"), Meta::U(*n as u64))).collect();
    for (k, x) in [
        ("hyper_connection.sinkhorn_iterations", 20u64),
        ("context_length", 1 << 20),
        ("rope.scaling.original_context_length", 65536),
        ("expert_gating_func", 4),
    ] {
        v.push((format!("{ARCH}.{k}"), Meta::U(x)));
    }
    for (k, x) in [
        ("hyper_connection.epsilon", 1e-6f32),
        ("attention.layer_norm_rms_epsilon", 1e-6),
        ("rope.freq_base", 10000.0),
        ("rope.scaling.factor", 16.0),
        ("rope.scaling.yarn_beta_fast", 32.0),
        ("rope.scaling.yarn_beta_slow", 1.0),
        ("attention.compress_rope_freq_base", 160000.0),
        ("expert_weights_scale", 1.5),
    ] {
        v.push((format!("{ARCH}.{k}"), Meta::F(x)));
    }
    v.push((format!("{ARCH}.rope.scaling.type"), Meta::S("yarn")));
    v.push((format!("{ARCH}.expert_weights_norm"), Meta::B(true)));
    v.push((format!("{ARCH}.attention.compress_ratios"), Meta::Us(ratios)));
    v.push((format!("{ARCH}.swiglu_clamp_exp"), Meta::Fs(vec![10.0; 43])));
    v.push(("tokenizer.ggml.bos_token_id".into(), Meta::U(0)));
    v.push(("tokenizer.ggml.eos_token_id".into(), Meta::U(1)));
    v
}

fn header(meta: &[(String, Meta)]) -> Vec<u8> {
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
    s(&mut f, ARCH);
    for (k, m) in meta {
        s(&mut f, k);
        match m {
            Meta::U(v) => {
                f.extend(10u32.to_le_bytes());
                f.extend(v.to_le_bytes());
            }
            Meta::F(v) => {
                f.extend(6u32.to_le_bytes());
                f.extend(v.to_le_bytes());
            }
            Meta::S(v) => {
                f.extend(8u32.to_le_bytes());
                s(&mut f, v);
            }
            Meta::B(v) => {
                f.extend(7u32.to_le_bytes());
                f.push(u8::from(*v));
            }
            Meta::Us(v) => {
                f.extend(9u32.to_le_bytes());
                f.extend(4u32.to_le_bytes()); // u32 elements
                f.extend((v.len() as u64).to_le_bytes());
                for x in v {
                    f.extend((*x as u32).to_le_bytes());
                }
            }
            Meta::Fs(v) => {
                f.extend(9u32.to_le_bytes());
                f.extend(6u32.to_le_bytes());
                f.extend((v.len() as u64).to_le_bytes());
                for x in v {
                    f.extend(x.to_le_bytes());
                }
            }
        }
    }
    f
}

fn parse(name: &str, meta: &[(String, Meta)]) -> Result<Config> {
    let dir = std::env::temp_dir().join(format!("lily-ds4-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, header(meta)).unwrap();
    let r = Gguf::open(&p).and_then(|g| Config::from_gguf(&g));
    std::fs::remove_file(&p).ok();
    r
}

#[test]
fn the_published_metadata_parses() {
    let c = parse("good.gguf", &metadata()).unwrap();
    let want = Config::deepseek4_flash();
    assert_eq!(format!("{c:?}"), format!("{want:?}"));
    assert_eq!(c.compress_ratios.iter().filter(|&&r| r == 4).count(), 21);
    assert_eq!(c.compress_ratios.iter().filter(|&&r| r == 128).count(), 20);
    assert_eq!((c.o_group_heads(), c.q_dim(), c.hc_mix()), (8, 32768, 24));
}

/// Metadata that would change the graph is refused.
#[test]
fn other_shapes_are_refused() {
    let set = |key: &str, m: Meta| {
        let mut v = metadata();
        let k = format!("{ARCH}.{key}");
        let i = v.iter().position(|(n, _)| *n == k).expect(key);
        v[i].1 = m;
        v
    };
    let mut ratios: Vec<u64> =
        Config::deepseek4_flash().compress_ratios.iter().map(|&r| r as u64).collect();
    ratios[5] = 4;
    let cases = [
        (
            "expert_used_count",
            set("expert_used_count", Meta::U(8)),
            "expert_used_count",
        ),
        (
            "ratios",
            set("attention.compress_ratios", Meta::Us(ratios)),
            "compress ratio",
        ),
        (
            "short ratios",
            set("attention.compress_ratios", Meta::Us(vec![0; 10])),
            "compress_ratios",
        ),
        ("gating", set("expert_gating_func", Meta::U(2)), "sqrt-softplus"),
        (
            "clamp",
            set(
                "swiglu_clamp_exp",
                Meta::Fs((0..43).map(|i| if i == 9 { 7.0 } else { 10.0 }).collect()),
            ),
            "SwiGLU clamp",
        ),
        ("eps", set("attention.layer_norm_rms_epsilon", Meta::F(1e-5)), "epsilon"),
        ("yarn", set("rope.scaling.factor", Meta::F(8.0)), "rope.scaling.factor"),
        ("kv heads", set("attention.head_count_kv", Meta::U(2)), "head_count_kv"),
        ("norm", set("expert_weights_norm", Meta::B(false)), "expert_weights_norm"),
    ];
    for (what, meta, msg) in cases {
        let e = parse("bad.gguf", &meta).unwrap_err();
        assert!(format!("{e:#}").contains(msg), "{what}: {e:#}");
    }
    let mut v = metadata();
    v.push((format!("{ARCH}.checkpoint_variant"), Meta::S("vision-exp")));
    let e = parse("variant.gguf", &v).unwrap_err();
    assert!(format!("{e:#}").contains("Vision-Exp"), "{e:#}");
}
