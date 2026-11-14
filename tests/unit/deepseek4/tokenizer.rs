use crate::tokenizer::{ChatRender, Tokenizer};

/// `LILY_DS4_GGUF=<path> cargo test --lib deepseek4::tokenizer -- --ignored`
/// (reads the header only).
fn tok() -> Tokenizer {
    let path = std::env::var("LILY_DS4_GGUF").expect("LILY_DS4_GGUF");
    Tokenizer::from_model_dir(std::path::Path::new(&path)).expect("tokenizer")
}

#[test]
#[ignore = "requires LILY_DS4_GGUF"]
fn encodes_and_renders_like_ds4() {
    let t = tok();
    assert_eq!(
        t.encode("The capital of France is").unwrap(),
        [671, 6102, 294, 8760, 344]
    );
    // Digits go in runs of three; punctuation keeps its newline; a run of
    // spaces gives its last to the next word.
    for s in [
        "12345678",
        "foo();\n    int x = 1;\n",
        "héllo, wörld! 日本語のテキスト",
        "a\n\n\nb",
    ] {
        let ids = t.encode(s).unwrap();
        assert_eq!(t.decode(&ids, false).unwrap(), s, "round trip of {s:?}");
    }
    let digits: Vec<String> = t
        .encode("12345678")
        .unwrap()
        .iter()
        .map(|&i| t.decode(&[i], false).unwrap())
        .collect();
    assert_eq!(digits, ["123", "456", "78"]);
    let spaces: Vec<String> = t
        .encode("x\n    int")
        .unwrap()
        .iter()
        .map(|&i| t.decode(&[i], false).unwrap())
        .collect();
    assert_eq!(spaces.last().unwrap(), " int", "{spaces:?}");
    let msgs = serde_json::json!([{"role": "user", "content": "Hi"}]);
    let msgs = msgs.as_array().unwrap();
    let mut chat = ChatRender {
        messages: msgs,
        tools: None,
        enable_thinking: false,
        reasoning_effort: None,
        preserve_thinking: None,
    };
    assert_eq!(
        t.render(&chat).unwrap(),
        "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>"
    );
    chat.enable_thinking = true;
    assert_eq!(
        t.render(&chat).unwrap(),
        "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜><think>"
    );
    let ids = t.encode_chat(&chat).unwrap();
    assert_eq!(ids[0], 0);
    assert_eq!(&ids[ids.len() - 2..], [128804, 128821]);
    assert!(t.stop_tokens().contains(&1) && t.stop_tokens().contains(&128803));
}
