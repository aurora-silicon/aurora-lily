//! Gemma 4's tokenizer and chat template from the GGUF metadata: the
//! `tokenizers` JSON lily's [`crate::tokenizer::Tokenizer`] loads, built in
//! memory from `tokenizer.ggml.*` in the shape of the model's own
//! `tokenizer.json` (BPE over raw UTF-8 with spaces as U+2581, byte
//! fallback, no pre-split; llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GEMMA4` splits
//! at newlines only, which gives the same ids), and the template from
//! `tokenizer.chat_template`.

use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

use crate::gguf::{Gguf, Value as GValue};

/// GGUF token types.
const TYPE_CONTROL: u64 = 3;
const TYPE_USER_DEFINED: u64 = 4;

fn strings(g: &Gguf, key: &str) -> Result<Vec<String>> {
    match g.get(key) {
        Some(GValue::Array(v)) => v
            .iter()
            .map(|s| s.as_str().map(str::to_string).context("non-string entry"))
            .collect::<Result<_>>()
            .with_context(|| format!("reading {key}")),
        _ => anyhow::bail!("GGUF metadata {key} missing"),
    }
}

/// The `tokenizer.json` of the file's vocabulary. Control and user-defined
/// tokens (the template's markup: `<|turn>`, `<|channel>`, `<|tool_call>`,
/// `<|"|>`, ...) and the BOS/EOS/pad/unk ids are special added tokens, as
/// in the model's own tokenizer (whose `<eos>` the GGUF types as normal).
pub fn tokenizer_json(g: &Gguf) -> Result<String> {
    ensure!(
        g.str("tokenizer.ggml.model")? == "gemma4",
        "Gemma tokenizer model is not gemma4"
    );
    let tokens = strings(g, "tokenizer.ggml.tokens")?;
    let merges = strings(g, "tokenizer.ggml.merges")?;
    let types: Vec<u64> = match g.get("tokenizer.ggml.token_type") {
        Some(GValue::Array(v)) => v.iter().map(|t| t.as_u64().unwrap_or(1)).collect(),
        _ => vec![1; tokens.len()],
    };
    let named: Vec<u64> = ["bos", "eos", "padding", "unknown"]
        .iter()
        .filter_map(|k| g.u64(&format!("tokenizer.ggml.{k}_token_id")).ok())
        .chain([1])
        .collect();
    let mut vocab = serde_json::Map::with_capacity(tokens.len());
    let mut added = Vec::new();
    for (id, tok) in tokens.iter().enumerate() {
        vocab.insert(tok.clone(), json!(id));
        let ty = types.get(id).copied().unwrap_or(1);
        if ty == TYPE_CONTROL || ty == TYPE_USER_DEFINED || named.contains(&(id as u64))
        {
            added.push(json!({
                "id": id,
                "content": tok,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            }));
        }
    }
    let doc = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": Value::Object(vocab),
            "merges": merges,
        },
    });
    Ok(doc.to_string())
}

/// The file's chat template, as written.
pub fn chat_template(g: &Gguf) -> Result<String> {
    let t = g.str("tokenizer.chat_template")?;
    ensure!(
        t.contains("<|turn>model\\n") || t.contains("<|turn>model\n"),
        "Gemma chat template lacks its generation prompt"
    );
    Ok(t.to_string())
}

/// Turn-ending ids: `<turn|>` (the GGUF's EOS), `<eos>` and
/// `<|tool_response>` (the model hands a tool call to the caller there),
/// as the model's generation config lists them.
pub fn stop_ids(g: &Gguf) -> Result<Vec<u32>> {
    let tokens = strings(g, "tokenizer.ggml.tokens")?;
    let eos = g.u64("tokenizer.ggml.eos_token_id")? as usize;
    ensure!(eos < tokens.len(), "eos token {eos} outside the vocabulary");
    let mut out = vec![eos as u32];
    for t in ["<eos>", "<|tool_response>"] {
        let id = tokens
            .iter()
            .position(|x| x == t)
            .with_context(|| format!("no {t} token"))?;
        if !out.contains(&(id as u32)) {
            out.push(id as u32);
        }
    }
    Ok(out)
}
