//! Qwen3.6's tokenizer and chat template from the GGUF metadata: byte-level
//! BPE with the `qwen35` pre-tokenizer split and NFC normalization (Qwen's
//! `tokenizer.json`, as llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35` has it),
//! and the template from `tokenizer.chat_template` as it is (Qwen's: the
//! `enable_thinking` switch, `<tool_call>` XML).

use anyhow::{Result, ensure};
use serde_json::json;

use crate::gguf::Gguf;

/// The `qwen35` pre-tokenizer split (the regex of Qwen3.5's
/// `tokenizer.json`).
const QWEN35_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The `tokenizer.json` of the file's vocabulary.
pub fn tokenizer_json(g: &Gguf) -> Result<String> {
    let vocab = super::config::Config::from_gguf(g)?.vocab;
    crate::glm53::tokenizer::bpe_json_normalized(
        g,
        "qwen35",
        &[QWEN35_SPLIT],
        vocab,
        json!({"type": "NFC"}),
    )
}

/// The file's template without transformers' `{% generation %}` /
/// `{% endgeneration %}` tags (they mark the assistant's tokens for
/// training masks and render nothing; minijinja does not know them).
pub fn chat_template(g: &Gguf) -> Result<String> {
    let mut t = g.str("tokenizer.chat_template")?.to_string();
    for tag in ["generation", "endgeneration"] {
        for (open, close) in
            [("{%-", "-%}"), ("{%-", "%}"), ("{%", "-%}"), ("{%", "%}")]
        {
            t = t.replace(&format!("{open} {tag} {close}"), "");
        }
    }
    ensure!(!t.contains("generation %}"), "unhandled generation tag in the template");
    Ok(t)
}

/// Turn-ending ids: `<|im_end|>` (the file's eos) and `<|endoftext|>`.
pub fn stop_ids(g: &Gguf) -> Result<Vec<u32>> {
    let c = super::config::Config::from_gguf(g)?;
    Ok(vec![c.eos, c.pad])
}
