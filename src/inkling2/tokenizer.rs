//! Inkling 2 Mini tokenizer and chat template from GGUF metadata:
//! byte-level BPE with GPT-4o (`o200k` / `inkling2`) pre-tokenizer split
//! (`LLAMA_VOCAB_PRE_TYPE_GPT4O`), and the template from
//! `tokenizer.chat_template`.

use anyhow::{Result, ensure};
use serde_json::Value;

use crate::gguf::Gguf;

/// GPT-4o (`o200k_base`) pre-tokenizer split regex used by `tokenizer.ggml.pre = "inkling2"`
/// (matches `llama.cpp`'s `LLAMA_VOCAB_PRE_TYPE_GPT4O`).
pub(crate) const GPT4O_SPLIT: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Builds the in-memory `tokenizer.json` for `crate::tokenizer::Tokenizer`.
pub fn tokenizer_json(g: &Gguf) -> Result<String> {
    let cfg = super::config::Config::from_gguf(g)?;
    crate::glm53::tokenizer::bpe_json_normalized(
        g,
        "inkling2",
        &[GPT4O_SPLIT],
        cfg.vocab_unpadded,
        Value::Null,
    )
}

/// The file's chat template without `{% generation %}` / `{% endgeneration %}` tags.
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

/// Turn-ending stop token ids: `<|content_model_end_sampling|>` (`eos`) and padding.
pub fn stop_ids(g: &Gguf) -> Result<Vec<u32>> {
    let c = super::config::Config::from_gguf(g)?;
    Ok(vec![c.eos, c.pad])
}
