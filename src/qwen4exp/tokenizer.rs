//! Qwen3.8-Flash-Next's tokenizer and chat template from a GGUF's metadata:
//! the byte-level BPE vocabulary with the `qwen35` pre-tokenizer (NFC, then
//! the split regex of the model's own `tokenizer.json`: Qwen3.6's), and the
//! template as the file carries it.

use anyhow::Result;
use serde_json::json;

use crate::gguf::Gguf;

/// The `tokenizer.json` of the file's vocabulary.
pub fn tokenizer_json(g: &Gguf) -> Result<String> {
    let vocab = g.u64(&format!("{}.vocab_size", super::gguf::ARCH))? as usize;
    crate::glm53::tokenizer::bpe_json_normalized(
        g,
        "qwen35",
        &[crate::qwen36::tokenizer::QWEN35_SPLIT],
        vocab,
        json!({"type": "NFC"}),
    )
}

/// The file's chat template.
pub fn chat_template(g: &Gguf) -> Result<String> {
    Ok(g.str("tokenizer.chat_template")?.to_string())
}
