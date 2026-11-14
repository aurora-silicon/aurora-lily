//! DeepSeek V4 Flash's tokenizer and chat template from the GGUF metadata:
//! byte-level BPE with the `joyai-llm` pre-tokenizer (DeepSeek V3's split
//! rules, which ds4's `bpe_tokenize_text` mirrors: digits in runs of up to
//! three, CJK and kana runs, then words, punctuation runs with their
//! trailing newlines, and whitespace), and the template from
//! `tokenizer.chat_template` (thinking switched by `enable_thinking`, DSML
//! tool calls).

use anyhow::{Result, ensure};

use crate::gguf::Gguf;

/// The pre-tokenizer's splits, applied in order (DeepSeek V3's
/// `tokenizer.json`).
const SPLITS: [&str; 3] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

/// The beginning-of-sentence token the template opens with.
pub const BOS: &str = "<｜begin▁of▁sentence｜>";

/// The `tokenizer.json` of the file's vocabulary.
pub fn tokenizer_json(g: &Gguf) -> Result<String> {
    let vocab = g.u64(&format!("{}.vocab_size", super::config::ARCH))? as usize;
    crate::glm53::tokenizer::bpe_json(g, "joyai-llm", &SPLITS, vocab)
}

/// The file's chat template, its `bos_token` spelled out (lily renders
/// without one).
pub fn chat_template(g: &Gguf) -> Result<String> {
    let t = g.str("tokenizer.chat_template")?;
    let open = "{{- bos_token -}}";
    ensure!(t.contains(open), "DeepSeek chat template lacks its bos_token");
    ensure!(
        t.contains("enable_thinking"),
        "DeepSeek chat template has no thinking switch"
    );
    Ok(t.replace(open, &format!("{{{{- '{BOS}' -}}}}")))
}

/// Turn-ending ids: `<｜end▁of▁sentence｜>` (the file's eos) and
/// `<｜User｜>`.
pub fn stop_ids(g: &Gguf) -> Result<Vec<u32>> {
    let vocab = g.u64(&format!("{}.vocab_size", super::config::ARCH))?;
    let eos = g.u64("tokenizer.ggml.eos_token_id")?;
    ensure!(eos < vocab, "eos token {eos} is outside the vocabulary of {vocab}");
    let mut ids = vec![eos as u32];
    if let Some(crate::gguf::Value::Array(toks)) = g.get("tokenizer.ggml.tokens")
        && let Some(user) = toks.iter().position(|t| t.as_str() == Some("<｜User｜>"))
    {
        ids.push(user as u32);
    }
    Ok(ids)
}

#[cfg(test)]
#[path = "../../tests/unit/deepseek4/tokenizer.rs"]
mod tests;
