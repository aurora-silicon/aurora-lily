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

/// The file's chat template without `{% generation %}` / `{% endgeneration %}`
/// tags and with `enable_thinking: false` mapped to thinking effort `0` plus an
/// open `<|message_model|><|content_text|>` generation prompt.
///
/// Inkling 2's GGUF template reads only `reasoning_effort` (defaulting to
/// `0.9` when omitted) and ends the prompt without opening a model content
/// block, so `--thinking false`, `enable_thinking: false`, and
/// `reasoning_effort: "none"` would otherwise still emit a
/// `<|content_thinking|>` block.
pub fn chat_template(g: &Gguf) -> Result<String> {
    patch_chat_template(g.str("tokenizer.chat_template")?)
}

fn patch_chat_template(raw: &str) -> Result<String> {
    let mut t = raw.to_string();
    for tag in ["generation", "endgeneration"] {
        for (open, close) in
            [("{%-", "-%}"), ("{%-", "%}"), ("{%", "-%}"), ("{%", "%}")]
        {
            t = t.replace(&format!("{open} {tag} {close}"), "");
        }
    }
    ensure!(!t.contains("generation %}"), "unhandled generation tag in the template");
    if !t.contains("enable_thinking") {
        const EFFORT_FALLBACK: &str = "{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}";
        const EFFORT_WITH_THINKING: &str = "{%- set eff = 0.0 if enable_thinking is defined and not enable_thinking else (reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9) -%}";
        const NOTHINK_GEN_PROMPT: &str = "{%- if add_generation_prompt is defined and add_generation_prompt and enable_thinking is defined and not enable_thinking -%}{{- \"<|message_model|><|content_text|>\" -}}{%- endif -%}";
        ensure!(
            t.contains(EFFORT_FALLBACK),
            "Inkling 2 chat template lacks its reasoning_effort fallback"
        );
        t = t.replace(EFFORT_FALLBACK, EFFORT_WITH_THINKING);
        t.push_str(NOTHINK_GEN_PROMPT);
    }
    Ok(t)
}

/// Turn-ending stop token ids: `<|content_model_end_sampling|>` (`eos`) and padding.
pub fn stop_ids(g: &Gguf) -> Result<Vec<u32>> {
    let c = super::config::Config::from_gguf(g)?;
    Ok(vec![c.eos, c.pad])
}

#[cfg(test)]
#[path = "../../tests/unit/inkling2/tokenizer.rs"]
mod tests;
