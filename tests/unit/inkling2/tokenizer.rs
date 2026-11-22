use minijinja::Environment;
use serde_json::json;

use super::patch_chat_template;

/// Minimal extract of Inkling 2 Mini's GGUF `tokenizer.chat_template`
/// covering `emit_thinking_effort()` and `{% generation %}` stripping.
const INKLING2_TEMPLATE: &str = concat!(
    r#"{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}"#,
    r#"{%- macro emit_thinking_effort() -%}"#,
    r#"{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}"#,
    r#"{%- if eff is string -%}{%- set num = effort_map[eff | trim] -%}{%- else -%}{%- set num = eff | float -%}{%- endif -%}"#,
    r#"{{- "<|message_system|><|content_text|>Thinking effort level: " -}}"#,
    r#"{%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}"#,
    r#"{{- "<|end_message|>" -}}"#,
    r#"{%- endmacro -%}"#,
    r#"{{- emit_thinking_effort() -}}"#,
    r#"{%- for m in messages -%}"#,
    r#"{%- if m.role == "assistant" -%}{%- generation -%}{{ m.content }}{%- endgeneration -%}"#,
    r#"{%- else -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"#,
    r#"{%- endfor -%}"#,
);

fn render(
    template: &str,
    add_generation_prompt: bool,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> String {
    let mut env = Environment::new();
    env.add_template("chat", template).unwrap();
    let mut ctx = serde_json::Map::new();
    ctx.insert("messages".into(), json!([{"role": "user", "content": "hi"}]));
    ctx.insert("add_generation_prompt".into(), json!(add_generation_prompt));
    ctx.insert("enable_thinking".into(), json!(enable_thinking));
    if let Some(effort) = reasoning_effort {
        ctx.insert("reasoning_effort".into(), json!(effort));
    }
    env.get_template("chat").unwrap().render(ctx).unwrap()
}

#[test]
fn enable_thinking_false_sets_effort_zero_and_opens_text() {
    let patched = patch_chat_template(INKLING2_TEMPLATE).unwrap();
    assert_eq!(
        render(&patched, true, false, None),
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>\
         <|message_model|><|content_text|>"
    );
    assert_eq!(
        render(&patched, true, false, Some("low")),
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>\
         <|message_model|><|content_text|>"
    );
    assert_eq!(
        render(&patched, false, false, None),
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>"
    );
    assert_eq!(
        render(&patched, true, true, None),
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>"
    );
    assert_eq!(
        render(&patched, true, true, Some("low")),
        "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>"
    );
    assert_eq!(
        render(&patched, true, true, Some("xhigh")),
        "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>"
    );
}

#[test]
fn patch_passes_through_existing_enable_thinking_and_rejects_unknown_shape() {
    let already_wired = "{%- if enable_thinking -%}on{%- else -%}off{%- endif -%}";
    assert_eq!(patch_chat_template(already_wired).unwrap(), already_wired);
    assert!(patch_chat_template("{{ messages }}").is_err());
}
