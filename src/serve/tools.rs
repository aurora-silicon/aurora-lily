//! Tool definitions and the model's `<tool_call>` XML (GLM's, DeepSeek's and
//! Gemma 4's call syntaxes further down).
//!
//! Qwen3.8 emits calls as
//!
//! ```text
//! <tool_call>
//! <function=name>
//! <parameter=key>
//! value
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Parameter values are plain text; the tool's JSON schema decides whether a
//! value is kept as a string or parsed as JSON (numbers, booleans, objects,
//! arrays), the way the reference parser in Qwen's own server does.

use anyhow::{Context as _, Result, bail, ensure};
use serde_json::{Map, Value};

/// One function the request offered, reduced to what parsing needs.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    /// `parameters.properties`, if the schema has any.
    pub properties: Map<String, Value>,
}

impl ToolSchema {
    /// Reads OpenAI `{"type": "function", "function": {...}}` tool entries.
    pub fn from_request(tools: &[Value]) -> Result<Vec<Self>> {
        tools
            .iter()
            .map(|tool| {
                let function = tool
                    .get("function")
                    .filter(|f| f.is_object())
                    .context("each tool needs a `function` object")?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .context("each tool function needs a `name`")?
                    .to_string();
                let properties = function
                    .get("parameters")
                    .and_then(|p| p.get("properties"))
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                Ok(Self { name, properties })
            })
            .collect()
    }

    fn param_is_string(&self, key: &str) -> bool {
        let Some(schema) = self.properties.get(key) else {
            // Unknown parameter: keep the text.
            return true;
        };
        match schema.get("type") {
            Some(Value::String(t)) => t == "string",
            Some(Value::Array(ts)) => {
                // Nullable strings are common: ["string", "null"].
                ts.iter().any(|t| t == "string") && ts.len() <= 2
            }
            // No type: enum-of-strings or anything else; text is the safe read.
            _ => schema.get("enum").is_some() || schema.get("anyOf").is_none(),
        }
    }
}

/// A parsed call: the function name and its arguments as a JSON object text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

/// The tool-call syntax a model writes between `<tool_call>` and
/// `</tool_call>`, chosen by its tokenizer
/// ([`crate::tokenizer::Tokenizer::tool_grammar`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolGrammar {
    /// Qwen3.8's `<function=name><parameter=key>...` ([`parse_tool_call`]).
    #[default]
    Qwen,
    /// GLM's `name<arg_key>k</arg_key><arg_value>v</arg_value>...`
    /// ([`parse_glm_tool_call`]).
    Glm,
    /// DeepSeek V4's DSML: a `<｜DSML｜tool_calls>` block of `invoke`s
    /// ([`parse_dsml_tool_calls`]).
    Dsml,
    /// Gemma 4's `call:name{key:value,...}` between `<|tool_call>` and
    /// `<tool_call|>`, strings quoted by `<|"|>` ([`parse_gemma_tool_call`]),
    /// its reasoning in `<|channel>thought\n...<channel|>`.
    Gemma,
}

impl ToolGrammar {
    /// The text that opens and closes a tool-call block.
    pub fn markers(self) -> (&'static str, &'static str) {
        match self {
            Self::Qwen | Self::Glm => ("<tool_call>", "</tool_call>"),
            Self::Dsml => ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"),
            Self::Gemma => ("<|tool_call>", "<tool_call|>"),
        }
    }

    /// The marker that ends the reasoning block.
    pub fn think_end(self) -> &'static str {
        match self {
            Self::Gemma => "<channel|>",
            _ => "</think>",
        }
    }

    /// What the model writes to open its reasoning when the prompt does not
    /// (Gemma's generation prompt with thinking on ends at the turn header).
    pub fn think_open(self) -> Option<&'static str> {
        match self {
            Self::Gemma => Some("<|channel>thought"),
            _ => None,
        }
    }
}

/// Parses a tool-call block in `grammar` into its calls (one, except
/// DSML's).
pub fn parse_tool_calls_as(
    grammar: ToolGrammar,
    block: &str,
    tools: &[ToolSchema],
) -> Result<Vec<ParsedToolCall>> {
    match grammar {
        ToolGrammar::Dsml => parse_dsml_tool_calls(block, tools),
        _ => Ok(vec![parse_tool_call_as(grammar, block, tools)?]),
    }
}

/// Parses a tool-call block in `grammar`.
pub fn parse_tool_call_as(
    grammar: ToolGrammar,
    block: &str,
    tools: &[ToolSchema],
) -> Result<ParsedToolCall> {
    match grammar {
        ToolGrammar::Qwen => parse_tool_call(block, tools),
        ToolGrammar::Glm => parse_glm_tool_call(block, tools),
        ToolGrammar::Gemma => parse_gemma_tool_call(block, tools),
        ToolGrammar::Dsml => {
            let mut calls = parse_dsml_tool_calls(block, tools)?;
            ensure!(
                calls.len() == 1,
                "{} DSML calls where one was expected",
                calls.len()
            );
            Ok(calls.remove(0))
        }
    }
}

/// Parses the text between `<tool_call>` and `</tool_call>`.
pub fn parse_tool_call(block: &str, tools: &[ToolSchema]) -> Result<ParsedToolCall> {
    let block = block.trim();
    let rest = block
        .strip_prefix("<function=")
        .context("tool call does not start with <function=")?;
    let (name, rest) = rest.split_once('>').context("unterminated <function= tag")?;
    let name = name.trim();
    let body = rest
        .strip_suffix("</function>")
        .or_else(|| rest.trim_end().strip_suffix("</function>"))
        .context("tool call does not end with </function>")?;
    let schema = tools.iter().find(|t| t.name == name);
    let mut arguments = Map::new();
    let mut cursor = body;
    while let Some(start) = cursor.find("<parameter=") {
        let after = &cursor[start + "<parameter=".len()..];
        let (key, after) =
            after.split_once('>').context("unterminated <parameter= tag")?;
        let end = after.find("</parameter>").context("unterminated parameter")?;
        let raw = &after[..end];
        // The template writes one newline around the value; strip exactly that.
        let value = raw.strip_prefix('\n').unwrap_or(raw);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let typed = match schema {
            Some(s) if s.param_is_string(key) => Value::String(value.to_string()),
            Some(_) => serde_json::from_str(value)
                .unwrap_or_else(|_| Value::String(value.to_string())),
            None => Value::String(value.to_string()),
        };
        arguments.insert(key.trim().to_string(), typed);
        cursor = &after[end + "</parameter>".len()..];
    }
    if cursor.trim().contains("<parameter") {
        bail!("malformed parameter block");
    }
    Ok(ParsedToolCall {
        name: name.to_string(),
        arguments: serde_json::to_string(&Value::Object(arguments))?,
    })
}

/// Gemma 4's string quote in tool calls.
const GEMMA_QUOTE: &str = "<|\"|>";

/// One value of Gemma's call syntax (the chat template's `format_argument`
/// with unquoted keys): a `<|"|>`-quoted string, `{key:value,...}`,
/// `[value,...]`, or a bare word (number, `true`, `false`, `null`; anything
/// else is kept as a string). Returns the value and the rest.
fn gemma_value(s: &str) -> Result<(Value, &str)> {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix(GEMMA_QUOTE) {
        let end = rest.find(GEMMA_QUOTE).context("unterminated Gemma string")?;
        return Ok((Value::String(rest[..end].to_string()), &rest[end + GEMMA_QUOTE.len()..]));
    }
    if let Some(rest) = s.strip_prefix('{') {
        let (map, rest) = gemma_object(rest)?;
        return Ok((Value::Object(map), rest));
    }
    if let Some(mut rest) = s.strip_prefix('[') {
        let mut items = Vec::new();
        loop {
            rest = rest.trim_start();
            if let Some(r) = rest.strip_prefix(']') {
                return Ok((Value::Array(items), r));
            }
            let (v, r) = gemma_value(rest)?;
            items.push(v);
            rest = r.trim_start();
            rest = rest.strip_prefix(',').unwrap_or(rest);
        }
    }
    let end = s.find([',', '}', ']']).unwrap_or(s.len());
    let word = s[..end].trim();
    let v = match word {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        _ => serde_json::from_str::<serde_json::Number>(word)
            .map(Value::Number)
            .unwrap_or_else(|_| Value::String(word.to_string())),
    };
    Ok((v, &s[end..]))
}

/// The pairs of an object after its `{`, up to and including its `}`.
fn gemma_object(mut s: &str) -> Result<(Map<String, Value>, &str)> {
    let mut map = Map::new();
    loop {
        s = s.trim_start();
        if let Some(r) = s.strip_prefix('}') {
            return Ok((map, r));
        }
        let colon = s.find(':').context("Gemma object key without a value")?;
        let key = s[..colon].trim();
        let key = key
            .strip_prefix(GEMMA_QUOTE)
            .and_then(|k| k.strip_suffix(GEMMA_QUOTE))
            .unwrap_or(key);
        let (v, r) = gemma_value(&s[colon + 1..])?;
        map.insert(key.to_string(), v);
        s = r.trim_start();
        s = s.strip_prefix(',').unwrap_or(s);
        anyhow::ensure!(!s.is_empty(), "unterminated Gemma object");
    }
}

/// Gemma 4's call: `call:name{key:value,...}`. Values keep their own types
/// (the syntax quotes strings), except that a string where the tool's
/// schema wants another type is read as JSON when it parses.
pub fn parse_gemma_tool_call(
    block: &str,
    tools: &[ToolSchema],
) -> Result<ParsedToolCall> {
    let block = block.trim();
    let rest = block.strip_prefix("call:").context("Gemma tool call does not start with call:")?;
    let brace = rest.find('{').context("Gemma tool call without arguments")?;
    let name = rest[..brace].trim();
    let is_name_char =
        |c: char| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':');
    if name.is_empty() || !name.chars().all(is_name_char) {
        bail!("Gemma tool call has no function name");
    }
    let (mut arguments, tail) = gemma_object(&rest[brace + 1..])?;
    if !tail.trim().is_empty() {
        bail!("text after a Gemma tool call");
    }
    if let Some(schema) = tools.iter().find(|t| t.name == name) {
        for (key, value) in arguments.iter_mut() {
            if let Value::String(text) = value
                && !schema.param_is_string(key)
                && let Ok(v) = serde_json::from_str::<Value>(text)
            {
                *value = v;
            }
        }
    }
    Ok(ParsedToolCall {
        name: name.to_string(),
        arguments: serde_json::to_string(&Value::Object(arguments))?,
    })
}

/// GLM's call: the function name, then key/value pairs (GLM's template
/// writes string values as they are and other values as JSON, which the
/// schema typing reads back). A call without arguments is the bare name.
/// The name must be one token of identifier characters (letters, digits,
/// `_`, `-`, `.`, `:`), so a block of prose is not a call.
pub fn parse_glm_tool_call(
    block: &str,
    tools: &[ToolSchema],
) -> Result<ParsedToolCall> {
    let block = block.trim();
    let (name, mut cursor) = match block.find("<arg_key>") {
        Some(at) => (block[..at].trim(), &block[at..]),
        None => (block, ""),
    };
    let is_name_char =
        |c: char| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':');
    if name.is_empty() || !name.chars().all(is_name_char) {
        bail!("GLM tool call does not start with a function name");
    }
    let schema = tools.iter().find(|t| t.name == name);
    let mut arguments = Map::new();
    while let Some(rest) = cursor.trim_start().strip_prefix("<arg_key>") {
        let (key, rest) =
            rest.split_once("</arg_key>").context("unterminated <arg_key>")?;
        let rest = rest
            .trim_start()
            .strip_prefix("<arg_value>")
            .context("<arg_key> without an <arg_value>")?;
        let (value, rest) =
            rest.split_once("</arg_value>").context("unterminated <arg_value>")?;
        let key = key.trim();
        let typed = match schema {
            Some(s) if s.param_is_string(key) => Value::String(value.to_string()),
            Some(_) => serde_json::from_str(value)
                .unwrap_or_else(|_| Value::String(value.to_string())),
            None => Value::String(value.to_string()),
        };
        arguments.insert(key.to_string(), typed);
        cursor = rest;
    }
    if !cursor.trim().is_empty() {
        bail!("malformed GLM tool call arguments");
    }
    Ok(ParsedToolCall {
        name: name.to_string(),
        arguments: serde_json::to_string(&Value::Object(arguments))?,
    })
}

/// Parses the inside of a DSML `tool_calls` block (DeepSeek V4's chat
/// template writes it): `<｜DSML｜invoke name="f">` elements, each of
/// `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`
/// (a JSON value unless `string="true"`), separated by newlines.
pub fn parse_dsml_tool_calls(
    block: &str,
    tools: &[ToolSchema],
) -> Result<Vec<ParsedToolCall>> {
    const INVOKE: &str = "<｜DSML｜invoke name=\"";
    const INVOKE_END: &str = "</｜DSML｜invoke>";
    const PARAM: &str = "<｜DSML｜parameter name=\"";
    const PARAM_END: &str = "</｜DSML｜parameter>";
    let mut calls = Vec::new();
    let mut cursor = block.trim();
    while !cursor.is_empty() {
        let rest =
            cursor.strip_prefix(INVOKE).context("DSML block without an invoke")?;
        let (name, rest) =
            rest.split_once("\">").context("unterminated invoke name")?;
        let (body, rest) =
            rest.split_once(INVOKE_END).context("unterminated invoke")?;
        let schema = tools.iter().find(|t| t.name == name);
        let mut arguments = Map::new();
        let mut params = body.trim();
        while !params.is_empty() {
            let p = params.strip_prefix(PARAM).context("malformed DSML parameter")?;
            let (key, p) = p.split_once('"').context("unterminated parameter name")?;
            let p = p.trim_start();
            let (is_string, p) = if let Some(p) = p.strip_prefix("string=\"true\">") {
                (true, p)
            } else if let Some(p) = p.strip_prefix("string=\"false\">") {
                (false, p)
            } else {
                bail!("DSML parameter without string=\"true|false\"");
            };
            let (value, p) =
                p.split_once(PARAM_END).context("unterminated DSML parameter")?;
            let typed = if is_string || schema.is_some_and(|s| s.param_is_string(key)) {
                Value::String(value.to_string())
            } else {
                serde_json::from_str(value)
                    .unwrap_or_else(|_| Value::String(value.to_string()))
            };
            arguments.insert(key.to_string(), typed);
            params = p.trim_start();
        }
        calls.push(ParsedToolCall {
            name: name.to_string(),
            arguments: serde_json::to_string(&Value::Object(arguments))?,
        });
        cursor = rest.trim_start();
    }
    ensure!(!calls.is_empty(), "empty DSML tool-call block");
    Ok(calls)
}

/// Parses OpenAI-style `tool_calls` on an incoming assistant message into
/// the shape the chat template wants (`arguments` as an object).
pub fn template_tool_calls(calls: &[Value]) -> Result<Vec<Value>> {
    calls
        .iter()
        .map(|call| {
            let function = call
                .get("function")
                .filter(|f| f.is_object())
                .context("assistant tool_calls entries need a `function` object")?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .context("tool call function needs a `name`")?;
            let arguments = match function.get("arguments") {
                None | Some(Value::Null) => Value::Object(Map::new()),
                Some(Value::String(text)) => {
                    if text.trim().is_empty() {
                        Value::Object(Map::new())
                    } else {
                        serde_json::from_str::<Value>(text).with_context(|| {
                            format!("tool call {name}: arguments are not valid JSON")
                        })?
                    }
                }
                Some(other) => other.clone(),
            };
            anyhow::ensure!(
                arguments.is_object(),
                "tool call {name}: arguments must be a JSON object"
            );
            Ok(serde_json::json!({
                "id": call.get("id").cloned().unwrap_or(Value::Null),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }))
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/serve/tools.rs"]
mod tests;
