//! JSON/YAML formatting for prompts and parsing of model output.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{json, Map, Value};
use thiserror::Error;

use crate::types::{AttributeValue, Extraction, FormatType, ATTRIBUTE_SUFFIX, EXTRACTIONS_KEY};

static FENCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"```(?<lang>[A-Za-z0-9_+-]+)?(?:\s*\n)?(?<body>[\s\S]*?)```")
        .expect("fence regex")
});
static THINK_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<think>[\s\S]*?</think>\s*").expect("think regex"));

/// Format handling errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FormatError {
    #[error("{0}")]
    Parse(String),
}

/// Handles all format-specific logic for prompts and parsing.
#[derive(Debug, Clone)]
pub struct FormatHandler {
    pub format_type: FormatType,
    pub use_wrapper: bool,
    pub wrapper_key: Option<String>,
    pub use_fences: bool,
    pub attribute_suffix: String,
    pub strict_fences: bool,
    pub allow_top_level_list: bool,
}

impl Default for FormatHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatHandler {
    pub fn new() -> Self {
        Self {
            format_type: FormatType::Json,
            use_wrapper: true,
            wrapper_key: Some(EXTRACTIONS_KEY.to_string()),
            use_fences: true,
            attribute_suffix: ATTRIBUTE_SUFFIX.to_string(),
            strict_fences: false,
            allow_top_level_list: true,
        }
    }

    pub fn with_format_type(mut self, format_type: FormatType) -> Self {
        self.format_type = format_type;
        self
    }

    pub fn with_wrapper(mut self, use_wrapper: bool, wrapper_key: Option<&str>) -> Self {
        self.use_wrapper = use_wrapper;
        self.wrapper_key = if use_wrapper {
            Some(
                wrapper_key
                    .unwrap_or(EXTRACTIONS_KEY)
                    .to_string(),
            )
        } else {
            None
        };
        self
    }

    pub fn with_fences(mut self, use_fences: bool) -> Self {
        self.use_fences = use_fences;
        self
    }

    pub fn with_strict_fences(mut self, strict_fences: bool) -> Self {
        self.strict_fences = strict_fences;
        self
    }

    pub fn with_attribute_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.attribute_suffix = suffix.into();
        self
    }

    /// Create from legacy resolver kwargs (`fence_output`, `format_type`, …).
    pub fn from_kwargs(
        fence_output: Option<bool>,
        format_type: Option<FormatType>,
        strict_fences: Option<bool>,
        require_extractions_key: Option<bool>,
        attribute_suffix: Option<String>,
    ) -> Self {
        let use_wrapper = require_extractions_key.unwrap_or(true);
        Self {
            format_type: format_type.unwrap_or(FormatType::Json),
            use_wrapper,
            wrapper_key: if use_wrapper {
                Some(EXTRACTIONS_KEY.to_string())
            } else {
                None
            },
            use_fences: fence_output.unwrap_or(true),
            attribute_suffix: attribute_suffix.unwrap_or_else(|| ATTRIBUTE_SUFFIX.to_string()),
            strict_fences: strict_fences.unwrap_or(false),
            allow_top_level_list: true,
        }
    }

    pub fn format_extraction_example(&self, extractions: &[Extraction]) -> String {
        let items: Vec<Value> = extractions
            .iter()
            .map(|ext| {
                let mut obj = Map::new();
                obj.insert(ext.extraction_class.clone(), Value::String(ext.extraction_text.clone()));
                let attrs = match &ext.attributes {
                    Some(map) => attributes_to_value(map),
                    None => json!({}),
                };
                obj.insert(
                    format!("{}{}", ext.extraction_class, self.attribute_suffix),
                    attrs,
                );
                Value::Object(obj)
            })
            .collect();

        let payload = if self.use_wrapper {
            let key = self
                .wrapper_key
                .clone()
                .unwrap_or_else(|| EXTRACTIONS_KEY.to_string());
            json!({ key: items })
        } else {
            Value::Array(items)
        };

        let formatted = match self.format_type {
            FormatType::Yaml => to_yaml(&payload),
            FormatType::Json => serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string()),
        };
        if self.use_fences {
            self.add_fences(&formatted)
        } else {
            formatted
        }
    }

    pub fn parse_output(&self, text: &str, strict: bool) -> Result<Vec<Map<String, Value>>, FormatError> {
        if text.is_empty() {
            return Err(FormatError::Parse(
                "Empty or invalid input string.".to_string(),
            ));
        }
        let content = self.extract_content(text)?;
        let parsed = self.parse_with_fallback(&content, strict)?;
        if parsed.is_null() {
            return if self.use_wrapper {
                Err(FormatError::Parse(format!(
                    "Content must be a mapping with an '{}' key.",
                    self.wrapper_key.as_deref().unwrap_or(EXTRACTIONS_KEY)
                )))
            } else {
                Err(FormatError::Parse(
                    "Content must be a list of extractions or a dict.".to_string(),
                ))
            };
        }

        let require_wrapper = self.wrapper_key.is_some() && (self.use_wrapper || strict);

        let items = if let Some(obj) = parsed.as_object() {
            if require_wrapper {
                let key = self.wrapper_key.as_deref().unwrap_or(EXTRACTIONS_KEY);
                obj.get(key).cloned().ok_or_else(|| {
                    FormatError::Parse(format!("Content must contain an '{key}' key."))
                })?
            } else if obj.contains_key(EXTRACTIONS_KEY) {
                obj[EXTRACTIONS_KEY].clone()
            } else if let Some(key) = &self.wrapper_key {
                if obj.contains_key(key) {
                    obj[key].clone()
                } else {
                    Value::Array(vec![parsed.clone()])
                }
            } else {
                Value::Array(vec![parsed.clone()])
            }
        } else if parsed.is_array() {
            if require_wrapper && (strict || !self.allow_top_level_list) {
                return Err(FormatError::Parse(format!(
                    "Content must be a mapping with an '{}' key.",
                    self.wrapper_key.as_deref().unwrap_or(EXTRACTIONS_KEY)
                )));
            }
            if strict && self.use_wrapper {
                return Err(FormatError::Parse(
                    "Strict mode requires a wrapper object.".to_string(),
                ));
            }
            if !self.allow_top_level_list {
                return Err(FormatError::Parse(
                    "Top-level list is not allowed.".to_string(),
                ));
            }
            parsed
        } else {
            return Err(FormatError::Parse(format!(
                "Expected list or dict, got {}",
                type_name(&parsed)
            )));
        };

        let Some(arr) = items.as_array() else {
            return Err(FormatError::Parse(
                "The extractions must be a sequence (list) of mappings.".to_string(),
            ));
        };

        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let Some(map) = item.as_object() else {
                return Err(FormatError::Parse(
                    "Each item in the sequence must be a mapping.".to_string(),
                ));
            };
            out.push(map.clone());
        }
        Ok(out)
    }

    fn add_fences(&self, content: &str) -> String {
        format!("```{}\n{}\n```", self.format_type.as_str(), content.trim())
    }

    fn parse_with_fallback(&self, content: &str, strict: bool) -> Result<Value, FormatError> {
        match self.decode(content) {
            Ok(v) => Ok(v),
            Err(e) => {
                if strict {
                    return Err(e);
                }
                if THINK_TAG_RE.is_match(content) {
                    let stripped = THINK_TAG_RE.replace_all(content, "").trim().to_string();
                    self.decode(&stripped)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn decode(&self, content: &str) -> Result<Value, FormatError> {
        match self.format_type {
            FormatType::Yaml => serde_yaml::from_str::<Value>(content).map_err(|e| {
                FormatError::Parse(format!(
                    "Failed to parse YAML content: {}",
                    truncate(&e.to_string(), 200)
                ))
            }),
            FormatType::Json => serde_json::from_str::<Value>(content).map_err(|e| {
                FormatError::Parse(format!(
                    "Failed to parse JSON content: {}",
                    truncate(&e.to_string(), 200)
                ))
            }),
        }
    }

    fn extract_content(&self, text: &str) -> Result<String, FormatError> {
        if !self.use_fences {
            return Ok(text.trim().to_string());
        }
        let matches: Vec<regex::Captures> = FENCE_RE.captures_iter(text).collect();
        let valid_tags: &[&str] = match self.format_type {
            FormatType::Yaml => &["yaml", "yml"],
            FormatType::Json => &["json"],
        };
        let candidates: Vec<&regex::Captures> = matches
            .iter()
            .filter(|m| is_valid_language_tag(m.name("lang").map(|s| s.as_str()), valid_tags))
            .collect();

        if self.strict_fences {
            if candidates.len() != 1 {
                if candidates.is_empty() {
                    return Err(FormatError::Parse(
                        "Input string does not contain valid fence markers.".to_string(),
                    ));
                }
                return Err(FormatError::Parse(
                    "Multiple fenced blocks found. Expected exactly one.".to_string(),
                ));
            }
            return Ok(candidates[0].name("body").unwrap().as_str().trim().to_string());
        }

        if candidates.len() == 1 {
            return Ok(candidates[0].name("body").unwrap().as_str().trim().to_string());
        }
        if candidates.len() > 1 {
            return Err(FormatError::Parse(
                "Multiple fenced blocks found. Expected exactly one.".to_string(),
            ));
        }
        if matches.len() == 1 {
            return Ok(matches[0].name("body").unwrap().as_str().trim().to_string());
        }
        if !matches.is_empty() {
            return Err(FormatError::Parse(format!(
                "No {} code block found.",
                self.format_type.as_str()
            )));
        }
        Ok(text.trim().to_string())
    }
}

fn is_valid_language_tag(lang: Option<&str>, valid_tags: &[&str]) -> bool {
    let Some(lang) = lang else {
        return true;
    };
    let tag = lang.trim().to_ascii_lowercase();
    valid_tags.contains(&tag.as_str())
}

fn attributes_to_value(map: &indexmap::IndexMap<String, AttributeValue>) -> Value {
    let mut obj = Map::new();
    for (k, v) in map {
        obj.insert(
            k.clone(),
            match v {
                AttributeValue::String(s) => Value::String(s.clone()),
                AttributeValue::Strings(ss) => {
                    Value::Array(ss.iter().map(|s| Value::String(s.clone())).collect())
                }
            },
        );
    }
    Value::Object(obj)
}

fn to_yaml(value: &Value) -> String {
    let mut out = String::new();
    emit_yaml(value, 0, &mut out);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn emit_yaml(value: &Value, indent: usize, out: &mut String) {
    match value {
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                out.push_str(&" ".repeat(indent));
                out.push_str(k);
                out.push(':');
                emit_yaml_value(v, indent, out);
            }
        }
        Value::Array(arr) => emit_yaml_seq(arr, indent, out),
        other => emit_yaml_scalar(other, out),
    }
}

fn emit_yaml_value(v: &Value, indent: usize, out: &mut String) {
    match v {
        Value::Object(m) if !m.is_empty() => {
            out.push('\n');
            emit_yaml(v, indent + 2, out);
        }
        Value::Array(a) if !a.is_empty() => {
            out.push('\n');
            emit_yaml_seq(a, indent, out);
        }
        Value::Array(a) if a.is_empty() => out.push_str(" []"),
        Value::Object(m) if m.is_empty() => out.push_str(" {}"),
        _ => {
            out.push(' ');
            emit_yaml_scalar(v, out);
        }
    }
}

fn emit_yaml_seq(arr: &[Value], indent: usize, out: &mut String) {
    for (i, item) in arr.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&" ".repeat(indent));
        out.push_str("- ");
        match item {
            Value::Object(map) if !map.is_empty() => {
                for (j, (k, v)) in map.iter().enumerate() {
                    if j > 0 {
                        out.push('\n');
                        out.push_str(&" ".repeat(indent + 2));
                    }
                    out.push_str(k);
                    out.push(':');
                    emit_yaml_value(v, indent + 2, out);
                }
            }
            other => emit_yaml_scalar(other, out),
        }
    }
}

fn emit_yaml_scalar(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            if yaml_plain_ok(s) {
                out.push_str(s);
            } else {
                out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\"")));
            }
        }
        Value::Array(_) | Value::Object(_) => {
            out.push_str(&serde_yaml::to_string(v).unwrap_or_default());
        }
    }
}

fn yaml_plain_ok(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    fn attrs_role(role: &str) -> IndexMap<String, AttributeValue> {
        let mut m = IndexMap::new();
        m.insert("role".into(), AttributeValue::String(role.into()));
        m
    }

    #[test]
    fn json_with_wrapper_and_fences() {
        let handler = FormatHandler::new()
            .with_format_type(FormatType::Json)
            .with_wrapper(true, Some("extractions"))
            .with_fences(true);
        let ext = Extraction::new("person", "Alice").with_attributes(attrs_role("engineer"));
        let formatted = handler.format_extraction_example(&[ext]);
        assert!(formatted.contains("```json"));
        assert!(formatted.contains("\"extractions\":"));
        assert!(formatted.contains("\"person\": \"Alice\""));

        let output = "Here is the result:\n```json\n{\n  \"extractions\": [\n    {\"person\": \"Bob\", \"person_attributes\": {\"role\": \"manager\"}}\n  ]\n}\n```";
        let parsed = handler.parse_output(output, false).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["person"], "Bob");
    }

    #[test]
    fn json_no_wrapper_no_fences() {
        let handler = FormatHandler::new()
            .with_wrapper(false, None)
            .with_fences(false);
        let ext = Extraction::new("item", "book");
        let formatted = handler.format_extraction_example(&[ext]);
        assert!(!formatted.contains("```"));
        assert!(formatted.contains("\"item\": \"book\""));
        let parsed = handler
            .parse_output("[{\"item\": \"pen\", \"item_attributes\": {}}]", false)
            .unwrap();
        assert_eq!(parsed[0]["item"], "pen");
    }

    #[test]
    fn yaml_with_wrapper_and_fences() {
        let handler = FormatHandler::new()
            .with_format_type(FormatType::Yaml)
            .with_wrapper(true, Some("extractions"))
            .with_fences(true);
        let ext = Extraction::new("city", "Paris");
        let formatted = handler.format_extraction_example(&[ext]);
        assert!(formatted.contains("```yaml"));
        assert!(formatted.contains("extractions:"));
        assert!(formatted.contains("city: Paris"));
        let output = "```yaml\nextractions:\n  - city: London\n    city_attributes: {}\n```";
        let parsed = handler.parse_output(output, false).unwrap();
        assert_eq!(parsed[0]["city"], "London");
    }

    #[test]
    fn format_parse_roundtrip_json() {
        let handler = FormatHandler::new()
            .with_wrapper(true, Some("extractions"))
            .with_fences(true);
        let mut attrs = IndexMap::new();
        attrs.insert("key".into(), AttributeValue::String("data".into()));
        let ext = Extraction::new("test", "value").with_attributes(attrs);
        let formatted = handler.format_extraction_example(&[ext]);
        let parsed = handler.parse_output(&formatted, false).unwrap();
        assert_eq!(parsed[0]["test"], "value");
        assert_eq!(parsed[0]["test_attributes"]["key"], "data");
    }

    #[test]
    fn format_parse_roundtrip_yaml() {
        for (use_wrapper, use_fences) in [(false, false), (true, false)] {
            let handler = FormatHandler::new()
                .with_format_type(FormatType::Yaml)
                .with_wrapper(use_wrapper, Some("extractions"))
                .with_fences(use_fences);
            let mut attrs = IndexMap::new();
            attrs.insert("key".into(), AttributeValue::String("data".into()));
            let ext = Extraction::new("test", "value").with_attributes(attrs);
            let formatted = handler.format_extraction_example(&[ext]);
            let parsed = handler.parse_output(&formatted, false).unwrap();
            assert_eq!(parsed[0]["test"], "value");
            assert_eq!(parsed[0]["test_attributes"]["key"], "data");
        }
    }

    #[test]
    fn think_tags_stripped() {
        let handler = FormatHandler::new().with_fences(false);
        let input = "<think>Let me analyze this text...</think>{\"extractions\": [{\"person\": \"Alice\"}]}";
        let parsed = handler.parse_output(input, false).unwrap();
        assert_eq!(parsed[0]["person"], "Alice");
    }

    #[test]
    fn top_level_list_fallback() {
        let handler = FormatHandler::new().with_fences(false);
        let parsed = handler
            .parse_output("[{\"person\": \"Bob\"}, {\"person\": \"Carol\"}]", false)
            .unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["person"], "Bob");
        assert_eq!(parsed[1]["person"], "Carol");
    }

    #[test]
    fn deepseek_r1_real_output() {
        let handler = FormatHandler::new().with_fences(false);
        let output = "<think>\nAlright, so I need to extract people from the given text.\nI see John Smith is mentioned as an engineer.\n</think>\n{\"extractions\": [{\"person\": \"John Smith\"}]}";
        let parsed = handler.parse_output(output, false).unwrap();
        assert_eq!(parsed[0]["person"], "John Smith");
    }
}
