//! Domain types for the extraction pipeline.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{TokenInterval, TokenizedText, Tokenizer};

/// Attribute values on an extraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    String(String),
    Strings(Vec<String>),
}

/// How an extraction sits in the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentStatus {
    MatchExact,
    MatchGreater,
    MatchLesser,
    MatchFuzzy,
}

impl AlignmentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MatchExact => "match_exact",
            Self::MatchGreater => "match_greater",
            Self::MatchLesser => "match_lesser",
            Self::MatchFuzzy => "match_fuzzy",
        }
    }
}

/// Character range in the source document. `None` means ungrounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CharInterval {
    pub start_pos: Option<i32>,
    pub end_pos: Option<i32>,
}

impl CharInterval {
    pub fn new(start_pos: i32, end_pos: i32) -> Self {
        Self {
            start_pos: Some(start_pos),
            end_pos: Some(end_pos),
        }
    }
}

/// Output format for prompts and parsed model responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatType {
    Json,
    Yaml,
}

impl FormatType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Yaml => "yaml",
        }
    }
}

/// Wrapper key for the model JSON envelope.
pub const EXTRACTIONS_KEY: &str = "extractions";
/// Suffix appended to an extraction class for its attributes object.
pub const ATTRIBUTE_SUFFIX: &str = "_attributes";

/// One extracted entity, optionally grounded to source offsets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extraction {
    pub extraction_class: String,
    pub extraction_text: String,
    #[serde(default)]
    pub char_interval: Option<CharInterval>,
    #[serde(default)]
    pub alignment_status: Option<AlignmentStatus>,
    #[serde(default)]
    pub extraction_index: Option<i32>,
    #[serde(default)]
    pub group_index: Option<i32>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub attributes: Option<IndexMap<String, AttributeValue>>,
    #[serde(default, skip)]
    token_interval: Option<TokenInterval>,
}

impl Extraction {
    pub fn new(extraction_class: impl Into<String>, extraction_text: impl Into<String>) -> Self {
        Self {
            extraction_class: extraction_class.into(),
            extraction_text: extraction_text.into(),
            char_interval: None,
            alignment_status: None,
            extraction_index: None,
            group_index: None,
            description: None,
            attributes: None,
            token_interval: None,
        }
    }

    pub fn token_interval(&self) -> Option<TokenInterval> {
        self.token_interval
    }

    pub fn set_token_interval(&mut self, interval: Option<TokenInterval>) {
        self.token_interval = interval;
    }

    pub fn with_index(mut self, index: i32) -> Self {
        self.extraction_index = Some(index);
        self
    }

    pub fn with_group(mut self, group: i32) -> Self {
        self.group_index = Some(group);
        self
    }

    pub fn with_attributes(mut self, attributes: IndexMap<String, AttributeValue>) -> Self {
        self.attributes = Some(attributes);
        self
    }

    pub fn with_char_interval(mut self, start: i32, end: i32) -> Self {
        self.char_interval = Some(CharInterval::new(start, end));
        self
    }

    pub fn with_alignment(mut self, status: AlignmentStatus) -> Self {
        self.alignment_status = Some(status);
        self
    }
}

/// One scored candidate from a language model.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredOutput {
    pub score: Option<f64>,
    pub output: Option<String>,
}

impl ScoredOutput {
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            score: Some(1.0),
            output: Some(output.into()),
        }
    }
}

fn generate_document_id() -> String {
    format!("doc_{}", &uuid::Uuid::new_v4().simple().to_string()[..8])
}

/// Input document.
#[derive(Debug, Clone)]
pub struct Document {
    pub text: String,
    pub additional_context: Option<String>,
    pub document_id: String,
    tokenized_text: Option<TokenizedText>,
}

impl Document {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            additional_context: None,
            document_id: generate_document_id(),
            tokenized_text: None,
        }
    }

    pub fn with_document_id(mut self, document_id: impl Into<String>) -> Self {
        self.document_id = document_id.into();
        self
    }

    pub fn with_additional_context(mut self, context: impl Into<String>) -> Self {
        self.additional_context = Some(context.into());
        self
    }

    pub fn tokenized_text(&mut self, tokenizer: &impl Tokenizer) -> &TokenizedText {
        if self.tokenized_text.is_none() {
            self.tokenized_text = Some(tokenizer.tokenize(&self.text));
        }
        self.tokenized_text.as_ref().unwrap()
    }

    pub fn tokenized_text_ref(&self) -> Option<&TokenizedText> {
        self.tokenized_text.as_ref()
    }

    pub fn set_tokenized_text(&mut self, tokenized: TokenizedText) {
        self.tokenized_text = Some(tokenized);
    }
}

/// Document plus extractions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedDocument {
    pub extractions: Option<Vec<Extraction>>,
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    document_id: Option<String>,
    #[serde(skip)]
    tokenized_text: Option<TokenizedText>,
}

impl AnnotatedDocument {
    pub fn new() -> Self {
        Self {
            extractions: None,
            text: None,
            document_id: None,
            tokenized_text: None,
        }
    }

    pub fn with_document_id(mut self, document_id: impl Into<String>) -> Self {
        self.document_id = Some(document_id.into());
        self
    }

    pub fn document_id(&mut self) -> &str {
        if self.document_id.is_none() {
            self.document_id = Some(generate_document_id());
        }
        self.document_id.as_deref().unwrap()
    }

    pub fn document_id_ref(&self) -> Option<&str> {
        self.document_id.as_deref()
    }

    pub fn tokenized_text(&mut self, tokenizer: &impl Tokenizer) -> Option<&TokenizedText> {
        if self.tokenized_text.is_none() {
            if let Some(text) = &self.text {
                self.tokenized_text = Some(tokenizer.tokenize(text));
            }
        }
        self.tokenized_text.as_ref()
    }

    pub fn set_tokenized_text(&mut self, tokenized: TokenizedText) {
        self.tokenized_text = Some(tokenized);
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_extractions(mut self, extractions: Vec<Extraction>) -> Self {
        self.extractions = Some(extractions);
        self
    }
}

impl Default for AnnotatedDocument {
    fn default() -> Self {
        Self::new()
    }
}

/// Few-shot example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleData {
    pub text: String,
    pub extractions: Vec<Extraction>,
}

impl ExampleData {
    pub fn new(text: impl Into<String>, extractions: Vec<Extraction>) -> Self {
        Self {
            text: text.into(),
            extractions,
        }
    }
}
