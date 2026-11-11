//! Few-shot Q/A prompt construction.

use std::collections::HashMap;
use std::path::Path;

use crate::format::FormatHandler;
use crate::types::{ExampleData, FormatType};
use crate::Error;

const CONTEXT_PREFIX: &str = "[Previous text]: ...";

/// Description plus few-shot examples.
#[derive(Debug, Clone)]
pub struct PromptTemplateStructured {
    pub description: String,
    pub examples: Vec<ExampleData>,
}

impl PromptTemplateStructured {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            examples: Vec::new(),
        }
    }

    pub fn with_examples(mut self, examples: Vec<ExampleData>) -> Self {
        self.examples = examples;
        self
    }
}

/// Reads a prompt template from a YAML or JSON file.
pub fn read_prompt_template_from_file(
    path: impl AsRef<Path>,
    format_type: FormatType,
) -> Result<PromptTemplateStructured, Error> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| {
        Error::Prompt(format!(
            "Failed to parse prompt template from file: {}: {e}",
            path.display()
        ))
    })?;
    let value: serde_json::Value = match format_type {
        FormatType::Yaml => serde_yaml::from_str(&content).map_err(|e| {
            Error::Prompt(format!(
                "Failed to parse prompt template from file: {}: {e}",
                path.display()
            ))
        })?,
        FormatType::Json => serde_json::from_str(&content).map_err(|e| {
            Error::Prompt(format!(
                "Failed to parse prompt template from file: {}: {e}",
                path.display()
            ))
        })?,
    };
    let description = value
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(PromptTemplateStructured {
        description,
        examples: Vec::new(),
    })
}

/// Renders a Q/A prompt from a template and format handler.
#[derive(Debug, Clone)]
pub struct QAPromptGenerator {
    pub template: PromptTemplateStructured,
    pub format_handler: FormatHandler,
    pub examples_heading: String,
    pub question_prefix: String,
    pub answer_prefix: String,
}

impl QAPromptGenerator {
    pub fn new(template: PromptTemplateStructured, format_handler: FormatHandler) -> Self {
        Self {
            template,
            format_handler,
            examples_heading: "Examples".into(),
            question_prefix: "Q: ".into(),
            answer_prefix: "A: ".into(),
        }
    }

    pub fn format_example_as_text(&self, example: &ExampleData) -> String {
        let answer = self
            .format_handler
            .format_extraction_example(&example.extractions);
        format!(
            "{}{}\n{}{}\n",
            self.question_prefix, example.text, self.answer_prefix, answer
        )
    }

    pub fn render(&self, question: &str, additional_context: Option<&str>) -> String {
        let mut prompt_lines: Vec<String> = vec![format!("{}\n", self.template.description)];
        if let Some(ctx) = additional_context {
            if !ctx.is_empty() {
                prompt_lines.push(format!("{ctx}\n"));
            }
        }
        if !self.template.examples.is_empty() {
            prompt_lines.push(self.examples_heading.clone());
            for ex in &self.template.examples {
                prompt_lines.push(self.format_example_as_text(ex));
            }
        }
        prompt_lines.push(format!("{}{question}", self.question_prefix));
        prompt_lines.push(self.answer_prefix.clone());
        prompt_lines.join("\n")
    }
}

/// Stateless prompt builder.
pub struct PromptBuilder {
    generator: QAPromptGenerator,
}

impl PromptBuilder {
    pub fn new(generator: QAPromptGenerator) -> Self {
        Self { generator }
    }

    pub fn build_prompt(
        &self,
        chunk_text: &str,
        _document_id: &str,
        additional_context: Option<&str>,
    ) -> String {
        self.generator.render(chunk_text, additional_context)
    }
}

/// Injects the tail of the previous chunk as coreference context.
pub struct ContextAwarePromptBuilder {
    generator: QAPromptGenerator,
    context_window_chars: Option<usize>,
    prev_chunk_by_doc_id: HashMap<String, String>,
}

impl ContextAwarePromptBuilder {
    pub fn new(generator: QAPromptGenerator, context_window_chars: Option<usize>) -> Self {
        Self {
            generator,
            context_window_chars,
            prev_chunk_by_doc_id: HashMap::new(),
        }
    }

    pub fn context_window_chars(&self) -> Option<usize> {
        self.context_window_chars
    }

    pub fn context_prefix() -> &'static str {
        CONTEXT_PREFIX
    }

    pub fn build_prompt(
        &mut self,
        chunk_text: &str,
        document_id: &str,
        additional_context: Option<&str>,
    ) -> String {
        let effective = self.build_effective_context(document_id, additional_context);
        let prompt = self.generator.render(chunk_text, effective.as_deref());
        self.update_state(document_id, chunk_text);
        prompt
    }

    fn build_effective_context(
        &self,
        document_id: &str,
        additional_context: Option<&str>,
    ) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(n) = self.context_window_chars {
            if n > 0 {
                if let Some(prev) = self.prev_chunk_by_doc_id.get(document_id) {
                    let window = tail_chars(prev, n);
                    parts.push(format!("{CONTEXT_PREFIX}{window}"));
                }
            }
        }
        if let Some(ctx) = additional_context {
            if !ctx.is_empty() {
                parts.push(ctx.to_string());
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    fn update_state(&mut self, document_id: &str, chunk_text: &str) {
        if self.context_window_chars.is_some_and(|n| n > 0) {
            self.prev_chunk_by_doc_id
                .insert(document_id.to_string(), chunk_text.to_string());
        }
    }
}

fn tail_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if n >= count {
        s.to_string()
    } else {
        s.chars().skip(count - n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::FormatHandler;
    use crate::types::{AttributeValue, Extraction, FormatType};
    use indexmap::IndexMap;

    fn attrs(pairs: &[(&str, &str)]) -> IndexMap<String, AttributeValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), AttributeValue::String((*v).to_string())))
            .collect()
    }

    fn yaml_handler() -> FormatHandler {
        FormatHandler::new()
            .with_format_type(FormatType::Yaml)
            .with_wrapper(true, Some("extractions"))
            .with_fences(true)
    }

    #[test]
    fn generate_prompt_yaml_examples() {
        let template = PromptTemplateStructured::new(
            "You are an assistant specialized in extracting key extractions from text.\nIdentify and extract important extractions such as people, places,\norganizations, dates, and medical conditions mentioned in the text.\n**Please ensure that the extractions are extracted in the same order as they\nappear in the source text.**\nProvide the extracted extractions in a structured YAML format.",
        )
        .with_examples(vec![ExampleData::new(
            "The patient was diagnosed with hypertension and diabetes.",
            vec![
                Extraction::new("medical_condition", "hypertension").with_attributes(attrs(&[
                    ("chronicity", "chronic"),
                    ("system", "cardiovascular"),
                ])),
                Extraction::new("medical_condition", "diabetes").with_attributes(attrs(&[
                    ("chronicity", "chronic"),
                    ("system", "endocrine"),
                ])),
            ],
        )]);
        let mut gen = QAPromptGenerator::new(template, yaml_handler());
        gen.examples_heading.clear();
        gen.question_prefix.clear();
        gen.answer_prefix.clear();
        let actual = gen.render(
            "The patient reports chest pain and shortness of breath.",
            None,
        );
        assert!(actual.contains("medical_condition: hypertension"));
        assert!(actual.contains("chronicity: chronic"));
        assert!(actual.contains("The patient reports chest pain and shortness of breath."));
        assert!(actual.contains("```yaml"));
        assert!(actual.contains("extractions:"));
    }

    #[test]
    fn format_example_json_and_yaml() {
        let example = ExampleData::new(
            "Patient has diabetes and is prescribed insulin.",
            vec![
                Extraction::new("medical_condition", "diabetes")
                    .with_attributes(attrs(&[("chronicity", "chronic")])),
                Extraction::new("medication", "insulin")
                    .with_attributes(attrs(&[("prescribed", "prescribed")])),
            ],
        );
        let json_handler = FormatHandler::new()
            .with_format_type(FormatType::Json)
            .with_wrapper(true, Some("extractions"))
            .with_fences(true);
        let mut gen = QAPromptGenerator::new(PromptTemplateStructured::new("x"), json_handler);
        gen.question_prefix.clear();
        gen.answer_prefix.clear();
        let formatted = gen.format_example_as_text(&example);
        assert!(formatted.contains("Patient has diabetes and is prescribed insulin."));
        assert!(formatted.contains("```json"));
        assert!(formatted.contains("\"medical_condition\": \"diabetes\""));

        let mut gen = QAPromptGenerator::new(PromptTemplateStructured::new("x"), yaml_handler());
        gen.question_prefix.clear();
        gen.answer_prefix.clear();
        let formatted = gen.format_example_as_text(&example);
        assert!(formatted.contains("```yaml"));
        assert!(formatted.contains("medical_condition: diabetes"));
        assert!(formatted.contains("medication: insulin"));
    }

    fn simple_generator() -> QAPromptGenerator {
        let template = PromptTemplateStructured::new("Extract entities.").with_examples(vec![
            ExampleData::new("Sample text.", vec![Extraction::new("entity", "Sample")]),
        ]);
        QAPromptGenerator::new(template, yaml_handler())
    }

    #[test]
    fn prompt_builder_includes_chunk_and_context() {
        let builder = PromptBuilder::new(simple_generator());
        let prompt = builder.build_prompt("Test input text.", "doc1", None);
        assert!(prompt.contains("Test input text."));
        assert!(prompt.contains("Extract entities."));
        let prompt = builder.build_prompt("Test input.", "doc1", Some("Important context here."));
        assert!(prompt.contains("Important context here."));
    }

    #[test]
    fn context_aware_builder() {
        let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(20));
        assert_eq!(builder.context_window_chars(), Some(20));
        let first = builder.build_prompt("First chunk ending.", "doc1", None);
        assert!(!first.contains(CONTEXT_PREFIX));
        let second = builder.build_prompt("Second chunk text.", "doc1", None);
        assert!(second.contains(CONTEXT_PREFIX));
        assert!(second.contains("chunk ending."));
    }

    #[test]
    fn context_disabled() {
        let mut builder = ContextAwarePromptBuilder::new(simple_generator(), None);
        builder.build_prompt("First chunk.", "doc1", None);
        let second = builder.build_prompt("Second chunk.", "doc1", None);
        assert!(!second.contains(CONTEXT_PREFIX));
    }

    #[test]
    fn context_isolated_per_document() {
        let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(50));
        builder.build_prompt("Doc A chunk one.", "docA", None);
        builder.build_prompt("Doc B chunk one.", "docB", None);
        let a2 = builder.build_prompt("Doc A chunk two.", "docA", None);
        let b2 = builder.build_prompt("Doc B chunk two.", "docB", None);
        assert!(a2.contains("Doc A chunk one"));
        assert!(!a2.contains("Doc B"));
        assert!(b2.contains("Doc B chunk one"));
        assert!(!b2.contains("Doc A"));
    }

    #[test]
    fn combines_previous_and_additional_context() {
        let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(30));
        builder.build_prompt("Previous chunk text.", "doc1", None);
        let prompt = builder.build_prompt("Current chunk.", "doc1", Some("Extra notes."));
        assert!(prompt.contains(CONTEXT_PREFIX));
        assert!(prompt.contains("Extra notes."));
    }
}
