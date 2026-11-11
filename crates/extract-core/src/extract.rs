//! High-level `extract()` entry point.

use crate::annotation::{AnnotateOptions, Annotator};
use crate::format::FormatHandler;
use crate::model::LanguageModel;
use crate::prompt_validation::{
    handle_alignment_report, validate_prompt_alignment, PromptValidationLevel,
};
use crate::prompting::PromptTemplateStructured;
use crate::resolver::Resolver;
use crate::tokenizer::Tokenizer;
use crate::types::{AnnotatedDocument, Document, ExampleData, FormatType, EXTRACTIONS_KEY};
use crate::Error;

/// What to extract from.
pub enum ExtractInput {
    Text(String),
    Documents(Vec<Document>),
}

/// Result of [`extract`].
#[derive(Debug)]
pub enum ExtractResult {
    Document(AnnotatedDocument),
    Documents(Vec<AnnotatedDocument>),
}

/// Options for [`extract`].
#[derive(Clone)]
pub struct ExtractOptions<'a> {
    pub max_char_buffer: i32,
    pub batch_length: usize,
    pub extraction_passes: usize,
    pub context_window_chars: Option<usize>,
    pub additional_context: Option<String>,
    pub format_type: FormatType,
    pub fence_output: bool,
    pub tokenizer: Option<&'a dyn Tokenizer>,
    pub suppress_parse_errors: bool,
    pub enable_fuzzy_alignment: bool,
    pub prompt_validation_level: PromptValidationLevel,
    pub prompt_validation_strict: bool,
}

impl Default for ExtractOptions<'_> {
    fn default() -> Self {
        Self {
            max_char_buffer: 1000,
            batch_length: 10,
            extraction_passes: 1,
            context_window_chars: None,
            additional_context: None,
            format_type: FormatType::Json,
            fence_output: true,
            tokenizer: None,
            suppress_parse_errors: true,
            enable_fuzzy_alignment: true,
            prompt_validation_level: PromptValidationLevel::Warning,
            prompt_validation_strict: false,
        }
    }
}

/// Extract structured entities from text or documents using `model`.
pub fn extract(
    input: ExtractInput,
    prompt_description: &str,
    examples: &[ExampleData],
    model: &mut dyn LanguageModel,
    options: ExtractOptions<'_>,
) -> Result<ExtractResult, Error> {
    if examples.is_empty() {
        return Err(Error::Config(
            "Examples are required to specify the extraction schema.".into(),
        ));
    }
    if options.prompt_validation_level != PromptValidationLevel::Off {
        let issues = validate_prompt_alignment(examples, options.tokenizer)?;
        handle_alignment_report(
            &issues,
            options.prompt_validation_level,
            options.prompt_validation_strict,
        )?;
    }

    let template =
        PromptTemplateStructured::new(prompt_description).with_examples(examples.to_vec());
    let format_handler = FormatHandler::new()
        .with_format_type(options.format_type)
        .with_wrapper(true, Some(EXTRACTIONS_KEY))
        .with_fences(options.fence_output);
    let annotator = Annotator::new(template, format_handler.clone());
    let resolver = Resolver::new(format_handler, None);

    let annotate_opts = AnnotateOptions {
        max_char_buffer: options.max_char_buffer,
        batch_length: options.batch_length,
        extraction_passes: options.extraction_passes,
        context_window_chars: options.context_window_chars,
        additional_context: options.additional_context.clone(),
        tokenizer: options.tokenizer,
        suppress_parse_errors: options.suppress_parse_errors,
        enable_fuzzy_alignment: options.enable_fuzzy_alignment,
        ..AnnotateOptions::default()
    };

    match input {
        ExtractInput::Text(text) => {
            let doc = annotator.annotate_text(&text, model, &resolver, annotate_opts)?;
            Ok(ExtractResult::Document(doc))
        }
        ExtractInput::Documents(docs) => {
            let docs = annotator.annotate_documents(docs, model, &resolver, annotate_opts)?;
            Ok(ExtractResult::Documents(docs))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ScriptedModel;
    use crate::types::Extraction;

    #[test]
    fn extract_text_json() {
        let examples = vec![ExampleData::new(
            "ROMEO loved Juliet.",
            vec![Extraction::new("character", "ROMEO")],
        )];
        let output = r#"```json
{"extractions": [{"character": "Juliet"}]}
```"#;
        let mut model = ScriptedModel::from_outputs([output]);
        let result = extract(
            ExtractInput::Text("Juliet gazed at the stars.".into()),
            "Extract characters.",
            &examples,
            &mut model,
            ExtractOptions::default(),
        )
        .unwrap();
        let ExtractResult::Document(doc) = result else {
            panic!("expected single document");
        };
        let xs = doc.extractions.unwrap();
        assert_eq!(xs[0].extraction_class, "character");
        assert_eq!(xs[0].extraction_text, "Juliet");
    }

    #[test]
    fn extract_requires_examples() {
        let mut model = ScriptedModel::from_outputs(["{}"]);
        let err = extract(
            ExtractInput::Text("hi".into()),
            "Extract.",
            &[],
            &mut model,
            ExtractOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("Examples are required"));
    }
}
