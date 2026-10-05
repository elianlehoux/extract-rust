//! Chunk → prompt → infer → resolve → align pipeline.

use std::collections::{HashMap, HashSet};

use crate::chunking::{make_batches_of_textchunk, ChunkIterator, TextChunk};
use crate::format::FormatHandler;
use crate::model::LanguageModel;
use crate::prompting::{ContextAwarePromptBuilder, PromptTemplateStructured, QAPromptGenerator};
use crate::resolver::{AlignOptions, Resolver, DEFAULT_INDEX_SUFFIX};
use crate::tokenizer::{RegexTokenizer, Tokenizer};
use crate::types::{
    AnnotatedDocument, Document, Extraction, FormatType, ATTRIBUTE_SUFFIX, EXTRACTIONS_KEY,
};
use crate::Error;

/// Options for [`Annotator`].
#[derive(Clone)]
pub struct AnnotateOptions<'a> {
    pub max_char_buffer: i32,
    pub batch_length: usize,
    pub extraction_passes: usize,
    pub context_window_chars: Option<usize>,
    pub additional_context: Option<String>,
    pub tokenizer: Option<&'a dyn Tokenizer>,
    pub suppress_parse_errors: bool,
    pub enable_fuzzy_alignment: bool,
    pub fuzzy_alignment_threshold: f64,
    pub accept_match_lesser: bool,
    pub exact_alignment_algorithm: &'a str,
    pub fuzzy_alignment_algorithm: &'a str,
    pub fuzzy_alignment_min_density: f64,
}

impl Default for AnnotateOptions<'_> {
    fn default() -> Self {
        Self {
            max_char_buffer: 200,
            batch_length: 1,
            extraction_passes: 1,
            context_window_chars: None,
            additional_context: None,
            tokenizer: None,
            suppress_parse_errors: false,
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: crate::resolver::FUZZY_ALIGNMENT_MIN_THRESHOLD,
            accept_match_lesser: true,
            exact_alignment_algorithm: "dp",
            fuzzy_alignment_algorithm: "lcs",
            fuzzy_alignment_min_density: crate::resolver::FUZZY_ALIGNMENT_MIN_DENSITY,
        }
    }
}

/// First-pass wins merge of overlapping extractions.
pub fn merge_non_overlapping_extractions(
    all_extractions: &[Vec<Extraction>],
) -> Vec<Extraction> {
    if all_extractions.is_empty() {
        return Vec::new();
    }
    if all_extractions.len() == 1 {
        return all_extractions[0].clone();
    }
    let mut merged = all_extractions[0].clone();
    for pass in &all_extractions[1..] {
        for extraction in pass {
            let overlaps = extraction.char_interval.is_some()
                && merged.iter().any(|existing| extractions_overlap(extraction, existing));
            if !overlaps {
                merged.push(extraction.clone());
            }
        }
    }
    merged
}

/// True if both have intervals and they overlap (`start1 < end2 && start2 < end1`).
pub fn extractions_overlap(a: &Extraction, b: &Extraction) -> bool {
    let (Some(ia), Some(ib)) = (a.char_interval, b.char_interval) else {
        return false;
    };
    let (Some(s1), Some(e1), Some(s2), Some(e2)) =
        (ia.start_pos, ia.end_pos, ib.start_pos, ib.end_pos)
    else {
        return false;
    };
    s1 < e2 && s2 < e1
}

/// Annotates documents by chunking, prompting, inferring, and aligning.
pub struct Annotator {
    prompt_generator: QAPromptGenerator,
}

impl Annotator {
    pub fn new(prompt_template: PromptTemplateStructured, format_handler: FormatHandler) -> Self {
        Self {
            prompt_generator: QAPromptGenerator::new(prompt_template, format_handler),
        }
    }

    pub fn with_defaults(prompt_template: PromptTemplateStructured) -> Self {
        let format_handler = FormatHandler::new()
            .with_format_type(FormatType::Yaml)
            .with_wrapper(true, Some(EXTRACTIONS_KEY))
            .with_fences(false)
            .with_attribute_suffix(ATTRIBUTE_SUFFIX);
        Self::new(prompt_template, format_handler)
    }

    pub fn annotate_text(
        &self,
        text: &str,
        model: &mut dyn LanguageModel,
        resolver: &Resolver,
        options: AnnotateOptions<'_>,
    ) -> Result<AnnotatedDocument, Error> {
        let mut doc = Document::new(text);
        if let Some(ctx) = &options.additional_context {
            doc = doc.with_additional_context(ctx.clone());
        }
        let mut docs = self.annotate_documents(vec![doc], model, resolver, options)?;
        if docs.len() != 1 {
            return Err(Error::Config(format!(
                "Expected 1 annotation but got {} annotations.",
                docs.len()
            )));
        }
        Ok(docs.remove(0))
    }

    pub fn annotate_documents(
        &self,
        documents: Vec<Document>,
        model: &mut dyn LanguageModel,
        resolver: &Resolver,
        options: AnnotateOptions<'_>,
    ) -> Result<Vec<AnnotatedDocument>, Error> {
        if options.extraction_passes <= 1 {
            self.annotate_single_pass(documents, model, resolver, &options)
        } else {
            self.annotate_sequential_passes(documents, model, resolver, &options)
        }
    }

    fn annotate_single_pass(
        &self,
        documents: Vec<Document>,
        model: &mut dyn LanguageModel,
        resolver: &Resolver,
        options: &AnnotateOptions<'_>,
    ) -> Result<Vec<AnnotatedDocument>, Error> {
        let tokenizer: &dyn Tokenizer = options.tokenizer.unwrap_or(&RegexTokenizer);
        let mut doc_order: Vec<String> = Vec::new();
        let mut doc_text: HashMap<String, String> = HashMap::new();
        let mut per_doc: HashMap<String, Vec<Extraction>> = HashMap::new();
        let mut visited = HashSet::new();

        let mut chunks: Vec<TextChunk> = Vec::new();
        for mut document in documents {
            let id = document.document_id.clone();
            if !visited.insert(id.clone()) {
                return Err(Error::InvalidDocument(format!(
                    "Document id {id} is already visited."
                )));
            }
            doc_order.push(id.clone());
            doc_text.insert(id.clone(), document.text.clone());
            let tokenized = tokenizer.tokenize(&document.text);
            document.set_tokenized_text(tokenized.clone());
            let iter = ChunkIterator::from_tokenized(
                tokenized,
                options.max_char_buffer,
                &RegexTokenizer,
                Some(document),
            )?;
            chunks.extend(iter);
        }

        let batches = make_batches_of_textchunk(chunks, options.batch_length.max(1));
        let mut prompt_builder = ContextAwarePromptBuilder::new(
            self.prompt_generator.clone(),
            options.context_window_chars,
        );

        for batch in batches {
            if batch.is_empty() {
                continue;
            }
            let mut prompts = Vec::with_capacity(batch.len());
            for chunk in &batch {
                let text = chunk.chunk_text()?;
                let doc_id = chunk.document_id().unwrap_or("");
                prompts.push(prompt_builder.build_prompt(
                    &text,
                    doc_id,
                    chunk.additional_context(),
                ));
            }
            let outputs = model.infer(&prompts)?;
            for (chunk, scored) in batch.iter().zip(outputs) {
                if scored.is_empty() {
                    return Err(Error::InferenceOutput(
                        "No scored outputs from language model.".into(),
                    ));
                }
                let output = scored[0].output.clone().unwrap_or_default();
                let resolved =
                    resolver.resolve(&output, options.suppress_parse_errors)?;
                let chunk_text = chunk.chunk_text()?;
                let token_offset = chunk.token_interval.start_index;
                let char_offset = chunk
                    .char_interval()
                    .ok()
                    .and_then(|c| c.start_pos)
                    .unwrap_or(0);
                let aligned = resolver.align(
                    resolved,
                    &chunk_text,
                    AlignOptions {
                        token_offset,
                        char_offset,
                        enable_fuzzy_alignment: options.enable_fuzzy_alignment,
                        fuzzy_alignment_threshold: options.fuzzy_alignment_threshold,
                        accept_match_lesser: options.accept_match_lesser,
                        tokenizer: options.tokenizer,
                        fuzzy_alignment_algorithm: options.fuzzy_alignment_algorithm,
                        fuzzy_alignment_min_density: options.fuzzy_alignment_min_density,
                        exact_alignment_algorithm: options.exact_alignment_algorithm,
                        ..AlignOptions::default()
                    },
                )?;
                if let Some(id) = chunk.document_id() {
                    per_doc.entry(id.to_string()).or_default().extend(aligned);
                }
            }
        }

        Ok(doc_order
            .into_iter()
            .map(|id| {
                let text = doc_text.remove(&id).unwrap_or_default();
                let extractions = per_doc.remove(&id).unwrap_or_default();
                AnnotatedDocument::new()
                    .with_document_id(id)
                    .with_text(text)
                    .with_extractions(extractions)
            })
            .collect())
    }

    fn annotate_sequential_passes(
        &self,
        documents: Vec<Document>,
        model: &mut dyn LanguageModel,
        resolver: &Resolver,
        options: &AnnotateOptions<'_>,
    ) -> Result<Vec<AnnotatedDocument>, Error> {
        let mut by_pass: HashMap<String, Vec<Vec<Extraction>>> = HashMap::new();
        let texts: HashMap<String, String> = documents
            .iter()
            .map(|d| (d.document_id.clone(), d.text.clone()))
            .collect();
        let order: Vec<String> = documents.iter().map(|d| d.document_id.clone()).collect();

        for _pass in 0..options.extraction_passes {
            let annotated = self.annotate_single_pass(
                documents.clone(),
                model,
                resolver,
                options,
            )?;
            for doc in annotated {
                let id = doc.document_id_ref().unwrap_or("").to_string();
                by_pass
                    .entry(id)
                    .or_default()
                    .push(doc.extractions.unwrap_or_default());
            }
        }

        Ok(order
            .into_iter()
            .map(|id| {
                let merged = merge_non_overlapping_extractions(
                    by_pass.get(&id).map(Vec::as_slice).unwrap_or(&[]),
                );
                let text = texts.get(&id).cloned().unwrap_or_default();
                AnnotatedDocument::new()
                    .with_document_id(id)
                    .with_text(text)
                    .with_extractions(merged)
            })
            .collect())
    }
}

/// Resolver used by tests that match the Python YAML+index-suffix setup.
pub fn yaml_index_resolver(fence_output: bool) -> Resolver {
    Resolver::from_legacy(
        fence_output,
        FormatType::Yaml,
        Some(DEFAULT_INDEX_SUFFIX.into()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ScriptedModel;
    use crate::types::{AlignmentStatus, ScoredOutput};

    fn yaml_out(body: &str) -> String {
        format!("```yaml\n{body}\n```")
    }

    #[test]
    fn merge_cases() {
        assert!(merge_non_overlapping_extractions(&[]).is_empty());
        let single = vec![vec![
            Extraction::new("class1", "text1").with_char_interval(0, 5),
            Extraction::new("class2", "text2").with_char_interval(10, 15),
        ]];
        assert_eq!(merge_non_overlapping_extractions(&single).len(), 2);

        let passes = vec![
            vec![Extraction::new("class1", "text1").with_char_interval(0, 5)],
            vec![Extraction::new("class2", "text2").with_char_interval(10, 15)],
        ];
        let merged = merge_non_overlapping_extractions(&passes);
        assert_eq!(merged.len(), 2);

        let overlap = vec![
            vec![Extraction::new("class1", "text1").with_char_interval(0, 10)],
            vec![
                Extraction::new("class2", "text2").with_char_interval(5, 15),
                Extraction::new("class3", "text3").with_char_interval(20, 25),
            ],
        ];
        let merged = merge_non_overlapping_extractions(&overlap);
        let classes: Vec<_> = merged.iter().map(|e| e.extraction_class.as_str()).collect();
        assert_eq!(classes, vec!["class1", "class3"]);
    }

    #[test]
    fn overlap_detection() {
        let a = Extraction::new("c1", "t1").with_char_interval(0, 10);
        let b = Extraction::new("c2", "t2").with_char_interval(5, 15);
        assert!(extractions_overlap(&a, &b));
        let b = Extraction::new("c2", "t2").with_char_interval(10, 15);
        assert!(!extractions_overlap(&a, &b));
        let b = Extraction::new("c2", "t2").with_char_interval(5, 10);
        let a = Extraction::new("c1", "t1").with_char_interval(0, 5);
        assert!(!extractions_overlap(&a, &b));
        let none = Extraction::new("c1", "t1");
        let some = Extraction::new("c2", "t2").with_char_interval(5, 15);
        assert!(!extractions_overlap(&none, &some));
        assert!(!extractions_overlap(&some, &none));
        assert!(!extractions_overlap(&none, &none));
    }

    #[test]
    fn annotate_text_single_chunk() {
        let text = "Patient Jane Doe, ID 67890, received 10mg of Lisinopril daily for hypertension diagnosed on 2023-03-15.";
        let output = yaml_out(
            "extractions:\n- patient: \"Jane Doe\"\n  patient_index: 1\n  patient_id: \"67890\"\n  patient_id_index: 4\n  dosage: \"10mg\"\n  dosage_index: 6\n  medication: \"Lisinopril\"\n  medication_index: 8\n  frequency: \"daily\"\n  frequency_index: 9\n  condition: \"hypertension\"\n  condition_index: 11\n  diagnosis_date: \"2023-03-15\"\n  diagnosis_date_index: 13",
        );
        let mut model = ScriptedModel::from_outputs([output]);
        let annotator = Annotator::with_defaults(PromptTemplateStructured::new(""));
        let resolver = yaml_index_resolver(true);
        let result = annotator
            .annotate_text(text, &mut model, &resolver, AnnotateOptions::default())
            .unwrap();
        let xs = result.extractions.unwrap();
        assert_eq!(xs.len(), 7);
        assert_eq!(xs[0].extraction_class, "patient");
        assert_eq!(xs[0].extraction_text, "Jane Doe");
        assert_eq!(xs[0].alignment_status, Some(AlignmentStatus::MatchExact));
        assert_eq!(xs[0].char_interval.unwrap(), crate::types::CharInterval::new(8, 16));
        assert_eq!(model.calls[0][0], format!("\n\nQ: {text}\nA: "));
    }

    #[test]
    fn multipass_non_overlapping() {
        let text = "Patient John Smith has diabetes and takes insulin daily.";
        let mut model = ScriptedModel::from_outputs([
            yaml_out("extractions:\n- patient: \"John Smith\"\n  patient_index: 1\n- condition: \"diabetes\"\n  condition_index: 4"),
            yaml_out("extractions:\n- medication: \"insulin\"\n  medication_index: 7\n- frequency: \"daily\"\n  frequency_index: 8"),
        ]);
        let annotator = Annotator::with_defaults(PromptTemplateStructured::new(""));
        let resolver = yaml_index_resolver(true);
        let opts = AnnotateOptions {
            extraction_passes: 2,
            ..AnnotateOptions::default()
        };
        let result = annotator
            .annotate_text(text, &mut model, &resolver, opts)
            .unwrap();
        let xs = result.extractions.unwrap();
        let classes: Vec<_> = xs.iter().map(|e| e.extraction_class.as_str()).collect();
        assert_eq!(classes.len(), 4);
        assert!(classes.contains(&"patient"));
        assert!(classes.contains(&"condition"));
        assert!(classes.contains(&"medication"));
        assert!(classes.contains(&"frequency"));
        assert_eq!(model.calls.len(), 2);
    }

    #[test]
    fn multipass_overlapping_first_wins() {
        let text = "Dr. Smith prescribed aspirin.";
        let mut model = ScriptedModel::from_outputs([
            yaml_out("extractions:\n- doctor: \"Dr. Smith\"\n  doctor_index: 0"),
            yaml_out("extractions:\n- patient: \"Smith\"\n  patient_index: 1\n- medication: \"aspirin\"\n  medication_index: 2"),
        ]);
        let annotator = Annotator::with_defaults(PromptTemplateStructured::new(""));
        let resolver = yaml_index_resolver(true);
        let opts = AnnotateOptions {
            extraction_passes: 2,
            ..AnnotateOptions::default()
        };
        let result = annotator
            .annotate_text(text, &mut model, &resolver, opts)
            .unwrap();
        let xs = result.extractions.unwrap();
        let classes: Vec<_> = xs.iter().map(|e| e.extraction_class.as_str()).collect();
        assert_eq!(xs.len(), 2);
        assert!(classes.contains(&"doctor"));
        assert!(classes.contains(&"medication"));
        let doctor = xs.iter().find(|e| e.extraction_class == "doctor").unwrap();
        assert_eq!(doctor.extraction_text, "Dr. Smith");
    }

    #[test]
    fn annotate_documents_isolated() {
        let docs = vec![
            Document::new("Patient took 400 mg PO Ibuprofen q4h for two days.")
                .with_document_id("doc1"),
            Document::new("Patient was given 250 mg IV Cefazolin TID for one week.")
                .with_document_id("doc2"),
        ];
        let mut model = crate::model::MappingModel::new(|prompt: &str| {
            if prompt.contains("Ibuprofen") {
                yaml_out("extractions:\n- medication: \"Ibuprofen\"\n  medication_index: 4")
            } else if prompt.contains("Cefazolin") {
                yaml_out("extractions:\n- medication: \"Cefazolin\"\n  medication_index: 4")
            } else {
                yaml_out("extractions: []")
            }
        });
        let annotator = Annotator::with_defaults(PromptTemplateStructured::new(""));
        let resolver = yaml_index_resolver(true);
        let results = annotator
            .annotate_documents(docs, &mut model, &resolver, AnnotateOptions::default())
            .unwrap();
        assert_eq!(results.len(), 2);
        let meds = |i: usize| {
            results[i]
                .extractions
                .as_ref()
                .unwrap()
                .iter()
                .filter(|e| e.extraction_class == "medication")
                .map(|e| e.extraction_text.as_str())
                .collect::<Vec<_>>()
        };
        assert!(meds(0).contains(&"Ibuprofen"));
        assert!(!meds(0).contains(&"Cefazolin"));
        assert!(meds(1).contains(&"Cefazolin"));
        assert!(!meds(1).contains(&"Ibuprofen"));
    }

    #[test]
    fn context_window_in_prompts() {
        let text =
            "Dr. Sarah Johnson is a cardiologist. She specializes in heart surgery.";
        let empty = yaml_out("extractions: []");
        let mut model = ScriptedModel::from_outputs([empty.clone(), empty]);
        let annotator = Annotator::with_defaults(PromptTemplateStructured::new(""));
        let resolver = yaml_index_resolver(true);
        let opts = AnnotateOptions {
            max_char_buffer: 40,
            batch_length: 1,
            context_window_chars: Some(30),
            enable_fuzzy_alignment: false,
            ..AnnotateOptions::default()
        };
        annotator
            .annotate_text(text, &mut model, &resolver, opts)
            .unwrap();
        assert_eq!(model.calls.len(), 2);
        assert!(!model.calls[0][0].contains(crate::prompting::ContextAwarePromptBuilder::context_prefix()));
        assert!(model.calls[1][0].contains(crate::prompting::ContextAwarePromptBuilder::context_prefix()));
        assert!(model.calls[1][0].contains("cardiologist"));
    }

    #[test]
    fn no_scored_outputs_errors() {
        let mut model = ScriptedModel::new(vec![vec![Vec::<ScoredOutput>::new()]]);
        let annotator = Annotator::with_defaults(PromptTemplateStructured::new(""));
        let resolver = yaml_index_resolver(true);
        let err = annotator
            .annotate_text("hello.", &mut model, &resolver, AnnotateOptions::default())
            .unwrap_err();
        assert!(err.to_string().contains("No scored outputs"));
    }

    #[test]
    fn duplicate_document_id_errors() {
        let docs = vec![
            Document::new("a").with_document_id("same"),
            Document::new("b").with_document_id("same"),
        ];
        let mut model = ScriptedModel::from_outputs([yaml_out("extractions: []")]);
        let annotator = Annotator::with_defaults(PromptTemplateStructured::new(""));
        let resolver = yaml_index_resolver(true);
        let err = annotator
            .annotate_documents(docs, &mut model, &resolver, AnnotateOptions::default())
            .unwrap_err();
        assert!(err.to_string().contains("already visited"));
    }
}
