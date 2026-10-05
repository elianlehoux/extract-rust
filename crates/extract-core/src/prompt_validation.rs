//! Pre-flight alignment checks on few-shot examples.

use crate::resolver::{AlignOptions, WordAligner};
use crate::tokenizer::{RegexTokenizer, Tokenizer};
use crate::types::{AlignmentStatus, ExampleData};
use crate::Error;

/// How strictly to treat alignment issues on examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptValidationLevel {
    Off,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueKind {
    Failed,
    NonExact,
}

#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub example_index: usize,
    pub extraction_text: String,
    pub kind: IssueKind,
}

/// Align each example's extractions against its own source text.
pub fn validate_prompt_alignment(
    examples: &[ExampleData],
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<Vec<ValidationIssue>, Error> {
    let tok: &dyn Tokenizer = tokenizer.unwrap_or(&RegexTokenizer);
    let mut issues = Vec::new();
    for (i, example) in examples.iter().enumerate() {
        if example.extractions.is_empty() {
            continue;
        }
        let groups = vec![example.extractions.clone()];
        let aligned = WordAligner.align_groups(
            groups,
            &example.text,
            AlignOptions {
                tokenizer: Some(tok),
                ..AlignOptions::default()
            },
        )?;
        for extraction in aligned.into_iter().flatten() {
            match extraction.alignment_status {
                None => issues.push(ValidationIssue {
                    example_index: i,
                    extraction_text: extraction.extraction_text,
                    kind: IssueKind::Failed,
                }),
                Some(AlignmentStatus::MatchExact) => {}
                Some(_) => issues.push(ValidationIssue {
                    example_index: i,
                    extraction_text: extraction.extraction_text,
                    kind: IssueKind::NonExact,
                }),
            }
        }
    }
    Ok(issues)
}

/// Log or raise based on validation level.
pub fn handle_alignment_report(
    issues: &[ValidationIssue],
    level: PromptValidationLevel,
    strict_non_exact: bool,
) -> Result<(), Error> {
    if level == PromptValidationLevel::Off || issues.is_empty() {
        return Ok(());
    }
    if level == PromptValidationLevel::Error {
        let failed: Vec<_> = issues.iter().filter(|i| i.kind == IssueKind::Failed).collect();
        if !failed.is_empty() {
            return Err(Error::PromptAlignment(format!(
                "Prompt alignment validation failed: {} extraction(s) could not be aligned (e.g., {})",
                failed.len(),
                failed[0].extraction_text
            )));
        }
        if strict_non_exact {
            let non_exact: Vec<_> = issues
                .iter()
                .filter(|i| i.kind == IssueKind::NonExact)
                .collect();
            if !non_exact.is_empty() {
                return Err(Error::PromptAlignment(format!(
                    "Prompt alignment validation failed under strict mode: {} non-exact match(es) found (e.g., {})",
                    non_exact.len(),
                    non_exact[0].extraction_text
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Extraction;

    #[test]
    fn exact_example_is_clean() {
        let examples = vec![ExampleData::new(
            "Patient takes aspirin daily.",
            vec![Extraction::new("med", "aspirin")],
        )];
        let issues = validate_prompt_alignment(&examples, None).unwrap();
        assert!(issues.is_empty());
    }

    #[test]
    fn missing_span_is_failed() {
        let examples = vec![ExampleData::new(
            "Patient takes aspirin daily.",
            vec![Extraction::new("med", "warfarin")],
        )];
        let issues = validate_prompt_alignment(&examples, None).unwrap();
        assert_eq!(issues[0].kind, IssueKind::Failed);
        assert!(handle_alignment_report(&issues, PromptValidationLevel::Error, false).is_err());
        assert!(handle_alignment_report(&issues, PromptValidationLevel::Warning, false).is_ok());
        assert!(handle_alignment_report(&issues, PromptValidationLevel::Off, false).is_ok());
    }
}
