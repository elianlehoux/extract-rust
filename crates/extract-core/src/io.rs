//! Input and output operations for annotated documents (JSONL format).
//!
//! Provides serialization and deserialization matching LangExtract's JSON Lines format.

use std::borrow::Borrow;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Result, Write};
use std::path::Path;

use crate::types::AnnotatedDocument;

/// Saves annotated documents to a JSON Lines file.
///
/// Documents without a `document_id` (or with an empty `document_id`) are skipped.
///
/// If `path` is an existing directory, documents will be written to `path/data.jsonl`.
/// Otherwise, `path` is treated as the file destination and any missing parent directories
/// will be created.
pub fn save_annotated_documents<I, D, P>(docs: I, path: P) -> Result<()>
where
    I: IntoIterator<Item = D>,
    D: Borrow<AnnotatedDocument>,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let file_path = if path.is_dir() {
        path.join("data.jsonl")
    } else {
        path.to_path_buf()
    };

    if let Some(parent) = file_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(&file_path)?;
    let mut writer = BufWriter::new(file);

    for doc in docs {
        let doc = doc.borrow();
        let Some(doc_id) = doc.document_id_ref() else {
            continue;
        };
        if doc_id.trim().is_empty() {
            continue;
        }

        serde_json::to_writer(&mut writer, doc)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writer.write_all(b"\n")?;
    }

    writer.flush()?;
    Ok(())
}

/// Loads annotated documents from a JSON Lines file.
///
/// Blank lines are skipped.
/// If `path` is an existing directory, it attempts to load `path/data.jsonl`.
pub fn load_annotated_documents_jsonl<P: AsRef<Path>>(path: P) -> Result<Vec<AnnotatedDocument>> {
    let path = path.as_ref();
    let file_path = if path.is_dir() {
        path.join("data.jsonl")
    } else {
        path.to_path_buf()
    };

    let file = File::open(&file_path)?;
    let reader = BufReader::new(file);
    let mut docs = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let doc: AnnotatedDocument = serde_json::from_str(trimmed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        docs.push(doc);
    }

    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AlignmentStatus, CharInterval, Extraction};
    use indexmap::IndexMap;
    use std::fs;

    fn temp_file_path(suffix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "extract_io_test_{}_{}.jsonl",
            uuid::Uuid::new_v4(),
            suffix
        ))
    }

    #[test]
    fn test_roundtrip_equal_fields() {
        let temp_path = temp_file_path("roundtrip");

        let mut attributes = IndexMap::new();
        attributes.insert(
            "key".to_string(),
            crate::types::AttributeValue::String("value".to_string()),
        );

        let ext1 = Extraction::new("medication", "aspirin")
            .with_char_interval(29, 36)
            .with_alignment(AlignmentStatus::MatchExact)
            .with_index(1)
            .with_group(0)
            .with_attributes(attributes);

        let ext2 = Extraction::new("dosage", "100mg")
            .with_char_interval(23, 28)
            .with_alignment(AlignmentStatus::MatchFuzzy)
            .with_index(2)
            .with_group(0);

        let ext3 = Extraction::new("frequency", "daily")
            .with_char_interval(37, 42)
            .with_alignment(AlignmentStatus::MatchGreater);

        let ext4 = Extraction::new("condition", "headache")
            .with_char_interval(47, 55)
            .with_alignment(AlignmentStatus::MatchLesser);

        let doc1 = AnnotatedDocument::new()
            .with_text("Patient was prescribed 100mg aspirin daily for headache.")
            .with_document_id("doc_test_1")
            .with_extractions(vec![ext1.clone(), ext2.clone()]);

        let doc2 = AnnotatedDocument::new()
            .with_text("Follow-up visit next week.")
            .with_document_id("doc_test_2")
            .with_extractions(vec![ext3.clone(), ext4.clone()]);

        let docs = vec![doc1.clone(), doc2.clone()];

        // Save documents
        save_annotated_documents(&docs, &temp_path).expect("failed to save documents");

        // Verify JSON content has snake_case enum values
        let content = fs::read_to_string(&temp_path).expect("failed to read temp file");
        assert!(content.contains("\"match_exact\""));
        assert!(content.contains("\"match_fuzzy\""));
        assert!(content.contains("\"match_greater\""));
        assert!(content.contains("\"match_lesser\""));
        assert!(!content.contains("\"MatchExact\""));
        assert!(!content.contains("\"MatchFuzzy\""));

        // Load documents back
        let loaded = load_annotated_documents_jsonl(&temp_path).expect("failed to load documents");

        assert_eq!(loaded.len(), 2);

        // Verify equal fields (text, extractions, char_interval, alignment_status)
        assert_eq!(
            loaded[0].extractions.as_ref().unwrap()[0].char_interval,
            Some(CharInterval::new(29, 36))
        );

        for (loaded_doc, orig_doc) in loaded.iter().zip(docs.iter()) {
            assert_eq!(loaded_doc.text, orig_doc.text);
            assert_eq!(loaded_doc.document_id_ref(), orig_doc.document_id_ref());

            let loaded_exts = loaded_doc
                .extractions
                .as_ref()
                .expect("missing extractions");
            let orig_exts = orig_doc.extractions.as_ref().expect("missing extractions");
            assert_eq!(loaded_exts.len(), orig_exts.len());

            for (l_ext, o_ext) in loaded_exts.iter().zip(orig_exts.iter()) {
                assert_eq!(l_ext.extraction_class, o_ext.extraction_class);
                assert_eq!(l_ext.extraction_text, o_ext.extraction_text);
                assert_eq!(l_ext.char_interval, o_ext.char_interval);
                assert_eq!(l_ext.alignment_status, o_ext.alignment_status);
                assert_eq!(l_ext, o_ext);
            }
        }

        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_skip_docs_without_document_id() {
        let temp_path = temp_file_path("skip_no_id");

        let doc_no_id = AnnotatedDocument::new().with_text("No document ID");
        let doc_empty_id = AnnotatedDocument::new()
            .with_text("Empty document ID")
            .with_document_id("");
        let doc_whitespace_id = AnnotatedDocument::new()
            .with_text("Whitespace document ID")
            .with_document_id("   ");
        let doc_valid = AnnotatedDocument::new()
            .with_text("Valid document ID")
            .with_document_id("valid_doc_1");

        let docs = vec![doc_no_id, doc_empty_id, doc_whitespace_id, doc_valid];

        save_annotated_documents(&docs, &temp_path).expect("failed to save documents");

        let lines: Vec<String> = fs::read_to_string(&temp_path)
            .expect("failed to read")
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(
            lines.len(),
            1,
            "Only the doc with valid document_id should be saved"
        );

        let loaded = load_annotated_documents_jsonl(&temp_path).expect("failed to load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].document_id_ref(), Some("valid_doc_1"));
        assert_eq!(loaded[0].text.as_deref(), Some("Valid document ID"));

        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_enum_serialization_snake_case() {
        let temp_path = temp_file_path("snake_case_enums");

        let statuses = vec![
            (AlignmentStatus::MatchExact, "match_exact"),
            (AlignmentStatus::MatchGreater, "match_greater"),
            (AlignmentStatus::MatchLesser, "match_lesser"),
            (AlignmentStatus::MatchFuzzy, "match_fuzzy"),
        ];

        let mut extractions = Vec::new();
        for (status, _) in &statuses {
            extractions.push(
                Extraction::new("entity", "text")
                    .with_char_interval(0, 4)
                    .with_alignment(*status),
            );
        }

        let doc = AnnotatedDocument::new()
            .with_text("test text")
            .with_document_id("doc_enums")
            .with_extractions(extractions);

        save_annotated_documents(&[doc], &temp_path).expect("failed to save");

        let content = fs::read_to_string(&temp_path).expect("failed to read");
        for (_, expected_str) in &statuses {
            assert!(
                content.contains(&format!("\"alignment_status\":\"{expected_str}\"")),
                "Expected content to contain snake_case status {expected_str}"
            );
        }

        let loaded = load_annotated_documents_jsonl(&temp_path).expect("failed to load");
        assert_eq!(loaded.len(), 1);
        let loaded_exts = loaded[0].extractions.as_ref().unwrap();
        for (i, (expected_status, _)) in statuses.iter().enumerate() {
            assert_eq!(loaded_exts[i].alignment_status, Some(*expected_status));
        }

        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_file_not_found() {
        let non_existent =
            std::env::temp_dir().join(format!("non_existent_{}.jsonl", uuid::Uuid::new_v4()));
        let result = load_annotated_documents_jsonl(&non_existent);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn test_blank_lines_ignored() {
        let temp_path = temp_file_path("blank_lines");

        let doc = AnnotatedDocument::new()
            .with_text("doc1")
            .with_document_id("doc_1");
        let json_line = serde_json::to_string(&doc).unwrap();

        let raw = format!("\n  \n{}\n\n   \n{}\n\n", json_line, json_line);
        fs::write(&temp_path, raw).unwrap();

        let loaded = load_annotated_documents_jsonl(&temp_path).expect("failed to load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].document_id_ref(), Some("doc_1"));
        assert_eq!(loaded[1].document_id_ref(), Some("doc_1"));

        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let temp_path = temp_file_path("invalid_json");

        fs::write(&temp_path, "not a valid json\n").unwrap();
        let result = load_annotated_documents_jsonl(&temp_path);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidData);

        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_save_and_load_with_directory() {
        let temp_dir = std::env::temp_dir().join(format!("test_io_dir_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let doc = AnnotatedDocument::new()
            .with_text("Directory save test")
            .with_document_id("dir_doc_1");

        // Save passing directory path
        save_annotated_documents(std::slice::from_ref(&doc), &temp_dir)
            .expect("failed to save to dir");
        assert!(temp_dir.join("data.jsonl").exists());

        // Load passing directory path
        let loaded = load_annotated_documents_jsonl(&temp_dir).expect("failed to load from dir");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].document_id_ref(), Some("dir_doc_1"));
        assert_eq!(loaded[0].text.as_deref(), Some("Directory save test"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_into_iterator_support() {
        let temp_path = temp_file_path("into_iter");

        let doc1 = AnnotatedDocument::new()
            .with_text("Doc 1")
            .with_document_id("d1");
        let doc2 = AnnotatedDocument::new()
            .with_text("Doc 2")
            .with_document_id("d2");

        // Pass slice
        save_annotated_documents(&[doc1.clone(), doc2.clone()][..], &temp_path).unwrap();
        let loaded = load_annotated_documents_jsonl(&temp_path).unwrap();
        assert_eq!(loaded.len(), 2);

        // Pass Vec by value
        let vec_owned = vec![doc1.clone(), doc2.clone()];
        save_annotated_documents(vec_owned, &temp_path).unwrap();
        let loaded = load_annotated_documents_jsonl(&temp_path).unwrap();
        assert_eq!(loaded.len(), 2);

        // Pass Vec by reference
        let vec_ref = vec![doc1, doc2];
        save_annotated_documents(&vec_ref, &temp_path).unwrap();
        let loaded = load_annotated_documents_jsonl(&temp_path).unwrap();
        assert_eq!(loaded.len(), 2);

        let _ = fs::remove_file(&temp_path);
    }
}
