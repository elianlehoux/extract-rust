//! Library usage: the README's crate example, runnable.
//!
//! ```text
//! ollama pull gemma2:2b   # once
//! cargo run -p extract-cli --example library
//! ```

use extract_core::extract::{extract, ExtractInput, ExtractOptions, ExtractResult};
use extract_core::types::{ExampleData, Extraction};
use extract_providers::{create_model, ModelOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let examples = vec![ExampleData::new(
        "Alan Turing worked at Bletchley Park.",
        vec![Extraction::new("person", "Alan Turing")],
    )];

    let mut model = create_model("gemma2:2b", ModelOptions::default())?;

    let result = extract(
        ExtractInput::Text("Ada Lovelace wrote the first algorithm with Charles Babbage.".into()),
        "Extract people mentioned in the source text",
        &examples,
        model.as_mut(),
        ExtractOptions {
            // Ollama's JSON mode returns raw JSON, without ```json fences.
            fence_output: false,
            ..ExtractOptions::default()
        },
    )?;

    if let ExtractResult::Document(doc) = result {
        let text = doc.text.clone().unwrap_or_default();
        for ext in doc.extractions.unwrap_or_default() {
            let (start, end) = ext
                .char_interval
                .and_then(|c| Some((c.start_pos?, c.end_pos?)))
                .unwrap_or((-1, -1));
            let span: String = if start >= 0 {
                text.chars()
                    .skip(start as usize)
                    .take((end - start) as usize)
                    .collect()
            } else {
                String::new()
            };
            println!(
                "{:<8} {:<16} [{start}..{end}] {:?} -> source: {span:?}",
                ext.extraction_class,
                ext.extraction_text,
                ext.alignment_status
                    .map(|s| s.as_str())
                    .unwrap_or("unaligned"),
            );
        }
    }
    Ok(())
}
