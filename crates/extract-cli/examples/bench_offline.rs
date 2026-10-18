//! Offline benchmark with a deterministic fake model (no LLM involved).
//!
//! The fake model reads the chunk after the last "Q: " in each prompt and
//! returns every "Firstname Lastname" pair found by a regex, so the timing
//! measures this crate's own work: chunking, prompting, parsing, alignment.
//! The Python twin is `evals/bench/fake_model_py.py`.
//!
//! ```text
//! python3 evals/bench/gen_corpus.py 2000 > /tmp/corpus.txt
//! cargo run --release -p extract-cli --example bench_offline -- /tmp/corpus.txt /tmp/rs_out.json
//! ```

use std::time::Instant;

use extract_core::extract::{extract, ExtractInput, ExtractOptions, ExtractResult};
use extract_core::model::MappingModel;
use extract_core::prompt_validation::PromptValidationLevel;
use extract_core::types::{ExampleData, Extraction};
use serde_json::json;

fn fake_answer(prompt: &str, re: &regex::Regex) -> String {
    let chunk = match (prompt.rfind("Q: "), prompt.rfind("\nA:")) {
        (Some(q), Some(a)) if a > q => &prompt[q + 3..a],
        _ => "",
    };
    let items: Vec<_> = re
        .find_iter(chunk)
        .map(|m| json!({ "person": m.as_str() }))
        .collect();
    json!({ "extractions": items }).to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1])?;
    let re = regex::Regex::new(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b")?;
    let examples = vec![ExampleData::new(
        "Alan Turing worked with Grace Hopper.",
        vec![
            Extraction::new("person", "Alan Turing"),
            Extraction::new("person", "Grace Hopper"),
        ],
    )];
    let mut model = MappingModel::new(|p: &str| fake_answer(p, &re));
    let options = ExtractOptions {
        max_char_buffer: 1000,
        batch_length: 10,
        fence_output: false,
        prompt_validation_level: PromptValidationLevel::Off,
        ..ExtractOptions::default()
    };
    let t0 = Instant::now();
    let result = extract(
        ExtractInput::Text(text),
        "Extract people.",
        &examples,
        &mut model,
        options,
    )?;
    let dt = t0.elapsed().as_secs_f64();
    let ExtractResult::Document(doc) = result else {
        unreachable!()
    };
    let exts: Vec<_> = doc
        .extractions
        .unwrap_or_default()
        .into_iter()
        .map(|e| {
            json!({
                "extraction_class": e.extraction_class,
                "extraction_text": e.extraction_text,
                "start": e.char_interval.and_then(|c| c.start_pos),
                "end": e.char_interval.and_then(|c| c.end_pos),
                "status": e.alignment_status.map(|s| s.as_str()),
            })
        })
        .collect();
    std::fs::write(&args[2], serde_json::to_string(&exts)?)?;
    println!("rust extract(): {dt:.3}s, {} extractions", exts.len());
    Ok(())
}
