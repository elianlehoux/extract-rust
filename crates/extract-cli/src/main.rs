//! Command-line interface for source-grounded extraction.

use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use extract_core::extract::{extract, ExtractInput, ExtractOptions, ExtractResult};
use extract_core::prompt_validation::PromptValidationLevel;
use extract_core::types::{ExampleData, FormatType};
use extract_providers::{create_model, ModelOptions, ProviderKind};

#[derive(Parser, Debug)]
#[command(
    name = "extract",
    about = "Extract structured entities from text and ground them to source offsets."
)]
struct Cli {
    /// Input text. Use - to read stdin.
    #[arg(long, conflicts_with = "file")]
    text: Option<String>,
    /// Path to a UTF-8 text file.
    #[arg(long)]
    file: Option<PathBuf>,
    /// Extraction instructions.
    #[arg(long, short = 'p')]
    prompt: String,
    /// JSON file with few-shot examples (array of {text, extractions}).
    #[arg(long, short = 'e')]
    examples: PathBuf,
    /// Model id (e.g. gemma2:2b, gemini-2.5-flash, gpt-4o-mini).
    #[arg(long, short = 'm')]
    model: String,
    /// Force provider instead of inferring from the model id.
    #[arg(long)]
    provider: Option<String>,
    /// API key. When omitted, each provider reads its own variable:
    /// Gemini: GEMINI_API_KEY or LANGEXTRACT_API_KEY; OpenAI: OPENAI_API_KEY;
    /// Ollama: none.
    #[arg(long)]
    api_key: Option<String>,
    /// Base URL (Ollama or OpenAI-compatible).
    #[arg(long)]
    base_url: Option<String>,
    /// Output format expected from the model.
    #[arg(long, value_enum, default_value_t = OutFormat::Json)]
    format: OutFormat,
    /// Maximum characters per chunk.
    #[arg(long, default_value_t = 1000)]
    max_char_buffer: i32,
    /// Chunks per inference batch.
    #[arg(long, default_value_t = 10)]
    batch_length: usize,
    /// Sequential extraction passes (first-pass wins on overlap).
    #[arg(long, default_value_t = 1)]
    extraction_passes: usize,
    /// Trailing characters of the previous chunk injected as context.
    #[arg(long)]
    context_window_chars: Option<usize>,
    /// Parallel HTTP workers.
    #[arg(long, default_value_t = 10)]
    max_workers: usize,
    /// Write JSON to this path, or - for stdout (default).
    #[arg(long, default_value = "-")]
    output: String,
    /// Write a self-contained HTML review file for the first document.
    #[arg(long)]
    html: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutFormat {
    Json,
    Yaml,
}

impl From<OutFormat> for FormatType {
    fn from(value: OutFormat) -> Self {
        match value {
            OutFormat::Json => FormatType::Json,
            OutFormat::Yaml => FormatType::Yaml,
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("extract: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let text = read_input(&cli)?;
    let examples: Vec<ExampleData> = serde_json::from_str(&fs::read_to_string(&cli.examples)?)?;
    let provider = cli
        .provider
        .as_deref()
        .map(ProviderKind::parse)
        .transpose()?;
    let format_type = FormatType::from(cli.format);
    let resolved_kind = provider.or_else(|| extract_providers::resolve_provider(&cli.model).ok());
    let mut model = create_model(
        &cli.model,
        ModelOptions {
            provider,
            api_key: cli.api_key.clone(),
            base_url: cli.base_url.clone(),
            format_type: Some(format_type),
            temperature: None,
            timeout_secs: None,
            max_workers: Some(cli.max_workers),
        },
    )?;
    let options = ExtractOptions {
        max_char_buffer: cli.max_char_buffer,
        batch_length: cli.batch_length,
        extraction_passes: cli.extraction_passes,
        context_window_chars: cli.context_window_chars,
        format_type,
        fence_output: expects_fences(resolved_kind, format_type),
        prompt_validation_level: PromptValidationLevel::Warning,
        ..ExtractOptions::default()
    };
    let result = extract(
        ExtractInput::Text(text),
        &cli.prompt,
        &examples,
        model.as_mut(),
        options,
    )?;
    if let Some(html_path) = &cli.html {
        let html = match &result {
            ExtractResult::Document(doc) => extract_core::visualize(doc),
            ExtractResult::Documents(docs) => docs
                .first()
                .map(extract_core::visualize)
                .unwrap_or_else(|| "<p>No documents.</p>".into()),
        };
        fs::write(html_path, html)?;
    }
    let json = match result {
        ExtractResult::Document(doc) => serde_json::to_string_pretty(&doc)?,
        ExtractResult::Documents(docs) => serde_json::to_string_pretty(&docs)?,
    };
    write_output(&cli.output, &json)?;
    Ok(())
}

/// Ollama's `format: "json"` mode returns raw JSON, so fences are only expected
/// when that constraint is not active (YAML, or other providers).
fn expects_fences(kind: Option<ProviderKind>, format_type: FormatType) -> bool {
    !(kind == Some(ProviderKind::Ollama) && format_type == FormatType::Json)
}

fn read_input(cli: &Cli) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(path) = &cli.file {
        return Ok(fs::read_to_string(path)?);
    }
    match cli.text.as_deref() {
        Some("-") | None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            Ok(buf)
        }
        Some(t) => Ok(t.to_string()),
    }
}

fn write_output(path: &str, json: &str) -> Result<(), Box<dyn std::error::Error>> {
    if path == "-" {
        let mut stdout = io::stdout();
        stdout.write_all(json.as_bytes())?;
        stdout.write_all(b"\n")?;
        return Ok(());
    }
    fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_off_only_for_ollama_json() {
        assert!(!expects_fences(
            Some(ProviderKind::Ollama),
            FormatType::Json
        ));
        assert!(expects_fences(Some(ProviderKind::Ollama), FormatType::Yaml));
        assert!(expects_fences(Some(ProviderKind::Gemini), FormatType::Json));
        assert!(expects_fences(Some(ProviderKind::OpenAI), FormatType::Json));
        assert!(expects_fences(None, FormatType::Json));
    }

    #[test]
    fn api_key_flag_does_not_read_env() {
        // A Gemini key in the environment must not be forwarded to other
        // providers through the generic --api-key flag.
        std::env::set_var("LANGEXTRACT_API_KEY", "gemini-secret");
        let cli = Cli::try_parse_from([
            "extract",
            "-m",
            "gpt-4o-mini",
            "-p",
            "x",
            "-e",
            "ex.json",
            "--text",
            "hi",
        ])
        .unwrap();
        std::env::remove_var("LANGEXTRACT_API_KEY");
        assert_eq!(cli.api_key, None);
    }
}
