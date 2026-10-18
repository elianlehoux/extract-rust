//! HTTP providers for [`extract_core::model::LanguageModel`].

pub mod gemini;
pub mod ollama;
pub mod openai;
pub mod router;

use extract_core::model::LanguageModel;
use extract_core::types::FormatType;
use extract_core::Error;

pub use gemini::GeminiLanguageModel;
pub use ollama::OllamaLanguageModel;
pub use openai::OpenAILanguageModel;
pub use router::{resolve_provider, ProviderKind};

/// Constructor knobs shared by [`create_model`].
#[derive(Debug, Clone, Default)]
pub struct ModelOptions {
    pub provider: Option<ProviderKind>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub format_type: Option<FormatType>,
    pub temperature: Option<f64>,
    pub timeout_secs: Option<u64>,
    pub max_workers: Option<usize>,
}

/// Build a provider from a model id (and optional explicit provider).
pub fn create_model(
    model_id: &str,
    options: ModelOptions,
) -> Result<Box<dyn LanguageModel>, Error> {
    let kind = match options.provider {
        Some(k) => k,
        None => resolve_provider(model_id)?,
    };
    match kind {
        ProviderKind::Ollama => Ok(Box::new(OllamaLanguageModel::new(
            model_id,
            options.base_url,
            options.format_type.unwrap_or(FormatType::Json),
            options.temperature,
            options.timeout_secs,
            options.max_workers.unwrap_or(10),
            options.api_key,
        )?)),
        ProviderKind::Gemini => Ok(Box::new(GeminiLanguageModel::new(
            model_id,
            options.api_key,
            options.format_type.unwrap_or(FormatType::Json),
            options.temperature.unwrap_or(0.0),
            options.timeout_secs.unwrap_or(120),
            options.max_workers.unwrap_or(10),
        )?)),
        ProviderKind::OpenAI => Ok(Box::new(OpenAILanguageModel::new(
            model_id,
            options.api_key,
            options.base_url,
            options.format_type.unwrap_or(FormatType::Json),
            options.temperature,
            options.timeout_secs.unwrap_or(120),
            options.max_workers.unwrap_or(10),
        )?)),
    }
}

fn env_first(keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
}

fn http_error(status: u16, body: &str, provider: &str) -> Error {
    Error::Inference(format!("{provider} HTTP {status}: {}", truncate(body, 300)))
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

fn infer_parallel<F>(
    prompts: &[String],
    max_workers: usize,
    f: F,
) -> Result<Vec<Vec<extract_core::types::ScoredOutput>>, Error>
where
    F: Fn(&str) -> Result<extract_core::types::ScoredOutput, Error> + Sync,
{
    use extract_core::types::ScoredOutput;
    if prompts.is_empty() {
        return Ok(Vec::new());
    }
    if prompts.len() == 1 || max_workers <= 1 {
        return prompts.iter().map(|p| f(p).map(|o| vec![o])).collect();
    }
    let workers = max_workers.min(prompts.len());
    let results: Vec<std::sync::Mutex<Option<Result<ScoredOutput, Error>>>> = (0..prompts.len())
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= prompts.len() {
                    break;
                }
                let out = f(&prompts[i]);
                *results[i].lock().unwrap() = Some(out);
            });
        }
    });
    let mut out = Vec::with_capacity(prompts.len());
    for slot in results {
        match slot.into_inner().unwrap() {
            Some(Ok(scored)) => out.push(vec![scored]),
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::Inference(
                    "Failed to process one or more prompts".into(),
                ))
            }
        }
    }
    Ok(out)
}
