//! Map a model id to a built-in provider.

use extract_core::Error;
use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Gemini,
    OpenAI,
    Ollama,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemini => "gemini",
            Self::OpenAI => "openai",
            Self::Ollama => "ollama",
        }
    }

    pub fn parse(name: &str) -> Result<Self, Error> {
        match name.to_ascii_lowercase().as_str() {
            "gemini" => Ok(Self::Gemini),
            "openai" => Ok(Self::OpenAI),
            "ollama" => Ok(Self::Ollama),
            other => Err(Error::Config(format!("Unknown provider '{other}'."))),
        }
    }
}

struct Entry {
    pattern: Regex,
    kind: ProviderKind,
}

static ENTRIES: LazyLock<Vec<Entry>> = LazyLock::new(|| {
    let mut out = Vec::new();
    out.push(Entry {
        pattern: Regex::new(r"^gemini").expect("gemini pattern"),
        kind: ProviderKind::Gemini,
    });
    for p in [r"^gpt-3\.5", r"^gpt-4", r"^gpt4\.", r"^gpt-5", r"^gpt5\.", r"^o[1-9]"] {
        out.push(Entry {
            pattern: Regex::new(p).expect("openai pattern"),
            kind: ProviderKind::OpenAI,
        });
    }
    for p in [
        r"^gemma",
        r"^llama",
        r"^mistral",
        r"^mixtral",
        r"^phi",
        r"^qwen",
        r"^deepseek",
        r"^command-r",
        r"^starcoder",
        r"^codellama",
        r"^codegemma",
        r"^tinyllama",
        r"^wizardcoder",
        r"^gpt-oss",
        r"^meta-llama/[Ll]lama",
        r"^google/gemma",
        r"^mistralai/[Mm]istral",
        r"^mistralai/[Mm]ixtral",
        r"^microsoft/phi",
        r"^Qwen/",
        r"^deepseek-ai/",
        r"^bigcode/starcoder",
        r"^codellama/",
        r"^TinyLlama/",
        r"^WizardLM/",
    ] {
        out.push(Entry {
            pattern: Regex::new(p).expect("ollama pattern"),
            kind: ProviderKind::Ollama,
        });
    }
    out
});

/// Resolve `model_id` against built-in patterns. First match wins (Gemini/OpenAI
/// are registered before the Ollama catch-all of local names).
pub fn resolve_provider(model_id: &str) -> Result<ProviderKind, Error> {
    ENTRIES
        .iter()
        .find(|e| e.pattern.is_match(model_id))
        .map(|e| e.kind)
        .ok_or_else(|| {
            Error::Config(format!(
                "No provider found matching: {model_id:?}. Pass --provider ollama|gemini|openai."
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_known_ids() {
        assert_eq!(resolve_provider("gemini-2.5-flash").unwrap(), ProviderKind::Gemini);
        assert_eq!(resolve_provider("gpt-4o-mini").unwrap(), ProviderKind::OpenAI);
        assert_eq!(resolve_provider("o3-mini").unwrap(), ProviderKind::OpenAI);
        assert_eq!(resolve_provider("gemma2:2b").unwrap(), ProviderKind::Ollama);
        assert_eq!(resolve_provider("llama3.1:8b").unwrap(), ProviderKind::Ollama);
        assert_eq!(resolve_provider("gpt-oss:20b").unwrap(), ProviderKind::Ollama);
        assert_eq!(
            resolve_provider("meta-llama/Llama-3.2-1B-Instruct").unwrap(),
            ProviderKind::Ollama
        );
    }

    #[test]
    fn unknown_id_errors() {
        assert!(resolve_provider("not-a-real-model").is_err());
    }
}
