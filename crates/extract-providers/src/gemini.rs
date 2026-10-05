//! Gemini `generateContent` REST (API key). Vertex/batch stay out of the default path.

use std::thread;
use std::time::Duration;

use extract_core::model::LanguageModel;
use extract_core::types::{FormatType, ScoredOutput};
use extract_core::Error;
use serde_json::{json, Value};

use crate::{env_first, http_error, infer_parallel};

const DEFAULT_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";
const RETRYABLE: [u16; 6] = [408, 429, 500, 502, 503, 504];

pub struct GeminiLanguageModel {
    model_id: String,
    api_key: String,
    format_type: FormatType,
    temperature: f64,
    max_workers: usize,
    max_retries: u32,
    retry_delay: Duration,
    max_retry_delay: Duration,
    agent: ureq::Agent,
}

impl GeminiLanguageModel {
    pub fn new(
        model_id: impl Into<String>,
        api_key: Option<String>,
        format_type: FormatType,
        temperature: f64,
        timeout_secs: u64,
        max_workers: usize,
    ) -> Result<Self, Error> {
        let api_key = api_key
            .or_else(|| env_first(&["GEMINI_API_KEY", "LANGEXTRACT_API_KEY", "GOOGLE_API_KEY"]))
            .ok_or_else(|| {
                Error::Config(
                    "Gemini models require an API key via --api-key or GEMINI_API_KEY / LANGEXTRACT_API_KEY."
                        .into(),
                )
            })?;
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout_secs))
            .build();
        Ok(Self {
            model_id: model_id.into(),
            api_key,
            format_type,
            temperature,
            max_workers: max_workers.max(1),
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            max_retry_delay: Duration::from_secs(16),
            agent,
        })
    }

    fn generate(&self, prompt: &str) -> Result<ScoredOutput, Error> {
        let mut gen = json!({ "temperature": self.temperature });
        if self.format_type == FormatType::Json {
            gen["responseMimeType"] = json!("application/json");
        }
        let payload = json!({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen,
        });
        let url = format!(
            "{DEFAULT_ENDPOINT}/models/{}:generateContent?key={}",
            self.model_id, self.api_key
        );
        let body = post_with_retry(&self.agent, &url, payload, self.max_retries, self.retry_delay, self.max_retry_delay)?;
        let text = extract_gemini_text(&body)?;
        Ok(ScoredOutput::new(text))
    }
}

impl LanguageModel for GeminiLanguageModel {
    fn infer(&mut self, batch_prompts: &[String]) -> Result<Vec<Vec<ScoredOutput>>, Error> {
        infer_parallel(batch_prompts, self.max_workers, |p| self.generate(p))
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.model_id)
    }
}

fn post_with_retry(
    agent: &ureq::Agent,
    url: &str,
    payload: Value,
    max_retries: u32,
    retry_delay: Duration,
    max_retry_delay: Duration,
) -> Result<Value, Error> {
    let mut delay = retry_delay;
    for attempt in 0..=max_retries {
        match agent
            .post(url)
            .set("Content-Type", "application/json")
            .send_json(payload.clone())
        {
            Ok(resp) => {
                return resp
                    .into_json()
                    .map_err(|e| Error::Inference(format!("Gemini returned invalid JSON: {e}")));
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                if RETRYABLE.contains(&code) && attempt < max_retries {
                    thread::sleep(delay);
                    delay = (delay * 2).min(max_retry_delay);
                    continue;
                }
                return Err(http_error(code, &body, "Gemini"));
            }
            Err(e) => {
                if attempt < max_retries {
                    thread::sleep(delay);
                    delay = (delay * 2).min(max_retry_delay);
                    continue;
                }
                return Err(Error::Inference(format!("Gemini request failed: {e}")));
            }
        }
    }
    Err(Error::Inference("Gemini retries exhausted.".into()))
}

pub(crate) fn extract_gemini_text(body: &Value) -> Result<String, Error> {
    if let Some(reason) = body.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Err(Error::Inference(format!(
            "Gemini blocked the prompt ({reason})."
        )));
    }
    let parts = body
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array);
    let mut text = String::new();
    if let Some(parts) = parts {
        for part in parts {
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                text.push_str(t);
            }
        }
    }
    if text.is_empty() {
        let finish = body
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        return Err(Error::Inference(format!(
            "Gemini response contained no text (finishReason={finish})."
        )));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_parts_text() {
        let v = json!({
            "candidates": [{"content": {"parts": [{"text": "{\"a\":1}"}]}, "finishReason": "STOP"}]
        });
        assert_eq!(extract_gemini_text(&v).unwrap(), "{\"a\":1}");
    }

    #[test]
    fn blocked_prompt() {
        let v = json!({"promptFeedback": {"blockReason": "SAFETY"}});
        assert!(extract_gemini_text(&v).unwrap_err().to_string().contains("SAFETY"));
    }
}
