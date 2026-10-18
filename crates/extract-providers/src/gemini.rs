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
    endpoint: String,
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
            endpoint: DEFAULT_ENDPOINT.to_string(),
            agent,
        })
    }

    /// Override the REST endpoint (e.g. a proxy or a local mock server).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into().trim_end_matches('/').to_string();
        self
    }

    /// Override retry behaviour (mainly for tests).
    pub fn with_retries(mut self, max_retries: u32, retry_delay: Duration) -> Self {
        self.max_retries = max_retries;
        self.retry_delay = retry_delay;
        self
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
        // The key travels in a header, never in the URL, so it cannot leak
        // through transport error messages (which include the URL).
        let url = format!("{}/models/{}:generateContent", self.endpoint, self.model_id);
        let body = post_with_retry(
            &self.agent,
            &url,
            &self.api_key,
            payload,
            self.max_retries,
            self.retry_delay,
            self.max_retry_delay,
        )?;
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
    api_key: &str,
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
            .set("x-goog-api-key", api_key)
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
                return Err(http_error(code, &redact(&body, api_key), "Gemini"));
            }
            Err(e) => {
                if attempt < max_retries {
                    thread::sleep(delay);
                    delay = (delay * 2).min(max_retry_delay);
                    continue;
                }
                return Err(Error::Inference(format!(
                    "Gemini request failed: {}",
                    redact(&e.to_string(), api_key)
                )));
            }
        }
    }
    Err(Error::Inference("Gemini retries exhausted.".into()))
}

/// Defense in depth: never echo the API key back in an error message.
fn redact(message: &str, api_key: &str) -> String {
    if api_key.is_empty() {
        return message.to_string();
    }
    message.replace(api_key, "[REDACTED]")
}

pub(crate) fn extract_gemini_text(body: &Value) -> Result<String, Error> {
    if let Some(reason) = body
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
    {
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
        assert!(extract_gemini_text(&v)
            .unwrap_err()
            .to_string()
            .contains("SAFETY"));
    }

    #[test]
    fn api_key_sent_as_header_not_in_url() {
        let (base, server) = crate::test_server::serve_once(
            200,
            r#"{"candidates":[{"content":{"parts":[{"text":"{\"extractions\":[]}"}]}}]}"#,
        );
        let mut m = GeminiLanguageModel::new(
            "gemini-2.5-flash",
            Some("SECRET_KEY_123".into()),
            FormatType::Json,
            0.0,
            5,
            1,
        )
        .unwrap()
        .with_endpoint(base);
        let out = m.infer(&["hi".into()]).unwrap();
        assert!(out[0][0].output.as_ref().unwrap().contains("extractions"));
        let request = server.join().unwrap();
        let request_line = request.lines().next().unwrap();
        assert!(!request_line.contains("SECRET_KEY_123"), "{request_line}");
        assert!(!request_line.contains("key="), "{request_line}");
        assert!(request
            .to_ascii_lowercase()
            .contains("x-goog-api-key: secret_key_123"));
    }

    #[test]
    fn api_key_not_in_transport_error() {
        // Nothing listens on this port, so the request fails at transport level.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut m = GeminiLanguageModel::new(
            "gemini-2.5-flash",
            Some("SECRET_KEY_123".into()),
            FormatType::Json,
            0.0,
            2,
            1,
        )
        .unwrap()
        .with_endpoint(format!("http://127.0.0.1:{port}"))
        .with_retries(0, Duration::from_millis(1));
        let err = m.infer(&["hi".into()]).unwrap_err().to_string();
        assert!(err.contains("Gemini request failed"), "{err}");
        assert!(!err.contains("SECRET_KEY_123"), "{err}");
    }

    #[test]
    fn api_key_redacted_from_http_error_body() {
        let (base, server) =
            crate::test_server::serve_once(400, r#"{"error":"bad key SECRET_KEY_123"}"#);
        let mut m = GeminiLanguageModel::new(
            "gemini-2.5-flash",
            Some("SECRET_KEY_123".into()),
            FormatType::Json,
            0.0,
            5,
            1,
        )
        .unwrap()
        .with_endpoint(base);
        let err = m.infer(&["hi".into()]).unwrap_err().to_string();
        server.join().unwrap();
        assert!(err.contains("HTTP 400"), "{err}");
        assert!(!err.contains("SECRET_KEY_123"), "{err}");
        assert!(err.contains("[REDACTED]"), "{err}");
    }
}
