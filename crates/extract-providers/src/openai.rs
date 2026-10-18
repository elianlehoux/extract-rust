//! OpenAI Chat Completions REST.

use std::time::Duration;

use extract_core::model::LanguageModel;
use extract_core::types::{FormatType, ScoredOutput};
use extract_core::Error;
use serde_json::{json, Value};

use crate::{env_first, http_error, infer_parallel};

const DEFAULT_URL: &str = "https://api.openai.com/v1";

pub struct OpenAILanguageModel {
    model_id: String,
    api_key: String,
    base_url: String,
    format_type: FormatType,
    temperature: Option<f64>,
    max_workers: usize,
    agent: ureq::Agent,
}

impl OpenAILanguageModel {
    pub fn new(
        model_id: impl Into<String>,
        api_key: Option<String>,
        base_url: Option<String>,
        format_type: FormatType,
        temperature: Option<f64>,
        timeout_secs: u64,
        max_workers: usize,
    ) -> Result<Self, Error> {
        let api_key = api_key
            .or_else(|| env_first(&["OPENAI_API_KEY"]))
            .ok_or_else(|| Error::Config("OpenAI API key not provided.".into()))?;
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout_secs))
            .build();
        Ok(Self {
            model_id: model_id.into(),
            api_key,
            base_url: resolve_base_url(base_url, env_first(&["OPENAI_BASE_URL"])),
            format_type,
            temperature,
            max_workers: max_workers.max(1),
            agent,
        })
    }

    fn complete(&self, prompt: &str) -> Result<ScoredOutput, Error> {
        let system = match self.format_type {
            FormatType::Json => "You are a helpful assistant that responds in JSON format.",
            FormatType::Yaml => "You are a helpful assistant that responds in YAML format.",
        };
        let mut payload = json!({
            "model": self.model_id,
            "n": 1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        });
        if let Some(t) = self.temperature {
            payload["temperature"] = json!(t);
        }
        if self.format_type == FormatType::Json {
            payload["response_format"] = json!({"type": "json_object"});
        }
        let url = format!("{}/chat/completions", self.base_url);
        let resp = match self
            .agent
            .post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(payload)
        {
            Ok(r) => r,
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                return Err(http_error(code, &body, "OpenAI"));
            }
            Err(e) => return Err(Error::Inference(format!("OpenAI request failed: {e}"))),
        };
        let body: Value = resp
            .into_json()
            .map_err(|e| Error::Inference(format!("OpenAI returned invalid JSON: {e}")))?;
        Ok(ScoredOutput::new(extract_openai_text(&body)?))
    }
}

impl LanguageModel for OpenAILanguageModel {
    fn infer(&mut self, batch_prompts: &[String]) -> Result<Vec<Vec<ScoredOutput>>, Error> {
        infer_parallel(batch_prompts, self.max_workers, |p| self.complete(p))
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.model_id)
    }
}

/// Explicit `base_url` wins, then `OPENAI_BASE_URL`, then the public API.
fn resolve_base_url(explicit: Option<String>, from_env: Option<String>) -> String {
    explicit
        .or(from_env)
        .unwrap_or_else(|| DEFAULT_URL.to_string())
        .trim_end_matches('/')
        .to_string()
}

pub(crate) fn extract_openai_text(body: &Value) -> Result<String, Error> {
    let choice = body
        .pointer("/choices/0")
        .ok_or_else(|| Error::Inference("OpenAI response contained no choices.".into()))?;
    if let Some(refusal) = choice.pointer("/message/refusal").and_then(Value::as_str) {
        if !refusal.is_empty() {
            return Err(Error::Inference(format!(
                "OpenAI refused the request: {refusal}"
            )));
        }
    }
    if let Some(content) = choice
        .pointer("/message/content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return Ok(content.to_string());
    }
    let finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let hint = if finish == "length" {
        " The output token budget was exhausted before any text was emitted; consider raising max_output_tokens."
    } else {
        ""
    };
    Err(Error::Inference(format!(
        "OpenAI response contained no message content (finish_reason={finish}).{hint}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_message_content() {
        let v =
            json!({"choices": [{"message": {"content": "{\"ok\":1}"}, "finish_reason": "stop"}]});
        assert_eq!(extract_openai_text(&v).unwrap(), "{\"ok\":1}");
    }

    #[test]
    fn refusal() {
        let v = json!({"choices": [{"message": {"refusal": "nope"}, "finish_reason": "stop"}]});
        assert!(extract_openai_text(&v)
            .unwrap_err()
            .to_string()
            .contains("refused"));
    }

    #[test]
    fn base_url_precedence() {
        assert_eq!(resolve_base_url(None, None), DEFAULT_URL);
        assert_eq!(
            resolve_base_url(None, Some("http://localhost:8000/v1/".into())),
            "http://localhost:8000/v1"
        );
        assert_eq!(
            resolve_base_url(Some("http://a/v1".into()), Some("http://b/v1".into())),
            "http://a/v1"
        );
    }

    #[test]
    fn openai_compatible_server_gets_bearer_and_path() {
        let (base, server) = crate::test_server::serve_once(
            200,
            r#"{"choices":[{"message":{"content":"{\"extractions\":[]}"},"finish_reason":"stop"}]}"#,
        );
        let mut m = OpenAILanguageModel::new(
            "gpt-4o-mini",
            Some("sk-test".into()),
            Some(format!("{base}/v1")),
            FormatType::Json,
            None,
            5,
            1,
        )
        .unwrap();
        m.infer(&["hi".into()]).unwrap();
        let request = server.join().unwrap();
        assert!(
            request.starts_with("POST /v1/chat/completions"),
            "{request}"
        );
        assert!(request
            .to_ascii_lowercase()
            .contains("authorization: bearer sk-test"));
    }
}
