//! Ollama `/api/generate` and gpt-oss `/api/chat`.

use std::time::Duration;

use extract_core::model::LanguageModel;
use extract_core::types::{FormatType, ScoredOutput};
use extract_core::Error;
use serde_json::{json, Value};

use crate::{http_error, infer_parallel};

const DEFAULT_URL: &str = "http://localhost:11434";
const DEFAULT_TEMPERATURE: f64 = 0.1;
const DEFAULT_TIMEOUT: u64 = 120;
const DEFAULT_KEEP_ALIVE: u64 = 300;
const DEFAULT_NUM_CTX: u64 = 2048;
const GPT_OSS_JSON_SYSTEM: &str = "Output a single JSON object matching the requested extraction format. Do not include code fences, prose, or reasoning.";

pub struct OllamaLanguageModel {
    model_id: String,
    base_url: String,
    format_type: FormatType,
    temperature: f64,
    max_workers: usize,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl OllamaLanguageModel {
    pub fn new(
        model_id: impl Into<String>,
        base_url: Option<String>,
        format_type: FormatType,
        temperature: Option<f64>,
        timeout_secs: Option<u64>,
        max_workers: usize,
        api_key: Option<String>,
    ) -> Result<Self, Error> {
        let timeout = Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT));
        let agent = ureq::AgentBuilder::new().timeout(timeout).build();
        let base_url = base_url
            .or_else(|| crate::env_first(&["OLLAMA_BASE_URL"]))
            .unwrap_or_else(|| DEFAULT_URL.to_string())
            .trim_end_matches('/')
            .to_string();
        Ok(Self {
            model_id: model_id.into(),
            base_url,
            format_type,
            temperature: temperature.unwrap_or(DEFAULT_TEMPERATURE),
            max_workers: max_workers.max(1),
            api_key,
            agent,
        })
    }

    fn is_gpt_oss(&self) -> bool {
        let id = self.model_id.to_ascii_lowercase();
        id == "gpt-oss" || id.starts_with("gpt-oss:")
    }

    fn headers(&self, req: ureq::Request) -> ureq::Request {
        let req = req
            .set("Content-Type", "application/json")
            .set("Accept", "application/json");
        if let Some(key) = &self.api_key {
            req.set("Authorization", &format!("Bearer {key}"))
        } else {
            req
        }
    }

    fn post_json(&self, path: &str, payload: Value) -> Result<Value, Error> {
        let url = format!("{}{path}", self.base_url);
        let req = self.headers(self.agent.post(&url));
        match req.send_json(payload) {
            Ok(resp) => resp
                .into_json()
                .map_err(|e| Error::Inference(format!("Ollama returned invalid JSON: {e}"))),
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                if code == 404 {
                    return Err(Error::Inference(format!(
                        "Ollama model not found: {}. Try: ollama run {}",
                        self.model_id, self.model_id
                    )));
                }
                Err(http_error(code, &body, "Ollama"))
            }
            Err(e) => Err(Error::Inference(format!("Ollama request failed: {e}"))),
        }
    }

    fn generate(&self, prompt: &str) -> Result<ScoredOutput, Error> {
        if self.is_gpt_oss() && self.format_type == FormatType::Json {
            return self.chat_gpt_oss(prompt);
        }
        let format = self.format_type.as_str();
        let payload = json!({
            "model": self.model_id,
            "prompt": prompt,
            "stream": false,
            "raw": false,
            "think": false,
            "format": format,
            "keep_alive": DEFAULT_KEEP_ALIVE,
            "options": {
                "temperature": self.temperature,
                "num_ctx": DEFAULT_NUM_CTX,
                "keep_alive": DEFAULT_KEEP_ALIVE,
            }
        });
        let body = self.post_json("/api/generate", payload)?;
        let output = extract_generate_text(&body)?;
        Ok(ScoredOutput::new(output))
    }

    fn chat_gpt_oss(&self, prompt: &str) -> Result<ScoredOutput, Error> {
        let payload = json!({
            "model": self.model_id,
            "stream": false,
            "think": false,
            "format": "json",
            "keep_alive": DEFAULT_KEEP_ALIVE,
            "messages": [
                {"role": "system", "content": GPT_OSS_JSON_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": DEFAULT_NUM_CTX,
            }
        });
        let body = self.post_json("/api/chat", payload)?;
        let output = extract_chat_text(&body)?;
        Ok(ScoredOutput::new(output))
    }
}

impl LanguageModel for OllamaLanguageModel {
    fn infer(&mut self, batch_prompts: &[String]) -> Result<Vec<Vec<ScoredOutput>>, Error> {
        infer_parallel(batch_prompts, self.max_workers, |p| self.generate(p))
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.model_id)
    }
}

pub(crate) fn extract_generate_text(body: &Value) -> Result<String, Error> {
    if let Some(s) = body.get("response").and_then(Value::as_str) {
        if !s.is_empty() {
            return Ok(s.to_string());
        }
    }
    if body.get("thinking").is_some() {
        return Err(Error::Inference(
            "Ollama returned an empty response with a thinking trace. Use think=false.".into(),
        ));
    }
    Err(Error::Inference(
        "Ollama response did not include generated text in the 'response' field.".into(),
    ))
}

pub(crate) fn extract_chat_text(body: &Value) -> Result<String, Error> {
    if let Some(s) = body
        .pointer("/message/content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return Ok(s.to_string());
    }
    if body.get("thinking").is_some() || body.pointer("/message/thinking").is_some() {
        return Err(Error::Inference(
            "Ollama returned an empty chat response with only a reasoning trace.".into(),
        ));
    }
    Err(Error::Inference(
        "Ollama chat response did not include generated text in the 'message.content' field."
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_text_from_response_field() {
        let v = json!({"response": "{\"extractions\":[]}"});
        assert_eq!(extract_generate_text(&v).unwrap(), "{\"extractions\":[]}");
    }

    #[test]
    fn generate_rejects_thinking_only() {
        let v = json!({"thinking": "hmm"});
        assert!(extract_generate_text(&v)
            .unwrap_err()
            .to_string()
            .contains("thinking"));
    }

    #[test]
    fn chat_text_from_message() {
        let v = json!({"message": {"content": "{\"ok\":true}"}});
        assert_eq!(extract_chat_text(&v).unwrap(), "{\"ok\":true}");
    }

    #[test]
    fn generate_against_local_http() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = r#"{"response":"{\"extractions\":[{\"person\":\"Ada\"}]}"}"#.to_string();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            // A single read races the client: responding before the request
            // body arrives makes ureq fail with EINVAL on macOS.
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                .unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 1024];
            loop {
                match stream.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut =>
                    {
                        break;
                    }
                    Err(e) => panic!("read request: {e}"),
                }
                let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let header_end = end + 4;
                let headers = String::from_utf8_lossy(&buf[..header_end]);
                let len = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                });
                match len {
                    Some(len) if buf.len() >= header_end + len => break,
                    None => break,
                    Some(_) => continue,
                }
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
        });
        let mut model = OllamaLanguageModel::new(
            "gemma2:2b",
            Some(format!("http://{addr}")),
            FormatType::Json,
            Some(0.0),
            Some(5),
            1,
            None,
        )
        .unwrap();
        let out = model.infer(&["hello".into()]).unwrap();
        assert!(out[0][0].output.as_ref().unwrap().contains("Ada"));
        server.join().unwrap();
    }
}
