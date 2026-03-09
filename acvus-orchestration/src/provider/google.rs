use rustc_hash::FxHashMap;

use crate::kind::GenerationParams;
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolSpec, Usage};

use super::{HttpRequest, ProviderConfig};

pub struct GoogleModel {
    config: ProviderConfig,
    model: String,
}

impl GoogleModel {
    pub fn new(config: ProviderConfig, model: String) -> Self {
        Self { config, model }
    }

    pub fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        generation: &GenerationParams,
        max_output_tokens: Option<u32>,
        cached_content: Option<&str>,
    ) -> HttpRequest {
        build_request(
            &self.config,
            &self.model,
            messages,
            tools,
            generation,
            max_output_tokens,
            cached_content,
        )
    }

    pub fn parse_response(
        &self,
        json: &serde_json::Value,
    ) -> Result<(ModelResponse, Usage), String> {
        parse_response(json)
    }

    pub fn build_count_tokens_request(&self, messages: &[Message]) -> Option<HttpRequest> {
        Some(build_count_tokens_request(
            &self.config,
            &self.model,
            messages,
        ))
    }

    pub fn parse_count_tokens_response(&self, json: &serde_json::Value) -> Result<u32, String> {
        parse_count_tokens_response(json)
    }
}

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
    cached_content: Option<&str>,
) -> HttpRequest {
    let body = match cached_content {
        Some(cache_name) => {
            format_cached_body(messages, tools, generation, max_output_tokens, cache_name)
        }
        None => format_body(messages, tools, generation, max_output_tokens),
    };
    let url = format!(
        "{}/v1beta/models/{}:generateContent",
        config.endpoint, model
    );
    HttpRequest {
        url,
        headers: vec![
            ("x-goog-api-key".into(), config.api_key.clone()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

pub fn build_cache_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    ttl: &str,
    cache_config: &FxHashMap<String, serde_json::Value>,
) -> HttpRequest {
    let mut system_text = String::new();
    let mut contents = Vec::new();

    for m in messages {
        if let Message::Content { role, content } = m
            && role == "system"
        {
            let Content::Text(text) = content else {
                panic!("system message must be text, got blob");
            };
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(text);
        } else {
            contents.push(format_content(m));
        }
    }

    let mut body = serde_json::json!({
        "model": format!("models/{model}"),
        "contents": contents,
        "ttl": ttl,
    });

    if !system_text.is_empty() {
        body["systemInstruction"] = serde_json::json!({
            "parts": [{ "text": system_text }]
        });
    }

    // Pass through provider-specific fields (e.g. display_name)
    for (k, v) in cache_config {
        body[k] = v.clone();
    }

    let url = format!("{}/v1beta/cachedContents", config.endpoint);
    HttpRequest {
        url,
        headers: vec![
            ("x-goog-api-key".into(), config.api_key.clone()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

pub fn parse_cache_response(json: &serde_json::Value) -> Result<String, String> {
    json.get("name")
        .and_then(|n| n.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "missing 'name' in cache response".into())
}

pub fn build_count_tokens_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
) -> HttpRequest {
    let mut system_text = String::new();
    let mut contents = Vec::new();

    for m in messages {
        if let Message::Content { role, content } = m
            && role == "system"
        {
            let Content::Text(text) = content else {
                panic!("system message must be text, got blob");
            };
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(text);
        } else {
            contents.push(format_content(m));
        }
    }

    let mut gen_request = serde_json::json!({
        "model": format!("models/{model}"),
        "contents": contents,
    });
    if !system_text.is_empty() {
        gen_request["systemInstruction"] = serde_json::json!({
            "parts": [{ "text": system_text }]
        });
    }
    let body = serde_json::json!({ "generateContentRequest": gen_request });

    let url = format!("{}/v1beta/models/{}:countTokens", config.endpoint, model);
    HttpRequest {
        url,
        headers: vec![
            ("x-goog-api-key".into(), config.api_key.clone()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

pub fn parse_count_tokens_response(json: &serde_json::Value) -> Result<u32, String> {
    json.get("totalTokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| "missing 'totalTokens' in count tokens response".into())
}

fn format_cached_body(
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
    cache_name: &str,
) -> serde_json::Value {
    let contents: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| !matches!(m, Message::Content { role, .. } if role == "system"))
        .map(format_content)
        .collect();

    let mut body = serde_json::json!({
        "cachedContent": cache_name,
        "contents": contents,
    });

    let mut gen_config = serde_json::Map::new();
    if let Some(t) = generation.temperature {
        gen_config.insert("temperature".into(), serde_json::json!(t));
    }
    if let Some(p) = generation.top_p {
        gen_config.insert("topP".into(), serde_json::json!(p));
    }
    if let Some(k) = generation.top_k {
        gen_config.insert("topK".into(), serde_json::json!(k));
    }
    if let Some(m) = max_output_tokens {
        gen_config.insert("maxOutputTokens".into(), serde_json::json!(m));
    }
    if !gen_config.is_empty() {
        body["generationConfig"] = serde_json::Value::Object(gen_config);
    }

    format_tools(&mut body, tools, generation.grounding);

    body
}

fn format_tools(body: &mut serde_json::Value, tools: &[ToolSpec], grounding: bool) {
    let mut tool_entries = Vec::new();

    if !tools.is_empty() {
        let declarations: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                let properties: serde_json::Map<String, serde_json::Value> = t
                    .params
                    .iter()
                    .map(|(name, type_name)| {
                        (name.clone(), serde_json::json!({ "type": type_name }))
                    })
                    .collect();

                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    }
                })
            })
            .collect();

        tool_entries.push(serde_json::json!({
            "function_declarations": declarations,
        }));
    }

    if grounding {
        tool_entries.push(serde_json::json!({ "google_search": {} }));
    }

    if !tool_entries.is_empty() {
        body["tools"] = serde_json::Value::Array(tool_entries);
    }
}

fn format_body(
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
) -> serde_json::Value {
    let mut system_text = String::new();
    let mut contents = Vec::new();

    for m in messages {
        if let Message::Content { role, content } = m
            && role == "system"
        {
            let Content::Text(text) = content else {
                panic!("system message must be text, got blob");
            };
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(text);
        } else {
            contents.push(format_content(m));
        }
    }

    let mut body = serde_json::json!({
        "contents": contents,
    });

    let mut gen_config = serde_json::Map::new();
    if let Some(t) = generation.temperature {
        gen_config.insert("temperature".into(), serde_json::json!(t));
    }
    if let Some(p) = generation.top_p {
        gen_config.insert("topP".into(), serde_json::json!(p));
    }
    if let Some(k) = generation.top_k {
        gen_config.insert("topK".into(), serde_json::json!(k));
    }
    if let Some(m) = max_output_tokens {
        gen_config.insert("maxOutputTokens".into(), serde_json::json!(m));
    }
    if !gen_config.is_empty() {
        body["generationConfig"] = serde_json::Value::Object(gen_config);
    }

    if !system_text.is_empty() {
        body["system_instruction"] = serde_json::json!({
            "parts": [{ "text": system_text }]
        });
    }

    format_tools(&mut body, tools, generation.grounding);

    body
}

fn format_content(m: &Message) -> serde_json::Value {
    match m {
        Message::Content { role, content } => match content {
            Content::Text(text) => {
                serde_json::json!({
                    "role": role,
                    "parts": [{ "text": text }],
                })
            }
            Content::Blob { mime_type, data } => {
                serde_json::json!({
                    "role": role,
                    "parts": [{
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": data,
                        }
                    }],
                })
            }
        },
        Message::ToolCalls(calls) => {
            let parts: Vec<serde_json::Value> = calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    })
                })
                .collect();
            serde_json::json!({ "role": "model", "parts": parts })
        }
        Message::ToolResult { call_id, content } => {
            serde_json::json!({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": call_id,
                        "response": { "content": content },
                    }
                }],
            })
        }
    }
}

pub fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), String> {
    let usage = parse_usage(json);

    let candidates = json
        .get("candidates")
        .and_then(|c| c.as_array())
        .ok_or("missing 'candidates' in response")?;

    let candidate = candidates.first().ok_or("empty candidates array")?;
    let content = candidate
        .get("content")
        .ok_or("missing 'content' in candidate")?;
    let parts = content
        .get("parts")
        .and_then(|p| p.as_array())
        .ok_or("missing 'parts' in content")?;

    let mut tool_calls = Vec::new();
    let mut content_parts = Vec::new();

    for part in parts {
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("missing functionCall name")?
                .to_string();
            let arguments = fc.get("args").cloned().unwrap_or(serde_json::json!({}));
            let id = format!("call_{name}");
            tool_calls.push(ToolCall {
                id,
                name,
                arguments,
            });
        } else if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            content_parts.push(Content::Text(text.to_string()));
        } else if let Some(inline) = part.get("inlineData") {
            let mime_type = inline
                .get("mimeType")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let data = inline
                .get("data")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            content_parts.push(Content::Blob { mime_type, data });
        }
    }

    if !tool_calls.is_empty() {
        Ok((ModelResponse::ToolCalls(tool_calls), usage))
    } else {
        let role = content
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("model")
            .to_string();
        let items = content_parts
            .into_iter()
            .map(|c| ContentItem {
                role: role.clone(),
                content: c,
            })
            .collect();
        Ok((ModelResponse::Content(items), usage))
    }
}

fn parse_usage(json: &serde_json::Value) -> Usage {
    let u = match json.get("usageMetadata") {
        Some(u) => u,
        None => return Usage::default(),
    };
    Usage {
        input_tokens: u
            .get("promptTokenCount")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        output_tokens: u
            .get("candidatesTokenCount")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {

    use crate::kind::GenerationParams;
    use crate::message::{Message, ToolSpec};

    use super::*;

    #[test]
    fn format_system_as_instruction() {
        let body = format_body(
            &[
                Message::Content {
                    role: "system".into(),
                    content: Content::Text("You are helpful.".into()),
                },
                Message::Content {
                    role: "user".into(),
                    content: Content::Text("Hello".into()),
                },
            ],
            &[],
            &GenerationParams::default(),
            None,
        );
        assert_eq!(
            body["system_instruction"]["parts"][0]["text"],
            "You are helpful."
        );
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn format_model_role() {
        let msg = Message::Content {
            role: "model".into(),
            content: Content::Text("I can help.".into()),
        };
        let content = format_content(&msg);
        assert_eq!(content["role"], "model");
    }

    #[test]
    fn format_with_tools() {
        let body = format_body(
            &[Message::Content {
                role: "user".into(),
                content: Content::Text("hi".into()),
            }],
            &[ToolSpec {
                name: "search".into(),
                description: "Search".into(),
                params: FxHashMap::from_iter([("query".into(), "string".into())]),
            }],
            &GenerationParams::default(),
            None,
        );
        let decls = &body["tools"][0]["function_declarations"];
        assert_eq!(decls.as_array().unwrap().len(), 1);
        assert_eq!(decls[0]["name"], "search");
    }

    #[test]
    fn format_function_call_message() {
        let msg = Message::ToolCalls(vec![ToolCall {
            id: "call_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"query": "rust"}),
        }]);
        let content = format_content(&msg);
        assert_eq!(content["role"], "model");
        let parts = content["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "search");
    }

    #[test]
    fn format_function_response_message() {
        let msg = Message::ToolResult {
            call_id: "search".into(),
            content: "result data".into(),
        };
        let content = format_content(&msg);
        assert_eq!(content["role"], "user");
        let parts = content["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionResponse"]["name"], "search");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{ "text": "Hello there!" }]
                }
            }]
        });
        let (resp, _) = parse_response(&json).unwrap();
        let ModelResponse::Content(items) = resp else {
            panic!("expected Content");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].role, "model");
        assert!(matches!(&items[0].content, Content::Text(s) if s == "Hello there!"));
    }

    #[test]
    fn parse_function_call_response() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "hello"}
                        }
                    }]
                }
            }]
        });
        let (resp, _) = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_usage_fields() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{ "text": "hi" }]
                }
            }],
            "usageMetadata": { "promptTokenCount": 20, "candidatesTokenCount": 12 }
        });
        let (_, usage) = parse_response(&json).unwrap();
        assert_eq!(usage.input_tokens, Some(20));
        assert_eq!(usage.output_tokens, Some(12));
    }

    #[test]
    fn count_tokens_request_format() {
        let config = ProviderConfig {
            api: crate::provider::ApiKind::Google,
            endpoint: "https://generativelanguage.googleapis.com".into(),
            api_key: "test-key".into(),
        };
        let req = build_count_tokens_request(
            &config,
            "gemini-2.0-flash",
            &[Message::Content {
                role: "user".into(),
                content: Content::Text("hello".into()),
            }],
        );
        assert!(req.url.contains(":countTokens"));
        assert!(req.url.contains("gemini-2.0-flash"));
        let gen_req = req.body.get("generateContentRequest").unwrap();
        assert!(gen_req.get("contents").is_some());
    }

    #[test]
    fn count_tokens_response_parsing() {
        let json = serde_json::json!({ "totalTokens": 42 });
        assert_eq!(parse_count_tokens_response(&json).unwrap(), 42);
    }
}
