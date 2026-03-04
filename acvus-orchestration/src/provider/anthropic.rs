use crate::dsl::GenerationParams;
use crate::message::{Message, ModelResponse, ToolCall, ToolSpec, Usage};

use super::{HttpRequest, LlmModel, ProviderConfig};

pub struct AnthropicModel {
    config: ProviderConfig,
    model: String,
}

impl AnthropicModel {
    pub fn new(config: ProviderConfig, model: String) -> Self {
        Self { config, model }
    }
}

impl LlmModel for AnthropicModel {
    fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        generation: &GenerationParams,
        cached_content: Option<&str>,
    ) -> HttpRequest {
        let _ = cached_content;
        build_request(&self.config, &self.model, messages, tools, generation)
    }

    fn parse_response(&self, json: &serde_json::Value) -> Result<(ModelResponse, Usage), String> {
        parse_response(json)
    }

    fn build_count_tokens_request(&self, messages: &[Message]) -> Option<HttpRequest> {
        Some(build_count_tokens_request(
            &self.config,
            &self.model,
            messages,
        ))
    }

    fn parse_count_tokens_response(&self, json: &serde_json::Value) -> Result<u32, String> {
        parse_count_tokens_response(json)
    }
}

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
) -> HttpRequest {
    let body = format_body(model, messages, tools, generation);
    let url = format!("{}/v1/messages", config.endpoint);
    HttpRequest {
        url,
        headers: vec![
            ("x-api-key".into(), config.api_key.clone()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

pub fn build_count_tokens_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
) -> HttpRequest {
    let mut system_text = String::new();
    let mut msgs = Vec::new();

    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(&m.content);
        } else {
            msgs.push(format_message(m));
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": msgs,
    });
    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
    }

    let url = format!("{}/v1/messages/count_tokens", config.endpoint);
    HttpRequest {
        url,
        headers: vec![
            ("x-api-key".into(), config.api_key.clone()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("anthropic-beta".into(), "token-counting-2024-11-01".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

pub fn parse_count_tokens_response(json: &serde_json::Value) -> Result<u32, String> {
    json.get("input_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| "missing 'input_tokens' in count tokens response".into())
}

fn format_body(
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
) -> serde_json::Value {
    let mut system_text = String::new();
    let mut msgs = Vec::new();

    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(&m.content);
        } else {
            msgs.push(format_message(m));
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": msgs,
        "max_tokens": generation.max_tokens.unwrap_or(4096),
    });

    if let Some(t) = generation.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = generation.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(k) = generation.top_k {
        body["top_k"] = serde_json::json!(k);
    }

    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
    }

    if !tools.is_empty() {
        let tool_specs: Vec<serde_json::Value> = tools
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
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                    }
                })
            })
            .collect();

        body["tools"] = serde_json::Value::Array(tool_specs);
    }

    body
}

fn format_message(m: &Message) -> serde_json::Value {
    // Tool call results -> user message with tool_result content blocks
    if let Some(ref tool_call_id) = m.tool_call_id {
        return serde_json::json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": m.content,
            }]
        });
    }

    // Assistant message with tool calls -> tool_use content blocks
    if !m.tool_calls.is_empty() {
        let mut content: Vec<serde_json::Value> = Vec::new();

        if !m.content.is_empty() {
            content.push(serde_json::json!({
                "type": "text",
                "text": m.content,
            }));
        }

        for tc in &m.tool_calls {
            content.push(serde_json::json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }));
        }

        return serde_json::json!({
            "role": "assistant",
            "content": content,
        });
    }

    // Plain text message
    serde_json::json!({
        "role": m.role,
        "content": m.content,
    })
}

pub fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), String> {
    let usage = parse_usage(json);

    let content = json
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or("missing 'content' array in response")?;

    let mut tool_calls = Vec::new();
    let mut text_parts = Vec::new();

    for block in content {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool_use id")?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool_use name")?
                    .to_string();
                let arguments = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            _ => {}
        }
    }

    if !tool_calls.is_empty() {
        Ok((ModelResponse::ToolCalls(tool_calls), usage))
    } else {
        Ok((ModelResponse::Text(text_parts.join("")), usage))
    }
}

fn parse_usage(json: &serde_json::Value) -> Usage {
    let u = match json.get("usage") {
        Some(u) => u,
        None => return Usage::default(),
    };
    Usage {
        input_tokens: u
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        output_tokens: u
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::dsl::GenerationParams;
    use crate::message::{Message, ToolSpec};

    use super::*;

    #[test]
    fn format_system_separated() {
        let body = format_body(
            "claude-sonnet-4-6",
            &[
                Message::text("system", "You are helpful."),
                Message::text("user", "Hello"),
            ],
            &[],
            &GenerationParams::default(),
        );
        assert_eq!(body["system"], "You are helpful.");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn format_with_tools() {
        let body = format_body(
            "claude-sonnet-4-6",
            &[Message::text("user", "hi")],
            &[ToolSpec {
                name: "search".into(),
                description: "Search".into(),
                params: HashMap::from([("query".into(), "string".into())]),
            }],
            &GenerationParams::default(),
        );
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "search");
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn format_tool_use_message() {
        let msg = Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "toolu_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"query": "rust"}),
            }],
            tool_call_id: None,
        };
        let formatted = format_message(&msg);
        let content = formatted["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "toolu_1");
    }

    #[test]
    fn format_tool_result_message() {
        let msg = Message {
            role: "tool".into(),
            content: "result data".into(),
            tool_calls: Vec::new(),
            tool_call_id: Some("toolu_1".into()),
        };
        let formatted = format_message(&msg);
        assert_eq!(formatted["role"], "user");
        let content = formatted["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "content": [{
                "type": "text",
                "text": "Hello there!"
            }],
            "stop_reason": "end_turn"
        });
        let (resp, _) = parse_response(&json).unwrap();
        assert!(matches!(resp, ModelResponse::Text(ref s) if s == "Hello there!"));
    }

    #[test]
    fn parse_tool_use_response() {
        let json = serde_json::json!({
            "content": [
                { "type": "text", "text": "Let me search." },
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                    "input": {"query": "hello"}
                }
            ],
            "stop_reason": "tool_use"
        });
        let (resp, _) = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].id, "toolu_123");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn parse_usage_fields() {
        let json = serde_json::json!({
            "content": [{ "type": "text", "text": "hi" }],
            "usage": { "input_tokens": 15, "output_tokens": 8 }
        });
        let (_, usage) = parse_response(&json).unwrap();
        assert_eq!(usage.input_tokens, Some(15));
        assert_eq!(usage.output_tokens, Some(8));
    }

    #[test]
    fn count_tokens_request_format() {
        let config = ProviderConfig {
            api: crate::provider::ApiKind::Anthropic,
            endpoint: "https://api.anthropic.com".into(),
            api_key: "test-key".into(),
        };
        let req = build_count_tokens_request(
            &config,
            "claude-sonnet-4-6",
            &[
                Message::text("system", "You are helpful."),
                Message::text("user", "hello"),
            ],
        );
        assert!(req.url.contains("/v1/messages/count_tokens"));
        assert_eq!(req.body["model"], "claude-sonnet-4-6");
        assert_eq!(req.body["system"], "You are helpful.");
        assert_eq!(req.body["messages"].as_array().unwrap().len(), 1);
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "anthropic-beta" && v.contains("token-counting"))
        );
    }

    #[test]
    fn count_tokens_response_parsing() {
        let json = serde_json::json!({ "input_tokens": 37 });
        assert_eq!(parse_count_tokens_response(&json).unwrap(), 37);
    }
}
