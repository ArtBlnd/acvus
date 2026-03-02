use std::collections::HashMap;

use crate::dsl::GenerationParams;
use crate::message::{Message, ModelResponse, ToolCall, ToolSpec};

use super::{HttpRequest, ProviderConfig};

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    cached_content: Option<&str>,
) -> HttpRequest {
    let body = match cached_content {
        Some(cache_name) => format_cached_body(messages, tools, generation, cache_name),
        None => format_body(messages, tools, generation),
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
    cache_config: &HashMap<String, serde_json::Value>,
) -> HttpRequest {
    let mut system_text = String::new();
    let mut contents = Vec::new();

    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(&m.content);
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

fn format_cached_body(
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    cache_name: &str,
) -> serde_json::Value {
    let contents: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(format_content)
        .collect();

    let mut body = serde_json::json!({
        "cachedContent": cache_name,
        "contents": contents,
    });

    let mut gen_config = serde_json::Map::new();
    if let Some(t) = generation.temperature { gen_config.insert("temperature".into(), serde_json::json!(t)); }
    if let Some(p) = generation.top_p { gen_config.insert("topP".into(), serde_json::json!(p)); }
    if let Some(k) = generation.top_k { gen_config.insert("topK".into(), serde_json::json!(k)); }
    if let Some(m) = generation.max_tokens { gen_config.insert("maxOutputTokens".into(), serde_json::json!(m)); }
    if !gen_config.is_empty() {
        body["generationConfig"] = serde_json::Value::Object(gen_config);
    }

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

        body["tools"] = serde_json::json!([{
            "function_declarations": declarations,
        }]);
    }

    body
}

fn format_body(messages: &[Message], tools: &[ToolSpec], generation: &GenerationParams) -> serde_json::Value {
    let mut system_text = String::new();
    let mut contents = Vec::new();

    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(&m.content);
        } else {
            contents.push(format_content(m));
        }
    }

    let mut body = serde_json::json!({
        "contents": contents,
    });

    let mut gen_config = serde_json::Map::new();
    if let Some(t) = generation.temperature { gen_config.insert("temperature".into(), serde_json::json!(t)); }
    if let Some(p) = generation.top_p { gen_config.insert("topP".into(), serde_json::json!(p)); }
    if let Some(k) = generation.top_k { gen_config.insert("topK".into(), serde_json::json!(k)); }
    if let Some(m) = generation.max_tokens { gen_config.insert("maxOutputTokens".into(), serde_json::json!(m)); }
    if !gen_config.is_empty() {
        body["generationConfig"] = serde_json::Value::Object(gen_config);
    }

    if !system_text.is_empty() {
        body["system_instruction"] = serde_json::json!({
            "parts": [{ "text": system_text }]
        });
    }

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

        body["tools"] = serde_json::json!([{
            "function_declarations": declarations,
        }]);
    }

    body
}

fn format_content(m: &Message) -> serde_json::Value {
    // Map roles: "assistant" -> "model", "tool" stays special
    let role = match m.role.as_str() {
        "assistant" => "model",
        other => other,
    };

    let mut parts = Vec::new();

    // Tool call results -> functionResponse parts
    if let Some(ref tool_call_id) = m.tool_call_id {
        parts.push(serde_json::json!({
            "functionResponse": {
                "name": tool_call_id,
                "response": {
                    "content": m.content,
                }
            }
        }));
        return serde_json::json!({ "role": "function", "parts": parts });
    }

    // Assistant message with tool calls -> functionCall parts
    if !m.tool_calls.is_empty() {
        for tc in &m.tool_calls {
            parts.push(serde_json::json!({
                "functionCall": {
                    "name": tc.name,
                    "args": tc.arguments,
                }
            }));
        }
        return serde_json::json!({ "role": role, "parts": parts });
    }

    // Plain text
    parts.push(serde_json::json!({ "text": m.content }));
    serde_json::json!({ "role": role, "parts": parts })
}

pub fn parse_response(json: &serde_json::Value) -> Result<ModelResponse, String> {
    let candidates = json
        .get("candidates")
        .and_then(|c| c.as_array())
        .ok_or("missing 'candidates' in response")?;

    let candidate = candidates.first().ok_or("empty candidates array")?;
    let content = candidate.get("content").ok_or("missing 'content' in candidate")?;
    let parts = content
        .get("parts")
        .and_then(|p| p.as_array())
        .ok_or("missing 'parts' in content")?;

    let mut tool_calls = Vec::new();
    let mut text_parts = Vec::new();

    for part in parts {
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("missing functionCall name")?
                .to_string();
            let arguments = fc.get("args").cloned().unwrap_or(serde_json::json!({}));
            // Gemini doesn't have explicit call IDs; generate from name
            let id = format!("call_{name}");
            tool_calls.push(ToolCall { id, name, arguments });
        } else if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            text_parts.push(text.to_string());
        }
    }

    if !tool_calls.is_empty() {
        Ok(ModelResponse::ToolCalls(tool_calls))
    } else {
        Ok(ModelResponse::Text(text_parts.join("")))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::dsl::GenerationParams;
    use crate::message::{Message, ToolSpec};

    use super::*;

    #[test]
    fn format_system_as_instruction() {
        let body = format_body(
            &[
                Message::text("system", "You are helpful."),
                Message::text("user", "Hello"),
            ],
            &[],
            &GenerationParams::default(),
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
    fn format_assistant_as_model() {
        let msg = Message::text("assistant", "I can help.");
        let content = format_content(&msg);
        assert_eq!(content["role"], "model");
    }

    #[test]
    fn format_with_tools() {
        let body = format_body(
            &[Message::text("user", "hi")],
            &[ToolSpec {
                name: "search".into(),
                description: "Search".into(),
                params: HashMap::from([("query".into(), "string".into())]),
            }],
            &GenerationParams::default(),
        );
        let decls = &body["tools"][0]["function_declarations"];
        assert_eq!(decls.as_array().unwrap().len(), 1);
        assert_eq!(decls[0]["name"], "search");
    }

    #[test]
    fn format_function_call_message() {
        let msg = Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"query": "rust"}),
            }],
            tool_call_id: None,
        };
        let content = format_content(&msg);
        assert_eq!(content["role"], "model");
        let parts = content["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "search");
    }

    #[test]
    fn format_function_response_message() {
        let msg = Message {
            role: "tool".into(),
            content: "result data".into(),
            tool_calls: Vec::new(),
            tool_call_id: Some("search".into()),
        };
        let content = format_content(&msg);
        assert_eq!(content["role"], "function");
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
        let resp = parse_response(&json).unwrap();
        assert!(matches!(resp, ModelResponse::Text(ref s) if s == "Hello there!"));
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
        let resp = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }
}
