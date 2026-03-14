use rustc_hash::FxHashMap;

use crate::kind::{GenerationParams, ThinkingConfig};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolCallExtras, ToolSpec, Usage};

use super::schema::google as schema;
use super::{HttpRequest, ProviderConfig, ProviderError, split_system_messages};

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
    ) -> Result<HttpRequest, ProviderError> {
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
    ) -> Result<(ModelResponse, Usage), ProviderError> {
        parse_response(json)
    }

    pub fn build_count_tokens_request(&self, messages: &[Message]) -> Option<HttpRequest> {
        Some(build_count_tokens_request(
            &self.config,
            &self.model,
            messages,
        ))
    }

    pub fn parse_count_tokens_response(
        &self,
        json: &serde_json::Value,
    ) -> Result<u32, ProviderError> {
        parse_count_tokens_response(json)
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Convert a `&Message` into a `schema::Content` for Gemini.
fn convert_content(m: &Message) -> schema::Content {
    match m {
        Message::Content { role, content } => match content {
            Content::Text(text) => schema::Content {
                role: role.clone(),
                parts: vec![schema::Part::Text { text: text.clone() }],
            },
            Content::Blob { mime_type, data } => schema::Content {
                role: role.clone(),
                parts: vec![schema::Part::InlineData {
                    inline_data: schema::InlineData {
                        mime_type: mime_type.clone(),
                        data: data.clone(),
                    },
                }],
            },
        },
        Message::ToolCalls(calls) => {
            let parts = calls
                .iter()
                .map(|tc| {
                    let thought_signature = match &tc.extras {
                        Some(ToolCallExtras::Gemini { thought_signature }) => {
                            thought_signature.clone()
                        }
                        _ => None,
                    };
                    schema::Part::FunctionCall {
                        function_call: schema::FunctionCallPayload {
                            name: tc.name.clone(),
                            args: tc.arguments.clone(),
                        },
                        thought_signature,
                    }
                })
                .collect();
            schema::Content {
                role: "model".into(),
                parts,
            }
        }
        Message::ToolResult { call_id, content } => schema::Content {
            role: "user".into(),
            parts: vec![schema::Part::FunctionResponse {
                function_response: schema::FunctionResponsePayload {
                    name: call_id.clone(),
                    response: schema::FunctionResponseContent {
                        content: content.clone(),
                    },
                },
            }],
        },
    }
}

/// Build a `schema::GenerationConfig` from generation params and optional max_output_tokens.
fn build_generation_config(
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
) -> Option<schema::GenerationConfig> {
    let thinking_config = match &generation.thinking {
        None | Some(ThinkingConfig::Off) => None,
        Some(ThinkingConfig::Low) => Some(schema::ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some("LOW".into()),
        }),
        Some(ThinkingConfig::Medium) => Some(schema::ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some("MEDIUM".into()),
        }),
        Some(ThinkingConfig::High) => Some(schema::ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some("HIGH".into()),
        }),
        Some(ThinkingConfig::Custom(n)) => Some(schema::ThinkingConfig {
            thinking_budget: Some(*n),
            thinking_level: None,
        }),
    };

    let has_any = generation.temperature.is_some()
        || generation.top_p.is_some()
        || generation.top_k.is_some()
        || max_output_tokens.is_some()
        || thinking_config.is_some();

    if !has_any {
        return None;
    }

    Some(schema::GenerationConfig {
        temperature: generation.temperature,
        top_p: generation.top_p,
        top_k: generation.top_k,
        max_output_tokens,
        thinking_config,
    })
}

/// Build tool declarations from `ToolSpec` slice and grounding flag.
fn build_tools(tools: &[ToolSpec], grounding: bool) -> Option<Vec<schema::ToolDeclaration>> {
    let mut entries = Vec::new();

    if !tools.is_empty() {
        let declarations: Vec<schema::FunctionDecl> = tools
            .iter()
            .map(|t| {
                let properties: serde_json::Map<String, serde_json::Value> = t
                    .params
                    .iter()
                    .filter_map(|(name, param)| {
                        let gemini_ty = to_gemini_type(&param.ty)?;
                        let prop = schema::GeminiPropertySchema {
                            ty: gemini_ty.into(),
                            description: param.description.clone(),
                        };
                        Some((name.clone(), serde_json::to_value(prop).unwrap()))
                    })
                    .collect();

                let required: Vec<String> = t.params.keys().cloned().collect();

                schema::FunctionDecl {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: schema::GeminiSchema {
                        schema_type: "OBJECT".into(),
                        properties,
                        required,
                    },
                }
            })
            .collect();

        entries.push(schema::ToolDeclaration::Functions {
            function_declarations: declarations,
        });
    }

    if grounding {
        entries.push(schema::ToolDeclaration::GoogleSearch {
            google_search: schema::GoogleSearchConfig {},
        });
    }

    if entries.is_empty() {
        None
    } else {
        Some(entries)
    }
}

/// Build a `schema::SystemInstruction` from system text.
fn build_system_instruction(system_text: Option<String>) -> Option<schema::SystemInstruction> {
    system_text.map(|text| schema::SystemInstruction {
        parts: vec![schema::TextPart { text }],
    })
}

/// Convert JSON Schema type to Gemini Schema enum type.
fn to_gemini_type(ty: &str) -> Option<&'static str> {
    match ty {
        "string" => Some("STRING"),
        "number" => Some("NUMBER"),
        "integer" => Some("INTEGER"),
        "boolean" => Some("BOOLEAN"),
        "array" => Some("ARRAY"),
        "object" => Some("OBJECT"),
        _ => None,
    }
}

// ── Request building ───────────────────────────────────────────────

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
    cached_content: Option<&str>,
) -> Result<HttpRequest, ProviderError> {
    let body = match cached_content {
        Some(cache_name) => {
            let (_, rest) = split_system_messages(messages);
            let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();

            let req = schema::CachedRequest {
                cached_content: cache_name.to_string(),
                contents,
                tools: build_tools(tools, generation.grounding),
                generation_config: build_generation_config(generation, max_output_tokens),
            };
            serde_json::to_value(req).unwrap()
        }
        None => {
            let (system_text, rest) = split_system_messages(messages);
            let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();

            let req = schema::Request {
                contents,
                system_instruction: build_system_instruction(system_text),
                tools: build_tools(tools, generation.grounding),
                generation_config: build_generation_config(generation, max_output_tokens),
            };
            serde_json::to_value(req).unwrap()
        }
    };

    let url = format!(
        "{}/v1beta/models/{}:generateContent",
        config.endpoint, model
    );
    Ok(HttpRequest {
        url,
        headers: vec![
            ("x-goog-api-key".into(), config.api_key.clone()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    })
}

pub fn build_cache_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    ttl: &str,
    cache_config: &FxHashMap<String, serde_json::Value>,
) -> HttpRequest {
    let (system_text, rest) = split_system_messages(messages);
    let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();

    let extra: serde_json::Map<String, serde_json::Value> = cache_config
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let req = schema::CacheRequest {
        model: format!("models/{model}"),
        contents,
        ttl: ttl.to_string(),
        system_instruction: build_system_instruction(system_text),
        extra,
    };
    let body = serde_json::to_value(req).unwrap();

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

pub fn parse_cache_response(json: &serde_json::Value) -> Result<String, ProviderError> {
    let resp: schema::CacheResponse = serde_json::from_value(json.clone()).map_err(|e| {
        ProviderError::ResponseParse {
            detail: e.to_string(),
        }
    })?;
    resp.name
        .ok_or(ProviderError::MissingField { field: "name" })
}

pub fn build_count_tokens_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
) -> HttpRequest {
    let (system_text, rest) = split_system_messages(messages);
    let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();

    let req = schema::CountTokensRequest {
        generate_content_request: schema::CountTokensInner {
            model: format!("models/{model}"),
            contents,
            system_instruction: build_system_instruction(system_text),
        },
    };
    let body = serde_json::to_value(req).unwrap();

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

pub fn parse_count_tokens_response(json: &serde_json::Value) -> Result<u32, ProviderError> {
    let resp: schema::CountTokensResponse =
        serde_json::from_value(json.clone()).map_err(|e| ProviderError::ResponseParse {
            detail: e.to_string(),
        })?;
    resp.total_tokens
        .ok_or(ProviderError::MissingField { field: "totalTokens" })
}

// ── Response parsing ───────────────────────────────────────────────

pub fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), ProviderError> {
    let resp: schema::Response =
        serde_json::from_value(json.clone()).map_err(|e| ProviderError::ResponseParse {
            detail: e.to_string(),
        })?;

    let usage = match &resp.usage_metadata {
        Some(u) => Usage {
            input_tokens: u.prompt_token_count,
            output_tokens: u.candidates_token_count,
        },
        None => Usage::default(),
    };

    let candidates = resp
        .candidates
        .as_ref()
        .ok_or(ProviderError::MissingField {
            field: "candidates",
        })?;
    let candidate = candidates
        .first()
        .ok_or(ProviderError::MissingField {
            field: "candidates",
        })?;
    let content = candidate
        .content
        .as_ref()
        .ok_or(ProviderError::MissingField { field: "content" })?;
    let parts = content
        .parts
        .as_ref()
        .ok_or(ProviderError::MissingField { field: "parts" })?;

    let mut tool_calls = Vec::new();
    let mut content_parts = Vec::new();

    for part in parts {
        if let Some(fc) = &part.function_call {
            let name = fc
                .name
                .as_deref()
                .ok_or(ProviderError::MissingField {
                    field: "functionCall name",
                })?
                .to_string();
            let arguments = fc.args.clone().unwrap_or(serde_json::json!({}));
            let thought_signature = part.thought_signature.clone();
            let id = format!("call_{name}");
            tool_calls.push(ToolCall {
                id,
                name,
                arguments,
                extras: Some(ToolCallExtras::Gemini { thought_signature }),
            });
        } else if let Some(text) = &part.text {
            content_parts.push(Content::Text(text.clone()));
        } else if let Some(inline) = &part.inline_data {
            let mime_type = inline.mime_type.clone().unwrap_or_default();
            let data = inline.data.clone().unwrap_or_default();
            content_parts.push(Content::Blob { mime_type, data });
        }
    }

    if !tool_calls.is_empty() {
        Ok((ModelResponse::ToolCalls(tool_calls), usage))
    } else {
        let role = content
            .role
            .as_deref()
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

#[cfg(test)]
mod tests {

    use crate::ApiKind;
    use crate::message::{Message, ToolSpec, ToolSpecParam};

    use super::*;

    #[test]
    fn format_system_as_instruction() {
        let body = {
            let messages = &[
                Message::Content {
                    role: "system".into(),
                    content: Content::Text("You are helpful.".into()),
                },
                Message::Content {
                    role: "user".into(),
                    content: Content::Text("Hello".into()),
                },
            ];
            let (system_text, rest) = split_system_messages(messages);
            let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();
            let req = schema::Request {
                contents,
                system_instruction: build_system_instruction(system_text),
                tools: None,
                generation_config: None,
            };
            serde_json::to_value(req).unwrap()
        };
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
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
        let content = serde_json::to_value(convert_content(&msg)).unwrap();
        assert_eq!(content["role"], "model");
    }

    #[test]
    fn format_with_tools() {
        let tools_list = &[ToolSpec {
            name: "search".into(),
            description: "Search".into(),
            params: FxHashMap::from_iter([(
                "query".into(),
                ToolSpecParam {
                    ty: "string".into(),
                    description: None,
                },
            )]),
        }];
        let tool_decls = build_tools(tools_list, false).unwrap();
        let body = serde_json::to_value(&tool_decls).unwrap();
        let decls = &body[0]["functionDeclarations"];
        assert_eq!(decls.as_array().unwrap().len(), 1);
        assert_eq!(decls[0]["name"], "search");
    }

    #[test]
    fn format_function_call_message() {
        let msg = Message::ToolCalls(vec![ToolCall {
            id: "call_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"query": "rust"}),
            extras: None,
        }]);
        let content = serde_json::to_value(convert_content(&msg)).unwrap();
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
        let content = serde_json::to_value(convert_content(&msg)).unwrap();
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
            api: ApiKind::Google,
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
