use std::sync::Arc;

use acvus_interpreter::{PureValue, RuntimeError, TypedValue, Value};
use acvus_utils::{Astr, Interner};

use rust_decimal::Decimal;
use rustc_hash::FxHashMap;
use tracing::{debug, info};

use super::Node;
use super::helpers::{
    allocate_token_budgets, content_to_value, eval_script_in_coroutine,
    execute_tool_calls, flatten_segments, make_tool_specs,
    render_messages, split_system_messages,
};
use super::schema::google as schema;
use crate::compile::{CompiledMessage, CompiledScript};
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::spec::{CompiledGoogleAI, CompiledToolBinding, MaxTokens, ThinkingConfig};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, ToolCallExtras, ToolSpec, Usage};

// ── Provider logic (absorbed from provider/google.rs) ───────────────

const THINKING_LEVEL_LOW: &str = "LOW";
const THINKING_LEVEL_MEDIUM: &str = "MEDIUM";
const THINKING_LEVEL_HIGH: &str = "HIGH";

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

/// Build a `schema::GenerationConfig` from individual params and optional max_output_tokens.
fn build_generation_config(
    temperature: Option<Decimal>,
    top_p: Option<Decimal>,
    top_k: Option<u32>,
    thinking: &Option<ThinkingConfig>,
    max_output_tokens: Option<u32>,
) -> Option<schema::GenerationConfig> {
    let thinking_config = match thinking {
        None | Some(ThinkingConfig::Off) => None,
        Some(ThinkingConfig::Low) => Some(schema::ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(THINKING_LEVEL_LOW.into()),
        }),
        Some(ThinkingConfig::Medium) => Some(schema::ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(THINKING_LEVEL_MEDIUM.into()),
        }),
        Some(ThinkingConfig::High) => Some(schema::ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(THINKING_LEVEL_HIGH.into()),
        }),
        Some(ThinkingConfig::Custom(n)) => Some(schema::ThinkingConfig {
            thinking_budget: Some(*n),
            thinking_level: None,
        }),
    };

    let has_any = temperature.is_some()
        || top_p.is_some()
        || top_k.is_some()
        || max_output_tokens.is_some()
        || thinking_config.is_some();

    if !has_any {
        return None;
    }

    Some(schema::GenerationConfig {
        temperature,
        top_p,
        top_k,
        max_output_tokens,
        thinking_config,
    })
}

/// Build tool declarations from `ToolSpec` slice and grounding flag.
fn build_tools(tools: &[ToolSpec], grounding: bool) -> Result<Option<Vec<schema::ToolDeclaration>>, RequestError> {
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
                        Some(serde_json::to_value(prop)
                            .map(|v| (name.clone(), v))
                            .map_err(|e| RequestError::Serialization { detail: e.to_string() }))
                    })
                    .collect::<Result<_, RequestError>>()?;

                let required: Vec<String> = t.params.keys().cloned().collect();

                Ok(schema::FunctionDecl {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: schema::GeminiSchema {
                        schema_type: "OBJECT".into(),
                        properties,
                        required,
                    },
                })
            })
            .collect::<Result<_, RequestError>>()?;

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
        Ok(None)
    } else {
        Ok(Some(entries))
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

fn build_request(
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    temperature: Option<Decimal>,
    top_p: Option<Decimal>,
    top_k: Option<u32>,
    thinking: &Option<ThinkingConfig>,
    grounding: bool,
    max_output_tokens: Option<u32>,
    cached_content: Option<&str>,
) -> Result<HttpRequest, RequestError> {
    let body = match cached_content {
        Some(cache_name) => {
            let (_, rest) = split_system_messages(messages)?;
            let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();

            let req = schema::CachedRequest {
                cached_content: cache_name.to_string(),
                contents,
                tools: build_tools(tools, grounding)?,
                generation_config: build_generation_config(temperature, top_p, top_k, thinking, max_output_tokens),
            };
            serde_json::to_value(req).map_err(|e| RequestError::Serialization { detail: e.to_string() })?
        }
        None => {
            let (system_text, rest) = split_system_messages(messages)?;
            let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();

            let req = schema::Request {
                contents,
                system_instruction: build_system_instruction(system_text),
                tools: build_tools(tools, grounding)?,
                generation_config: build_generation_config(temperature, top_p, top_k, thinking, max_output_tokens),
            };
            serde_json::to_value(req).map_err(|e| RequestError::Serialization { detail: e.to_string() })?
        }
    };

    let url = format!("{endpoint}/v1beta/models/{model}:generateContent");
    Ok(HttpRequest {
        url,
        headers: vec![
            ("x-goog-api-key".into(), api_key.to_string()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    })
}

pub(super) fn build_cache_request(
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    ttl: &str,
    cache_config: &FxHashMap<String, serde_json::Value>,
) -> Result<HttpRequest, RequestError> {
    let (system_text, rest) = split_system_messages(messages)?;
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
    let body = serde_json::to_value(req).map_err(|e| RequestError::Serialization { detail: e.to_string() })?;

    let url = format!("{endpoint}/v1beta/cachedContents");
    Ok(HttpRequest {
        url,
        headers: vec![
            ("x-goog-api-key".into(), api_key.to_string()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    })
}

pub(super) fn parse_cache_response(json: &serde_json::Value) -> Result<String, RequestError> {
    let resp: schema::CacheResponse = serde_json::from_value(json.clone()).map_err(|e| {
        RequestError::ResponseParse {
            detail: e.to_string(),
        }
    })?;
    resp.name
        .ok_or(RequestError::MissingField { field: "name" })
}

pub(super) fn build_count_tokens_request(
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
) -> Result<HttpRequest, RequestError> {
    let (system_text, rest) = split_system_messages(messages)?;
    let contents: Vec<schema::Content> = rest.into_iter().map(convert_content).collect();

    let req = schema::CountTokensRequest {
        generate_content_request: schema::CountTokensInner {
            model: format!("models/{model}"),
            contents,
            system_instruction: build_system_instruction(system_text),
        },
    };
    let body = serde_json::to_value(req).map_err(|e| RequestError::Serialization { detail: e.to_string() })?;

    let url = format!("{endpoint}/v1beta/models/{model}:countTokens");
    Ok(HttpRequest {
        url,
        headers: vec![
            ("x-goog-api-key".into(), api_key.to_string()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    })
}

pub(super) fn parse_count_tokens_response(json: &serde_json::Value) -> Result<u32, RequestError> {
    let resp: schema::CountTokensResponse =
        serde_json::from_value(json.clone()).map_err(|e| RequestError::ResponseParse {
            detail: e.to_string(),
        })?;
    resp.total_tokens
        .ok_or(RequestError::MissingField { field: "totalTokens" })
}

// ── Response parsing ───────────────────────────────────────────────

fn parse_response(json: &serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response =
        serde_json::from_value(json.clone()).map_err(|e| RequestError::ResponseParse {
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
        .ok_or(RequestError::MissingField {
            field: "candidates",
        })?;
    let candidate = candidates
        .first()
        .ok_or(RequestError::MissingField {
            field: "candidates",
        })?;
    let content = candidate
        .content
        .as_ref()
        .ok_or(RequestError::MissingField { field: "content" })?;
    let parts = content
        .parts
        .as_ref()
        .ok_or(RequestError::MissingField { field: "parts" })?;

    let mut tool_calls = Vec::new();
    let mut content_parts = Vec::new();

    for part in parts {
        if let Some(fc) = &part.function_call {
            let name = fc
                .name
                .as_deref()
                .ok_or(RequestError::MissingField {
                    field: "functionCall name",
                })?
                .to_string();
            let arguments = fc.args.clone()
                .ok_or(RequestError::MissingField { field: "functionCall.args" })?;
            let thought_signature = part.thought_signature.clone();
            // Google Gemini API does not provide a tool call ID in responses
            // (unlike OpenAI). We synthesize one from the function name.
            // This is a known fabrication — if multiple calls share the same
            // name within a single response, IDs will collide.
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
            let mime_type = inline.mime_type.clone().ok_or(RequestError::MissingField { field: "mime_type" })?;
            let data = inline.data.clone().ok_or(RequestError::MissingField { field: "data" })?;
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

// ── Node ────────────────────────────────────────────────────────────

const MAX_TOOL_ROUNDS: usize = 10;

struct GoogleAIConfig {
    endpoint: String,
    api_key: String,
    model: String,
    messages: Vec<CompiledMessage>,
    tools: Vec<CompiledToolBinding>,
    temperature: Option<rust_decimal::Decimal>,
    top_p: Option<rust_decimal::Decimal>,
    top_k: Option<u32>,
    max_tokens: MaxTokens,
    thinking: Option<ThinkingConfig>,
    grounding: bool,
    cache_key: Option<CompiledScript>,
}

pub struct GoogleAINode<F> {
    config: Arc<GoogleAIConfig>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> GoogleAINode<F>
where
    F: Fetch + 'static,
{
    pub fn new(
        compiled: &CompiledGoogleAI,
        fetch: Arc<F>,
        interner: &Interner,
    ) -> Self {
        Self {
            config: Arc::new(GoogleAIConfig {
                endpoint: compiled.endpoint.clone(),
                api_key: compiled.api_key.clone(),
                model: compiled.model.clone(),
                messages: compiled.messages.clone(),
                tools: compiled.tools.clone(),
                temperature: compiled.temperature,
                top_p: compiled.top_p,
                top_k: compiled.top_k,
                max_tokens: compiled.max_tokens.clone(),
                thinking: compiled.thinking.clone(),
                grounding: compiled.grounding,
                cache_key: compiled.cache_key.clone(),
            }),
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F> Node for GoogleAINode<F>
where
    F: Fetch + 'static,
{
    fn spawn(
        &self,
        local: FxHashMap<Astr, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError> {
        let config = Arc::clone(&self.config);
        let fetch = Arc::clone(&self.fetch);
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            let model_name = config.model.clone();

            let cached_content = if let Some(ref ck_script) = config.cache_key {
                let val =
                    eval_script_in_coroutine(&interner, &ck_script.module, &local, &handle)
                        .await?;
                match val.value() {
                    Value::Pure(PureValue::String(s)) => Some(s.clone()),
                    _ => None,
                }
            } else {
                None
            };

            // Render messages
            let mut segments = render_messages(&config.messages, &interner, &local, &handle).await?;

            allocate_token_budgets(
                &config.endpoint, &config.api_key, &config.model,
                "google",
                &*fetch, &mut segments, config.max_tokens.input,
            ).await;

            let mut rendered = flatten_segments(segments);
            let specs = make_tool_specs(&config.tools);

            info!(model = %model_name, messages = rendered.len(), tools = specs.len(), "google request");
            let request = build_request(
                &config.endpoint,
                &config.api_key,
                &config.model,
                &rendered,
                &specs,
                config.temperature,
                config.top_p,
                config.top_k,
                &config.thinking,
                config.grounding,
                config.max_tokens.output,
                cached_content.as_deref(),
            ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
            let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
            let (mut response, _usage) = parse_response(&json)
                .map_err(|e| RuntimeError::fetch(e.to_string()))?;
            debug!(
                input_tokens = _usage.input_tokens,
                output_tokens = _usage.output_tokens,
                "google response received",
            );

            let mut tool_rounds = 0usize;
            loop {
                match response {
                    ModelResponse::Content(items) => {
                        debug!(items = items.len(), "google returned content");
                        handle.yield_val(content_to_value(&interner, &items)).await;
                        return Ok(());
                    }
                    ModelResponse::ToolCalls(calls) => {
                        tool_rounds += 1;
                        info!(round = tool_rounds, count = calls.len(), "google tool calls");
                        if tool_rounds > MAX_TOOL_ROUNDS {
                            return Err(RuntimeError::tool_call_limit(MAX_TOOL_ROUNDS));
                        }

                        rendered.push(Message::ToolCalls(calls.clone()));
                        let tool_results = execute_tool_calls(&calls, &config.tools, &interner, &handle).await;
                        rendered.extend(tool_results);

                        debug!(
                            round = tool_rounds,
                            "google follow-up request after tool results"
                        );
                        let request = build_request(
                            &config.endpoint,
                            &config.api_key,
                            &config.model,
                            &rendered,
                            &specs,
                            config.temperature,
                            config.top_p,
                            config.top_k,
                            &config.thinking,
                            config.grounding,
                            config.max_tokens.output,
                            cached_content.as_deref(),
                        ).map_err(|e| RuntimeError::fetch(e.to_string()))?;
                        let json = fetch.fetch(&request).await.map_err(RuntimeError::fetch)?;
                        let parsed = parse_response(&json)
                            .map_err(|e| RuntimeError::fetch(e.to_string()))?;
                        response = parsed.0;
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {

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
            let (system_text, rest) = split_system_messages(messages).unwrap();
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
        let tool_decls = build_tools(tools_list, false).unwrap().unwrap();
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
        let req = build_count_tokens_request(
            "https://generativelanguage.googleapis.com",
            "test-key",
            "gemini-2.0-flash",
            &[Message::Content {
                role: "user".into(),
                content: Content::Text("hello".into()),
            }],
        ).unwrap();
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
