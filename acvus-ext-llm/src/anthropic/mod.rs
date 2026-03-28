mod schema;

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::ty::{Effect, Param, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::extract::{obj_get_decimal, obj_get_str, obj_get_u32, split_system, values_to_messages};
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, Usage};

const DEFAULT_MAX_TOKENS: u32 = 4096;
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

// ── Message conversion ──────────────────────────────────────────────

fn convert_message(m: &Message) -> schema::RequestMessage {
    match m {
        Message::Content { role, content } => match content {
            Content::Text(text) => schema::RequestMessage {
                role: role.clone(),
                content: schema::RequestContent::Text(text.clone()),
            },
            Content::Blob { mime_type, data } => schema::RequestMessage {
                role: role.clone(),
                content: schema::RequestContent::Blocks(vec![schema::ContentBlock::Image {
                    source: schema::ImageSource {
                        source_type: "base64".into(),
                        media_type: mime_type.clone(),
                        data: data.clone(),
                    },
                }]),
            },
        },
        Message::ToolCalls(calls) => schema::RequestMessage {
            role: "assistant".into(),
            content: schema::RequestContent::Blocks(
                calls
                    .iter()
                    .map(|tc| schema::ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input: tc.arguments.clone(),
                    })
                    .collect(),
            ),
        },
        Message::ToolResult { call_id, content } => schema::RequestMessage {
            role: "user".into(),
            content: schema::RequestContent::Blocks(vec![schema::ContentBlock::ToolResult {
                tool_use_id: call_id.clone(),
                content: content.clone(),
            }]),
        },
    }
}

// ── Response parsing ────────────────────────────────────────────────

fn parse_response(json: serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response =
        serde_json::from_value(json).map_err(|e| RequestError::ResponseParse {
            detail: e.to_string(),
        })?;

    let usage = Usage {
        input_tokens: Some(resp.usage.input_tokens),
        output_tokens: Some(resp.usage.output_tokens),
    };

    let role = resp.role;

    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in resp.content {
        match block {
            schema::ResponseContentBlock::Text { text } => texts.push(text),
            schema::ResponseContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input,
                });
            }
            _ => {}
        }
    }

    if !tool_calls.is_empty() {
        return Ok((ModelResponse::ToolCalls(tool_calls), usage));
    }

    let text = texts.join("");
    Ok((
        ModelResponse::Content(vec![ContentItem {
            role,
            content: Content::Text(text),
        }]),
        usage,
    ))
}

// ── Value extraction helpers ────────────────────────────────────────

/// Convert a ModelResponse into a Value (List of Objects with role/content/content_type).
fn response_to_value(resp: &ModelResponse, interner: &Interner) -> Value {
    let role_key = interner.intern("role");
    let content_key = interner.intern("content");
    let content_type_key = interner.intern("content_type");

    match resp {
        ModelResponse::Content(parts) => {
            let items: Vec<Value> = parts
                .iter()
                .map(|item| {
                    let text = match &item.content {
                        Content::Text(t) => t.clone(),
                        Content::Blob { data, .. } => data.clone(),
                    };
                    Value::object(FxHashMap::from_iter([
                        (role_key, Value::string(item.role.clone())),
                        (content_key, Value::string(text)),
                        (content_type_key, Value::string("text")),
                    ]))
                })
                .collect();
            Value::list(items)
        }
        ModelResponse::ToolCalls(_) => Value::list(vec![]),
    }
}

// ── Registry ────────────────────────────────────────────────────────

pub fn anthropic_registry<F: Fetch + Send + Sync + 'static>(fetch: Arc<F>) -> ExternRegistry {
    ExternRegistry::new(move |interner| {
        let endpoint_key = interner.intern("endpoint");
        let api_key_key = interner.intern("api_key");
        let model_key = interner.intern("model");
        let max_tokens_key = interner.intern("max_tokens");
        let temperature_key = interner.intern("temperature");

        let msg_elem_ty = Ty::Object(
            [
                (interner.intern("role"), Ty::String),
                (interner.intern("content"), Ty::String),
                (interner.intern("content_type"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let config_ty = Ty::Object(
            [
                (interner.intern("endpoint"), Ty::String),
                (interner.intern("api_key"), Ty::String),
                (interner.intern("model"), Ty::String),
                (interner.intern("max_tokens"), Ty::Int),
            ]
            .into_iter()
            .collect(),
        );

        let input_msg_ty = Ty::Object(
            [
                (interner.intern("role"), Ty::String),
                (interner.intern("content"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let fetch = Arc::clone(&fetch);

        let params = vec![Ty::List(Box::new(input_msg_ty)), config_ty];
        let ret = Ty::List(Box::new(msg_elem_ty));
        let named: Vec<Param> = params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
            .collect();
        let constraint = FnConstraint {
            signature: Some(Signature { params: named.clone() }),
            output: Constraint::Exact(Ty::Fn {
                params: named,
                ret: Box::new(ret),
                captures: vec![],
                effect: Effect::io(),
            }),
            effect: None,
        };

        vec![
            ExternFnBuilder::new("anthropic", constraint)
                .handler_async(
                    move |interner: Interner,
                          (messages_val, config_val): (Value, Value),
                          Uses(()): Uses<()>| {
                        let fetch = Arc::clone(&fetch);
                        async move {
                            // Extract messages from Value::List
                            let messages_list = match &messages_val {
                                Value::List(l) => l.as_slice(),
                                other => {
                                    return Err(RuntimeError::fetch(format!(
                                        "anthropic: expected List for messages, got {:?}",
                                        other.kind()
                                    )));
                                }
                            };
                            let messages =
                                values_to_messages(messages_list, &interner, "anthropic")?;

                            // Split system message (first "system" role -> system param)
                            let (system, rest) = split_system(&messages);

                            // Extract config from Value::Object
                            let config_obj = match &config_val {
                                Value::Object(o) => o,
                                other => {
                                    return Err(RuntimeError::fetch(format!(
                                        "anthropic: expected Object for config, got {:?}",
                                        other.kind()
                                    )));
                                }
                            };

                            let endpoint =
                                obj_get_str(config_obj, endpoint_key).ok_or_else(|| {
                                    RuntimeError::fetch("anthropic: missing 'endpoint' in config")
                                })?;
                            let api_key =
                                obj_get_str(config_obj, api_key_key).ok_or_else(|| {
                                    RuntimeError::fetch("anthropic: missing 'api_key' in config")
                                })?;
                            let model = obj_get_str(config_obj, model_key).ok_or_else(|| {
                                RuntimeError::fetch("anthropic: missing 'model' in config")
                            })?;
                            let max_tokens = obj_get_u32(config_obj, max_tokens_key)
                                .unwrap_or(DEFAULT_MAX_TOKENS);
                            let temperature = obj_get_decimal(config_obj, temperature_key);

                            // Build schema::Request
                            let request_body = schema::Request {
                                model,
                                messages: rest.iter().map(|m| convert_message(m)).collect(),
                                max_tokens,
                                system,
                                tools: None,
                                temperature,
                                top_p: None,
                                top_k: None,
                                thinking: None,
                            };

                            let http_request = HttpRequest {
                                url: endpoint,
                                headers: vec![
                                    ("x-api-key".into(), api_key),
                                    ("anthropic-version".into(), ANTHROPIC_API_VERSION.into()),
                                    ("Content-Type".into(), "application/json".into()),
                                ],
                                body: serde_json::to_value(&request_body).map_err(|e| {
                                    RuntimeError::fetch(format!(
                                        "anthropic: serialization failed: {e}"
                                    ))
                                })?,
                            };

                            let response_json = fetch
                                .fetch(&http_request)
                                .await
                                .map_err(RuntimeError::fetch)?;

                            let (response, _usage) = parse_response(response_json)
                                .map_err(|e| RuntimeError::fetch(e.to_string()))?;

                            let result = response_to_value(&response, &interner);
                            Ok((result, Defs(())))
                        }
                    },
                ),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockFetch {
        response: serde_json::Value,
    }

    impl Fetch for MockFetch {
        async fn fetch(&self, _request: &HttpRequest) -> Result<serde_json::Value, String> {
            Ok(self.response.clone())
        }
    }

    #[test]
    fn convert_text_message() {
        let msg = Message::Content {
            role: "user".into(),
            content: Content::Text("hello".into()),
        };
        let schema_msg = convert_message(&msg);
        let json = serde_json::to_value(&schema_msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn convert_blob_message() {
        let msg = Message::Content {
            role: "user".into(),
            content: Content::Blob {
                mime_type: "image/png".into(),
                data: "base64data".into(),
            },
        };
        let schema_msg = convert_message(&msg);
        let json = serde_json::to_value(&schema_msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "image");
        assert_eq!(json["content"][0]["source"]["data"], "base64data");
    }

    #[test]
    fn convert_tool_calls_message() {
        let msg = Message::ToolCalls(vec![ToolCall {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Seoul"}),
        }]);
        let schema_msg = convert_message(&msg);
        let json = serde_json::to_value(&schema_msg).unwrap();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "tool_use");
        assert_eq!(json["content"][0]["name"], "get_weather");
    }

    #[test]
    fn convert_tool_result_message() {
        let msg = Message::ToolResult {
            call_id: "call_1".into(),
            content: "sunny".into(),
        };
        let schema_msg = convert_message(&msg);
        let json = serde_json::to_value(&schema_msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "tool_result");
        assert_eq!(json["content"][0]["tool_use_id"], "call_1");
        assert_eq!(json["content"][0]["content"], "sunny");
    }

    #[test]
    fn parse_content_response() {
        let json = serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let (resp, usage) = parse_response(json).unwrap();
        match resp {
            ModelResponse::Content(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0].role, "assistant");
                match &parts[0].content {
                    Content::Text(t) => assert_eq!(t, "Hello from Claude!"),
                    _ => panic!("expected text"),
                }
            }
            _ => panic!("expected content response"),
        }
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(5));
    }

    #[test]
    fn parse_tool_call_response() {
        let json = serde_json::json!({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "call_1",
                "name": "get_weather",
                "input": {"city": "Seoul"}
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let (resp, _usage) = parse_response(json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].name, "get_weather");
            }
            _ => panic!("expected tool calls response"),
        }
    }

    #[test]
    fn split_system_extracts_first() {
        let messages = vec![
            Message::Content {
                role: "system".into(),
                content: Content::Text("You are helpful.".into()),
            },
            Message::Content {
                role: "user".into(),
                content: Content::Text("Hi".into()),
            },
        ];
        let (system, rest) = split_system(&messages);
        assert_eq!(system.as_deref(), Some("You are helpful."));
        assert_eq!(rest.len(), 1);
    }

    #[test]
    fn split_system_none_when_absent() {
        let messages = vec![Message::Content {
            role: "user".into(),
            content: Content::Text("Hi".into()),
        }];
        let (system, rest) = split_system(&messages);
        assert!(system.is_none());
        assert_eq!(rest.len(), 1);
    }

    #[test]
    fn registry_produces_function() {
        let fetch = Arc::new(MockFetch {
            response: serde_json::json!({}),
        });
        let interner = Interner::new();
        let registry = anthropic_registry(fetch);
        let registered = registry.register(&interner);
        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.executables.len(), 1);

        let func = &registered.functions[0];
        assert_eq!(interner.resolve(func.qref.name), "anthropic");
    }
}
