//! OpenAI provider — ExternFn handler for chat completions.

pub mod schema;

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::ty::{Effect, Param, Ty};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::extract::{obj_get_decimal, obj_get_str, obj_get_u32, values_to_messages};
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, Usage};

// ── Message conversion ──────────────────────────────────────────────

fn convert_message(m: &Message) -> schema::RequestMessage {
    match m {
        Message::Content { role, content } => match content {
            Content::Text(text) => schema::RequestMessage::Content {
                role: role.clone(),
                content: text.clone(),
            },
            Content::Blob { mime_type, data } => schema::RequestMessage::ContentArray {
                role: role.clone(),
                content: vec![schema::ContentPart::ImageUrl {
                    image_url: schema::ImageUrlData {
                        url: format!("data:{mime_type};base64,{data}"),
                    },
                }],
            },
        },
        Message::ToolCalls(calls) => schema::RequestMessage::ToolCalls {
            role: "assistant".into(),
            tool_calls: calls
                .iter()
                .map(|tc| schema::RequestToolCall {
                    id: tc.id.clone(),
                    call_type: "function".into(),
                    function: schema::RequestToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect(),
        },
        Message::ToolResult { call_id, content } => schema::RequestMessage::ToolResult {
            role: "tool".into(),
            tool_call_id: call_id.clone(),
            content: content.clone(),
        },
    }
}

// ── Response parsing ────────────────────────────────────────────────

fn parse_response(json: serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response =
        serde_json::from_value(json).map_err(|e| RequestError::ResponseParse {
            detail: e.to_string(),
        })?;

    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or(RequestError::EmptyResponse)?;

    let usage = Usage {
        input_tokens: resp.usage.as_ref().map(|u| u.prompt_tokens),
        output_tokens: resp.usage.as_ref().map(|u| u.completion_tokens),
    };

    // Check for tool calls first
    if let Some(tool_calls) = choice.message.tool_calls {
        let calls: Result<Vec<ToolCall>, RequestError> = tool_calls
            .into_iter()
            .map(|tc| {
                let arguments = serde_json::from_str(&tc.function.arguments).map_err(|e| {
                    RequestError::ResponseParse {
                        detail: format!("tool call arguments: {e}"),
                    }
                })?;
                Ok(ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments,
                })
            })
            .collect();
        let calls = calls?;
        if !calls.is_empty() {
            return Ok((ModelResponse::ToolCalls(calls), usage));
        }
    }

    // Content
    let text = match choice.message.content {
        Some(schema::ResponseContent::Text(t)) => t,
        Some(schema::ResponseContent::Parts(parts)) => parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    };

    let role = choice.message.role;
    Ok((
        ModelResponse::Content(vec![ContentItem {
            role,
            content: Content::Text(text),
        }]),
        usage,
    ))
}

// ── Value extraction helpers ────────────────────────────────────────

fn usage_to_value(usage: &Usage, input_tokens_key: Astr, output_tokens_key: Astr) -> Value {
    let input = match usage.input_tokens {
        Some(n) => Value::Int(n as i64),
        None => Value::Unit,
    };
    let output = match usage.output_tokens {
        Some(n) => Value::Int(n as i64),
        None => Value::Unit,
    };
    Value::object(FxHashMap::from_iter([
        (input_tokens_key, input),
        (output_tokens_key, output),
    ]))
}

/// Convert a ModelResponse + Usage into a Value::Object.
fn response_to_value(resp: &ModelResponse, usage: &Usage, interner: &Interner) -> Value {
    let role_key = interner.intern("role");
    let content_key = interner.intern("content");
    let content_type_key = interner.intern("content_type");
    let tool_calls_key = interner.intern("tool_calls");
    let usage_key = interner.intern("usage");
    let input_tokens_key = interner.intern("input_tokens");
    let output_tokens_key = interner.intern("output_tokens");

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

            let usage_obj = usage_to_value(usage, input_tokens_key, output_tokens_key);

            Value::object(FxHashMap::from_iter([
                (content_key, Value::list(items)),
                (tool_calls_key, Value::list(vec![])),
                (usage_key, usage_obj),
            ]))
        }
        ModelResponse::ToolCalls(calls) => {
            let name_key = interner.intern("name");
            let id_key = interner.intern("id");
            let arguments_key = interner.intern("arguments");

            let tc_values: Vec<Value> = calls
                .iter()
                .map(|tc| {
                    Value::object(FxHashMap::from_iter([
                        (id_key, Value::string(tc.id.clone())),
                        (name_key, Value::string(tc.name.clone())),
                        (arguments_key, Value::string(tc.arguments.to_string())),
                    ]))
                })
                .collect();

            let usage_obj = usage_to_value(usage, input_tokens_key, output_tokens_key);

            Value::object(FxHashMap::from_iter([
                (content_key, Value::list(vec![])),
                (tool_calls_key, Value::list(tc_values)),
                (usage_key, usage_obj),
            ]))
        }
    }
}

// ── Registry ────────────────────────────────────────────────────────

/// Create an ExternRegistry for the OpenAI chat completion handler.
///
/// The registered function `openai_chat` takes `(messages: Value, config: Value)`
/// and returns a Value (Object with content, tool_calls, usage fields).
pub fn openai_registry<F: Fetch + Send + Sync + 'static>(fetch: Arc<F>) -> ExternRegistry {
    ExternRegistry::new(move |interner| {
        let endpoint_key = interner.intern("endpoint");
        let api_key_key = interner.intern("api_key");
        let model_key = interner.intern("model");
        let temperature_key = interner.intern("temperature");
        let top_p_key = interner.intern("top_p");
        let max_tokens_key = interner.intern("max_tokens");

        let fetch = Arc::clone(&fetch);

        let input_msg_ty = Ty::Object(
            [
                (interner.intern("role"), Ty::String),
                (interner.intern("content"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let config_ty = Ty::Object(
            [
                (interner.intern("endpoint"), Ty::String),
                (interner.intern("api_key"), Ty::String),
                (interner.intern("model"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let msg_elem_ty = Ty::Object(
            [
                (interner.intern("role"), Ty::String),
                (interner.intern("content"), Ty::String),
                (interner.intern("content_type"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let params = vec![Ty::List(Box::new(input_msg_ty)), config_ty];
        let ret = Ty::List(Box::new(msg_elem_ty));
        let named: Vec<Param> = params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
            .collect();
        let constraint = FnConstraint {
            signature: Some(Signature {
                params: named.clone(),
            }),
            output: Constraint::Exact(Ty::Fn {
                params: named,
                ret: Box::new(ret),
                captures: vec![],
                effect: Effect::pure(),
            }),
            effect: None,
            hint: Some(acvus_mir::ty::Hint::Io),
        };

        vec![
            ExternFnBuilder::new("openai_chat", constraint).handler_async(
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
                                    "openai_chat: expected List for messages, got {:?}",
                                    other.kind()
                                )));
                            }
                        };
                        let messages = values_to_messages(messages_list, &interner, "openai_chat")?;

                        // Extract config from Value::Object
                        let config_obj = match &config_val {
                            Value::Object(o) => o,
                            other => {
                                return Err(RuntimeError::fetch(format!(
                                    "openai_chat: expected Object for config, got {:?}",
                                    other.kind()
                                )));
                            }
                        };

                        let endpoint = obj_get_str(config_obj, endpoint_key).ok_or_else(|| {
                            RuntimeError::fetch("openai_chat: missing 'endpoint' in config")
                        })?;
                        let api_key = obj_get_str(config_obj, api_key_key).ok_or_else(|| {
                            RuntimeError::fetch("openai_chat: missing 'api_key' in config")
                        })?;
                        let model = obj_get_str(config_obj, model_key).ok_or_else(|| {
                            RuntimeError::fetch("openai_chat: missing 'model' in config")
                        })?;
                        let temperature = obj_get_decimal(config_obj, temperature_key);
                        let top_p = obj_get_decimal(config_obj, top_p_key);
                        let max_tokens = obj_get_u32(config_obj, max_tokens_key);

                        // Build schema::Request
                        let request_body = schema::Request {
                            model,
                            messages: messages.iter().map(convert_message).collect(),
                            tools: None,
                            temperature,
                            top_p,
                            max_tokens,
                            reasoning_effort: None,
                        };

                        let http_request = HttpRequest {
                            url: endpoint,
                            headers: vec![
                                ("Authorization".into(), format!("Bearer {api_key}")),
                                ("Content-Type".into(), "application/json".into()),
                            ],
                            body: serde_json::to_value(&request_body).map_err(|e| {
                                RuntimeError::fetch(format!(
                                    "openai_chat: serialization failed: {e}"
                                ))
                            })?,
                        };

                        let response_json = fetch
                            .fetch(&http_request)
                            .await
                            .map_err(RuntimeError::fetch)?;

                        let (response, usage) = parse_response(response_json)
                            .map_err(|e| RuntimeError::fetch(e.to_string()))?;

                        let result = response_to_value(&response, &usage, &interner);
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
        assert_eq!(
            json["content"][0]["image_url"]["url"],
            "data:image/png;base64,base64data"
        );
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
        assert_eq!(json["tool_calls"][0]["id"], "call_1");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn parse_content_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        });
        let (resp, usage) = parse_response(json).unwrap();
        match resp {
            ModelResponse::Content(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0].role, "assistant");
                match &parts[0].content {
                    Content::Text(t) => assert_eq!(t, "Hello!"),
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
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Seoul\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
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
    fn parse_empty_response_errors() {
        let json = serde_json::json!({
            "choices": [],
            "usage": null
        });
        assert!(parse_response(json).is_err());
    }

    #[test]
    fn registry_produces_function() {
        let fetch = Arc::new(MockFetch {
            response: serde_json::json!({}),
        });
        let interner = Interner::new();
        let registry = openai_registry(fetch);
        let registered = registry.register(&interner);
        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.executables.len(), 1);

        let func = &registered.functions[0];
        assert_eq!(interner.resolve(func.qref.name), "openai_chat");
    }
}
