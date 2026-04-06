mod schema;

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::ty::{Effect, Hint, ParamTerm, Poly, Ty, TyTerm, lift_effect_to_poly, lift_to_poly};
use acvus_utils::Interner;

use crate::extract::{obj_get_decimal, obj_get_str, obj_get_u32, split_system, values_to_messages};
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::*;

// ── Message conversion ──────────────────────────────────────────────

fn convert_message(m: &Message) -> schema::Content {
    match m {
        Message::Content { role, content } => {
            let role = if role == "assistant" { "model" } else { role };
            match content {
                Content::Text(text) => schema::Content {
                    role: role.to_string(),
                    parts: vec![schema::Part::Text { text: text.clone() }],
                },
                Content::Blob { mime_type, data } => schema::Content {
                    role: role.to_string(),
                    parts: vec![schema::Part::InlineData {
                        inline_data: schema::InlineData {
                            mime_type: mime_type.clone(),
                            data: data.clone(),
                        },
                    }],
                },
            }
        }
        Message::ToolCalls(calls) => schema::Content {
            role: "model".into(),
            parts: calls
                .iter()
                .map(|tc| schema::Part::FunctionCall {
                    function_call: schema::FunctionCallPayload {
                        name: tc.name.clone(),
                        args: tc.arguments.clone(),
                    },
                    thought_signature: None,
                })
                .collect(),
        },
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

// ── Response parsing ────────────────────────────────────────────────

fn parse_response(json: serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response =
        serde_json::from_value(json).map_err(|e| RequestError::ResponseParse {
            detail: e.to_string(),
        })?;

    let candidate = resp
        .candidates
        .and_then(|mut c| {
            if c.is_empty() {
                None
            } else {
                Some(c.remove(0))
            }
        })
        .ok_or(RequestError::EmptyResponse)?;

    let usage = Usage {
        input_tokens: resp
            .usage_metadata
            .as_ref()
            .and_then(|u| u.prompt_token_count),
        output_tokens: resp
            .usage_metadata
            .as_ref()
            .and_then(|u| u.candidates_token_count),
    };

    let content = candidate.content.ok_or(RequestError::EmptyResponse)?;
    let role = content.role.unwrap_or_else(|| "model".into());
    let parts = content.parts.unwrap_or_default();

    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        if let Some(text) = part.text {
            texts.push(text);
        }
        if let Some(fc) = part.function_call {
            tool_calls.push(ToolCall {
                id: fc.name.clone(),
                name: fc.name,
                arguments: fc.args,
            });
        }
    }

    if !tool_calls.is_empty() {
        return Ok((ModelResponse::ToolCalls(tool_calls), usage));
    }

    Ok((
        ModelResponse::Content(vec![ContentItem {
            role,
            content: Content::Text(texts.join("")),
        }]),
        usage,
    ))
}

// ── Value helpers ───────────────────────────────────────────────────

/// Build the response `Value::Object` from a `ModelResponse`.
fn response_to_value(resp: &ModelResponse, interner: &Interner) -> Value {
    let role_key = interner.intern("role");
    let content_key = interner.intern("content");
    let content_type_key = interner.intern("content_type");

    match resp {
        ModelResponse::Content(parts) => {
            let first = parts.first().map(|item| {
                let text = match &item.content {
                    Content::Text(t) => t.clone(),
                    Content::Blob { data, .. } => data.clone(),
                };
                Value::object(
                    [
                        (role_key, Value::string(item.role.clone())),
                        (content_key, Value::string(text)),
                        (content_type_key, Value::string("text")),
                    ]
                    .into_iter()
                    .collect(),
                )
            });
            first.unwrap_or_else(|| {
                Value::object(
                    [
                        (role_key, Value::string("model")),
                        (content_key, Value::string("")),
                        (content_type_key, Value::string("text")),
                    ]
                    .into_iter()
                    .collect(),
                )
            })
        }
        ModelResponse::ToolCalls(_) => Value::object(
            [
                (role_key, Value::string("model")),
                (content_key, Value::string("")),
                (content_type_key, Value::string("text")),
            ]
            .into_iter()
            .collect(),
        ),
    }
}

// ── Registry ────────────────────────────────────────────────────────

/// Create an `ExternRegistry` for the Google/Gemini chat completion handler.
///
/// The registered function `google_llm` takes `(messages, config)` where:
/// - `messages`: list of objects with `{role, content, content_type}` fields
/// - `config`: object with `{endpoint, api_key, model, temperature?, top_p?, top_k?, max_tokens?}`
///
/// Gemini specifics:
/// - API key goes in URL query param: `{endpoint}/models/{model}:generateContent?key={api_key}`
/// - System messages are extracted into the `system_instruction` field (separate from `contents`)
/// - Role `"assistant"` is mapped to `"model"` for the Gemini API
/// - Returns `Value::Object` with `{role, content, content_type}` fields
pub fn google_registry<F: Fetch + Send + Sync + 'static>(fetch: Arc<F>) -> ExternRegistry {
    ExternRegistry::new(move |interner| {
        let role_key = interner.intern("role");
        let content_key = interner.intern("content");
        let content_type_key = interner.intern("content_type");
        let endpoint_key = interner.intern("endpoint");
        let api_key_key = interner.intern("api_key");
        let model_key = interner.intern("model");
        let temperature_key = interner.intern("temperature");
        let top_p_key = interner.intern("top_p");
        let top_k_key = interner.intern("top_k");
        let max_tokens_key = interner.intern("max_tokens");

        let msg_ty = Ty::Object(
            [
                (role_key, Ty::String),
                (content_key, Ty::String),
                (content_type_key, Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let config_ty = Ty::Object(
            [
                (endpoint_key, Ty::String),
                (api_key_key, Ty::String),
                (model_key, Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let fetch = Arc::clone(&fetch);

        let params = vec![
            Ty::List(Box::new(Ty::Object(
                [(role_key, Ty::String), (content_key, Ty::String)]
                    .into_iter()
                    .collect(),
            ))),
            config_ty,
        ];
        let named: Vec<ParamTerm<Poly>> = params
            .iter()
            .enumerate()
            .map(|(i, ty)| ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), lift_to_poly(ty)))
            .collect();
        let ty = TyTerm::Fn {
            params: named,
            ret: Box::new(lift_to_poly(&msg_ty)),
            captures: vec![],
            effect: lift_effect_to_poly(&Effect::pure()),
            hint: Some(Hint::Io),
        };

        vec![
            ExternFnBuilder::new("google_llm", ty).handler_async(
                move |interner: Interner,
                      (messages, config): (Value, Value),
                      Uses(()): Uses<()>| {
                    let fetch = Arc::clone(&fetch);
                    async move {
                        let messages_list = match &messages {
                            Value::List(l) => l.as_slice(),
                            other => {
                                return Err(RuntimeError::fetch(format!(
                                    "google_llm: expected List for messages, got {:?}",
                                    other.kind()
                                )));
                            }
                        };
                        let msgs = values_to_messages(messages_list, &interner, "google_llm")?;
                        let (system, rest) = split_system(&msgs);

                        let config_obj = match &config {
                            Value::Object(o) => o,
                            other => {
                                return Err(RuntimeError::fetch(format!(
                                    "google_llm: expected Object for config, got {:?}",
                                    other.kind()
                                )));
                            }
                        };

                        let endpoint = obj_get_str(config_obj, endpoint_key).ok_or_else(|| {
                            RuntimeError::fetch("google_llm: missing 'endpoint' in config")
                        })?;
                        let api_key = obj_get_str(config_obj, api_key_key).ok_or_else(|| {
                            RuntimeError::fetch("google_llm: missing 'api_key' in config")
                        })?;
                        let model = obj_get_str(config_obj, model_key).ok_or_else(|| {
                            RuntimeError::fetch("google_llm: missing 'model' in config")
                        })?;
                        let temperature = obj_get_decimal(config_obj, temperature_key);
                        let top_p = obj_get_decimal(config_obj, top_p_key);
                        let top_k = obj_get_u32(config_obj, top_k_key);
                        let max_tokens = obj_get_u32(config_obj, max_tokens_key);

                        // Build the Gemini request body.
                        let request_body = schema::Request {
                            contents: rest.iter().map(|m| convert_message(m)).collect(),
                            system_instruction: system.map(|s| schema::SystemInstruction {
                                parts: vec![schema::TextPart { text: s }],
                            }),
                            tools: None,
                            generation_config: Some(schema::GenerationConfig {
                                temperature,
                                top_p,
                                top_k,
                                max_output_tokens: max_tokens,
                                thinking_config: None,
                            }),
                        };

                        // API key goes in URL query param (not in header).
                        let url = format!(
                            "{}/models/{}:generateContent?key={}",
                            endpoint, model, api_key
                        );

                        let body = serde_json::to_value(&request_body).map_err(|e| {
                            acvus_interpreter::RuntimeError::fetch(format!(
                                "google_llm: serialization failed: {e}"
                            ))
                        })?;

                        let http_request = HttpRequest {
                            url,
                            // Content-Type only — no auth header (key is in URL).
                            headers: vec![("Content-Type".into(), "application/json".into())],
                            body,
                        };

                        let response_json = fetch
                            .fetch(&http_request)
                            .await
                            .map_err(acvus_interpreter::RuntimeError::fetch)?;

                        let (response, _usage) = parse_response(response_json)
                            .map_err(|e| acvus_interpreter::RuntimeError::fetch(e.to_string()))?;

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
        assert_eq!(json["parts"][0]["text"], "hello");
    }

    #[test]
    fn convert_assistant_maps_to_model() {
        let msg = Message::Content {
            role: "assistant".into(),
            content: Content::Text("hi".into()),
        };
        let schema_msg = convert_message(&msg);
        let json = serde_json::to_value(&schema_msg).unwrap();
        assert_eq!(json["role"], "model");
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
        assert_eq!(json["parts"][0]["inlineData"]["mimeType"], "image/png");
        assert_eq!(json["parts"][0]["inlineData"]["data"], "base64data");
    }

    #[test]
    fn convert_tool_calls_message() {
        let msg = Message::ToolCalls(vec![ToolCall {
            id: "get_weather".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Seoul"}),
        }]);
        let schema_msg = convert_message(&msg);
        let json = serde_json::to_value(&schema_msg).unwrap();
        assert_eq!(json["role"], "model");
        assert_eq!(json["parts"][0]["functionCall"]["name"], "get_weather");
    }

    #[test]
    fn parse_content_response() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello from Gemini!"}]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        });
        let (resp, usage) = parse_response(json).unwrap();
        match resp {
            ModelResponse::Content(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0].role, "model");
                match &parts[0].content {
                    Content::Text(t) => assert_eq!(t, "Hello from Gemini!"),
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
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Seoul"}
                        }
                    }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 12
            }
        });
        let (resp, _usage) = parse_response(json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].id, "get_weather");
            }
            _ => panic!("expected tool calls response"),
        }
    }

    #[test]
    fn parse_empty_candidates_errors() {
        let json = serde_json::json!({
            "candidates": [],
            "usageMetadata": null
        });
        assert!(parse_response(json).is_err());
    }

    #[test]
    fn parse_no_candidates_errors() {
        let json = serde_json::json!({});
        assert!(parse_response(json).is_err());
    }

    #[test]
    fn registry_produces_function() {
        let fetch = Arc::new(MockFetch {
            response: serde_json::json!({}),
        });
        let interner = Interner::new();
        let registry = google_registry(fetch);
        let registered = registry.register(&interner);
        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.executables.len(), 1);

        let func = &registered.functions[0];
        assert_eq!(interner.resolve(func.qref.name), "google_llm");
    }
}
