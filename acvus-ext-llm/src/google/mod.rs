mod schema;

use std::sync::Arc;

use acvus_extern::{Registry, Runtime, TyArg, extern_fn, extern_registry};

use crate::extract::{input_messages, split_system};
use crate::http::{Fetch, FetchClient, HttpRequest, RequestError};
use crate::message::*;

// -- Message conversion ----------------------------------------------

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

// -- Response parsing ------------------------------------------------

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

// -- Registry --------------------------------------------------------

#[derive(Debug, Clone, TyArg)]
pub struct GoogleConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
}

fn first_message(resp: ModelResponse) -> OutputMessage {
    match resp {
        ModelResponse::Content(parts) => parts
            .first()
            .map(OutputMessage::text)
            .expect("google_llm: response has no content parts"),
        ModelResponse::ToolCalls(_) => {
            panic!("google_llm: tool calls are not representable as a message")
        }
    }
}

/// Gemini specifics:
/// - API key goes in URL query param: `{endpoint}/models/{model}:generateContent?key={api_key}`
/// - System messages are extracted into the `system_instruction` field (separate from `contents`)
/// - Role `"assistant"` is mapped to `"model"` for the Gemini API
#[extern_fn]
async fn google_llm(
    #[state] fetch: &FetchClient,
    messages: Vec<InputMessage>,
    config: GoogleConfig,
) -> OutputMessage {
    let msgs = input_messages(messages);
    let (system, rest) = split_system(&msgs);

    let request_body = schema::Request {
        contents: rest.iter().map(|m| convert_message(m)).collect(),
        system_instruction: system.map(|s| schema::SystemInstruction {
            parts: vec![schema::TextPart { text: s }],
        }),
        tools: None,
        generation_config: None,
    };

    let url = format!(
        "{}/models/{}:generateContent?key={}",
        config.endpoint, config.model, config.api_key
    );
    let body = serde_json::to_value(&request_body)
        .unwrap_or_else(|e| panic!("google_llm: serialization failed: {e}"));
    let http_request = HttpRequest {
        url,
        headers: vec![("Content-Type".into(), "application/json".into())],
        body,
    };

    let response_json = fetch
        .fetch(&http_request)
        .await
        .unwrap_or_else(|e| panic!("google_llm: {e}"));
    let (response, _usage) =
        parse_response(response_json).unwrap_or_else(|e| panic!("google_llm: {e}"));
    first_message(response)
}

/// The registry holding the Google/Gemini chat completion extern.
pub fn google_registry<F, R>(fetch: Arc<F>) -> Registry<R>
where
    F: Fetch + Send + Sync + 'static,
    R: Runtime + Clone,
{
    extern_registry! {
        ns: "llm",
        fns: [google_llm(FetchClient::new(fetch))],
    }
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
        let interner = acvus_extern::Interner::new();
        let registry = google_registry::<_, acvus_extern::TypesOnly>(fetch);
        let registered =
            acvus_extern::Externs::combine(vec![registry], &interner).expect("registry combines");
        let core = interner.intern("core");
        let mut implemented = registered.functions.iter().filter(|f| {
            f.qref.namespace != Some(core)
                && registered
                    .handlers
                    .get(&f.qref)
                    .is_some_and(|h| !h.is_empty())
        });
        let func = implemented
            .next()
            .expect("the registry implements its function");
        assert!(
            implemented.next().is_none(),
            "the registry implements exactly one function of its own; the shared signatures it combines with have no handler and `core` is what `combine` prepends"
        );
        assert_eq!(interner.resolve(func.qref.name), "google_llm");
    }
}
