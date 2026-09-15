mod schema;

use std::sync::Arc;

use acvus_ext::List;
use acvus_extern::{ExternError, ExternFn, ExternItems, ExternRegistry, Runtime, TyArg};

use crate::extract::{input_messages, split_system};
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::{
    Content, ContentItem, InputMessage, Message, ModelResponse, OutputMessage, ToolCall, Usage,
};
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

// -- Message conversion ----------------------------------------------

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

// -- Response parsing ------------------------------------------------

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

// -- Registry --------------------------------------------------------

#[derive(Debug, Clone, TyArg)]
pub struct AnthropicConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: i64,
}

fn response_messages(resp: ModelResponse) -> Result<List<OutputMessage>, ExternError> {
    match resp {
        ModelResponse::Content(parts) => Ok(List(parts.iter().map(OutputMessage::text).collect())),
        ModelResponse::ToolCalls(_) => Err(ExternError::call(
            "anthropic",
            "anthropic: tool calls are not representable as messages",
        )),
    }
}

pub fn anthropic_registry<F, R>(fetch: Arc<F>) -> ExternRegistry<R>
where
    F: Fetch + Send + Sync + 'static,
    R: Runtime + Clone,
{
    ExternRegistry::new(move |interner| {
        let handler = move |_: R, messages: List<InputMessage>, config: AnthropicConfig| {
            let fetch = Arc::clone(&fetch);
            async move {
                let messages = input_messages(messages.0);
                let (system, rest) = split_system(&messages);
                let max_tokens = u32::try_from(config.max_tokens).map_err(|_| {
                    ExternError::call(
                        "anthropic",
                        format!("max_tokens {} out of range", config.max_tokens),
                    )
                })?;

                let request_body = schema::Request {
                    model: config.model,
                    messages: rest.iter().map(|m| convert_message(m)).collect(),
                    max_tokens,
                    system,
                    tools: None,
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    thinking: None,
                };

                let http_request = HttpRequest {
                    url: config.endpoint,
                    headers: vec![
                        ("x-api-key".into(), config.api_key),
                        ("anthropic-version".into(), ANTHROPIC_API_VERSION.into()),
                        ("Content-Type".into(), "application/json".into()),
                    ],
                    body: serde_json::to_value(&request_body).map_err(|e| {
                        ExternError::call("anthropic", format!("serialization failed: {e}"))
                    })?,
                };

                let response_json = fetch
                    .fetch(&http_request)
                    .await
                    .map_err(|e| ExternError::call("anthropic", e))?;
                let (response, _usage) = parse_response(response_json)
                    .map_err(|e| ExternError::call("anthropic", e.to_string()))?;
                response_messages(response).map_err(R::Error::from)
            }
        };
        ExternItems {
            types: vec![],
            fns: vec![ExternFn::r#async(interner, "anthropic", handler)],
        }
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
        let interner = acvus_extern::Interner::new();
        let registry = anthropic_registry::<_, acvus_extern::TypesOnly>(fetch);
        let registered = registry.register(&interner, &mut acvus_extern::TypeRegistry::new());
        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.handlers.len(), 1);

        let func = &registered.functions[0];
        assert_eq!(interner.resolve(func.qref.name), "anthropic");
    }
}
