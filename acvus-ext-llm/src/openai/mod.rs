//! OpenAI provider - ExternFn handler for chat completions.

pub mod schema;

use std::sync::Arc;

use acvus_ext::List;
use acvus_extern::{ExternFn, ExternItems, ExternRegistry, Interner, RuntimeError, TyArg};

use crate::extract::input_messages;
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::{
    Content, ContentItem, InputMessage, Message, ModelResponse, OutputMessage, ToolCall, Usage,
};

// -- Message conversion ----------------------------------------------

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

// -- Response parsing ------------------------------------------------

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

// -- Registry --------------------------------------------------------

#[derive(Debug, Clone, TyArg)]
pub struct OpenAiConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Clone, TyArg)]
pub struct ToolCallValue {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, TyArg)]
pub struct UsageValue {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
}

/// What `openai_chat` returns: content messages, or tool calls, with usage.
#[derive(Debug, Clone, TyArg)]
pub struct ChatResponse {
    pub content: List<OutputMessage>,
    pub tool_calls: List<ToolCallValue>,
    pub usage: UsageValue,
}

fn chat_response(resp: ModelResponse, usage: Usage) -> ChatResponse {
    let usage = UsageValue {
        input_tokens: usage.input_tokens.map(i64::from),
        output_tokens: usage.output_tokens.map(i64::from),
    };
    match resp {
        ModelResponse::Content(parts) => ChatResponse {
            content: List(parts.iter().map(OutputMessage::text).collect()),
            tool_calls: List(vec![]),
            usage,
        },
        ModelResponse::ToolCalls(calls) => ChatResponse {
            content: List(vec![]),
            tool_calls: List(
                calls
                    .into_iter()
                    .map(|tc| ToolCallValue {
                        id: tc.id,
                        name: tc.name,
                        arguments: tc.arguments.to_string(),
                    })
                    .collect(),
            ),
            usage,
        },
    }
}

/// Create an ExternRegistry for the OpenAI chat completion handler.
pub fn openai_registry<F: Fetch + Send + Sync + 'static>(fetch: Arc<F>) -> ExternRegistry {
    ExternRegistry::new(move |interner| {
        let handler = move |_: Interner, messages: List<InputMessage>, config: OpenAiConfig| {
            let fetch = Arc::clone(&fetch);
            async move {
                let messages = input_messages(messages.0);
                let request_body = schema::Request {
                    model: config.model,
                    messages: messages.iter().map(convert_message).collect(),
                    tools: None,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    reasoning_effort: None,
                };

                let http_request = HttpRequest {
                    url: config.endpoint,
                    headers: vec![
                        ("Authorization".into(), format!("Bearer {}", config.api_key)),
                        ("Content-Type".into(), "application/json".into()),
                    ],
                    body: serde_json::to_value(&request_body).map_err(|e| {
                        RuntimeError::fetch(format!("openai_chat: serialization failed: {e}"))
                    })?,
                };

                let response_json = fetch.fetch(&http_request).await.map_err(RuntimeError::fetch)?;
                let (response, usage) =
                    parse_response(response_json).map_err(|e| RuntimeError::fetch(e.to_string()))?;
                Ok(chat_response(response, usage))
            }
        };
        ExternItems {
            types: vec![],
            fns: vec![ExternFn::r#async(interner, "openai_chat", handler)],
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
        let registered = registry.register(&interner, &mut acvus_extern::TypeRegistry::new());
        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.executables.len(), 1);

        let func = &registered.functions[0];
        assert_eq!(interner.resolve(func.qref.name), "openai_chat");
    }
}
