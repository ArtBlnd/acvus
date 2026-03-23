use std::sync::Arc;

use acvus_interpreter::{RuntimeError, TypedValue, Value};
use acvus_mir::graph::ContextId;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rust_decimal::Decimal;
use rustc_hash::FxHashMap;

use super::Unit;
use crate::dsl::message_elem_ty;
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, Usage};

mod schema {
    pub use crate::unit::schema::anthropic::*;
}

const DEFAULT_MAX_TOKENS: u32 = 4096;
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub message_ids: Vec<(Astr, ContextId)>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
}

pub struct AnthropicUnit<F> {
    config: Freeze<AnthropicConfig>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> AnthropicUnit<F> {
    pub fn new(config: Freeze<AnthropicConfig>, fetch: Arc<F>, interner: &Interner) -> Self {
        Self {
            config,
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F: Fetch + 'static> Unit for AnthropicUnit<F> {
    fn spawn(
        &self,
        _local_context: FxHashMap<ContextId, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError, ContextId> {
        let config = self.config.clone();
        let fetch = Arc::clone(&self.fetch);
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            let mut messages = Vec::new();
            for (role, msg_id) in &config.message_ids {
                let rendered: TypedValue = handle.request_context(*msg_id).await;
                let text = rendered.value().expect_ref::<str>("anthropic message").to_string();
                messages.push(Message::Content {
                    role: interner.resolve(*role).to_string(),
                    content: Content::Text(text),
                });
            }

            // Split system message (first "system" role → system param)
            let (system, rest) = split_system(&messages);

            let request_body = schema::Request {
                model: config.model.clone(),
                messages: rest.iter().map(|m| convert_message(m)).collect(),
                max_tokens: config.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
                system,
                tools: None,
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: config.top_k,
                thinking: None,
            };

            let http_request = HttpRequest {
                url: config.endpoint.clone(),
                headers: vec![
                    ("x-api-key".into(), config.api_key.clone()),
                    ("anthropic-version".into(), ANTHROPIC_API_VERSION.into()),
                    ("Content-Type".into(), "application/json".into()),
                ],
                body: serde_json::to_value(&request_body).unwrap(),
            };

            let response_json = fetch.fetch(&http_request).await
                .map_err(RuntimeError::fetch)?;

            let (response, _usage) = parse_response(response_json)
                .map_err(|e| RuntimeError::fetch(e.to_string()))?;

            let result = response_to_value(&response, &interner);
            handle.yield_val(result).await;
            Ok(())
        })
    }
}

fn split_system(messages: &[Message]) -> (Option<String>, Vec<&Message>) {
    let mut system = None;
    let mut rest = Vec::new();
    for m in messages {
        if let Message::Content { role, content: Content::Text(text) } = m {
            if role == "system" && system.is_none() {
                system = Some(text.clone());
                continue;
            }
        }
        rest.push(m);
    }
    (system, rest)
}

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
                calls.iter().map(|tc| schema::ContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    input: tc.arguments.clone(),
                }).collect(),
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

fn parse_response(json: serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response = serde_json::from_value(json)
        .map_err(|e| RequestError::ResponseParse { detail: e.to_string() })?;

    let usage = Usage {
        input_tokens: resp.usage.as_ref().and_then(|u| u.input_tokens),
        output_tokens: resp.usage.as_ref().and_then(|u| u.output_tokens),
    };

    let role = resp.role.unwrap_or_else(|| "assistant".into());

    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in resp.content {
        match block {
            schema::ResponseContentBlock::Text { text } => texts.push(text),
            schema::ResponseContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input.unwrap_or_default(),
                    extras: None,
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

fn response_to_value(resp: &ModelResponse, interner: &Interner) -> TypedValue {
    let role_key = interner.intern("role");
    let content_key = interner.intern("content");
    let content_type_key = interner.intern("content_type");

    let items: Vec<Value> = match resp {
        ModelResponse::Content(parts) => parts.iter().map(|item| {
            let text = match &item.content {
                Content::Text(t) => t.clone(),
                Content::Blob { data, .. } => data.clone(),
            };
            Value::object(FxHashMap::from_iter([
                (role_key, Value::string(item.role.clone())),
                (content_key, Value::string(text)),
                (content_type_key, Value::string("text")),
            ]))
        }).collect(),
        ModelResponse::ToolCalls(_) => vec![],
    };

    TypedValue::new(
        Value::list(items),
        Ty::List(Box::new(message_elem_ty(interner))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_interpreter::Stepped;

    struct MockFetch {
        response: serde_json::Value,
    }

    impl Fetch for MockFetch {
        async fn fetch(&self, _request: &HttpRequest) -> Result<serde_json::Value, String> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn basic_response() {
        let interner = Interner::new();
        let msg_id = ContextId::alloc();
        let role = interner.intern("user");

        let fetch = Arc::new(MockFetch {
            response: serde_json::json!({
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi from Claude!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }),
        });

        let unit = AnthropicUnit::new(
            Freeze::new(AnthropicConfig {
                endpoint: "http://test".into(),
                api_key: "key".into(),
                model: "claude-3".into(),
                message_ids: vec![(role, msg_id)],
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: None,
            }),
            fetch,
            &interner,
        );

        let co = unit.spawn(FxHashMap::default());
        let (co, stepped) = co.step().await;
        match stepped {
            Stepped::NeedContext(r) => r.resolve(TypedValue::string("Hello")),
            _ => panic!("expected NeedContext"),
        };

        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => {
                let items = value.value().expect_ref::<[acvus_interpreter::Value]>("test");
                assert_eq!(items.len(), 1);
                let obj = items[0].expect_ref::<FxHashMap<Astr, acvus_interpreter::Value>>("item");
                let content = obj.get(&interner.intern("content")).unwrap()
                    .expect_ref::<str>("content");
                assert_eq!(content, "Hi from Claude!");
            }
            _ => panic!("expected Emit"),
        }
    }
}
