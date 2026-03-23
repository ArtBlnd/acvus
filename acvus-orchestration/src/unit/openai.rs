use std::sync::Arc;

use acvus_interpreter::{RuntimeError, TypedValue, Value};
use acvus_mir::graph::ContextId;
use acvus_utils::{Astr, Freeze, Interner};
use rust_decimal::Decimal;
use rustc_hash::FxHashMap;

use acvus_mir::ty::Ty;

use super::Unit;
use crate::dsl::message_elem_ty;
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, Usage};

mod schema {
    pub use crate::unit::schema::openai::*;
}

#[derive(Debug, Clone)]
pub struct ToolBinding {
    pub name: String,
    pub description: String,
    pub unit_id: ContextId,
}

#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub message_ids: Vec<(Astr, ContextId)>,
    pub tools: Vec<ToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub max_tokens: Option<u32>,
    pub max_tool_rounds: u32,
}

pub struct OpenAIUnit<F> {
    config: Freeze<OpenAIConfig>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> OpenAIUnit<F> {
    pub fn new(config: Freeze<OpenAIConfig>, fetch: Arc<F>, interner: &Interner) -> Self {
        Self {
            config,
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F: Fetch + 'static> Unit for OpenAIUnit<F> {
    fn spawn(
        &self,
        _local_context: FxHashMap<ContextId, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError, ContextId> {
        let config = self.config.clone();
        let fetch = Arc::clone(&self.fetch);
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            // 1. Gather rendered messages via NeedContext
            let mut messages = Vec::new();
            for (role, msg_id) in &config.message_ids {
                let rendered: TypedValue = handle.request_context(*msg_id).await;
                let text = rendered.value().expect_ref::<str>("openai message").to_string();
                messages.push(Message::Content {
                    role: interner.resolve(*role).to_string(),
                    content: Content::Text(text),
                });
            }

            // Tool name → ContextId mapping
            let tool_id_map: FxHashMap<String, ContextId> = config.tools.iter()
                .map(|t| (t.name.clone(), t.unit_id))
                .collect();

            let tool_specs: Option<Vec<schema::Tool>> = if config.tools.is_empty() {
                None
            } else {
                Some(config.tools.iter().map(|t| schema::Tool {
                    tool_type: "function".into(),
                    function: schema::FunctionDecl {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: schema::FunctionParams {
                            schema_type: "object".into(),
                            properties: Default::default(),
                        },
                    },
                }).collect())
            };

            // 2. Request loop (handles tool calls)
            for _ in 0..=config.max_tool_rounds {
                let request_body = schema::Request {
                    model: config.model.clone(),
                    messages: messages.iter().map(convert_message).collect(),
                    tools: tool_specs.clone(),
                    temperature: config.temperature,
                    top_p: config.top_p,
                    max_tokens: config.max_tokens,
                    reasoning_effort: None,
                };

                let http_request = HttpRequest {
                    url: config.endpoint.clone(),
                    headers: vec![
                        ("Authorization".into(), format!("Bearer {}", config.api_key)),
                        ("Content-Type".into(), "application/json".into()),
                    ],
                    body: serde_json::to_value(&request_body).unwrap(),
                };

                let response_json = fetch.fetch(&http_request).await
                    .map_err(RuntimeError::fetch)?;

                let (response, _usage) = parse_response(response_json)
                    .map_err(|e| RuntimeError::fetch(e.to_string()))?;

                match response {
                    ModelResponse::Content(_) => {
                        let result = response_to_value(&response, &interner);
                        handle.yield_val(result).await;
                        return Ok(());
                    }
                    ModelResponse::ToolCalls(calls) => {
                        // Add assistant's tool call message
                        messages.push(Message::ToolCalls(calls.clone()));

                        // Execute each tool call
                        for call in &calls {
                            let tool_id = tool_id_map.get(&call.name)
                                .ok_or_else(|| RuntimeError::fetch(
                                    format!("unknown tool: {}", call.name)
                                ))?;
                            let result = handle
                                .request_extern_call(*tool_id, vec![TypedValue::new(
                                    Value::Pure(acvus_interpreter::PureValue::String(call.arguments.to_string())),
                                    acvus_mir::ty::Ty::String,
                                )])
                                .await;
                            let result_str = result.value().expect_ref::<str>("tool result").to_string();
                            messages.push(Message::ToolResult {
                                call_id: call.id.clone(),
                                content: result_str,
                            });
                        }
                    }
                }
            }

            Err(RuntimeError::tool_call_limit(config.max_tool_rounds as usize))
        })
    }
}

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
            tool_calls: calls.iter().map(|tc| schema::RequestToolCall {
                id: tc.id.clone(),
                call_type: "function".into(),
                function: schema::RequestToolCallFunction {
                    name: tc.name.clone(),
                    arguments: tc.arguments.to_string(),
                },
            }).collect(),
        },
        Message::ToolResult { call_id, content } => schema::RequestMessage::ToolResult {
            role: "tool".into(),
            tool_call_id: call_id.clone(),
            content: content.clone(),
        },
    }
}

fn parse_response(json: serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response = serde_json::from_value(json)
        .map_err(|e| RequestError::ResponseParse { detail: e.to_string() })?;

    let choice = resp.choices.into_iter().next()
        .ok_or(RequestError::EmptyResponse)?;

    let usage = Usage {
        input_tokens: resp.usage.as_ref().and_then(|u| u.prompt_tokens),
        output_tokens: resp.usage.as_ref().and_then(|u| u.completion_tokens),
    };

    // Check for tool calls first
    if let Some(tool_calls) = choice.message.tool_calls {
        let calls: Vec<ToolCall> = tool_calls.into_iter().filter_map(|tc| {
            Some(ToolCall {
                id: tc.id?,
                name: tc.function.as_ref()?.name.clone()?,
                arguments: serde_json::from_str(tc.function?.arguments.as_deref()?)
                    .unwrap_or_default(),
                extras: None,
            })
        }).collect();
        if !calls.is_empty() {
            return Ok((ModelResponse::ToolCalls(calls), usage));
        }
    }

    // Content
    let text = match choice.message.content {
        Some(schema::ResponseContent::Text(t)) => t,
        Some(schema::ResponseContent::Parts(parts)) => {
            parts.into_iter().filter_map(|p| p.text).collect::<Vec<_>>().join("")
        }
        None => String::new(),
    };

    let role = choice.message.role.unwrap_or_else(|| "assistant".into());
    Ok((
        ModelResponse::Content(vec![ContentItem {
            role,
            content: Content::Text(text),
        }]),
        usage,
    ))
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

    fn openai_response(content: &str) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        })
    }

    #[tokio::test]
    async fn basic_response() {
        let interner = Interner::new();
        let msg_id = ContextId::alloc();
        let role = interner.intern("user");

        let fetch = Arc::new(MockFetch {
            response: openai_response("Hello!"),
        });

        let unit = OpenAIUnit::new(
            Freeze::new(OpenAIConfig {
                endpoint: "http://test".into(),
                api_key: "key".into(),
                model: "gpt-4".into(),
                message_ids: vec![(role, msg_id)],
                tools: vec![],
                temperature: None,
                top_p: None,
                max_tokens: None,
                max_tool_rounds: 0,
            }),
            fetch,
            &interner,
        );

        let co = unit.spawn(FxHashMap::default());

        // Step 1: NeedContext for message
        let (co, stepped) = co.step().await;
        let request = match stepped {
            Stepped::NeedContext(r) => {
                assert_eq!(r.key(), msg_id);
                r
            }
            _ => panic!("expected NeedContext"),
        };
        request.resolve(TypedValue::string("Hi"));

        // Step 2: Emit result
        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => {
                let items = value.value().expect_ref::<[acvus_interpreter::Value]>("test");
                assert_eq!(items.len(), 1);
                let obj = items[0].expect_ref::<FxHashMap<Astr, acvus_interpreter::Value>>("item");
                let content_key = interner.intern("content");
                let content = obj.get(&content_key).unwrap().expect_ref::<str>("content");
                assert_eq!(content, "Hello!");
            }
            _ => panic!("expected Emit"),
        }
    }

    #[tokio::test]
    async fn tool_call_loop() {
        let interner = Interner::new();
        let msg_id = ContextId::alloc();
        let tool_unit_id = ContextId::alloc();
        let role = interner.intern("user");

        // First response: tool call. Second response: content.
        let responses = std::sync::Arc::new(std::sync::Mutex::new(vec![
            // 2nd call → content
            serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "The weather is sunny."}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}
            }),
            // 1st call → tool call
            serde_json::json!({
                "choices": [{"message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Seoul\"}"}
                    }]
                }}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            }),
        ]));

        struct SeqFetch(std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>);
        impl Fetch for SeqFetch {
            async fn fetch(&self, _req: &HttpRequest) -> Result<serde_json::Value, String> {
                Ok(self.0.lock().unwrap().pop().unwrap())
            }
        }

        let fetch = Arc::new(SeqFetch(responses));

        let unit = OpenAIUnit::new(
            Freeze::new(OpenAIConfig {
                endpoint: "http://test".into(),
                api_key: "key".into(),
                model: "gpt-4".into(),
                message_ids: vec![(role, msg_id)],
                tools: vec![ToolBinding {
                    name: "get_weather".into(),
                    description: "Get weather".into(),
                    unit_id: tool_unit_id,
                }],
                temperature: None,
                top_p: None,
                max_tokens: None,
                max_tool_rounds: 3,
            }),
            fetch,
            &interner,
        );

        let co = unit.spawn(FxHashMap::default());

        // Step 1: NeedContext for message
        let (co, stepped) = co.step().await;
        match stepped {
            Stepped::NeedContext(r) => r.resolve(TypedValue::string("What's the weather?")),
            _ => panic!("expected NeedContext"),
        };

        // Step 2: NeedExternCall for tool
        let (co, stepped) = co.step().await;
        match stepped {
            Stepped::NeedExternCall(r) => {
                assert_eq!(r.key(), tool_unit_id);
                r.resolve(TypedValue::string("sunny, 25°C"));
            }
            _ => panic!("expected NeedExternCall for tool"),
        };

        // Step 3: Emit final content
        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => {
                let items = value.value().expect_ref::<[acvus_interpreter::Value]>("test");
                assert_eq!(items.len(), 1);
                let obj = items[0].expect_ref::<FxHashMap<Astr, acvus_interpreter::Value>>("item");
                let content = obj.get(&interner.intern("content")).unwrap()
                    .expect_ref::<str>("content");
                assert_eq!(content, "The weather is sunny.");
            }
            _ => panic!("expected Emit"),
        }
    }
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
