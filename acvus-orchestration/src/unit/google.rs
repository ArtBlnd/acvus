use std::sync::Arc;

use acvus_interpreter::{RuntimeError, TypedValue, Value};
use acvus_mir::graph::ContextId;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rust_decimal::Decimal;
use rustc_hash::FxHashMap;

use super::Unit;
use super::openai::ToolBinding;
use crate::dsl::message_elem_ty;
use crate::http::{Fetch, HttpRequest, RequestError};
use crate::message::{Content, ContentItem, Message, ModelResponse, ToolCall, Usage};

mod schema {
    pub use crate::unit::schema::google::*;
}

#[derive(Debug, Clone)]
pub struct GoogleConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub message_ids: Vec<(Astr, ContextId)>,
    pub tools: Vec<ToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub grounding: bool,
    pub max_tool_rounds: u32,
}

pub struct GoogleUnit<F> {
    config: Freeze<GoogleConfig>,
    fetch: Arc<F>,
    interner: Interner,
}

impl<F> GoogleUnit<F> {
    pub fn new(config: Freeze<GoogleConfig>, fetch: Arc<F>, interner: &Interner) -> Self {
        Self {
            config,
            fetch,
            interner: interner.clone(),
        }
    }
}

impl<F: Fetch + 'static> Unit for GoogleUnit<F> {
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
                let text = rendered.value().expect_ref::<str>("google message").to_string();
                messages.push(Message::Content {
                    role: interner.resolve(*role).to_string(),
                    content: Content::Text(text),
                });
            }

            let tool_id_map: FxHashMap<String, ContextId> = config.tools.iter()
                .map(|t| (t.name.clone(), t.unit_id))
                .collect();

            let url = format!("{}{}:generateContent?key={}", config.endpoint, config.model, config.api_key);

            for _ in 0..=config.max_tool_rounds {
                let (system, rest) = split_system(&messages);
                let request_body = schema::Request {
                    contents: rest.iter().map(|m| convert_message(m)).collect(),
                    system_instruction: system.map(|s| schema::SystemInstruction {
                        parts: vec![schema::TextPart { text: s }],
                    }),
                    tools: None,
                    generation_config: Some(schema::GenerationConfig {
                        temperature: config.temperature,
                        top_p: config.top_p,
                        top_k: config.top_k,
                        max_output_tokens: config.max_tokens,
                        thinking_config: None,
                    }),
                };

                let http_request = HttpRequest {
                    url: url.clone(),
                    headers: vec![("Content-Type".into(), "application/json".into())],
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
                        messages.push(Message::ToolCalls(calls.clone()));
                        for call in &calls {
                            let tool_id = tool_id_map.get(&call.name)
                                .ok_or_else(|| RuntimeError::fetch(format!("unknown tool: {}", call.name)))?;
                            let result = handle
                                .request_extern_call(*tool_id, vec![TypedValue::string(call.arguments.to_string())])
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
            parts: calls.iter().map(|tc| schema::Part::FunctionCall {
                function_call: schema::FunctionCallPayload {
                    name: tc.name.clone(),
                    args: tc.arguments.clone(),
                },
                thought_signature: None,
            }).collect(),
        },
        Message::ToolResult { call_id, content } => schema::Content {
            role: "user".into(),
            parts: vec![schema::Part::FunctionResponse {
                function_response: schema::FunctionResponsePayload {
                    name: call_id.clone(),
                    response: schema::FunctionResponseContent { content: content.clone() },
                },
            }],
        },
    }
}

fn parse_response(json: serde_json::Value) -> Result<(ModelResponse, Usage), RequestError> {
    let resp: schema::Response = serde_json::from_value(json)
        .map_err(|e| RequestError::ResponseParse { detail: e.to_string() })?;

    let candidate = resp.candidates
        .and_then(|mut c| if c.is_empty() { None } else { Some(c.remove(0)) })
        .ok_or(RequestError::EmptyResponse)?;

    let usage = Usage {
        input_tokens: resp.usage_metadata.as_ref().and_then(|u| u.prompt_token_count),
        output_tokens: resp.usage_metadata.as_ref().and_then(|u| u.candidates_token_count),
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
            if let Some(name) = fc.name {
                tool_calls.push(ToolCall {
                    id: name.clone(),
                    name,
                    arguments: fc.args.unwrap_or_default(),
                    extras: None,
                });
            }
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

    TypedValue::new(Value::list(items), Ty::List(Box::new(message_elem_ty(interner))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_interpreter::Stepped;

    struct MockFetch { response: serde_json::Value }
    impl Fetch for MockFetch {
        async fn fetch(&self, _req: &HttpRequest) -> Result<serde_json::Value, String> {
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
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello from Gemini!"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
            }),
        });

        let unit = GoogleUnit::new(
            Freeze::new(GoogleConfig {
                endpoint: "http://test/".into(),
                api_key: "key".into(),
                model: "gemini-pro".into(),
                message_ids: vec![(role, msg_id)],
                tools: vec![],
                temperature: None, top_p: None, top_k: None, max_tokens: None,
                grounding: false, max_tool_rounds: 0,
            }),
            fetch, &interner,
        );

        let co = unit.spawn(FxHashMap::default());
        let (co, stepped) = co.step().await;
        match stepped {
            Stepped::NeedContext(r) => r.resolve(TypedValue::string("Hi")),
            _ => panic!("expected NeedContext"),
        };

        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => {
                let items = value.value().expect_ref::<[acvus_interpreter::Value]>("test");
                assert_eq!(items.len(), 1);
                let obj = items[0].expect_ref::<FxHashMap<Astr, acvus_interpreter::Value>>("item");
                let content = obj.get(&interner.intern("content")).unwrap().expect_ref::<str>("c");
                assert_eq!(content, "Hello from Gemini!");
            }
            _ => panic!("expected Emit"),
        }
    }
}
