use std::collections::HashMap;

use acvus_interpreter::{ExternFnRegistry, Interpreter, Value};
use acvus_mir::extern_module::ExternRegistry;

use crate::compile::CompiledNode;
use crate::dag::Dag;
use crate::error::{OrchError, OrchErrorKind};
use crate::message::{Message, ModelResponse, Output, ToolCall, ToolResult, ToolSpec};
use crate::provider::Fetch;
use crate::storage::Storage;

pub struct Executor<F, S>
where
    F: Fetch,
    S: Storage,
{
    nodes: Vec<CompiledNode>,
    dag: Dag,
    storage: S,
    fetch: F,
    mir_registry: ExternRegistry,
    fuel_limit: u64,
    fuel_consumed: u64,
}

impl<F, S> Executor<F, S>
where
    F: Fetch,
    S: Storage,
{
    pub fn new(
        nodes: Vec<CompiledNode>,
        dag: Dag,
        storage: S,
        fetch: F,
        mir_registry: ExternRegistry,
        fuel_limit: u64,
    ) -> Self {
        Self {
            nodes,
            dag,
            storage,
            fetch,
            mir_registry,
            fuel_limit,
            fuel_consumed: 0,
        }
    }

    /// Run the full DAG in topological order, returning the final storage.
    pub async fn run(mut self) -> Result<S, OrchError> {
        let topo_order = self.dag.topo_order.clone();
        for &idx in &topo_order {
            self.execute_node(idx).await?;
        }
        Ok(self.storage)
    }

    async fn execute_node(&mut self, idx: usize) -> Result<(), OrchError> {
        let node = &self.nodes[idx];
        let model = node.model.clone();
        let node_name = node.name.clone();
        let blocks: Vec<_> = node
            .blocks
            .iter()
            .map(|b| (b.module.clone(), b.role.clone(), b.context_keys.clone()))
            .collect();
        let tools: Vec<ToolSpec> = node
            .tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name.clone(),
                description: String::new(),
                params: t.params.clone(),
            })
            .collect();

        // Build context from storage based on context keys
        let context = self.build_context(idx);

        // Render each block into a message
        let mut messages = Vec::new();
        for (module, role, _context_keys) in &blocks {
            let context_values: HashMap<String, Value> = context
                .iter()
                .map(|(k, v)| (k.clone(), output_to_value(v)))
                .collect();

            let interp =
                Interpreter::new(module.clone(), context_values, ExternFnRegistry::new());
            let output = interp.execute_to_string().await;

            messages.push(Message { role: role.clone(), content: output });
        }

        // Format request, call fetch, parse response
        self.consume_fuel(1)?;

        let body = format_request(&messages, &tools, &model);
        let response_json = self
            .fetch
            .fetch(body)
            .await
            .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
        let mut response = parse_response(&response_json)
            .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;

        // Tool call loop
        let mut all_messages = messages;
        while let ModelResponse::ToolCalls(ref calls) = response {
            self.consume_fuel(1)?;

            let tool_results = self.handle_tool_calls(calls)?;

            // Append assistant tool_calls message
            all_messages.push(Message {
                role: "assistant".into(),
                content: format_tool_calls_content(calls),
            });

            // Append tool results as messages
            for result in &tool_results {
                all_messages.push(Message {
                    role: "tool".into(),
                    content: result.content.clone(),
                });
            }

            // Re-call model
            let body = format_request(&all_messages, &tools, &model);
            let resp_json = self
                .fetch
                .fetch(body)
                .await
                .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
            response = parse_response(&resp_json)
                .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
        }

        // Store output
        if let ModelResponse::Text(text) = response {
            self.storage.set(node_name, Output::Text(text));
        }

        Ok(())
    }

    fn build_context(&self, idx: usize) -> HashMap<String, Output> {
        let node = &self.nodes[idx];
        let mut context = HashMap::new();
        for key in &node.all_context_keys {
            if let Some(value) = self.storage.get(key) {
                context.insert(key.clone(), value);
            }
        }
        context
    }

    fn consume_fuel(&mut self, amount: u64) -> Result<(), OrchError> {
        self.fuel_consumed += amount;
        if self.fuel_consumed > self.fuel_limit {
            Err(OrchError::new(OrchErrorKind::FuelExhausted))
        } else {
            Ok(())
        }
    }

    fn handle_tool_calls(&self, calls: &[ToolCall]) -> Result<Vec<ToolResult>, OrchError> {
        let mut results = Vec::new();
        for call in calls {
            results.push(ToolResult {
                call_id: call.id.clone(),
                content: format!("tool '{}' not implemented", call.name),
            });
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Output -> Value conversion
// ---------------------------------------------------------------------------

fn output_to_value(output: &Output) -> Value {
    match output {
        Output::Text(s) => Value::String(s.clone()),
        Output::Json(v) => json_to_value(v),
        Output::Image(bytes) => Value::List(bytes.iter().map(|&b| Value::Byte(b)).collect()),
    }
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Unit
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            Value::Object(obj.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Request formatting (OpenAI-compatible)
// ---------------------------------------------------------------------------

fn format_request(messages: &[Message], tools: &[ToolSpec], model: &str) -> serde_json::Value {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content,
            })
        })
        .collect();

    let mut body = serde_json::json!({
        "model": model,
        "messages": msgs,
    });

    if !tools.is_empty() {
        let tool_specs: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                let properties: serde_json::Map<String, serde_json::Value> = t
                    .params
                    .iter()
                    .map(|(name, type_name)| {
                        (name.clone(), serde_json::json!({ "type": type_name }))
                    })
                    .collect();

                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                        }
                    }
                })
            })
            .collect();

        body["tools"] = serde_json::Value::Array(tool_specs);
    }

    body
}

fn format_tool_calls_content(calls: &[ToolCall]) -> String {
    calls
        .iter()
        .map(|c| format!("{}({})", c.name, c.arguments))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Response parsing (OpenAI-compatible)
// ---------------------------------------------------------------------------

fn parse_response(json: &serde_json::Value) -> Result<ModelResponse, String> {
    let choices = json
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or("missing 'choices' in response")?;

    let choice = choices.first().ok_or("empty choices array")?;

    let message = choice.get("message").ok_or("missing 'message' in choice")?;

    // Check for tool calls
    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        let calls: Result<Vec<ToolCall>, String> = tool_calls
            .iter()
            .map(|tc| {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool call id")?
                    .to_string();
                let func = tc.get("function").ok_or("missing function")?;
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing function name")?
                    .to_string();
                let arguments = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Object(Default::default()));

                Ok(ToolCall { id, name, arguments })
            })
            .collect();

        return Ok(ModelResponse::ToolCalls(calls?));
    }

    // Text response
    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    Ok(ModelResponse::Text(content))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_to_value_text() {
        let v = output_to_value(&Output::Text("hello".into()));
        assert!(matches!(v, Value::String(ref s) if s == "hello"));
    }

    #[test]
    fn output_to_value_json() {
        let v = output_to_value(&Output::Json(serde_json::json!({"name": "alice", "age": 30})));
        match v {
            Value::Object(obj) => {
                assert!(matches!(obj.get("name"), Some(Value::String(s)) if s == "alice"));
                assert!(matches!(obj.get("age"), Some(Value::Int(30))));
            }
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn output_to_value_image() {
        let v = output_to_value(&Output::Image(vec![0xff, 0x00]));
        match v {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(items[0], Value::Byte(0xff)));
            }
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn format_request_basic() {
        let messages = vec![
            Message { role: "system".into(), content: "You are helpful.".into() },
            Message { role: "user".into(), content: "Hello".into() },
        ];
        let body = format_request(&messages, &[], "gpt-4");
        assert_eq!(body["model"], "gpt-4");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn format_request_with_tools() {
        let messages = vec![Message { role: "user".into(), content: "hi".into() }];
        let tools = vec![ToolSpec {
            name: "search".into(),
            description: "Search the web".into(),
            params: HashMap::from([("query".into(), "string".into())]),
        }];
        let body = format_request(&messages, &tools, "gpt-4");
        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["function"]["name"], "search");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                }
            }]
        });
        let resp = parse_response(&json).unwrap();
        assert!(matches!(resp, ModelResponse::Text(ref s) if s == "Hello there!"));
    }

    #[test]
    fn parse_tool_call_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "search",
                            "arguments": "{\"query\": \"hello\"}"
                        }
                    }]
                }
            }]
        });
        let resp = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }
}
