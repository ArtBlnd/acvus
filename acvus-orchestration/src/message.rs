use std::collections::HashMap;

/// Message content — text or binary blob.
#[derive(Debug, Clone)]
pub enum Content {
    Text(String),
    Blob { mime_type: String, data: String },
}

/// A chat message — explicit variants, no implicit fields.
#[derive(Debug, Clone)]
pub enum Message {
    Content { role: String, content: Content },
    ToolCalls(Vec<ToolCall>),
    ToolResult { call_id: String, content: String },
}

/// A tool call requested by the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A single content part from a model response.
#[derive(Debug, Clone)]
pub struct ContentItem {
    pub role: String,
    pub content: Content,
}

/// Model response: either content parts or tool calls.
#[derive(Debug, Clone)]
pub enum ModelResponse {
    Content(Vec<ContentItem>),
    ToolCalls(Vec<ToolCall>),
}

/// Tool specification passed to the model.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub params: HashMap<String, String>,
}

/// Token usage from a model response.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

/// Node output stored in storage.
#[derive(Debug, Clone)]
pub enum Output {
    Text(String),
    Json(serde_json::Value),
    Image(Vec<u8>),
}
