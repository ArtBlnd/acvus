/// A chat message with role and content.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A tool call requested by the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
}

/// Model response: either text or tool calls.
#[derive(Debug, Clone)]
pub enum ModelResponse {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

/// Tool specification passed to the model.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub params: Vec<(String, String)>,
}

/// Node output stored in storage.
#[derive(Debug, Clone)]
pub enum Output {
    Text(String),
    Json(serde_json::Value),
    Image(Vec<u8>),
}
