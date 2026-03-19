use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ── Request ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum RequestMessage {
    Content {
        role: String,
        content: String,
    },
    ContentArray {
        role: String,
        content: Vec<ContentPart>,
    },
    ToolCalls {
        role: String,
        tool_calls: Vec<RequestToolCall>,
    },
    ToolResult {
        role: String,
        tool_call_id: String,
        content: String,
    },
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlData },
}

#[derive(Serialize)]
pub struct ImageUrlData {
    pub url: String,
}

#[derive(Serialize)]
pub struct RequestToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: RequestToolCallFunction,
}

#[derive(Serialize)]
pub struct RequestToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDecl,
}

#[derive(Serialize)]
pub struct FunctionDecl {
    pub name: String,
    pub description: String,
    pub parameters: FunctionParams,
}

#[derive(Serialize)]
pub struct FunctionParams {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

// ── Response ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct Response {
    pub choices: Vec<Choice>,
    pub usage: Option<ResponseUsage>,
}

#[derive(Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
}

#[derive(Deserialize)]
pub struct ResponseMessage {
    pub role: Option<String>,
    pub content: Option<ResponseContent>,
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum ResponseContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

#[derive(Deserialize)]
pub struct ResponseContentPart {
    #[serde(rename = "type")]
    pub part_type: Option<String>,
    pub text: Option<String>,
}

#[derive(Deserialize)]
pub struct ResponseToolCall {
    pub id: Option<String>,
    pub function: Option<ResponseToolCallFunction>,
}

#[derive(Deserialize)]
pub struct ResponseToolCallFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Deserialize)]
pub struct ResponseUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}
