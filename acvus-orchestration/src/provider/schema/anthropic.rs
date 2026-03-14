use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ── Request ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingParam>,
}

#[derive(Serialize)]
pub struct RequestMessage {
    pub role: String,
    pub content: RequestContent,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum RequestContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ThinkingParam {
    #[serde(rename = "enabled")]
    Enabled { budget_tokens: u32 },
    #[serde(rename = "disabled")]
    Disabled {},
}

#[derive(Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: InputSchema,
}

#[derive(Serialize)]
pub struct InputSchema {
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

// ── Count Tokens Request ────────────────────────────────────────────

#[derive(Serialize)]
pub struct CountTokensRequest {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
}

// ── Response ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct Response {
    pub role: Option<String>,
    pub content: Vec<ResponseContentBlock>,
    pub usage: Option<ResponseUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ResponseImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Option<serde_json::Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
pub struct ResponseImageSource {
    pub media_type: Option<String>,
    pub data: Option<String>,
}

#[derive(Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

// ── Count Tokens Response ───────────────────────────────────────────

#[derive(Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: Option<u32>,
}
