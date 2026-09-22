use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// -- Request ---------------------------------------------------------

#[derive(Serialize)]
pub struct Request {
    pub contents: Vec<Content>,
    #[serde(rename = "systemInstruction")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(rename = "generationConfig")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum Part {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallPayload,
        #[serde(rename = "thoughtSignature")]
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponsePayload,
    },
}

#[derive(Serialize)]
pub struct InlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Serialize)]
pub struct FunctionCallPayload {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Serialize)]
pub struct FunctionResponsePayload {
    pub name: String,
    pub response: FunctionResponseContent,
}

#[derive(Serialize)]
pub struct FunctionResponseContent {
    pub content: String,
}

#[derive(Serialize)]
pub struct SystemInstruction {
    pub parts: Vec<TextPart>,
}

#[derive(Serialize)]
pub struct TextPart {
    pub text: String,
}

#[derive(Serialize)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Decimal>,
    #[serde(rename = "topP")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Decimal>,
    #[serde(rename = "topK")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(rename = "maxOutputTokens")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(rename = "thinkingConfig")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[derive(Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    #[serde(rename = "thinkingLevel")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,
}

// -- Response --------------------------------------------------------

#[derive(Deserialize)]
pub struct Response {
    pub candidates: Option<Vec<Candidate>>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
pub struct Candidate {
    pub content: Option<CandidateContent>,
}

#[derive(Deserialize)]
pub struct CandidateContent {
    pub role: Option<String>,
    pub parts: Option<Vec<ResponsePart>>,
}

#[derive(Deserialize)]
pub struct ResponsePart {
    pub text: Option<String>,
    #[serde(rename = "functionCall")]
    pub function_call: Option<ResponseFunctionCall>,
}

#[derive(Deserialize)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Deserialize)]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
}
