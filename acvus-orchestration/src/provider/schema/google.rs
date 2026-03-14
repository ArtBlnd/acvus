use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ── Request ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Request {
    pub contents: Vec<Content>,
    #[serde(rename = "systemInstruction")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDeclaration>>,
    #[serde(rename = "generationConfig")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
pub struct CachedRequest {
    #[serde(rename = "cachedContent")]
    pub cached_content: String,
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDeclaration>>,
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
#[serde(untagged)]
pub enum ToolDeclaration {
    Functions {
        #[serde(rename = "functionDeclarations")]
        function_declarations: Vec<FunctionDecl>,
    },
    GoogleSearch {
        google_search: GoogleSearchConfig,
    },
}

#[derive(Serialize)]
pub struct GoogleSearchConfig {}

#[derive(Serialize)]
pub struct FunctionDecl {
    pub name: String,
    pub description: String,
    pub parameters: GeminiSchema,
}

#[derive(Serialize)]
pub struct GeminiSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub required: Vec<String>,
}

#[derive(Serialize)]
pub struct GeminiPropertySchema {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
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

// ── Cache Request ───────────────────────────────────────────────────

#[derive(Serialize)]
pub struct CacheRequest {
    pub model: String,
    pub contents: Vec<Content>,
    pub ttl: String,
    #[serde(rename = "systemInstruction")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ── Count Tokens Request ────────────────────────────────────────────

#[derive(Serialize)]
pub struct CountTokensRequest {
    #[serde(rename = "generateContentRequest")]
    pub generate_content_request: CountTokensInner,
}

#[derive(Serialize)]
pub struct CountTokensInner {
    pub model: String,
    pub contents: Vec<Content>,
    #[serde(rename = "systemInstruction")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
}

// ── Response ────────────────────────────────────────────────────────

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
    #[serde(rename = "inlineData")]
    pub inline_data: Option<ResponseInlineData>,
    #[serde(rename = "functionCall")]
    pub function_call: Option<ResponseFunctionCall>,
    #[serde(rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
}

#[derive(Deserialize)]
pub struct ResponseInlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
    pub data: Option<String>,
}

#[derive(Deserialize)]
pub struct ResponseFunctionCall {
    pub name: Option<String>,
    pub args: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
}

// ── Cache Response ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CacheResponse {
    pub name: Option<String>,
}

// ── Count Tokens Response ───────────────────────────────────────────

#[derive(Deserialize)]
pub struct CountTokensResponse {
    #[serde(rename = "totalTokens")]
    pub total_tokens: Option<u32>,
}
