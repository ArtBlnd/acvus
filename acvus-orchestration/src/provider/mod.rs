mod openai;
mod anthropic;
mod google;

use crate::message::{Message, ModelResponse, ToolSpec};

#[derive(Debug, Clone)]
pub enum ApiKind {
    OpenAI,
    Anthropic,
    Google,
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api: ApiKind,
    pub endpoint: String,
    pub api_key: String,
}

pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: serde_json::Value,
}

/// Raw HTTP fetch — implementors only handle transport.
#[trait_variant::make(Send)]
pub trait Fetch: Sync {
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String>;
}

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
) -> HttpRequest {
    match config.api {
        ApiKind::OpenAI => openai::build_request(config, model, messages, tools),
        ApiKind::Anthropic => anthropic::build_request(config, model, messages, tools),
        ApiKind::Google => google::build_request(config, model, messages, tools),
    }
}

pub fn parse_response(
    api: &ApiKind,
    json: &serde_json::Value,
) -> Result<ModelResponse, String> {
    match api {
        ApiKind::OpenAI => openai::parse_response(json),
        ApiKind::Anthropic => anthropic::parse_response(json),
        ApiKind::Google => google::parse_response(json),
    }
}
