mod anthropic;
mod google;
mod openai;

use std::collections::HashMap;

use crate::dsl::GenerationParams;
use crate::message::{Message, ModelResponse, ToolSpec, Usage};

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
    generation: &GenerationParams,
    cached_content: Option<&str>,
) -> HttpRequest {
    match config.api {
        ApiKind::OpenAI => openai::build_request(config, model, messages, tools, generation),
        ApiKind::Anthropic => anthropic::build_request(config, model, messages, tools, generation),
        ApiKind::Google => {
            google::build_request(config, model, messages, tools, generation, cached_content)
        }
    }
}

pub fn build_cache_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    ttl: &str,
    cache_config: &HashMap<String, serde_json::Value>,
) -> HttpRequest {
    match config.api {
        ApiKind::Google => google::build_cache_request(config, model, messages, ttl, cache_config),
        _ => panic!("context caching not supported for {:?}", config.api),
    }
}

pub fn parse_cache_response(api: &ApiKind, json: &serde_json::Value) -> Result<String, String> {
    match api {
        ApiKind::Google => google::parse_cache_response(json),
        _ => Err("context caching not supported".into()),
    }
}

pub fn parse_response(
    api: &ApiKind,
    json: &serde_json::Value,
) -> Result<(ModelResponse, Usage), String> {
    match api {
        ApiKind::OpenAI => openai::parse_response(json),
        ApiKind::Anthropic => anthropic::parse_response(json),
        ApiKind::Google => google::parse_response(json),
    }
}

/// Provider-specific model abstraction — handles request building, response parsing, and token counting.
pub trait LlmModel {
    fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        generation: &GenerationParams,
        cached_content: Option<&str>,
    ) -> HttpRequest;

    fn parse_response(&self, json: &serde_json::Value) -> Result<(ModelResponse, Usage), String>;

    /// Returns `None` if the provider doesn't support token counting.
    fn build_count_tokens_request(&self, messages: &[Message]) -> Option<HttpRequest>;

    fn parse_count_tokens_response(&self, json: &serde_json::Value) -> Result<u32, String>;
}

pub fn create_llm_model(config: ProviderConfig, model: String) -> Box<dyn LlmModel> {
    match config.api {
        ApiKind::OpenAI => Box::new(openai::OpenAiModel::new(config, model)),
        ApiKind::Anthropic => Box::new(anthropic::AnthropicModel::new(config, model)),
        ApiKind::Google => Box::new(google::GoogleModel::new(config, model)),
    }
}
