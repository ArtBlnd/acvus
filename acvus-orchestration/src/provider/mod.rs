mod anthropic;
mod google;
mod openai;

use std::collections::{BTreeMap, HashMap};

use acvus_interpreter::Value;
use acvus_mir::ty::Ty;

use crate::kind::GenerationParams;
use crate::message::{Message, ModelResponse, ToolSpec, Usage};

#[derive(Debug, Clone)]
pub enum ApiKind {
    OpenAI,
    Anthropic,
    Google,
}

impl ApiKind {
    pub fn parse(s: &str) -> Option<ApiKind> {
        match s {
            "openai" => Some(ApiKind::OpenAI),
            "anthropic" => Some(ApiKind::Anthropic),
            "google" => Some(ApiKind::Google),
            _ => None,
        }
    }

    pub fn message_elem_ty(&self) -> Ty {
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("role".into(), Ty::String),
            ("content".into(), Ty::String),
            ("content_type".into(), Ty::String),
        ]))))
    }

    pub fn item_fields<'a>(&self, item: &'a Value) -> (&'a str, &'a str, &'a str) {
        let Value::Object(obj) = item else {
            panic!("item_fields: expected Object, got {item:?}");
        };
        let Some(Value::String(role)) = obj.get("role") else {
            panic!("item_fields: missing or non-string 'role'");
        };
        let Some(Value::String(content)) = obj.get("content") else {
            panic!("item_fields: missing or non-string 'content'");
        };
        let Some(Value::String(content_type)) = obj.get("content_type") else {
            panic!("item_fields: missing or non-string 'content_type'");
        };
        (role.as_str(), content.as_str(), content_type.as_str())
    }
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

impl<F> Fetch for std::sync::Arc<F>
where
    F: Fetch,
{
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
        (**self).fetch(request).await
    }
}

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
    cached_content: Option<&str>,
) -> HttpRequest {
    match config.api {
        ApiKind::OpenAI => openai::build_request(
            config,
            model,
            messages,
            tools,
            generation,
            max_output_tokens,
        ),
        ApiKind::Anthropic => anthropic::build_request(
            config,
            model,
            messages,
            tools,
            generation,
            max_output_tokens,
        ),
        ApiKind::Google => google::build_request(
            config,
            model,
            messages,
            tools,
            generation,
            max_output_tokens,
            cached_content,
        ),
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

/// Provider-specific model abstraction — enum dispatch over provider implementations.
pub enum LlmModelKind {
    OpenAI(openai::OpenAiModel),
    Anthropic(anthropic::AnthropicModel),
    Google(google::GoogleModel),
}

impl LlmModelKind {
    pub fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        generation: &GenerationParams,
        max_output_tokens: Option<u32>,
        cached_content: Option<&str>,
    ) -> HttpRequest {
        match self {
            LlmModelKind::OpenAI(m) => m.build_request(
                messages,
                tools,
                generation,
                max_output_tokens,
                cached_content,
            ),
            LlmModelKind::Anthropic(m) => m.build_request(
                messages,
                tools,
                generation,
                max_output_tokens,
                cached_content,
            ),
            LlmModelKind::Google(m) => m.build_request(
                messages,
                tools,
                generation,
                max_output_tokens,
                cached_content,
            ),
        }
    }

    pub fn parse_response(
        &self,
        json: &serde_json::Value,
    ) -> Result<(ModelResponse, Usage), String> {
        match self {
            LlmModelKind::OpenAI(m) => m.parse_response(json),
            LlmModelKind::Anthropic(m) => m.parse_response(json),
            LlmModelKind::Google(m) => m.parse_response(json),
        }
    }

    pub fn build_count_tokens_request(&self, messages: &[Message]) -> Option<HttpRequest> {
        match self {
            LlmModelKind::OpenAI(m) => m.build_count_tokens_request(messages),
            LlmModelKind::Anthropic(m) => m.build_count_tokens_request(messages),
            LlmModelKind::Google(m) => m.build_count_tokens_request(messages),
        }
    }

    pub fn parse_count_tokens_response(&self, json: &serde_json::Value) -> Result<u32, String> {
        match self {
            LlmModelKind::OpenAI(m) => m.parse_count_tokens_response(json),
            LlmModelKind::Anthropic(m) => m.parse_count_tokens_response(json),
            LlmModelKind::Google(m) => m.parse_count_tokens_response(json),
        }
    }
}

pub fn create_llm_model(config: ProviderConfig, model: String) -> LlmModelKind {
    match config.api {
        ApiKind::OpenAI => LlmModelKind::OpenAI(openai::OpenAiModel::new(config, model)),
        ApiKind::Anthropic => {
            LlmModelKind::Anthropic(anthropic::AnthropicModel::new(config, model))
        }
        ApiKind::Google => LlmModelKind::Google(google::GoogleModel::new(config, model)),
    }
}
