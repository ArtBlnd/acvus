mod anthropic;
mod google;
mod openai;
mod schema;

use std::fmt;

use acvus_interpreter::{LazyValue, PureValue, Value};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::kind::{GenerationParams, ThinkingConfig};
use crate::message::{Content, Message, ModelResponse, ToolSpec, Usage};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiKind {
    OpenAI,
    Anthropic,
    Google,
}

impl ApiKind {

    pub fn message_elem_ty(&self, interner: &Interner) -> Ty {
        Ty::Object(FxHashMap::from_iter([
            (interner.intern("role"), Ty::String),
            (interner.intern("content"), Ty::String),
            (interner.intern("content_type"), Ty::String),
        ]))
    }

    pub fn item_fields<'a>(
        &self,
        interner: &Interner,
        item: &'a Value,
    ) -> (&'a str, &'a str, &'a str) {
        let Value::Lazy(LazyValue::Object(obj)) = item else {
            panic!("item_fields: expected Object, got {item:?}");
        };
        let role_key = interner.intern("role");
        let content_key = interner.intern("content");
        let content_type_key = interner.intern("content_type");
        let Some(Value::Pure(PureValue::String(role))) = obj.get(&role_key) else {
            panic!("item_fields: missing or non-string 'role'");
        };
        let Some(Value::Pure(PureValue::String(content))) = obj.get(&content_key) else {
            panic!("item_fields: missing or non-string 'content'");
        };
        let Some(Value::Pure(PureValue::String(content_type))) = obj.get(&content_type_key) else {
            panic!("item_fields: missing or non-string 'content_type'");
        };
        (role.as_str(), content.as_str(), content_type.as_str())
    }
}

// ── Provider errors ─────────────────────────────────────────────────

/// Errors that can occur when building a provider request.
#[derive(Debug, Clone)]
pub enum ProviderError {
    /// The provider does not support the given thinking config.
    UnsupportedThinkingConfig {
        provider: &'static str,
        config: ThinkingConfig,
    },
    /// Response failed to deserialize.
    ResponseParse {
        detail: String,
    },
    /// Expected field missing in response.
    MissingField {
        field: &'static str,
    },
    /// Provider does not support this operation.
    Unsupported {
        provider: &'static str,
        operation: &'static str,
    },
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::UnsupportedThinkingConfig { provider, config } => {
                write!(f, "{provider}: unsupported thinking config {config:?}")
            }
            ProviderError::ResponseParse { detail } => {
                write!(f, "response parse error: {detail}")
            }
            ProviderError::MissingField { field } => {
                write!(f, "missing field '{field}' in response")
            }
            ProviderError::Unsupported { provider, operation } => {
                write!(f, "{provider}: {operation} not supported")
            }
        }
    }
}

impl std::error::Error for ProviderError {}

// ── Config & Transport ──────────────────────────────────────────────

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

// ── Free-function dispatch ──────────────────────────────────────────

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
    max_output_tokens: Option<u32>,
    cached_content: Option<&str>,
) -> Result<HttpRequest, ProviderError> {
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
    cache_config: &FxHashMap<String, serde_json::Value>,
) -> HttpRequest {
    match config.api {
        ApiKind::Google => google::build_cache_request(config, model, messages, ttl, cache_config),
        _ => panic!("context caching not supported for {:?}", config.api),
    }
}

pub fn parse_cache_response(
    api: &ApiKind,
    json: &serde_json::Value,
) -> Result<String, ProviderError> {
    match api {
        ApiKind::Google => google::parse_cache_response(json),
        _ => Err(ProviderError::Unsupported {
            provider: "non-google",
            operation: "context caching",
        }),
    }
}

pub fn parse_response(
    api: &ApiKind,
    json: &serde_json::Value,
) -> Result<(ModelResponse, Usage), ProviderError> {
    match api {
        ApiKind::OpenAI => openai::parse_response(json),
        ApiKind::Anthropic => anthropic::parse_response(json),
        ApiKind::Google => google::parse_response(json),
    }
}

// ── LlmModelKind ────────────────────────────────────────────────────

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
    ) -> Result<HttpRequest, ProviderError> {
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
    ) -> Result<(ModelResponse, Usage), ProviderError> {
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

    pub fn parse_count_tokens_response(
        &self,
        json: &serde_json::Value,
    ) -> Result<u32, ProviderError> {
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

/// Split system messages out of a message list.
///
/// Returns `(system_text, non_system_messages)` where system_text is the
/// concatenation of all system message texts (joined by newline), or `None`
/// if there are no system messages.
///
/// Panics if a system message contains non-text content — system messages
/// must always be text.
pub(crate) fn split_system_messages(messages: &[Message]) -> (Option<String>, Vec<&Message>) {
    let mut system_text = String::new();
    let mut rest = Vec::new();

    for m in messages {
        if let Message::Content { role, content } = m
            && role == "system"
        {
            let Content::Text(text) = content else {
                panic!("system message must be text, got blob");
            };
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(text);
        } else {
            rest.push(m);
        }
    }

    let system = if system_text.is_empty() {
        None
    } else {
        Some(system_text)
    };
    (system, rest)
}
