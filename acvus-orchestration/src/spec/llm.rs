use rust_decimal::Decimal;

/// An LLM spec — provider-specific configuration and prompt.
pub struct LlmSpec {
    pub name: String,
    pub provider: Provider,
}

/// LLM provider with its full configuration and prompt structure.
pub enum Provider {
    OpenAI(OpenAISpec),
    Anthropic(AnthropicSpec),
    Google(GoogleSpec),
}

pub struct OpenAISpec {}

pub struct AnthropicSpec {}

pub struct GoogleSpec {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub system: Option<Content>,
    pub messages: Vec<GoogleMessage>,
}

pub struct GoogleMessage {
    pub role: GoogleRole,
    pub content: Content,
}

pub enum GoogleRole {
    User,
    Model,
}

/// Content source for a prompt message.
pub enum Content {
    /// Literal string.
    Inline(String),
    /// Reference to a Block by name.
    Ref(String),
    /// Acvus expression yielding Iterator<T> — flattens into messages.
    Iterator(String),
}
