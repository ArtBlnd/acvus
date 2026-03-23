use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rust_decimal::Decimal;

use crate::dsl::MessageSpec;

use super::{MaxTokens, ToolBinding};

/// OpenAI-compatible LLM node spec.
#[derive(Debug, Clone)]
pub struct OpenAICompatibleSpec {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<MessageSpec>,
    pub tools: Vec<ToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub max_tokens: MaxTokens,
    pub cache_key: Option<String>,
}

impl OpenAICompatibleSpec {
    pub fn output_ty(&self, interner: &Interner) -> Ty {
        Ty::List(Box::new(crate::dsl::message_elem_ty(interner)))
    }
}
