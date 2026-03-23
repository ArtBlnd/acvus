use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rust_decimal::Decimal;

use crate::dsl::MessageSpec;

use super::{MaxTokens, ThinkingConfig, ToolBinding};

/// Google AI (Gemini) LLM node spec.
#[derive(Debug, Clone)]
pub struct GoogleAISpec {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<MessageSpec>,
    pub tools: Vec<ToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub top_k: Option<u32>,
    pub max_tokens: MaxTokens,
    pub thinking: Option<ThinkingConfig>,
    pub grounding: bool,
    pub cache_key: Option<String>,
}

impl GoogleAISpec {
    pub fn output_ty(&self, interner: &Interner) -> Ty {
        Ty::List(Box::new(crate::dsl::message_elem_ty(interner)))
    }
}
