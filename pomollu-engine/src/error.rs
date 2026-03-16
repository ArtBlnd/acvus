use serde::Serialize;
use tsify::Tsify;

use acvus_orchestration::{OrchError, OrchErrorKind};
use acvus_utils::Interner;

#[derive(Debug, Clone, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct EngineError {
    pub category: ErrorCategory,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<ErrorSpan>,
}

#[derive(Debug, Clone, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ErrorSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub enum ErrorCategory {
    Parse,
    Type,
    Runtime,
}

impl EngineError {
    pub fn from_parse(e: &acvus_ast::ParseError) -> Self {
        Self {
            category: ErrorCategory::Parse,
            message: e.kind.to_string(),
            span: Some(ErrorSpan {
                start: e.span.start,
                end: e.span.end,
            }),
        }
    }

    pub fn from_mir(e: &acvus_mir::error::MirError, interner: &Interner) -> Self {
        Self {
            category: ErrorCategory::Type,
            message: e.display(interner).to_string(),
            span: Some(ErrorSpan {
                start: e.span.start,
                end: e.span.end,
            }),
        }
    }

    pub fn from_runtime(e: &acvus_interpreter::RuntimeError) -> Self {
        Self {
            category: ErrorCategory::Runtime,
            message: e.to_string(),
            span: None,
        }
    }

    pub fn from_mir_errors(errs: &[acvus_mir::error::MirError], interner: &Interner) -> Vec<Self> {
        errs.iter().map(|e| Self::from_mir(e, interner)).collect()
    }

    pub fn general(category: ErrorCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
            span: None,
        }
    }

    /// Flatten an OrchError into one or more EngineErrors.
    /// OrchErrors that contain nested MirErrors are expanded.
    pub fn from_orch(e: &OrchError, interner: &Interner) -> Vec<Self> {
        match &e.kind {
            OrchErrorKind::TemplateParse { .. } | OrchErrorKind::ScriptParse { .. } => {
                vec![Self::general(ErrorCategory::Parse, e.display(interner).to_string())]
            }
            OrchErrorKind::TemplateCompile { errors, .. }
            | OrchErrorKind::ScriptCompile { errors, .. } => {
                Self::from_mir_errors(errors, interner)
            }
            OrchErrorKind::FuelExhausted
            | OrchErrorKind::ModelError(_)
            | OrchErrorKind::ToolNotFound(_) => {
                vec![Self::general(ErrorCategory::Runtime, e.display(interner).to_string())]
            }
            _ => {
                // Config/DAG/structural errors → Type category
                vec![Self::general(ErrorCategory::Type, e.display(interner).to_string())]
            }
        }
    }

    pub fn from_orch_errors(errs: &[OrchError], interner: &Interner) -> Vec<Self> {
        errs.iter().flat_map(|e| Self::from_orch(e, interner)).collect()
    }

    pub fn from_lsp(category: ErrorCategory, message: &str, span: Option<(usize, usize)>) -> Self {
        Self {
            category,
            message: message.to_string(),
            span: span.map(|(start, end)| ErrorSpan { start, end }),
        }
    }
}
