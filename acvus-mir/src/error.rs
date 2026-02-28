use std::fmt;

use acvus_ast::Span;

use crate::ty::Ty;

#[derive(Debug, Clone)]
pub struct MirError {
    pub kind: MirErrorKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum MirErrorKind {
    // Type errors
    TypeMismatchBinOp {
        op: &'static str,
        left: Ty,
        right: Ty,
    },
    EmitNotString {
        actual: Ty,
    },
    RangeBoundsNotInt {
        actual: Ty,
    },
    HeterogeneousList {
        expected: Ty,
        got: Ty,
    },
    AmbiguousEmptyList,
    UnificationFailure {
        expected: Ty,
        got: Ty,
    },

    // Name errors
    UndefinedVariable(String),
    UndefinedFunction(String),
    UndefinedField {
        object_ty: Ty,
        field: String,
    },
    UndefinedStorage(String),

    // Pattern errors
    MissingCatchAll,
    PatternTypeMismatch {
        pattern_ty: Ty,
        source_ty: Ty,
    },
    StorageRefNotDerived(String),

    // Lowering errors
    ArityMismatch {
        func: String,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for MirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            MirErrorKind::TypeMismatchBinOp { op, left, right } => {
                write!(f, "type mismatch in `{op}`: {left} vs {right}")
            }
            MirErrorKind::EmitNotString { actual } => {
                write!(f, "emit requires String, got {actual}")
            }
            MirErrorKind::RangeBoundsNotInt { actual } => {
                write!(f, "range bounds must be Int, got {actual}")
            }
            MirErrorKind::HeterogeneousList { expected, got } => {
                write!(f, "heterogeneous list: expected {expected}, got {got}")
            }
            MirErrorKind::AmbiguousEmptyList => {
                write!(f, "cannot infer type of empty list `[]` without context")
            }
            MirErrorKind::UnificationFailure { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            MirErrorKind::UndefinedVariable(name) => {
                write!(f, "undefined variable `{name}`")
            }
            MirErrorKind::UndefinedFunction(name) => {
                write!(f, "undefined function `{name}`")
            }
            MirErrorKind::UndefinedField { object_ty, field } => {
                write!(f, "no field `{field}` on type {object_ty}")
            }
            MirErrorKind::UndefinedStorage(name) => {
                write!(f, "undefined storage `${name}`")
            }
            MirErrorKind::MissingCatchAll => {
                write!(f, "match block must have a catch-all `{{{{_}}}}` arm")
            }
            MirErrorKind::PatternTypeMismatch {
                pattern_ty,
                source_ty,
            } => {
                write!(
                    f,
                    "pattern type {pattern_ty} incompatible with source type {source_ty}"
                )
            }
            MirErrorKind::StorageRefNotDerived(name) => {
                write!(f, "storage ref `${name}` must be derived from an existing storage variable")
            }
            MirErrorKind::ArityMismatch {
                func,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function `{func}` expects {expected} arguments, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for MirError {}
