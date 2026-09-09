use std::fmt;

use acvus_ast::Span;
use acvus_utils::Interner;

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
    HeterogeneousList {
        expected: Ty,
        got: Ty,
    },
    AmbiguousType {
        resolved_ty: Ty,
    },
    UnificationFailure {
        expected: Ty,
        got: Ty,
    },
    /// A declared type variable resolved to a type outside its bound.
    TypeOutOfBound {
        ty: Ty,
        bound: crate::ty::TyVarBound,
    },
    EffectExceeded(crate::ty::EffectConflict),
    ArrayLengthMismatch {
        pattern_min: usize,
        exact: bool,
        got: usize,
    },
    ArrayLengthUnknown,
    RestInArrayLiteral,

    // Name errors
    UndefinedVariable(String),
    UndefinedFunction(String),
    UndefinedField {
        object_ty: Ty,
        field: String,
    },
    UndefinedContext(String),

    // Pattern errors
    MissingCatchAll,
    PatternTypeMismatch {
        pattern_ty: Ty,
        source_ty: Ty,
    },
    ContextWriteAttempt(String),
    ExternParamAssign(String),
    SourceNotIterable {
        actual: Ty,
    },

    // Value errors
    NonPureContextLoad {
        name: String,
        ty: Ty,
    },

    // Lowering errors
    ArityMismatch {
        func: String,
        expected: usize,
        got: usize,
    },

    // Validation errors (from MIR pass type checking)
    ValidationCheck {
        scope: String,
        inst_index: usize,
        message: String,
    },

    // Graph engine errors
    ParseError(String),
}

impl MirError {
    pub fn display<'a>(&'a self, interner: &'a Interner) -> MirErrorDisplay<'a> {
        MirErrorDisplay {
            error: self,
            interner,
        }
    }
}

pub struct MirErrorDisplay<'a> {
    error: &'a MirError,
    interner: &'a Interner,
}

impl<'a> fmt::Display for MirErrorDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let interner = self.interner;
        match &self.error.kind {
            MirErrorKind::TypeMismatchBinOp { op, left, right } => {
                write!(
                    f,
                    "type mismatch in `{op}`: {} vs {}",
                    left.display(interner),
                    right.display(interner)
                )
            }
            MirErrorKind::EmitNotString { actual } => {
                write!(f, "emit requires String, got {}", actual.display(interner))
            }
            MirErrorKind::HeterogeneousList { expected, got } => {
                write!(
                    f,
                    "heterogeneous list: expected {}, got {}",
                    expected.display(interner),
                    got.display(interner)
                )
            }
            MirErrorKind::AmbiguousType { resolved_ty } => {
                write!(
                    f,
                    "cannot infer type: resolved to {} which contains unresolved type variables",
                    resolved_ty.display(interner)
                )
            }
            MirErrorKind::UnificationFailure { expected, got } => {
                write!(
                    f,
                    "type mismatch: expected {}, got {}",
                    expected.display(interner),
                    got.display(interner)
                )
            }
            MirErrorKind::EffectExceeded(c) => {
                write!(
                    f,
                    "effect {:?} exceeds the allowed {:?}",
                    c.required, c.allowed
                )
            }
            MirErrorKind::ArrayLengthMismatch {
                pattern_min,
                exact,
                got,
            } => {
                if *exact {
                    write!(f, "array pattern needs length {pattern_min}, got {got}")
                } else {
                    write!(
                        f,
                        "array pattern needs length at least {pattern_min}, got {got}"
                    )
                }
            }
            MirErrorKind::ArrayLengthUnknown => write!(f, "array length is not known here"),
            MirErrorKind::RestInArrayLiteral => {
                write!(f, "`..` is a pattern, not an array element")
            }
            MirErrorKind::UndefinedVariable(name) => {
                write!(f, "undefined variable `{name}`")
            }
            MirErrorKind::UndefinedFunction(name) => {
                write!(f, "undefined function `{name}`")
            }
            MirErrorKind::TypeOutOfBound { ty, bound } => {
                write!(
                    f,
                    "type {} is outside the declared bound ",
                    ty.display(interner)
                )?;
                match bound {
                    crate::ty::TyVarBound::Any => write!(f, "(any)"),
                    crate::ty::TyVarBound::OneOf(tys) => {
                        write!(f, "one of ")?;
                        for (i, t) in tys.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}", t.display(interner))?;
                        }
                        Ok(())
                    }
                }
            }
            MirErrorKind::UndefinedField { object_ty, field } => {
                write!(
                    f,
                    "no field `{field}` on type {}",
                    object_ty.display(interner)
                )
            }
            MirErrorKind::UndefinedContext(name) => {
                write!(f, "undefined context `@{name}`")
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
                    "pattern type {} incompatible with source type {}",
                    pattern_ty.display(interner),
                    source_ty.display(interner)
                )
            }
            MirErrorKind::ContextWriteAttempt(name) => {
                write!(f, "context `@{name}` is read-only and cannot be assigned")
            }
            MirErrorKind::ExternParamAssign(name) => {
                write!(
                    f,
                    "extern param `${name}` is immutable and cannot be assigned"
                )
            }
            MirErrorKind::NonPureContextLoad { name, ty } => {
                write!(
                    f,
                    "`@{name}` has non-pure type {} and cannot be used as a value; it can only be called directly",
                    ty.display(interner)
                )
            }
            MirErrorKind::SourceNotIterable { actual } => {
                write!(
                    f,
                    "source type `{}` is not iterable",
                    actual.display(interner)
                )
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
            MirErrorKind::ValidationCheck {
                scope,
                inst_index,
                message,
            } => {
                write!(f, "[{scope}] inst #{inst_index}: {message}")
            }
            MirErrorKind::ParseError(msg) => {
                write!(f, "parse error: {msg}")
            }
        }
    }
}
