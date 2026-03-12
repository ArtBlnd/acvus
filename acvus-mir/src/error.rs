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
    RangeBoundsNotInt {
        actual: Ty,
    },
    HeterogeneousList {
        expected: Ty,
        got: Ty,
    },
    AmbiguousType,
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
    UndefinedContext(String),

    // Pattern errors
    MissingCatchAll,
    PatternTypeMismatch {
        pattern_ty: Ty,
        source_ty: Ty,
    },
    ContextWriteAttempt(String),
    SourceNotIterable {
        actual: Ty,
    },

    // Builtin constraint errors
    BuiltinConstraint(String),

    // Overload resolution errors
    NoMatchingOverload {
        name: String,
        arg_tys: Vec<Ty>,
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
            MirErrorKind::RangeBoundsNotInt { actual } => {
                write!(
                    f,
                    "range bounds must be Int, got {}",
                    actual.display(interner)
                )
            }
            MirErrorKind::HeterogeneousList { expected, got } => {
                write!(
                    f,
                    "heterogeneous list: expected {}, got {}",
                    expected.display(interner),
                    got.display(interner)
                )
            }
            MirErrorKind::AmbiguousType => {
                write!(f, "cannot infer type: not enough context to determine the type of this expression")
            }
            MirErrorKind::UnificationFailure { expected, got } => {
                write!(
                    f,
                    "type mismatch: expected {}, got {}",
                    expected.display(interner),
                    got.display(interner)
                )
            }
            MirErrorKind::UndefinedVariable(name) => {
                write!(f, "undefined variable `{name}`")
            }
            MirErrorKind::UndefinedFunction(name) => {
                write!(f, "undefined function `{name}`")
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
                    "source type `{}` is not iterable (expected List or Range)",
                    actual.display(interner)
                )
            }
            MirErrorKind::BuiltinConstraint(msg) => {
                write!(f, "{msg}")
            }
            MirErrorKind::NoMatchingOverload { name, arg_tys } => {
                write!(f, "no matching overload for `{name}` with arguments (")?;
                for (i, ty) in arg_tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty.display(interner))?;
                }
                write!(f, ")")
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
