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
    /// An integer literal's value is not representable in the type its use
    /// gave it (RFC-0037).
    IntegerLiteralOutOfRange {
        value: i128,
        ty: Ty,
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
    NoOperatorInstance {
        op: &'static str,
        ty: Ty,
    },
    /// A call of a shared signature that no instance matches (RFC-0027).
    NoInstance {
        ty: Ty,
    },
    StoreThroughSharedReference(Ty),
    /// `&mut r` where `r: &T` (RFC-0029).
    MutableBorrowOfShared,
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
    ExternParamAssign(String),
    SourceNotIterable {
        actual: Ty,
    },

    /// A context's type is not data (RFC-0014).
    ContextNotData {
        name: String,
        ty: Ty,
    },

    // Lowering errors
    ArityMismatch {
        func: String,
        expected: usize,
        got: usize,
    },
    /// `&` or `&mut` on something that is not a place.
    NotAPlace,
    /// `*r` where `r` is not a reference.
    DerefOfNonReference(Ty),
    /// `*r` where the reference names a value that is not a primitive.
    DerefOfNonPrimitive(Ty),
    /// A lambda captured a reference.
    ReferenceCaptured,
    /// A reference inside a list, object, or tuple.
    ReferenceInData,
    /// A lambda returned a reference.
    ReferenceReturned,
    /// A bare name that two namespaces both declare.
    AmbiguousFunction {
        name: String,
        candidates: Vec<String>,
    },
    /// One call names the same place twice.
    PlaceNamedTwice(String),

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
                write!(f, "effect {} exceeds the allowed {}", c.required, c.allowed)
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
            MirErrorKind::NotAPlace => {
                write!(
                    f,
                    "only a variable, a context, or a field of one can be referenced"
                )
            }
            MirErrorKind::DerefOfNonReference(ty) => {
                write!(f, "`*` needs a reference, got {}", ty.display(interner))
            }
            MirErrorKind::DerefOfNonPrimitive(ty) => {
                write!(
                    f,
                    "`*` reads only a primitive; {} is used through the reference or cloned",
                    ty.display(interner)
                )
            }
            MirErrorKind::ReferenceCaptured => {
                write!(f, "a lambda cannot capture a reference")
            }
            MirErrorKind::ReferenceInData => {
                write!(
                    f,
                    "a reference cannot be stored in a list, object, or tuple"
                )
            }
            MirErrorKind::ReferenceReturned => {
                write!(f, "a lambda cannot return a reference")
            }
            MirErrorKind::AmbiguousFunction { name, candidates } => {
                write!(f, "`{name}` is declared by {}", candidates.join(" and "))
            }
            MirErrorKind::PlaceNamedTwice(place) => {
                write!(f, "`{place}` is named twice in one call")
            }
            MirErrorKind::UndefinedFunction(name) => {
                write!(f, "undefined function `{name}`")
            }
            MirErrorKind::StoreThroughSharedReference(ty) => {
                write!(
                    f,
                    "cannot store through {}: not a `&mut`",
                    ty.display(interner)
                )
            }
            MirErrorKind::MutableBorrowOfShared => {
                write!(f, "a shared reference cannot be borrowed mutably")
            }
            MirErrorKind::NoInstance { ty } => {
                write!(
                    f,
                    "no instance of the signature has the call type {}",
                    ty.display(interner)
                )
            }
            MirErrorKind::NoOperatorInstance { op, ty } => {
                write!(
                    f,
                    "`{op}` has no instance of core::eq for {}",
                    ty.display(interner)
                )
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
                    crate::ty::TyVarBound::Integer { signed, among } => {
                        if among.len() == crate::ty::IntTy::ALL.len() {
                            write!(f, "an integer")
                        } else if *signed && among.iter().all(|k| k.signed()) && among.len() == 4 {
                            write!(f, "a signed integer")
                        } else {
                            let names: Vec<&str> = among.iter().map(|k| k.name()).collect();
                            write!(f, "one of {}", names.join(", "))
                        }
                    }
                }
            }
            MirErrorKind::IntegerLiteralOutOfRange { value, ty } => {
                write!(f, "literal {value} does not fit {}", ty.display(interner))
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
            MirErrorKind::ExternParamAssign(name) => {
                write!(
                    f,
                    "extern param `${name}` is immutable and cannot be assigned"
                )
            }
            MirErrorKind::ContextNotData { name, ty } => {
                write!(
                    f,
                    "`@{name}` has type {}, which is not data; a context holds only what a host keeps between runs",
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
