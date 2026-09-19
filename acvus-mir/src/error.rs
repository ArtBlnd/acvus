use std::fmt;

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_utils::Interner;

use crate::graph::QualifiedRef;
use crate::ty::Ty;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShownValue {
    Named(String),
    Anonymous,
}

impl fmt::Display for ShownValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShownValue::Named(name) => write!(f, "`{name}`"),
            ShownValue::Anonymous => write!(f, "this value"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataShape {
    Aggregate,
    Payload,
}

impl fmt::Display for DataShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DataShape::Aggregate => "a list, object, or tuple",
            DataShape::Payload => "an Option or a Result",
        })
    }
}

#[derive(Debug, Clone)]
pub struct MirError {
    pub kind: MirErrorKind,
    pub span: Span,
    pub labels: Vec<Label>,
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
    /// This refusal carries no type, and that is a decision rather than
    /// an omission: the two types are the same as the program writes
    /// them, so printing both would show a reader two identical lines and
    /// say nothing about the difference. The labels say where each source
    /// began instead.
    OneTypeTwoSources {
        left: ShownValue,
        right: ShownValue,
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
    /// More than one declared conversion takes the value to the type
    /// (RFC-0023).
    AmbiguousConversion {
        from: Ty,
        to: Ty,
        rules: Vec<QualifiedRef>,
    },
    /// The conversion rewrites the place a reference names, and the
    /// argument is a reference value with no place behind it.
    ConversionNeedsPlace {
        from: Ty,
        to: Ty,
    },
    /// `?` where nothing returns: a template body (RFC-0038).
    TryOutsideFunction,
    /// `?` on a value that is neither a `Result` nor an `Option`.
    TryOnNonResult(Ty),
    /// `?` would leave with a type the function does not return.
    TryReturnMismatch {
        leaves: Ty,
        returns: Ty,
    },
    EffectExceeded(crate::ty::EffectConflict),
    TaskTooHigh {
        required: crate::ty::Task,
        found: crate::ty::Task,
    },
    ArrayLengthMismatch {
        pattern_min: usize,
        exact: bool,
        got: usize,
    },
    ArrayLengthUnknown,
    RestInArrayLiteral,

    // Name errors
    UndefinedVariable(String),
    /// `x = e;` where no `x` is bound in this body (RFC-0045).
    AssignToUnbound(String),
    /// `x = e;` inside a lambda, where `x` is bound outside it. A capture is
    /// by value, so the store would write the lambda's copy (RFC-0045).
    AssignToCapture(String),
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
    /// An object at a parameter of a declared struct's type lacks a field
    /// the struct declares (RFC-0042).
    ObjectLacksDeclaredField {
        declared: String,
        field: String,
    },
    /// An object at a parameter of a declared struct's type has a field the
    /// struct does not declare (RFC-0042).
    ObjectFieldNotDeclared {
        declared: String,
        field: String,
    },
    /// An object at a projection parameter lacks a field the projection
    /// borrows (RFC-0050 rule 6).
    ProjectionLacksField {
        object: String,
        field: String,
    },
    UndefinedContext(String),
    /// A field the body reads on a path that never stored it
    /// (`validate::init_check`, RFC-0042).
    FieldNotStored {
        subject: ShownValue,
        fields: Vec<String>,
    },

    // Pattern errors
    MissingCatchAll,
    /// A `match` whose arms are not one dispatch over a tag -- a literal,
    /// a tuple, a nested refutable payload -- has no shape `validate` can
    /// decide exhaustiveness on, so it must say so itself (RFC-0051 §3).
    MatchIsNotADispatch,
    /// A `match` arm names a variant the scrutinee cannot hold (RFC-0051
    /// §2). The arms contribute no variant: the scrutinee's type is its
    /// own, so an arm outside it can never be taken.
    UnreachablePattern {
        pattern: String,
        scrutinee_ty: Ty,
    },
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
    /// `a[i]` where `a`'s type has no `as_slice` instance (RFC-0047 §2).
    CannotIndex {
        ty: Ty,
    },
    /// `a[i]` read as a value where the element type moves (RFC-0047 §5);
    /// `clone(&a[i])` is the way (RFC-0028).
    MoveOutOfIndex {
        ty: Ty,
    },
    /// A `for` head that is none of the four the language has (RFC-0057
    /// Decision 1).
    ForSourceNotAdmitted {
        ty: Ty,
    },
    /// `for x in v` where `v` is a container held by value: a loop borrows
    /// its container and does not consume it, the one exception being an
    /// `Array` (RFC-0057 Decision 1).
    ForConsumesContainer,
    /// `break` or `continue` outside a loop (RFC-0057 Decision 4).
    OutsideLoop {
        keyword: &'static str,
    },
    /// A `for` over an `Array` whose element owns something, left early:
    /// the elements the loop did not take have no release, because how many
    /// were taken is a run-time number and the array's own release does not
    /// know it (RFC-0057 Decision 2).
    ArrayLoopLeftEarly {
        keyword: &'static str,
        element: Ty,
    },
    /// `e as T` where `T` is not one of the types `as` converts between
    /// (RFC-0049).
    CastToUnknownType(String),
    /// `e as T` where `e` is neither a number nor a `char`.
    CastOfWhatDoesNotCast(Ty),
    /// `e as T` where both types cast but Rust does not join this pair:
    /// only `u8 as char` reaches a `char`, and a `char` reaches an
    /// integer and not `f64` (RFC-0058).
    CastNotAdmitted {
        from: Ty,
        to: Ty,
    },
    /// `*r` where `r` is not a reference.
    DerefOfNonReference(Ty),
    /// `*r` where the reference names a value that is not a primitive.
    DerefOfNonPrimitive(Ty),
    /// A lambda captured a name the enclosing lambda captured, and the
    /// owned type is not a word (RFC-0018).
    MoveOutOfCapture {
        name: String,
        ty: Ty,
    },
    /// A lambda captured a reference.
    ReferenceCaptured,
    ReferenceInData(DataShape),
    ViewInData(DataShape),
    /// A lambda returned a reference.
    ReferenceReturned,
    /// A script body returned a reference.
    ReferenceReturnedFromBody(Ty),
    /// RFC-0043.
    AmbiguousFunction {
        name: String,
        candidates: Vec<String>,
    },
    /// RFC-0043.
    NoMatchingFunction {
        name: String,
        ty: Ty,
    },
    /// One call names the same place twice.
    PlaceNamedTwice(String),

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
                    "type mismatch in `{op}`: {} vs {}{}",
                    left.display(interner),
                    right.display(interner),
                    match (holds_text(left), holds_text(right)) {
                        (true, true) => COPY_OF_A_VIEW,
                        _ => "",
                    }
                )
            }
            MirErrorKind::EmitNotString { actual } => {
                write!(
                    f,
                    "emit requires String, got {}{}",
                    actual.display(interner),
                    match is_text_view(actual) {
                        true => COPY_OF_A_VIEW,
                        false => "",
                    }
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
            MirErrorKind::OneTypeTwoSources { left, right } => {
                write!(
                    f,
                    "{left} and {right} are values of one type from two different sources, \
                     and one place cannot hold both"
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
                    "type mismatch: expected {}, got {}{}",
                    expected.display(interner),
                    got.display(interner),
                    copy_of_a_view(expected, got)
                )
            }
            MirErrorKind::EffectExceeded(c) => {
                write!(f, "effect {} exceeds the allowed {}", c.required, c.allowed)
            }
            MirErrorKind::TaskTooHigh { required, found } => {
                write!(
                    f,
                    "a function whose task is {found} where {required} is required"
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
            MirErrorKind::AssignToUnbound(name) => {
                write!(
                    f,
                    "cannot assign to `{name}`: no binding named `{name}` is in scope; `let {name} = ...;` binds it"
                )
            }
            MirErrorKind::AssignToCapture(name) => {
                write!(
                    f,
                    "cannot assign to `{name}`: it is captured by the lambda, not bound in it"
                )
            }
            MirErrorKind::NotAPlace => {
                write!(
                    f,
                    "only a variable, a context, or a field of one can be referenced"
                )
            }
            MirErrorKind::CannotIndex { ty } => {
                write!(
                    f,
                    "cannot index into a value of type `{}`",
                    ty.display(interner)
                )
            }
            MirErrorKind::MoveOutOfIndex { ty } => {
                write!(f, "cannot move out of index of `{}`", ty.display(interner))
            }
            MirErrorKind::ForSourceNotAdmitted { ty } => {
                write!(
                    f,
                    "a `for` traverses `&v`, `&mut v`, an array by value, or `lo..hi`; `{}` is none of them",
                    ty.display(interner)
                )
            }
            MirErrorKind::ForConsumesContainer => {
                write!(
                    f,
                    "a container is not consumed by a loop; write `&v` or `&mut v`"
                )
            }
            MirErrorKind::OutsideLoop { keyword } => {
                write!(f, "`{keyword}` is only inside a loop")
            }
            MirErrorKind::ArrayLoopLeftEarly { keyword, element } => {
                write!(
                    f,
                    "a `for` over an array of `{}` cannot `{keyword}`: the elements the loop has not taken would have no release",
                    element.display(interner)
                )
            }
            MirErrorKind::CastToUnknownType(name) => {
                write!(
                    f,
                    "`as` converts to {}, not to `{name}`",
                    crate::ty::CastTy::NAMES
                )
            }
            MirErrorKind::CastOfWhatDoesNotCast(ty) => {
                write!(
                    f,
                    "`as` converts a number or a char; {} is neither",
                    ty.display(interner)
                )
            }
            MirErrorKind::CastNotAdmitted { from, to } => match to {
                Ty::Char => write!(
                    f,
                    "only `u8 as char` reaches a char; {} does not",
                    from.display(interner)
                ),
                _ => write!(
                    f,
                    "`char as` reaches an integer; {} is not one",
                    to.display(interner)
                ),
            },
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
            MirErrorKind::MoveOutOfCapture { name, ty } => {
                write!(
                    f,
                    "cannot move `{name}` out of a closure's capture (type {}); \
                     act through the reference, or clone it",
                    ty.display(interner)
                )
            }
            MirErrorKind::ReferenceCaptured => {
                write!(f, "a lambda cannot capture a reference")
            }
            MirErrorKind::ReferenceInData(shape) => {
                write!(f, "a reference cannot be stored in {shape}")
            }
            MirErrorKind::ViewInData(shape) => {
                write!(
                    f,
                    "a reference cannot be stored in {shape}; \
                     write `.to_string()` to store the text"
                )
            }
            MirErrorKind::ReferenceReturnedFromBody(ty) => {
                write!(
                    f,
                    "a body does not return a reference{}",
                    match is_text_view(ty) {
                        true => COPY_OF_A_VIEW,
                        false => "",
                    }
                )
            }
            MirErrorKind::ReferenceReturned => {
                write!(f, "a lambda cannot return a reference")
            }
            MirErrorKind::AmbiguousFunction { name, candidates } => {
                write!(f, "`{name}` is declared by {}", candidates.join(" and "))
            }
            MirErrorKind::NoMatchingFunction { name, ty } => {
                write!(
                    f,
                    "no `{name}` takes a call of type {}",
                    ty.display(interner)
                )
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
            MirErrorKind::AmbiguousConversion { from, to, rules } => {
                write!(
                    f,
                    "more than one conversion takes {} to {}: ",
                    from.display(interner),
                    to.display(interner)
                )?;
                for (i, rule) in rules.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", interner.resolve(rule.name))?;
                }
                Ok(())
            }
            MirErrorKind::ConversionNeedsPlace { from, to } => {
                write!(
                    f,
                    "converting {} to {} rewrites the place the reference names, and this argument is a reference value, not a borrow of a place",
                    from.display(interner),
                    to.display(interner)
                )
            }
            MirErrorKind::TryOutsideFunction => {
                write!(
                    f,
                    "`?` needs a function to return from; a template has none"
                )
            }
            MirErrorKind::TryOnNonResult(ty) => write!(
                f,
                "`?` takes a Result or an Option, not {}",
                ty.display(interner)
            ),
            MirErrorKind::TryReturnMismatch { leaves, returns } => write!(
                f,
                "`?` leaves with {} but the function returns {}",
                leaves.display(interner),
                returns.display(interner)
            ),
            MirErrorKind::UndefinedField { object_ty, field } => {
                write!(
                    f,
                    "no field `{field}` on type {}",
                    object_ty.display(interner)
                )
            }
            MirErrorKind::ObjectLacksDeclaredField { declared, field } => {
                write!(f, "object lacks field `{field}` that `{declared}` declares")
            }
            MirErrorKind::ObjectFieldNotDeclared { declared, field } => {
                write!(
                    f,
                    "object has field `{field}` that `{declared}` does not declare"
                )
            }
            MirErrorKind::ProjectionLacksField { object, field } => {
                write!(
                    f,
                    "`{object}` lacks field `{field}`, which the projection parameter borrows"
                )
            }
            MirErrorKind::UndefinedContext(name) => {
                write!(f, "`@{name}` is not a declared context")
            }
            MirErrorKind::FieldNotStored { subject, fields } => {
                let named: Vec<String> = fields.iter().map(|f| format!("`{f}`")).collect();
                write!(
                    f,
                    "{subject} has no {} stored on every path that reaches here",
                    named.join(", ")
                )
            }
            MirErrorKind::MatchIsNotADispatch => {
                write!(
                    f,
                    "non-exhaustive match: these arms are not one dispatch over a tag, so the variants they cover are not known; add a `_` arm"
                )
            }
            MirErrorKind::UnreachablePattern {
                pattern,
                scrutinee_ty,
            } => {
                write!(
                    f,
                    "unreachable pattern: `{pattern}` is not a variant of `{}`",
                    scrutinee_ty.display(interner)
                )
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
            MirErrorKind::ParseError(msg) => {
                write!(f, "parse error: {msg}")
            }
        }
    }
}

/// The spelling that turns a view into the owned text (RFC-0062 Decision 3).
const COPY_OF_A_VIEW: &str = "; write `.to_string()` for the owned text";

/// `str`, or a reference to it: what a string literal is.
fn is_text_view(ty: &Ty) -> bool {
    match ty {
        Ty::Str => true,
        Ty::Ref(_, inner) => is_text_view(&inner.ty),
        _ => false,
    }
}

/// One of the two representations of text, or a reference to one. A
/// mismatch between two of these is the one `.to_string()` settles; a
/// mismatch between text and a number is not.
fn holds_text(ty: &Ty) -> bool {
    match ty {
        Ty::String | Ty::Str => true,
        Ty::Ref(_, inner) => holds_text(&inner.ty),
        _ => false,
    }
}

fn copy_of_a_view(expected: &Ty, got: &Ty) -> &'static str {
    match (expected, is_text_view(got)) {
        (Ty::String, true) => COPY_OF_A_VIEW,
        _ => "",
    }
}
