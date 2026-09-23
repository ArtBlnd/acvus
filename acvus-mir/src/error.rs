use std::fmt;

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_utils::Interner;

use crate::graph::QualifiedRef;
use crate::ir::SwitchKey;
use crate::ty::Ty;

/// A name as a script writes it: `ns::name`, or the bare name where the
/// declaration has no namespace.
fn qualified(interner: &Interner, qref: QualifiedRef) -> String {
    match qref.namespace {
        Some(ns) => format!("{}::{}", interner.resolve(ns), interner.resolve(qref.name)),
        None => interner.resolve(qref.name).to_string(),
    }
}

/// Each spelling once, in the order first met: two instances that differ
/// only in what a shown type leaves out, an identity or a requirement, are
/// one line to a reader.
fn spelled_once<I>(spellings: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut once: Vec<String> = Vec::new();
    for spelling in spellings {
        if !once.contains(&spelling) {
            once.push(spelling);
        }
    }
    once
}

/// The shapes among `shapes` no other shape there covers: a specialized
/// instance is of its generic instance's shape, and one of two equal shapes
/// stays.
fn most_general(shapes: &[crate::ty::PolyTy]) -> Vec<&crate::ty::PolyTy> {
    shapes
        .iter()
        .enumerate()
        .filter(|(i, shape)| {
            !shapes.iter().enumerate().any(|(j, other)| {
                j != *i
                    && crate::ty::subsumes(other, shape)
                    && (!crate::ty::subsumes(shape, other) || j < *i)
            })
        })
        .map(|(_, shape)| shape)
        .collect()
}

/// How many alternatives a listing shows before it says how many more
/// there are: past this many, a reader is looking at a catalogue, not at
/// the one they meant.
const LISTED: usize = 12;

/// The alternatives a refusal enumerates, one per line under the sentence
/// that introduces them: the instances a call could reach, the shapes a
/// bound admits, the declarations sharing a name. The first `LISTED` are
/// shown and the rest counted.
fn listed<I>(f: &mut fmt::Formatter<'_>, items: I) -> fmt::Result
where
    I: IntoIterator,
    I::Item: fmt::Display,
{
    let items: Vec<I::Item> = items.into_iter().collect();
    for item in items.iter().take(LISTED) {
        write!(f, "\n  {item}")?;
    }
    match items.len().saturating_sub(LISTED) {
        0 => Ok(()),
        1 => write!(f, "\n  and 1 other"),
        more => write!(f, "\n  and {more} others"),
    }
}

/// Whose instance a refused instance decision was looking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceWanted {
    /// The callee's own shared signature (RFC-0019).
    Callee(Option<QualifiedRef>),
    /// RFC-0068 rule 5.
    Requirement {
        signature: QualifiedRef,
        required_by: Option<QualifiedRef>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorSignature {
    /// `==` and `!=`. `acvus_extern::core` declares it as
    /// `eq<T>(a: &T, b: &T) -> bool`.
    Eq,
    /// `<`, `<=`, `>` and `>=`. `acvus_extern::core` declares it as
    /// `cmp<T>(a: &T, b: &T) -> i64`, whose answer is `-1`, `0` or `1`, and
    /// lowering reads the operator off the sign of that answer.
    Cmp,
    /// `+`. `acvus_extern::core` declares it as
    /// `add<T, O>(a: &T, b: &T) -> O`, whose answer is the operator's value.
    Add,
    /// `-`, declared as `add` is.
    Sub,
    /// `*`, declared as `add` is.
    Mul,
    /// `/`, declared as `add` is.
    Div,
    /// `%`, declared as `add` is.
    Rem,
    /// Unary `-`. `acvus_extern::core` declares it as
    /// `neg<T, O>(a: &T) -> O`, whose answer is the operator's value.
    Neg,
}

impl OperatorSignature {
    pub fn operand_count(self) -> usize {
        match self {
            OperatorSignature::Neg => 1,
            OperatorSignature::Eq
            | OperatorSignature::Cmp
            | OperatorSignature::Add
            | OperatorSignature::Sub
            | OperatorSignature::Mul
            | OperatorSignature::Div
            | OperatorSignature::Rem => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            OperatorSignature::Eq => "eq",
            OperatorSignature::Cmp => "cmp",
            OperatorSignature::Add => "add",
            OperatorSignature::Sub => "sub",
            OperatorSignature::Mul => "mul",
            OperatorSignature::Div => "div",
            OperatorSignature::Rem => "rem",
            OperatorSignature::Neg => "neg",
        }
    }
}

impl fmt::Display for OperatorSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "core::{}", self.name())
    }
}

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

/// The names a refusal offers in place of the one it did not find. A
/// candidate is within an edit distance of two of what was written, or has
/// it as a prefix; the nearest three are kept, and an empty list prints
/// nothing, so a refusal with no near name keeps the sentence it had.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DidYouMean(Vec<String>);

impl DidYouMean {
    /// At most this many names, so the sentence stays one line.
    pub const SHOWN: usize = 3;

    /// The names near `wanted` among `known`, nearest first and then
    /// alphabetically, with `wanted` itself never among them.
    pub fn of<I>(wanted: &str, known: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let mut near: Vec<(usize, String)> = known
            .into_iter()
            .filter(|name| name != wanted)
            .filter_map(|name| near(wanted, &name).map(|d| (d, name)))
            .collect();
        near.sort();
        near.dedup_by(|a, b| a.1 == b.1);
        near.truncate(Self::SHOWN);
        Self(near.into_iter().map(|(_, name)| name).collect())
    }
}

impl fmt::Display for DidYouMean {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [first, rest @ ..] = self.0.as_slice() else {
            return Ok(());
        };
        write!(f, "; did you mean `{first}`")?;
        for (i, name) in rest.iter().enumerate() {
            match i + 2 == self.0.len() {
                true => write!(f, " or `{name}`")?,
                false => write!(f, ", `{name}`")?,
            }
        }
        write!(f, "?")
    }
}

/// How far `name` is from `wanted`, where near enough to offer: an edit
/// distance of at most two, or `wanted` written as a prefix of it or as
/// one of its `_`-separated words (`iter` names `as_iter` and `into_iter`,
/// not `filter`). A name of one or two characters is near everything of
/// its length, so only a prefix or a word counts there.
fn near(wanted: &str, name: &str) -> Option<usize> {
    if wanted.len() >= 3 && (name.starts_with(wanted) || name.split('_').any(|word| word == wanted))
    {
        return Some(0);
    }
    if wanted.len() < 3 {
        return None;
    }
    match edit_distance(wanted, name) {
        d @ 0..=2 => Some(d),
        _ => None,
    }
}

/// The Levenshtein distance, over characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let next = match ca == *cb {
                true => diagonal,
                false => 1 + diagonal.min(row[j]).min(row[j + 1]),
            };
            diagonal = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
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
    /// RFC-0037.
    IntegerLiteralWidthUnsettled {
        among: Vec<crate::ty::IntTy>,
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
    /// `return` where nothing returns: a template body.
    ReturnOutsideFunction,
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
    UndefinedVariable {
        name: String,
        near: DidYouMean,
    },
    /// `x = e;` where no `x` is bound in this body (RFC-0045).
    AssignToUnbound(String),
    /// `x = e;` inside a lambda, where `x` is bound outside it. A capture is
    /// by value, so the store would write the lambda's copy (RFC-0045).
    AssignToCapture(String),
    UndefinedFunction {
        name: String,
        near: DidYouMean,
    },
    /// A call of a value that is not a function.
    NotCallable(Ty),
    NoOperatorInstance {
        op: &'static str,
        signature: OperatorSignature,
        ty: Ty,
    },
    /// A comparison operator on a language-owned type whose representation
    /// carries no order. An extension type reaches `core::cmp` instead of
    /// this refusal (RFC-0070 rule 5). Both operands are text often enough that
    /// the refusal names the ordering functions `string::cmp` offers, which
    /// is what a program reaching for `<` on a `String` wants.
    NoOrdering {
        op: &'static str,
        ty: Ty,
    },
    /// A call of a shared signature that no instance matches (RFC-0019):
    /// the call's type, and the instances the call could have reached.
    NoInstance {
        ty: Ty,
        instances: Vec<crate::ty::PolyTy>,
        of: InstanceWanted,
    },
    StoreThroughSharedReference {
        subject: ShownValue,
        ty: Ty,
    },
    /// `&mut r` where `r: &T` (RFC-0029).
    MutableBorrowOfShared {
        subject: ShownValue,
    },
    UndefinedField {
        object_ty: Ty,
        field: String,
        near: DidYouMean,
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
    /// A value whose type was laid out outside the body met a type with a
    /// member it lacks.
    FixedLacks {
        member: String,
    },
    /// An object type with more fields than `ty::ObjectTy::MAX_FIELDS`, whose
    /// positions the machine cannot name.
    ObjectTooWide {
        fields: usize,
        most: usize,
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
        near: DidYouMean,
    },

    // Pattern errors
    MissingCatchAll,
    /// A `match` whose arms are not one dispatch over a tag -- a literal,
    /// a tuple, a nested refutable payload -- has no shape `validate` can
    /// decide exhaustiveness on, so it must say so itself (RFC-0051 rule 3).
    MatchIsNotADispatch,
    /// The later arm can never be taken, and the language has no warning
    /// axis (RFC-0051 rule 2).
    MatchArmKeyRepeated {
        key: SwitchKey,
    },
    /// A `match` arm names a variant the scrutinee cannot hold (RFC-0051
    /// rule 2). The arms contribute no variant: the scrutinee's type is its
    /// own, so an arm outside it can never be taken.
    UnreachablePattern {
        pattern: String,
        scrutinee_ty: Ty,
        near: DidYouMean,
    },
    PatternTypeMismatch {
        pattern_ty: Ty,
        source_ty: Ty,
    },
    ExternParamAssign(String),
    /// One pattern binds a name twice.
    NameBoundTwice(String),
    /// An operator checked where its operand's type was still open, whose
    /// operand settled to a type the operator on words does not take.
    OperatorDecidedBeforeItsOperand {
        op: String,
        ty: Ty,
    },
    /// A `$` input nothing that reads it gives a type, so no value can be
    /// supplied for it.
    InputTypeUndecided(String),
    BindingTypeMismatch {
        name: String,
        value: acvus_ast::Literal,
        ty: Ty,
    },
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
        func: ShownValue,
        expected: usize,
        got: usize,
    },
    /// `a[i]` where `a`'s type has no `as_slice` instance (RFC-0047 rule 3).
    CannotIndex {
        ty: Ty,
    },
    /// `a[i]` read as a value where the element type moves (RFC-0047 rule 5);
    /// `clone(&a[i])` is the way (RFC-0028).
    MoveOutOfIndex {
        ty: Ty,
    },
    /// A `for` head that is none of the four the language has (RFC-0057
    /// rule 1).
    ForSourceNotAdmitted {
        ty: Ty,
    },
    /// `for x in v` where `v` is a container held by value: a loop borrows
    /// its container and does not consume it, the one exception being an
    /// `Array` (RFC-0057 rule 1).
    ForConsumesContainer,
    /// `break` or `continue` outside a loop (RFC-0057 rule 4).
    OutsideLoop {
        keyword: &'static str,
    },
    /// A `for` over an `Array` whose element owns something, left early:
    /// the elements the loop did not take have no release, because how many
    /// were taken is a run-time number and the array's own release does not
    /// know it (RFC-0057 rule 6).
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
    /// A lambda captured a string or slice view.
    ViewCaptured,
    ReferenceInData(DataShape),
    ViewInData(DataShape),
    /// A script body returned a reference.
    ReferenceReturnedFromBody(Ty),
    /// RFC-0069 rule 1: the entry's result reaches the host, which outlives
    /// the run a closure belongs to.
    ClosureReturnedToTheHost(Ty),
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

impl MirErrorKind {
    /// The words the primary marker carries. A refusal whose message names
    /// two places says nothing at either of them, so it gives the marker a
    /// sentence of its own; every other refusal's marker repeats the
    /// message, which is `Report`'s rule with no primary text.
    pub fn primary(&self) -> Option<String> {
        match self {
            MirErrorKind::OneTypeTwoSources { left, right } => {
                Some(format!("{left} and {right} meet here"))
            }
            _ => None,
        }
    }
}

impl MirError {
    pub fn display<'a>(&'a self, interner: &'a Interner) -> MirErrorDisplay<'a> {
        MirErrorDisplay {
            error: self,
            interner,
        }
    }

    pub fn primary(&self) -> Option<String> {
        self.kind.primary()
    }
}

/// Everything one compilation refused, whichever stage raised it: the stages
/// that read the source and its types speak `MirError`, and the validator
/// that reads a lowered body speaks `ValidationError`.
#[derive(Debug, Clone)]
pub enum Refusal {
    Mir(MirError),
    Invalid(crate::validate::ValidationError),
}

impl Refusal {
    pub fn span(&self) -> Span {
        match self {
            Refusal::Mir(error) => error.span,
            Refusal::Invalid(error) => error.span,
        }
    }

    pub fn labels(&self) -> &[Label] {
        match self {
            Refusal::Mir(error) => &error.labels,
            Refusal::Invalid(error) => error.labels(),
        }
    }

    pub fn primary(&self) -> Option<String> {
        match self {
            Refusal::Mir(error) => error.primary(),
            Refusal::Invalid(_) => None,
        }
    }

    pub fn display<'a>(&'a self, interner: &'a Interner) -> RefusalDisplay<'a> {
        RefusalDisplay {
            refusal: self,
            interner,
        }
    }
}

pub struct RefusalDisplay<'a> {
    refusal: &'a Refusal,
    interner: &'a Interner,
}

impl fmt::Display for RefusalDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.refusal {
            Refusal::Mir(error) => error.display(self.interner).fmt(f),
            Refusal::Invalid(error) => error.display(self.interner).fmt(f),
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
                    left.shown(interner),
                    right.shown(interner),
                    match (holds_text(left), holds_text(right)) {
                        (true, true) => COPY_OF_A_VIEW.to_string(),
                        _ => open_payload(left)
                            .or_else(|| open_payload(right))
                            .unwrap_or_default(),
                    }
                )
            }
            MirErrorKind::EmitNotString { actual } => {
                write!(
                    f,
                    "emit requires String, got {}{}",
                    actual.shown(interner),
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
                    expected.shown(interner),
                    got.shown(interner)
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
                    resolved_ty.shown(interner)
                )
            }
            MirErrorKind::UnificationFailure { expected, got } => {
                write!(
                    f,
                    "type mismatch: expected {}, got {}{}{}",
                    expected.shown(interner),
                    got.shown(interner),
                    copy_of_a_view(expected, got),
                    borrow_mode(expected, got)
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
            MirErrorKind::UndefinedVariable { name, near } => {
                write!(f, "undefined variable `{name}`{near}")
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
            MirErrorKind::CannotIndex { ty } => {
                write!(
                    f,
                    "cannot index into a value of type `{}`",
                    ty.shown(interner)
                )
            }
            MirErrorKind::MoveOutOfIndex { ty } => {
                write!(f, "cannot move out of index of `{}`", ty.shown(interner))
            }
            MirErrorKind::ForSourceNotAdmitted { ty } => {
                write!(
                    f,
                    "a `for` traverses `&v`, `&mut v`, an array by value, or `lo..hi`; `{}` is none of them",
                    ty.shown(interner)
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
                    element.shown(interner)
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
                    ty.shown(interner)
                )
            }
            MirErrorKind::CastNotAdmitted { from, to } => match to {
                Ty::Char => write!(
                    f,
                    "only `u8 as char` reaches a char; {} does not",
                    from.shown(interner)
                ),
                _ => write!(
                    f,
                    "`char as` reaches an integer; {} is not one",
                    to.shown(interner)
                ),
            },
            MirErrorKind::DerefOfNonReference(ty) => {
                write!(f, "`*` needs a reference, got {}", ty.shown(interner))
            }
            MirErrorKind::DerefOfNonPrimitive(ty) => {
                write!(
                    f,
                    "`*` reads only a primitive; {} is used through the reference or cloned",
                    ty.shown(interner)
                )
            }
            MirErrorKind::MoveOutOfCapture { name, ty } => {
                write!(
                    f,
                    "cannot move `{name}` out of a closure's capture (type {}); \
                     act through the reference, or clone it",
                    ty.shown(interner)
                )
            }
            MirErrorKind::ViewCaptured => {
                write!(
                    f,
                    "a lambda cannot capture a string or slice view{COPY_OF_A_VIEW}"
                )
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
            MirErrorKind::ClosureReturnedToTheHost(ty) => {
                write!(
                    f,
                    "the program's result holds a closure, {}, and a closure does not leave \
                     the run it was made in",
                    ty.display(self.interner)
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
            MirErrorKind::AmbiguousFunction { name, candidates } => {
                write!(f, "`{name}` is declared by")?;
                listed(f, candidates)
            }
            MirErrorKind::NoMatchingFunction { name, ty } => {
                write!(
                    f,
                    "no `{name}` takes a call of type {}{}",
                    ty.shown(interner),
                    view_in(ty)
                )
            }
            MirErrorKind::PlaceNamedTwice(place) => {
                write!(f, "`{place}` is named twice in one call")
            }
            MirErrorKind::UndefinedFunction { name, near } => {
                write!(f, "undefined function `{name}`{near}")
            }
            MirErrorKind::NotCallable(ty) => {
                write!(f, "cannot call a value of type {}", ty.shown(interner))
            }
            MirErrorKind::StoreThroughSharedReference { subject, ty } => {
                write!(
                    f,
                    "cannot store through {subject}, of type {}: not a `&mut`; bind it with `&mut`",
                    ty.shown(interner)
                )
            }
            MirErrorKind::MutableBorrowOfShared { subject } => {
                write!(
                    f,
                    "{subject} is a shared reference and cannot be borrowed mutably; bind it with `&mut`"
                )
            }
            MirErrorKind::NoInstance { ty, instances, of } => {
                let wanted = match of {
                    InstanceWanted::Callee(None) => "the signature".to_string(),
                    InstanceWanted::Callee(Some(callee)) => qualified(interner, *callee),
                    InstanceWanted::Requirement {
                        signature,
                        required_by: None,
                    } => qualified(interner, *signature),
                    InstanceWanted::Requirement {
                        signature,
                        required_by: Some(by),
                    } => format!(
                        "{} required by {}",
                        qualified(interner, *signature),
                        qualified(interner, *by)
                    ),
                };
                write!(
                    f,
                    "no instance of {wanted} has the call type {}{}",
                    ty.shown(interner),
                    view_in(ty)
                )?;
                let spelled = spelled_once(
                    most_general(instances)
                        .into_iter()
                        .map(|t| t.shown(interner).to_string()),
                );
                if spelled.is_empty() {
                    return Ok(());
                }
                write!(f, "; the instances it could reach are")?;
                listed(f, &spelled)
            }
            MirErrorKind::NoOperatorInstance { op, signature, ty } => {
                write!(
                    f,
                    "`{op}` has no instance of {signature} for {}",
                    ty.shown(interner)
                )
            }
            MirErrorKind::NoOrdering { op, ty } => {
                write!(
                    f,
                    "`{op}` is not defined on {}{}",
                    ty.shown(interner),
                    match holds_text(ty) {
                        true => USE_STRING_CMP,
                        false => "",
                    }
                )
            }
            MirErrorKind::TypeOutOfBound { ty, bound } => {
                write!(
                    f,
                    "type {} is outside the declared bound",
                    ty.shown(interner)
                )?;
                match bound {
                    crate::ty::TyVarBound::Any => write!(f, " (any)"),
                    crate::ty::TyVarBound::OneOf { shapes, .. } => {
                        write!(f, ", one of")?;
                        listed(
                            f,
                            most_general(shapes).into_iter().map(|t| t.shown(interner)),
                        )
                    }
                    crate::ty::TyVarBound::Integer { signed, among } => {
                        if among.len() == crate::ty::IntTy::ALL.len() {
                            write!(f, ", an integer")
                        } else if *signed && among.iter().all(|k| k.signed()) && among.len() == 4 {
                            write!(f, ", a signed integer")
                        } else {
                            write!(f, ", one of")?;
                            listed(f, among.iter().map(|k| k.name()))
                        }
                    }
                }
            }
            MirErrorKind::IntegerLiteralOutOfRange { value, ty } => {
                write!(f, "literal {value} does not fit {}", ty.shown(interner))
            }
            MirErrorKind::IntegerLiteralWidthUnsettled { among } => {
                let names: Vec<&str> = among.iter().map(|k| k.name()).collect();
                write!(
                    f,
                    "an integer literal here may be {}: write a suffix",
                    match names.split_last() {
                        Some((last, before)) if !before.is_empty() =>
                            format!("{} or {last}", before.join(", ")),
                        _ => names.join(", "),
                    }
                )
            }
            MirErrorKind::AmbiguousConversion { from, to, rules } => {
                write!(
                    f,
                    "more than one conversion takes {} to {}: ",
                    from.shown(interner),
                    to.shown(interner)
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
                    from.shown(interner),
                    to.shown(interner)
                )
            }
            MirErrorKind::TryOutsideFunction => {
                write!(
                    f,
                    "`?` needs a function to return from; a template has none"
                )
            }
            MirErrorKind::ReturnOutsideFunction => {
                write!(
                    f,
                    "`return` needs a function to return from; a template has none"
                )
            }
            MirErrorKind::TryOnNonResult(ty) => write!(
                f,
                "`?` takes a Result or an Option, not {}",
                ty.shown(interner)
            ),
            MirErrorKind::TryReturnMismatch { leaves, returns } => write!(
                f,
                "`?` leaves with {} but the function returns {}",
                leaves.shown(interner),
                returns.shown(interner)
            ),
            MirErrorKind::UndefinedField {
                object_ty,
                field,
                near,
            } => {
                write!(
                    f,
                    "no field `{field}` on type {}{near}",
                    object_ty.shown(interner)
                )
            }
            MirErrorKind::FixedLacks { member } => {
                write!(
                    f,
                    "`{member}` cannot be added to a type laid out outside this body"
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
            MirErrorKind::ObjectTooWide { fields, most } => {
                write!(
                    f,
                    "object has {fields} fields and an object has at most {most}"
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
            MirErrorKind::FieldNotStored {
                subject,
                fields,
                near,
            } => {
                let named: Vec<String> = fields.iter().map(|f| format!("`{f}`")).collect();
                write!(
                    f,
                    "{subject} has no {} stored on every path that reaches here{near}",
                    named.join(", ")
                )
            }
            MirErrorKind::MatchIsNotADispatch => {
                write!(
                    f,
                    "non-exhaustive match: these arms are not one dispatch over a tag, so the variants they cover are not known; add a `_` arm"
                )
            }
            MirErrorKind::MatchArmKeyRepeated { key } => {
                write!(
                    f,
                    "unreachable pattern: `{}` is already covered by an earlier arm",
                    key.shown(interner)
                )
            }
            MirErrorKind::UnreachablePattern {
                pattern,
                scrutinee_ty,
                near,
            } => {
                write!(
                    f,
                    "unreachable pattern: `{pattern}` is not a variant of `{}`{near}",
                    scrutinee_ty.shown(interner)
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
                    pattern_ty.shown(interner),
                    source_ty.shown(interner)
                )
            }
            MirErrorKind::BindingTypeMismatch { name, value, ty } => {
                write!(
                    f,
                    "`${name}` is bound to {value:?}, which is not a value of {}, the type its uses require",
                    ty.shown(interner)
                )
            }
            MirErrorKind::InputTypeUndecided(name) => {
                write!(
                    f,
                    "nothing that reads `${name}` decides its type; use it where its type is known"
                )
            }
            MirErrorKind::OperatorDecidedBeforeItsOperand { op, ty } => {
                write!(
                    f,
                    "`{op}` is decided where it is written, and its operand is known to be {} only later",
                    ty.shown(interner)
                )
            }
            MirErrorKind::NameBoundTwice(name) => {
                write!(f, "`{name}` is bound twice in one pattern")
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
                    ty.shown(interner)
                )
            }
            MirErrorKind::SourceNotIterable { actual } => {
                write!(
                    f,
                    "source type `{}` is not iterable",
                    actual.shown(interner)
                )
            }
            MirErrorKind::ArityMismatch {
                func,
                expected,
                got,
            } => match func {
                ShownValue::Named(name) => write!(
                    f,
                    "function `{name}` expects {expected} arguments, got {got}"
                ),
                ShownValue::Anonymous => {
                    write!(f, "this closure expects {expected} arguments, got {got}")
                }
            },
            MirErrorKind::ParseError(msg) => {
                write!(f, "parse error: {msg}")
            }
        }
    }
}

/// The spelling that turns a view into the owned text (RFC-0062 rule 3).
const COPY_OF_A_VIEW: &str = "; write `.to_string()` for the owned text";

/// The bytewise ordering `acvus-ext`'s `string` module offers in place of
/// the operator.
const USE_STRING_CMP: &str = "; use `string::cmp`, `lt`, `le`, `gt` or `ge`";

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

/// A payload still in its wrapper where a plain value was wanted: the
/// spelling that opens it.
fn open_payload(ty: &Ty) -> Option<String> {
    match ty {
        Ty::Option(_) => {
            Some("; an Option is not its payload -- write `.unwrap()`, `?` or match it".to_string())
        }
        Ty::Result(..) => {
            Some("; a Result is not its payload -- write `?`, `.unwrap()` or match it".to_string())
        }
        Ty::Ref(_, inner) => open_payload(&inner.ty),
        _ => None,
    }
}

/// A `&` where a `&mut` was wanted is the one mismatch the argument's own
/// spelling settles.
fn borrow_mode(expected: &Ty, got: &Ty) -> &'static str {
    match (expected, got) {
        (Ty::Ref(crate::ty::Mutability::Mut, _), Ty::Ref(crate::ty::Mutability::Shared, _)) => {
            "; write `&mut` where the `&` is"
        }
        _ => "",
    }
}

/// A call type that carries a view where the owned text was wanted.
fn view_in(ty: &Ty) -> &'static str {
    match ty {
        Ty::Fn { params, ret, .. } => {
            match params.iter().any(|p| mentions_view(&p.ty)) || mentions_view(ret) {
                true => COPY_OF_A_VIEW,
                false => "",
            }
        }
        _ => "",
    }
}

fn mentions_view(ty: &Ty) -> bool {
    match ty {
        Ty::Str => true,
        Ty::Ref(_, inner) => mentions_view(&inner.ty),
        Ty::Option(inner) | Ty::Array(inner, _) | Ty::Slice(inner) => mentions_view(inner),
        Ty::UserDefined { type_args, .. } => type_args.iter().any(|arg| mentions_view(&arg.ty)),
        _ => false,
    }
}

fn copy_of_a_view(expected: &Ty, got: &Ty) -> &'static str {
    match (expected, is_text_view(got)) {
        (Ty::String, true) => COPY_OF_A_VIEW,
        _ => "",
    }
}
