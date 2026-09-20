use acvus_ast::report::Label;
use acvus_ast::{
    AstId, BinOp, Expr, Literal, MatchBlock, Node, ObjectExprField, ObjectPatternField, Pattern,
    RefKind, Span, SuffixedInt, Template, TupleElem, TuplePatternElem,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::{DataShape, MirError, MirErrorKind, ShownValue};
use crate::graph::QualifiedRef;
use crate::ir::{Callee, CastKind, ExternCast, ForKind, IndexAccess, IndexMode};
use crate::solver::{
    Admission, Answer, CallShape, Candidate, CaptureOutcome, CaptureRead, Conversion,
    ConvertedArgument, Decision, DecisionId, EffectRelation, InstanceChoice, InstanceKind,
    LendOutcome, MatchBinding, MatchMode, MatchOutcome, Mismatch, MismatchReason, ReceiverMode,
    ReferencePair, SettledSignature, SignatureCandidate, SignatureName, SignatureOption,
    UndecidedCall, UnjoinedArgument, Unsettled,
};
use crate::ty::generalize_patterns;
use crate::ty::{
    CastTy, Effect, EffectTerm, Infer, InferTy, IntTy, LenTerm, Mutability, ObjectTy, Param,
    ParamTerm, Solver, Task, Ty, TyTerm, TyVarBound, TypeArg, TypeEnv, lift_ty,
};
use crate::variant::VariantPayload;

/// Maps each AST node id to its inferred type.
pub type TypeMap = FxHashMap<AstId, Ty>;

/// Maps expression AST ids to the coercion needed at that point.
/// Produced by the type checker, consumed by the lowerer.
pub type CoercionMap = Vec<(AstId, CastKind)>;

/// Maps callee expression AstId -> QualifiedRef for direct calls.
/// Present only when typeck resolved the callee to a named function.
/// Absent = indirect call (local variable, closure, etc.).
pub type DirectCallMap = FxHashMap<AstId, Callee>;

// -- TypeResolution: boundary between TypeChecker and Lowerer ----------

/// Result of type checking a single script or template.
///
/// Contains concrete `Ty` (frozen from `InferTy` at `check_template`/`check_script`).
/// `Ty = TyTerm<Concrete>` cannot contain unresolved variables by construction
/// (`TyVar = Infallible`), so completeness is guaranteed structurally.
/// One use of a scheme, with the two spans it answers for: where the
/// program uses the scheme, and where a source the use mints begins.
/// `@items | into_iter` is used at the whole pipe and begins a source at
/// `into_iter`, which is the place a refusal over sources points at.
#[derive(Clone, Copy)]
struct SchemeUse {
    at: Span,
    source_begins: Span,
}

/// The callee of a call as the program wrote it: the id the checker
/// records its type under, and the span of the name. A method call and an
/// operator write no name of their own, so both spans are the call's.
#[derive(Clone, Copy)]
struct CalleeSite {
    id: AstId,
    span: Span,
}

/// An `if` whose two branches did not join, with the two branch types as
/// the solver still holds them.
struct BranchMismatch {
    then: InferTy,
    else_: InferTy,
    span: Span,
}

/// A refusal with the second places it points at.
struct Refusal {
    kind: MirErrorKind,
    labels: Vec<Label>,
}

/// One source as a refusal shows it.
struct ShownSource {
    value: ShownValue,
    begins: Option<Label>,
}

/// A bounded type variable and where a violation of its bound is reported.
struct BoundSite {
    var: crate::ty::TypeBoundId,
    span: Span,
}

/// An integer literal awaiting its width, checked against its value once
/// the width is known (RFC-0037).
struct IntLiteral {
    ty: InferTy,
    value: i128,
    span: Span,
}

/// An `e as T`, kept until the solve, because whether `e` is a number is
/// a question its type answers only once the body is solved (RFC-0049).
struct CastSite {
    from: InferTy,
    to: CastTy,
    span: Span,
    /// The span of the target name, where the refusal is about the pair
    /// rather than about the source.
    target_span: Span,
}

/// The first argument of a call that was checked before the callee's
/// parameters were seen: a piped value or a method receiver.
struct FirstArg {
    ty: InferTy,
    site: ArgSite,
}

/// One `a[i]` whose refusal waits for the solve (RFC-0047).
struct IndexUse {
    id: AstId,
    span: Span,
    container: InferTy,
    element: InferTy,
    demand: PlaceDemand,
}

/// One argument whose container was still a variable where the argument met
/// its parameter. What an `a[i]` argument names is settled by the index's own
/// signature decision, so the coercion waits for the solve as `IndexUse`
/// does.
struct SliceArg {
    at: AstId,
    span: Span,
    arg: InferTy,
    param: InferTy,
    view: DeferredView,
}

/// What coercion a deferred argument owes once the solve has named what it
/// lends.
enum DeferredView {
    /// The parameter is the view already (RFC-0047 rule 6): the referent owes
    /// the declaration, and a referent that declares none is the argument
    /// mismatch.
    Asked(Viewed),
    /// The parameter belongs to whichever candidate the signature decision
    /// settled on (RFC-0043 rule 2), so a coercion is owed only where that
    /// parameter turned out to be a view; where it did not, the decision
    /// joined the argument with it.
    OfSettledParam,
}

/// One `a[i]` as the checker reads it (RFC-0047).
struct IndexSite<'a> {
    id: AstId,
    callee_id: AstId,
    object: &'a Expr,
    index: &'a Expr,
    span: Span,
    demand: PlaceDemand,
}

/// What a `&C` names, and `C` itself where the type is not a reference.
fn referent_of(ty: &InferTy) -> &InferTy {
    match ty {
        TyTerm::Ref(_, referent) => &referent.ty,
        other => other,
    }
}

/// Whether one of the container's own `as_slice` signatures takes this
/// container. A head that is still open is nobody's refusal yet.
fn takes_container(candidates: &[SignatureCandidate], container: &InferTy) -> bool {
    let Some(head) = sliceable_head(container) else {
        return true;
    };
    candidates.iter().any(|candidate| {
        let SignatureCandidate::Named { scheme, .. } = candidate else {
            return true;
        };
        match scheme.params().first().map(|param| &param.ty) {
            Some(TyTerm::Ref(_, takes)) => sliceable_head(&takes.ty) == Some(head),
            _ => false,
        }
    })
}

/// A container the machine can slice is named by its head alone: two
/// `Vec`s of different elements have one `as_slice` between them.
fn sliceable_head<V>(ty: &TyTerm<V>) -> Option<Option<QualifiedRef>>
where
    V: crate::ty::Phase,
{
    match ty {
        TyTerm::UserDefined { id, .. } => Some(Some(*id)),
        TyTerm::Array(..) => Some(None),
        _ => None,
    }
}

/// What the pointee of a reference parameter asks of the argument: the run
/// of a container's elements (RFC-0047 rule 6), or the run of a `String`'s
/// bytes (RFC-0062 Decision 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Slice,
    Str,
}

impl View {
    fn of<V>(ty: &TyTerm<V>) -> Option<View>
    where
        V: crate::ty::Phase,
    {
        match ty {
            TyTerm::Slice(_) => Some(View::Slice),
            TyTerm::Str => Some(View::Str),
            _ => None,
        }
    }
}

/// A view at the mutability the parameter asks for, which names the one
/// declaration that takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Viewed {
    view: View,
    mutability: Mutability,
}

impl Viewed {
    fn declaration(self) -> &'static str {
        match (self.view, self.mutability) {
            (View::Str, _) => "as_str",
            (View::Slice, Mutability::Shared) => "as_slice",
            (View::Slice, Mutability::Mut) => "as_slice_mut",
        }
    }
}

/// What an expression under check is read for (RFC-0018): its value, or a
/// reference to the place it names, at that mutability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaceDemand {
    Value,
    Borrow(Mutability),
}

impl PlaceDemand {
    /// The slice `a[i]` takes of its container under this demand.
    fn slice_mutability(self) -> Mutability {
        match self {
            Self::Value | Self::Borrow(Mutability::Shared) => Mutability::Shared,
            Self::Borrow(Mutability::Mut) => Mutability::Mut,
        }
    }

    /// A mutable projection demands its object mutably (RFC-0018); every
    /// other read borrows the object of `a.f` shared (RFC-0047 §3).
    fn through_a_field(self) -> Self {
        match self {
            Self::Borrow(Mutability::Mut) => Self::Borrow(Mutability::Mut),
            Self::Value | Self::Borrow(Mutability::Shared) => Self::Borrow(Mutability::Shared),
        }
    }
}

/// RFC-0030.
struct CandidateReceiver {
    candidate: SignatureCandidate,
    mode: ReceiverMode,
}

/// RFC-0043.
struct AdmittedReceiver {
    candidates: Vec<SignatureCandidate>,
    first: FirstArg,
}

impl AdmittedReceiver {
    fn taking_every_candidate(kept: Vec<CandidateReceiver>, first: FirstArg) -> Self {
        Self {
            candidates: kept.into_iter().map(|seen| seen.candidate).collect(),
            first,
        }
    }
}

/// What an overloaded call's arguments left of its candidate set
/// (RFC-0043).
/// The candidate set as the arguments narrow it, left to right (RFC-0043).
struct Narrowing {
    options: Vec<SignatureOption>,
    awaiting_head: Vec<UnjoinedArgument>,
}

struct Admitted {
    narrowing: Narrowing,
    params: Vec<ParamTerm<Infer>>,
}

/// The set an argument emptied, with the argument types the call was written
/// with, which is what the report names (RFC-0043).
struct ArgumentsRefused {
    types: Vec<InferTy>,
}

/// A call argument at its site, with the place it borrows when it is
/// `&place` or a receiver lent as one: a conversion answered through the
/// reference consumes that place for the call.
#[derive(Debug, Clone)]
struct ArgSite {
    id: AstId,
    span: Span,
    /// The expression that takes this argument -- the call, the `a[i]`, or
    /// the `for` head. The registry holds no declaration span for an
    /// extern parameter, so a refusal about the parameter marks the whole
    /// expression that named it and labels the argument inside it.
    taken_by: Span,
    place: Option<LentPlace>,
}

#[derive(Debug, Clone)]
struct LentPlace {
    id: AstId,
    place: Place,
}

impl LentPlace {
    fn of(expr: &Expr) -> Option<Self> {
        place_of(expr).map(|place| Self {
            id: expr.id(),
            place,
        })
    }
}

impl ArgSite {
    fn of(expr: &Expr, taken_by: Span) -> Self {
        let place = match expr {
            Expr::Borrow { place, .. } => LentPlace::of(place),
            _ => None,
        };
        Self {
            id: expr.id(),
            span: expr.span(),
            taken_by,
            place,
        }
    }

    fn value(expr: &Expr, taken_by: Span) -> Self {
        Self {
            id: expr.id(),
            span: expr.span(),
            taken_by,
            place: None,
        }
    }

    /// A receiver lent to a reference parameter: the receiver is the place.
    fn lent(expr: &Expr, taken_by: Span) -> Self {
        Self {
            id: expr.id(),
            span: expr.span(),
            taken_by,
            place: LentPlace::of(expr),
        }
    }
}

/// A place consumed by a call argument's conversion (RFC-0041): until the
/// call it holds `to`, the referent type the parameter names.
struct Hold {
    root: HeldRoot,
    path: Vec<Astr>,
    to: InferTy,
}

/// The binding a held place's root resolves to.
#[derive(Clone, PartialEq, Eq)]
enum HeldRoot {
    Local { name: Astr, scope: usize },
    Param(Astr),
    Context(QualifiedRef),
}

/// RFC-0043: a candidate whose arity is not yet known is one the call's
/// own arity fixes, so the filter keeps it.
fn takes_arity(candidate: &SignatureCandidate, arity: usize) -> bool {
    candidate.arity().is_none_or(|declared| declared == arity)
}

/// RFC-0030.
fn declared_receiver_mode(ty: &crate::ty::PolyTy) -> ReceiverMode {
    let TyTerm::Fn { params, .. } = ty else {
        return ReceiverMode::Value;
    };
    match params.first().map(|p| &p.ty) {
        Some(TyTerm::Ref(mutability, _)) => ReceiverMode::Lent(*mutability),
        _ => ReceiverMode::Value,
    }
}

/// RFC-0020: `StringEq` and `StringConcat` lend their operand places
/// instead of copying them.
fn operand_stays_lent(op: BinOp) -> bool {
    match op {
        BinOp::Eq | BinOp::Neq | BinOp::Add => true,
        BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte
        | BinOp::Xor
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::And
        | BinOp::Or => false,
    }
}

/// The bound an operator imposes on an operand still open at the operator
/// (RFC-0020, RFC-0043). `Add` admits `String` because `+` on strings is
/// the concatenation.
fn operand_bound(op: BinOp) -> Option<TyVarBound> {
    let integers = || crate::ty::IntTy::ALL.iter().copied().map(TyTerm::Int);
    match op {
        BinOp::Add => Some(TyVarBound::OneOf(
            integers().chain([TyTerm::Float, TyTerm::String]).collect(),
        )),
        BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte => Some(TyVarBound::OneOf(
            integers().chain([TyTerm::Float, TyTerm::Char]).collect(),
        )),
        BinOp::Xor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
            Some(TyVarBound::OneOf(integers().collect()))
        }
        BinOp::Eq | BinOp::Neq | BinOp::And | BinOp::Or => None,
    }
}

/// RFC-0043.
fn agreed_receiver_mode(mut modes: impl Iterator<Item = ReceiverMode>) -> Option<ReceiverMode> {
    let first = modes.next()?;
    modes.all(|mode| mode == first).then_some(first)
}

/// RFC-0030.
fn lent_only_if_agreed(per_candidate: &[CandidateReceiver]) -> ReceiverMode {
    match agreed_receiver_mode(per_candidate.iter().map(|seen| seen.mode)) {
        Some(lent @ ReceiverMode::Lent(_)) => lent,
        Some(ReceiverMode::Value) | None => ReceiverMode::Value,
    }
}

pub use crate::ir::Intrinsic;

/// An operator resolved to a shared signature (RFC-0020).
#[derive(Debug, Clone)]
pub struct OperatorCall<C, T> {
    pub callee: C,
    pub ty: T,
    pub at: Span,
}

/// The reference a body's result names, itself or inside the data it holds.
///
/// A reference inside data is already refused where the data is built
/// (`MirErrorKind::ReferenceInData`), and a variant payload is the one data
/// constructor that check does not reach; `nope::len(&xs)` builds one whose
/// referent the body releases before it returns, and reading the result then
/// reads freed storage. A function type is not searched: its parameters and
/// return are a signature, not storage this run holds.
fn reference_in_result(ty: &InferTy) -> Option<&InferTy> {
    match ty {
        TyTerm::Ref(..) => Some(ty),
        TyTerm::Array(inner, _) | TyTerm::Slice(inner) | TyTerm::Option(inner) => {
            reference_in_result(inner)
        }
        TyTerm::Result(ok, err) => reference_in_result(ok).or_else(|| reference_in_result(err)),
        TyTerm::Tuple(elems) => elems.iter().find_map(reference_in_result),
        TyTerm::Object(object) => object
            .iter()
            .find_map(|(_, field)| reference_in_result(field)),
        TyTerm::Enum { variants, .. } => variants
            .iter()
            .filter_map(|(_, payload)| payload.as_ref())
            .find_map(|payload| reference_in_result(payload)),
        _ => None,
    }
}

/// The reference a body's result names that the machine has no destination
/// for.
///
/// A bare reference is the one `Kind::Ref` word `control::Return` writes and
/// a direct call's destination receives, so RFC-0064 rule 1 admits it here
/// and the borrow check says which storage it names. A view is the register
/// pair only an extern call's `CallShape::Pair*` opens, which is what
/// RFC-0062 meant by a result leaving in the one register a caller reads; a
/// body returning one waits for the machine half of RFC-0064. Below the top
/// level nothing changes: a reference inside data is RFC-0062 Decision 5's
/// refusal whatever its shape.
fn unreturnable_reference(ty: &InferTy) -> Option<&InferTy> {
    match ty {
        TyTerm::Ref(_, target) => is_view(&target.ty).then_some(ty),
        _ => reference_in_result(ty),
    }
}

/// The two adjacent word registers a run occupies (RFC-0047 amended rule 1,
/// RFC-0062 Decision 1). `acvus_interpreter::prepare::is_slice` is the same
/// predicate on a frozen `Ty`, and the two must name the same types: this
/// one decides what a body may return, that one lays the registers out, and
/// a type admitted here that it calls a pair returns into one register and
/// loses its length.
fn is_view(ty: &InferTy) -> bool {
    match ty {
        TyTerm::Slice(_) | TyTerm::Str => true,
        TyTerm::Ref(_, target) => is_view(&target.ty),
        _ => false,
    }
}

/// A `$name` the body reads. A parameter a Signature declared arrives with
/// its type already closed and no place in this body; one the body
/// discovered carries the place it was first read, which is where a type
/// that does not close is refused.
struct ExternParam {
    name: Astr,
    ty: InferTy,
    first_read: Option<Span>,
}

/// A call resolved to a named function, at the instance the solver fixed
/// or is still settling.
#[derive(Debug, Clone, Copy)]
struct ResolvedCallee {
    qref: QualifiedRef,
    instance: Option<InstanceChoice>,
}

/// RFC-0043.
#[derive(Debug, Clone, Copy)]
enum CalleeChoice {
    Resolved(ResolvedCallee),
    Decided(DecisionId),
}

/// A coercion recorded at the value it converts.
struct PendingCoercion {
    at: AstId,
    cast: PendingCast,
}

enum PendingCast {
    Value(PendingExternCast),
    ThroughRef {
        mutability: Mutability,
        cast: PendingExternCast,
        back: PendingExternCast,
    },
    Slice {
        mutability: Mutability,
        as_slice: PendingExternCast,
    },
    Str {
        as_str: PendingExternCast,
    },
}

/// A cast function at the type of one call of it, its instance still the
/// solver's to name.
struct PendingExternCast {
    callee: ResolvedCallee,
    callee_ty: Ty,
}

#[derive(Debug, Clone)]
pub struct TypeResolution {
    pub type_map: TypeMap,
    pub coercion_map: CoercionMap,
    /// Direct call resolution: callee AstId -> the function and instance.
    /// Only contains entries for calls resolved to named functions.
    pub direct_calls: DirectCallMap,
    pub operator_calls: FxHashMap<AstId, OperatorCall<Callee, Ty>>,
    pub intrinsic_calls: FxHashMap<AstId, Intrinsic>,
    /// How each `a[i]` reaches its element (RFC-0047). The checker decided
    /// it from the container's evidence and the element type; the lowering
    /// reads it and decides nothing.
    pub index_access: FxHashMap<AstId, IndexAccess>,
    /// Which of the four heads each `for` was written with (RFC-0057),
    /// keyed by the statement's own id.
    pub for_kinds: FxHashMap<AstId, ForKind>,
    /// Calls `ns::tag(payload)` that are structural variants (RFC-0030).
    pub structural_variant_calls: FxHashSet<AstId>,
    /// The return type of the function each `?` leaves early from (RFC-0038).
    pub try_returns: FxHashMap<AstId, Ty>,
    pub tail_ty: Ty,
    /// Extern parameters ($name) discovered during typecheck.
    pub extern_params: Vec<(Astr, Ty)>,
    /// A lambda's captures, keyed by the lambda's own `AstId`. The lowering
    /// takes this list as the closure's captures, and the same lambda's
    /// `Fn { captures }` in `type_map` holds those names' types positionally,
    /// so the two must stay in one order: the MakeClosure arity and type
    /// check in `validate::type_check` is where a divergence is reported.
    pub lambda_captures: FxHashMap<AstId, Vec<CapturedName>>,
    /// Join of the effects of every call in the body.
    pub effect: Effect,
    pub context_types: FxHashMap<QualifiedRef, Ty>,
}

impl TypeResolution {
    fn new(
        type_map: TypeMap,
        coercion_map: CoercionMap,
        direct_calls: DirectCallMap,
        operator_calls: FxHashMap<AstId, OperatorCall<Callee, Ty>>,
        intrinsic_calls: FxHashMap<AstId, Intrinsic>,
        index_access: FxHashMap<AstId, IndexAccess>,
        for_kinds: FxHashMap<AstId, ForKind>,
        structural_variant_calls: FxHashSet<AstId>,
        try_returns: FxHashMap<AstId, Ty>,
        tail_ty: Ty,
        extern_params: Vec<(Astr, Ty)>,
        lambda_captures: FxHashMap<AstId, Vec<CapturedName>>,
        effect: Effect,
        context_types: FxHashMap<QualifiedRef, Ty>,
    ) -> Self {
        Self {
            type_map,
            coercion_map,
            direct_calls,
            operator_calls,
            intrinsic_calls,
            index_access,
            for_kinds,
            structural_variant_calls,
            try_returns,
            tail_ty,
            extern_params,
            lambda_captures,
            effect,
            context_types,
        }
    }
}

/// A name a closure captures and the reading its body was checked at, so
/// that the lowering emits that reading rather than deciding one.
#[derive(Debug, Clone, Copy)]
pub struct CapturedName {
    pub name: Astr,
    pub read: CaptureRead,
}

/// What `x = e;` found for `x`.
enum AssignTarget {
    /// A binding of the body that carries the statement: its type.
    Bound(InferTy),
    /// Bound outside the innermost lambda, so the lambda sees a copy.
    Captured,
    /// No binding of that name anywhere in scope.
    Unbound,
}

struct LambdaScope {
    depth: usize,
    body_span: Span,
    captures: Vec<Capture>,
    moves_out_of_capture: Vec<CaptureMove>,
}

/// One name a lambda reads from outside its body: the type the name
/// binds, how the body reads it, and the type it reads it at.
struct Capture {
    name: Astr,
    ty: InferTy,
    read: CaptureSource,
    seen: InferTy,
}

/// The reading is the solver's answer where the captured type's head was
/// open at the first use, and the checker's own where it was not.
#[derive(Debug, Clone, Copy)]
enum CaptureSource {
    Read(CaptureRead),
    Decided(DecisionId),
}

impl LambdaScope {
    fn capture(&mut self, name: Astr, ty: &InferTy, read: CaptureSource, seen: &InferTy) {
        if self.captures.iter().any(|c| c.name == name) {
            return;
        }
        self.captures.push(Capture {
            name,
            ty: ty.clone(),
            read,
            seen: seen.clone(),
        });
    }
}

/// A name a lambda takes out of the enclosing closure's captures: the inner
/// closure owns what it captures, so the name leaves a value the enclosing
/// closure owns and is called again (RFC-0018).
struct CaptureMove {
    name: Astr,
    owned: InferTy,
    span: Span,
    /// The enclosing lambda whose capture the name leaves. A lambda directly
    /// inside the body carries no enclosing lambda, and the refusal then
    /// names one place.
    captured_at: Option<Span>,
}

/// Where a value meets a type it may need converting to, and how a
/// conversion that does not exist is reported there.
#[derive(Debug, Clone, Copy)]
struct ConversionSite {
    id: AstId,
    span: Span,
    report: ConversionReport,
}

/// The error a failed conversion is reported as, by the kind of site.
#[derive(Debug, Clone, Copy)]
enum ConversionReport {
    /// A call argument, a branch of an `if`: "expected .., got ..".
    Value,
    /// A template emits the value: it must be a `String`.
    Emit,
    /// An element of a list literal against the first element.
    ListElement,
    /// A value stored into a typed place: a context, a variable, a field,
    /// a dereference.
    Store,
    /// A body's value into its declared return type.
    Return,
    /// A pattern's source against the type the pattern names.
    Pattern,
    /// A lend of a place a call holds (RFC-0041): the held reference
    /// against the parameter.
    HeldLend,
}

/// The expression a pattern's source is: a conversion of the source is a
/// cast of that expression before the match. A member of a destructured
/// source is no expression.
#[derive(Debug, Clone, Copy)]
enum PatternSource {
    Expr(AstId),
    Member,
}

/// How the pattern being checked reads its scrutinee (RFC-0024).
/// `Deferred` is the mode of a pattern whose scrutinee's head is still a
/// variable: a `Decision::Match` settles it into one of the other two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternMode {
    Value,
    Through,
    Deferred,
}

/// A `ContextBind` checked under an open head, with the decision whose
/// answer says whether it is refused (RFC-0014).
struct DeferredContextBind {
    decision: DecisionId,
    span: Span,
}

/// A branch of an `if`: its type and the expression whose value it is.
struct Branch {
    ty: InferTy,
    value: Option<AstId>,
}

/// A conversion decision registered at a site; a cast the solver settles
/// on becomes a `PendingCoercion` when the body is solved.
struct PendingConversion {
    site: ConversionSite,
    decision: DecisionId,
    from: InferTy,
    to: InferTy,
    /// The place the value borrows, at a call argument that is `&place`.
    place: Option<AstId>,
}

/// State only active in analysis mode (partial inference for unknown params).
struct AnalysisState {
    /// Declared parameter types from Signature, consumed in order as $params are discovered.
    declared_param_types: Vec<Ty>,
    /// Next index into declared_param_types.
    next_declared_param: usize,
}

pub struct TypeChecker<'a, 's, 'src> {
    /// Interner for string interning.
    interner: &'a Interner,
    /// Unified type environment: contexts + functions.
    env: &'a TypeEnv,
    /// Namespace this function belongs to.
    namespace: Option<Astr>,
    /// Stack of scopes: each scope maps variable names to types.
    scopes: Vec<FxHashMap<Astr, InferTy>>,
    /// Extern parameters in Signature order, which is the order iteration
    /// must keep.
    param_types: smallvec::SmallVec<[ExternParam; 4]>,
    /// Solver state (borrowed - may be shared across compilations).
    solver: &'s mut Solver<'src>,
    /// Accumulated type map (internal, uses InferTy during inference).
    type_map: FxHashMap<AstId, InferTy>,
    /// Accumulated coercions, each through the cast function at its
    /// call type.
    coercions: Vec<PendingCoercion>,
    /// Direct call resolutions (callee AstId -> the function and instance).
    direct_calls: FxHashMap<AstId, CalleeChoice>,
    operator_calls: FxHashMap<AstId, OperatorCall<ResolvedCallee, InferTy>>,
    /// How each `a[i]` reaches its element (RFC-0047), keyed by the index
    /// expression's own id.
    index_access: FxHashMap<AstId, IndexAccess>,
    /// Which head each `for` under check was written with (RFC-0057).
    for_kinds: FxHashMap<AstId, ForKind>,
    /// The loops enclosing the statement under check, innermost last:
    /// `break` and `continue` name the last one and are refused where there
    /// is none (RFC-0057 Decision 4). A `Some` carries the element type of a
    /// `for x in a` whose array owns what it holds, and leaving such a loop
    /// early would leave the elements it has not taken without a release.
    loops: Vec<Option<Ty>>,
    /// How the pattern being checked reads its scrutinee (RFC-0024).
    pattern_mode: PatternMode,
    /// The bindings of the pattern being checked under `Deferred`, each
    /// waiting for the mode that gives it its type.
    deferred_bindings: Vec<MatchBinding>,
    /// The spans of the `ContextBind`s of the pattern being checked under
    /// `Deferred`: a context cannot hold a reference (RFC-0014), which is
    /// a refusal only once the mode is known.
    deferred_context_binds: Vec<Span>,
    context_binds_under_open_head: Vec<DeferredContextBind>,
    /// Accumulated errors.
    errors: Vec<MirError>,
    /// What the expression under check is read for. One field, so
    /// "borrowed" and "at which mutability" cannot disagree: `a[i]`
    /// settles `as_slice` or `as_slice_mut` on it (RFC-0047 §3).
    demand: PlaceDemand,
    /// Every context the program names, at the span of its first use. A
    /// context's type must be data (RFC-0014); the check runs once the
    /// types are known.
    context_uses: FxHashMap<QualifiedRef, Span>,
    /// Bounded variables instantiated so far, each at the span that will
    /// report a violation.
    bound_sites: Vec<BoundSite>,
    /// Each `if` whose two branches did not join, held until `solve_body`
    /// has solved: the branch types name variables the instance decisions
    /// bind, and a type frozen before they settle prints `!` where the
    /// program wrote a type the solve went on to find.
    branch_mismatches: Vec<BranchMismatch>,
    /// The return type of the function being checked, where one can be
    /// left early: a script's or a lambda's, never a template's (RFC-0038).
    return_ty: Option<InferTy>,
    /// Each `?` with the return type it leaves through (RFC-0038).
    try_sites: FxHashMap<AstId, InferTy>,
    int_literals: Vec<IntLiteral>,
    casts: Vec<CastSite>,
    /// Calls `ns::tag(payload)` that resolved to a structural variant
    /// (RFC-0030), for the lowering.
    structural_variant_calls: FxHashSet<AstId>,
    /// Conversion decisions registered so far, at their sites.
    conversions: Vec<PendingConversion>,
    /// Every `a[i]`, kept for the refusals that can only name their type
    /// once the body is solved (RFC-0047 §2, §5).
    index_uses: Vec<IndexUse>,
    /// RFC-0047 rule 6, drained by `settle_slice_args`.
    slice_args: Vec<SliceArg>,
    /// The places the calls being checked have consumed, innermost last.
    holds: Vec<Hold>,
    /// Decisions instantiated so far, each at the span that will report a
    /// failure.
    decision_sites: FxHashMap<DecisionId, Span>,
    /// Analysis mode state. `None` = normal mode, `Some` = partial inference enabled.
    analysis: Option<AnalysisState>,
    lambda_stack: Vec<LambdaScope>,
    lambda_captures: FxHashMap<AstId, Vec<(Astr, CaptureSource)>>,
    capture_moves: Vec<CaptureMove>,
    /// Maps lambda expression AstId -> body expression AstId.
    /// Effect variable of the body being checked; every call raises its lower bound.
    body_effect: EffectTerm<Infer>,
}

impl<'a, 's, 'src> TypeChecker<'a, 's, 'src> {
    pub fn new(interner: &'a Interner, env: &'a TypeEnv, solver: &'s mut Solver<'src>) -> Self {
        let body_effect = solver.fresh_effect_var();
        Self {
            interner,
            scopes: vec![FxHashMap::default()],
            env,
            namespace: None,
            param_types: smallvec::smallvec![],
            solver,
            type_map: FxHashMap::default(),
            coercions: Vec::new(),
            direct_calls: FxHashMap::default(),
            operator_calls: FxHashMap::default(),
            index_access: FxHashMap::default(),
            for_kinds: FxHashMap::default(),
            loops: Vec::new(),
            pattern_mode: PatternMode::Value,
            deferred_bindings: Vec::new(),
            deferred_context_binds: Vec::new(),
            context_binds_under_open_head: Vec::new(),
            demand: PlaceDemand::Value,
            bound_sites: Vec::new(),
            branch_mismatches: Vec::new(),
            return_ty: None,
            try_sites: FxHashMap::default(),
            int_literals: Vec::new(),
            casts: Vec::new(),
            structural_variant_calls: FxHashSet::default(),
            conversions: Vec::new(),
            index_uses: Vec::new(),
            slice_args: Vec::new(),
            holds: Vec::new(),
            decision_sites: FxHashMap::default(),
            errors: Vec::new(),
            context_uses: FxHashMap::default(),
            analysis: None,
            lambda_stack: Vec::new(),
            lambda_captures: FxHashMap::default(),
            capture_moves: Vec::new(),
            body_effect,
        }
    }

    pub fn with_body_effect(mut self, effect: EffectTerm<Infer>) -> Self {
        self.body_effect = effect;
        self
    }

    fn note_call_effect(&mut self, callee: &EffectTerm<Infer>, span: Span) {
        let body = self.body_effect.clone();
        if let Err(conflict) = self
            .solver
            .unify_effect(callee, &body, EffectRelation::AtMost)
        {
            self.error(MirErrorKind::EffectExceeded(conflict), span);
        }
    }

    /// A context access of the body: joined into its effect (RFC-0017).
    fn note_access(&mut self, access: Effect, span: Span) {
        self.note_call_effect(&EffectTerm::Known(access), span);
    }

    fn close_body_effect(&self) -> Effect {
        self.solver.freeze_effect(&self.body_effect)
    }

    /// Pre-bind function parameters as local variables.
    /// Called before typecheck to inject parameter names+types from Signature.
    pub fn with_params(mut self, params: &[Param]) -> Self {
        for param in params {
            self.param_types.push(ExternParam {
                name: param.name,
                ty: lift_ty(&param.ty),
                first_read: None,
            });
        }
        self
    }

    /// Set the namespace for context lookups.
    pub fn with_namespace(mut self, namespace: Option<Astr>) -> Self {
        self.namespace = namespace;
        self
    }

    /// Enable analysis mode: unknown `$param`s produce fresh type variables
    /// instead of errors, allowing partial type inference.
    pub fn with_analysis_mode(mut self) -> Self {
        self.analysis = Some(AnalysisState {
            declared_param_types: Vec::new(),
            next_declared_param: 0,
        });
        self
    }

    /// Provide declared parameter types from Signature.
    /// In analysis mode, these are consumed in order as free params are discovered.
    pub fn with_declared_param_types(mut self, types: Vec<Ty>) -> Self {
        if let Some(ref mut state) = self.analysis {
            state.declared_param_types = types;
        }
        self
    }

    /// The type a report shows (RFC-0043): as written, a variable
    /// nothing constrained closed to `!`.
    fn type_as_written(&self, ty: &InferTy) -> Ty {
        self.solver.written_ty(ty).unwrap_or_else(|_| Ty::error())
    }

    /// A type the resolution carries into lowering (RFC-0043). A type the
    /// solve left with only a bound is one `Ty` cannot express and the
    /// machine cannot run, so the refusal is where it stops.
    fn closed_or_refused(&mut self, ty: &InferTy, span: Span) -> Ty {
        match self.solver.close_ty(ty) {
            Ok(closed) => closed,
            Err(_) => {
                let resolved_ty = self.type_as_written(ty);
                self.error(MirErrorKind::AmbiguousType { resolved_ty }, span);
                Ty::error()
            }
        }
    }

    /// Freeze the internal InferTy type_map to a concrete TypeMap.
    fn freeze_type_map(&self) -> TypeMap {
        self.type_map
            .iter()
            .map(|(id, ty)| {
                let resolved = self.solver.resolve_ty(ty);
                (
                    *id,
                    self.solver
                        .close_ty(&resolved)
                        .unwrap_or_else(|_| Ty::error()),
                )
            })
            .collect()
    }

    /// Construct an InferTy error token.
    fn infer_error() -> InferTy {
        TyTerm::Error(crate::ty::ErrorToken::new())
    }

    /// Check if an InferTy is an error.
    fn is_error(ty: &InferTy) -> bool {
        matches!(ty, TyTerm::Error(_))
    }

    /// Type check a template. Consumes self, returns TypeResolution.
    /// Template tail type is always String (templates emit text).
    pub fn check_template(
        mut self,
        template: &Template,
    ) -> Result<Freeze<TypeResolution>, Vec<MirError>> {
        self.check_nodes(&template.body);
        self.solve_body();
        self.check_moves_out_of_captures();
        self.check_context_binds_under_open_head();
        self.check_contexts_are_data();
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let resolved: TypeMap = self.freeze_type_map();
        let extern_params = self.frozen_extern_params(template.span);
        let effect = self.close_body_effect();
        let context_types = self.named_context_types();
        let operator_calls = self.frozen_operator_calls();
        let coercion_map = self.frozen_coercions();
        let direct_calls = self.frozen_direct_calls();
        let lambda_captures = self.frozen_lambda_captures();
        // A second gate, because freezing is itself a check: a type the
        // solve left open is found only where it is closed, and every such
        // type is closed above.
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        Ok(Freeze::new(TypeResolution::new(
            resolved,
            coercion_map,
            direct_calls,
            operator_calls,
            self.frozen_intrinsic_calls(),
            self.index_access.clone(),
            self.for_kinds.clone(),
            self.structural_variant_calls,
            FxHashMap::default(),
            Ty::String,
            extern_params,
            lambda_captures,
            effect,
            context_types,
        )))
    }

    /// Type check a script. Consumes self, returns TypeResolution.
    /// `expected_tail`: if provided, the script's tail expression is unified with this type.
    pub fn check_script(
        mut self,
        script: &acvus_ast::Script,
        expected_tail: Option<&Ty>,
    ) -> Result<Freeze<TypeResolution>, Vec<MirError>> {
        // A declared `ret` is the body's return type itself, so every
        // `return` joins against it exactly as the tail does; undeclared, it
        // is the fresh variable the tail resolves.
        let return_ty = match expected_tail {
            Some(declared) => lift_ty(declared),
            None => self.solver.fresh_ty_var(),
        };
        self.return_ty = Some(return_ty.clone());
        for stmt in &script.stmts {
            self.check_stmt(stmt);
        }
        let tail_ty = if let Some(tail) = &script.tail {
            let ty = self.check_expr(tail);
            let site = ConversionSite {
                id: tail.id(),
                span: tail.span(),
                report: ConversionReport::Return,
            };
            if self.flow(&ty, &return_ty, site).is_err() {
                let resolved = self.solver.resolve_ty(&ty);
                let expected = self.solver.resolve_ty(&return_ty);
                self.error(
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&expected),
                        got: self.type_as_written(&resolved),
                    },
                    tail.span(),
                );
            }
            ty
        } else {
            // No tail: the body returns unit, and a declaration says whether
            // that is what the caller reads.
            if let Err(Mismatch { expected, got, .. }) =
                self.solver.unify(&TyTerm::Unit, &return_ty)
            {
                self.error(
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&expected),
                        got: self.type_as_written(&got),
                    },
                    script.span,
                );
            }
            TyTerm::Unit
        };
        self.solve_body();
        if let Some(tail) = &script.tail {
            let resolved = self.solver.resolve_ty(&tail_ty);
            if let Some(reference) = unreturnable_reference(&resolved) {
                let ty = self.type_as_written(reference);
                self.error(MirErrorKind::ReferenceReturnedFromBody(ty), tail.span());
            }
        }
        self.check_moves_out_of_captures();
        self.check_context_binds_under_open_head();
        self.check_contexts_are_data();
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let resolved: TypeMap = self.freeze_type_map();
        let extern_params = self.frozen_extern_params(script.span);
        let tail_at = script.tail.as_ref().map_or(script.span, |tail| tail.span());
        let frozen_tail = self.closed_or_refused(&self.solver.resolve_ty(&tail_ty), tail_at);
        let try_returns = self.frozen_try_returns();
        let effect = self.close_body_effect();
        let context_types = self.named_context_types();
        let operator_calls = self.frozen_operator_calls();
        let coercion_map = self.frozen_coercions();
        let direct_calls = self.frozen_direct_calls();
        let lambda_captures = self.frozen_lambda_captures();
        // A second gate, because freezing is itself a check: a type the
        // solve left open is found only where it is closed, and every such
        // type is closed above.
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        Ok(Freeze::new(TypeResolution::new(
            resolved,
            coercion_map,
            direct_calls,
            operator_calls,
            self.frozen_intrinsic_calls(),
            self.index_access.clone(),
            self.for_kinds.clone(),
            self.structural_variant_calls,
            try_returns,
            frozen_tail,
            extern_params,
            lambda_captures,
            effect,
            context_types,
        )))
    }

    /// A value flows into a position that must have its type (solver.md
    /// R1): the two join at a value position. A join whose one
    /// disagreement is a signature's open representation is the conversion
    /// decision at the site (R4, hash-types.md), answered when that
    /// representation is decided; every other mismatch is the caller's to
    /// report. A flow into a fresh variable or a primitive has no site and
    /// joins through `Solver::unify` directly: no representation is open
    /// there.
    fn flow(
        &mut self,
        value_ty: &InferTy,
        expected_ty: &InferTy,
        site: ConversionSite,
    ) -> Result<(), Mismatch> {
        match self.solver.unify(value_ty, expected_ty) {
            Err(Mismatch {
                reason: MismatchReason::ReprOpen(_),
                ..
            }) => {
                self.convert_at(value_ty, expected_ty, site);
                Ok(())
            }
            joined => joined,
        }
    }

    /// A value meets a type it may need converting to (solver.md R4): the
    /// conversion is a decision the body's solve answers, reported at the
    /// site if none exists.
    fn convert_at(&mut self, value_ty: &InferTy, expected_ty: &InferTy, site: ConversionSite) {
        let decision = self.decide_conversion(value_ty, expected_ty, site.span);
        self.conversions.push(PendingConversion {
            site,
            decision,
            from: value_ty.clone(),
            to: expected_ty.clone(),
            place: None,
        });
    }

    /// `convert_at` for a call argument: a conversion answered through the
    /// reference rewrites the place the argument borrows.
    fn convert_argument_at(&mut self, value_ty: &InferTy, expected_ty: &InferTy, arg: &ArgSite) {
        let site = ConversionSite {
            id: arg.id,
            span: arg.span,
            report: ConversionReport::Value,
        };
        let decision = self.decide_conversion(value_ty, expected_ty, site.span);
        self.conversions.push(PendingConversion {
            site,
            decision,
            from: value_ty.clone(),
            to: expected_ty.clone(),
            place: arg.place.as_ref().map(|lent| lent.id),
        });
    }

    /// How a declared struct's field set refused an object (RFC-0042), for
    /// the join that refused it and no other.
    fn declared_field_refusal(&self, mismatch: &Mismatch) -> Option<MirErrorKind> {
        let shown = |name: Astr| self.interner.resolve(name).to_string();
        match mismatch.reason {
            MismatchReason::ObjectLacksDeclaredField { declared, field } => {
                Some(MirErrorKind::ObjectLacksDeclaredField {
                    declared: shown(declared),
                    field: shown(field),
                })
            }
            MismatchReason::ObjectFieldNotDeclared { declared, field } => {
                Some(MirErrorKind::ObjectFieldNotDeclared {
                    declared: shown(declared),
                    field: shown(field),
                })
            }
            MismatchReason::NoJoin
            | MismatchReason::UnionWithoutHome
            | MismatchReason::ReprOpen(_)
            | MismatchReason::TaskTooHigh { .. } => None,
        }
    }

    /// An argument meets its parameter (solver.md R1, R4). A `&place`
    /// argument's conversion consumes the place (RFC-0041): the place holds
    /// the parameter's referent type until the call ends, and a later lend
    /// of it inside the call is a conversion decision from the held
    /// reference, resolved as a `HeldLend`.
    fn meet_argument(&mut self, arg_ty: &InferTy, param_ty: &InferTy, site: &ArgSite) {
        if self.meet_slice_parameter(arg_ty, param_ty, site) {
            return;
        }
        if self.refused_projection_parameter(param_ty, arg_ty, site) {
            return;
        }
        let Some(lent) = &site.place else {
            if self.refused_field_set(param_ty, arg_ty, site.span) {
                return;
            }
            self.convert_argument_at(arg_ty, param_ty, site);
            return;
        };
        let TyTerm::Ref(mutability, lent_referent) = self.solver.shallow_resolve_ty(arg_ty) else {
            unreachable!("a lent argument is typed by check_borrow, a reference")
        };
        let Some(root) = self.held_root(&lent.place.root) else {
            let TyTerm::Error(_) = self.solver.shallow_resolve_ty(&lent_referent.ty) else {
                unreachable!("a place whose root is bound nowhere is an undefined name")
            };
            return;
        };
        let path = &lent.place.path;
        if let Some(held) = self.held(&root, path) {
            let held_lend_ty =
                TyTerm::Ref(mutability, Box::new(TypeArg::new(lent_referent.repr, held)));
            let site = ConversionSite {
                id: site.id,
                span: site.span,
                report: ConversionReport::HeldLend,
            };
            self.convert_at(&held_lend_ty, param_ty, site);
            return;
        }
        if self.refused_field_set(param_ty, arg_ty, site.span) {
            return;
        }
        self.convert_argument_at(arg_ty, param_ty, site);
        let to = match self.solver.shallow_resolve_ty(param_ty) {
            TyTerm::Ref(_, param_referent) => param_referent.ty,
            _ => lent_referent.ty,
        };
        self.holds.push(Hold {
            root,
            path: path.clone(),
            to,
        });
    }

    /// RFC-0050 rule 6. This must stay ahead of every other join of the
    /// argument with the parameter: `ObjectTy::meet` in `ty.rs` widens a
    /// `Written` set by union, so a call placed after one still compiles
    /// and finds nothing left to refuse.
    ///
    fn refused_projection_parameter(
        &mut self,
        param_ty: &InferTy,
        arg_ty: &InferTy,
        site: &ArgSite,
    ) -> bool {
        let TyTerm::Ref(_, borrowed) = self.solver.shallow_resolve_ty(param_ty) else {
            return false;
        };
        let TyTerm::Object(projection) = self.solver.shallow_resolve_ty(&borrowed.ty) else {
            return false;
        };
        let TyTerm::Ref(_, referent) = self.solver.shallow_resolve_ty(arg_ty) else {
            return false;
        };
        let TyTerm::Object(argument) = self.solver.shallow_resolve_ty(&referent.ty) else {
            return false;
        };
        let Some(field) = projection.borrowed_field_missing_from(&argument) else {
            return false;
        };
        let borrows = self.object_as_written(&projection);
        let has = self.object_as_written(&argument);
        let object = match &site.place {
            Some(lent) => lent.place.display(self.interner),
            None => has.clone(),
        };
        let note =
            format!("the parameter borrows at least `{borrows}`, and `{object}` has `{has}`");
        self.labeled_error(
            MirErrorKind::ProjectionLacksField {
                object,
                field: self.interner.resolve(field).to_string(),
            },
            site.taken_by,
            vec![Label::at(site.span, "this argument"), Label::note(note)],
        );
        true
    }

    fn object_as_written(&self, object: &ObjectTy<Infer>) -> String {
        self.type_as_written(&TyTerm::Object(object.clone()))
            .display(self.interner)
            .to_string()
    }

    /// Whether the argument met a declared struct's field set and disagreed
    /// with it, reported at `span`. No conversion answers such a join: the
    /// field set is the type (RFC-0042).
    fn refused_field_set(&mut self, param_ty: &InferTy, arg_ty: &InferTy, span: Span) -> bool {
        let Err(mismatch) = self.solver.unify(param_ty, arg_ty) else {
            return false;
        };
        let Some(kind) = self.declared_field_refusal(&mismatch) else {
            return false;
        };
        self.error(kind, span);
        true
    }

    /// `&v` at a `&[T]` parameter is the container's own `as_slice` of it,
    /// and `&s` at a `&str` parameter is the `String`'s own `as_str`,
    /// recorded at the argument (RFC-0047 rule 6, RFC-0062 Decision 3). The
    /// two mutabilities must agree, so `&v` at a `&mut [T]` parameter is
    /// refused where every other argument mismatch is; so is a container
    /// that declares no `as_slice`, and an argument already of the
    /// parameter's own type unifies as it is.
    fn meet_slice_parameter(
        &mut self,
        arg_ty: &InferTy,
        param_ty: &InferTy,
        site: &ArgSite,
    ) -> bool {
        let TyTerm::Ref(mutability, wanted) = self.solver.shallow_resolve_ty(param_ty) else {
            return false;
        };
        let Some(view) = View::of(&self.solver.shallow_resolve_ty(&wanted.ty)) else {
            return false;
        };
        if view == View::Str && mutability != Mutability::Shared {
            return false;
        }
        let TyTerm::Ref(lent, container) = self.solver.shallow_resolve_ty(arg_ty) else {
            return false;
        };
        if lent != mutability {
            return false;
        }
        let referent = self.solver.resolve_ty(&container.ty);
        if matches!(referent, TyTerm::Var(_)) {
            self.slice_args.push(SliceArg {
                at: site.id,
                span: site.span,
                arg: arg_ty.clone(),
                param: param_ty.clone(),
                view: DeferredView::Asked(Viewed { view, mutability }),
            });
            return true;
        }
        self.slice_coercion(
            &referent,
            Viewed { view, mutability },
            arg_ty,
            param_ty,
            site.id,
            site.span,
        )
    }

    /// The declaration the referent's evidence settles on, recorded as the
    /// coercion at `at`.
    fn slice_coercion(
        &mut self,
        referent: &InferTy,
        viewed: Viewed,
        arg_ty: &InferTy,
        param_ty: &InferTy,
        at: AstId,
        span: Span,
    ) -> bool {
        let Viewed { view, mutability } = viewed;
        let name = self.interner.intern(viewed.declaration());
        let head = match view {
            View::Str => None,
            View::Slice => match sliceable_head(referent) {
                Some(head) => Some(head),
                None => return false,
            },
        };
        let takes_referent = |takes: &crate::ty::PolyTy| match view {
            View::Str => matches!(takes, TyTerm::String),
            View::Slice => sliceable_head(takes) == head,
        };
        let taker: Option<(QualifiedRef, crate::ty::Scheme)> = self
            .env
            .machine_set(name)
            .into_iter()
            .find(
                |(_, scheme)| match scheme.params().first().map(|param| &param.ty) {
                    Some(TyTerm::Ref(_, takes)) => takes_referent(&takes.ty),
                    _ => false,
                },
            )
            .map(|(qref, scheme)| (qref, scheme.clone()));
        let Some((qref, scheme)) = taker else {
            return false;
        };
        if view == View::Str && !matches!(referent, TyTerm::String) {
            return false;
        }
        let taken = self.applied_at(qref, &scheme, arg_ty, param_ty, span);
        self.coercions.push(PendingCoercion {
            at,
            cast: match view {
                View::Str => PendingCast::Str { as_str: taken },
                View::Slice => PendingCast::Slice {
                    mutability,
                    as_slice: taken,
                },
            },
        });
        true
    }

    /// Every argument whose container the solve has now named.
    fn settle_slice_args(&mut self) {
        let deferred = std::mem::take(&mut self.slice_args);
        for SliceArg {
            at,
            span,
            arg,
            param,
            view,
        } in deferred
        {
            let referent = match self.solver.shallow_resolve_ty(&arg) {
                TyTerm::Ref(_, container) => self.solver.resolve_ty(&container.ty),
                other => other,
            };
            let viewed = match view {
                DeferredView::Asked(viewed) => viewed,
                DeferredView::OfSettledParam => {
                    let TyTerm::Ref(mutability, wanted) = self.solver.shallow_resolve_ty(&param)
                    else {
                        continue;
                    };
                    let Some(view) = View::of(&self.solver.shallow_resolve_ty(&wanted.ty)) else {
                        continue;
                    };
                    Viewed { view, mutability }
                }
            };
            if self.slice_coercion(&referent, viewed, &arg, &param, at, span) {
                continue;
            }
            self.meet_settled_argument(&arg, &param, span);
        }
    }

    /// No conversion decision is asked here, and that is a decision, not an
    /// omission: this runs inside `solve_body` after `Solver::solve`, where
    /// nothing would answer one. A conversion answers `Identity` exactly
    /// where its two sides unify, so a settled argument takes that
    /// unification directly, and a conversion that needs a cast is out of
    /// reach on this path.
    fn meet_settled_argument(&mut self, arg: &InferTy, param: &InferTy, span: Span) {
        let mismatch = MirErrorKind::UnificationFailure {
            expected: self.type_as_written(&self.solver.shallow_resolve_ty(param)),
            got: self.type_as_written(&self.solver.shallow_resolve_ty(arg)),
        };
        if self.solver.unify(arg, param).is_ok() {
            return;
        }
        self.error(mismatch, span);
    }

    fn held_root(&self, root: &PlaceRoot) -> Option<HeldRoot> {
        match root {
            PlaceRoot::Context(qref) => Some(HeldRoot::Context(*qref)),
            PlaceRoot::Local(name) => {
                if let Some(scope) = self.scopes.iter().rposition(|s| s.contains_key(name)) {
                    return Some(HeldRoot::Local { name: *name, scope });
                }
                self.param_types
                    .iter()
                    .any(|param| param.name == *name)
                    .then_some(HeldRoot::Param(*name))
            }
        }
    }

    fn held(&self, root: &HeldRoot, path: &[Astr]) -> Option<InferTy> {
        self.holds
            .iter()
            .rev()
            .find(|hold| hold.root == *root && hold.path == path)
            .map(|hold| hold.to.clone())
    }

    fn decide_conversion(&mut self, from: &InferTy, to: &InferTy, span: Span) -> DecisionId {
        let decision = self.solver.decide(Decision::conversion(from, to));
        self.decision_sites.insert(decision, span);
        decision
    }

    fn push_scope(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Where `x = e;` stores. A binding is assignable only from the body
    /// that introduced it: a name the innermost lambda captured is the
    /// lambda's own copy (captures are by value, RFC-0018), so a store
    /// there would not reach the binding the writer named (RFC-0045).
    fn assign_target(&self, name: Astr) -> AssignTarget {
        let Some(depth) = self.scopes.iter().rposition(|s| s.contains_key(&name)) else {
            return AssignTarget::Unbound;
        };
        if self.lambda_stack.iter().any(|ls| depth < ls.depth) {
            return AssignTarget::Captured;
        }
        AssignTarget::Bound(self.scopes[depth][&name].clone())
    }

    /// A name for a value: its type is a variable of the solver, so what
    /// the value grows to (a field stored, a variant added) is seen by
    /// every use of the name.
    fn define_var(&mut self, name: Astr, ty: InferTy) {
        let ty = match ty {
            TyTerm::Var(_) => ty,
            term => {
                let home = self.solver.fresh_ty_var();
                self.solver
                    .unify(&term, &home)
                    .expect("a fresh variable takes any type");
                home
            }
        };
        let mut held = Vec::new();
        self.solver
            .resolve_ty(&ty)
            .for_each_source(&mut |id| held.push(id));
        for id in held {
            self.solver.name_source(id, name);
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    /// The type a name has where it is used. A name a lambda captured is
    /// seen in its body as the word it is, copied at each use, and as a
    /// shared reference into the closure at every other type (RFC-0018).
    fn lookup_var(&mut self, name: Astr) -> Option<InferTy> {
        for (depth, scope) in self.scopes.iter().enumerate().rev() {
            let Some(ty) = scope.get(&name) else { continue };
            let ty = ty.clone();
            let capturing_lambdas = self
                .lambda_stack
                .iter()
                .filter(|ls| depth < ls.depth)
                .count();
            if capturing_lambdas == 0 {
                return Some(ty);
            }
            let capturing: Vec<usize> = self
                .lambda_stack
                .iter()
                .enumerate()
                .filter(|(_, ls)| depth < ls.depth)
                .map(|(i, _)| i)
                .collect();
            let seen = self.captured_by(name, &ty, &capturing);
            let taken_from_an_enclosing_capture = capturing_lambdas >= 2;
            if taken_from_an_enclosing_capture {
                let captured_at = capturing
                    .iter()
                    .rev()
                    .nth(1)
                    .map(|enclosing| self.lambda_stack[*enclosing].body_span);
                let inner = self
                    .lambda_stack
                    .last_mut()
                    .expect("a lambda deeper than the scope is on the stack");
                if !inner.moves_out_of_capture.iter().any(|m| m.name == name) {
                    let span = inner.body_span;
                    inner.moves_out_of_capture.push(CaptureMove {
                        name,
                        owned: ty.clone(),
                        span,
                        captured_at,
                    });
                }
            }
            return Some(seen);
        }
        None
    }

    /// Records a name of type `ty` as a capture of each lambda in
    /// `capturing`, innermost last, and answers the type the innermost
    /// one's body reads it at. The reading is the solver's answer where
    /// the type's head was open at the first use and the checker's own
    /// where it was not; a second use takes the first use's.
    fn captured_by(&mut self, name: Astr, ty: &InferTy, capturing: &[usize]) -> InferTy {
        let Some(&innermost) = capturing.last() else {
            return ty.clone();
        };
        let inner = &self.lambda_stack[innermost];
        let body_span = inner.body_span;
        let (read, seen) = match inner.captures.iter().find(|c| c.name == name) {
            Some(recorded) => (recorded.read, recorded.seen.clone()),
            None => match self.solver.capture_read(ty) {
                CaptureOutcome::Reads { read, seen } => (CaptureSource::Read(read), seen),
                CaptureOutcome::Refused => (
                    CaptureSource::Read(CaptureRead::Lent),
                    TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(ty.clone()))),
                ),
                CaptureOutcome::HeadOpen => {
                    let seen = self.solver.fresh_ty_var();
                    let decision = self.solver.decide(Decision::Capture {
                        of: ty.clone(),
                        seen: seen.clone(),
                    });
                    self.decision_sites.insert(decision, body_span);
                    (CaptureSource::Decided(decision), seen)
                }
            },
        };
        for lambda in capturing {
            self.lambda_stack[*lambda].capture(name, ty, read, &seen);
        }
        seen
    }

    /// RFC-0043.
    fn signature_set(&mut self, name: QualifiedRef) -> Vec<SignatureCandidate> {
        let mut candidates: Vec<SignatureCandidate> = match self.env.resolve_fn(name) {
            crate::ty::FnLookup::Found(qref, scheme) => vec![SignatureCandidate::Named {
                qref,
                scheme: scheme.clone(),
            }],
            crate::ty::FnLookup::Overloaded(declared) => declared
                .iter()
                .map(|(qref, scheme)| SignatureCandidate::Named {
                    qref: *qref,
                    scheme: (*scheme).clone(),
                })
                .collect(),
            crate::ty::FnLookup::Missing => Vec::new(),
        };
        if name.namespace.is_none()
            && let Some(ty) = self.local_signature(name.name)
        {
            candidates.push(SignatureCandidate::Local { ty });
        }
        candidates
    }

    /// The binding that is one more signature of a bare name (RFC-0043).
    /// A callable is never a word, so a call of a capture whose type is
    /// still open is a call through the reference the body reads it as
    /// (RFC-0018).
    fn local_signature(&mut self, name: Astr) -> Option<InferTy> {
        let seen = self.lookup_var(name)?;
        if self.is_an_open_capture(name, &seen) {
            let lent = self.solver.fresh_ty_var();
            let reference =
                TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(lent.clone())));
            self.solver
                .unify(&seen, &reference)
                .expect("an open capture's type takes a reference");
            return Some(lent);
        }
        Some(self.lent_fn(&seen))
    }

    fn is_an_open_capture(&self, name: Astr, seen: &InferTy) -> bool {
        matches!(self.solver.shallow_resolve_ty(seen), TyTerm::Var(_))
            && self.lambda_stack.last().is_some_and(|ls| {
                ls.captures
                    .iter()
                    .any(|c| c.name == name && matches!(c.read, CaptureSource::Decided(_)))
            })
    }

    /// A call of the binding itself: no signature is named, so no
    /// `direct_calls` entry is frozen and the lowering takes its indirect
    /// path off the variable.
    fn check_local_call(
        &mut self,
        callee_id: AstId,
        callable: &InferTy,
        first: Option<&FirstArg>,
        args: &[Expr],
        call_span: Span,
    ) -> InferTy {
        self.record(callee_id, callable.clone());
        self.check_callable(callable, args, first, call_span)
    }

    /// A lambda's parameter types come from the parameter that receives it
    /// (RFC-0018) when that is known; otherwise they are inference variables.
    /// A lambda, at the function type the position expects where there is
    /// one: its parameters are the expected ones and its body's value
    /// converts into the expected return (solver.md R4), else fresh.
    fn check_lambda(&mut self, expr: &Expr, expected: Option<&InferTy>) -> InferTy {
        let Expr::Lambda {
            id, params, body, ..
        } = expr
        else {
            unreachable!("check_lambda takes a lambda");
        };
        let expected = expected.map(|e| self.solver.shallow_resolve_ty(e));
        let (expected_params, expected_ret): (Vec<InferTy>, Option<InferTy>) = match &expected {
            Some(TyTerm::Fn { params, ret, .. }) => (
                params.iter().map(|p| p.ty.clone()).collect(),
                Some((**ret).clone()),
            ),
            Some(_) | None => (Vec::new(), None),
        };
        self.push_scope();
        let mut param_types = Vec::new();
        for (i, p) in params.iter().enumerate() {
            let pt = match expected_params.get(i) {
                Some(t) => t.clone(),
                None => self.solver.fresh_ty_var(),
            };
            self.define_var(p.name, pt.clone());
            self.record(p.id, pt.clone());
            param_types.push(ParamTerm::new(p.name, pt));
        }
        self.lambda_stack.push(LambdaScope {
            depth: self.scopes.len() - 1,
            body_span: body.span(),
            captures: Vec::new(),
            moves_out_of_capture: Vec::new(),
        });

        let outer_effect = self.body_effect.clone();
        self.body_effect = self.solver.fresh_effect_var();
        let outer_return = self.return_ty.replace(self.solver.fresh_ty_var());
        let outer_holds = std::mem::take(&mut self.holds);
        let outer_loops = std::mem::take(&mut self.loops);
        let body_ty = self.check_expr(body);
        self.loops = outer_loops;
        self.holds = outer_holds;
        let ret = self
            .return_ty
            .take()
            .expect("the lambda's return type was set above");
        self.return_ty = outer_return;
        match &expected_ret {
            Some(expected_ret) => {
                self.solver
                    .unify(&ret, expected_ret)
                    .expect("a lambda's fresh return variable takes the expected return");
                self.convert_at(
                    &body_ty,
                    &ret,
                    ConversionSite {
                        id: body.id(),
                        span: body.span(),
                        report: ConversionReport::Value,
                    },
                );
            }
            None => {
                let site = ConversionSite {
                    id: body.id(),
                    span: body.span(),
                    report: ConversionReport::Return,
                };
                if let Err(Mismatch { expected, got, .. }) = self.flow(&body_ty, &ret, site) {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.type_as_written(&expected),
                            got: self.type_as_written(&got),
                        },
                        body.span(),
                    );
                }
            }
        }
        let lambda_effect = self.body_effect.clone();
        self.body_effect = outer_effect;

        let mut ls = self.lambda_stack.pop().expect("this lambda's scope");
        self.capture_moves.append(&mut ls.moves_out_of_capture);
        self.lambda_captures
            .insert(*id, ls.captures.iter().map(|c| (c.name, c.read)).collect());
        let capture_types: Vec<InferTy> = ls
            .captures
            .iter()
            .map(|c| self.solver.resolve_ty(&c.ty))
            .collect();
        let captured_a_reference = capture_types.iter().zip(&ls.captures).any(|(t, c)| {
            matches!(t, TyTerm::Ref(..)) && !matches!(c.read, CaptureSource::Decided(_))
        });
        if captured_a_reference {
            self.error(MirErrorKind::ReferenceCaptured, body.span());
        }
        if matches!(self.solver.resolve_ty(&ret), TyTerm::Ref(..)) {
            self.error(MirErrorKind::ReferenceReturned, body.span());
        }

        self.pop_scope();
        let ty = TyTerm::Fn {
            params: param_types,
            ret: Box::new(ret),
            captures: capture_types,
            effect: lambda_effect,
        };
        self.record_ret(*id, ty)
    }

    /// A call argument, checked against the parameter that receives it.
    fn check_arg(&mut self, arg: &Expr, expected: Option<&InferTy>) -> InferTy {
        if let (Expr::Lambda { .. }, Some(expected)) = (arg, expected) {
            return self.check_lambda(arg, Some(expected));
        }
        self.check_expr(arg)
    }

    /// Every argument, the first (piped or a receiver) first, checked left
    /// to right; each meets its parameter as soon as it is checked, so a
    /// lambda later in the list is checked at parameter types the earlier
    /// arguments have already fixed, and a place an earlier argument
    /// consumed is held through the later ones.
    fn check_args_in_order(
        &mut self,
        fn_ty: &InferTy,
        first: Option<&FirstArg>,
        args: &[Expr],
        call_span: Span,
    ) -> Vec<InferTy> {
        let params: Vec<InferTy> = match fn_ty {
            TyTerm::Fn { params, .. } => params.iter().map(|p| p.ty.clone()).collect(),
            _ => Vec::new(),
        };
        let offset = usize::from(first.is_some());
        let arity_holds = params.len() == args.len() + offset;
        let meet = |this: &mut Self, param: Option<&InferTy>, arg: &InferTy, site: &ArgSite| {
            let Some(param) = param else {
                return;
            };
            if !arity_holds {
                let _ = this.solver.unify(param, arg);
                return;
            }
            this.meet_argument(arg, param, site);
        };
        let outer_holds = self.holds.len();
        let mut types = Vec::with_capacity(args.len() + offset);
        if let Some(first) = first {
            meet(self, params.first(), &first.ty, &first.site);
            types.push(first.ty.clone());
        }
        for (i, arg) in args.iter().enumerate() {
            let expected = params.get(i + offset);
            let ty = self.check_arg(arg, expected);
            meet(self, expected, &ty, &ArgSite::of(arg, call_span));
            types.push(ty);
        }
        self.holds.truncate(outer_holds);
        types
    }

    /// A reference is never data (RFC-0018).
    fn reject_reference_in_data(&mut self, ty: &InferTy, span: Span, shape: DataShape) {
        let kind = match self.solver.resolve_ty(ty) {
            TyTerm::Ref(_, inner) if matches!(inner.ty, TyTerm::Str) => {
                MirErrorKind::ViewInData(shape)
            }
            TyTerm::Ref(..) => MirErrorKind::ReferenceInData(shape),
            _ => return,
        };
        self.error(kind, span);
    }

    /// Walk a field path on a type, resolving each step.
    /// Returns the leaf type, or an error type if any step fails.
    /// The type stored at `base.path`: each step joins the object with a
    /// partial one naming the field, so a store to a field the object did
    /// not have grows the object (solver.md R1). A base that is not an
    /// object is a type error at the store.
    fn field_path_for_store(&mut self, base: &InferTy, path: &[Astr], span: Span) -> InferTy {
        let mut current = base.clone();
        for field in path {
            let field_ty = self.solver.fresh_ty_var();
            let partial = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter([(
                *field,
                field_ty.clone(),
            )])));
            if let Err(mismatch) = self.solver.unify(&current, &partial) {
                let kind = self.declared_field_refusal(&mismatch).unwrap_or_else(|| {
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&mismatch.got),
                        got: self.type_as_written(&mismatch.expected),
                    }
                });
                self.error(kind, span);
                return Self::infer_error();
            }
            current = field_ty;
        }
        current
    }

    /// The type of the place a store writes: one step per `.field` and
    /// `[index]` of the path, from the root's storage. An index step demands
    /// the container's `as_slice_mut`, which is what makes the element write
    /// exclusive -- the loan the slice holds is the write's, so a container
    /// already held shared is refused there and a root holding `&T` here.
    fn store_place(&mut self, place: &acvus_ast::Place, span: Span) -> InferTy {
        match place {
            acvus_ast::Place::Field {
                id, object, field, ..
            } => {
                let base = self.store_place(object, span);
                let ty = self.field_path_for_store(&base, &[*field], span);
                self.record_ret(*id, ty)
            }
            acvus_ast::Place::Base(acvus_ast::PlaceBase::Element {
                id,
                callee_id,
                container,
                index,
                span: index_span,
            }) => self.check_index(IndexSite {
                id: *id,
                callee_id: *callee_id,
                object: container.expr(),
                index,
                span: *index_span,
                demand: PlaceDemand::Borrow(Mutability::Mut),
            }),
            acvus_ast::Place::Base(acvus_ast::PlaceBase::Root {
                id,
                root: acvus_ast::Root::Context(name),
                ..
            }) => {
                let qref = QualifiedRef::root(*name);
                self.note_context_use(qref, span);
                self.note_access(Effect::write(qref), span);
                let ty = self
                    .env
                    .contexts
                    .get(&qref)
                    .cloned()
                    .unwrap_or_else(|| self.solver.fresh_ty_var());
                self.record_ret(*id, ty)
            }
            acvus_ast::Place::Base(acvus_ast::PlaceBase::Root {
                root: acvus_ast::Root::ExternParam(name),
                ..
            }) => {
                self.error(
                    MirErrorKind::ExternParamAssign(self.interner.resolve(*name).to_string()),
                    span,
                );
                Self::infer_error()
            }
            acvus_ast::Place::Base(acvus_ast::PlaceBase::Root {
                id,
                root: acvus_ast::Root::Local(name),
                ..
            }) => {
                let var_ty = self.lookup_var(*name).unwrap_or_else(|| {
                    self.error(
                        MirErrorKind::UndefinedVariable(self.interner.resolve(*name).to_string()),
                        span,
                    );
                    Self::infer_error()
                });
                self.record(*id, var_ty.clone());
                match self.solver.shallow_resolve_ty(&var_ty) {
                    TyTerm::Ref(Mutability::Mut, inner) => inner.ty,
                    TyTerm::Ref(Mutability::Shared, _) => {
                        let shown = self.type_as_written(&var_ty);
                        self.error(MirErrorKind::StoreThroughSharedReference(shown), span);
                        Self::infer_error()
                    }
                    _ => var_ty,
                }
            }
        }
    }

    /// Every refusal the checker raises without labels of its own passes
    /// here, so a mismatch that is really one type from two sources is
    /// restated wherever it is raised and not only where this run looked.
    fn error(&mut self, kind: MirErrorKind, span: Span) {
        let sources = match &kind {
            MirErrorKind::UnificationFailure { expected, got }
            | MirErrorKind::HeterogeneousList { expected, got } => {
                self.one_type_two_sources(expected, got)
            }
            _ => None,
        };
        match sources {
            Some(refusal) => self.labeled_error(refusal.kind, span, refusal.labels),
            None => self.errors.push(MirError {
                kind,
                span,
                labels: Vec::new(),
            }),
        }
    }

    fn labeled_error(&mut self, kind: MirErrorKind, span: Span, labels: Vec<Label>) {
        self.errors.push(MirError { kind, span, labels });
    }

    fn one_type_two_sources(&self, expected: &Ty, got: &Ty) -> Option<Refusal> {
        if !expected.same_erased(got) {
            return None;
        }
        let (mut left, mut right) = (Vec::new(), Vec::new());
        expected.for_each_source(&mut |id| left.push(id));
        got.for_each_source(&mut |id| right.push(id));
        if left.len() != right.len() {
            return None;
        }
        let (first, second) = left.into_iter().zip(right).find(|(a, b)| a != b)?;
        let (first, second) = (self.shown_source(first), self.shown_source(second));
        Some(Refusal {
            kind: MirErrorKind::OneTypeTwoSources {
                left: first.value.clone(),
                right: second.value.clone(),
            },
            labels: [first.begins, second.begins]
                .into_iter()
                .flatten()
                .collect(),
        })
    }

    fn shown_source(&self, id: crate::ty::IdentityId) -> ShownSource {
        let origin = self.solver.source_origin(id);
        let value = match origin.and_then(|origin| origin.name) {
            Some(name) => ShownValue::Named(self.interner.resolve(name).to_string()),
            None => ShownValue::Anonymous,
        };
        let begins = origin
            .filter(|origin| origin.span != Span::ZERO)
            .map(|origin| Label::at(origin.span, format!("{value}'s source begins here")));
        ShownSource { value, begins }
    }

    /// Instantiate a scheme for a use at `site.at`; its bounded variables
    /// are verified and its instance decided when the body is solved. The
    /// compiler's own instances of a shared signature (RFC-0020) join the
    /// declared ones, and the bound admits their shapes.
    fn instantiate_at(
        &mut self,
        qref: QualifiedRef,
        scheme: &crate::ty::Scheme,
        site: SchemeUse,
    ) -> (InferTy, Option<InstanceChoice>) {
        let compiler_instances = self.compiler_instances(qref);
        let mut scheme = scheme.clone();
        if !compiler_instances.is_empty()
            && let Some(TyVarBound::OneOf(shapes)) = scheme.bounds.first_mut()
        {
            shapes.extend(compiler_instances.iter().filter_map(|c| match &c.ty {
                TyTerm::Fn { ret, .. } => Some((**ret).clone()),
                _ => None,
            }));
        }
        let inst = self
            .solver
            .instantiate_scheme_with(&scheme, compiler_instances);
        self.bound_sites.extend(
            inst.bounded
                .into_iter()
                .map(|var| BoundSite { var, span: site.at }),
        );
        if let Some(InstanceChoice::Decided(decision)) = inst.instance {
            self.decision_sites.insert(decision, site.at);
        }
        let mut minted = Vec::new();
        inst.ty.for_each_source(&mut |id| minted.push(id));
        for id in minted {
            self.solver.source_begins_at(id, site.source_begins);
        }
        (inst.ty, inst.instance)
    }

    /// The callee a resolved call lowers to; `None` while the call's type
    /// is too open to choose an instance, which the lowering treats as it
    /// treats any unresolved call.
    fn callee_of(&self, resolved: ResolvedCallee) -> Option<Callee> {
        let instance = match resolved.instance {
            None => return Some(Callee::Direct(resolved.qref)),
            Some(InstanceChoice::Fixed(instance)) => instance,
            Some(InstanceChoice::Decided(decision)) => match self.solver.answer(decision)? {
                Answer::Instance(InstanceKind::Extern(instance)) => instance,
                Answer::Instance(InstanceKind::Intrinsic(_)) => return None,
                Answer::Conversion(_)
                | Answer::Signature { .. }
                | Answer::Lend(_)
                | Answer::Capture(_)
                | Answer::Match(_) => {
                    unreachable!("an instance decision answers with an instance")
                }
            },
        };
        Some(Callee::Extern {
            id: resolved.qref,
            instance,
        })
    }

    /// RFC-0043.
    fn resolved_of(&self, choice: CalleeChoice) -> Option<ResolvedCallee> {
        match choice {
            CalleeChoice::Resolved(resolved) => Some(resolved),
            CalleeChoice::Decided(decision) => match self.solver.answer(decision)? {
                Answer::Signature {
                    settled: SettledSignature::Named { qref, instance, .. },
                    ..
                } => Some(ResolvedCallee { qref, instance }),
                Answer::Signature {
                    settled: SettledSignature::Local,
                    ..
                } => None,
                Answer::Instance(_)
                | Answer::Conversion(_)
                | Answer::Lend(_)
                | Answer::Capture(_)
                | Answer::Match(_) => {
                    unreachable!("a signature decision answers with a signature")
                }
            },
        }
    }

    /// Verified by `solve_body` as an instantiation's bounded variables are.
    fn settled_signature_bounds(&self) -> Vec<BoundSite> {
        self.direct_calls
            .values()
            .filter_map(|choice| {
                let CalleeChoice::Decided(decision) = choice else {
                    return None;
                };
                let Answer::Signature { settled, .. } = self.solver.answer(*decision)? else {
                    unreachable!("a signature decision answers with a signature")
                };
                let SettledSignature::Named { bounded, .. } = settled else {
                    return None;
                };
                let span = self.decision_sites[decision];
                Some(bounded.into_iter().map(move |var| BoundSite { var, span }))
            })
            .flatten()
            .collect()
    }

    /// The calls whose instance decision settled on an instruction of the
    /// language (RFC-0020).
    fn frozen_intrinsic_calls(&self) -> FxHashMap<AstId, Intrinsic> {
        self.direct_calls
            .iter()
            .filter_map(|(id, choice)| {
                let Some(InstanceChoice::Decided(decision)) = self.resolved_of(*choice)?.instance
                else {
                    return None;
                };
                match self.solver.answer(decision)? {
                    Answer::Instance(InstanceKind::Intrinsic(intrinsic)) => Some((*id, intrinsic)),
                    Answer::Instance(InstanceKind::Extern(_))
                    | Answer::Conversion(_)
                    | Answer::Signature { .. }
                    | Answer::Lend(_)
                    | Answer::Capture(_)
                    | Answer::Match(_) => None,
                }
            })
            .collect()
    }

    /// The compiler's own instances of a shared signature (RFC-0020):
    /// `core::clone` on a `String` is an instruction.
    fn compiler_instances(&self, qref: QualifiedRef) -> Vec<Candidate> {
        let core_clone =
            QualifiedRef::qualified(self.interner.intern("core"), self.interner.intern("clone"));
        if qref != core_clone {
            return Vec::new();
        }
        vec![Candidate {
            instance: InstanceKind::Intrinsic(Intrinsic::StringClone),
            ty: TyTerm::Fn {
                params: vec![ParamTerm::new(
                    self.interner.intern("a"),
                    TyTerm::Ref(
                        Mutability::Shared,
                        Box::new(TypeArg::uniform(TyTerm::String)),
                    ),
                )],
                ret: Box::new(TyTerm::String),
                captures: vec![],
                effect: Effect::PURE.into(),
            },
            admits: Task::Heavy,
        }]
    }

    fn frozen_direct_calls(&self) -> DirectCallMap {
        self.direct_calls
            .iter()
            .filter_map(|(id, choice)| Some((*id, self.callee_of(self.resolved_of(*choice)?)?)))
            .collect()
    }

    fn frozen_coercions(&self) -> CoercionMap {
        self.coercions
            .iter()
            .filter_map(|coercion| {
                let kind = match &coercion.cast {
                    PendingCast::Value(cast) => {
                        let ExternCast {
                            fn_ref,
                            instance,
                            callee_ty,
                        } = self.frozen_cast(cast)?;
                        CastKind::Extern {
                            fn_ref,
                            instance,
                            callee_ty,
                        }
                    }
                    PendingCast::ThroughRef {
                        mutability,
                        cast,
                        back,
                    } => CastKind::ThroughRef {
                        mutability: *mutability,
                        cast: self.frozen_cast(cast)?,
                        back: self.frozen_cast(back)?,
                    },
                    PendingCast::Slice {
                        mutability,
                        as_slice,
                    } => CastKind::Slice {
                        mutability: *mutability,
                        as_slice: self.frozen_cast(as_slice)?,
                    },
                    PendingCast::Str { as_str } => CastKind::Str {
                        as_str: self.frozen_cast(as_str)?,
                    },
                };
                Some((coercion.at, kind))
            })
            .collect()
    }

    fn frozen_cast(&self, cast: &PendingExternCast) -> Option<ExternCast> {
        let Callee::Extern {
            id: fn_ref,
            instance,
        } = self.callee_of(cast.callee)?
        else {
            unreachable!("a cast is an Extern function (RFC-0023)")
        };
        Some(ExternCast {
            fn_ref,
            instance,
            callee_ty: cast.callee_ty.clone(),
        })
    }

    /// Solve the body once it is checked (solver.md): every decision
    /// settles or is reported at its site, casts the conversions settled on
    /// become coercions, and every bounded variable and literal is verified.
    fn solve_body(&mut self) {
        let unsettled = self.solver.solve();
        self.report_unsettled(unsettled);
        for BranchMismatch { then, else_, span } in std::mem::take(&mut self.branch_mismatches) {
            self.error(
                MirErrorKind::UnificationFailure {
                    expected: self.type_as_written(&then),
                    got: self.type_as_written(&else_),
                },
                span,
            );
        }
        self.record_decided_call_types();
        self.resolve_conversions();
        self.settle_slice_args();
        self.settle_index_uses();
        let settled = self.settled_signature_bounds();
        self.bound_sites.extend(settled);
        let sites = std::mem::take(&mut self.bound_sites);
        for site in sites {
            if let Err(crate::ty::FreezeError::OutOfBound { ty, bound, .. }) =
                self.solver.freeze_ty(&TyTerm::Var(site.var))
            {
                self.error(MirErrorKind::TypeOutOfBound { ty, bound }, site.span);
            }
        }
        let casts = std::mem::take(&mut self.casts);
        for CastSite {
            from,
            to,
            span,
            target_span,
        } in casts
        {
            let Ok(from) = self.solver.freeze_ty(&from) else {
                continue;
            };
            let Some(source) = CastTy::of_ty(&from) else {
                if !from.is_error() {
                    self.error(MirErrorKind::CastOfWhatDoesNotCast(from), span);
                }
                continue;
            };
            if !source.admits(to) {
                self.error(
                    MirErrorKind::CastNotAdmitted {
                        from,
                        to: Ty::from(to),
                    },
                    target_span,
                );
            }
        }
        let literals = std::mem::take(&mut self.int_literals);
        for IntLiteral { ty, value, span } in literals {
            let Ok(Ty::Int(k)) = self.solver.freeze_ty(&ty) else {
                continue;
            };
            if !k.holds(value) {
                self.error(
                    MirErrorKind::IntegerLiteralOutOfRange {
                        value,
                        ty: Ty::Int(k),
                    },
                    span,
                );
            }
        }
    }

    /// Every capture with the reading the body was checked at. A decided
    /// reading is settled by now: `solve_body` reports every decision that
    /// stayed open or failed, and a body with errors never reaches here.
    fn frozen_lambda_captures(&mut self) -> FxHashMap<AstId, Vec<CapturedName>> {
        let captures = std::mem::take(&mut self.lambda_captures);
        captures
            .into_iter()
            .map(|(id, names)| {
                let names = names
                    .into_iter()
                    .map(|(name, source)| CapturedName {
                        name,
                        read: match source {
                            CaptureSource::Read(read) => read,
                            CaptureSource::Decided(decision) => {
                                let Some(Answer::Capture(read)) = self.solver.answer(decision)
                                else {
                                    unreachable!("an unsettled capture is an error of the body")
                                };
                                read
                            }
                        },
                    })
                    .collect();
                (id, names)
            })
            .collect()
    }

    /// The callee of an overloaded call is typed here and nowhere else: its
    /// type is the instance the decision settled on, which does not exist
    /// before the solve.
    fn record_decided_call_types(&mut self) {
        let settled: Vec<(AstId, InferTy)> = self
            .direct_calls
            .iter()
            .filter_map(|(id, choice)| {
                let CalleeChoice::Decided(decision) = choice else {
                    return None;
                };
                Some((*id, self.solver.settled_callee_ty(*decision)?))
            })
            .collect();
        for (id, ty) in settled {
            self.record(id, ty);
        }
    }

    /// Every conversion the solve settled on a cast becomes a coercion
    /// through that cast at its call type; at a `HeldLend` a cast is the
    /// call demanding a second representation of a place it holds, a
    /// mismatch of the two referent types.
    fn resolve_conversions(&mut self) {
        let conversions = std::mem::take(&mut self.conversions);
        for conversion in &conversions {
            let Some(Answer::Conversion(answer)) = self.solver.answer(conversion.decision) else {
                continue;
            };
            let span = conversion.site.span;
            if let (
                ConversionReport::HeldLend,
                Conversion::Cast(_) | Conversion::ThroughRef { .. },
            ) = (conversion.site.report, answer)
            {
                self.report_held_lend_mismatch(&conversion.from, &conversion.to, span);
                continue;
            }
            let cast = match answer {
                Conversion::Identity => continue,
                Conversion::Cast(fn_ref) => {
                    PendingCast::Value(self.cast_at(fn_ref, &conversion.from, &conversion.to, span))
                }
                Conversion::ThroughRef {
                    mutability,
                    cast,
                    back,
                } => {
                    let owned_place = conversion.place.filter(|place| {
                        !matches!(
                            self.solver.resolve_ty(&self.type_map[place]),
                            TyTerm::Ref(..)
                        )
                    });
                    if owned_place.is_none() {
                        self.report_unsettled(vec![Unsettled::ConversionNeedsPlace {
                            decision: conversion.decision,
                            from: conversion.from.clone(),
                            to: conversion.to.clone(),
                        }]);
                        continue;
                    }
                    let from = self.solver.shallow_resolve_ty(&conversion.from);
                    let to = self.solver.shallow_resolve_ty(&conversion.to);
                    let Some(references) = ReferencePair::of(&from, &to) else {
                        unreachable!("a ThroughRef answer was formed from a reference pair")
                    };
                    let named_from = references.from.ty.clone();
                    let named_to = references.to.ty.clone();
                    PendingCast::ThroughRef {
                        mutability,
                        cast: self.cast_at(cast, &named_from, &named_to, span),
                        back: self.cast_at(back, &named_to, &named_from, span),
                    }
                }
            };
            self.coercions.push(PendingCoercion {
                at: conversion.site.id,
                cast,
            });
        }
        self.conversions = conversions;
    }

    fn report_held_lend_mismatch(
        &mut self,
        held_lend_ty: &InferTy,
        param_ty: &InferTy,
        span: Span,
    ) {
        let held_lend_ty = self.solver.shallow_resolve_ty(held_lend_ty);
        let param_ty = self.solver.shallow_resolve_ty(param_ty);
        let (expected, got) = match ReferencePair::of(&held_lend_ty, &param_ty) {
            Some(references) => (&references.to.ty, &references.from.ty),
            None => (&param_ty, &held_lend_ty),
        };
        self.error(
            MirErrorKind::UnificationFailure {
                expected: self.type_as_written(expected),
                got: self.type_as_written(got),
            },
            span,
        );
    }

    fn cast_at(
        &mut self,
        fn_ref: QualifiedRef,
        from: &InferTy,
        to: &InferTy,
        span: Span,
    ) -> PendingExternCast {
        let Some(scheme) = self.env.functions.get(&fn_ref) else {
            unreachable!("a cast rule names a declared function (RFC-0023)")
        };
        let scheme = scheme.clone();
        self.applied_at(fn_ref, &scheme, from, to, span)
    }

    /// One declared function at the types of one call of it: the instance the
    /// solver settles for those types, and the closed call type it settles
    /// at. `from` meets the first parameter and `to` the return.
    fn applied_at(
        &mut self,
        fn_ref: QualifiedRef,
        scheme: &crate::ty::Scheme,
        from: &InferTy,
        to: &InferTy,
        span: Span,
    ) -> PendingExternCast {
        let site = SchemeUse {
            at: span,
            source_begins: span,
        };
        let (inst, instance) = self.instantiate_at(fn_ref, scheme, site);
        if let TyTerm::Fn {
            params,
            ret,
            effect,
            ..
        } = &inst
        {
            if let Some(p) = params.first() {
                let _ = self.solver.unify(&p.ty, from);
            }
            let _ = self.solver.unify(ret, to);
            let effect = effect.clone();
            self.note_call_effect(&effect, span);
        }
        let unsettled = self.solver.settle();
        self.report_unsettled(unsettled);
        let callee_ty = self.closed_or_refused(&self.solver.resolve_ty(&inst), span);
        PendingExternCast {
            callee: ResolvedCallee {
                qref: fn_ref,
                instance,
            },
            callee_ty,
        }
    }

    fn decision_span(&self, decision: DecisionId) -> Span {
        if let Some(span) = self.decision_sites.get(&decision) {
            return *span;
        }
        let parent = self.direct_calls.values().find_map(|choice| {
            let CalleeChoice::Decided(parent) = choice else {
                return None;
            };
            let Answer::Signature { settled, .. } = self.solver.answer(*parent)? else {
                unreachable!("a signature decision answers with a signature")
            };
            let SettledSignature::Named { instance, .. } = settled else {
                return None;
            };
            (instance == Some(InstanceChoice::Decided(decision))).then_some(*parent)
        });
        let Some(parent) = parent else {
            unreachable!("a decision is opened at a site or by a settled signature decision")
        };
        self.decision_sites[&parent]
    }

    /// RFC-0043.
    fn shown_candidates(
        &self,
        name: Astr,
        candidates: impl Iterator<Item = SignatureName>,
    ) -> Vec<String> {
        let bare = self.interner.resolve(name);
        let mut shown: Vec<String> = candidates
            .map(|candidate| match candidate {
                SignatureName::Named(qref) => self.shown_name(qref),
                SignatureName::Local => format!("the binding `{bare}`"),
            })
            .collect();
        shown.sort();
        shown
    }

    fn shown_name(&self, qref: QualifiedRef) -> String {
        let name = self.interner.resolve(qref.name);
        match qref.namespace {
            Some(ns) => format!("{}::{name}", self.interner.resolve(ns)),
            None => name.to_string(),
        }
    }

    fn report_unsettled(&mut self, unsettled: Vec<Unsettled>) {
        for failure in unsettled {
            let decision = failure.decision();
            let span = self.decision_span(decision);
            let report = self
                .conversions
                .iter()
                .find(|c| c.decision == decision)
                .map(|c| c.site.report);
            let kind = match failure {
                Unsettled::NoInstance { call, .. } => MirErrorKind::NoInstance {
                    ty: self.type_as_written(&call),
                },
                Unsettled::InstanceMismatch { expected, got, .. } => {
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&expected),
                        got: self.type_as_written(&got),
                    }
                }
                Unsettled::AmbiguousInstance { .. } | Unsettled::ConversionOpen { .. } => {
                    // The call's type stayed open: the lowering treats the
                    // call as unresolved, and the open type is reported
                    // where it is frozen.
                    continue;
                }
                Unsettled::TaskTooHigh {
                    required, found, ..
                } => MirErrorKind::TaskTooHigh { required, found },
                Unsettled::NoConversion { from, to, .. } => {
                    if let TyTerm::Var(var) = self.solver.resolve_ty(&to)
                        && let bound @ (TyVarBound::OneOf(_) | TyVarBound::Integer { .. }) =
                            self.solver.bound_of_var(var)
                    {
                        self.error(
                            MirErrorKind::TypeOutOfBound {
                                ty: self.type_as_written(&from),
                                bound,
                            },
                            span,
                        );
                        continue;
                    }
                    let from = self.type_as_written(&from);
                    let to = self.type_as_written(&to);
                    match report {
                        Some(ConversionReport::Emit) => {
                            MirErrorKind::EmitNotString { actual: from }
                        }
                        Some(ConversionReport::ListElement) => MirErrorKind::HeterogeneousList {
                            expected: to,
                            got: from,
                        },
                        Some(ConversionReport::Pattern) => MirErrorKind::PatternTypeMismatch {
                            pattern_ty: to,
                            source_ty: from,
                        },
                        Some(
                            ConversionReport::Value
                            | ConversionReport::Store
                            | ConversionReport::Return
                            | ConversionReport::HeldLend,
                        )
                        | None => MirErrorKind::UnificationFailure {
                            expected: to,
                            got: from,
                        },
                    }
                }
                Unsettled::AmbiguousConversion {
                    from, to, rules, ..
                } => MirErrorKind::AmbiguousConversion {
                    from: self.type_as_written(&from),
                    to: self.type_as_written(&to),
                    rules,
                },
                Unsettled::ConversionNeedsPlace { from, to, .. } => {
                    MirErrorKind::ConversionNeedsPlace {
                        from: self.type_as_written(&from),
                        to: self.type_as_written(&to),
                    }
                }
                Unsettled::NoSignature { name, call, .. } => MirErrorKind::NoMatchingFunction {
                    name: self.interner.resolve(name).to_string(),
                    ty: self.call_type_as_written(&call),
                },
                Unsettled::EffectExceeded { conflict, .. } => {
                    MirErrorKind::EffectExceeded(conflict)
                }
                Unsettled::AmbiguousSignature {
                    name, candidates, ..
                } => MirErrorKind::AmbiguousFunction {
                    name: self.interner.resolve(name).to_string(),
                    candidates: self.shown_candidates(name, candidates.iter().copied()),
                },
                Unsettled::ReferenceCaptured { .. } => MirErrorKind::ReferenceCaptured,
                Unsettled::MutableBorrowOfShared { .. } => MirErrorKind::MutableBorrowOfShared,
                Unsettled::LendMismatch { expected, got, .. }
                | Unsettled::MatchMismatch { expected, got, .. } => {
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&expected),
                        got: self.type_as_written(&got),
                    }
                }
            };
            self.error(kind, span);
        }
    }

    /// An integer literal's type: a variable any integer width may fill,
    /// checked against the literal's value once it is known (RFC-0037).
    fn int_literal(&mut self, value: i128, span: Span) -> InferTy {
        let ty = self.solver.fresh_int_var();
        self.int_literals.push(IntLiteral {
            ty: ty.clone(),
            value,
            span,
        });
        ty
    }

    /// `10u64`: the width the suffix names, and the value checked against
    /// it here rather than after the solve, because nothing is left for a
    /// use to decide (RFC-0058).
    fn suffixed_int_literal(&mut self, literal: SuffixedInt, span: Span) -> InferTy {
        let width = IntTy::from(literal.width);
        if !width.holds(literal.value) {
            self.error(
                MirErrorKind::IntegerLiteralOutOfRange {
                    value: literal.value,
                    ty: Ty::Int(width),
                },
                span,
            );
        }
        TyTerm::Int(width)
    }

    fn record(&mut self, id: AstId, ty: InferTy) {
        self.type_map.insert(id, ty);
    }

    /// Record the type at an AST id and return it.
    fn record_ret(&mut self, id: AstId, ty: InferTy) -> InferTy {
        self.record(id, ty.clone());
        ty
    }

    fn resolve_context_type(&mut self, qref: QualifiedRef, span: Span) -> InferTy {
        self.note_context_use(qref, span);
        if let Some(ty) = self.env.contexts.get(&qref) {
            return ty.clone();
        }
        let labels = self.declared_contexts().into_iter().collect();
        self.labeled_error(
            MirErrorKind::UndefinedContext(self.interner.resolve(qref.name).to_string()),
            span,
            labels,
        );
        Self::infer_error()
    }

    /// The contexts a reader can reach instead, as a note. A long list is
    /// not one: past this many names the note is a wall of text where the
    /// reader wanted the one name they misspelled, so it is left out.
    fn declared_contexts(&self) -> Option<Label> {
        const SHOWN: usize = 8;
        let mut names: Vec<&str> = self
            .env
            .contexts
            .keys()
            .map(|qref| self.interner.resolve(qref.name))
            .collect();
        if names.is_empty() || names.len() > SHOWN {
            return None;
        }
        names.sort_unstable();
        let written: Vec<String> = names.iter().map(|name| format!("`@{name}`")).collect();
        Some(Label::note(format!("declared: {}", written.join(", "))))
    }

    fn note_context_use(&mut self, qref: QualifiedRef, span: Span) {
        self.context_uses.entry(qref).or_insert(span);
    }

    fn frozen_operator_calls(&mut self) -> FxHashMap<AstId, OperatorCall<Callee, Ty>> {
        let calls = std::mem::take(&mut self.operator_calls);
        calls
            .iter()
            .filter_map(|(id, call)| {
                let callee = self.callee_of(call.callee)?;
                let resolved = self.solver.resolve_ty(&call.ty);
                let ty = self.closed_or_refused(&resolved, call.at);
                Some((
                    *id,
                    OperatorCall {
                        callee,
                        ty,
                        at: call.at,
                    },
                ))
            })
            .collect()
    }

    /// Every `$name` this body reads, at the type the solve closed it to.
    /// A parameter a Signature declared has no place of its own, so a type
    /// that does not close is refused at the body.
    fn frozen_extern_params(&mut self, body: Span) -> Vec<(Astr, Ty)> {
        let params = std::mem::take(&mut self.param_types);
        params
            .iter()
            .map(|param| {
                let resolved = self.solver.resolve_ty(&param.ty);
                let at = param.first_read.unwrap_or(body);
                (param.name, self.closed_or_refused(&resolved, at))
            })
            .collect()
    }

    fn named_context_types(&self) -> FxHashMap<QualifiedRef, Ty> {
        self.context_uses
            .keys()
            .map(|qref| (*qref, self.context_type(*qref).unwrap_or_else(Ty::error)))
            .collect()
    }

    /// The type a context resolved to, once every use has been checked.
    /// `None` while it is still open, or when the context is unknown.
    fn context_type(&self, qref: QualifiedRef) -> Option<Ty> {
        let ty = self.env.contexts.get(&qref)?;
        let resolved = self.solver.resolve_ty(ty);
        self.solver.freeze_ty(&resolved).ok()
    }

    /// A name a lambda took out of the enclosing closure's capture
    /// (RFC-0018). Checked after inference, as a context's data-ness is.
    fn check_moves_out_of_captures(&mut self) {
        let moves = std::mem::take(&mut self.capture_moves);
        for m in &moves {
            let owned = self.solver.resolve_ty(&m.owned);
            if owned.is_primitive() {
                continue;
            }
            let at_the_enclosing_lambda = m
                .captured_at
                .map(|span| Label::at(span, "captured here"))
                .into_iter()
                .collect();
            self.labeled_error(
                MirErrorKind::MoveOutOfCapture {
                    name: self.interner.resolve(m.name).to_string(),
                    ty: self.type_as_written(&owned),
                },
                m.span,
                at_the_enclosing_lambda,
            );
        }
    }

    /// A context holds only data (RFC-0014): a function, a handle, an
    /// order, or a reference cannot be kept by a host from one run to the
    /// next. Checked after inference, at the first use of each context.
    fn check_contexts_are_data(&mut self) {
        let uses: Vec<(QualifiedRef, Span)> =
            self.context_uses.iter().map(|(q, s)| (*q, *s)).collect();
        for (qref, span) in uses {
            let Some(ty) = self.context_type(qref) else {
                continue;
            };
            if ty.is_error() || ty.is_data() {
                continue;
            }
            self.error(
                MirErrorKind::ContextNotData {
                    name: self.interner.resolve(qref.name).to_string(),
                    ty,
                },
                span,
            );
        }
    }

    /// A call that is an instruction of the language (RFC-0020).
    /// `&place` / `&mut place`: the reference type, with the place checked
    /// as a place.
    fn check_borrow(&mut self, place: &Expr, mutable: bool, span: Span) -> InferTy {
        if place_of(place).is_none() {
            self.error(MirErrorKind::NotAPlace, span);
        }
        let mutability = if mutable {
            Mutability::Mut
        } else {
            Mutability::Shared
        };
        let outer = std::mem::replace(&mut self.demand, PlaceDemand::Borrow(mutability));
        let ty = self.check_expr(place);
        self.demand = outer;
        self.lend_place(&ty, place, mutability, span)
    }

    /// `a[i]` (RFC-0047): the container's own `as_slice` settled by its
    /// evidence, the element read at `i: u64`. The instance goes on
    /// `callee_id`, where every other resolved call's does, and the mode
    /// on the expression's own id, so the lowering decides nothing.
    fn check_index(&mut self, site: IndexSite<'_>) -> InferTy {
        let IndexSite {
            id,
            callee_id,
            object,
            index,
            span,
            demand,
        } = site;
        let mutability = demand.slice_mutability();
        let name = self.interner.intern(match mutability {
            Mutability::Shared => "as_slice",
            Mutability::Mut => "as_slice_mut",
        });
        let candidates: Vec<SignatureCandidate> = self
            .env
            .machine_set(name)
            .into_iter()
            .map(|(qref, scheme)| SignatureCandidate::Named {
                qref,
                scheme: scheme.clone(),
            })
            .collect();

        let first = self.receiver_arg(object, ReceiverMode::Lent(mutability), span);
        // The index is a `u64` and nothing else (RFC-0047 §4), so it is the
        // expected type here.
        let position = TyTerm::Int(IntTy::U64);
        let index_ty = self.check_arg(index, Some(&position));
        if self.solver.unify(&index_ty, &position).is_err() {
            self.error(
                MirErrorKind::UnificationFailure {
                    expected: Ty::Int(IntTy::U64),
                    got: self.type_as_written(&index_ty),
                },
                index.span(),
            );
        }

        let container = self.solver.resolve_ty(&first.ty);
        if Self::is_error(referent_of(&container)) {
            return self.record_ret(id, Self::infer_error());
        }
        if !takes_container(
            &candidates,
            &self.solver.resolve_ty(referent_of(&container)),
        ) {
            self.index_uses.push(IndexUse {
                id,
                span,
                container,
                element: Self::infer_error(),
                demand,
            });
            return self.record_ret(id, Self::infer_error());
        }
        let slice = self.check_overloaded_call(
            candidates,
            CalleeSite {
                id: callee_id,
                span,
            },
            name,
            Some(first),
            &[],
            span,
        );
        let element = match self.solver.shallow_resolve_ty(&slice) {
            TyTerm::Ref(_, inner) => match self.solver.shallow_resolve_ty(&inner.ty) {
                TyTerm::Slice(element) => *element,
                _ => Self::infer_error(),
            },
            _ => Self::infer_error(),
        };
        self.index_uses.push(IndexUse {
            id,
            span,
            container,
            element: element.clone(),
            demand,
        });
        self.record_ret(id, element)
    }

    /// How every `a[i]` yields its element, and the refusals that name a
    /// type. A container's element type and its representation are not
    /// settled where the index is written, so both wait for the solve.
    fn settle_index_uses(&mut self) {
        let uses = std::mem::take(&mut self.index_uses);
        for IndexUse {
            id,
            span,
            container,
            element,
            demand,
        } in uses
        {
            let referent = self.solver.resolve_ty(referent_of(&container));
            if Self::is_error(&element) {
                let ty = self.type_as_written(&referent);
                self.error(MirErrorKind::CannotIndex { ty }, span);
                continue;
            }
            let Ok(element) = self.solver.freeze_ty(&element) else {
                continue;
            };
            let moves = crate::validate::is_move_only(&element) == Some(true);
            self.index_access.insert(
                id,
                IndexAccess {
                    mutability: demand.slice_mutability(),
                    mode: match moves {
                        // A word is the element itself; anything else is a
                        // reference into the slice (RFC-0047 §4).
                        false => IndexMode::Copy,
                        true => IndexMode::Ref,
                    },
                },
            );
            if moves && demand == PlaceDemand::Value {
                let ty = self.type_as_written(&referent);
                let note = Label::note(format!(
                    "the element is {}, which moves; take a reference with `&a[i]`",
                    element.display(self.interner)
                ));
                self.labeled_error(MirErrorKind::MoveOutOfIndex { ty }, span, vec![note]);
            }
        }
    }

    /// The reference a borrow of an already-checked place names: a borrow
    /// of a reference is a reborrow of what it names (RFC-0029).
    fn lend_place(
        &mut self,
        of: &InferTy,
        place: &Expr,
        mutability: Mutability,
        span: Span,
    ) -> InferTy {
        if mutability == Mutability::Mut
            && let Some(Place {
                root: PlaceRoot::Context(qref),
                ..
            }) = place_of(place)
        {
            self.note_access(Effect::write(qref), span);
        }
        let referent = match self.solver.lend(of, mutability) {
            LendOutcome::Names { referent, .. } => referent,
            LendOutcome::MutableBorrowOfShared { referent } => {
                self.error(MirErrorKind::MutableBorrowOfShared, span);
                referent
            }
            LendOutcome::HeadOpen => {
                return self.open_lend(of, mutability, span);
            }
        };
        TyTerm::Ref(mutability, Box::new(TypeArg::uniform(referent)))
    }

    /// The place's head is still a variable, so what the reference names
    /// cannot be read off it: the solver settles that when the head
    /// resolves, and refuses it if nothing ever does.
    fn open_lend(&mut self, of: &InferTy, mutability: Mutability, span: Span) -> InferTy {
        let referent = self.solver.fresh_ty_var();
        let decision = self.solver.decide(Decision::Lend {
            of: of.clone(),
            referent: referent.clone(),
            mutability,
        });
        self.decision_sites.insert(decision, span);
        TyTerm::Ref(mutability, Box::new(TypeArg::uniform(referent)))
    }

    /// `recv.f(args)` is `f(recv', args)`, `recv'` borrowed when `f`'s
    /// first parameter is a reference (RFC-0030).
    fn check_method_call(
        &mut self,
        callee_id: AstId,
        receiver: &Expr,
        name: Astr,
        args: &[Expr],
        call_span: Span,
    ) -> InferTy {
        let name_str = self.interner.resolve(name).to_string();
        let candidates = self.signature_set(QualifiedRef::root(name));
        match candidates.as_slice() {
            [] => {}
            [SignatureCandidate::Named { qref, scheme }] => {
                let (qref, scheme) = (*qref, scheme.clone());
                let first =
                    self.receiver_arg(receiver, declared_receiver_mode(&scheme.ty), call_span);
                return self.check_resolved_call(
                    qref,
                    &scheme,
                    CalleeSite {
                        id: callee_id,
                        span: call_span,
                    },
                    &name_str,
                    Some(first),
                    args,
                    call_span,
                );
            }
            [SignatureCandidate::Local { ty }] => {
                let ty = ty.clone();
                let first = FirstArg {
                    ty: self.check_expr(receiver),
                    site: ArgSite::value(receiver, call_span),
                };
                return self.check_local_call(callee_id, &ty, Some(&first), args, call_span);
            }
            _ => {
                let candidates: Vec<SignatureCandidate> = candidates
                    .into_iter()
                    .filter(|candidate| takes_arity(candidate, args.len() + 1))
                    .collect();
                let Some(AdmittedReceiver { candidates, first }) =
                    self.admit_receiver(candidates, receiver, name, call_span)
                else {
                    self.check_args_in_order(&Self::infer_error(), None, args, call_span);
                    return Self::infer_error();
                };
                return self.check_overloaded_call(
                    candidates,
                    CalleeSite {
                        id: callee_id,
                        span: call_span,
                    },
                    name,
                    Some(first),
                    args,
                    call_span,
                );
            }
        }
        self.error(MirErrorKind::UndefinedFunction(name_str), call_span);
        Self::infer_error()
    }

    /// A receiver that is a place is lent as the parameter asks; a receiver
    /// that is already a reference value is passed as it is (RFC-0030).
    fn receiver_arg(&mut self, receiver: &Expr, mode: ReceiverMode, taken_by: Span) -> FirstArg {
        match mode {
            ReceiverMode::Lent(mutability) if place_of(receiver).is_some() => FirstArg {
                ty: self.check_borrow(receiver, mutability == Mutability::Mut, receiver.span()),
                site: ArgSite::lent(receiver, taken_by),
            },
            ReceiverMode::Lent(_) => {
                let ty = self.check_expr(receiver);
                if !matches!(
                    self.solver.resolve_ty(&ty),
                    TyTerm::Ref(..) | TyTerm::Var(_) | TyTerm::Error(_)
                ) {
                    self.error(MirErrorKind::NotAPlace, receiver.span());
                }
                FirstArg {
                    ty,
                    site: ArgSite::value(receiver, taken_by),
                }
            }
            ReceiverMode::Value => FirstArg {
                ty: self.check_expr(receiver),
                site: ArgSite::value(receiver, taken_by),
            },
        }
    }

    /// The receiver of a method call whose name is still a set: one more
    /// argument, admitted per candidate in that candidate's own mode
    /// before the mode the call takes is fixed (RFC-0043).
    fn admit_receiver(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        receiver: &Expr,
        name: Astr,
        call_span: Span,
    ) -> Option<AdmittedReceiver> {
        let per_candidate: Vec<CandidateReceiver> = candidates
            .into_iter()
            .map(|candidate| CandidateReceiver {
                mode: self.solver.receiver_mode(&candidate),
                candidate,
            })
            .collect();
        if place_of(receiver).is_none() {
            let first = self.receiver_arg(receiver, lent_only_if_agreed(&per_candidate), call_span);
            return Some(AdmittedReceiver::taking_every_candidate(
                per_candidate,
                first,
            ));
        }
        let outer = std::mem::replace(
            &mut self.demand,
            match lent_only_if_agreed(&per_candidate) {
                ReceiverMode::Lent(mutability) => PlaceDemand::Borrow(mutability),
                ReceiverMode::Value => PlaceDemand::Borrow(Mutability::Shared),
            },
        );
        let owned = self.check_expr(receiver);
        self.demand = outer;
        let trials: Option<Vec<InferTy>> = per_candidate
            .iter()
            .map(|seen| self.receiver_as(&owned, seen.mode))
            .collect();
        let Some(trials) = trials else {
            let mode = lent_only_if_agreed(&per_candidate);
            let first = self.receiver_in(receiver, owned, mode, call_span);
            return Some(AdmittedReceiver::taking_every_candidate(
                per_candidate,
                first,
            ));
        };
        let kept: Vec<(CandidateReceiver, InferTy)> = per_candidate
            .into_iter()
            .zip(trials)
            .filter(|(seen, ty)| {
                !matches!(
                    self.solver.admits(&seen.candidate, 0, ty),
                    Admission::Refused
                )
            })
            .collect();
        let Some(mode) = self.one_receiver_mode(&kept) else {
            let shown = self.shown_candidates(name, kept.iter().map(|(s, _)| s.candidate.name()));
            self.error(
                MirErrorKind::AmbiguousFunction {
                    name: self.interner.resolve(name).to_string(),
                    candidates: shown,
                },
                call_span,
            );
            return None;
        };
        let first = self.receiver_in(receiver, owned, mode, call_span);
        let kept = kept.into_iter().map(|(seen, _)| seen).collect();
        Some(AdmittedReceiver::taking_every_candidate(kept, first))
    }

    /// The one mode survivors that see the receiver as one type take it
    /// in, or `None` where the types they see differ (RFC-0043). Where one
    /// type is both the lend and the move, the lend is the mode: that type
    /// is the reference already.
    fn one_receiver_mode(&self, kept: &[(CandidateReceiver, InferTy)]) -> Option<ReceiverMode> {
        let Some((_, first)) = kept.first() else {
            return Some(ReceiverMode::Value);
        };
        let first = self.solver.resolve_ty(first);
        if kept
            .iter()
            .any(|(_, ty)| self.solver.resolve_ty(ty) != first)
        {
            return None;
        }
        Some(
            kept.iter()
                .map(|(seen, _)| seen.mode)
                .find(|mode| matches!(mode, ReceiverMode::Lent(_)))
                .unwrap_or(ReceiverMode::Value),
        )
    }

    /// The type one candidate's mode sees the checked receiver as, or
    /// `None` where the place's head is still a variable and the lend
    /// cannot be read off it. No bookkeeping: this is the trial the
    /// candidate is admitted against (RFC-0043).
    fn receiver_as(&self, owned: &InferTy, mode: ReceiverMode) -> Option<InferTy> {
        let ReceiverMode::Lent(mutability) = mode else {
            return Some(owned.clone());
        };
        let referent = match self.solver.lend(owned, mutability) {
            LendOutcome::Names { referent, .. }
            | LendOutcome::MutableBorrowOfShared { referent } => referent,
            LendOutcome::HeadOpen => return None,
        };
        Some(TyTerm::Ref(
            mutability,
            Box::new(TypeArg::uniform(referent)),
        ))
    }

    /// The mode the call settled on, with its bookkeeping: the lend
    /// (RFC-0029, RFC-0041) or the place by value.
    fn receiver_in(
        &mut self,
        receiver: &Expr,
        owned: InferTy,
        mode: ReceiverMode,
        taken_by: Span,
    ) -> FirstArg {
        match mode {
            ReceiverMode::Lent(mutability) => FirstArg {
                ty: self.lend_place(&owned, receiver, mutability, receiver.span()),
                site: ArgSite::lent(receiver, taken_by),
            },
            ReceiverMode::Value => {
                let refused = self.reads_through_reference(receiver)
                    && self.refuse_deref_of_non_primitive(&owned, receiver.span());
                FirstArg {
                    ty: if refused { Self::infer_error() } else { owned },
                    site: ArgSite::value(receiver, taken_by),
                }
            }
        }
    }

    /// RFC-0018: a read that takes a non-primitive out of a place named
    /// through a reference is not a copy. `true` where it was refused.
    fn refuse_deref_of_non_primitive(&mut self, read: &InferTy, span: Span) -> bool {
        let read = self.solver.resolve_ty(read);
        if read.is_primitive() || matches!(read, TyTerm::String | TyTerm::Var(_) | TyTerm::Error(_))
        {
            return false;
        }
        let shown = self.type_as_written(&read);
        self.error(MirErrorKind::DerefOfNonPrimitive(shown), span);
        true
    }

    /// Whether the place names its value through a reference, which is the
    /// case the refusal above is about: a field of an object a reference
    /// names.
    fn reads_through_reference(&self, place: &Expr) -> bool {
        let Expr::FieldAccess { object, .. } = place else {
            return false;
        };
        let object_ty = self
            .type_map
            .get(&object.id())
            .expect("the receiver place was checked before its mode was fixed");
        matches!(self.solver.shallow_resolve_ty(object_ty), TyTerm::Ref(..))
    }

    /// RFC-0043.
    fn check_overloaded_call(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        callee: CalleeSite,
        name: Astr,
        first: Option<FirstArg>,
        args: &[Expr],
        call_span: Span,
    ) -> InferTy {
        let name_str = self.interner.resolve(name).to_string();
        let arity = args.len() + usize::from(first.is_some());
        let candidates: Vec<SignatureCandidate> = candidates
            .into_iter()
            .filter(|candidate| takes_arity(candidate, arity))
            .collect();
        match candidates.as_slice() {
            [] => {
                let types =
                    self.check_args_in_order(&Self::infer_error(), first.as_ref(), args, call_span);
                return self.no_matching_function(&name_str, types, call_span);
            }
            [SignatureCandidate::Named { qref, scheme }] => {
                let (qref, scheme) = (*qref, scheme.clone());
                return self
                    .check_resolved_call(qref, &scheme, callee, &name_str, first, args, call_span);
            }
            [SignatureCandidate::Local { ty }] => {
                let ty = ty.clone();
                return self.check_local_call(callee.id, &ty, first.as_ref(), args, call_span);
            }
            _ => {}
        }
        let admitted = match self.admit_args(candidates, first.as_ref(), args, call_span) {
            Ok(admitted) => admitted,
            Err(ArgumentsRefused { types }) => {
                return self.no_matching_function(&name_str, types, call_span);
            }
        };
        let Admitted {
            narrowing:
                Narrowing {
                    options,
                    awaiting_head,
                },
            params,
        } = admitted;
        let declared_returns: Option<Vec<&crate::ty::PolyTy>> = options
            .iter()
            .map(|option| match &option.candidate {
                SignatureCandidate::Named { scheme, .. } => Some(scheme.ret()),
                SignatureCandidate::Local { .. } => None,
            })
            .collect();
        let ret = match declared_returns.as_deref() {
            Some([first, rest @ ..]) => {
                let shape = rest.iter().fold((*first).clone(), |shape, ret| {
                    generalize_patterns(&shape, ret)
                });
                self.solver.fresh_shape(&shape)
            }
            Some([]) | None => self.solver.fresh_ty_var(),
        };
        let decision = self.solver.decide_signature(UndecidedCall {
            name,
            call: CallShape {
                params,
                ret: ret.clone(),
            },
            options,
            awaiting_head,
            body_effect: self.body_effect.clone(),
        });
        self.decision_sites.insert(decision, call_span);
        self.direct_calls
            .insert(callee.id, CalleeChoice::Decided(decision));
        ret
    }

    /// No signature of the name takes the call (RFC-0043), reported at the
    /// call with the argument types it was written with.
    fn no_matching_function(
        &mut self,
        name: &str,
        types: Vec<InferTy>,
        call_span: Span,
    ) -> InferTy {
        let call = CallShape {
            params: types
                .into_iter()
                .enumerate()
                .map(|(index, ty)| ParamTerm::new(self.interner.intern(&index.to_string()), ty))
                .collect(),
            ret: self.solver.fresh_ty_var(),
        };
        self.error(
            MirErrorKind::NoMatchingFunction {
                name: name.to_string(),
                ty: self.call_type_as_written(&call),
            },
            call_span,
        );
        Self::infer_error()
    }

    /// The function type the call was written as. A call that matched no
    /// signature ran no instance, so it has no effect, and `Effect::PURE`
    /// is what the printer writes as nothing.
    fn call_type_as_written(&self, call: &CallShape) -> Ty {
        Ty::Fn {
            params: call
                .params
                .iter()
                .map(|param| ParamTerm::new(param.name, self.type_as_written(&param.ty)))
                .collect(),
            ret: Box::new(self.type_as_written(&call.ret)),
            captures: vec![],
            effect: EffectTerm::Known(Effect::PURE),
        }
    }

    /// Every argument of an overloaded call, checked left to right against
    /// the set that still takes it (RFC-0043), so that a candidate an
    /// earlier argument refused bounds no later parameter.
    fn admit_args(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        first: Option<&FirstArg>,
        args: &[Expr],
        call_span: Span,
    ) -> Result<Admitted, ArgumentsRefused> {
        let mut narrowing = Narrowing {
            options: candidates
                .into_iter()
                .map(SignatureOption::taking_every_argument_directly)
                .collect(),
            awaiting_head: Vec::new(),
        };
        let offset = usize::from(first.is_some());
        let mut params: Vec<ParamTerm<Infer>> = Vec::with_capacity(args.len() + offset);
        let mut types = Vec::with_capacity(args.len() + offset);
        let outer_holds = self.holds.len();
        if let Some(first) = first {
            let param = self.call_param(&narrowing.options, 0);
            self.admit_arg(&mut narrowing, 0, &param, &first.ty, &first.site);
            params.push(param);
            types.push(first.ty.clone());
        }
        for (i, arg) in args.iter().enumerate() {
            if narrowing.options.is_empty() {
                types.push(self.check_arg(arg, None));
                continue;
            }
            let index = i + offset;
            let param = self.call_param(&narrowing.options, index);
            let ty = self.check_arg(arg, Some(&param.ty));
            self.admit_arg(
                &mut narrowing,
                index,
                &param,
                &ty,
                &ArgSite::of(arg, call_span),
            );
            params.push(param);
            types.push(ty);
        }
        self.holds.truncate(outer_holds);
        if narrowing.options.is_empty() {
            return Err(ArgumentsRefused { types });
        }
        Ok(Admitted { narrowing, params })
    }

    /// The call's parameter at `index` (RFC-0043). Its bound is the
    /// parameter's own range; nothing reads it to decide how an argument
    /// is admitted.
    fn call_param(&mut self, options: &[SignatureOption], index: usize) -> ParamTerm<Infer> {
        let name = options
            .iter()
            .find_map(|option| option.candidate.param_name(index))
            .unwrap_or_else(|| self.interner.intern(&index.to_string()));
        let bound = options
            .iter()
            .map(|option| option.candidate.param_bound(index))
            .fold(TyVarBound::OneOf(vec![]), TyVarBound::union);
        ParamTerm::new(name, self.solver.fresh_var_with(bound))
    }

    /// One argument against the set (RFC-0043): Direct first, then a weaker
    /// admission from a resolved head, and an argument whose head the solve
    /// has not named held for the decision to ask again.
    fn admit_arg(
        &mut self,
        narrowing: &mut Narrowing,
        index: usize,
        param: &ParamTerm<Infer>,
        ty: &InferTy,
        site: &ArgSite,
    ) {
        let options = &mut narrowing.options;
        if options
            .iter()
            .any(|option| self.solver.admission_waits(&option.candidate, index, ty))
        {
            options.retain(|option| {
                !matches!(
                    self.solver.admits(&option.candidate, index, ty),
                    Admission::Refused
                )
            });
            if options.is_empty() {
                return;
            }
            narrowing.awaiting_head.push(UnjoinedArgument {
                index,
                ty: ty.clone(),
            });
            self.slice_args.push(SliceArg {
                at: site.id,
                span: site.span,
                arg: ty.clone(),
                param: param.ty.clone(),
                view: DeferredView::OfSettledParam,
            });
            return;
        }
        let admissions: Vec<Admission> = options
            .iter()
            .map(|option| self.solver.admits(&option.candidate, index, ty))
            .collect();
        let direct = admissions.contains(&Admission::Direct);
        let mut converts = false;
        let mut views = false;
        *options = std::mem::take(options)
            .into_iter()
            .zip(admissions)
            .filter_map(|(mut option, admission)| match admission {
                Admission::Direct => Some(option),
                Admission::Converted | Admission::Viewed if direct => None,
                Admission::Converted => {
                    option.converted.push(ConvertedArgument {
                        index,
                        ty: ty.clone(),
                    });
                    converts = true;
                    Some(option)
                }
                Admission::Viewed => {
                    option.viewed.push(ConvertedArgument {
                        index,
                        ty: ty.clone(),
                    });
                    views = true;
                    Some(option)
                }
                Admission::Refused => None,
            })
            .collect();
        if options.is_empty() {
            return;
        }
        if views {
            self.slice_args.push(SliceArg {
                at: site.id,
                span: site.span,
                arg: ty.clone(),
                param: param.ty.clone(),
                view: DeferredView::Asked(Viewed {
                    view: View::Str,
                    mutability: Mutability::Shared,
                }),
            });
            return;
        }
        if converts {
            self.convert_argument_at(ty, &param.ty, site);
        } else {
            self.meet_argument(ty, &param.ty, site);
        }
    }

    /// A resolved named call, once its first argument (piped or a method
    /// receiver) has been checked.
    #[allow(clippy::too_many_arguments)]
    fn check_resolved_call(
        &mut self,
        resolved_qref: QualifiedRef,
        fn_sig: &crate::ty::Scheme,
        callee: CalleeSite,
        name_str: &str,
        first: Option<FirstArg>,
        args: &[Expr],
        call_span: Span,
    ) -> InferTy {
        let site = SchemeUse {
            at: call_span,
            source_begins: callee.span,
        };
        let (fn_ty, instance) = self.instantiate_at(resolved_qref, fn_sig, site);
        let arg_types = self.check_args_in_order(&fn_ty, first.as_ref(), args, call_span);
        match &fn_ty {
            TyTerm::Fn {
                params: param_tys,
                ret,
                effect,
                ..
            } => {
                if !self.check_arity(name_str, arg_types.len(), param_tys.len(), call_span) {
                    return Self::infer_error();
                }
                let effect = effect.clone();
                self.note_call_effect(&effect, call_span);
                self.record(callee.id, fn_ty.clone());
                self.direct_calls.insert(
                    callee.id,
                    CalleeChoice::Resolved(ResolvedCallee {
                        qref: resolved_qref,
                        instance,
                    }),
                );
                (**ret).clone()
            }
            _ => {
                self.error(
                    MirErrorKind::UndefinedFunction(name_str.to_string()),
                    call_span,
                );
                Self::infer_error()
            }
        }
    }

    /// `ns::tag(payload)` where `ns` names no function: the structural
    /// variant it was before namespaces (RFC-0030).
    fn check_structural_variant(
        &mut self,
        id: AstId,
        enum_name: Astr,
        tag: Astr,
        args: &[Expr],
        span: Span,
    ) -> InferTy {
        let [payload] = args else {
            self.error(
                MirErrorKind::UndefinedFunction(format!(
                    "{}::{}",
                    self.interner.resolve(enum_name),
                    self.interner.resolve(tag)
                )),
                span,
            );
            return Self::infer_error();
        };
        let payload_ty = self.check_expr(payload);
        let mut variants = FxHashMap::default();
        variants.insert(tag, Some(Box::new(payload_ty)));
        self.structural_variant_calls.insert(id);
        TyTerm::Enum {
            name: enum_name,
            variants,
        }
    }

    /// An operator on a non-primitive is a call of its shared signature
    /// (RFC-0020).
    fn check_operator_call(
        &mut self,
        id: AstId,
        op: &'static str,
        operand: &InferTy,
        operands: &[InferTy],
        span: Span,
    ) {
        let qref =
            QualifiedRef::qualified(self.interner.intern("core"), self.interner.intern("eq"));
        let Some(scheme) = self.env.functions.get(&qref).cloned() else {
            self.error(
                MirErrorKind::UndefinedFunction(format!("core::eq for {op}")),
                span,
            );
            return;
        };
        let site = SchemeUse {
            at: span,
            source_begins: span,
        };
        let (fn_ty, instance) = self.instantiate_at(qref, &scheme, site);
        let TyTerm::Fn { params, effect, .. } = &fn_ty else {
            unreachable!("a shared signature is a function type");
        };
        let params: Vec<InferTy> = params.iter().map(|p| p.ty.clone()).collect();
        for (given, param) in operands.iter().zip(params.iter()) {
            let borrowed = match given {
                TyTerm::Ref(..) => given.clone(),
                other => TyTerm::Ref(
                    Mutability::Shared,
                    Box::new(TypeArg::uniform(other.clone())),
                ),
            };
            if self.solver.unify(&borrowed, param).is_err() {
                let shown = self.type_as_written(operand);
                self.error(MirErrorKind::NoOperatorInstance { op, ty: shown }, span);
                return;
            }
        }
        let effect = effect.clone();
        self.note_call_effect(&effect, span);
        self.operator_calls.insert(
            id,
            OperatorCall {
                callee: ResolvedCallee { qref, instance },
                ty: fn_ty,
                at: span,
            },
        );
    }

    /// The type an operand is read at. A reference the program wrote is
    /// read through it, which is the word copy of RFC-0018 where the
    /// referent is a word and the lend of RFC-0020 where the operator
    /// keeps the reference. A referent still open is read through as well,
    /// because the operand bound the operator then imposes leaves it a
    /// word.
    fn read_operand_through(&mut self, operand: &InferTy, op: BinOp) -> InferTy {
        let resolved = self.solver.resolve_ty(operand);
        let TyTerm::Ref(_, inner) = &resolved else {
            return resolved;
        };
        let referent = self.solver.resolve_ty(&inner.ty);
        let word = referent.is_primitive() || matches!(referent, TyTerm::Var(_) | TyTerm::Error(_));
        if operand_stays_lent(op) || word {
            referent
        } else {
            resolved
        }
    }

    /// RFC-0043: `false` where no type satisfies both operands and the
    /// operator's bound.
    fn unify_operands(&mut self, op: BinOp, lt: &InferTy, rt: &InferTy, span: Span) -> bool {
        if self.solver.unify(lt, rt).is_err() {
            return false;
        }
        let Some(bound) = operand_bound(op) else {
            return true;
        };
        if !matches!(self.solver.resolve_ty(lt), TyTerm::Var(_)) {
            return true;
        }
        let bounded = self.solver.fresh_var_with(bound);
        if self.solver.unify(lt, &bounded).is_err() {
            return false;
        }
        let TyTerm::Var(var) = self.solver.resolve_ty(lt) else {
            unreachable!("two open variables unify into an open variable");
        };
        self.bound_sites.push(BoundSite { var, span });
        true
    }

    fn binop_error(&mut self, op: &'static str, left: InferTy, right: InferTy, span: Span) {
        self.error(
            MirErrorKind::TypeMismatchBinOp {
                op,
                left: self.type_as_written(&left),
                right: self.type_as_written(&right),
            },
            span,
        );
    }

    fn check_arity(&mut self, func: &str, got: usize, expected: usize, call_span: Span) -> bool {
        if got == expected {
            return true;
        }
        self.error(
            MirErrorKind::ArityMismatch {
                func: func.to_string(),
                expected,
                got,
            },
            call_span,
        );
        false
    }

    fn check_nodes(&mut self, nodes: &[Node]) {
        for node in nodes {
            self.check_node(node);
        }
    }

    fn check_node(&mut self, node: &Node) {
        match node {
            Node::Text { .. } | Node::Comment { .. } => {}
            Node::InlineExpr { expr, span, .. } => {
                let ty = self.check_expr(expr);
                let resolved = self.solver.resolve_ty(&ty);
                // A `&str` emits as it is: the node joins the template's
                // other parts in one `StringConcat`, which reads a part's
                // bytes through either representation (RFC-0062 Decision 3).
                match &resolved {
                    TyTerm::String | TyTerm::Error(_) => {}
                    TyTerm::Ref(_, inner)
                        if matches!(
                            self.solver.resolve_ty(&inner.ty),
                            TyTerm::String | TyTerm::Str
                        ) => {}
                    TyTerm::Var(_) => self.convert_at(
                        &ty,
                        &TyTerm::String,
                        ConversionSite {
                            id: expr.id(),
                            span: *span,
                            report: ConversionReport::Emit,
                        },
                    ),
                    _ => self.error(
                        MirErrorKind::EmitNotString {
                            actual: self.type_as_written(&resolved),
                        },
                        *span,
                    ),
                }
            }
            Node::MatchBlock(mb) => self.check_match_block(mb),
        }
    }

    /// Type-check a single script statement.
    fn check_stmt(&mut self, stmt: &acvus_ast::Stmt) {
        match stmt {
            acvus_ast::Stmt::Store {
                id,
                place,
                expr,
                span,
            } => {
                let ty = self.check_expr(expr);
                let target_ty = self.store_place(place, *span);
                let site = ConversionSite {
                    id: expr.id(),
                    span: *span,
                    report: ConversionReport::Store,
                };
                if self.flow(&ty, &target_ty, site).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.type_as_written(&target_ty),
                            got: self.type_as_written(&ty),
                        },
                        *span,
                    );
                }
                self.record(*id, ty);
            }
            acvus_ast::Stmt::DerefStore {
                id,
                target,
                expr,
                span,
            } => {
                let ty = self.check_expr(expr);
                let tt = self.check_expr(target);
                let inner = match self.solver.resolve_ty(&tt) {
                    TyTerm::Ref(Mutability::Mut, inner) => inner.ty,
                    TyTerm::Error(_) => Self::infer_error(),
                    other => {
                        let shown = self.type_as_written(&other);
                        self.error(MirErrorKind::StoreThroughSharedReference(shown), *span);
                        Self::infer_error()
                    }
                };
                let site = ConversionSite {
                    id: expr.id(),
                    span: *span,
                    report: ConversionReport::Store,
                };
                if self.flow(&ty, &inner, site).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.type_as_written(&inner),
                            got: self.type_as_written(&ty),
                        },
                        *span,
                    );
                }
                self.record(*id, ty);
            }
            acvus_ast::Stmt::Expr(expr) => {
                self.check_expr(expr);
            }

            // -- Script mode statements ------------------------------
            acvus_ast::Stmt::LetBind {
                id,
                name,
                expr,
                span: _,
            } => {
                let ty = self.check_expr(expr);
                self.define_var(*name, ty.clone());
                self.record(*id, ty);
            }
            acvus_ast::Stmt::LetUninit { id, name, span: _ } => {
                let ty = self.solver.fresh_ty_var();
                self.define_var(*name, ty.clone());
                self.record(*id, ty);
            }
            acvus_ast::Stmt::Assign {
                id,
                name,
                expr,
                span,
            } => {
                let ty = self.check_expr(expr);
                let var_ty = match self.assign_target(*name) {
                    AssignTarget::Bound(var_ty) => var_ty,
                    AssignTarget::Captured => {
                        let at_the_lambda = self
                            .lambda_stack
                            .last()
                            .map(|ls| Label::at(ls.body_span, "captured here"))
                            .into_iter()
                            .collect();
                        self.labeled_error(
                            MirErrorKind::AssignToCapture(self.interner.resolve(*name).to_string()),
                            *span,
                            at_the_lambda,
                        );
                        Self::infer_error()
                    }
                    AssignTarget::Unbound => {
                        self.error(
                            MirErrorKind::AssignToUnbound(self.interner.resolve(*name).to_string()),
                            *span,
                        );
                        Self::infer_error()
                    }
                };
                let site = ConversionSite {
                    id: expr.id(),
                    span: *span,
                    report: ConversionReport::Store,
                };
                if self.flow(&ty, &var_ty, site).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.type_as_written(&var_ty),
                            got: self.type_as_written(&ty),
                        },
                        *span,
                    );
                }
                self.record(*id, ty);
            }
            acvus_ast::Stmt::While {
                cond, body, span, ..
            } => {
                let cond_ty = self.check_expr(cond);
                if self.solver.unify(&cond_ty, &TyTerm::Bool).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: Ty::Bool,
                            got: self.type_as_written(&cond_ty),
                        },
                        *span,
                    );
                }
                self.push_scope();
                self.loops.push(None);
                for s in body {
                    self.check_stmt(s);
                }
                self.loops.pop();
                self.pop_scope();
            }
            acvus_ast::Stmt::Anyorder { body, .. } => {
                self.push_scope();
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            acvus_ast::Stmt::WhileLet {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                let source_ty = self.check_expr(source);
                let resolved = self.solver.resolve_ty(&source_ty);
                self.push_scope();
                self.loops.push(None);
                self.check_pattern(pattern, &resolved, PatternSource::Expr(source.id()), *span);
                for s in body {
                    self.check_stmt(s);
                }
                self.loops.pop();
                self.pop_scope();
            }
            acvus_ast::Stmt::For {
                id,
                callee_id,
                binding,
                head,
                body,
                span,
            } => self.check_for(*id, *callee_id, *binding, head, body, *span),
            acvus_ast::Stmt::Break { span, .. } => self.check_loop_jump("break", *span),
            acvus_ast::Stmt::Continue { span, .. } => self.check_loop_jump("continue", *span),
        }
    }

    /// One traversal (RFC-0057 Decision 1). The head decides the source and
    /// the element: `&v` and `&mut v` through the container's own `as_slice`,
    /// which the index expressions settle the same way, an array by value,
    /// and a range of one integer width.
    fn check_for(
        &mut self,
        id: AstId,
        callee_id: AstId,
        binding: Astr,
        head: &acvus_ast::ForHead,
        body: &[acvus_ast::Stmt],
        span: Span,
    ) {
        let (kind, element) = match head {
            acvus_ast::ForHead::Range { lo, hi } => {
                (ForKind::Range, self.check_range(lo, hi, span))
            }
            acvus_ast::ForHead::Value(Expr::Borrow { mutable, place, .. }) => {
                let mutability = match mutable {
                    true => Mutability::Mut,
                    false => Mutability::Shared,
                };
                (
                    ForKind::Slice(mutability),
                    self.check_for_slice(callee_id, place, mutability, span),
                )
            }
            acvus_ast::ForHead::Value(array) => (ForKind::Array, self.check_for_array(array)),
        };
        self.for_kinds.insert(id, kind);
        // The statement's own id carries the element the source yields, which
        // is what the lowering builds the slice's type from, exactly as an
        // index expression's id carries its element (RFC-0047).
        self.record(id, element.clone());
        let binding_ty = match kind {
            ForKind::Slice(mutability) => {
                TyTerm::Ref(mutability, Box::new(TypeArg::uniform(element.clone())))
            }
            ForKind::Array | ForKind::Range => element.clone(),
        };
        let consumed = self
            .solver
            .freeze_ty(&element)
            .ok()
            .filter(|_| kind == ForKind::Array)
            .filter(|element| crate::validate::is_move_only(element) == Some(true));
        self.push_scope();
        self.loops.push(consumed);
        self.define_var(binding, binding_ty);
        for s in body {
            self.check_stmt(s);
        }
        self.loops.pop();
        self.pop_scope();
    }

    /// `lo..hi`: one integer width at both bounds, which is the element's
    /// type (RFC-0057 Decision 1).
    fn check_range(&mut self, lo: &Expr, hi: &Expr, span: Span) -> InferTy {
        let lo_ty = self.check_expr(lo);
        let hi_ty = self.check_expr(hi);
        if self.solver.unify(&lo_ty, &hi_ty).is_err() {
            self.error(
                MirErrorKind::UnificationFailure {
                    expected: self.type_as_written(&lo_ty),
                    got: self.type_as_written(&hi_ty),
                },
                span,
            );
            return Self::infer_error();
        }
        let resolved = self.solver.resolve_ty(&lo_ty);
        if !matches!(resolved, TyTerm::Int(_) | TyTerm::Var(_) | TyTerm::Error(_)) {
            self.error(
                MirErrorKind::ForSourceNotAdmitted {
                    ty: self.type_as_written(&lo_ty),
                },
                span,
            );
            return Self::infer_error();
        }
        lo_ty
    }

    /// `&v` or `&mut v`: the container's own `as_slice`, settled on the
    /// statement's `callee_id` where an index expression settles its own
    /// (RFC-0047 rule 6). The slice's element is what comes back; the
    /// binding is a reference to it.
    fn check_for_slice(
        &mut self,
        callee_id: AstId,
        place: &Expr,
        mutability: Mutability,
        span: Span,
    ) -> InferTy {
        let name = self.interner.intern(match mutability {
            Mutability::Shared => "as_slice",
            Mutability::Mut => "as_slice_mut",
        });
        let candidates: Vec<SignatureCandidate> = self
            .env
            .machine_set(name)
            .into_iter()
            .map(|(qref, scheme)| SignatureCandidate::Named {
                qref,
                scheme: scheme.clone(),
            })
            .collect();
        let first = self.receiver_arg(place, ReceiverMode::Lent(mutability), span);
        let container = self.solver.resolve_ty(&first.ty);
        if !takes_container(
            &candidates,
            &self.solver.resolve_ty(referent_of(&container)),
        ) {
            self.error(
                MirErrorKind::ForSourceNotAdmitted {
                    ty: self.type_as_written(&container),
                },
                span,
            );
            return Self::infer_error();
        }
        let slice = self.check_overloaded_call(
            candidates,
            CalleeSite {
                id: callee_id,
                span,
            },
            name,
            Some(first),
            &[],
            span,
        );
        match self.solver.shallow_resolve_ty(&slice) {
            TyTerm::Ref(_, inner) => match self.solver.shallow_resolve_ty(&inner.ty) {
                TyTerm::Slice(element) => *element,
                _ => Self::infer_error(),
            },
            _ => Self::infer_error(),
        }
    }

    /// An array by value: the loop takes its elements out, and the element
    /// is the array's own (RFC-0057 Decision 1). Every other value held by
    /// value is refused here, a container by value with the message that
    /// names the borrow it wanted.
    fn check_for_array(&mut self, array: &Expr) -> InferTy {
        let array_ty = self.check_expr(array);
        let resolved = self.solver.resolve_ty(&array_ty);
        match &resolved {
            TyTerm::Array(element, _) => (**element).clone(),
            TyTerm::UserDefined { .. } => {
                self.error(MirErrorKind::ForConsumesContainer, array.span());
                Self::infer_error()
            }
            _ => {
                self.error(
                    MirErrorKind::ForSourceNotAdmitted {
                        ty: self.type_as_written(&resolved),
                    },
                    array.span(),
                );
                Self::infer_error()
            }
        }
    }

    /// `break` and `continue` name the innermost loop, so outside every loop
    /// they name none (RFC-0057 Decision 4). A `break` out of a `for x in a`
    /// whose element owns something is refused: the elements the loop has
    /// not taken would have no release.
    fn check_loop_jump(&mut self, keyword: &'static str, span: Span) {
        let Some(innermost) = self.loops.last() else {
            self.error(MirErrorKind::OutsideLoop { keyword }, span);
            return;
        };
        if let Some(element) = innermost
            && keyword == "break"
        {
            self.error(
                MirErrorKind::ArrayLoopLeftEarly {
                    keyword,
                    element: element.clone(),
                },
                span,
            );
        }
    }

    /// `?` returns from the body, so it leaves every enclosing loop at once:
    /// a `for x in a` among them whose element owns something is refused for
    /// the reason `break` is.
    fn check_try_leaves_loops(&mut self, span: Span) {
        let Some(element) = self.loops.iter().rev().find_map(|loop_| loop_.clone()) else {
            return;
        };
        self.error(
            MirErrorKind::ArrayLoopLeftEarly {
                keyword: "?",
                element,
            },
            span,
        );
    }

    fn check_match_block(&mut self, mb: &MatchBlock) {
        // Body-less variable binding: define in current scope (no push/pop).
        if self.is_bodyless_var_binding(mb) {
            let source_ty = self.check_expr(&mb.source);
            self.check_pattern(
                &mb.arms[0].pattern,
                &source_ty,
                PatternSource::Expr(mb.source.id()),
                mb.arms[0].tag_span,
            );
            return;
        }

        let source_ty = self.check_expr(&mb.source);
        let resolved_source = self.solver.resolve_ty(&source_ty);

        for arm in &mb.arms {
            let match_ty = self.pattern_match_type(&arm.pattern, &resolved_source);
            // For patterns that destructure the source as a whole and may
            // contain nested variants (Variant, Tuple, List), pass the
            // unresolved source so unify can trace the Var chain and rebind
            // the merged type. This ensures variant sets from all arms are
            // accumulated into the same type variable.
            // Object patterns are NOT included because they go through the
            // iteration path (pattern_match_type extracts element types).
            let pattern_source = match &arm.pattern {
                Pattern::Variant { .. }
                | Pattern::Tuple { .. }
                | Pattern::List { .. }
                | Pattern::Binding { .. } => source_ty.clone(),
                _ => match_ty,
            };

            self.push_scope();
            self.check_pattern(
                &arm.pattern,
                &pattern_source,
                PatternSource::Expr(mb.source.id()),
                arm.tag_span,
            );
            self.check_nodes(&arm.body);
            self.pop_scope();
        }

        if let Some(catch_all) = &mb.catch_all {
            self.push_scope();
            self.check_nodes(&catch_all.body);
            self.pop_scope();
        }
    }

    fn is_bodyless_var_binding(&self, mb: &MatchBlock) -> bool {
        mb.arms.len() == 1
            && mb.arms[0].body.is_empty()
            && matches!(&mb.arms[0].pattern, Pattern::Binding { .. })
    }

    /// A pattern matches the source's type as it is (RFC-0024).
    fn pattern_match_type(&self, _pattern: &Pattern, source_ty: &InferTy) -> InferTy {
        source_ty.clone()
    }

    fn check_expr(&mut self, expr: &Expr) -> InferTy {
        match expr {
            // `&place` / `&mut place`: a reference to the place's storage.
            Expr::Borrow {
                id,
                mutable,
                place,
                span,
            } => {
                let ty = self.check_borrow(place, *mutable, *span);
                self.record_ret(*id, ty)
            }
            Expr::Literal { id, value, span } => {
                let ty = match value {
                    Literal::Int(n) => self.int_literal(*n, *span),
                    Literal::IntOf(n) => self.suffixed_int_literal(*n, *span),
                    Literal::Float(_) => TyTerm::Float,
                    Literal::Char(_) => TyTerm::Char,
                    Literal::Bytes(bytes) => {
                        TyTerm::Array(Box::new(TyTerm::U8), LenTerm::Known(bytes.len()))
                    }
                    Literal::String(_) => str_literal_ty(),
                    Literal::Bool(_) => TyTerm::Bool,
                    Literal::Unit => TyTerm::Unit,
                    Literal::List(elems) => {
                        if elems.is_empty() {
                            TyTerm::Array(Box::new(self.solver.fresh_ty_var()), LenTerm::Known(0))
                        } else {
                            let element = self.solver.fresh_ty_var();
                            let first_ty = self.literal_ty(&elems[0], *span);
                            self.solver
                                .unify(&first_ty, &element)
                                .expect("a fresh variable takes any type");
                            for elem in &elems[1..] {
                                let elem_ty = self.literal_ty(elem, *span);
                                self.convert_at(
                                    &elem_ty,
                                    &element,
                                    ConversionSite {
                                        id: *id,
                                        span: *span,
                                        report: ConversionReport::ListElement,
                                    },
                                );
                            }
                            TyTerm::Array(Box::new(element), LenTerm::Known(elems.len()))
                        }
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::ContextRef {
                id,
                name: qref,
                span,
            } => {
                let ty = self.resolve_context_type(*qref, *span);
                self.note_access(Effect::read(*qref), *span);
                self.record_ret(*id, ty)
            }

            Expr::Ident {
                id,
                name,
                ref_kind,
                span,
            } => {
                let ty = match ref_kind {
                    RefKind::ExternParam => {
                        let ty = match self.param_types.iter().find(|p| p.name == name.name) {
                            Some(param) => param.ty.clone(),
                            None => {
                                if let Some(ref mut state) = self.analysis {
                                    // In analysis mode, unknown $params are inferred.
                                    // Use declared type from Signature if available.
                                    let ty = if state.next_declared_param
                                        < state.declared_param_types.len()
                                    {
                                        let t =
                                            &state.declared_param_types[state.next_declared_param];
                                        let lifted = lift_ty(t);
                                        state.next_declared_param += 1;
                                        lifted
                                    } else {
                                        self.solver.fresh_ty_var()
                                    };
                                    self.param_types.push(ExternParam {
                                        name: name.name,
                                        ty: ty.clone(),
                                        first_read: Some(*span),
                                    });
                                    ty
                                } else {
                                    self.error(
                                        MirErrorKind::UndefinedVariable(format!(
                                            "${}",
                                            self.interner.resolve(name.name)
                                        )),
                                        *span,
                                    );
                                    Self::infer_error()
                                }
                            }
                        };
                        let capturing: Vec<usize> = (0..self.lambda_stack.len()).collect();
                        self.captured_by(name.name, &ty, &capturing);
                        ty
                    }
                    RefKind::Value => match self.lookup_var(name.name) {
                        Some(ty) => ty,
                        None => {
                            // Undefined local variable - always an error.
                            // Use $name for extern params, @name for context.
                            self.error(
                                MirErrorKind::UndefinedVariable(
                                    self.interner.resolve(name.name).to_string(),
                                ),
                                *span,
                            );
                            Self::infer_error()
                        }
                    },
                };
                self.record_ret(*id, ty)
            }

            Expr::BinaryOp {
                id,
                left,
                op,
                right,
                span,
            } => {
                let outer = std::mem::replace(
                    &mut self.demand,
                    match operand_stays_lent(*op) {
                        true => PlaceDemand::Borrow(Mutability::Shared),
                        false => PlaceDemand::Value,
                    },
                );
                let lt = self.check_expr(left);
                let rt = self.check_expr(right);
                self.demand = outer;
                let lt = self.read_operand_through(&lt, *op);
                let rt = self.read_operand_through(&rt, *op);

                // Early guard: if either operand is Error, suppress cascading errors.
                if Self::is_error(&lt) || Self::is_error(&rt) {
                    let ty = match op {
                        BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Mod
                        | BinOp::Xor
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::Shl
                        | BinOp::Shr => Self::infer_error(),
                        _ => TyTerm::Bool,
                    };
                    return self.record_ret(*id, ty);
                }

                // `+` on text is the concatenation, and `==` on text is the
                // byte comparison: both read their operands and neither needs
                // the two sides to be the same representation of text
                // (RFC-0062 Decision 3).
                let both_text =
                    is_text(&self.solver.resolve_ty(&lt)) && is_text(&self.solver.resolve_ty(&rt));
                let ty = match op {
                    BinOp::Add if both_text => TyTerm::String,
                    BinOp::Eq | BinOp::Neq if both_text => TyTerm::Bool,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        if !self.unify_operands(*op, &lt, &rt, *span) {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                            return self.record_ret(*id, Self::infer_error());
                        }
                        let rl = self.solver.resolve_ty(&lt);
                        match &rl {
                            TyTerm::Int(_) | TyTerm::Float | TyTerm::Var(_) => rl,
                            TyTerm::String if *op == BinOp::Add => TyTerm::String,
                            _ => {
                                self.binop_error(
                                    op_str(*op),
                                    rl,
                                    self.solver.resolve_ty(&rt),
                                    *span,
                                );
                                Self::infer_error()
                            }
                        }
                    }
                    BinOp::Eq | BinOp::Neq => {
                        if self.solver.unify(&lt, &rt).is_err() {
                            self.binop_error(op_str(*op), lt, rt, *span);
                            return self.record_ret(*id, TyTerm::Bool);
                        }
                        let operand = self.solver.resolve_ty(&lt);
                        if !operand.is_primitive()
                            && !matches!(operand, TyTerm::Var(_) | TyTerm::String)
                        {
                            self.check_operator_call(*id, op_str(*op), &operand, &[lt, rt], *span);
                        }
                        TyTerm::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        let lok = self.solver.unify(&lt, &TyTerm::Bool).is_ok();
                        let rok = self.solver.unify(&rt, &TyTerm::Bool).is_ok();
                        if !lok || !rok {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                        }
                        TyTerm::Bool
                    }
                    BinOp::Xor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
                        let ok = self.unify_operands(*op, &lt, &rt, *span)
                            && matches!(
                                self.solver.resolve_ty(&lt),
                                TyTerm::Int(_) | TyTerm::Var(_)
                            );
                        if !ok {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                            Self::infer_error()
                        } else {
                            self.solver.resolve_ty(&lt)
                        }
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        let ok = self.unify_operands(*op, &lt, &rt, *span)
                            && matches!(
                                self.solver.resolve_ty(&lt),
                                TyTerm::Int(_) | TyTerm::Float | TyTerm::Char | TyTerm::Var(_)
                            );
                        if !ok {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                        }
                        TyTerm::Bool
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::UnaryOp {
                id,
                op,
                operand,
                span,
            } => {
                let ot = self.check_expr(operand);
                let ot = self.solver.resolve_ty(&ot);

                // Early guard: if operand is Error, suppress cascading errors.
                if Self::is_error(&ot) {
                    let ty = match op {
                        acvus_ast::UnaryOp::Neg | acvus_ast::UnaryOp::Deref => Self::infer_error(),
                        acvus_ast::UnaryOp::Not => TyTerm::Bool,
                    };
                    return self.record_ret(*id, ty);
                }

                let ty = match op {
                    acvus_ast::UnaryOp::Deref => match &ot {
                        TyTerm::Var(_) => {
                            let inner = self.solver.fresh_ty_var();
                            let reference = TyTerm::Ref(
                                Mutability::Shared,
                                Box::new(TypeArg::new(self.solver.fresh_repr_var(), inner.clone())),
                            );
                            let _ = self.solver.unify(&ot, &reference);
                            inner
                        }
                        TyTerm::Ref(_, inner) => {
                            let inner = self.solver.resolve_ty(&inner.ty);
                            if inner.is_primitive()
                                || matches!(inner, TyTerm::Var(_) | TyTerm::Ref(..))
                            {
                                inner
                            } else {
                                let shown = self.type_as_written(&inner);
                                self.error(MirErrorKind::DerefOfNonPrimitive(shown), *span);
                                Self::infer_error()
                            }
                        }
                        _ => {
                            let shown = self.type_as_written(&ot);
                            self.error(MirErrorKind::DerefOfNonReference(shown), *span);
                            Self::infer_error()
                        }
                    },
                    acvus_ast::UnaryOp::Neg => match &ot {
                        TyTerm::Int(k) if k.signed() => ot.clone(),
                        TyTerm::Float => TyTerm::Float,
                        TyTerm::Var(v) => {
                            self.solver.require_signed(*v);
                            ot.clone()
                        }
                        _ => {
                            self.binop_error("-", ot, Self::infer_error(), *span);
                            Self::infer_error()
                        }
                    },
                    acvus_ast::UnaryOp::Not => {
                        match &ot {
                            TyTerm::Bool => {}
                            TyTerm::Var(_) => {
                                let _ = self.solver.unify(&ot, &TyTerm::Bool);
                            }
                            _ => self.binop_error("!", ot, Self::infer_error(), *span),
                        }
                        TyTerm::Bool
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::Index {
                id,
                callee_id,
                object,
                index,
                span,
            } => {
                let demand = self.demand;
                self.check_index(IndexSite {
                    id: *id,
                    callee_id: *callee_id,
                    object,
                    index,
                    span: *span,
                    demand,
                })
            }

            Expr::FieldAccess {
                id,
                object,
                field,
                span,
            } => {
                let through = self.demand.through_a_field();
                let outer = std::mem::replace(&mut self.demand, through);
                let ot_raw = self.check_expr(object);
                self.demand = outer;
                let ot = self.solver.shallow_resolve_ty(&ot_raw);
                if outer == PlaceDemand::Borrow(Mutability::Mut)
                    && let TyTerm::Ref(Mutability::Shared, _) = &ot
                {
                    let shown = self.type_as_written(&ot_raw);
                    self.error(MirErrorKind::StoreThroughSharedReference(shown), *span);
                    return self.record_ret(*id, Self::infer_error());
                }
                let field_key = *field;
                let field_str = || self.interner.resolve(*field).to_string();
                if let TyTerm::Ref(_, inner) = &ot {
                    let field_ty = match self.solver.shallow_resolve_ty(&inner.ty) {
                        TyTerm::Object(fields) if fields.contains_key(&field_key) => {
                            fields[&field_key].clone()
                        }
                        TyTerm::Error(_) => Self::infer_error(),
                        TyTerm::Object(_) | TyTerm::Var(_) => {
                            let fresh = self.solver.fresh_ty_var();
                            let partial = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter(
                                [(field_key, fresh.clone())],
                            )));
                            if self.solver.unify(&inner.ty, &partial).is_err() {
                                self.error(
                                    MirErrorKind::UndefinedField {
                                        object_ty: self.type_as_written(&ot),
                                        field: field_str(),
                                    },
                                    *span,
                                );
                                return self.record_ret(*id, Self::infer_error());
                            }
                            fresh
                        }
                        _ => {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: self.type_as_written(&ot),
                                    field: field_str(),
                                },
                                *span,
                            );
                            return self.record_ret(*id, Self::infer_error());
                        }
                    };
                    if self.demand == PlaceDemand::Value
                        && self.refuse_deref_of_non_primitive(&field_ty, *span)
                    {
                        return self.record_ret(*id, Self::infer_error());
                    }
                    return self.record_ret(*id, field_ty);
                }
                let ty = match &ot {
                    TyTerm::Error(_) => Self::infer_error(),
                    TyTerm::Object(fields) if fields.contains_key(&field_key) => {
                        fields[&field_key].clone()
                    }
                    TyTerm::Object(_) | TyTerm::Var(_) => {
                        let fresh = self.solver.fresh_ty_var();
                        let partial_obj = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter(
                            [(field_key, fresh.clone())],
                        )));
                        if self.solver.unify(&ot_raw, &partial_obj).is_err() {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: self.type_as_written(&ot),
                                    field: field_str(),
                                },
                                *span,
                            );
                        }
                        fresh
                    }
                    _ => {
                        self.error(
                            MirErrorKind::UndefinedField {
                                object_ty: self.type_as_written(&ot),
                                field: field_str(),
                            },
                            *span,
                        );
                        Self::infer_error()
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::FuncCall {
                id,
                func,
                args,
                span,
            } => {
                let ty = self.check_func_call(*id, func, args, None, *span);
                self.record_ret(*id, ty)
            }

            Expr::MethodCall {
                id,
                callee_id,
                receiver,
                name,
                args,
                span,
            } => {
                let ty = self.check_method_call(*callee_id, receiver, *name, args, *span);
                self.record_ret(*id, ty)
            }

            Expr::Pipe {
                id,
                left,
                right,
                span,
            } => {
                // Desugar: `a | f(b, c)` -> `f(a, b, c)`
                // `a | f` -> `f(a)`
                let pipe_left = Some(left.as_ref());
                let ty = match right.as_ref() {
                    Expr::FuncCall { func, args, .. } => {
                        self.check_func_call(*id, func, args, pipe_left, *span)
                    }
                    Expr::Ident {
                        ref_kind: RefKind::Value,
                        ..
                    } => self.check_func_call(*id, right, &[], pipe_left, *span),
                    _ => {
                        let first = FirstArg {
                            ty: self.check_expr(left),
                            site: ArgSite::of(left, *span),
                        };
                        let rt = self.check_expr(right);
                        self.check_callable(&rt, &[], Some(&first), *span)
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::Lambda { .. } => self.check_lambda(expr, None),

            Expr::Paren { id, inner, span: _ } => {
                let ty = self.check_expr(inner);
                self.record_ret(*id, ty)
            }

            Expr::Cast {
                id,
                expr,
                target,
                target_span,
                span: _,
            } => {
                let from = self.check_expr(expr);
                let Some(to) = CastTy::of_name(self.interner.resolve(*target)) else {
                    let name = self.interner.resolve(*target).to_string();
                    self.error(MirErrorKind::CastToUnknownType(name), *target_span);
                    return self.record_ret(*id, Self::infer_error());
                };
                self.casts.push(CastSite {
                    from,
                    to,
                    span: expr.span(),
                    target_span: *target_span,
                });
                self.record_ret(*id, InferTy::from(to))
            }

            Expr::Try { id, inner, span } => {
                self.check_try_leaves_loops(*span);
                let ty = self.check_try(inner, *span);
                if !Self::is_error(&ty) {
                    let ret = self.return_ty.clone().expect("check_try admitted a return");
                    self.try_sites.insert(*id, ret);
                }
                self.record_ret(*id, ty)
            }

            Expr::List {
                id,
                head,
                rest,
                tail,
                span,
            } => {
                let all_elems: Vec<_> = head.iter().chain(tail.iter()).collect();
                if let Some(rest_span) = rest {
                    self.error(MirErrorKind::RestInArrayLiteral, *rest_span);
                }
                if all_elems.is_empty() {
                    let ty = TyTerm::Array(Box::new(self.solver.fresh_ty_var()), LenTerm::Known(0));
                    return self.record_ret(*id, ty);
                }

                let element = self.solver.fresh_ty_var();
                let first_ty = self.check_expr(all_elems[0]);
                self.reject_reference_in_data(&first_ty, all_elems[0].span(), DataShape::Aggregate);
                self.solver
                    .unify(&first_ty, &element)
                    .expect("a fresh variable takes any type");

                for elem in all_elems.iter().skip(1) {
                    let et = self.check_expr(elem);
                    self.convert_at(
                        &et,
                        &element,
                        ConversionSite {
                            id: elem.id(),
                            span: *span,
                            report: ConversionReport::ListElement,
                        },
                    );
                }

                let ty = TyTerm::Array(Box::new(element), LenTerm::Known(all_elems.len()));
                self.record_ret(*id, ty)
            }

            Expr::Object {
                id,
                fields,
                span: _,
            } => {
                let mut field_types = FxHashMap::default();
                for ObjectExprField { key, value, .. } in fields {
                    let ft = self.check_expr(value);
                    self.reject_reference_in_data(&ft, value.span(), DataShape::Aggregate);
                    field_types.insert(*key, ft);
                }
                let ty = TyTerm::Object(ObjectTy::written(field_types));
                self.record_ret(*id, ty)
            }

            Expr::Tuple {
                id,
                elements,
                span: _,
            } => {
                let elem_types: Vec<InferTy> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => {
                            let et = self.check_expr(e);
                            self.reject_reference_in_data(&et, e.span(), DataShape::Aggregate);
                            et
                        }
                        TupleElem::Wildcard(_) => self.solver.fresh_ty_var(),
                    })
                    .collect();
                let ty = TyTerm::Tuple(elem_types);
                self.record_ret(*id, ty)
            }

            Expr::Group {
                id,
                elements,
                span: _,
            } => {
                // Group is only valid as lambda param list (handled by parser).
                let Some(last) = elements.last() else {
                    self.record(*id, TyTerm::Unit);
                    return TyTerm::Unit;
                };
                for e in &elements[..elements.len() - 1] {
                    self.check_expr(e);
                }
                let ty = self.check_expr(last);
                self.record_ret(*id, ty)
            }

            Expr::Variant {
                id,
                enum_name: ast_enum_name,
                tag,
                payload,
                span,
            } => {
                // Try builtin (Option) first.
                if let Some((_enum_name, type_params, variant_payload)) =
                    self.resolve_builtin_variant(ast_enum_name, *tag)
                {
                    match &variant_payload {
                        VariantPayload::TypeParam(idx) => {
                            let Some(inner_expr) = payload else {
                                self.error(
                                    MirErrorKind::UnificationFailure {
                                        expected: Ty::error(),
                                        got: Ty::Unit,
                                    },
                                    *span,
                                );
                                return Self::infer_error();
                            };
                            let inner_ty = self.check_expr(inner_expr);
                            self.reject_reference_in_data(
                                &inner_ty,
                                inner_expr.span(),
                                DataShape::Payload,
                            );
                            if self.solver.unify(&type_params[*idx], &inner_ty).is_err() {
                                let resolved_tp = self.solver.resolve_ty(&type_params[*idx]);
                                let resolved_inner = self.solver.resolve_ty(&inner_ty);
                                self.error(
                                    MirErrorKind::UnificationFailure {
                                        expected: self.type_as_written(&resolved_tp),
                                        got: self.type_as_written(&resolved_inner),
                                    },
                                    *span,
                                );
                            }
                        }
                        VariantPayload::None => {}
                    }
                    let ty = self.builtin_enum_ty(&type_params);
                    return self.record_ret(*id, ty);
                }

                // Structural enum: requires qualified name (A::B).
                let Some(enum_name) = ast_enum_name else {
                    self.error(
                        MirErrorKind::UndefinedFunction(format!(
                            "unknown variant: {}",
                            self.interner.resolve(*tag)
                        )),
                        *span,
                    );
                    return Self::infer_error();
                };

                let payload_ty = match payload {
                    Some(expr) => {
                        let ty = self.check_expr(expr);
                        Some(Box::new(ty))
                    }
                    None => None,
                };

                let mut variants = FxHashMap::default();
                variants.insert(*tag, payload_ty);
                let ty = TyTerm::Enum {
                    name: *enum_name,
                    variants,
                };
                self.record_ret(*id, ty)
            }

            Expr::Block {
                id,
                stmts,
                tail,
                span: _,
            } => {
                self.push_scope();
                for stmt in stmts {
                    self.check_stmt(stmt);
                }
                let ty = self.check_expr(tail);
                self.pop_scope();
                self.record_ret(*id, ty)
            }

            Expr::If {
                id,
                cond,
                then_body,
                then_tail,
                else_branch,
                span,
            } => {
                let cond_ty = self.check_expr(cond);
                if self.solver.unify(&cond_ty, &TyTerm::Bool).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: Ty::Bool,
                            got: self.type_as_written(&cond_ty),
                        },
                        *span,
                    );
                }
                self.push_scope();
                for s in then_body {
                    self.check_stmt(s);
                }
                let then = match then_tail {
                    Some(tail) => Branch {
                        ty: self.check_expr(tail),
                        value: Some(tail.id()),
                    },
                    None => Branch {
                        ty: TyTerm::Unit,
                        value: None,
                    },
                };
                self.pop_scope();
                let result_ty = match else_branch {
                    Some(eb) => {
                        let else_ = self.check_else_branch(eb);
                        self.join_branches(&then, &else_, *span)
                    }
                    None => then.ty,
                };
                self.record_ret(*id, result_ty)
            }

            Expr::Match {
                id,
                scrutinee,
                arms,
                span,
            } => {
                let ty = self.check_match(scrutinee, arms, *span);
                self.record_ret(*id, ty)
            }

            Expr::IfLet {
                id,
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                span,
            } => {
                let source_ty = self.check_expr(source);
                let resolved = self.solver.resolve_ty(&source_ty);
                self.push_scope();
                self.check_pattern(pattern, &resolved, PatternSource::Expr(source.id()), *span);
                for s in then_body {
                    self.check_stmt(s);
                }
                let then = match then_tail {
                    Some(tail) => Branch {
                        ty: self.check_expr(tail),
                        value: Some(tail.id()),
                    },
                    None => Branch {
                        ty: TyTerm::Unit,
                        value: None,
                    },
                };
                self.pop_scope();
                let result_ty = match else_branch {
                    Some(eb) => {
                        let else_ = self.check_else_branch(eb);
                        self.join_branches(&then, &else_, *span)
                    }
                    None => then.ty,
                };
                self.record_ret(*id, result_ty)
            }
        }
    }

    /// A `match` is one dispatch (RFC-0051): the scrutinee's type is
    /// settled first, every arm's pattern is checked *against* it, and the
    /// arms' types join into one. Exhaustiveness is not decided here --
    /// the union is still open while checking -- it is decided in
    /// `validate`, where the body's definitions are visible.
    fn check_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[acvus_ast::MatchExprArm],
        span: Span,
    ) -> InferTy {
        let source_ty = self.check_expr(scrutinee);
        let resolved = self.solver.resolve_ty(&source_ty);
        let mut joined: Option<Branch> = None;
        for arm in arms {
            self.check_arm_is_reachable(&arm.pattern, &resolved, arm.span);
            self.push_scope();
            self.check_pattern(
                &arm.pattern,
                &resolved,
                PatternSource::Expr(scrutinee.id()),
                arm.span,
            );
            for s in &arm.body {
                self.check_stmt(s);
            }
            let branch = match &arm.tail {
                Some(tail) => Branch {
                    ty: self.check_expr(tail),
                    value: Some(tail.id()),
                },
                None => Branch {
                    ty: TyTerm::Unit,
                    value: None,
                },
            };
            self.pop_scope();
            joined = Some(match joined {
                None => branch,
                Some(acc) => Branch {
                    ty: self.join_branches(&acc, &branch, span),
                    value: None,
                },
            });
        }
        if !crate::lower::Dispatch::is_decidable(arms) {
            self.error(MirErrorKind::MatchIsNotADispatch, span);
        }
        joined.map_or(TyTerm::Unit, |b| b.ty)
    }

    /// RFC-0051 §2: an arm contributes no variant. A pattern naming a tag
    /// the scrutinee's enum does not carry can never be taken, and the
    /// language has no warning axis, so it is refused. The question is
    /// asked only where the answer is written down: a scrutinee whose head
    /// is still open says nothing, and no arm is refused on it.
    fn check_arm_is_reachable(&mut self, pattern: &Pattern, source_ty: &InferTy, span: Span) {
        let Pattern::Variant { tag, payload, .. } = pattern else {
            return;
        };
        let TyTerm::Enum { name, variants } = self.solver.shallow_resolve_ty(source_ty) else {
            return;
        };
        if variants.contains_key(tag) {
            return;
        }
        let written = match payload {
            Some(_) => format!(
                "{}::{}(_)",
                self.interner.resolve(name),
                self.interner.resolve(*tag)
            ),
            None => format!(
                "{}::{}",
                self.interner.resolve(name),
                self.interner.resolve(*tag)
            ),
        };
        let scrutinee_ty = self.type_as_written(source_ty);
        self.error(
            MirErrorKind::UnreachablePattern {
                pattern: written,
                scrutinee_ty,
            },
            span,
        );
    }

    fn check_else_branch(&mut self, eb: &acvus_ast::ElseBranch) -> Branch {
        match eb {
            acvus_ast::ElseBranch::ElseIf(expr) => Branch {
                ty: self.check_expr(expr),
                value: Some(expr.id()),
            },
            acvus_ast::ElseBranch::Else { body, tail, .. } => {
                self.push_scope();
                for s in body {
                    self.check_stmt(s);
                }
                let branch = match tail {
                    Some(tail) => Branch {
                        ty: self.check_expr(tail),
                        value: Some(tail.id()),
                    },
                    None => Branch {
                        ty: TyTerm::Unit,
                        value: None,
                    },
                };
                self.pop_scope();
                branch
            }
        }
    }

    fn check_func_call(
        &mut self,
        call_id: AstId,
        func: &Expr,
        args: &[Expr],
        pipe_left: Option<&Expr>,
        call_span: Span,
    ) -> InferTy {
        // Collect argument types, prepending pipe_left if present.
        let first = pipe_left.map(|e| FirstArg {
            ty: self.check_expr(e),
            site: ArgSite::of(e, call_span),
        });

        // Try to resolve as a named function (builtin or extern).
        let Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } = func
        else {
            // Not a simple name - evaluate the function expression.
            let ft = self.check_expr(func);
            let resolved = self.solver.shallow_resolve_ty(&ft);
            return self.check_callable(&resolved, args, first.as_ref(), call_span);
        };

        let name_str = self.interner.resolve(name.name);
        let candidates = self.signature_set(*name);
        match candidates.as_slice() {
            [] => {}
            [SignatureCandidate::Named { qref, scheme }] => {
                let scheme = scheme.clone();
                return self.check_resolved_call(
                    *qref,
                    &scheme,
                    CalleeSite {
                        id: func.id(),
                        span: func.span(),
                    },
                    name_str,
                    first,
                    args,
                    call_span,
                );
            }
            [SignatureCandidate::Local { ty }] => {
                let ty = ty.clone();
                return self.check_local_call(func.id(), &ty, first.as_ref(), args, call_span);
            }
            _ => {
                return self.check_overloaded_call(
                    candidates,
                    CalleeSite {
                        id: func.id(),
                        span: func.span(),
                    },
                    name.name,
                    first,
                    args,
                    call_span,
                );
            }
        }
        if let Some(ns) = name.namespace {
            return self.check_structural_variant(call_id, ns, name.name, args, call_span);
        }
        self.error(
            MirErrorKind::UndefinedFunction(name_str.to_string()),
            call_span,
        );
        Self::infer_error()
    }

    /// A call lends the closure a `&Fn` names instead of moving it
    /// (RFC-0018), which is the reborrow of RFC-0029. A referent still open
    /// is lent the same way, and the call is what decides it is a function.
    fn lent_fn(&mut self, ty: &InferTy) -> InferTy {
        let resolved = self.solver.shallow_resolve_ty(ty);
        let TyTerm::Ref(_, inner) = &resolved else {
            return resolved;
        };
        match self.solver.shallow_resolve_ty(&inner.ty) {
            lent @ (TyTerm::Fn { .. } | TyTerm::Var(_)) => lent,
            _ => resolved,
        }
    }

    fn check_callable(
        &mut self,
        func_ty: &InferTy,
        args: &[Expr],
        first: Option<&FirstArg>,
        call_span: Span,
    ) -> InferTy {
        let func_ty = &self.lent_fn(func_ty);
        match func_ty {
            TyTerm::Fn { .. } | TyTerm::Var(_) => {}
            TyTerm::Error(_) => {
                for a in args {
                    self.check_expr(a);
                }
                return Self::infer_error();
            }
            _ => {
                self.error(
                    MirErrorKind::UndefinedFunction("<not callable>".to_string()),
                    call_span,
                );
                return Self::infer_error();
            }
        }

        let arg_types = self.check_args_in_order(func_ty, first, args, call_span);

        match func_ty {
            TyTerm::Fn {
                params,
                ret,
                effect,
                ..
            } => {
                if !self.check_arity("<closure>", arg_types.len(), params.len(), call_span) {
                    return Self::infer_error();
                }
                let effect = effect.clone();
                self.note_call_effect(&effect, call_span);
                (**ret).clone()
            }
            TyTerm::Var(_) => {
                let ret = self.solver.fresh_ty_var();
                let effect = self.solver.fresh_effect_var();
                let dummy = self.interner.intern("_");
                let fn_ty = TyTerm::Fn {
                    params: arg_types
                        .into_iter()
                        .map(|ty| ParamTerm::new(dummy, ty))
                        .collect(),
                    ret: Box::new(ret.clone()),
                    captures: vec![],
                    effect: effect.clone(),
                };
                self.note_call_effect(&effect, call_span);
                if self.solver.unify(func_ty, &fn_ty).is_err() {
                    self.error(
                        MirErrorKind::UndefinedFunction("<expr>".to_string()),
                        call_span,
                    );
                    return Self::infer_error();
                }
                ret
            }
            _ => unreachable!(),
        }
    }

    /// RFC-0024. A reference scrutinee is read through: the pattern is
    /// checked against what it names, and every name under it binds a
    /// reference. A scrutinee whose head is still a variable settles the
    /// same question later, through a decision; a part of a pattern is
    /// read in the mode its whole was.
    fn check_pattern(
        &mut self,
        pattern: &Pattern,
        source_ty: &InferTy,
        source: PatternSource,
        span: Span,
    ) {
        match self.solver.match_mode(source_ty) {
            MatchOutcome::Reads(reads) if reads.mode == MatchMode::Through => {
                let outer = std::mem::replace(&mut self.pattern_mode, PatternMode::Through);
                self.check_pattern_inner(pattern, &reads.names, source, span);
                self.pattern_mode = outer;
            }
            MatchOutcome::Reads(_) => self.check_pattern_inner(pattern, source_ty, source, span),
            MatchOutcome::HeadOpen => match source {
                PatternSource::Expr(_) => {
                    self.check_pattern_deferred(pattern, source_ty, source, span)
                }
                PatternSource::Member => self.check_pattern_inner(pattern, source_ty, source, span),
            },
        }
    }

    /// The scrutinee's head is still a variable, so how the pattern reads
    /// it cannot be read off it: the pattern is checked against a referent
    /// of its own, and the solver joins that referent with the scrutinee
    /// -- and every name the pattern binds with the part it stands for --
    /// once the head resolves, or on the least element if nothing ever
    /// resolves it.
    fn check_pattern_deferred(
        &mut self,
        pattern: &Pattern,
        source_ty: &InferTy,
        source: PatternSource,
        span: Span,
    ) {
        let referent = self.solver.fresh_ty_var();
        let outer_mode = std::mem::replace(&mut self.pattern_mode, PatternMode::Deferred);
        let outer_bindings = std::mem::take(&mut self.deferred_bindings);
        let outer_context_binds = std::mem::take(&mut self.deferred_context_binds);
        self.check_pattern_inner(pattern, &referent, source, span);
        self.pattern_mode = outer_mode;
        let bindings = std::mem::replace(&mut self.deferred_bindings, outer_bindings);
        let context_binds =
            std::mem::replace(&mut self.deferred_context_binds, outer_context_binds);
        let decision = self.solver.decide(Decision::Match {
            scrutinee: source_ty.clone(),
            referent,
            bindings,
        });
        self.decision_sites.insert(decision, span);
        self.context_binds_under_open_head.extend(
            context_binds
                .into_iter()
                .map(|span| DeferredContextBind { decision, span }),
        );
    }

    /// A context holds data, and data holds no reference (RFC-0014): a
    /// context bound by a pattern that read through a reference is
    /// refused, whether the head said so at once or only once it settled.
    fn check_context_binds_under_open_head(&mut self) {
        let binds = std::mem::take(&mut self.context_binds_under_open_head);
        for DeferredContextBind { decision, span } in binds {
            if let Some(Answer::Match(MatchMode::Through)) = self.solver.answer(decision) {
                self.error(MirErrorKind::ReferenceInData(DataShape::Aggregate), span);
            }
        }
    }

    fn check_pattern_inner(
        &mut self,
        pattern: &Pattern,
        source_ty: &InferTy,
        source: PatternSource,
        span: Span,
    ) {
        let source_resolved = self.solver.resolve_ty(source_ty);
        match pattern {
            Pattern::ContextBind { name: qref, .. } => {
                match self.pattern_mode {
                    PatternMode::Through => {
                        self.error(MirErrorKind::ReferenceInData(DataShape::Aggregate), span);
                        return;
                    }
                    PatternMode::Deferred => self.deferred_context_binds.push(span),
                    PatternMode::Value => {}
                }
                self.note_context_use(*qref, span);
                self.note_access(Effect::write(*qref), span);
                let ctx_ty = self
                    .env
                    .contexts
                    .get(qref)
                    .cloned()
                    .unwrap_or_else(|| self.solver.fresh_ty_var());
                let joined = match source {
                    PatternSource::Expr(id) => {
                        let site = ConversionSite {
                            id,
                            span,
                            report: ConversionReport::Pattern,
                        };
                        self.flow(source_ty, &ctx_ty, site)
                    }
                    PatternSource::Member => self.solver.unify(source_ty, &ctx_ty),
                };
                if joined.is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: self.type_as_written(&ctx_ty),
                            source_ty: self.type_as_written(&source_resolved),
                        },
                        span,
                    );
                }
            }
            Pattern::Binding { name, ref_kind, .. } => match ref_kind {
                RefKind::ExternParam => {
                    self.error(
                        MirErrorKind::ExternParamAssign(self.interner.resolve(*name).to_string()),
                        span,
                    );
                }
                RefKind::Value => {
                    let ty = match self.pattern_mode {
                        PatternMode::Through => TyTerm::Ref(
                            Mutability::Shared,
                            Box::new(TypeArg::uniform(source_ty.clone())),
                        ),
                        PatternMode::Value => source_ty.clone(),
                        PatternMode::Deferred => {
                            let binding = self.solver.fresh_ty_var();
                            self.deferred_bindings.push(MatchBinding {
                                binding: binding.clone(),
                                part: source_ty.clone(),
                            });
                            binding
                        }
                    };
                    self.define_var(*name, ty);
                }
            },

            // `_` reads the position and asks nothing of it.
            Pattern::Wildcard { .. } => {}

            // Obligation across artifacts: what makes this admission sound
            // is that a string literal pattern never becomes a value.
            // `lower.rs` emits `InstKind::TestLiteral`, which holds the text
            // as an immediate and compares the scrutinee's bytes, so the two
            // representations of one text are one comparison (RFC-0062
            // Decision 2).
            Pattern::Literal {
                value: Literal::String(_),
                ..
            } if holds_text(&source_resolved) => {}

            Pattern::Literal { value, .. } => {
                let pat_ty = self.literal_ty(value, span);
                if self.solver.unify_pattern(source_ty, &pat_ty).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: self.type_as_written(&pat_ty),
                            source_ty: self.type_as_written(&source_resolved),
                        },
                        span,
                    );
                }
            }

            Pattern::List {
                head, rest, tail, ..
            } => {
                let shallow = self.solver.shallow_resolve_ty(source_ty);
                let (elem_ty, len) = match shallow {
                    TyTerm::Array(ref inner, len) => ((**inner).clone(), len),
                    _ => {
                        let var = self.solver.fresh_ty_var();
                        let len = self.solver.fresh_len_var();
                        let array_ty = TyTerm::Array(Box::new(var.clone()), len);
                        if self.solver.unify_pattern(source_ty, &array_ty).is_err() {
                            self.error(
                                MirErrorKind::PatternTypeMismatch {
                                    pattern_ty: self.type_as_written(&array_ty),
                                    source_ty: self.type_as_written(&source_resolved),
                                },
                                span,
                            );
                            return;
                        }
                        (var, len)
                    }
                };
                let pattern_min = head.len() + tail.len();
                let exact = rest.is_none();
                match self.solver.resolve_len(&len) {
                    LenTerm::Known(got) => {
                        let fits = if exact {
                            got == pattern_min
                        } else {
                            got >= pattern_min
                        };
                        if !fits {
                            self.error(
                                MirErrorKind::ArrayLengthMismatch {
                                    pattern_min,
                                    exact,
                                    got,
                                },
                                span,
                            );
                        }
                    }
                    LenTerm::Var(_) => self.error(MirErrorKind::ArrayLengthUnknown, span),
                }
                for p in head.iter().chain(tail.iter()) {
                    self.check_pattern(p, &elem_ty, PatternSource::Member, span);
                }
            }

            Pattern::Object { fields, .. } => {
                // If source is already a concrete Object, match fields directly (open/subset).
                // Otherwise, build an Object from pattern fields and unify to infer the type.
                let obj_fields = if let TyTerm::Object(object) = &source_resolved {
                    object.iter().map(|(k, v)| (*k, v.clone())).collect()
                } else {
                    let field_vars: FxHashMap<Astr, InferTy> = fields
                        .iter()
                        .map(|f| (f.key, self.solver.fresh_ty_var()))
                        .collect();
                    let obj_ty = TyTerm::Object(ObjectTy::at_least(field_vars.clone()));
                    if self.solver.unify_pattern(source_ty, &obj_ty).is_err() {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: self.type_as_written(&obj_ty),
                                source_ty: self.type_as_written(&source_resolved),
                            },
                            span,
                        );
                        return;
                    }
                    field_vars
                };
                for ObjectPatternField { key, pattern, .. } in fields {
                    let Some(field_ty) = obj_fields.get(key) else {
                        self.error(
                            MirErrorKind::UndefinedField {
                                object_ty: self.type_as_written(&source_resolved),
                                field: self.interner.resolve(*key).to_string(),
                            },
                            span,
                        );
                        continue;
                    };
                    let resolved = self.solver.resolve_ty(field_ty);
                    self.check_pattern(pattern, &resolved, PatternSource::Member, span);
                }
            }

            Pattern::Tuple { elements, .. } => {
                // Reuse existing element Vars when source already resolves to a
                // Tuple. This preserves the Var chain so nested Variant patterns
                // can accumulate merged variant sets across match arms via
                // find_leaf_var.
                let shallow = self.solver.shallow_resolve_ty(source_ty);
                let elem_tys = match shallow {
                    TyTerm::Tuple(ref existing) if existing.len() == elements.len() => {
                        existing.clone()
                    }
                    _ => {
                        let vars: Vec<InferTy> = elements
                            .iter()
                            .map(|_| self.solver.fresh_ty_var())
                            .collect();
                        let tuple_ty = TyTerm::Tuple(vars.clone());
                        if self.solver.unify_pattern(source_ty, &tuple_ty).is_err() {
                            self.error(
                                MirErrorKind::PatternTypeMismatch {
                                    pattern_ty: self.type_as_written(&tuple_ty),
                                    source_ty: self.type_as_written(&source_resolved),
                                },
                                span,
                            );
                            return;
                        }
                        vars
                    }
                };
                for (i, elem) in elements.iter().enumerate() {
                    let TuplePatternElem::Pattern(pat) = elem else {
                        continue; // Wildcard: no binding, skip.
                    };
                    self.check_pattern(pat, &elem_tys[i], PatternSource::Member, span);
                }
            }

            Pattern::Variant {
                enum_name: ast_enum_name,
                tag,
                payload,
                ..
            } => {
                // Try builtin (Option) first.
                if let Some((_enum_name, type_params, variant_payload)) =
                    self.resolve_builtin_variant(ast_enum_name, *tag)
                {
                    let enum_ty = self.builtin_enum_ty(&type_params);
                    if self.solver.unify_pattern(source_ty, &enum_ty).is_err() {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: self.type_as_written(&enum_ty),
                                source_ty: self.type_as_written(&source_resolved),
                            },
                            span,
                        );
                        return;
                    }

                    if let VariantPayload::TypeParam(idx) = &variant_payload {
                        let resolved_inner = self.solver.resolve_ty(&type_params[*idx]);
                        if let Some(inner_pat) = payload {
                            self.check_pattern(
                                inner_pat,
                                &resolved_inner,
                                PatternSource::Member,
                                span,
                            );
                        }
                    }
                    return;
                }

                // Structural enum: requires qualified name.
                let Some(enum_name) = ast_enum_name else {
                    self.error(
                        MirErrorKind::UndefinedFunction(format!(
                            "unknown variant: {}",
                            self.interner.resolve(*tag)
                        )),
                        span,
                    );
                    return;
                };

                // Build Ty::Enum with this single variant.
                let payload_ty = if payload.is_some() {
                    Some(Box::new(self.solver.fresh_ty_var()))
                } else {
                    None
                };
                let mut variants = FxHashMap::default();
                variants.insert(*tag, payload_ty.clone());
                let enum_ty = TyTerm::Enum {
                    name: *enum_name,
                    variants,
                };
                // Unify against the original (unresolved) source_ty so that
                // find_leaf_var can trace the Var chain and rebind the merged type.
                if self.solver.unify_pattern(source_ty, &enum_ty).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: self.type_as_written(&enum_ty),
                            source_ty: self.type_as_written(&source_resolved),
                        },
                        span,
                    );
                    return;
                }

                // Bind payload pattern if present.
                if let Some(inner_pat) = payload {
                    let inner_ty = payload_ty
                        .map(|ty| self.solver.resolve_ty(&ty))
                        .unwrap_or_else(Self::infer_error);
                    self.check_pattern(inner_pat, &inner_ty, PatternSource::Member, span);
                }
            }
        }
    }

    /// Try to resolve a variant tag as a builtin enum (Option).
    /// Returns None if the tag is not a builtin variant.
    fn resolve_builtin_variant(
        &mut self,
        ast_enum_name: &Option<Astr>,
        tag: Astr,
    ) -> Option<(Astr, Vec<InferTy>, VariantPayload)> {
        let (name, payload, arity) = match self.interner.resolve(tag) {
            "Some" => ("Option", VariantPayload::TypeParam(0), 1),
            "None" => ("Option", VariantPayload::None, 1),
            "Ok" => ("Result", VariantPayload::TypeParam(0), 2),
            "Err" => ("Result", VariantPayload::TypeParam(1), 2),
            _ => return None,
        };
        let name = self.interner.intern(name);
        if let Some(ename) = ast_enum_name
            && *ename != name
        {
            return None;
        }
        let type_params = (0..arity).map(|_| self.solver.fresh_ty_var()).collect();
        Some((name, type_params, payload))
    }

    /// The type of an `if` with both branches: each flows into one fresh
    /// variable, so a branch typed `!` (a call that traps) leaves the other
    /// branch's type standing (RFC-0038). The `then` branch fills the
    /// variable; the `else` branch meets it, and a conversion there is a
    /// cast of the `else` value.
    fn join_branches(&mut self, then: &Branch, else_: &Branch, span: Span) -> InferTy {
        let joined = self.solver.fresh_ty_var();
        let then_ok = self.solver.unify(&then.ty, &joined).is_ok();
        let else_ok = then_ok
            && match else_.value {
                Some(id) => {
                    let site = ConversionSite {
                        id,
                        span,
                        report: ConversionReport::Value,
                    };
                    self.flow(&else_.ty, &joined, site).is_ok()
                }
                None => self.solver.unify(&else_.ty, &joined).is_ok(),
            };
        if !else_ok {
            self.branch_mismatches.push(BranchMismatch {
                then: then.ty.clone(),
                else_: else_.ty.clone(),
                span,
            });
        }
        self.solver.resolve_ty(&joined)
    }

    /// `inner?` (RFC-0038): the payload of an `Ok` or a `Some`, while the
    /// `Err` or `None` leaves the function early, so the function returns a
    /// `Result` whose error type unifies with the operand's, or an `Option`.
    fn check_try(&mut self, inner: &Expr, span: Span) -> InferTy {
        let operand = self.check_expr(inner);
        let mut operand = self.solver.resolve_ty(&operand);
        let Some(return_ty) = self.return_ty.clone() else {
            self.error(MirErrorKind::TryOutsideFunction, span);
            return Self::infer_error();
        };
        if let TyTerm::Var(_) = operand {
            operand = self.assume_result_operand(&operand);
        }
        let (payload, leaves) = match &operand {
            TyTerm::Result(ok, err) => (
                (**ok).clone(),
                TyTerm::Result(Box::new(self.solver.fresh_ty_var()), err.clone()),
            ),
            TyTerm::Option(some) => (
                (**some).clone(),
                TyTerm::Option(Box::new(self.solver.fresh_ty_var())),
            ),
            TyTerm::Error(_) => return Self::infer_error(),
            other => {
                let shown = self.type_as_written(other);
                self.error(MirErrorKind::TryOnNonResult(shown), span);
                return Self::infer_error();
            }
        };
        if self.solver.unify(&leaves, &return_ty).is_err() {
            let expected = self.solver.resolve_ty(&return_ty);
            self.error(
                MirErrorKind::TryReturnMismatch {
                    leaves: self.type_as_written(&leaves),
                    returns: self.type_as_written(&expected),
                },
                span,
            );
            return Self::infer_error();
        }
        payload
    }

    /// A `?` operand whose type nothing has fixed yet — a lambda parameter
    /// checked before any call — is a `Result` (RFC-0038); an `Option`
    /// operand must be known where `?` is applied.
    fn assume_result_operand(&mut self, operand: &InferTy) -> InferTy {
        let assumed = TyTerm::Result(
            Box::new(self.solver.fresh_ty_var()),
            Box::new(self.solver.fresh_ty_var()),
        );
        if self.solver.unify(operand, &assumed).is_err() {
            return Self::infer_error();
        }
        self.solver.resolve_ty(&assumed)
    }

    /// The return type each `?` leaves through, frozen for the lowering.
    fn frozen_try_returns(&self) -> FxHashMap<AstId, Ty> {
        self.try_sites
            .iter()
            .map(|(id, ty)| {
                let resolved = self.solver.resolve_ty(ty);
                (
                    *id,
                    self.solver
                        .close_ty(&resolved)
                        .unwrap_or_else(|_| Ty::error()),
                )
            })
            .collect()
    }

    /// `Option<T>` for one type parameter, `Result<T, E>` for two: the
    /// builtin enums `Some`/`None` and `Ok`/`Err` construct.
    fn builtin_enum_ty(&self, type_params: &[InferTy]) -> InferTy {
        match type_params {
            [inner] => TyTerm::Option(Box::new(inner.clone())),
            [ok, err] => TyTerm::Result(Box::new(ok.clone()), Box::new(err.clone())),
            _ => unreachable!("a builtin enum has one or two type parameters"),
        }
    }

    fn literal_ty(&mut self, lit: &Literal, span: Span) -> InferTy {
        match lit {
            Literal::Int(n) => self.int_literal(*n, span),
            Literal::IntOf(n) => self.suffixed_int_literal(*n, span),
            Literal::Float(_) => TyTerm::Float,
            Literal::Char(_) => TyTerm::Char,
            Literal::Bytes(bytes) => {
                TyTerm::Array(Box::new(TyTerm::U8), LenTerm::Known(bytes.len()))
            }
            Literal::String(_) => str_literal_ty(),
            Literal::Bool(_) => TyTerm::Bool,
            Literal::Unit => TyTerm::Unit,
            Literal::List(elems) => match elems.first() {
                Some(first) => TyTerm::Array(
                    Box::new(self.literal_ty(first, span)),
                    LenTerm::Known(elems.len()),
                ),
                None => TyTerm::Array(Box::new(self.solver.fresh_ty_var()), LenTerm::Known(0)),
            },
        }
    }
}

/// Whether this type is one of the two representations of text: a `String`
/// or the `str` a `&str` names. An operand of `+` or `==` has already been
/// read through its reference, so a reference is not one of them.
fn is_text(ty: &InferTy) -> bool {
    matches!(ty, TyTerm::String | TyTerm::Str)
}

/// Whether this type is one of the two representations of text, or a
/// reference to one: `String`, `&String`, `str`, `&str`.
fn holds_text(ty: &InferTy) -> bool {
    match ty {
        TyTerm::String | TyTerm::Str => true,
        TyTerm::Ref(_, inner) => holds_text(&inner.ty),
        _ => false,
    }
}

fn str_literal_ty() -> InferTy {
    TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(TyTerm::Str)))
}

fn op_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Eq => "==",
        BinOp::Neq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Lte => "<=",
        BinOp::Gte => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::Xor => "^",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Mod => "%",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::QualifiedRef;
    use crate::ty::TypeRegistry;

    /// Test helper: create a `Param` with name "_".
    fn p(i: &Interner, ty: Ty) -> Param {
        Param::new(i.intern("_"), ty)
    }

    fn check(source: &str) -> Result<TypeMap, String> {
        let interner = Interner::new();
        check_with_interner(source, &FxHashMap::default(), &interner)
    }

    fn check_with_interner(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> Result<TypeMap, String> {
        check_with_env(source, context, &FxHashMap::default(), interner)
    }

    fn check_with_env(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        functions: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> Result<TypeMap, String> {
        let template = acvus_ast::parse(interner, source).expect("parse failed");
        let mut sources = crate::ty::Sources::new();
        let registry = TypeRegistry::default();
        let mut solver = Solver::new(&mut sources, &registry);
        let qref_contexts: FxHashMap<QualifiedRef, InferTy> = context
            .iter()
            .map(|(&name, ty)| (QualifiedRef::root(name), crate::ty::lift_ty(ty)))
            .collect();
        let qref_functions = functions
            .iter()
            .map(|(&name, ty)| {
                (
                    QualifiedRef::root(name),
                    crate::ty::Scheme::unbounded(crate::ty::lift_ty(ty)),
                )
            })
            .collect();

        let env = crate::ty::TypeEnv {
            contexts: qref_contexts,
            functions: qref_functions,
            machine: FxHashMap::default(),
        };
        let checker = TypeChecker::new(interner, &env, &mut solver);
        let resolution = checker.check_template(&template).map_err(|errs| {
            errs.iter()
                .map(|e| {
                    format!(
                        "[typeck] [{}..{}] {}",
                        e.span.start,
                        e.span.end,
                        e.display(interner)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        Ok(resolution.type_map.clone())
    }

    #[test]
    fn literal_string_emit() {
        check("{{ \"hello\" }}").unwrap();
    }

    #[test]
    fn literal_int_emit_fails() {
        assert!(check("{{ 42 }}").is_err());
    }

    #[test]
    fn arithmetic_int() {
        // Int arithmetic result is Int, not String - emit should fail.
        assert!(check("{{ 1 + 2 }}").is_err());
    }

    #[test]
    fn arithmetic_mixed_fails() {
        // We need the result to be used somewhere. Let's use a match to avoid emit errors.
        // Actually, let's test directly: Int + Float is a type error.
        let src = r#"{{ x = 1 + 2.0 }}{{_}}{{/}}"#;
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn catch_all_optional() {
        // Catch-all is optional - match blocks without {{_}} should type-check fine.
        let src = "{{ x = 42 }}hello{{/}}";
        let result = check(src);
        result.unwrap();
    }

    #[test]
    fn context_read() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let src = "{{ @name }}";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    fn undefined_variable() {
        let src = "{{ x = unknown }}{{_}}{{/}}";
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn extern_param_write_rejected() {
        let src = "{{ $count = 42 }}";
        let err = check(src).expect_err("should reject extern param write");
        assert!(
            err.contains("$count"),
            "expected ExternParamAssign error, got: {err}"
        );
    }

    #[test]
    fn extern_fn_call() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("fetch_user"),
            Ty::Fn {
                params: vec![p(&i, Ty::I64)],
                ret: Box::new(Ty::String),

                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        )]);
        let src = "{{ x = fetch_user(1) }}{{ x }}{{_}}{{/}}";
        check_with_env(src, &FxHashMap::default(), &context, &i).unwrap();
    }

    #[test]
    fn field_access() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("user"),
            Ty::Object(ObjectTy::written(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::I64),
            ]))),
        )]);
        let src = "{{ @user.name }}";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    /// A field the object did not have grows the object (solver.md R1);
    /// whether it is initialized where it is read is the
    /// definite-assignment check's question, not the type checker's.
    fn field_access_grows_the_object() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("user"),
            Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
                i.intern("name"),
                Ty::String,
            )]))),
        )]);
        let src = "{{ @user.unknown }}";
        let result = check_with_interner(src, &context, &i);
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn pattern_binding_captures_type() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let src = "{{ x = @name }}{{ x }}{{_}}{{/}}";
        check_with_interner(src, &context, &i).unwrap();
    }

    // -- Variant (Option) --------------------------------------------

    #[test]
    fn some_int_is_option_int() {
        let src = "{{ x = Some(42) }}{{_}}{{/}}";
        check(src).unwrap();
    }

    #[test]
    fn none_is_option() {
        let src = "{{ x = None }}{{_}}{{/}}";
        check(src).unwrap();
    }

    #[test]
    fn some_pattern_extracts_inner() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::String)))]);
        let src = "{{ Some(x) = @opt }}{{ x }}{{_}}{{/}}";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    fn none_pattern_matches_option() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::I64)))]);
        let src = "{{ None = @opt }}none{{_}}has value{{/}}";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    fn some_type_mismatch() {
        let i = Interner::new();
        // Some(42) is Option<Int>, cannot match against String
        let context = FxHashMap::from_iter([(i.intern("s"), Ty::String)]);
        let src = "{{ Some(x) = @s }}{{ x }}{{_}}{{/}}";
        assert!(check_with_interner(src, &context, &i).is_err());
    }

    // -- Named extern functions --

    /// A string literal's type (RFC-0062 Decision 2).
    fn str_view() -> Ty {
        Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::Str)))
    }

    fn my_fn(interner: &Interner) -> FxHashMap<Astr, Ty> {
        FxHashMap::from_iter([(
            interner.intern("my_fn"),
            Ty::Fn {
                params: vec![p(interner, str_view())],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        )])
    }

    #[test]
    fn extern_fn_call_ok() {
        let i = Interner::new();
        let src = r#"{{ my_fn("hello") }}"#;
        check_with_env(src, &FxHashMap::default(), &my_fn(&i), &i).unwrap();
    }

    #[test]
    fn opaque_callback_rejected_where_pure_required() {
        let i = Interner::new();
        let pure_fn_ty = Ty::Fn {
            params: vec![p(&i, Ty::I64)],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: crate::ty::Effect::PURE.into(),
        };
        let fns = FxHashMap::from_iter([
            (
                i.intern("hof"),
                Ty::Fn {
                    params: vec![p(&i, pure_fn_ty.clone())],
                    ret: Box::new(Ty::String),
                    captures: vec![],
                    effect: crate::ty::Effect::PURE.into(),
                },
            ),
            (
                i.intern("io_fn"),
                Ty::Fn {
                    params: vec![p(&i, Ty::I64)],
                    ret: Box::new(Ty::I64),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                },
            ),
        ]);
        let none = FxHashMap::default();
        let err = check_with_env("{{ hof(|x| -> io_fn(x)) }}", &none, &fns, &i).unwrap_err();
        assert!(err.contains("with Opaque"), "{err}");
        check_with_env("{{ hof(|x| -> x + 1) }}", &none, &fns, &i).unwrap();
    }

    #[test]
    fn extern_fn_pipe_call_ok() {
        let i = Interner::new();
        let src = r#"{{ "hello" | my_fn }}"#;
        check_with_env(src, &FxHashMap::default(), &my_fn(&i), &i).unwrap();
    }

    #[test]
    fn extern_fn_pipe_with_args_ok() {
        let i = Interner::new();
        let fns = FxHashMap::from_iter([(
            i.intern("my_fn"),
            Ty::Fn {
                params: vec![p(&i, str_view()), p(&i, Ty::I64)],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        )]);
        let src = r#"{{ "hello" | my_fn(42) }}"#;
        check_with_env(src, &FxHashMap::default(), &fns, &i).unwrap();
    }

    // -- A context holds data (RFC-0014) --

    #[test]
    fn context_fn_rejected() {
        let i = Interner::new();
        let ctx = my_fn(&i);
        let err = check_with_interner("{{ f = @my_fn }}{{_}}{{/}}", &ctx, &i).unwrap_err();
        assert!(err.contains("not data"), "{err}");
        let err = check_with_interner(r#"{{ @my_fn("hello") }}"#, &ctx, &i).unwrap_err();
        assert!(err.contains("not data"), "{err}");
    }

    #[test]
    fn context_data_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("items"),
            Ty::Array(Box::new(Ty::I64), LenTerm::Known(3)),
        )]);
        let src = "{{ x = @items }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    // Iterator/Sequence context load tests migrated to acvus-mir-test
    // (requires TypeRegistry + Interner for UserDefined construction).

    #[test]
    fn lazy_option_context_load_ok() {
        // @opt : Option<Int> - Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::I64)))]);
        let src = "{{ x = @opt }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn lazy_tuple_context_load_ok() {
        // @pair : (Int, String) - Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("pair"), Ty::Tuple(vec![Ty::I64, Ty::String]))]);
        let src = "{{ x = @pair }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn lazy_object_context_load_ok() {
        // @obj : {x: Int} - Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("obj"),
            Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
                i.intern("x"),
                Ty::I64,
            )]))),
        )]);
        let src = "{{ x = @obj }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn context_list_of_fn_rejected() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("fns"),
            Ty::Array(
                Box::new(Ty::Fn {
                    params: vec![p(&i, Ty::I64)],
                    ret: Box::new(Ty::I64),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                }),
                LenTerm::Known(3),
            ),
        )]);
        let src = "{{ x = @fns }}{{_}}{{/}}";
        let err = check_with_interner(src, &ctx, &i).unwrap_err();
        assert!(err.contains("not data"), "{err}");
    }

    #[test]
    fn context_user_defined_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("conn"),
            Ty::UserDefined {
                id: QualifiedRef::root(i.intern("TestOpaque")),
                type_args: vec![],
                effect_args: vec![],
                identity_args: vec![],
            },
        )]);
        let src = "{{ x = @conn }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn context_user_defined_as_argument_ok() {
        let i = Interner::new();
        let conn_ty = Ty::UserDefined {
            id: QualifiedRef::root(i.intern("TestOpaque")),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
        };
        let ctx = FxHashMap::from_iter([(i.intern("conn"), conn_ty.clone())]);
        let fns = FxHashMap::from_iter([(
            i.intern("handler"),
            Ty::Fn {
                params: vec![p(&i, conn_ty)],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        )]);
        let src = "{{ handler(@conn) }}";
        check_with_env(src, &ctx, &fns, &i).unwrap();
    }

    #[test]
    fn pure_int_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("count"), Ty::I64)]);
        let src = "{{ x = @count }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn pure_string_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("msg"), Ty::String)]);
        let src = "{{ x = @msg }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }
}

/// Where a lent argument points: a local, a context, or an extern
/// parameter, and the field path below it (RFC-0015).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Place {
    root: PlaceRoot,
    path: Vec<Astr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PlaceRoot {
    Local(Astr),
    Context(QualifiedRef),
}

impl Place {
    fn display(&self, interner: &Interner) -> String {
        let mut out = match &self.root {
            PlaceRoot::Local(name) => interner.resolve(*name).to_string(),
            PlaceRoot::Context(qref) => format!("@{}", interner.resolve(qref.name)),
        };
        for f in &self.path {
            out.push('.');
            out.push_str(interner.resolve(*f));
        }
        out
    }
}

/// The place an expression denotes, if it denotes one. An extern
/// parameter is a value, not a place.
/// Whether an expression denotes a place.
pub(crate) fn is_place(expr: &Expr) -> bool {
    place_of(expr).is_some()
}

fn place_of(expr: &Expr) -> Option<Place> {
    match expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value | RefKind::ExternParam,
            ..
        } => Some(Place {
            root: PlaceRoot::Local(name.name),
            path: Vec::new(),
        }),
        Expr::ContextRef { name, .. } => Some(Place {
            root: PlaceRoot::Context(*name),
            path: Vec::new(),
        }),
        Expr::FieldAccess { object, field, .. } => {
            let mut place = place_of(object)?;
            place.path.push(*field);
            Some(place)
        }
        // `a[i]` is the place `a` with the index left off: two elements of
        // one container are one loan, which is what the slice holds
        // (RFC-0047 §3).
        Expr::Index { object, .. } => place_of(object),
        Expr::Paren { inner, .. } => place_of(inner),
        _ => None,
    }
}
