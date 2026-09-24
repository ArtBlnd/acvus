use acvus_ast::report::Label;
use acvus_ast::{
    AstId, BinOp, Clean, ErrorNode, Expr, Literal, ObjectExprField, ObjectPatternField, Pattern,
    RefKind, Slot, Span, SuffixedInt, Template, TupleElem, TuplePatternElem,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::error::{
    DataShape, DidYouMean, InstanceWanted, MirError, MirErrorKind, OperatorSignature, ShownValue,
};
use crate::graph::bind::{DeferredJoin, Typed, type_bound};
use crate::graph::{Bindings, BoundValue, Inputs, QualifiedRef};
use crate::ir::{Callee, CastKind, Chosen, ExternCast, ForKind, IndexAccess, IndexMode};
use crate::place::{
    Loan, PlaceBase, Storage, WrittenBase, names_a_place, projected, projected_store,
};
use crate::solver::{
    Admission, Answer, CallShape, Candidate, CaptureOutcome, CaptureRead, CompilerInstances,
    ComponentOffer, Conversion, ConvertedArgument, Decision, DecisionId, EffectRelation, Handed,
    InstanceChoice, InstanceKind, Kept, LendOutcome, MatchBinding, MatchMode, MatchOutcome,
    Mismatch, MismatchReason, ReceiverMode, ReferencePair, RequiredDecision, SettledSignature,
    SignatureCandidate, SignatureName, SignatureOption, UndecidedCall, UnjoinedArgument, Unsettled,
    Withholds, takes_a_language_owned_type,
};
use crate::structural::{StructuralSignature, structural_leaves};
use crate::ty::generalize_patterns;
use crate::ty::{
    CastTy, Effect, EffectTerm, Flows, HeldTy, Infer, InferTy, IntTy, LenTerm, Mutability,
    ObjectTy, ParamTerm, Phase, Solver, Task, Ty, TyTerm, TyVarBound, TypeArg, TypeEnv, View,
    Viewed,
};
use crate::variant::VariantPayload;

/// Maps each AST node id to its inferred type.
pub type TypeMap = FxHashMap<AstId, Ty>;

mod body_flows;

/// Maps expression AST ids to the coercion needed at that point.
/// Produced by the type checker, consumed by the lowerer.
pub type CoercionMap = Vec<(AstId, CastKind)>;

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

/// A bounded effect variable and where a violation of its bound is
/// reported (RFC-0011 rule 5).
struct EffectBoundSite {
    var: crate::ty::EffectVarId,
    span: Span,
}

/// An integer literal awaiting its width, checked against its value once
/// the width is known (RFC-0037).
/// How a pattern source is read: known where it was checked, or the answer
/// of the match decision its open head opened.
#[derive(Clone, Copy)]
enum SourceMode {
    Read(MatchMode),
    Decided(DecisionId),
}

/// Every name a pattern binds, once per binding.
fn bound_names<S>(pattern: &Pattern<S>, on_name: &mut impl FnMut(Astr)) {
    match pattern {
        Pattern::Binding { name, .. } => on_name(*name),
        Pattern::ContextBind { .. }
        | Pattern::Literal { .. }
        | Pattern::Wildcard { .. }
        | Pattern::Error(_) => {}
        Pattern::List { head, tail, .. } => {
            for part in head.iter().chain(tail) {
                bound_names(part, on_name);
            }
        }
        Pattern::Object { fields, .. } => {
            for field in fields {
                bound_names(&field.pattern, on_name);
            }
        }
        Pattern::Tuple { elements, .. } => {
            for element in elements {
                if let TuplePatternElem::Pattern(part) = element {
                    bound_names(part, on_name);
                }
            }
        }
        Pattern::Variant { payload, .. } => {
            if let Some(part) = payload {
                bound_names(part, on_name);
            }
        }
    }
}

struct LentWhole {
    place: AstId,
    mutability: Mutability,
}

struct OpenDecision {
    open: InferTy,
    span: Span,
}

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
    /// The `as_slice` call the index is; a refusal of that call is
    /// reported as the index's own (`MirErrorKind::CannotIndex`).
    callee_id: AstId,
    span: Span,
    container: InferTy,
    element: InferTy,
    demand: PlaceDemand,
}

/// One argument whose container was still a variable where the argument met
/// its parameter. What an `a[i]` argument names is settled by the index's own
/// signature decision, so the coercion waits for the solve as `IndexUse`
/// does.
///
/// The parameter is the view already (RFC-0047 rule 6): the referent owes
/// the declaration, and a referent that declares none is the argument
/// mismatch.
struct SliceArg {
    at: AstId,
    span: Span,
    arg: InferTy,
    param: InferTy,
    viewed: Viewed,
}

/// An argument of an overloaded call whose parameter belongs to whichever
/// candidate the signature decision settles on (RFC-0043 rule 2), so what
/// it owes is that candidate's admission of it, asked once the solve has
/// named its head.
struct HeldArgument<S> {
    site: ArgSite,
    index: usize,
    arg: InferTy,
    param: InferTy,
    handed: HeldHanded<S>,
}

/// How a held argument reaches the candidate the decision settles on.
enum HeldHanded<S> {
    AsWritten,
    /// A receiver place whose head was open where the call met it: the
    /// settled candidate's mode lends it or moves it (RFC-0043 rule 6), and
    /// that lend or move is recorded once the mode is known. `referent` is
    /// what the lend decision opened at the place names.
    Receiver {
        place: Expr<S>,
        referent: InferTy,
    },
}

impl<S> HeldArgument<S> {
    fn unjoined(&self) -> UnjoinedArgument {
        UnjoinedArgument {
            index: self.index,
            ty: self.arg.clone(),
            handed: match &self.handed {
                HeldHanded::AsWritten => Handed::AsIs,
                HeldHanded::Receiver { referent, .. } => Handed::Receiver {
                    referent: referent.clone(),
                },
            },
        }
    }
}

/// A receiver place whose head is open where the call meets it, held for
/// the candidate the decision settles on (RFC-0043 rule 6).
struct OpenReceiver<S> {
    place: Expr<S>,
    referent: InferTy,
}

/// A receiver place as the mode the call settled on passes it.
enum Handing {
    Lent {
        mutability: Mutability,
        lent: InferTy,
    },
    Value,
}

/// One `a[i]` as the checker reads it (RFC-0047).
struct IndexSite<'a, S> {
    id: AstId,
    callee_id: AstId,
    object: &'a Expr<S>,
    index: &'a Expr<S>,
    span: Span,
    demand: PlaceDemand,
}

/// What a `&C` names, and `C` itself where the type is not a reference.
/// A call's arguments as the source wrote them, with the receiver of a
/// method call counted but not written: `recv.f(a)` has one argument at
/// offset one, because the receiver's own spelling is not in `args`.
struct CallAsWritten<'a, S> {
    args: &'a [Expr<S>],
    offset: usize,
}

fn written_place<S>(interner: &Interner, expr: &Expr<S>) -> ShownValue {
    match expr {
        Expr::Ident { name, .. } => ShownValue::Named(interner.resolve(name.name).to_string()),
        Expr::ContextRef { name, .. } => {
            ShownValue::Named(format!("@{}", interner.resolve(name.name)))
        }
        _ => ShownValue::Anonymous,
    }
}

fn behind_a_reference(ty: &Ty) -> Cow<'_, Ty> {
    match ty {
        TyTerm::Ref(_, referent) => referent.ty(),
        other => Cow::Borrowed(other),
    }
}

/// What a lent operand's reference names, as the operator's report shows it.
fn referent_shown(ty: Ty) -> Ty {
    match ty {
        Ty::Ref(_, named) => named.into_ty(),
        other => other,
    }
}

fn referent_of(ty: &InferTy) -> Cow<'_, InferTy> {
    match ty {
        TyTerm::Ref(_, referent) => referent.ty(),
        other => Cow::Borrowed(other),
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
            Some(TyTerm::Ref(_, takes)) => sliceable_head(&takes.ty()) == Some(head),
            _ => false,
        }
    })
}

/// Whether a declared type says what it is, as against a bare variable
/// that takes or yields anything: `dedup<I>(it: I)` takes anything, and
/// `unwrap` yields anything.
fn names_a_shape<V>(ty: &TyTerm<V>) -> bool
where
    V: crate::ty::Phase,
{
    !matches!(ty, TyTerm::Var(_))
}

/// The type behind a reference, or the type itself.
fn strip_ref<V>(ty: &TyTerm<V>) -> Cow<'_, TyTerm<V>>
where
    V: crate::ty::Phase,
{
    match ty {
        TyTerm::Ref(_, inner) => inner.ty(),
        _ => Cow::Borrowed(ty),
    }
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

/// What the referent's evidence answered when a view was asked of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SliceCoercion {
    /// The referent declares the view, and the cast is recorded at the
    /// argument.
    Coerced,
    /// Nothing the referent declares takes it at that view and mutability,
    /// so the argument is left to unification against the parameter.
    NoDeclaration,
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
    /// other read borrows the object of `a.f` shared (RFC-0047 rule 3).
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
enum Signatures {
    One(SignatureCandidate),
    Several(Vec<SignatureCandidate>),
}

impl Signatures {
    fn of(candidates: Vec<SignatureCandidate>) -> Option<Self> {
        match <[SignatureCandidate; 1]>::try_from(candidates) {
            Ok([one]) => Some(Self::One(one)),
            Err(several) if several.is_empty() => None,
            Err(several) => Some(Self::Several(several)),
        }
    }

    fn candidates(&self) -> &[SignatureCandidate] {
        match self {
            Self::One(one) => std::slice::from_ref(one),
            Self::Several(several) => several,
        }
    }
}

struct NamedCall<'e, S> {
    callee: CalleeSite,
    name: Astr,
    args: &'e [Expr<S>],
    span: Span,
    refused_as: CallRefusal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallRefusal {
    NoMatchingFunction,
    CannotIndex,
}

enum FirstOperand<'e, S> {
    Absent,
    Checked(FirstArg),
    Receiver(&'e Expr<S>),
}

/// RFC-0030, RFC-0043.
enum Received<S> {
    Taken(Signatures, FirstArg),
    Held(Vec<SignatureCandidate>, FirstArg, OpenReceiver<S>),
    Refused(FirstArg),
    AmbiguityReported,
}

/// RFC-0043.
struct Narrowing<S> {
    options: Vec<SignatureOption>,
    awaiting_head: Vec<UnjoinedArgument>,
    held: Vec<HeldArgument<S>>,
    params: Vec<ParamTerm<Infer>>,
    receiver: Option<OpenReceiver<S>>,
}

enum Parameters<'p, S> {
    Instantiated(&'p [InferTy]),
    Narrowing(&'p mut Narrowing<S>),
}

enum Reach {
    Joined,
    /// Not joined with the call's parameter: the signature decision joins
    /// that parameter with the settled candidate's own, and a join here
    /// would give it the argument's type, which that candidate takes only
    /// through the conversion (RFC-0043).
    Converted,
    /// RFC-0062 rule 3.
    Viewed(Viewed),
}

/// How an argument of an overloaded call meets the call's parameter.
enum Narrowed<S> {
    Reaches(Reach),
    /// RFC-0043 rule 2: as the candidate the decision settles on admits it.
    Held(HeldHanded<S>),
}

struct CallType {
    ty: InferTy,
    params: Vec<InferTy>,
    ret: InferTy,
    effect: EffectTerm<Infer>,
}

impl CallType {
    fn of(ty: InferTy) -> Option<Self> {
        let TyTerm::Fn {
            params,
            ret,
            effect,
            ..
        } = &ty
        else {
            return None;
        };
        Some(Self {
            params: params.iter().map(|param| param.ty.clone()).collect(),
            ret: (**ret).clone(),
            effect: effect.clone(),
            ty,
        })
    }
}

enum Uncallable {
    Poisoned,
    NotCallable(Ty),
    Arity(usize),
}

/// A call argument at its site, with what it borrows when it is `&expr`
/// or a receiver lent as one: a conversion answered through the reference
/// takes that value out for the call (RFC-0041).
#[derive(Debug, Clone)]
struct ArgSite {
    id: AstId,
    span: Span,
    /// The expression that takes this argument -- the call, the `a[i]`, or
    /// the `for` head. The registry holds no declaration span for an
    /// extern parameter, so a refusal about the parameter marks the whole
    /// expression that named it and labels the argument inside it.
    taken_by: Span,
    borrowed: Option<Borrowed>,
}

#[derive(Debug, Clone)]
struct Borrowed {
    /// The name or value the borrowed place is projected from.
    base: AstId,
    loan: Option<Loan>,
}

impl Borrowed {
    fn of<S>(expr: &Expr<S>) -> Self
    where
        S: Slot,
    {
        Self {
            base: projected(expr).base.id(),
            loan: Loan::of(expr),
        }
    }
}

impl ArgSite {
    fn of<S>(expr: &Expr<S>, taken_by: Span) -> Self
    where
        S: Slot,
    {
        let borrowed = match expr {
            Expr::Borrow { place, .. } => Some(Borrowed::of(place)),
            _ => None,
        };
        Self {
            id: expr.id(),
            span: expr.span(),
            taken_by,
            borrowed,
        }
    }

    fn value<S>(expr: &Expr<S>, taken_by: Span) -> Self
    where
        S: Slot,
    {
        Self {
            id: expr.id(),
            span: expr.span(),
            taken_by,
            borrowed: None,
        }
    }

    /// A receiver lent to a reference parameter.
    fn lent<S>(expr: &Expr<S>, taken_by: Span) -> Self
    where
        S: Slot,
    {
        Self {
            id: expr.id(),
            span: expr.span(),
            taken_by,
            borrowed: Some(Borrowed::of(expr)),
        }
    }
}

/// The arity a candidate declares where it is not the call's (RFC-0043). A
/// candidate whose arity is not yet known is one the call's own arity
/// fixes, so it is never refused here.
fn arity_refusal(candidate: &SignatureCandidate, arity: usize) -> Option<usize> {
    candidate.arity().filter(|declared| *declared != arity)
}

/// How a receiver reaches the call that takes it (RFC-0030).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Passing {
    Value,
    /// Lent from its place, or from a temporary holding it where it names
    /// no place.
    Lent(Mutability),
    /// Already a reference, passed as it is.
    AsIs,
}

/// RFC-0020: an operator's `core` signature takes its operands by
/// reference, and `StringEq` and `StringConcat` lend them too, so the
/// operand places are lent instead of copied.
fn operand_stays_lent(op: BinOp) -> bool {
    match op {
        BinOp::Eq
        | BinOp::Neq
        | BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte => true,
        BinOp::Xor
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::And
        | BinOp::Or => false,
    }
}

/// The union of the shapes the candidates take at `index` directly: the
/// call's parameter's range there (RFC-0043).
fn param_range(options: &[SignatureOption], index: usize) -> TyVarBound {
    options
        .iter()
        .map(|option| option.candidate.param_bound(index))
        .fold(TyVarBound::one_of(vec![]), TyVarBound::union)
}

/// RFC-0043.
fn agreed_receiver_mode(mut modes: impl Iterator<Item = ReceiverMode>) -> Option<ReceiverMode> {
    let first = modes.next()?;
    modes.all(|mode| mode == first).then_some(first)
}

/// RFC-0030.
fn lent_only_if_agreed(agreed: Option<ReceiverMode>) -> ReceiverMode {
    match agreed {
        Some(lent @ ReceiverMode::Lent(_)) => lent,
        Some(ReceiverMode::Value) | None => ReceiverMode::Value,
    }
}

/// Where the candidates disagree, the mode is fixed only after the place is
/// checked, and a shared borrow is what every mode can be read off
/// (RFC-0043).
fn receiver_demand(agreed: Option<ReceiverMode>) -> PlaceDemand {
    match agreed {
        Some(ReceiverMode::Lent(mutability)) => PlaceDemand::Borrow(mutability),
        Some(ReceiverMode::Value) => PlaceDemand::Value,
        None => PlaceDemand::Borrow(Mutability::Shared),
    }
}

pub use crate::ir::Intrinsic;

/// An operator resolved to a shared signature (RFC-0020).
#[derive(Debug, Clone)]
pub struct OperatorCall<C, T> {
    pub callee: C,
    pub signature: OperatorSignature,
    pub ty: T,
    pub at: Span,
}

/// The operator a decision was opened for, as its refusal names it.
#[derive(Clone, Copy)]
struct OperatorSite {
    op: &'static str,
    signature: OperatorSignature,
}

/// An operator's operand as its call takes it: the type it is read at,
/// what it names (`read_operand_through`), and the expression a
/// conversion to the parameter is recorded at.
#[derive(Clone, Copy)]
struct Operand<'t> {
    ty: &'t InferTy,
    at: AstId,
}

/// Keyed by the id `lower.rs` reads each call at: a named call's name, the
/// `as_slice` id of an index or of a `for` over a borrow, and an operator's
/// own expression. A call of an expression's value has no entry, and the
/// lowering calls that value.
#[derive(Debug, Clone)]
pub enum CallTarget {
    Declared(Callee),
    /// RFC-0020.
    Intrinsic(Intrinsic),
    Binding,
    /// RFC-0030.
    StructuralVariant,
    /// RFC-0020.
    Operator(OperatorCall<Callee, Ty>),
    /// RFC-0020.
    Structural(StructuralCall),
}

/// Obligation across artifacts: `leaves` is in `structural_leaves` order over
/// the type the call's first parameter lends, which is the order
/// `acvus_interpreter::prepare` reads them in.
#[derive(Debug, Clone)]
pub struct StructuralCall {
    pub signature: StructuralSignature,
    pub leaves: Vec<Chosen>,
}

pub type CallMap = FxHashMap<AstId, CallTarget>;

/// The reference inside the data a result to the host holds. A function
/// type is not searched: its parameters and return are a signature, not
/// storage this run holds.
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

/// A lambda that holds a loan: RFC-0064 rule 5 makes it a holder like a
/// reference, so it is refused where a reference is. `reference_in_result`
/// stops at a function type because a signature is not storage; a capture
/// list is, and this is where it is walked.
fn holds_a_loan(ty: &InferTy) -> bool {
    let TyTerm::Fn { captures, .. } = ty else {
        return false;
    };
    captures
        .iter()
        .any(|c| reference_in_result(c).is_some() || holds_a_loan(c))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResultCrossing {
    /// Into the registers `control::Return` writes: one word, or the two
    /// adjacent words a view occupies (RFC-0047 rule 6).
    Registers,
    /// Into one `Value` read by kind: a lambda's result, which
    /// `acvus_extern::Runtime::call_now` hands a handler and
    /// `interpreter::code::Code::call` is the signature of.
    OneValue,
    /// Into one `Value` the host reads by kind (RFC-0054): the graph's
    /// entry. The value outlives the run, which a closure does not
    /// (RFC-0014, RFC-0069 rule 1).
    Host,
}

/// Why a body's result does not cross.
enum Unreturnable<'t> {
    Reference(&'t InferTy),
    ClosureToTheHost(&'t InferTy),
}

impl ResultCrossing {
    /// A lambda's or a named function's result may hold references: the
    /// borrow check admits only loans on the body's inputs there, each one
    /// its flows name (RFC-0079 rules 5 and 9). What stays refused is
    /// representation: a view where one value is read, or a view inside
    /// data (RFC-0062 rules 5 and 6, RFC-0047 rule 6). The host's result
    /// outlives the run and holds no loan and no closure.
    fn unreturnable<'t>(self, ty: &'t InferTy) -> Option<Unreturnable<'t>> {
        match self {
            Self::Host => {
                let reference = match ty {
                    TyTerm::Ref(_, target) => is_view(&target.ty()).then_some(ty),
                    _ => reference_in_result(ty),
                };
                match reference {
                    Some(reference) => Some(Unreturnable::Reference(reference)),
                    None => closure_in_result(ty).map(Unreturnable::ClosureToTheHost),
                }
            }
            Self::Registers if is_pair(ty) => None,
            Self::Registers | Self::OneValue => match ty {
                TyTerm::Ref(_, target) => is_view(&target.ty()).then_some(ty),
                _ => view_in_result(ty),
            }
            .map(Unreturnable::Reference),
        }
    }
}

/// A view inside a result's data, walked as `reference_in_result` walks it.
fn view_in_result(ty: &InferTy) -> Option<&InferTy> {
    match ty {
        TyTerm::Ref(_, target) => is_view(&target.ty()).then_some(ty),
        TyTerm::Array(inner, _) | TyTerm::Slice(inner) | TyTerm::Option(inner) => {
            view_in_result(inner)
        }
        TyTerm::Result(ok, err) => view_in_result(ok).or_else(|| view_in_result(err)),
        TyTerm::Tuple(elems) => elems.iter().find_map(view_in_result),
        TyTerm::Object(object) => object.iter().find_map(|(_, field)| view_in_result(field)),
        TyTerm::Enum { variants, .. } => variants
            .iter()
            .filter_map(|(_, payload)| payload.as_ref())
            .find_map(|payload| view_in_result(payload)),
        _ => None,
    }
}

/// A closure anywhere in a result's data, walked as `reference_in_result`
/// walks it.
fn closure_in_result(ty: &InferTy) -> Option<&InferTy> {
    match ty {
        TyTerm::Fn { .. } => Some(ty),
        TyTerm::Array(inner, _) | TyTerm::Slice(inner) | TyTerm::Option(inner) => {
            closure_in_result(inner)
        }
        TyTerm::Result(ok, err) => closure_in_result(ok).or_else(|| closure_in_result(err)),
        TyTerm::Tuple(elems) => elems.iter().find_map(closure_in_result),
        TyTerm::Object(object) => object
            .iter()
            .find_map(|(_, field)| closure_in_result(field)),
        TyTerm::Enum { variants, .. } => variants
            .iter()
            .filter_map(|(_, payload)| payload.as_ref())
            .find_map(|payload| closure_in_result(payload)),
        TyTerm::UserDefined { type_args, .. } => type_args.iter().find_map(closure_in_arg),
        _ => None,
    }
}

/// A closure anywhere in an argument, part by part. A `#` part over a type
/// variable is no closure, as the variable is not.
fn closure_in_arg(arg: &TypeArg<Infer>) -> Option<&InferTy> {
    match arg {
        TypeArg::Uniform(ty) | TypeArg::Open(_, ty) => closure_in_result(ty),
        TypeArg::Specialized(held) => match held {
            HeldTy::Leaf(leaf) => closure_in_result(leaf.ty()),
            HeldTy::Held(_) => None,
            HeldTy::Tuple(_)
            | HeldTy::Option(_)
            | HeldTy::Result(..)
            | HeldTy::Array(..)
            | HeldTy::RustArray(..) => held.parts().into_iter().find_map(closure_in_arg),
        },
    }
}

/// A view found at an extern type's type argument.
struct ViewArgument<'t> {
    taker: &'t InferTy,
    view: &'t InferTy,
}

/// An extern type's argument is one runtime value when uniform, or its Rust
/// type when `#` (RFC-0041); a view is the two registers `is_pair` names,
/// which are neither (RFC-0062 rule 5), so it is refused there whether the
/// type stores it or not. A reference is one word whatever it names, so the
/// storage below one is searched as a value. A function type is not
/// searched: its parameters and return are a signature, not a value this
/// run holds.
fn view_argument<'t>(
    ty: &'t InferTy,
    argument_of: Option<&'t InferTy>,
) -> Option<ViewArgument<'t>> {
    if let Some(taker) = argument_of
        && is_pair(ty)
    {
        return Some(ViewArgument { taker, view: ty });
    }
    match ty {
        TyTerm::UserDefined { type_args, .. } => {
            type_args.iter().find_map(|arg| view_in_arg(arg, Some(ty)))
        }
        TyTerm::Ref(_, target) => view_in_arg(target, None),
        TyTerm::Array(inner, _)
        | TyTerm::Slice(inner)
        | TyTerm::Option(inner)
        | TyTerm::Handle(inner) => view_argument(inner, argument_of),
        TyTerm::Result(ok, err) => {
            view_argument(ok, argument_of).or_else(|| view_argument(err, argument_of))
        }
        TyTerm::Tuple(elems) => elems.iter().find_map(|e| view_argument(e, argument_of)),
        TyTerm::Object(object) => object
            .iter()
            .find_map(|(_, field)| view_argument(field, argument_of)),
        TyTerm::Enum { variants, .. } => variants
            .iter()
            .filter_map(|(_, payload)| payload.as_ref())
            .find_map(|payload| view_argument(payload, argument_of)),
        _ => None,
    }
}

/// A `#` part over a type variable is no view, as the variable is not.
fn view_in_arg<'t>(
    arg: &'t TypeArg<Infer>,
    argument_of: Option<&'t InferTy>,
) -> Option<ViewArgument<'t>> {
    match arg {
        TypeArg::Uniform(ty) | TypeArg::Open(_, ty) => view_argument(ty, argument_of),
        TypeArg::Specialized(held) => match held {
            HeldTy::Leaf(leaf) => view_argument(leaf.ty(), argument_of),
            HeldTy::Held(_) => None,
            HeldTy::Tuple(_)
            | HeldTy::Option(_)
            | HeldTy::Result(..)
            | HeldTy::Array(..)
            | HeldTy::RustArray(..) => held
                .parts()
                .into_iter()
                .find_map(|part| view_in_arg(part, argument_of)),
        },
    }
}

/// Obligation across artifacts: the same predicate as
/// `acvus_interpreter::prepare::is_slice` on a frozen `Ty` — one level of
/// reference over `Str` or `Slice`, and the two adjacent registers
/// `assign_slots` lays for it (RFC-0062 rule 1). `is_view` recurses
/// through a reference and this does not: a reference *to* a view is the one
/// `Kind::Ref` word, which reaches half a pair.
fn is_pair(ty: &InferTy) -> bool {
    matches!(ty, TyTerm::Ref(_, target) if matches!(*target.ty(), TyTerm::Str | TyTerm::Slice(_)))
}

/// The two adjacent word registers a run occupies (RFC-0047 rule 6,
/// RFC-0062 rule 1). `acvus_interpreter::prepare::is_slice` is the same
/// predicate on a frozen `Ty`, and the two must name the same types: this
/// one decides what a body may return, that one lays the registers out, and
/// a type admitted here that it calls a pair returns into one register and
/// loses its length.
fn is_view(ty: &InferTy) -> bool {
    match ty {
        TyTerm::Slice(_) | TyTerm::Str => true,
        TyTerm::Ref(_, target) => is_view(&target.ty()),
        _ => false,
    }
}

/// A `$name` the body may read. A parameter the declaration named arrives
/// before the body is checked and has no place in it; one the body
/// discovered carries the place it was first read, which is where a type
/// that does not close is refused.
struct ExternParam {
    name: Astr,
    ty: InferTy,
    first_read: Option<Span>,
    origin: ParamOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamOrigin {
    Declared,
    Bound,
    Read(Reader),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reader {
    ThisBody,
    Reached(QualifiedRef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedParam {
    pub name: Astr,
    pub ty: Ty,
    pub origin: ParamOrigin,
}

struct InputRead {
    ty: InferTy,
    admitted: bool,
}

/// A `$name` the host bound, typed in this body before it was checked
/// (RFC-0087 rule 2). Its enum positions join the variants the value holds
/// once the body is checked, and a join the uses refuse is reported at the
/// first read, which is the only place those uses reach the input.
struct BoundInput {
    name: Astr,
    value: BoundValue,
    deferred: Vec<DeferredJoin>,
    first_read: Option<Span>,
}

/// How one candidate of a method call takes the receiver, seen as the type
/// that candidate's mode gives it (RFC-0043).
struct ReceiverAdmission {
    mode: ReceiverMode,
    seen_as: InferTy,
    admission: Admission,
}

struct CandidateAdmission {
    candidate: SignatureCandidate,
    receiver: ReceiverAdmission,
}

/// A call resolved to a named function, at the instance the solver fixed
/// or is still settling.
#[derive(Debug, Clone)]
struct ResolvedCallee {
    qref: QualifiedRef,
    instance: Option<InstanceChoice>,
    /// One instance decision per requirement the declaration states.
    requirements: Vec<RequiredDecision>,
}

/// A scheme instantiated at a use: its type and the decisions the use opened.
struct SchemeAt {
    ty: InferTy,
    instance: Option<InstanceChoice>,
    requirements: Vec<RequiredDecision>,
}

/// RFC-0043.
#[derive(Debug, Clone)]
enum CallChoice {
    Resolved(ResolvedCallee),
    Decided(DecisionId),
    Binding,
    StructuralVariant,
    Operator(OperatorCall<ResolvedCallee, InferTy>),
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
    Reborrow {
        shared: InferTy,
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
    pub calls: CallMap,
    /// How each `a[i]` reaches its element (RFC-0047). The checker decided
    /// it from the container's evidence and the element type; the lowering
    /// reads it and decides nothing.
    pub index_access: FxHashMap<AstId, IndexAccess>,
    /// Which of the four heads each `for` was written with (RFC-0057),
    /// keyed by the statement's own id.
    pub for_kinds: FxHashMap<AstId, ForKind>,
    /// How each receiver and each operator operand, keyed by its own
    /// expression, reaches what takes it.
    pub passing: FxHashMap<AstId, Passing>,
    /// How each `match` / `if let` / `while let` source, keyed by its own
    /// expression, is read by its patterns, and how each member a
    /// sub-pattern stands for, keyed by that pattern, is read by it
    /// (RFC-0024).
    pub pattern_modes: FxHashMap<AstId, MatchMode>,
    /// Where the base of each expression the lowering reads as a place
    /// lives, keyed by the base's own id. The checker settled it from the
    /// base's form and type; the lowering reads it and decides nothing.
    pub place_bases: FxHashMap<AstId, PlaceBase>,
    /// The return type of the function each `?` leaves early from (RFC-0038).
    pub try_returns: FxHashMap<AstId, Ty>,
    pub tail_ty: Ty,
    /// Every `$name` of the body in the order its module takes them.
    pub extern_params: Vec<ResolvedParam>,
    /// A lambda's captures, keyed by the lambda's own `AstId`. The lowering
    /// takes this list as the closure's captures, and the same lambda's
    /// `Fn { captures }` in `type_map` holds those names' types positionally,
    /// so the two must stay in one order: the MakeClosure arity and type
    /// check in `validate::type_check` is where a divergence is reported.
    pub lambda_captures: FxHashMap<AstId, Vec<CapturedName>>,
    /// Join of the effects of every call in the body.
    pub effect: Effect,
    pub context_types: FxHashMap<QualifiedRef, Ty>,
    /// The body's flows as a named function's (RFC-0079 rule 5), what its
    /// type states once its call-graph component settles.
    pub flows: Flows,
}

/// A name a closure captures and the reading its body was checked at, so
/// that the lowering emits that reading rather than deciding one.
#[derive(Debug, Clone, Copy)]
pub struct CapturedName {
    pub name: Astr,
    pub read: CaptureRead,
}

#[derive(Debug, Clone)]
pub struct BodyView {
    /// An accepted body's `types` is its resolution's `type_map`, the map
    /// the lowering reads, and beside it the type each binder is bound at.
    /// A refused body's are the types as written, as its refusals show
    /// them, so a poisoned position is `Ty::Error`.
    pub types: FxHashMap<AstId, Ty>,
    /// What each name the checker resolved refers to, keyed by the node
    /// the name is; for a call's callee, by the id the callee is recorded
    /// under, which for a method call, an index or a `for` is the AST's
    /// `callee_id` and not the call's. Every binder the body defines is in
    /// it, a use of it or not, as `Resolved::Local` of itself.
    pub resolved: FxHashMap<AstId, Resolved>,
}

/// What a name refers to, as the checker resolved it. `B` is what names a
/// local's binder: its node in `BodyView::resolved`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resolved<B = AstId> {
    Local(B),
    /// The declaration a call settled on.
    Function(QualifiedRef),
    /// The declared context an `@name` reads, writes or binds.
    Context(QualifiedRef),
    /// The input a `$name` the checker admitted reads.
    Input(Astr),
}

/// The resolutions a body's names have so far. A name refers to one
/// thing, so every resolution is recorded through `record`, which holds
/// that.
#[derive(Debug, Clone, Default)]
struct Resolutions(FxHashMap<AstId, Resolved>);

impl Resolutions {
    /// Record that the name at `id` refers to `resolved`. Recording it
    /// again is recording the same fact; recording another referent for
    /// it is a fault of the checker.
    fn record(&mut self, id: AstId, resolved: Resolved) {
        let held = *self.0.entry(id).or_insert(resolved);
        assert_eq!(
            held, resolved,
            "the name at {id:?} resolves to one referent"
        );
    }
}

enum SettledCallee {
    Declared(QualifiedRef),
    Binding,
}

/// A tree that holds an error node has no resolution to lower against,
/// only the refusals of what parsed around its error nodes (RFC-0078
/// rule 5). Its parse errors are reported beside it by whoever parsed it.
#[derive(Debug, Clone)]
pub struct Unresolved {
    pub refusals: Vec<MirError>,
}

pub trait Checks: Slot {
    type Resolution;

    fn conclude(
        checker: TypeChecker<'_, '_, '_, Self>,
        body: Span,
        tail: BodyTail,
        flows: Flows,
    ) -> Checked<Self::Resolution>;
}

pub enum BodyTail {
    Text,
    Value { ty: InferTy, at: Span },
}

impl Checks for Clean {
    type Resolution = Result<Freeze<TypeResolution>, Vec<MirError>>;

    fn conclude(
        mut checker: TypeChecker<'_, '_, '_, Self>,
        body: Span,
        tail: BodyTail,
        flows: Flows,
    ) -> Checked<Self::Resolution> {
        if !checker.errors.is_empty() {
            return checker.refused(Err);
        }
        let tail_ty = match tail {
            BodyTail::Text => Ty::String,
            BodyTail::Value { ty, at } => {
                let resolved = checker.solver.resolve_ty(&ty);
                checker.closed_or_refused(&resolved, at)
            }
        };
        checker.into_resolution(body, tail_ty, flows)
    }
}

impl Checks for ErrorNode {
    type Resolution = Unresolved;

    fn conclude(
        checker: TypeChecker<'_, '_, '_, Self>,
        _: Span,
        _: BodyTail,
        _: Flows,
    ) -> Checked<Unresolved> {
        checker.refused(|refusals| Unresolved { refusals })
    }
}

pub struct Checked<R = Result<Freeze<TypeResolution>, Vec<MirError>>> {
    pub resolution: R,
    pub view: Freeze<BodyView>,
    /// What the checker saw at the node `with_probe` marked; `None` where
    /// no node was marked or checking never reached the marked one.
    pub probe: Option<ProbeProduct>,
}

/// What a name written at the marked node could be, answered by the rules
/// checking applies there. Types are as written (RFC-0043), so a refused
/// body, which a body holding an unknown name is, still answers.
///
/// Where the solve fixed the type the marked position takes, `expected`
/// is that type and each candidate's `fits` says whether the checker
/// joins the candidate there: the flow a value makes at its use
/// (`TypeChecker::flow`), tried on a copy of the solve. Where it did not,
/// `expected` is `None` and nothing fits.
#[derive(Debug, Clone, PartialEq)]
pub enum ProbeProduct {
    /// The marker is a name read as a value, a `$` input, or a `@` context.
    Value {
        expected: Option<Ty>,
        /// The innermost binding of each name in scope at the marker, in
        /// name order.
        scope: Vec<Visible>,
        /// The `$` inputs of the body other than the marker itself: the
        /// declared parameters and bound inputs, or where none is declared,
        /// the ones the body reads.
        inputs: Vec<ProbedInput>,
        /// Every context the body can read, in name order; one fits if its
        /// type flows into `expected`.
        contexts: Vec<DeclarationFit>,
        /// Every function a call can name, in name order; one fits if the
        /// return type of a call to it flows into `expected`, the call
        /// instantiated as the checker instantiates one with its arguments
        /// still unwritten.
        functions: Vec<DeclarationFit>,
    },
    /// The marker is the name a field access or a method call reads off a
    /// receiver.
    Member {
        receiver: Ty,
        /// The type the field access or the method call takes where it is
        /// written.
        expected: Option<Ty>,
        /// The fields of the object the receiver is or references other
        /// than the marker, which reading it may have added, in name order;
        /// empty where it is no object.
        fields: Vec<ProbedField>,
        /// Every function name with a declaration whose first parameter
        /// admits the receiver as `receive` admits it, in name order. Empty
        /// where the receiver's head is still open, since no candidate's
        /// mode can be admitted then.
        methods: Vec<ProbedMethod>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Visible {
    pub name: Astr,
    pub binder: AstId,
    pub ty: Ty,
    pub fits: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbedInput {
    pub name: Astr,
    pub ty: Ty,
    pub fits: bool,
}

/// A declaration of the environment, which the reader looks up by `qref`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclarationFit {
    pub qref: QualifiedRef,
    pub fits: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbedField {
    pub name: Astr,
    pub ty: Ty,
    pub fits: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbedMethod {
    pub name: Astr,
    pub admitted: Vec<AdmittedDeclaration>,
    /// Whether a call of some admitted declaration on the receiver
    /// returns a type that flows into `expected`.
    pub fits: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedDeclaration {
    pub qref: QualifiedRef,
    pub ty: crate::ty::PolyTy,
}

/// What checking recorded at the marked node, held as the solver's types
/// until the body is solved.
enum Probed<S> {
    Value {
        marker: Astr,
        site: CalleeSite,
        scope: Vec<InScope>,
    },
    Member(ProbedMember<S>),
}

struct ProbedMember<S> {
    marker: Astr,
    receiver: ProbedReceiver,
    receiver_as_written: Expr<S>,
    marker_site: CalleeSite,
}

impl<S> ProbedMember<S> {
    /// The call of the function `name` a method call at the marker makes,
    /// its arguments after the receiver still unwritten.
    fn call_named(&self, name: Astr) -> NamedCall<'_, S> {
        NamedCall {
            callee: self.marker_site,
            name,
            args: &[],
            span: self.marker_site.span,
            refused_as: CallRefusal::NoMatchingFunction,
        }
    }
}

struct InScope {
    name: Astr,
    binder: AstId,
    ty: InferTy,
}

/// RFC-0030, RFC-0043: `receive` admits a place in each candidate's own
/// mode, and a value in the mode the candidates agree on.
enum ProbedReceiver {
    Place(InferTy),
    Value(InferTy),
}

struct Bound {
    ty: InferTy,
    binder: AstId,
}

/// What `x = e;` found for `x`.
enum AssignTarget {
    /// A binding of the body that carries the statement.
    Bound { ty: InferTy, binder: AstId },
    /// Bound outside the innermost lambda, so the lambda sees a copy.
    Captured { binder: AstId },
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

impl ConversionSite {
    /// A value written at `site`, reported as a call argument is.
    fn value_at(site: CalleeSite) -> Self {
        Self {
            id: site.id,
            span: site.span,
            report: ConversionReport::Value,
        }
    }
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
    /// An operator's operand against the operator's parameter: the operand
    /// is lent as it stands, so a conversion there is the identity or a
    /// mismatch of the operator's.
    Operand { op: &'static str },
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

/// A read under a head that was still open — a field, or `*` — whose
/// `Match` decision says whether it went through a reference; the demand
/// says what such a read may do (RFC-0018 rules 4 and 8).
struct ReadUnderOpenHead {
    decision: DecisionId,
    head: InferTy,
    read: InferTy,
    demand: PlaceDemand,
    through: ReadThrough,
    subject: ShownValue,
    span: Span,
}

/// A list pattern's demand on its source's length: exactly its parts
/// without `..`, at least them with it. The length is read once the solve
/// has joined every pattern with its source, so a source still open at the
/// pattern is held to it as a known one is.
struct ListPatternLength {
    len: LenTerm<Infer>,
    pattern_min: usize,
    exact: bool,
    span: Span,
}

/// A value stored in data, returned from a body, captured, or defined, read
/// once the solve has settled its type: a type still open where the value
/// was written is held to RFC-0062 rules 5 and 6 and RFC-0064 rule 5 as a
/// known one is.
struct ValueSite {
    at: ValuePosition,
    span: Span,
}

enum ValuePosition {
    /// A value a name binds or a call returns.
    Defined(InferTy),
    /// A component of data built here.
    Data(InferTy, DataShape),
    /// A body's or a lambda's result.
    Result(InferTy, ResultCrossing),
    /// What a lambda captures, one word each.
    Captures(Vec<InferTy>),
}

/// What an open head was read by: a field reads a value or through a
/// reference, and `*` only through one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadThrough {
    Field,
    Deref,
}

impl ReadThrough {
    /// Whether a read of `ty` out of what a reference names is a copy:
    /// `*r` needs a word (RFC-0018 rule 4), a place under the reference a
    /// word or a `String` (rule 2).
    fn copies<V>(self, ty: &TyTerm<V>) -> Option<bool>
    where
        V: Phase,
    {
        match self {
            Self::Field => ty.copies(),
            Self::Deref => ty.is_word(),
        }
    }

    fn refusal(self, ty: Ty) -> MirErrorKind {
        match self {
            Self::Field => MirErrorKind::MoveOutOfReference(ty),
            Self::Deref => MirErrorKind::DerefOfNonWord(ty),
        }
    }
}

/// A read by value that the lowering takes out of what a reference names
/// where its source is one, checked once every type and every place's base
/// is frozen.
#[derive(Clone, Copy)]
enum ValueRead {
    /// `*r`.
    Deref { span: Span },
    /// A place path read by value, through a reference where `base`
    /// settles to one (RFC-0047 rule 5).
    Place { base: AstId, span: Span },
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
    /// The base of the place the value borrows, at a call argument that
    /// borrows one.
    base: Option<AstId>,
}

pub struct TypeChecker<'a, 's, 'src, S = Clean> {
    /// Interner for string interning.
    interner: &'a Interner,
    /// Unified type environment: contexts + functions.
    env: &'a TypeEnv,
    /// Namespace this function belongs to.
    namespace: Option<Astr>,
    scopes: Vec<FxHashMap<Astr, Bound>>,
    /// What each name resolved to, but a call's callee, which `view`
    /// records from `calls` once the solve settled which one it is.
    resolved: Resolutions,
    candidate_binder_of: FxHashMap<AstId, AstId>,
    binder_types: FxHashMap<AstId, InferTy>,
    /// Extern parameters in Signature order, which is the order iteration
    /// must keep.
    param_types: smallvec::SmallVec<[ExternParam; 4]>,
    bound_inputs: Vec<BoundInput>,
    /// Solver state (borrowed - may be shared across compilations).
    solver: &'s mut Solver<'src>,
    /// Accumulated type map (internal, uses InferTy during inference).
    type_map: FxHashMap<AstId, InferTy>,
    /// Accumulated coercions, each through the cast function at its
    /// call type.
    coercions: Vec<PendingCoercion>,
    calls: FxHashMap<AstId, CallChoice>,
    /// How each `a[i]` reaches its element (RFC-0047), keyed by the index
    /// expression's own id.
    index_access: FxHashMap<AstId, IndexAccess>,
    /// Which head each `for` under check was written with (RFC-0057).
    for_kinds: FxHashMap<AstId, ForKind>,
    /// How many loops enclose the statement under check: `break` and
    /// `continue` name the innermost one and are refused where there is
    /// none (RFC-0057 rule 4).
    loop_depth: usize,
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
    reads_under_open_head: Vec<ReadUnderOpenHead>,
    list_pattern_lengths: Vec<ListPatternLength>,
    /// Drained by `check_value_sites`.
    value_sites: Vec<ValueSite>,
    /// Accumulated errors.
    errors: Vec<MirError>,
    /// What the expression under check is read for. One field, so
    /// "borrowed" and "at which mutability" cannot disagree: `a[i]`
    /// settles `as_slice` or `as_slice_mut` on it (RFC-0047 rule 3).
    demand: PlaceDemand,
    /// Every context the program names, at the span of its first use. A
    /// context's type must be data (RFC-0014); the check runs once the
    /// types are known.
    context_uses: FxHashMap<QualifiedRef, Span>,
    /// Bounded variables instantiated so far, each at the span that will
    /// report a violation.
    bound_sites: Vec<BoundSite>,
    /// Bounded effect variables instantiated so far, as `bound_sites`.
    effect_bound_sites: Vec<EffectBoundSite>,
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
    open_decisions: Vec<OpenDecision>,
    refused_open: FxHashSet<crate::ty::TypeBoundId>,
    casts: Vec<CastSite>,
    passing: FxHashMap<AstId, Passing>,
    source_modes: FxHashMap<AstId, SourceMode>,
    /// Places lent whole, whose passing is read off their settled types.
    lent_places: Vec<LentWhole>,
    place_bases: FxHashMap<AstId, WrittenBase>,
    /// Conversion decisions registered so far, at their sites.
    conversions: Vec<PendingConversion>,
    /// What a settle inside the body refused, reported with the solve's own:
    /// a type is written out only once its identities are minted.
    refused_in_body: Vec<Unsettled>,
    /// Every `a[i]`, kept for the refusals that can only name their type
    /// once the body is solved (RFC-0047 rules 3 and 5).
    index_uses: Vec<IndexUse>,
    /// Every read by value that may copy out of what a reference names, at
    /// the read's own id; drained by `check_value_reads`.
    value_reads: FxHashMap<AstId, ValueRead>,
    /// RFC-0047 rule 6, drained by `settle_slice_args`.
    slice_args: Vec<SliceArg>,
    /// RFC-0043 rule 2, drained by `settle_held_arguments`.
    held_arguments: Vec<(DecisionId, HeldArgument<S>)>,
    /// Decisions instantiated so far, each at the span that will report a
    /// failure.
    decision_sites: FxHashMap<DecisionId, Span>,
    /// The operator each of its call's instance decisions was opened for:
    /// a failure of one is that operator's refusal.
    operator_decisions: FxHashMap<DecisionId, OperatorSite>,
    /// The operator each of its operands' head decisions was opened for:
    /// a failure of one is that operator's mismatch.
    operand_decisions: FxHashMap<DecisionId, &'static str>,
    requirer_of: FxHashMap<DecisionId, DecisionId>,
    /// The declaration whose scheme opened each instance decision: the
    /// callee for a call's own instance, and the declaration that carries
    /// the requirement for a requirement's instance. A refusal names it.
    decision_callees: FxHashMap<DecisionId, QualifiedRef>,
    /// Where a source begins if this instance decision is what mints it.
    source_begins_by_decision: FxHashMap<DecisionId, Span>,
    inputs: Inputs,
    lambda_stack: Vec<LambdaScope>,
    lambda_captures: FxHashMap<AstId, Vec<(Astr, CaptureSource)>>,
    capture_moves: Vec<CaptureMove>,
    /// Maps lambda expression AstId -> body expression AstId.
    /// Effect variable of the body being checked; every call raises its lower bound.
    body_effect: EffectTerm<Infer>,
    probe: Option<AstId>,
    probed: Option<Probed<S>>,
    /// The type checking recorded at the marked node: a fresh variable
    /// where the name it holds, which nothing binds, would be poison.
    marked_ty: Option<InferTy>,
}

impl TypeChecker<'_, '_, '_, Clean> {
    /// What the lowering reads of a checked body.
    fn into_resolution(mut self, body: Span, tail_ty: Ty, flows: Flows) -> Checked {
        // A `$` input nothing typed is refused here, in its own words, so it
        // is refused before the freezes below take every type as closed.
        let extern_params = self.frozen_extern_params(body);
        if !self.errors.is_empty() {
            return self.refused(Err);
        }
        let type_map = self.freeze_type_map();
        let try_returns = self.frozen_try_returns();
        let effect = self.close_body_effect();
        let context_types = self.named_context_types();
        let calls = self.frozen_calls();
        let coercion_map = self.frozen_coercions();
        let lambda_captures = self.frozen_lambda_captures();
        let place_bases = self.frozen_place_bases(&type_map);
        self.check_value_reads(&type_map, &place_bases);
        self.check_conversion_places(&place_bases);
        let passing = self.frozen_passing(&type_map);
        let pattern_modes = self.frozen_pattern_modes();
        // A second gate, because freezing is itself a check: a type the
        // solve left open is found only where it is closed, and every such
        // type is closed above.
        if !self.errors.is_empty() {
            return self.refused(Err);
        }
        let view = self.view(type_map.clone(), |ty| self.closed(ty));
        let probe = self.probe_product();
        let resolution = Freeze::new(TypeResolution {
            type_map,
            coercion_map,
            calls,
            index_access: self.index_access,
            for_kinds: self.for_kinds,
            passing,
            pattern_modes,
            place_bases,
            try_returns,
            tail_ty,
            extern_params,
            lambda_captures,
            effect,
            context_types,
            flows,
        });
        Checked {
            resolution: Ok(resolution),
            view,
            probe,
        }
    }
}

impl<'a, 's, 'src, S> TypeChecker<'a, 's, 'src, S>
where
    S: Slot,
{
    pub fn new(
        interner: &'a Interner,
        env: &'a TypeEnv,
        solver: &'s mut Solver<'src>,
        inputs: Inputs,
    ) -> Self {
        let body_effect = solver.fresh_effect_var();
        Self {
            interner,
            scopes: vec![FxHashMap::default()],
            resolved: Resolutions::default(),
            candidate_binder_of: FxHashMap::default(),
            binder_types: FxHashMap::default(),
            env,
            namespace: None,
            param_types: smallvec::smallvec![],
            bound_inputs: Vec::new(),
            solver,
            type_map: FxHashMap::default(),
            coercions: Vec::new(),
            calls: FxHashMap::default(),
            index_access: FxHashMap::default(),
            for_kinds: FxHashMap::default(),
            loop_depth: 0,
            pattern_mode: PatternMode::Value,
            deferred_bindings: Vec::new(),
            deferred_context_binds: Vec::new(),
            context_binds_under_open_head: Vec::new(),
            reads_under_open_head: Vec::new(),
            list_pattern_lengths: Vec::new(),
            value_sites: Vec::new(),
            demand: PlaceDemand::Value,
            bound_sites: Vec::new(),
            effect_bound_sites: Vec::new(),
            branch_mismatches: Vec::new(),
            return_ty: None,
            try_sites: FxHashMap::default(),
            int_literals: Vec::new(),
            open_decisions: Vec::new(),
            refused_open: FxHashSet::default(),
            casts: Vec::new(),
            passing: FxHashMap::default(),
            source_modes: FxHashMap::default(),
            lent_places: Vec::new(),
            place_bases: FxHashMap::default(),
            conversions: Vec::new(),
            refused_in_body: Vec::new(),
            index_uses: Vec::new(),
            value_reads: FxHashMap::default(),
            slice_args: Vec::new(),
            held_arguments: Vec::new(),
            decision_sites: FxHashMap::default(),
            operator_decisions: FxHashMap::default(),
            operand_decisions: FxHashMap::default(),
            requirer_of: FxHashMap::default(),
            decision_callees: FxHashMap::default(),
            source_begins_by_decision: FxHashMap::default(),
            errors: Vec::new(),
            context_uses: FxHashMap::default(),
            inputs,
            lambda_stack: Vec::new(),
            lambda_captures: FxHashMap::default(),
            capture_moves: Vec::new(),
            body_effect,
            probe: None,
            probed: None,
            marked_ty: None,
        }
    }

    pub fn with_body_effect(mut self, effect: EffectTerm<Infer>) -> Self {
        self.body_effect = effect;
        self
    }

    /// Mark the node whose name completion asks about: an identifier, a
    /// context, or a field access or method call whose name is the marker.
    /// Checking is unchanged but for the marked node's type, a fresh
    /// variable where it would be poison, and `Checked::probe` carries what
    /// it saw there.
    pub fn with_probe(mut self, marker: AstId) -> Self {
        self.probe = Some(marker);
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

    /// A context access of the body: joined into its effect (RFC-0025 rule 5).
    fn note_access(&mut self, access: Effect, span: Span) {
        self.note_call_effect(&EffectTerm::Known(access), span);
    }

    fn close_body_effect(&self) -> Effect {
        self.solver.freeze_effect(&self.body_effect)
    }

    /// Set the namespace for context lookups.
    pub fn with_namespace(mut self, namespace: Option<Astr>) -> Self {
        self.namespace = namespace;
        self
    }

    /// Bind the parameters the function's declaration names: a `$name` in the
    /// body names the declared parameter of that name, and the declared order
    /// is the order the call passes its arguments in.
    pub fn with_declared_params(mut self, declared: Vec<ParamTerm<Infer>>) -> Self {
        self.param_types
            .extend(declared.into_iter().map(|param| ExternParam {
                name: param.name,
                ty: param.ty,
                first_read: None,
                origin: ParamOrigin::Declared,
            }));
        self
    }

    /// Give the `$` names a host has already bound the type their value
    /// has (RFC-0087 rule 2), so the body is checked against the value it
    /// will hold rather than against an open variable. A name the
    /// declaration already bound keeps the declared type.
    pub fn with_bound_inputs(mut self, bindings: &Bindings) -> Self {
        for (name, value) in bindings.iter() {
            if self.param_types.iter().any(|p| p.name == name) {
                continue;
            }
            let Typed { ty, deferred } = type_bound(self.solver, value)
                .expect("`Bindings::bind` holds only a value that types (RFC-0087 rule 3)");
            self.param_types.push(ExternParam {
                name,
                ty,
                first_read: None,
                origin: ParamOrigin::Bound,
            });
            self.bound_inputs.push(BoundInput {
                name,
                value: value.clone(),
                deferred,
                first_read: None,
            });
        }
        self
    }

    /// Each bound input's enum positions join the variants its value holds,
    /// now that the body's uses have typed them (RFC-0087 rule 4).
    fn join_bound_variants(&mut self) {
        for bound in std::mem::take(&mut self.bound_inputs) {
            for DeferredJoin { var, variant } in &bound.deferred {
                let uses = self.type_as_written(var);
                if self.solver.unify(variant, var).is_ok() {
                    continue;
                }
                let at = bound.first_read.expect(
                    "a join fails only on a variable a read constrained, and the read recorded its span",
                );
                self.error(
                    MirErrorKind::BindingTypeMismatch {
                        name: self.interner.resolve(bound.name).to_string(),
                        value: bound.value.clone(),
                        ty: uses,
                    },
                    at,
                );
                break;
            }
        }
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
            .map(|(id, ty)| (*id, self.closed(ty)))
            .collect()
    }

    /// A type the solve closed. Freezing runs once no refusal was raised,
    /// and every variable left open is a decision `report_unsettled`
    /// refused, so a type that does not close is a checker defect.
    fn closed(&self, ty: &InferTy) -> Ty {
        let resolved = self.solver.resolve_ty(ty);
        match self.solver.close_ty(&resolved) {
            Ok(closed) => closed,
            Err(open) => panic!("an admitted body left {resolved:?} open: {open:?}"),
        }
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
    pub fn check_template(mut self, template: &Template<S>) -> Checked<S::Resolution>
    where
        S: Checks,
    {
        for stmt in &template.body {
            self.check_stmt(stmt);
        }
        self.solve_body();
        self.check_moves_out_of_captures();
        self.check_context_binds_under_open_head();
        self.check_contexts_are_data();
        let flows = self.infer_flows(body_flows::Body {
            stmts: &template.body,
            tail: None,
        });
        S::conclude(self, template.span, BodyTail::Text, flows)
    }

    /// Type check a script. Consumes self, returns TypeResolution.
    /// `expected_tail`: if provided, the script's tail expression is unified with this type.
    pub fn check_script(
        mut self,
        script: &acvus_ast::Script<S>,
        expected_tail: Option<InferTy>,
        crossing: ResultCrossing,
    ) -> Checked<S::Resolution>
    where
        S: Checks,
    {
        // A declared `ret` is the body's return type itself, so every
        // `return` joins against it exactly as the tail does; undeclared, it
        // is the fresh variable the tail resolves.
        let return_ty = match expected_tail {
            Some(declared) => declared,
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
            self.value_sites.push(ValueSite {
                at: ValuePosition::Result(ty.clone(), crossing),
                span: tail.span(),
            });
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
        self.check_moves_out_of_captures();
        self.check_context_binds_under_open_head();
        self.check_contexts_are_data();
        let flows = self.infer_flows(body_flows::Body {
            stmts: &script.stmts,
            tail: script.tail.as_deref(),
        });
        let at = script.tail.as_ref().map_or(script.span, |tail| tail.span());
        S::conclude(
            self,
            script.span,
            BodyTail::Value { ty: tail_ty, at },
            flows,
        )
    }

    fn refused<R, F>(mut self, resolution: F) -> Checked<R>
    where
        F: FnOnce(Vec<MirError>) -> R,
    {
        let written = self
            .type_map
            .iter()
            .map(|(id, ty)| (*id, self.type_as_written(ty)))
            .collect();
        let view = self.view(written, |ty| self.type_as_written(ty));
        let probe = self.probe_product();
        Checked {
            resolution: resolution(self.reported()),
            view,
            probe,
        }
    }

    fn probe_product(&self) -> Option<ProbeProduct> {
        let expected = self.expected_at_marker();
        let written = expected.as_ref().map(|ty| self.type_as_written(ty));
        Some(match self.probed.as_ref()? {
            Probed::Value {
                marker,
                site,
                scope,
            } => {
                let fits = |ty: &InferTy| {
                    expected
                        .as_ref()
                        .is_some_and(|expected| self.flows_on_trial(ty, expected, *site))
                };
                let mut contexts: Vec<&QualifiedRef> = self.env.contexts.keys().collect();
                contexts.sort_by(|a, b| self.written_order(a, b));
                let mut functions: Vec<&QualifiedRef> = self.env.functions.keys().collect();
                functions.sort_by(|a, b| self.written_order(a, b));
                ProbeProduct::Value {
                    expected: written,
                    scope: scope
                        .iter()
                        .map(|in_scope| Visible {
                            name: in_scope.name,
                            binder: in_scope.binder,
                            ty: self.type_as_written(&in_scope.ty),
                            fits: fits(&in_scope.ty),
                        })
                        .collect(),
                    inputs: self
                        .param_types
                        .iter()
                        .filter(|param| param.name != *marker)
                        .map(|param| ProbedInput {
                            name: param.name,
                            ty: self.type_as_written(&param.ty),
                            fits: fits(&param.ty),
                        })
                        .collect(),
                    contexts: contexts
                        .into_iter()
                        .map(|qref| DeclarationFit {
                            qref: *qref,
                            fits: fits(&self.env.contexts[qref]),
                        })
                        .collect(),
                    functions: functions
                        .into_iter()
                        .map(|qref| DeclarationFit {
                            qref: *qref,
                            fits: expected.as_ref().is_some_and(|expected| {
                                let scheme = &self.env.functions[qref];
                                self.call_returns_on_trial(*qref, scheme, expected, *site)
                            }),
                        })
                        .collect(),
                }
            }
            Probed::Member(member) => {
                let (ProbedReceiver::Place(owned) | ProbedReceiver::Value(owned)) =
                    &member.receiver;
                let resolved = self.solver.resolve_ty(owned);
                let mut fields: Vec<ProbedField> = match referent_of(&resolved).as_ref() {
                    TyTerm::Object(object) => object
                        .iter()
                        .filter(|(name, _)| **name != member.marker)
                        .map(|(name, ty)| ProbedField {
                            name: *name,
                            ty: self.type_as_written(ty),
                            fits: expected.as_ref().is_some_and(|expected| {
                                self.flows_on_trial(ty, expected, member.marker_site)
                            }),
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                fields.sort_by(|a, b| {
                    self.interner
                        .resolve(a.name)
                        .cmp(self.interner.resolve(b.name))
                });
                ProbeProduct::Member {
                    receiver: self.type_as_written(owned),
                    expected: written,
                    fields,
                    methods: self.methods_receiving(member, expected.as_ref()),
                }
            }
        })
    }

    /// The type the solve gave the marked node; `None` where it left it
    /// open, or poison, which joins with anything.
    fn expected_at_marker(&self) -> Option<InferTy> {
        let expected = self.solver.resolve_ty(self.marked_ty.as_ref()?);
        (!matches!(expected, TyTerm::Var(_) | TyTerm::Error(_))).then_some(expected)
    }

    /// By name as written, then by namespace.
    fn written_order(&self, a: &QualifiedRef, b: &QualifiedRef) -> std::cmp::Ordering {
        let name = |qref: &QualifiedRef| self.interner.resolve(qref.name);
        let namespace = |qref: &QualifiedRef| qref.namespace.map(|ns| self.interner.resolve(ns));
        name(a)
            .cmp(name(b))
            .then_with(|| namespace(a).cmp(&namespace(b)))
    }

    fn methods_receiving(
        &self,
        member: &ProbedMember<S>,
        expected: Option<&InferTy>,
    ) -> Vec<ProbedMethod> {
        let mut names: Vec<Astr> = self
            .env
            .functions
            .keys()
            .chain(self.env.machine.keys())
            .map(|qref| qref.name)
            .collect();
        names.sort_by(|a, b| self.interner.resolve(*a).cmp(self.interner.resolve(*b)));
        names.dedup();
        names
            .into_iter()
            .filter_map(|name| {
                let mut by_arity: BTreeMap<usize, Vec<SignatureCandidate>> = BTreeMap::new();
                for candidate in self.declared_signatures(QualifiedRef::root(name)) {
                    let Some(arity) = candidate.arity().filter(|arity| *arity >= 1) else {
                        continue;
                    };
                    by_arity.entry(arity).or_default().push(candidate);
                }
                let admitted: Vec<SignatureCandidate> = by_arity
                    .into_values()
                    .flat_map(|taking| self.admitted_receiving(taking, member, name))
                    .collect();
                let fits = expected.is_some_and(|expected| {
                    admitted
                        .iter()
                        .any(|candidate| self.receives_on_trial(candidate, member, name, expected))
                });
                let admitted: Vec<AdmittedDeclaration> = admitted
                    .into_iter()
                    .filter_map(|candidate| match candidate {
                        SignatureCandidate::Named { qref, scheme } => Some(AdmittedDeclaration {
                            qref,
                            ty: scheme.ty,
                        }),
                        SignatureCandidate::Local { .. } => None,
                    })
                    .collect();
                (!admitted.is_empty()).then_some(ProbedMethod {
                    name,
                    admitted,
                    fits,
                })
            })
            .collect()
    }

    fn admitted_receiving(
        &self,
        candidates: Vec<SignatureCandidate>,
        member: &ProbedMember<S>,
        name: Astr,
    ) -> Vec<SignatureCandidate> {
        match Signatures::of(candidates) {
            None => Vec::new(),
            Some(Signatures::One(one)) => {
                // Admission stays in front of the trial, as a method call
                // admits its receiver against the first parameter before
                // it admits the call.
                let admitted = self.admitted_at_receiver(vec![one.clone()], &member.receiver);
                match !admitted.is_empty() && self.one_takes_receiver(&one, member, name) {
                    true => admitted,
                    false => Vec::new(),
                }
            }
            Some(Signatures::Several(several)) => {
                self.admitted_at_receiver(several, &member.receiver)
            }
        }
    }

    fn one_takes_receiver(
        &self,
        candidate: &SignatureCandidate,
        member: &ProbedMember<S>,
        name: Astr,
    ) -> bool {
        let call = member.call_named(name);
        self.accepted_on_trial(|trial| {
            trial.call_on_receiver(candidate, member, &call);
            true
        })
    }

    /// A call of `candidate` on the probed receiver with its other
    /// arguments unwritten, admitted as a method call admits one: the
    /// type it returns.
    fn call_on_receiver(
        &mut self,
        candidate: &SignatureCandidate,
        member: &ProbedMember<S>,
        call: &NamedCall<'_, S>,
    ) -> InferTy {
        let mode = self.solver.receiver_mode(candidate);
        let written = &member.receiver_as_written;
        let first = match &member.receiver {
            ProbedReceiver::Place(owned) => {
                self.receiver_in(written, owned.clone(), mode, call.span)
            }
            ProbedReceiver::Value(ty) => {
                self.value_receiver_in(written, ty.clone(), mode, call.span)
            }
        };
        self.admit_call(Signatures::One(candidate.clone()), Some(first), call)
    }

    /// Whether a value of type `value` written at the marker joins the type
    /// the position expects, as a value joins at its use.
    fn flows_on_trial(&self, value: &InferTy, expected: &InferTy, site: CalleeSite) -> bool {
        self.accepted_on_trial(|trial| {
            trial
                .flow(value, expected, ConversionSite::value_at(site))
                .is_ok()
        })
    }

    /// Whether a call of `qref` written at the marker, its arguments still
    /// unwritten, returns a type that joins the one the position expects.
    fn call_returns_on_trial(
        &self,
        qref: QualifiedRef,
        scheme: &crate::ty::Scheme,
        expected: &InferTy,
        site: CalleeSite,
    ) -> bool {
        let call = NamedCall {
            callee: site,
            name: qref.name,
            args: &[],
            span: site.span,
            refused_as: CallRefusal::NoMatchingFunction,
        };
        let candidate = SignatureCandidate::Named {
            qref,
            scheme: scheme.clone(),
        };
        self.accepted_on_trial(|trial| {
            let returned = trial.admit_call(Signatures::One(candidate), None, &call);
            trial
                .flow(&returned, expected, ConversionSite::value_at(site))
                .is_ok()
        })
    }

    /// Whether a call of `candidate` on the probed receiver returns a type
    /// that joins the one the member's position expects.
    fn receives_on_trial(
        &self,
        candidate: &SignatureCandidate,
        member: &ProbedMember<S>,
        name: Astr,
        expected: &InferTy,
    ) -> bool {
        let call = member.call_named(name);
        self.accepted_on_trial(|trial| {
            let returned = trial.call_on_receiver(candidate, member, &call);
            trial
                .flow(
                    &returned,
                    expected,
                    ConversionSite::value_at(member.marker_site),
                )
                .is_ok()
        })
    }

    /// Whether what `admit` adds to a copy of the solve is refused
    /// nowhere: `admit` answers its own join, no refusal is raised, no
    /// decision it opens fails when the copy settles, and no variable it
    /// bounds is out of its bound, as `solve_body` verifies one. Every
    /// offer the probe makes on a trial is held to this one verdict.
    ///
    /// `settle`, not `solve`: the arguments of a call written at the
    /// marker are absent, and closing what they would fix by its least
    /// element would refuse calls that those arguments complete.
    fn accepted_on_trial<F>(&self, admit: F) -> bool
    where
        F: for<'t, 'u, 'v> FnOnce(&mut TypeChecker<'t, 'u, 'v, S>) -> bool,
    {
        self.solver.trial(|solver| {
            let opened = solver.decisions_opened();
            let mut trial = TypeChecker::new(self.interner, self.env, solver, self.inputs)
                .with_namespace(self.namespace);
            trial.type_map = self.type_map.clone();
            admit(&mut trial)
                && trial.errors.is_empty()
                && trial
                    .solver
                    .settle()
                    .iter()
                    .all(|failure| (failure.decision().0 as usize) < opened)
                && trial.bound_sites.iter().all(|site| {
                    !matches!(
                        trial.solver.close_ty(&TyTerm::Var(site.var)),
                        Err(crate::ty::FreezeError::OutOfBound { .. })
                    )
                })
        })
    }

    fn admitted_at_receiver(
        &self,
        candidates: Vec<SignatureCandidate>,
        receiver: &ProbedReceiver,
    ) -> Vec<SignatureCandidate> {
        match receiver {
            ProbedReceiver::Place(owned) => {
                let Some(referent) = self.named_referent(owned, Mutability::Shared) else {
                    return Vec::new();
                };
                candidates
                    .into_iter()
                    .filter(|candidate| {
                        !matches!(
                            self.receiver_admission(candidate, owned, &referent)
                                .admission,
                            Admission::Refused
                        )
                    })
                    .collect()
            }
            ProbedReceiver::Value(ty) => {
                let agreed = agreed_receiver_mode(
                    candidates
                        .iter()
                        .map(|candidate| self.solver.receiver_mode(candidate)),
                );
                let Some(handed) = self.value_receiver_handed(ty, lent_only_if_agreed(agreed))
                else {
                    return Vec::new();
                };
                candidates
                    .into_iter()
                    .filter(|candidate| {
                        !matches!(
                            self.solver.admits(candidate, 0, &handed),
                            Admission::Refused
                        )
                    })
                    .collect()
            }
        }
    }

    fn probe_scope(&mut self, id: AstId, span: Span, marker: Astr) {
        if self.probe != Some(id) {
            return;
        }
        let mut scope: Vec<InScope> = Vec::new();
        for bound_here in self.scopes.iter().rev() {
            for (&name, bound) in bound_here {
                if scope.iter().all(|in_scope| in_scope.name != name) {
                    scope.push(InScope {
                        name,
                        binder: bound.binder,
                        ty: bound.ty.clone(),
                    });
                }
            }
        }
        scope.sort_by(|a, b| {
            self.interner
                .resolve(a.name)
                .cmp(self.interner.resolve(b.name))
        });
        self.probed = Some(Probed::Value {
            marker,
            site: CalleeSite { id, span },
            scope,
        });
    }

    fn probe_receiver(
        &mut self,
        marker_site: CalleeSite,
        marker: Astr,
        receiver: &Expr<S>,
        ty: &InferTy,
    ) {
        if self.probe != Some(marker_site.id) {
            return;
        }
        let probed = match names_a_place(receiver) {
            true => ProbedReceiver::Place(ty.clone()),
            false => ProbedReceiver::Value(ty.clone()),
        };
        self.probed = Some(Probed::Member(ProbedMember {
            marker,
            receiver: probed,
            receiver_as_written: receiver.clone(),
            marker_site,
        }));
    }

    /// A method call whose name is the marker: its receiver is checked for
    /// the probe, and the name, which no function has, is refused as
    /// `check_method_call` refuses it.
    fn check_probed_method_call(
        &mut self,
        id: AstId,
        receiver: &Expr<S>,
        name: Astr,
        args: &[Expr<S>],
        call_span: Span,
    ) -> InferTy {
        let owned = self.check_expr(receiver);
        let marker_site = CalleeSite {
            id,
            span: call_span,
        };
        self.probe_receiver(marker_site, name, receiver, &owned);
        self.check_unadmitted_args(None, args);
        self.undefined_function(QualifiedRef::root(name), call_span)
    }

    fn view<F>(&self, mut types: FxHashMap<AstId, Ty>, bound: F) -> Freeze<BodyView>
    where
        F: Fn(&InferTy) -> Ty,
    {
        for (&binder, ty) in &self.binder_types {
            types.entry(binder).or_insert_with(|| bound(ty));
        }
        let mut resolved = self.resolved.clone();
        for (&callee, choice) in &self.calls {
            let settled = match choice {
                CallChoice::Resolved(resolved) => Some(SettledCallee::Declared(resolved.qref)),
                CallChoice::Decided(decision) => match self.solver.answer(*decision) {
                    Some(Answer::Signature {
                        settled: SettledSignature::Named { qref, .. },
                        ..
                    }) => Some(SettledCallee::Declared(qref)),
                    Some(Answer::Signature {
                        settled: SettledSignature::Local,
                        ..
                    }) => Some(SettledCallee::Binding),
                    None => None,
                    Some(
                        Answer::Instance(_)
                        | Answer::Conversion(_)
                        | Answer::Lend(_)
                        | Answer::Capture(_)
                        | Answer::Match(_),
                    ) => unreachable!("a signature decision answers with a signature"),
                },
                CallChoice::Binding => Some(SettledCallee::Binding),
                CallChoice::StructuralVariant | CallChoice::Operator(_) => None,
            };
            match settled {
                Some(SettledCallee::Declared(qref)) => {
                    resolved.record(callee, Resolved::Function(qref));
                }
                Some(SettledCallee::Binding) => {
                    let binder = self
                        .candidate_binder_of
                        .get(&callee)
                        .expect("a call settles on a binding only among the candidates it found");
                    resolved.record(callee, Resolved::Local(*binder));
                }
                None => {}
            }
        }
        Freeze::new(BodyView {
            types,
            resolved: resolved.0,
        })
    }

    fn frozen_place_bases(&self, type_map: &TypeMap) -> FxHashMap<AstId, PlaceBase> {
        self.place_bases
            .iter()
            .map(|(&id, &written)| {
                let is_a_reference = || {
                    let ty = type_map
                        .get(&id)
                        .expect("a place's base is checked before it is noted");
                    matches!(ty, Ty::Ref(..))
                };
                let base = match written {
                    WrittenBase::Storage(storage) if is_a_reference() => {
                        PlaceBase::ThroughReferenceIn(storage)
                    }
                    WrittenBase::Storage(storage) => PlaceBase::Storage(storage),
                    WrittenBase::Element => {
                        let access = self
                            .index_access
                            .get(&id)
                            .expect("type checking settles every index expression");
                        PlaceBase::Element(IndexAccess {
                            mode: IndexMode::Ref,
                            ..*access
                        })
                    }
                    WrittenBase::Value if is_a_reference() => PlaceBase::ThroughReference,
                    WrittenBase::Value => PlaceBase::Temporary,
                };
                (id, base)
            })
            .collect()
    }

    /// A read by value out of what a reference names copies, so its type
    /// must be one that does (RFC-0018 rules 2 and 4). Checked here, where
    /// a place's base is the one the lowering reads, so a type or a base
    /// still open where the read was written is checked as it settled.
    fn check_value_reads(&mut self, type_map: &TypeMap, place_bases: &FxHashMap<AstId, PlaceBase>) {
        for (read, value_read) in std::mem::take(&mut self.value_reads) {
            let (through, span) = match value_read {
                ValueRead::Deref { span } => (ReadThrough::Deref, span),
                ValueRead::Place { base, span } => {
                    let base = place_bases
                        .get(&base)
                        .expect("a place read by value notes its base");
                    if !base.is_through_a_reference() {
                        continue;
                    }
                    (ReadThrough::Field, span)
                }
            };
            let ty = type_map
                .get(&read)
                .expect("a read is checked before it is noted");
            if through.copies(ty) == Some(false) {
                self.error(through.refusal(ty.clone()), span);
            }
        }
    }

    /// RFC-0041: a conversion answered through the reference takes its
    /// value out of the place the argument borrows and restores it after
    /// the call, so that place is storage the body owns directly or a
    /// temporary. Checked here, where every conversion is answered, the
    /// held ones included, and every place's base is settled.
    fn check_conversion_places(&mut self, place_bases: &FxHashMap<AstId, PlaceBase>) {
        let conversions = std::mem::take(&mut self.conversions);
        for conversion in &conversions {
            let Some(Answer::Conversion(Conversion::ThroughRef { .. })) =
                self.solver.answer(conversion.decision)
            else {
                continue;
            };
            let base = conversion
                .base
                .map(|base| place_bases.get(&base).expect("a lent place notes its base"));
            match base {
                Some(PlaceBase::Storage(_) | PlaceBase::Temporary) => {}
                Some(
                    PlaceBase::ThroughReferenceIn(_)
                    | PlaceBase::ThroughReference
                    | PlaceBase::Element(_),
                )
                | None => {
                    let from = self.solver.resolve_ty(&conversion.from);
                    let to = self.solver.resolve_ty(&conversion.to);
                    self.error(
                        MirErrorKind::ConversionNeedsPlace {
                            from: self.type_as_written(&from),
                            to: self.type_as_written(&to),
                        },
                        conversion.site.span,
                    );
                }
            }
        }
        self.conversions = conversions;
    }

    fn frozen_pattern_modes(&self) -> FxHashMap<AstId, MatchMode> {
        self.source_modes
            .iter()
            .map(|(&source, &mode)| {
                let mode = match mode {
                    SourceMode::Read(mode) => mode,
                    SourceMode::Decided(decision) => match self.solver.answer(decision) {
                        Some(Answer::Match(mode)) => mode,
                        other => panic!("a match decision settles before freezing, not {other:?}"),
                    },
                };
                (source, mode)
            })
            .collect()
    }

    fn frozen_passing(&mut self, type_map: &TypeMap) -> FxHashMap<AstId, Passing> {
        let mut passing = std::mem::take(&mut self.passing);
        for LentWhole { place, mutability } in &self.lent_places {
            let ty = type_map
                .get(place)
                .expect("a lent place is checked before it is noted");
            // A place that holds a shared reference is lent as that
            // reference: the lend reborrows nothing. A `&mut` is reborrowed,
            // so exclusion sees the lend (RFC-0029).
            let lent = match ty {
                Ty::Ref(Mutability::Shared, _) => Passing::AsIs,
                _ => Passing::Lent(*mutability),
            };
            passing.insert(*place, lent);
        }
        passing
    }

    fn note_place(&mut self, expr: &Expr<S>) {
        let base = projected(expr).base;
        self.place_bases.insert(base.id(), WrittenBase::of(base));
    }

    fn note_value_read(&mut self, place: &Expr<S>) {
        let path = projected(place);
        if path.fields.is_empty() {
            return;
        }
        self.value_reads.insert(
            place.id(),
            ValueRead::Place {
                base: path.base.id(),
                span: place.span(),
            },
        );
    }

    fn pass_receiver(&mut self, receiver: &Expr<S>, passing: Passing) {
        if let Passing::Lent(_) = passing {
            self.note_place(receiver);
        }
        self.passing.insert(receiver.id(), passing);
    }

    /// A value flows into a position that must have its type (RFC-0042
    /// rule 1): the two join at a value position. A join whose one
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
                reason: MismatchReason::ReprOpen(_) | MismatchReason::Weakens,
                ..
            }) => {
                self.convert_at(value_ty, expected_ty, site);
                Ok(())
            }
            joined => joined,
        }
    }

    /// A value meets a type it may need converting to (RFC-0042 rule 4): the
    /// conversion is a decision the body's solve answers, reported at the
    /// site if none exists.
    fn convert_at(&mut self, value_ty: &InferTy, expected_ty: &InferTy, site: ConversionSite) {
        let decision = self.decide_conversion(value_ty, expected_ty, site.span);
        self.conversions.push(PendingConversion {
            site,
            decision,
            from: value_ty.clone(),
            to: expected_ty.clone(),
            base: None,
        });
    }

    /// `convert_at` for a call argument: a conversion answered through the
    /// reference takes out the value the argument borrows.
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
            base: arg.borrowed.as_ref().map(|borrowed| borrowed.base),
        });
    }

    /// How an object type refused a join, for the join that refused it and no
    /// other: a declared struct's field set disagreeing (RFC-0042), or a union
    /// over `ObjectTy::MAX_FIELDS`.
    fn object_refusal(&self, mismatch: &Mismatch) -> Option<MirErrorKind> {
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
            MismatchReason::ObjectTooWide { fields } => Some(MirErrorKind::ObjectTooWide {
                fields,
                most: ObjectTy::<Infer>::MAX_FIELDS,
            }),
            MismatchReason::FixedLacks { member } => Some(MirErrorKind::FixedLacks {
                member: shown(member),
            }),
            MismatchReason::NoJoin
            | MismatchReason::ReprOpen(_)
            | MismatchReason::TaskTooHigh { .. }
            | MismatchReason::Weakens => None,
        }
    }

    /// An argument meets its parameter (RFC-0042 rules 1 and 4).
    fn meet_argument(
        &mut self,
        arg_ty: &InferTy,
        param_ty: &InferTy,
        site: &ArgSite,
        reach: Reach,
    ) {
        let joins = match reach {
            Reach::Viewed(viewed) => {
                match self.coerce_viewed(arg_ty, param_ty, site, viewed) {
                    SliceCoercion::Coerced => {}
                    SliceCoercion::NoDeclaration => {
                        self.meet_settled_argument(arg_ty, param_ty, site.span)
                    }
                }
                return;
            }
            Reach::Joined => true,
            Reach::Converted => false,
        };
        if self.meet_slice_parameter(arg_ty, param_ty, site) {
            return;
        }
        if self.refused_projection_parameter(param_ty, arg_ty, site) {
            return;
        }
        if joins && self.refused_field_set(param_ty, arg_ty, site.span) {
            return;
        }
        self.convert_argument_at(arg_ty, param_ty, site);
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
        let TyTerm::Object(projection) = self.solver.shallow_resolve_ty(&borrowed.ty()) else {
            return false;
        };
        let TyTerm::Ref(_, referent) = self.solver.shallow_resolve_ty(arg_ty) else {
            return false;
        };
        let TyTerm::Object(argument) = self.solver.shallow_resolve_ty(&referent.ty()) else {
            return false;
        };
        let Some(field) = projection.borrowed_field_missing_from(&argument) else {
            return false;
        };
        let borrows = self.object_as_written(&projection);
        let has = self.object_as_written(&argument);
        let object = match site.borrowed.as_ref().and_then(|lent| lent.loan.as_ref()) {
            Some(loan) => loan.display(self.interner),
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
        let Some(kind) = self.object_refusal(&mismatch) else {
            return false;
        };
        self.error(kind, span);
        true
    }

    /// `&v` at a `&[T]` parameter is the container's own `as_slice` of it,
    /// and `&s` at a `&str` parameter is the `String`'s own `as_str`,
    /// recorded at the argument (RFC-0047 rule 6, RFC-0062 rule 3). A `&v`
    /// at a `&mut [T]` parameter is refused where every other argument
    /// mismatch is; so is a container that declares no `as_slice`, and an
    /// argument already of the parameter's own type unifies as it is.
    fn meet_slice_parameter(
        &mut self,
        arg_ty: &InferTy,
        param_ty: &InferTy,
        site: &ArgSite,
    ) -> bool {
        let TyTerm::Ref(mutability, wanted) = self.solver.shallow_resolve_ty(param_ty) else {
            return false;
        };
        let Some(view) = View::of(&self.solver.shallow_resolve_ty(&wanted.ty())) else {
            return false;
        };
        if view == View::Str && mutability != Mutability::Shared {
            return false;
        }
        let TyTerm::Ref(lent, _) = self.solver.shallow_resolve_ty(arg_ty) else {
            return false;
        };
        if !lent.reaches(mutability) {
            return false;
        }
        match self.coerce_viewed(arg_ty, param_ty, site, Viewed { view, mutability }) {
            SliceCoercion::Coerced => true,
            SliceCoercion::NoDeclaration => false,
        }
    }

    /// The argument's coercion to the view, recorded where the referent's
    /// head is named and held until it is where it is not.
    fn coerce_viewed(
        &mut self,
        arg_ty: &InferTy,
        param_ty: &InferTy,
        site: &ArgSite,
        viewed: Viewed,
    ) -> SliceCoercion {
        if self.solver.lends_an_unnamed_head(arg_ty) {
            self.slice_args.push(SliceArg {
                at: site.id,
                span: site.span,
                arg: arg_ty.clone(),
                param: param_ty.clone(),
                viewed,
            });
            return SliceCoercion::Coerced;
        }
        let TyTerm::Ref(_, container) = self.solver.shallow_resolve_ty(arg_ty) else {
            return SliceCoercion::NoDeclaration;
        };
        let referent = self.solver.resolve_ty(&container.ty());
        let lent_as = TyTerm::Ref(viewed.mutability, container);
        self.slice_coercion(&referent, viewed, &lent_as, param_ty, site.id, site.span)
    }

    /// The one declaration that takes the referent, recorded as the coercion
    /// at `at`. A referent the solve left open is storage, which is never a
    /// view (RFC-0062 rule 1), so it is what the declaration lending the view
    /// takes.
    fn slice_coercion(
        &mut self,
        referent: &InferTy,
        viewed: Viewed,
        arg_ty: &InferTy,
        param_ty: &InferTy,
        at: AstId,
        span: Span,
    ) -> SliceCoercion {
        let Viewed { view, mutability } = viewed;
        let open = matches!(referent, TyTerm::Var(_));
        let takes_referent = |takes: &crate::ty::PolyTy| {
            open || match view {
                View::Str => matches!((takes, referent), (TyTerm::String, TyTerm::String)),
                View::Slice => {
                    sliceable_head(referent).is_some_and(|head| sliceable_head(takes) == Some(head))
                }
            }
        };
        let takers: Vec<(QualifiedRef, crate::ty::Scheme)> = self
            .env
            .machine_views(viewed)
            .into_iter()
            .filter(
                |(_, scheme)| match scheme.params().first().map(|param| &param.ty) {
                    Some(TyTerm::Ref(_, takes)) => takes_referent(&takes.ty()),
                    _ => false,
                },
            )
            .map(|(qref, scheme)| (qref, scheme.clone()))
            .collect();
        let [(qref, scheme)] = takers.as_slice() else {
            return SliceCoercion::NoDeclaration;
        };
        let (qref, scheme) = (*qref, scheme.clone());
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
        SliceCoercion::Coerced
    }

    /// Every argument whose container the solve has now named.
    fn settle_slice_args(&mut self) {
        let deferred = std::mem::take(&mut self.slice_args);
        for SliceArg {
            at,
            span,
            arg,
            param,
            viewed,
        } in deferred
        {
            self.settle_view(at, span, &arg, &param, viewed);
        }
    }

    /// Every argument a signature decision held, met as the candidate it
    /// settled on admits it now that the solve has named its head (RFC-0043
    /// rule 2): a view is the checker's coercion at the argument, a reborrow
    /// is the shared reborrow the argument reaches the parameter as, a
    /// conversion is the conversion decision the known path opens at the
    /// argument (`convert_argument_at`), which the solve that follows
    /// answers, and a direct or refused admission meets the parameter as it
    /// is.
    fn settle_held_arguments(&mut self) {
        let held = std::mem::take(&mut self.held_arguments);
        for (
            decision,
            HeldArgument {
                site,
                index,
                arg,
                param,
                handed,
            },
        ) in held
        {
            let Some(candidate) = self.solver.settled_candidate(decision).cloned() else {
                continue;
            };
            let FirstArg { ty: arg, site } = match handed {
                HeldHanded::AsWritten => FirstArg { ty: arg, site },
                HeldHanded::Receiver { place, referent } => {
                    self.settle_receiver(&candidate, &place, arg, &referent, site.taken_by)
                }
            };
            let (at, span) = (site.id, site.span);
            match self.solver.admits(&candidate, index, &arg) {
                Admission::Viewed(viewed) => self.settle_view(at, span, &arg, &param, viewed),
                Admission::Reborrowed { shared } => {
                    self.meet_settled_argument(&shared, &param, span);
                    self.coercions.push(PendingCoercion {
                        at,
                        cast: PendingCast::Reborrow { shared: param },
                    });
                }
                Admission::Converted => self.convert_argument_at(&arg, &param, &site),
                Admission::Direct | Admission::Refused => {
                    self.meet_settled_argument(&arg, &param, span)
                }
            }
        }
    }

    /// A held receiver taken in the settled candidate's mode, with the
    /// bookkeeping `receiver_in` does for a mode chosen at the call. A head
    /// the solve left open is lent as the lend decision opened where the
    /// receiver was held names it.
    fn settle_receiver(
        &mut self,
        candidate: &SignatureCandidate,
        place: &Expr<S>,
        owned: InferTy,
        referent: &InferTy,
        taken_by: Span,
    ) -> FirstArg {
        let (mode, seen) = self.solver.receiver_seen_by(candidate, &owned, referent);
        let handing = match mode {
            ReceiverMode::Lent(mutability) => Handing::Lent {
                mutability,
                lent: match self.lend_named(&owned, place, mutability, place.span()) {
                    Some(named) => TyTerm::Ref(mutability, Box::new(TypeArg::uniform(named))),
                    None => seen,
                },
            },
            ReceiverMode::Value => Handing::Value,
        };
        self.receiver_handed(place, owned, handing, taken_by)
    }

    fn settle_view(
        &mut self,
        at: AstId,
        span: Span,
        arg: &InferTy,
        param: &InferTy,
        viewed: Viewed,
    ) {
        let referent = match self.solver.shallow_resolve_ty(arg) {
            TyTerm::Ref(_, container) => self.solver.resolve_ty(&container.ty()),
            other => other,
        };
        let open = matches!(referent, TyTerm::Var(_));
        match self.slice_coercion(&referent, viewed, arg, param, at, span) {
            SliceCoercion::Coerced => {}
            SliceCoercion::NoDeclaration if open => {
                let resolved_ty = self.type_as_written(arg);
                self.error(MirErrorKind::AmbiguousType { resolved_ty }, span);
            }
            SliceCoercion::NoDeclaration => self.meet_settled_argument(arg, param, span),
        }
    }

    /// No conversion decision is asked here, and that is a decision, not an
    /// omission: no argument met here is one its candidate converts, and a
    /// conversion answers `Identity` exactly where its two sides unify, so
    /// the argument takes that unification directly. A held argument the
    /// settled candidate converts opens its decision in
    /// `settle_held_arguments`.
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

    /// The head of a type the body is about to read a field off, after the
    /// decisions already open have had their say about it.
    fn head_once_decided(&mut self, ty: &InferTy) -> InferTy {
        let head = self.solver.shallow_resolve_ty(ty);
        if !matches!(head, TyTerm::Var(_)) {
            return head;
        }
        self.settle_in_body();
        self.solver.shallow_resolve_ty(ty)
    }

    fn settle_in_body(&mut self) {
        let refused = self.solver.settle();
        self.refused_in_body.extend(refused);
        self.place_begun_sources();
        self.place_opened_children();
    }

    fn place_opened_children(&mut self) {
        for crate::solver::OpenedChild {
            parent,
            child,
            component,
        } in self.solver.take_opened_children()
        {
            self.requirer_of.insert(child, parent);
            if component.is_some()
                && let Some(site) = self.operator_decisions.get(&parent).copied()
            {
                self.operator_decisions.insert(child, site);
            }
        }
    }

    fn place_instance_bounds(&mut self) {
        for crate::solver::InstanceBound { decision, var } in self.solver.take_instance_bounds() {
            let span = self.decision_span(decision);
            self.effect_bound_sites.push(EffectBoundSite { var, span });
        }
    }

    fn place_begun_sources(&mut self) {
        for crate::solver::BegunSource { decision, source } in self.solver.take_begun_sources() {
            if let Some(begins) = self.source_begins_by_decision.get(&decision) {
                self.solver.source_begins_at(source, *begins);
            }
        }
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
        let bound = &self.scopes[depth][&name];
        if self.lambda_stack.iter().any(|ls| depth < ls.depth) {
            return AssignTarget::Captured {
                binder: bound.binder,
            };
        }
        AssignTarget::Bound {
            ty: bound.ty.clone(),
            binder: bound.binder,
        }
    }

    /// A name for a value: its type is a variable of the solver, so what
    /// the value grows to (a field stored, a variant added) is seen by
    /// every use of the name.
    fn define_var(&mut self, name: Astr, ty: InferTy, binder: AstId, span: Span) {
        let ty = self.as_defined(ty, span);
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
        self.head_once_decided(&ty);
        let mut held = Vec::new();
        self.solver
            .resolve_ty(&ty)
            .for_each_source(&mut |id| held.push(id));
        for id in held {
            self.solver.name_source(id, name);
        }
        self.binder_types.insert(binder, ty.clone());
        self.resolved.record(binder, Resolved::Local(binder));
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, Bound { ty, binder });
        }
    }

    /// The type a name has where it is used, and the node that binds it. A
    /// name a lambda captured is seen in its body as the word it is, copied
    /// at each use, and as a shared reference into the closure at every
    /// other type (RFC-0018).
    fn lookup_var(&mut self, name: Astr) -> Option<(InferTy, AstId)> {
        for (depth, scope) in self.scopes.iter().enumerate().rev() {
            let Some(bound) = scope.get(&name) else {
                continue;
            };
            let (ty, binder) = (bound.ty.clone(), bound.binder);
            let capturing_lambdas = self
                .lambda_stack
                .iter()
                .filter(|ls| depth < ls.depth)
                .count();
            if capturing_lambdas == 0 {
                return Some((ty, binder));
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
            return Some((seen, binder));
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
    fn signature_set(&mut self, name: QualifiedRef, callee: AstId) -> Vec<SignatureCandidate> {
        let mut candidates = self.declared_signatures(name);
        if name.namespace.is_none()
            && let Some(ty) = self.local_signature(name.name, callee)
        {
            candidates.push(SignatureCandidate::Local { ty });
        }
        candidates
    }

    fn declared_signatures(&self, name: QualifiedRef) -> Vec<SignatureCandidate> {
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
        candidates.extend(self.view_signatures(name));
        candidates
    }

    /// The coercions of `TypeEnv::machine` (RFC-0047 rule 3, RFC-0062
    /// rule 3), offered to a script's own call as candidates alongside
    /// `resolve_fn`'s, so that `v.as_slice()` and `s.as_str()` resolve as
    /// they do in Rust.
    ///
    /// The offer is made here and not by putting these in
    /// `TypeEnv::functions`: a name the script did not write must not reach
    /// one, and `resolve_fn` is what a bare name, a `did you mean`, and
    /// every other lookup go through.
    fn view_signatures(&self, name: QualifiedRef) -> Vec<SignatureCandidate> {
        self.env
            .machine_coercions()
            .into_iter()
            .filter(|(qref, _)| qref.name == name.name)
            .filter(|(qref, _)| name.namespace.is_none_or(|ns| qref.namespace == Some(ns)))
            .map(|(qref, coercion)| SignatureCandidate::Named {
                qref,
                scheme: coercion.scheme.clone(),
            })
            .collect()
    }

    /// The binding that is one more signature of a bare name (RFC-0043).
    /// A callable is never a word, so a call of a capture whose type is
    /// still open is a call through the reference the body reads it as
    /// (RFC-0018).
    fn local_signature(&mut self, name: Astr, callee: AstId) -> Option<InferTy> {
        let (seen, binder) = self.lookup_var(name)?;
        self.candidate_binder_of.insert(callee, binder);
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

    /// A lambda's parameter types come from the parameter that receives it
    /// (RFC-0018) when that is known; otherwise they are inference variables.
    /// A lambda, at the function type the position expects where there is
    /// one: its parameters are the expected ones and its body's value
    /// converts into the expected return (RFC-0042 rule 4), else fresh.
    fn check_lambda(&mut self, expr: &Expr<S>, expected: Option<&InferTy>) -> InferTy {
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
            self.define_var(p.name, pt.clone(), p.id, p.span);
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
        let outer_loop_depth = std::mem::take(&mut self.loop_depth);
        let body_ty = self.check_expr(body);
        self.loop_depth = outer_loop_depth;
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
        self.value_sites.push(ValueSite {
            at: ValuePosition::Result(ret.clone(), ResultCrossing::OneValue),
            span: body.span(),
        });
        self.value_sites.push(ValueSite {
            at: ValuePosition::Captures(capture_types.clone()),
            span: body.span(),
        });

        self.pop_scope();
        let ty = TyTerm::Fn {
            params: param_types,
            ret: Box::new(ret),
            captures: capture_types,
            effect: lambda_effect,
            flows: self.solver.fresh_flow_var(),
        };
        self.record_ret(*id, ty)
    }

    /// A call argument, checked against the parameter that receives it.
    fn check_arg(&mut self, arg: &Expr<S>, expected: Option<&InferTy>) -> InferTy {
        if let (Expr::Lambda { .. }, Some(expected)) = (arg, expected) {
            self.settle_in_body();
            return self.check_lambda(arg, Some(expected));
        }
        let outer = std::mem::replace(&mut self.demand, PlaceDemand::Value);
        let ty = self.check_expr(arg);
        self.demand = outer;
        ty
    }

    /// An aggregate's refusal of a reference and of a lambda that holds one
    /// is RFC-0064 rule 5, kept by decision: the loans carry an element's
    /// positions as they carry a payload's (RFC-0079 rule 2), and admitting
    /// them there is not built yet. A view is refused in a payload as well,
    /// because a payload is one `Value` and a view is two registers
    /// (RFC-0047 rule 6).
    fn reference_in_data(&self, ty: &InferTy, shape: DataShape) -> Option<MirErrorKind> {
        let resolved = self.solver.resolve_ty(ty);
        match (&resolved, shape) {
            (TyTerm::Ref(_, inner), _) if matches!(*inner.ty(), TyTerm::Str) => {
                Some(MirErrorKind::ViewInData(shape))
            }
            (TyTerm::Ref(..), _) if is_pair(&resolved) => {
                Some(MirErrorKind::ReferenceInData(shape))
            }
            (_, DataShape::Payload | DataShape::Variant) => None,
            (TyTerm::Ref(..), DataShape::Aggregate) => Some(MirErrorKind::ReferenceInData(shape)),
            (_, DataShape::Aggregate) if holds_a_loan(&resolved) => {
                Some(MirErrorKind::ReferenceInData(shape))
            }
            (_, DataShape::Aggregate) => None,
        }
    }

    /// A value a name binds or a call returns, held to `view_argument` once
    /// its type is settled.
    fn as_defined(&mut self, ty: InferTy, span: Span) -> InferTy {
        self.value_sites.push(ValueSite {
            at: ValuePosition::Defined(ty.clone()),
            span,
        });
        ty
    }

    /// A component of data, held to `reference_in_data` once its type is
    /// settled.
    fn as_data(&mut self, ty: InferTy, span: Span, shape: DataShape) -> InferTy {
        self.value_sites.push(ValueSite {
            at: ValuePosition::Data(ty.clone(), shape),
            span,
        });
        ty
    }

    /// Walk a field path on a type, resolving each step.
    /// Returns the leaf type, or an error type if any step fails.
    /// The type stored at `base.path`: each step joins the object with a
    /// partial one naming the field, so a store to a field the object did
    /// not have grows the object (RFC-0042 rule 1), and each field it
    /// reaches is a component of data. A base that is not an object is a
    /// type error at the store.
    fn field_path_for_store(&mut self, base: &InferTy, path: &[Astr], span: Span) -> InferTy {
        let mut current = base.clone();
        for field in path {
            let field_ty = self.solver.fresh_ty_var();
            let partial = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter([(
                *field,
                field_ty.clone(),
            )])));
            if let Err(mismatch) = self.solver.unify(&current, &partial) {
                let kind = self.object_refusal(&mismatch).unwrap_or_else(|| {
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&mismatch.got),
                        got: self.type_as_written(&mismatch.expected),
                    }
                });
                self.error(kind, span);
                return Self::infer_error();
            }
            current = self.as_data(field_ty, span, DataShape::Aggregate);
        }
        current
    }

    /// The type of the place a store writes: one step per `.field` and
    /// `[index]` of the path, from the root's storage. An index step demands
    /// the container's `as_slice_mut`, which is what makes the element write
    /// exclusive -- the loan the slice holds is the write's, so a container
    /// already held shared is refused there and a root holding `&T` here.
    fn store_place(&mut self, place: &acvus_ast::Place<S>, span: Span) -> InferTy {
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
                self.note_access(Effect::write(qref), span);
                let ty = self.resolve_context_type(*id, qref, span);
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
                let var_ty = match self.lookup_var(*name) {
                    Some((ty, binder)) => {
                        self.resolved.record(*id, Resolved::Local(binder));
                        ty
                    }
                    None => {
                        let near = self.near_bindings(*name);
                        self.error(
                            MirErrorKind::UndefinedVariable {
                                name: self.interner.resolve(*name).to_string(),
                                near,
                            },
                            span,
                        );
                        Self::infer_error()
                    }
                };
                self.record(*id, var_ty.clone());
                match self.solver.shallow_resolve_ty(&var_ty) {
                    TyTerm::Ref(Mutability::Mut, inner) => inner.into_ty(),
                    TyTerm::Ref(Mutability::Shared, _) => {
                        let shown = self.type_as_written(&var_ty);
                        let subject = ShownValue::Named(self.interner.resolve(*name).to_string());
                        self.error(
                            MirErrorKind::StoreThroughSharedReference { subject, ty: shown },
                            span,
                        );
                        Self::infer_error()
                    }
                    TyTerm::Var(_) => {
                        let referent = self.solver.fresh_ty_var();
                        let decision = self.solver.decide(Decision::Match {
                            scrutinee: var_ty.clone(),
                            referent: referent.clone(),
                            bindings: Vec::new(),
                        });
                        self.decision_sites.insert(decision, span);
                        self.reads_under_open_head.push(ReadUnderOpenHead {
                            decision,
                            head: var_ty,
                            read: referent.clone(),
                            demand: PlaceDemand::Borrow(Mutability::Mut),
                            through: ReadThrough::Field,
                            subject: ShownValue::Named(self.interner.resolve(*name).to_string()),
                            span,
                        });
                        referent
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

    /// The refusals a reader is shown. A refusal on a poisoned operand is
    /// not raised at all -- that is the poison contract on `Ty::Error`
    /// (`ty.rs`), held at each site. What is left here is the one
    /// consequence a type cannot carry: an unbound name assigned to is
    /// afterwards read as an undefined variable, and the two refusals are
    /// the same fact told twice under different kinds.
    ///
    /// A deferred decision settles after the text that follows it was read,
    /// so raise order is not an order a reader has: what is left is sorted
    /// by span, stably.
    fn reported(&mut self) -> Vec<MirError> {
        let raised = std::mem::take(&mut self.errors);
        let mut shown: Vec<MirError> = Vec::with_capacity(raised.len());
        for error in raised {
            let consequence = shown.iter().any(|earlier| match &error.kind {
                MirErrorKind::UndefinedVariable { name, .. } => {
                    matches!(&earlier.kind, MirErrorKind::AssignToUnbound(unbound) if unbound == name)
                }
                _ => false,
            });
            if !consequence {
                shown.push(error);
            }
        }
        shown.sort_by_key(|error| (error.span.start, error.span.end));
        shown
    }

    /// The candidate set mirrors `TypeEnv::resolve_fn`: a bare name reaches
    /// every namespace's declaration of it, a qualified one only its own
    /// namespace's. Change that rule in `ty.rs` and this offers names the
    /// call could not have resolved to.
    fn near_functions(&self, wanted: QualifiedRef) -> DidYouMean {
        let known = self
            .env
            .functions
            .keys()
            .filter_map(|q| match wanted.namespace {
                Some(ns) => {
                    (q.namespace == Some(ns)).then(|| self.interner.resolve(q.name).to_string())
                }
                None => Some(self.interner.resolve(q.name).to_string()),
            });
        let bound = self
            .scopes
            .iter()
            .flat_map(|scope| scope.keys())
            .map(|name| self.interner.resolve(*name).to_string());
        DidYouMean::of(
            self.interner.resolve(wanted.name),
            known.chain(bound).collect::<Vec<_>>(),
        )
    }

    fn near_namespaces(&self, wanted: Astr, of: Astr) -> DidYouMean {
        let declaring = self.env.functions.keys().filter_map(|q| {
            (q.name == of)
                .then(|| q.namespace)
                .flatten()
                .map(|ns| self.interner.resolve(ns).to_string())
        });
        DidYouMean::of(self.interner.resolve(wanted), declaring.collect::<Vec<_>>())
    }

    fn near_bindings(&self, wanted: Astr) -> DidYouMean {
        let bound = self
            .scopes
            .iter()
            .flat_map(|scope| scope.keys())
            .map(|name| self.interner.resolve(*name).to_string());
        DidYouMean::of(self.interner.resolve(wanted), bound.collect::<Vec<_>>())
    }

    fn near_fields(&self, object_ty: &Ty, wanted: &str) -> DidYouMean {
        let TyTerm::Object(object) = &*behind_a_reference(object_ty) else {
            return DidYouMean::default();
        };
        DidYouMean::of(
            wanted,
            object
                .keys()
                .map(|name| self.interner.resolve(*name).to_string())
                .collect::<Vec<_>>(),
        )
    }

    fn near_variants(
        &self,
        scrutinee_ty: &Ty,
        enum_name: Option<Astr>,
        wanted: Astr,
    ) -> DidYouMean {
        let TyTerm::Enum { name, variants, .. } = &*behind_a_reference(scrutinee_ty) else {
            return DidYouMean::default();
        };
        let qualified = |tag: &Astr| match enum_name {
            Some(_) => format!(
                "{}::{}",
                self.interner.resolve(name.name),
                self.interner.resolve(*tag)
            ),
            None => self.interner.resolve(*tag).to_string(),
        };
        let written = match enum_name {
            Some(ns) => format!(
                "{}::{}",
                self.interner.resolve(ns),
                self.interner.resolve(wanted)
            ),
            None => self.interner.resolve(wanted).to_string(),
        };
        DidYouMean::of(&written, variants.keys().map(qualified).collect::<Vec<_>>())
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
    ) -> SchemeAt {
        let compiler = self.compiler_offer(qref, scheme);
        self.instantiate_with_instances(qref, scheme, site, compiler)
    }

    /// `instantiate_at` with the compiler's instances the site admits: a
    /// named call's (`compiler_instances`) or an operator's
    /// (`operator_instances`). Each widens the signature's bound by the type
    /// its first parameter takes.
    fn instantiate_with_instances(
        &mut self,
        qref: QualifiedRef,
        scheme: &crate::ty::Scheme,
        site: SchemeUse,
        compiler: CompilerInstances,
    ) -> SchemeAt {
        let mut scheme = scheme.clone();
        let compiler_instances = &compiler.candidates;
        if compiler.structural.is_some()
            && let Some(first) = scheme.bounds.first_mut()
        {
            *first = TyVarBound::Any;
        } else if !compiler_instances.is_empty()
            && let Some(TyVarBound::OneOf { shapes, .. }) = scheme.bounds.first_mut()
        {
            shapes.extend(compiler_instances.iter().filter_map(|c| match &c.ty {
                TyTerm::Fn { params, .. } => match &params.first()?.ty {
                    TyTerm::Ref(_, taken) => Some(taken.ty().into_owned()),
                    taken => Some(taken.clone()),
                },
                _ => None,
            }));
        }
        let inst = self.solver.instantiate_scheme_with(&scheme, compiler);
        self.bound_sites.extend(
            inst.bounded
                .into_iter()
                .map(|var| BoundSite { var, span: site.at }),
        );
        self.effect_bound_sites.extend(
            inst.bounded_effects
                .into_iter()
                .map(|var| EffectBoundSite { var, span: site.at }),
        );
        if let Some(InstanceChoice::Decided(decision)) = inst.instance {
            self.decision_sites.insert(decision, site.at);
            self.decision_callees.insert(decision, qref);
            self.source_begins_by_decision
                .insert(decision, site.source_begins);
        }
        for required in &inst.requirements {
            self.decision_sites.insert(required.id, site.at);
            self.decision_callees.insert(required.id, qref);
        }
        let mut minted = Vec::new();
        inst.ty.for_each_source(&mut |id| minted.push(id));
        for id in minted {
            self.solver.source_begins_at(id, site.source_begins);
        }
        SchemeAt {
            ty: inst.ty,
            instance: inst.instance,
            requirements: inst.requirements,
        }
    }

    fn instantiate_call(
        &mut self,
        qref: QualifiedRef,
        scheme: &crate::ty::Scheme,
        site: SchemeUse,
    ) -> (CallType, ResolvedCallee) {
        let compiler = self.compiler_offer(qref, scheme);
        self.instantiate_call_with(qref, scheme, site, compiler)
    }

    fn instantiate_call_with(
        &mut self,
        qref: QualifiedRef,
        scheme: &crate::ty::Scheme,
        site: SchemeUse,
        compiler: CompilerInstances,
    ) -> (CallType, ResolvedCallee) {
        let SchemeAt {
            ty,
            instance,
            requirements,
        } = self.instantiate_with_instances(qref, scheme, site, compiler);
        let call_type = CallType::of(ty).expect("a declared signature is a function type");
        let callee = ResolvedCallee {
            qref,
            instance,
            requirements,
        };
        (call_type, callee)
    }

    /// The callee a resolved call lowers to; `None` while the call's type
    /// is too open to choose an instance, which the lowering treats as it
    /// treats any unresolved call.
    fn callee_of(&self, resolved: &ResolvedCallee) -> Option<Callee> {
        let (instance, own_requirements) = match resolved.instance {
            None => return Some(Callee::Direct(resolved.qref)),
            Some(InstanceChoice::Fixed(instance)) => (instance, [].as_slice()),
            Some(InstanceChoice::Decided(decision)) => (
                self.settled_instance(decision)?,
                self.solver.children_of(decision),
            ),
        };
        let required = resolved
            .requirements
            .iter()
            .chain(own_requirements)
            .map(|required| self.chosen_of(*required))
            .collect::<Option<Vec<_>>>()?;
        Some(Callee::Extern {
            id: resolved.qref,
            instance,
            required,
        })
    }

    fn settled_instance(&self, decision: DecisionId) -> Option<usize> {
        match self.solver.answer(decision)? {
            Answer::Instance(InstanceKind::Extern(instance)) => Some(instance),
            Answer::Instance(
                InstanceKind::Intrinsic(_) | InstanceKind::Operator | InstanceKind::Structural,
            ) => None,
            Answer::Conversion(_)
            | Answer::Signature { .. }
            | Answer::Lend(_)
            | Answer::Capture(_)
            | Answer::Match(_) => {
                unreachable!("an instance decision answers with an instance")
            }
        }
    }

    fn chosen_of(&self, required: RequiredDecision) -> Option<Chosen> {
        Some(Chosen {
            signature: required.signature,
            instance: self.settled_instance(required.id)?,
            required: self
                .solver
                .children_of(required.id)
                .iter()
                .map(|child| self.chosen_of(*child))
                .collect::<Option<Vec<_>>>()?,
        })
    }

    fn frozen_call(&mut self, choice: &CallChoice) -> Option<CallTarget> {
        match choice {
            CallChoice::Resolved(resolved) => self.declared_or_intrinsic(resolved),
            CallChoice::Decided(decision) => match self.solver.answer(*decision)? {
                Answer::Signature {
                    settled:
                        SettledSignature::Named {
                            qref,
                            instance,
                            requirements,
                            ..
                        },
                    ..
                } => self.declared_or_intrinsic(&ResolvedCallee {
                    qref,
                    instance,
                    requirements,
                }),
                Answer::Signature {
                    settled: SettledSignature::Local,
                    ..
                } => Some(CallTarget::Binding),
                Answer::Instance(_)
                | Answer::Conversion(_)
                | Answer::Lend(_)
                | Answer::Capture(_)
                | Answer::Match(_) => {
                    unreachable!("a signature decision answers with a signature")
                }
            },
            CallChoice::Binding => Some(CallTarget::Binding),
            CallChoice::StructuralVariant => Some(CallTarget::StructuralVariant),
            CallChoice::Operator(call) => {
                if let Some(structural) = self.structural_call(&call.callee) {
                    return Some(CallTarget::Structural(structural));
                }
                let callee = self.callee_of(&call.callee)?;
                let resolved = self.solver.resolve_ty(&call.ty);
                let ty = self.closed_or_refused(&resolved, call.at);
                Some(CallTarget::Operator(OperatorCall {
                    callee,
                    signature: call.signature,
                    ty,
                    at: call.at,
                }))
            }
        }
    }

    /// RFC-0020.
    fn declared_or_intrinsic(&self, resolved: &ResolvedCallee) -> Option<CallTarget> {
        if let Some(InstanceChoice::Decided(decision)) = resolved.instance
            && let Some(Answer::Instance(InstanceKind::Intrinsic(intrinsic))) =
                self.solver.answer(decision)
        {
            return Some(CallTarget::Intrinsic(intrinsic));
        }
        if let Some(structural) = self.structural_call(resolved) {
            return Some(CallTarget::Structural(structural));
        }
        self.callee_of(resolved).map(CallTarget::Declared)
    }

    /// `None` also where a leaf's decision did not settle: the solve
    /// reported that refusal.
    fn structural_call(&self, resolved: &ResolvedCallee) -> Option<StructuralCall> {
        let Some(InstanceChoice::Decided(decision)) = resolved.instance else {
            return None;
        };
        let settled = self.solver.settled_structural(decision)?;
        let TyTerm::Fn { params, .. } = self.solver.resolve_ty(&settled.call) else {
            return None;
        };
        let TyTerm::Ref(_, lent) = &params.first()?.ty else {
            return None;
        };
        let ty = self.solver.close_ty(&lent.ty()).ok()?;
        let leaves = structural_leaves(&ty, self.interner)
            .into_iter()
            .map(|leaf| {
                let mut at = decision;
                for component in &leaf.path {
                    let (_, child) = self
                        .solver
                        .settled_structural(at)?
                        .children
                        .iter()
                        .find(|(held, _)| held == component)?;
                    at = *child;
                }
                self.chosen_of(RequiredDecision {
                    signature: settled.offer.signature,
                    id: at,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(StructuralCall {
            signature: settled.offer.shape,
            leaves,
        })
    }

    fn frozen_calls(&mut self) -> CallMap {
        let calls: Vec<(AstId, CallChoice)> = self
            .calls
            .iter()
            .map(|(id, choice)| (*id, choice.clone()))
            .collect();
        calls
            .into_iter()
            .filter_map(|(id, choice)| Some((id, self.frozen_call(&choice)?)))
            .collect()
    }

    /// Verified by `solve_body` as an instantiation's bounded variables are.
    fn settled_signature_bounds(&self) -> (Vec<BoundSite>, Vec<EffectBoundSite>) {
        let mut types = Vec::new();
        let mut effects = Vec::new();
        for choice in self.calls.values() {
            let CallChoice::Decided(decision) = choice else {
                continue;
            };
            let Some(answer) = self.solver.answer(*decision) else {
                continue;
            };
            let Answer::Signature { settled, .. } = answer else {
                unreachable!("a signature decision answers with a signature")
            };
            let SettledSignature::Named {
                bounded,
                bounded_effects,
                ..
            } = settled
            else {
                continue;
            };
            let span = self.decision_sites[decision];
            types.extend(bounded.into_iter().map(|var| BoundSite { var, span }));
            effects.extend(
                bounded_effects
                    .into_iter()
                    .map(|var| EffectBoundSite { var, span }),
            );
        }
        (types, effects)
    }

    /// An operator's `core` signature as `acvus_extern::core` declares it,
    /// for an environment that registers none: an operator over a word or
    /// text needs only the language's own instances.
    fn operator_scheme(&self, signature: OperatorSignature) -> crate::ty::Scheme {
        let taken = || {
            TyTerm::Ref(
                Mutability::Shared,
                Box::new(TypeArg::uniform(TyTerm::Var(0))),
            )
        };
        let operands = self.operand_names(signature.operand_count());
        let (ret, bounds) = match signature {
            OperatorSignature::Eq => (TyTerm::Bool, vec![TyVarBound::one_of(vec![])]),
            OperatorSignature::Cmp => (
                TyTerm::Int(crate::ty::IntTy::I64),
                vec![TyVarBound::one_of(vec![])],
            ),
            OperatorSignature::Add
            | OperatorSignature::Sub
            | OperatorSignature::Mul
            | OperatorSignature::Div
            | OperatorSignature::Rem
            | OperatorSignature::Neg => (
                TyTerm::Var(1),
                vec![TyVarBound::one_of(vec![]), TyVarBound::Any],
            ),
        };
        crate::ty::Scheme {
            ty: TyTerm::Fn {
                params: operands
                    .into_iter()
                    .map(|name| ParamTerm::new(name, taken()))
                    .collect(),
                ret: Box::new(ret),
                captures: vec![],
                effect: Effect::PURE.into(),
                flows: Flows::Every.into(),
            },
            bounds,
            effect_bounds: vec![],
            instances: Some(crate::ty::Instances {
                concrete: Vec::new(),
                generic: None,
            }),
            requires: Vec::new(),
        }
    }

    /// `a`, `b`: the parameters of an operator's signature, as
    /// `acvus_extern::core` names them.
    fn operand_names(&self, count: usize) -> Vec<Astr> {
        ["a", "b"]
            .into_iter()
            .take(count)
            .map(|name| self.interner.intern(name))
            .collect()
    }

    /// The language's instances of an operator's `core` signature over a
    /// word or text (RFC-0020): the operator's instruction. A named call of
    /// the signature does not see them and reaches the registry's instances.
    fn operator_instances(&self, signature: OperatorSignature) -> Vec<Candidate> {
        let integers = || {
            crate::ty::IntTy::ALL
                .iter()
                .map(|width| TyTerm::Int(*width))
        };
        let at = |words: Vec<crate::ty::PolyTy>, ret: crate::ty::PolyTy| {
            words
                .into_iter()
                .map(|word| (word, ret.clone()))
                .collect::<Vec<_>>()
        };
        let answered_at_the_operand = |words: Vec<crate::ty::PolyTy>| {
            words
                .into_iter()
                .map(|word| (word.clone(), word))
                .collect::<Vec<_>>()
        };
        let instances = match signature {
            OperatorSignature::Eq => at(
                integers()
                    .chain([
                        TyTerm::Float,
                        TyTerm::Bool,
                        TyTerm::Char,
                        TyTerm::Unit,
                        TyTerm::String,
                        TyTerm::Str,
                    ])
                    .collect(),
                TyTerm::Bool,
            ),
            OperatorSignature::Cmp => at(
                integers().chain([TyTerm::Float, TyTerm::Char]).collect(),
                TyTerm::Int(crate::ty::IntTy::I64),
            ),
            OperatorSignature::Add => {
                let mut instances =
                    answered_at_the_operand(integers().chain([TyTerm::Float]).collect());
                instances.extend(at(vec![TyTerm::String, TyTerm::Str], TyTerm::String));
                instances
            }
            OperatorSignature::Sub
            | OperatorSignature::Mul
            | OperatorSignature::Div
            | OperatorSignature::Rem => {
                answered_at_the_operand(integers().chain([TyTerm::Float]).collect())
            }
            OperatorSignature::Neg => answered_at_the_operand(
                integers()
                    .filter(|word| matches!(word, TyTerm::Int(width) if width.signed()))
                    .chain([TyTerm::Float])
                    .collect(),
            ),
        };
        let operands = self.operand_names(signature.operand_count());
        instances
            .into_iter()
            .map(|(word, ret)| {
                let taken =
                    || TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(word.clone())));
                Candidate {
                    instance: InstanceKind::Operator,
                    requires: Vec::new(),
                    ty: TyTerm::Fn {
                        params: operands
                            .iter()
                            .map(|name| ParamTerm::new(*name, taken()))
                            .collect(),
                        ret: Box::new(ret),
                        captures: vec![],
                        effect: Effect::PURE.into(),
                        flows: Flows::none().into(),
                    },
                    admits: Task::Heavy,
                    effect_bounds: Vec::new(),
                }
            })
            .collect()
    }

    fn compiler_offer(&self, qref: QualifiedRef, scheme: &crate::ty::Scheme) -> CompilerInstances {
        let core = |name: &str| {
            QualifiedRef::qualified(self.interner.intern("core"), self.interner.intern(name))
        };
        let shape = [
            (core("eq"), StructuralSignature::Eq),
            (core("clone"), StructuralSignature::Clone),
        ]
        .into_iter()
        .find_map(|(named, shape)| (named == qref).then_some(shape));
        CompilerInstances {
            candidates: self.compiler_instances(qref),
            withholds: Withholds::SameType,
            structural: shape.map(|shape| self.structural_offer(shape, qref, scheme)),
        }
    }

    /// Each component of a structural type is decided among the language's
    /// own instances at a word or text and the registry's at every other
    /// type, whether the structural type was reached by an operator or by
    /// name: the component has no site of its own to withhold by. A
    /// component's instance is called from inside the language's own
    /// instruction, which does not suspend, so an instance that does is not
    /// offered.
    fn structural_offer(
        &self,
        shape: StructuralSignature,
        signature: QualifiedRef,
        scheme: &crate::ty::Scheme,
    ) -> ComponentOffer {
        let (params, own) = match shape {
            StructuralSignature::Eq => (
                self.operand_names(2),
                self.operator_instances(OperatorSignature::Eq),
            ),
            StructuralSignature::Clone => (self.operand_names(1), self.language_clones()),
        };
        let declared = scheme
            .instances
            .iter()
            .flat_map(|instances| instances.concrete.iter().enumerate())
            .filter(|(_, sig)| sig.task <= Task::Sync && !takes_a_language_owned_type(&sig.ty))
            .map(|(instance, sig)| Candidate {
                instance: InstanceKind::Extern(instance),
                ty: sig.ty.clone(),
                admits: sig.admits,
                requires: sig.requires.clone(),
                effect_bounds: sig.effect_bounds.clone(),
            });
        ComponentOffer {
            signature,
            shape,
            params,
            candidates: own.into_iter().chain(declared).collect(),
        }
    }

    fn language_clones(&self) -> Vec<Candidate> {
        let a = self.interner.intern("a");
        crate::ty::IntTy::ALL
            .iter()
            .map(|width| TyTerm::Int(*width))
            .chain([
                TyTerm::Float,
                TyTerm::Bool,
                TyTerm::Char,
                TyTerm::Unit,
                TyTerm::String,
            ])
            .map(|word| Candidate {
                instance: InstanceKind::Operator,
                requires: Vec::new(),
                ty: TyTerm::Fn {
                    params: vec![ParamTerm::new(
                        a,
                        TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(word.clone()))),
                    )],
                    ret: Box::new(word),
                    captures: vec![],
                    effect: Effect::PURE.into(),
                    flows: Flows::none().into(),
                },
                admits: Task::Heavy,
                effect_bounds: Vec::new(),
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
            requires: Vec::new(),
            effect_bounds: Vec::new(),
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
                flows: Flows::none().into(),
            },
            admits: Task::Heavy,
        }]
    }

    fn frozen_coercions(&self) -> CoercionMap {
        self.coercions
            .iter()
            .filter_map(|coercion| {
                let kind = match &coercion.cast {
                    PendingCast::Value(cast) => CastKind::Extern(self.frozen_cast(cast)?),
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
                    PendingCast::Reborrow { shared } => CastKind::Reborrow {
                        shared: self.solver.close_ty(shared).ok()?,
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
            required,
        } = self.callee_of(&cast.callee)?
        else {
            unreachable!("a cast is an Extern function (RFC-0023)")
        };
        Some(ExternCast {
            fn_ref,
            instance,
            required,
            callee_ty: cast.callee_ty.clone(),
        })
    }

    /// Solve the body once it is checked (RFC-0042): every decision
    /// settles or is reported at its site, casts the conversions settled on
    /// become coercions, and every bounded variable and literal is verified.
    /// The held arguments are settled between two solves: the first names
    /// their heads and settles their candidates, and the second answers the
    /// conversions the settled candidates take them by.
    fn solve_body(&mut self) {
        self.join_bound_variants();
        let mut unsettled = std::mem::take(&mut self.refused_in_body);
        unsettled.extend(self.solver.solve());
        let mut refused: FxHashSet<DecisionId> =
            unsettled.iter().map(Unsettled::decision).collect();
        self.place_begun_sources();
        self.place_opened_children();
        self.report_unsettled(unsettled);
        self.check_reads_under_open_head();
        self.check_list_pattern_lengths();
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
        self.settle_held_arguments();
        let converted = self.solver.solve();
        refused.extend(converted.iter().map(Unsettled::decision));
        self.report_unsettled(converted);
        self.resolve_conversions();
        self.settle_slice_args();
        self.settle_index_uses();
        self.check_value_sites();
        let (settled, settled_effects) = self.settled_signature_bounds();
        self.bound_sites.extend(settled);
        self.effect_bound_sites.extend(settled_effects);
        self.place_instance_bounds();
        let effect_sites = std::mem::take(&mut self.effect_bound_sites);
        // As a type variable's bound below: a violation where a decision
        // was refused is that refusal's consequence (RFC-0011 rule 5).
        if refused.is_empty() {
            for EffectBoundSite { var, span } in effect_sites {
                if let Err(out) = self.solver.close_effect_var(var) {
                    self.error(
                        MirErrorKind::TaskBelowBound {
                            found: out.effect.task,
                        },
                        span,
                    );
                }
            }
        }
        let sites = std::mem::take(&mut self.bound_sites);
        let mut never_settled: Vec<BoundSite> = Vec::new();
        for site in sites {
            // A bound that fails where a decision was refused is that
            // refusal's consequence: the refused call's types are the ones
            // that reach the bounded variable.
            if !refused.is_empty() {
                continue;
            }
            match self.solver.close_ty(&TyTerm::Var(site.var)) {
                Err(crate::ty::FreezeError::OutOfBound { ty, bound, .. }) => {
                    self.error(MirErrorKind::TypeOutOfBound { ty, bound }, site.span);
                }
                Err(_)
                    if !self
                        .solver
                        .resolve_ty(&TyTerm::Var(site.var))
                        .mentions_error() =>
                {
                    never_settled.push(site);
                }
                Err(_) | Ok(_) => {}
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
        self.settle_int_literals();
        self.refuse_open_decisions();
        // An operand no use ever settled is refused only in a body nothing
        // else was refused in: a refused call leaves its arguments' types
        // open, and that is the refusal the reader was already told.
        if self.errors.is_empty() {
            for site in never_settled {
                let resolved_ty = self.type_as_written(&TyTerm::Var(site.var));
                self.error(MirErrorKind::AmbiguousType { resolved_ty }, site.span);
            }
        }
    }

    /// Poison rule 3 and one refusal per open variable: a decision whose
    /// type holds poison, or whose every open variable was already refused
    /// by its own rule, is not refused again.
    fn refuse_open_decisions(&mut self) {
        for OpenDecision { open, span } in std::mem::take(&mut self.open_decisions) {
            if self.solver.resolve_ty(&open).mentions_error() {
                continue;
            }
            let vars = self.solver.open_vars(&open);
            if !vars.is_empty() && vars.iter().all(|var| self.refused_open.contains(var)) {
                continue;
            }
            self.refused_open.extend(vars);
            let resolved_ty = self.type_as_written(&open);
            self.error(MirErrorKind::AmbiguousType { resolved_ty }, span);
        }
    }

    /// A literal takes a width here or is refused here (RFC-0042 rule 3).
    ///
    /// The refusal is here because of what its absence did. `freeze_ty`
    /// answers `UnresolvedType` for a literal whose admitted widths have no
    /// default, no caller reported that answer, and the frozen type became
    /// poison instead — so `8.is_power_of_two()`, whose instances are the
    /// unsigned widths only, was admitted and stopped the machine at a
    /// poison instruction.
    fn settle_int_literals(&mut self) {
        let literals = std::mem::take(&mut self.int_literals);
        for IntLiteral { ty, value, span } in literals {
            let Ok(Ty::Int(k)) = self.solver.freeze_ty(&ty) else {
                let TyTerm::Var(var) = self.solver.shallow_resolve_ty(&ty) else {
                    continue;
                };
                let TyVarBound::Integer { among, .. } = self.solver.bound_of_var(var) else {
                    continue;
                };
                if self.refused_open.insert(self.solver.find_ty_root(var)) {
                    self.error(MirErrorKind::IntegerLiteralWidthUnsettled { among }, span);
                }
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
            .calls
            .iter()
            .filter_map(|(id, choice)| {
                let CallChoice::Decided(decision) = choice else {
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
    /// through that cast at its call type; at an operator's operand, which
    /// is lent as it stands, a cast is a mismatch of the two referent types.
    fn resolve_conversions(&mut self) {
        let conversions = std::mem::take(&mut self.conversions);
        for conversion in &conversions {
            let Some(Answer::Conversion(answer)) = self.solver.answer(conversion.decision) else {
                continue;
            };
            let span = conversion.site.span;
            if let (
                ConversionReport::Operand { .. },
                Conversion::Cast(_) | Conversion::ThroughRef { .. },
            ) = (conversion.site.report, answer)
            {
                self.report_lent_mismatch(&conversion.from, &conversion.to, span);
                continue;
            }
            let cast = match answer {
                Conversion::Identity => continue,
                Conversion::Reborrow => PendingCast::Reborrow {
                    shared: conversion.to.clone(),
                },
                Conversion::Cast(fn_ref) => {
                    PendingCast::Value(self.cast_at(fn_ref, &conversion.from, &conversion.to, span))
                }
                Conversion::ThroughRef {
                    mutability,
                    cast,
                    back,
                } => {
                    let from = self.solver.shallow_resolve_ty(&conversion.from);
                    let to = self.solver.shallow_resolve_ty(&conversion.to);
                    let Some(references) = ReferencePair::of(&from, &to) else {
                        unreachable!("a ThroughRef answer was formed from a reference pair")
                    };
                    let named_from = references.from.ty().clone();
                    let named_to = references.to.ty().clone();
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

    fn report_lent_mismatch(&mut self, lent_ty: &InferTy, param_ty: &InferTy, span: Span) {
        let lent_ty = self.solver.shallow_resolve_ty(lent_ty);
        let param_ty = self.solver.shallow_resolve_ty(param_ty);
        let (expected, got) = match ReferencePair::of(&lent_ty, &param_ty) {
            Some(references) => (references.to.ty(), references.from.ty()),
            None => (Cow::Borrowed(&param_ty), Cow::Borrowed(&lent_ty)),
        };
        self.error(
            MirErrorKind::UnificationFailure {
                expected: self.type_as_written(&expected),
                got: self.type_as_written(&got),
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
        let SchemeAt {
            ty: inst,
            instance,
            requirements,
        } = self.instantiate_at(fn_ref, scheme, site);
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
        self.place_opened_children();
        self.report_unsettled(unsettled);
        let callee_ty = self.closed_or_refused(&self.solver.resolve_ty(&inst), span);
        PendingExternCast {
            callee: ResolvedCallee {
                qref: fn_ref,
                instance,
                requirements,
            },
            callee_ty,
        }
    }

    fn decision_span(&self, decision: DecisionId) -> Span {
        if let Some(span) = self.decision_sites.get(&decision) {
            return *span;
        }
        if let Some(parent) = self.requirer_of.get(&decision) {
            return self.decision_span(*parent);
        }
        let Some((parent, _)) = self.opening_signature(decision) else {
            unreachable!("a decision is opened at a site or by a settled signature decision")
        };
        self.decision_sites[&parent]
    }

    /// The settled signature decision that opened `decision`, with the
    /// declaration it settled on (RFC-0043).
    fn opening_signature(&self, decision: DecisionId) -> Option<(DecisionId, QualifiedRef)> {
        self.calls.values().find_map(|choice| {
            let CallChoice::Decided(parent) = choice else {
                return None;
            };
            let Answer::Signature { settled, .. } = self.solver.answer(*parent)? else {
                unreachable!("a signature decision answers with a signature")
            };
            let SettledSignature::Named {
                qref,
                instance,
                requirements,
                ..
            } = settled
            else {
                return None;
            };
            (instance == Some(InstanceChoice::Decided(decision))
                || requirements.iter().any(|required| required.id == decision))
            .then_some((*parent, qref))
        })
    }

    /// The declaration whose scheme opened an instance decision.
    fn decision_callee(&self, decision: DecisionId) -> Option<QualifiedRef> {
        if let Some(qref) = self.decision_callees.get(&decision) {
            return Some(*qref);
        }
        if let Some(parent) = self.requirer_of.get(&decision) {
            return self.decision_callee(*parent);
        }
        self.opening_signature(decision).map(|(_, qref)| qref)
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

    /// Whether the call at `callee` was decided and refused.
    fn call_refused(&self, callee: AstId) -> bool {
        let Some(CallChoice::Decided(decision)) = self.calls.get(&callee) else {
            return false;
        };
        self.solver.answer(*decision).is_none()
    }

    /// The decisions of the `as_slice` calls an index is lowered to: their
    /// refusal is the index's, `CannotIndex`, in `settle_index_uses`.
    fn index_decisions(&self) -> FxHashSet<DecisionId> {
        self.index_uses
            .iter()
            .filter_map(|index| {
                let Some(CallChoice::Decided(decision)) = self.calls.get(&index.callee_id) else {
                    return None;
                };
                Some(*decision)
            })
            .collect()
    }

    /// The signatures one call of which takes `arg`, or a borrow of it,
    /// and yields what an instance among `takes` takes: the call a reader
    /// writes before the refused one. `Vec<i64>` is not what `iter::next`
    /// takes, and `into_iter` yields `Items<T>` of it, which is.
    fn a_call_that_reaches(&self, arg: &Ty, takes: &[crate::ty::PolyTy]) -> Vec<String> {
        let taken: Vec<Cow<'_, crate::ty::PolyTy>> = takes
            .iter()
            .filter_map(|t| match t {
                TyTerm::Fn { params, .. } => params.first().map(|p| strip_ref(&p.ty)),
                _ => None,
            })
            .collect();
        let borrowed =
            |mutability: Mutability| Ty::Ref(mutability, Box::new(TypeArg::uniform(arg.clone())));
        let spellings = [
            arg.clone(),
            borrowed(Mutability::Shared),
            borrowed(Mutability::Mut),
        ];
        let mut reaches: Vec<(String, String)> = Vec::new();
        for (qref, scheme) in &self.env.functions {
            let Some(instances) = &scheme.instances else {
                continue;
            };
            let concrete = instances.concrete.iter().map(|c| &c.ty);
            let generic = instances.generic.as_ref().map(|_| &scheme.ty);
            for instance in concrete.chain(generic) {
                let TyTerm::Fn { params, ret, .. } = instance else {
                    continue;
                };
                let [param] = params.as_slice() else {
                    continue;
                };
                let yielded = strip_ref(ret);
                if !names_a_shape(&strip_ref(&param.ty)) || !names_a_shape(&yielded) {
                    continue;
                }
                if !taken
                    .iter()
                    .any(|t| crate::ty::could_match_pattern(&yielded, t))
                {
                    continue;
                }
                let Some(spelled) = spellings
                    .iter()
                    .find(|spelling| crate::ty::matches_pattern(spelling, &param.ty))
                else {
                    continue;
                };
                let name = self.interner.resolve(qref.name).to_string();
                let spelled = spelled.shown(self.interner).to_string();
                if !reaches.iter().any(|(n, _)| *n == name) {
                    reaches.push((name, spelled));
                }
            }
        }
        reaches.sort();
        reaches
            .into_iter()
            .map(|(name, spelled)| format!("what `{name}` yields of {spelled}"))
            .collect()
    }

    fn report_unsettled(&mut self, unsettled: Vec<Unsettled>) {
        let index_decisions = self.index_decisions();
        let mut refused: Vec<InferTy> = Vec::new();
        for failure in unsettled {
            let decision = failure.decision();
            if index_decisions.contains(&decision) {
                continue;
            }
            if !matches!(
                failure,
                Unsettled::AmbiguousInstance { .. } | Unsettled::ConversionOpen { .. }
            ) {
                refused.extend(failure.types().into_iter().cloned());
            }
            let mut labels: Vec<Label> = Vec::new();
            let span = self.decision_span(decision);
            let report = self
                .conversions
                .iter()
                .find(|c| c.decision == decision)
                .map(|c| c.site.report);
            let kind = match failure {
                Unsettled::NoInstance {
                    call: TyTerm::Fn { ref params, .. },
                    ..
                } if let Some(&OperatorSite { op, signature }) =
                    self.operator_decisions.get(&decision)
                    && let Some(first) = params.first() =>
                {
                    MirErrorKind::NoOperatorInstance {
                        op,
                        signature,
                        ty: referent_shown(self.type_as_written(&first.ty)),
                    }
                }
                Unsettled::NoInstance {
                    call,
                    instances,
                    required,
                    ..
                } => {
                    let opened_by = self.decision_callee(decision);
                    if let (Some(signature), TyTerm::Fn { params, .. }) = (required, &call)
                        && let Some(first) = params.first()
                    {
                        let arg = self.type_as_written(&first.ty);
                        let reaches = self.a_call_that_reaches(&strip_ref(&arg), &instances);
                        if !reaches.is_empty() {
                            labels.push(Label::note(format!(
                                "{} takes {}",
                                self.shown_name(signature),
                                reaches.join(", or ")
                            )));
                        }
                    }
                    MirErrorKind::NoInstance {
                        ty: self.type_as_written(&call),
                        instances,
                        of: match required {
                            Some(signature) => InstanceWanted::Requirement {
                                signature,
                                required_by: opened_by,
                            },
                            None => InstanceWanted::Callee(opened_by),
                        },
                    }
                }
                Unsettled::InstanceMismatch { expected, got, .. } => {
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&expected),
                        got: self.type_as_written(&got),
                    }
                }
                Unsettled::AmbiguousInstance { call: open, .. }
                | Unsettled::ConversionOpen { to: open, .. } => {
                    self.open_decisions.push(OpenDecision { open, span });
                    continue;
                }
                Unsettled::TaskTooHigh {
                    required, found, ..
                } => MirErrorKind::TaskTooHigh { required, found },
                Unsettled::NoConversion { from, to, .. } => {
                    if let TyTerm::Var(var) = self.solver.resolve_ty(&to)
                        && let bound @ (TyVarBound::OneOf { .. } | TyVarBound::Integer { .. }) =
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
                        Some(ConversionReport::Operand { op }) => MirErrorKind::TypeMismatchBinOp {
                            op,
                            left: referent_shown(from),
                            right: referent_shown(to),
                        },
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
                            | ConversionReport::Return,
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
                Unsettled::MutableBorrowOfShared { .. } => MirErrorKind::MutableBorrowOfShared {
                    subject: ShownValue::Anonymous,
                },
                Unsettled::MatchMismatch { expected, got, .. }
                    if let Some(&op) = self.operand_decisions.get(&decision) =>
                {
                    MirErrorKind::TypeMismatchBinOp {
                        op,
                        left: self.type_as_written(&expected),
                        right: self.type_as_written(&got),
                    }
                }
                Unsettled::LendMismatch { expected, got, .. }
                | Unsettled::MatchMismatch { expected, got, .. } => {
                    MirErrorKind::UnificationFailure {
                        expected: self.type_as_written(&expected),
                        got: self.type_as_written(&got),
                    }
                }
            };
            match labels.is_empty() {
                true => self.error(kind, span),
                false => self.labeled_error(kind, span, labels),
            }
        }
        for ty in refused {
            self.solver.poison(&ty);
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
        let ty = match self.probe == Some(id) {
            true => self.marked(ty),
            false => ty,
        };
        self.record(id, ty.clone());
        ty
    }

    /// The marked node's type. The name it holds is one nothing binds, so
    /// where it would be poison it is a fresh variable instead: what the
    /// position joins it with is then the type the position expects.
    fn marked(&mut self, ty: InferTy) -> InferTy {
        let ty = match Self::is_error(&ty) {
            true => self.solver.fresh_ty_var(),
            false => ty,
        };
        self.marked_ty = Some(ty.clone());
        ty
    }

    /// The type of the context `@name` at `id` names, recording `id` as a
    /// reference to it where it is declared.
    fn resolve_context_type(&mut self, id: AstId, qref: QualifiedRef, span: Span) -> InferTy {
        self.note_context_use(qref, span);
        if let Some(ty) = self.env.contexts.get(&qref) {
            self.resolved.record(id, Resolved::Context(qref));
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

    /// A parameter a Signature declared has no place of its own, so a type
    /// that does not close is refused at the body.
    fn frozen_extern_params(&mut self, body: Span) -> Vec<ResolvedParam> {
        let params = std::mem::take(&mut self.param_types);
        params
            .iter()
            .map(|param| {
                let resolved = self.solver.resolve_ty(&param.ty);
                let at = param.first_read.unwrap_or(body);
                let closed = self.closed_or_refused(&resolved, at);
                if matches!(closed, Ty::Never) {
                    let name = self.interner.resolve(param.name).to_string();
                    self.error(MirErrorKind::InputTypeUndecided(name), at);
                }
                ResolvedParam {
                    name: param.name,
                    ty: closed,
                    origin: param.origin,
                }
            })
            .collect()
    }

    fn read_input(&mut self, name: Astr, span: Span, reader: Reader) -> InputRead {
        if let Some(bound) = self.bound_inputs.iter_mut().find(|b| b.name == name) {
            bound.first_read.get_or_insert(span);
        }
        let read = match self.param_types.iter().find(|p| p.name == name) {
            Some(param) => InputRead {
                ty: param.ty.clone(),
                admitted: true,
            },
            None => match self.inputs {
                Inputs::FromReads => {
                    let ty = self.solver.fresh_ty_var();
                    self.param_types.push(ExternParam {
                        name,
                        ty: ty.clone(),
                        first_read: Some(span),
                        origin: ParamOrigin::Read(reader),
                    });
                    InputRead { ty, admitted: true }
                }
                Inputs::Declared => {
                    let input = format!("${}", self.interner.resolve(name));
                    let refusal = match reader {
                        Reader::ThisBody => MirErrorKind::UndefinedVariable {
                            name: input,
                            near: DidYouMean::default(),
                        },
                        Reader::Reached(reader) => MirErrorKind::UndeclaredInput {
                            input,
                            reader: self.interner.resolve(reader.name).to_string(),
                        },
                    };
                    self.error(refusal, span);
                    InputRead {
                        ty: Self::infer_error(),
                        admitted: false,
                    }
                }
            },
        };
        let capturing: Vec<usize> = (0..self.lambda_stack.len()).collect();
        self.captured_by(name, &read.ty, &capturing);
        read
    }

    /// A call to a local function reads, in the caller, each `$` the
    /// callee's reachable code reads, and the lowering passes those values
    /// (RFC-0071 rule 4).
    fn pass_inputs(&mut self, callee: QualifiedRef, span: Span) {
        let env = self.env;
        let Some(inputs) = env.inputs.get(&callee) else {
            return;
        };
        for input in inputs {
            let read = self.read_input(input.name, span, Reader::Reached(input.reader));
            if !read.admitted {
                continue;
            }
            let passed = crate::ty::lift_ty(&input.ty);
            if self.solver.unify(&read.ty, &passed).is_err() {
                let got = self.type_as_written(&read.ty);
                self.error(
                    MirErrorKind::UnificationFailure {
                        expected: input.ty.clone(),
                        got,
                    },
                    span,
                );
            }
        }
    }

    fn named_context_types(&self) -> FxHashMap<QualifiedRef, Ty> {
        self.context_uses
            .keys()
            .map(|qref| {
                let declared = self
                    .env
                    .contexts
                    .get(qref)
                    .expect("an undeclared context was refused before freezing");
                (*qref, self.closed(declared))
            })
            .collect()
    }

    /// The type a context resolved to, once every use has been checked.
    /// `None` while it is still open, or when the context is unknown.
    fn context_type(&self, qref: QualifiedRef) -> Option<Result<Ty, crate::ty::FreezeError>> {
        let ty = self.env.contexts.get(&qref)?;
        let resolved = self.solver.resolve_ty(ty);
        Some(self.solver.close_ty(&resolved))
    }

    /// A name a lambda took out of the enclosing closure's capture
    /// (RFC-0018). Checked after inference, as a context's data-ness is.
    fn check_moves_out_of_captures(&mut self) {
        let moves = std::mem::take(&mut self.capture_moves);
        for m in &moves {
            let owned = self.solver.resolve_ty(&m.owned);
            if matches!(
                self.solver.capture_read(&owned),
                CaptureOutcome::Reads {
                    read: CaptureRead::Word,
                    ..
                }
            ) {
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
            let ty = match self.context_type(qref) {
                None => continue,
                Some(Ok(ty)) => ty,
                Some(Err(_)) => {
                    let declared = &self.env.contexts[&qref];
                    let resolved_ty = self.type_as_written(declared);
                    self.error(MirErrorKind::AmbiguousType { resolved_ty }, span);
                    continue;
                }
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
    /// `&e` / `&mut e`: the reference type. Where `e` names no place, the
    /// value it produces is bound to a temporary storage and the reference
    /// names that temporary, which lives as long as the loans on it do.
    fn check_borrow(&mut self, place: &Expr<S>, mutable: bool, span: Span) -> InferTy {
        let mutability = if mutable {
            Mutability::Mut
        } else {
            Mutability::Shared
        };
        let outer = std::mem::replace(&mut self.demand, PlaceDemand::Borrow(mutability));
        let ty = self.check_expr(place);
        self.demand = outer;
        self.note_place(place);
        self.lend_place(&ty, place, mutability, span)
    }

    /// `a[i]` (RFC-0047): the container's own `as_slice` settled by its
    /// evidence, the element read at `i: u64`. The instance goes on
    /// `callee_id`, where every other resolved call's does, and the mode
    /// on the expression's own id, so the lowering decides nothing.
    fn check_index(&mut self, site: IndexSite<'_, S>) -> InferTy {
        let IndexSite {
            id,
            callee_id,
            object,
            index,
            span,
            demand,
        } = site;
        let mutability = demand.slice_mutability();
        let viewed = Viewed {
            view: View::Slice,
            mutability,
        };
        let candidates = self.as_slice_candidates(viewed);

        let first = self.receiver_arg(object, ReceiverMode::Lent(mutability), span);
        // The index is a `u64` and nothing else (RFC-0047 rule 4), so it is the
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
        if Self::is_error(&referent_of(&container)) {
            return self.record_ret(id, Self::infer_error());
        }
        if !takes_container(
            &candidates,
            &self.solver.resolve_ty(&referent_of(&container)),
        ) {
            self.index_uses.push(IndexUse {
                id,
                callee_id,
                span,
                container,
                element: Self::infer_error(),
                demand,
            });
            return self.record_ret(id, Self::infer_error());
        }
        let call = CalleeSite {
            id: callee_id,
            span,
        };
        let element =
            self.check_as_slice(candidates, viewed, call, first, CallRefusal::CannotIndex);
        self.index_uses.push(IndexUse {
            id,
            callee_id,
            span,
            container,
            element: element.clone(),
            demand,
        });
        self.record_ret(id, element)
    }

    fn as_slice_candidates(&self, viewed: Viewed) -> Vec<SignatureCandidate> {
        let views = self.env.machine_views(viewed).into_iter();
        views
            .map(|(qref, scheme)| SignatureCandidate::Named {
                qref,
                scheme: scheme.clone(),
            })
            .collect()
    }

    /// The element of the run the container's own `as_slice` lends
    /// (RFC-0047), or poison where that call is refused.
    fn check_as_slice(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        viewed: Viewed,
        callee: CalleeSite,
        container: FirstArg,
        refused_as: CallRefusal,
    ) -> InferTy {
        let call = NamedCall {
            callee,
            name: self.interner.intern(viewed.spelling()),
            args: &[],
            span: callee.span,
            refused_as,
        };
        let slice = self.check_named_call(candidates, &call, FirstOperand::Checked(container));
        let TyTerm::Ref(_, run) = self.solver.shallow_resolve_ty(&slice) else {
            return Self::infer_error();
        };
        match self.solver.shallow_resolve_ty(&run.ty()) {
            TyTerm::Slice(element) => *element,
            _ => Self::infer_error(),
        }
    }

    /// How every `a[i]` yields its element, and the refusals that name a
    /// type. A container's element type and its representation are not
    /// settled where the index is written, so both wait for the solve.
    fn settle_index_uses(&mut self) {
        let uses = std::mem::take(&mut self.index_uses);
        for IndexUse {
            id,
            callee_id,
            span,
            container,
            element,
            demand,
        } in uses
        {
            let referent = self.solver.resolve_ty(&referent_of(&container));
            if Self::is_error(&element) || self.call_refused(callee_id) {
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
                        // reference into the slice (RFC-0047 rule 4).
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
        place: &Expr<S>,
        mutability: Mutability,
        span: Span,
    ) -> InferTy {
        let referent = match self.lend_named(of, place, mutability, span) {
            Some(referent) => referent,
            None => self.open_lend(of, mutability, span),
        };
        TyTerm::Ref(mutability, Box::new(TypeArg::uniform(referent)))
    }

    /// What a borrow of the place names, where its head does: `None` where
    /// the head is still a variable.
    fn lend_named(
        &mut self,
        of: &InferTy,
        place: &Expr<S>,
        mutability: Mutability,
        span: Span,
    ) -> Option<InferTy> {
        if mutability == Mutability::Mut
            && let Some(Loan {
                root: Storage::Context(qref),
                ..
            }) = Loan::of(place)
        {
            self.note_access(Effect::write(qref), span);
        }
        match self.solver.lend(of, mutability) {
            LendOutcome::Names { referent, .. } => Some(referent),
            LendOutcome::MutableBorrowOfShared { referent } => {
                let subject = written_place(self.interner, place);
                self.error(MirErrorKind::MutableBorrowOfShared { subject }, span);
                Some(referent)
            }
            LendOutcome::HeadOpen => None,
        }
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
        referent
    }

    /// `recv.f(args)` is `f(recv', args)`, `recv'` borrowed when `f`'s
    /// first parameter is a reference (RFC-0030).
    fn check_method_call(
        &mut self,
        callee_id: AstId,
        receiver: &Expr<S>,
        name: Astr,
        args: &[Expr<S>],
        call_span: Span,
    ) -> InferTy {
        let candidates = self.signature_set(QualifiedRef::root(name), callee_id);
        if candidates.is_empty() {
            return self.undefined_function(QualifiedRef::root(name), call_span);
        }
        let call = NamedCall {
            callee: CalleeSite {
                id: callee_id,
                span: call_span,
            },
            name,
            args,
            span: call_span,
            refused_as: CallRefusal::NoMatchingFunction,
        };
        self.check_named_call(candidates, &call, FirstOperand::Receiver(receiver))
    }

    fn undefined_function(&mut self, name: QualifiedRef, call_span: Span) -> InferTy {
        let near = self.near_functions(name);
        let name = self.interner.resolve(name.name).to_string();
        self.error(MirErrorKind::UndefinedFunction { name, near }, call_span);
        Self::infer_error()
    }

    /// RFC-0043.
    fn check_named_call(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        call: &NamedCall<'_, S>,
        first: FirstOperand<'_, S>,
    ) -> InferTy {
        let arity = call.args.len() + usize::from(!matches!(first, FirstOperand::Absent));
        let mut refused = Vec::new();
        let taking: Vec<SignatureCandidate> = candidates
            .into_iter()
            .filter_map(|candidate| match arity_refusal(&candidate, arity) {
                None => Some(candidate),
                Some(declared) => {
                    refused.push((candidate, declared));
                    None
                }
            })
            .collect();
        let Some(signatures) = Signatures::of(taking) else {
            return self.refuse_arity(refused, call, first, arity);
        };
        let (signatures, first) = match first {
            FirstOperand::Absent => (signatures, None),
            FirstOperand::Checked(first) => (signatures, Some(first)),
            FirstOperand::Receiver(receiver) => match self.receive(signatures, receiver, call) {
                Received::Taken(signatures, first) => (signatures, Some(first)),
                Received::Held(candidates, first, receiver) => {
                    return self.admit_several(candidates, Some(first), Some(receiver), call);
                }
                Received::Refused(first) => {
                    let types = self.check_unadmitted_args(Some(&first), call.args);
                    return self.refuse_call(call, types, 1);
                }
                Received::AmbiguityReported => {
                    self.check_unadmitted_args(None, call.args);
                    return Self::infer_error();
                }
            },
        };
        self.admit_call(signatures, first, call)
    }

    fn refuse_arity(
        &mut self,
        refused: Vec<(SignatureCandidate, usize)>,
        call: &NamedCall<'_, S>,
        first: FirstOperand<'_, S>,
        arity: usize,
    ) -> InferTy {
        let one = <[(SignatureCandidate, usize); 1]>::try_from(refused).ok();
        let first = match first {
            FirstOperand::Absent => None,
            FirstOperand::Checked(first) => Some(first),
            FirstOperand::Receiver(receiver) => {
                let mode = match &one {
                    Some([(candidate, _)]) => self.solver.receiver_mode(candidate),
                    None => ReceiverMode::Value,
                };
                Some(self.receiver_arg(receiver, mode, call.span))
            }
        };
        let types = self.check_unadmitted_args(first.as_ref(), call.args);
        let Some([(candidate, declared)]) = one else {
            return self.refuse_call(call, types, usize::from(first.is_some()));
        };
        let func = match candidate {
            SignatureCandidate::Named { .. } => {
                ShownValue::Named(self.interner.resolve(call.name).to_string())
            }
            SignatureCandidate::Local { .. } => ShownValue::Anonymous,
        };
        self.error(
            MirErrorKind::ArityMismatch {
                func,
                expected: declared,
                got: arity,
            },
            call.span,
        );
        Self::infer_error()
    }

    fn check_unadmitted_args(
        &mut self,
        first: Option<&FirstArg>,
        args: &[Expr<S>],
    ) -> Vec<InferTy> {
        first
            .map(|first| first.ty.clone())
            .into_iter()
            .chain(args.iter().map(|arg| self.check_arg(arg, None)))
            .collect()
    }

    fn refuse_call(
        &mut self,
        call: &NamedCall<'_, S>,
        types: Vec<InferTy>,
        offset: usize,
    ) -> InferTy {
        match call.refused_as {
            CallRefusal::CannotIndex => Self::infer_error(),
            CallRefusal::NoMatchingFunction => self.no_matching_function(
                self.interner.resolve(call.name),
                types,
                CallAsWritten {
                    args: call.args,
                    offset,
                },
                call.span,
            ),
        }
    }

    fn admit_call(
        &mut self,
        signatures: Signatures,
        first: Option<FirstArg>,
        call: &NamedCall<'_, S>,
    ) -> InferTy {
        match signatures {
            Signatures::One(SignatureCandidate::Named { qref, scheme }) => {
                let site = SchemeUse {
                    at: call.span,
                    source_begins: call.callee.span,
                };
                let (call_type, callee) = self.instantiate_call(qref, &scheme, site);
                self.pass_inputs(qref, call.span);
                let ret = self.call_one(&call_type, first.as_ref(), call.args, call.span);
                self.record(call.callee.id, call_type.ty);
                self.calls
                    .insert(call.callee.id, CallChoice::Resolved(callee));
                ret
            }
            Signatures::One(SignatureCandidate::Local { ty }) => {
                self.record(call.callee.id, ty.clone());
                self.calls.insert(call.callee.id, CallChoice::Binding);
                self.check_callable(&ty, first.as_ref(), call.args, call.span)
            }
            Signatures::Several(candidates) => self.admit_several(candidates, first, None, call),
        }
    }

    /// A binding's or an expression's value called; the lowering calls the
    /// value.
    fn check_callable(
        &mut self,
        ty: &InferTy,
        first: Option<&FirstArg>,
        args: &[Expr<S>],
        call_span: Span,
    ) -> InferTy {
        let arity = args.len() + usize::from(first.is_some());
        let uncallable = match self.callable_type(ty, arity) {
            Ok(call_type) => return self.call_one(&call_type, first, args, call_span),
            Err(uncallable) => uncallable,
        };
        self.check_unadmitted_args(first, args);
        let refusal = match uncallable {
            Uncallable::Poisoned => return Self::infer_error(),
            Uncallable::NotCallable(shown) => MirErrorKind::NotCallable(shown),
            Uncallable::Arity(declared) => MirErrorKind::ArityMismatch {
                func: ShownValue::Anonymous,
                expected: declared,
                got: arity,
            },
        };
        self.error(refusal, call_span);
        Self::infer_error()
    }

    /// Its function type, or the one an open variable takes at the call's
    /// arity.
    fn callable_type(&mut self, ty: &InferTy, arity: usize) -> Result<CallType, Uncallable> {
        let lent = self.lent_fn(ty);
        let callable = SignatureCandidate::Local { ty: lent.clone() };
        if let Some(declared) = arity_refusal(&callable, arity) {
            return Err(Uncallable::Arity(declared));
        }
        match &lent {
            TyTerm::Var(_) => {
                let params: Vec<InferTy> = (0..arity).map(|_| self.solver.fresh_ty_var()).collect();
                let ret = self.solver.fresh_ty_var();
                let effect = self.solver.fresh_effect_var();
                let unnamed = self.interner.intern("_");
                let fn_ty = TyTerm::Fn {
                    params: params
                        .iter()
                        .map(|param| ParamTerm::new(unnamed, param.clone()))
                        .collect(),
                    ret: Box::new(ret.clone()),
                    captures: vec![],
                    effect: effect.clone(),
                    flows: self.solver.fresh_flow_var(),
                };
                if self.solver.unify(&lent, &fn_ty).is_err() {
                    return Err(Uncallable::NotCallable(self.type_as_written(&lent)));
                }
                Ok(CallType {
                    ty: fn_ty,
                    params,
                    ret,
                    effect,
                })
            }
            TyTerm::Error(_) => Err(Uncallable::Poisoned),
            _ => match CallType::of(lent.clone()) {
                Some(call_type) => Ok(call_type),
                None => Err(Uncallable::NotCallable(self.type_as_written(&lent))),
            },
        }
    }

    fn call_one(
        &mut self,
        call_type: &CallType,
        first: Option<&FirstArg>,
        args: &[Expr<S>],
        call_span: Span,
    ) -> InferTy {
        let mut parameters = Parameters::Instantiated(&call_type.params);
        self.admit_arguments(&mut parameters, first, args, call_span);
        self.note_call_effect(&call_type.effect, call_span);
        call_type.ret.clone()
    }

    /// The effect is the settled instance's, which the decision raises the
    /// body by (`UndecidedCall`).
    fn admit_several(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        first: Option<FirstArg>,
        receiver: Option<OpenReceiver<S>>,
        call: &NamedCall<'_, S>,
    ) -> InferTy {
        // Which candidate the call settles on is known only once the solve
        // has run, after this body's inputs are fixed, so the inputs of
        // every local function among the candidates are read here. A
        // candidate the call does not settle on leaves its reads unread,
        // and `graph::optimize` removes such an input from the module.
        for candidate in &candidates {
            if let SignatureCandidate::Named { qref, .. } = candidate {
                self.pass_inputs(*qref, call.span);
            }
        }
        let mut narrowing = Narrowing {
            options: candidates
                .into_iter()
                .map(SignatureOption::taking_every_argument_directly)
                .collect(),
            awaiting_head: Vec::new(),
            held: Vec::new(),
            params: Vec::new(),
            receiver,
        };
        let types = self.admit_arguments(
            &mut Parameters::Narrowing(&mut narrowing),
            first.as_ref(),
            call.args,
            call.span,
        );
        if narrowing.options.is_empty() {
            return self.refuse_call(call, types, usize::from(first.is_some()));
        }
        let declared_returns: Option<Vec<&crate::ty::PolyTy>> = narrowing
            .options
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
            name: call.name,
            call: CallShape {
                params: narrowing.params,
                ret: ret.clone(),
            },
            options: narrowing.options,
            awaiting_head: narrowing.awaiting_head,
            body_effect: self.body_effect.clone(),
        });
        self.decision_sites.insert(decision, call.span);
        self.calls
            .insert(call.callee.id, CallChoice::Decided(decision));
        self.held_arguments
            .extend(narrowing.held.into_iter().map(|held| (decision, held)));
        ret
    }

    /// Every argument, the first (piped or a receiver) first, checked left
    /// to right; each is admitted as soon as it is checked, so a lambda
    /// later in the list is checked at parameter types the earlier
    /// arguments have already fixed.
    fn admit_arguments(
        &mut self,
        parameters: &mut Parameters<'_, S>,
        first: Option<&FirstArg>,
        args: &[Expr<S>],
        call_span: Span,
    ) -> Vec<InferTy> {
        let offset = usize::from(first.is_some());
        let mut types = Vec::with_capacity(args.len() + offset);
        if let Some(first) = first {
            if let Some(param) = self.parameter_at(parameters, 0) {
                self.admit_argument(parameters, 0, &param, &first.ty, &first.site);
            }
            types.push(first.ty.clone());
        }
        for (i, arg) in args.iter().enumerate() {
            let index = i + offset;
            let param = self.parameter_at(parameters, index);
            let ty = self.check_arg(arg, param.as_ref());
            if let Some(param) = param {
                self.admit_argument(parameters, index, &param, &ty, &ArgSite::of(arg, call_span));
            }
            types.push(ty);
        }
        types
    }

    fn parameter_at(
        &mut self,
        parameters: &mut Parameters<'_, S>,
        index: usize,
    ) -> Option<InferTy> {
        match parameters {
            Parameters::Instantiated(params) => Some(
                params
                    .get(index)
                    .expect("the arity step keeps a signature only at the call's arity")
                    .clone(),
            ),
            Parameters::Narrowing(narrowing) => {
                if narrowing.options.is_empty() {
                    return None;
                }
                let param = self.call_param(&narrowing.options, index);
                let ty = param.ty.clone();
                narrowing.params.push(param);
                Some(ty)
            }
        }
    }

    fn admit_argument(
        &mut self,
        parameters: &mut Parameters<'_, S>,
        index: usize,
        param: &InferTy,
        ty: &InferTy,
        site: &ArgSite,
    ) {
        let reach = match parameters {
            Parameters::Instantiated(_) => Reach::Joined,
            Parameters::Narrowing(narrowing) => match self.narrow(narrowing, index, ty) {
                None => return,
                Some(Narrowed::Reaches(reach)) => reach,
                Some(Narrowed::Held(handed)) => {
                    let held = HeldArgument {
                        site: site.clone(),
                        index,
                        arg: ty.clone(),
                        param: param.clone(),
                        handed,
                    };
                    narrowing.awaiting_head.push(held.unjoined());
                    narrowing.held.push(held);
                    return;
                }
            },
        };
        self.meet_argument(ty, param, site, reach);
    }

    /// RFC-0043: a candidate that takes the argument directly drives out one
    /// that takes it only through a conversion or a view, and an argument
    /// whose head the solve has not named is held for the decision to ask
    /// again. `None` where the argument empties the set.
    fn narrow(
        &mut self,
        narrowing: &mut Narrowing<S>,
        index: usize,
        ty: &InferTy,
    ) -> Option<Narrowed<S>> {
        if index == 0
            && let Some(receiver) = narrowing.receiver.take()
        {
            return self.hold_receiver(narrowing, ty, receiver);
        }
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
                return None;
            }
            // Rule 2: a variable is admitted directly where the bounds
            // intersect, and not joined with the parameter: the settled
            // candidate may take the head the solve names only by a
            // reborrow or a view.
            if let TyTerm::Var(_) = self.solver.shallow_resolve_ty(ty) {
                let range = self.solver.fresh_var_with(param_range(options, index));
                if self.solver.unify(ty, &range).is_err() {
                    return None;
                }
            }
            return Some(Narrowed::Held(HeldHanded::AsWritten));
        }
        let admissions: Vec<Admission> = options
            .iter()
            .map(|option| self.solver.admits(&option.candidate, index, ty))
            .collect();
        let kept = Kept::among(&admissions);
        let mut converts = false;
        let mut wanted: Vec<Viewed> = Vec::new();
        *options = std::mem::take(options)
            .into_iter()
            .zip(admissions)
            .filter_map(|(mut option, admission)| match admission {
                _ if !kept.keeps(&admission) => None,
                Admission::Direct => Some(option),
                Admission::Reborrowed { .. } | Admission::Converted => {
                    option.converted.push(ConvertedArgument {
                        index,
                        ty: ty.clone(),
                    });
                    converts = true;
                    Some(option)
                }
                Admission::Viewed(viewed) => {
                    option.viewed.push(ConvertedArgument {
                        index,
                        ty: ty.clone(),
                    });
                    wanted.push(viewed);
                    Some(option)
                }
                Admission::Refused => None,
            })
            .collect();
        if options.is_empty() {
            return None;
        }
        wanted.dedup();
        Some(match wanted.as_slice() {
            [] if converts => Narrowed::Reaches(Reach::Converted),
            [] => Narrowed::Reaches(Reach::Joined),
            [viewed] if !converts => Narrowed::Reaches(Reach::Viewed(*viewed)),
            _ => Narrowed::Held(HeldHanded::AsWritten),
        })
    }

    /// A receiver whose head is open is held whatever the candidates take
    /// it as: each sees it in its own mode (RFC-0043 rule 6), so no one
    /// type meets the call's parameter before the decision settles on a
    /// candidate. One that refuses what its mode sees leaves the set here.
    fn hold_receiver(
        &mut self,
        narrowing: &mut Narrowing<S>,
        owned: &InferTy,
        OpenReceiver { place, referent }: OpenReceiver<S>,
    ) -> Option<Narrowed<S>> {
        narrowing.options.retain(|option| {
            !matches!(
                self.receiver_admission(&option.candidate, owned, &referent)
                    .admission,
                Admission::Refused
            )
        });
        if narrowing.options.is_empty() {
            return None;
        }
        Some(Narrowed::Held(HeldHanded::Receiver { place, referent }))
    }

    /// RFC-0030, RFC-0043.
    fn receive(
        &mut self,
        signatures: Signatures,
        receiver: &Expr<S>,
        call: &NamedCall<'_, S>,
    ) -> Received<S> {
        let agreed = agreed_receiver_mode(
            signatures
                .candidates()
                .iter()
                .map(|candidate| self.solver.receiver_mode(candidate)),
        );
        if !names_a_place(receiver) {
            let first = self.receiver_arg(receiver, lent_only_if_agreed(agreed), call.span);
            return Received::Taken(signatures, first);
        }
        let owned = self.check_receiver_place(receiver, agreed);
        let (kept, mode) = match signatures {
            Signatures::One(candidate) => {
                let mode = self.solver.receiver_mode(&candidate);
                (vec![candidate], mode)
            }
            Signatures::Several(candidates) => {
                let Some(referent) = self.named_referent(&owned, Mutability::Shared) else {
                    return self.hold_open_receiver(candidates, receiver, owned, call);
                };
                let Some(admitted) = self.admit_receiver(candidates, &owned, &referent, call)
                else {
                    return Received::AmbiguityReported;
                };
                admitted
            }
        };
        let first = self.receiver_in(receiver, owned, mode, call.span);
        match Signatures::of(kept) {
            Some(signatures) => Received::Taken(signatures, first),
            None => Received::Refused(first),
        }
    }

    fn check_receiver_place(
        &mut self,
        receiver: &Expr<S>,
        agreed: Option<ReceiverMode>,
    ) -> InferTy {
        let outer = std::mem::replace(&mut self.demand, receiver_demand(agreed));
        let owned = self.check_expr(receiver);
        self.demand = outer;
        owned
    }

    /// The place's head is open, so no candidate's mode can be admitted
    /// yet and none is chosen: the receiver is held for the candidate the
    /// decision settles on, and what a lend of the place names is the lend
    /// decision opened here, which every mode that lends it reads.
    fn hold_open_receiver(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        receiver: &Expr<S>,
        owned: InferTy,
        call: &NamedCall<'_, S>,
    ) -> Received<S> {
        let referent = self.open_lend(&owned, Mutability::Shared, receiver.span());
        let first = FirstArg {
            ty: owned,
            site: ArgSite::of(receiver, call.span),
        };
        let receiver = OpenReceiver {
            place: receiver.clone(),
            referent,
        };
        Received::Held(candidates, first, receiver)
    }

    /// Each candidate admitted at the receiver in its own mode (RFC-0043).
    /// `None` where the ones kept see it as different types, reported here.
    fn admit_receiver(
        &mut self,
        candidates: Vec<SignatureCandidate>,
        owned: &InferTy,
        referent: &InferTy,
        call: &NamedCall<'_, S>,
    ) -> Option<(Vec<SignatureCandidate>, ReceiverMode)> {
        let admitted: Vec<CandidateAdmission> = candidates
            .into_iter()
            .map(|candidate| CandidateAdmission {
                receiver: self.receiver_admission(&candidate, owned, referent),
                candidate,
            })
            .collect();
        let kept_among = Kept::among(admitted.iter().map(|one| &one.receiver.admission));
        let kept: Vec<(CandidateReceiver, InferTy)> = admitted
            .into_iter()
            .filter(|one| kept_among.keeps(&one.receiver.admission))
            .map(|one| {
                let seen = CandidateReceiver {
                    candidate: one.candidate,
                    mode: one.receiver.mode,
                };
                (seen, one.receiver.seen_as)
            })
            .collect();
        let Some(mode) = self.one_receiver_mode(&kept) else {
            let shown =
                self.shown_candidates(call.name, kept.iter().map(|(s, _)| s.candidate.name()));
            self.error(
                MirErrorKind::AmbiguousFunction {
                    name: self.interner.resolve(call.name).to_string(),
                    candidates: shown,
                },
                call.span,
            );
            return None;
        };
        Some((
            kept.into_iter().map(|(seen, _)| seen.candidate).collect(),
            mode,
        ))
    }

    /// The one admission rule of a receiver place, which a method call and
    /// completion's probe both apply.
    fn receiver_admission(
        &self,
        candidate: &SignatureCandidate,
        owned: &InferTy,
        referent: &InferTy,
    ) -> ReceiverAdmission {
        let (mode, seen_as) = self.solver.receiver_seen_by(candidate, owned, referent);
        ReceiverAdmission {
            admission: self.solver.admits(candidate, 0, &seen_as),
            mode,
            seen_as,
        }
    }

    /// What a lend of a place of type `owned` names; `None` where its head
    /// is still open.
    fn named_referent(&self, owned: &InferTy, mutability: Mutability) -> Option<InferTy> {
        match self.solver.lend(owned, mutability) {
            LendOutcome::Names { referent, .. }
            | LendOutcome::MutableBorrowOfShared { referent } => Some(referent),
            LendOutcome::HeadOpen => None,
        }
    }

    /// The type a receiver that is a value is handed to the first parameter
    /// as, in `mode`, as `receiver_arg` hands it; `None` where a lend of it
    /// waits on a head still open.
    fn value_receiver_handed(&self, ty: &InferTy, mode: ReceiverMode) -> Option<InferTy> {
        match mode {
            ReceiverMode::Lent(_) if self.passes_as_is(ty) => Some(ty.clone()),
            ReceiverMode::Lent(mutability) => {
                let referent = self.named_referent(ty, mutability)?;
                Some(TyTerm::Ref(
                    mutability,
                    Box::new(TypeArg::uniform(referent)),
                ))
            }
            ReceiverMode::Value => Some(ty.clone()),
        }
    }

    /// RFC-0030: a receiver value that is a reference already is passed as
    /// it is where the parameter takes one.
    fn passes_as_is(&self, ty: &InferTy) -> bool {
        matches!(
            self.solver.resolve_ty(ty),
            TyTerm::Ref(..) | TyTerm::Error(_)
        )
    }

    /// A receiver that is a place is lent as the parameter asks; a receiver
    /// that is already a reference value is passed as it is (RFC-0030); a
    /// receiver that is a value is bound to a temporary storage and that
    /// temporary is lent.
    fn receiver_arg(&mut self, receiver: &Expr<S>, mode: ReceiverMode, taken_by: Span) -> FirstArg {
        if names_a_place(receiver) {
            let owned = self.check_receiver_place(receiver, Some(mode));
            return self.receiver_in(receiver, owned, mode, taken_by);
        }
        let ty = self.check_expr(receiver);
        self.value_receiver_in(receiver, ty, mode, taken_by)
    }

    fn value_receiver_in(
        &mut self,
        receiver: &Expr<S>,
        ty: InferTy,
        mode: ReceiverMode,
        taken_by: Span,
    ) -> FirstArg {
        let (first, passing) = match mode {
            ReceiverMode::Lent(_) if self.passes_as_is(&ty) => {
                let first = FirstArg {
                    ty,
                    site: ArgSite::value(receiver, taken_by),
                };
                (first, Passing::AsIs)
            }
            ReceiverMode::Lent(mutability) => {
                let first = FirstArg {
                    ty: self.lend_place(&ty, receiver, mutability, receiver.span()),
                    site: ArgSite::lent(receiver, taken_by),
                };
                (first, Passing::Lent(mutability))
            }
            ReceiverMode::Value => {
                let first = FirstArg {
                    ty,
                    site: ArgSite::value(receiver, taken_by),
                };
                (first, Passing::Value)
            }
        };
        self.pass_receiver(receiver, passing);
        first
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

    /// The mode the call settled on, with its bookkeeping: the lend
    /// (RFC-0029, RFC-0041) or the place by value.
    fn receiver_in(
        &mut self,
        receiver: &Expr<S>,
        owned: InferTy,
        mode: ReceiverMode,
        taken_by: Span,
    ) -> FirstArg {
        let handing = match mode {
            ReceiverMode::Lent(mutability) => Handing::Lent {
                mutability,
                lent: self.lend_place(&owned, receiver, mutability, receiver.span()),
            },
            ReceiverMode::Value => Handing::Value,
        };
        self.receiver_handed(receiver, owned, handing, taken_by)
    }

    fn receiver_handed(
        &mut self,
        receiver: &Expr<S>,
        owned: InferTy,
        handing: Handing,
        taken_by: Span,
    ) -> FirstArg {
        let (first, passing) = match handing {
            Handing::Lent { mutability, lent } => {
                let first = FirstArg {
                    ty: lent,
                    site: ArgSite::lent(receiver, taken_by),
                };
                self.lent_places.push(LentWhole {
                    place: receiver.id(),
                    mutability,
                });
                (first, Passing::Lent(mutability))
            }
            Handing::Value => {
                self.note_value_read(receiver);
                let refused = self.reads_through_reference(receiver)
                    && self.refuse_read_through(&owned, ReadThrough::Field, receiver.span());
                let first = FirstArg {
                    ty: if refused { Self::infer_error() } else { owned },
                    site: ArgSite::value(receiver, taken_by),
                };
                (first, Passing::Value)
            }
        };
        self.pass_receiver(receiver, passing);
        first
    }

    /// RFC-0018 rules 2 and 4: a read out of what a reference names is a
    /// copy, refused where the type read is known not to copy. `true` where
    /// it was refused; a type still open is checked by `check_value_reads`.
    fn refuse_read_through(&mut self, read: &InferTy, through: ReadThrough, span: Span) -> bool {
        let read = self.solver.resolve_ty(read);
        if through.copies(&read) != Some(false) {
            return false;
        }
        let shown = self.type_as_written(&read);
        self.error(through.refusal(shown), span);
        true
    }

    /// Whether the place names its value through a reference, which is the
    /// case the refusal above is about: a field of an object a reference
    /// names.
    fn reads_through_reference(&self, place: &Expr<S>) -> bool {
        let Expr::FieldAccess { object, .. } = place else {
            return false;
        };
        let object_ty = self
            .type_map
            .get(&object.id())
            .expect("the receiver place was checked before its mode was fixed");
        matches!(self.solver.shallow_resolve_ty(object_ty), TyTerm::Ref(..))
    }

    /// No signature of the name takes the call (RFC-0043), reported at the
    /// call with the argument types it was written with.
    fn no_matching_function(
        &mut self,
        name: &str,
        types: Vec<InferTy>,
        written: CallAsWritten<'_, S>,
        call_span: Span,
    ) -> InferTy {
        if types
            .iter()
            .any(|ty| Self::is_error(&self.solver.resolve_ty(ty)))
        {
            return Self::infer_error();
        }

        let call = CallShape {
            params: types
                .into_iter()
                .enumerate()
                .map(|(index, ty)| ParamTerm::new(self.interner.intern(&index.to_string()), ty))
                .collect(),
            ret: self.solver.fresh_ty_var(),
        };
        let labels = self
            .borrow_that_would_fit(name, written)
            .into_iter()
            .collect();
        self.labeled_error(
            MirErrorKind::NoMatchingFunction {
                name: name.to_string(),
                ty: self.call_type_as_written(&call),
            },
            call_span,
            labels,
        );
        Self::infer_error()
    }

    /// The call rewritten with the borrow every declaration of the name
    /// asks for at one argument, where the argument is a plain name and so
    /// has a spelling to put it in front of.
    ///
    /// The mode comes from the scheme's parameter, whose type is a
    /// `PolyTy`. Its variables have no reader-facing spelling, so the note
    /// says the mode rather than printing the parameter.
    fn borrow_that_would_fit(&self, name: &str, written: CallAsWritten<'_, S>) -> Option<Label> {
        let CallAsWritten { args, offset } = written;
        let spelled: Option<Vec<&str>> = args
            .iter()
            .map(|arg| match arg {
                Expr::Ident {
                    name,
                    ref_kind: RefKind::Value,
                    ..
                } => Some(self.interner.resolve(name.name)),
                _ => None,
            })
            .collect();
        let spelled = spelled?;
        let wanted = self.interner.intern(name);
        let arity = args.len() + offset;
        let declarations: Vec<&crate::ty::Scheme> = self
            .env
            .functions
            .iter()
            .filter(|(qref, scheme)| qref.name == wanted && scheme.params().len() == arity)
            .map(|(_, scheme)| scheme)
            .collect();
        if declarations.is_empty() {
            return None;
        }
        let (borrowed, mode) = (offset..arity).find_map(|index| {
            let mut modes = declarations
                .iter()
                .map(|scheme| match scheme.params()[index].ty {
                    TyTerm::Ref(mutability, _) => Some(mutability),
                    _ => None,
                });
            let first = modes.next()??;
            modes
                .all(|mode| mode == Some(first))
                .then_some((index, first))
        })?;
        let borrow = match mode {
            Mutability::Shared => "&",
            Mutability::Mut => "&mut ",
        };
        let call: Vec<String> = spelled
            .iter()
            .enumerate()
            .map(|(i, arg)| match i + offset == borrowed {
                true => format!("{borrow}{arg}"),
                false => (*arg).to_string(),
            })
            .collect();
        Some(Label::note(format!(
            "the parameter is a `{}`; write `{name}({})`",
            borrow.trim_end(),
            call.join(", ")
        )))
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
            flows: Flows::Every.into(),
        }
    }

    /// The call's parameter at `index` (RFC-0043). Its bound is the
    /// parameter's own range; nothing reads it to decide how an argument
    /// is admitted.
    fn call_param(&mut self, options: &[SignatureOption], index: usize) -> ParamTerm<Infer> {
        let name = options
            .iter()
            .find_map(|option| option.candidate.param_name(index))
            .unwrap_or_else(|| self.interner.intern(&index.to_string()));
        ParamTerm::new(
            name,
            self.solver.fresh_var_with(param_range(options, index)),
        )
    }

    /// `ns::tag(payload)` where `ns` names no function: the structural
    /// variant it was before namespaces (RFC-0030).
    fn check_structural_variant(
        &mut self,
        callee_id: AstId,
        enum_name: Astr,
        tag: Astr,
        first: Option<&FirstArg>,
        args: &[Expr<S>],
        span: Span,
    ) -> InferTy {
        let payload_ty = match (first, args) {
            (None, [payload]) => Some(self.check_arg(payload, None)),
            (Some(first), []) => Some(first.ty.clone()),
            _ => None,
        };
        let Some(payload_ty) = payload_ty else {
            let near = self.near_namespaces(enum_name, tag);
            self.error(
                MirErrorKind::UndefinedFunction {
                    name: format!(
                        "{}::{}",
                        self.interner.resolve(enum_name),
                        self.interner.resolve(tag)
                    ),
                    near,
                },
                span,
            );
            return Self::infer_error();
        };
        let payload_span = args.first().map_or(span, Expr::span);
        let payload_ty = self.as_data(payload_ty, payload_span, DataShape::Variant);
        let mut variants = FxHashMap::default();
        variants.insert(tag, Some(Box::new(payload_ty)));
        self.calls.insert(callee_id, CallChoice::StructuralVariant);
        self.solver.construct(TyTerm::Enum {
            name: QualifiedRef::root(enum_name),
            variants,
            home: crate::ty::Home::NONE,
        })
    }

    /// What an operand whose head is still open names: itself, or what it
    /// references, as the solve settles it — the decision a pattern opens on
    /// an open source (RFC-0024 rule 5). An operator borrows its operands,
    /// and a borrow of a reference is that reference (RFC-0029 rule 3).
    fn named_by_open_operand(
        &mut self,
        operand: &InferTy,
        op: &'static str,
        span: Span,
    ) -> InferTy {
        let named = self.solver.fresh_ty_var();
        let decision = self.solver.decide(Decision::Match {
            scrutinee: operand.clone(),
            referent: named.clone(),
            bindings: Vec::new(),
        });
        self.decision_sites.insert(decision, span);
        self.operand_decisions.insert(decision, op);
        named
    }

    /// An operator is a call of its `core` signature (RFC-0020).
    fn check_operator_call(
        &mut self,
        id: AstId,
        op: &'static str,
        signature: OperatorSignature,
        operand: &InferTy,
        operands: &[Operand<'_>],
        span: Span,
    ) -> Option<InferTy> {
        let qref = QualifiedRef::qualified(
            self.interner.intern("core"),
            self.interner.intern(signature.name()),
        );
        let scheme = self
            .env
            .functions
            .get(&qref)
            .cloned()
            .unwrap_or_else(|| self.operator_scheme(signature));
        let site = SchemeUse {
            at: span,
            source_begins: span,
        };
        let structural = (signature == OperatorSignature::Eq)
            .then(|| self.structural_offer(StructuralSignature::Eq, qref, &scheme));
        let compiler = CompilerInstances {
            candidates: self.operator_instances(signature),
            withholds: Withholds::LanguageOwned,
            structural,
        };
        let (call_type, callee) = self.instantiate_call_with(qref, &scheme, site, compiler);
        let operator = OperatorSite { op, signature };
        if let Some(InstanceChoice::Decided(decision)) = callee.instance {
            self.operator_decisions.insert(decision, operator);
        }
        let no_instance = |this: &mut Self| {
            let shown = this.type_as_written(operand);
            this.error(
                MirErrorKind::NoOperatorInstance {
                    op,
                    signature,
                    ty: shown,
                },
                span,
            );
        };
        if call_type.params.len() != operands.len() {
            no_instance(self);
            return None;
        }
        for (&Operand { ty: named, at }, param) in operands.iter().zip(&call_type.params) {
            let borrowed = TyTerm::Ref(
                Mutability::Shared,
                Box::new(TypeArg::uniform(named.clone())),
            );
            let site = ConversionSite {
                id: at,
                span,
                report: ConversionReport::Operand { op },
            };
            if self.flow(&borrowed, param, site).is_err() {
                no_instance(self);
                return None;
            }
        }
        self.note_call_effect(&call_type.effect, span);
        self.calls.insert(
            id,
            CallChoice::Operator(OperatorCall {
                callee,
                signature,
                ty: call_type.ty,
                at: span,
            }),
        );
        Some(call_type.ret)
    }

    /// The type an operand is read at. A reference the program wrote is
    /// read through it, which is the word copy of RFC-0018 where the
    /// referent is a word and the lend of RFC-0020 where the operator
    /// keeps the reference. A referent still open is read through as well,
    /// because the operand bound the operator then imposes leaves it a
    /// word. An operand whose head is still open is read at what it names
    /// once the head settles, so the two operands meet at their referents
    /// whether the head is known here or later.
    fn read_operand_through(
        &mut self,
        operand: &InferTy,
        op: &'static str,
        stays_lent: bool,
        span: Span,
    ) -> InferTy {
        let resolved = self.solver.resolve_ty(operand);
        let inner = match &resolved {
            TyTerm::Ref(_, inner) => inner,
            TyTerm::Var(var) if self.solver.bound_of_var(*var).admits_a_reference() => {
                return self.named_by_open_operand(operand, op, span);
            }
            _ => return resolved,
        };
        let referent = self.solver.resolve_ty(&inner.ty());
        let word = referent.is_word() != Some(false);
        if stays_lent || word {
            referent
        } else {
            resolved
        }
    }

    /// The two operands of one operator meet at one type. An open
    /// representation a signature left on either side is the operator's
    /// instance decision to join, as a call argument's is (RFC-0042 rule 4).
    fn operands_meet(&mut self, lt: &InferTy, rt: &InferTy) -> bool {
        match self.solver.unify(lt, rt) {
            Ok(())
            | Err(Mismatch {
                reason: MismatchReason::ReprOpen(_),
                ..
            }) => true,
            Err(_) => false,
        }
    }

    /// An operand still open is bounded by the types its operator takes, and
    /// the bound is checked where the operand settles (`bound_sites`).
    fn bound_operand(&mut self, operand: &InferTy, bound: TyVarBound, span: Span) -> bool {
        if !matches!(self.solver.resolve_ty(operand), TyTerm::Var(_)) {
            return true;
        }
        let bounded = self.solver.fresh_var_with(bound);
        if self.solver.unify(operand, &bounded).is_err() {
            return false;
        }
        let TyTerm::Var(var) = self.solver.resolve_ty(operand) else {
            unreachable!("two open variables unify into an open variable");
        };
        self.bound_sites.push(BoundSite { var, span });
        true
    }

    fn open_beside_text(&self, lt: &InferTy, rt: &InferTy) -> Option<InferTy> {
        match (self.solver.resolve_ty(lt), self.solver.resolve_ty(rt)) {
            (TyTerm::Var(_), text) if is_text(&text) => Some(lt.clone()),
            (text, TyTerm::Var(_)) if is_text(&text) => Some(rt.clone()),
            _ => None,
        }
    }

    /// `+`, `-`, `*`, `/` and `%` over operands that are not both text:
    /// the two operands meet, and the operator's value is its call's answer.
    fn check_arithmetic(
        &mut self,
        id: AstId,
        op: &'static str,
        signature: OperatorSignature,
        [left, right]: [Operand<'_>; 2],
        span: Span,
    ) -> InferTy {
        if !self.operands_meet(left.ty, right.ty) {
            self.binop_error(
                op,
                self.solver.resolve_ty(left.ty),
                self.solver.resolve_ty(right.ty),
                span,
            );
            return Self::infer_error();
        }
        let operand = self.solver.resolve_ty(left.ty);
        self.check_operator_call(id, op, signature, &operand, &[left, right], span)
            .unwrap_or_else(Self::infer_error)
    }

    fn bound_to_text(&mut self, open: &InferTy, span: Span) -> bool {
        let text = TyVarBound::one_of(vec![TyTerm::String.into(), TyTerm::Str.into()]);
        let bounded = self.solver.fresh_var_with(text);
        if self.solver.unify(open, &bounded).is_err() {
            return false;
        }
        if let TyTerm::Var(var) = self.solver.resolve_ty(open) {
            self.bound_sites.push(BoundSite { var, span });
        }
        true
    }

    fn no_ordering(&mut self, op: &'static str, operand: InferTy, span: Span) {
        self.error(
            MirErrorKind::NoOrdering {
                op,
                ty: self.type_as_written(&operand),
            },
            span,
        );
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

    /// Type-check a single script statement.
    fn check_stmt(&mut self, stmt: &acvus_ast::Stmt<S>) {
        match stmt {
            acvus_ast::Stmt::Append { expr, span, .. } => self.check_append(expr, *span),
            acvus_ast::Stmt::Store {
                id,
                place,
                expr,
                span,
            } => {
                let ty = self.check_expr(expr);
                let target_ty = self.store_place(place, *span);
                let (base, _) = projected_store(place);
                self.place_bases
                    .insert(base.id(), WrittenBase::of_store(base));
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
                    TyTerm::Ref(Mutability::Mut, inner) => inner.into_ty(),
                    TyTerm::Error(_) => Self::infer_error(),
                    // A store through `*r` writes through a `&mut`, so a head
                    // still open is one (RFC-0018 rule 4).
                    TyTerm::Var(_) => {
                        let inner = self.solver.fresh_ty_var();
                        let written =
                            TyTerm::Ref(Mutability::Mut, Box::new(TypeArg::uniform(inner.clone())));
                        match self.solver.unify(&tt, &written) {
                            Ok(()) => inner,
                            Err(_) => Self::infer_error(),
                        }
                    }
                    other => {
                        let shown = self.type_as_written(&other);
                        let subject = written_place(self.interner, target);
                        self.error(
                            MirErrorKind::StoreThroughSharedReference { subject, ty: shown },
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
                binder,
                expr,
                span: _,
            } => {
                let ty = self.check_expr(expr);
                self.define_var(binder.name, ty.clone(), binder.id, binder.span);
                self.record(*id, ty);
            }
            acvus_ast::Stmt::LetUninit {
                id,
                binder,
                span: _,
            } => {
                let ty = self.solver.fresh_ty_var();
                self.define_var(binder.name, ty.clone(), binder.id, binder.span);
                self.record(*id, ty);
            }
            acvus_ast::Stmt::Assign {
                id,
                name,
                name_span: _,
                expr,
                span,
            } => {
                let ty = self.check_expr(expr);
                let var_ty = match self.assign_target(*name) {
                    AssignTarget::Bound { ty, binder } => {
                        self.resolved.record(*id, Resolved::Local(binder));
                        ty
                    }
                    AssignTarget::Captured { binder } => {
                        self.resolved.record(*id, Resolved::Local(binder));
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
                self.loop_depth += 1;
                for s in body {
                    self.check_stmt(s);
                }
                self.loop_depth -= 1;
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
                self.note_place(source);
                self.push_scope();
                self.loop_depth += 1;
                self.check_pattern(pattern, &source_ty, PatternSource::Expr(source.id()), *span);
                for s in body {
                    self.check_stmt(s);
                }
                self.loop_depth -= 1;
                self.pop_scope();
            }
            acvus_ast::Stmt::For {
                id,
                callee_id,
                binder,
                head,
                body,
                span,
            } => self.check_for(*id, *callee_id, binder, head, body, *span),
            acvus_ast::Stmt::Break { span, .. } => self.check_loop_jump("break", *span),
            acvus_ast::Stmt::Continue { span, .. } => self.check_loop_jump("continue", *span),
            // Its parse error is its refusal (RFC-0078 rule 5).
            acvus_ast::Stmt::Error(_) => {}
        }
    }

    /// One traversal (RFC-0057 rule 1). The head decides the source and
    /// the element: `&v` and `&mut v` through the container's own `as_slice`,
    /// which the index expressions settle the same way, an array by value,
    /// and a range of one integer width.
    fn check_for(
        &mut self,
        id: AstId,
        callee_id: AstId,
        binder: &acvus_ast::Binder,
        head: &acvus_ast::ForHead<S>,
        body: &[acvus_ast::Stmt<S>],
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
        self.push_scope();
        self.loop_depth += 1;
        self.define_var(binder.name, binding_ty, binder.id, binder.span);
        for s in body {
            self.check_stmt(s);
        }
        self.loop_depth -= 1;
        self.pop_scope();
    }

    /// `lo..hi`: one integer width at both bounds, which is the element's
    /// type (RFC-0057 rule 1).
    fn check_range(&mut self, lo: &Expr<S>, hi: &Expr<S>, span: Span) -> InferTy {
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
        place: &Expr<S>,
        mutability: Mutability,
        span: Span,
    ) -> InferTy {
        let viewed = Viewed {
            view: View::Slice,
            mutability,
        };
        let candidates = self.as_slice_candidates(viewed);
        let first = self.receiver_arg(place, ReceiverMode::Lent(mutability), span);
        let container = self.solver.resolve_ty(&first.ty);
        if !takes_container(
            &candidates,
            &self.solver.resolve_ty(&referent_of(&container)),
        ) {
            self.error(
                MirErrorKind::ForSourceNotAdmitted {
                    ty: self.type_as_written(&container),
                },
                span,
            );
            return Self::infer_error();
        }
        let call = CalleeSite {
            id: callee_id,
            span,
        };
        self.check_as_slice(
            candidates,
            viewed,
            call,
            first,
            CallRefusal::NoMatchingFunction,
        )
    }

    /// An array by value: the loop takes its elements out, and the element
    /// is the array's own (RFC-0057 rule 1). Every other value held by
    /// value is refused here, a container by value with the message that
    /// names the borrow it wanted.
    fn check_for_array(&mut self, array: &Expr<S>) -> InferTy {
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
    /// they name none (RFC-0057 rule 4).
    fn check_loop_jump(&mut self, keyword: &'static str, span: Span) {
        if self.loop_depth == 0 {
            self.error(MirErrorKind::OutsideLoop { keyword }, span);
        }
    }

    /// A template's append reads a `String` or a `&str`; nothing is
    /// converted to text implicitly (RFC-0071 rule 3).
    fn check_append(&mut self, expr: &Expr<S>, span: Span) {
        let ty = self.check_expr(expr);
        let resolved = self.solver.resolve_ty(&ty);
        let site = ConversionSite {
            id: expr.id(),
            span,
            report: ConversionReport::Emit,
        };
        match &resolved {
            TyTerm::String | TyTerm::Error(_) => {}
            TyTerm::Ref(_, inner) => match self.solver.resolve_ty(&inner.ty()) {
                TyTerm::String | TyTerm::Str | TyTerm::Error(_) => {}
                TyTerm::Var(_) => self.convert_at(&inner.ty(), &TyTerm::String, site),
                _ => self.error(
                    MirErrorKind::EmitNotString {
                        actual: self.type_as_written(&resolved),
                    },
                    span,
                ),
            },
            TyTerm::Var(_) => self.convert_at(&ty, &TyTerm::String, site),
            _ => self.error(
                MirErrorKind::EmitNotString {
                    actual: self.type_as_written(&resolved),
                },
                span,
            ),
        }
    }

    /// A demand reaches a place and the places it projects from; any other
    /// expression, and every operand inside it, is read as a value.
    fn check_expr(&mut self, expr: &Expr<S>) -> InferTy {
        if names_a_place(expr) {
            return self.check_expr_at_demand(expr);
        }
        let outer = std::mem::replace(&mut self.demand, PlaceDemand::Value);
        let ty = self.check_expr_at_demand(expr);
        self.demand = outer;
        ty
    }

    fn check_expr_at_demand(&mut self, expr: &Expr<S>) -> InferTy {
        match expr {
            // Poison, so what reads it is not refused a second time
            // (RFC-0078 rule 5).
            Expr::Error(_) => Self::infer_error(),
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
                self.probe_scope(*id, *span, qref.name);
                let ty = self.resolve_context_type(*id, *qref, *span);
                self.note_access(Effect::read(*qref), *span);
                self.record_ret(*id, ty)
            }

            Expr::Ident {
                id,
                name,
                ref_kind,
                span,
            } => {
                self.probe_scope(*id, *span, name.name);
                let ty = match ref_kind {
                    RefKind::ExternParam => {
                        let read = self.read_input(name.name, *span, Reader::ThisBody);
                        if read.admitted {
                            self.resolved.record(*id, Resolved::Input(name.name));
                        }
                        read.ty
                    }
                    RefKind::Value => match self.lookup_var(name.name) {
                        Some((ty, binder)) => {
                            self.resolved.record(*id, Resolved::Local(binder));
                            ty
                        }
                        None => {
                            let near = self.near_bindings(name.name);
                            self.error(
                                MirErrorKind::UndefinedVariable {
                                    name: self.interner.resolve(name.name).to_string(),
                                    near,
                                },
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
                for operand in [left, right] {
                    self.note_place(operand);
                    self.lent_places.push(LentWhole {
                        place: operand.id(),
                        mutability: Mutability::Shared,
                    });
                }
                let lt =
                    self.read_operand_through(&lt, op_str(*op), operand_stays_lent(*op), *span);
                let rt =
                    self.read_operand_through(&rt, op_str(*op), operand_stays_lent(*op), *span);

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
                // (RFC-0062 rule 3).
                let both_text =
                    is_text(&self.solver.resolve_ty(&lt)) && is_text(&self.solver.resolve_ty(&rt));
                let operands = [
                    Operand {
                        ty: &lt,
                        at: left.id(),
                    },
                    Operand {
                        ty: &rt,
                        at: right.id(),
                    },
                ];
                let ty = match op {
                    BinOp::Add if both_text => TyTerm::String,
                    BinOp::Eq | BinOp::Neq if both_text => TyTerm::Bool,
                    // Text's `+` and `==` read `String` and `str` alike, so
                    // an operand still open beside text is bound to text
                    // rather than met with the other operand: meeting would
                    // fix it to that one representation (RFC-0062 rule 3).
                    BinOp::Add | BinOp::Eq | BinOp::Neq
                        if let Some(open) = self.open_beside_text(&lt, &rt) =>
                    {
                        if !self.bound_to_text(&open, *span) {
                            self.binop_error(op_str(*op), lt, rt, *span);
                        }
                        match op {
                            BinOp::Add => TyTerm::String,
                            _ => TyTerm::Bool,
                        }
                    }
                    BinOp::Add => self.check_arithmetic(
                        *id,
                        op_str(*op),
                        OperatorSignature::Add,
                        operands,
                        *span,
                    ),
                    BinOp::Sub => self.check_arithmetic(
                        *id,
                        op_str(*op),
                        OperatorSignature::Sub,
                        operands,
                        *span,
                    ),
                    BinOp::Mul => self.check_arithmetic(
                        *id,
                        op_str(*op),
                        OperatorSignature::Mul,
                        operands,
                        *span,
                    ),
                    BinOp::Div => self.check_arithmetic(
                        *id,
                        op_str(*op),
                        OperatorSignature::Div,
                        operands,
                        *span,
                    ),
                    BinOp::Mod => self.check_arithmetic(
                        *id,
                        op_str(*op),
                        OperatorSignature::Rem,
                        operands,
                        *span,
                    ),
                    BinOp::Eq | BinOp::Neq => {
                        if !self.operands_meet(&lt, &rt) {
                            self.binop_error(op_str(*op), lt, rt, *span);
                            return self.record_ret(*id, TyTerm::Bool);
                        }
                        let operand = self.solver.resolve_ty(&lt);
                        if !matches!(operand, TyTerm::String) {
                            self.check_operator_call(
                                *id,
                                op_str(*op),
                                OperatorSignature::Eq,
                                &operand,
                                &operands,
                                *span,
                            );
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
                        let integers = TyVarBound::one_of(
                            crate::ty::IntTy::ALL
                                .iter()
                                .copied()
                                .map(TyTerm::Int)
                                .collect(),
                        );
                        let ok = self.operands_meet(&lt, &rt)
                            && self.bound_operand(&lt, integers, *span)
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
                    // Two operands of one type that the operator has no
                    // ordering for is a missing ordering, not a mismatch:
                    // `.to_string()` settles a mismatch and settles nothing
                    // here. Both operands text is the same refusal, because
                    // no representation of text is ordered either.
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte
                        if holds_text(&self.solver.resolve_ty(&lt))
                            && holds_text(&self.solver.resolve_ty(&rt)) =>
                    {
                        self.no_ordering(op_str(*op), self.solver.resolve_ty(&lt), *span);
                        TyTerm::Bool
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        if !self.operands_meet(&lt, &rt) {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                            return self.record_ret(*id, TyTerm::Bool);
                        }
                        let operand = self.solver.resolve_ty(&lt);
                        match &operand {
                            TyTerm::Bool | TyTerm::Unit => {
                                self.no_ordering(op_str(*op), operand.clone(), *span)
                            }
                            _ => {
                                self.check_operator_call(
                                    *id,
                                    op_str(*op),
                                    OperatorSignature::Cmp,
                                    &operand,
                                    &operands,
                                    *span,
                                );
                            }
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
                let outer = match op {
                    acvus_ast::UnaryOp::Neg => Some(std::mem::replace(
                        &mut self.demand,
                        PlaceDemand::Borrow(Mutability::Shared),
                    )),
                    acvus_ast::UnaryOp::Not | acvus_ast::UnaryOp::Deref => None,
                };
                let ot = self.check_expr(operand);
                let ot = match outer {
                    Some(outer) => {
                        self.demand = outer;
                        self.note_place(operand);
                        self.lent_places.push(LentWhole {
                            place: operand.id(),
                            mutability: Mutability::Shared,
                        });
                        self.read_operand_through(&ot, "-", true, *span)
                    }
                    None => self.solver.resolve_ty(&ot),
                };

                // Early guard: if operand is Error, suppress cascading errors.
                if Self::is_error(&ot) {
                    let ty = match op {
                        acvus_ast::UnaryOp::Neg | acvus_ast::UnaryOp::Deref => Self::infer_error(),
                        acvus_ast::UnaryOp::Not => TyTerm::Bool,
                    };
                    return self.record_ret(*id, ty);
                }

                if let acvus_ast::UnaryOp::Deref = op {
                    self.value_reads
                        .insert(*id, ValueRead::Deref { span: *span });
                }
                let ty = match op {
                    acvus_ast::UnaryOp::Deref => match &ot {
                        TyTerm::Var(_) => {
                            let inner = self.solver.fresh_ty_var();
                            let decision = self.solver.decide(Decision::Match {
                                scrutinee: ot.clone(),
                                referent: inner.clone(),
                                bindings: Vec::new(),
                            });
                            self.decision_sites.insert(decision, *span);
                            self.reads_under_open_head.push(ReadUnderOpenHead {
                                decision,
                                head: ot.clone(),
                                read: inner.clone(),
                                demand: PlaceDemand::Value,
                                through: ReadThrough::Deref,
                                subject: written_place(self.interner, operand),
                                span: *span,
                            });
                            inner
                        }
                        TyTerm::Ref(_, inner) => {
                            let inner = self.solver.resolve_ty(&inner.ty());
                            match self.refuse_read_through(&inner, ReadThrough::Deref, *span) {
                                true => Self::infer_error(),
                                false => inner,
                            }
                        }
                        _ => {
                            let shown = self.type_as_written(&ot);
                            self.error(MirErrorKind::DerefOfNonReference(shown), *span);
                            Self::infer_error()
                        }
                    },
                    acvus_ast::UnaryOp::Neg => self
                        .check_operator_call(
                            *id,
                            "-",
                            OperatorSignature::Neg,
                            &ot,
                            &[Operand {
                                ty: &ot,
                                at: operand.id(),
                            }],
                            *span,
                        )
                        .unwrap_or_else(Self::infer_error),
                    acvus_ast::UnaryOp::Not => {
                        match &ot {
                            TyTerm::Bool => {}
                            TyTerm::Var(_) if self.solver.unify(&ot, &TyTerm::Bool).is_ok() => {}
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
                let marker_site = CalleeSite {
                    id: *id,
                    span: *span,
                };
                self.probe_receiver(marker_site, *field, object, &ot_raw);
                self.note_place(expr);
                if outer == PlaceDemand::Value {
                    self.note_value_read(expr);
                }
                let ot = self.head_once_decided(&ot_raw);
                if outer == PlaceDemand::Borrow(Mutability::Mut)
                    && let TyTerm::Ref(Mutability::Shared, _) = &ot
                {
                    let shown = self.type_as_written(&ot_raw);
                    let subject = written_place(self.interner, object);
                    self.error(
                        MirErrorKind::StoreThroughSharedReference { subject, ty: shown },
                        *span,
                    );
                    return self.record_ret(*id, Self::infer_error());
                }
                let field_key = *field;
                let field_str = || self.interner.resolve(*field).to_string();
                if let TyTerm::Ref(_, inner) = &ot {
                    let field_ty = match self.solver.shallow_resolve_ty(&inner.ty()) {
                        TyTerm::Object(fields) if fields.contains_key(&field_key) => {
                            fields[&field_key].clone()
                        }
                        TyTerm::Error(_) => Self::infer_error(),
                        TyTerm::Object(_) | TyTerm::Var(_) => {
                            let fresh = self.solver.fresh_ty_var();
                            let partial = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter(
                                [(field_key, fresh.clone())],
                            )));
                            if self.solver.unify(&inner.ty(), &partial).is_err() {
                                self.error(
                                    MirErrorKind::UndefinedField {
                                        near: self
                                            .near_fields(&self.type_as_written(&ot), &field_str()),
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
                                    near: self
                                        .near_fields(&self.type_as_written(&ot), &field_str()),
                                    object_ty: self.type_as_written(&ot),
                                    field: field_str(),
                                },
                                *span,
                            );
                            return self.record_ret(*id, Self::infer_error());
                        }
                    };
                    if self.demand == PlaceDemand::Value
                        && self.refuse_read_through(&field_ty, ReadThrough::Field, *span)
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
                    TyTerm::Var(_) => {
                        let fresh = self.solver.fresh_ty_var();
                        let partial_obj = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter(
                            [(field_key, fresh.clone())],
                        )));
                        let decision = self.solver.decide(Decision::Match {
                            scrutinee: ot_raw.clone(),
                            referent: partial_obj,
                            bindings: Vec::new(),
                        });
                        self.decision_sites.insert(decision, *span);
                        self.reads_under_open_head.push(ReadUnderOpenHead {
                            decision,
                            head: ot_raw.clone(),
                            read: fresh.clone(),
                            demand: outer,
                            through: ReadThrough::Field,
                            subject: written_place(self.interner, object),
                            span: *span,
                        });
                        fresh
                    }
                    TyTerm::Object(_) => {
                        let fresh = self.solver.fresh_ty_var();
                        let partial_obj = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter(
                            [(field_key, fresh.clone())],
                        )));
                        if self.solver.unify(&ot_raw, &partial_obj).is_err() {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    near: self
                                        .near_fields(&self.type_as_written(&ot), &field_str()),
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
                                near: self.near_fields(&self.type_as_written(&ot), &field_str()),
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
                let ty = self.check_func_call(func, args, None, *span);
                let ty = self.as_defined(ty, *span);
                self.record_ret(*id, ty)
            }

            Expr::MethodCall {
                id,
                callee_id,
                receiver,
                name,
                name_span: _,
                args,
                span,
            } => {
                let ty = match self.probe == Some(*id) {
                    true => self.check_probed_method_call(*id, receiver, *name, args, *span),
                    false => self.check_method_call(*callee_id, receiver, *name, args, *span),
                };
                let ty = self.as_defined(ty, *span);
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
                // The call is the stage, so a refusal of it marks `f(b, c)`
                // and not the pipeline that feeds it.
                let pipe_left = Some(left.as_ref());
                let stage = right.span();
                let ty = match right.as_ref() {
                    Expr::FuncCall { func, args, .. } => {
                        self.check_func_call(func, args, pipe_left, stage)
                    }
                    Expr::Ident {
                        ref_kind: RefKind::Value,
                        ..
                    } => self.check_func_call(right, &[], pipe_left, stage),
                    Expr::Error(_) => {
                        self.check_expr(left);
                        Self::infer_error()
                    }
                    _ => {
                        let first = FirstArg {
                            ty: self.check_expr(left),
                            site: ArgSite::of(left, stage),
                        };
                        let rt = self.check_expr(right);
                        self.check_callable(&rt, Some(&first), &[], stage)
                    }
                };
                let ty = self.as_defined(ty, *span);
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
                let ty = self.check_try(inner, *span);
                if !Self::is_error(&ty) {
                    let ret = self.return_ty.clone().expect("check_try admitted a return");
                    self.try_sites.insert(*id, ret);
                }
                self.record_ret(*id, ty)
            }

            Expr::Return { id, value, span } => {
                let ty = self.check_expr(value);
                let Some(return_ty) = self.return_ty.clone() else {
                    self.error(MirErrorKind::ReturnOutsideFunction, *span);
                    return self.record_ret(*id, Self::infer_error());
                };
                let site = ConversionSite {
                    id: value.id(),
                    span: value.span(),
                    report: ConversionReport::Return,
                };
                if self.flow(&ty, &return_ty, site).is_err() {
                    let got = self.solver.resolve_ty(&ty);
                    let expected = self.solver.resolve_ty(&return_ty);
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.type_as_written(&expected),
                            got: self.type_as_written(&got),
                        },
                        value.span(),
                    );
                }
                self.record_ret(*id, TyTerm::Never)
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
                let first_ty = self.as_data(first_ty, all_elems[0].span(), DataShape::Aggregate);
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

            Expr::Object { id, fields, span } => {
                let mut field_types = FxHashMap::default();
                for ObjectExprField { key, value, .. } in fields {
                    let ft = self.check_expr(value);
                    let ft = self.as_data(ft, value.span(), DataShape::Aggregate);
                    field_types.insert(*key, ft);
                }
                if let Some(fields) = ObjectTy::<Infer>::too_wide(&field_types) {
                    self.error(
                        MirErrorKind::ObjectTooWide {
                            fields,
                            most: ObjectTy::<Infer>::MAX_FIELDS,
                        },
                        *span,
                    );
                    return self.record_ret(*id, Self::infer_error());
                }
                let ty = self
                    .solver
                    .construct(TyTerm::Object(ObjectTy::written(field_types)));
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
                            self.as_data(et, e.span(), DataShape::Aggregate)
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
                            let inner_ty =
                                self.as_data(inner_ty, inner_expr.span(), DataShape::Payload);
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
                        MirErrorKind::UndefinedFunction {
                            name: format!("unknown variant: {}", self.interner.resolve(*tag)),
                            near: DidYouMean::default(),
                        },
                        *span,
                    );
                    return Self::infer_error();
                };

                let payload_ty = match payload {
                    Some(expr) => {
                        let ty = self.check_expr(expr);
                        let ty = self.as_data(ty, expr.span(), DataShape::Variant);
                        Some(Box::new(ty))
                    }
                    None => None,
                };

                let mut variants = FxHashMap::default();
                variants.insert(*tag, payload_ty);
                let ty = self.solver.construct(TyTerm::Enum {
                    name: QualifiedRef::root(*enum_name),
                    variants,
                    home: crate::ty::Home::NONE,
                });
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
                        self.join_branches(&[then, else_], *span)
                    }
                    None => TyTerm::Unit,
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
                self.note_place(source);
                self.push_scope();
                self.check_pattern(pattern, &source_ty, PatternSource::Expr(source.id()), *span);
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
                        self.join_branches(&[then, else_], *span)
                    }
                    None => TyTerm::Unit,
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
        scrutinee: &Expr<S>,
        arms: &[acvus_ast::MatchExprArm<S>],
        span: Span,
    ) -> InferTy {
        let source_ty = self.check_expr(scrutinee);
        self.note_place(scrutinee);
        let mut branches: Vec<Branch> = Vec::with_capacity(arms.len());
        for arm in arms {
            let arm_source = self.reachable_arm_source(&arm.pattern, &source_ty, arm.span);
            self.push_scope();
            self.check_pattern(
                &arm.pattern,
                &arm_source,
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
            branches.push(branch);
        }
        let joined = match branches.as_slice() {
            [] => TyTerm::Unit,
            [only] => only.ty.clone(),
            all => self.join_branches(all, span),
        };
        if !crate::lower::Dispatch::is_decidable(arms, self.interner) {
            self.error(MirErrorKind::MatchIsNotADispatch, span);
        }
        if let Some(key) = crate::lower::Dispatch::repeated_key(arms, self.interner) {
            self.error(MirErrorKind::MatchArmKeyRepeated { key }, span);
        }
        joined
    }

    /// RFC-0051 rule 2: an arm contributes no variant. A pattern naming a tag
    /// the scrutinee's enum does not carry can never be taken, and the
    /// language has no warning axis, so it is refused. The question is
    /// asked only where the answer is written down: a scrutinee whose head
    /// is still open says nothing, and no arm is refused on it.
    ///
    /// An arm this refused is read against poison, so the tag that is not
    /// there is not unified against a second time and the payload name
    /// still binds.
    fn reachable_arm_source(
        &mut self,
        pattern: &Pattern<S>,
        source_ty: &InferTy,
        span: Span,
    ) -> InferTy {
        let Pattern::Variant { tag, payload, .. } = pattern else {
            return source_ty.clone();
        };
        let TyTerm::Enum { name, variants, .. } = self.solver.shallow_resolve_ty(source_ty) else {
            return source_ty.clone();
        };
        if variants.contains_key(tag) {
            return source_ty.clone();
        }
        let written = match payload {
            Some(_) => format!(
                "{}::{}(_)",
                self.interner.resolve(name.name),
                self.interner.resolve(*tag)
            ),
            None => format!(
                "{}::{}",
                self.interner.resolve(name.name),
                self.interner.resolve(*tag)
            ),
        };
        let scrutinee_ty = self.type_as_written(source_ty);
        let near = self.near_variants(&scrutinee_ty, Some(name.name), *tag);
        self.error(
            MirErrorKind::UnreachablePattern {
                pattern: written,
                scrutinee_ty,
                near,
            },
            span,
        );
        Self::infer_error()
    }

    fn check_else_branch(&mut self, eb: &acvus_ast::ElseBranch<S>) -> Branch {
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
        func: &Expr<S>,
        args: &[Expr<S>],
        pipe_left: Option<&Expr<S>>,
        call_span: Span,
    ) -> InferTy {
        let first = pipe_left.map(|e| FirstArg {
            ty: self.check_expr(e),
            site: ArgSite::of(e, call_span),
        });
        let Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } = func
        else {
            let ft = self.check_expr(func);
            let resolved = self.solver.shallow_resolve_ty(&ft);
            return self.check_callable(&resolved, first.as_ref(), args, call_span);
        };
        self.probe_scope(func.id(), func.span(), name.name);
        let candidates = self.signature_set(*name, func.id());
        if candidates.is_empty() {
            if let Some(ns) = name.namespace {
                return self.check_structural_variant(
                    func.id(),
                    ns,
                    name.name,
                    first.as_ref(),
                    args,
                    call_span,
                );
            }
            return self.undefined_function(*name, call_span);
        }
        let call = NamedCall {
            callee: CalleeSite {
                id: func.id(),
                span: func.span(),
            },
            name: name.name,
            args,
            span: call_span,
            refused_as: CallRefusal::NoMatchingFunction,
        };
        let first = match first {
            Some(first) => FirstOperand::Checked(first),
            None => FirstOperand::Absent,
        };
        self.check_named_call(candidates, &call, first)
    }

    /// A call lends the closure a `&Fn` names instead of moving it
    /// (RFC-0018), which is the reborrow of RFC-0029. A referent still open
    /// is lent the same way, and the call is what decides it is a function.
    fn lent_fn(&mut self, ty: &InferTy) -> InferTy {
        let resolved = self.solver.shallow_resolve_ty(ty);
        let TyTerm::Ref(_, inner) = &resolved else {
            return resolved;
        };
        match self.solver.shallow_resolve_ty(&inner.ty()) {
            lent @ (TyTerm::Fn { .. } | TyTerm::Var(_)) => lent,
            _ => resolved,
        }
    }

    /// RFC-0024. A reference scrutinee is read through: the pattern is
    /// checked against what it names, and every name under it binds a
    /// reference. A scrutinee whose head is still a variable settles the
    /// same question later, through a decision; a part of a pattern is
    /// read in the mode its whole was.
    fn refuse_names_bound_twice(&mut self, pattern: &Pattern<S>, span: Span) {
        let mut bound = FxHashSet::default();
        let mut twice = Vec::new();
        bound_names(pattern, &mut |name| {
            if !bound.insert(name) {
                twice.push(name);
            }
        });
        for name in twice {
            let name = self.interner.resolve(name).to_string();
            self.error(MirErrorKind::NameBoundTwice(name), span);
        }
    }

    fn check_pattern(
        &mut self,
        pattern: &Pattern<S>,
        source_ty: &InferTy,
        source: PatternSource,
        span: Span,
    ) {
        if let PatternSource::Expr(_) = source {
            self.refuse_names_bound_twice(pattern, span);
        }
        let mode = match self.solver.match_mode(source_ty) {
            MatchOutcome::Reads(reads) if reads.mode == MatchMode::Through => {
                let outer = std::mem::replace(&mut self.pattern_mode, PatternMode::Through);
                self.check_pattern_inner(pattern, &reads.names, source, span);
                self.pattern_mode = outer;
                SourceMode::Read(MatchMode::Through)
            }
            MatchOutcome::Reads(_) => {
                self.check_pattern_inner(pattern, source_ty, source, span);
                SourceMode::Read(MatchMode::Value)
            }
            MatchOutcome::HeadOpen => match source {
                PatternSource::Expr(_) => SourceMode::Decided(
                    self.check_pattern_deferred(pattern, source_ty, source, span),
                ),
                PatternSource::Member => {
                    self.check_pattern_inner(pattern, source_ty, source, span);
                    SourceMode::Read(MatchMode::Value)
                }
            },
        };
        let read = match source {
            PatternSource::Expr(id) => id,
            PatternSource::Member => pattern.id(),
        };
        self.source_modes.insert(read, mode);
    }

    /// The scrutinee's head is still a variable, so how the pattern reads
    /// it cannot be read off it: the pattern is checked against a referent
    /// of its own, and the solver joins that referent with the scrutinee
    /// -- and every name the pattern binds with the part it stands for --
    /// once the head resolves, or on the least element if nothing ever
    /// resolves it.
    fn check_pattern_deferred(
        &mut self,
        pattern: &Pattern<S>,
        source_ty: &InferTy,
        source: PatternSource,
        span: Span,
    ) -> DecisionId {
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
        decision
    }

    /// A field read through a reference its head settled to reads what that
    /// reference allows: a store or a `&mut` needs a `&mut`, and a value
    /// read needs a word (RFC-0018 rules 4 and 8), as for a head known at
    /// the read.
    fn check_reads_under_open_head(&mut self) {
        let reads = std::mem::take(&mut self.reads_under_open_head);
        for read in reads {
            let head = self.solver.resolve_ty(&read.head);
            // A head nothing settled is the ambiguity's refusal, not this one.
            if matches!(head, TyTerm::Var(_)) {
                continue;
            }
            match self.solver.answer(read.decision) {
                Some(Answer::Match(MatchMode::Through)) => {}
                Some(Answer::Match(MatchMode::Value)) if read.through == ReadThrough::Deref => {
                    let shown = self.type_as_written(&head);
                    self.error(MirErrorKind::DerefOfNonReference(shown), read.span);
                    continue;
                }
                _ => continue,
            }
            match read.demand {
                PlaceDemand::Borrow(Mutability::Mut)
                    if matches!(head, TyTerm::Ref(Mutability::Shared, _)) =>
                {
                    let shown = self.type_as_written(&head);
                    self.error(
                        MirErrorKind::StoreThroughSharedReference {
                            subject: read.subject,
                            ty: shown,
                        },
                        read.span,
                    );
                }
                PlaceDemand::Value => {
                    self.refuse_read_through(&read.read, read.through, read.span);
                }
                PlaceDemand::Borrow(_) => {}
            }
        }
    }

    /// Every list pattern's length, as the solve left it: a known length
    /// is held to the pattern's parts, and one nothing settled is refused,
    /// since the lowering places a tail part by it.
    fn check_list_pattern_lengths(&mut self) {
        let lengths = std::mem::take(&mut self.list_pattern_lengths);
        for ListPatternLength {
            len,
            pattern_min,
            exact,
            span,
        } in lengths
        {
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
        }
    }

    /// Every value stored in data, returned from a body, captured, or
    /// defined, at the type the solve settled (RFC-0062 rules 5 and 6,
    /// RFC-0064 rule 5).
    fn check_value_sites(&mut self) {
        let sites = std::mem::take(&mut self.value_sites);
        let mut refused_in_data = false;
        for ValueSite { at, span } in &sites {
            let ValuePosition::Data(ty, shape) = at else {
                continue;
            };
            if let Some(kind) = self.reference_in_data(ty, *shape) {
                self.error(kind, *span);
                refused_in_data = true;
            }
        }
        for ValueSite { at, span } in &sites {
            let span = *span;
            match at {
                ValuePosition::Data(..) | ValuePosition::Defined(_) => {}
                ValuePosition::Result(ty, crossing) => {
                    let resolved = self.solver.resolve_ty(ty);
                    match crossing.unreturnable(&resolved) {
                        // A reference inside the result's data, where a
                        // component of data was refused, is that refusal's
                        // consequence (RFC-0011 rule 5).
                        Some(Unreturnable::Reference(_))
                            if refused_in_data && !matches!(resolved, TyTerm::Ref(..)) => {}
                        Some(Unreturnable::Reference(reference)) => {
                            let ty = self.type_as_written(reference);
                            self.error(MirErrorKind::ReferenceReturnedFromBody(ty), span);
                        }
                        Some(Unreturnable::ClosureToTheHost(closure)) => {
                            let ty = self.type_as_written(closure);
                            self.error(MirErrorKind::ClosureReturnedToTheHost(ty), span);
                        }
                        None => {}
                    }
                }
                // `MakeClosure` writes one word per capture and
                // `machine::bind_captures` hands the body one reference to
                // that word, so a view — the two adjacent registers of
                // RFC-0047 rule 6 — has nowhere to put its length.
                ValuePosition::Captures(captures) => {
                    let captured_a_view = captures.iter().any(|c| {
                        matches!(self.solver.resolve_ty(c), TyTerm::Ref(_, target)
                            if is_view(&target.ty()))
                    });
                    if captured_a_view {
                        self.error(MirErrorKind::ViewCaptured, span);
                    }
                }
            }
        }
        // A view at a type's argument in a program already refused is that
        // refusal's consequence: a lambda refused for returning `&str` still
        // makes the stage mapping with it take `&str`, and a call refused
        // for its result `Vec<&str>` still gives the name that type. Every
        // value that holds the view carries it on, so the first is the one
        // refusal.
        if !self.errors.is_empty() {
            return;
        }
        for ValueSite { at, span } in &sites {
            let ValuePosition::Defined(ty) = at else {
                continue;
            };
            let resolved = self.solver.resolve_ty(ty);
            let Some(ViewArgument { taker, view }) = view_argument(&resolved, None) else {
                continue;
            };
            let kind = MirErrorKind::ViewAsTypeArgument {
                view: self.type_as_written(view),
                taker: self.type_as_written(taker),
            };
            self.error(kind, *span);
            return;
        }
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

    /// A pattern refused against its source still binds its names, as
    /// poison, so a use of one is not refused a second time (docs/solver.md,
    /// Poison).
    fn bind_as_poison(&mut self, pattern: &Pattern<S>, span: Span) {
        self.check_pattern_inner(pattern, &Self::infer_error(), PatternSource::Member, span);
    }

    fn check_pattern_inner(
        &mut self,
        pattern: &Pattern<S>,
        source_ty: &InferTy,
        source: PatternSource,
        span: Span,
    ) {
        let source_resolved = self.solver.resolve_ty(source_ty);
        match pattern {
            Pattern::ContextBind { id, name: qref, .. } => {
                match self.pattern_mode {
                    PatternMode::Through => {
                        self.error(MirErrorKind::ReferenceInData(DataShape::Aggregate), span);
                        return;
                    }
                    PatternMode::Deferred => self.deferred_context_binds.push(span),
                    PatternMode::Value => {}
                }
                self.note_access(Effect::write(*qref), span);
                let ctx_ty = self.resolve_context_type(*id, *qref, span);
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
            Pattern::Binding {
                id, name, ref_kind, ..
            } => match ref_kind {
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
                    self.define_var(*name, ty, *id, span);
                }
            },

            // `_` reads the position and asks nothing of it, and so does a
            // pattern that did not parse, whose parse error is its refusal.
            Pattern::Wildcard { .. } | Pattern::Error(_) => {}

            // Obligation across artifacts: what makes this admission sound
            // is that a string literal pattern never becomes a value.
            // `lower.rs` emits `InstKind::TestLiteral`, which holds the text
            // as an immediate and compares the scrutinee's bytes, so the two
            // representations of one text are one comparison (RFC-0062
            // rule 3).
            Pattern::Literal {
                value: Literal::String(_),
                ..
            } if holds_text(&source_resolved) => {}
            Pattern::Literal {
                value: Literal::String(_),
                ..
            } if matches!(source_resolved, TyTerm::Var(_)) => {
                if !self.bound_to_text(source_ty, span) {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: Ty::String,
                            source_ty: self.type_as_written(&source_resolved),
                        },
                        span,
                    );
                }
            }

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
                            return self.bind_as_poison(pattern, span);
                        }
                        (var, len)
                    }
                };
                self.list_pattern_lengths.push(ListPatternLength {
                    len,
                    pattern_min: head.len() + tail.len(),
                    exact: rest.is_none(),
                    span,
                });
                for p in head.iter().chain(tail.iter()) {
                    self.check_pattern(p, &elem_ty, PatternSource::Member, span);
                }
            }

            Pattern::Object { fields, .. } => {
                // If source is already a concrete Object, match fields directly (open/subset).
                // Otherwise, build an Object from pattern fields and unify to infer the type.
                let obj_fields =
                    if let TyTerm::Object(object) = self.solver.shallow_resolve_ty(source_ty) {
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
                            return self.bind_as_poison(pattern, span);
                        }
                        field_vars
                    };
                for ObjectPatternField { key, pattern, .. } in fields {
                    let Some(field_ty) = obj_fields.get(key) else {
                        self.error(
                            MirErrorKind::UndefinedField {
                                near: self.near_fields(
                                    &self.type_as_written(&source_resolved),
                                    self.interner.resolve(*key),
                                ),
                                object_ty: self.type_as_written(&source_resolved),
                                field: self.interner.resolve(*key).to_string(),
                            },
                            span,
                        );
                        continue;
                    };
                    let field_ty = field_ty.clone();
                    self.check_pattern(pattern, &field_ty, PatternSource::Member, span);
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
                            return self.bind_as_poison(pattern, span);
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
                        return self.bind_as_poison(pattern, span);
                    }

                    if let (VariantPayload::TypeParam(idx), Some(inner_pat)) =
                        (&variant_payload, payload)
                    {
                        self.check_pattern(
                            inner_pat,
                            &type_params[*idx],
                            PatternSource::Member,
                            span,
                        );
                    }
                    return;
                }

                // Structural enum: requires qualified name.
                let Some(enum_name) = ast_enum_name else {
                    self.error(
                        MirErrorKind::UndefinedFunction {
                            name: format!("unknown variant: {}", self.interner.resolve(*tag)),
                            near: DidYouMean::default(),
                        },
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
                    name: QualifiedRef::root(*enum_name),
                    variants,
                    home: crate::ty::Home::NONE,
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
                    return self.bind_as_poison(pattern, span);
                }

                // Bind payload pattern if present.
                if let Some(inner_pat) = payload {
                    let inner_ty = payload_ty.map_or_else(Self::infer_error, |ty| *ty);
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

    /// The type the branches of an `if` or the arms of a `match` meet at:
    /// each branch flows into one variable, so a branch typed `!` (a call
    /// that traps) leaves the others' type standing (RFC-0038), and a
    /// conversion is a cast of that branch's value. References of both
    /// mutabilities meet at a `&` of their own, which a `&mut` branch
    /// reaches by a reborrow (RFC-0029 rule 3).
    fn join_branches(&mut self, branches: &[Branch], span: Span) -> InferTy {
        let joined = match self.shared_meeting(branches) {
            Some(shared) => shared,
            None => self.solver.fresh_ty_var(),
        };
        let failed = branches.iter().find(|branch| {
            let met = match branch.value {
                Some(id) => {
                    let site = ConversionSite {
                        id,
                        span,
                        report: ConversionReport::Value,
                    };
                    self.flow(&branch.ty, &joined, site)
                }
                None => self.solver.unify(&branch.ty, &joined),
            };
            met.is_err()
        });
        if let (Some(first), Some(failed)) = (branches.first(), failed) {
            self.branch_mismatches.push(BranchMismatch {
                then: first.ty.clone(),
                else_: failed.ty.clone(),
                span,
            });
        }
        joined
    }

    /// A `&` of a fresh referent where every branch is a reference or `!`
    /// and the references do not agree on their mutability.
    fn shared_meeting(&mut self, branches: &[Branch]) -> Option<InferTy> {
        let mut mutabilities = Vec::with_capacity(branches.len());
        for branch in branches {
            match self.solver.shallow_resolve_ty(&branch.ty) {
                TyTerm::Ref(mutability, _) => {
                    mutabilities.push(mutability);
                }
                TyTerm::Never => {}
                _ => return None,
            }
        }
        if !mutabilities.contains(&Mutability::Shared) || !mutabilities.contains(&Mutability::Mut) {
            return None;
        }
        Some(TyTerm::Ref(
            Mutability::Shared,
            Box::new(TypeArg::Open(
                self.solver.fresh_repr_var(),
                self.solver.fresh_ty_var(),
            )),
        ))
    }

    /// `inner?` (RFC-0038): the payload of an `Ok` or a `Some`, while the
    /// `Err` or `None` leaves the function early, so the function returns a
    /// `Result` whose error type unifies with the operand's, or an `Option`.
    fn check_try(&mut self, inner: &Expr<S>, span: Span) -> InferTy {
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
            .map(|(id, ty)| (*id, self.closed(ty)))
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
        TyTerm::Ref(_, inner) => holds_text(&inner.ty()),
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
    use crate::ty::{Param, TypeRegistry};

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
        let signatures = FxHashMap::default();
        let mut solver = Solver::new(&mut sources, &registry, &signatures);
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
            inputs: FxHashMap::default(),
        };
        let checker = TypeChecker::new(interner, &env, &mut solver, Inputs::Declared);
        let resolution = checker
            .check_template(&template)
            .resolution
            .map_err(|errs| {
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

    /// On an accepted body the view keeps every type the resolution
    /// holds and adds each binder's, which the resolution does not carry.
    #[test]
    fn an_accepted_view_extends_the_type_map_with_binders() {
        let interner = Interner::new();
        let source = "let a = 1; let b; b = a; for c in 0..2 { } \
                      let d = match Some(a) { Some(e) => e, None => 0, }; \
                      let f = |g| -> g + d; f(b)";
        let script = acvus_ast::parse_script(&interner, source).expect("the source parses");
        let mut sources = crate::ty::Sources::new();
        let registry = TypeRegistry::default();
        let signatures = FxHashMap::default();
        let mut solver = Solver::new(&mut sources, &registry, &signatures);
        let env = crate::ty::TypeEnv {
            contexts: FxHashMap::default(),
            functions: FxHashMap::default(),
            machine: FxHashMap::default(),
            inputs: FxHashMap::default(),
        };
        let checked = TypeChecker::new(&interner, &env, &mut solver, Inputs::Declared)
            .check_script(&script, None, ResultCrossing::Registers);
        let Ok(resolution) = &checked.resolution else {
            panic!("the body is accepted: {:?}", checked.resolution.err());
        };
        for (id, ty) in &resolution.type_map {
            assert_eq!(checked.view.types.get(id), Some(ty), "{id:?}");
        }
        let nodes = acvus_ast::locate::Nodes::of_script(&script);
        let spans = nodes.spans();
        let binder_type = |introduced: &str| {
            let found = source
                .find(introduced)
                .expect("the binder is in the source");
            let at = found + introduced.len() - 1;
            let binder = nodes.at(at)[0];
            assert_eq!(spans[&binder.id], acvus_ast::Span::new(at, at + 1));
            checked.view.types.get(&binder.id).cloned()
        };
        for introduced in ["let a", "let b", "for c", "let d", "Some(e", "|g"] {
            assert_eq!(binder_type(introduced), Some(Ty::I64), "{introduced}");
        }
        assert!(
            matches!(binder_type("let f"), Some(TyTerm::Fn { .. })),
            "{:?}",
            binder_type("let f")
        );
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
        let src = "% let x = 1 + 2.0";
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn match_without_a_wildcard_arm_ok() {
        let src = "% match 42\n% x =>\nhello\n% end\n";
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
        let src = "% let x = unknown";
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn extern_param_write_rejected() {
        let src = "% $count = 42";
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
                flows: crate::ty::Flows::Every.into(),
            },
        )]);
        let src = "% let x = fetch_user(1)\n{{ x }}";
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
    fn a_field_the_context_type_lacks_is_refused() {
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
        let err = result.expect_err("a host's type does not grow");
        assert!(err.contains("no field `unknown`"), "{err}");
    }

    #[test]
    fn pattern_binding_captures_type() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let src = "% let x = @name\n{{ x }}";
        check_with_interner(src, &context, &i).unwrap();
    }

    // -- Variant (Option) --------------------------------------------

    #[test]
    fn some_int_is_option_int() {
        let src = "% let x = Some(42)";
        check(src).unwrap();
    }

    #[test]
    fn none_is_option() {
        let src = "% let x = None";
        check(src).unwrap();
    }

    #[test]
    fn some_pattern_extracts_inner() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::String)))]);
        let src = "% match @opt\n% Some(x) =>\n{{ x }}\n% None =>\n% end\n";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    fn none_pattern_matches_option() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::I64)))]);
        let src = "% match @opt\n% None =>\nnone\n% _ =>\nhas value\n% end\n";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    fn some_type_mismatch() {
        let i = Interner::new();
        // Some(42) is Option<Int>, cannot match against String
        let context = FxHashMap::from_iter([(i.intern("s"), Ty::String)]);
        let src = "% match @s\n% Some(x) =>\n{{ x }}\n% None =>\n% end\n";
        assert!(check_with_interner(src, &context, &i).is_err());
    }

    // -- Named extern functions --

    /// A string literal's type (RFC-0062 rule 2).
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
                flows: crate::ty::Flows::Every.into(),
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
            flows: crate::ty::Flows::Every.into(),
        };
        let fns = FxHashMap::from_iter([
            (
                i.intern("hof"),
                Ty::Fn {
                    params: vec![p(&i, pure_fn_ty.clone())],
                    ret: Box::new(Ty::String),
                    captures: vec![],
                    effect: crate::ty::Effect::PURE.into(),
                    flows: crate::ty::Flows::Every.into(),
                },
            ),
            (
                i.intern("io_fn"),
                Ty::Fn {
                    params: vec![p(&i, Ty::I64)],
                    ret: Box::new(Ty::I64),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                    flows: crate::ty::Flows::Every.into(),
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
                flows: crate::ty::Flows::Every.into(),
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
        let err = check_with_interner("% let f = @my_fn", &ctx, &i).unwrap_err();
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
        let src = "% let x = @items";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    // Iterator/Sequence context load tests migrated to acvus-mir-test
    // (requires TypeRegistry + Interner for UserDefined construction).

    #[test]
    fn lazy_option_context_load_ok() {
        // @opt : Option<Int> - Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::I64)))]);
        let src = "% let x = @opt";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn lazy_tuple_context_load_ok() {
        // @pair : (Int, String) - Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("pair"), Ty::Tuple(vec![Ty::I64, Ty::String]))]);
        let src = "% let x = @pair";
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
        let src = "% let x = @obj";
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
                    flows: crate::ty::Flows::Every.into(),
                }),
                LenTerm::Known(3),
            ),
        )]);
        let src = "% let x = @fns";
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
                region_params: 0,
            },
        )]);
        let src = "% let x = @conn";
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
            region_params: 0,
        };
        let ctx = FxHashMap::from_iter([(i.intern("conn"), conn_ty.clone())]);
        let fns = FxHashMap::from_iter([(
            i.intern("handler"),
            Ty::Fn {
                params: vec![p(&i, conn_ty)],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
                flows: crate::ty::Flows::Every.into(),
            },
        )]);
        let src = "{{ handler(@conn) }}";
        check_with_env(src, &ctx, &fns, &i).unwrap();
    }

    #[test]
    fn pure_int_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("count"), Ty::I64)]);
        let src = "% let x = @count";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn pure_string_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("msg"), Ty::String)]);
        let src = "% let x = @msg";
        check_with_interner(src, &ctx, &i).unwrap();
    }
}
