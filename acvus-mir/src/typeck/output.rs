//! What typeck decides (typeck-obligations.md), keyed by the AST site.
//! `Output` is built while checking, at terms; `Typeck` is the frozen form
//! `writeback` alone makes.

use acvus_ast::{AstId, Span};
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use super::refusal::Refusal;
use crate::infer::{FlowId, Term};
use crate::infer::ShapeCoercion;
use crate::ty::structural::PartKey;
use crate::ty::{CallsVarId, CastTy, Calls, Effect, EffectTerm, HoldsVarId, Infer as InferPhase, Mutability, Ty};

/// What a name at a site is (types §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Name {
    /// A binding, by its binder's site.
    Binding(AstId),
    Declaration(QualifiedRef),
    /// A unit a call reaches by its whole name (programs).
    Unit(QualifiedRef),
    Context(QualifiedRef),
    Input(Astr),
}

/// A call of a unit (programs): the unit, and for each of its parameters,
/// in order, the index of the argument written for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitCall {
    pub unit: QualifiedRef,
    pub arguments: Box<[usize]>,
}

/// Which impl of a generic function a call reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Impl {
    Concrete(usize),
    Generic,
    /// The impl a transparent structured type has by its generic
    /// function's lift (types §1): each part's impl is reached at a node of
    /// its own, recorded in `structures` at the call's site.
    Structural,
}

/// A reached declaration at the impl chosen, with what its
/// requirements chose (extern-fn §6).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Chosen {
    pub declaration: QualifiedRef,
    pub r#impl: Option<Impl>,
    pub required: Box<[Chosen]>,
}

/// What a `for` iterates by (control-flow). By reference, its elements are
/// reached through the `as_slice` or `as_slice_mut` recorded at its callee
/// site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iteration {
    ByRef(Mutability),
    ByValue,
    Range,
}

/// What a `?` unwraps and the type it leaves through (control-flow).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryExit<T> {
    Option(T),
    Result(T),
}

/// A coercion applied without being written: a view (operators §5) or a
/// declared cast or conversion (types §3, §7), each through the function
/// that performs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion<T> {
    pub via: Chosen,
    /// `via`'s function type, instantiated at this coercion's two ends.
    pub via_ty: T,
    pub to: T,
}

/// A declared conversion applied while checking (types §7): its `via`
/// reached at `node`, where `reached` records what it chose, and its
/// function type there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Converted {
    pub node: AstId,
    pub via_ty: Term,
    pub to: Term,
}

/// How a lambda holds a name it captured (functions §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capture {
    Move,
    Ref(Mutability),
}

/// What a captured name names: a binding, by its binder's site, or an
/// input (programs §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureOf {
    Binding(AstId),
    Input(Astr),
}

/// One name a lambda captured, what it names, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Captured {
    pub name: Astr,
    pub binder: CaptureOf,
    pub capture: Capture,
}

/// How a method call's receiver is taken (types §6): as written, or lent
/// `&` or `&mut` where only that reaches a declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjustment {
    AsWritten,
    Ref(Mutability),
}

/// A name read inside the lambdas `crossed`, innermost first, each of which
/// captures it; `mutability` is how the read uses it. The capture's mode is
/// set from every such use once the types are solved (functions §2).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CaptureUse {
    pub name: Astr,
    pub of: CaptureOf,
    pub crossed: Box<[AstId]>,
    pub mutability: Mutability,
    /// The captured value's term, whose loans a capture by move holds.
    pub term: Term,
}

/// A value of term `term` read whole out of what the `move` lambdas
/// `lambdas` captured: where its type does not copy, each of them moves it
/// out and is called once (functions §3).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WholeUse {
    pub lambdas: Box<[AstId]>,
    pub term: Term,
}

/// A lambda checked: whether it is written `move`, the variables its
/// function type's `holds` and `calls` are, which settling its captures
/// binds (functions §2-3), and its effect, where a refusal of what it
/// needs is made at its span (D94).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Lambda {
    pub site: AstId,
    pub span: Span,
    pub by_move: bool,
    pub holds: HoldsVarId,
    pub calls: CallsVarId,
    pub effect: EffectTerm<InferPhase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Closure {
    pub captures: Box<[Captured]>,
    pub calls: Calls,
}

/// A transparent structured type taken apart down to the impls its parts
/// reach, node by node, in `ty::structural`'s order: how `==` or an
/// ordering compares (operators §4), and how a lifted impl answers
/// (types §1, `Impl::Structural`). A reference compares what it names, and
/// `!` has no part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructureTree {
    Parts(Box<[(PartKey, StructureTree)]>),
    /// The impl reached, and its function type as instantiated there.
    Reached { chosen: Chosen, callee: Ty },
    /// The language's own answer, where no impl is reached.
    Own(Own),
}

/// What the language itself answers at a node of a lifted call, where it
/// defines the answer and no impl does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Own {
    /// A `clone` of a value that copies is its copy (ownership §2).
    Copy,
    /// `Option`'s `default` is `None` (operators §2).
    OptionNone,
    /// `as_ref` or `as_mut`: a structure's parts, each lent at this
    /// mutability through the reference taken, one level (operators §2,
    /// ownership §5).
    Lend(Mutability),
}

/// A structure-tree node while checking: its parts' nodes, or a reach at
/// the node itself, recorded in `reached` once it fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum StructureNode {
    Parts(Box<[(PartKey, AstId)]>),
    Reached,
    Own(Own),
}

/// How a value is coerced to its slot (types §3, §7): a concrete value by
/// the steps up the tree or its one declared conversion, in order; a
/// transparent structured value by its shape into the slot's type `to`,
/// and each part it keeps by its own coercion, `None` where the part
/// enters as it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coercion {
    Steps(Box<[Conversion<Ty>]>),
    /// `through` is `Some` where the value is a reference whose target is
    /// coerced, the result a reference built of references to the kept
    /// parts (ownership §5).
    Structured { shape: ShapeCoercion, through: Option<Mutability>, to: Ty, parts: Box<[(PartKey, Option<Coercion>)]> },
}

impl Coercion {
    /// The type the value is coerced to.
    ///
    /// # Panics
    /// No step: writeback records a coercion of at least one step.
    pub fn to(&self) -> Ty {
        match self {
            Coercion::Steps(steps) => steps.last().expect("a recorded coercion has a step").to,
            Coercion::Structured { to, .. } => *to,
        }
    }
}

/// A structured value's coercion while checking: its shape, the slot it
/// enters, and each part it keeps at a site of its own, where that part's
/// coercion is recorded.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CoercionNode {
    pub shape: ShapeCoercion,
    pub through: Option<Mutability>,
    pub slot: Term,
    pub parts: Box<[(PartKey, AstId)]>,
}

/// A value that entered a slot and was folded there (types §3); its
/// coercion is read at writeback, from the value's final type up to the
/// slot's, since the slot may rise after the value entered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Entered {
    pub site: AstId,
    pub span: Span,
    pub value: Term,
    pub slot: Term,
    /// The effect of the body the value entered in, which each step's
    /// function joins (effects §1).
    pub body: EffectTerm<InferPhase>,
}

/// An `as` cast, at either end a `CastTy` (operators §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsCast {
    pub from: CastTy,
    pub to: CastTy,
}

/// The output while checking.
#[derive(Default)]
pub(super) struct Output {
    /// Each generalized binding (functions §2), by `SchemeId`.
    pub schemes: Vec<super::generalize::LetScheme>,
    /// Each unit whose interface this check brought in, by the scheme it is
    /// here (`generalize::import`).
    pub imported: FxHashMap<acvus_utils::QualifiedRef, super::generalize::SchemeId>,
    /// Every scheme an import brought in: another check's, whose values are
    /// not this check's to make.
    pub foreign: rustc_hash::FxHashSet<super::generalize::SchemeId>,
    /// Each instance of one, by `InstanceId`.
    pub instances: Vec<super::generalize::Instance>,
    /// The instance whose requirement is firing, where one is: what fires
    /// records at the lambda body's sites within it.
    pub firing_in: Option<super::generalize::InstanceId>,
    /// Each owner's walks (S7): a lambda's body, or the values given a
    /// binding, by the lambda's or the binder's site.
    pub walks: FxHashMap<AstId, super::generalize::Walk>,
    /// Each site recorded while an owner's walk ran, with that owner.
    pub site_owners: FxHashMap<AstId, AstId>,
    /// Whether a generalized binding's local fixpoint is running.
    pub settling: bool,
    /// Every term a place's step reached through a reference, recorded by
    /// the decision that made it as it fired: a `*`'s target, and a field,
    /// index or pattern part past a reference or past a term here. Every
    /// step after one on its path is reached so too (types §1). Each holds
    /// the least mutability of the references on its path, what the path
    /// reaches exclusively only where every one is `&mut` (ownership §5).
    pub through: FxHashMap<Term, Mutability>,
    pub types: FxHashMap<AstId, Term>,
    pub names: FxHashMap<AstId, Name>,
    pub reached: FxHashMap<AstId, Chosen>,
    /// Each call of a unit, at the call's site.
    pub units: FxHashMap<AstId, UnitCall>,
    /// Each reached call's function type as its declaration or its unit's
    /// interface was instantiated there, at the site `reached` or `units`
    /// has it.
    pub callees: FxHashMap<AstId, Term>,
    /// Each reached call whose result as declared holds a mark its value
    /// starts in a fresh cell of its own, with that declared result
    /// (types §8).
    pub origins: FxHashMap<AstId, Term>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Term>>,
    /// Each value converted by a declared conversion (types §7), at its
    /// site.
    pub conversions: FxHashMap<AstId, Converted>,
    /// Each structured value's coercion (types §3), at its site.
    pub coercions: FxHashMap<AstId, CoercionNode>,
    /// Each value folded into a slot, in the order it entered.
    pub entered: Vec<Entered>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub closures: FxHashMap<AstId, Closure>,
    /// Each lambda, in the order its body's check ended.
    pub lambdas: Vec<Lambda>,
    /// Each flow of a call's effect into its body's (effects §1), at the
    /// call's span, where a refusal of it is made.
    pub effect_flows: FxHashMap<FlowId, Span>,
    pub capture_uses: Vec<CaptureUse>,
    pub whole_uses: Vec<WholeUse>,
    /// Each method call's receiver adjustment, at the call's callee site.
    pub adjustments: FxHashMap<AstId, Adjustment>,
    /// Each comparison's and each lifted impl's structure node.
    pub structures: FxHashMap<AstId, StructureNode>,
    pub body_effect: Option<EffectTerm<InferPhase>>,
    /// The unit's interface, where its inputs were open (`generalize::export`).
    pub interface: Option<super::interface::Interface>,
    pub refusals: Vec<Refusal>,
}

/// What typeck hands on: every decision at concrete types.
pub struct Typeck {
    pub types: FxHashMap<AstId, Ty>,
    pub names: FxHashMap<AstId, Name>,
    pub reached: FxHashMap<AstId, Chosen>,
    /// Each call of a unit, at the call's site.
    pub units: FxHashMap<AstId, UnitCall>,
    /// Each reached call's function type at its site, the site `reached`
    /// or `units` has it at. A call's result is its declared one.
    pub callees: FxHashMap<AstId, Ty>,
    /// Each reached call whose uses settled its result at a type other than
    /// its declared result, which it differs from only in marks raised
    /// (types §8): the type its value is taken at.
    pub unmarks: FxHashMap<AstId, Ty>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Ty>>,
    /// Each coercion, at the site of the value coerced.
    pub coercions: FxHashMap<AstId, Coercion>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub closures: FxHashMap<AstId, Closure>,
    /// Each method call's receiver adjustment, at the call's callee site,
    /// where `reached` has the call (types §6).
    pub adjustments: FxHashMap<AstId, Adjustment>,
    /// Each comparison's tree, at the operator's site, and each lifted
    /// impl's, at the site of the call that reached it.
    pub structure_trees: FxHashMap<AstId, StructureTree>,
    /// Each instance of a generalized binding (functions §2): what was
    /// decided at the lambda bodies' sites within it, which lower reads to
    /// specialize the code to it over the one environment.
    pub instances: FxHashMap<super::generalize::InstanceId, InstanceDecided>,
    /// Each generalized binding, by its binder: the lambdas written as its
    /// values, whose code lower makes at each of its instances' signatures.
    pub generalized: FxHashMap<AstId, Box<[AstId]>>,
    /// Each site a generalized binding's value passes through where it is
    /// written, its binder included, with that binder: a value there holds
    /// every signature the binding has (functions §2).
    pub value_sites: FxHashMap<AstId, AstId>,
    pub body_effect: Effect,
    /// The unit's interface, where its inputs were open: its scheme, which
    /// every caller instantiates (programs §1-§2).
    pub interface: Option<super::interface::Interface>,
    /// The run's result type: the slot every tail and `return` enters, and
    /// for a template the `String` it builds. `None` only where a refusal
    /// left it open.
    pub result: Option<Ty>,
    pub refusals: Vec<Refusal>,
}

/// What an instance decided at the sites of the lambda bodies it
/// specializes: each site's type, and each decision recorded there, as
/// `Typeck` keeps them for the rest of the run.
#[derive(Debug, Clone, Default)]
pub struct InstanceDecided {
    /// The binder of the generalized binding this is an instance of.
    pub binder: Option<AstId>,
    /// The use this instance is.
    pub use_site: Option<AstId>,
    /// The instance of an enclosing binding it was made within.
    pub within: Option<super::generalize::InstanceId>,
    pub types: FxHashMap<AstId, Ty>,
    pub reached: FxHashMap<AstId, Chosen>,
    pub callees: FxHashMap<AstId, Ty>,
    pub unmarks: FxHashMap<AstId, Ty>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Ty>>,
    pub coercions: FxHashMap<AstId, Coercion>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub adjustments: FxHashMap<AstId, Adjustment>,
    pub structure_trees: FxHashMap<AstId, StructureTree>,
}
