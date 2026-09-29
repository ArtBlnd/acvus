//! What typeck decides (typeck-obligations.md), keyed by the AST site.
//! `Output` is built while checking, at terms; `Typeck` is the frozen form
//! `writeback` alone makes.

use acvus_ast::{AstId, Span};
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use super::refusal::Refusal;
use crate::infer::{FlowId, Term};
use crate::ty::{CallsVarId, CastTy, Calls, Effect, EffectTerm, HoldsVarId, Infer as InferPhase, Mutability, Ty};

/// What a name at a site is (types §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Name {
    /// A binding, by its binder's site.
    Binding(AstId),
    Declaration(QualifiedRef),
    Context(QualifiedRef),
    Input(Astr),
}

/// Which impl of a generic function a call reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Impl {
    Concrete(usize),
    Generic,
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

/// How `==`, an ordering or `.clone()` treats its operand's type, node by
/// node: part by part for a structural type in `ty::structural`'s order, an
/// array by its one element part (operators §4, ownership §1-2); through
/// the `core::eq`, `core::cmp` or `core::clone` impl reached; or, for a
/// clone, copied where the node's type copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartTree {
    Parts(Box<[PartTree]>),
    /// The impl reached, and its function type as instantiated there.
    Reached { chosen: Chosen, callee: Ty },
    /// A clone of a node whose type copies (ownership §1).
    Copied,
}

/// A part-tree node while checking: its parts' nodes, a reach at the node
/// itself, recorded in `reached` once it fires, or a copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PartNode {
    Parts(Box<[AstId]>),
    Reached,
    Copied,
}

/// A clone's node while checking, and its type: a structural node whose
/// type turns out to copy is copied whole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CloneNode {
    pub node: PartNode,
    pub ty: Term,
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
    pub types: FxHashMap<AstId, Term>,
    /// Each field access's and index's part, at its site; `types` has what
    /// a read of it gives (D108 rule 1).
    pub parts: FxHashMap<AstId, Term>,
    pub names: FxHashMap<AstId, Name>,
    pub reached: FxHashMap<AstId, Chosen>,
    /// Each reached call's function type as its declaration was
    /// instantiated there, at the site `reached` has it.
    pub callees: FxHashMap<AstId, Term>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Term>>,
    /// Each value converted by a declared conversion (types §7), at its
    /// site.
    pub conversions: FxHashMap<AstId, Converted>,
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
    pub comparisons: FxHashMap<AstId, PartNode>,
    pub clones: FxHashMap<AstId, CloneNode>,
    pub body_effect: Option<EffectTerm<InferPhase>>,
    pub refusals: Vec<Refusal>,
}

/// What typeck hands on: every decision at concrete types.
pub struct Typeck {
    pub types: FxHashMap<AstId, Ty>,
    /// Each field access's and index's part, at its site; `types` has what
    /// a read of it gives (D108 rule 1).
    pub parts: FxHashMap<AstId, Ty>,
    pub names: FxHashMap<AstId, Name>,
    pub reached: FxHashMap<AstId, Chosen>,
    /// Each reached call's function type at its site, the site `reached`
    /// has it at.
    pub callees: FxHashMap<AstId, Ty>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Ty>>,
    /// Each coercion at a site, its steps in order.
    pub conversions: FxHashMap<AstId, Box<[Conversion<Ty>]>>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub closures: FxHashMap<AstId, Closure>,
    /// Each method call's receiver adjustment, at the call's callee site,
    /// where `reached` has the call (types §6).
    pub adjustments: FxHashMap<AstId, Adjustment>,
    /// Each comparison's tree, at the operator's site, and each
    /// `.clone()`'s, at its callee site.
    pub part_trees: FxHashMap<AstId, PartTree>,
    pub body_effect: Effect,
    /// The run's result type: the slot every tail and `return` enters, and
    /// for a template the `String` it builds. `None` only where a refusal
    /// left it open.
    pub result: Option<Ty>,
    pub refusals: Vec<Refusal>,
}
