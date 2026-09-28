//! What typeck decides (typeck-obligations.md), keyed by the AST site.
//! `Output` is built while checking, at terms; `Typeck` is the frozen form
//! `writeback` alone makes.

use acvus_ast::{AstId, Span};
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use super::refusal::Refusal;
use crate::infer::Term;
use crate::ty::{CastTy, Calls, Effect, EffectTerm, Infer as InferPhase, Mutability, Ty};

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
    pub to: T,
}

/// How a lambda holds a name it captured (functions §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capture {
    Move,
    Ref(Mutability),
}

/// One name a lambda captured, the binding it names by its binder's
/// site, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Captured {
    pub name: Astr,
    pub binder: AstId,
    pub capture: Capture,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Closure {
    pub captures: Box<[Captured]>,
    pub calls: Calls,
}

/// How `==` or an ordering compares its operands' type (operators §4):
/// part by part, each part as it compares, or through the `core::eq` or
/// `core::cmp` impl reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Comparison {
    Parts(Box<[Comparison]>),
    Reached(Chosen),
}

/// A comparison node while checking: its parts' nodes, or a reach at the
/// node itself, recorded in `reached` once it fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Comparing {
    Parts(Box<[AstId]>),
    Reached,
}

/// A value that entered a slot and was folded there (types §3); its
/// coercion is read at writeback, from the value's final type up to the
/// slot's, since the slot may rise after the value entered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Entered {
    pub site: AstId,
    pub span: Span,
    pub value: Term,
    pub slot: Term,
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
    pub names: FxHashMap<AstId, Name>,
    pub reached: FxHashMap<AstId, Chosen>,
    /// Each reached call's function type as its declaration was
    /// instantiated there, at the site `reached` has it.
    pub callees: FxHashMap<AstId, Term>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Term>>,
    /// Each value converted by a declared conversion (types §7), at its
    /// site.
    pub conversions: FxHashMap<AstId, Conversion<Term>>,
    /// Each value folded into a slot, in the order it entered.
    pub entered: Vec<Entered>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub closures: FxHashMap<AstId, Closure>,
    pub comparisons: FxHashMap<AstId, Comparing>,
    pub body_effect: Option<EffectTerm<InferPhase>>,
    pub refusals: Vec<Refusal>,
}

/// What typeck hands on: every decision at concrete types.
pub struct Typeck {
    pub types: FxHashMap<AstId, Ty>,
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
    pub comparisons: FxHashMap<AstId, Comparison>,
    pub body_effect: Effect,
    pub refusals: Vec<Refusal>,
}
