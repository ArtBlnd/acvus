//! What typeck decides (typeck-obligations.md), keyed by the AST site.
//! `Output` is built while checking, at terms; `Typeck` is the frozen form
//! `writeback` alone makes.

use acvus_ast::AstId;
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

/// One name a lambda captured, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Captured {
    pub name: Astr,
    pub capture: Capture,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Closure {
    pub captures: Box<[Captured]>,
    pub calls: Calls,
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
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Term>>,
    pub conversions: FxHashMap<AstId, Conversion<Term>>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub closures: FxHashMap<AstId, Closure>,
    pub body_effect: Option<EffectTerm<InferPhase>>,
    pub refusals: Vec<Refusal>,
}

/// What typeck hands on: every decision at concrete types.
pub struct Typeck {
    pub types: FxHashMap<AstId, Ty>,
    pub names: FxHashMap<AstId, Name>,
    pub reached: FxHashMap<AstId, Chosen>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Ty>>,
    pub conversions: FxHashMap<AstId, Conversion<Ty>>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub closures: FxHashMap<AstId, Closure>,
    pub body_effect: Effect,
    pub refusals: Vec<Refusal>,
}
