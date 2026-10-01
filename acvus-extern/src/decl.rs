//! A declaration as the extern side states it (compiler: extern fn; rfc_extern
//! macros, registry): plain values built from Rust types, which the registry
//! combines into the compiler's input. Nothing here checks a declaration
//! against its body; the compiler trusts it absolutely.
//!
//! A declared type is never written by hand: a Rust type states its own
//! through `DeclType`, and a parameter's Rust type picks its mode through
//! `ParamMode` (macros §4). A declaration is built with its generics filled by
//! stand-ins: `Nth<N>` for the N-th type parameter, `EffectNth<N>` and
//! `IdentityNth<N>`.

pub mod ty;

use acvus_mir::decl::NamedFact;
use acvus_mir::ty::IntTy;
use acvus_utils::Astr;

pub use acvus_mir::decl::{DeclaredFacts, TyVarBound};
pub use acvus_mir::ty::{Contexts, Effect, Reissue, SplitCost};

pub use acvus_utils::{Interner, QualifiedRef};
pub use ty::{DeclCx, DeclType, EffectNth, IdentityNth, Nth, ParamDecl, ParamMode, Part, Passed, Var, kind, part};

/// A declared type (compiler: extern fn §1, §8; types §1-§3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclTy {
    Int(IntTy),
    F64,
    Bool,
    Char,
    Unit,
    String,
    /// The text a `String` lends; only under a shared `Ref`.
    Str,
    /// The declaration's N-th type parameter.
    Param(u32),
    /// No value: the result of a call that never returns.
    Never,
    Ref { mutable: bool, target: Box<DeclTy> },
    /// The run a program's array lends; only under a `Ref` (crossing §2).
    Slice(Box<DeclTy>),
    Tuple(Vec<DeclTy>),
    Option(Box<DeclTy>),
    Result(Box<DeclTy>, Box<DeclTy>),
    /// A callback: its flows are every flow (types §5).
    Fn { params: Vec<DeclTy>, result: Box<DeclTy>, effect: DeclEffect },
    /// An extern-defined type (extern fn §8).
    Extern {
        name: QualifiedRef,
        args: Vec<DeclTy>,
        effect: Option<DeclEffect>,
        identity: Option<u32>,
        lifetimes: u32,
    },
}

/// An effect a declared type carries: stated, or the N-th effect variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclEffect {
    Known(Effect),
    Var(u32),
}

/// A declaration's binders, each by position (compiler: extern fn §1-§2,
/// types §3).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Binders {
    pub types: Vec<TyVarBound>,
    pub effects: u32,
    pub identities: u32,
}

/// An impl a declaration requires at its own type variables (extern fn §6):
/// the generic function, its pattern at the declaration's variables, and the
/// fact the impl must state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub function: QualifiedRef,
    pub pattern: Vec<DeclTy>,
    pub fact: Option<NamedFact>,
}

/// What else a declaration is (extern fn §6-§7).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Role {
    #[default]
    Function,
    /// An impl of this generic function; reached only through it.
    ImplOf(QualifiedRef),
    /// A step up the tree of casts (types §3).
    Cast,
    /// A conversion, no part of that tree (types §7).
    Conversion,
}

/// Everything the compiler knows of one extern fn.
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    /// The Rust module path and the function's name, or its stated name for
    /// an overload within one module.
    pub path: QualifiedRef,
    pub binders: Binders,
    pub params: Vec<(Astr, DeclTy)>,
    pub result: DeclTy,
    /// The call's own effect (extern fn §2).
    pub effect: Effect,
    /// The effect variables whose work the call runs before its result is
    /// taken.
    pub runs: Vec<u32>,
    /// The declared facts, the lifetimes and how a call ends (extern fn
    /// §3-§4), as the compiler holds them.
    pub facts: DeclaredFacts,
    pub requires: Vec<Requirement>,
    pub role: Role,
}
