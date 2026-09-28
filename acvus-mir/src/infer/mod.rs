//! B, the typing rule (rfc_compiler: types): terms under solving and the
//! relations between them. It knows no syntax. C's label is on the terms it
//! relates (`ty::Label`); it asks nothing of the program.
//!
//! A variable is bound at most once to a term, or rebound only to a larger
//! term of one head under `join` where the label allows growth, or under
//! `enter` along the cast tree. Knowledge only grows (types §6).

mod closed;
mod freeze;
mod relate;
mod scheme;

use crate::decl::TyVarBound;
use crate::ty::{
    EffectTerm, EffectVarId, FlowVarId, IdentityId, IdentityVarId, Infer as InferPhase, IntTy, LenVarId, ReprVarId, TyId,
    TyKind, TyVarId, Types,
};

pub use closed::{Closed, OpenSources};
pub use freeze::{FreezeMemo, Unsettled};
pub use relate::{Entered, Mismatch};
pub use scheme::{Instantiated, RequiredAt};

pub type Term = TyId<InferPhase>;

/// What a type variable is while solving.
#[derive(Debug, Clone, PartialEq)]
pub enum VarState {
    Unbound(Bound),
    Bound(Term),
}

/// A variable's bound, instantiated (extern-fn §1, types §5).
#[derive(Debug, Clone, PartialEq)]
pub enum Bound {
    Any,
    OneOf(Box<[Term]>),
    Integer { signed: bool, among: Box<[IntTy]> },
    Settled,
    /// An integer literal's width, which a use narrows and the default
    /// settles to `i64` (types §5).
    Literal,
}

/// What an open `#` settles to (types §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Uniform,
    Specialized,
}

/// The solving state of one checked unit.
pub struct Infer {
    types: Types<InferPhase>,
    ty_vars: Vec<VarState>,
    effect_vars: Vec<Option<EffectTerm<InferPhase>>>,
    len_vars: Vec<Option<usize>>,
    identity_vars: Vec<Option<IdentityId>>,
    repr_vars: Vec<Option<Mark>>,
    flow_vars: Vec<Option<crate::flows::Flows>>,
}

impl Infer {
    pub fn new() -> Self {
        Self {
            types: Types::new(),
            ty_vars: Vec::new(),
            effect_vars: Vec::new(),
            len_vars: Vec::new(),
            identity_vars: Vec::new(),
            repr_vars: Vec::new(),
            flow_vars: Vec::new(),
        }
    }

    pub fn types(&self) -> &Types<InferPhase> {
        &self.types
    }

    /// A term of a written shape. A structural object or enum the program
    /// writes carries `Label::Transparent`; one instantiated from a
    /// declaration carries `Label::Opaque` (`scheme`).
    pub fn intern(&mut self, kind: TyKind<InferPhase>) -> Term {
        self.types.intern(kind)
    }

    pub fn fresh(&mut self, bound: Bound) -> Term {
        let var = TyVarId(u32::try_from(self.ty_vars.len()).expect("Infer::fresh: u32::MAX variables"));
        self.ty_vars.push(VarState::Unbound(bound));
        self.types.intern(TyKind::Var(var))
    }

    pub fn fresh_effect(&mut self) -> EffectVarId {
        let var = EffectVarId(u32::try_from(self.effect_vars.len()).expect("Infer::fresh_effect: u32::MAX variables"));
        self.effect_vars.push(None);
        var
    }

    pub fn fresh_len(&mut self) -> LenVarId {
        let var = LenVarId(u32::try_from(self.len_vars.len()).expect("Infer::fresh_len: u32::MAX variables"));
        self.len_vars.push(None);
        var
    }

    pub fn fresh_identity(&mut self) -> IdentityVarId {
        let var = IdentityVarId(
            u32::try_from(self.identity_vars.len()).expect("Infer::fresh_identity: u32::MAX variables"),
        );
        self.identity_vars.push(None);
        var
    }

    pub fn fresh_repr(&mut self) -> ReprVarId {
        let var = ReprVarId(u32::try_from(self.repr_vars.len()).expect("Infer::fresh_repr: u32::MAX variables"));
        self.repr_vars.push(None);
        var
    }

    pub fn fresh_flow(&mut self) -> FlowVarId {
        let var = FlowVarId(u32::try_from(self.flow_vars.len()).expect("Infer::fresh_flow: u32::MAX variables"));
        self.flow_vars.push(None);
        var
    }

    /// `ty` with every bound variable at its root followed; a term whose
    /// root is an unbound variable is that variable.
    pub fn shallow(&self, ty: Term) -> Term {
        let mut at = ty;
        while let TyKind::Var(var) = self.types.kind(at)
            && let VarState::Bound(to) = &self.ty_vars[var.0 as usize]
        {
            at = *to;
        }
        at
    }

    /// The unbound variables, with their bounds: what defaults read at a
    /// stall (types §5).
    pub fn unbound(&self) -> impl Iterator<Item = (TyVarId, &Bound)> {
        self.ty_vars.iter().enumerate().filter_map(|(index, state)| match state {
            VarState::Unbound(bound) => Some((
                TyVarId(u32::try_from(index).expect("Infer::unbound: indices are u32 by fresh")),
                bound,
            )),
            VarState::Bound(_) => None,
        })
    }

    /// A declaration's bound as a variable's (`scheme`).
    pub(crate) fn bound_of(&mut self, _bound: &TyVarBound) -> Bound {
        todo!("infer::Infer::bound_of: instantiate a declared bound's shapes")
    }
}

impl Default for Infer {
    fn default() -> Self {
        Self::new()
    }
}
