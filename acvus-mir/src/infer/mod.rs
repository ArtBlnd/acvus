//! B, the typing rule (rfc_compiler: types): terms under solving and the
//! relations between them. It knows no syntax. C's label is on the terms it
//! relates (`ty::Label`); it asks nothing of the program.
//!
//! A variable is bound at most once to a term, or rebound only: to a larger
//! term of one head under `join` where the label allows growth; up the cast
//! tree under `enter`; or to the more opaque of two equal shapes under
//! `eq` (types §1, §6). Knowledge only grows.

mod bound;
mod cells;
mod closed;
mod freeze;
mod relate;
mod scheme;
#[cfg(test)]
mod tests;

use crate::decl::TyVarBound;
use crate::flows::FlowTerm;
use crate::ty::{
    EffectTerm, EffectVarId, FlowVarId, IdentityTerm, IdentityVarId, Infer as InferPhase, IntTy, LenTerm, LenVarId,
    ReprVarId, TyId, TyKind, TyVarId, Types,
};

use cells::{Cells, MarkTerm};

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
    /// An integer type among `among`, signed only where `signed`. A literal
    /// (`literal`) is one whose width a use narrows; at a stall it is `i64`
    /// if `among` has it, the one width if one remains, and refused
    /// otherwise (types §5).
    Integer { signed: bool, among: Box<[IntTy]>, literal: bool },
    Settled,
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
    effect_vars: Cells<EffectTerm<InferPhase>>,
    len_vars: Cells<LenTerm<InferPhase>>,
    identity_vars: Cells<IdentityTerm<InferPhase>>,
    repr_vars: Cells<MarkTerm>,
    flow_vars: Cells<FlowTerm<InferPhase>>,
}

/// Where a term's chain of bound variables ends: at an unbound variable,
/// or at a node, reached through `holder`, the last variable on the way.
#[derive(Debug, Clone, Copy)]
enum End {
    Unbound(TyVarId, Term),
    Node { holder: Option<(TyVarId, Term)>, node: Term },
}

impl Infer {
    pub fn new() -> Self {
        Self {
            types: Types::new(),
            ty_vars: Vec::new(),
            effect_vars: Cells::new(),
            len_vars: Cells::new(),
            identity_vars: Cells::new(),
            repr_vars: Cells::new(),
            flow_vars: Cells::new(),
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
        self.effect_vars.fresh("Infer::fresh_effect: u32::MAX variables")
    }

    pub fn fresh_len(&mut self) -> LenVarId {
        self.len_vars.fresh("Infer::fresh_len: u32::MAX variables")
    }

    pub fn fresh_identity(&mut self) -> IdentityVarId {
        self.identity_vars.fresh("Infer::fresh_identity: u32::MAX variables")
    }

    pub fn fresh_repr(&mut self) -> ReprVarId {
        self.repr_vars.fresh("Infer::fresh_repr: u32::MAX variables")
    }

    pub fn fresh_flow(&mut self) -> FlowVarId {
        self.flow_vars.fresh("Infer::fresh_flow: u32::MAX variables")
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

    fn end(&self, term: Term) -> End {
        let mut at = term;
        let mut holder = None;
        while let TyKind::Var(var) = self.types.kind(at) {
            match &self.ty_vars[var.0 as usize] {
                VarState::Unbound(_) => return End::Unbound(*var, at),
                VarState::Bound(to) => {
                    holder = Some((*var, at));
                    at = *to;
                }
            }
        }
        End::Node { holder, node: at }
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
    pub(crate) fn bound_of(&mut self, bound: &TyVarBound) -> Bound {
        match bound {
            TyVarBound::Any => Bound::Any,
            TyVarBound::Settled => Bound::Settled,
            TyVarBound::Integer { signed, among } => Bound::Integer {
                signed: *signed,
                among: among.clone(),
            },
            TyVarBound::OneOf(shapes) => todo!(
                "stop: a OneOf bound's {} shapes are TyId<Poly> in the declarations' table, which bound_of \
                 cannot read, and a shape may name the scheme's own variables, which bound_of cannot map",
                shapes.len()
            ),
        }
    }
}

impl Default for Infer {
    fn default() -> Self {
        Self::new()
    }
}
