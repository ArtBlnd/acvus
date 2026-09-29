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

use crate::ty::{
    EffectTerm, EffectVarId, FlowTerm, FlowVarId, IdentityTerm, IdentityVarId, Infer as InferPhase, IntTy, LenTerm,
    LenVarId, ReprVarId, TyId, TyKind, TyVarId, Types,
};

use rustc_hash::FxHashSet;

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
    /// Every variable a reference's target is, which types §4 keeps from
    /// being bound to a reference.
    reference_targets: FxHashSet<TyVarId>,
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
            reference_targets: FxHashSet::default(),
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
    /// declaration carries `Label::Opaque` (`scheme`). A product with a
    /// part of type `!` is `!` (types §4). A reference's target is never a
    /// reference (types §4): a variable it targets is marked so, and is
    /// never bound to one.
    ///
    /// # Panics
    /// `kind` is a reference whose target is a reference, through bound
    /// variables or not. Every caller builds a reference over a target it
    /// already holds to types §4.
    pub fn intern(&mut self, kind: TyKind<InferPhase>) -> Term {
        if let TyKind::Ref(_, target) = &kind {
            match self.end(target.ty()) {
                End::Unbound(var, _) => {
                    self.reference_targets.insert(var);
                }
                End::Node { node, .. } => assert!(
                    !matches!(self.types.kind(node), TyKind::Ref(..)),
                    "a reference's target is never a reference (types §4)"
                ),
            }
        }
        let bottom = product_parts(&kind).any(|part| matches!(self.types.kind(part), TyKind::Never));
        self.types.intern(if bottom { TyKind::Never } else { kind })
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

    /// types §5: an effect nothing constrains is opaque, and an argument
    /// whose `#` nothing decides is not specialized. Every such variable
    /// still open takes its default; whether any did. Solving calls this
    /// only at a stall.
    pub fn default_effects_and_marks(&mut self) -> bool {
        let effects = self.effect_vars.default_unbound(&crate::ty::Effect::opaque());
        let marks = self.repr_vars.default_unbound(&Mark::Uniform);
        effects || marks
    }

    /// types §5: a function type's flows nothing decides state every flow,
    /// as a declaration's default does (extern-fn §4). Every flow set still
    /// open takes it, but those `claimed`, which a lambda's claim decides
    /// (ownership §5); whether any did. Solving calls this only at a stall.
    pub(crate) fn default_unclaimed_flows(&mut self, claimed: &[FlowTerm<InferPhase>]) -> bool {
        let kept: Vec<FlowVarId> = claimed.iter().filter_map(|flows| self.flow_vars.unbound_end(flows)).collect();
        self.flow_vars.default_unbound_but(&crate::ty::Flows::Every, &kept)
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
}

/// The parts of a product: a tuple's items, an object's fields. An array is
/// not among them, since one of length zero holds no part.
pub(super) fn product_parts<V: crate::ty::Phase>(kind: &TyKind<V>) -> impl Iterator<Item = TyId<V>> + '_ {
    let parts: &[TyId<V>] = match kind {
        TyKind::Tuple(items) => items,
        _ => &[],
    };
    let fields = match kind {
        TyKind::Object(object) => object.fields(),
        _ => &[],
    };
    parts.iter().copied().chain(fields.iter().map(|(_, field)| *field))
}

impl Default for Infer {
    fn default() -> Self {
        Self::new()
    }
}
