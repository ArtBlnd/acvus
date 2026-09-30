//! One store's terms brought into another: every variable of every kind
//! appended after the store's own, each kind moved by one count, and every
//! type re-interned with its parts and variables moved. What a unit's check
//! settled is read by its callers' checks this way (typeck::Interface): a
//! scheme's terms are the other store's, and after the import they are
//! this store's, the same terms, so the one instantiation a scheme has
//! copies them as it copies a scheme of this store's own.

use rustc_hash::FxHashMap;

use super::cells::Shift;
use super::{Infer, Term, VarState};
use crate::ty::{
    CallsTerm, EffectTerm, EffectVarId, FlowTerm, FlowVarId, HoldsTerm, IdentityTerm, IdentityVarId, Infer as InferPhase,
    LenTerm, LenVarId, MarkTerm, MarkVarId, ObjectTy, TyKind, TyVarId, ValuesVarId,
};

/// How an import moved the other store's terms into this one.
#[derive(Debug, Clone)]
pub struct Imported {
    /// Each of the other store's types, and its term here.
    terms: FxHashMap<Term, Term>,
    types: u32,
    effects: Shift,
    lengths: Shift,
    identities: Shift,
    marks: Shift,
    flows: Shift,
    holds: Shift,
    calls: Shift,
    values: Shift,
}

impl Infer {
    /// A store of `from`'s terms alone, for a unit's interface
    /// (typeck::Interface). It is no check, so each identity keeps the
    /// level `from` made it at: what `from`'s bodies made stays told from
    /// what came from outside them (types §3), which a canonical form
    /// needs. A check importing this store takes every one as its outside.
    pub fn copied_out(from: &Infer) -> (Infer, Imported) {
        let mut store = Infer::new();
        let imported = store.import(from);
        store.made_at.clone_from(&from.made_at);
        (store, imported)
    }

    /// Every term of `from` brought into this store (the module doc).
    pub fn import(&mut self, from: &Infer) -> Imported {
        let types = u32::try_from(self.ty_vars.len()).expect("Infer::import: u32::MAX type variables");
        let mut imported = Imported {
            terms: FxHashMap::default(),
            types,
            effects: self.effect_vars.import(&from.effect_vars),
            lengths: self.len_vars.import(&from.len_vars),
            identities: self.identity_vars.import(&from.identity_vars),
            marks: self.mark_vars.import(&from.mark_vars),
            flows: self.flow_vars.import(&from.flow_vars),
            holds: self.holds_vars.import(&from.holds_vars),
            calls: self.calls_vars.import(&from.calls_vars),
            values: self.values_vars.import(&from.values_vars),
        };
        for id in from.types.ids() {
            let kind = imported.kind(from.types.kind(id));
            let term = self.types.intern(kind);
            imported.terms.insert(id, term);
        }
        // A variable's state names terms, so the terms come first.
        self.ty_vars.extend(from.ty_vars.iter().map(|state| match state {
            VarState::Unbound(bound) => VarState::Unbound(bound.clone()),
            VarState::Bound(to) => VarState::Bound(imported.term(*to)),
        }));
        // A level is a depth of body instantiation, not a number of a
        // variable: each variable keeps the level it was made at in the
        // other store. An identity that store made came from outside this
        // check.
        self.ty_levels.extend(from.ty_levels.iter().copied());
        self.identity_levels.extend(from.identity_levels.iter().copied());
        for identity in from.made_at.keys() {
            self.made_at.entry(*identity).or_insert(super::Level::OUTSIDE);
        }
        self.reference_targets.extend(from.reference_targets.iter().map(|var| imported.ty_var(*var)));
        self.unmarked.extend(from.unmarked.iter().map(|var| imported.ty_var(*var)));
        imported
    }
}

impl Imported {
    /// The term the other store's `term` is here.
    ///
    /// # Panics
    /// `term` is not a term of the store imported.
    pub fn term(&self, term: Term) -> Term {
        *self.terms.get(&term).expect("Imported::term: a term of the store imported")
    }

    pub fn ty_var(&self, var: TyVarId) -> TyVarId {
        TyVarId(var.0 + self.types)
    }

    pub fn effect(&self, effect: &EffectTerm<InferPhase>) -> EffectTerm<InferPhase> {
        match effect {
            EffectTerm::Known(known) => EffectTerm::Known(known.clone()),
            EffectTerm::Var(var) => EffectTerm::Var(self.effect_var(*var)),
        }
    }

    pub fn effect_var(&self, var: EffectVarId) -> EffectVarId {
        EffectVarId(var.0 + self.effects.vars)
    }

    pub fn length_var(&self, var: LenVarId) -> LenVarId {
        LenVarId(var.0 + self.lengths.vars)
    }

    pub fn identity_var(&self, var: IdentityVarId) -> IdentityVarId {
        IdentityVarId(var.0 + self.identities.vars)
    }

    pub fn mark(&self, mark: MarkTerm<InferPhase>) -> MarkTerm<InferPhase> {
        match mark {
            MarkTerm::Known(known) => MarkTerm::Known(known),
            MarkTerm::Var(var) => MarkTerm::Var(MarkVarId(var.0 + self.marks.vars)),
        }
    }

    pub fn flow_var(&self, var: FlowVarId) -> FlowVarId {
        FlowVarId(var.0 + self.flows.vars)
    }

    pub fn values_var(&self, var: ValuesVarId) -> ValuesVarId {
        ValuesVarId(var.0 + self.values.vars)
    }

    /// The other store's `kind` with its parts and variables moved here.
    fn kind(&self, kind: &TyKind<InferPhase>) -> TyKind<InferPhase> {
        match kind {
            TyKind::Var(var) => TyKind::Var(self.ty_var(*var)),
            TyKind::Handle(inner) => TyKind::Handle(self.term(*inner)),
            TyKind::Array(element, len, label) => TyKind::Array(
                self.term(*element),
                match len {
                    LenTerm::Known(len) => LenTerm::Known(*len),
                    LenTerm::Var(var) => LenTerm::Var(self.length_var(*var)),
                },
                *label,
            ),
            TyKind::Slice(element) => TyKind::Slice(self.term(*element)),
            TyKind::Tuple(items, label) => TyKind::Tuple(items.iter().map(|item| self.term(*item)).collect(), *label),
            TyKind::Option(payload, label) => TyKind::Option(self.term(*payload), *label),
            TyKind::Result(ok, err, label) => TyKind::Result(self.term(*ok), self.term(*err), *label),
            TyKind::Object(ObjectTy::Structural(fields, label)) => {
                TyKind::Object(ObjectTy::Structural(fields.iter().map(|(name, field)| (*name, self.term(*field))).collect(), *label))
            }
            TyKind::Enum { name, variants, label } => TyKind::Enum {
                name: *name,
                variants: variants.iter().map(|(tag, payload)| (*tag, payload.map(|payload| self.term(payload)))).collect(),
                label: *label,
            },
            TyKind::Fn { params, ret, effect, flows, calls, holds, values } => TyKind::Fn {
                params: params.iter().map(|param| self.term(*param)).collect(),
                ret: self.term(*ret),
                effect: self.effect(effect),
                flows: match flows {
                    FlowTerm::Known(known) => FlowTerm::Known(known.clone()),
                    FlowTerm::Var(var) => FlowTerm::Var(self.flow_var(*var)),
                },
                calls: match calls {
                    CallsTerm::Known(known) => CallsTerm::Known(*known),
                    CallsTerm::Var(var) => CallsTerm::Var(crate::ty::CallsVarId(var.0 + self.calls.vars)),
                },
                holds: match holds {
                    HoldsTerm::Known(known) => HoldsTerm::Known(*known),
                    HoldsTerm::Var(var) => HoldsTerm::Var(crate::ty::HoldsVarId(var.0 + self.holds.vars)),
                },
                values: self.values_var(*values),
            },
            TyKind::UserDefined { id, type_args, effect, identity, marked } => TyKind::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|arg| self.term(*arg)).collect(),
                effect: effect.as_ref().map(|effect| self.effect(effect)),
                identity: identity.map(|identity| match identity {
                    IdentityTerm::Known(known) => IdentityTerm::Known(known),
                    IdentityTerm::Var(var) => IdentityTerm::Var(self.identity_var(var)),
                }),
                marked: self.mark(*marked),
            },
            TyKind::Ref(mutability, target) => TyKind::Ref(*mutability, self.term(*target)),
            TyKind::Closures(signatures) => TyKind::Closures(signatures.iter().map(|signature| self.term(*signature)).collect()),
            leaf @ (TyKind::Int(_)
            | TyKind::Float
            | TyKind::Char
            | TyKind::String
            | TyKind::Bool
            | TyKind::Unit
            | TyKind::Never
            | TyKind::Order
            | TyKind::Str) => leaf.clone(),
        }
    }
}
