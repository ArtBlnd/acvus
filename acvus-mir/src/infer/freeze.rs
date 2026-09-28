//! Terms to concrete types, once solving ends (compilation: O(size)).

use rustc_hash::FxHashMap;

use super::cells::{Cells, MarkTerm, VarTerm};
use super::{Infer, Mark, Term, VarState};
use crate::flows::FlowTerm;
use crate::ty::{
    Concrete, EffectArg, EffectTerm, IdentityTerm, Infer as InferPhase, LenTerm, ObjectTy, ParamTerm, Ty, TyKind,
    TyVarId, TypeArg, Types,
};

/// Each term already frozen, so a shared subterm freezes once.
#[derive(Default)]
pub struct FreezeMemo {
    done: FxHashMap<Term, Ty>,
}

/// A variable still unbound at freeze: defaults and the final refusal ran
/// before freezing, so this is the checker's defect, reported as such.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsettled {
    Ty(TyVarId),
    Effect(crate::ty::EffectVarId),
    Len(crate::ty::LenVarId),
    Identity(crate::ty::IdentityVarId),
    Mark(crate::ty::ReprVarId),
    Flow(crate::ty::FlowVarId),
}

impl Infer {
    pub fn freeze(&self, into: &mut Types<Concrete>, memo: &mut FreezeMemo, term: Term) -> Result<Ty, Unsettled> {
        acvus_utils::grow(|| self.freeze_level(into, memo, term))
    }

    fn freeze_level(&self, into: &mut Types<Concrete>, memo: &mut FreezeMemo, term: Term) -> Result<Ty, Unsettled> {
        if let Some(done) = memo.done.get(&term) {
            return Ok(*done);
        }
        let frozen = match self.types.kind(term) {
            TyKind::Var(var) => match &self.ty_vars[var.0 as usize] {
                VarState::Unbound(_) => return Err(Unsettled { var: *var }),
                VarState::Bound(to) => self.freeze(into, memo, *to)?,
            },
            kind => {
                let kind = self.freeze_kind(into, memo, kind)?;
                into.intern(kind)
            }
        };
        memo.done.insert(term, frozen);
        Ok(frozen)
    }

    fn freeze_kind(
        &self,
        into: &mut Types<Concrete>,
        memo: &mut FreezeMemo,
        kind: &TyKind<InferPhase>,
    ) -> Result<TyKind<Concrete>, Unsettled> {
        let mut freeze = |term: Term| self.freeze(into, memo, term);
        Ok(match kind {
            TyKind::Int(width) => TyKind::Int(*width),
            TyKind::Float => TyKind::Float,
            TyKind::Char => TyKind::Char,
            TyKind::String => TyKind::String,
            TyKind::Bool => TyKind::Bool,
            TyKind::Unit => TyKind::Unit,
            TyKind::Never => TyKind::Never,
            TyKind::Order => TyKind::Order,
            TyKind::Str => TyKind::Str,
            TyKind::Handle(result) => TyKind::Handle(freeze(*result)?),
            TyKind::Array(element, len) => TyKind::Array(
                freeze(*element)?,
                LenTerm::Known(self.settled(&self.len_vars, len)),
            ),
            TyKind::Slice(element) => TyKind::Slice(freeze(*element)?),
            TyKind::Option(inner) => TyKind::Option(freeze(*inner)?),
            TyKind::Result(ok, err) => TyKind::Result(freeze(*ok)?, freeze(*err)?),
            TyKind::Tuple(items) => TyKind::Tuple(items.iter().map(|item| freeze(*item)).collect::<Result<_, _>>()?),
            TyKind::Object(ObjectTy::Structural(fields, _)) => TyKind::Object(ObjectTy::Structural(
                fields
                    .iter()
                    .map(|(name, field)| Ok((*name, freeze(*field)?)))
                    .collect::<Result<_, _>>()?,
                (),
            )),
            TyKind::Object(ObjectTy::Declared(id, fields)) => TyKind::Object(ObjectTy::Declared(
                *id,
                fields
                    .iter()
                    .map(|(name, field)| Ok((*name, freeze(*field)?)))
                    .collect::<Result<_, _>>()?,
            )),
            TyKind::Enum { name, variants, .. } => TyKind::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(tag, payload)| Ok((*tag, payload.map(&mut freeze).transpose()?)))
                    .collect::<Result<_, _>>()?,
                label: (),
            },
            TyKind::Fn {
                params,
                ret,
                effect,
                flows,
                calls,
                holds,
            } => TyKind::Fn {
                params: params
                    .iter()
                    .map(|param| {
                        Ok(ParamTerm {
                            name: param.name,
                            ty: freeze(param.ty)?,
                        })
                    })
                    .collect::<Result<_, _>>()?,
                ret: freeze(*ret)?,
                effect: EffectTerm::Known(self.settled(&self.effect_vars, effect)),
                flows: FlowTerm::Known(self.settled(&self.flow_vars, flows)),
                calls: *calls,
                holds: *holds,
            },
            TyKind::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => TyKind::UserDefined {
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|arg| self.freeze_type_arg(arg, &mut freeze))
                    .collect::<Result<_, _>>()?,
                effect_args: effect_args
                    .iter()
                    .map(|arg| match arg {
                        EffectArg::Uniform(effect) => EffectArg::Uniform(EffectTerm::Known(
                            self.settled(&self.effect_vars, effect),
                        )),
                        EffectArg::Specialized(effect) => EffectArg::Specialized(EffectTerm::Known(
                            self.settled(&self.effect_vars, effect),
                        )),
                    })
                    .collect(),
                identity_args: identity_args
                    .iter()
                    .map(|identity| IdentityTerm::Known(self.settled(&self.identity_vars, identity)))
                    .collect(),
            },
            TyKind::Ref(mutability, target) => TyKind::Ref(*mutability, self.freeze_type_arg(target, &mut freeze)?),
            TyKind::Var(_) => unreachable!("freeze_kind: freeze_level follows a variable before reading its kind"),
        })
    }

    fn freeze_type_arg(
        &self,
        arg: &TypeArg<InferPhase>,
        freeze: &mut impl FnMut(Term) -> Result<Ty, Unsettled>,
    ) -> Result<TypeArg<Concrete>, Unsettled> {
        let mark = match arg {
            TypeArg::Uniform(_) => Mark::Uniform,
            TypeArg::Specialized(_) => Mark::Specialized,
            TypeArg::Open(var, _) => self.settled(&self.repr_vars, &MarkTerm::Var(*var)),
        };
        let ty = freeze(arg.ty())?;
        Ok(match mark {
            Mark::Uniform => TypeArg::Uniform(ty),
            Mark::Specialized => TypeArg::Specialized(ty),
        })
    }

    /// The value a variable of one of the term kinds settled to.
    fn settled<T: VarTerm>(&self, cells: &Cells<T>, term: &T) -> T::Value {
        match cells.end(term).value() {
            Some(value) => value,
            None => todo!(
                "stop: an effect, length, identity, mark or flow variable unbound at freeze has no \
                 Unsettled form: Unsettled names only a type variable"
            ),
        }
    }
}
