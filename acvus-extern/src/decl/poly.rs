//! A declared type as the compiler holds it (`Types<Poly>`), so that what the
//! compiler reads off a declared type, its segments first (`segments`), is
//! read by the compiler's own code, written once. A registry lowers every
//! declared type here.

use std::collections::HashMap;

use acvus_mir::decl::declared_fn_flows;
use acvus_mir::ty::{
    Calls, EffectTerm, Holds, IdentityTerm, LifetimesOf, Mark, MarkTerm, Mutability, Poly, TyId, TyKind, Types,
};
use acvus_utils::QualifiedRef;

use crate::decl::{DeclEffect, DeclTy};

/// Declared types lowered into one table, with the lifetimes each
/// extern-defined type they name declares.
pub struct Lowered {
    pub types: Types<Poly>,
    lifetimes: HashMap<QualifiedRef, usize>,
}

impl Lowered {
    pub fn new() -> Self {
        Lowered { types: Types::new(), lifetimes: HashMap::new() }
    }

    /// `ty` in the compiler's form. A declared function type is called many
    /// times (functions §3), may hold any loan in its captures, and has every
    /// flow (extern-fn §4); no declared type is marked.
    pub fn lower(&mut self, ty: &DeclTy) -> TyId<Poly> {
        let kind = match ty {
            DeclTy::Int(int) => TyKind::Int(*int),
            DeclTy::F64 => TyKind::Float,
            DeclTy::Bool => TyKind::Bool,
            DeclTy::Char => TyKind::Char,
            DeclTy::Unit => TyKind::Unit,
            DeclTy::String => TyKind::String,
            DeclTy::Str => TyKind::Str,
            DeclTy::Param(n) => TyKind::Var(*n),
            DeclTy::Never => TyKind::Never,
            DeclTy::Ref { mutable, target } => {
                let mutability = if *mutable { Mutability::Mut } else { Mutability::Shared };
                TyKind::Ref(mutability, self.lower(target))
            }
            DeclTy::Slice(element) => TyKind::Slice(self.lower(element)),
            DeclTy::Tuple(parts) => TyKind::Tuple(parts.iter().map(|part| self.lower(part)).collect(), ()),
            DeclTy::Option(part) => TyKind::Option(self.lower(part), ()),
            DeclTy::Result(ok, err) => TyKind::Result(self.lower(ok), self.lower(err), ()),
            DeclTy::Fn { params, result, effect } => TyKind::Fn {
                params: params.iter().map(|param| self.lower(param)).collect(),
                ret: self.lower(result),
                effect: effect_term(effect),
                flows: declared_fn_flows().into(),
                calls: Calls::Many,
                holds: Holds::Mut,
                values: (),
            },
            DeclTy::Extern { name, args, effect, identity, lifetimes } => {
                self.lifetimes.insert(*name, *lifetimes as usize);
                TyKind::UserDefined {
                    id: *name,
                    type_args: args.iter().map(|arg| self.lower(arg)).collect(),
                    effect: effect.as_ref().map(effect_term),
                    identity: identity.map(IdentityTerm::Var),
                    marked: MarkTerm::Known(Mark::Unmarked),
                }
            }
        };
        self.types.intern(kind)
    }
}

impl Default for Lowered {
    fn default() -> Self {
        Lowered::new()
    }
}

impl LifetimesOf for Lowered {
    /// # Panics
    /// `id` names no extern-defined type lowered here.
    fn lifetimes_of(&self, id: QualifiedRef) -> usize {
        *self.lifetimes.get(&id).expect("a lowered type names a lowered extern-defined type")
    }
}

fn effect_term(effect: &DeclEffect) -> EffectTerm<Poly> {
    match effect {
        DeclEffect::Known(effect) => EffectTerm::Known(effect.clone()),
        DeclEffect::Var(n) => EffectTerm::Var(*n),
    }
}

#[cfg(test)]
mod tests {
    use acvus_mir::decl::{Segment, segments};
    use acvus_mir::ty::IntTy;
    use acvus_utils::{Interner, QualifiedRef};

    use super::Lowered;
    use crate::decl::{DeclEffect, DeclTy};

    fn refs(names: &Interner) -> DeclTy {
        DeclTy::Extern {
            name: QualifiedRef::root(names.intern("Refs")),
            args: vec![DeclTy::Param(0)],
            effect: None,
            identity: None,
            lifetimes: 1,
        }
    }

    #[test]
    fn a_lowered_type_has_the_compilers_segments() {
        let names = Interner::new();
        let ty = DeclTy::Ref { mutable: false, target: Box::new(DeclTy::Tuple(vec![refs(&names), DeclTy::Int(IntTy::I64)])) };

        let mut lowered = Lowered::new();
        let id = lowered.lower(&ty);

        assert_eq!(segments(&lowered.types, &lowered, id), [Segment::Reference, Segment::Lifetime, Segment::TypeVar(0)]);
    }

    #[test]
    fn a_function_value_is_one_segment_whatever_its_type_holds() {
        let names = Interner::new();
        let ty = DeclTy::Fn {
            params: vec![DeclTy::Ref { mutable: false, target: Box::new(refs(&names)) }],
            result: Box::new(DeclTy::Param(1)),
            effect: DeclEffect::Var(0),
        };

        let mut lowered = Lowered::new();
        let id = lowered.lower(&ty);

        assert_eq!(segments(&lowered.types, &lowered, id), [Segment::FunctionValue]);
    }
}
