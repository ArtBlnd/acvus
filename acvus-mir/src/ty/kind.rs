use acvus_utils::{Astr, QualifiedRef};

use super::arg::{EffectArg, IdentityTerm, LenTerm, TypeArg};
use super::effect::EffectTerm;
use super::num::IntTy;
use super::{Phase, TyId};
use crate::flows::FlowTerm;

/// One shape of a type. Its parts are ids of the same table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TyKind<V: Phase> {
    Int(IntTy),
    Float,
    Char,
    String,
    Bool,
    Unit,
    /// The bottom type (types §4).
    Never,
    /// The order token (effects §3). No program writes it.
    Order,
    /// A started call, taken exactly once (effects §6). No program writes it.
    Handle(TyId<V>),
    Array(TyId<V>, LenTerm<V>),
    /// The run of elements a container lends; only under a `Ref`.
    Slice(TyId<V>),
    /// The run of bytes a `String` lends; only under a shared `Ref`.
    Str,
    Tuple(Box<[TyId<V>]>),
    /// A primitive, closed like an extern enum (types §2).
    Option(TyId<V>),
    /// A primitive, closed like an extern enum (types §2).
    Result(TyId<V>, TyId<V>),
    Object(ObjectTy<V>),
    /// An enum the program names; its variants sorted by tag (types §2).
    Enum {
        name: QualifiedRef,
        variants: Box<[(Astr, Option<TyId<V>>)]>,
        label: V::Label,
    },
    Fn {
        params: Box<[ParamTerm<V>]>,
        ret: TyId<V>,
        effect: EffectTerm<V>,
        flows: FlowTerm<V>,
        calls: Calls,
    },
    /// An extern-defined type (extern-fn §8).
    UserDefined {
        id: QualifiedRef,
        type_args: Box<[TypeArg<V>]>,
        effect_args: Box<[EffectArg<V>]>,
        identity_args: Box<[IdentityTerm<V>]>,
    },
    Ref(Mutability, TypeArg<V>),
    Var(V::TyVar),
}

/// An object's fields, sorted by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ObjectTy<V: Phase> {
    /// A structural object; its label says whether it may grow.
    Structural(Box<[(Astr, TyId<V>)]>, V::Label),
    /// A struct a declaration names: extern-defined, opaque (types §1).
    Declared(QualifiedRef, Box<[(Astr, TyId<V>)]>),
}

impl<V: Phase> ObjectTy<V> {
    pub fn fields(&self) -> &[(Astr, TyId<V>)] {
        match self {
            ObjectTy::Structural(fields, _) | ObjectTy::Declared(_, fields) => fields,
        }
    }
}

/// Whether a function value may be called more than once (functions §3).
/// Two meet at `Once` where either is (types §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Calls {
    Once,
    Many,
}

impl Calls {
    pub fn join(self, other: Calls) -> Calls {
        match (self, other) {
            (Calls::Many, Calls::Many) => Calls::Many,
            (Calls::Once, _) | (_, Calls::Once) => Calls::Once,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParamTerm<V: Phase> {
    pub name: Astr,
    pub ty: TyId<V>,
}

/// Whether a reference may write through to its storage (ownership §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Mutability {
    Shared,
    Mut,
}

impl Mutability {
    /// Whether a reference of this mutability reaches a position of
    /// `wanted`'s: a `&mut` reaches a `&` by reborrow, a `&` never a `&mut`.
    pub fn reaches(self, wanted: Mutability) -> bool {
        self == wanted || wanted == Mutability::Shared
    }
}

impl<V: Phase> TyKind<V> {
    /// Every type id this shape holds directly.
    pub fn for_each_child(&self, mut on_child: impl FnMut(TyId<V>)) {
        match self {
            TyKind::Int(_)
            | TyKind::Float
            | TyKind::Char
            | TyKind::String
            | TyKind::Bool
            | TyKind::Unit
            | TyKind::Never
            | TyKind::Order
            | TyKind::Str
            | TyKind::Var(_) => {}
            TyKind::Handle(inner) | TyKind::Array(inner, _) | TyKind::Slice(inner) | TyKind::Option(inner) => {
                on_child(*inner)
            }
            TyKind::Result(ok, err) => {
                on_child(*ok);
                on_child(*err);
            }
            TyKind::Tuple(items) => items.iter().copied().for_each(on_child),
            TyKind::Object(object) => object.fields().iter().for_each(|(_, ty)| on_child(*ty)),
            TyKind::Enum { variants, .. } => variants
                .iter()
                .filter_map(|(_, payload)| *payload)
                .for_each(on_child),
            TyKind::Fn { params, ret, .. } => {
                params.iter().for_each(|param| on_child(param.ty));
                on_child(*ret);
            }
            TyKind::UserDefined { type_args, .. } => {
                type_args.iter().for_each(|arg| on_child(arg.ty()));
            }
            TyKind::Ref(_, target) => on_child(target.ty()),
        }
    }
}
