use acvus_utils::{Astr, QualifiedRef};

use super::arg::{IdentityTerm, LenTerm, MarkTerm};
use super::effect::EffectTerm;
use super::flow::FlowTerm;
use super::num::IntTy;
use super::{CallsVarId, HoldsVarId, Phase, TyId};

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
    Array(TyId<V>, LenTerm<V>, V::Label),
    /// The run of elements a container lends; only under a `Ref`.
    Slice(TyId<V>),
    /// The run of bytes a `String` lends; only under a shared `Ref`.
    Str,
    Tuple(Box<[TyId<V>]>, V::Label),
    /// A primitive, closed like an extern enum (types §2).
    Option(TyId<V>, V::Label),
    /// A primitive, closed like an extern enum (types §2).
    Result(TyId<V>, TyId<V>, V::Label),
    Object(ObjectTy<V>),
    /// An enum the program names; its variants sorted by tag (types §2).
    Enum {
        name: QualifiedRef,
        variants: Box<[(Astr, Option<TyId<V>>)]>,
        label: V::Label,
    },
    Fn {
        params: Box<[TyId<V>]>,
        ret: TyId<V>,
        effect: EffectTerm<V>,
        flows: FlowTerm<V>,
        calls: V::CallsTerm,
        holds: V::HoldsTerm,
        /// Which function values it may be (types §2).
        values: V::ValuesTerm,
    },
    /// An extern-defined type (extern-fn §8).
    UserDefined {
        id: QualifiedRef,
        type_args: Box<[TyId<V>]>,
        /// The one effect it carries, apart from its type arguments
        /// (`T<…> with E`), where it declares one: an upper bound on the
        /// work its values hold. The type is covariant in it, as a function
        /// type is in its effect (extern-fn §8, types §2).
        effect: Option<EffectTerm<V>>,
        /// The one identity it carries, where it declares one (types §3).
        identity: Option<IdentityTerm<V>>,
        /// `#` (types §8): a member of the type, not its constructor, known
        /// once nothing can meet in it (§6). Marked below unmarked, it joins
        /// as an effect does: a value an extern fn gives marked starts so,
        /// each use taking it unmarked raises it, and a `#` parameter fixes
        /// it. Only a whole concrete extern-defined type is ever marked.
        marked: MarkTerm<V>,
    },
    Ref(Mutability, TyId<V>),
    /// A closure with code at each of a set of function types over one
    /// environment: what a generalized binding holds once lowered, each a
    /// signature one of its instances has (functions §2). The set is
    /// written in one canonical order, by `TyId::canonical_key`, only so
    /// that one set interns to one type; the order is no layout, and how a
    /// backend holds such a closure is its own (undefined behaviour §2).
    /// A set of one signature is written as that function type, never as a
    /// `Closures` of one, so that one set is written one way; a `Closures`
    /// holds none or at least two. None is the closure of a binding no use
    /// reaches: making it is still an effect, its captures moved or lent,
    /// and it has no code, only its environment. Only lower writes it; no
    /// program and no declaration does (`InternalType`).
    Closures(Box<[TyId<V>]>),
    Var(V::TyVar),
}

/// An object's fields, sorted by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ObjectTy<V: Phase> {
    /// A structural object; its label says who wrote it (`Label`). An
    /// object a declaration writes is this shape, written elsewhere, so
    /// closed as written (rfc_extern types §1, types §1).
    Structural(Box<[(Astr, TyId<V>)]>, V::Label),
}

impl<V: Phase> ObjectTy<V> {
    pub fn fields(&self) -> &[(Astr, TyId<V>)] {
        match self {
            ObjectTy::Structural(fields, _) => fields,
        }
    }
}

/// The strongest loan a function value's captures hold, at its one
/// position (ownership §5). Two join at the stronger (types §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Holds {
    Nothing,
    Shared,
    Mut,
}

impl Holds {
    pub fn join(self, other: Holds) -> Holds {
        self.max(other)
    }
}

/// A function type's `holds` while solving: known, or a variable that
/// every copy of the type shares and that joins where two function types
/// join (types §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HoldsTerm {
    Known(Holds),
    Var(HoldsVarId),
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

/// A function type's `calls` while solving, as `HoldsTerm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallsTerm {
    Known(Calls),
    Var(CallsVarId),
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

    /// The weaker of the two: what a path through both reaches, exclusive
    /// only where every reference on it is (ownership §5).
    pub fn least(self, other: Mutability) -> Mutability {
        match (self, other) {
            (Mutability::Mut, Mutability::Mut) => Mutability::Mut,
            _ => Mutability::Shared,
        }
    }
}

impl<V: Phase> TyKind<V> {
    /// This shape with each type id it holds directly replaced by `map`'s,
    /// everything else as it is.
    pub fn with_children(&self, map: &mut impl FnMut(TyId<V>) -> TyId<V>) -> TyKind<V> {
        match self {
            TyKind::Handle(inner) => TyKind::Handle(map(*inner)),
            TyKind::Array(element, len, label) => TyKind::Array(map(*element), *len, *label),
            TyKind::Slice(element) => TyKind::Slice(map(*element)),
            TyKind::Option(payload, label) => TyKind::Option(map(*payload), *label),
            TyKind::Result(ok, err, label) => {
                let ok = map(*ok);
                TyKind::Result(ok, map(*err), *label)
            }
            TyKind::Tuple(items, label) => TyKind::Tuple(items.iter().map(|item| map(*item)).collect(), *label),
            TyKind::Object(ObjectTy::Structural(fields, label)) => {
                TyKind::Object(ObjectTy::Structural(fields.iter().map(|(name, field)| (*name, map(*field))).collect(), *label))
            }
            TyKind::Enum { name, variants, label } => TyKind::Enum {
                name: *name,
                variants: variants.iter().map(|(tag, payload)| (*tag, payload.map(|payload| map(payload)))).collect(),
                label: *label,
            },
            TyKind::Fn { params, ret, effect, flows, calls, holds, values } => TyKind::Fn {
                params: params.iter().map(|param| map(*param)).collect(),
                ret: map(*ret),
                effect: effect.clone(),
                flows: flows.clone(),
                calls: *calls,
                holds: *holds,
                values: values.clone(),
            },
            TyKind::UserDefined { id, type_args, effect, identity, marked } => TyKind::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|type_arg| map(*type_arg)).collect(),
                effect: effect.clone(),
                identity: *identity,
                marked: *marked,
            },
            TyKind::Ref(mutability, target) => TyKind::Ref(*mutability, map(*target)),
            TyKind::Closures(signatures) => TyKind::Closures(signatures.iter().map(|signature| map(*signature)).collect()),
            leaf @ (TyKind::Int(_)
            | TyKind::Float
            | TyKind::Char
            | TyKind::String
            | TyKind::Bool
            | TyKind::Unit
            | TyKind::Never
            | TyKind::Order
            | TyKind::Str
            | TyKind::Var(_)) => leaf.clone(),
        }
    }

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
            TyKind::Handle(inner) | TyKind::Array(inner, _, _) | TyKind::Slice(inner) | TyKind::Option(inner, _) => {
                on_child(*inner)
            }
            TyKind::Result(ok, err, _) => {
                on_child(*ok);
                on_child(*err);
            }
            TyKind::Tuple(items, _) => items.iter().copied().for_each(on_child),
            TyKind::Object(object) => object.fields().iter().for_each(|(_, ty)| on_child(*ty)),
            TyKind::Enum { variants, .. } => variants
                .iter()
                .filter_map(|(_, payload)| *payload)
                .for_each(on_child),
            TyKind::Fn { params, ret, .. } => {
                params.iter().copied().for_each(&mut on_child);
                on_child(*ret);
            }
            TyKind::UserDefined { type_args, .. } => type_args.iter().copied().for_each(on_child),
            TyKind::Ref(_, target) => on_child(*target),
            TyKind::Closures(signatures) => signatures.iter().copied().for_each(on_child),
        }
    }
}
