//! The parts of a transparent structured type (types §1): an object, an
//! enum, a tuple, an array, an `Option` and a `Result`, and no other. Each
//! kind's parts are in the order the type table stores them, so no interner
//! is read. This is the one place such a type is taken apart; typeck, lower
//! and every later reader ask it.
//!
//! A struct a declaration names is not one: it is defined outside the
//! program and equals only itself (types §1).

use acvus_utils::Astr;
use smallvec::SmallVec;

use super::arg::LenTerm;
use super::kind::{ObjectTy, TyKind};
use super::{Phase, TyId, Types};

/// A transparent structured type, by kind, with its parts and who wrote
/// it (`Label`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structure<'t, V: Phase> {
    Tuple(&'t [TyId<V>], V::Label),
    /// The fields in the order the object's type stores them.
    Object(&'t [(Astr, TyId<V>)], V::Label),
    /// The variants sorted by tag, each with its payload if it has one.
    Enum(&'t [(Astr, Option<TyId<V>>)], V::Label),
    /// Every element is of `element`; there are `len` of them.
    Array { element: TyId<V>, len: LenTerm<V>, label: V::Label },
    /// The `Some` payload.
    Option(TyId<V>, V::Label),
    Result { ok: TyId<V>, err: TyId<V>, label: V::Label },
}

/// Which part of a structure a part is: what a coercion, a lifted impl and
/// a comparison match two structures' parts by (types §1, §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartKey {
    /// A tuple's element at its position.
    Index(usize),
    Field(Astr),
    /// An enum variant's payload.
    Variant(Astr),
    /// An array's elements, which share one part.
    Element,
    Some,
    Ok,
    Err,
}

/// One part of a structure: which it is, and its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part<V: Phase> {
    pub key: PartKey,
    pub ty: TyId<V>,
}

/// `ty`'s structure, where `ty` is a transparent structured type.
pub fn structure<V>(types: &Types<V>, ty: TyId<V>) -> Option<Structure<'_, V>>
where
    V: Phase,
{
    match types.kind(ty) {
        TyKind::Tuple(items, label) => Some(Structure::Tuple(items, *label)),
        TyKind::Object(ObjectTy::Structural(fields, label)) => Some(Structure::Object(fields, *label)),
        TyKind::Enum { variants, label, .. } => Some(Structure::Enum(variants, *label)),
        TyKind::Array(element, len, label) => Some(Structure::Array { element: *element, len: *len, label: *label }),
        TyKind::Option(payload, label) => Some(Structure::Option(*payload, *label)),
        TyKind::Result(ok, err, label) => Some(Structure::Result { ok: *ok, err: *err, label: *label }),
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Order
        | TyKind::Handle(_)
        | TyKind::Slice(_)
        | TyKind::Str
        | TyKind::Object(ObjectTy::Declared(..))
        | TyKind::Fn { .. }
        | TyKind::UserDefined { .. }
        | TyKind::Ref(..)
        | TyKind::Var(_) => None,
    }
}

impl<V: Phase> Structure<'_, V> {
    /// Who wrote it (`Label`).
    pub fn label(&self) -> V::Label {
        match *self {
            Structure::Tuple(_, label)
            | Structure::Object(_, label)
            | Structure::Enum(_, label)
            | Structure::Array { label, .. }
            | Structure::Option(_, label)
            | Structure::Result { label, .. } => label,
        }
    }

    /// Each part, keyed, in order: a tuple's elements, an object's fields,
    /// an enum's variant payloads, an `Option`'s `Some` payload, a
    /// `Result`'s `Ok` payload then its `Err` payload. An array has one part,
    /// `Element`, which every one of its elements has.
    pub fn parts(&self) -> SmallVec<[Part<V>; 4]> {
        let part = |key, ty| Part { key, ty };
        match *self {
            Structure::Tuple(items, _) => items.iter().enumerate().map(|(at, ty)| part(PartKey::Index(at), *ty)).collect(),
            Structure::Object(fields, _) => fields.iter().map(|(name, ty)| part(PartKey::Field(*name), *ty)).collect(),
            Structure::Enum(variants, _) => variants
                .iter()
                .filter_map(|(tag, payload)| payload.map(|ty| part(PartKey::Variant(*tag), ty)))
                .collect(),
            Structure::Array { element, .. } => smallvec::smallvec![part(PartKey::Element, element)],
            Structure::Option(payload, _) => smallvec::smallvec![part(PartKey::Some, payload)],
            Structure::Result { ok, err, .. } => smallvec::smallvec![part(PartKey::Ok, ok), part(PartKey::Err, err)],
        }
    }
}

#[cfg(test)]
mod tests {
    use acvus_utils::{Interner, QualifiedRef};

    use super::{PartKey, Structure, structure};
    use crate::ty::{Concrete, IntTy, LenTerm, Mutability, ObjectTy, Ty, TyKind, TypeArg, Types};

    fn parts(types: &Types<Concrete>, ty: Ty) -> Vec<Ty> {
        structure(types, ty).expect("a structural type").parts().iter().map(|part| part.ty).collect()
    }

    #[test]
    fn a_tuples_parts_are_its_elements_in_order() {
        let mut types = Types::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let tuple = types.intern(TyKind::Tuple(Box::new([string, int, string]), ()));
        assert_eq!(parts(&types, tuple), vec![string, int, string]);
    }

    #[test]
    fn an_objects_parts_are_its_fields_in_the_order_its_type_stores() {
        let names = Interner::new();
        let (a, b) = (names.intern("a"), names.intern("b"));
        let mut types = Types::new();
        let (int, bool_ty) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::Bool));
        let object = types.intern(TyKind::Object(ObjectTy::Structural(Box::new([(b, bool_ty), (a, int)]), ())));
        assert_eq!(parts(&types, object), vec![bool_ty, int]);
        let Some(Structure::Object(fields, _)) = structure(&types, object) else {
            panic!("a structural object is an object");
        };
        let order: Vec<_> = fields.iter().map(|(name, _)| *name).collect();
        assert_eq!(order, vec![b, a]);
    }

    #[test]
    fn an_arrays_one_part_is_its_element_type() {
        let mut types = Types::new();
        let string = types.intern(TyKind::String);
        let array = types.intern(TyKind::Array(string, LenTerm::Known(3), ()));
        assert_eq!(parts(&types, array), vec![string]);
        assert_eq!(structure(&types, array), Some(Structure::Array { element: string, len: LenTerm::Known(3), label: () }));
    }

    #[test]
    fn an_options_part_is_its_payload_and_a_results_are_ok_then_err() {
        let mut types = Types::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let option = types.intern(TyKind::Option(string, ()));
        assert_eq!(parts(&types, option), vec![string]);
        let result = types.intern(TyKind::Result(string, int, ()));
        assert_eq!(parts(&types, result), vec![string, int]);
    }

    /// types §1: an enum's parts are its variants' payloads, each keyed by
    /// its variant; a variant of no payload has none.
    #[test]
    fn an_enums_parts_are_its_variants_payloads() {
        let names = Interner::new();
        let (none, some) = (names.intern("Empty"), names.intern("Full"));
        let mut types = Types::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let variants = Box::new([(none, None), (some, Some(int))]);
        let r#enum = types.intern(TyKind::Enum { name: QualifiedRef::root(names.intern("E")), variants, label: () });
        let keys: Vec<_> = structure(&types, r#enum).expect("an enum is structured").parts().iter().map(|part| part.key).collect();
        assert_eq!(keys, vec![PartKey::Variant(some)]);
        assert_eq!(parts(&types, r#enum), vec![int]);
    }

    /// types §1: a declared struct is opaque, so it has no parts; nor has a
    /// reference, `!` or a primitive.
    #[test]
    fn a_declared_struct_a_reference_and_a_primitive_are_not_structural() {
        let names = Interner::new();
        let a = names.intern("a");
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let declared = types.intern(TyKind::Object(ObjectTy::Declared(QualifiedRef::root(names.intern("S")), Box::new([(a, int)]))));
        let reference = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
        let never = types.intern(TyKind::Never);
        let string = types.intern(TyKind::String);
        for ty in [declared, reference, never, string, int] {
            assert_eq!(structure(&types, ty), None);
        }
    }
}
