//! The parts of a structural type (types §1, ownership §1): a tuple, an
//! object the program shapes, an array of a fixed length, an `Option` and a
//! `Result`. Each kind's parts are in the order the type table stores them,
//! so no interner is read. This is the one place a structural type is taken
//! apart; typeck, lower and every later reader ask it.
//!
//! A struct a declaration names is not structural: it is defined outside
//! the program and equals only itself (types §1).

use acvus_utils::Astr;
use smallvec::SmallVec;

use super::arg::LenTerm;
use super::kind::{ObjectTy, TyKind};
use super::{Phase, TyId, Types};

/// A structural type, by kind, with its parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structure<'t, V: Phase> {
    Tuple(&'t [TyId<V>]),
    /// The fields in the order the object's type stores them.
    Object(&'t [(Astr, TyId<V>)]),
    /// Every element is of `element`; there are `len` of them.
    Array { element: TyId<V>, len: LenTerm<V> },
    /// The `Some` payload.
    Option(TyId<V>),
    Result { ok: TyId<V>, err: TyId<V> },
}

/// `ty`'s structure, where `ty` is structural.
pub fn structure<V>(types: &Types<V>, ty: TyId<V>) -> Option<Structure<'_, V>>
where
    V: Phase,
{
    match types.kind(ty) {
        TyKind::Tuple(items) => Some(Structure::Tuple(items)),
        TyKind::Object(ObjectTy::Structural(fields, _)) => Some(Structure::Object(fields)),
        TyKind::Array(element, len) => Some(Structure::Array { element: *element, len: *len }),
        TyKind::Option(payload) => Some(Structure::Option(*payload)),
        TyKind::Result(ok, err) => Some(Structure::Result { ok: *ok, err: *err }),
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
        | TyKind::Enum { .. }
        | TyKind::Fn { .. }
        | TyKind::UserDefined { .. }
        | TyKind::Ref(..)
        | TyKind::Var(_) => None,
    }
}

impl<V: Phase> Structure<'_, V> {
    /// Each part's type, in order: a tuple's elements, an object's fields,
    /// an `Option`'s payload, a `Result`'s `Ok` payload then its `Err`
    /// payload. An array has one part, its element type, which every one of
    /// its elements has.
    pub fn parts(&self) -> SmallVec<[TyId<V>; 4]> {
        match *self {
            Structure::Tuple(items) => items.iter().copied().collect(),
            Structure::Object(fields) => fields.iter().map(|(_, field)| *field).collect(),
            Structure::Array { element, .. } | Structure::Option(element) => smallvec::smallvec![element],
            Structure::Result { ok, err } => smallvec::smallvec![ok, err],
        }
    }
}

#[cfg(test)]
mod tests {
    use acvus_utils::{Interner, QualifiedRef};

    use super::{Structure, structure};
    use crate::ty::{Concrete, IntTy, LenTerm, Mutability, ObjectTy, Ty, TyKind, TypeArg, Types};

    fn parts(types: &Types<Concrete>, ty: Ty) -> Vec<Ty> {
        structure(types, ty).expect("a structural type").parts().to_vec()
    }

    #[test]
    fn a_tuples_parts_are_its_elements_in_order() {
        let mut types = Types::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let tuple = types.intern(TyKind::Tuple(Box::new([string, int, string])));
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
        let Some(Structure::Object(fields)) = structure(&types, object) else {
            panic!("a structural object is an object");
        };
        let order: Vec<_> = fields.iter().map(|(name, _)| *name).collect();
        assert_eq!(order, vec![b, a]);
    }

    #[test]
    fn an_arrays_one_part_is_its_element_type() {
        let mut types = Types::new();
        let string = types.intern(TyKind::String);
        let array = types.intern(TyKind::Array(string, LenTerm::Known(3)));
        assert_eq!(parts(&types, array), vec![string]);
        assert_eq!(structure(&types, array), Some(Structure::Array { element: string, len: LenTerm::Known(3) }));
    }

    #[test]
    fn an_options_part_is_its_payload_and_a_results_are_ok_then_err() {
        let mut types = Types::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let option = types.intern(TyKind::Option(string));
        assert_eq!(parts(&types, option), vec![string]);
        let result = types.intern(TyKind::Result(string, int));
        assert_eq!(parts(&types, result), vec![string, int]);
    }

    /// types §1: a declared struct is opaque, so it has no parts; nor has a
    /// reference, an enum, `!` or a primitive.
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
