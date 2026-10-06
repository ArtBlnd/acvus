//! The parts of a transparent structured type (types §1): an object, an
//! enum, a tuple, an array, an `Option` and a `Result`, and no other. Each
//! kind's parts are in the order the type table stores them, so no interner
//! is read. This is the one place such a type is taken apart; typeck, lower
//! and every later reader ask it. A sequence's element is read here too
//! (`element`), a slice's among them, though a slice is no structure.
//!
//! A struct a declaration names is not one: it is defined outside the
//! program and equals only itself (types §1).

use acvus_utils::Astr;
use smallvec::SmallVec;

use super::arg::LenTerm;
use super::kind::TyKind;
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
    /// An array's elements, which share one part.
    Element,
    /// The payload of a variant, where it carries one.
    Payload(VariantKey),
}

/// Which variant of a variant type a value holds (types §1-2): an
/// `Option`'s `Some` or `None`, a `Result`'s `Ok` or `Err`, or an enum's
/// variant by its name. It names the variant by the type alone, so no
/// reader needs the names the program writes `Option`'s and `Result`'s
/// variants with to tell which one it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VariantKey {
    Some,
    None,
    Ok,
    Err,
    /// An enum's variant of this name.
    Enum(Astr),
}

/// One variant of a variant type: which it is, and its payload's type
/// where it carries one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variant<V: Phase> {
    pub key: VariantKey,
    pub payload: Option<TyId<V>>,
}

/// One part of a structure: which it is, and its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part<V: Phase> {
    pub key: PartKey,
    pub ty: TyId<V>,
}

/// The element of a sequence (operators §3): an array's, its one part
/// `Element` (types §1), and a slice's, which is no structure; `None` for
/// any other type. The one place a sequence's element is read.
pub fn element<V>(types: &Types<V>, ty: TyId<V>) -> Option<TyId<V>>
where
    V: Phase,
{
    match types.kind(ty) {
        TyKind::Slice(element) => Some(*element),
        _ => structure(types, ty)?.part(PartKey::Element),
    }
}

/// `ty`'s structure, where `ty` is a transparent structured type.
pub fn structure<V>(types: &Types<V>, ty: TyId<V>) -> Option<Structure<'_, V>>
where
    V: Phase,
{
    match types.kind(ty) {
        TyKind::Tuple(items, label) => Some(Structure::Tuple(items, *label)),
        TyKind::Object { fields, label } => Some(Structure::Object(fields, *label)),
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
        | TyKind::Fn { .. }
        | TyKind::Closures(_)
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

    /// The type of the part `key` names, where the structure has it: a
    /// tuple's element at its position, a field, an array's element, or the
    /// payload of a variant (`variants`).
    pub fn part(&self, key: PartKey) -> Option<TyId<V>> {
        match (*self, key) {
            (Structure::Tuple(items, _), PartKey::Index(at)) => items.get(at).copied(),
            (Structure::Object(fields, _), PartKey::Field(name)) => {
                fields.iter().find(|(field, _)| *field == name).map(|(_, ty)| *ty)
            }
            (Structure::Array { element, .. }, PartKey::Element) => Some(element),
            (_, PartKey::Payload(variant)) => self.variants()?.into_iter().find(|held| held.key == variant)?.payload,
            _ => None,
        }
    }

    /// The variants of a variant type, in order, each with its payload's
    /// type where it has one: `Some` then `None`, `Ok` then `Err`, an enum's
    /// in the order its type stores them. `None` for a structure that is no
    /// variant type. The one place a variant type's variants are listed.
    pub fn variants(&self) -> Option<SmallVec<[Variant<V>; 4]>> {
        let variant = |key, payload| Variant { key, payload };
        match *self {
            Structure::Option(payload, _) => {
                Some(smallvec::smallvec![variant(VariantKey::Some, Some(payload)), variant(VariantKey::None, None)])
            }
            Structure::Result { ok, err, .. } => {
                Some(smallvec::smallvec![variant(VariantKey::Ok, Some(ok)), variant(VariantKey::Err, Some(err))])
            }
            Structure::Enum(variants, _) => Some(variants.iter().map(|(name, payload)| variant(VariantKey::Enum(*name), *payload)).collect()),
            Structure::Tuple(..) | Structure::Object(..) | Structure::Array { .. } => None,
        }
    }

    /// Whether the structure is a variant type with the variant `variant`.
    pub fn has_variant(&self, variant: VariantKey) -> bool {
        self.variants().is_some_and(|variants| variants.iter().any(|held| held.key == variant))
    }

    /// A product's fields, keyed, in order: a tuple's elements by their
    /// positions, an object's fields in the order its type stores them.
    /// `None` for a structure that is no tuple or object. The one place a
    /// product's fields are listed.
    pub fn fields(&self) -> Option<SmallVec<[Part<V>; 4]>> {
        let part = |key, ty| Part { key, ty };
        match *self {
            Structure::Tuple(items, _) => Some(items.iter().enumerate().map(|(at, ty)| part(PartKey::Index(at), *ty)).collect()),
            Structure::Object(fields, _) => Some(fields.iter().map(|(name, ty)| part(PartKey::Field(*name), *ty)).collect()),
            Structure::Enum(..) | Structure::Array { .. } | Structure::Option(..) | Structure::Result { .. } => None,
        }
    }

    /// Each part, keyed, in order: a product's fields (`fields`), a variant
    /// type's payloads in its variants' order (`variants`). An array has one
    /// part, `Element`, which every one of its elements has.
    pub fn parts(&self) -> SmallVec<[Part<V>; 4]> {
        let part = |key, ty| Part { key, ty };
        match *self {
            Structure::Tuple(..) | Structure::Object(..) => self.fields().expect("a tuple and an object are products"),
            Structure::Array { element, .. } => smallvec::smallvec![part(PartKey::Element, element)],
            Structure::Enum(..) | Structure::Option(..) | Structure::Result { .. } => self
                .variants()
                .expect("an enum, an `Option` and a `Result` are variant types")
                .into_iter()
                .filter_map(|variant| Some(part(PartKey::Payload(variant.key), variant.payload?)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use acvus_utils::{Interner, QualifiedRef};

    use super::{PartKey, Structure, VariantKey, element, structure};
    use crate::ty::{Mark, MarkTerm};
    use crate::ty::{Concrete, IntTy, LenTerm, Mutability, Ty, TyKind, Types};

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
        let object = types.intern(TyKind::Object { fields: Box::new([(b, bool_ty), (a, int)]), label: () });
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
        assert_eq!(keys, vec![PartKey::Payload(VariantKey::Enum(some))]);
        assert_eq!(parts(&types, r#enum), vec![int]);
    }

    /// `part` names each part `parts` lists by its key, and nothing for a key
    /// of another kind or a position, field or variant the type does not
    /// have.
    #[test]
    fn a_part_by_its_key_is_the_part_parts_lists() {
        let names = Interner::new();
        let (a, full, empty) = (names.intern("a"), names.intern("Full"), names.intern("Empty"));
        let mut types = Types::<Concrete>::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let structured = [
            types.intern(TyKind::Tuple(Box::new([string, int]), ())),
            types.intern(TyKind::Object { fields: Box::new([(a, int)]), label: () }),
            types.intern(TyKind::Enum {
                name: QualifiedRef::root(names.intern("E")),
                variants: Box::new([(empty, None), (full, Some(int))]),
                label: (),
            }),
            types.intern(TyKind::Array(string, LenTerm::Known(2), ())),
            types.intern(TyKind::Option(string, ())),
            types.intern(TyKind::Result(string, int, ())),
        ];
        for ty in structured {
            let structure = structure(&types, ty).expect("a structural type");
            for part in structure.parts() {
                assert_eq!(structure.part(part.key), Some(part.ty), "{:?}", part.key);
            }
        }
        let tuple = structure(&types, structured[0]).expect("a tuple");
        assert_eq!(tuple.part(PartKey::Index(2)), None);
        assert_eq!(tuple.part(PartKey::Element), None);
        let r#enum = structure(&types, structured[2]).expect("an enum");
        assert_eq!(r#enum.part(PartKey::Payload(VariantKey::Enum(empty))), None, "a variant of no payload");
        assert_eq!(r#enum.part(PartKey::Payload(VariantKey::Enum(a))), None, "no such variant");
    }

    /// types §1-2: a variant type's variants are named by the type alone, in
    /// order, each with its payload's type; a key of another kind of variant
    /// type, or a variant the type does not have, names none.
    #[test]
    fn a_variant_types_variants_are_keyed_by_the_type() {
        let names = Interner::new();
        let (full, empty) = (names.intern("Full"), names.intern("Empty"));
        let mut types = Types::<Concrete>::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let option = types.intern(TyKind::Option(string, ()));
        let result = types.intern(TyKind::Result(string, int, ()));
        let r#enum = types.intern(TyKind::Enum {
            name: QualifiedRef::root(names.intern("E")),
            variants: Box::new([(empty, None), (full, Some(int))]),
            label: (),
        });
        let variants = |ty| {
            let variants = structure(&types, ty).and_then(|structure| structure.variants()).expect("a variant type");
            variants.iter().map(|variant| (variant.key, variant.payload)).collect::<Vec<_>>()
        };
        assert_eq!(variants(option), vec![(VariantKey::Some, Some(string)), (VariantKey::None, None)]);
        assert_eq!(variants(result), vec![(VariantKey::Ok, Some(string)), (VariantKey::Err, Some(int))]);
        assert_eq!(variants(r#enum), vec![(VariantKey::Enum(empty), None), (VariantKey::Enum(full), Some(int))]);
        let result = structure(&types, result).expect("a result");
        assert_eq!(result.part(PartKey::Payload(VariantKey::Err)), Some(int));
        assert_eq!(result.part(PartKey::Payload(VariantKey::Some)), None, "an `Option`'s key names no `Result` variant");
        assert!(result.has_variant(VariantKey::Ok) && !result.has_variant(VariantKey::None));
        let r#enum = structure(&types, r#enum).expect("an enum");
        assert_eq!(r#enum.part(PartKey::Payload(VariantKey::Enum(empty))), None, "a variant of no payload");
        assert!(!r#enum.has_variant(VariantKey::Enum(names.intern("Other"))));
        let tuple = types.intern(TyKind::Tuple(Box::new([int]), ()));
        assert_eq!(structure(&types, tuple).expect("a tuple").variants(), None);
    }

    /// A product's fields are a tuple's elements by position and an
    /// object's fields in its stored order, and they are its parts; an
    /// array and a variant type are no product.
    #[test]
    fn a_products_fields_are_its_parts_and_no_other_structure_has_fields() {
        let names = Interner::new();
        let (a, b) = (names.intern("a"), names.intern("b"));
        let mut types = Types::<Concrete>::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let tuple = types.intern(TyKind::Tuple(Box::new([string, int]), ()));
        let object = types.intern(TyKind::Object { fields: Box::new([(b, int), (a, string)]), label: () });
        let others = [
            types.intern(TyKind::Array(int, LenTerm::Known(2), ())),
            types.intern(TyKind::Option(int, ())),
            types.intern(TyKind::Result(int, string, ())),
        ];
        let fields = |ty| {
            let structure = structure(&types, ty).expect("a structural type");
            structure.fields().map(|fields| fields.iter().map(|field| (field.key, field.ty)).collect::<Vec<_>>())
        };
        assert_eq!(fields(tuple), Some(vec![(PartKey::Index(0), string), (PartKey::Index(1), int)]));
        assert_eq!(fields(object), Some(vec![(PartKey::Field(b), int), (PartKey::Field(a), string)]));
        for product in [tuple, object] {
            let structure = structure(&types, product).expect("a product");
            assert_eq!(structure.fields().expect("a product").to_vec(), structure.parts().to_vec());
        }
        for other in others {
            assert_eq!(fields(other), None, "{:?}", types.kind(other));
        }
    }

    /// operators §3: an array's and a slice's element is a sequence's; a
    /// tuple, even of one type, and any other type are no sequence.
    #[test]
    fn a_sequences_element_is_an_arrays_or_a_slices() {
        let mut types = Types::<Concrete>::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let array = types.intern(TyKind::Array(string, LenTerm::Known(2), ()));
        let slice = types.intern(TyKind::Slice(int));
        assert_eq!(element(&types, array), Some(string));
        assert_eq!(element(&types, slice), Some(int));
        let tuple = types.intern(TyKind::Tuple(Box::new([int, int]), ()));
        let reference = types.intern(TyKind::Ref(Mutability::Shared, slice));
        for other in [tuple, reference, string, int] {
            assert_eq!(element(&types, other), None, "{:?}", types.kind(other));
        }
    }

    /// types §1: an extern-defined type is opaque, so it has no parts; nor
    /// has a reference, `!` or a primitive.
    #[test]
    fn an_extern_defined_type_a_reference_and_a_primitive_are_not_structural() {
        let names = Interner::new();
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let declared = types.intern(TyKind::UserDefined {
            id: QualifiedRef::root(names.intern("S")),
            type_args: Box::new([]),
            effect: None,
            identity: None,
            marked: MarkTerm::Known(Mark::Unmarked),
        });
        let reference = types.intern(TyKind::Ref(Mutability::Shared, int));
        let never = types.intern(TyKind::Never);
        let string = types.intern(TyKind::String);
        for ty in [declared, reference, never, string, int] {
            assert_eq!(structure(&types, ty), None);
        }
    }
}
