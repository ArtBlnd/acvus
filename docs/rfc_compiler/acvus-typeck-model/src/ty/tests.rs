use std::collections::{BTreeMap, BTreeSet};

use super::*;

fn ty(term: Term<Ty>) -> Ty {
    Ty::new(term).expect("the term is a closed type")
}

fn name(text: &str) -> Name {
    Name::new(text)
}

fn int(int: IntTy) -> Ty {
    ty(Term::Scalar(Scalar::Int(int)))
}

fn identity(text: &str) -> Ty {
    ty(Term::Identity(name(text)))
}

fn extern_ty(text: &str, args: Vec<Ty>, identity: Option<Ty>) -> Ty {
    ty(Term::Extern {
        name: name(text),
        args,
        identity,
    })
}

/// One type, with what the shape table gives it, and its notation.
struct Shape {
    ty: Ty,
    constructor: Constructor,
    members: Vec<Member>,
    parts: Vec<PartKey>,
    structural: bool,
    notation: &'static str,
}

fn shapes() -> Vec<Shape> {
    let u8_ = int(IntTy::U8);
    let u16_ = int(IntTy::U16);
    let bool_ = ty(Term::Scalar(Scalar::Bool));
    let field = |text| Member::Field(name(text));
    let variant = |text| Member::Variant(name(text));
    let payload = |text| PartKey::Payload(name(text));
    vec![
        Shape {
            ty: ty(Term::Ref(Mutability::Exclusive, u8_.clone())),
            constructor: Constructor::Ref(Mutability::Exclusive),
            members: vec![],
            parts: vec![PartKey::Target],
            structural: false,
            notation: "&mut u8",
        },
        Shape {
            ty: ty(Term::Tuple(vec![u8_.clone(), u16_.clone()])),
            constructor: Constructor::Tuple(2),
            members: vec![],
            parts: vec![PartKey::Component(0), PartKey::Component(1)],
            structural: true,
            notation: "(u8, u16)",
        },
        Shape {
            ty: ty(Term::Tuple(vec![u8_.clone()])),
            constructor: Constructor::Tuple(1),
            members: vec![],
            parts: vec![PartKey::Component(0)],
            structural: true,
            notation: "(u8,)",
        },
        Shape {
            ty: ty(Term::Array(4, u8_.clone())),
            constructor: Constructor::Array(4),
            members: vec![],
            parts: vec![PartKey::Element],
            structural: true,
            notation: "[u8; 4]",
        },
        Shape {
            ty: ty(Term::Slice(u8_.clone())),
            constructor: Constructor::Slice,
            members: vec![],
            parts: vec![PartKey::Element],
            structural: false,
            notation: "[u8]",
        },
        Shape {
            ty: ty(Term::Option(u8_.clone())),
            constructor: Constructor::Option,
            members: vec![variant("None"), variant("Some")],
            parts: vec![payload("Some")],
            structural: true,
            notation: "Option<u8>",
        },
        Shape {
            ty: ty(Term::Result {
                ok: u8_.clone(),
                err: u16_.clone(),
            }),
            constructor: Constructor::Result,
            members: vec![variant("Err"), variant("Ok")],
            parts: vec![payload("Ok"), payload("Err")],
            structural: true,
            notation: "Result<u8, u16>",
        },
        Shape {
            ty: ty(Term::Function {
                params: vec![u8_.clone(), u16_.clone()],
                ret: bool_.clone(),
            }),
            constructor: Constructor::Function(2),
            members: vec![],
            parts: vec![PartKey::Param(0), PartKey::Param(1), PartKey::Return],
            structural: false,
            notation: "Fn(u8, u16 -> bool)",
        },
        Shape {
            ty: ty(Term::FunctionValue { arity: 2 }),
            constructor: Constructor::Function(2),
            members: vec![],
            parts: vec![],
            structural: false,
            notation: "Fn/2",
        },
        Shape {
            ty: ty(Term::Object(
                [(name("y"), u16_.clone()), (name("x"), u8_.clone())].into(),
            )),
            constructor: Constructor::Object,
            members: vec![field("x"), field("y")],
            parts: vec![PartKey::Field(name("x")), PartKey::Field(name("y"))],
            structural: true,
            notation: "{x: u8, y: u16}",
        },
        Shape {
            ty: ty(Term::Enum(
                name("E"),
                [(name("B"), u8_.clone()), (name("A"), ty(Term::Never))].into(),
            )),
            constructor: Constructor::Enum(name("E")),
            members: vec![variant("A"), variant("B")],
            parts: vec![payload("A"), payload("B")],
            structural: true,
            notation: "E{A(!), B(u8)}",
        },
        Shape {
            ty: ty(Term::ErrorOf(u8_.clone())),
            constructor: Constructor::ErrorOf,
            members: vec![],
            parts: vec![PartKey::Wrapped],
            structural: true,
            notation: "Error<u8>",
        },
        Shape {
            ty: ty(Term::Struct(
                name("Pair"),
                vec![
                    u8_.clone(),
                    ty(Term::Ref(Mutability::Shared, ty(Term::Str))),
                ],
            )),
            constructor: Constructor::Struct(name("Pair")),
            members: vec![],
            parts: vec![PartKey::Arg(0), PartKey::Arg(1)],
            structural: false,
            notation: "Pair<u8, &str>",
        },
        Shape {
            ty: extern_ty("F", vec![u8_.clone()], Some(identity("ι"))),
            constructor: Constructor::Extern(name("F")),
            members: vec![],
            parts: vec![PartKey::Arg(0), PartKey::Identity],
            structural: false,
            notation: "F<u8, ι>",
        },
        Shape {
            ty: bool_,
            constructor: Constructor::Scalar(Scalar::Bool),
            members: vec![],
            parts: vec![],
            structural: false,
            notation: "bool",
        },
        Shape {
            ty: ty(Term::String),
            constructor: Constructor::String,
            members: vec![],
            parts: vec![],
            structural: false,
            notation: "String",
        },
        Shape {
            ty: ty(Term::Str),
            constructor: Constructor::Str,
            members: vec![],
            parts: vec![],
            structural: false,
            notation: "str",
        },
        Shape {
            ty: ty(Term::Unit),
            constructor: Constructor::Unit,
            members: vec![],
            parts: vec![],
            structural: false,
            notation: "()",
        },
        Shape {
            ty: ty(Term::Never),
            constructor: Constructor::Never,
            members: vec![],
            parts: vec![],
            structural: false,
            notation: "!",
        },
        Shape {
            ty: identity("ι"),
            constructor: Constructor::Identity(name("ι")),
            members: vec![],
            parts: vec![],
            structural: false,
            notation: "ι",
        },
    ]
}

/// The position of `term`'s variant in `Term`. A variant added to `Term`
/// stops this from compiling until `shapes` is given one of it.
fn variant_index(term: &Term<Ty>) -> usize {
    match term {
        Term::Ref(..) => 0,
        Term::Tuple(_) => 1,
        Term::Array(..) => 2,
        Term::Slice(_) => 3,
        Term::Option(_) => 4,
        Term::Result { .. } => 5,
        Term::Function { .. } => 6,
        Term::FunctionValue { .. } => 7,
        Term::Object(_) => 8,
        Term::Enum(..) => 9,
        Term::Struct(..) => 10,
        Term::Extern { .. } => 11,
        Term::Scalar(_) => 12,
        Term::String => 13,
        Term::Str => 14,
        Term::Unit => 15,
        Term::Never => 16,
        Term::Identity(_) => 17,
        Term::ErrorOf(_) => 18,
    }
}

#[test]
fn shapes_cover_every_variant_of_term() {
    let covered: BTreeSet<usize> = shapes()
        .iter()
        .map(|shape| variant_index(shape.ty.term()))
        .collect();
    assert_eq!(covered, (0..19).collect::<BTreeSet<usize>>());
}

#[test]
fn each_constructor_has_its_members_and_parts() {
    for shape in shapes() {
        let term = shape.ty.term();
        assert_eq!(term.constructor(), shape.constructor, "{}", shape.ty);
        assert_eq!(
            term.members().into_iter().collect::<Vec<_>>(),
            shape.members,
            "{}",
            shape.ty
        );
        let parts: Vec<PartKey> = term.parts().into_iter().map(|(key, _)| key).collect();
        assert_eq!(parts, shape.parts, "{}", shape.ty);
    }
}

#[test]
fn map_names_each_part_as_parts_does_and_keeps_the_rest() {
    for shape in shapes() {
        let mut named = Vec::new();
        let mapped = shape.ty.term().clone().map(|key, part| {
            named.push(key);
            part
        });
        assert_eq!(named, shape.parts, "{}", shape.ty);
        assert_eq!(&mapped, shape.ty.term(), "{}", shape.ty);
    }
}

#[test]
fn part_finds_each_part_by_its_name_and_no_other() {
    for shape in shapes() {
        let term = shape.ty.term();
        for (key, part) in term.parts() {
            assert_eq!(term.part(&key), Some(part), "{}", shape.ty);
        }
        assert_eq!(term.part(&PartKey::Arg(7)), None, "{}", shape.ty);
    }
}

#[test]
fn types_built_apart_are_equal_by_constructor_members_and_parts() {
    let u8_ = int(IntTy::U8);
    assert_eq!(
        ty(Term::Option(u8_.clone())),
        ty(Term::Option(int(IntTy::U8)))
    );
    assert_ne!(
        ty(Term::Option(u8_.clone())),
        ty(Term::Option(int(IntTy::U16)))
    );
    assert_ne!(
        ty(Term::Array(4, u8_.clone())),
        ty(Term::Array(5, u8_.clone()))
    );
    assert_ne!(
        ty(Term::Ref(Mutability::Shared, u8_.clone())),
        ty(Term::Ref(Mutability::Exclusive, u8_.clone()))
    );
    assert_ne!(
        ty(Term::Object([(name("x"), u8_.clone())].into())),
        ty(Term::Object([(name("y"), u8_.clone())].into()))
    );
}

#[test]
fn an_object_has_one_representation_whatever_order_its_fields_come_in() {
    let fields = [(name("x"), int(IntTy::U8)), (name("y"), int(IntTy::U16))];
    let forward: BTreeMap<Name, Ty> = fields.clone().into_iter().collect();
    let backward: BTreeMap<Name, Ty> = fields.into_iter().rev().collect();
    assert_eq!(ty(Term::Object(forward)), ty(Term::Object(backward)));
}

#[test]
fn a_function_value_is_not_the_written_function_type_of_its_arity() {
    let value = ty(Term::FunctionValue { arity: 1 });
    let written = ty(Term::Function {
        params: vec![int(IntTy::U8)],
        ret: int(IntTy::U8),
    });
    assert_eq!(value.term().constructor(), written.term().constructor());
    assert_ne!(value, written);
    assert_eq!(value, ty(Term::FunctionValue { arity: 1 }));
}

#[test]
fn a_tuple_of_no_component_is_refused() {
    assert_eq!(Ty::new(Term::Tuple(Vec::new())), Err(TyError::EmptyTuple));
}

#[test]
fn a_reference_to_a_reference_is_refused_but_not_a_reference_inside_a_target() {
    let shared = ty(Term::Ref(Mutability::Shared, int(IntTy::U8)));
    assert_eq!(
        Ty::new(Term::Ref(Mutability::Exclusive, shared.clone())),
        Err(TyError::RefTarget)
    );
    let holder = extern_ty("N", vec![shared], None);
    assert!(Ty::new(Term::Ref(Mutability::Shared, holder)).is_ok());
}

#[test]
fn an_identity_lies_only_at_an_extern_defined_types_identity() {
    assert!(
        Ty::new(Term::Extern {
            name: name("F"),
            args: vec![],
            identity: Some(identity("ι")),
        })
        .is_ok()
    );
    assert_eq!(
        Ty::new(Term::Extern {
            name: name("F"),
            args: vec![identity("ι")],
            identity: None,
        }),
        Err(TyError::IdentityPlace)
    );
    assert_eq!(
        Ty::new(Term::Extern {
            name: name("F"),
            args: vec![],
            identity: Some(int(IntTy::U8)),
        }),
        Err(TyError::IdentityPlace)
    );
    assert_eq!(
        Ty::new(Term::Option(identity("ι"))),
        Err(TyError::IdentityPlace)
    );
}

#[test]
fn identities_are_equal_exactly_when_their_names_are() {
    assert_eq!(identity("ι₀"), identity("ι₀"));
    assert_ne!(identity("ι₀"), identity("ι₁"));
}

#[test]
fn object_enum_tuple_array_option_and_result_alone_are_structural() {
    for shape in shapes() {
        assert_eq!(
            shape.constructor.is_structural(),
            shape.structural,
            "{}",
            shape.ty
        );
    }
}

#[test]
fn each_variant_has_the_part_its_payload_lies_in() {
    let payload = |text| Some(PartKey::Payload(name(text)));
    assert_eq!(
        Constructor::Option.payload_part(&name("Some")),
        payload("Some")
    );
    assert_eq!(
        Constructor::Option.payload_part(&name("None")),
        payload("Some")
    );
    assert_eq!(Constructor::Option.payload_part(&name("Ok")), None);
    assert_eq!(Constructor::Result.payload_part(&name("Ok")), payload("Ok"));
    assert_eq!(
        Constructor::Result.payload_part(&name("Err")),
        payload("Err")
    );
    assert_eq!(Constructor::Result.payload_part(&name("Some")), None);
    assert_eq!(
        Constructor::Enum(name("E")).payload_part(&name("A")),
        payload("A")
    );
    assert_eq!(
        Constructor::Struct(name("Pair")).payload_part(&name("A")),
        None
    );
}

#[test]
fn the_payload_part_of_a_variant_is_a_part_of_its_term() {
    for shape in shapes() {
        let term = shape.ty.term();
        for member in term.members() {
            if let Member::Variant(variant) = member {
                let key = term
                    .constructor()
                    .payload_part(&variant)
                    .expect("a variant has a payload part");
                assert!(term.part(&key).is_some(), "{} {variant}", shape.ty);
            }
        }
    }
}

#[test]
fn display_writes_the_specs_notation_in_ascii() {
    for shape in shapes() {
        assert_eq!(shape.ty.to_string(), shape.notation);
    }
}
