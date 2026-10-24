//! Declarations and types the tests of several primitives share.

use crate::decl::{Decls, DeclsBuilder, Role};
use crate::join::ERROR;
use crate::sig::{DeclaredTypes, ExternDecl, SigTy, Signature};
use crate::ty::{IntTy, Mutability, Name, Scalar, Term, Ty};

pub fn name(text: &str) -> Name {
    Name::new(text)
}

pub fn ty(term: Term<Ty>) -> Ty {
    Ty::new(term).expect("the term is a closed type")
}

pub fn int(int: IntTy) -> Ty {
    ty(Term::Scalar(Scalar::Int(int)))
}

pub fn named(text: &str) -> Ty {
    ty(Term::Extern {
        name: name(text),
        args: vec![],
        identity: None,
    })
}

pub fn boxed(arg: Ty) -> Ty {
    ty(Term::Extern {
        name: name("Box"),
        args: vec![arg],
        identity: None,
    })
}

pub fn reference(mutability: Mutability, target: Ty) -> Ty {
    ty(Term::Ref(mutability, target))
}

pub fn object(fields: &[(&str, Ty)]) -> Ty {
    ty(Term::Object(
        fields
            .iter()
            .map(|(field, ty)| (name(field), ty.clone()))
            .collect(),
    ))
}

pub fn enumeration(text: &str, variants: &[(&str, Ty)]) -> Ty {
    ty(Term::Enum(
        name(text),
        variants
            .iter()
            .map(|(variant, payload)| (name(variant), payload.clone()))
            .collect(),
    ))
}

pub fn result(ok: Ty, err: Ty) -> Ty {
    ty(Term::Result { ok, err })
}

/// The `Error` whose variants are the elements, each named after its type.
pub fn error(elements: &[Ty]) -> Ty {
    let variants = elements
        .iter()
        .map(|element| (type_name(element.term()).clone(), element.clone()))
        .collect();
    ty(Term::Enum(name(ERROR), variants))
}

/// The name of an enum, a struct or an extern-defined type.
fn type_name(term: &Term<Ty>) -> &Name {
    match term {
        Term::Enum(name, _) | Term::Struct(name, _) | Term::Extern { name, .. } => name,
        _ => panic!("an element of a fixture's `Error` is a type of a name"),
    }
}

/// `A`, `B` and `C` of no argument and `Box` of one; the casts `A → B` and
/// `C → B`, and `&A → &B` and `&mut A → &mut B`.
pub fn decls() -> Decls {
    let mut types = DeclaredTypes::default();
    for text in ["A", "B", "C"] {
        types
            .declare_extern(
                name(text),
                ExternDecl {
                    arity: 0,
                    identity: false,
                },
            )
            .unwrap();
    }
    types
        .declare_extern(
            name("Box"),
            ExternDecl {
                arity: 1,
                identity: false,
            },
        )
        .unwrap();
    let mut builder = DeclsBuilder::new(types);
    let plain = |text: &str| {
        SigTy::term(Term::Extern {
            name: name(text),
            args: vec![],
            identity: None,
        })
    };
    let to_ref = |mutability, target| SigTy::term(Term::Ref(mutability, target));
    let casts = [
        (plain("A"), plain("B")),
        (plain("C"), plain("B")),
        (
            to_ref(Mutability::Shared, plain("A")),
            to_ref(Mutability::Shared, plain("B")),
        ),
        (
            to_ref(Mutability::Exclusive, plain("A")),
            to_ref(Mutability::Exclusive, plain("B")),
        ),
    ];
    for (i, (from, to)) in casts.into_iter().enumerate() {
        let sig = Signature::new(vec![], false, vec![from], to, builder.types()).unwrap();
        builder.declare(name(&format!("cast{i}")), sig, Role::Cast);
    }
    builder.finish().unwrap()
}

pub fn never() -> Ty {
    ty(Term::Never)
}

/// Types up to one constructor deep over leaves that the cast tree and the
/// reference row each relate, and a `Result` whose error part is an `Error`.
pub fn universe() -> Vec<Ty> {
    let leaves = [never(), int(IntTy::U8), named("A"), named("B"), named("C")];
    let mut types: Vec<Ty> = leaves.to_vec();
    for mutability in [Mutability::Shared, Mutability::Exclusive] {
        for target in ["A", "B", "C"] {
            types.push(reference(mutability, named(target)));
        }
    }
    for leaf in &leaves {
        types.push(ty(Term::Option(leaf.clone())));
        types.push(object(&[("x", leaf.clone())]));
        types.push(object(&[("x", leaf.clone()), ("y", int(IntTy::U8))]));
        types.push(enumeration("E", &[("V", leaf.clone())]));
        types.push(enumeration("E", &[("W", leaf.clone())]));
        types.push(result(int(IntTy::U8), leaf.clone()));
        types.push(boxed(leaf.clone()));
    }
    types.push(result(int(IntTy::U8), error(&[named("A"), named("C")])));
    types
}
