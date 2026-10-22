use super::*;
use crate::sig::{ExternDecl, SigVar};
use crate::ty::{IntTy, Scalar};

fn name(text: &str) -> Name {
    Name::new(text)
}

fn ty(term: Term<Ty>) -> Ty {
    Ty::new(term).expect("the term is a closed type")
}

fn int(int: IntTy) -> Ty {
    ty(Term::Scalar(Scalar::Int(int)))
}

fn closed_extern(text: &str, args: Vec<Ty>) -> Ty {
    ty(Term::Extern {
        name: name(text),
        args,
        identity: None,
    })
}

fn sig_extern(text: &str, args: Vec<SigTy>) -> SigTy {
    SigTy::term(Term::Extern {
        name: name(text),
        args,
        identity: None,
    })
}

fn sig_ref(mutability: Mutability, target: SigTy) -> SigTy {
    SigTy::term(Term::Ref(mutability, target))
}

fn var(i: usize) -> SigTy {
    SigTy::Var(SigVar::Type(i))
}

/// Extern-defined types of no argument `A` to `E`, and `Vec`, `Box`, `Seq` and
/// `Bytes` of one argument but `Bytes`.
fn new_builder() -> DeclsBuilder {
    let mut types = DeclaredTypes::default();
    for text in ["A", "B", "C", "D", "E", "Bytes"] {
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
    for text in ["Vec", "Box", "Seq"] {
        types
            .declare_extern(
                name(text),
                ExternDecl {
                    arity: 1,
                    identity: false,
                },
            )
            .unwrap();
    }
    DeclsBuilder::new(types)
}

fn cast(
    builder: &mut DeclsBuilder,
    text: &str,
    params: Vec<Name>,
    from: SigTy,
    to: SigTy,
) -> DeclId {
    let sig = Signature::new(params, false, vec![from], to, builder.types()).unwrap();
    builder.declare(name(text), sig, Role::Cast)
}

fn plain(text: &str) -> SigTy {
    sig_extern(text, vec![])
}

#[test]
fn casts_that_step_up_once_from_each_node_form_a_tree() {
    let mut builder = new_builder();
    cast(&mut builder, "ab", vec![], plain("A"), plain("B"));
    cast(&mut builder, "cb", vec![], plain("C"), plain("B"));
    cast(&mut builder, "bd", vec![], plain("B"), plain("D"));
    let decls = builder.finish().unwrap();
    let node = |text: &str| Node::Owned(Constructor::Extern(name(text)));
    let tree = decls.tree();
    assert_eq!(
        tree.ancestors(&node("A")),
        vec![node("A"), node("B"), node("D")]
    );
    assert_eq!(tree.nca(&node("A"), &node("C")), Some(node("B")));
    assert_eq!(tree.nca(&node("A"), &node("D")), Some(node("D")));
    assert_eq!(tree.nca(&node("A"), &node("E")), None);
    let path = tree.path(&node("A"), &node("D")).unwrap();
    assert_eq!(
        path.iter().map(|id| id.name.as_str()).collect::<Vec<_>>(),
        ["ab", "bd"]
    );
    assert_eq!(tree.path(&node("A"), &node("A")), Some(vec![]));
    assert_eq!(tree.path(&node("D"), &node("A")), None);
}

#[test]
fn a_second_cast_up_from_one_node_is_refused() {
    let mut builder = new_builder();
    let first = cast(&mut builder, "ab", vec![], plain("A"), plain("B"));
    let second = cast(&mut builder, "ac", vec![], plain("A"), plain("C"));
    assert_eq!(
        builder.finish().err(),
        Some(TreeError::TwoUp { first, second })
    );
}

#[test]
fn casts_forming_a_cycle_are_refused() {
    let mut builder = new_builder();
    cast(&mut builder, "ab", vec![], plain("A"), plain("B"));
    cast(&mut builder, "ba", vec![], plain("B"), plain("A"));
    assert!(matches!(builder.finish(), Err(TreeError::Cycle(_))));
}

#[test]
fn a_cast_goes_from_an_extern_defined_type_and_keeps_its_mode() {
    let mut builder = new_builder();
    let from_int = cast(
        &mut builder,
        "int",
        vec![],
        SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8))),
        plain("A"),
    );
    assert_eq!(
        builder.finish().err(),
        Some(TreeError::NotFromExtern(from_int))
    );
    let mut builder = new_builder();
    let to_ref = cast(
        &mut builder,
        "to_ref",
        vec![],
        plain("A"),
        sig_ref(Mutability::Shared, plain("B")),
    );
    assert_eq!(builder.finish().err(), Some(TreeError::ChangesMode(to_ref)));
    let mut builder = new_builder();
    let weaker = cast(
        &mut builder,
        "weaker",
        vec![],
        sig_ref(Mutability::Exclusive, plain("A")),
        sig_ref(Mutability::Shared, plain("B")),
    );
    assert_eq!(builder.finish().err(), Some(TreeError::ChangesMode(weaker)));
    let mut builder = new_builder();
    let to_var = cast(&mut builder, "to_var", vec![name("T")], plain("A"), var(0));
    assert_eq!(
        builder.finish().err(),
        Some(TreeError::NoTargetNode(to_var))
    );
}

#[test]
fn a_cast_of_exclusive_references_has_its_shared_counterpart() {
    let mut builder = new_builder();
    cast(
        &mut builder,
        "mut_ab",
        vec![],
        sig_ref(Mutability::Exclusive, plain("A")),
        sig_ref(Mutability::Exclusive, plain("B")),
    );
    cast(
        &mut builder,
        "ab",
        vec![],
        sig_ref(Mutability::Shared, plain("A")),
        sig_ref(Mutability::Shared, plain("B")),
    );
    assert!(builder.finish().is_ok());

    let mut builder = new_builder();
    let alone = cast(
        &mut builder,
        "mut_ab",
        vec![],
        sig_ref(Mutability::Exclusive, plain("A")),
        sig_ref(Mutability::Exclusive, plain("B")),
    );
    assert_eq!(
        builder.finish().err(),
        Some(TreeError::NotCommuting(alone.clone()))
    );

    let mut builder = new_builder();
    let elsewhere = cast(
        &mut builder,
        "mut_ab",
        vec![],
        sig_ref(Mutability::Exclusive, plain("A")),
        sig_ref(Mutability::Exclusive, plain("B")),
    );
    cast(
        &mut builder,
        "ac",
        vec![],
        sig_ref(Mutability::Shared, plain("A")),
        sig_ref(Mutability::Shared, plain("C")),
    );
    assert_eq!(
        builder.finish().err(),
        Some(TreeError::NotCommuting(elsewhere))
    );
}

#[test]
fn a_view_steps_a_reference_to_a_container_up_to_a_slice_of_its_elements() {
    let mut builder = new_builder();
    cast(
        &mut builder,
        "core::as_slice",
        vec![name("T")],
        sig_ref(Mutability::Shared, sig_extern("Vec", vec![var(0)])),
        sig_ref(Mutability::Shared, SigTy::term(Term::Slice(var(0)))),
    );
    let decls = builder.finish().unwrap();
    let vec_u8 = ty(Term::Ref(
        Mutability::Shared,
        closed_extern("Vec", vec![int(IntTy::U8)]),
    ));
    let (cast, up) = decls.step_up(&vec_u8).unwrap();
    assert_eq!(cast.name, name("core::as_slice"));
    assert_eq!(
        up,
        ty(Term::Ref(
            Mutability::Shared,
            ty(Term::Slice(int(IntTy::U8)))
        ))
    );
    assert_eq!(
        Node::of(&up),
        Node::Ref(Mutability::Shared, Constructor::Slice)
    );
}

#[test]
fn raising_takes_each_cast_up_to_the_node() {
    let mut builder = new_builder();
    cast(
        &mut builder,
        "box_seq",
        vec![name("T")],
        sig_extern("Box", vec![var(0)]),
        sig_extern("Seq", vec![var(0)]),
    );
    let decls = builder.finish().unwrap();
    let box_u16 = closed_extern("Box", vec![int(IntTy::U16)]);
    let seq = Node::Owned(Constructor::Extern(name("Seq")));
    let (casts, raised) = decls.raise(&box_u16, &seq).unwrap();
    assert_eq!(casts.len(), 1);
    assert_eq!(raised, closed_extern("Seq", vec![int(IntTy::U16)]));
}

#[test]
fn a_cast_writes_its_node_generally_and_its_result_by_its_parameter() {
    let mut narrow = new_builder();
    let bytes = cast(
        &mut narrow,
        "bytes",
        vec![],
        sig_extern(
            "Vec",
            vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8)))],
        ),
        plain("Bytes"),
    );
    assert!(
        matches!(narrow.finish(), Err(TreeError::NarrowerThanNode(id)) if id == bytes),
        "Vec<u8> is not Vec's every type (l.78; holes.md, H1)"
    );
    let mut unbound = new_builder();
    let boxed = cast(
        &mut unbound,
        "boxed",
        vec![name("T")],
        plain("A"),
        sig_extern("Box", vec![var(0)]),
    );
    assert!(
        matches!(unbound.finish(), Err(TreeError::ResultUnbound(id)) if id == boxed),
        "where A goes up is not decided"
    );
}

#[test]
fn conversions_between_two_types_are_found_by_their_signatures() {
    let mut builder = new_builder();
    let u16_to_u8 = Signature::new(
        vec![],
        false,
        vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U16)))],
        SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8))),
        builder.types(),
    )
    .unwrap();
    let first = builder.declare(name("narrow"), u16_to_u8.clone(), Role::Conversion);
    let second = builder.declare(name("truncate"), u16_to_u8, Role::Conversion);
    let decls = builder.finish().unwrap();
    assert_eq!(
        decls.conversions_between(&int(IntTy::U16), &int(IntTy::U8)),
        vec![first, second]
    );
    assert!(
        decls
            .conversions_between(&int(IntTy::U8), &int(IntTy::U16))
            .is_empty()
    );
}

#[test]
fn the_slots_type_binds_what_a_conversions_parameter_leaves_unbound() {
    let mut builder = new_builder();
    let a_to_box = Signature::new(
        vec![name("T")],
        false,
        vec![plain("A")],
        sig_extern("Box", vec![var(0)]),
        builder.types(),
    )
    .unwrap();
    let boxed = builder.declare(name("boxed"), a_to_box, Role::Conversion);
    let decls = builder.finish().unwrap();
    let a = closed_extern("A", vec![]);
    assert_eq!(
        decls.conversions_between(&a, &closed_extern("Box", vec![int(IntTy::U8)])),
        vec![boxed]
    );
    assert!(
        decls
            .conversions_between(&a, &closed_extern("Seq", vec![int(IntTy::U8)]))
            .is_empty()
    );
}

#[test]
fn a_name_reaches_each_of_its_declarations_in_the_order_declared() {
    let mut builder = new_builder();
    let unit = SigTy::term(Term::Unit);
    let take_u8 = Signature::new(
        vec![],
        false,
        vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8)))],
        unit.clone(),
        builder.types(),
    )
    .unwrap();
    let take_u16 = Signature::new(
        vec![],
        false,
        vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U16)))],
        unit,
        builder.types(),
    )
    .unwrap();
    builder.declare(name("take"), take_u8.clone(), Role::Fn);
    builder.declare(name("take"), take_u16.clone(), Role::Fn);
    let decls = builder.finish().unwrap();
    let candidates = decls.candidates(&name("take"));
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].sig, take_u8);
    assert_eq!(candidates[1].id.overload, 1);
    assert!(decls.candidates(&name("other")).is_empty());
}
