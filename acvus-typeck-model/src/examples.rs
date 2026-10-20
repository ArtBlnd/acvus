//! The spec's examples (l.360–596), each encoded as the constraints its
//! text gives and solved in every order.

use crate::decl::{DeclId, Decls, DeclsBuilder, Role};
use crate::explore::{Exploration, explore};
use crate::node::Decision;
use crate::program::{Program, ProgramBuilder};
use crate::sig::{DeclaredTypes, ExternDecl, SigTy, SigVar, Signature};
use crate::solve::Outcome;
use crate::ty::{IntTy, Mutability, Name, Scalar, Term, Ty};

const BUDGET: usize = 100_000;

fn no_decls() -> Decls {
    DeclsBuilder::new(DeclaredTypes::default())
        .finish()
        .unwrap()
}

/// Declarations over the extern-defined types `externs`, each with its
/// number of arguments and no identity.
fn declare(externs: &[(&str, usize)]) -> DeclsBuilder {
    let mut types = DeclaredTypes::default();
    for (name, arity) in externs {
        types
            .declare_extern(
                Name::new(name),
                ExternDecl {
                    arity: *arity,
                    identity: false,
                },
            )
            .unwrap();
    }
    DeclsBuilder::new(types)
}

fn function(
    builder: &mut DeclsBuilder,
    name: &str,
    type_params: &[&str],
    params: Vec<SigTy>,
    result: SigTy,
    role: Role,
) -> DeclId {
    let sig = Signature::new(
        type_params.iter().map(|param| Name::new(param)).collect(),
        false,
        params,
        result,
        builder.types(),
    )
    .unwrap();
    builder.declare(Name::new(name), sig, role)
}

fn t(i: usize) -> SigTy {
    SigTy::Var(SigVar::Type(i))
}

fn s_extern(name: &str, args: Vec<SigTy>) -> SigTy {
    SigTy::term(Term::Extern {
        name: Name::new(name),
        args,
        identity: None,
    })
}

fn s_ref(mutability: Mutability, target: SigTy) -> SigTy {
    SigTy::term(Term::Ref(mutability, target))
}

fn s_int(int: IntTy) -> SigTy {
    SigTy::term(Term::Scalar(Scalar::Int(int)))
}

fn s_unit() -> SigTy {
    SigTy::term(Term::Unit)
}

fn decision(outcome: &Outcome, node: &str) -> Decision {
    outcome.decisions[&Name::new(node)]
        .clone()
        .unwrap_or_else(|| panic!("{node} is decided"))
}

fn int(int: IntTy) -> Term<Name> {
    Term::Scalar(Scalar::Int(int))
}

fn closed(term: Term<Ty>) -> Ty {
    Ty::new(term).unwrap()
}

fn closed_int(int: IntTy) -> Ty {
    closed(Term::Scalar(Scalar::Int(int)))
}

/// The one outcome every order of solving reaches.
fn only_outcome(program: &Program) -> Outcome {
    let Exploration {
        outcomes,
        states,
        complete,
    } = explore(program, BUDGET);
    assert!(complete, "explored {states} states within the budget");
    assert_eq!(
        outcomes.len(),
        1,
        "every order reaches one outcome: {outcomes:#?}"
    );
    outcomes.into_iter().next().unwrap()
}

fn value(outcome: &Outcome, name: &str) -> Option<Ty> {
    outcome
        .values
        .get(&crate::class::ClassName::Text(Name::new(name)))
        .unwrap_or_else(|| panic!("the program has the class {name}"))
        .clone()
}

/// `let a = if c { Some(1u8) } else { None };`
#[test]
fn example_1_an_if_with_information_on_one_side() {
    let mut p = ProgramBuilder::new(no_decls());
    let a = p.var("A");
    let s = p.var("S");
    let u8_ = p.written("u8", int(IntTy::U8));
    let some = p.written("Some(1u8)", Term::Option(s.clone()));
    let k = p.var("?k");
    let none = p.written("None", Term::Option(k));
    p.pass(&u8_, &s);
    p.pass(&some, &a);
    p.pass(&none, &a);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "A"),
        Some(closed(Term::Option(closed_int(IntTy::U8))))
    );
    assert_eq!(value(&outcome, "?k"), Some(closed(Term::Never)));
}

/// ```text
/// let x = None;
/// let y = if c { x } else { Some(1u16) };
/// take(x);                 // take(Option⟨u8⟩)
/// ```
/// `take(x)` is encoded as the boundary it records.
#[test]
fn example_4_a_covariant_boundary_is_a_check() {
    let mut p = ProgramBuilder::new(no_decls());
    let x = p.var("X");
    let k = p.var("?k");
    let none = p.written("None", Term::Option(k));
    let y = p.var("Y");
    let s = p.var("S");
    let u16_ = p.written("u16", int(IntTy::U16));
    let some = p.written("Some(1u16)", Term::Option(s.clone()));
    let u8_ = p.written("u8", int(IntTy::U8));
    let param = p.written("Option⟨u8⟩", Term::Option(u8_));
    p.pass(&none, &x);
    p.pass(&x, &y);
    p.pass(&u16_, &s);
    p.pass(&some, &y);
    p.pass(&x, &param);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "X"),
        Some(closed(Term::Option(closed(Term::Never))))
    );
    assert_eq!(
        value(&outcome, "Y"),
        Some(closed(Term::Option(closed_int(IntTy::U16))))
    );
}

/// ```text
/// let mut x = None;
/// x = Some(x);
/// ```
#[test]
fn example_5_a_lower_bound_reaching_a_part_of_its_own() {
    let mut p = ProgramBuilder::new(no_decls());
    let x = p.var("X");
    let k = p.var("?k");
    let none = p.written("None", Term::Option(k));
    let s = p.var("S");
    let some = p.written("Some(x)", Term::Option(s.clone()));
    p.pass(&none, &x);
    p.pass(&x, &s);
    p.pass(&some, &x);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    assert!(
        outcome
            .refusals
            .iter()
            .any(|refusal| matches!(refusal, crate::solve::Refusal::Infinite(_))),
        "{outcome:#?}"
    );
}

fn closed_extern(name: &str, args: Vec<Ty>) -> Ty {
    closed(Term::Extern {
        name: Name::new(name),
        args,
        identity: None,
    })
}

/// ```text
/// let a = Vec::new();      // new() -> Vec⟨T⟩
/// fill_all(&mut a);        // fill_all(&mut Vec⟨u8⟩)
/// ```
#[test]
fn example_2_a_boundarys_invariant_part() {
    let mut d = declare(&[("Vec", 1)]);
    function(
        &mut d,
        "new",
        &["T"],
        vec![],
        s_extern("Vec", vec![t(0)]),
        Role::Fn,
    );
    let vec_u8 = s_extern("Vec", vec![s_int(IntTy::U8)]);
    function(
        &mut d,
        "fill_all",
        &[],
        vec![s_ref(Mutability::Exclusive, vec_u8)],
        s_unit(),
        Role::Fn,
    );
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let a = p.var("A");
    let new = p.var("new()");
    let lent = p.var("&mut a");
    let filled = p.var("fill_all(..)");
    p.call("new", "new", &[], &new);
    p.pass(&new, &a);
    p.lend("lend", Mutability::Exclusive, &a, &lent);
    p.call("fill_all", "fill_all", &[&lent], &filled);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "A"),
        Some(closed_extern("Vec", vec![closed_int(IntTy::U8)]))
    );
    assert_eq!(decision(&outcome, "lend"), Decision::LendPlace);
}

/// ```text
/// let mut buf = Vec::new();
/// push(&mut buf, 1u16);    // push(&mut Vec⟨T⟩, T)
/// consume(buf);            // consume(Vec⟨u8⟩)
/// ```
/// with the declared conversion `u16 → u8`, and without it.
#[test]
fn example_3_a_conversion_where_an_equation_wins() {
    let program = |conversion: bool| {
        let mut d = declare(&[("Vec", 1)]);
        function(
            &mut d,
            "new",
            &["T"],
            vec![],
            s_extern("Vec", vec![t(0)]),
            Role::Fn,
        );
        let vec_t = s_extern("Vec", vec![t(0)]);
        function(
            &mut d,
            "push",
            &["T"],
            vec![s_ref(Mutability::Exclusive, vec_t), t(0)],
            s_unit(),
            Role::Fn,
        );
        let vec_u8 = s_extern("Vec", vec![s_int(IntTy::U8)]);
        function(&mut d, "consume", &[], vec![vec_u8], s_unit(), Role::Fn);
        if conversion {
            function(
                &mut d,
                "narrow",
                &[],
                vec![s_int(IntTy::U16)],
                s_int(IntTy::U8),
                Role::Conversion,
            );
        }
        let mut p = ProgramBuilder::new(d.finish().unwrap());
        let buf = p.var("B");
        let new = p.var("new()");
        let lent = p.var("&mut buf");
        let u16_ = p.written("1u16", int(IntTy::U16));
        let pushed = p.var("push(..)");
        let consumed = p.var("consume(..)");
        p.call("new", "new", &[], &new);
        p.pass(&new, &buf);
        p.lend("lend", Mutability::Exclusive, &buf, &lent);
        p.call("push", "push", &[&lent, &u16_], &pushed);
        p.call("consume", "consume", &[&buf], &consumed);
        p.build()
    };
    let outcome = only_outcome(&program(true));
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "B"),
        Some(closed_extern("Vec", vec![closed_int(IntTy::U8)]))
    );
    let narrow = DeclId {
        name: Name::new("narrow"),
        overload: 0,
    };
    assert!(
        outcome
            .checks
            .iter()
            .any(|check| check.coercion == Ok(crate::coerce::Coercion::Conversion(narrow.clone()))),
        "{outcome:#?}"
    );
    let refused = only_outcome(&program(false));
    assert!(!refused.accepted, "{refused:#?}");
}

/// ```text
/// let mut v = Vec::new();
/// let r = &mut v;
/// let s = &r;
/// ```
#[test]
fn example_9_a_lend_that_reborrows_a_reference() {
    let mut d = declare(&[("Vec", 1)]);
    function(
        &mut d,
        "new",
        &["T"],
        vec![],
        s_extern("Vec", vec![t(0)]),
        Role::Fn,
    );
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let v = p.var("V");
    let new = p.var("new()");
    let lent_v = p.var("&mut v");
    let r = p.var("R");
    let lent_r = p.var("&r");
    let s = p.var("S");
    p.call("new", "new", &[], &new);
    p.pass(&new, &v);
    p.lend("lend v", Mutability::Exclusive, &v, &lent_v);
    p.pass(&lent_v, &r);
    p.lend("lend r", Mutability::Shared, &r, &lent_r);
    p.pass(&lent_r, &s);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        decision(&outcome, "lend r"),
        Decision::LendThrough(Mutability::Exclusive)
    );
    let vec_never = closed_extern("Vec", vec![closed(Term::Never)]);
    assert_eq!(
        value(&outcome, "S"),
        Some(closed(Term::Ref(Mutability::Shared, vec_never)))
    );
}

/// `holes.md`, H3: a view at a boundary.
/// ```text
/// let v = make();          // make() -> Vec⟨u8⟩; the view as_slice(&Vec⟨T⟩) -> &[T]
/// take(&v);                // take(&[u8])
/// ```
/// The boundary `&V ⇝ &[u8]` joins two references, whose constructors are
/// their mutabilities alone (l.66), so its target is an equation (l.289),
/// `V = [u8]`, and the view's cast is never taken.
#[test]
fn hole_3_a_view_at_a_boundary_equates_the_targets() {
    let mut d = declare(&[("Vec", 1)]);
    let vec_t = s_extern("Vec", vec![t(0)]);
    let slice_t = SigTy::term(Term::Slice(t(0)));
    function(
        &mut d,
        "core::as_slice",
        &["T"],
        vec![s_ref(Mutability::Shared, vec_t)],
        s_ref(Mutability::Shared, slice_t),
        Role::Cast,
    );
    let vec_u8 = s_extern("Vec", vec![s_int(IntTy::U8)]);
    function(&mut d, "make", &[], vec![], vec_u8, Role::Fn);
    let slice_u8 = SigTy::term(Term::Slice(s_int(IntTy::U8)));
    function(
        &mut d,
        "take",
        &[],
        vec![s_ref(Mutability::Shared, slice_u8)],
        s_unit(),
        Role::Fn,
    );
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let v = p.var("V");
    let made = p.var("make()");
    let lent = p.var("&v");
    let taken = p.var("take(..)");
    p.call("make", "make", &[], &made);
    p.pass(&made, &v);
    p.lend("lend", Mutability::Shared, &v, &lent);
    p.call("take", "take", &[&lent], &taken);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    let slice_u8 = closed(Term::Slice(closed_int(IntTy::U8)));
    assert_eq!(value(&outcome, "V"), Some(slice_u8));
}

/// `holes.md`, H4: a reborrow at a boundary.
/// ```text
/// let mut v = Vec::new();  // new() -> Vec⟨T⟩
/// look(&mut v);            // look(&Vec⟨u8⟩)
/// ```
/// The constructors `&mut` and `&` differ and no cast links them (l.291);
/// the model reads the reborrow row `&mut T ⇝ &T` as an equation of the
/// targets, so `V = Vec⟨u8⟩`.
#[test]
fn hole_4_a_reborrow_at_a_boundary_equates_the_targets() {
    let mut d = declare(&[("Vec", 1)]);
    function(
        &mut d,
        "new",
        &["T"],
        vec![],
        s_extern("Vec", vec![t(0)]),
        Role::Fn,
    );
    let vec_u8 = s_extern("Vec", vec![s_int(IntTy::U8)]);
    function(
        &mut d,
        "look",
        &[],
        vec![s_ref(Mutability::Shared, vec_u8)],
        s_unit(),
        Role::Fn,
    );
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let v = p.var("V");
    let new = p.var("new()");
    let lent = p.var("&mut v");
    let looked = p.var("look(..)");
    p.call("new", "new", &[], &new);
    p.pass(&new, &v);
    p.lend("lend", Mutability::Exclusive, &v, &lent);
    p.call("look", "look", &[&lent], &looked);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "V"),
        Some(closed_extern("Vec", vec![closed_int(IntTy::U8)]))
    );
}
