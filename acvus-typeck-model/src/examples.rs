//! The spec's examples (l.360–596), each encoded as the constraints its
//! text gives and solved in every order.

use crate::decl::{DeclId, Decls, DeclsBuilder, Role};
use crate::explore::{Exploration, explore};
use crate::node::{Decision, LendTarget};
use std::collections::BTreeSet;

use crate::program::{
    BindingPath, Deferral, Program, ProgramBuilder, Projection, Readings, Recording,
};
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
    assert_eq!(
        decision(&outcome, "lend"),
        Decision::Lend {
            target: LendTarget::Place,
            mutability: Mutability::Exclusive
        }
    );
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
        Decision::Lend {
            target: LendTarget::Through(Some(Mutability::Exclusive)),
            mutability: Mutability::Shared
        }
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

/// Every outcome some order of solving reaches.
fn outcomes(program: &Program) -> Vec<Outcome> {
    let Exploration {
        outcomes,
        states,
        complete,
    } = explore(program, BUDGET);
    assert!(complete, "explored {states} states within the budget");
    outcomes
}

fn int_lit(value: u128) -> crate::lit::LitVar {
    crate::lit::LitVar::of(crate::lit::Literal::Int(value))
}

fn field(name: &str) -> Projection {
    Projection::Field(Name::new(name))
}

fn variant(name: &str) -> Projection {
    Projection::Payload {
        enum_name: None,
        variant: Name::new(name),
    }
}

fn object(fields: &[(&str, Ty)]) -> Ty {
    closed(Term::Object(
        fields
            .iter()
            .map(|(name, ty)| (Name::new(name), ty.clone()))
            .collect(),
    ))
}

fn has_refusal<F>(outcome: &Outcome, which: F) -> bool
where
    F: Fn(&crate::solve::Refusal) -> bool,
{
    outcome.refusals.iter().any(which)
}

/// ```text
/// let mut x = None;
/// let r = &mut x;
/// *r = Some(1u8);
/// ```
#[test]
fn example_7_a_write_through_a_reference() {
    let mut p = ProgramBuilder::new(no_decls());
    let x = p.var("X");
    let k = p.var("?k");
    let none = p.written("None", Term::Option(k));
    let lent = p.var("&mut x");
    let r = p.var("R");
    let deref = p.var("*r");
    let s = p.var("S");
    let some = p.written("Some(1u8)", Term::Option(s.clone()));
    let u8_ = p.written("1u8", int(IntTy::U8));
    p.pass(&none, &x);
    p.lend("&mut x", Mutability::Exclusive, &x, &lent);
    p.pass(&lent, &r);
    p.project("*r", Projection::Deref, &r, &deref);
    p.pass(&u8_, &s);
    p.pass(&some, &deref);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "X"),
        Some(closed(Term::Option(closed_int(IntTy::U8))))
    );
    assert_eq!(
        decision(&outcome, "&mut x"),
        Decision::Lend {
            target: LendTarget::Place,
            mutability: Mutability::Exclusive
        }
    );
}

/// ```text
/// let a;
/// a.x = 0;
/// ```
#[test]
fn example_8_a_field_written_by_the_text() {
    let mut p = ProgramBuilder::new(no_decls());
    let a = p.var("A");
    let x = p.var("a.x");
    let zero = p.lit("0", int_lit(0));
    p.project("a.x", field("x"), &a, &x);
    p.pass(&zero, &x);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "A"),
        Some(object(&[("x", closed_int(IntTy::I32))]))
    );
    assert_eq!(decision(&outcome, "a.x"), Decision::Applied);
}

/// `@cfg.extra = 1;` with `@cfg` the host's `{name: String}`.
#[test]
fn example_8_a_field_of_a_type_the_host_gives() {
    let mut p = ProgramBuilder::new(no_decls());
    let string = p.written("String", Term::String);
    let cfg = p.host("@cfg", Term::Object([(Name::new("name"), string)].into()));
    let extra = p.var("@cfg.extra");
    let one = p.lit("1", int_lit(1));
    p.project("@cfg.extra", field("extra"), &cfg, &extra);
    p.pass(&one, &extra);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    assert!(
        has_refusal(&outcome, |refusal| matches!(
            refusal,
            crate::solve::Refusal::Conflict { .. }
        )),
        "{outcome:#?}"
    );
}

/// `if let Ok(v) = Some(1u8) { }`
#[test]
fn example_8_a_variant_the_language_writes() {
    let mut p = ProgramBuilder::new(no_decls());
    let s = p.var("S");
    let some = p.written("Some(1u8)", Term::Option(s.clone()));
    let u8_ = p.written("1u8", int(IntTy::U8));
    let payload = p.var("Ok.0");
    let v = p.var("V");
    p.pass(&u8_, &s);
    p.project("Ok", variant("Ok"), &some, &payload);
    p.bind("v", &["Ok"], &payload, &v);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    assert!(
        has_refusal(&outcome, |refusal| matches!(
            refusal,
            crate::solve::Refusal::Conflict { .. }
        )),
        "{outcome:#?}"
    );
}

/// `holes.md`, T5.
/// ```text
/// let a = @cfg;            // @cfg is { name: String }, given by the host
/// a.extra = 1;
/// ```
/// Read by the member, `extra` is written by the text and visible, and `a`
/// grows to `{extra, name}`; read by the sources, a lower bound the host gives
/// makes no member visible, and the kind conflicts with `{name}`.
#[test]
fn hole_t5_a_field_on_a_class_the_host_bounds() {
    let program = |visibility: crate::program::Visibility| {
        let mut p = ProgramBuilder::new(no_decls());
        p.readings(Readings {
            visibility,
            ..Readings::default()
        });
        let string = p.written("String", Term::String);
        let cfg = p.host("@cfg", Term::Object([(Name::new("name"), string)].into()));
        let a = p.var("A");
        let extra = p.var("a.extra");
        let one = p.lit("1", int_lit(1));
        p.pass(&cfg, &a);
        p.project("a.extra", field("extra"), &a, &extra);
        p.pass(&one, &extra);
        p.build()
    };
    let by_member = only_outcome(&program(crate::program::Visibility::ByMember));
    assert!(by_member.accepted, "{by_member:#?}");
    assert_eq!(
        value(&by_member, "A"),
        Some(object(&[
            ("extra", closed_int(IntTy::I32)),
            ("name", closed(Term::String)),
        ]))
    );
    let by_sources = only_outcome(&program(crate::program::Visibility::BySources));
    assert!(!by_sources.accepted, "{by_sources:#?}");
}

/// ```text
/// let mut v = Vec::new();  // new() -> Vec⟨T⟩; the view as_slice(&Vec⟨T⟩) -> &[T]
/// v[0] = 1u8;
/// ```
/// and without the view, which l.161 refuses.
#[test]
fn an_index_through_a_view() {
    let program = |view: bool| {
        let mut d = declare(&[("Vec", 1)]);
        function(
            &mut d,
            "new",
            &["T"],
            vec![],
            s_extern("Vec", vec![t(0)]),
            Role::Fn,
        );
        if view {
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
        }
        let mut p = ProgramBuilder::new(d.finish().unwrap());
        let v = p.var("V");
        let new = p.var("new()");
        let element = p.var("v[0]");
        let u8_ = p.written("1u8", int(IntTy::U8));
        p.call("new", "new", &[], &new);
        p.pass(&new, &v);
        p.project("v[0]", Projection::Elem, &v, &element);
        p.pass(&u8_, &element);
        p.build()
    };
    let outcome = only_outcome(&program(true));
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "V"),
        Some(closed_extern("Vec", vec![closed_int(IntTy::U8)]))
    );
    let refused = only_outcome(&program(false));
    assert!(!refused.accepted, "{refused:#?}");
}

/// ```text
/// let a;
/// a[0] = 1;
/// ```
/// The kind "indexes `τ`" names no member, so `A` has no source that gives a
/// constructor and falls back to `!`, which holds no element.
#[test]
fn an_index_alone_names_no_constructor() {
    let mut p = ProgramBuilder::new(no_decls());
    let a = p.var("A");
    let element = p.var("a[0]");
    let one = p.lit("1", int_lit(1));
    p.project("a[0]", Projection::Elem, &a, &element);
    p.pass(&one, &element);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
}

/// ```text
/// let o = Some(1u8);
/// if let Some(x) = &o { }
/// ```
/// The payload's projection applies to the target of `&o` (l.309), so the
/// path passes a shared reference and `x` lends its position (l.317).
#[test]
fn a_binding_behind_a_reference_lends_its_position() {
    let program = |binding_path: BindingPath| {
        let mut p = ProgramBuilder::new(no_decls());
        p.readings(Readings {
            binding_path,
            ..Readings::default()
        });
        let s = p.var("S");
        let some = p.written("Some(1u8)", Term::Option(s.clone()));
        let u8_ = p.written("1u8", int(IntTy::U8));
        let o = p.var("O");
        let lent = p.var("&o");
        let payload = p.var("Some.0");
        let x = p.var("X");
        p.pass(&u8_, &s);
        p.pass(&some, &o);
        p.lend("&o", Mutability::Shared, &o, &lent);
        p.project("Some", variant("Some"), &lent, &payload);
        p.bind("x", &["Some"], &payload, &x);
        p.build()
    };
    let outcome = only_outcome(&program(BindingPath::AfterPath));
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "X"),
        Some(closed(Term::Ref(Mutability::Shared, closed_int(IntTy::U8))))
    );
    assert_eq!(
        decision(&outcome, "Some"),
        Decision::AppliedToTarget(Some(Mutability::Shared))
    );
    assert_eq!(
        decision(&outcome, "x"),
        Decision::BindLend {
            target: LendTarget::Place,
            mutability: Mutability::Shared
        }
    );
    let as_stated = outcomes(&program(BindingPath::AsStated));
    let values: BTreeSet<Option<Ty>> = as_stated
        .iter()
        .map(|outcome| value(outcome, "X"))
        .collect();
    assert_eq!(
        values,
        BTreeSet::from([
            Some(closed_int(IntTy::U8)),
            Some(closed(Term::Ref(Mutability::Shared, closed_int(IntTy::U8)))),
        ]),
        "{as_stated:#?}"
    );
}

/// `holes.md`, T6.
/// ```text
/// let n = never();         // never() -> !
/// if let Some(x) = n { }
/// ```
/// The kind "has variant `Some`" gives no member, so `N` is `!`, which holds
/// no payload, and the model refuses; patterns §4 has the pattern decide
/// nothing.
#[test]
fn hole_t6_a_pattern_on_a_scrutinee_of_type_never() {
    let mut d = declare(&[]);
    function(
        &mut d,
        "never",
        &[],
        vec![],
        SigTy::term(Term::Never),
        Role::Fn,
    );
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let n = p.var("N");
    let called = p.var("never()");
    let payload = p.var("Some.0");
    let x = p.var("X");
    p.call("never", "never", &[], &called);
    p.pass(&called, &n);
    p.project("Some", variant("Some"), &n, &payload);
    p.bind("x", &["Some"], &payload, &x);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    assert!(
        has_refusal(&outcome, |refusal| matches!(
            refusal,
            crate::solve::Refusal::Conflict { at, .. } if at.contains("N")
        )),
        "{outcome:#?}"
    );
}

fn s_object(fields: &[(&str, SigTy)]) -> SigTy {
    SigTy::term(Term::Object(
        fields
            .iter()
            .map(|(name, ty)| (Name::new(name), ty.clone()))
            .collect(),
    ))
}

fn overload(name: &str, index: usize) -> DeclId {
    DeclId {
        name: Name::new(name),
        overload: index,
    }
}

/// The example of l.254–265.
/// ```text
/// let mut a;
/// a.x = 0;
/// a = pick(a);             // pick({x: i32}) -> {x: i32}, pick(u8) -> u8
/// ```
/// and with `pick(u8) -> &u8` in place of the second, which is refused.
#[test]
fn example_of_l254_a_projection_and_an_overload() {
    let program = |second_is_reference: bool| {
        let mut d = declare(&[]);
        let object = s_object(&[("x", s_int(IntTy::I32))]);
        function(&mut d, "pick", &[], vec![object.clone()], object, Role::Fn);
        let second = if second_is_reference {
            s_ref(Mutability::Shared, s_int(IntTy::U8))
        } else {
            s_int(IntTy::U8)
        };
        function(
            &mut d,
            "pick",
            &[],
            vec![s_int(IntTy::U8)],
            second,
            Role::Fn,
        );
        let mut p = ProgramBuilder::new(d.finish().unwrap());
        let a = p.var("A");
        let x = p.var("a.x");
        let zero = p.lit("0", int_lit(0));
        let picked = p.var("pick(a)");
        p.project("a.x", field("x"), &a, &x);
        p.pass(&zero, &x);
        p.call("pick", "pick", &[&a], &picked);
        p.pass(&picked, &a);
        p.build()
    };
    let outcome = only_outcome(&program(false));
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        decision(&outcome, "pick"),
        Decision::Use(overload("pick", 0))
    );
    assert_eq!(decision(&outcome, "a.x"), Decision::Applied);
    assert_eq!(
        value(&outcome, "A"),
        Some(object(&[("x", closed_int(IntTy::I32))]))
    );
    let refused = only_outcome(&program(true));
    assert!(!refused.accepted, "{refused:#?}");
    assert!(
        matches!(decision(&refused, "pick"), Decision::Unreached(_)),
        "{refused:#?}"
    );
}

/// ```text
/// fill_all(&mut parse(s));     // parse(&str) -> Vec⟨u8⟩, parse(&str) -> Vec⟨u16⟩; fill_all(&mut Vec⟨u8⟩)
/// let _ = &mut parse(s);
/// ```
#[test]
fn example_13_an_overload_chosen_by_its_result() {
    let mut d = declare(&[("Vec", 1)]);
    let str_ref = s_ref(Mutability::Shared, SigTy::term(Term::Str));
    for element in [IntTy::U8, IntTy::U16] {
        function(
            &mut d,
            "parse",
            &[],
            vec![str_ref.clone()],
            s_extern("Vec", vec![s_int(element)]),
            Role::Fn,
        );
    }
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
    let str_ = p.written("str", Term::Str);
    let s = p.written("&str", Term::Ref(Mutability::Shared, str_));
    let parsed = p.var("parse(s)");
    let lent = p.var("&mut parse(s)");
    let filled = p.var("fill_all(..)");
    let parsed_2 = p.var("parse(s) 2");
    let lent_2 = p.var("&mut parse(s) 2");
    p.call("parse", "parse", &[&s], &parsed);
    p.lend("lend", Mutability::Exclusive, &parsed, &lent);
    p.call("fill_all", "fill_all", &[&lent], &filled);
    p.call("parse 2", "parse", &[&s], &parsed_2);
    p.lend("lend 2", Mutability::Exclusive, &parsed_2, &lent_2);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    assert_eq!(
        decision(&outcome, "parse"),
        Decision::Use(overload("parse", 0))
    );
    assert_eq!(
        decision(&outcome, "lend"),
        Decision::Lend {
            target: LendTarget::Place,
            mutability: Mutability::Exclusive
        }
    );
    assert!(
        matches!(decision(&outcome, "parse 2"), Decision::Unreached(_)),
        "{outcome:#?}"
    );
}

/// Declarations of `core::add(&T, &T) -> T` for each of `numbers`.
fn add_impls(builder: &mut DeclsBuilder, numbers: &[IntTy]) {
    for number in numbers {
        let operand = s_ref(Mutability::Shared, s_int(*number));
        function(
            builder,
            "core::add",
            &[],
            vec![operand.clone(), operand],
            s_int(*number),
            Role::Fn,
        );
    }
}

/// ```text
/// let mut x = 0;
/// let mut y = 1u16;
/// while c {
///     x = x + y;
///     y = y + 1;
/// }
/// ```
/// A literal operand is a temporary of the literal's type, which `&` lends.
/// `x + y` and `y + 1` depend on each other and break their dependencies
/// (l.269); either reading of what a broken dependency defers reaches the
/// spec's outcome.
#[test]
fn example_12_accumulation_that_starts_from_a_literal() {
    for deferral in [Deferral::Recording, Deferral::Reading] {
        let readings = Readings {
            deferral,
            ..Readings::default()
        };
        let outcome = only_outcome(&example_12(readings, false));
        assert!(outcome.accepted, "{outcome:#?}");
        for class in ["X", "Y", "?a", "?b", "R1", "R2"] {
            assert_eq!(
                value(&outcome, class),
                Some(closed_int(IntTy::U16)),
                "{class}: {outcome:#?}"
            );
        }
        let add_u16 = overload("core::add", 1);
        assert_eq!(decision(&outcome, "x + y"), Decision::Use(add_u16.clone()));
        assert_eq!(decision(&outcome, "y + 1"), Decision::Use(add_u16));
    }
}

/// `holes.md`, H7: Example 12 followed by `probe(x)`, with `probe(u16)` and
/// `probe(i32)`. `probe` and `x + y` both read `X`, each lying at a position the
/// other reads, so each breaks its dependency on the other (l.250, l.269), and
/// `probe` reads `X` without `x + y`'s relations, as `?a` alone, takes the
/// default `i32`, and is refused in every order. Where only the nodes that
/// would record a relation into a class record into it (`Recording::Relation`),
/// `probe`, which records nothing into its argument, is no dependency of
/// `x + y`, waits for it, and reads `X` as `u16`.
#[test]
fn hole_7_two_uses_that_read_one_class() {
    let by_position = only_outcome(&example_12(Readings::default(), true));
    assert!(!by_position.accepted, "{by_position:#?}");
    assert_eq!(
        decision(&by_position, "probe"),
        Decision::Use(overload("probe", 1))
    );
    let by_relation = Readings {
        recording: Recording::Relation,
        deferral: Deferral::Reading,
        ..Readings::default()
    };
    let outcome = only_outcome(&example_12(by_relation, true));
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        decision(&outcome, "probe"),
        Decision::Use(overload("probe", 0))
    );
}

/// `holes.md`, O2: the program of H7, read by relation (`Recording::Relation`).
/// `x + y` and `y + 1` break their dependencies on each other. Decided after
/// `x + y` and before `y + 1`, `probe` reads `X` without the relations `x + y`
/// defers until `y + 1` is decided (`Deferral::Recording`), and takes `i32`;
/// where only the breaking node's read leaves them out (`Deferral::Reading`),
/// every order reaches `probe(u16)`.
#[test]
fn hole_o2_a_deferred_relation_read_by_a_third_node() {
    let readings = |deferral| Readings {
        recording: Recording::Relation,
        deferral,
        ..Readings::default()
    };
    let recording = outcomes(&example_12(readings(Deferral::Recording), true));
    let decisions: BTreeSet<Decision> = recording
        .iter()
        .map(|outcome| decision(outcome, "probe"))
        .collect();
    assert_eq!(
        decisions,
        BTreeSet::from([
            Decision::Use(overload("probe", 0)),
            Decision::Use(overload("probe", 1)),
        ]),
        "{recording:#?}"
    );
    let outcome = only_outcome(&example_12(readings(Deferral::Reading), true));
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        decision(&outcome, "probe"),
        Decision::Use(overload("probe", 0))
    );
}

fn example_12(readings: Readings, probe: bool) -> Program {
    let mut d = declare(&[]);
    add_impls(&mut d, &[IntTy::U8, IntTy::U16, IntTy::I32]);
    for number in [IntTy::U16, IntTy::I32] {
        function(
            &mut d,
            "probe",
            &[],
            vec![s_int(number)],
            s_unit(),
            Role::Fn,
        );
    }
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    p.readings(readings);
    let x = p.var("X");
    let a = p.lit("?a", int_lit(0));
    let y = p.var("Y");
    let u16_ = p.written("1u16", int(IntTy::U16));
    let b = p.lit("?b", int_lit(1));
    let lx = p.var("&x");
    let ly = p.var("&y");
    let r1 = p.var("R1");
    let ly_2 = p.var("&y 2");
    let lb = p.var("&1");
    let r2 = p.var("R2");
    p.pass(&a, &x);
    p.pass(&u16_, &y);
    p.lend("&x", Mutability::Shared, &x, &lx);
    p.lend("&y", Mutability::Shared, &y, &ly);
    p.call("x + y", "core::add", &[&lx, &ly], &r1);
    p.pass(&r1, &x);
    p.lend("&y 2", Mutability::Shared, &y, &ly_2);
    p.lend("&1", Mutability::Shared, &b, &lb);
    p.call("y + 1", "core::add", &[&ly_2, &lb], &r2);
    p.pass(&r2, &y);
    if probe {
        let probed = p.var("probe(x)");
        p.call("probe", "probe", &[&x], &probed);
    }
    p.build()
}

/// `holes.md`, O3.
/// ```text
/// let r = first(x);        // x: A; first(A) -> &u8, first(B) -> &mut u8
/// let s = &mut r;
/// ```
/// Both candidates return a reference, so `first` records at once that its
/// result is one (l.252), and the lend no longer depends on it (l.248). Decided
/// before `first`, the lend reads `r` as a reference whose mutability the value
/// read does not decide, and `&mut` gives no result (l.307).
#[test]
fn hole_o3_a_reference_recorded_at_once_has_no_mutability() {
    let mut d = declare(&[("A", 0), ("B", 0)]);
    for (param, mutability) in [("A", Mutability::Shared), ("B", Mutability::Exclusive)] {
        function(
            &mut d,
            "first",
            &[],
            vec![s_extern(param, vec![])],
            s_ref(mutability, s_int(IntTy::U8)),
            Role::Fn,
        );
    }
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let x = p.written(
        "x",
        Term::Extern {
            name: Name::new("A"),
            args: vec![],
            identity: None,
        },
    );
    let first = p.var("first(x)");
    let r = p.var("R");
    let lent = p.var("&mut r");
    p.call("first", "first", &[&x], &first);
    p.pass(&first, &r);
    p.lend("&mut r", Mutability::Exclusive, &r, &lent);
    let outcomes = outcomes(&p.build());
    let decisions: BTreeSet<Decision> = outcomes
        .iter()
        .map(|outcome| decision(outcome, "&mut r"))
        .collect();
    assert_eq!(
        decisions,
        BTreeSet::from([
            Decision::NoResult,
            Decision::Lend {
                target: LendTarget::Through(Some(Mutability::Shared)),
                mutability: Mutability::Shared
            },
        ]),
        "{outcomes:#?}"
    );
}

/// ```text
/// let r = f(never());      // never() -> !; f(u8) -> u8, f(u16) -> u16
/// ```
/// The argument reads as `!`, so the use is decided with the result `!`
/// (l.344).
#[test]
fn a_use_whose_argument_is_never() {
    let mut d = declare(&[]);
    function(
        &mut d,
        "never",
        &[],
        vec![],
        SigTy::term(Term::Never),
        Role::Fn,
    );
    for number in [IntTy::U8, IntTy::U16] {
        function(
            &mut d,
            "f",
            &[],
            vec![s_int(number)],
            s_int(number),
            Role::Fn,
        );
    }
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let never = p.var("never()");
    let r = p.var("R");
    p.call("never", "never", &[], &never);
    p.call("f", "f", &[&never], &r);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(decision(&outcome, "f"), Decision::Never);
    assert_eq!(value(&outcome, "R"), Some(closed(Term::Never)));
}

fn s_named(name: &str) -> SigTy {
    s_extern(name, vec![])
}

/// ```text
/// fn g(a, b) { g(b, a) }
/// ```
/// ```text
/// g(1u8, "s")
/// ```
#[test]
fn example_10_recursion_by_name() {
    let mut p = ProgramBuilder::new(no_decls());
    p.group("g");
    let a = p.var("A");
    let b = p.var("B");
    let r = p.var("R");
    let called = p.var("g(b, a)");
    p.function("g", &[&a, &b], &r);
    p.call("g(b, a)", "g", &[&b, &a], &called);
    p.pass(&called, &r);
    p.group(crate::program::ENTRY);
    let u8_ = p.written("1u8", int(IntTy::U8));
    let str_ = p.written("str", Term::Str);
    let s = p.written("\"s\"", Term::Ref(Mutability::Shared, str_));
    let out = p.var("g(1u8, \"s\")");
    p.call("g(1u8, \"s\")", "g", &[&u8_, &s], &out);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    let failed: Vec<&str> = outcome
        .checks
        .iter()
        .filter(|check| check.coercion.is_err())
        .map(|check| check.relation.as_str())
        .collect();
    assert_eq!(failed, ["g@2/A ⇝ g@2/B", "g@2/B ⇝ g@2/A"], "{outcome:#?}");
}

/// Review 2's P4 (`holes.md`, T2).
/// ```text
/// fn g(a, b) { g(b, b); 0u8 }
/// ```
/// ```text
/// g(Some(f1()), Some(f2()))    // f1() -> F1, f2() -> F2; casts F1 → G, F2 → G
/// ```
/// The copied boundary `B' ⇝ A'` relates the payloads once both are solved,
/// so `A'`'s payload joins `F1` and `F2` to `G` in every order.
#[test]
fn review_2_p4_a_boundary_between_two_instances_universals() {
    let mut d = declare(&[("F1", 0), ("F2", 0), ("G", 0)]);
    function(&mut d, "f1", &[], vec![], s_named("F1"), Role::Fn);
    function(&mut d, "f2", &[], vec![], s_named("F2"), Role::Fn);
    function(
        &mut d,
        "up1",
        &[],
        vec![s_named("F1")],
        s_named("G"),
        Role::Cast,
    );
    function(
        &mut d,
        "up2",
        &[],
        vec![s_named("F2")],
        s_named("G"),
        Role::Cast,
    );
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    p.group("g");
    let a = p.var("A");
    let b = p.var("B");
    let r = p.var("R");
    let called = p.var("g(b, b)");
    let u8_ = p.written("0u8", int(IntTy::U8));
    p.function("g", &[&a, &b], &r);
    p.call("g(b, b)", "g", &[&b, &b], &called);
    p.pass(&u8_, &r);
    p.group(crate::program::ENTRY);
    let f1 = p.var("f1()");
    let f2 = p.var("f2()");
    let s1 = p.var("S1");
    let s2 = p.var("S2");
    let some_1 = p.written("Some(f1())", Term::Option(s1.clone()));
    let some_2 = p.written("Some(f2())", Term::Option(s2.clone()));
    let out = p.var("g(..)");
    p.call("f1", "f1", &[], &f1);
    p.call("f2", "f2", &[], &f2);
    p.pass(&f1, &s1);
    p.pass(&f2, &s2);
    p.call("g(..)", "g", &[&some_1, &some_2], &out);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(value(&outcome, "g(..)"), Some(closed_int(IntTy::U8)));
}

/// ```text
/// let mut last = None;
/// let remember = |x| last = Some(x);
/// remember(1u8);
/// ```
#[test]
fn example_6_a_capture_written_by_name() {
    let mut p = ProgramBuilder::new(no_decls());
    p.group_in("λ", Some(crate::program::ENTRY));
    let x = p.var("X");
    let s = p.var("S");
    let some = p.written("Some(x)", Term::Option(s.clone()));
    let unit = p.written("()", Term::Unit);
    let returned = p.var("λ result");
    p.function("λ", &[&x], &returned);
    p.pass(&x, &s);
    p.group(crate::program::ENTRY);
    let last = p.var("L");
    let k = p.var("?k");
    let none = p.written("None", Term::Option(k));
    let lambda = p.class("|x| ..", crate::program::Origin::Lambda(Name::new("λ")));
    let remember = p.var("remember");
    let u8_ = p.written("1u8", int(IntTy::U8));
    let called = p.var("remember(1u8)");
    p.pass(&none, &last);
    p.pass(&lambda, &remember);
    p.call_value("remember(1u8)", &remember, &[&u8_], &called);
    p.group("λ");
    p.pass_at(&some, &last, crate::program::Place::ByName);
    p.pass(&unit, &returned);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "L"),
        Some(closed(Term::Option(closed_int(IntTy::U8))))
    );
    assert_eq!(value(&outcome, "S"), None, "S stays in the lambda's C");
}

/// ```text
/// let mut x = None;
/// let r = &mut x;
/// let f = || *r = Some(1u8);
/// f();
/// ```
/// The write through the captured `r` is a boundary (l.464), whose payload,
/// an unknown of the enclosing group, takes a lower bound (l.286).
#[test]
fn example_7_a_write_through_a_captured_reference() {
    let mut p = ProgramBuilder::new(no_decls());
    p.group_in("λ", Some(crate::program::ENTRY));
    let deref = p.var("*r");
    let s = p.var("S");
    let some = p.written("Some(1u8)", Term::Option(s.clone()));
    let u8_ = p.written("1u8", int(IntTy::U8));
    let unit = p.written("()", Term::Unit);
    let returned = p.var("λ result");
    p.function("λ", &[], &returned);
    p.pass(&u8_, &s);
    p.pass(&unit, &returned);
    p.group(crate::program::ENTRY);
    let x = p.var("X");
    let k = p.var("?k");
    let none = p.written("None", Term::Option(k));
    let lent = p.var("&mut x");
    let r = p.var("R");
    let lambda = p.class("|| ..", crate::program::Origin::Lambda(Name::new("λ")));
    let f = p.var("f");
    let called = p.var("f()");
    p.pass(&none, &x);
    p.lend("&mut x", Mutability::Exclusive, &x, &lent);
    p.pass(&lent, &r);
    p.pass(&lambda, &f);
    p.call_value("f()", &f, &[], &called);
    p.group("λ");
    p.project("*r", Projection::Deref, &r, &deref);
    p.pass_at(&some, &deref, crate::program::Place::ThroughRef);
    let outcome = only_outcome(&p.build());
    assert!(outcome.accepted, "{outcome:#?}");
    assert_eq!(
        value(&outcome, "X"),
        Some(closed(Term::Option(closed_int(IntTy::U8))))
    );
}

/// `holes.md`, O4.
/// ```text
/// let mut x = make_a();    // make_a() -> A, make_b() -> B; a cast A → B, a conversion B → A
/// let put = |v| x = v;
/// probe(x);                // probe(A), probe(B)
/// put(make_b());
/// ```
/// `probe` reads `X`, and the call of `put` records into `X` only once it is
/// decided and makes its instance, so nothing makes `probe` wait for it.
#[test]
fn hole_o4_a_capture_written_after_a_use_reads_it() {
    let mut d = declare(&[("A", 0), ("B", 0)]);
    function(&mut d, "make_a", &[], vec![], s_named("A"), Role::Fn);
    function(&mut d, "make_b", &[], vec![], s_named("B"), Role::Fn);
    function(
        &mut d,
        "up",
        &[],
        vec![s_named("A")],
        s_named("B"),
        Role::Cast,
    );
    function(
        &mut d,
        "down",
        &[],
        vec![s_named("B")],
        s_named("A"),
        Role::Conversion,
    );
    for param in ["A", "B"] {
        function(
            &mut d,
            "probe",
            &[],
            vec![s_named(param)],
            s_unit(),
            Role::Fn,
        );
    }
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    p.group_in("λ", Some(crate::program::ENTRY));
    let v = p.var("V");
    let unit = p.written("()", Term::Unit);
    let returned = p.var("λ result");
    p.function("λ", &[&v], &returned);
    p.pass(&unit, &returned);
    p.group(crate::program::ENTRY);
    let x = p.var("X");
    let made_a = p.var("make_a()");
    let lambda = p.class("|v| ..", crate::program::Origin::Lambda(Name::new("λ")));
    let put = p.var("put");
    let probed = p.var("probe(x)");
    let made_b = p.var("make_b()");
    let called = p.var("put(..)");
    p.call("make_a", "make_a", &[], &made_a);
    p.pass(&made_a, &x);
    p.pass(&lambda, &put);
    p.call("probe", "probe", &[&x], &probed);
    p.call("make_b", "make_b", &[], &made_b);
    p.call_value("put(..)", &put, &[&made_b], &called);
    p.group("λ");
    p.pass_at(&v, &x, crate::program::Place::ByName);
    let outcomes = outcomes(&p.build());
    let decisions: BTreeSet<Decision> = outcomes
        .iter()
        .map(|outcome| decision(outcome, "probe"))
        .collect();
    assert_eq!(
        decisions,
        BTreeSet::from([
            Decision::Use(overload("probe", 0)),
            Decision::Use(overload("probe", 1)),
        ]),
        "{outcomes:#?}"
    );
}

/// `holes.md`, H5.
/// ```text
/// let x = make();          // make() -> A
/// let f = || probe(x);     // probe(A), probe(B)
/// f();
/// ```
/// The lambda's group is solved before the enclosing one (l.220), and no
/// dependency makes its `probe` wait for what the enclosing group records into
/// `X` (l.216, l.244–250), so `probe` reads `X` as nothing and is refused.
#[test]
fn hole_5_a_lambda_reads_a_capture() {
    let mut d = declare(&[("A", 0), ("B", 0)]);
    function(&mut d, "make", &[], vec![], s_named("A"), Role::Fn);
    for param in ["A", "B"] {
        function(
            &mut d,
            "probe",
            &[],
            vec![s_named(param)],
            s_unit(),
            Role::Fn,
        );
    }
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    p.group_in("λ", Some(crate::program::ENTRY));
    let returned = p.var("λ result");
    let probed = p.var("probe(x)");
    p.function("λ", &[], &returned);
    p.pass(&probed, &returned);
    p.group(crate::program::ENTRY);
    let x = p.var("X");
    let made = p.var("make()");
    let lambda = p.class("|| ..", crate::program::Origin::Lambda(Name::new("λ")));
    let f = p.var("f");
    let called = p.var("f()");
    p.call("make", "make", &[], &made);
    p.pass(&made, &x);
    p.pass(&lambda, &f);
    p.call_value("f()", &f, &[], &called);
    p.group("λ");
    p.call("probe", "probe", &[&x], &probed);
    let outcome = only_outcome(&p.build());
    assert!(!outcome.accepted, "{outcome:#?}");
    assert!(
        matches!(decision(&outcome, "probe"), Decision::Unreached(_)),
        "{outcome:#?}"
    );
}

/// Declarations of `core::eq(&T, &T) -> bool` and `core::sub(&T, &T) -> T` for
/// each of `numbers`.
fn comparison_impls(builder: &mut DeclsBuilder, numbers: &[IntTy]) {
    for number in numbers {
        let operand = s_ref(Mutability::Shared, s_int(*number));
        let bool_ = SigTy::term(Term::Scalar(Scalar::Bool));
        function(
            builder,
            "core::eq",
            &[],
            vec![operand.clone(), operand.clone()],
            bool_,
            Role::Fn,
        );
        function(
            builder,
            "core::sub",
            &[],
            vec![operand.clone(), operand],
            s_int(*number),
            Role::Fn,
        );
    }
}

/// ```text
/// struct Pair { l, r }
/// impl Pair {
///     fn new(l, r) { Self { l, r } }
///     fn swap(self) { Self { l: self.r, r: self.l } }
///     fn nest(self, n) { if n == 0 { 0u8 } else { Self { l: self, r: 0u8 }.nest(n - 1) } }
/// }
/// ```
/// ```text
/// let p = Pair::new(1u8, "s");
/// let q = p.swap();
/// q.nest(3)
/// ```
/// Its states are too many to explore, so the orders are sampled. Refused, as
/// the spec has it, but where `q` has no value (`holes.md`, H6).
#[test]
fn example_14_a_struct_and_recursion_that_grows() {
    let pair = Name::new("Pair");
    let mut types = DeclaredTypes::default();
    types
        .declare_struct(
            pair.clone(),
            crate::sig::StructDecl {
                fields: vec![Name::new("l"), Name::new("r")],
            },
        )
        .unwrap();
    let mut d = DeclsBuilder::new(types);
    comparison_impls(&mut d, &[IntTy::U8, IntTy::I32]);
    let mut p = ProgramBuilder::new(d.finish().unwrap());
    let u8_term = || int(IntTy::U8);
    let made = |p: &mut ProgramBuilder, name: &str| {
        let l = p.var(&format!("{name}.l"));
        let r = p.var(&format!("{name}.r"));
        let made = p.written(name, Term::Struct(pair.clone(), vec![l.clone(), r.clone()]));
        (made, l, r)
    };
    let receiver = |p: &mut ProgramBuilder, group: &str| {
        let a = p.var(&format!("{group} A"));
        let b = p.var(&format!("{group} B"));
        p.universal(&a);
        p.universal(&b);
        p.written(
            &format!("{group} self"),
            Term::Struct(pair.clone(), vec![a, b]),
        )
    };

    p.group("new");
    let l = p.var("new L");
    let r = p.var("new R");
    let returned = p.var("new result");
    p.function("new", &[&l, &r], &returned);
    let (value_, part_l, part_r) = made(&mut p, "Self { l, r }");
    p.pass(&l, &part_l);
    p.pass(&r, &part_r);
    p.pass(&value_, &returned);

    p.group("swap");
    let self_ = receiver(&mut p, "swap");
    let returned = p.var("swap result");
    p.function("swap", &[&self_], &returned);
    let self_r = p.var("self.r");
    let self_l = p.var("self.l");
    p.project("self.r", field("r"), &self_, &self_r);
    p.project("self.l", field("l"), &self_, &self_l);
    let (value_, part_l, part_r) = made(&mut p, "Self { l: self.r, r: self.l }");
    p.pass(&self_r, &part_l);
    p.pass(&self_l, &part_r);
    p.pass(&value_, &returned);

    p.group("nest");
    let self_ = receiver(&mut p, "nest");
    let n = p.var("N");
    let returned = p.var("nest result");
    p.function("nest", &[&self_, &n], &returned);
    let lent_n = p.var("&n");
    let zero = p.lit("0", int_lit(0));
    let lent_zero = p.var("&0");
    let compared = p.var("n == 0");
    p.lend("&n", Mutability::Shared, &n, &lent_n);
    p.lend("&0", Mutability::Shared, &zero, &lent_zero);
    p.call("n == 0", "core::eq", &[&lent_n, &lent_zero], &compared);
    let tail = p.written("0u8", u8_term());
    p.pass(&tail, &returned);
    let (value_, part_l, part_r) = made(&mut p, "Self { l: self, r: 0u8 }");
    let r_u8 = p.written("r: 0u8", u8_term());
    p.pass(&self_, &part_l);
    p.pass(&r_u8, &part_r);
    let lent_n_2 = p.var("&n 2");
    let one = p.lit("1", int_lit(1));
    let lent_one = p.var("&1");
    let less = p.var("n - 1");
    p.lend("&n 2", Mutability::Shared, &n, &lent_n_2);
    p.lend("&1", Mutability::Shared, &one, &lent_one);
    p.call("n - 1", "core::sub", &[&lent_n_2, &lent_one], &less);
    let recursed = p.var("….nest(n - 1)");
    p.call("….nest(n - 1)", "nest", &[&value_, &less], &recursed);
    p.pass(&recursed, &returned);

    p.group(crate::program::ENTRY);
    let u8_ = p.written("1u8", u8_term());
    let str_ = p.written("str", Term::Str);
    let s = p.written("\"s\"", Term::Ref(Mutability::Shared, str_));
    let new = p.var("Pair::new(..)");
    let p_ = p.var("p");
    p.call("Pair::new", "new", &[&u8_, &s], &new);
    p.pass(&new, &p_);
    let swapped = p.var("p.swap()");
    let q = p.var("q");
    p.call("p.swap()", "swap", &[&p_], &swapped);
    p.pass(&swapped, &q);
    let three = p.lit("3", int_lit(3));
    let nested = p.var("q.nest(3)");
    p.call("q.nest(3)", "nest", &[&q, &three], &nested);

    let outcomes = crate::explore::sample(&p.build(), 200);
    assert_eq!(
        outcomes.len(),
        1,
        "every sampled order reaches one outcome: {outcomes:#?}"
    );
    let outcome = outcomes.into_iter().next().unwrap();
    assert!(!outcome.accepted, "{outcome:#?}");
    let str_ref = closed(Term::Ref(Mutability::Shared, closed(Term::Str)));
    let pair_of = |l: Ty, r: Ty| closed(Term::Struct(pair.clone(), vec![l, r]));
    assert_eq!(
        value(&outcome, "p"),
        Some(pair_of(closed_int(IntTy::U8), str_ref))
    );
    // `holes.md`, H6: the copy of `nest`'s `A` is `q`'s first part, whose
    // lower bounds are `&str` and `Pair⟨A', B'⟩`, so it reaches a part of its
    // own (l.275) and `q` has no value.
    assert_eq!(value(&outcome, "q"), None, "{outcome:#?}");
    assert!(
        has_refusal(&outcome, |refusal| matches!(
            refusal,
            crate::solve::Refusal::Infinite(_)
        )),
        "{outcome:#?}"
    );
    let narrowed: Vec<Decision> = ["n == 0", "n - 1"]
        .iter()
        .map(|node| {
            let copied = outcome
                .decisions
                .keys()
                .find(|name| name.as_str().ends_with(&format!("/{node}")))
                .unwrap_or_else(|| panic!("{node} is copied: {outcome:#?}"));
            decision(&outcome, copied.as_str())
        })
        .collect();
    assert_eq!(
        narrowed,
        [
            Decision::Use(overload("core::eq", 1)),
            Decision::Use(overload("core::sub", 1)),
        ]
    );
}
