//! A `&[T]` parameter takes `&v` by coercion (RFC-0047 rule 6): where the
//! container's `as_slice` ends up, and what a container that declares none is
//! refused with.

use acvus_extern::vec_ty;
use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    Effect, EffectTerm, Instances, Mutability, ParamTerm, Poly, PolyTy, Ty, TypeArg, lift_to_poly,
};
use acvus_mir_test::compile_script_mode_ir_with;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn slice_of_i64(mutability: Mutability) -> Ty {
    Ty::Ref(
        mutability,
        Box::new(TypeArg::uniform(Ty::Slice(Box::new(Ty::I64)))),
    )
}

fn takes_slice(i: &Interner, name: &str, mutability: Mutability) -> Function {
    let ty = PolyTy::Fn {
        params: vec![ParamTerm::new(
            i.intern("s"),
            lift_to_poly(&slice_of_i64(mutability)),
        )],
        ret: Box::new(lift_to_poly(&Ty::I64)),
        captures: vec![],
        effect: EffectTerm::<Poly>::Known(Effect::PURE),
    };
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty,
    }
}

fn two_slices(i: &Interner, name: &str) -> Function {
    let ty = PolyTy::Fn {
        params: vec![
            ParamTerm::new(
                i.intern("a"),
                lift_to_poly(&slice_of_i64(Mutability::Shared)),
            ),
            ParamTerm::new(
                i.intern("b"),
                lift_to_poly(&slice_of_i64(Mutability::Shared)),
            ),
        ],
        ret: Box::new(lift_to_poly(&Ty::I64)),
        captures: vec![],
        effect: EffectTerm::<Poly>::Known(Effect::PURE),
    };
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty,
    }
}

fn ir(source: &str) -> String {
    let i = Interner::new();
    let externs = [
        takes_slice(&i, "total", Mutability::Shared),
        takes_slice(&i, "clear", Mutability::Mut),
        two_slices(&i, "two"),
    ];
    compile_script_mode_ir_with(&i, source, &FxHashMap::default(), &externs)
        .unwrap_or_else(|e| panic!("{source}\n{e}"))
}

fn refusal(source: &str) -> String {
    let i = Interner::new();
    let externs = [
        takes_slice(&i, "total", Mutability::Shared),
        takes_slice(&i, "clear", Mutability::Mut),
    ];
    let reported = compile_script_mode_ir_with(&i, source, &FxHashMap::default(), &externs)
        .expect_err(&format!("{source} is refused"));
    let found: Vec<&str> = reported
        .lines()
        .map(|line| line.split("] ").nth(1).unwrap_or(line))
        .collect();
    assert_eq!(found.len(), 1, "{source}: {found:?}");
    found[0].to_string()
}

fn main_body(ir: &str) -> &str {
    ir.split("=== main ===")
        .nth(1)
        .expect("an entry body")
        .split("=== ")
        .next()
        .expect("the entry body ends at the next section")
}

fn count(body: &str, needle: &str) -> usize {
    body.matches(needle).count()
}

#[test]
fn a_container_argument_at_a_slice_parameter_is_one_as_slice_before_the_call() {
    let ir = ir("let v = vec([1, 2, 3]); total(&v)");
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 1, "{body}");
    let sliced = body
        .lines()
        .position(|line| line.contains("as_slice"))
        .expect("the as_slice");
    let called = body
        .lines()
        .position(|line| line.contains("total"))
        .expect("the call");
    assert!(sliced < called, "the slice is the argument:\n{body}");
}

#[test]
fn an_exclusive_slice_parameter_takes_the_container_exclusively() {
    let ir = ir("let v = vec([1, 2, 3]); clear(&mut v)");
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice &mut"), 1, "{body}");
    assert!(body.contains("ref &mut v"), "{body}");
}

#[test]
fn an_element_of_a_nested_container_reaches_a_slice_parameter() {
    let ir = ir("let m = vec([vec([1, 2]), vec([3, 4])]); total(&m[1])");
    let body = main_body(&ir);
    assert_eq!(
        count(body, "as_slice"),
        2,
        "one for the row's index and one for the argument:\n{body}"
    );
}

#[test]
fn a_context_container_reaches_a_slice_parameter() {
    let i = Interner::new();
    let externs = [takes_slice(&i, "total", Mutability::Shared)];
    let context: FxHashMap<Astr, Ty> = [(i.intern("rows"), vec_ty(&i, Ty::I64))]
        .into_iter()
        .collect();
    let ir = compile_script_mode_ir_with(&i, "total(&@rows)", &context, &externs)
        .unwrap_or_else(|e| panic!("{e}"));
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 1, "{body}");
}

#[test]
fn a_value_that_is_no_container_is_refused_at_the_argument() {
    assert_eq!(
        refusal("let n = 1; total(&n)"),
        "type mismatch: expected &[i64], got &i64"
    );
}

#[test]
fn a_shared_borrow_does_not_reach_an_exclusive_slice_parameter() {
    assert_eq!(
        refusal("let v = vec([1, 2, 3]); clear(&v)"),
        "type mismatch: expected &mut [i64], got &Vec<i64>; write `&mut` where the `&` is"
    );
}

/// The coercion a `&[T]` parameter takes is a declaration a script may
/// also name, and naming it takes that one view and no second one.
#[test]
fn a_script_names_the_coercion_it_would_have_got() {
    let ir = ir("let v = vec([1, 2, 3]); total(as_slice(&v))");
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 1, "{body}");
    let sliced = body
        .lines()
        .position(|line| line.contains("as_slice"))
        .expect("the as_slice");
    let called = body
        .lines()
        .position(|line| line.contains("total"))
        .expect("the call");
    assert!(sliced < called, "the slice is the argument:\n{body}");
}

#[test]
fn two_slice_parameters_take_two_as_slices_into_one_call() {
    let ir = ir("let a = vec([1, 2, 3]); let b = vec([10, 20, 30]); two(&a, &b)");
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 2, "{body}");
    let calls: Vec<usize> = body
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("call #"))
        .map(|(at, _)| at)
        .collect();
    let call = *calls.last().expect("the call");
    let sliced: Vec<usize> = body
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("as_slice"))
        .map(|(at, _)| at)
        .collect();
    assert!(
        sliced.iter().all(|at| *at < call),
        "both slices are the call's arguments:\n{body}"
    );
}
