//! A shared signature with polymorphic instances, at the contract
//! (RFC-0027): the instance the argument's shape picks fixes the variables
//! the signature left open, a call no instance matches is an error at the
//! call, and a call whose argument is still open stays open.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{
    LenTerm, ParamTerm, Poly, PolyBuilder, PolyTy, Ty, TyTerm, TyVarBound, TypeRegistry,
};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

fn fn_of(i: &Interner, params: &[(&str, PolyTy)], ret: PolyTy) -> PolyTy {
    TyTerm::Fn {
        params: params
            .iter()
            .map(|(n, t)| ParamTerm::<Poly>::new(i.intern(n), t.clone()))
            .collect(),
        ret: Box::new(ret),
        captures: vec![],
        effect: acvus_mir::ty::Effect::PURE.into(),
    }
}

/// `pick<C, T>(c: C, f: Fn(T) -> T) -> T` with instances for `Array<T, N>`
/// and `Option<T>`.
fn pick_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let c = pb.fresh_ty_var();
    let t = pb.fresh_ty_var();
    let f = |elem: &PolyTy| fn_of(i, &[("x", elem.clone())], elem.clone());
    let array = TyTerm::Array(Box::new(t.clone()), LenTerm::Var(0));
    let option = TyTerm::Option(Box::new(t.clone()));
    let instances = vec![
        fn_of(i, &[("c", array.clone()), ("f", f(&t))], t.clone()),
        fn_of(i, &[("c", option.clone()), ("f", f(&t))], t.clone()),
    ];
    Function {
        qref: QualifiedRef::root(i.intern("pick")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::OneOf(vec![array, option])],
            instances: acvus_mir::ty::Instances {
                concrete: instances,
                generic: false,
            },
        },
        ty: fn_of(i, &[("c", c), ("f", f(&t))], t),
    }
}

fn script_fn(i: &Interner, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("f")),
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(i, source).expect("parse"),
        )),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
        },
    }
}

/// The script's return type, or every error the checker reported.
fn check(i: &Interner, source: &str) -> Result<Ty, Vec<String>> {
    let f = script_fn(i, source);
    let qref = f.qref;
    let graph = CompilationGraph {
        functions: Freeze::new(vec![pick_fn(i), f]),
        contexts: Freeze::new(vec![]),
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(
        i,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(TypeRegistry::new()),
    );
    if inf.has_errors() {
        return Err(inf
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
            .collect());
    }
    match &inf.outcomes[&qref].meta().ty {
        Ty::Fn { ret, .. } => Ok((**ret).clone()),
        other => panic!("expected Fn, got {other:?}"),
    }
}

#[test]
fn the_argument_s_shape_picks_the_instance_and_fixes_the_element_type() {
    let i = Interner::new();
    assert_eq!(check(&i, "pick([1, 2], |x| -> x)").unwrap(), Ty::I64);
    assert_eq!(
        check(&i, "pick(Some(\"a\"), |x| -> x)").unwrap(),
        Ty::String
    );
}

#[test]
fn a_lambda_after_the_argument_sees_the_element_type_the_instance_fixed() {
    let i = Interner::new();
    assert_eq!(check(&i, "pick([1, 2], |x| -> x + 1)").unwrap(), Ty::I64);
}

#[test]
fn a_call_no_instance_matches_is_an_error_at_the_call() {
    let i = Interner::new();
    let errs = check(&i, "pick(1, |x| -> x)").unwrap_err();
    assert!(
        errs.iter()
            .any(|e| e.contains("outside the declared bound")),
        "{errs:?}"
    );
}

#[test]
fn the_choice_waits_for_the_argument_to_resolve() {
    let i = Interner::new();
    assert_eq!(
        check(&i, "let g = |c| -> pick(c, |x| -> x); g([1])").unwrap(),
        Ty::I64
    );
}
