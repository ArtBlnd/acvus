//! Array<T, N> at its contracts.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{LenTerm, PolyBuilder, Ty, TyTerm};
use acvus_mir_test::compile_script_ir;
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

fn local_fn(i: &Interner, name: &str, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern(name)),
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

fn return_type(i: &Interner, source: &str) -> Ty {
    let f = local_fn(i, "f", source);
    let qref = f.qref;
    let graph = CompilationGraph {
        functions: Freeze::new(vec![f]),
        contexts: Freeze::new(vec![]),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: None,
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(
        i,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(acvus_mir::ty::TypeRegistry::new()),
    );
    assert!(!inf.has_errors(), "infer errors: {:?}", inf.errors());
    match &inf.outcomes[&qref].meta().ty {
        Ty::Fn { ret, .. } => (**ret).clone(),
        other => panic!("expected Fn, got {other:?}"),
    }
}

#[test]
fn literal_types_as_array_with_its_length() {
    let i = Interner::new();
    assert_eq!(
        return_type(&i, "[1, 2, 3]"),
        Ty::Array(Box::new(Ty::I64), LenTerm::Known(3))
    );
    assert_eq!(
        return_type(&i, "[[1], [2]]"),
        Ty::Array(
            Box::new(Ty::Array(Box::new(Ty::I64), LenTerm::Known(1))),
            LenTerm::Known(2)
        )
    );
}

#[test]
fn pattern_length_is_checked_statically() {
    let i = Interner::new();
    let items = Ty::Array(Box::new(Ty::I64), LenTerm::Known(3));
    let ctx = FxHashMap::from_iter([(i.intern("items"), items)]);

    let err = compile_script_ir(&i, "if let [a, b] = @items { let x = a; }; 0", &ctx).unwrap_err();
    assert!(err.contains("array pattern needs length 2, got 3"), "{err}");

    compile_script_ir(&i, "if let [a, b, ..] = @items { let x = a; }; 0", &ctx).unwrap();
    compile_script_ir(&i, "if let [a, b, c] = @items { let x = a; }; 0", &ctx).unwrap();
}

#[test]
fn array_flows_into_list_and_iterator() {
    let i = Interner::new();
    let ctx = FxHashMap::default();
    compile_script_ir(&i, "let xs = [1, 2]; len(&xs)", &ctx).unwrap();
    compile_script_ir(
        &i,
        "let xs = [1, 2] | into_iter | collect | into_iter | collect; len(&xs)",
        &ctx,
    )
    .unwrap();
}
