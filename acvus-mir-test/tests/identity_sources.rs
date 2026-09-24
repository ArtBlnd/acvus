//! A source number names one identity for the whole compilation. A frozen
//! type that leaves one solver and enters another still names the source
//! it was frozen with, and no solver mints that number again.

use acvus_mir::graph::incremental::IncrementalGraph;
use acvus_mir::graph::{Bindings, CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef};
use acvus_mir::ty::{
    IdentityTerm, ParamTerm, Poly, PolyBuilder, PolyTy, Solver, Sources, Ty, TyTerm, TypeArg,
    TypeRegistry, lift_to_poly,
};
use acvus_mir_test::inferred_function;
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

fn iterator_registry(i: &Interner) -> TypeRegistry {
    let mut reg = TypeRegistry::new();
    reg.register(acvus_mir::ty::UserDefinedDecl {
        qref: QualifiedRef::root(i.intern("Iterator")),
        type_params: vec![acvus_mir::ty::TyVarBound::Any],
        effect_params: 1,
        identity_params: 1,
        region_params: 0,
        specializable: vec![false],
    })
    .expect("one declaration per name");
    reg
}

fn iter_poly(i: &Interner, identity: IdentityTerm<Poly>) -> PolyTy {
    TyTerm::UserDefined {
        id: QualifiedRef::root(i.intern("Iterator")),
        type_args: vec![TypeArg::uniform(TyTerm::I64)],
        effect_args: vec![acvus_mir::ty::EffectArg::uniform(
            acvus_mir::ty::Effect::PURE.into(),
        )],
        identity_args: vec![identity],
        region_params: 0,
    }
}

fn source_of(ty: &Ty) -> acvus_mir::ty::IdentityId {
    let Ty::UserDefined { identity_args, .. } = ty else {
        panic!("expected a UserDefined type, got {ty:?}");
    };
    let [IdentityTerm::Known(id)] = identity_args.as_slice() else {
        panic!("expected one known source, got {identity_args:?}");
    };
    *id
}

#[test]
fn a_source_frozen_in_one_solver_is_never_minted_by_another() {
    let i = Interner::new();
    let reg = iterator_registry(&i);
    let mut sources = Sources::new();

    let frozen_y = {
        let signatures = FxHashMap::default();
        let mut a = Solver::new(&mut sources, &reg, &signatures);
        let mut pb = PolyBuilder::new();
        let y = a.instantiate_poly(&iter_poly(&i, pb.fresh_identity_var()));
        a.freeze_ty(&y).unwrap()
    };

    let signatures = FxHashMap::default();
    let mut b = Solver::new(&mut sources, &reg, &signatures);
    let imported_y = b.instantiate_poly(&lift_to_poly(&frozen_y));
    for _ in 0..8 {
        let mut pb = PolyBuilder::new();
        let fresh = b.instantiate_poly(&iter_poly(&i, pb.fresh_identity_var()));
        assert!(
            b.unify(&imported_y, &fresh).is_err(),
            "a source minted in solver B unified with a source imported from solver A"
        );
        assert_ne!(
            source_of(&b.freeze_ty(&fresh).unwrap()),
            source_of(&frozen_y)
        );
    }
}

/// `mk()` returns a new source at every call; `same(x, y)` accepts only two
/// values of one source. A function in an earlier SCC returning `mk()`
/// crosses into a later SCC's solver as a frozen type.
#[test]
fn a_source_returned_across_sccs_stays_distinct_from_new_ones() {
    let i = Interner::new();
    let mut pb = PolyBuilder::new();
    let mk = Function {
        qref: QualifiedRef::root(i.intern("mk")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(iter_poly(&i, pb.fresh_identity_var())),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    };
    let mut pb = PolyBuilder::new();
    let shared = pb.fresh_identity_var();
    let same = Function {
        qref: QualifiedRef::root(i.intern("same")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![
                ParamTerm::<Poly>::new(i.intern("x"), iter_poly(&i, shared.clone())),
                ParamTerm::<Poly>::new(i.intern("y"), iter_poly(&i, shared)),
            ],
            ret: Box::new(TyTerm::I64),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    };
    let get = inferred_function(
        QualifiedRef::root(i.intern("get")),
        FnKind::Local(
            ParsedAst::Script(acvus_ast::parse_script(&i, "mk()").expect("parse")),
            acvus_mir::graph::Inputs::FromReads,
        ),
        vec![],
    );
    let main = inferred_function(
        QualifiedRef::root(i.intern("main")),
        FnKind::Local(
            ParsedAst::Script(
                acvus_ast::parse_script(&i, "let x = get(); let y = mk(); same(x, y)")
                    .expect("parse"),
            ),
            acvus_mir::graph::Inputs::FromReads,
        ),
        vec![],
    );
    let mut graph = IncrementalGraph::new(
        &i,
        CompilationGraph {
            functions: Freeze::new(vec![mk, same]),
            contexts: Freeze::default(),
            types: Freeze::new(iterator_registry(&i)),
            bindings: Bindings::default(),
            access: acvus_mir::graph::Access::Sync,
            entries: Vec::new(),
        },
    );
    graph.add_function(get);
    graph.add_function(main);

    let get_diags = graph.diagnostics(QualifiedRef::root(i.intern("get")));
    assert!(get_diags.is_empty(), "get: {get_diags:?}");
    let main_diags = graph.diagnostics(QualifiedRef::root(i.intern("main")));
    assert!(
        !main_diags.is_empty(),
        "same(get(), mk()) joins two sources and must be rejected"
    );
}

/// A declared context names no source. Whatever number the declaration
/// carries, the compilation mints the context's source itself, so a new
/// source from `iter` never coincides with it.
#[test]
fn a_declared_context_never_shares_a_source_with_a_new_one() {
    let i = Interner::new();
    let declared = Ty::UserDefined {
        id: QualifiedRef::root(i.intern("Items")),
        type_args: vec![TypeArg::uniform(Ty::I64)],
        effect_args: vec![],
        identity_args: vec![IdentityTerm::Known(
            <acvus_mir::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
        )],
        region_params: 0,
    };
    let ctx = rustc_hash::FxHashMap::from_iter([
        (i.intern("src"), declared),
        (
            i.intern("items"),
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        ),
    ]);
    let result = acvus_mir_test::compile_script_ir(&i, "@src = @items | into_iter; 0", &ctx);
    assert!(
        result.is_err(),
        "storing a new source into a context must be rejected: {result:?}"
    );
    assert!(
        acvus_mir_test::compile_script_ir(&i, "let a = @src; @src = a; 0", &ctx).is_ok(),
        "storing the context's own source back must be accepted"
    );
}

/// A source minted where no expression is in hand has no origin, and naming
/// one that has no origin does not invent one: a refusal over such a source
/// shows `this value` and points at no second place.
#[test]
fn a_source_no_expression_minted_has_no_origin() {
    let i = Interner::new();
    let mut sources = Sources::new();
    let unplaced = sources.next();
    let placed = sources.next();
    sources.begins_at(placed, acvus_ast::Span::new(4, 9));

    sources.named(unplaced, i.intern("a"));
    assert_eq!(sources.origin(unplaced), None);

    sources.named(placed, i.intern("b"));
    let origin = sources.origin(placed).expect("its origin was recorded");
    assert_eq!(origin.span, acvus_ast::Span::new(4, 9));
    assert_eq!(origin.name, Some(i.intern("b")));
}

/// The first expression to name a source is its origin, and the first name to
/// hold one is the name a refusal prints.
#[test]
fn an_origin_and_a_name_are_written_once() {
    let i = Interner::new();
    let mut sources = Sources::new();
    let id = sources.next();
    sources.begins_at(id, acvus_ast::Span::new(4, 9));
    sources.begins_at(id, acvus_ast::Span::new(20, 30));
    sources.named(id, i.intern("a"));
    sources.named(id, i.intern("b"));

    let origin = sources.origin(id).expect("its origin was recorded");
    assert_eq!(origin.span, acvus_ast::Span::new(4, 9));
    assert_eq!(origin.name, Some(i.intern("a")));
}
