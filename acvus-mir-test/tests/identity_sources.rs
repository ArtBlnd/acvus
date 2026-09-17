//! A source number names one identity for the whole compilation. A frozen
//! type that leaves one solver and enters another still names the source
//! it was frozen with, and no solver mints that number again.

use acvus_mir::graph::incremental::IncrementalGraph;
use acvus_mir::graph::{FnKind, Function, ParsedAst, QualifiedRef};
use acvus_mir::ty::{
    IdentityTerm, ParamTerm, Poly, PolyBuilder, PolyTy, Solver, Sources, Ty, TyTerm, TypeArg,
    TypeRegistry, lift_to_poly,
};
use acvus_mir_test::inferred_function;
use acvus_utils::Interner;

fn iterator_registry(i: &Interner) -> TypeRegistry {
    let mut reg = TypeRegistry::new();
    reg.register(acvus_mir::ty::UserDefinedDecl {
        qref: QualifiedRef::root(i.intern("Iterator")),
        type_params: vec![acvus_mir::ty::TyVarBound::Any],
        effect_params: 1,
        identity_params: 1,
        specializable: vec![false],
    });
    reg
}

fn iter_poly(i: &Interner, identity: IdentityTerm<Poly>) -> PolyTy {
    TyTerm::UserDefined {
        id: QualifiedRef::root(i.intern("Iterator")),
        type_args: vec![TypeArg::uniform(TyTerm::I64)],
        effect_args: vec![acvus_mir::ty::Effect::PURE.into()],
        identity_args: vec![identity],
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
        let mut a = Solver::new(&mut sources, &reg);
        let mut pb = PolyBuilder::new();
        let y = a.instantiate_poly(&iter_poly(&i, pb.fresh_identity_var()));
        a.freeze_ty(&y).unwrap()
    };

    let mut b = Solver::new(&mut sources, &reg);
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
    let mut graph = IncrementalGraph::with_type_registry(&i, iterator_registry(&i));

    let mut pb = PolyBuilder::new();
    let mk = Function {
        qref: QualifiedRef::root(i.intern("mk")),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
        },
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(iter_poly(&i, pb.fresh_identity_var())),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
        },
    };
    let mut pb = PolyBuilder::new();
    let shared = pb.fresh_identity_var();
    let same = Function {
        qref: QualifiedRef::root(i.intern("same")),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
        },
        ty: TyTerm::Fn {
            params: vec![
                ParamTerm::<Poly>::new(i.intern("x"), iter_poly(&i, shared.clone())),
                ParamTerm::<Poly>::new(i.intern("y"), iter_poly(&i, shared)),
            ],
            ret: Box::new(TyTerm::I64),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
        },
    };
    let get = inferred_function(
        QualifiedRef::root(i.intern("get")),
        FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(&i, "mk()").expect("parse"),
        )),
        vec![],
    );
    let main = inferred_function(
        QualifiedRef::root(i.intern("main")),
        FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(&i, "let x = get(); let y = mk(); same(x, y)").expect("parse"),
        )),
        vec![],
    );
    graph.add_function(mk);
    graph.add_function(same);
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
        id: QualifiedRef::root(i.intern("Iterator")),
        type_args: vec![TypeArg::uniform(Ty::I64)],
        effect_args: vec![acvus_mir::ty::Effect::PURE.into()],
        identity_args: vec![IdentityTerm::Known(
            <acvus_mir::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
        )],
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
