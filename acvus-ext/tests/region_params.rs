//! An extension type's lifetime parameters are its region parameters
//! (RFC-0079 rules 2 and 6): the declaration counts them, and every type
//! the checker resolves at that name carries the count.

use acvus_ext::std_registries;
use acvus_extern::Externs;
use acvus_interpreter::AcvusRuntime;
use acvus_mir::graph::{CompilationGraph, FnKind, Function, ParsedAst, extract, infer};
use acvus_mir::ty::{PolyBuilder, PolyTy, Ty};
use acvus_utils::{Freeze, Interner, QualifiedRef};

fn externs(i: &Interner) -> Externs<AcvusRuntime> {
    Externs::combine(std_registries::<AcvusRuntime>(), i).expect("the std registries combine")
}

#[test]
fn refs_declares_one_region_parameter() {
    let i = Interner::new();
    let externs = externs(&i);
    let region_params = |name: &str| externs.types.get(QualifiedRef::root(i.intern(name))).region_params;
    assert_eq!(region_params("Refs"), 1, "Refs<'a, C, I, Rt> holds a Ref<'a, C, Shared, Rt>");
    assert_eq!(region_params("Map"), 1, "Map<'a, ..> holds a Closure<'a, ..> and an Instance<'a, ..>");
    assert_eq!(region_params("Items"), 0, "Items<T, I, Rt> owns its elements");
    assert_eq!(region_params("Vec"), 0, "Vec<T> names no lifetime");
}

#[test]
fn the_resolved_type_of_a_borrowed_source_carries_the_count() {
    let i = Interner::new();
    let Externs {
        mut functions,
        types,
        ..
    } = externs(&i);
    let entry = QualifiedRef::root(i.intern("test"));
    let source = "let xs = [1, 2, 3]; xs.as_iter()";
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse"));
    functions.push(Function {
        qref: entry,
        kind: FnKind::Local(ast),
        ty: PolyTy::Fn {
            params: vec![],
            ret: Box::new(PolyBuilder::new().fresh_ty_var()),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    });
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(types),
        bindings: acvus_mir::graph::Bindings::default(),
        entries: vec![entry],
    };
    let ext = extract::extract(&i, &graph);
    let inf = infer::infer(&i, &graph, &ext);
    let tail = inf.outcomes[&entry]
        .tail_ty()
        .unwrap_or_else(|| panic!("the entry checks: {:?}", inf.errors()));
    let Ty::UserDefined {
        id, region_params, ..
    } = tail
    else {
        panic!("`xs.as_iter()` is typed {}", tail.display(&i))
    };
    assert_eq!(i.resolve(id.name), "Refs");
    assert_eq!(*region_params, 1);
}
