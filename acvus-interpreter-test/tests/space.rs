//! A space at its contract (RFC-0033): a value is laid out by its type and
//! comes back equal; a deque's changes are its ops, replayed from the
//! nearest checkpoint; a deque nested in a deque has its own log; a head
//! that moved refuses a commit.

use std::sync::Arc;

use acvus_ext::Deque;
use acvus_extern::{Externs, Owned, Runtime};
use acvus_interpreter::{AcvusRuntime, InterpreterContext, Mode, SequentialExecutor, Space, Value};
use acvus_interpreter_test::*;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{LenTerm, Ty, TypeArg};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn runtime(i: &Interner) -> AcvusRuntime {
    let externs = Externs::combine(acvus_ext::std_registries(), i).expect("registries combine");
    InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor))
        .with_space(externs.space)
        .runtime()
}

fn deque_ty(i: &Interner, elem: Ty) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(i.intern("Deque")),
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
        identity_args: vec![],
    }
}

/// The store a `Deque` context holds: its elements are owned runtime
/// values (RFC-0048), which is what the deque's externs read back.
type ValueDeque = Deque<Owned<AcvusRuntime>>;

fn deque_of(rt: &AcvusRuntime, items: impl IntoIterator<Item = Value>) -> Value {
    let mut d = ValueDeque::default();
    for v in items {
        d.push_back(Owned::from_value(v));
    }
    // SAFETY: a `ValueDeque` erased as itself; the space's hooks read it back as that.
    unsafe { rt.erase::<ValueDeque>(d) }
}

fn ints(rt: &AcvusRuntime, value: &Value) -> Vec<i64> {
    // SAFETY: the value is a `ValueDeque` of Int, as its type says.
    let reference = unsafe { rt.reference(value) };
    let d: &ValueDeque = unsafe { rt.deref::<ValueDeque>(&reference) };
    d.iter().map(|v| v.as_int()).collect()
}

fn with_deque(rt: &AcvusRuntime, value: &Value, f: impl FnOnce(&mut ValueDeque)) {
    // SAFETY: as in `ints`; the test alone holds the value.
    let reference = unsafe { rt.reference(value) };
    f(unsafe { rt.deref_mut::<ValueDeque>(&reference) });
}

#[test]
fn a_value_of_a_language_shape_comes_back_equal() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Plain);
    let ty = Ty::Object(
        [
            (i.intern("name"), Ty::String),
            (
                i.intern("scores"),
                Ty::Array(Box::new(Ty::I64), LenTerm::Known(2)),
            ),
            (i.intern("tag"), Ty::Option(Box::new(Ty::Bool))),
        ]
        .into_iter()
        .collect(),
    );
    let mut value = Value::object(
        [
            (i.intern("name"), Owned::from_value(Value::string("acvus"))),
            (
                i.intern("scores"),
                Owned::from_value(Value::array(vec![
                    Owned::from_value(Value::int(7)),
                    Owned::from_value(Value::int(-3)),
                ])),
            ),
            (
                i.intern("tag"),
                Owned::from_value(Value::some(Value::bool_(true))),
            ),
        ]
        .into_iter()
        .collect(),
    );
    let head = space.commit(&rt, "o", &ty, &mut value).unwrap();
    assert_eq!(space.head("o"), Some(head));
    let back = space.load(&rt, "o", &ty).unwrap().expect("committed");
    let fields = unsafe { back.as_object() };
    assert_eq!(unsafe { fields[&i.intern("name")].as_str() }, "acvus");
    assert_eq!(
        unsafe { fields[&i.intern("scores")].as_array() }
            .iter()
            .map(|v| v.as_int())
            .collect::<Vec<_>>(),
        [7, -3]
    );
    assert!(fields[&i.intern("tag")].as_bool());
}

#[test]
fn a_deque_s_commit_is_its_ops_replayed_from_the_last_checkpoint() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Log {
        checkpoint_every: 100,
    });
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, [Value::int(1), Value::int(2)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    assert_eq!(
        space.node_count(),
        1,
        "a value never held is one state node"
    );

    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        d.push_back(Owned::from_value(Value::int(3)));
        assert_eq!(d.pop_front().map(|v| v.as_int()), Some(1));
        d.push_front(Owned::from_value(Value::int(0)));
    });
    space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    assert_eq!(space.node_count(), 4, "one state and three ops");

    let again = space.load(&rt, "d", &ty).unwrap().expect("held");
    assert_eq!(ints(&rt, &again), [0, 2, 3]);
}

#[test]
fn a_checkpoint_is_written_every_n_ops_and_loading_starts_there() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Log {
        checkpoint_every: 2,
    });
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, []);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        for n in 1..=5 {
            d.push_back(Owned::from_value(Value::int(n)));
        }
    });
    space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    // state, then five ops, then the checkpoint the fifth op crosses into
    assert_eq!(space.node_count(), 7);
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        d.push_back(Owned::from_value(Value::int(6)))
    });
    space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    assert_eq!(
        space.node_count(),
        8,
        "one op after the checkpoint, no new state yet"
    );
    let again = space.load(&rt, "d", &ty).unwrap().expect("held");
    assert_eq!(ints(&rt, &again), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn in_plain_mode_a_commit_is_one_state_node() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Plain);
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, [Value::int(1)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        d.push_back(Owned::from_value(Value::int(2)))
    });
    space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    assert_eq!(space.node_count(), 2);
    assert_eq!(
        ints(&rt, &space.load(&rt, "d", &ty).unwrap().unwrap()),
        [1, 2]
    );
}

#[test]
fn a_deque_nested_in_a_deque_has_its_own_log() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Log {
        checkpoint_every: 100,
    });
    let inner_ty = deque_ty(&i, Ty::I64);
    let ty = deque_ty(&i, inner_ty.clone());
    let inner_a = deque_of(&rt, [Value::int(1)]);
    let inner_b = deque_of(&rt, [Value::int(10), Value::int(20)]);
    let mut outer = deque_of(&rt, [inner_a, inner_b]);
    space.commit(&rt, "dd", &ty, &mut outer).unwrap();
    assert_eq!(
        space.node_count(),
        3,
        "two inner states and the outer state"
    );

    let mut loaded = space.load(&rt, "dd", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |outer| {
        let first = outer.get_mut(0).expect("two inner deques");
        with_deque(&rt, first, |inner| {
            inner.push_back(Owned::from_value(Value::int(2)))
        });
    });
    space.commit(&rt, "dd", &ty, &mut loaded).unwrap();
    assert_eq!(
        space.node_count(),
        5,
        "one op on the inner log and an outer state naming the new head"
    );

    let again = space.load(&rt, "dd", &ty).unwrap().expect("held");
    let reference = unsafe { rt.reference(&again) };
    let outer: &ValueDeque = unsafe { rt.deref::<ValueDeque>(&reference) };
    assert_eq!(ints(&rt, outer.get(0).unwrap()), [1, 2]);
    assert_eq!(ints(&rt, outer.get(1).unwrap()), [10, 20]);
}

#[test]
fn a_head_that_moved_refuses_the_commit() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Log {
        checkpoint_every: 100,
    });
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, [Value::int(1)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut a = space.load(&rt, "d", &ty).unwrap().unwrap();
    let mut b = space.load(&rt, "d", &ty).unwrap().unwrap();
    with_deque(&rt, &a, |d| d.push_back(Owned::from_value(Value::int(2))));
    with_deque(&rt, &b, |d| d.push_back(Owned::from_value(Value::int(3))));
    space.commit(&rt, "d", &ty, &mut a).unwrap();
    let err = space
        .commit(&rt, "d", &ty, &mut b)
        .expect_err("b was loaded at the old head");
    assert!(err.0.contains("head moved"), "{err}");
    assert_eq!(
        ints(&rt, &space.load(&rt, "d", &ty).unwrap().unwrap()),
        [1, 2]
    );
}

#[tokio::test]
async fn a_script_s_change_to_a_deque_context_is_committed_as_its_ops() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Log {
        checkpoint_every: 100,
    });
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, [Value::int(1), Value::int(2)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let loaded = space.load(&rt, "d", &ty).unwrap().unwrap();

    let ran = run_script_with_externs(
        &i,
        "push_back(&mut @d, 3); pop_front(&mut @d); len(&@d)",
        [(i.intern("d"), typed(ty.clone(), loaded))]
            .into_iter()
            .collect(),
        acvus_ext::std_registries(),
    )
    .await;
    assert_eq!(ran.value.as_int(), 2);
    let mut written = ran
        .writes
        .into_iter()
        .find(|w| w.key == "d")
        .expect("d was written")
        .value;
    space.commit(&rt, "d", &ty, &mut written).unwrap();
    assert_eq!(space.node_count(), 3, "the state and two ops");
    assert_eq!(
        ints(&rt, &space.load(&rt, "d", &ty).unwrap().unwrap()),
        [2, 3]
    );
}

#[test]
fn a_directory_store_holds_nodes_and_heads_across_openings() {
    let i = Interner::new();
    let rt = runtime(&i);
    let dir = tempfile::tempdir().unwrap();
    let ty = deque_ty(&i, Ty::I64);
    {
        let store = acvus_interpreter::DirStore::open(dir.path(), &i).unwrap();
        let space = Space::over(
            Mode::Log {
                checkpoint_every: 100,
            },
            Box::new(store),
        );
        let mut d = deque_of(&rt, [Value::int(1)]);
        space.commit(&rt, "d", &ty, &mut d).unwrap();
        let mut loaded = space.load(&rt, "d", &ty).unwrap().unwrap();
        with_deque(&rt, &loaded, |d| {
            d.push_back(Owned::from_value(Value::int(2)))
        });
        space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    }
    let store = acvus_interpreter::DirStore::open(dir.path(), &i).unwrap();
    let space = Space::over(
        Mode::Log {
            checkpoint_every: 100,
        },
        Box::new(store),
    );
    assert_eq!(space.node_count(), 2);
    let ids = space.identities().unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0].0, "d");
    assert_eq!(ids[0].1, ty);
    assert_eq!(
        ints(&rt, &space.load(&rt, "d", &ty).unwrap().unwrap()),
        [1, 2]
    );
}

#[tokio::test]
async fn a_run_over_a_space_page_fetches_from_the_space_and_commits_its_ops() {
    use acvus_interpreter::{Interpreter, InterpreterContext, SpacePage};
    let i = Interner::new();
    let space = Arc::new(Space::new(Mode::Log {
        checkpoint_every: 100,
    }));
    let ty = deque_ty(&i, Ty::I64);
    let rt = runtime(&i);
    let mut d = deque_of(&rt, [Value::int(1)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();

    let externs = Externs::combine(acvus_ext::std_registries(), &i).unwrap();
    let compiled = compile_source_with_externs(
        &i,
        acvus_mir::graph::ParsedAst::Script(
            acvus_ast::parse_script(
                &i,
                "push_back(&mut @d, 2); push_back(&mut @d, 3); pop_front(&mut @d); len(&@d)",
            )
            .unwrap(),
        ),
        &[(i.intern("d"), ty.clone())].into_iter().collect(),
        acvus_ext::std_registries(),
    );
    let mut functions = compiled.extern_executables;
    let prepare_ctx = acvus_interpreter::PrepareCtx {
        interner: &i,
        externs: &functions,
        context_names: &compiled.context_names,
    };
    let prepared: Vec<(
        acvus_mir::graph::QualifiedRef,
        acvus_interpreter::Executable,
    )> = compiled
        .modules
        .iter()
        .map(|(qref, module)| {
            let prepared = acvus_interpreter::prepare_module(module, &prepare_ctx);
            (
                *qref,
                acvus_interpreter::Executable::Module(Arc::new(prepared)),
            )
        })
        .collect();
    functions.extend(prepared);
    let shared = InterpreterContext::new(&i, functions, Arc::new(SequentialExecutor))
        .with_fn_types(compiled.fn_types)
        .with_context_names(compiled.context_names)
        .with_space(externs.space);
    let page =
        Arc::new(SpacePage::new(Arc::clone(&space), shared.runtime(), Default::default()).unwrap());
    let mut interp = Interpreter::on_page(
        shared,
        compiled.entry_qref,
        Arc::clone(&page) as Arc<dyn acvus_interpreter::RuntimeContext>,
    );
    let value = interp.execute().await;
    assert_eq!(value.as_int(), 2);
    let committed = page.commit().unwrap();
    assert_eq!(committed.len(), 1);
    assert_eq!(committed[0].0, "d");
    assert_eq!(space.node_count(), 4, "the state and three ops");
    assert_eq!(
        ints(&rt, &space.load(&rt, "d", &ty).unwrap().unwrap()),
        [2, 3]
    );
}

#[test]
fn a_deque_inside_an_object_inside_a_deque_has_its_own_log() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Log {
        checkpoint_every: 100,
    });
    let inner_ty = deque_ty(&i, Ty::I64);
    let obj_ty = Ty::Object(
        [
            (i.intern("name"), Ty::String),
            (i.intern("log"), inner_ty.clone()),
        ]
        .into_iter()
        .collect(),
    );
    let ty = deque_ty(&i, obj_ty);
    let obj = Value::object(
        [
            (i.intern("name"), Owned::from_value(Value::string("a"))),
            (
                i.intern("log"),
                Owned::from_value(deque_of(&rt, [Value::int(1)])),
            ),
        ]
        .into_iter()
        .collect(),
    );
    let mut outer = deque_of(&rt, [obj]);
    space.commit(&rt, "o", &ty, &mut outer).unwrap();
    assert_eq!(
        space.node_count(),
        2,
        "the inner deque's state and the outer's"
    );

    let mut loaded = space.load(&rt, "o", &ty).unwrap().unwrap();
    with_deque(&rt, &loaded, |outer| {
        let obj = outer.get_mut(0).unwrap();
        let log = unsafe { obj.as_object_mut() }
            .get_mut(&i.intern("log"))
            .unwrap();
        with_deque(&rt, log, |inner| {
            inner.push_back(Owned::from_value(Value::int(2)))
        });
    });
    space.commit(&rt, "o", &ty, &mut loaded).unwrap();
    assert_eq!(
        space.node_count(),
        4,
        "one inner op and one outer state naming the inner's new head"
    );

    let again = space.load(&rt, "o", &ty).unwrap().unwrap();
    let reference = unsafe { rt.reference(&again) };
    let outer: &ValueDeque = unsafe { rt.deref::<ValueDeque>(&reference) };
    let obj = unsafe { outer.get(0).unwrap().as_object() };
    assert_eq!(unsafe { obj[&i.intern("name")].as_str() }, "a");
    assert_eq!(ints(&rt, &obj[&i.intern("log")]), [1, 2]);
}
