//! A space at its contract (RFC-0033): a value is laid out by its type and
//! comes back equal; a deque's changes are its ops, replayed from the
//! nearest checkpoint; a deque nested in a deque has its own log; a head
//! that moved refuses a commit.

use std::sync::Arc;

use acvus_ext::Deque;
use acvus_extern::{Externs, Runtime};
use acvus_interpreter::{AcvusRuntime, InterpreterContext, Mode, SequentialExecutor, Space, Value};
use acvus_interpreter_test::*;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{LenTerm, Ty};
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
        type_args: vec![elem],
        effect_args: vec![],
        identity_args: vec![],
    }
}

fn deque_of(rt: &AcvusRuntime, items: impl IntoIterator<Item = Value>) -> Value {
    let mut d = Deque::<Value>::default();
    for v in items {
        d.push_back(v);
    }
    // SAFETY: a `Deque<Value>` erased as itself; the space's hooks read it back as that.
    unsafe { rt.erase::<Deque<Value>>(d) }
}

fn ints(rt: &AcvusRuntime, value: &Value) -> Vec<i64> {
    // SAFETY: the value is a `Deque<Value>` of Int, as its type says.
    let reference = unsafe { rt.reference(value) };
    let d: &Deque<Value> = unsafe { rt.deref::<Deque<Value>>(&reference) };
    d.iter().map(|v| v.as_int()).collect()
}

fn with_deque(rt: &AcvusRuntime, value: &Value, f: impl FnOnce(&mut Deque<Value>)) {
    // SAFETY: as in `ints`; the test alone holds the value.
    let reference = unsafe { rt.reference(value) };
    f(unsafe { rt.deref_mut::<Deque<Value>>(&reference) });
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
                Ty::Array(Box::new(Ty::Int), LenTerm::Known(2)),
            ),
            (i.intern("tag"), Ty::Option(Box::new(Ty::Bool))),
        ]
        .into_iter()
        .collect(),
    );
    let mut value = Value::object(
        [
            (i.intern("name"), Value::string("acvus")),
            (
                i.intern("scores"),
                Value::array(vec![Value::int(7), Value::int(-3)]),
            ),
            (i.intern("tag"), Value::option(Some(Value::bool_(true)))),
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
    assert!(
        unsafe { fields[&i.intern("tag")].as_option() }
            .as_ref()
            .unwrap()
            .as_bool()
    );
}

#[test]
fn a_deque_s_commit_is_its_ops_replayed_from_the_last_checkpoint() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Mode::Log {
        checkpoint_every: 100,
    });
    let ty = deque_ty(&i, Ty::Int);
    let mut d = deque_of(&rt, [Value::int(1), Value::int(2)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    assert_eq!(
        space.node_count(),
        1,
        "a value never held is one state node"
    );

    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        d.push_back(Value::int(3));
        assert_eq!(d.pop_front().map(|v| v.as_int()), Some(1));
        d.push_front(Value::int(0));
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
    let ty = deque_ty(&i, Ty::Int);
    let mut d = deque_of(&rt, []);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        for n in 1..=5 {
            d.push_back(Value::int(n));
        }
    });
    space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    // state, then five ops, then the checkpoint the fifth op crosses into
    assert_eq!(space.node_count(), 7);
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| d.push_back(Value::int(6)));
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
    let ty = deque_ty(&i, Ty::Int);
    let mut d = deque_of(&rt, [Value::int(1)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| d.push_back(Value::int(2)));
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
    let inner_ty = deque_ty(&i, Ty::Int);
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
        with_deque(&rt, first, |inner| inner.push_back(Value::int(2)));
    });
    space.commit(&rt, "dd", &ty, &mut loaded).unwrap();
    assert_eq!(
        space.node_count(),
        5,
        "one op on the inner log and an outer state naming the new head"
    );

    let again = space.load(&rt, "dd", &ty).unwrap().expect("held");
    let reference = unsafe { rt.reference(&again) };
    let outer: &Deque<Value> = unsafe { rt.deref::<Deque<Value>>(&reference) };
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
    let ty = deque_ty(&i, Ty::Int);
    let mut d = deque_of(&rt, [Value::int(1)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut a = space.load(&rt, "d", &ty).unwrap().unwrap();
    let mut b = space.load(&rt, "d", &ty).unwrap().unwrap();
    with_deque(&rt, &a, |d| d.push_back(Value::int(2)));
    with_deque(&rt, &b, |d| d.push_back(Value::int(3)));
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
    let ty = deque_ty(&i, Ty::Int);
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
