//! A space at its contract (RFC-0033): a value is laid out by its type and
//! comes back equal; a deque's changes are its ops, replayed from the
//! nearest checkpoint; a deque nested in a deque has its own log; a head
//! that moved refuses a commit.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_ext::Deque;
use acvus_extern::{
    Borrowable, Decode, Encode, ExternType, ExternTypeDecl, Externs, Journaled, NodeHash,
    OneValue, Owned, Registry, Runtime, SpaceError, SpaceResult, UniformPayload, Var, Visit,
    extern_fn, extern_registry, kind,
};
use acvus_interpreter::{
    AcvusRuntime, Commit, InterpreterContext, Log, Mode, NodeKind, Plain, Record,
    SequentialExecutor, Space, Value,
};
use acvus_interpreter_test::*;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{IdentityId, IdentityTerm, LenTerm, ObjectTy, Ty, TypeArg};
use acvus_utils::{Interner, LocalIdOps};
use rustc_hash::FxHashMap;

fn runtime(i: &Interner) -> AcvusRuntime {
    let externs = Externs::combine(acvus_ext::std_registries(), i).expect("registries combine");
    InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor))
        .with_space(externs.space)
        .runtime_over_an_empty_page()
}

fn deque_ty(i: &Interner, elem: Ty) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(i.intern("Deque")),
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
        identity_args: vec![],
        region_params: 0,
    }
}

/// The store a `Deque` context holds: its elements are owned runtime
/// values (RFC-0048), which is what the deque's externs read back.
type ValueDeque = Deque<Owned<AcvusRuntime>>;

fn deque_of(rt: &AcvusRuntime, items: impl IntoIterator<Item = Value>) -> Value {
    let mut d = ValueDeque::default();
    for v in items {
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), v) });
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
    let space = Space::new(Plain);
    let ty = Ty::Object(ObjectTy::written(
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
    ));
    let mut value = Value::object_by_name(
        &i,
        [
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            (i.intern("name"), unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::string("acvus")) }),
            (
                i.intern("scores"),
                // SAFETY: the word was made for this holder and moved in; no other holder owns it.
                unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::array(vec![
                    Owned::from_value(acvus_extern::Holding::new(), Value::int(7)),
                    Owned::from_value(acvus_extern::Holding::new(), Value::int(-3)),
                ])) },
            ),
            (
                i.intern("tag"),
                // SAFETY: the word was made for this holder and moved in; no other holder owns it.
                unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::some(Value::bool_(true))) },
            ),
        ],
    );
    let head = space.commit(&rt, "o", &ty, &mut value).unwrap();
    assert_eq!(space.head("o"), Some(head));
    let back = space.load(&rt, "o", &ty).unwrap().expect("committed");
    let field = |name: &str| unsafe { back.field_by_name(i.intern(name)) }.expect("the field");
    assert_eq!(unsafe { field("name").as_str() }, "acvus");
    assert_eq!(
        unsafe { field("scores").as_array() }
            .iter()
            .map(|v| v.as_int())
            .collect::<Vec<_>>(),
        [7, -3]
    );
    assert!(field("tag").as_bool());
}

#[test]
fn a_deque_s_commit_is_its_ops_replayed_from_the_last_checkpoint() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(Log {
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
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(3)) });
        assert_eq!(d.pop_front().map(|v| v.as_int()), Some(1));
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        d.push_front(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(0)) });
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
    let space = Space::new(Log {
        checkpoint_every: 2,
    });
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, []);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        for n in 1..=5 {
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(n)) });
        }
    });
    space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    // state, then five ops, then the checkpoint the fifth op crosses into
    assert_eq!(space.node_count(), 7);
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(6)) })
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
    let space = Space::new(Plain);
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, [Value::int(1)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
    with_deque(&rt, &loaded, |d| {
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(2)) })
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
    let space = Space::new(Log {
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
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            inner.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(2)) })
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
    let space = Space::new(Log {
        checkpoint_every: 100,
    });
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, [Value::int(1)]);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    let mut a = space.load(&rt, "d", &ty).unwrap().unwrap();
    let mut b = space.load(&rt, "d", &ty).unwrap().unwrap();
    // SAFETY: the word was made for this holder and moved in; no other holder owns it.
    with_deque(&rt, &a, |d| d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(2)) }));
    // SAFETY: the word was made for this holder and moved in; no other holder owns it.
    with_deque(&rt, &b, |d| d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(3)) }));
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
    let space = Space::new(Log {
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
        Ty::U64,
    )
    .await;
    assert_eq!(ran.value.as_int(), 2);
    let mut written = ran
        .writes
        .into_iter()
        .find(|w| w.key == "d")
        .expect("d was written")
        .value;
    // SAFETY: `commit` edits the value in place and writes no word into it.
    space.commit(&rt, "d", &ty, unsafe { written.value_mut(acvus_extern::Holding::new()) }).unwrap();
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
            Log {
                checkpoint_every: 100,
            },
            Box::new(store),
        );
        let mut d = deque_of(&rt, [Value::int(1)]);
        space.commit(&rt, "d", &ty, &mut d).unwrap();
        let mut loaded = space.load(&rt, "d", &ty).unwrap().unwrap();
        with_deque(&rt, &loaded, |d| {
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(2)) })
        });
        space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    }
    let store = acvus_interpreter::DirStore::open(dir.path(), &i).unwrap();
    let space = Space::over(
        Log {
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
    let space = Arc::new(Space::new(Log {
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
        Ty::U64,
    );
    let mut functions = compiled.extern_executables;
    let prepare_ctx = acvus_interpreter::PrepareCtx {
        interner: &i,
        externs: &functions,
        context_names: &compiled.context_names,
        instances: &acvus_extern::NoInstances,
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
    let page = Arc::new(SpacePage::new(Arc::clone(&space), Default::default()).unwrap());
    let mut interp = Interpreter::on_page(
        shared,
        compiled.entry_qref,
        Arc::clone(&page) as Arc<dyn acvus_interpreter::RuntimeContext>,
    );
    let value = interp.execute().await;
    assert_eq!(value.as_int(), 2);
    let committed = page.commit(&interp.runtime()).unwrap();
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
    let space = Space::new(Log {
        checkpoint_every: 100,
    });
    let inner_ty = deque_ty(&i, Ty::I64);
    let obj_ty = Ty::Object(ObjectTy::written(
        [
            (i.intern("name"), Ty::String),
            (i.intern("log"), inner_ty.clone()),
        ]
        .into_iter()
        .collect(),
    ));
    let ty = deque_ty(&i, obj_ty);
    let obj = Value::object_by_name(
        &i,
        [
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            (i.intern("name"), unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::string("a")) }),
            (
                i.intern("log"),
                // SAFETY: the word was made for this holder and moved in; no other holder owns it.
                unsafe { Owned::from_value(acvus_extern::Holding::new(), deque_of(&rt, [Value::int(1)])) },
            ),
        ],
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
        let log = unsafe { obj.value_mut(acvus_extern::Holding::new()).field_by_name_mut(i.intern("log")) }.unwrap();
        with_deque(&rt, log, |inner| {
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            inner.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(2)) })
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
    let obj = outer.get(0).unwrap();
    let field = |name: &str| unsafe { obj.field_by_name(i.intern(name)) }.expect("the field");
    assert_eq!(unsafe { field("name").as_str() }, "a");
    assert_eq!(ints(&rt, field("log")), [1, 2]);
}

/// A mode a host writes: every commit's ops, and a state only every third
/// commit, counted by the mode itself.
struct EveryThirdCommit(std::sync::atomic::AtomicUsize);

impl Mode for EveryThirdCommit {
    fn record(&self, _: &Commit) -> Record {
        let n = self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        Record::Ops {
            then_state: n % 3 == 0,
        }
    }
}

/// A host's own mode decides what a commit writes, and the history is read
/// back node by node from the head through the parents.
#[test]
fn a_host_s_mode_decides_the_nodes_and_the_history_reads_back() {
    let i = Interner::new();
    let rt = runtime(&i);
    let space = Space::new(EveryThirdCommit(std::sync::atomic::AtomicUsize::new(0)));
    let ty = deque_ty(&i, Ty::I64);
    let mut d = deque_of(&rt, []);
    space.commit(&rt, "d", &ty, &mut d).unwrap();
    for n in 1..=3 {
        let mut loaded = space.load(&rt, "d", &ty).unwrap().expect("held");
        with_deque(&rt, &loaded, |d| {
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            d.push_back(unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(n)) })
        });
        space.commit(&rt, "d", &ty, &mut loaded).unwrap();
    }

    let mut kinds = Vec::new();
    let mut at = space.head("d");
    while let Some(hash) = at {
        let node = space.get(hash).unwrap();
        kinds.push(node.kind());
        at = node.parent();
    }
    kinds.reverse();
    assert_eq!(
        kinds,
        [
            NodeKind::State,
            NodeKind::Op,
            NodeKind::Op,
            NodeKind::Op,
            NodeKind::State,
        ],
        "the first state, one op per commit, and the state the third commit closes with"
    );
    let again = space.load(&rt, "d", &ty).unwrap().expect("held");
    assert_eq!(ints(&rt, &again), [1, 2, 3]);
}

/// A context type written the way the derive lets it be: the struct, the
/// `space` switch, and `Journaled`. It has one type parameter and one
/// identity parameter, and its box is keyed by its payload.
#[derive(ExternType)]
#[extern_type(name = "Tally", space)]
#[repr(transparent)]
struct Tally<T, I>(TallyState<T>, PhantomData<I>)
where
    T: Var<kind::Type>,
    I: Var<kind::Identity>;

#[derive(UniformPayload, acvus_extern::Within)]
struct TallyState<T> {
    items: Vec<T>,
    settled: usize,
    head: Option<NodeHash>,
}

/// The form the runtime holds a `Tally` at, which its space hooks are
/// registered for.
type HeldTally = Tally<Owned<AcvusRuntime>, ()>;

fn tally_element(type_args: &[Ty]) -> SpaceResult<&Ty> {
    match type_args {
        [elem] => Ok(elem),
        other => Err(SpaceError::new(format!(
            "Tally has one type argument, got {}",
            other.len()
        ))),
    }
}

/// State: `u64` count, then the items in order. Op: one pushed item.
impl<Rt> Journaled<Rt> for Tally<Owned<Rt>, ()>
where
    Rt: Runtime,
{
    fn encode_state(
        &self,
        _: &Rt,
        type_args: &[Ty],
        elem: &Encode<'_, Rt>,
        out: &mut Vec<u8>,
    ) -> SpaceResult<()> {
        let ty = tally_element(type_args)?;
        out.extend_from_slice(&(self.0.items.len() as u64).to_le_bytes());
        for item in &self.0.items {
            elem(ty, item, out)?;
        }
        Ok(())
    }

    fn decode_state(
        _: &Rt,
        type_args: &[Ty],
        elem: &Decode<'_, Rt>,
        input: &mut &[u8],
    ) -> SpaceResult<Self> {
        let ty = tally_element(type_args)?;
        let (count, rest) = input
            .split_first_chunk::<8>()
            .ok_or_else(|| SpaceError::new("Tally: truncated count"))?;
        *input = rest;
        let items = (0..u64::from_le_bytes(*count))
            .map(|_| elem(ty, input))
            .collect::<SpaceResult<Vec<_>>>()?;
        let settled = items.len();
        Ok(Tally(
            TallyState {
                items,
                settled,
                head: None,
            },
            PhantomData,
        ))
    }

    fn take_ops(
        &mut self,
        _: &Rt,
        type_args: &[Ty],
        elem: &Encode<'_, Rt>,
    ) -> SpaceResult<Vec<Vec<u8>>> {
        let ty = tally_element(type_args)?;
        let ops = self.0.items[self.0.settled..]
            .iter()
            .map(|item| {
                let mut op = Vec::new();
                elem(ty, item, &mut op).map(|()| op)
            })
            .collect::<SpaceResult<Vec<_>>>()?;
        self.0.settled = self.0.items.len();
        Ok(ops)
    }

    fn apply_op(
        &mut self,
        _: &Rt,
        type_args: &[Ty],
        elem: &Decode<'_, Rt>,
        op: &mut &[u8],
    ) -> SpaceResult<()> {
        let ty = tally_element(type_args)?;
        self.0.items.push(elem(ty, op)?);
        self.0.settled = self.0.items.len();
        Ok(())
    }

    fn children(&mut self, type_args: &[Ty], visit: &mut Visit<'_, Rt>) -> SpaceResult<()> {
        let ty = tally_element(type_args)?;
        for item in &mut self.0.items {
            visit(ty, item)?;
        }
        Ok(())
    }

    fn head(&self) -> Option<NodeHash> {
        self.0.head
    }

    fn set_head(&mut self, head: NodeHash) {
        self.0.head = Some(head);
    }
}

#[extern_fn(effect = pure)]
fn tally_push<T, I>(t: &mut Tally<T, I>, item: T)
where
    T: Var<kind::Type>,
    I: Var<kind::Identity>,
{
    t.0.items.push(item);
}

fn tally_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "tally",
        types: [Tally<_, _>],
        fns: [tally_push],
    }
}

fn registries_with_tally() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(tally_registry());
    registries
}

fn tally_runtime(i: &Interner) -> AcvusRuntime {
    let externs = Externs::combine(registries_with_tally(), i).expect("registries combine");
    InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor))
        .with_space(externs.space)
        .runtime_over_an_empty_page()
}

fn tally_ty(i: &Interner) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(i.intern("Tally")),
        type_args: vec![TypeArg::uniform(Ty::I64)],
        effect_args: vec![],
        identity_args: vec![IdentityTerm::Known(IdentityId::from_raw(0))],
        region_params: 0,
    }
}

/// A `Tally` made as a program makes one: through its own crossing.
fn tally_of(rt: &AcvusRuntime, items: impl IntoIterator<Item = i64>) -> Value {
    let t: HeldTally = Tally(
        TallyState {
            items: items
                .into_iter()
                // SAFETY: the word was made for this holder and moved in; no other holder owns it.
                .map(|n| unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(n)) })
                .collect(),
            settled: 0,
            head: None,
        },
        PhantomData,
    );
    <HeldTally as OneValue<AcvusRuntime>>::erase(t, unsafe { acvus_extern::Crossing::new(rt) })
}

fn tally_ints(rt: &AcvusRuntime, value: &Value) -> Vec<i64> {
    // SAFETY: the value is a `Tally` of Int, as its type says, read through
    // its own crossing.
    let reference = unsafe { rt.reference(value) };
    let t: &HeldTally = unsafe { <HeldTally as Borrowable<AcvusRuntime>>::deref(rt, &reference) };
    t.0.items.iter().map(|v| v.as_int()).collect()
}

async fn push_in_a_script(i: &Interner, ty: &Ty, loaded: Value, source: &str) -> Value {
    let ran = run_script_with_externs(
        i,
        source,
        [(i.intern("t"), typed(ty.clone(), loaded))]
            .into_iter()
            .collect(),
        registries_with_tally(),
        Ty::I64,
    )
    .await;
    ran.writes
        .into_iter()
        .find(|w| w.key == "t")
        .expect("t was written")
        .value
        // SAFETY: this test is the runtime, reading the committed word out
        // of the write that holds it.
        .into_value(unsafe { acvus_extern::Holding::new() })
}

/// A derived context type round-trips through a directory space: its
/// state, the ops a script records on it, a checkpoint the ops cross, and
/// a reopened store that replays from that checkpoint.
#[tokio::test]
async fn a_derived_context_commits_reloads_and_replays_across_a_checkpoint() {
    let i = Interner::new();
    let rt = tally_runtime(&i);
    let dir = tempfile::tempdir().unwrap();
    let ty = tally_ty(&i);
    let open = || {
        Space::over(
            Log {
                checkpoint_every: 2,
            },
            Box::new(acvus_interpreter::DirStore::open(dir.path(), &i).unwrap()),
        )
    };
    {
        let space = open();
        let mut t = tally_of(&rt, [1]);
        space.commit(&rt, "t", &ty, &mut t).unwrap();
        let loaded = space.load(&rt, "t", &ty).unwrap().expect("held");
        let mut written = push_in_a_script(
            &i,
            &ty,
            loaded,
            "tally_push(&mut @t, 2); tally_push(&mut @t, 3); 0",
        )
        .await;
        space.commit(&rt, "t", &ty, &mut written).unwrap();
        // state, then two ops, then the checkpoint the second op crosses into
        assert_eq!(space.node_count(), 4);
    }
    {
        let space = open();
        let loaded = space.load(&rt, "t", &ty).unwrap().expect("held");
        assert_eq!(tally_ints(&rt, &loaded), [1, 2, 3]);
        let mut written =
            push_in_a_script(&i, &ty, loaded, "tally_push(&mut @t, 4); 0").await;
        space.commit(&rt, "t", &ty, &mut written).unwrap();
        assert_eq!(
            space.node_count(),
            5,
            "one op after the checkpoint, no new state yet"
        );
    }
    let space = open();
    let mut kinds = Vec::new();
    let mut at = space.head("t");
    while let Some(hash) = at {
        let node = space.get(hash).unwrap();
        kinds.push(node.kind());
        at = node.parent();
    }
    kinds.reverse();
    assert_eq!(
        kinds,
        [
            NodeKind::State,
            NodeKind::Op,
            NodeKind::Op,
            NodeKind::State,
            NodeKind::Op,
        ]
    );
    let again = space.load(&rt, "t", &ty).unwrap().expect("held");
    assert_eq!(tally_ints(&rt, &again), [1, 2, 3, 4]);
}

/// Without the switch a derived type declares no space hooks.
#[derive(ExternType)]
#[repr(transparent)]
struct Untallied<I>(i64, PhantomData<I>)
where
    I: Var<kind::Identity>;

#[test]
fn a_derived_type_without_the_switch_has_no_space_hooks() {
    assert!(<Untallied<()> as ExternTypeDecl>::space::<AcvusRuntime>().is_none());
    assert!(<Tally<(), ()> as ExternTypeDecl>::space::<AcvusRuntime>().is_some());
}

/// A declared context of a derived type with an identity parameter names
/// its own source: another context's value is refused, its own is accepted.
#[test]
fn a_derived_context_keeps_its_identity() {
    let i = Interner::new();
    let ctx: FxHashMap<_, _> = [(i.intern("t"), tally_ty(&i)), (i.intern("u"), tally_ty(&i))]
        .into_iter()
        .collect();
    let check = |source: &str| {
        check_source(
            &i,
            acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).unwrap()),
            &ctx,
            registries_with_tally(),
            Ty::I64,
            acvus_mir::graph::optimize::Opt::Full,
            |_| {},
        )
        .map(|_| ())
        .map_err(|refusal| refusal.messages)
    };
    let refused = check("@t = @u; 0").expect_err("another context's value is refused");
    assert!(
        refused.iter().any(|m| m.contains("different sources")),
        "{refused:?}"
    );
    check("let a = @t; @t = a; 0").expect("the context's own value is accepted");
}
