//! A slice is the one thing the machine indexes (RFC-0047).
//!
//! Nothing lowers `a[i]` yet, so every body here is built by hand, as
//! `prepare`'s and `code_motion`'s own tests build theirs.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{Externs, Owned};
use acvus_interpreter::{
    AcvusRuntime, Executable, InMemoryContext, Interpreter, InterpreterContext, PrepareCtx, Value,
    prepare_module,
};
use acvus_mir::ir::{
    DebugInfo, ExternInstance, IndexBound, IndexMode, Inst, InstKind, MirBody, MirModule,
    RefTarget, ValueId,
};
use acvus_mir::ty::{IntTy, LenTerm, Mutability, Task, Ty, TypeArg};
use acvus_mir::validate::{ValidationErrorKind, validate};
use acvus_utils::{Interner, LocalFactory, LocalIdOps, QualifiedRef};
use rustc_hash::FxHashMap;

// -- A body built by hand, and the run of it -------------------------

const CONTAINER: &str = "c";

struct Body {
    insts: Vec<Inst>,
    val_types: FxHashMap<ValueId, Ty>,
}

fn v(n: usize) -> ValueId {
    ValueId::from_raw(n)
}

impl Body {
    fn new() -> Self {
        Self {
            insts: Vec::new(),
            val_types: FxHashMap::default(),
        }
    }

    fn typed(mut self, id: ValueId, ty: Ty) -> Self {
        self.val_types.insert(id, ty);
        self
    }

    fn inst(mut self, kind: InstKind) -> Self {
        self.insts.push(Inst {
            span: acvus_ast::Span::ZERO,
            kind,
        });
        self
    }

    /// `Ref` mode leaves a reference into the container; the word behind
    /// it is read here, inside the frame that still holds the container.
    fn read_through(self, mode: IndexMode, reference: ValueId, word: ValueId) -> Self {
        match mode {
            IndexMode::Copy => self,
            IndexMode::Ref => self.inst(InstKind::Take {
                dst: word,
                target: RefTarget::Through(reference),
                path: Vec::new(),
                taken_out: false,
            }),
        }
    }

    fn mir(self) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..32 {
            factory.next();
        }
        MirBody {
            demoted_diamonds: Default::default(),
            insts: self.insts,
            val_types: self.val_types,
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            task: Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        }
    }
}

/// The extern this container's `as_slice` resolves to: one declaration per
/// namespace, so the instance the checker would settle on is its only one.
fn as_slice_of(interner: &Interner, namespace: &str, name: &str) -> ExternInstance {
    ExternInstance {
        id: QualifiedRef::qualified(interner.intern(namespace), interner.intern(name)),
        instance: 0,
    }
}

async fn run_with(
    interner: &Interner,
    body: MirBody,
    page: HashMap<String, (Ty, Owned<AcvusRuntime>)>,
    ret: Ty,
) -> Value {
    let combined = Externs::combine(acvus_ext::std_registries::<AcvusRuntime>(), interner)
        .expect("the standard registries combine");
    let mut functions: FxHashMap<QualifiedRef, Executable> = combined
        .handlers
        .into_iter()
        .map(|(qref, handlers)| (qref, Executable::Extern(handlers)))
        .collect();

    let entry = QualifiedRef::root(interner.intern("test"));
    let context_names: FxHashMap<QualifiedRef, acvus_utils::Astr> = [CONTAINER, REPLACEMENT]
        .into_iter()
        .map(|name| QualifiedRef::root(interner.intern(name)))
        .map(|qref| (qref, qref.name))
        .collect();
    let module = MirModule {
        declared_params: body.params.len(),
        main: body,
        closures: FxHashMap::default(),
        ret,
        flows: acvus_mir::ty::Flows::Every,
        fetched_first: Vec::new(),
    };
    let prepared = prepare_module(
        &module,
        &PrepareCtx {
            interner,
            externs: &functions,
            context_names: &context_names,
            instances: &acvus_extern::NoInstances,
        },
    );
    functions.insert(entry, Executable::Module(Arc::new(prepared)));

    let shared = InterpreterContext::new(
        interner,
        functions,
        Arc::new(acvus_interpreter::SequentialExecutor),
    )
    .with_context_names(context_names);
    let mut interpreter = Interpreter::new(shared, entry, InMemoryContext::new(page));
    interpreter
        .execute()
        .await
        .expect("the page holds every context the run fetches first")
}

async fn run(interner: &Interner, body: MirBody, ret: Ty) -> Value {
    run_with(interner, body, HashMap::new(), ret).await
}

// -- The containers -------------------------------------------------

/// `Array<i64, 2>` built in the body itself: `MakeArray` leaves the
/// runtime's own `Arr<Value, ()>` in a register, which is the storage the
/// `Ref` names.
fn array_body(interner: &Interner, mode: IndexMode, index: u64, dst_ty: Ty) -> MirBody {
    let elem = Ty::Int(IntTy::I64);
    let array = Ty::Array(Box::new(elem.clone()), LenTerm::Known(2));
    Body::new()
        .typed(v(0), elem.clone())
        .typed(v(1), elem.clone())
        .typed(v(2), array.clone())
        .typed(v(3), reference(Mutability::Shared, array))
        .typed(v(4), slice(Mutability::Shared, elem))
        .typed(v(5), Ty::U64)
        .typed(v(6), dst_ty)
        .typed(v(7), Ty::I64)
        .inst(InstKind::Const {
            dst: v(0),
            value: acvus_ast::Literal::Int(10),
        })
        .inst(InstKind::Const {
            dst: v(1),
            value: acvus_ast::Literal::Int(20),
        })
        .inst(InstKind::MakeArray {
            dst: v(2),
            elements: vec![v(0), v(1)],
        })
        .inst(InstKind::Ref {
            dst: v(3),
            target: RefTarget::Var(v(2)),
            path: Vec::new(),
            mutability: Mutability::Shared,
        })
        .inst(InstKind::AsSlice {
            dst: v(4),
            container: v(3),
            mutability: Mutability::Shared,
            instance: as_slice_of(interner, "array", "as_slice"),
        })
        .inst(InstKind::Const {
            dst: v(5),
            value: acvus_ast::Literal::Int(index.into()),
        })
        .inst(InstKind::Index {
            dst: v(6),
            slice: v(4),
            index: v(5),
            mode,
            bound: IndexBound::Checked,
        })
        .read_through(mode, v(6), v(7))
        .inst(InstKind::Drop { src: v(4) })
        .inst(InstKind::Return {
            value: element_of(mode, v(6), v(7)),
            order: None,
        })
        .mir()
}

/// `Vec<i64>` fetched whole out of the page: its storage is the runtime's
/// `Vec<Owned<AcvusRuntime>>`, which is what `vec::as_slice` reads
/// (RFC-0039).
fn vec_body(interner: &Interner, mode: IndexMode, index: u64, dst_ty: Ty) -> MirBody {
    let elem = Ty::Int(IntTy::I64);
    let vec_ty = acvus_extern::vec_ty(interner, elem.clone());
    Body::new()
        .typed(v(0), vec_ty.clone())
        .typed(v(1), reference(Mutability::Shared, vec_ty))
        .typed(v(2), slice(Mutability::Shared, elem))
        .typed(v(3), Ty::U64)
        .typed(v(4), dst_ty)
        .typed(v(5), Ty::I64)
        .inst(InstKind::Fetch {
            dst: v(0),
            context: QualifiedRef::root(interner.intern(CONTAINER)),
        })
        .inst(InstKind::Ref {
            dst: v(1),
            target: RefTarget::Var(v(0)),
            path: Vec::new(),
            mutability: Mutability::Shared,
        })
        .inst(InstKind::AsSlice {
            dst: v(2),
            container: v(1),
            mutability: Mutability::Shared,
            instance: as_slice_of(interner, "vec", "as_slice"),
        })
        .inst(InstKind::Const {
            dst: v(3),
            value: acvus_ast::Literal::Int(index.into()),
        })
        .inst(InstKind::Index {
            dst: v(4),
            slice: v(2),
            index: v(3),
            mode,
            bound: IndexBound::Checked,
        })
        .read_through(mode, v(4), v(5))
        .inst(InstKind::Drop { src: v(2) })
        .inst(InstKind::Return {
            value: element_of(mode, v(4), v(5)),
            order: None,
        })
        .mir()
}

fn element_of(mode: IndexMode, indexed: ValueId, word: ValueId) -> ValueId {
    match mode {
        IndexMode::Copy => indexed,
        IndexMode::Ref => word,
    }
}

fn reference(mutability: Mutability, target: Ty) -> Ty {
    Ty::Ref(mutability, Box::new(TypeArg::uniform(target)))
}

fn slice(mutability: Mutability, element: Ty) -> Ty {
    reference(mutability, Ty::Slice(Box::new(element)))
}

/// A `Vec<i64>` as the runtime holds it: one box over a
/// `Vec<Owned<AcvusRuntime>>`.
fn stored_vec(items: &[i64]) -> Value {
    let values: Vec<Owned<AcvusRuntime>> = items
        .iter()
        .copied()
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        .map(|n| unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::int(n)) })
        .collect();
    // SAFETY: read back only as this same `Vec<Owned<AcvusRuntime>>`, which
    // is what `vec::as_slice`'s glue derefs.
    unsafe { Value::erase(values) }
}

fn page_with(interner: &Interner, items: &[i64]) -> HashMap<String, (Ty, Owned<AcvusRuntime>)> {
    let ty = acvus_extern::vec_ty(interner, Ty::Int(IntTy::I64));
    // SAFETY: the word was made for this holder and moved in; no other holder owns it.
    let held = unsafe { Owned::from_value(acvus_extern::Holding::new(), stored_vec(items)) };
    [(CONTAINER.to_string(), (ty, held))].into_iter().collect()
}

// -- Reading an element ----------------------------------------------

#[tokio::test]
async fn an_array_element_is_read_by_copy() {
    let interner = Interner::new();
    let got = run(
        &interner,
        array_body(&interner, IndexMode::Copy, 1, Ty::I64),
        Ty::I64,
    )
    .await;
    assert_eq!(got.as_int(), 20);
}

#[tokio::test]
async fn an_array_element_is_read_through_a_reference() {
    let interner = Interner::new();
    let dst = reference(Mutability::Shared, Ty::I64);
    let got = run(
        &interner,
        array_body(&interner, IndexMode::Ref, 0, dst.clone()),
        dst,
    )
    .await;
    assert_eq!(got.as_int(), 10);
}

#[tokio::test]
async fn a_vec_element_is_read_by_copy() {
    let interner = Interner::new();
    let got = run_with(
        &interner,
        vec_body(&interner, IndexMode::Copy, 2, Ty::I64),
        page_with(&interner, &[7, 8, 9]),
        Ty::I64,
    )
    .await;
    assert_eq!(got.as_int(), 9);
}

#[tokio::test]
async fn a_vec_element_is_read_through_a_reference() {
    let interner = Interner::new();
    let dst = reference(Mutability::Shared, Ty::I64);
    let got = run_with(
        &interner,
        vec_body(&interner, IndexMode::Ref, 1, dst.clone()),
        page_with(&interner, &[7, 8, 9]),
        dst,
    )
    .await;
    assert_eq!(got.as_int(), 8);
}

#[tokio::test]
#[should_panic(expected = "index out of bounds: the len is 2 but the index is 2")]
async fn an_index_at_the_length_panics_with_rusts_text() {
    let interner = Interner::new();
    run(
        &interner,
        array_body(&interner, IndexMode::Copy, 2, Ty::I64),
        Ty::I64,
    )
    .await;
}

// -- Writing an element ----------------------------------------------

/// An element whose drop is counted: `IndexSet` assigns, so the element it
/// replaces is released there and not at the end of the run (RFC-0047 rule 4).
struct Counted;

static ELEMENTS_DROPPED: AtomicUsize = AtomicUsize::new(0);

impl Drop for Counted {
    fn drop(&mut self) {
        ELEMENTS_DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

fn counted_ty(interner: &Interner) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(interner.intern("Counted")),
        type_args: Vec::new(),
        effect_args: Vec::new(),
        identity_args: Vec::new(),
        region_params: 0,
    }
}

fn counted_page(interner: &Interner, len: usize) -> HashMap<String, (Ty, Owned<AcvusRuntime>)> {
    // SAFETY: each element is read back only as this same `Counted`, and
    // the buffer only as the `Vec<Owned<AcvusRuntime>>` `vec::as_slice_mut`
    // derefs.
    let values: Vec<Owned<AcvusRuntime>> = (0..len)
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        .map(|_| unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::erase(Counted)) })
        .collect();
    let stored = unsafe { Value::erase(values) };
    let ty = acvus_extern::vec_ty(interner, counted_ty(interner));
    // SAFETY: the word was made for this holder and moved in; no other holder owns it.
    let held = unsafe { Owned::from_value(acvus_extern::Holding::new(), stored) };
    [(CONTAINER.to_string(), (ty, held))].into_iter().collect()
}

/// `c[0] = Counted` over a `&mut [Counted]` taken from the page.
fn index_set_body(interner: &Interner) -> MirBody {
    let element = counted_ty(interner);
    let vec_ty = acvus_extern::vec_ty(interner, element.clone());
    Body::new()
        .typed(v(0), vec_ty.clone())
        .typed(v(1), reference(Mutability::Mut, vec_ty))
        .typed(v(2), slice(Mutability::Mut, element.clone()))
        .typed(v(3), Ty::U64)
        .typed(v(4), element)
        .typed(v(5), Ty::I64)
        .inst(InstKind::Fetch {
            dst: v(0),
            context: QualifiedRef::root(interner.intern(CONTAINER)),
        })
        .inst(InstKind::Ref {
            dst: v(1),
            target: RefTarget::Var(v(0)),
            path: Vec::new(),
            mutability: Mutability::Mut,
        })
        .inst(InstKind::AsSlice {
            dst: v(2),
            container: v(1),
            mutability: Mutability::Mut,
            instance: as_slice_of(interner, "vec", "as_slice_mut"),
        })
        .inst(InstKind::Const {
            dst: v(3),
            value: acvus_ast::Literal::Int(0),
        })
        .inst(InstKind::Fetch {
            dst: v(4),
            context: QualifiedRef::root(interner.intern(REPLACEMENT)),
        })
        .inst(InstKind::IndexSet {
            slice: v(2),
            index: v(3),
            value: v(4),
            bound: IndexBound::Checked,
        })
        .inst(InstKind::Drop { src: v(2) })
        .inst(InstKind::Const {
            dst: v(5),
            value: acvus_ast::Literal::Int(0),
        })
        .inst(InstKind::Return {
            value: v(5),
            order: None,
        })
        .mir()
}

const REPLACEMENT: &str = "r";

#[tokio::test]
async fn index_set_drops_the_element_it_replaces() {
    let interner = Interner::new();
    let mut page = counted_page(&interner, 3);
    // SAFETY: as `counted_page`.
    page.insert(
        REPLACEMENT.to_string(),
        (
            counted_ty(&interner),
            // SAFETY: the word was made for this holder and moved in; no other holder owns it.
            unsafe { Owned::from_value(acvus_extern::Holding::new(), Value::erase(Counted)) },
        ),
    );

    ELEMENTS_DROPPED.store(0, Ordering::Relaxed);
    let got = run_with(&interner, index_set_body(&interner), page, Ty::I64).await;
    assert_eq!(got.as_int(), 0);
    assert_eq!(
        ELEMENTS_DROPPED.load(Ordering::Relaxed),
        4,
        "the replaced element and the three the page still owns"
    );
}

// -- Which containers lend a slice -----------------------------------

#[tokio::test]
async fn a_deque_lends_no_slice() {
    let interner = Interner::new();
    let combined = Externs::combine(acvus_ext::std_registries::<AcvusRuntime>(), &interner)
        .expect("the standard registries combine");
    let lends = |namespace: &str, name: &str| {
        combined.handlers.contains_key(&QualifiedRef::qualified(
            interner.intern(namespace),
            interner.intern(name),
        ))
    };
    assert!(lends("vec", "as_slice"));
    assert!(lends("vec", "as_slice_mut"));
    assert!(lends("array", "as_slice"));
    assert!(lends("array", "as_slice_mut"));
    assert!(!lends("deque", "as_slice"));
    assert!(!lends("deque", "as_slice_mut"));
}

// -- What the checker refuses ----------------------------------------

/// A body that takes a slice of `@c` and reads element `v(3)` of it.
fn indexing_body(interner: &Interner, element: Ty, mode: IndexMode, dst_ty: Ty) -> MirBody {
    let vec_ty = acvus_extern::vec_ty(interner, element.clone());
    Body::new()
        .typed(v(0), vec_ty.clone())
        .typed(v(1), reference(Mutability::Shared, vec_ty))
        .typed(v(2), slice(Mutability::Shared, element))
        .typed(v(3), Ty::U64)
        .typed(v(4), dst_ty)
        .inst(InstKind::Fetch {
            dst: v(0),
            context: QualifiedRef::root(interner.intern(CONTAINER)),
        })
        .inst(InstKind::Ref {
            dst: v(1),
            target: RefTarget::Var(v(0)),
            path: Vec::new(),
            mutability: Mutability::Shared,
        })
        .inst(InstKind::AsSlice {
            dst: v(2),
            container: v(1),
            mutability: Mutability::Shared,
            instance: as_slice_of(interner, "vec", "as_slice"),
        })
        .inst(InstKind::Const {
            dst: v(3),
            value: acvus_ast::Literal::Int(0),
        })
        .inst(InstKind::Index {
            dst: v(4),
            slice: v(2),
            index: v(3),
            mode,
            bound: IndexBound::Checked,
        })
        .inst(InstKind::Return {
            value: v(4),
            order: None,
        })
        .mir()
}

fn refusals(body: MirBody, ret: Ty) -> Vec<ValidationErrorKind> {
    let module = MirModule {
        declared_params: body.params.len(),
        main: body,
        closures: FxHashMap::default(),
        ret,
        flows: acvus_mir::ty::Flows::Every,
        fetched_first: Vec::new(),
    };
    validate(&module).into_iter().map(|e| e.kind).collect()
}

fn names(kinds: &[ValidationErrorKind]) -> Vec<String> {
    kinds
        .iter()
        .map(|kind| match kind {
            ValidationErrorKind::TypeMismatch {
                inst_name, desc, ..
            } => format!("TypeMismatch({inst_name}.{desc})"),
            ValidationErrorKind::InvalidConstructor { inst_name, .. } => {
                format!("InvalidConstructor({inst_name})")
            }
            ValidationErrorKind::BorrowConflict { .. } => "BorrowConflict".to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn a_well_typed_index_passes() {
    let interner = Interner::new();
    let body = indexing_body(&interner, Ty::I64, IndexMode::Copy, Ty::I64);
    assert_eq!(names(&refusals(body, Ty::I64)), Vec::<String>::new());
}

#[tokio::test]
async fn a_copy_mode_index_whose_element_moves_is_refused() {
    let interner = Interner::new();
    let body = indexing_body(&interner, Ty::String, IndexMode::Copy, Ty::String);
    assert_eq!(
        names(&refusals(body, Ty::String)),
        ["InvalidConstructor(Index)"]
    );
}

#[tokio::test]
async fn a_ref_mode_index_whose_dst_is_not_a_reference_is_refused() {
    let interner = Interner::new();
    let body = indexing_body(&interner, Ty::I64, IndexMode::Ref, Ty::I64);
    assert_eq!(names(&refusals(body, Ty::I64)), ["TypeMismatch(Index.dst)"]);
}

/// A slice holds its container's loan, so taking the container exclusively
/// while the slice is live is the conflict a live `Ref` raises (RFC-0018).
#[tokio::test]
async fn a_live_slice_refuses_an_exclusive_take_of_its_container() {
    let interner = Interner::new();
    let vec_ty = acvus_extern::vec_ty(&interner, Ty::I64);
    let body = Body::new()
        .typed(v(0), vec_ty.clone())
        .typed(v(1), reference(Mutability::Shared, vec_ty.clone()))
        .typed(v(2), slice(Mutability::Shared, Ty::I64))
        .typed(v(3), reference(Mutability::Mut, vec_ty))
        .typed(v(4), Ty::U64)
        .typed(v(5), Ty::I64)
        .inst(InstKind::Fetch {
            dst: v(0),
            context: QualifiedRef::root(interner.intern(CONTAINER)),
        })
        .inst(InstKind::Ref {
            dst: v(1),
            target: RefTarget::Var(v(0)),
            path: Vec::new(),
            mutability: Mutability::Shared,
        })
        .inst(InstKind::AsSlice {
            dst: v(2),
            container: v(1),
            mutability: Mutability::Shared,
            instance: as_slice_of(&interner, "vec", "as_slice"),
        })
        .inst(InstKind::Ref {
            dst: v(3),
            target: RefTarget::Var(v(0)),
            path: Vec::new(),
            mutability: Mutability::Mut,
        })
        .inst(InstKind::Const {
            dst: v(4),
            value: acvus_ast::Literal::Int(0),
        })
        .inst(InstKind::Index {
            dst: v(5),
            slice: v(2),
            index: v(4),
            mode: IndexMode::Copy,
            bound: IndexBound::Checked,
        })
        .inst(InstKind::Return {
            value: v(5),
            order: None,
        })
        .mir();
    assert_eq!(names(&refusals(body, Ty::I64)), ["BorrowConflict"]);
}
