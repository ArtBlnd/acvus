//! RFC-0050 rule 4's first judgment: how many aggregates the run keeps off the
//! heap.
//!
//! The count is per iteration, taken as the difference between two loop
//! lengths, so whatever the compile and the frame allocate once does not enter
//! it. Obligation across artifacts: the shape these scripts prepare to is
//! `run_shape.rs`'s subject, and a run form that stopped taking a run would
//! pass every assertion there and fail the numbers here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{Erased, ExternType, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor};
use acvus_interpreter_test::{
    Context, Helper, check_graph, execute_compiled, int_context, run_script, split_context,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{ParamTerm, Poly, Ty, lift_to_poly};
use acvus_utils::Interner;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static RELEASES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every call forwards to `System`, which is a sound allocator, and the
// counter is not read inside the allocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        RELEASES.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTER: Counting = Counting;

/// The inner loop is what makes `e` a storage whose address is taken, which is
/// the web `run_shape.rs` pins: without it `optimize::sroa` scalarizes the
/// aggregate in the MIR and no aggregate reaches the machine at all.
const RUN_RESIDENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let e = if i % 3 == 0 { E::A(i) } else { if i % 3 == 1 { E::B(i + 1) } else { E::C(i + 2) } }; \
let j = 0; while j < 1 { match e { E::A(v) => { acc = acc + v; }, \
E::B(v) => { acc = acc + v; }, E::C(v) => { acc = acc + v; } }; j = j + 1; } \
i = i + 1; } acc";

/// `RUN_RESIDENT`'s shape at the one variant type whose tags the language names
/// rather than a declaration.
const RESULT_RUN_RESIDENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let r = if i % 2 == 0 { Ok(i) } else { Err(i + 1) }; \
let j = 0; while j < 1 { match r { Ok(v) => { acc = acc + v; }, \
Err(v) => { acc = acc + v; } }; j = j + 1; } \
i = i + 1; } acc";

/// The same three-armed construction with one member crossing into an array,
/// which `prepare::runs::Sites` refuses, so every iteration realizes a variant
/// on the heap (RFC-0050 rule 4) and reads its tag and payload there.
const HEAP_RESIDENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let e = if i % 3 == 0 { E::A(i) } else { if i % 3 == 1 { E::B(i + 1) } else { E::C(i + 2) } }; \
let v = [e]; let m = len(&v); let z = m - m; \
match &v[z] { E::A(w) => { acc = acc + *w; }, E::B(w) => { acc = acc + *w; }, \
E::C(w) => { acc = acc + *w; } }; i = i + 1; } acc";

async fn allocations(source: &str, n: i64) -> usize {
    let interner = Interner::new();
    let context: Context = int_context(&interner, "n", n);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let answer = run_script(&interner, source, context, Ty::I64).await;
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    std::hint::black_box(answer);
    after - before
}

/// `ALLOCATIONS` counts the whole process, so two measurements running at once
/// each read the other's allocations. This is what keeps them apart, and
/// without it a second test in this file silently moves the first's number.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

async fn per_iteration(source: &str) -> f64 {
    let measuring = ONE_AT_A_TIME.lock().expect("no measurement panicked");
    let (few, many) = (10_000i64, 110_000i64);
    let low = allocations(source, few).await;
    let high = allocations(source, many).await;
    drop(measuring);
    (high as f64 - low as f64) / (many - few) as f64
}

#[tokio::test]
async fn a_run_resident_variant_allocates_nothing_per_iteration() {
    let run = per_iteration(RUN_RESIDENT).await;
    println!("allocations per iteration: {run:.3}");
    assert!(
        run < 0.01,
        "a run-resident variant allocates {run} per iteration"
    );
}

#[tokio::test]
async fn a_run_resident_result_allocates_nothing_per_iteration() {
    let run = per_iteration(RESULT_RUN_RESIDENT).await;
    println!("allocations per iteration: {run:.3}");
    assert!(
        run < 0.01,
        "a run-resident Result allocates {run} per iteration"
    );
}

/// Before RFC-0050 rule 4's flat variant, a variant with a payload was two
/// allocations: a header holding the tag and a box holding the payload. This
/// script read 4.000 at `bc5841db` and reads 3.000 here; the two it keeps are
/// the array and the slice the match indexes through, which no build changes.
#[tokio::test]
async fn a_heap_resident_variant_is_one_allocation() {
    let heap = per_iteration(HEAP_RESIDENT).await;
    println!("allocations per iteration: {heap:.3}");
    assert!(
        (heap - 3.0).abs() < 0.05,
        "the heap variant script allocates {heap} per iteration"
    );
}

/// A `String` payload of a run-resident variant, read by a pattern in a loop
/// and once more after it. At `Opt::None` no pass scalarizes `e`, so every
/// read is a take of a register of the run.
const PAYLOAD_READ_BY_PATTERNS: &str = "\
let acc = 0; let i = 0; while i < @n { \
let e = if i % 2 == 0 { E::A(i.to_string()) } else { E::B((i + 1).to_string()) }; \
let j = 0; while j < 1 { match e { E::A(v) => { acc = acc + v.len(); }, \
E::B(v) => { acc = acc + v.len(); } }; j = j + 1; } \
let c = match e { E::A(t) => t, E::B(t) => t, }; acc = acc + c.len(); \
i = i + 1; } acc";

struct Balance {
    allocations: usize,
    releases: usize,
    answer: u64,
}

type Helpers = fn(&Interner) -> Vec<Helper<'static>>;

fn no_helpers(_: &Interner) -> Vec<Helper<'static>> {
    Vec::new()
}

type Registries = fn() -> Vec<Registry<AcvusRuntime>>;

async fn balance(
    source: &str,
    helpers: Helpers,
    n: i64,
    opt: Opt,
    registries: Registries,
) -> Balance {
    let interner = Interner::new();
    let (context_types, snapshot) = split_context(&interner, int_context(&interner, "n", n));
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parse error"));
    let compiled = check_graph(
        &interner,
        ast,
        &helpers(&interner),
        &context_types,
        registries(),
        Ty::U64,
        opt,
        |_| {},
    )
    .unwrap_or_else(|refusal| panic!("at {opt:?}: {}", refusal.messages.join("\n")));
    let (_shared, mut interp) =
        execute_compiled(&interner, compiled, snapshot, Arc::new(SequentialExecutor));
    let allocated = ALLOCATIONS.load(Ordering::Relaxed);
    let released = RELEASES.load(Ordering::Relaxed);
    let answer = interp.execute().await.expect("the seeds hold every context the run fetches");
    Balance {
        allocations: ALLOCATIONS.load(Ordering::Relaxed) - allocated,
        releases: RELEASES.load(Ordering::Relaxed) - released,
        answer: answer.bits(),
    }
}

fn std_registries() -> Vec<Registry<AcvusRuntime>> {
    acvus_ext::std_registries::<AcvusRuntime>()
}

#[tokio::test]
async fn a_string_payload_read_by_patterns_leaves_nothing_behind() {
    for opt in [Opt::None, Opt::Full] {
        let measuring = ONE_AT_A_TIME.lock().expect("no measurement panicked");
        let (few, many) = (1_000i64, 5_000i64);
        let low = balance(
            PAYLOAD_READ_BY_PATTERNS,
            no_helpers,
            few,
            opt,
            std_registries,
        )
        .await;
        let high = balance(
            PAYLOAD_READ_BY_PATTERNS,
            no_helpers,
            many,
            opt,
            std_registries,
        )
        .await;
        drop(measuring);
        let span = (many - few) as f64;
        let allocated = (high.allocations as f64 - low.allocations as f64) / span;
        let left = ((high.allocations as f64 - high.releases as f64)
            - (low.allocations as f64 - low.releases as f64))
            / span;
        println!("at {opt:?}: allocations per iteration: {allocated:.3}, left behind: {left:.3}");
        assert!(
            left.abs() < 0.01,
            "at {opt:?}, an iteration leaves {left} allocations behind"
        );
    }
}

// -- Leaving a `for` over an owned array early (RFC-0057 rule 6) --------

/// What one iteration of `source` leaves allocated, at `opt`, with the run's
/// answer at the larger count checked against `answer`.
async fn left_per_iteration<F>(
    source: &str,
    helpers: Helpers,
    opt: Opt,
    answer: F,
    registries: Registries,
) -> f64
where
    F: Fn(i64) -> u64,
{
    let measuring = ONE_AT_A_TIME.lock().expect("no measurement panicked");
    let (few, many) = (1_000i64, 5_000i64);
    let low = balance(source, helpers, few, opt, registries).await;
    let high = balance(source, helpers, many, opt, registries).await;
    drop(measuring);
    assert_eq!(low.answer, answer(few), "at {opt:?}, the answer at {few}");
    assert_eq!(
        high.answer,
        answer(many),
        "at {opt:?}, the answer at {many}"
    );
    ((high.allocations as f64 - high.releases as f64)
        - (low.allocations as f64 - low.releases as f64))
        / (many - few) as f64
}

/// Each script leaves a `for` over an array of owners by the edge its name
/// gives, at both levels, and every iteration of the enclosing `while` gives
/// back all it allocated: the elements the loop took are released by their
/// own scopes, and the array's `Drop` on the leaving edge releases the rest.
async fn leaves_nothing_behind<F>(source: &str, answer: F)
where
    F: Fn(i64) -> u64,
{
    each_iteration_gives_back_all(source, no_helpers, answer).await;
}

async fn each_iteration_gives_back_all<F>(source: &str, helpers: Helpers, answer: F)
where
    F: Fn(i64) -> u64,
{
    for opt in [Opt::None, Opt::Full] {
        let left = left_per_iteration(source, helpers, opt, &answer, std_registries).await;
        println!("at {opt:?}: left behind per iteration: {left:.3}");
        assert!(
            left.abs() < 0.01,
            "at {opt:?}, an iteration leaves {left} allocations behind"
        );
    }
}

/// Two taken, "ab" read and "c" released by its scope at the `break`, "d"
/// never taken.
const BREAK_IN_OWNERS: &str = "\
let acc = 0; let i = 0; while i < @n { \
let a = [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()]; let k = 0; \
for s in a { if k == 1 { break; }; acc = acc + len(&s); k = k + 1; } \
i = i + 1; } acc";

#[tokio::test]
async fn a_break_out_of_an_array_of_strings_releases_the_elements_not_taken() {
    leaves_nothing_behind(BREAK_IN_OWNERS, |n| 2 * n as u64).await;
}

/// The control: a `continue` does not leave the loop, so the terminator's
/// exit is the one edge out, as it is without a jump.
const CONTINUE_IN_OWNERS: &str = "\
let acc = 0; let i = 0; while i < @n { \
let a = [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()]; \
for s in a { if len(&s) == 1 { continue; }; acc = acc + len(&s); } \
i = i + 1; } acc";

#[tokio::test]
async fn a_continue_in_an_array_of_strings_releases_every_element() {
    leaves_nothing_behind(CONTINUE_IN_OWNERS, |n| 2 * n as u64).await;
}

/// The `?` leaves the lambda from its second element, "c".
const TRY_IN_OWNERS: &str = "\
let f = |k| -> { let acc = 0; \
for s in [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()] { \
let r = if len(&s) == k { Err(1) } else { Ok(len(&s)) }; acc = acc + r?; } Ok(acc) }; \
let acc = 0; let i = 0; while i < @n { \
acc = acc + match f(1) { Ok(v) => v, Err(e) => e }; \
i = i + 1; } acc";

#[tokio::test]
async fn a_try_out_of_an_array_of_strings_releases_the_elements_not_taken() {
    leaves_nothing_behind(TRY_IN_OWNERS, |n| n as u64).await;
}

/// The `return` leaves the lambda from its second element, "c".
const RETURN_IN_OWNERS: &str = "\
let f = |k| -> { for s in [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()] { \
if len(&s) == k { return 1; }; } 0 }; \
let acc = 0; let i = 0; while i < @n { acc = acc + f(1); \
i = i + 1; } acc";

#[tokio::test]
async fn a_return_out_of_an_array_of_strings_releases_the_elements_not_taken() {
    leaves_nothing_behind(RETURN_IN_OWNERS, |n| n as u64).await;
}

/// Each outer element runs the inner loop to its `break` at "yz", and the
/// outer loop breaks at "c": per turn of the `while`, 1 + 2 + 1.
const NESTED_BREAKS_IN_OWNERS: &str = "\
let acc = 0; let i = 0; while i < @n { \
for s in [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()] { \
for t in [\"x\".to_string(), \"yz\".to_string(), \"w\".to_string()] { \
if len(&t) == 2 { break; }; acc = acc + len(&t); } \
if len(&s) == 1 { break; }; acc = acc + len(&s); } \
i = i + 1; } acc";

#[tokio::test]
async fn nested_breaks_out_of_arrays_of_strings_release_the_elements_not_taken() {
    leaves_nothing_behind(NESTED_BREAKS_IN_OWNERS, |n| 4 * n as u64).await;
}

/// The element's type is open where the checker reaches the `break`: the
/// lambda's parameters are settled by the call.
const BREAK_IN_OPEN_ELEMENTS: &str = "\
let f = |x, z| -> { let k = 0; for y in [x, z] { k = k + 1; break; } k }; \
let acc = 0; let i = 0; while i < @n { \
acc = acc + f(\"a\".to_string(), \"b\".to_string()); i = i + 1; } acc";

#[tokio::test]
async fn a_break_out_of_an_array_of_open_elements_releases_the_elements_not_taken() {
    leaves_nothing_behind(BREAK_IN_OPEN_ELEMENTS, |n| n as u64).await;
}

/// An element that is itself an aggregate owning a `String`.
const BREAK_IN_OBJECTS: &str = "\
let acc = 0; let i = 0; while i < @n { \
for o in [{ s: \"ab\".to_string(), }, { s: \"c\".to_string(), }, { s: \"d\".to_string(), }] { \
acc = acc + len(&o.s); break; } \
i = i + 1; } acc";

#[tokio::test]
async fn a_break_out_of_an_array_of_objects_releases_the_elements_not_taken() {
    leaves_nothing_behind(BREAK_IN_OBJECTS, |n| 2 * n as u64).await;
}

fn owned_string(i: &Interner, source: &'static str) -> Vec<Helper<'static>> {
    vec![Helper {
        name: "h",
        source,
        params: vec![ParamTerm::<Poly>::new(
            i.intern("p"),
            lift_to_poly(&Ty::String),
        )],
    }]
}

fn lends_its_string(i: &Interner) -> Vec<Helper<'static>> {
    owned_string(i, "len(&$p)\n")
}

fn compares_its_string(i: &Interner) -> Vec<Helper<'static>> {
    owned_string(i, "(if $p == \"ab\" { 2 } else { 0 }) as u64\n")
}

fn reads_then_returns_its_string(i: &Interner) -> Vec<Helper<'static>> {
    owned_string(i, "if len(&$p) == 2 { $p } else { \"\".to_string() }\n")
}

const FRESH_ARGUMENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
acc = acc + h(\"ab\".to_string()); i = i + 1; } acc";

const REUSED_ARGUMENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let s = \"ab\".to_string(); acc = acc + h(s) + h(s); i = i + 1; } acc";

const RETURNED_ARGUMENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let r = h(\"ab\".to_string()); acc = acc + len(&r); i = i + 1; } acc";

#[tokio::test]
async fn a_body_lending_its_fresh_string_releases_it() {
    each_iteration_gives_back_all(FRESH_ARGUMENT, lends_its_string, |n| 2 * n as u64).await;
}

#[tokio::test]
async fn a_body_comparing_its_fresh_string_releases_it() {
    each_iteration_gives_back_all(FRESH_ARGUMENT, compares_its_string, |n| 2 * n as u64).await;
}

#[tokio::test]
async fn a_body_lending_a_reused_string_releases_both_copies() {
    each_iteration_gives_back_all(REUSED_ARGUMENT, lends_its_string, |n| 4 * n as u64).await;
}

#[tokio::test]
async fn a_body_returning_its_string_releases_it_once() {
    each_iteration_gives_back_all(RETURNED_ARGUMENT, reads_then_returns_its_string, |n| {
        2 * n as u64
    })
    .await;
}

// -- Conversions of an argument (RFC-0041, RFC-0043 rule 2) ---------------

#[derive(ExternType)]
#[extern_type(name = "Bag")]
#[repr(transparent)]
struct Bag(Vec<i64>);

#[extern_fn(effect = pure)]
#[extern_cast]
fn bag(items: Vec<i64>) -> Bag {
    Bag(items)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn unbag(b: Bag) -> Vec<i64> {
    b.0
}

#[extern_fn(name = "weigh", effect = pure)]
fn weigh_bag(b: Bag, n: u64) -> u64 {
    b.0.len() as u64 * 100 + n
}

#[extern_fn(name = "peek", effect = pure)]
fn peek_bag(b: &Bag, n: u64) -> u64 {
    b.0.len() as u64 * 100 + n
}

#[extern_fn(name = "pair", effect = pure)]
fn pair_bag(a: &Bag, b: &Bag, n: u64) -> u64 {
    (a.0.len() + b.0.len()) as u64 * 100 + n
}

#[extern_fn(name = "stuff", effect = pure)]
fn stuff_bag(b: &mut Bag, n: i64) -> u64 {
    b.0.push(n);
    b.0.len() as u64
}

#[extern_fn(name = "weigh", effect = pure)]
fn weigh_slice<Rt>(s: &[Erased<Rt, i64>], wide: bool) -> u64
where
    Rt: Runtime,
{
    let n = s.len() as u64;
    if wide { n * 10 } else { n }
}

#[extern_fn(name = "peek", effect = pure)]
fn peek_slice<Rt>(s: &[Erased<Rt, i64>], wide: bool) -> u64
where
    Rt: Runtime,
{
    let n = s.len() as u64;
    if wide { n * 10 } else { n }
}

#[extern_fn(name = "pair", effect = pure)]
fn pair_slice<Rt>(a: &[Erased<Rt, i64>], b: &[Erased<Rt, i64>], wide: bool) -> u64
where
    Rt: Runtime,
{
    let n = (a.len() + b.len()) as u64;
    if wide { n * 10 } else { n }
}

/// `t` alone: every argument `t::peek` or `t::weigh` converts meets its
/// parameter on the known path.
fn converting_only() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = std_registries();
    regs.push(extern_registry! {
        ns: "t",
        types: [Bag],
        fns: [bag, unbag, weigh_bag, peek_bag, pair_bag, stuff_bag],
    });
    regs
}

/// `t` with `u`, whose `peek` and `weigh` view the argument `t`'s convert,
/// so that argument is held until the rest of the call settles.
fn weighing() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = converting_only();
    regs.push(extern_registry! {
        ns: "u",
        fns: [weigh_slice, peek_slice, pair_slice],
    });
    regs
}

/// The lambda parameter's head is open where `weigh` meets it, so the
/// argument is held, and the settled `t::weigh` takes the `Vec` each call
/// passes through `bag`: the `Vec` is moved into the cast and the `Bag`
/// into the callee, and every iteration gives back what it allocated.
const HELD_BY_VALUE: &str = "\
let f = |x| -> weigh(x, 1); \
let acc = 0; let i = 0; while i < @n { acc = acc + f(vec([1, 2])); i = i + 1; } acc";

#[tokio::test]
async fn a_held_argument_converted_by_value_leaves_nothing_behind() {
    for opt in [Opt::None, Opt::Full] {
        let left =
            left_per_iteration(HELD_BY_VALUE, no_helpers, opt, |n| 201 * n as u64, weighing).await;
        println!("at {opt:?}: left behind per iteration: {left:.3}");
        assert!(
            left.abs() < 0.01,
            "at {opt:?}, an iteration leaves {left} allocations behind"
        );
    }
}

/// RFC-0041: `peek` takes `v` through the reference: each call takes the
/// `Vec` out of `v`, casts it into a `Bag` in a temporary the call borrows,
/// and casts it back into `v` after the call, and every iteration gives
/// back what it allocated.
const THROUGH_A_REFERENCE: &str = "\
let v = vec([1, 2]); \
let acc = 0; let i = 0; while i < @n { acc = acc + peek(&v, 1); i = i + 1; } acc";

/// RFC-0041: a borrowed temporary is cast into the `Bag` the call borrows
/// and is not cast back; the `Bag` is released after the call.
const TEMPORARY_THROUGH_A_REFERENCE: &str = "\
let acc = 0; let i = 0; while i < @n { acc = acc + peek(&vec([1, 2]), 1); i = i + 1; } acc";

/// RFC-0041: the place is read at its own type after the call restores it.
const THROUGH_A_REFERENCE_THEN_READ: &str = "\
let v = vec([1, 2]); \
let acc = 0; let i = 0; while i < @n { acc = acc + peek(&v, 1) + len(&v); i = i + 1; } acc";

/// RFC-0041: two shared lends of `v` in one call lend one temporary, and `v`
/// is cast back once.
const SHARED_LENDS_IN_ONE_CALL: &str = "\
let v = vec([1, 2]); \
let acc = 0; let i = 0; while i < @n { acc = acc + pair(&v, &v, 1); i = i + 1; } acc";

/// RFC-0041: a nested call's shared lend of `v` lends the outer call's
/// temporary, and `v` is cast back once, after the outer call.
const SHARED_LEND_IN_A_NESTED_CALL: &str = "\
let v = vec([1, 2]); \
let acc = 0; let i = 0; while i < @n { acc = acc + peek(&v, peek(&v, 1)); i = i + 1; } acc";

/// RFC-0041: a `&mut` conversion writes back through the cast.
const MUTABLE_THROUGH_A_REFERENCE: &str = "\
let acc = 0; let i = 0; \
while i < @n { let v = vec([1, 2]); acc = acc + stuff(&mut v, 7) + len(&v); i = i + 1; } acc";

#[tokio::test]
async fn a_conversion_through_a_reference_leaves_nothing_behind() {
    let sets: [(&str, Registries); 2] = [("t", converting_only), ("t+u", weighing)];
    let sources: [(&str, u64); 6] = [
        (THROUGH_A_REFERENCE, 201),
        (TEMPORARY_THROUGH_A_REFERENCE, 201),
        (THROUGH_A_REFERENCE_THEN_READ, 203),
        (SHARED_LENDS_IN_ONE_CALL, 401),
        (SHARED_LEND_IN_A_NESTED_CALL, 401),
        (MUTABLE_THROUGH_A_REFERENCE, 6),
    ];
    for (source, each) in sources {
        for (set, registries) in sets {
            for opt in [Opt::None, Opt::Full] {
                let left =
                    left_per_iteration(source, no_helpers, opt, |n| each * n as u64, registries)
                        .await;
                println!("{set} at {opt:?}: left behind per iteration: {left:.3}");
                assert!(
                    left.abs() < 0.01,
                    "{set} at {opt:?}, an iteration leaves {left} allocations behind: {source}"
                );
            }
        }
    }
}
