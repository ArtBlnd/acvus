//! A page lends a `Vec<String>` context as the language's `&[String]`, and a
//! read of an element allocates nothing: the closure reads each string where
//! the page holds it (RFC-0090 rule 3). The measurement is its own binary
//! because a global allocator counts every allocation in the binary.

use acvus_extern::{Ctx, Erased};
use acvus_interpreter::{
    AcvusRuntime, Host, MemoryStorage, Program, Scope, SequentialExecutor, Source,
};

type Rt = AcvusRuntime;

thread_local! {
    static ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn allocations() -> usize {
    ALLOCATIONS.with(std::cell::Cell::get)
}

struct Counting;

// SAFETY: every method forwards to the system allocator with the same
// arguments; the counter is a side effect on thread-local state and changes
// no pointer.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new: usize) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn program() -> Program {
    let host = Host::new(acvus_ext::std_registries::<AcvusRuntime>())
        .init("log", Source::Expr("vec([])"))
        .entry::<(), ()>("one", Source::Script(r#"@log.push("one".to_string());"#))
        .entry::<(), ()>("two", Source::Script(r#"@log.push("two".to_string());"#));
    match host.compile(SequentialExecutor) {
        Ok(program) => program,
        Err(error) => panic!("the program is refused: {error:?}"),
    }
}

/// Store in `storage` a `@log` that holds `one`, `two`, repeated `pairs`
/// times.
async fn log_pairs(scope: Scope<'_>, storage: &mut MemoryStorage, pairs: usize) {
    let mut page = scope.open(storage);
    for name in std::iter::repeat_n(["one", "two"], pairs).flatten() {
        let entry = scope.entry::<(), ()>(name).expect("the entry returns `()`");
        entry.run(&mut page, ()).await.expect("the init fills `@log`");
    }
}

fn block_on<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime")
        .block_on(future)
}

/// The bytes of every element, and how many allocations the reads of the
/// elements made, counted inside the closure.
struct Read {
    bytes: usize,
    first: bool,
    allocations_in_reads: usize,
}

async fn read(scope: Scope<'_>, storage: &mut MemoryStorage) -> Read {
    let mut page = scope.open(storage);
    page.with("log", |ctx: &mut Ctx<'_, Rt>, xs: &[Erased<Rt, String>]| {
        let before = allocations();
        let bytes = xs.iter().map(|x| x.as_ref(ctx.rt).len()).sum();
        let first = xs[0].as_ref(ctx.rt) == "one" && xs[1].as_ref(ctx.rt) == "two";
        Read {
            bytes,
            first,
            allocations_in_reads: allocations() - before,
        }
    })
    .await
    .expect("`@log` holds a `Vec<String>`")
}

#[test]
fn two_strings_are_read_in_place_with_no_allocation() {
    let program = program();
    block_on(program.scope(async |s| {
        let mut storage = MemoryStorage::new();
        log_pairs(s, &mut storage, 1).await;
        let read = read(s, &mut storage).await;
        assert!(read.first, "the elements are `one` and `two` in order");
        assert_eq!(read.bytes, 6);
        assert_eq!(read.allocations_in_reads, 0, "an element read allocates nothing");
    }));
}

/// What a `with` allocates is the type comparison before the closure runs,
/// once per call: a page of 64 elements costs what a page of 2 does.
#[test]
fn a_lend_allocates_the_same_for_two_elements_and_for_sixty_four() {
    let program = program();
    block_on(program.scope(async |s| {
        let mut small = MemoryStorage::new();
        let mut large = MemoryStorage::new();
        log_pairs(s, &mut small, 1).await;
        log_pairs(s, &mut large, 32).await;
        read(s, &mut small).await;
        read(s, &mut large).await;

        let before = allocations();
        let small_read = read(s, &mut small).await;
        let small_cost = allocations() - before;
        let before = allocations();
        let large_read = read(s, &mut large).await;
        let large_cost = allocations() - before;

        assert_eq!(small_read.bytes, 6);
        assert_eq!(large_read.bytes, 6 * 32);
        assert_eq!(large_read.allocations_in_reads, 0);
        assert_eq!(small_cost, large_cost, "no allocation scales with the elements read");
    }));
}
