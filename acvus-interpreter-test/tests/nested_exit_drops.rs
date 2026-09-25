//! RFC-0066 rule 1's move of an exit out of a nested loop, at the drops: a
//! `String` alive across the moved exit is released exactly once on every
//! path. The pass runs before drop insertion, so the drops on the moved edge
//! are the inserter's; a release left out leaves the string allocated behind
//! the run, and one made twice releases more than the run allocated.
//!
//! The measurement needs the prepared program apart from its run, since a
//! global allocator counts every allocation in the binary; it lives in its
//! own test binary for that reason.

use std::sync::Arc;

use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{Context, check_source, execute_compiled, split_context};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

thread_local! {
    static LIVE: std::cell::Cell<isize> = const { std::cell::Cell::new(0) };
}

struct Counting;

// SAFETY: every method forwards to the system allocator with the same
// arguments; the counter is a side effect on thread-local state and changes
// no pointer.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        LIVE.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        LIVE.with(|n| n.set(n.get() - 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `tag` lives across both loops and the moved code reads it, `row` lives
/// across the inner loop and the moved code reads it, and `cell` lives in
/// the inner loop's iteration, where the moved edge leaves it. Each is long
/// enough to live on the heap, where the count sees it.
const FIND: &str = "\
let find = |m| -> { \
    let tag = \"tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt\".to_string(); \
    for i in 0u64..m.len() { \
        let row = \"rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr\".to_string(); \
        for j in 0u64..m[i].len() { \
            let cell = \"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\".to_string(); \
            if m[i][j] == 7 && cell.len() == 64 { \
                return tag.len() + row.len() + i * 10 + j; \
            }; \
        } \
    } \
    tag.len() \
}; \
find(vec([vec([1, 2, 3]), vec([4, 7, 7])])) * 1000 + find(vec([vec([1]), vec([2])]))";

struct Measured {
    listing: String,
    leaked: isize,
    answer: Value,
}

fn measure(source: &str, opt: Opt) -> Measured {
    let interner = Interner::new();
    let (context_types, snapshot) = split_context(&interner, Context::default());
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("it parses"));
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    let Ok(compiled) = check_source(&interner, ast, &context_types, registries, Ty::U64, opt, |_| {})
    else {
        panic!("the checker admits it at {opt:?}")
    };
    let listing = compiled
        .modules
        .values()
        .map(|module| acvus_mir::printer::dump_with(&interner, module))
        .collect();
    let (_shared, mut interp) = execute_compiled(
        &interner,
        compiled,
        snapshot,
        Arc::new(SequentialExecutor) as Arc<dyn acvus_interpreter::Executor>,
    );
    let live = LIVE.with(std::cell::Cell::get);
    let answer = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime")
        .block_on(interp.execute())
        .expect("the program reads no context");
    Measured {
        listing,
        leaked: LIVE.with(std::cell::Cell::get) - live,
        answer,
    }
}

/// The count settles after the first run warms the allocator's cold paths,
/// so the number a level is judged by is the least of three runs.
fn least_left_behind(opt: Opt) -> Measured {
    (0..3)
        .map(|_| measure(FIND, opt))
        .min_by_key(|measured| measured.leaked)
        .expect("three runs")
}

#[test]
fn a_string_alive_across_an_exit_moved_out_of_a_nested_loop_is_released_once() {
    for opt in [Opt::Full, Opt::None] {
        let measured = least_left_behind(opt);
        assert_eq!(measured.answer.as_int(), (64 + 64 + 11) * 1000 + 64, "{}", measured.listing);
        assert_eq!(
            measured.leaked, 0,
            "at {opt:?} the run leaves nothing allocated and releases nothing twice:\n{}",
            measured.listing
        );
    }
    let full = measure(FIND, Opt::Full);
    assert!(
        full.listing.contains("variant Some") && full.listing.contains("variant None"),
        "the exit moves into the inner loop's own exit:\n{}",
        full.listing
    );
}
