//! RFC-0066 rule 1's move of an exit out of a nested loop, at the drops: a
//! `String` alive across the moved exit is released exactly once, and on the
//! edge it dies on. The pass runs before drop insertion, so the drops on the
//! moved edge are the inserter's.
//!
//! A count taken once the run ends cannot see a drop left out: the frame's
//! exit releases every register still marked, and an `assign` to a storage
//! releases the value it overwrites (RFC-0048 rules 4 and 6). So each string
//! here has a length no other allocation of the run has, and the allocator
//! reads, at chosen allocations inside the run, which strings are alive.
//!
//! The measurement needs the prepared program apart from its run, since a
//! global allocator counts every allocation in the binary; it lives in its
//! own test binary for that reason.

use std::cell::Cell;
use std::sync::Arc;

use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{Context, check_source, execute_compiled, split_context};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// The strings the program allocates, each by the one length it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// Lives across both loops; the moved code reads it.
    Tag,
    /// Lives across the inner loop; the moved code reads it.
    Row,
    /// Lives in one inner iteration.
    Cell,
    /// An element of `m`, which the moved code does not read: the edge that
    /// leaves by the moved exit releases `m`.
    Element,
    /// Allocated by the moved code, once the exit has been taken.
    AfterTheExit,
    /// Allocated once the outer loop has run out.
    AfterTheLoops,
}

impl Held {
    const ALL: [Held; 6] = [
        Held::Tag,
        Held::Row,
        Held::Cell,
        Held::Element,
        Held::AfterTheExit,
        Held::AfterTheLoops,
    ];

    /// Odd lengths no allocation the machine makes for itself has in this
    /// run; the exact alive sets the test asserts would count one that did.
    const fn len(self) -> usize {
        match self {
            Held::Tag => 101,
            Held::Row => 107,
            Held::Cell => 109,
            Held::Element => 113,
            Held::AfterTheExit => 127,
            Held::AfterTheLoops => 131,
        }
    }

    fn of_size(size: usize) -> Option<Held> {
        Held::ALL.into_iter().find(|held| held.len() == size)
    }

    const fn at(self) -> usize {
        self as usize
    }

    fn literal(self) -> String {
        let letter = match self {
            Held::Tag => 't',
            Held::Row => 'r',
            Held::Cell => 'c',
            Held::Element => 'e',
            Held::AfterTheExit => 'x',
            Held::AfterTheLoops => 'd',
        };
        format!("\"{}\"", String::from(letter).repeat(self.len()))
    }
}

type Alive = [isize; Held::ALL.len()];

/// What the allocator saw of the strings while a run was watched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Seen {
    /// Every allocation, the machine's own included, less every release.
    net: isize,
    /// A `Row` or `Cell` allocated while one of its kind was still alive:
    /// the one before it was released late, or not at all.
    overlapped: Option<Held>,
    after_the_exit: Option<Alive>,
    after_the_loops: Option<Alive>,
}

thread_local! {
    static WATCHING: Cell<bool> = const { Cell::new(false) };
    static ALIVE: Cell<Alive> = const { Cell::new([0; Held::ALL.len()]) };
    static SEEN: Cell<Seen> = const {
        Cell::new(Seen {
            net: 0,
            overlapped: None,
            after_the_exit: None,
            after_the_loops: None,
        })
    };
}

fn allocated(size: usize) {
    if !WATCHING.with(Cell::get) {
        return;
    }
    let mut seen = SEEN.with(Cell::get);
    seen.net += 1;
    if let Some(held) = Held::of_size(size) {
        let mut alive = ALIVE.with(Cell::get);
        match held {
            Held::Row | Held::Cell if alive[held.at()] != 0 && seen.overlapped.is_none() => {
                seen.overlapped = Some(held);
            }
            Held::AfterTheExit => seen.after_the_exit = Some(alive),
            Held::AfterTheLoops => seen.after_the_loops = Some(alive),
            _ => {}
        }
        alive[held.at()] += 1;
        ALIVE.with(|cell| cell.set(alive));
    }
    SEEN.with(|cell| cell.set(seen));
}

fn released(size: usize) {
    if !WATCHING.with(Cell::get) {
        return;
    }
    SEEN.with(|cell| {
        let mut seen = cell.get();
        seen.net -= 1;
        cell.set(seen);
    });
    if let Some(held) = Held::of_size(size) {
        ALIVE.with(|cell| {
            let mut alive = cell.get();
            alive[held.at()] -= 1;
            cell.set(alive);
        });
    }
}

struct Counting;

// SAFETY: every method forwards to the system allocator with the same
// arguments; the bookkeeping is a side effect on thread-local state, which
// allocates nothing, and changes no pointer.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        allocated(layout.size());
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        released(layout.size());
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `find` over `[[e, 1, 2, 3], [4, 7, 7]]` leaves by the moved exit at
/// `(1, 1)`; over `[[1], [2]]` it runs out and leaves by the outer loop's
/// header. The inner loop compares lengths, so `m`'s elements are strings.
fn find_source() -> String {
    let tag = Held::Tag.literal();
    let row = Held::Row.literal();
    let cell = Held::Cell.literal();
    let element = Held::Element.literal();
    let after_the_exit = Held::AfterTheExit.literal();
    let after_the_loops = Held::AfterTheLoops.literal();
    let digits = |lens: &[usize]| {
        lens.iter()
            .map(|len| format!("\"{}\".to_string()", "n".repeat(*len)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let first = digits(&[1, 2, 3]);
    let second = digits(&[4, 7, 7]);
    format!(
        "let find = |m| -> {{ \
            let tag = {tag}.to_string(); \
            for i in 0u64..m.len() {{ \
                let row = {row}.to_string(); \
                for j in 0u64..m[i].len() {{ \
                    let cell = {cell}.to_string(); \
                    if m[i][j].len() == 7 && cell.len() == {cell_len} {{ \
                        return {after_the_exit}.to_string().len() + tag.len() + row.len() \
                            + i * 10 + j; \
                    }}; \
                }} \
            }} \
            {after_the_loops}.to_string().len() + tag.len() \
        }}; \
        find(vec([vec([{element}.to_string(), {first}]), vec([{second}])])) * 1000 \
            + find(vec([vec([\"n\".to_string()]), vec([\"nn\".to_string()])]))",
        cell_len = Held::Cell.len(),
    )
}

/// `(1, 1)` is `i * 10 + j = 11` past the three lengths the moved code adds.
const ANSWER: i64 = (101 + 107 + 127 + 11) * 1000 + 101 + 131;

struct Measured {
    listing: String,
    answer: Value,
    seen: Seen,
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
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    ALIVE.with(|cell| cell.set([0; Held::ALL.len()]));
    SEEN.with(|cell| cell.set(Seen::default()));
    WATCHING.with(|cell| cell.set(true));
    let answer = runtime
        .block_on(interp.execute())
        .expect("the program reads no context");
    WATCHING.with(|cell| cell.set(false));
    Measured {
        listing,
        answer,
        seen: SEEN.with(Cell::get),
    }
}

/// The net count settles after the first run warms the allocator's cold
/// paths, so the run a level is judged by is the one of three that leaves
/// the least behind.
fn least_left_behind(source: &str, opt: Opt) -> Measured {
    (0..3)
        .map(|_| measure(source, opt))
        .min_by_key(|measured| measured.seen.net)
        .expect("three runs")
}

fn alive(pairs: &[(Held, isize)]) -> Alive {
    let mut alive = [0; Held::ALL.len()];
    for (held, count) in pairs {
        alive[held.at()] = *count;
    }
    alive
}

#[test]
fn a_string_alive_across_an_exit_moved_out_of_a_nested_loop_is_released_once() {
    let source = find_source();
    for opt in [Opt::Full, Opt::None] {
        let Measured {
            listing,
            answer,
            seen,
        } = least_left_behind(&source, opt);
        assert_eq!(answer.as_int(), ANSWER, "{listing}");
        assert_eq!(
            seen.overlapped, None,
            "at {opt:?} each row and cell is released before the next one is made:\n{listing}"
        );
        assert_eq!(
            seen.after_the_exit,
            Some(alive(&[(Held::Tag, 1), (Held::Row, 1)])),
            "at {opt:?} past the moved exit only `tag` and `row` are alive; `cell` and `m` \
             were released on the way out:\n{listing}"
        );
        assert_eq!(
            seen.after_the_loops,
            Some(alive(&[(Held::Tag, 1)])),
            "at {opt:?} past the loops only `tag` is alive:\n{listing}"
        );
        assert_eq!(
            seen.net, 0,
            "at {opt:?} the run leaves nothing allocated and releases nothing twice:\n{listing}"
        );
    }
    let full = measure(&source, Opt::Full);
    assert!(
        full.listing.contains("variant Some") && full.listing.contains("variant None"),
        "the exit moves into the inner loop's own exit:\n{}",
        full.listing
    );
}
