//! A log-line parser: the flat, record-oriented program.
//!
//! `N` generated log lines are handed to the script as a `Vec<Vec<i64>>` --
//! bytes as `i64`, because the language has no byte literal and no character
//! API. Over them runs a glob matcher (`*` any run, `?` one byte, literal
//! otherwise) written the way a user of the language writes one: two indices,
//! a backtrack position, `while`, no `break`. A matching line's trailing
//! latency field is parsed and summed.
//!
//! Each line is independent of every other, and the aggregate is exact, so
//! this is the shape a `for` -- an iteration space, no overlap, no order --
//! would take. Four ways of writing it:
//!
//! - `inline`: the matcher's `while` written into the kernel's `while`.
//! - `closure`: the same matcher as a named closure, called once per line --
//!   the `CallIndirect` boundary.
//! - `heavy ext`: the matcher as a `Heavy` Rust extern, one call per line.
//!   This is what `optimize::spawn_split` can reach today without a `for`.
//! - `sync ext`: the same matcher in the caller's frame, taking the pattern
//!   and the line as `&[i64]` -- the script holds the corpus and the call
//!   crosses two register pairs (RFC-0047 rule 6).
//! - `sync native`: the same matcher in the caller's frame over a corpus Rust
//!   holds, named by a `u64` line index. One register of argument and no
//!   container crossing, so it is the floor an extern call can cost, and the
//!   difference from `sync ext` is what a script pays to hold its own
//!   container.
//!
//! The Rust twins are the same matcher over the same `Vec<Vec<i64>>`, once
//! sequentially and once over `std::thread::scope` chunks -- the parallel
//! ceiling a parallel `for` would be measured against.
//!
//! These timings hold only under one pinned core and a fixed load base;
//! `benches/README.md` states the protocol.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::{Erased, Owned, Registry, extern_fn, extern_registry, vec_ty};
use acvus_interpreter::{
    AcvusRuntime, Executor, Interpreter, InterpreterContext, SequentialExecutor, TokioExecutor,
    Value,
};
use acvus_interpreter_test::listing::{BlockListing, PartListing, script_listing_with_externs};
use acvus_interpreter_test::{
    Context, TypedValue, compile_source_with_externs, execute_compiled, split_context, typed,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use tokio::runtime::Runtime;

// -- The answer ---------------------------------------------------------

/// What one run produced. It travels between the script and Rust as a single
/// `i64` because a script returns one value: `count` above `SCALE`, `total`
/// below it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Counted {
    count: i64,
    total: i64,
}

/// A latency is at most `MAX_LATENCY`, so the sum over the largest size this
/// bench runs is at most `MAX_LATENCY * 100 000 = 99 900 000`; the scale is
/// the next power of ten above it, and `packed` asserts the gap.
const SCALE: i64 = 1_000_000_000;

impl Counted {
    const ZERO: Counted = Counted { count: 0, total: 0 };

    fn plus(self, other: Counted) -> Counted {
        Counted {
            count: self.count + other.count,
            total: self.total + other.total,
        }
    }

    fn packed(self) -> i64 {
        assert!(
            self.total < SCALE,
            "the latency sum {} reached the scale {SCALE} that separates it from the count",
            self.total
        );
        self.count * SCALE + self.total
    }

    fn unpack(packed: i64) -> Counted {
        Counted {
            count: packed / SCALE,
            total: packed % SCALE,
        }
    }
}

// -- The data -----------------------------------------------------------

const LCG_MUL: u64 = 6_364_136_223_846_793_005;
const LCG_ADD: u64 = 1_442_695_040_888_963_407;

/// The seed every size starts from, so the line at index `k` is the same line
/// in the script, in the Rust twin and under `perf`.
const SEED: u64 = 0x5eed_0f_10_91_11;

/// A deterministic stream of words: the only source of variation in the data.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        self.0 >> 17
    }

    fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "an empty range has no member to draw");
        self.next() % n
    }

    fn index(&mut self, n: usize) -> usize {
        let drawn = self.below(n as u64);
        usize::try_from(drawn).expect("a draw below a usize is a usize")
    }
}

/// The largest latency a generated line carries, in milliseconds: three
/// digits, so `latency_of`'s `place` never passes 100.
const MAX_LATENCY: i64 = 999;

const METHODS: [&str; 4] = ["GET", "POST", "PUT", "DELETE"];
const STATUSES: [&str; 4] = ["200", "404", "500", "201"];

/// One line, 44-78 bytes:
///
/// ```text
/// 2026-09-19T12:34:56 10.0.0.17 GET /api/users/1234 200 37
/// ```
///
/// Nothing here is tuned toward a mispredict rate: the mix is what a request
/// log holds -- four methods, three path shapes, four statuses, a latency.
fn line_text(rng: &mut Lcg) -> String {
    let method = METHODS[rng.index(METHODS.len())];
    let status = STATUSES[rng.index(STATUSES.len())];
    let id = rng.below(10_000);
    let path = match rng.below(4) {
        0 | 1 => format!("/api/users/{id}"),
        2 => format!("/api/orders/{id}/items"),
        _ => "/health".to_string(),
    };
    let hour = rng.below(24);
    let minute = rng.below(60);
    let second = rng.below(60);
    let host = rng.below(250) + 1;
    let ms = rng.below(MAX_LATENCY as u64) + 1;
    format!(
        "2026-09-19T{hour:02}:{minute:02}:{second:02} 10.0.0.{host} {method} {path} {status} {ms}"
    )
}

fn bytes(s: &str) -> Vec<i64> {
    s.bytes().map(i64::from).collect()
}

/// A leading `*` over the timestamp and the host, a `?` over the method's
/// first byte, literal runs, two more `*`. A `POST`, `PUT` or `DELETE` line
/// fails only after the leading `*` has walked the whole line, which is the
/// backtracking this bench is made of.
const PATTERN: &str = "* ?ET /api/users/* 200 *";

fn lines_of(n: usize) -> Vec<Vec<i64>> {
    let mut rng = Lcg(SEED);
    (0..n).map(|_| bytes(&line_text(&mut rng))).collect()
}

// -- The matcher, in Rust -----------------------------------------------

const STAR: i64 = b'*' as i64;
const QMARK: i64 = b'?' as i64;
const ZERO_DIGIT: i64 = b'0' as i64;
const NINE_DIGIT: i64 = b'9' as i64;

fn glob(pat: &[i64], line: &[i64]) -> bool {
    glob_over(pat.len(), |at| pat[at], line.len(), |at| line[at])
}

/// The same algorithm the script runs, statement for statement: one forward
/// index into each of pattern and line, the last `*` with the position it was
/// matched at, and a backtrack that advances that position by one.
///
/// It reads its two sequences through an accessor because the extern reads
/// them through a `Slice`'s view of the script's own container, where a Rust
/// `&[i64]` would be a copy per call.
#[inline(always)]
fn glob_over<P, L>(plen: usize, pat: P, llen: usize, line: L) -> bool
where
    P: Fn(usize) -> i64,
    L: Fn(usize) -> i64,
{
    let mut i = 0usize;
    let mut j = 0usize;
    let mut has_star = false;
    let mut star = 0usize;
    let mut mark = 0usize;
    loop {
        if i >= llen {
            while j < plen && pat(j) == STAR {
                j += 1;
            }
            return j == plen;
        }
        let mut advanced = false;
        if j < plen {
            let pj = pat(j);
            if pj == STAR {
                has_star = true;
                star = j;
                mark = i;
                j += 1;
                advanced = true;
            } else if pj == QMARK || pj == line(i) {
                i += 1;
                j += 1;
                advanced = true;
            }
        }
        if !advanced {
            if !has_star {
                return false;
            }
            j = star + 1;
            mark += 1;
            i = mark;
        }
    }
}

/// The trailing integer field, read backwards from the end of the line. A
/// byte that is not a digit ends the field; the digits read so far are the
/// answer, and a line with no trailing digit has a latency of zero.
fn latency_of(line: &[i64]) -> i64 {
    let mut lat = 0i64;
    let mut place = 1i64;
    let mut k = line.len();
    while k > 0 {
        let c = line[k - 1];
        if !(ZERO_DIGIT..=NINE_DIGIT).contains(&c) {
            return lat;
        }
        lat += (c - ZERO_DIGIT) * place;
        place *= 10;
        k -= 1;
    }
    lat
}

fn fold_lines(lines: &[Vec<i64>], pat: &[i64]) -> Counted {
    lines
        .iter()
        .filter(|line| glob(pat, line))
        .fold(Counted::ZERO, |acc, line| {
            acc.plus(Counted {
                count: 1,
                total: latency_of(line),
            })
        })
}

fn rust_seq(lines: &[Vec<i64>], pat: &[i64]) -> Counted {
    fold_lines(lines, pat)
}

/// The parallel ceiling: the same work split into one chunk per core, no
/// order and no overlap -- the shape a parallel `for` would emit.
fn rust_chunked(lines: &[Vec<i64>], pat: &[i64], chunks: usize) -> Counted {
    assert!(chunks > 0, "a chunked run needs at least one chunk");
    assert!(!lines.is_empty(), "a chunked run needs at least one line");
    let size = lines.len().div_ceil(chunks);
    std::thread::scope(|scope| {
        let handles: Vec<_> = lines
            .chunks(size)
            .map(|chunk| scope.spawn(move || fold_lines(chunk, pat)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a chunk thread does not panic"))
            .fold(Counted::ZERO, Counted::plus)
    })
}

// -- The matcher, as an extern ------------------------------------------

/// The corpus the cases that name a line by its index match against. A slice
/// borrows the frame the call laid its arguments on, and a `heavy` call is
/// awaited, so a parameter of one is refused there (RFC-0047 rule 6); the
/// `sync native` case takes an index by choice, to hold its boundary at one
/// register. The bytes live on the Rust side, built from the same `SEED`.
struct Corpus {
    lines: Vec<Vec<i64>>,
    pat: Vec<i64>,
}

impl Corpus {
    fn of_size(n: usize) -> Arc<Corpus> {
        Arc::new(Corpus {
            lines: lines_of(n),
            pat: bytes(PATTERN),
        })
    }

    fn matches(&self, index: u64) -> bool {
        let index = usize::try_from(index).expect("a line index is a usize");
        let line = self.lines.get(index).unwrap_or_else(|| {
            panic!(
                "line {index} is past the {} the corpus holds",
                self.lines.len()
            )
        });
        glob(&self.pat, line)
    }
}

/// The `Heavy` declaration (RFC-0046): the runtime hands the call to the
/// blocking pool and awaits it.
#[extern_fn(heavy, effect = pure)]
fn match_heavy(#[state] corpus: &Arc<Corpus>, index: u64) -> bool {
    corpus.matches(index)
}

/// The same body, still `heavy`, declared `opaque`. One variable apart from
/// `match_heavy`, and that variable is what `spawn_split::is_io_call` reads.
#[extern_fn(heavy, effect = opaque)]
fn match_heavy_opaque(#[state] corpus: &Arc<Corpus>, index: u64) -> bool {
    corpus.matches(index)
}

/// The same body as `match_heavy` in the caller's frame, one declaration
/// variable apart from it: the sequential floor the `heavy` rows are read
/// against, and the extern boundary at one `u64` of argument.
#[extern_fn(effect = pure)]
fn match_sync(#[state] corpus: &Arc<Corpus>, index: u64) -> bool {
    corpus.matches(index)
}

/// The same matcher in the caller's frame, over the pattern and the line the
/// script lends it rather than a corpus of its own.
#[extern_fn(effect = pure)]
fn glob_match<Rt>(pat: &[Erased<Rt, i64>], line: &[Erased<Rt, i64>]) -> bool
where
    Rt: acvus_extern::Runtime,
{
    // The elements are the script's own (RFC-0047 rule 6, RFC-0068 rule 4), and
    // an `Erased<Rt, i64>` is read in place through `get`, so the
    // accessors still index the container the script lent.
    glob_over(pat.len(), |at| pat[at].get(), line.len(), |at| line[at].get())
}

fn registries(corpus: &Arc<Corpus>) -> Vec<Registry<AcvusRuntime>> {
    let for_heavy = Arc::clone(corpus);
    let for_opaque = Arc::clone(corpus);
    let for_sync = Arc::clone(corpus);
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [
            match_heavy(for_heavy),
            match_heavy_opaque(for_opaque),
            match_sync(for_sync),
            glob_match,
        ],
    });
    regs
}

// -- The script ---------------------------------------------------------

/// `a[i]` takes a `u64` (RFC-0047) and RFC-0058 gives one a literal, so the
/// zero and the one every index arithmetic needs are written down. The
/// workaround this recorded -- `one = plen / plen` -- is what
/// `benches/programs.rs` and `benches/shapes.rs` still show.
const PRELUDE: &str = "\
let plen = len(&@pat); \
let one = 1u64; \
let zero = 0u64; \
let n = len(&@lines); \
let count = 0; \
let total = 0; \
let li = zero; \
";

/// The three script expressions a generated fragment reads its data through:
/// where the pattern is, where the line is, and where the line's length is.
/// They differ between the cases -- `@pat` and `@lines[li]` in the kernel's
/// own body, the closure's parameters inside the closure.
struct Places {
    pat: &'static str,
    line: &'static str,
    len: &'static str,
}

/// The matcher over the places, leaving `ok`. `while` is the only loop the
/// language has and there is no `break`, so the exit is the `alive` flag.
fn matcher(places: &Places) -> String {
    let Places {
        pat,
        line,
        len: slen,
    } = places;
    format!(
        "\
let i = zero; \
let j = zero; \
let hs = false; \
let star = zero; \
let mark = zero; \
let alive = true; \
let ok = false; \
while alive {{ \
if i < {slen} {{ \
let adv = false; \
if j < plen {{ \
let pj = {pat}[j]; \
if pj == {STAR} {{ hs = true; star = j; mark = i; j = j + one; adv = true; }} \
else {{ if pj == {QMARK} || pj == {line}[i] {{ i = i + one; j = j + one; adv = true; }}; }}; \
}}; \
if !adv {{ \
if hs {{ j = star + one; mark = mark + one; i = mark; }} else {{ alive = false; }}; \
}}; \
}} else {{ \
while j < plen && {pat}[j] == {STAR} {{ j = j + one; }} \
ok = j == plen; \
alive = false; \
}}; \
}} \
"
    )
}

/// The same matcher with `break` in place of the `alive` flag (RFC-0057
/// rule 4). The flag was the exit the language had; the two exits are
/// `break` now, and the `Mov` per iteration that carried the flag through
/// the loop's back edge goes with it --
/// `acvus-interpreter-test/tests/loop_exit_moves.rs` counts them.
fn matcher_break(places: &Places) -> String {
    let Places {
        pat,
        line,
        len: slen,
    } = places;
    format!(
        "\
let i = zero; \
let j = zero; \
let hs = false; \
let star = zero; \
let mark = zero; \
let ok = false; \
while true {{ \
if i < {slen} {{ \
let adv = false; \
if j < plen {{ \
let pj = {pat}[j]; \
if pj == {STAR} {{ hs = true; star = j; mark = i; j = j + one; adv = true; }} \
else {{ if pj == {QMARK} || pj == {line}[i] {{ i = i + one; j = j + one; adv = true; }}; }}; \
}}; \
if !adv {{ \
if hs {{ j = star + one; mark = mark + one; i = mark; }} else {{ break; }}; \
}}; \
}} else {{ \
while j < plen && {pat}[j] == {STAR} {{ j = j + one; }} \
ok = j == plen; \
break; \
}}; \
}} \
"
    )
}

/// The trailing integer field, read backwards. Runs on a matching line only.
fn latency(places: &Places) -> String {
    let Places {
        line, len: slen, ..
    } = places;
    format!(
        "\
let lat = 0; \
let place = 1; \
let k = {slen}; \
let scanning = true; \
while scanning {{ \
if k > zero {{ \
let c = {line}[k - one]; \
if c >= {ZERO_DIGIT} && c <= {NINE_DIGIT} {{ \
lat = lat + (c - {ZERO_DIGIT}) * place; place = place * 10; k = k - one; \
}} else {{ scanning = false; }}; \
}} else {{ scanning = false; }}; \
}} \
"
    )
}

fn tail() -> String {
    format!("count * {SCALE} + total")
}

/// Where the kernel's own body reads its data: the context names, and the
/// `slen` it binds per line.
const KERNEL_PLACES: Places = Places {
    pat: "@pat",
    line: "@lines[li]",
    len: "slen",
};

/// Case 1: the matcher's `while` written into the kernel's `while`.
fn inline_source() -> String {
    format!(
        "{PRELUDE}\
while li < n {{ \
let slen = len(&@lines[li]); \
{matcher} \
if ok {{ {latency} count = count + 1; total = total + lat; }}; \
li = li + one; \
}} \
{tail}",
        matcher = matcher(&KERNEL_PLACES),
        latency = latency(&KERNEL_PLACES),
        tail = tail(),
    )
}

/// Case 2: case 1 with the matcher's two exits written as `break`
/// (RFC-0057). The old case stays: the two are the measurement's pair.
pub fn inline_break_source() -> String {
    format!(
        "{PRELUDE}\
while li < n {{ \
let slen = len(&@lines[li]); \
{matcher} \
if ok {{ {latency} count = count + 1; total = total + lat; }}; \
li = li + one; \
}} \
{tail}",
        matcher = matcher_break(&KERNEL_PLACES),
        latency = latency(&KERNEL_PLACES),
        tail = tail(),
    )
}

/// Case 3: the same matcher as a named closure, one `CallIndirect` per line.
/// The pattern and the line are its two arguments, and the `u64` `plen`,
/// `one` and `zero` the index arithmetic needs are captures of the outer
/// body (RFC-0018: a captured word is a copy).
fn closure_source() -> String {
    format!(
        "{PRELUDE}\
let matches = |pp, ln| -> {{ \
let slen = len(ln); \
{matcher} \
ok \
}}; \
while li < n {{ \
if matches(&@pat, &@lines[li]) {{ \
let slen = len(&@lines[li]); \
{latency} count = count + 1; total = total + lat; \
}}; \
li = li + one; \
}} \
{tail}",
        matcher = matcher(&Places {
            pat: "pp",
            line: "ln",
            len: "slen",
        }),
        latency = latency(&KERNEL_PLACES),
        tail = tail(),
    )
}

/// Case 4: the matcher as a `sync` Rust extern over the corpus the script
/// holds, the pattern and the line crossing as `&[i64]`.
fn slice_source() -> String {
    format!(
        "{PRELUDE}\
while li < n {{ \
if glob_match(&@pat, &@lines[li]) {{ \
let slen = len(&@lines[li]); \
{latency} count = count + 1; total = total + lat; \
}}; \
li = li + one; \
}} \
{tail}",
        latency = latency(&KERNEL_PLACES),
        tail = tail(),
    )
}

/// Cases 5, 6 and 7: the matcher as a Rust extern over the corpus Rust holds,
/// the script naming a line by its index -- `sync` in the caller's frame, then
/// `heavy` on the blocking pool at each of the two effect levels.
fn extern_source(call: &str) -> String {
    format!(
        "{PRELUDE}\
while li < n {{ \
if {call}(li) {{ \
let slen = len(&@lines[li]); \
{latency} count = count + 1; total = total + lat; \
}}; \
li = li + one; \
}} \
{tail}",
        latency = latency(&KERNEL_PLACES),
        tail = tail(),
    )
}

// -- The context --------------------------------------------------------

/// A language `Vec<T>` crosses as the runtime's `Vec<Owned<Rt>>`
/// (`acvus-ext/src/vec.rs`), so a context value of that type is that vector,
/// erased.
fn vec_value(items: Vec<Value>) -> Value {
    // SAFETY: each word is moved in from `items`, and no other holder owns it.
    let items: Vec<Owned<AcvusRuntime>> =
        items.into_iter().map(|item| unsafe { Owned::from_value(acvus_extern::Holding::new(), item) }).collect();
    // SAFETY: the only reader is `Vec<T>`'s `Cross` impl, which materializes
    // a language `Vec` as this exact `Vec<Owned<AcvusRuntime>>`; `context`
    // declares the matching `vec_ty` for both names below.
    unsafe { Value::erase(items) }
}

fn line_value(line: &[i64]) -> Value {
    vec_value(line.iter().copied().map(Value::int).collect())
}

/// Building the data here rather than in the script's prelude keeps the
/// generation out of the measured `execute`: 5.6 M `push` calls at
/// `n = 100 000` would be as large as the kernel itself.
fn context(interner: &Interner, corpus: &Corpus) -> Context {
    let bytes_ty = vec_ty(interner, Ty::I64);
    let lines_ty = vec_ty(interner, bytes_ty.clone());
    let lines_value = vec_value(corpus.lines.iter().map(|line| line_value(line)).collect());
    let entries: [(_, TypedValue); 2] = [
        (interner.intern("lines"), typed(lines_ty, lines_value)),
        (
            interner.intern("pat"),
            typed(bytes_ty, line_value(&corpus.pat)),
        ),
    ];
    entries.into_iter().collect()
}

fn snapshot(interner: &Interner, corpus: &Corpus) -> HashMap<String, (acvus_mir::ty::Ty, Owned<AcvusRuntime>)> {
    split_context(interner, context(interner, corpus)).1
}

// -- The harness --------------------------------------------------------

/// Which executor a case runs on. The `Heavy` case needs a real pool; the
/// rest are one thread and are the cases worth pinning.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exec {
    Sequential,
    Tokio,
}

impl Exec {
    fn executor(self) -> Arc<dyn Executor> {
        match self {
            Exec::Sequential => Arc::new(SequentialExecutor),
            Exec::Tokio => Arc::new(TokioExecutor),
        }
    }

    fn runtime(self, cores: usize) -> Runtime {
        match self {
            Exec::Sequential => tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("a current-thread tokio runtime"),
            Exec::Tokio => tokio::runtime::Builder::new_multi_thread()
                .worker_threads(cores)
                .enable_all()
                .build()
                .expect("a multi-thread tokio runtime"),
        }
    }
}

struct Case {
    name: &'static str,
    source: String,
    exec: Exec,
}

const CASE_NAMES: [&str; 7] = [
    "inline",
    "inline break",
    "closure",
    "sync ext",
    "sync native",
    "heavy pure",
    "heavy opq",
];

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: CASE_NAMES[0],
            source: inline_source(),
            exec: Exec::Sequential,
        },
        Case {
            name: CASE_NAMES[1],
            source: inline_break_source(),
            exec: Exec::Sequential,
        },
        Case {
            name: CASE_NAMES[2],
            source: closure_source(),
            exec: Exec::Sequential,
        },
        Case {
            name: CASE_NAMES[3],
            source: slice_source(),
            exec: Exec::Sequential,
        },
        Case {
            name: CASE_NAMES[4],
            source: extern_source("match_sync"),
            exec: Exec::Sequential,
        },
        Case {
            name: CASE_NAMES[5],
            source: extern_source("match_heavy"),
            exec: Exec::Tokio,
        },
        Case {
            name: CASE_NAMES[6],
            source: extern_source("match_heavy_opaque"),
            exec: Exec::Tokio,
        },
    ]
}

fn case_named(name: &str) -> Case {
    cases()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no case named {name:?}: it is one of {CASE_NAMES:?}"))
}

fn median(mut samples: Vec<Duration>) -> Duration {
    assert!(!samples.is_empty(), "a median needs at least one sample");
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// One `execute`, timed.
struct Run {
    elapsed: Duration,
    answer: Counted,
}

/// Everything a run does before the kernel: the context types, the parse, the
/// compile, the page snapshot and the prepare. It scales with `n` as the
/// kernel does, so the `perf` difference subtracts it as its own side.
fn prepare_run(
    interner: &Interner,
    case: &Case,
    corpus: &Arc<Corpus>,
) -> (InterpreterContext, Interpreter) {
    let context_types = split_context(interner, context(interner, corpus)).0;
    let ast =
        ParsedAst::Script(acvus_ast::parse_script(interner, &case.source).expect("parse error"));
    let cr =
        compile_source_with_externs(interner, ast, &context_types, registries(corpus), Ty::I64);
    execute_compiled(
        interner,
        cr,
        snapshot(interner, corpus),
        case.exec.executor(),
    )
}

fn run_once(rt: &Runtime, interner: &Interner, case: &Case, corpus: &Arc<Corpus>) -> Run {
    let (_shared, mut interp) = prepare_run(interner, case, corpus);
    let start = Instant::now();
    let value = rt.block_on(interp.execute()).expect("the page holds every context the run fetches first");
    let elapsed = start.elapsed();
    Run {
        elapsed,
        answer: Counted::unpack(value.as_int()),
    }
}

struct Timing {
    execute: Duration,
    rust: Duration,
    chunked: Duration,
    answer: Counted,
}

const REPS: NonZeroUsize = reps(3);

const fn reps(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(reps) => reps,
        None => panic!("a size measures at least one rep"),
    }
}

struct Answers {
    script: Counted,
    rust: Counted,
    chunked: Counted,
}

struct Elapsed {
    execute: Duration,
    rust: Duration,
    chunked: Duration,
}

fn measure(rt: &Runtime, case: &Case, corpus: &Arc<Corpus>, cores: usize) -> Timing {
    let lines = corpus.lines.as_slice();
    let pat = corpus.pat.as_slice();
    let interner = Interner::new();
    let run_rep = || {
        let run = run_once(rt, &interner, case, corpus);

        let start = Instant::now();
        let rust = black_box(rust_seq(black_box(lines), black_box(pat)));
        let rust_elapsed = start.elapsed();

        let start = Instant::now();
        let chunked = black_box(rust_chunked(black_box(lines), black_box(pat), cores));
        let chunked_elapsed = start.elapsed();

        (
            Answers {
                script: run.answer,
                rust,
                chunked,
            },
            Elapsed {
                execute: run.elapsed,
                rust: rust_elapsed,
                chunked: chunked_elapsed,
            },
        )
    };

    let (mut answers, _warm_up) = run_rep();
    let mut execute = Vec::with_capacity(REPS.get());
    let mut rust = Vec::with_capacity(REPS.get());
    let mut chunked = Vec::with_capacity(REPS.get());
    for _ in 0..REPS.get() {
        let (this, elapsed) = run_rep();
        answers = this;
        execute.push(elapsed.execute);
        rust.push(elapsed.rust);
        chunked.push(elapsed.chunked);
    }

    let Answers {
        script: script_answer,
        rust: rust_answer,
        chunked: chunked_answer,
    } = answers;
    assert!(
        script_answer == rust_answer,
        "{}: script produced {script_answer:?}, Rust produced {rust_answer:?}",
        case.name
    );
    assert!(
        chunked_answer == rust_answer,
        "{}: chunked Rust produced {chunked_answer:?}, sequential Rust produced {rust_answer:?}",
        case.name
    );
    Timing {
        execute: median(execute),
        rust: median(rust),
        chunked: median(chunked),
        answer: rust_answer,
    }
}

/// Which machine a `perf stat` run counts, and whether it runs the kernel at
/// all. A run under `perf` executes one side alone, so this path makes no
/// comparison; that check is `measure`'s. `ScriptSetup` and `RustSetup` do
/// everything their kernel side does except the kernel, so the two-size
/// difference of a side minus the difference of its setup is the kernel.
#[derive(Clone, Copy)]
enum Side {
    Script,
    ScriptSetup,
    Rust,
    RustSetup,
}

impl Side {
    fn named(name: &str) -> Side {
        match name {
            "script" => Side::Script,
            "script-setup" => Side::ScriptSetup,
            "rust" => Side::Rust,
            "rust-setup" => Side::RustSetup,
            other => panic!(
                "no side named {other:?}: it is \"script\", \"script-setup\", \"rust\" or \"rust-setup\""
            ),
        }
    }
}

/// Runs a `perf` run makes. The per-line figure is the difference between two
/// sizes, so the warm-up cancels.
const PERF_REPS: usize = 4;

fn perf_run(rt: &Runtime, case: &Case, n: usize, side: Side) {
    let corpus = Corpus::of_size(n);
    let answer = match side {
        Side::Script => {
            let interner = Interner::new();
            let mut answer = Counted::ZERO;
            for _ in 0..PERF_REPS {
                answer = run_once(rt, &interner, case, &corpus).answer;
            }
            Some(answer)
        }
        Side::ScriptSetup => {
            let interner = Interner::new();
            for _ in 0..PERF_REPS {
                black_box(prepare_run(&interner, case, &corpus));
            }
            None
        }
        Side::Rust => {
            let mut answer = Counted::ZERO;
            for _ in 0..PERF_REPS {
                answer = black_box(rust_seq(black_box(&corpus.lines), black_box(&corpus.pat)));
            }
            Some(answer)
        }
        Side::RustSetup => {
            black_box(corpus.lines.len());
            None
        }
    };
    match answer {
        Some(answer) => println!(
            "{} n={n} reps={PERF_REPS} matched={} total={}",
            case.name, answer.count, answer.total
        ),
        None => println!("{} n={n} reps={PERF_REPS} setup only", case.name),
    }
}

// -- The op list --------------------------------------------------------

fn print_part(part: &PartListing, indent: usize) {
    let pad = " ".repeat(indent);
    println!("{pad}{}: {} ops", part.part, part.ops.len());
    for op in &part.ops {
        println!("{pad}  {op}");
    }
    for region in &part.regions {
        println!("{pad}  region {}", region.name);
        for owned in &region.owns {
            print_part(owned, indent + 4);
        }
    }
}

fn print_listing(blocks: &[BlockListing]) {
    for (i, block) in blocks.iter().enumerate() {
        println!("block {i}: {} ops, end {}", block.ops.len(), block.end);
        for op in &block.ops {
            println!("  {op}");
        }
        for region in &block.regions {
            println!("  region {}", region.name);
            for owned in &region.owns {
                print_part(owned, 4);
            }
        }
    }
}

/// The listing does not depend on the data's size, only on its types, so the
/// op-list paths compile against four lines.
const LISTING_LINES: usize = 4;

fn oplist_corpus() -> Arc<Corpus> {
    Corpus::of_size(LISTING_LINES)
}

fn oplist(name: &str) {
    let interner = Interner::new();
    let corpus = oplist_corpus();
    let blocks = script_listing_with_externs(
        &interner,
        &case_named(name).source,
        context(&interner, &corpus),
        registries(&corpus),
        Ty::I64,
    );
    println!("== oplist: {name}");
    print_listing(&blocks);
}

fn mir(name: &str) {
    let interner = Interner::new();
    let source = case_named(name).source;
    let corpus = oplist_corpus();
    let context_types = split_context(&interner, context(&interner, &corpus)).0;
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, &source).expect("parse error"));
    let cr =
        compile_source_with_externs(&interner, ast, &context_types, registries(&corpus), Ty::I64);
    println!("== mir: {name}");
    let entry = cr.modules.get(&cr.entry_qref).expect("the entry module");
    println!("{}", acvus_mir::printer::dump_with(&interner, entry));
    for (qref, module) in &cr.modules {
        if *qref != cr.entry_qref {
            println!("-- module {qref:?}");
            println!("{}", acvus_mir::printer::dump_with(&interner, module));
        }
    }
}

// -- Main ---------------------------------------------------------------

const SIZES: [usize; 2] = [10_000, 100_000];

fn main() {
    if let Ok(name) = std::env::var("LOGS_OPLIST") {
        oplist(&name);
        return;
    }
    if let Ok(name) = std::env::var("LOGS_SOURCE") {
        println!("{}", case_named(&name).source);
        return;
    }
    if let Ok(name) = std::env::var("LOGS_MIR") {
        mir(&name);
        return;
    }

    let cores = std::thread::available_parallelism()
        .expect("this bench reports the core count it ran on; without it there is no measurement")
        .get();
    let only = std::env::var("LOGS_CASE").ok();
    let sizes: Vec<usize> = match std::env::var("LOGS_N").ok() {
        Some(n) => vec![n.parse().expect("LOGS_N is an integer")],
        None => SIZES.to_vec(),
    };

    if let Ok(side) = std::env::var("LOGS_ONLY") {
        let side = Side::named(&side);
        let case = case_named(&only.expect("LOGS_ONLY names a case too"));
        let rt = case.exec.runtime(cores);
        for n in sizes {
            perf_run(&rt, &case, n, side);
        }
        return;
    }

    println!(
        "cores: {cores}; sequential cases: SequentialExecutor on a current-thread \
         runtime (pin them with taskset); heavy case: TokioExecutor on a \
         multi_thread runtime with {cores} workers, threads not pinned"
    );
    println!("pattern: {PATTERN:?}");
    println!();
    println!(
        "{:>10} {:>8} {:>8} {:>12} {:>12} {:>12} {:>9} {:>9} {:>9}",
        "case",
        "n",
        "matched",
        "execute/us",
        "rust/us",
        "chunk/us",
        "ratio",
        "ns/line",
        "rust/line"
    );
    for case in cases()
        .iter()
        .filter(|c| only.as_deref().is_none_or(|o| c.name == o))
    {
        let rt = case.exec.runtime(cores);
        for n in &sizes {
            let corpus = Corpus::of_size(*n);
            let Timing {
                execute,
                rust,
                chunked,
                answer,
            } = measure(&rt, case, &corpus, cores);
            println!(
                "{:>10} {:>8} {:>8} {:>12.1} {:>12.1} {:>12.1} {:>9.1} {:>9.1} {:>9.2}",
                case.name,
                n,
                answer.count,
                micros(execute),
                micros(rust),
                micros(chunked),
                micros(execute) / micros(rust),
                execute.as_secs_f64() * 1e9 / *n as f64,
                rust.as_secs_f64() * 1e9 / *n as f64,
            );
        }
    }
}
