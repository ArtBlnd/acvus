//! Object and enum shapes: the per-operation cost of a field access, an
//! object construction and a variant match, next to the same loop in Rust.
//!
//! The Rust references use a struct and an enum, so their field access is a
//! fixed offset and their match one tag test. Today's `Value` holds an
//! object as a boxed `FxHashMap<Astr, Value>` and an enum as a boxed
//! `Variant<Value>`, so the ratio is the distance between the two layouts,
//! not interpreter overhead alone.
//!
//! These timings hold only under one pinned core and a fixed load base;
//! `benches/README.md` states the protocol.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::{Owned, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::listing::{BlockListing, PartListing, script_listing_with_externs};
use acvus_interpreter_test::{
    Context, compile_source_with_externs, execute_compiled, split_context, typed,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use tokio::runtime::Runtime;

#[extern_fn(effect = pure)]
fn some_of(i: i64) -> Option<i64> {
    if i % 2 == 0 { Some(i) } else { None }
}

/// The aggregate an extern returns. The two rows below run one declaration
/// at two destinations: the frame run the placement gives a result that
/// stays in the body, and the heap object rule 4 realizes one that crosses
/// back by value into (RFC-0050 rules 4 and 6).
#[derive(acvus_extern::TyArg)]
pub struct Made {
    x: i64,
    y: i64,
}

#[extern_fn(effect = pure)]
fn made_of(a: i64, b: i64) -> Made {
    Made { x: a, y: b }
}

#[extern_fn(effect = pure)]
fn sum_made(m: Made) -> i64 {
    m.x + m.y
}

/// The aggregate a handler borrows, and the projection RFC-0050 rule 6 gives
/// it. Three fields, so the row measures a table of three positions against
/// the same three field reads in Rust.
#[derive(acvus_extern::TyArg)]
#[projection]
pub struct Row {
    a: i64,
    b: i64,
    c: i64,
}

#[extern_fn(effect = pure)]
fn sum_row(r: RowRef<'_>) -> i64 {
    *r.a + *r.b + *r.c
}

/// The enum a handler borrows: the dispatch is one `u64` compare per arm
/// against the site datum, where `sum_row`'s is three positions read off it.
#[derive(acvus_extern::TyArg)]
#[projection]
pub enum Tagged {
    Zero,
    One(i64),
    Two(i64),
}

#[extern_fn(effect = pure)]
fn tagged_of(t: TaggedRef<'_>) -> i64 {
    match t {
        TaggedRef::Zero => 0,
        TaggedRef::One(n) => *n,
        TaggedRef::Two(n) => *n,
    }
}

fn with_projections() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = std_only();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [sum_row, tagged_of],
    });
    regs
}

fn std_only() -> Vec<Registry<AcvusRuntime>> {
    acvus_ext::std_registries::<AcvusRuntime>()
}

fn with_made() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = std_only();
    regs.push(extern_registry! {
        ns: "bench",
        types: [],
        fns: [made_of, sum_made],
    });
    regs
}

fn with_some_of() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = std_only();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [some_of],
    });
    regs
}

const FIELD_READ: &str = "let p = { x: 1, y: 2, }; let acc = 0; let i = 0; while i < @n { acc = acc + p.x + p.y; i = i + 1; } acc";
const FIELD_WRITE: &str =
    "let p = { x: 0, y: 0, }; let i = 0; while i < @n { p.x = p.x + i; i = i + 1; } p.x";
const CONSTRUCT: &str = "let acc = 0; let i = 0; while i < @n { let q = { x: i, y: i + 1, }; acc = acc + q.x; i = i + 1; } acc";

/// The language has no multi-arm `match`: a refutable pattern in script
/// position is the match-bind statement `Pattern = Expr { body };`, so the
/// two arms are two statements and both tags are tested every iteration.
const ENUM_MATCH: &str = "let acc = 0; let i = 0; while i < @n { let e = if i % 2 == 0 { E::A(i) } else { E::B(i + 1) }; match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v; } }; i = i + 1; } acc";
/// Three arms, and a scrutinee whose three edges each carry a constant tag.
const ENUM_MATCH_THREE: &str = "let acc = 0; let i = 0; while i < @n { let e = match i % 3 { 0 => E::A(i), 1 => E::B(i + 1), _ => E::C(i + 2) }; match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v; }, E::C(v) => { acc = acc + v; } }; i = i + 1; } acc";
/// The three arms of `ENUM_MATCH_THREE` with the scrutinee held across an
/// inner loop, so its address is taken and `optimize::sroa` leaves the
/// aggregate to the machine. Every row above it holds no aggregate at all by
/// the time `prepare` sees it, which is why they do not move when the machine's
/// aggregate representation does.
const ENUM_MATCH_HELD: &str = "let acc = 0; let i = 0; while i < @n { \
let e = if i % 3 == 0 { E::A(i) } else { if i % 3 == 1 { E::B(i + 1) } else { E::C(i + 2) } }; \
let j = 0; while j < 1 { match e { E::A(v) => { acc = acc + v; }, \
E::B(v) => { acc = acc + v; }, E::C(v) => { acc = acc + v; } }; j = j + 1; } \
i = i + 1; } acc";
const ENUM_MATCH_HEAPED: &str = "let acc = 0; let i = 0; while i < @n { \
let e = if i % 3 == 0 { E::A(i) } else { if i % 3 == 1 { E::B(i + 1) } else { E::C(i + 2) } }; \
let v = [e]; let m = len(&v); let z = m - m; \
match &v[z] { E::A(w) => { acc = acc + *w; }, E::B(w) => { acc = acc + *w; }, \
E::C(w) => { acc = acc + *w; } }; i = i + 1; } acc";
/// `ENUM_MATCH_HELD`'s shape at the one variant type the language names the
/// tags of. The scrutinee is held across an inner loop for the same reason it
/// is there: without it `optimize::sroa` scalarizes the `Result` in the MIR and
/// no aggregate reaches the machine.
const RESULT_MATCH: &str = "let acc = 0; let i = 0; while i < @n { \
let r = if i % 2 == 0 { Ok(i) } else { Err(i + 1) }; \
let j = 0; while j < 1 { match r { Ok(v) => { acc = acc + v; }, \
Err(v) => { acc = acc + v; } }; j = j + 1; } \
i = i + 1; } acc";
/// `ENUM_MATCH_HEAPED`'s shape at the one variant type the language names the
/// tags of: the `Result` escapes into an array, so it is built and matched on
/// the heap rather than in a run.
const RESULT_MATCH_HEAPED: &str = "let acc = 0; let i = 0; while i < @n { \
let r = if i % 2 == 0 { Ok(i) } else { Err(i + 1) }; \
let v = [r]; let m = len(&v); let z = m - m; \
match &v[z] { Ok(w) => { acc = acc + *w; }, Err(w) => { acc = acc + *w; } }; i = i + 1; } acc";
const OPTION_MATCH: &str = "let i = 0; let acc = 0; while i < @n { if let Some(v) = some_of(i) { acc = acc + v; }; i = i + 1; } acc";

/// `v[i]` takes a `u64` index and integer literals are `i64`, so the index
/// counter is derived from `len`, and the sweep over the thousand objects
/// repeats `n / 1000` times to reach `n` iterations of the inner body.
/// The result stays in the body, so it takes the registers the run
/// allocation placed for it and the call writes them.
const CONSTRUCT_VIA_EXTERN: &str = "let acc = 0; let i = 0; while i < @n { \
let p = made_of(i, i + 1); acc = acc + p.x + p.y; i = i + 1; } acc";
/// The result crosses back into a handler by value, which is one of rule 4's
/// escape sites, so the call writes the flat body of a heap object instead.
/// The row also pays one more extern call than the row above it, so the
/// distance between the two bounds the destination's cost from above rather
/// than measuring it alone; the allocation counts are what measure the
/// destination.
const CONSTRUCT_VIA_EXTERN_HEAPED: &str = "let acc = 0; let i = 0; while i < @n { \
let p = made_of(i, i + 1); acc = acc + sum_made(p); i = i + 1; } acc";

/// The object is lent, so `prepare::runs::Sites` refuses its web and it is a
/// heap object the handler borrows in place. The Rust reference reads the
/// same three fields behind a reference.
const PROJECT_VIA_EXTERN: &str = "let acc = 0; let i = 0; while i < @n { \
let p = { a: i, b: i + 1, c: i + 2, }; acc = acc + sum_row(&p); i = i + 1; } acc";
/// The enum half of the row above: three variants meeting at the `if`, lent
/// to a handler that dispatches on the tag.
const SWITCH_VIA_EXTERN: &str = "let acc = 0; let i = 0; while i < @n { \
let e = if i % 3 == 0 { Tagged::Zero } else { if i % 3 == 1 { Tagged::One(i) } else { Tagged::Two(i + 1) } }; \
acc = acc + tagged_of(&e); i = i + 1; } acc";

const VEC_OF_OBJECTS: &str = "let v = range(0, 1000) | map(|k| -> { x: k, y: k + 1, }) | collect; let m = len(&v); let one = m / m; let acc = 0; let r = 0; while r < @n / 1000 { let i = m - m; while i < m { acc = acc + v[i].x; i = i + one; } r = r + 1; } acc";

struct Point {
    x: i64,
    y: i64,
}

struct Row3 {
    a: i64,
    b: i64,
    c: i64,
}

enum E {
    A(i64),
    B(i64),
}

enum E3 {
    A(i64),
    B(i64),
    C(i64),
}

struct Case {
    name: &'static str,
    source: &'static str,
    registries: fn() -> Vec<Registry<AcvusRuntime>>,
    rust: fn(i64) -> f64,
    read: fn(&Value) -> f64,
    /// What the host declares `main` returns for this source (RFC-0054).
    ret: Ty,
}

fn rust_field_read(n: i64) -> f64 {
    let p = Point { x: 1, y: 2 };
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let p = black_box(&p);
        acc += p.x + p.y;
        i += 1;
    }
    acc as f64
}

fn rust_field_write(n: i64) -> f64 {
    let mut p = Point { x: 0, y: 0 };
    let mut i = 0i64;
    while i < n {
        p.x = p.x + black_box(i);
        i += 1;
    }
    p.x as f64
}

fn rust_construct(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let q = black_box(Point { x: i, y: i + 1 });
        acc += q.x;
        i += 1;
    }
    acc as f64
}

fn rust_construct_via_extern(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let p = black_box(Point { x: i, y: i + 1 });
        acc += p.x + p.y;
        i += 1;
    }
    acc as f64
}

fn rust_enum_match(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let e = if black_box(i) % 2 == 0 {
            E::A(i)
        } else {
            E::B(i + 1)
        };
        match e {
            E::A(v) => acc += v,
            E::B(v) => acc += v,
        }
        i += 1;
    }
    acc as f64
}

fn rust_enum_match_three(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let e = match black_box(i) % 3 {
            0 => E3::A(i),
            1 => E3::B(i + 1),
            _ => E3::C(i + 2),
        };
        match e {
            E3::A(v) => acc += v,
            E3::B(v) => acc += v,
            E3::C(v) => acc += v,
        }
        i += 1;
    }
    acc as f64
}

fn rust_enum_match_held(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let e = match black_box(i) % 3 {
            0 => E3::A(i),
            1 => E3::B(i + 1),
            _ => E3::C(i + 2),
        };
        let mut j = 0i64;
        while j < 1 {
            match &e {
                E3::A(v) => acc += *v,
                E3::B(v) => acc += *v,
                E3::C(v) => acc += *v,
            }
            j += 1;
        }
        i += 1;
    }
    acc as f64
}

/// The array is what `prepare::runs::Sites` refuses, so this row's variant is
/// realized on the heap where `enum match held`'s takes a run. The Rust
/// reference carries the same `Vec` so the ratio compares like with like: the
/// `Vec` passes through `black_box`, so LLVM keeps its allocation, the store
/// into it and the read back through it, which it removes otherwise.
fn rust_enum_match_heaped(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let e = match black_box(i) % 3 {
            0 => E3::A(i),
            1 => E3::B(i + 1),
            _ => E3::C(i + 2),
        };
        let v = black_box(vec![e]);
        match &v[0] {
            E3::A(w) => acc += *w,
            E3::B(w) => acc += *w,
            E3::C(w) => acc += *w,
        }
        i += 1;
    }
    acc as f64
}

fn rust_result_match(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let r: Result<i64, i64> = match black_box(i) % 2 {
            0 => Ok(i),
            _ => Err(i + 1),
        };
        let mut j = 0i64;
        while j < 1 {
            match &r {
                Ok(v) => acc += *v,
                Err(v) => acc += *v,
            }
            j += 1;
        }
        i += 1;
    }
    acc as f64
}

fn rust_project_via_extern(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let p = Row3 {
            a: i,
            b: i + 1,
            c: i + 2,
        };
        let p = black_box(&p);
        acc += p.a + p.b + p.c;
        i += 1;
    }
    acc as f64
}

/// `rust_enum_match_heaped`'s `Vec`, at `Result`.
fn rust_result_match_heaped(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let r: Result<i64, i64> = match black_box(i) % 2 {
            0 => Ok(i),
            _ => Err(i + 1),
        };
        let v = black_box(vec![r]);
        match &v[0] {
            Ok(w) => acc += *w,
            Err(w) => acc += *w,
        }
        i += 1;
    }
    acc as f64
}

fn rust_switch_via_extern(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let e = match black_box(i) % 3 {
            0 => E3::A(0),
            1 => E3::B(i),
            _ => E3::C(i + 1),
        };
        acc += match &e {
            E3::A(_) => 0,
            E3::B(v) => *v,
            E3::C(v) => *v,
        };
        i += 1;
    }
    acc as f64
}

fn rust_option_match(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        if let Some(v) = some_of(black_box(i)) {
            acc += v;
        }
        i += 1;
    }
    acc as f64
}

fn rust_vec_of_objects(n: i64) -> f64 {
    let v: Vec<Point> = (0..1000).map(|k| Point { x: k, y: k + 1 }).collect();
    let m = v.len();
    let mut acc = 0i64;
    let mut r = 0i64;
    while r < n / 1000 {
        let mut i = 0usize;
        while i < m {
            acc += black_box(&v)[i].x;
            i += 1;
        }
        r += 1;
    }
    acc as f64
}

fn context(interner: &Interner, n: i64) -> Context {
    [(interner.intern("n"), typed(Ty::I64, Value::int(n)))]
        .into_iter()
        .collect()
}

fn snapshot(interner: &Interner, n: i64) -> HashMap<String, (acvus_mir::ty::Ty, Owned<AcvusRuntime>)> {
    split_context(interner, context(interner, n)).1
}

struct Timing {
    execute: Duration,
    rust: Duration,
}

struct Size {
    n: i64,
    reps: NonZeroUsize,
}

const fn reps(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(reps) => reps,
        None => panic!("a size measures at least one rep"),
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn measure(rt: &Runtime, case: &Case, size: &Size) -> Timing {
    let Size { n, reps } = *size;
    let interner = Interner::new();
    let context_types = split_context(&interner, context(&interner, n)).0;
    let run_script = || {
        let ast = ParsedAst::Script(
            acvus_ast::parse_script(&interner, case.source).expect("parse error"),
        );
        let cr = compile_source_with_externs(
            &interner,
            ast,
            &context_types,
            (case.registries)(),
            case.ret.clone(),
        );
        let (_shared, mut interp) = execute_compiled(
            &interner,
            cr,
            snapshot(&interner, n),
            Arc::new(SequentialExecutor),
        );
        let start = Instant::now();
        let value = rt.block_on(interp.execute()).expect("the seeds hold every context the run fetches");
        ((case.read)(&value), start.elapsed())
    };
    let run_rust = || {
        let start = Instant::now();
        let value = black_box((case.rust)(black_box(n)));
        (value, start.elapsed())
    };

    let (mut script_value, _warm_up) = run_script();
    let mut execute = Vec::with_capacity(reps.get());
    for _ in 0..reps.get() {
        let (value, elapsed) = run_script();
        script_value = value;
        execute.push(elapsed);
    }

    let (mut rust_value, _warm_up) = run_rust();
    let mut rust = Vec::with_capacity(reps.get());
    for _ in 0..reps.get() {
        let (value, elapsed) = run_rust();
        rust_value = value;
        rust.push(elapsed);
    }

    assert!(
        script_value == rust_value,
        "{} n={n}: script produced {script_value}, Rust produced {rust_value}",
        case.name
    );
    Timing {
        execute: median(execute),
        rust: median(rust),
    }
}

fn cases_named<'a>(cases: &'a [Case], var: &str) -> Vec<&'a Case> {
    let Ok(name) = std::env::var(var) else {
        return cases.iter().collect();
    };
    let picked: Vec<&Case> = cases.iter().filter(|c| c.name == name).collect();
    if picked.is_empty() {
        eprintln!("{var}={name:?} names no case. The cases are:");
        for case in cases {
            eprintln!("  {}", case.name);
        }
        std::process::exit(1);
    }
    picked
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");
    let cases = [
        Case {
            name: "field read",
            source: FIELD_READ,
            registries: std_only,
            rust: rust_field_read,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "field write",
            source: FIELD_WRITE,
            registries: std_only,
            rust: rust_field_write,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "construct",
            source: CONSTRUCT,
            registries: std_only,
            rust: rust_construct,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "enum match",
            source: ENUM_MATCH,
            registries: std_only,
            rust: rust_enum_match,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "enum match three",
            source: ENUM_MATCH_THREE,
            registries: std_only,
            rust: rust_enum_match_three,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "enum match held",
            source: ENUM_MATCH_HELD,
            registries: std_only,
            rust: rust_enum_match_held,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "enum match heaped",
            source: ENUM_MATCH_HEAPED,
            registries: std_only,
            rust: rust_enum_match_heaped,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "result match",
            source: RESULT_MATCH,
            registries: std_only,
            rust: rust_result_match,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "result match heaped",
            source: RESULT_MATCH_HEAPED,
            registries: std_only,
            rust: rust_result_match_heaped,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "option match",
            source: OPTION_MATCH,
            registries: with_some_of,
            rust: rust_option_match,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "construct via extern",
            source: CONSTRUCT_VIA_EXTERN,
            registries: with_made,
            rust: rust_construct_via_extern,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "construct via extern heaped",
            source: CONSTRUCT_VIA_EXTERN_HEAPED,
            registries: with_made,
            rust: rust_construct_via_extern,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "project via extern",
            source: PROJECT_VIA_EXTERN,
            registries: with_projections,
            rust: rust_project_via_extern,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "switch via extern",
            source: SWITCH_VIA_EXTERN,
            registries: with_projections,
            rust: rust_switch_via_extern,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "vec of objects",
            source: VEC_OF_OBJECTS,
            registries: std_only,
            rust: rust_vec_of_objects,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
    ];
    if std::env::var_os("SHAPES_OPLIST").is_some() {
        for case in cases_named(&cases, "SHAPES_OPLIST") {
            oplist(case);
        }
        return;
    }
    let selected = cases_named(&cases, "SHAPES_CASE");
    println!(
        "{:>14} {:>10} {:>14} {:>12} {:>12} {:>14}",
        "case", "n", "execute/us", "rust/us", "ratio", "ns/iteration"
    );
    for case in selected {
        for size in [
            Size {
                n: 100_000,
                reps: reps(19),
            },
            Size {
                n: 1_000_000,
                reps: reps(4),
            },
        ] {
            let Timing { execute, rust } = measure(&rt, case, &size);
            let n = size.n;
            println!(
                "{:>14} {:>10} {:>14.1} {:>12.2} {:>12.1} {:>14.1}",
                case.name,
                n,
                micros(execute),
                micros(rust),
                micros(execute) / micros(rust),
                execute.as_secs_f64() * 1e9 / n as f64
            );
        }
    }
}

// -- the op list --------------------------------------------------

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

/// The prepared operations of one case, as `benches/programs.rs` prints
/// them: what a shape costs is read here and measured above.
fn oplist(case: &Case) {
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        case.source,
        context(&interner, 1_000_000),
        (case.registries)(),
        case.ret.clone(),
    );
    println!("== oplist: {}", case.name);
    print_listing(&blocks);
}
