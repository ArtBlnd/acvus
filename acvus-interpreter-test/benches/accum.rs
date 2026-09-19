//! Accumulation loops: the per-operation cost of the machine with no
//! extern call in the loop, next to the same loop in Rust.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::{Owned, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
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

#[extern_fn(effect = pure)]
fn id_of(i: i64) -> i64 {
    i
}

#[extern_fn(effect = pure)]
fn even_of(i: i64) -> bool {
    i % 2 == 0
}

/// One variable apart: the same body, the same argument, the same work,
/// and the result crossing back as the two registers a view occupies or as
/// the one a `String` does (RFC-0062 Decision 4). The owned side pays the
/// copy and the `as_str` a later `&str` parameter then needs; that pair is
/// what a script paid for `string::trim` before the view return landed.
#[extern_fn(effect = pure)]
fn view_trim(s: &str) -> &str {
    s.trim()
}

#[extern_fn(effect = pure)]
fn owned_trim(s: &str) -> String {
    s.trim().to_owned()
}

#[extern_fn(effect = pure)]
fn view_cut(s: &str, from: u64, to: u64) -> &str {
    &s[from as usize..to as usize]
}

#[extern_fn(effect = pure)]
fn owned_cut(s: &str, from: u64, to: u64) -> String {
    s[from as usize..to as usize].to_owned()
}

fn std_only() -> Vec<Registry<AcvusRuntime>> {
    acvus_ext::std_registries::<AcvusRuntime>()
}

fn with_some_of() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = std_only();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [some_of, id_of, even_of],
    });
    regs
}

fn with_views() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = std_only();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [view_trim, owned_trim, view_cut, owned_cut],
    });
    regs
}

const TEXT: &str = "  hello world  ";

const TRIM_VIEW: &str = "let acc = 0; let i = 0; while i < @n { let v = view_trim(\"  hello world  \"); acc = acc + len(&v) as i64; i = i + 1; } acc";
const TRIM_OWNED: &str = "let acc = 0; let i = 0; while i < @n { let v = owned_trim(\"  hello world  \"); acc = acc + len(&v) as i64; i = i + 1; } acc";
const CUT_VIEW: &str = "let acc = 0; let i = 0; while i < @n { let v = view_cut(\"  hello world  \", 2, 7); acc = acc + len(&v) as i64; i = i + 1; } acc";
const CUT_OWNED: &str = "let acc = 0; let i = 0; while i < @n { let v = owned_cut(\"  hello world  \", 2, 7); acc = acc + len(&v) as i64; i = i + 1; } acc";

fn rust_trim(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        acc += black_box(TEXT).trim().len() as i64;
        i += 1;
    }
    acc as f64
}

fn rust_cut(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        acc += black_box(TEXT)[2..7].len() as i64;
        i += 1;
    }
    acc as f64
}

const INT_WHILE: &str = "let acc = 0; let i = 0; while i < @n { acc = acc + i; i = i + 1; } acc";
const FLOAT_WHILE: &str =
    "let acc = 0.0; let i = 0; while i < @n { acc = acc + i as f64; i = i + 1; } acc";
const RANGE_SUM: &str = "range(0, @n) | sum";
const MAP_ID_SUM: &str = "range(0, @n) | map(|x| -> x) | sum";
const MAP_ADD_SUM: &str = "range(0, @n) | map(|x| -> x + 1) | sum";
const MAP_CAP_SUM: &str = "let k = 1; range(0, @n) | map(|x| -> x + k) | sum";
/// The three traversals RFC-0057 replaces a pipeline with: one `For` region
/// per loop, and no `Iter` extern per element.
const FOR_RANGE: &str = "let acc = 0; for i in 0..@n { acc = acc + i; } acc";
/// The joints path of RFC-0057 Decision 4 beside the region: the `break` is
/// never taken, so the two rows run the same traversal and the difference is
/// what a terminator-shaped header costs over a region.
const FOR_RANGE_BREAK: &str =
    "let acc = 0; for i in 0..@n { if i == @n { break; }; acc = acc + i; } acc";
/// The `else if` chain over variables, which is the shape every example's
/// loop body has. The references at the tail keep `a` and `b` variables
/// rather than block parameters, and that is what decides the shape: with
/// block parameters the inner join carries them and stays a block of its
/// own, while without them `optimize::forward` collapses that join into the
/// outer one, so the inner branch names the same join as the branch above
/// it. Before the recognizer read that shape the whole body ran as joints,
/// at a dispatch per block edge per iteration.
const FOR_IF_CHAIN: &str = "let a = 0; let b = 0; for i in 0..@n { \
    if i % 3 == 0 { a = a + 1; } else if i % 3 == 1 { b = b + 1; } else { a = a + 2; }; \
    } let ra = &a; let rb = &b; *ra + *rb";
const FOR_SLICE: &str =
    "let v = range(0, @n) | collect; let acc = 0; for x in &v { acc = acc + *x; } acc";
const FOR_SLICE_ADD: &str =
    "let v = range(0, @n) | collect; let acc = 0; for x in &v { acc = acc + *x + 1; } acc";

const EXTERN_WHILE: &str =
    "let i = 0; let acc = 0; while i < @n { acc = acc + id_of(i); i = i + 1; } acc";
const BRANCH_WHILE: &str =
    "let i = 0; let acc = 0; while i < @n { if even_of(i) { acc = acc + i; }; i = i + 1; } acc";
const OPTION_WHILE: &str = "let i = 0; let acc = 0; while i < @n { if let Some(v) = some_of(i) { acc = acc + v; }; i = i + 1; } acc";

/// The language's iteration idiom over a container: a `while let` head
/// whose `next` is one extern call (RFC-0046).
const WHILE_LET_VEC: &str = "let v = range(0, @n) | collect; let it = as_iter(&v); let acc = 0; while let Some(x) = next(&mut it) { acc = acc + *x; } acc";
/// The same head over a lazy pipeline of synchronous stages.
const WHILE_LET_MAP: &str = "let it = range(0, @n) | map(|x| -> x + 1); let acc = 0; while let Some(x) = next(&mut it) { acc = acc + x; } acc";

/// The measurement RFC-0052 §"a synchronous call is an operation" is judged
/// against: a `Sync` call to a user function, once per iteration.
const CALL_WHILE: &str = "let step = |x| -> x + 1; let i = 0; while i < @n { i = step(i); } i";
/// A value diamond in the body: the collatz step, both arms non-empty and
/// the join carrying one parameter.
const COLLATZ_WHILE: &str = "let i = 0; let acc = 0; while i < @n { let d = if i % 2 == 0 { i / 2 } else { i * 3 + 1 }; acc = acc + d; i = i + 1; } acc";
/// A three-way statement diamond: `else if` joins on two parameters, the
/// shape `grade_classifier` runs in its `while let` body.
const GRADE_WHILE: &str = "let i = 0; let a = 0; let b = 0; while i < @n { if i % 3 == 0 { a = a + 1; } else if i % 3 == 1 { b = b + 1; } else { a = a + 2; }; i = i + 1; } a + b";
struct Case {
    name: &'static str,
    source: &'static str,
    registries: fn() -> Vec<Registry<AcvusRuntime>>,
    rust: fn(i64) -> f64,
    read: fn(&Value) -> f64,
    /// What the host declares `main` returns for this source (RFC-0054).
    ret: Ty,
}

fn rust_int_while(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        acc += black_box(i);
        i += 1;
    }
    acc as f64
}

fn rust_float_while(n: i64) -> f64 {
    let mut acc = 0.0f64;
    let mut i = 0i64;
    while i < n {
        acc += black_box(i) as f64;
        i += 1;
    }
    acc
}

fn rust_range_sum(n: i64) -> f64 {
    (0..n).map(black_box).sum::<i64>() as f64
}

fn rust_map_id_sum(n: i64) -> f64 {
    (0..n).map(black_box).map(|x| x).sum::<i64>() as f64
}

fn rust_map_add_sum(n: i64) -> f64 {
    (0..n).map(black_box).map(|x| x + 1).sum::<i64>() as f64
}

fn rust_map_cap_sum(n: i64) -> f64 {
    let k = 1i64;
    (0..n).map(black_box).map(|x| x + k).sum::<i64>() as f64
}

fn rust_for_range(n: i64) -> f64 {
    let mut acc = 0i64;
    for i in 0..n {
        acc += black_box(i);
    }
    acc as f64
}

fn rust_for_range_break(n: i64) -> f64 {
    let mut acc = 0i64;
    for i in 0..n {
        if i == n {
            break;
        }
        acc += black_box(i);
    }
    acc as f64
}

fn rust_for_if_chain(n: i64) -> f64 {
    let mut a = 0i64;
    let mut b = 0i64;
    for i in 0..n {
        if black_box(i) % 3 == 0 {
            a += 1;
        } else if i % 3 == 1 {
            b += 1;
        } else {
            a += 2;
        }
    }
    (a + b) as f64
}

fn rust_for_slice(n: i64) -> f64 {
    let v: Vec<i64> = (0..n).collect();
    let mut acc = 0i64;
    for x in &v {
        acc += black_box(*x);
    }
    acc as f64
}

fn rust_for_slice_add(n: i64) -> f64 {
    let v: Vec<i64> = (0..n).collect();
    let mut acc = 0i64;
    for x in &v {
        acc += black_box(*x) + 1;
    }
    acc as f64
}

fn rust_extern_while(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        acc += id_of(black_box(i));
        i += 1;
    }
    acc as f64
}

fn rust_branch_while(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        if even_of(black_box(i)) {
            acc += i;
        }
        i += 1;
    }
    acc as f64
}

fn rust_option_while(n: i64) -> f64 {
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

fn rust_while_let_vec(n: i64) -> f64 {
    let v: Vec<i64> = (0..n).collect();
    let mut it = v.iter();
    let mut acc = 0i64;
    while let Some(x) = it.next() {
        acc += black_box(*x);
    }
    acc as f64
}

fn rust_while_let_map(n: i64) -> f64 {
    let mut it = (0..n).map(|x| x + 1);
    let mut acc = 0i64;
    while let Some(x) = it.next() {
        acc += black_box(x);
    }
    acc as f64
}

fn step(i: i64) -> i64 {
    i + 1
}

fn rust_call_while(n: i64) -> f64 {
    let mut i = 0i64;
    while i < n {
        i = step(black_box(i));
    }
    i as f64
}

fn rust_collatz_while(n: i64) -> f64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < n {
        let d = if black_box(i) % 2 == 0 {
            i / 2
        } else {
            i * 3 + 1
        };
        acc += d;
        i += 1;
    }
    acc as f64
}

fn rust_grade_while(n: i64) -> f64 {
    let mut a = 0i64;
    let mut b = 0i64;
    let mut i = 0i64;
    while i < n {
        if black_box(i) % 3 == 0 {
            a += 1;
        } else if i % 3 == 1 {
            b += 1;
        } else {
            a += 2;
        }
        i += 1;
    }
    (a + b) as f64
}

fn context(interner: &Interner, n: i64) -> Context {
    [(interner.intern("n"), typed(Ty::I64, Value::int(n)))]
        .into_iter()
        .collect()
}

fn snapshot(interner: &Interner, n: i64) -> HashMap<String, Owned<AcvusRuntime>> {
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
        let value = rt.block_on(interp.execute());
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

fn sizes() -> Vec<Size> {
    let Ok(raw) = std::env::var("ACCUM_N") else {
        return vec![
            Size {
                n: 100_000,
                reps: reps(19),
            },
            Size {
                n: 1_000_000,
                reps: reps(4),
            },
        ];
    };
    let Ok(n) = raw.parse::<NonZeroU32>() else {
        eprintln!("ACCUM_N={raw:?} is not a positive integer");
        std::process::exit(1);
    };
    vec![Size {
        n: i64::from(n.get()),
        reps: reps(1),
    }]
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");
    let cases = [
        Case {
            name: "trim view",
            source: TRIM_VIEW,
            registries: with_views,
            rust: rust_trim,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "trim owned",
            source: TRIM_OWNED,
            registries: with_views,
            rust: rust_trim,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "cut view",
            source: CUT_VIEW,
            registries: with_views,
            rust: rust_cut,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "cut owned",
            source: CUT_OWNED,
            registries: with_views,
            rust: rust_cut,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "int while",
            source: INT_WHILE,
            registries: std_only,
            rust: rust_int_while,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "float while",
            source: FLOAT_WHILE,
            registries: std_only,
            rust: rust_float_while,
            read: Value::as_float,
            ret: Ty::Float,
        },
        Case {
            name: "range | sum",
            source: RANGE_SUM,
            registries: std_only,
            rust: rust_range_sum,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "for range",
            source: FOR_RANGE,
            registries: std_only,
            rust: rust_for_range,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "for range break",
            source: FOR_RANGE_BREAK,
            registries: std_only,
            rust: rust_for_range_break,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "for if chain",
            source: FOR_IF_CHAIN,
            registries: std_only,
            rust: rust_for_if_chain,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "map id | sum",
            source: MAP_ID_SUM,
            registries: std_only,
            rust: rust_map_id_sum,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "for slice",
            source: FOR_SLICE,
            registries: std_only,
            rust: rust_for_slice,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "map add | sum",
            source: MAP_ADD_SUM,
            registries: std_only,
            rust: rust_map_add_sum,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "for slice add",
            source: FOR_SLICE_ADD,
            registries: std_only,
            rust: rust_for_slice_add,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "map cap | sum",
            source: MAP_CAP_SUM,
            registries: std_only,
            rust: rust_map_cap_sum,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "extern while",
            source: EXTERN_WHILE,
            registries: with_some_of,
            rust: rust_extern_while,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "branch while",
            source: BRANCH_WHILE,
            registries: with_some_of,
            rust: rust_branch_while,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "option while",
            source: OPTION_WHILE,
            registries: with_some_of,
            rust: rust_option_while,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "while let vec",
            source: WHILE_LET_VEC,
            registries: std_only,
            rust: rust_while_let_vec,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "while let map",
            source: WHILE_LET_MAP,
            registries: std_only,
            rust: rust_while_let_map,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "call while",
            source: CALL_WHILE,
            registries: std_only,
            rust: rust_call_while,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "collatz while",
            source: COLLATZ_WHILE,
            registries: std_only,
            rust: rust_collatz_while,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
        Case {
            name: "grade while",
            source: GRADE_WHILE,
            registries: std_only,
            rust: rust_grade_while,
            read: |v| v.as_int() as f64,
            ret: Ty::I64,
        },
    ];
    let selected = cases_named(&cases, "ACCUM_CASE");
    let sizes = sizes();
    println!(
        "{:>12} {:>10} {:>14} {:>12} {:>12} {:>14}",
        "case", "n", "execute/us", "rust/us", "ratio", "ns/iteration"
    );
    for case in selected {
        for size in &sizes {
            let Timing { execute, rust } = measure(&rt, case, size);
            let n = size.n;
            println!(
                "{:>12} {:>10} {:>14.1} {:>12.2} {:>12.1} {:>14.1}",
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
