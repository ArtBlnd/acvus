//! Accumulation loops: the per-operation cost of the machine with no
//! extern call in the loop, next to the same loop in Rust.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::{Registry, extern_fn, extern_registry};
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

const INT_WHILE: &str = "let acc = 0; let i = 0; while i < @n { acc = acc + i; i = i + 1; } acc";
const FLOAT_WHILE: &str =
    "let acc = 0.0; let i = 0; while i < @n { acc = acc + i.to_float(); i = i + 1; } acc";
const RANGE_SUM: &str = "range(0, @n) | sum";
const MAP_ID_SUM: &str = "range(0, @n) | map(|x| -> x) | sum";
const MAP_ADD_SUM: &str = "range(0, @n) | map(|x| -> x + 1) | sum";
const MAP_CAP_SUM: &str = "let k = 1; range(0, @n) | map(|x| -> x + k) | sum";
const EXTERN_WHILE: &str =
    "let i = 0; let acc = 0; while i < @n { acc = acc + id_of(i); i = i + 1; } acc";
const BRANCH_WHILE: &str =
    "let i = 0; let acc = 0; while i < @n { if even_of(i) { acc = acc + i; }; i = i + 1; } acc";
const OPTION_WHILE: &str = "let i = 0; let acc = 0; while i < @n { if let Some(v) = some_of(i) { acc = acc + v; }; i = i + 1; } acc";

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

fn snapshot(interner: &Interner, n: i64) -> HashMap<String, Value> {
    split_context(interner, context(interner, n)).1
}

struct Timing {
    execute: Duration,
    rust: Duration,
}

struct Size {
    n: i64,
    reps: usize,
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
    let mut execute = Vec::new();
    let mut script_value = f64::NAN;
    for rep in 0..reps {
        let ast = ParsedAst::Script(
            acvus_ast::parse_script(&interner, case.source).expect("parse error"),
        );
        let cr = compile_source_with_externs(&interner, ast, &context_types, (case.registries)());
        let (_shared, mut interp) = execute_compiled(
            &interner,
            cr,
            snapshot(&interner, n),
            Arc::new(SequentialExecutor),
        );
        let start = Instant::now();
        let value = rt.block_on(interp.execute());
        let elapsed = start.elapsed();
        script_value = (case.read)(&value);
        if rep > 0 {
            execute.push(elapsed);
        }
    }
    let mut rust = Vec::new();
    let mut rust_value = f64::NAN;
    for rep in 0..reps {
        let start = Instant::now();
        rust_value = black_box((case.rust)(black_box(n)));
        let elapsed = start.elapsed();
        if rep > 0 {
            rust.push(elapsed);
        }
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

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");
    let cases = [
        Case {
            name: "int while",
            source: INT_WHILE,
            registries: std_only,
            rust: rust_int_while,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "float while",
            source: FLOAT_WHILE,
            registries: std_only,
            rust: rust_float_while,
            read: Value::as_float,
        },
        Case {
            name: "range | sum",
            source: RANGE_SUM,
            registries: std_only,
            rust: rust_range_sum,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "map id | sum",
            source: MAP_ID_SUM,
            registries: std_only,
            rust: rust_map_id_sum,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "map add | sum",
            source: MAP_ADD_SUM,
            registries: std_only,
            rust: rust_map_add_sum,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "map cap | sum",
            source: MAP_CAP_SUM,
            registries: std_only,
            rust: rust_map_cap_sum,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "extern while",
            source: EXTERN_WHILE,
            registries: with_some_of,
            rust: rust_extern_while,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "branch while",
            source: BRANCH_WHILE,
            registries: with_some_of,
            rust: rust_branch_while,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "option while",
            source: OPTION_WHILE,
            registries: with_some_of,
            rust: rust_option_while,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "collatz while",
            source: COLLATZ_WHILE,
            registries: std_only,
            rust: rust_collatz_while,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "grade while",
            source: GRADE_WHILE,
            registries: std_only,
            rust: rust_grade_while,
            read: |v| v.as_int() as f64,
        },
    ];
    println!(
        "{:>12} {:>10} {:>14} {:>12} {:>12} {:>14}",
        "case", "n", "execute/us", "rust/us", "ratio", "ns/iteration"
    );
    for case in &cases {
        for size in [
            Size {
                n: 100_000,
                reps: 20,
            },
            Size {
                n: 1_000_000,
                reps: 5,
            },
        ] {
            let Timing { execute, rust } = measure(&rt, case, &size);
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
