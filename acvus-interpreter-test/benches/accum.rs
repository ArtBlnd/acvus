//! Accumulation loops: the per-operation cost of the machine with no
//! extern call in the loop, next to the same loop in Rust.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_interpreter::{SequentialExecutor, Value};
use acvus_interpreter_test::{
    Context, compile_script_mode, execute_compiled, split_context, typed,
};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use tokio::runtime::Runtime;

const INT_WHILE: &str = "let acc = 0; let i = 0; while i < @n { acc = acc + i; i = i + 1; } acc";
const FLOAT_WHILE: &str =
    "let acc = 0.0; let i = 0; while i < @n { acc = acc + i.to_float(); i = i + 1; } acc";
const RANGE_SUM: &str = "range(0, @n) | sum";
const MAP_ID_SUM: &str = "range(0, @n) | map(|x| -> x) | sum";
const MAP_ADD_SUM: &str = "range(0, @n) | map(|x| -> x + 1) | sum";

struct Case {
    name: &'static str,
    source: &'static str,
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
        let cr = compile_script_mode(&interner, case.source, &context_types);
        let (_shared, mut interp) = execute_compiled(
            &interner,
            cr,
            snapshot(&interner, n),
            Arc::new(SequentialExecutor),
        );
        let start = Instant::now();
        let value = rt.block_on(interp.execute()).expect("execution failed");
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
            rust: rust_int_while,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "float while",
            source: FLOAT_WHILE,
            rust: rust_float_while,
            read: Value::as_float,
        },
        Case {
            name: "range | sum",
            source: RANGE_SUM,
            rust: rust_range_sum,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "map id | sum",
            source: MAP_ID_SUM,
            rust: rust_map_id_sum,
            read: |v| v.as_int() as f64,
        },
        Case {
            name: "map add | sum",
            source: MAP_ADD_SUM,
            rust: rust_map_add_sum,
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
