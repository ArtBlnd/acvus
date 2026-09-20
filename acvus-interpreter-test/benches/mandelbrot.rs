//! Mandelbrot escape counts over a grid: three nested `while`s of pure
//! float arithmetic with a compound loop condition, next to the same
//! loops in Rust.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::Owned;
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{
    Context, compile_script_mode, execute_compiled, split_context, typed,
};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use tokio::runtime::Runtime;

const MANDELBROT: &str = "
let total = 0;
let py = 0;
while py < @h {
    let px = 0;
    while px < @w {
        let cx = -2.0 + 3.0 * px as f64 / @w as f64;
        let cy = -1.2 + 2.4 * py as f64 / @h as f64;
        let x = 0.0;
        let y = 0.0;
        let i = 0;
        while i < @max && x * x + y * y < 4.0 {
            let xt = x * x - y * y + cx;
            y = 2.0 * x * y + cy;
            x = xt;
            i = i + 1;
        }
        total = total + i;
        px = px + 1;
    }
    py = py + 1;
}
total
";

struct Grid {
    w: i64,
    h: i64,
    max: i64,
    reps: NonZeroUsize,
}

const fn reps(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(reps) => reps,
        None => panic!("a size measures at least one rep"),
    }
}

struct Timing {
    execute: Duration,
    rust: Duration,
    iterations: i64,
}

fn rust_mandelbrot(w: i64, h: i64, max: i64) -> i64 {
    let mut total = 0i64;
    let mut py = 0i64;
    while py < h {
        let mut px = 0i64;
        while px < w {
            let cx = -2.0 + 3.0 * black_box(px) as f64 / w as f64;
            let cy = -1.2 + 2.4 * py as f64 / h as f64;
            let mut x = 0.0f64;
            let mut y = 0.0f64;
            let mut i = 0i64;
            while i < max && x * x + y * y < 4.0 {
                let xt = x * x - y * y + cx;
                y = 2.0 * x * y + cy;
                x = xt;
                i += 1;
            }
            total += i;
            px += 1;
        }
        py += 1;
    }
    total
}

fn context(interner: &Interner, grid: &Grid) -> Context {
    [
        (interner.intern("w"), typed(Ty::I64, Value::int(grid.w))),
        (interner.intern("h"), typed(Ty::I64, Value::int(grid.h))),
        (interner.intern("max"), typed(Ty::I64, Value::int(grid.max))),
    ]
    .into_iter()
    .collect()
}

fn snapshot(interner: &Interner, grid: &Grid) -> HashMap<String, Owned<AcvusRuntime>> {
    split_context(interner, context(interner, grid)).1
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn measure(rt: &Runtime, grid: &Grid) -> Timing {
    let interner = Interner::new();
    let context_types = split_context(&interner, context(&interner, grid)).0;
    let run_script = || {
        let cr = compile_script_mode(&interner, MANDELBROT, &context_types, Ty::I64);
        let (_shared, mut interp) = execute_compiled(
            &interner,
            cr,
            snapshot(&interner, grid),
            Arc::new(SequentialExecutor),
        );
        let start = Instant::now();
        let value = rt.block_on(interp.execute());
        (value.as_int(), start.elapsed())
    };
    let run_rust = || {
        let start = Instant::now();
        let total = black_box(rust_mandelbrot(grid.w, grid.h, grid.max));
        (total, start.elapsed())
    };

    let (mut script_total, _warm_up) = run_script();
    let mut execute = Vec::with_capacity(grid.reps.get());
    for _ in 0..grid.reps.get() {
        let (total, elapsed) = run_script();
        script_total = total;
        execute.push(elapsed);
    }

    let (mut rust_total, _warm_up) = run_rust();
    let mut rust = Vec::with_capacity(grid.reps.get());
    for _ in 0..grid.reps.get() {
        let (total, elapsed) = run_rust();
        rust_total = total;
        rust.push(elapsed);
    }
    assert!(
        script_total == rust_total,
        "{}x{} max {}: script produced {script_total}, Rust produced {rust_total}",
        grid.w,
        grid.h,
        grid.max
    );
    Timing {
        execute: median(execute),
        rust: median(rust),
        iterations: rust_total,
    }
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");
    let grids = [
        Grid {
            w: 80,
            h: 40,
            max: 100,
            reps: reps(9),
        },
        Grid {
            w: 200,
            h: 100,
            max: 200,
            reps: reps(4),
        },
    ];
    println!(
        "{:>14} {:>12} {:>14} {:>12} {:>10} {:>14}",
        "grid", "iterations", "execute/us", "rust/us", "ratio", "ns/iteration"
    );
    for grid in &grids {
        let Timing {
            execute,
            rust,
            iterations,
        } = measure(&rt, grid);
        println!(
            "{:>14} {:>12} {:>14.1} {:>12.1} {:>10.1} {:>14.1}",
            format!("{}x{}x{}", grid.w, grid.h, grid.max),
            iterations,
            micros(execute),
            micros(rust),
            micros(execute) / micros(rust),
            execute.as_secs_f64() * 1e9 / iterations as f64
        );
    }
}
