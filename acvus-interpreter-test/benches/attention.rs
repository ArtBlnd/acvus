use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::Owned;
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::listing::{regions_named, script_listing};
use acvus_interpreter_test::scripts::{ATTENTION, ATTENTION_VEC};
use acvus_interpreter_test::{
    Context, compile_script_mode, execute_compiled, split_context, value_from_json,
};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use tokio::runtime::Runtime;

struct Inputs {
    query: Vec<f64>,
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
}

fn inputs(n: usize, d: usize) -> Inputs {
    let query = (0..d).map(|i| (i as f64).sin()).collect();
    let keys = (0..n)
        .map(|t| (0..d).map(|i| ((t + i) as f64).cos()).collect())
        .collect();
    let values = (0..n)
        .map(|t| {
            (0..d)
                .map(|i| (t * d + i) as f64 / (n * d) as f64)
                .collect()
        })
        .collect();
    Inputs {
        query,
        keys,
        values,
    }
}

fn json_number(x: f64) -> serde_json::Value {
    serde_json::Value::Number(serde_json::Number::from_f64(x).expect("a finite input"))
}

fn json_row(row: &[f64]) -> serde_json::Value {
    serde_json::Value::Array(row.iter().copied().map(json_number).collect())
}

fn json_rows(rows: &[Vec<f64>]) -> serde_json::Value {
    serde_json::Value::Array(rows.iter().map(|row| json_row(row)).collect())
}

fn context_json(inputs: &Inputs) -> serde_json::Value {
    serde_json::json!({
        "query": json_row(&inputs.query),
        "keys": json_rows(&inputs.keys),
        "values": json_rows(&inputs.values),
    })
}

fn context_of(interner: &Interner, json: &serde_json::Value) -> Context {
    json.as_object()
        .expect("an object")
        .iter()
        .map(|(k, v)| (interner.intern(k), value_from_json(interner, v)))
        .collect()
}

/// `Value` is not `Clone`, so each run that consumes a page snapshot rebuilds
/// one from the JSON rather than cloning it.
fn snapshot_of(
    interner: &Interner,
    json: &serde_json::Value,
) -> HashMap<String, Owned<AcvusRuntime>> {
    split_context(interner, context_of(interner, json)).1
}

fn rust_attention(inputs: &Inputs) -> f64 {
    let d = inputs.query.len();
    let n = inputs.keys.len();
    let scale = 1.0 / (d as f64).sqrt();

    let mut scores = Vec::with_capacity(n);
    for t in 0..n {
        let mut s = 0.0;
        for i in 0..d {
            s = s + inputs.query[i] * inputs.keys[t][i];
        }
        scores.push(s * scale);
    }

    let mut m = f64::NEG_INFINITY;
    for &s in &scores {
        if s > m {
            m = s;
        }
    }

    let mut weights = Vec::with_capacity(n);
    for &s in &scores {
        weights.push((s - m).exp());
    }

    let mut z = 0.0;
    for &w in &weights {
        z = z + w;
    }

    let mut acc = 0.0;
    for t in 0..n {
        acc = acc + weights[t] / z * inputs.values[t][0];
    }
    acc
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Container {
    Deque,
    Vec,
}

impl Container {
    fn script(self) -> &'static str {
        match self {
            Container::Deque => ATTENTION,
            Container::Vec => ATTENTION_VEC,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Container::Deque => "deque",
            Container::Vec => "vec",
        }
    }
}

/// `*out.get(0)`, the read `ATTENTION`'s own tests use, is not available to
/// both variants: the `vec` registry exports no `get`, and the `deque`
/// registry exports no `as_slice`, so `out[0]` has no instance to lower
/// through either. `first` is the one read both registries do export, so the
/// bench uses it and the two variants differ in the container alone.
fn source_of(container: Container) -> String {
    format!(
        "{} if let Some(x) = out.first() {{ *x }} else {{ 0.0 }}",
        container.script()
    )
}

struct Case {
    container: Container,
    n: usize,
    d: usize,
    reps: usize,
}

struct Row {
    container: Container,
    n: usize,
    d: usize,
    compile: Duration,
    setup: Duration,
    execute: Duration,
    rust: Duration,
}

fn measure(rt: &Runtime, case: &Case) -> Row {
    let Case {
        container,
        n,
        d,
        reps,
    } = *case;
    let inputs = inputs(n, d);
    let json = context_json(&inputs);
    let interner = Interner::new();
    let source = source_of(container);
    let context_types: FxHashMap<Astr, Ty> =
        split_context(&interner, context_of(&interner, &json)).0;

    let mut compile_samples = Vec::new();
    for rep in 0..reps {
        let start = Instant::now();
        let cr = compile_script_mode(&interner, &source, &context_types, Ty::Float);
        let elapsed = start.elapsed();
        drop(black_box(cr));
        if rep > 0 {
            compile_samples.push(elapsed);
        }
    }

    let mut setup_samples = Vec::new();
    for rep in 0..reps {
        let cr = compile_script_mode(&interner, &source, &context_types, Ty::Float);
        let snapshot = snapshot_of(&interner, &json);
        let start = Instant::now();
        let built = execute_compiled(&interner, cr, snapshot, Arc::new(SequentialExecutor));
        let elapsed = start.elapsed();
        drop(black_box(built));
        if rep > 0 {
            setup_samples.push(elapsed);
        }
    }

    let mut execute_samples = Vec::new();
    let mut script_value = f64::NAN;
    for rep in 0..reps {
        let cr = compile_script_mode(&interner, &source, &context_types, Ty::Float);
        let snapshot = snapshot_of(&interner, &json);
        let (_shared, mut interp) =
            execute_compiled(&interner, cr, snapshot, Arc::new(SequentialExecutor));
        let start = Instant::now();
        let value = rt.block_on(interp.execute());
        let elapsed = start.elapsed();
        script_value = value.as_float();
        if rep > 0 {
            execute_samples.push(elapsed);
        }
    }

    let mut rust_samples = Vec::new();
    let mut rust_value = f64::NAN;
    for rep in 0..reps {
        let start = Instant::now();
        let value = rust_attention(black_box(&inputs));
        let elapsed = start.elapsed();
        rust_value = black_box(value);
        if rep > 0 {
            rust_samples.push(elapsed);
        }
    }

    let difference = (script_value - rust_value).abs();
    assert!(
        difference < 1e-9,
        "{} n={n} d={d}: script produced {script_value:.17e}, \
         Rust reference produced {rust_value:.17e}, abs diff {difference:.17e}",
        container.name()
    );

    Row {
        container,
        n,
        d,
        compile: median(compile_samples),
        setup: median(setup_samples),
        execute: median(execute_samples),
        rust: median(rust_samples),
    }
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// This mode exists so a profiler sees one phase. Pointed at the default
/// bench, `perf record` attributes most of the process to the compiler, and
/// the interpreter symbols the stage is about sit under the noise.
fn execute_only(rt: &Runtime, case: &Case) -> Duration {
    let Case {
        container,
        n,
        d,
        reps,
    } = *case;
    let inputs = inputs(n, d);
    let json = context_json(&inputs);
    let interner = Interner::new();
    let source = source_of(container);
    let context_types: FxHashMap<Astr, Ty> =
        split_context(&interner, context_of(&interner, &json)).0;
    let cr = compile_script_mode(&interner, &source, &context_types, Ty::Float);

    let mut samples = Vec::new();
    for rep in 0..reps {
        let snapshot = snapshot_of(&interner, &json);
        let (_shared, mut interp) = execute_compiled(
            &interner,
            compile_script_mode(&interner, &source, &context_types, Ty::Float),
            snapshot,
            Arc::new(SequentialExecutor),
        );
        let start = Instant::now();
        let value = rt.block_on(interp.execute());
        let elapsed = start.elapsed();
        black_box(value);
        if rep > 0 {
            samples.push(elapsed);
        }
    }
    drop(black_box(cr));
    median(samples)
}

/// The operations the register selector leaves in each `while`, nested loops
/// first, so a difference between the two containers can be read off the
/// inner loops instead of inferred from a duration.
fn print_loop_listing(container: Container) {
    let interner = Interner::new();
    let json = context_json(&inputs(2, 2));
    let blocks = script_listing(
        &interner,
        &source_of(container),
        context_of(&interner, &json),
        Ty::Float,
    );
    for (index, region) in regions_named(&blocks, "Loop").into_iter().enumerate() {
        let head = region.part("head").expect("a Loop holds a head");
        let body = region.part("body").expect("a Loop holds a body");
        println!(
            "{} loop {index}: head [{}] body [{}]",
            container.name(),
            head.ops.join(", "),
            body.ops.join(", ")
        );
    }
}

fn container_from_env() -> Container {
    match std::env::var("ATTENTION_CONTAINER")
        .unwrap_or_else(|_| "deque".to_string())
        .as_str()
    {
        "deque" => Container::Deque,
        "vec" => Container::Vec,
        other => panic!("ATTENTION_CONTAINER is deque or vec, got {other:?}"),
    }
}

fn case_from_env() -> Case {
    let size = std::env::var("ATTENTION_SIZE").unwrap_or_else(|_| "64x64".to_string());
    let (n, d) = size
        .split_once('x')
        .unwrap_or_else(|| panic!("ATTENTION_SIZE is <n>x<d>, got {size:?}"));
    Case {
        container: container_from_env(),
        n: n.parse()
            .unwrap_or_else(|e| panic!("ATTENTION_SIZE n {n:?}: {e}")),
        d: d.parse()
            .unwrap_or_else(|e| panic!("ATTENTION_SIZE d {d:?}: {e}")),
        reps: match std::env::var("ATTENTION_REPS") {
            Ok(text) => text
                .parse()
                .unwrap_or_else(|e| panic!("ATTENTION_REPS {text:?}: {e}")),
            Err(_) => 20,
        },
    }
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");

    match std::env::var("ATTENTION_PHASE").as_deref() {
        Ok("execute") => {
            println!("{:.1}", micros(execute_only(&rt, &case_from_env())));
            return;
        }
        Ok("listing") => {
            print_loop_listing(container_from_env());
            return;
        }
        _ => {}
    }

    let sizes = [(2, 2, 20), (64, 64, 20), (256, 128, 5)];
    let cases: Vec<Case> = [Container::Deque, Container::Vec]
        .into_iter()
        .flat_map(|container| {
            sizes.into_iter().map(move |(n, d, reps)| Case {
                container,
                n,
                d,
                reps,
            })
        })
        .collect();
    let rows: Vec<Row> = cases.iter().map(|case| measure(&rt, case)).collect();

    println!(
        "{:>8}  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "container", "size", "compile/us", "setup/us", "execute/us", "rust/us", "execute/rust"
    );
    for row in &rows {
        println!(
            "{:>8}  {:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}",
            row.container.name(),
            format!("{}x{}", row.n, row.d),
            micros(row.compile),
            micros(row.setup),
            micros(row.execute),
            micros(row.rust),
            micros(row.execute) / micros(row.rust)
        );
    }
    println!(
        "profile: {}",
        if cfg!(debug_assertions) {
            "debug_assertions on"
        } else {
            "debug_assertions off"
        }
    );
}
