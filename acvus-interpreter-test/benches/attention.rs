//! One program in three forms, one table row per size: `in-language` and
//! `dot` are execute times of the script whose inner loop is acvus and of
//! the script whose inner loop is one extern call, `rust` is the same
//! arithmetic in Rust, and the two ratio columns divide each script form by
//! `rust`.
//!
//! `dot` is declared here rather than in `acvus-ext` because a math library
//! is a design of its own and is not started by a bench. The signature this
//! file measures is the one such a library would export.
//!
//! These timings hold only under one pinned core and a fixed load base;
//! `benches/README.md` states the protocol.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::{Erased, Owned, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor};
use acvus_interpreter_test::listing::{regions_named, script_listing_with_externs};
use acvus_interpreter_test::scripts::{ATTENTION, ATTENTION_VEC};
use acvus_interpreter_test::{
    CompileResult, Context, compile_source_with_externs, execute_compiled, split_context,
    value_from_json,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use tokio::runtime::Runtime;

/// The elements reach the body as the caller's own run (RFC-0047 rule 6,
/// RFC-0068 rule 4): an `Erased<Rt, f64>` is read in place through its deref,
/// so the loop is the same multiply-add over the script's container that
/// the view-based form ran.
#[extern_fn(effect = pure)]
fn dot<Rt>(a: &[Erased<Rt, f64>], b: &[Erased<Rt, f64>]) -> f64
where
    Rt: acvus_extern::Runtime,
{
    assert_eq!(a.len(), b.len(), "dot takes two views of one length");
    a.iter().zip(b).map(|(x, y)| x.get() * y.get()).sum()
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [dot],
    });
    regs
}

fn compile_form(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    compile_source_with_externs(interner, ast, context_types, registries(), Ty::Float)
}

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
) -> HashMap<String, (acvus_mir::ty::Ty, Owned<AcvusRuntime>)> {
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

    // The script computes every output column and returns the first; the
    // twin does the same work so the ratio divides like by like.
    let mut out = Vec::with_capacity(d);
    for j in 0..d {
        let mut acc = 0.0;
        for t in 0..n {
            acc = acc + weights[t] / z * inputs.values[t][j];
        }
        out.push(acc);
    }
    out[0]
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    InLanguage,
    Dot,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::InLanguage => "in-language",
            Kernel::Dot => "dot",
        }
    }
}

/// Obligation across artifacts: this text is the `i` loop of both
/// `scripts::ATTENTION` and `scripts::ATTENTION_VEC`, character for
/// character. Editing either script's inner loop drops the count the
/// assertion in `source_of` takes.
const IN_LANGUAGE_INNER: &str = "    let key = &@keys[t];
    let s = 0.0;
    let i = 0;
    while i < d {
        s = s + @query[i] * key[i];
        i = i + 1;
    }
";

const DOT_INNER: &str = "    let s = dot(&@query, &@keys[t]);\n";

/// `*out.get(0)`, the read `ATTENTION`'s own tests use, is not available to
/// both variants: the `vec` registry exports no `get`, and the `deque`
/// registry exports no `as_slice`, so `out[0]` has no instance to lower
/// through either. `first` is the one read both registries do export, so the
/// bench uses it and the two variants differ in the container alone.
fn source_of(container: Container, kernel: Kernel) -> String {
    let script = container.script();
    let script = match kernel {
        Kernel::InLanguage => script.to_string(),
        Kernel::Dot => {
            assert_eq!(
                script.matches(IN_LANGUAGE_INNER).count(),
                1,
                "the {} script holds the in-language inner loop once",
                container.name()
            );
            script.replace(IN_LANGUAGE_INNER, DOT_INNER)
        }
    };
    format!("{script} if let Some(x) = out.first() {{ *x }} else {{ 0.0 }}")
}

struct Case {
    container: Container,
    n: usize,
    d: usize,
    reps: NonZeroUsize,
}

const fn reps(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(reps) => reps,
        None => panic!("a size measures at least one rep"),
    }
}

struct Timing {
    compile: Duration,
    setup: Duration,
    execute: Duration,
}

struct Row {
    container: Container,
    n: usize,
    d: usize,
    in_language: Timing,
    dot: Timing,
    rust: Duration,
}

struct Samples {
    kernel: Kernel,
    source: String,
    compile: Vec<Duration>,
    setup: Vec<Duration>,
    execute: Vec<Duration>,
    value: f64,
}

impl Samples {
    fn of(container: Container, kernel: Kernel) -> Self {
        Self {
            kernel,
            source: source_of(container, kernel),
            compile: Vec::new(),
            setup: Vec::new(),
            execute: Vec::new(),
            value: f64::NAN,
        }
    }

    fn timing(self) -> Timing {
        Timing {
            compile: median(self.compile),
            setup: median(self.setup),
            execute: median(self.execute),
        }
    }
}

struct Phases {
    compile: Duration,
    setup: Duration,
    execute: Duration,
    value: f64,
}

fn timed<T, F>(body: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let value = body();
    (value, start.elapsed())
}

/// The forms alternate inside one rep so that a drift of the box between the
/// first phase and the last lands on both of them.
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
    let context_types: FxHashMap<Astr, Ty> =
        split_context(&interner, context_of(&interner, &json)).0;

    let mut forms = [
        Samples::of(container, Kernel::InLanguage),
        Samples::of(container, Kernel::Dot),
    ];

    let run_form = |source: &str| {
        let (cr, compile) = timed(|| compile_form(&interner, source, &context_types));
        drop(black_box(cr));

        let cr = compile_form(&interner, source, &context_types);
        let snapshot = snapshot_of(&interner, &json);
        let (built, setup) =
            timed(|| execute_compiled(&interner, cr, snapshot, Arc::new(SequentialExecutor)));
        drop(black_box(built));

        let cr = compile_form(&interner, source, &context_types);
        let snapshot = snapshot_of(&interner, &json);
        let (_shared, mut interp) =
            execute_compiled(&interner, cr, snapshot, Arc::new(SequentialExecutor));
        let (value, execute) = timed(|| rt.block_on(interp.execute()).expect("the seeds hold every context the run fetches"));
        Phases {
            compile,
            setup,
            execute,
            value: value.as_float(),
        }
    };
    let run_rust = || timed(|| rust_attention(black_box(&inputs)));

    for form in &forms {
        let _warm_up = run_form(&form.source);
    }
    let (warm_up_value, _warm_up) = run_rust();
    let mut rust_value = black_box(warm_up_value);
    let mut rust_samples = Vec::with_capacity(reps.get());

    for _ in 0..reps.get() {
        for form in &mut forms {
            let Phases {
                compile,
                setup,
                execute,
                value,
            } = run_form(&form.source);
            form.value = value;
            form.compile.push(compile);
            form.setup.push(setup);
            form.execute.push(execute);
        }

        let (value, elapsed) = run_rust();
        rust_value = black_box(value);
        rust_samples.push(elapsed);
    }

    for form in &forms {
        let difference = (form.value - rust_value).abs();
        assert!(
            difference < 1e-9,
            "{} {} n={n} d={d}: script produced {:.17e}, \
             Rust reference produced {rust_value:.17e}, abs diff {difference:.17e}",
            container.name(),
            form.kernel.name(),
            form.value
        );
    }

    let [in_language, dot] = forms;
    Row {
        container,
        n,
        d,
        in_language: in_language.timing(),
        dot: dot.timing(),
        rust: median(rust_samples),
    }
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// This mode exists so a profiler sees one phase. Pointed at the default
/// bench, `perf record` attributes most of the process to the compiler, and
/// the interpreter symbols the stage is about sit under the noise.
fn execute_only(rt: &Runtime, case: &Case, kernel: Kernel) -> Duration {
    let Case {
        container,
        n,
        d,
        reps,
    } = *case;
    let inputs = inputs(n, d);
    let json = context_json(&inputs);
    let interner = Interner::new();
    let source = source_of(container, kernel);
    let context_types: FxHashMap<Astr, Ty> =
        split_context(&interner, context_of(&interner, &json)).0;
    let cr = compile_form(&interner, &source, &context_types);

    let run_once = || {
        let snapshot = snapshot_of(&interner, &json);
        let (_shared, mut interp) = execute_compiled(
            &interner,
            compile_form(&interner, &source, &context_types),
            snapshot,
            Arc::new(SequentialExecutor),
        );
        let start = Instant::now();
        let value = rt.block_on(interp.execute()).expect("the seeds hold every context the run fetches");
        (value, start.elapsed())
    };

    let (warm_up_value, _warm_up) = run_once();
    black_box(warm_up_value);
    let mut samples = Vec::with_capacity(reps.get());
    for _ in 0..reps.get() {
        let (value, elapsed) = run_once();
        black_box(value);
        samples.push(elapsed);
    }
    drop(black_box(cr));
    median(samples)
}

/// The operations the register selector leaves in each `while`, nested loops
/// first, so a difference between the two containers can be read off the
/// inner loops instead of inferred from a duration.
fn print_loop_listing(container: Container) {
    for kernel in [Kernel::InLanguage, Kernel::Dot] {
        let interner = Interner::new();
        let json = context_json(&inputs(2, 2));
        let blocks = script_listing_with_externs(
            &interner,
            &source_of(container, kernel),
            context_of(&interner, &json),
            registries(),
            Ty::Float,
        );
        for (index, region) in regions_named(&blocks, "Loop").into_iter().enumerate() {
            let head = region.part("head").expect("a Loop holds a head");
            let body = region.part("body").expect("a Loop holds a body");
            println!(
                "{} {} loop {index}: head [{}] body [{}]",
                container.name(),
                kernel.name(),
                head.ops.join(", "),
                body.ops.join(", ")
            );
        }
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

fn kernel_from_env() -> Kernel {
    match std::env::var("ATTENTION_KERNEL")
        .unwrap_or_else(|_| "in-language".to_string())
        .as_str()
    {
        "in-language" => Kernel::InLanguage,
        "dot" => Kernel::Dot,
        other => panic!("ATTENTION_KERNEL is in-language or dot, got {other:?}"),
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
            Err(_) => reps(19),
        },
    }
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");

    match std::env::var("ATTENTION_PHASE").as_deref() {
        Ok("execute") => {
            println!(
                "{:.1}",
                micros(execute_only(&rt, &case_from_env(), kernel_from_env()))
            );
            return;
        }
        Ok("listing") => {
            print_loop_listing(container_from_env());
            return;
        }
        _ => {}
    }

    let sizes = [(2, 2, reps(19)), (64, 64, reps(19)), (256, 128, reps(4))];
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
        "{:>8}  {:>10}  {:>22}  {:>16}  {:>22}  {:>16}  {:>22}  {:>16}  {:>10}  {:>18}  {:>10}",
        "container",
        "size",
        "in-language.compile/us",
        "dot.compile/us",
        "in-language.setup/us",
        "dot.setup/us",
        "in-language.execute/us",
        "dot.execute/us",
        "rust/us",
        "in-language/rust",
        "dot/rust",
    );
    for row in &rows {
        println!(
            "{:>8}  {:>10}  {:>22.1}  {:>16.1}  {:>22.1}  {:>16.1}  \
             {:>22.1}  {:>16.1}  {:>10.1}  {:>18.1}  {:>10.1}",
            row.container.name(),
            format!("{}x{}", row.n, row.d),
            micros(row.in_language.compile),
            micros(row.dot.compile),
            micros(row.in_language.setup),
            micros(row.dot.setup),
            micros(row.in_language.execute),
            micros(row.dot.execute),
            micros(row.rust),
            micros(row.in_language.execute) / micros(row.rust),
            micros(row.dot.execute) / micros(row.rust),
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
