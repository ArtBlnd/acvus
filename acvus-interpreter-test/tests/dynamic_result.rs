//! A `dynamic` extern's result is typed by its call site (RFC-0097 rule 3):
//! the script sees `Option<T>`, `T` settled by how the call's result is
//! used, and the handler fills an `Output` at that type. A mismatched leaf, a
//! second write, a name the type lacks or a missing field is `None`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Finished, Output, Registry, Runtime, Var, extern_fn, extern_registry, kind};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::{Context, Refusal, check_source, run_script_with_externs};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[extern_fn(dynamic, effect = pure)]
fn describe<'c, T, R>(n: i64, mut out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    out.field("name", |name| name.write(format!("n{n}")));
    out.field("n", |at| at.write(n));
    out.finish()
}

#[extern_fn(dynamic, effect = pure)]
fn name_only<'c, T, R>(n: i64, mut out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    out.field("name", |name| name.write(format!("n{n}")));
    out.finish()
}

#[extern_fn(dynamic, effect = pure)]
fn nested<'c, T, R>(n: i64, mut out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    out.field("inner", |inner| {
        inner.field("k", |k| k.write(n * 2));
        inner.field("label", |label| label.write("in".to_string()));
    });
    out.field("outer", |outer| outer.write(n));
    out.finish()
}

#[extern_fn(dynamic, effect = pure)]
fn whole<'c, T, R>(n: i64, mut out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    out.write(n);
    out.finish()
}

#[extern_fn(dynamic, effect = pure)]
fn twice<'c, T, R>(n: i64, mut out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    out.field("n", |at| at.write(n));
    out.field("n", |at| at.write(n + 1));
    out.finish()
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        types: [Held],
        fns: [describe, name_only, nested, whole, twice, held, held_then_mismatch, fuzz],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(registry());
    registries
}

async fn run(source: &str, ret: Ty) -> acvus_interpreter::Value {
    let i = Interner::new();
    run_script_with_externs(&i, source, Context::default(), registries(), ret)
        .await
        .value
}

async fn run_i64(source: &str) -> i64 {
    run(source, Ty::I64).await.as_int()
}

fn refused(source: &str) -> Vec<String> {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the source parses"));
    match check_source(&i, parsed, &FxHashMap::default(), registries(), Ty::I64, Opt::Full, |_| {}) {
        Ok(_) => panic!("`{source}` compiled"),
        Err(Refusal { messages, .. }) => messages,
    }
}

/// `{name: String, n: i64}`, settled by the `else` branch's literal.
const NAME_AND_N: &str = r#"let r = if let Some(r) = describe(7) { r } else { { name: "none".to_string(), n: -1, } };
r.name.len() as i64 * 100 + r.n"#;

#[tokio::test]
async fn a_site_settling_every_written_field_reads_them_from_the_some() {
    assert_eq!(run_i64(NAME_AND_N).await, 207, "`n7` is two bytes, and `n` is 7");
    let read = run_i64(
        r#"if let Some(r) = describe(7) { if r.name == "n7".to_string() { r.n } else { -2 } } else { -1 }"#,
    )
    .await;
    assert_eq!(read, 7, "the fields are read from the `Some` alone");
}

#[tokio::test]
async fn a_site_settling_fewer_fields_than_the_handler_writes_is_none() {
    let source = r#"let r = if let Some(r) = describe(7) { r } else { { name: "none".to_string(), } };
r.name.len() as i64"#;
    assert_eq!(run_i64(source).await, 4, "the `n` write names a field `{{name: String}}` lacks");
}

#[tokio::test]
async fn a_leaf_of_another_type_is_none() {
    let source = r#"let r = if let Some(r) = describe(7) { r } else { { name: -5, n: 0, } };
r.name"#;
    assert_eq!(run_i64(source).await, -5, "a `String` written where `i64` was settled");
}

#[tokio::test]
async fn a_missing_field_is_none() {
    let source = r#"let r = if let Some(r) = name_only(7) { r } else { { name: "none".to_string(), n: -1, } };
r.n"#;
    assert_eq!(run_i64(source).await, -1, "nothing wrote `n`");
}

#[tokio::test]
async fn a_second_write_of_a_field_is_none() {
    let source = "let r = if let Some(r) = twice(7) { r } else { { n: -1, } };\nr.n";
    assert_eq!(run_i64(source).await, -1);
}

#[tokio::test]
async fn nested_objects_fill_field_by_field() {
    let source = r#"let r = if let Some(r) = nested(4) { r } else { { inner: { k: 0, label: "".to_string(), }, outer: -1, } };
r.inner.k * 100 + r.inner.label.len() as i64 * 10 + r.outer"#;
    assert_eq!(run_i64(source).await, 824);
}

#[tokio::test]
async fn a_leaf_result_is_one_whole_write() {
    assert_eq!(run_i64("if let Some(n) = whole(9) { n } else { -1 }").await, 9);
    let source = r#"let r = if let Some(s) = whole(9) { s } else { "none".to_string() };
r.len() as i64"#;
    assert_eq!(run_i64(source).await, 4, "an `i64` written where `String` was settled");
}

#[test]
fn a_site_that_settles_no_type_is_a_compile_error() {
    let messages = refused("let r = describe(7);\n0");
    assert_eq!(
        messages,
        [
            "[main] the result of `t::describe` is typed by how this call's result is used, and \
             no use here settles its type: read its fields, return it, or pass it where a type is \
             declared (RFC-0097 rule 3)"
        ]
    );
}

// -- A sealed output releases what it filled ---------------------------------

static RELEASES: AtomicUsize = AtomicUsize::new(0);
static COUNTED_SCRIPTS: Mutex<()> = Mutex::new(());

fn counted_script() -> MutexGuard<'static, ()> {
    COUNTED_SCRIPTS.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Counted;

impl Drop for Counted {
    fn drop(&mut self) {
        RELEASES.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(ExternType)]
#[repr(transparent)]
struct Held(Vec<Counted>);

#[extern_fn(effect = pure)]
fn held(n: i64) -> Held {
    Held((0..n).map(|_| Counted).collect())
}

#[extern_fn(dynamic, effect = pure)]
fn held_then_mismatch<'c, T, R>(n: i64, mut out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    out.field("h", |h| h.write(Held((0..n).map(|_| Counted).collect())));
    out.field("n", |at| at.write("not an i64".to_string()));
    out.finish()
}

#[tokio::test]
async fn a_sealed_output_releases_the_values_it_filled() {
    let _only = counted_script();
    let before = RELEASES.load(Ordering::SeqCst);
    let source = "let r = if let Some(r) = held_then_mismatch(3) { r } else { { h: held(0), n: -1, } };\nr.n";
    assert_eq!(run_i64(source).await, -1);
    assert_eq!(RELEASES.load(Ordering::SeqCst) - before, 3);
}

// -- Random orders and mismatches never panic --------------------------------

/// The leaves of `{a: i64, b: String, c: {d: i64, e: bool}}`, the type the
/// fuzz script settles.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Leaf {
    A,
    B,
    D,
    E,
}

const LEAVES: [Leaf; 4] = [Leaf::A, Leaf::B, Leaf::D, Leaf::E];

#[derive(Clone, Copy, Debug)]
enum Step {
    Right(Leaf),
    Wrong(Leaf),
    UnknownField,
    WholeWriteAtObjectC,
    FieldIntoLeafA,
}

/// What the fuzz script reads out of one result.
struct Read {
    a: i64,
    b: String,
    d: i64,
    e: bool,
}

impl Read {
    /// The object the script's `else` branch builds for a `None`.
    fn script_default() -> Read {
        Read {
            a: -1,
            b: String::new(),
            d: -1,
            e: false,
        }
    }

    fn filled_by(seed: i64) -> Read {
        Read {
            a: seed * 3,
            b: format!("s{seed}"),
            d: seed * 5 + 1,
            e: seed % 2 == 0,
        }
    }

    /// The script's own sum over one result, at seed `i`.
    fn weighed(&self, i: i64) -> i64 {
        (i + 1) * (self.a * 7 + self.b.len() as i64 * 11 + self.d * 13 + i64::from(self.e) * 17)
    }
}

struct Plan {
    steps: Vec<Step>,
    reads: Option<Read>,
}

/// xorshift64, seeded per plan: the fuzz is the same on every run.
struct XorShift(u64);

/// Two odd constants that spread consecutive seeds over the state; any odd
/// pair does, these are the golden ratio's and xorshift64*'s multipliers.
const SEED_OFFSET: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_SPREAD: u64 = 0x2545_f491_4f6c_dd1d;

impl XorShift {
    fn seeded(seed: i64) -> XorShift {
        XorShift((SEED_OFFSET ^ (seed as u64).wrapping_mul(SEED_SPREAD)) | 1)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

fn plan(seed: i64) -> Plan {
    let mut rng = XorShift::seeded(seed);
    let mut steps = Vec::new();
    for leaf in LEAVES {
        match rng.below(10) {
            0 => {}
            1 => steps.push(Step::Wrong(leaf)),
            2 => steps.extend([Step::Right(leaf), Step::Right(leaf)]),
            _ => steps.push(Step::Right(leaf)),
        }
    }
    match rng.below(12) {
        0 => steps.push(Step::UnknownField),
        1 => steps.push(Step::WholeWriteAtObjectC),
        2 => steps.push(Step::FieldIntoLeafA),
        _ => {}
    }
    for at in (1..steps.len()).rev() {
        let other = rng.below(at as u64 + 1) as usize;
        steps.swap(at, other);
    }
    let each_leaf_once = LEAVES.iter().all(|leaf| {
        steps
            .iter()
            .filter(|step| matches!(step, Step::Right(right) if right == leaf))
            .count()
            == 1
    });
    let only_right_writes = steps.iter().all(|step| matches!(step, Step::Right(_)));
    let reads = (each_leaf_once && only_right_writes).then(|| Read::filled_by(seed));
    Plan { steps, reads }
}

fn write_leaf<R>(out: &mut Output<'_, impl Sized, R>, leaf: Leaf, seed: i64)
where
    R: Runtime,
{
    let Read { a, b, d, e } = Read::filled_by(seed);
    match leaf {
        Leaf::A => out.field("a", |at| at.write(a)),
        Leaf::B => out.field("b", |at| at.write(b)),
        Leaf::D => out.field("c", |c| c.field("d", |at| at.write(d))),
        Leaf::E => out.field("c", |c| c.field("e", |at| at.write(e))),
    }
}

fn write_leaf_mistyped<R>(out: &mut Output<'_, impl Sized, R>, leaf: Leaf, seed: i64)
where
    R: Runtime,
{
    match leaf {
        Leaf::A => out.field("a", |at| at.write(true)),
        Leaf::B => out.field("b", |at| at.write(seed)),
        Leaf::D => out.field("c", |c| c.field("d", |at| at.write(1.5f64))),
        Leaf::E => out.field("c", |c| c.field("e", |at| at.write('x'))),
    }
}

#[extern_fn(dynamic, effect = pure)]
fn fuzz<'c, T, R>(seed: i64, mut out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    for step in plan(seed).steps {
        match step {
            Step::Right(leaf) => write_leaf(&mut out, leaf, seed),
            Step::Wrong(leaf) => write_leaf_mistyped(&mut out, leaf, seed),
            Step::UnknownField => out.field("z", |z| z.write(seed)),
            Step::WholeWriteAtObjectC => out.field("c", |c| c.write(seed)),
            Step::FieldIntoLeafA => out.field("a", |a| a.field("x", |x| x.write(seed))),
        }
    }
    out.finish()
}

const SEEDS: i64 = 1000;

#[tokio::test]
async fn a_thousand_random_orders_and_mismatches_answer_as_the_plan_says() {
    let source = format!(
        r#"let total = 0;
for i in 0..{SEEDS} {{
    let r = if let Some(r) = fuzz(i) {{ r }} else {{ {{ a: -1, b: "".to_string(), c: {{ d: -1, e: false, }}, }} }};
    let e = if r.c.e {{ 1 }} else {{ 0 }};
    total = total + (i + 1) * (r.a * 7 + r.b.len() as i64 * 11 + r.c.d * 13 + e * 17);
}}
total"#
    );
    let plans: Vec<Plan> = (0..SEEDS).map(plan).collect();
    let clean = plans.iter().filter(|plan| plan.reads.is_some()).count();
    assert!(
        (100..900).contains(&clean),
        "the plans mix clean fills and failures: {clean} of {SEEDS} are clean"
    );
    let expected: i64 = plans
        .into_iter()
        .zip(0..)
        .map(|(plan, i)| plan.reads.unwrap_or_else(Read::script_default).weighed(i))
        .sum();
    assert_eq!(run_i64(&source).await, expected);
}
