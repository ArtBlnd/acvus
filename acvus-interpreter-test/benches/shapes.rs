//! Object and enum shapes: the per-operation cost of a field access, an
//! object construction and a variant match, next to the same loop in Rust.
//!
//! The Rust references use a struct and an enum, so their field access is a
//! fixed offset and their match one tag test. Today's `Value` holds an
//! object as a boxed `FxHashMap<Astr, Value>` and an enum as a boxed
//! `Variant<Value>`, so the ratio is the distance between the two layouts,
//! not interpreter overhead alone.

use std::collections::HashMap;
use std::hint::black_box;
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

fn std_only() -> Vec<Registry<AcvusRuntime>> {
    acvus_ext::std_registries::<AcvusRuntime>()
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
const OPTION_MATCH: &str = "let i = 0; let acc = 0; while i < @n { if let Some(v) = some_of(i) { acc = acc + v; }; i = i + 1; } acc";

/// `v[i]` takes a `u64` index and integer literals are `i64`, so the index
/// counter is derived from `len`, and the sweep over the thousand objects
/// repeats `n / 1000` times to reach `n` iterations of the inner body.
const VEC_OF_OBJECTS: &str = "let v = range(0, 1000) | map(|k| -> { x: k, y: k + 1, }) | collect; let m = len(&v); let one = m / m; let acc = 0; let r = 0; while r < @n / 1000 { let i = m - m; while i < m { acc = acc + v[i].x; i = i + one; } r = r + 1; } acc";

struct Point {
    x: i64,
    y: i64,
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

fn snapshot(interner: &Interner, n: i64) -> HashMap<String, Owned<AcvusRuntime>> {
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
            name: "option match",
            source: OPTION_MATCH,
            registries: with_some_of,
            rust: rust_option_match,
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
    if let Ok(name) = std::env::var("SHAPES_OPLIST") {
        oplist(&cases, &name);
        return;
    }
    println!(
        "{:>14} {:>10} {:>14} {:>12} {:>12} {:>14}",
        "case", "n", "execute/us", "rust/us", "ratio", "ns/iteration"
    );
    let only = std::env::var("SHAPES_CASE").ok();
    for case in cases
        .iter()
        .filter(|c| only.as_deref().is_none_or(|o| c.name == o))
    {
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
fn oplist(cases: &[Case], name: &str) {
    let case = cases
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no case named {name:?}"));
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        case.source,
        context(&interner, 1_000_000),
        (case.registries)(),
        case.ret.clone(),
    );
    println!("== oplist: {name}");
    print_listing(&blocks);
}
