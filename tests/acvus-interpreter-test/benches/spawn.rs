//! What `optimize::spawn_split` + `optimize::reorder` buy today.
//!
//! Every other bench in this crate is `Sync` on one thread, so the split
//! pass had nothing to split. Here one extern burns a fixed slice of pure
//! integer work, and the same eight calls are written three ways —
//! straight-line, a dependent chain, and a `while` loop — against three
//! declarations of the same Rust body:
//!
//! - `heavy_pure`: `#[extern_fn(heavy, effect = pure)]`, the declaration
//!   RFC-0046 names for CPU work handed to the blocking pool.
//! - `sync_pure`: `#[extern_fn(effect = pure)]`, the sequential floor.
//! - `heavy_opaque`: `#[extern_fn(heavy, effect = opaque)]`, one variable
//!   apart from `heavy_pure` — the reissue level, which is what
//!   `spawn_split::is_io_call` reads.
//!
//! The listings are printed before the table: what the pass emitted is the
//! measurement, and the numbers are what that emission costs.
//!
//! These timings hold only under one pinned core and a fixed load base;
//! `benches/README.md` states the protocol.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, TokioExecutor, Value};
use acvus_interpreter_test::listing::{BlockListing, script_listing_with_externs};
use acvus_interpreter_test::{
    Context, compile_source_with_externs, execute_compiled, split_context,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use tokio::runtime::Runtime;

// -- The work ---------------------------------------------------------

/// How many LCG steps one call burns. Read once per call; set once per
/// size, before both the script run and its Rust twin, so the two always
/// burn the same amount.
static SPIN: AtomicI64 = AtomicI64::new(SPIN_SMALL);

/// Calibrated on this machine so one call is about 50 µs.
const SPIN_SMALL: i64 = 45_000;
/// Ten times that: the second line Would-refute item 3 asks for, to show
/// where a per-spawn fixed cost stops mattering.
const SPIN_BIG: i64 = 450_000;

const LCG_MUL: i64 = 6_364_136_223_846_793_005;
const LCG_ADD: i64 = 1_442_695_040_888_963_407;

/// A dependent multiply-add chain: no allocation, nothing to fold, and the
/// answer depends on the seed. The loop is a state machine, so it is a
/// `for`.
fn burn(seed: i64) -> i64 {
    let mut x = seed.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
    let mut acc = 0i64;
    for _ in 0..SPIN.load(Ordering::Relaxed) {
        x = x.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        acc ^= x >> 17;
    }
    acc
}

/// The `Heavy` declaration: the runtime hands it to the blocking pool and
/// awaits (RFC-0046). `pure` is what a burner of CPU with no page and no
/// reissue hazard truthfully declares.
#[extern_fn(heavy, effect = pure)]
fn heavy_pure(seed: i64) -> i64 {
    burn(seed)
}

/// The sequential floor: the same body called in the caller's frame.
#[extern_fn(effect = pure)]
fn sync_pure(seed: i64) -> i64 {
    burn(seed)
}

/// The same body, still `heavy`, declared `opaque`. Only the reissue level
/// differs from `heavy_pure`, and that is the field `spawn_split`'s gate
/// reads.
#[extern_fn(heavy, effect = opaque)]
fn heavy_opaque(seed: i64) -> i64 {
    burn(seed)
}

/// The same body again, `heavy` and `opaque`, declared to commute. One
/// variable apart from `heavy_opaque`: the `commutes` axis, which is what
/// `optimize::commute` reads to release a run of calls onto one entry
/// `Order` instead of chaining each to the last one's.
#[extern_fn(heavy, effect = opaque, commutative)]
fn heavy_commutes(seed: i64) -> i64 {
    burn(seed)
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "bench",
        fns: [heavy_pure, sync_pure, heavy_opaque, heavy_commutes],
    });
    regs
}

// -- The sources ------------------------------------------------------

/// The seeds every case calls, so all three cases that sum agree on one
/// answer.
const SEEDS: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// Eight independent calls in one block: `spawn_split` + `reorder`'s home
/// ground, if the calls are split at all.
fn straight_line(f: &str) -> String {
    let lets: String = SEEDS
        .iter()
        .enumerate()
        .map(|(k, seed)| format!("let v{k} = {f}({seed}); "))
        .collect();
    let sum: String = (0..SEEDS.len())
        .map(|k| format!("v{k}"))
        .collect::<Vec<_>>()
        .join(" + ");
    format!("{lets}{sum}")
}

/// Eight calls each needing the last: no parallelism exists, so the
/// difference from the `Sync` line is eight times the fixed cost.
fn dependent_chain(f: &str) -> String {
    let mut source = format!("let v0 = {f}(1); ");
    for k in 1..SEEDS.len() {
        source.push_str(&format!("let v{k} = {f}(v{prev}); ", prev = k - 1));
    }
    source.push_str(&format!("v{}", SEEDS.len() - 1));
    source
}

/// The same eight independent calls, one per iteration. `reorder` schedules
/// within a block, so no two iterations can overlap today.
fn while_loop(f: &str) -> String {
    format!("let acc = 0; let i = 1; while i < 9 {{ acc = acc + {f}(i); i = i + 1; }} acc")
}

// -- The Rust twins ---------------------------------------------------

fn rust_straight_seq() -> i64 {
    SEEDS.iter().map(|seed| burn(black_box(*seed))).sum()
}

/// Eight threads, one per call, joined at the end of the scope: no new
/// dependency, and the ceiling case 1 would reach if its calls overlapped.
fn rust_straight_scope() -> i64 {
    std::thread::scope(|scope| {
        let handles: Vec<_> = SEEDS
            .iter()
            .map(|seed| scope.spawn(|| burn(black_box(*seed))))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a burn thread does not panic"))
            .sum()
    })
}

fn rust_chain() -> i64 {
    let mut v = burn(black_box(1));
    for _ in 1..SEEDS.len() {
        v = burn(black_box(v));
    }
    v
}

// -- Cases ------------------------------------------------------------

/// One declaration of the burner, by the name the source calls.
#[derive(Clone, Copy)]
struct Flavor {
    name: &'static str,
    call: &'static str,
}

const FLAVORS: [Flavor; 4] = [
    Flavor {
        name: "heavy/pure",
        call: "heavy_pure",
    },
    Flavor {
        name: "sync/pure",
        call: "sync_pure",
    },
    Flavor {
        name: "heavy/opaque",
        call: "heavy_opaque",
    },
    Flavor {
        name: "heavy/commutes",
        call: "heavy_commutes",
    },
];

struct Case {
    name: &'static str,
    source: fn(&str) -> String,
    /// The sequential Rust twin: the floor the same work costs with no
    /// machine and no spawn.
    rust_seq: fn() -> i64,
    /// The threaded Rust twin, where the case has independent calls.
    rust_par: Option<fn() -> i64>,
}

const CASES: [Case; 3] = [
    Case {
        name: "straight-8",
        source: straight_line,
        rust_seq: rust_straight_seq,
        rust_par: Some(rust_straight_scope),
    },
    Case {
        name: "chain-8",
        source: dependent_chain,
        rust_seq: rust_chain,
        rust_par: None,
    },
    Case {
        name: "loop-8",
        source: while_loop,
        rust_seq: rust_straight_seq,
        rust_par: Some(rust_straight_scope),
    },
];

// -- Measurement ------------------------------------------------------

/// acvus and Rust alternate inside one rep so a drift in the machine lands
/// on both.
const REPS: NonZeroUsize = reps(3);

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

struct Row {
    acvus: Duration,
    rust_seq: Duration,
    rust_par: Option<Duration>,
}

struct Values {
    script: i64,
    rust_seq: i64,
    rust_par: Option<i64>,
}

struct Timings {
    acvus: Duration,
    seq: Duration,
    par: Option<Duration>,
}

fn run_once(rt: &Runtime, source: &str) -> (Duration, i64) {
    let interner = Interner::new();
    let context_types = split_context(&interner, Context::default()).0;
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parse error"));
    let cr = compile_source_with_externs(&interner, ast, &context_types, registries(), Ty::I64);
    let (_shared, mut interp) =
        execute_compiled(&interner, cr, HashMap::new(), Arc::new(TokioExecutor));
    let start = Instant::now();
    let value: Value = rt.block_on(interp.execute()).expect("the seeds hold every context the run fetches");
    let elapsed = start.elapsed();
    (elapsed, value.as_int())
}

fn measure(rt: &Runtime, case: &Case, flavor: &Flavor) -> Row {
    let source = (case.source)(flavor.call);
    let run_rep = || {
        let (acvus, script) = run_once(rt, &source);

        let start = Instant::now();
        let rust_seq = black_box((case.rust_seq)());
        let seq = start.elapsed();

        let par = case.rust_par.map(|f| {
            let start = Instant::now();
            let value = black_box(f());
            (value, start.elapsed())
        });

        (
            Values {
                script,
                rust_seq,
                rust_par: par.map(|(value, _)| value),
            },
            Timings {
                acvus,
                seq,
                par: par.map(|(_, elapsed)| elapsed),
            },
        )
    };

    let (mut values, _warm_up) = run_rep();
    let mut acvus = Vec::with_capacity(REPS.get());
    let mut seq = Vec::with_capacity(REPS.get());
    let mut par = Vec::with_capacity(REPS.get());
    for _ in 0..REPS.get() {
        let (this, timings) = run_rep();
        values = this;
        acvus.push(timings.acvus);
        seq.push(timings.seq);
        par.extend(timings.par);
    }

    let Values {
        script,
        rust_seq,
        rust_par,
    } = values;
    assert!(
        script == rust_seq,
        "{} {}: script produced {script}, Rust produced {rust_seq}",
        case.name,
        flavor.name
    );
    if let Some(par_value) = rust_par {
        assert!(
            par_value == rust_seq,
            "{} {}: threaded Rust produced {par_value}, sequential Rust produced {rust_seq}",
            case.name,
            flavor.name
        );
    }
    Row {
        acvus: median(acvus),
        rust_seq: median(seq),
        rust_par: (!par.is_empty()).then(|| median(par)),
    }
}

// -- Listings ---------------------------------------------------------

/// Every block of the prepared entry body: the operations it runs and the
/// terminator it leaves at. `SpawnExternSync` + `Eval` is a split call;
/// `CallHeavy` is one the pass left whole.
fn print_listing(case: &Case, flavor: &Flavor) {
    let interner = Interner::new();
    let source = (case.source)(flavor.call);
    let blocks: Vec<BlockListing> = script_listing_with_externs(
        &interner,
        &source,
        Context::default(),
        registries(),
        Ty::I64,
    );
    println!("--- {} / {} ---", case.name, flavor.name);
    println!("    source: {source}");
    for (index, block) in blocks.iter().enumerate() {
        println!("    b{index}: {:?} end={}", block.ops, block.end);
    }
    println!();
}

// -- Main -------------------------------------------------------------

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a multi-thread tokio runtime");

    let cores = std::thread::available_parallelism()
        .expect("this bench reports the core count it ran on; without it there is no measurement")
        .get();
    println!(
        "executor: TokioExecutor; runtime: tokio multi_thread, {cores} worker threads \
         (tokio's default is the core count); cores: {cores}; threads not pinned"
    );
    println!();

    println!("== listings after spawn_split + reorder ==");
    for case in &CASES {
        for flavor in &FLAVORS {
            print_listing(case, flavor);
        }
    }

    println!(
        "{:>12} {:>14} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "case", "extern", "spin", "call/us", "acvus/us", "rust seq/us", "rust par/us", "acvus/seq"
    );
    for spin in [SPIN_SMALL, SPIN_BIG] {
        SPIN.store(spin, Ordering::Relaxed);
        let one_call = {
            let start = Instant::now();
            black_box(burn(black_box(1)));
            start.elapsed()
        };
        for case in &CASES {
            for flavor in &FLAVORS {
                let row = measure(&rt, case, flavor);
                println!(
                    "{:>12} {:>14} {:>10} {:>12.1} {:>12.1} {:>12.1} {:>12} {:>12.2}",
                    case.name,
                    flavor.name,
                    spin,
                    micros(one_call),
                    micros(row.acvus),
                    micros(row.rust_seq),
                    match row.rust_par {
                        Some(par) => format!("{:.1}", micros(par)),
                        None => "-".to_string(),
                    },
                    micros(row.acvus) / micros(row.rust_seq),
                );
            }
        }
    }
}
