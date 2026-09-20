//! An interpreter inside the interpreter: a Brainfuck machine written in
//! acvus, next to the same machine in Rust.
//!
//! The point of this bench is not a number to improve but a hot body to
//! read. One loop carries a program counter and a tape pointer, dispatches
//! an eight-variant `enum Op` through `match`, reads and writes a
//! `Vec<i64>` tape through `a[i]`, and jumps through a `Vec<u64>` bracket
//! table built by a user function -- every construct the machine has, in
//! one body, with a data-dependent branch at its head.
//!
//! The Brainfuck program is a nested-loop multiply, built as data by the
//! script itself:
//!
//! ```text
//! +^A [ > +^B [ - > + < ] < - ] > > .
//! ```
//!
//! `A` scales with `@n`. A `.` adds the cell under the pointer to `out`
//! rather than printing it, and `,` is in the alphabet, is never executed,
//! and is what the `match`'s catch-all arm stands for.
//!
//! These timings hold only under one pinned core and a fixed load base;
//! `benches/README.md` states the protocol.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::{Owned, Registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::listing::{BlockListing, PartListing, script_listing_with_externs};
use acvus_interpreter_test::{
    Context, compile_source_with_externs, execute_compiled, split_context, typed,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use tokio::runtime::Runtime;

fn std_only() -> Vec<Registry<AcvusRuntime>> {
    acvus_ext::std_registries::<AcvusRuntime>()
}

// -- the Brainfuck program ----------------------------------------

/// The inner loop's trip count.
const B: i64 = 32;
/// Steps the machine executes per outer iteration, the `+` that built its
/// counter included: `> +^B` is `B + 1`, the inner loop is its one `[`
/// plus `B` runs of `- > + <` and `]`, and `< - ]` closes it. A `]` jumps
/// to its `[` and the following `pc + 1` enters the body, so a loop tests
/// its `[` once and its `]` every iteration. `@n / STEPS_PER_OUTER` is how
/// the script and the twin both size the program; `steps` in the printed
/// table is what the machines counted, and `measure` asserts they agree.
const STEPS_PER_OUTER: i64 = 6 * B + 6;
/// A nested-loop program has an outer loop, so the count never falls to
/// zero however small `n` is.
const MIN_OUTER: i64 = 1;
/// The tape, in cells. The program uses three of them.
const TAPE: i64 = 30000;

const INC: i64 = 0;
const DEC: i64 = 1;
const LEFT: i64 = 2;
const RIGHT: i64 = 3;
const OPEN: i64 = 4;
const CLOSE: i64 = 5;
const OUT: i64 = 6;

fn outer_count(n: i64) -> i64 {
    (n / STEPS_PER_OUTER).max(MIN_OUTER)
}

/// The same program the script's prelude builds. That the two agree is not
/// asserted here but in `measure`, where the two machines' `steps + out`
/// must be equal.
fn program(n: i64) -> Vec<i64> {
    let mut code = Vec::new();
    for _ in 0..outer_count(n) {
        code.push(INC);
    }
    code.push(OPEN);
    code.push(RIGHT);
    for _ in 0..B {
        code.push(INC);
    }
    code.extend([OPEN, DEC, RIGHT, INC, LEFT, CLOSE]);
    code.extend([LEFT, DEC, CLOSE]);
    code.extend([RIGHT, RIGHT, OUT]);
    code
}

// -- the Rust twin ------------------------------------------------

#[derive(Clone, Copy)]
enum Op {
    Inc,
    Dec,
    Left,
    Right,
    Open,
    Close,
    Out,
    In,
}

fn decode(code: &[i64]) -> Vec<Op> {
    code.iter()
        .map(|c| match *c {
            INC => Op::Inc,
            DEC => Op::Dec,
            LEFT => Op::Left,
            RIGHT => Op::Right,
            OPEN => Op::Open,
            CLOSE => Op::Close,
            OUT => Op::Out,
            _ => Op::In,
        })
        .collect()
}

fn bracket_table(code: &[i64]) -> Vec<u64> {
    let mut jumps = vec![0u64; code.len()];
    let mut stack: Vec<u64> = Vec::with_capacity(code.len());
    for (i, c) in code.iter().enumerate() {
        if *c == OPEN {
            stack.push(i as u64);
        }
        if *c == CLOSE
            && let Some(o) = stack.pop()
        {
            jumps[o as usize] = i as u64;
            jumps[i as usize] = o as u64;
        }
    }
    jumps
}

/// What a run produced: the steps it took and the sum of its `.` writes.
struct Ran {
    steps: i64,
    out: i64,
}

fn rust_table(n: i64) -> Ran {
    let code = program(n);
    let prog = decode(black_box(&code));
    let jumps = bracket_table(&code);
    let mut tape = vec![0i64; TAPE as usize];
    let mut pc = 0u64;
    let mut ptr = 0u64;
    let mut steps = 0i64;
    let mut out = 0i64;
    let plen = prog.len() as u64;
    while pc < plen {
        match prog[pc as usize] {
            Op::Inc => tape[ptr as usize] += 1,
            Op::Dec => tape[ptr as usize] -= 1,
            Op::Left => ptr -= 1,
            Op::Right => ptr += 1,
            Op::Open => {
                if tape[ptr as usize] == 0 {
                    pc = jumps[pc as usize];
                }
            }
            Op::Close => {
                if tape[ptr as usize] != 0 {
                    pc = jumps[pc as usize];
                }
            }
            Op::Out => out += tape[ptr as usize],
            Op::In => {}
        }
        pc += 1;
        steps += 1;
    }
    Ran { steps, out }
}

fn rust_scan(n: i64) -> Ran {
    let code = program(n);
    let prog = decode(black_box(&code));
    let mut tape = vec![0i64; TAPE as usize];
    let mut pc = 0u64;
    let mut ptr = 0u64;
    let mut steps = 0i64;
    let mut out = 0i64;
    let plen = prog.len() as u64;
    while pc < plen {
        match prog[pc as usize] {
            Op::Inc => tape[ptr as usize] += 1,
            Op::Dec => tape[ptr as usize] -= 1,
            Op::Left => ptr -= 1,
            Op::Right => ptr += 1,
            Op::Open => {
                if tape[ptr as usize] == 0 {
                    let mut d = 1u64;
                    while d > 0 {
                        pc += 1;
                        match prog[pc as usize] {
                            Op::Open => d += 1,
                            Op::Close => d -= 1,
                            _ => {}
                        }
                    }
                }
            }
            Op::Close => {
                if tape[ptr as usize] != 0 {
                    let mut d = 1u64;
                    while d > 0 {
                        pc -= 1;
                        match prog[pc as usize] {
                            Op::Open => d -= 1,
                            Op::Close => d += 1,
                            _ => {}
                        }
                    }
                }
            }
            Op::Out => out += tape[ptr as usize],
            Op::In => {}
        }
        pc += 1;
        steps += 1;
    }
    Ran { steps, out }
}

/// The twin of the script's `bump` lambda: the machine does not inline a
/// user function, so neither does this one.
#[inline(never)]
fn bump(x: i64, d: i64) -> i64 {
    x + d
}

fn rust_call(n: i64) -> Ran {
    let code = program(n);
    let prog = decode(black_box(&code));
    let jumps = bracket_table(&code);
    let mut tape = vec![0i64; TAPE as usize];
    let mut pc = 0u64;
    let mut ptr = 0u64;
    let mut steps = 0i64;
    let mut out = 0i64;
    let plen = prog.len() as u64;
    while pc < plen {
        match prog[pc as usize] {
            Op::Inc => tape[ptr as usize] = bump(tape[ptr as usize], 1),
            Op::Dec => tape[ptr as usize] = bump(tape[ptr as usize], -1),
            Op::Left => ptr -= 1,
            Op::Right => ptr += 1,
            Op::Open => {
                if tape[ptr as usize] == 0 {
                    pc = jumps[pc as usize];
                }
            }
            Op::Close => {
                if tape[ptr as usize] != 0 {
                    pc = jumps[pc as usize];
                }
            }
            Op::Out => out += tape[ptr as usize],
            Op::In => {}
        }
        pc += 1;
        steps += 1;
    }
    Ran { steps, out }
}

// -- the interpreter in acvus -------------------------------------

/// The tape, the `u64` zero and one that `a[i]` needs (an integer literal
/// is `i64` and the index is a `u64`, and the language has no `u64`
/// literal, so both are derived from `len`), the program built as data,
/// and the decode pass that turns the opcode numbers into the `enum Op`
/// the hot loop matches.
fn prelude() -> String {
    format!(
        "\
let tape = filled({TAPE}, 0); \
let one = len(&tape) / len(&tape); \
let zero = len(&tape) - len(&tape); \
let a0 = @n / {STEPS_PER_OUTER}; \
let a = if a0 < {MIN_OUTER} {{ {MIN_OUTER} }} else {{ a0 }}; \
let code = with_capacity(64); \
let k = 0; \
while k < a {{ code.push({INC}); k = k + 1; }} \
code.extend(vec([{OPEN}, {RIGHT}])); \
k = 0; \
while k < {B} {{ code.push({INC}); k = k + 1; }} \
code.extend(vec([{OPEN}, {DEC}, {RIGHT}, {INC}, {LEFT}, {CLOSE}, \
{LEFT}, {DEC}, {CLOSE}, {RIGHT}, {RIGHT}, {OUT}])); \
let prog = as_iter(&code) | map(|c| -> \
if *c == {INC} {{ Op::Inc }} \
else if *c == {DEC} {{ Op::Dec }} \
else if *c == {LEFT} {{ Op::Left }} \
else if *c == {RIGHT} {{ Op::Right }} \
else if *c == {OPEN} {{ Op::Open }} \
else if *c == {CLOSE} {{ Op::Close }} \
else if *c == {OUT} {{ Op::Out }} \
else {{ Op::In }}) | collect; \
let plen = len(&prog); \
"
    )
}

/// The bracket table, as a user function over the opcode vector. That this
/// is one `CallDirect` is a claim about the listing; `PROGRAMS_OPLIST=...`
/// prints the listing that decides it.
fn jumps_pass() -> String {
    format!(
        "\
let build_jumps = |cc| -> {{ \
let m = len(&cc); \
let one2 = m / m; \
let z = m - m; \
let jm = with_capacity(m); \
let i = z; \
while i < m {{ jm.push(z); i = i + one2; }} \
let st = with_capacity(m); \
i = z; \
while i < m {{ \
let c = cc[i]; \
if c == {OPEN} {{ st.push(i); }}; \
if c == {CLOSE} {{ if let Some(o) = st.pop() {{ jm[o] = i; jm[i] = o; }}; }}; \
i = i + one2; \
}} \
jm \
}}; \
let jumps = build_jumps(code); \
"
    )
}

const STATE: &str = "\
let pc = zero; \
let ptr = zero; \
let steps = 0; \
let out = 0; \
";

const TAIL: &str = "steps + out";

/// The hot body, dispatching through the bracket table.
const BODY_TABLE: &str = "\
while pc < plen { \
match &prog[pc] { \
Op::Inc => { tape[ptr] = tape[ptr] + 1; }, \
Op::Dec => { tape[ptr] = tape[ptr] - 1; }, \
Op::Left => { ptr = ptr - one; }, \
Op::Right => { ptr = ptr + one; }, \
Op::Open => { if tape[ptr] == 0 { pc = jumps[pc]; }; }, \
Op::Close => { if tape[ptr] != 0 { pc = jumps[pc]; }; }, \
Op::Out => { out = out + tape[ptr]; }, \
_ => { steps = steps; } \
}; \
pc = pc + one; \
steps = steps + 1; \
} \
";

/// The same body with the bracket table replaced by a scan: a nested
/// `while` per jump, with a second `match` in it.
const BODY_SCAN: &str = "\
while pc < plen { \
match &prog[pc] { \
Op::Inc => { tape[ptr] = tape[ptr] + 1; }, \
Op::Dec => { tape[ptr] = tape[ptr] - 1; }, \
Op::Left => { ptr = ptr - one; }, \
Op::Right => { ptr = ptr + one; }, \
Op::Open => { if tape[ptr] == 0 { \
let d = one; \
while d > zero { \
pc = pc + one; \
match &prog[pc] { \
Op::Open => { d = d + one; }, \
Op::Close => { d = d - one; }, \
_ => { d = d; } \
}; \
} \
}; }, \
Op::Close => { if tape[ptr] != 0 { \
let d = one; \
while d > zero { \
pc = pc - one; \
match &prog[pc] { \
Op::Open => { d = d - one; }, \
Op::Close => { d = d + one; }, \
_ => { d = d; } \
}; \
} \
}; }, \
Op::Out => { out = out + tape[ptr]; }, \
_ => { steps = steps; } \
}; \
pc = pc + one; \
steps = steps + 1; \
} \
";

/// The same scan with the inner `while`'s condition written as a `break`:
/// the one hot body in this bench whose loop holds an exit edge.
const BODY_BREAK: &str = "\
while pc < plen { \
match &prog[pc] { \
Op::Inc => { tape[ptr] = tape[ptr] + 1; }, \
Op::Dec => { tape[ptr] = tape[ptr] - 1; }, \
Op::Left => { ptr = ptr - one; }, \
Op::Right => { ptr = ptr + one; }, \
Op::Open => { if tape[ptr] == 0 { \
let d = one; \
while d > zero { \
pc = pc + one; \
match &prog[pc] { \
Op::Open => { d = d + one; }, \
Op::Close => { d = d - one; }, \
_ => { d = d; } \
}; \
if d == zero { break; }; \
} \
}; }, \
Op::Close => { if tape[ptr] != 0 { \
let d = one; \
while d > zero { \
pc = pc - one; \
match &prog[pc] { \
Op::Open => { d = d - one; }, \
Op::Close => { d = d + one; }, \
_ => { d = d; } \
}; \
if d == zero { break; }; \
} \
}; }, \
Op::Out => { out = out + tape[ptr]; }, \
_ => { steps = steps; } \
}; \
pc = pc + one; \
steps = steps + 1; \
} \
";

/// The same body with the cell update routed through a user function: one
/// `CallDirect` per `+`/`-` step.
const BODY_CALL: &str = "\
let bump = |x, d| -> x + d; \
while pc < plen { \
match &prog[pc] { \
Op::Inc => { tape[ptr] = bump(tape[ptr], 1); }, \
Op::Dec => { tape[ptr] = bump(tape[ptr], -1); }, \
Op::Left => { ptr = ptr - one; }, \
Op::Right => { ptr = ptr + one; }, \
Op::Open => { if tape[ptr] == 0 { pc = jumps[pc]; }; }, \
Op::Close => { if tape[ptr] != 0 { pc = jumps[pc]; }; }, \
Op::Out => { out = out + tape[ptr]; }, \
_ => { steps = steps; } \
}; \
pc = pc + one; \
steps = steps + 1; \
} \
";

fn table_source() -> String {
    format!("{}{}{STATE}{BODY_TABLE}{TAIL}", prelude(), jumps_pass())
}

fn scan_source() -> String {
    format!("{}{STATE}{BODY_SCAN}{TAIL}", prelude())
}

fn break_source() -> String {
    format!("{}{STATE}{BODY_BREAK}{TAIL}", prelude())
}

fn call_source() -> String {
    format!("{}{}{STATE}{BODY_CALL}{TAIL}", prelude(), jumps_pass())
}

// -- the harness --------------------------------------------------

struct Case {
    name: &'static str,
    source: String,
    rust: fn(i64) -> Ran,
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
    steps: i64,
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
            acvus_ast::parse_script(&interner, &case.source).expect("parse error"),
        );
        let cr = compile_source_with_externs(&interner, ast, &context_types, std_only(), Ty::I64);
        let (_shared, mut interp) = execute_compiled(
            &interner,
            cr,
            snapshot(&interner, n),
            Arc::new(SequentialExecutor),
        );
        let start = Instant::now();
        let value = rt.block_on(interp.execute());
        (value.as_int(), start.elapsed())
    };
    let run_rust = || {
        let start = Instant::now();
        let this = (case.rust)(black_box(n));
        black_box(this.out);
        (this, start.elapsed())
    };

    let (mut script_value, _warm_up) = run_script();
    let mut execute = Vec::with_capacity(reps.get());
    for _ in 0..reps.get() {
        let (value, elapsed) = run_script();
        script_value = value;
        execute.push(elapsed);
    }

    let (mut ran, _warm_up) = run_rust();
    let mut rust = Vec::with_capacity(reps.get());
    for _ in 0..reps.get() {
        let (this, elapsed) = run_rust();
        ran = this;
        rust.push(elapsed);
    }

    assert!(
        ran.out == outer_count(n) * B,
        "{} n={n}: the nested-loop multiply left {} under the pointer, not {}",
        case.name,
        ran.out,
        outer_count(n) * B
    );
    let rust_value = ran.steps + ran.out;
    assert!(
        script_value == rust_value,
        "{} n={n}: script produced {script_value}, Rust produced {rust_value}",
        case.name
    );
    Timing {
        execute: median(execute),
        rust: median(rust),
        steps: ran.steps,
    }
}

/// Which machine a `perf stat` run is to count. A run under `perf` must
/// execute one side alone, so this path makes no comparison; the check
/// that the two agree is `measure`'s, on the default path.
enum Side {
    Script,
    Rust,
}

/// Runs a `perf` run makes, the first being the warm-up `measure` also
/// drops. `perf` counts them all, and the per-step figure is the
/// difference between two sizes, so the warm-up cancels.
const PERF_REPS: usize = 6;

fn perf_run(rt: &Runtime, case: &Case, n: i64, side: Side) {
    let steps = match side {
        Side::Script => {
            let interner = Interner::new();
            let context_types = split_context(&interner, context(&interner, n)).0;
            let mut value = None;
            for _ in 0..PERF_REPS {
                let ast = ParsedAst::Script(
                    acvus_ast::parse_script(&interner, &case.source).expect("parse error"),
                );
                let cr = compile_source_with_externs(
                    &interner,
                    ast,
                    &context_types,
                    std_only(),
                    Ty::I64,
                );
                let (_shared, mut interp) = execute_compiled(
                    &interner,
                    cr,
                    snapshot(&interner, n),
                    Arc::new(SequentialExecutor),
                );
                value = Some(rt.block_on(interp.execute()).as_int());
            }
            value.expect("PERF_REPS is at least one") - outer_count(n) * B
        }
        Side::Rust => {
            let mut ran = None;
            for _ in 0..PERF_REPS {
                let this = (case.rust)(black_box(n));
                black_box(this.out);
                ran = Some(this);
            }
            ran.expect("PERF_REPS is at least one").steps
        }
    };
    println!("{} n={n} reps={PERF_REPS} steps={steps}", case.name);
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "bf table",
            source: table_source(),
            rust: rust_table,
        },
        Case {
            name: "bf scan",
            source: scan_source(),
            rust: rust_scan,
        },
        Case {
            name: "bf call",
            source: call_source(),
            rust: rust_call,
        },
        Case {
            name: "bf break",
            source: break_source(),
            rust: rust_scan,
        },
    ]
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

fn source_named(name: &str) -> String {
    let found = cases().into_iter().find(|case| case.name == name);
    found
        .unwrap_or_else(|| {
            let all: Vec<&str> = cases().iter().map(|case| case.name).collect();
            panic!("no case named {name:?}: it is one of {all:?}")
        })
        .source
}

/// The block labels and predecessors the op list's blocks are prepared
/// from, which the listing itself does not carry.
fn mir(name: &str) {
    let interner = Interner::new();
    let source = source_named(name);
    let context_types = split_context(&interner, context(&interner, 1_000_000)).0;
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, &source).expect("parse error"));
    let cr = compile_source_with_externs(&interner, ast, &context_types, std_only(), Ty::I64);
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    println!("== mir: {name}");
    println!("{}", acvus_mir::printer::dump_with(&interner, module));
}

fn oplist(name: &str) {
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        &source_named(name),
        context(&interner, 1_000_000),
        std_only(),
        Ty::I64,
    );
    println!("== oplist: {name}");
    print_listing(&blocks);
}

fn main() {
    if let Ok(name) = std::env::var("PROGRAMS_OPLIST") {
        oplist(&name);
        return;
    }
    if let Ok(name) = std::env::var("PROGRAMS_SOURCE") {
        println!("{}", source_named(&name));
        return;
    }
    if let Ok(name) = std::env::var("PROGRAMS_MIR") {
        mir(&name);
        return;
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");
    if let Ok(side) = std::env::var("PROGRAMS_ONLY") {
        let side = match side.as_str() {
            "script" => Side::Script,
            "rust" => Side::Rust,
            other => panic!("no side named {other:?}: it is \"script\" or \"rust\""),
        };
        let name = std::env::var("PROGRAMS_CASE").expect("PROGRAMS_ONLY names a case too");
        let n: i64 = std::env::var("PROGRAMS_N")
            .expect("PROGRAMS_ONLY names a size too")
            .parse()
            .expect("PROGRAMS_N is an integer");
        let case = cases()
            .into_iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no case named {name:?}"));
        perf_run(&rt, &case, n, side);
        return;
    }
    println!(
        "{:>12} {:>10} {:>12} {:>14} {:>12} {:>12} {:>12}",
        "case", "n", "steps", "execute/us", "rust/us", "ratio", "ns/step"
    );
    let only = std::env::var("PROGRAMS_CASE").ok();
    for case in cases()
        .iter()
        .filter(|c| only.as_deref().is_none_or(|o| c.name == o))
    {
        let sizes = match std::env::var("PROGRAMS_N").ok() {
            Some(n) => vec![Size {
                n: n.parse().expect("PROGRAMS_N is an integer"),
                reps: reps(1),
            }],
            None => vec![
                Size {
                    n: 1_000_000,
                    reps: reps(3),
                },
                Size {
                    n: 5_000_000,
                    reps: reps(2),
                },
            ],
        };
        for size in sizes {
            let Timing {
                execute,
                rust,
                steps,
            } = measure(&rt, case, &size);
            println!(
                "{:>12} {:>10} {:>12} {:>14.1} {:>12.1} {:>12.1} {:>12.1}",
                case.name,
                size.n,
                steps,
                micros(execute),
                micros(rust),
                micros(execute) / micros(rust),
                execute.as_secs_f64() * 1e9 / steps as f64
            );
        }
    }
}
