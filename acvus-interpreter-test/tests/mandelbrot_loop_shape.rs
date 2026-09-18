//! What the register selector makes of the three nested `while`s the
//! mandelbrot bench runs: which of them it recognizes as loops, and how
//! many operations one iteration of each prepares.
//!
//! RFC-0020 lowers the innermost `while`'s condition `i < @max && ...` to
//! a branch diamond, and `prepare::recognize_loop` takes a head that is a
//! straight run ending in the one `jump_if` whose else-edge is the exit.
//! The diamond's `jump_if` is neither, so that loop is not recognized, and
//! an unrecognized loop inside a body stops the body from being a straight
//! run — so neither enclosing `while` is recognized either. Restoring the
//! recognition is the diamond op's work; until then this file measures the
//! absence, and the bench pays for it (50.3 ns/iteration against 41.1 with
//! the three loops recognized, 80x40 at max 100).
//!
//! What the counts used to guard: `total = total + i` is written after the
//! innermost loop and must be prepared after it. `optimize::code_motion`
//! used to hoist it into that loop's head, where it ran once per escape
//! step instead of once per pixel, because a loop's exit post-dominates
//! its header and post-dominance was the whole hoist condition. With no
//! loop recognized there is no head to hoist into, so that guard comes
//! back with the recognition.

use std::sync::Arc;

use acvus_interpreter::code::Code;
use acvus_interpreter::code::Payload;
use acvus_interpreter::prepare_module;
use acvus_interpreter::{PrepareCtx, Value};
use acvus_interpreter_test::{Context, compile_script_mode, split_context, typed};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// The bench's program, character for character. `benches/mandelbrot.rs`
/// holds the other copy and times it; the counts below describe that copy,
/// so the two texts must stay equal.
const MANDELBROT: &str = "
let total = 0;
let py = 0;
while py < @h {
    let px = 0;
    while px < @w {
        let cx = -2.0 + 3.0 * px.to_float() / @w.to_float();
        let cy = -1.2 + 2.4 * py.to_float() / @h.to_float();
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

/// One `while` as the machine runs it.
struct LoopShape {
    head_ops: usize,
    body_ops: usize,
}

fn mandelbrot_loops() -> Vec<LoopShape> {
    let interner = Interner::new();
    let context: Context = [
        (interner.intern("w"), typed(Ty::I64, Value::int(16))),
        (interner.intern("h"), typed(Ty::I64, Value::int(16))),
        (interner.intern("max"), typed(Ty::I64, Value::int(32))),
    ]
    .into_iter()
    .collect();
    let (context_types, _snapshot) = split_context(&interner, context);

    let cr = compile_script_mode(&interner, MANDELBROT, &context_types);
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    let prepared = Arc::new(prepare_module(module, &ctx));

    let Code::Body(main) = &*prepared.main else {
        panic!("the entry module's main is a body, not a one-chain expression")
    };
    main.payloads
        .iter()
        .filter_map(|payload| match payload {
            Payload::Loop(body) => Some(LoopShape {
                head_ops: body.head.iter().count(),
                body_ops: body.body.iter().count(),
            }),
            _ => None,
        })
        .collect()
}

#[test]
fn a_short_circuit_condition_leaves_no_while_for_the_recognizer() {
    let shapes: Vec<String> = mandelbrot_loops()
        .iter()
        .map(|l| format!("head {} body {}", l.head_ops, l.body_ops))
        .collect();
    let none: [String; 0] = [];
    assert_eq!(
        shapes, none,
        "a `while` recognized here runs as one loop op; the short-circuit diamond in the innermost head is what stops the recognizer, and its cost is the bench's"
    );
}
