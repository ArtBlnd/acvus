//! What the recognizer makes of the three nested `while`s the mandelbrot
//! bench runs: how many operations one iteration of each prepares, and how
//! many of them the short-circuit diamond in the innermost head holds.
//!
//! The counts are also the hoist guard. `total = total + i` is written
//! after the innermost `while` and belongs to the middle loop's body.
//! `optimize::code_motion` used to hoist it into the innermost head, where
//! it ran once per escape step instead of once per pixel, because a loop's
//! exit post-dominates its header and post-dominance was the whole hoist
//! condition. That hoist would show here as an innermost head of three and
//! a middle body of eighteen.

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
#[derive(Debug, PartialEq)]
struct LoopShape {
    head_ops: usize,
    body_ops: usize,
    diamonds_in_head: usize,
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
                diamonds_in_head: body
                    .head
                    .iter()
                    .filter(|op| matches!(&main.payloads[op.p], Payload::Diamond(_)))
                    .count(),
            }),
            _ => None,
        })
        .collect()
}

#[test]
fn every_while_is_one_loop_operation_and_the_diamond_is_one_more() {
    assert_eq!(
        mandelbrot_loops(),
        vec![
            LoopShape {
                head_ops: 2,
                body_ops: 4,
                diamonds_in_head: 1,
            },
            LoopShape {
                head_ops: 1,
                body_ops: 19,
                diamonds_in_head: 0,
            },
            LoopShape {
                head_ops: 1,
                body_ops: 4,
                diamonds_in_head: 0,
            },
        ],
        "the loops are listed innermost first: `i < @max` and the short-circuit \
         diamond, then the pixel loop whose body ends in `total = total + i`, \
         then the row loop"
    );
}
