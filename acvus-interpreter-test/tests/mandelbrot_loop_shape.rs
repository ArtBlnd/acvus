//! What the recognizer makes of the three nested `while`s the mandelbrot
//! bench runs: how many operations one iteration of each prepares, and how
//! many of them the short-circuit diamond in the innermost head holds.
//!
//! Under RFC-0052 §3 a region is an operation of the list it sits in, so
//! each count includes the regions that list holds: the innermost head is
//! its compare and its short-circuit `Diamond`, and each outer body counts
//! the nested `Loop` among its own operations.
//!
//! The counts are also the hoist guard. `total = total + i` is written
//! after the innermost `while` and belongs to the middle loop's body.
//! `optimize::code_motion` used to hoist it into the innermost head, where
//! it ran once per escape step instead of once per pixel, because a loop's
//! exit post-dominates its header and post-dominance was the whole hoist
//! condition. That hoist would show here as an innermost head of three and
//! a middle body of seventeen.

use acvus_interpreter::Value;
use acvus_interpreter_test::listing::{regions_named, script_listing};
use acvus_interpreter_test::{Context, typed};
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

    let blocks = script_listing(&interner, MANDELBROT, context, Ty::I64);
    regions_named(&blocks, "Loop")
        .into_iter()
        .map(|region| {
            let head = region.part("head").expect("a Loop holds a head");
            let body = region.part("body").expect("a Loop holds a body");
            LoopShape {
                head_ops: head.ops.len(),
                body_ops: body.ops.len(),
                diamonds_in_head: head.ops.iter().filter(|name| *name == "Diamond").count(),
            }
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
                body_ops: 5,
                diamonds_in_head: 1,
            },
            LoopShape {
                head_ops: 1,
                body_ops: 18,
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
