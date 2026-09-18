//! What the register selector makes of the attention loops: how many
//! operations one iteration of each `while` prepares, and how many slot
//! moves its back edge carries.
//!
//! `optimize::code_motion` moves work between blocks in two directions that
//! these counts measure. A borrow hoisted out of a loop lengthens a
//! register's live range, and a longer live range is what would force the
//! selector to add a move on the back edge; an instruction hoisted into a
//! loop head runs once per iteration instead of once. The counts are the
//! evidence that the first happened without the moves and the second does
//! not happen at all.

use acvus_interpreter_test::listing::{regions_named, script_listing};
use acvus_interpreter_test::scripts::ATTENTION;
use acvus_interpreter_test::value_from_json;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// One `while` as the machine runs it.
struct LoopShape {
    head_ops: usize,
    body_ops: usize,
    back_moves: usize,
}

fn attention_loops() -> Vec<LoopShape> {
    let interner = Interner::new();
    let json = serde_json::json!({
        "query": [0.1, 0.2],
        "keys": [[0.1, 0.2], [0.3, 0.4]],
        "values": [[1.0, 2.0], [3.0, 4.0]],
    });
    let contexts: acvus_interpreter_test::Context = json
        .as_object()
        .expect("an object of contexts")
        .iter()
        .map(|(name, value)| (interner.intern(name), value_from_json(&interner, value)))
        .collect();

    let source = format!("{ATTENTION} *out.get(0)");
    let blocks = script_listing(&interner, &source, contexts, Ty::Float);
    regions_named(&blocks, "Loop")
        .into_iter()
        .map(|region| {
            let head = region.part("head").expect("a Loop holds a head");
            let body = region.part("body").expect("a Loop holds a body");
            LoopShape {
                head_ops: head.ops.len(),
                body_ops: body.ops.len(),
                back_moves: body.leaves_with,
            }
        })
        .collect()
}

/// A region is an operation of the list it sits in (RFC-0052 §3), so each
/// outer body that holds a nested `while` counts that `Loop` among its own
/// operations.
///
/// The two middle bodies are the large ones because of the loop order, not
/// the instruction: an `AsSlice` hoists out of a loop that does not define
/// its container, and `@values[t]` is indexed by the *inner* variable, so
/// its row is taken inside the inner loop. Each of those bodies holds one
/// `AsSlice` and no drop — a slice is a register pair the frame never owns
/// (RFC-0047 amended, rules 1 and 2).
///
/// The last loop's back edge carries one move: its body's `CallExtern2`
/// writes the accumulator the head reads.
#[test]
fn each_loop_runs_only_what_its_own_nesting_level_holds() {
    let shapes: Vec<String> = attention_loops()
        .iter()
        .map(|l| {
            format!(
                "head {} body {} back {}",
                l.head_ops, l.body_ops, l.back_moves
            )
        })
        .collect();
    assert_eq!(
        shapes,
        [
            "head 1 body 4 back 0",
            "head 1 body 9 back 0",
            "head 1 body 7 back 0",
            "head 1 body 7 back 1",
        ],
        "an operation in a head it does not belong to, or a back edge that moves, is a hoist that went too deep or a register it lengthened"
    );
}
