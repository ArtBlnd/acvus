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

use std::sync::Arc;

use acvus_interpreter::code::Code;
use acvus_interpreter::code::Payload;
use acvus_interpreter::{PrepareCtx, prepare_module};
use acvus_interpreter_test::scripts::ATTENTION;
use acvus_interpreter_test::{compile_script_mode, split_context, value_from_json};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

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
    let (context_types, _snapshot): (FxHashMap<Astr, Ty>, _) = split_context(&interner, contexts);

    let source = format!("{ATTENTION} *out.get(0)");
    let cr = compile_script_mode(&interner, &source, &context_types);
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
                back_moves: body.back.len(),
            }),
            _ => None,
        })
        .collect()
}

/// History. On `63b31a42`, before the borrows left the loops, the four
/// loops were `head 2 body 11 back 0`, `head 1 body 8 back 1`,
/// `head 1 body 12 back 0`, `head 1 body 8 back 1`. Hoisting the borrows,
/// moving `s * scale` out of the inner head, and folding the arithmetic
/// chain took the inner bodies to 4 and 5.
///
/// RFC-0047 then replaced `get` with `a[i]`. The scores pass reads
/// `@keys[t]` once per row into a binding, so its outer body grew by the
/// row's `Index` and `AsSlice` while its inner body kept 4. The out pass
/// indexes `@values[t]` with `t` as the *inner* variable, so its row is
/// taken inside the inner loop and that body grew from 5 to 9. That is the
/// cost of the loop order, not of the instruction: an `AsSlice` hoists out
/// of a loop that does not define its container, and here the container is
/// defined by the loop itself.
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
            "head 1 body 12 back 0",
            "head 1 body 9 back 0",
            "head 1 body 7 back 1",
        ],
        "an operation in a head it does not belong to, or a back edge that moves, is a hoist that went too deep or a register it lengthened"
    );
}
