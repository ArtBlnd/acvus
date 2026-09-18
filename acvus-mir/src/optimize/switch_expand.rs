//! Expand `Terminator::Switch` into the `TestVariant` + `JumpIf` chain the
//! machine runs today (RFC-0051, first half).
//!
//! The lowering emits one `Switch` per `match`, because that is the shape
//! that names a dispatch and nothing else: `validate`'s exhaustiveness pass
//! reads it, and an `if let` -- two arms, always exhaustive -- never wears
//! one. The interpreter's `switch` operation is the second half; until it
//! lands this pass runs first in the optimization pipeline, so nothing
//! downstream of it ever sees a `Switch`.
//!
//! Cross-artifact obligation: the second half deletes the call to this pass
//! in `graph::optimize::run_pass1_body` and adds the `switch` handler in
//! `acvus-interpreter/src/prepare.rs`.

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::cfg::{Block, BlockIdx, CfgBody, ENTRY_LABEL, Terminator};
use crate::ir::{Inst, InstKind, Label, ValueId};
use crate::ty::Ty;

/// The first label a body can hold: `Label(0)`. A body whose blocks carry no
/// label but the entry sentinel has all of the space free.
const FIRST_LABEL: u32 = 0;

/// The span a synthesized instruction carries: it stands for no source text,
/// which is what `cfg::demote` writes for the terminators it rebuilds.
const SYNTHESIZED: acvus_ast::Span = acvus_ast::Span::ZERO;

/// Where one edge of a `Switch` goes and what it hands the block's
/// parameters. `Terminator::Switch` stores the same three fields
/// positionally (RFC-0051 §5 fixes that shape); this is them named, for the
/// expansion to carry around.
#[derive(Clone)]
struct Edge {
    target: Label,
    args: Vec<ValueId>,
}

/// One tag test of the expanded chain: `cond = test_variant tag, variant`
/// followed by `jump_if cond then <taken> else <next>`.
struct Test {
    variant: Astr,
    taken: Edge,
}

pub fn run(cfg: &mut CfgBody) {
    let heads: Vec<usize> = cfg
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.terminator, Terminator::Switch { .. }))
        .map(|(bi, _)| bi)
        .collect();
    if heads.is_empty() {
        return;
    }
    // The label space above every label the body already uses. `max` is
    // `None` only when no block but the entry carries a label, and then the
    // whole space is free.
    let mut next_label = cfg
        .blocks
        .iter()
        .map(|b| b.label)
        .filter(|l| *l != ENTRY_LABEL)
        .map(|l| l.0 + 1)
        .max()
        .unwrap_or(FIRST_LABEL);

    let mut inserted: FxHashMap<BlockIdx, Vec<Block>> = FxHashMap::default();
    for bi in heads {
        let Terminator::Switch { tag, arms, default } =
            std::mem::replace(&mut cfg.blocks[bi].terminator, Terminator::Fallthrough)
        else {
            unreachable!("`heads` holds only the blocks that end in a Switch")
        };

        // The edge the chain falls out of: the catch-all if there is one,
        // otherwise the last arm, which is then not tested. That is sound
        // exactly because the exhaustiveness pass decided some arm holds.
        let (tests, fallthrough) = split_last_edge(arms, default);

        // One bridge block per test after the first; the first test stays in
        // the head block.
        let bridges: Vec<Label> = (1..tests.len())
            .map(|_| {
                let label = Label(next_label);
                next_label += 1;
                label
            })
            .collect();

        let mut bridge_blocks: Vec<Block> = Vec::with_capacity(bridges.len());
        for (i, test) in tests.iter().enumerate() {
            let cond = cfg.val_factory.next();
            // A fresh ValueId names no earlier type, so the previous entry
            // this returns is always `None`.
            cfg.val_types.insert(cond, Ty::Bool);
            let inst = Inst {
                span: SYNTHESIZED,
                kind: InstKind::TestVariant {
                    dst: cond,
                    src: tag,
                    tag: test.variant,
                },
            };
            let next = match bridges.get(i) {
                Some(label) => Edge {
                    target: *label,
                    args: Vec::new(),
                },
                None => fallthrough.clone(),
            };
            let terminator = Terminator::JumpIf {
                cond,
                then_label: test.taken.target,
                then_args: test.taken.args.clone(),
                else_label: next.target,
                else_args: next.args,
            };
            match i.checked_sub(1) {
                None => {
                    cfg.blocks[bi].insts.push(inst);
                    cfg.blocks[bi].terminator = terminator;
                }
                Some(bridge) => bridge_blocks.push(Block {
                    label: bridges[bridge],
                    params: Vec::new(),
                    insts: vec![inst],
                    terminator,
                    merge_of: None,
                }),
            }
        }
        if !bridge_blocks.is_empty() {
            inserted.insert(BlockIdx(bi), bridge_blocks);
        }
    }

    if inserted.is_empty() {
        return;
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(cfg.blocks.len() + inserted.len());
    for (bi, block) in std::mem::take(&mut cfg.blocks).into_iter().enumerate() {
        blocks.push(block);
        // A block with nothing inserted after it takes no bridge: the empty
        // list is the answer, not a stand-in for one.
        blocks.extend(inserted.remove(&BlockIdx(bi)).unwrap_or_default());
    }
    cfg.label_to_block = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label, BlockIdx(i)))
        .collect();
    cfg.blocks = blocks;
}

/// The tests the chain runs, and the edge it falls out of. A `Switch` with
/// neither an arm nor a default names no successor at all; the lowering
/// never builds one, and the expansion cannot invent a target for it.
fn split_last_edge(
    arms: Vec<(Astr, Label, Vec<ValueId>)>,
    default: Option<(Label, Vec<ValueId>)>,
) -> (Vec<Test>, Edge) {
    let test = |(variant, target, args): (Astr, Label, Vec<ValueId>)| Test {
        variant,
        taken: Edge { target, args },
    };
    match default {
        Some((target, args)) => (arms.into_iter().map(test).collect(), Edge { target, args }),
        None => {
            let mut arms = arms;
            let (_, target, args) = arms
                .pop()
                .expect("a Switch names at least one successor: an arm or a default");
            (arms.into_iter().map(test).collect(), Edge { target, args })
        }
    }
}

// -- Tests ---------------------------------------------------------

#[cfg(test)]
mod tests {
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::cfg::{demote, promote};
    use crate::ir::{DebugInfo, MirBody, MirModule};
    use crate::ty::Task;

    /// `switch r1 { A -> L0, B -> L1 }` over a value built in this body,
    /// with a merge block that takes the arms' results.
    fn hand_built(interner: &Interner) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val = || factory.next();
        let (one, variant, ten, twenty, result) = (val(), val(), val(), val(), val());
        let a = interner.intern("A");
        let b = interner.intern("B");
        let enum_ty = Ty::Enum {
            name: interner.intern("E"),
            variants: FxHashMap::from_iter([
                (a, Some(Box::new(Ty::I64))),
                (b, Some(Box::new(Ty::I64))),
            ]),
        };
        let inst = |kind| Inst {
            span: SYNTHESIZED,
            kind,
        };
        MirBody {
            insts: vec![
                inst(InstKind::Const {
                    dst: one,
                    value: acvus_ast::Literal::Int(1),
                }),
                inst(InstKind::MakeVariant {
                    dst: variant,
                    tag: a,
                    payload: Some(one),
                }),
                inst(InstKind::Switch {
                    tag: variant,
                    arms: vec![(a, Label(0), Vec::new()), (b, Label(1), Vec::new())],
                    default: None,
                }),
                inst(InstKind::BlockLabel {
                    label: Label(0),
                    params: Vec::new(),
                    merge_of: None,
                }),
                inst(InstKind::Const {
                    dst: ten,
                    value: acvus_ast::Literal::Int(10),
                }),
                inst(InstKind::Jump {
                    label: Label(2),
                    args: vec![ten],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(1),
                    params: Vec::new(),
                    merge_of: None,
                }),
                inst(InstKind::Const {
                    dst: twenty,
                    value: acvus_ast::Literal::Int(20),
                }),
                inst(InstKind::Jump {
                    label: Label(2),
                    args: vec![twenty],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![result],
                    merge_of: Some(Label(0)),
                }),
                inst(InstKind::Return {
                    value: result,
                    order: None,
                }),
            ],
            val_types: FxHashMap::from_iter([
                (one, Ty::I64),
                (variant, enum_ty),
                (ten, Ty::I64),
                (twenty, Ty::I64),
                (result, Ty::I64),
            ]),
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            task: Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 3,
        }
    }

    #[test]
    fn a_switch_survives_every_pass_and_prints() {
        let interner = Interner::new();
        let body = hand_built(&interner);
        let module = MirModule {
            main: body.clone(),
            closures: FxHashMap::default(),
            ret: crate::ty::Ty::Unit,
        };
        let printed = format!("{}", crate::printer::dump(&interner, &module));
        assert!(
            printed.contains("switch r0 { A -> L0, B -> L1 }"),
            "{printed}"
        );

        // The checks and passes that see a `Switch` before this pass runs.
        assert!(crate::validate::move_check::check_moves(&module).is_empty());
        assert!(crate::validate::borrow_check::check_borrows(&module).is_empty());
        let mut cfg = promote(body);
        assert_eq!(cfg.successors(BlockIdx(0)).len(), 2, "both arms are edges");
        crate::optimize::ssa_pass::run(&mut cfg);
        crate::optimize::string_copy::run(&mut cfg);
        crate::optimize::dce::run(&mut cfg);
        crate::optimize::code_motion::run(&mut cfg);
        let val_types = cfg.val_types.clone();
        crate::optimize::drop_insertion::insert_drops(&mut cfg, &val_types);
        assert!(
            cfg.blocks
                .iter()
                .any(|b| matches!(b.terminator, Terminator::Switch { .. })),
            "no pass drops the dispatch"
        );
        let roundtripped = demote(cfg);
        assert!(
            roundtripped
                .insts
                .iter()
                .any(|i| matches!(i.kind, InstKind::Switch { .. })),
            "promote and demote carry it"
        );
    }

    /// Without a `default`, the last arm is the chain's else: two arms cost
    /// one `TestVariant`.
    #[test]
    fn expansion_tests_every_arm_but_the_last() {
        let interner = Interner::new();
        let mut cfg = promote(hand_built(&interner));
        run(&mut cfg);
        assert!(
            cfg.blocks
                .iter()
                .all(|b| !matches!(b.terminator, Terminator::Switch { .. }))
        );
        let tests = cfg
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.kind, InstKind::TestVariant { .. }))
            .count();
        assert_eq!(tests, 1, "two arms, one tag test");
        let module = MirModule {
            main: demote(cfg),
            closures: FxHashMap::default(),
            ret: crate::ty::Ty::Unit,
        };
        let printed = format!("{}", crate::printer::dump(&interner, &module));
        assert!(printed.contains("is A"), "{printed}");
        assert!(!printed.contains("switch "), "{printed}");
    }

    /// With a `default`, every arm is tested and the catch-all is the else.
    #[test]
    fn expansion_tests_every_arm_when_a_default_is_present() {
        let interner = Interner::new();
        let mut body = hand_built(&interner);
        let InstKind::Switch { arms, default, .. } = &mut body.insts[2].kind else {
            unreachable!("the hand-built body's third instruction is the Switch")
        };
        *default = Some((Label(1), Vec::new()));
        arms.pop();
        let mut cfg = promote(body);
        run(&mut cfg);
        let tests = cfg
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.kind, InstKind::TestVariant { .. }))
            .count();
        assert_eq!(tests, 1, "one arm and a default: one tag test");
    }
}
