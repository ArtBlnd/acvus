//! SSA construction (Braun et al. 2013, through `Editor<Storage>::promote`):
//! every slot the editor lets go is promoted to values, which is every slot
//! `Escapes` names no use of.
//!
//! Form: F0, lowered, in; F1, SSA, out.
//!
//! Cost: O(blocks + edges + uses · log uses) per slot promoted, and
//! `Escapes` computed again for the next slot where a promotion made it
//! stale.
//!
//! Meaning: a promoted slot is reached through no reference, so its loads
//! and stores are exactly its values, and the order of effects, whether the
//! run ends and whether it traps are unchanged. Types are unchanged: each
//! value the promotion makes has the slot's type.

use crate::ir::ValueId;
use crate::ir::cfg::edit::{CannotPromote, Editor, Storage};
use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Mem2Reg;

impl BodyTransform for Mem2Reg {
    const NAME: &'static str = "mem2reg";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let slots: Vec<ValueId> = module.body(body).slots().to_vec();
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Storage>::new(module, body, &mut facts);
        for slot in slots {
            match editor.promote(slot) {
                Ok(()) | Err(CannotPromote::Escapes(_)) => {}
            }
        }
        editor.finish()
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::RefTarget;
    use crate::ir::cfg::{Op, Terminator};
    use crate::passes::optimize::tests::{Fixture, add, assert_whole, assign, int_const, jump, ret, take};
    use crate::ty::Mutability;

    /// `x = 1; if c { x = 2 }; while c { x = x + 1 }; return x`, with `x` a
    /// slot.
    #[test]
    fn a_slot_across_a_diamond_and_a_loop_becomes_values() {
        let mut f = Fixture::new();
        let (int, bool, nc) = (f.int, f.bool, f.name("c"));
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let x = builder.slot(int, Span::ZERO);
        let one = builder.push(int_const(1), int, Span::ZERO);
        builder.push_unit(assign(x, one), Span::ZERO);
        let [then_block, else_block, join, header, loop_body, exit] = [(); 6].map(|()| builder.new_block(&[]));
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        let two = builder.push(int_const(2), int, Span::ZERO);
        builder.push_unit(assign(x, two), Span::ZERO);
        builder.terminate(jump(join), acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        builder.terminate(jump(join), acvus_ast::Span::ZERO);
        builder.switch_to(join);
        builder.terminate(jump(header), acvus_ast::Span::ZERO);
        builder.switch_to(header);
        builder.terminate(Terminator::While { cond: c, body: loop_body, exit, exit_args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(loop_body);
        let read = builder.push(take(x), int, Span::ZERO);
        let step = builder.push(int_const(1), int, Span::ZERO);
        let next = builder.push(add(read, step), int, Span::ZERO);
        builder.push_unit(assign(x, next), Span::ZERO);
        builder.terminate(jump(header), acvus_ast::Span::ZERO);
        builder.switch_to(exit);
        let last = builder.push(take(x), int, Span::ZERO);
        builder.terminate(ret(last), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = f.run::<Mem2Reg>(id);

        let body = f.module.body(id);
        assert!(body.slots().is_empty(), "the slot is promoted");
        assert!(changes.touched(crate::ir::cfg::Aspect::Storage));
        let storage_ops = body
            .blocks()
            .flat_map(|(_, block)| block.insts())
            .filter(|inst| matches!(inst.op(), Op::Take { .. } | Op::Assign { .. }))
            .count();
        assert_eq!(storage_ops, 0, "every load and store of the slot is gone");
        let [joined] = body.block(join).params() else {
            panic!("the join takes the slot's value from its arms: {:?}", body.block(join).params())
        };
        let passed = |block| match body.block(block).terminator() {
            Terminator::Jump { args, .. } => args.clone(),
            other => panic!("{other:?} is a jump"),
        };
        assert_eq!((passed(then_block), passed(else_block)), (vec![two], vec![one]));
        assert_eq!(passed(join), vec![*joined]);
        let [carried] = body.block(header).params() else {
            panic!("the loop header takes the slot's value around the loop: {:?}", body.block(header).params())
        };
        let added = body.block(loop_body).insts().find(|inst| matches!(inst.op(), Op::BinOp { .. }));
        let Some(added) = added else { panic!("the loop body keeps its addition") };
        assert!(matches!(added.op(), Op::BinOp { left, .. } if left == carried));
        assert_eq!(passed(loop_body), vec![added.result().expect("an addition yields a value")]);
        assert!(matches!(body.block(exit).terminator(), Terminator::Return { value, .. } if value == carried));
        assert_whole(&f.module, id);
    }

    /// `x = 1; r = &x; t = *r; return t`: the reference escapes the
    /// promotion, so `x` stays a slot with its store and its reference.
    #[test]
    fn a_slot_a_reference_names_stays() {
        let mut f = Fixture::new();
        let (int, ref_int) = (f.int, f.ref_int);
        let mut builder = f.module.build_body(false, int);
        let x = builder.slot(int, Span::ZERO);
        let one = builder.push(int_const(1), int, Span::ZERO);
        builder.push_unit(assign(x, one), Span::ZERO);
        let r = builder.push(
            Op::Ref { target: RefTarget::Var(x), path: Vec::new(), mutability: Mutability::Shared },
            ref_int,
            Span::ZERO,
        );
        let t = builder.push(Op::Take { target: RefTarget::Through(r), path: Vec::new(), taken_out: false }, int, Span::ZERO);
        builder.terminate(ret(t), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);

        let changes = f.run::<Mem2Reg>(id);

        assert_eq!(f.module.body(id).slots(), &[x]);
        assert_eq!(f.snapshot(id), before, "nothing changes");
        assert!(!changes.touched(crate::ir::cfg::Aspect::Storage));
        assert_whole(&f.module, id);
    }
}
