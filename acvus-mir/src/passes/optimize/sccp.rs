//! Sparse conditional constant propagation (Wegman and Zadeck), reading
//! `ConstantValues`:
//! - a `JumpIf` or a `Diamond` whose condition is known takes its one arm,
//!   and a `Switch` whose tag is known takes its key's arm, or its default
//!   where no arm has that key (`take_case`). A `Switch` on a string keeps
//!   every arm, since the analysis takes every arm of one (D60), and so
//!   does one whose known key neither an arm nor a default takes, since no
//!   arm of it runs;
//! - an instruction's result known constant, where the instruction is not
//!   already a `Const` and something reads the result, gets a `Const`
//!   inserted just before the instruction, with its span, and its uses take
//!   that constant; the computation, left unread, is `dce`'s;
//! - blocks no edge reaches any more are pruned.
//!
//! The arms are taken first. Taking an arm replaces no value, so every value
//! the constants step then asks about is one the analysis saw.
//!
//! A block parameter known constant is not folded yet. Its constant would
//! stand at the block's start, and no span of the program belongs to a
//! block parameter, while every instruction carries one; which span such a
//! constant takes is an open question to the coordinator.
//!
//! Form: F1, SSA, in and out.
//!
//! Cost: the analysis's, O(uses × 3); then O(blocks + instructions), one
//! insertion and one replacement per constant result, and one prune.
//!
//! Meaning: a value proven constant is that constant on every run that
//! reaches it, and an arm no run takes is never run. An operation that
//! traps on its known operands is not a constant of the domain, so no trap
//! is folded away. Every inserted constant has the type of the value it
//! stands for.

use acvus_ast::{Literal, Span};

use crate::ir::{SwitchKey, ValueId};
use crate::ir::cfg::edit::{DoesNotDominate, Editor, Position, Sccp as SccpScope};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Module, Op, Terminator};
use crate::passes::analysis::{AnalysisManager, ConstantValues, SparseFacts, switch_key};
use crate::passes::domain::ConstantDomain;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Sccp;

impl BodyTransform for Sccp {
    const NAME: &'static str = "sccp";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let constants = am.get::<ConstantValues>(module, body);
        let known_arms = known_arms(module.body(body), &constants);
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<SccpScope>::new(module, body, &mut facts);
        for (block, arm) in known_arms {
            match arm {
                KnownArm::Branch(then) => editor.take_arm(block, then),
                KnownArm::Case(key) => editor.take_case(block, &key),
            }
        }
        fold_constants(&mut editor, &constants);
        editor.prune();
        editor.finish()
    }
}

/// The arm a branch whose condition or tag is known takes.
enum KnownArm {
    Branch(bool),
    Case(SwitchKey),
}

/// Every `JumpIf`, `Diamond` and `Switch` whose condition or tag the
/// analysis knows, with the arm it takes. Taking an arm renames no block,
/// so these stay valid while the arms are taken.
fn known_arms(body: &Body, constants: &SparseFacts<ConstantDomain>) -> Vec<(BlockId, KnownArm)> {
    body.blocks()
        .filter_map(|(id, block)| match block.terminator() {
            Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } => match constants.constant(*cond) {
                Some(Literal::Bool(then)) => Some((id, KnownArm::Branch(*then))),
                Some(other) => panic!("{id:?} branches on {cond:?}, known as {other:?}, which is not a bool"),
                None => None,
            },
            Terminator::Switch { tag, arms, default, .. } => {
                let key = switch_key(&constants.constant(*tag)?.desugared())?;
                let taken = arms.iter().any(|(arm, _, _)| *arm == key) || default.is_some();
                taken.then_some((id, KnownArm::Case(key)))
            }
            Terminator::Jump { .. }
            | Terminator::For { .. }
            | Terminator::While { .. }
            | Terminator::Return { .. }
            | Terminator::Diverge => None,
        })
        .collect()
}

/// Walks the body once, folding every instruction's result where it
/// stands.
fn fold_constants(editor: &mut Editor<'_, SccpScope>, constants: &SparseFacts<ConstantDomain>) {
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        let editor = cursor.editor();
        let record = editor.body().inst(inst.id());
        let (span, result) = (record.span(), record.result());
        match (record.op(), result) {
            (Op::Const { .. }, _) | (_, None) => {}
            (_, Some(value)) => fold(editor, constants, value, Position::Before(inst), span),
        }
    }
}

/// `value`'s uses take a `Const` inserted at `at`, where the analysis knows
/// `value` and something reads it.
fn fold(
    editor: &mut Editor<'_, SccpScope>,
    constants: &SparseFacts<ConstantDomain>,
    value: ValueId,
    at: Position,
    span: Span,
) {
    let body = editor.body();
    if body.is_token(value) || body.use_count(value) == 0 {
        return;
    }
    let Some(literal) = constants.constant(value) else {
        return;
    };
    let (literal, ty) = (literal.clone(), body.ty_of(value));
    let constant = editor.insert(at, Op::Const { value: literal }, ty, span);
    match editor.replace_uses(value, constant) {
        Ok(()) | Err(DoesNotDominate { .. }) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, Overflow};
    use crate::passes::optimize::tests::{Fixture, assert_whole, bool_const, int_const, ret, trapping_add};

    /// `a = 1; b = 2; c = a + b; d = c * c; return d`: `d` is 9.
    #[test]
    fn a_constant_chain_folds() {
        let mut f = Fixture::new();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let b = builder.push(int_const(2), int, Span::ZERO);
        let c = builder.push(trapping_add(a, b), int, Span::ZERO);
        let d = builder.push(Op::BinOp { op: BinOp::Mul(Overflow::Trap), left: c, right: c }, int, Span::ZERO);
        builder.terminate(ret(d), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();

        let changes = f.run::<Sccp>(id);

        let body = f.module.body(id);
        let Terminator::Return { value, .. } = body.block(entry).terminator() else {
            panic!("the entry returns")
        };
        let returned = body.block(entry).insts().find(|inst| inst.result() == Some(*value));
        assert!(
            matches!(returned.map(|inst| inst.op()), Some(Op::Const { value: Literal::Int(9) })),
            "the return reads the constant 9: {}",
            f.snapshot(id)
        );
        assert_eq!(body.use_count(c), 0, "c's computation is left unread, for dce");
        assert_eq!(body.use_count(d), 0, "d's computation is left unread, for dce");
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
        assert_whole(&f.module, id);
    }

    /// ```text
    /// entry:    c = true; diamond c { then } else { else } join
    /// then:     one = 1; jump join(one)
    /// else:     two = 2; jump join(two)
    /// join(p):  return p
    /// ```
    #[test]
    fn a_known_diamond_keeps_one_arm_and_prunes_the_other() {
        let mut f = Fixture::new();
        let (int, bool) = (f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.push(bool_const(true), bool, Span::ZERO);
        let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        let one = builder.push(int_const(1), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: vec![one] }, acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        let two = builder.push(int_const(2), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: vec![two] }, acvus_ast::Span::ZERO);
        builder.switch_to(join);
        let p = builder.block_params(join)[0];
        builder.terminate(ret(p), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let blocks_before = f.module.body(id).blocks().count();

        let changes = f.run::<Sccp>(id);

        let body = f.module.body(id);
        assert_eq!(body.blocks().count(), blocks_before - 1, "the else arm is pruned: {}", f.snapshot(id));
        let Terminator::Jump { to: taken, .. } = body.block(body.entry()).terminator() else {
            panic!("the entry jumps into its known arm: {}", f.snapshot(id))
        };
        assert!(matches!(f.ops(id, *taken)[..], [Op::Const { value: Literal::Int(1) }]), "the then arm is kept");
        let twos = body
            .blocks()
            .flat_map(|(_, block)| block.insts())
            .filter(|inst| matches!(inst.op(), Op::Const { value: Literal::Int(2) }))
            .count();
        assert_eq!(twos, 0, "nothing of the else arm is left");
        assert!(changes.touched(crate::ir::cfg::Aspect::CfgShape));
        assert_whole(&f.module, id);
    }

    /// ```text
    /// entry:    a = 2; switch a { 1 => one, 2 => two, _ => other } join
    /// one:      k = 10; jump join(k)
    /// two:      k = 20; jump join(k)
    /// other:    k = 30; jump join(k)
    /// join(p):  return p
    /// ```
    #[test]
    fn a_known_switch_keeps_its_key_s_arm_and_prunes_the_others() {
        let mut f = Fixture::new();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let a = builder.push(int_const(2), int, Span::ZERO);
        let [one, two, other] = [(); 3].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        builder.terminate(Terminator::Switch {
            tag: a,
            arms: vec![(SwitchKey::Int(1), one, Vec::new()), (SwitchKey::Int(2), two, Vec::new())],
            default: Some((other, Vec::new())),
            join,
        }, acvus_ast::Span::ZERO);
        for (block, n) in [(one, 10), (two, 20), (other, 30)] {
            builder.switch_to(block);
            let k = builder.push(int_const(n), int, Span::ZERO);
            builder.terminate(Terminator::Jump { to: join, args: vec![k] }, acvus_ast::Span::ZERO);
        }
        builder.switch_to(join);
        let p = builder.block_params(join)[0];
        builder.terminate(ret(p), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let blocks_before = f.module.body(id).blocks().count();

        let changes = f.run::<Sccp>(id);

        let body = f.module.body(id);
        assert_eq!(body.blocks().count(), blocks_before - 2, "the other two arms are pruned: {}", f.snapshot(id));
        let Terminator::Jump { to: taken, .. } = body.block(body.entry()).terminator() else {
            panic!("the entry jumps into its key's arm: {}", f.snapshot(id))
        };
        assert!(matches!(f.ops(id, *taken)[..], [Op::Const { value: Literal::Int(20) }]), "the arm of 2 is kept");
        let others = body
            .blocks()
            .flat_map(|(_, block)| block.insts())
            .filter(|inst| matches!(inst.op(), Op::Const { value: Literal::Int(10 | 30) }))
            .count();
        assert_eq!(others, 0, "nothing of the other arms is left");
        assert!(changes.touched(crate::ir::cfg::Aspect::CfgShape));
        assert_whole(&f.module, id);
    }

    /// `a = 100i8; b = a + a; return b`: the addition traps, so it is no
    /// constant and stays.
    #[test]
    fn a_trapping_addition_is_not_folded() {
        let mut f = Fixture::new();
        let i8 = f.i8;
        let mut builder = f.module.build_body(false, i8);
        let a = builder.push(int_const(100), i8, Span::ZERO);
        let b = builder.push(trapping_add(a, a), i8, Span::ZERO);
        builder.terminate(ret(b), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);

        f.run::<Sccp>(id);

        assert_eq!(f.snapshot(id), before);
        assert_whole(&f.module, id);
    }
}
