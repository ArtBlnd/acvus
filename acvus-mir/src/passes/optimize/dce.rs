//! Dead code elimination over a worklist: an instruction whose result
//! nothing uses is offered to `Editor<Remove>::remove`, and a removal that
//! is made may leave its operands unused, so their definitions are offered
//! in turn. The editor refuses what traps, writes storage, touches a context
//! or calls a callee whose reissue is not pure (D54); those stay.
//!
//! Form: any, kept.
//!
//! Cost: O(instructions + operands) offers, each an O(1) check, and each
//! removal the core's cost of taking one instruction out.
//!
//! Meaning: only an instruction that neither traps, writes, nor has an
//! effect goes, and its value was read by nothing, so the order of effects,
//! whether the run ends and whether it traps are unchanged. No type changes.

use rustc_hash::FxHashSet;

use crate::ir::ValueId;
use crate::ir::cfg::edit::{Editor, NoFacts, Remove, Unremovable};
use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Dce;

impl BodyTransform for Dce {
    const NAME: &'static str = "dce";

    fn run(module: &mut Module, body: BodyId, _am: &mut AnalysisManager<'_>) -> Changes {
        let mut no_facts = NoFacts;
        let mut editor = Editor::<Remove>::new(module, body, &mut no_facts);
        let mut pending: Vec<ValueId> = editor
            .body()
            .blocks()
            .flat_map(|(_, block)| block.insts())
            .filter_map(|inst| inst.result())
            .collect();
        let mut removed: FxHashSet<ValueId> = FxHashSet::default();
        while let Some(value) = pending.pop() {
            if removed.contains(&value) || editor.body().use_count(value) > 0 {
                continue;
            }
            let Some(inst) = editor.defining_inst(value) else {
                continue;
            };
            let defined = editor.body().inst(inst.id());
            if defined.result() != Some(value) {
                continue;
            }
            let operands = defined.operands();
            match editor.remove(inst) {
                Ok(()) => {
                    removed.insert(value);
                    pending.extend(operands);
                }
                Err(Unremovable::StillUsed | Unremovable::Traps | Unremovable::Writes | Unremovable::Effect) => {}
            }
        }
        editor.finish()
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::cfg::Op;
    use crate::passes::optimize::tests::{Fixture, assert_whole, assign, int_const, ret, trapping_add, wrapping_add};

    /// ```text
    /// a = 1; b = a + a (wrapping); c = b + a (wrapping)   dead chain
    /// d = a + a (trapping)                                 unused, traps
    /// x = a                                                a store
    /// e = f()                                              unused, opaque
    /// return a
    /// ```
    #[test]
    fn a_dead_chain_goes_and_a_trap_a_store_and_an_effect_stay() {
        let mut f = Fixture::new();
        let (int, opaque_call) = (f.int, f.call(f.opaque));
        let mut builder = f.module.build_body(true, int);
        let x = builder.slot(int, Span::ZERO);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let b = builder.push(wrapping_add(a, a), int, Span::ZERO);
        builder.push(wrapping_add(b, a), int, Span::ZERO);
        builder.push(trapping_add(a, a), int, Span::ZERO);
        builder.push_unit(assign(x, a), Span::ZERO);
        builder.push_effect(opaque_call, int, Span::ZERO);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();

        let changes = f.run::<Dce>(id);

        let ops = f.ops(id, entry);
        assert_eq!(ops.len(), 4, "{ops:?}");
        assert!(matches!(ops[0], Op::Const { .. }), "the constant is returned");
        assert!(matches!(ops[1], Op::BinOp { left, right, .. } if *left == a && *right == a), "the trapping add stays");
        assert!(matches!(ops[2], Op::Assign { .. }), "the store stays");
        assert!(matches!(ops[3], Op::FunctionCall { .. }), "the opaque call stays");
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
        assert_whole(&f.module, id);
    }

    /// A pure call whose result nothing reads goes.
    #[test]
    fn an_unused_pure_call_goes() {
        let mut f = Fixture::new();
        let (int, pure_call) = (f.int, f.call(f.pure));
        let mut builder = f.module.build_body(false, int);
        builder.push(pure_call, int, Span::ZERO);
        let a = builder.push(int_const(1), int, Span::ZERO);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();

        f.run::<Dce>(id);

        assert!(matches!(f.ops(id, entry)[..], [Op::Const { .. }]));
        assert_whole(&f.module, id);
    }
}
