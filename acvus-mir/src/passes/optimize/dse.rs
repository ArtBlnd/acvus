//! Dead store elimination, scope `Discharge`. SSA form in (F1), the same
//! form out.
//!
//! Algorithm: every store, in block order, is offered to `discharge` (G1),
//! which removes it where `EditFacts` proves it `NeverRead`: no read of the
//! storage it writes is reached from it before a write that replaces what it
//! stored, or before the body's end for the body's own storage (`Loans`).
//! A removal drops the facts it made stale, so a store whose only reader
//! was a removed one is judged on the body as it now stands.
//!
//! Cost: O(stores) discharges; each is judged alone, O(storages it writes ·
//! the body's sites and edges), on `Loans` and `IntervalValues`, which a
//! removal before it made stale and the judgement computes again.
//!
//! Meaning: a store no read reaches is unobservable, so removing it keeps
//! the effects, whether the run ends and whether it traps (compilation §1);
//! a store neither traps nor carries an effect. Types are untouched
//! (compilation §3).

use crate::ir::cfg::edit::{CannotDischarge, Discharge, Editor, InstRef};
use crate::ir::cfg::{Body, BodyId, Changes, Module, Op};
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Dse;

impl BodyTransform for Dse {
    const NAME: &'static str = "dse";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Discharge>::new(module, body, &mut facts);
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            let editor = cursor.editor();
            if !is_store(editor.body(), inst) {
                continue;
            }
            match editor.discharge(inst) {
                Ok(()) | Err(CannotDischarge::NotProven | CannotDischarge::WrongFact) => {}
            }
        }
        editor.finish()
    }
}

fn is_store(body: &Body, inst: InstRef) -> bool {
    matches!(body.inst(inst.id()).op(), Op::Assign { .. })
}

#[cfg(test)]
mod tests {
    use acvus_ast::{Literal, Span};
    use acvus_utils::Interner;

    use super::*;
    use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
    use crate::ir::cfg::{Inst, Terminator};
    use crate::ir::{RefTarget, ValueId};
    use crate::ty::{Concrete, IntTy, Mutability, Ty, TyKind, TypeArg, Types};

    struct Fixture {
        module: Module,
        interner: Interner,
        int: Ty,
        ref_int: Ty,
    }

    fn fixture() -> Fixture {
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let ref_int = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
        Fixture { module: Module::new(types), interner: Interner::new(), int, ref_int }
    }

    fn declarations(interner: &Interner) -> Declarations {
        Declarations::verify(RawDeclarations {
            types: Types::new(),
            functions: Vec::new(),
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            inputs: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .expect("empty declarations verify")
    }

    fn constant(n: i128) -> Op {
        Op::Const { value: Literal::Int(n) }
    }

    fn assign(slot: ValueId, value: ValueId) -> Op {
        Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
    }

    fn take(target: RefTarget) -> Op {
        Op::Take { target, path: Vec::new(), taken_out: false }
    }

    fn ret(value: ValueId) -> Terminator {
        Terminator::Return { value, order: None, span: Span::ZERO }
    }

    fn run(f: &mut Fixture, body: BodyId) -> Changes {
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        Dse::run(&mut f.module, body, &mut am)
    }

    /// The stored value of every store left in `body`, in block order.
    fn stored(body: &crate::ir::cfg::Body) -> Vec<ValueId> {
        body.blocks()
            .flat_map(|(_, block)| block.insts())
            .map(Inst::op)
            .filter_map(|op| match op {
                Op::Assign { value, .. } => Some(*value),
                _ => None,
            })
            .collect()
    }

    /// `one = 1; two = 2; s = one; s = two; t = s; return t`: the first
    /// store is overwritten before any read.
    #[test]
    fn a_store_overwritten_before_any_read_goes() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let s = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        let two = builder.push(constant(2), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push_unit(assign(s, two), Span::ZERO);
        let t = builder.push(take(RefTarget::Var(s)), int, Span::ZERO);
        builder.terminate(ret(t), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = run(&mut f, id);

        assert_eq!(stored(f.module.body(id)), vec![two]);
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
    }

    /// `one = 1; s = one; t = s; two = 2; s = two; return t`: the first
    /// store is read; the second is left at the end of the body, whose own
    /// slot nothing reads after it.
    #[test]
    fn a_store_read_later_stays() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let s = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        let t = builder.push(take(RefTarget::Var(s)), int, Span::ZERO);
        let two = builder.push(constant(2), int, Span::ZERO);
        builder.push_unit(assign(s, two), Span::ZERO);
        builder.terminate(ret(t), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        run(&mut f, id);

        assert_eq!(stored(f.module.body(id)), vec![one]);
    }

    /// `one = 1; s = one; r = &s; t = *r; return t`: the store is read
    /// through the reference.
    #[test]
    fn a_store_into_a_slot_read_through_a_reference_stays() {
        let mut f = fixture();
        let (int, ref_int) = (f.int, f.ref_int);
        let mut builder = f.module.build_body(false, int);
        let s = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        let r = builder.push(
            Op::Ref { target: RefTarget::Var(s), path: Vec::new(), mutability: Mutability::Shared },
            ref_int,
            Span::ZERO,
        );
        let t = builder.push(take(RefTarget::Through(r)), int, Span::ZERO);
        builder.terminate(ret(t), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = run(&mut f, id);

        assert_eq!(stored(f.module.body(id)), vec![one]);
        assert!(!changes.touched(crate::ir::cfg::Aspect::Instructions), "nothing was removed");
    }
}
