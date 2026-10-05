//! Reborrow elimination: a `Ref` of an empty path through `r`, of `r`'s own
//! type, so of `r`'s mutability, reaches the whole of what `r` names (types
//! §4: a reference's target is never a reference, so `&*r` names `r`'s
//! target and nothing else). Where no use moves it, its uses take `r`; the
//! `Ref`, left unread, is `dce`'s. A use that moves it, a `&mut` at a
//! by-value position (`ir::cfg::takings`), keeps it: there `r` would be
//! moved, and its holder keeps `r`.
//!
//! Form: F1, SSA, in and out.
//!
//! Cost: O(instructions), and one replacement per such `Ref`.
//!
//! Meaning: `&*r` and `r` reach the same storage, and a `Ref` neither traps
//! nor has an effect, so where every use only reads it, nothing about the
//! run changes. The replacement has the replaced value's type.

use crate::ir::RefTarget;
use crate::ir::cfg::edit::{DoesNotDominate, Editor, Value};
use crate::ir::cfg::takings::{Taken, by_value, inst_takings, terminator_takings};
use crate::ir::cfg::{Body, BodyId, Changes, Module, Op, Site};
use crate::ir::ValueId;
use crate::ty::{Concrete, Types};
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Reborrow;

impl BodyTransform for Reborrow {
    const NAME: &'static str = "reborrow";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Value>::new(module, body, &mut facts);
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            let editor = cursor.editor();
            let body = editor.body();
            let record = body.inst(inst.id());
            let Op::Ref { target: RefTarget::Through(reference), path, .. } = record.op() else {
                continue;
            };
            let reborrow = record.result().expect("a Ref yields a value");
            if !path.is_empty() || body.ty_of(reborrow) != body.ty_of(*reference) || moved(body, editor.types(), reborrow) {
                continue;
            }
            let reference = *reference;
            match editor.replace_uses(reborrow, reference) {
                Ok(()) | Err(DoesNotDominate { .. }) => {}
            }
        }
        editor.finish()
    }
}

/// Whether some use of `value` moves it.
fn moved(body: &Body, types: &Types<Concrete>, value: ValueId) -> bool {
    body.users(value).any(|site| match site {
        Site::Inst(inst) => {
            inst_takings(body, types, body.inst(inst)).iter().any(|operand| operand.value == value && operand.taken == Taken::Moved)
        }
        Site::Term(block) => {
            let terminator = body.block(block).terminator();
            let named = terminator_takings(body, types, terminator).iter().any(|named| named.value == value && named.taken == Taken::Moved);
            let passed = || terminator.edges().iter().flat_map(|edge| edge.args.iter()).any(|arg| *arg == value);
            named || (passed() && by_value(types, body.ty_of(value)) == Taken::Moved)
        }
    })
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::cfg::{Aspect, Terminator};
    use crate::passes::optimize::tests::{Fixture, assert_whole};
    use crate::ty::Mutability;

    fn reborrow_of(reference: crate::ir::ValueId, mutability: Mutability) -> Op {
        Op::Ref { target: RefTarget::Through(reference), path: Vec::new(), mutability }
    }

    fn read_through(reference: crate::ir::ValueId) -> Op {
        Op::Take { target: RefTarget::Through(reference), path: Vec::new(), taken_out: false }
    }

    /// `(r: &i64, m: &mut i64) s = &*r; t = &mut *m; a = *s; b = *t; return a + b`:
    /// both reborrows are their references.
    #[test]
    fn a_reborrow_of_the_whole_is_its_reference() {
        let mut f = Fixture::new();
        let (int, ref_int, mut_ref_int) = (f.int, f.ref_int, f.mut_ref_int);
        let (nr, nm) = (f.name("r"), f.name("m"));
        let mut builder = f.module.build_body(false, int);
        let r = builder.param(nr, ref_int, acvus_ast::Span::ZERO);
        let m = builder.param(nm, mut_ref_int, acvus_ast::Span::ZERO);
        let s = builder.push(reborrow_of(r, Mutability::Shared), ref_int, Span::ZERO);
        let t = builder.push(reborrow_of(m, Mutability::Mut), mut_ref_int, Span::ZERO);
        let a = builder.push(read_through(s), int, Span::ZERO);
        let b = builder.push(read_through(t), int, Span::ZERO);
        let sum = builder.push(crate::passes::optimize::tests::add(a, b), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: sum, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();

        let changes = f.run::<Reborrow>(id);

        let body = f.module.body(id);
        assert_eq!(body.use_count(s), 0);
        assert_eq!(body.use_count(t), 0);
        let ops = f.ops(id, entry);
        assert!(matches!(ops[2], Op::Take { target: RefTarget::Through(of), .. } if *of == r));
        assert!(matches!(ops[3], Op::Take { target: RefTarget::Through(of), .. } if *of == m));
        assert!(changes.touched(Aspect::Instructions));
        assert_whole(&f.module, id);
    }

    /// `(m: &mut i64) t = &mut *m; return t`: the `Return` moves `t`, the
    /// reborrow made for that use, so `t` stays: returning `m` there would
    /// move `m`, which its holder keeps.
    #[test]
    fn a_mut_reborrow_a_use_moves_stays() {
        let mut f = Fixture::new();
        let (mut_ref_int, nm) = (f.mut_ref_int, f.name("m"));
        let mut builder = f.module.build_body(false, mut_ref_int);
        let m = builder.param(nm, mut_ref_int, acvus_ast::Span::ZERO);
        let t = builder.push(reborrow_of(m, Mutability::Mut), mut_ref_int, Span::ZERO);
        builder.terminate(Terminator::Return { value: t, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);

        f.run::<Reborrow>(id);

        assert_eq!(f.snapshot(id), before);
        assert_whole(&f.module, id);
    }

    /// `(m: &mut i64) s = &*m; return *s`: a shared reborrow of a `&mut` is
    /// not `m`, whose type is another.
    #[test]
    fn a_shared_reborrow_of_a_mut_reference_stays() {
        let mut f = Fixture::new();
        let (int, ref_int, mut_ref_int, nm) = (f.int, f.ref_int, f.mut_ref_int, f.name("m"));
        let mut builder = f.module.build_body(false, int);
        let m = builder.param(nm, mut_ref_int, acvus_ast::Span::ZERO);
        let s = builder.push(reborrow_of(m, Mutability::Shared), ref_int, Span::ZERO);
        let a = builder.push(read_through(s), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: a, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);

        f.run::<Reborrow>(id);

        assert_eq!(f.snapshot(id), before);
        assert_whole(&f.module, id);
    }
}
