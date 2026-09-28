//! Reborrow elimination: a `Ref` of an empty path through `r`, of `r`'s own
//! type, so of `r`'s mutability, is a reference to the whole of what `r`
//! names, which is `r` (types §4: a reference's target is never a
//! reference, so `&*r` names `r`'s target and nothing else). Its uses take
//! `r`; the `Ref`, left unread, is `dce`'s.
//!
//! Form: F1, SSA, in and out.
//!
//! Cost: O(instructions), and one replacement per such `Ref`.
//!
//! Meaning: `&*r` and `r` are the same loan of the same storage (ownership
//! §5), and a `Ref` neither traps nor has an effect, so nothing about the run
//! changes. The replacement has the replaced value's type.

use crate::ir::RefTarget;
use crate::ir::cfg::edit::{DoesNotDominate, Editor, Value};
use crate::ir::cfg::{BodyId, Changes, Module, Op};
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Reborrow;

impl BodyTransform for Reborrow {
    const NAME: &'static str = "reborrow";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let mut facts = EditFacts::new(am, module, body);
        let mut editor = Editor::<Value>::new(module, body, &mut facts);
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            let editor = cursor.editor();
            let body = editor.body();
            let record = &body.block(inst.block()).insts()[inst.index()];
            let Op::Ref { target: RefTarget::Through(reference), path, .. } = record.op() else {
                continue;
            };
            let reborrow = record.result().expect("a Ref yields a value");
            if !path.is_empty() || body.ty_of(reborrow) != body.ty_of(*reference) {
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
        let r = builder.param(nr, ref_int);
        let m = builder.param(nm, mut_ref_int);
        let s = builder.push(reborrow_of(r, Mutability::Shared), ref_int, Span::ZERO);
        let t = builder.push(reborrow_of(m, Mutability::Mut), mut_ref_int, Span::ZERO);
        let a = builder.push(read_through(s), int, Span::ZERO);
        let b = builder.push(read_through(t), int, Span::ZERO);
        let sum = builder.push(crate::passes::optimize::tests::wrapping_add(a, b), int, Span::ZERO);
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

    /// `(m: &mut i64) s = &*m; return *s`: a shared reborrow of a `&mut` is
    /// not `m`, whose type is another.
    #[test]
    fn a_shared_reborrow_of_a_mut_reference_stays() {
        let mut f = Fixture::new();
        let (int, ref_int, mut_ref_int, nm) = (f.int, f.ref_int, f.mut_ref_int, f.name("m"));
        let mut builder = f.module.build_body(false, int);
        let m = builder.param(nm, mut_ref_int);
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
