//! Scalar replacement of aggregates, scope `Storage`. SSA form in (F1), the
//! same form out; the field slots it makes are promoted by the `mem2reg`
//! the pipeline runs after it, never by this transform.
//!
//! Algorithm: every slot is offered to `split_slot` (S1), which splits a slot
//! of a tuple or a structural object into one slot per field wherever every
//! use names a field of it, and refuses an extern-defined type and a slot
//! some use names whole. The field slots a split makes are offered in turn,
//! so a nested structural value splits down to its leaves.
//!
//! Cost: O(the fields of the slots' types, nested ones included) splits,
//! each O(fields × uses of the slot), and `Escapes` computed again where a
//! split before it made it stale.
//!
//! Meaning: the fields of a structural value are independent places (types
//! §1), so a field's slot holds exactly what the field held, and every read
//! and write of the field reads and writes it; loads, stores and their
//! order are unchanged. Identity: an extern-defined type is never split,
//! and no value is rebuilt (compilation §3).

use crate::ir::ValueId;
use crate::ir::cfg::edit::{CannotSplitSlot, Editor, Storage};
use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Sroa;

impl BodyTransform for Sroa {
    const NAME: &'static str = "sroa";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Storage>::new(module, body, &mut facts);
        let mut pending: Vec<ValueId> = editor.body().slots().to_vec();
        while let Some(slot) = pending.pop() {
            match editor.split_slot(slot) {
                Ok(fields) => pending.extend(fields),
                Err(CannotSplitSlot::NotStructural | CannotSplitSlot::Escapes(_)) => {}
            }
        }
        editor.finish()
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::{Literal, Span};
    use acvus_utils::{Interner, QualifiedRef};

    use super::*;
    use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
    use crate::ir::cfg::{Body, Inst, Op, Terminator};
    use crate::ir::{PathSeg, RefTarget};
    use crate::ty::{Concrete, IntTy, Mutability, ObjectTy, Ty, TyKind, TypeArg, Types};

    struct Fixture {
        module: Module,
        interner: Interner,
        int: Ty,
        pair: Ty,
        ref_pair: Ty,
        /// A struct a declaration names, with one `i64` field `a`.
        declared: Ty,
    }

    fn fixture() -> Fixture {
        let interner = Interner::new();
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let pair = types.intern(TyKind::Tuple(Box::new([int, int]), ()));
        let ref_pair = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(pair)));
        let declared = types.intern(TyKind::Object(ObjectTy::Declared(
            QualifiedRef::root(interner.intern("Point")),
            Box::new([(interner.intern("a"), int)]),
        )));
        Fixture { module: Module::new(types), interner, int, pair, ref_pair, declared }
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
            entries: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .expect("empty declarations verify")
    }

    fn constant(n: i128) -> Op {
        Op::Const { value: Literal::Int(n) }
    }

    fn assign_at(slot: ValueId, path: Vec<PathSeg>, value: ValueId) -> Op {
        Op::Assign { target: RefTarget::Var(slot), path, value, restores: false }
    }

    fn take_at(slot: ValueId, path: Vec<PathSeg>) -> Op {
        Op::Take { target: RefTarget::Var(slot), path, taken_out: false }
    }

    fn ret(value: ValueId) -> Terminator {
        Terminator::Return { value, order: None, span: Span::ZERO }
    }

    fn run(f: &mut Fixture, body: BodyId) -> Changes {
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        Sroa::run(&mut f.module, body, &mut am)
    }

    /// The slot and path of every `Take` and `Assign` in `body`, in block
    /// order.
    fn places(body: &Body) -> Vec<(ValueId, Vec<PathSeg>)> {
        body.blocks()
            .flat_map(|(_, block)| block.insts())
            .map(Inst::op)
            .filter_map(|op| match op {
                Op::Take { target: RefTarget::Var(slot), path, .. }
                | Op::Assign { target: RefTarget::Var(slot), path, .. } => Some((*slot, path.clone())),
                _ => None,
            })
            .collect()
    }

    /// `p.0 = 1; p.1 = 2; t = p.0; u = p.1; v = t + u; return v`, with `p`
    /// a slot of `(i64, i64)`: `p` becomes one `i64` slot per field.
    #[test]
    fn a_tuple_slot_splits_into_its_fields() {
        let mut f = fixture();
        let (int, pair) = (f.int, f.pair);
        let mut builder = f.module.build_body(false, int);
        let p = builder.slot(pair, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        let two = builder.push(constant(2), int, Span::ZERO);
        builder.push_unit(assign_at(p, vec![PathSeg::Index(0)], one), Span::ZERO);
        builder.push_unit(assign_at(p, vec![PathSeg::Index(1)], two), Span::ZERO);
        let t = builder.push(take_at(p, vec![PathSeg::Index(0)]), int, Span::ZERO);
        let u = builder.push(take_at(p, vec![PathSeg::Index(1)]), int, Span::ZERO);
        let v = builder.push(
            Op::BinOp { op: crate::ir::BinOp::Add(crate::ir::Overflow::Trap), left: t, right: u },
            int,
            Span::ZERO,
        );
        builder.terminate(ret(v), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = run(&mut f, id);

        let body = f.module.body(id);
        let [first, second] = body.slots()[..] else {
            panic!("the pair's slot became two, and nothing else is a slot")
        };
        assert!(!body.slots().contains(&p));
        assert_eq!(body.ty_of(first), int);
        assert_eq!(body.ty_of(second), int);
        assert_eq!(
            places(body),
            vec![(first, Vec::new()), (second, Vec::new()), (first, Vec::new()), (second, Vec::new())],
            "each access names its field's slot whole"
        );
        assert!(changes.touched(crate::ir::cfg::Aspect::Storage));
    }

    /// `d.a = 1; t = d.a; return t`, with `d` a slot of a declared struct:
    /// an extern-defined type is never split (compilation §3).
    #[test]
    fn an_extern_defined_types_slot_is_refused() {
        let mut f = fixture();
        let (int, declared) = (f.int, f.declared);
        let a = f.interner.intern("a");
        let mut builder = f.module.build_body(false, int);
        let d = builder.slot(declared, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign_at(d, vec![PathSeg::Field(a)], one), Span::ZERO);
        let t = builder.push(take_at(d, vec![PathSeg::Field(a)]), int, Span::ZERO);
        builder.terminate(ret(t), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = run(&mut f, id);

        assert_eq!(f.module.body(id).slots(), &[d]);
        assert!(!changes.touched(crate::ir::cfg::Aspect::Storage));
    }

    /// `p.0 = 1; p.1 = 2; r = &p; t = p.0; return t`: the reference names
    /// the whole of `p`.
    #[test]
    fn a_slot_whose_whole_is_referenced_is_refused() {
        let mut f = fixture();
        let (int, pair, ref_pair) = (f.int, f.pair, f.ref_pair);
        let mut builder = f.module.build_body(false, int);
        let p = builder.slot(pair, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        let two = builder.push(constant(2), int, Span::ZERO);
        builder.push_unit(assign_at(p, vec![PathSeg::Index(0)], one), Span::ZERO);
        builder.push_unit(assign_at(p, vec![PathSeg::Index(1)], two), Span::ZERO);
        builder.push(
            Op::Ref { target: RefTarget::Var(p), path: Vec::new(), mutability: Mutability::Shared },
            ref_pair,
            Span::ZERO,
        );
        let t = builder.push(take_at(p, vec![PathSeg::Index(0)]), int, Span::ZERO);
        builder.terminate(ret(t), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = run(&mut f, id);

        assert_eq!(f.module.body(id).slots(), &[p]);
        assert!(!changes.touched(crate::ir::cfg::Aspect::Storage));
    }
}
