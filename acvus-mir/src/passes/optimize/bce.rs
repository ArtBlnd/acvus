//! Bounds-check elimination, scope `Discharge`. SSA form in (F1), the same
//! form out.
//!
//! Algorithm: every check, in block order, is offered to `discharge` (G1):
//! the bound of a checked `Index` or `IndexSet`, a `Check` and a
//! `CheckSteps`. `EditFacts` proves one `NeverFires` from `ValueBounds`, the
//! reduced product of `IntervalValues` and `AffineValues`, whose calls read
//! the declared Relations of their impls (extern-fn §3): a bound where the
//! index is never negative and lies below the slice's length, by their
//! intervals or by the length less the index as a combination, and a check
//! where the exact result of its operation fits the operands' width. So
//! `i < len(s)` holds for a `For` counter below `len(s)` whatever the
//! length is (D66, D83). A checked bound becomes proven and stays where it
//! is; a check goes.
//!
//! Cost: O(checks) discharges; each is judged alone, O(the symbols of the
//! index's combination · the relations of a slice's impl), on
//! `IntervalValues` and `AffineValues`, O(uses × widened height), and
//! `Loans`, which a discharge before it made stale and the judgement
//! computes again.
//!
//! Meaning: a check that never fires on any run that reaches it does not
//! change whether a run traps (compilation §1), and it has no other
//! effect. Types are untouched (compilation §3).

use crate::ir::IndexBound;
use crate::ir::cfg::edit::{CannotDischarge, Discharge, Editor, InstRef};
use crate::ir::cfg::{Body, BodyId, Changes, Module, Op};
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Bce;

impl BodyTransform for Bce {
    const NAME: &'static str = "bce";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Discharge>::new(module, body, &mut facts);
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            let editor = cursor.editor();
            if !is_check(editor.body(), inst) {
                continue;
            }
            match editor.discharge(inst) {
                Ok(()) | Err(CannotDischarge::NotProven | CannotDischarge::WrongFact) => {}
            }
        }
        editor.finish()
    }
}

fn is_check(body: &Body, inst: InstRef) -> bool {
    matches!(
        body.inst(inst.id()).op(),
        Op::Check { .. }
            | Op::CheckSteps { .. }
            | Op::Index { bound: IndexBound::Checked, .. }
            | Op::IndexSet { bound: IndexBound::Checked, .. }
    )
}

#[cfg(test)]
mod tests {
    use acvus_ast::{Literal, Span};
    use acvus_utils::{Interner, QualifiedRef};

    use super::*;
    use crate::decl::declared::{Atom, Comparison, Relation, Subject, Term};
    use crate::decl::{Core, DeclaredFacts, GenericImplSig, Impls, LoopMarkers, RawDeclarations, Scheme};
    use crate::decl::Declarations;
    use crate::ir::cfg::{BlockId, Inst, Stages, Terminator};
    use crate::ir::{Callee, Checked, ExitTrip, ExternImpl, ForSource, IndexMode, RefTarget, ValueId};
    use crate::ty::{
        Calls, Concrete, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, LenTerm, Mutability, Poly, Ty, TyKind, TypeArg,
        Types,
    };

    /// `make(n: u64) -> &[i64]`, declared with `len(ret) = n`, and
    /// `as_slice`, declared with no impl facts.
    struct Fixture {
        module: Module,
        decls: Declarations,
        make: QualifiedRef,
        as_slice: QualifiedRef,
        int: Ty,
        index: Ty,
        slice_ref: Ty,
        make_ty: Ty,
        /// `Array<i64, 4>`.
        array: Ty,
        array_ref: Ty,
    }

    fn fixture() -> Fixture {
        let interner = Interner::new();
        let make = QualifiedRef::root(interner.intern("make"));
        let as_slice = QualifiedRef::root(interner.intern("as_slice"));

        let mut poly = Types::<Poly>::new();
        let poly_int = poly.intern(TyKind::Int(IntTy::I64));
        let poly_index = poly.intern(TyKind::Int(IntTy::U64));
        let poly_slice = poly.intern(TyKind::Slice(poly_int));
        let poly_ref = poly.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(poly_slice)));
        let declared = poly.intern(TyKind::Fn {
            params: Box::new([poly_index]),
            ret: poly_ref,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });
        let relations = vec![Relation {
            left: Term::Atom(Atom::Len(Subject::Result)),
            comparison: Comparison::Eq,
            right: Term::Atom(Atom::Arg(0)),
        }];
        let scheme = Scheme {
            ty: declared,
            param_names: Box::new([interner.intern("n")]),
            bounds: Box::new([]),
            impls: Some(Impls {
                concrete: Box::new([]),
                generic: Some(GenericImplSig { facts: DeclaredFacts { relations: relations.into(), ..DeclaredFacts::default() } }),
            }),
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
        };
        let as_slice_scheme = Scheme {
            ty: declared,
            param_names: Box::new([interner.intern("n")]),
            bounds: Box::new([]),
            impls: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
        };
        let decls = Declarations::verify(RawDeclarations {
            types: poly,
            functions: vec![(make, scheme), (as_slice, as_slice_scheme)],
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            inputs: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(&interner),
        })
        .expect("the declarations verify");

        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let index = types.intern(TyKind::Int(IntTy::U64));
        let slice = types.intern(TyKind::Slice(int));
        let slice_ref = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(slice)));
        let make_ty = types.intern(TyKind::Fn {
            params: Box::new([index]),
            ret: slice_ref,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });
        let array = types.intern(TyKind::Array(int, LenTerm::Known(4)));
        let array_ref = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(array)));
        Fixture { module: Module::new(types), decls, make, as_slice, int, index, slice_ref, make_ty, array, array_ref }
    }

    fn constant(n: i128) -> Op {
        Op::Const { value: Literal::Int(n) }
    }

    fn ret(value: ValueId) -> Terminator {
        Terminator::Return { value, order: None, span: Span::ZERO }
    }

    fn run(f: &mut Fixture, body: BodyId) -> Changes {
        let mut am = AnalysisManager::new(&f.decls);
        Bce::run(&mut f.module, body, &mut am)
    }

    fn ops(body: &Body) -> Vec<&Op> {
        body.blocks().flat_map(|(_, block)| block.insts()).map(Inst::op).collect()
    }

    /// The bound of the one `Index` of `body`.
    fn index_bound(body: &Body) -> IndexBound {
        let bounds: Vec<IndexBound> = ops(body)
            .into_iter()
            .filter_map(|op| match op {
                Op::Index { bound, .. } => Some(*bound),
                _ => None,
            })
            .collect();
        let [bound] = bounds[..] else {
            panic!("the body holds one Index, not {}", bounds.len())
        };
        bound
    }

    /// Where the indexed slice comes from.
    enum Made {
        /// `n = 10; s = make(n)`, with `len(s) = n` declared.
        ByADeclaredCall,
        /// `a` a slot of `Array<i64, 4>`; `r = &a; s = as_slice(r)`.
        FromAnArray,
    }

    /// The loop's end over a slice `s` whose length no fact bounds.
    enum SymbolicEnd {
        /// `for i in 0..len(s)`, `s` a parameter.
        Len,
        /// `for i in 0..len(s) + 1`, `s` a parameter.
        LenPlusOne,
        /// `for i in 0..n`, `n` a parameter and `s = make(n)`, with
        /// `len(s) = n` declared.
        DeclaredLen,
    }

    /// ```text
    /// entry(..):  s = <slice>; at = 0; hi = <end>; jump header
    /// header:     for i in at..hi { body } else exit
    /// body(i):    e = s[i]; jump header
    /// exit:       z = 0; return z
    /// ```
    /// with `at`, `hi` and `i` of `u64`.
    fn range_loop_to_a_symbolic_end(f: &mut Fixture, end: SymbolicEnd) -> BodyId {
        let (int, index, slice_ref, make, make_ty) = (f.int, f.index, f.slice_ref, f.make, f.make_ty);
        let interner = Interner::new();
        let mut builder = f.module.build_body(false, int);
        let (s, hi) = match end {
            SymbolicEnd::Len => {
                let s = builder.param(interner.intern("s"), slice_ref, Span::ZERO);
                (s, builder.push(Op::SliceLen { src: s }, index, Span::ZERO))
            }
            SymbolicEnd::LenPlusOne => {
                let s = builder.param(interner.intern("s"), slice_ref, Span::ZERO);
                let len = builder.push(Op::SliceLen { src: s }, index, Span::ZERO);
                let one = builder.push(constant(1), index, Span::ZERO);
                let add = Op::BinOp { op: crate::ir::BinOp::Add(crate::ir::Overflow::Trap), left: len, right: one };
                (s, builder.push(add, index, Span::ZERO))
            }
            SymbolicEnd::DeclaredLen => {
                let n = builder.param(interner.intern("n"), index, Span::ZERO);
                let call = Op::FunctionCall {
                    callee: Callee::Extern { id: make, r#impl: 0, required: Vec::new() },
                    callee_ty: make_ty,
                    args: vec![n],
                };
                (builder.push(call, slice_ref, Span::ZERO), n)
            }
        };
        let at = builder.push(constant(0), index, Span::ZERO);
        let header = builder.new_block(&[]);
        let loop_body = builder.new_block(&[index]);
        let exit = builder.new_block(&[]);
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() }, Span::ZERO);
        builder.switch_to(header);
        builder.terminate(
            Terminator::For {
                source: ForSource::Range { at, hi },
                stages: Stages::lowered(loop_body),
                exit,
                exit_trip: ExitTrip::Absent,
                exit_args: Vec::new(),
            },
            Span::ZERO,
        );
        builder.switch_to(loop_body);
        let i = builder.block_params(loop_body)[0];
        builder.push(Op::Index { slice: s, index: i, mode: IndexMode::Copy, bound: IndexBound::Checked }, int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() }, Span::ZERO);
        builder.switch_to(exit);
        let z = builder.push(constant(0), int, Span::ZERO);
        builder.terminate(ret(z), Span::ZERO);
        builder.seal().expect("the body has no open state")
    }

    /// D66's limit, lifted: `i` is `len(s) - 1` less the steps left, so
    /// `len(s) - i - 1` is the steps left, never negative, whatever the
    /// length is.
    #[test]
    fn a_range_loop_to_a_symbolic_length_loses_its_check() {
        let mut f = fixture();
        let id = range_loop_to_a_symbolic_end(&mut f, SymbolicEnd::Len);

        let changes = run(&mut f, id);

        assert_eq!(index_bound(f.module.body(id)), IndexBound::Proven, "i < len(s) for i in 0..len(s)");
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
    }

    #[test]
    fn a_range_loop_one_past_a_symbolic_length_keeps_its_check() {
        let mut f = fixture();
        let id = range_loop_to_a_symbolic_end(&mut f, SymbolicEnd::LenPlusOne);

        run(&mut f, id);

        assert_eq!(index_bound(f.module.body(id)), IndexBound::Checked, "i may be len(s)");
    }

    #[test]
    fn a_range_loop_to_a_declared_symbolic_length_loses_its_check() {
        let mut f = fixture();
        let id = range_loop_to_a_symbolic_end(&mut f, SymbolicEnd::DeclaredLen);

        run(&mut f, id);

        assert_eq!(index_bound(f.module.body(id)), IndexBound::Proven, "len(make(n)) = n and i in 0..n");
    }

    /// ```text
    /// entry:     s = <made>; at = 0; hi = <hi>; jump header
    /// header:    for i in at..hi { body } else exit
    /// body(i):   e = s[i]; jump header
    /// exit:      z = 0; return z
    /// ```
    /// with `at`, `hi` and `i` of `u64`.
    fn range_loop_indexing(f: &mut Fixture, made: Made, hi: i128) -> BodyId {
        let (int, index, slice_ref, make, make_ty) = (f.int, f.index, f.slice_ref, f.make, f.make_ty);
        let (array, array_ref, as_slice) = (f.array, f.array_ref, f.as_slice);
        let mut builder = f.module.build_body(false, int);
        let s = match made {
            Made::ByADeclaredCall => {
                let n = builder.push(constant(10), index, Span::ZERO);
                let call = Op::FunctionCall {
                    callee: Callee::Extern { id: make, r#impl: 0, required: Vec::new() },
                    callee_ty: make_ty,
                    args: vec![n],
                };
                builder.push(call, slice_ref, Span::ZERO)
            }
            Made::FromAnArray => {
                let a = builder.slot(array, Span::ZERO);
                let r = builder.push(
                    Op::Ref { target: RefTarget::Var(a), path: Vec::new(), mutability: Mutability::Shared },
                    array_ref,
                    Span::ZERO,
                );
                let view = Op::AsSlice {
                    container: r,
                    mutability: Mutability::Shared,
                    r#impl: ExternImpl { id: as_slice, r#impl: 0 },
                };
                builder.push(view, slice_ref, Span::ZERO)
            }
        };
        let at = builder.push(constant(0), index, Span::ZERO);
        let hi = builder.push(constant(hi), index, Span::ZERO);
        let header = builder.new_block(&[]);
        let loop_body = builder.new_block(&[index]);
        let exit = builder.new_block(&[]);
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(header);
        builder.terminate(Terminator::For {
            source: ForSource::Range { at, hi },
            stages: Stages::lowered(loop_body),
            exit,
            exit_trip: ExitTrip::Absent,
            exit_args: Vec::new(),
        }, acvus_ast::Span::ZERO);
        builder.switch_to(loop_body);
        let i = builder.block_params(loop_body)[0];
        builder.push(Op::Index { slice: s, index: i, mode: IndexMode::Copy, bound: IndexBound::Checked }, int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(exit);
        let z = builder.push(constant(0), int, Span::ZERO);
        builder.terminate(ret(z), acvus_ast::Span::ZERO);
        builder.seal().expect("the body has no open state")
    }

    #[test]
    fn a_range_loop_indexing_below_a_declared_length_loses_its_check() {
        let mut f = fixture();
        let id = range_loop_indexing(&mut f, Made::ByADeclaredCall, 10);

        let changes = run(&mut f, id);

        assert_eq!(index_bound(f.module.body(id)), IndexBound::Proven, "i in [0, 9] and len(s) = 10");
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
    }

    #[test]
    fn an_index_that_may_exceed_the_length_keeps_its_check() {
        let mut f = fixture();
        let id = range_loop_indexing(&mut f, Made::ByADeclaredCall, 11);

        let changes = run(&mut f, id);

        assert_eq!(index_bound(f.module.body(id)), IndexBound::Checked, "i may be 10, and len(s) = 10");
        assert!(!changes.touched(crate::ir::cfg::Aspect::Instructions));
    }

    #[test]
    fn an_index_below_an_arrays_length_loses_its_check() {
        let mut f = fixture();
        let id = range_loop_indexing(&mut f, Made::FromAnArray, 4);

        run(&mut f, id);

        assert_eq!(index_bound(f.module.body(id)), IndexBound::Proven, "i in [0, 3] and the array holds 4");
    }

    #[test]
    fn an_index_past_an_arrays_length_keeps_its_check() {
        let mut f = fixture();
        let id = range_loop_indexing(&mut f, Made::FromAnArray, 5);

        run(&mut f, id);

        assert_eq!(index_bound(f.module.body(id)), IndexBound::Checked, "i may be 4, and the array holds 4");
    }

    /// `one = 1; two = 2; check one + two; check x + one; return x`, with
    /// `x` a parameter: the first check never fires, the second may.
    #[test]
    fn a_check_proven_never_to_fire_goes() {
        let mut f = fixture();
        let int = f.int;
        let name = Interner::new().intern("x");
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(name, int, acvus_ast::Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        let two = builder.push(constant(2), int, Span::ZERO);
        builder.push_unit(Op::Check { op: Checked::Add, left: one, right: two }, Span::ZERO);
        builder.push_unit(Op::Check { op: Checked::Add, left: x, right: one }, Span::ZERO);
        builder.terminate(ret(x), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        run(&mut f, id);

        let checked: Vec<ValueId> = ops(f.module.body(id))
            .into_iter()
            .filter_map(|op| match op {
                Op::Check { left, .. } => Some(*left),
                _ => None,
            })
            .collect();
        assert_eq!(checked, vec![x], "only the check of x + 1 stays");
    }

    /// A discharged bound changes only the bound: every instruction and
    /// block stays.
    #[test]
    fn a_proven_bound_keeps_its_place() {
        let mut f = fixture();
        let id = range_loop_indexing(&mut f, Made::ByADeclaredCall, 10);
        let blocks_before: Vec<BlockId> = f.module.body(id).blocks().map(|(block, _)| block).collect();

        run(&mut f, id);

        let body = f.module.body(id);
        assert_eq!(body.blocks().map(|(block, _)| block).collect::<Vec<_>>(), blocks_before);
        assert_eq!(ops(body).len(), 6, "n, s, at, hi, the index and z all stay");
    }
}
