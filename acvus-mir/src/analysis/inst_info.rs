//! Per-instruction def/use extraction.
//!
//! `defs(inst)` — ValueIds defined (written) by this instruction.
//! `uses(inst)` — ValueIds used (read) by this instruction.
//!
//! These are the building blocks for use-def analysis, DCE, reordering, etc.

use smallvec::{SmallVec, smallvec};

use crate::ir::{Callee, InstKind, ValueId};

/// ValueIds defined by this instruction.
pub fn defs(kind: &InstKind) -> SmallVec<[ValueId; 2]> {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::Ref { dst, .. }
        | InstKind::Load { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::FieldSet { dst, .. }
        | InstKind::LoadFunction { dst, .. }
        | InstKind::MakeDeque { dst, .. }
        | InstKind::MakeObject { dst, .. }
        | InstKind::MakeRange { dst, .. }
        | InstKind::MakeTuple { dst, .. }
        | InstKind::TupleIndex { dst, .. }
        | InstKind::TestLiteral { dst, .. }
        | InstKind::TestListLen { dst, .. }
        | InstKind::TestObjectKey { dst, .. }
        | InstKind::TestRange { dst, .. }
        | InstKind::ListIndex { dst, .. }
        | InstKind::ListGet { dst, .. }
        | InstKind::ListSlice { dst, .. }
        | InstKind::ObjectGet { dst, .. }
        | InstKind::MakeClosure { dst, .. }
        | InstKind::MakeVariant { dst, .. }
        | InstKind::TestVariant { dst, .. }
        | InstKind::UnwrapVariant { dst, .. }
        | InstKind::Cast { dst, .. }
        | InstKind::Clone { dst, .. }
        | InstKind::Spawn { dst, .. }
        | InstKind::Poison { dst }
        | InstKind::Undef { dst } => smallvec![*dst],

        // FunctionCall defines dst + context_defs (new SSA values for contexts after call).
        InstKind::FunctionCall {
            dst, context_defs, ..
        } => {
            let mut d: SmallVec<[ValueId; 2]> = smallvec![*dst];
            d.extend(context_defs.iter().map(|(_, v)| *v));
            d
        }

        // Eval defines dst + context_defs.
        InstKind::Eval {
            dst, context_defs, ..
        } => {
            let mut d: SmallVec<[ValueId; 2]> = smallvec![*dst];
            d.extend(context_defs.iter().map(|(_, v)| *v));
            d
        }

        InstKind::ListStep { dst, index_dst, .. } => smallvec![*dst, *index_dst],

        InstKind::BlockLabel { params, .. } => params.iter().copied().collect(),

        InstKind::Store { .. }
        | InstKind::Drop { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Return(_)
        | InstKind::Nop => smallvec![],
    }
}

/// ValueIds used (read) by this instruction.
pub fn uses(kind: &InstKind) -> SmallVec<[ValueId; 4]> {
    match kind {
        // No uses
        InstKind::Const { .. }
        | InstKind::Ref { .. }
        | InstKind::LoadFunction { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Nop
        | InstKind::Poison { .. }
        | InstKind::Undef { .. } => smallvec![],

        // Single use
        InstKind::Load { src, .. } => smallvec![*src],
        InstKind::Store { dst, value, .. } => smallvec![*dst, *value],
        InstKind::UnaryOp { operand, .. } => smallvec![*operand],
        InstKind::FieldGet { object, .. } => smallvec![*object],
        InstKind::FieldSet { object, value, .. } => smallvec![*object, *value],
        InstKind::Cast { src, .. }
        | InstKind::Clone { src, .. }
        | InstKind::Drop { src } => smallvec![*src],
        InstKind::Return(val) => smallvec![*val],
        InstKind::TestLiteral { src, .. } => smallvec![*src],
        InstKind::TestVariant { src, .. } => smallvec![*src],
        InstKind::UnwrapVariant { src, .. } => smallvec![*src],
        InstKind::ObjectGet { object, .. } => smallvec![*object],

        // Two uses
        InstKind::BinOp { left, right, .. } => smallvec![*left, *right],
        InstKind::TestObjectKey { src, .. } => smallvec![*src],
        InstKind::ListGet { list, index, .. } => smallvec![*list, *index],

        // TestListLen / TestRange
        InstKind::TestListLen { src, .. } => smallvec![*src],
        InstKind::TestRange { src, .. } => smallvec![*src],
        InstKind::ListIndex { list, .. } => smallvec![*list],
        InstKind::ListSlice { list, .. } => smallvec![*list],

        // Composite constructors
        InstKind::MakeDeque { elements, .. } => elements.iter().copied().collect(),
        InstKind::MakeObject { fields, .. } => fields.iter().map(|(_, v)| *v).collect(),
        InstKind::MakeRange { start, end, .. } => smallvec![*start, *end],
        InstKind::MakeTuple { elements, .. } => elements.iter().copied().collect(),
        InstKind::TupleIndex { tuple, .. } => smallvec![*tuple],

        // Variant
        InstKind::MakeVariant { payload, .. } => match payload {
            Some(v) => smallvec![*v],
            None => smallvec![],
        },

        // Closure
        InstKind::MakeClosure { captures, .. } => captures.iter().copied().collect(),

        // Function calls — context_defs are DEFS, not uses.
        InstKind::FunctionCall {
            callee,
            args,
            context_uses,
            ..
        } => {
            let mut v: SmallVec<[ValueId; 4]> = SmallVec::new();
            if let Callee::Indirect(f) = callee {
                v.push(*f);
            }
            v.extend(args.iter().copied());
            v.extend(context_uses.iter().map(|(_, val)| *val));
            v
        }
        InstKind::Spawn {
            callee,
            args,
            context_uses,
            ..
        } => {
            let mut v: SmallVec<[ValueId; 4]> = SmallVec::new();
            if let Callee::Indirect(f) = callee {
                v.push(*f);
            }
            v.extend(args.iter().copied());
            v.extend(context_uses.iter().map(|(_, val)| *val));
            v
        }
        // Eval — context_defs are DEFS, not uses.
        InstKind::Eval { src, .. } => smallvec![*src],

        // Iterator
        InstKind::ListStep {
            list,
            index_src,
            done_args,
            ..
        } => {
            let mut v: SmallVec<[ValueId; 4]> = smallvec![*list, *index_src];
            v.extend(done_args.iter().copied());
            v
        }

        // Control flow
        InstKind::Jump { args, .. } => args.iter().copied().collect(),
        InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v: SmallVec<[ValueId; 4]> = smallvec![*cond];
            v.extend(then_args.iter().copied());
            v.extend(else_args.iter().copied());
            v
        }
    }
}

/// Is this instruction a control flow boundary (block label, jump, branch, return)?
pub fn is_control_flow(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::BlockLabel { .. }
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::ListStep { .. }
            | InstKind::Return(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::LocalIdOps;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    #[test]
    fn const_defs_dst_uses_nothing() {
        let inst = InstKind::Const {
            dst: v(0),
            value: acvus_ast::Literal::Int(42),
        };
        assert_eq!(defs(&inst).as_slice(), &[v(0)]);
        assert!(uses(&inst).is_empty());
    }

    #[test]
    fn binop_defs_dst_uses_operands() {
        let inst = InstKind::BinOp {
            dst: v(2),
            op: acvus_ast::BinOp::Add,
            left: v(0),
            right: v(1),
        };
        assert_eq!(defs(&inst).as_slice(), &[v(2)]);
        assert_eq!(uses(&inst).as_slice(), &[v(0), v(1)]);
    }

    #[test]
    fn function_call_uses_include_callee_and_args() {
        let i = acvus_utils::Interner::new();
        let qref = acvus_utils::QualifiedRef::root(i.intern("f"));
        let inst = InstKind::FunctionCall {
            dst: v(3),
            callee: crate::ir::Callee::Direct(qref),
            args: vec![v(0), v(1)],
            context_uses: vec![(qref, v(2))],
            context_defs: vec![],
        };
        assert_eq!(defs(&inst).as_slice(), &[v(3)]);
        let u = uses(&inst);
        assert!(u.contains(&v(0)));
        assert!(u.contains(&v(1)));
        assert!(u.contains(&v(2)));
    }

    #[test]
    fn spawn_eval_defs_uses() {
        let i = acvus_utils::Interner::new();
        let qref = acvus_utils::QualifiedRef::root(i.intern("f"));

        let spawn = InstKind::Spawn {
            dst: v(1),
            callee: crate::ir::Callee::Direct(qref),
            args: vec![v(0)],
            context_uses: vec![],
        };
        assert_eq!(defs(&spawn).as_slice(), &[v(1)]);
        assert_eq!(uses(&spawn).as_slice(), &[v(0)]);

        let eval = InstKind::Eval {
            dst: v(2),
            src: v(1),
            context_defs: vec![],
        };
        assert_eq!(defs(&eval).as_slice(), &[v(2)]);
        assert_eq!(uses(&eval).as_slice(), &[v(1)]);
    }

    #[test]
    fn return_uses_value() {
        let inst = InstKind::Return(v(5));
        assert!(defs(&inst).is_empty());
        assert_eq!(uses(&inst).as_slice(), &[v(5)]);
    }

    #[test]
    fn list_step_multiple_defs() {
        let inst = InstKind::ListStep {
            dst: v(0),
            list: v(1),
            index_src: v(2),
            index_dst: v(3),
            done: crate::ir::Label(0),
            done_args: vec![v(4)],
        };
        let d = defs(&inst);
        assert!(d.contains(&v(0)));
        assert!(d.contains(&v(3)));
        let u = uses(&inst);
        assert!(u.contains(&v(1)));
        assert!(u.contains(&v(2)));
        assert!(u.contains(&v(4)));
    }

    #[test]
    fn control_flow_detection() {
        assert!(is_control_flow(&InstKind::Return(v(0))));
        assert!(is_control_flow(&InstKind::Jump {
            label: crate::ir::Label(0),
            args: vec![],
        }));
        assert!(!is_control_flow(&InstKind::Const {
            dst: v(0),
            value: acvus_ast::Literal::Int(1),
        }));
    }

    /// Regression: ListStep is a CFG terminator and must be classified as control flow.
    #[test]
    fn list_step_is_control_flow() {
        let inst = InstKind::ListStep {
            dst: v(0),
            list: v(1),
            index_src: v(2),
            index_dst: v(3),
            done: crate::ir::Label(0),
            done_args: vec![],
        };
        assert!(is_control_flow(&inst), "ListStep must be control flow");
    }

    /// Regression: FunctionCall context_defs are defs (caller's new SSA values),
    /// not uses. They must appear in defs() and NOT in uses().
    #[test]
    fn function_call_context_defs_are_defs_not_uses() {
        let i = acvus_utils::Interner::new();
        let qref = acvus_utils::QualifiedRef::root(i.intern("f"));
        let inst = InstKind::FunctionCall {
            dst: v(3),
            callee: crate::ir::Callee::Direct(qref),
            args: vec![v(0)],
            context_uses: vec![],
            context_defs: vec![(qref, v(4)), (qref, v(5))],
        };
        let d = defs(&inst);
        assert!(d.contains(&v(3)), "dst must be in defs");
        assert!(d.contains(&v(4)), "context_defs[0] must be in defs");
        assert!(d.contains(&v(5)), "context_defs[1] must be in defs");
        let u = uses(&inst);
        assert!(!u.contains(&v(4)), "context_defs must NOT be in uses");
        assert!(!u.contains(&v(5)), "context_defs must NOT be in uses");
    }

    /// Regression: Eval context_defs are defs, not uses.
    #[test]
    fn eval_context_defs_are_defs_not_uses() {
        let i = acvus_utils::Interner::new();
        let qref = acvus_utils::QualifiedRef::root(i.intern("f"));
        let inst = InstKind::Eval {
            dst: v(2),
            src: v(1),
            context_defs: vec![(qref, v(3))],
        };
        let d = defs(&inst);
        assert!(d.contains(&v(2)), "dst must be in defs");
        assert!(d.contains(&v(3)), "context_defs must be in defs");
        let u = uses(&inst);
        assert!(u.contains(&v(1)), "src must be in uses");
        assert!(!u.contains(&v(3)), "context_defs must NOT be in uses");
    }

    #[test]
    fn indirect_call_uses_callee_value() {
        let inst = InstKind::FunctionCall {
            dst: v(2),
            callee: crate::ir::Callee::Indirect(v(0)),
            args: vec![v(1)],
            context_uses: vec![],
            context_defs: vec![],
        };
        let u = uses(&inst);
        assert!(u.contains(&v(0)), "indirect callee must be in uses");
        assert!(u.contains(&v(1)));
    }
}
