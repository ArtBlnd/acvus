//! Per-instruction def/use extraction.
//!
//! `defs(inst)` - ValueIds defined (written) by this instruction.
//! `uses(inst)` - ValueIds used (read) by this instruction.
//!
//! These are the building blocks for use-def analysis, DCE, reordering, etc.

use smallvec::{SmallVec, smallvec};

use crate::ir::{Callee, InstKind, ValueId};

/// ValueIds defined by this instruction.
pub fn defs(kind: &InstKind) -> SmallVec<[ValueId; 2]> {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::Ref { dst, .. }
        | InstKind::Take { dst, .. }
        | InstKind::Fetch { dst, .. }
        | InstKind::Load { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::FieldSet { dst, .. }
        | InstKind::LoadFunction { dst, .. }
        | InstKind::MakeArray { dst, .. }
        | InstKind::MakeObject { dst, .. }
        | InstKind::MakeTuple { dst, .. }
        | InstKind::TupleIndex { dst, .. }
        | InstKind::TestLiteral { dst, .. }
        | InstKind::TestObjectKey { dst, .. }
        | InstKind::ArrayIndex { dst, .. }
        | InstKind::ArrayGet { dst, .. }
        | InstKind::ObjectGet { dst, .. }
        | InstKind::MakeClosure { dst, .. }
        | InstKind::MakeVariant { dst, .. }
        | InstKind::TestVariant { dst, .. }
        | InstKind::UnwrapVariant { dst, .. }
        | InstKind::Spawn { dst, .. }
        | InstKind::Poison { dst }
        | InstKind::Undef { dst } => smallvec![*dst],

        InstKind::FunctionCall { dst, order, .. } => {
            let mut v: SmallVec<[ValueId; 2]> = smallvec![*dst];
            v.extend(order.map(|edge| edge.after));
            v
        }
        InstKind::Eval { dst, order, .. } => {
            let mut v: SmallVec<[ValueId; 2]> = smallvec![*dst];
            v.extend(*order);
            v
        }
        InstKind::Merge { dst, .. } => smallvec![*dst],

        InstKind::BlockLabel { params, .. } => params.iter().copied().collect(),

        InstKind::Store { .. }
        | InstKind::Assign { .. }
        | InstKind::Commit { .. }
        | InstKind::Drop { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Return { .. }
        | InstKind::Nop => smallvec![],
    }
}

/// ValueIds used (read) by this instruction.
pub fn uses(kind: &InstKind) -> SmallVec<[ValueId; 4]> {
    match kind {
        // No uses
        InstKind::Const { .. }
        | InstKind::Ref { .. }
        | InstKind::Take { .. }
        | InstKind::Fetch { .. }
        | InstKind::LoadFunction { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Nop
        | InstKind::Poison { .. }
        | InstKind::Undef { .. } => smallvec![],

        // Single use
        InstKind::Load { src, .. } => smallvec![*src],
        InstKind::Store { dst, value, .. } => smallvec![*dst, *value],
        InstKind::Assign { value, .. } => smallvec![*value],
        InstKind::Commit { value, .. } => smallvec![*value],
        InstKind::UnaryOp { operand, .. } => smallvec![*operand],
        InstKind::FieldGet { object, .. } => smallvec![*object],
        InstKind::FieldSet { object, value, .. } => smallvec![*object, *value],
        InstKind::Drop { src } => smallvec![*src],
        InstKind::Return { value, order } => {
            let mut v: SmallVec<[ValueId; 4]> = smallvec![*value];
            v.extend(*order);
            v
        }
        InstKind::Merge { orders, .. } => orders.iter().copied().collect(),
        InstKind::TestLiteral { src, .. } => smallvec![*src],
        InstKind::TestVariant { src, .. } => smallvec![*src],
        InstKind::UnwrapVariant { src, .. } => smallvec![*src],
        InstKind::ObjectGet { object, .. } => smallvec![*object],

        // Two uses
        InstKind::BinOp { left, right, .. } => smallvec![*left, *right],
        InstKind::TestObjectKey { src, .. } => smallvec![*src],
        InstKind::ArrayGet {
            array: list, index, ..
        } => smallvec![*list, *index],

        InstKind::ArrayIndex { array: list, .. } => smallvec![*list],

        // Composite constructors
        InstKind::MakeArray { elements, .. } => elements.iter().copied().collect(),
        InstKind::MakeObject { fields, .. } => fields.iter().map(|(_, v)| *v).collect(),
        InstKind::MakeTuple { elements, .. } => elements.iter().copied().collect(),
        InstKind::TupleIndex { tuple, .. } => smallvec![*tuple],

        // Variant
        InstKind::MakeVariant { payload, .. } => match payload {
            Some(v) => smallvec![*v],
            None => smallvec![],
        },

        // Closure
        InstKind::MakeClosure { captures, .. } => captures.iter().copied().collect(),

        // Function calls
        InstKind::FunctionCall {
            callee,
            args,
            order,
            ..
        } => {
            let mut v: SmallVec<[ValueId; 4]> = SmallVec::new();
            if let Callee::Indirect(f) = callee {
                v.push(*f);
            }
            v.extend(args.iter().copied());
            v.extend(order.map(|edge| edge.before));
            v
        }
        InstKind::Spawn {
            callee,
            args,
            order,
            ..
        } => {
            let mut v: SmallVec<[ValueId; 4]> = SmallVec::new();
            if let Callee::Indirect(f) = callee {
                v.push(*f);
            }
            v.extend(args.iter().copied());
            v.extend(*order);
            v
        }
        InstKind::Eval { src, .. } => smallvec![*src],

        // Iterator

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
            | InstKind::Return { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::Ty;
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
            callee_ty: Ty::error(),
            args: vec![v(0), v(1)],
            order: None,
        };
        assert_eq!(defs(&inst).as_slice(), &[v(3)]);
        let u = uses(&inst);
        assert!(u.contains(&v(0)));
        assert!(u.contains(&v(1)));
    }

    #[test]
    fn spawn_eval_defs_uses() {
        let i = acvus_utils::Interner::new();
        let qref = acvus_utils::QualifiedRef::root(i.intern("f"));

        let spawn = InstKind::Spawn {
            dst: v(1),
            callee: crate::ir::Callee::Direct(qref),
            callee_ty: Ty::error(),
            args: vec![v(0)],
            order: None,
        };
        assert_eq!(defs(&spawn).as_slice(), &[v(1)]);
        assert_eq!(uses(&spawn).as_slice(), &[v(0)]);

        let eval = InstKind::Eval {
            dst: v(2),
            src: v(1),
            order: None,
        };
        assert_eq!(defs(&eval).as_slice(), &[v(2)]);
        assert_eq!(uses(&eval).as_slice(), &[v(1)]);
    }

    #[test]
    fn return_uses_value() {
        let inst = InstKind::Return {
            value: v(5),
            order: None,
        };
        assert!(defs(&inst).is_empty());
        assert_eq!(uses(&inst).as_slice(), &[v(5)]);
    }

    #[test]
    fn control_flow_detection() {
        assert!(is_control_flow(&InstKind::Return {
            value: v(0),
            order: None
        }));
        assert!(is_control_flow(&InstKind::Jump {
            label: crate::ir::Label(0),
            args: vec![],
        }));
        assert!(!is_control_flow(&InstKind::Const {
            dst: v(0),
            value: acvus_ast::Literal::Int(1),
        }));
    }

    #[test]
    fn indirect_call_uses_callee_value() {
        let inst = InstKind::FunctionCall {
            dst: v(2),
            callee: crate::ir::Callee::Indirect(v(0)),
            callee_ty: Ty::error(),
            args: vec![v(1)],
            order: None,
        };
        let u = uses(&inst);
        assert!(u.contains(&v(0)), "indirect callee must be in uses");
        assert!(u.contains(&v(1)));
    }
}
