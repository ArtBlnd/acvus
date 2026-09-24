//! Per-instruction def/use extraction.
//!
//! `defs(inst)` - ValueIds defined (written) by this instruction.
//! `uses(inst)` - ValueIds used (read) by this instruction.
//!
//! These are the building blocks for use-def analysis, DCE, reordering, etc.

use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};

use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Callee, InstKind, RefTarget, ValueId};

/// ValueIds defined by this instruction.
pub fn defs(kind: &InstKind) -> SmallVec<[ValueId; 2]> {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::ConstStr { dst, .. }
        | InstKind::Ref { dst, .. }
        | InstKind::Take { dst, .. }
        | InstKind::Fetch { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::Cast { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::FieldSet { dst, .. }
        | InstKind::LoadFunction { dst, .. }
        | InstKind::MakeArray { dst, .. }
        | InstKind::StringConcat { dst, .. }
        | InstKind::StringEq { dst, .. }
        | InstKind::StringClone { dst, .. }
        | InstKind::StructuralEq { dst, .. }
        | InstKind::StructuralClone { dst, .. }
        | InstKind::MakeObject { dst, .. }
        | InstKind::MakeTuple { dst, .. }
        | InstKind::TupleIndex { dst, .. }
        | InstKind::TestLiteral { dst, .. }
        | InstKind::TestObjectKey { dst, .. }
        | InstKind::ArrayIndex { dst, .. }
        | InstKind::AsSlice { dst, .. }
        | InstKind::Index { dst, .. }
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

        InstKind::Assign { target, .. } => storage(target).into_iter().collect(),

        InstKind::Commit { .. }
        | InstKind::Drop { .. }
        | InstKind::IndexSet { .. }
        | InstKind::StringAppend { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Diamond { .. }
        | InstKind::Switch { .. }
        | InstKind::For { .. }
        | InstKind::Return { .. }
        | InstKind::Diverge
        | InstKind::Nop => smallvec![],
    }
}

/// ValueIds used (read) by this instruction.
pub fn uses(kind: &InstKind) -> SmallVec<[ValueId; 4]> {
    match kind {
        // No uses
        InstKind::Const { .. }
        | InstKind::ConstStr { .. }
        | InstKind::Diverge
        | InstKind::Fetch { .. }
        | InstKind::LoadFunction { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Nop
        | InstKind::Poison { .. }
        | InstKind::Undef { .. } => smallvec![],

        // A place through a reference uses the reference; the storage a
        // place names directly is not a value (`loans::Loans` counts it).
        InstKind::Ref { target, .. } | InstKind::Take { target, .. } => {
            through(target).into_iter().collect()
        }
        InstKind::Assign { target, value, .. } => {
            let mut v: SmallVec<[ValueId; 4]> = SmallVec::new();
            v.extend(through(target));
            v.push(*value);
            v
        }

        // Single use
        InstKind::Commit { value, .. } => smallvec![*value],
        InstKind::UnaryOp { operand, .. } => smallvec![*operand],
        InstKind::Cast { src, .. } => smallvec![*src],
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
        InstKind::ArrayIndex { array: list, .. } => smallvec![*list],

        // Slices (RFC-0047)
        InstKind::AsSlice { container, .. } => smallvec![*container],
        InstKind::Index { slice, index, .. } => smallvec![*slice, *index],
        InstKind::IndexSet {
            slice,
            index,
            value,
            bound: _,
        } => smallvec![*slice, *index, *value],

        // Composite constructors
        InstKind::MakeArray { elements, .. } => elements.iter().copied().collect(),
        InstKind::StringConcat { parts, .. } => parts.iter().copied().collect(),
        InstKind::StringAppend { target, part } => smallvec![*target, *part],
        InstKind::StringEq { a, b, .. } => smallvec![*a, *b],
        InstKind::StringClone { src, .. } => smallvec![*src],
        InstKind::StructuralEq { a, b, .. } => smallvec![*a, *b],
        InstKind::StructuralClone { src, .. } => smallvec![*src],
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
        }
        | InstKind::Diamond {
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
        // The source is read once, and the exit edge carries the block
        // arguments its target takes after the trip count the terminator
        // fills itself (RFC-0057).
        InstKind::For {
            source, exit_args, ..
        } => traversal_uses(source, exit_args),
        // The tag is read once; every edge carries the block arguments
        // its target takes (RFC-0051).
        InstKind::Switch { tag, arms, default } => {
            let mut v: SmallVec<[ValueId; 4]> = smallvec![*tag];
            for (_, _, args) in arms {
                v.extend(args.iter().copied());
            }
            if let Some((_, args)) = default {
                v.extend(args.iter().copied());
            }
            v
        }
    }
}

/// ValueIds a block's terminator reads: what [`uses`] gives the control
/// flow instruction the terminator was promoted from.
pub fn terminator_uses(term: &Terminator) -> SmallVec<[ValueId; 4]> {
    match term {
        Terminator::Jump { args, .. } => args.iter().copied().collect(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
            cond,
            then_args,
            else_args,
            ..
        } => std::iter::once(*cond)
            .chain(then_args.iter().copied())
            .chain(else_args.iter().copied())
            .collect(),
        Terminator::For {
            source, exit_args, ..
        } => traversal_uses(source, exit_args),
        Terminator::Switch { tag, arms, default } => std::iter::once(*tag)
            .chain(arms.iter().flat_map(|(_, _, args)| args.iter().copied()))
            .chain(default.iter().flat_map(|(_, args)| args.iter().copied()))
            .collect(),
        Terminator::Return { value, order, .. } => std::iter::once(*value).chain(*order).collect(),
        Terminator::Diverge | Terminator::Fallthrough => SmallVec::new(),
    }
}

fn traversal_uses(source: &crate::ir::ForSource, exit_args: &[ValueId]) -> SmallVec<[ValueId; 4]> {
    source
        .uses()
        .into_iter()
        .chain(exit_args.iter().copied())
        .collect()
}

/// The storage slot a place names directly, if any.
pub fn storage(target: &RefTarget) -> Option<ValueId> {
    match target {
        RefTarget::Var(s) | RefTarget::Param(s) => Some(*s),
        RefTarget::Through(_) => None,
    }
}

fn through(target: &RefTarget) -> Option<ValueId> {
    match target {
        RefTarget::Through(r) => Some(*r),
        RefTarget::Var(_) | RefTarget::Param(_) => None,
    }
}

/// Is this instruction a control flow boundary (block label, jump, branch, return)?
pub fn is_control_flow(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::BlockLabel { .. }
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::Diamond { .. }
            | InstKind::Return { .. }
            | InstKind::Diverge
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
            op: crate::ir::BinOp::Add(crate::ir::Overflow::Trap),
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

/// How many times the instructions and terminators of some blocks read each
/// value.
pub(crate) struct Reads {
    by_value: FxHashMap<ValueId, usize>,
}

impl Reads {
    pub(crate) fn of<I>(cfg: &CfgBody, blocks: I) -> Self
    where
        I: IntoIterator<Item = BlockIdx>,
    {
        let mut by_value: FxHashMap<ValueId, usize> = FxHashMap::default();
        for block in blocks {
            let block = &cfg.blocks[block.0];
            let insts = block.insts.iter().flat_map(|inst| uses(&inst.kind));
            for value in insts.chain(terminator_uses(&block.terminator)) {
                *by_value.entry(value).or_default() += 1;
            }
        }
        Self { by_value }
    }

    pub(crate) fn in_body(cfg: &CfgBody) -> Self {
        Self::of(cfg, (0..cfg.blocks.len()).map(BlockIdx))
    }

    pub(crate) fn count(&self, value: ValueId) -> usize {
        self.by_value.get(&value).copied().unwrap_or(0)
    }
}
