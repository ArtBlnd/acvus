use acvus_ast::{BinOp, UnaryOp};
use acvus_mir::ir::{InstKind, ValueId};
use acvus_mir::ty::Ty;
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::analysis::dataflow::{DataflowState, TransferFunction};
use crate::analysis::domain::{
    AbstractValue, FiniteSet, abstract_and, abstract_not, abstract_or,
};
use crate::analysis::reachable_context::KnownValue;
use smallvec::SmallVec;

pub struct ValueDomainTransfer<'a> {
    pub val_types: &'a FxHashMap<ValueId, Ty>,
    pub known_context: &'a FxHashMap<Astr, KnownValue>,
}

impl<'a> TransferFunction<AbstractValue> for ValueDomainTransfer<'a> {
    fn transfer_inst(
        &self,
        inst: &acvus_mir::ir::Inst,
        state: &mut DataflowState<AbstractValue>,
    ) {
        match &inst.kind {
            InstKind::Const { dst, value } => {
                state.set(*dst, AbstractValue::from_literal(value));
            }

            InstKind::ContextLoad { dst, name } => {
                let val = if let Some(kv) = self.known_context.get(name) {
                    AbstractValue::from_known_value(kv)
                } else {
                    AbstractValue::Top
                };
                state.set(*dst, val);
            }

            InstKind::TestLiteral { dst, src, value } => {
                let src_val = state.get(*src);
                state.set(*dst, src_val.test_literal(value));
            }

            InstKind::TestRange {
                dst,
                src,
                start,
                end,
                kind,
            } => {
                let src_val = state.get(*src);
                state.set(*dst, src_val.test_range(*start, *end, *kind));
            }

            InstKind::TestVariant { dst, src, tag } => {
                // Type-based pruning first
                if let Some(ty) = self.val_types.get(src) {
                    match ty {
                        Ty::Enum { variants, .. } => {
                            if !variants.contains_key(tag) {
                                state.set(
                                    *dst,
                                    AbstractValue::Finite(FiniteSet::Bools(
                                        SmallVec::from_elem(false, 1),
                                    )),
                                );
                                return;
                            }
                            if variants.len() == 1 {
                                state.set(
                                    *dst,
                                    AbstractValue::Finite(FiniteSet::Bools(
                                        SmallVec::from_elem(true, 1),
                                    )),
                                );
                                return;
                            }
                        }
                        Ty::Option(_) => {
                            // No pruning from type alone for Option
                        }
                        _ => {}
                    }
                }

                // Value-based test
                let src_val = state.get(*src);
                state.set(*dst, src_val.test_variant(*tag));
            }

            InstKind::BinOp {
                dst,
                op: BinOp::And,
                left,
                right,
            } => {
                let l = state.get(*left);
                let r = state.get(*right);
                state.set(*dst, abstract_and(&l, &r));
            }

            InstKind::BinOp {
                dst,
                op: BinOp::Or,
                left,
                right,
            } => {
                let l = state.get(*left);
                let r = state.get(*right);
                state.set(*dst, abstract_or(&l, &r));
            }

            InstKind::UnaryOp {
                dst,
                op: UnaryOp::Not,
                operand,
            } => {
                let v = state.get(*operand);
                state.set(*dst, abstract_not(&v));
            }

            InstKind::MakeVariant { dst, tag, payload } => {
                let payload_val = match payload {
                    Some(p) => state.get(*p),
                    None => AbstractValue::Bottom,
                };
                state.set(*dst, AbstractValue::variant(*tag, payload_val));
            }

            // All other instructions that produce a value → Top
            InstKind::VarLoad { dst, .. }
            | InstKind::FieldGet { dst, .. }
            | InstKind::ObjectGet { dst, .. }
            | InstKind::BuiltinCall { dst, .. }
            | InstKind::ExternCall { dst, .. }
            | InstKind::ClosureCall { dst, .. }
            | InstKind::MakeList { dst, .. }
            | InstKind::MakeObject { dst, .. }
            | InstKind::MakeRange { dst, .. }
            | InstKind::MakeTuple { dst, .. }
            | InstKind::TupleIndex { dst, .. }
            | InstKind::ListIndex { dst, .. }
            | InstKind::ListGet { dst, .. }
            | InstKind::ListSlice { dst, .. }
            | InstKind::MakeClosure { dst, .. }
            | InstKind::IterInit { dst, .. }
            | InstKind::UnwrapVariant { dst, .. }
            | InstKind::TestListLen { dst, .. }
            | InstKind::TestObjectKey { dst, .. }
            | InstKind::Poison { dst } => {
                state.set(*dst, AbstractValue::Top);
            }

            InstKind::BinOp { dst, .. } => {
                // Other BinOps (Add, Sub, Eq, Lt, etc.) → Top for now
                state.set(*dst, AbstractValue::Top);
            }

            InstKind::UnaryOp { dst, .. } => {
                // Other UnaryOps → Top
                state.set(*dst, AbstractValue::Top);
            }

            InstKind::IterNext {
                dst_value,
                dst_done,
                ..
            } => {
                state.set(*dst_value, AbstractValue::Top);
                state.set(*dst_done, AbstractValue::Top);
            }

            // Instructions that don't produce values
            InstKind::Yield(_)
            | InstKind::VarStore { .. }
            | InstKind::Return(_)
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::BlockLabel { .. }
            | InstKind::Nop => {}
        }
    }
}
