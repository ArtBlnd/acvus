use crate::graph::QualifiedRef;
use crate::ir::{InstKind, ValueId};
use crate::ty::Ty;
use acvus_ast::{BinOp, UnaryOp};
use rustc_hash::FxHashMap;

use crate::analysis::dataflow::{DataflowState, TransferFunction};
use crate::analysis::domain::{AbstractValue, FiniteSet, abstract_and, abstract_not, abstract_or};
use crate::analysis::reachable_context::KnownValue;
use smallvec::SmallVec;

pub struct ValueDomainTransfer<'a> {
    pub val_types: &'a FxHashMap<ValueId, Ty>,
    pub known_context: &'a FxHashMap<QualifiedRef, KnownValue>,
}

impl<'a> TransferFunction<AbstractValue> for ValueDomainTransfer<'a> {
    fn transfer_inst(&self, inst: &crate::ir::Inst, state: &mut DataflowState<AbstractValue>) {
        match &inst.kind {
            InstKind::Const { dst, value } => {
                state.set(*dst, AbstractValue::from_literal(value));
            }

            InstKind::ContextProject { dst, ctx, .. } => {
                let val = if let Some(kv) = self.known_context.get(ctx) {
                    AbstractValue::from_known_value(kv)
                } else {
                    AbstractValue::Top
                };
                state.set(*dst, val);
            }

            InstKind::ContextLoad { dst, src } => {
                // Copy the abstract value from the projection source.
                state.set(*dst, state.get(*src));
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
                                    AbstractValue::Finite(FiniteSet::Bools(SmallVec::from_elem(
                                        false, 1,
                                    ))),
                                );
                                return;
                            }
                            if variants.len() == 1 {
                                state.set(
                                    *dst,
                                    AbstractValue::Finite(FiniteSet::Bools(SmallVec::from_elem(
                                        true, 1,
                                    ))),
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

            InstKind::MakeTuple { dst, elements } => {
                let elems: Vec<AbstractValue> = elements.iter().map(|e| state.get(*e)).collect();
                state.set(*dst, AbstractValue::tuple(elems));
            }

            InstKind::TupleIndex { dst, tuple, index } => {
                let tuple_val = state.get(*tuple);
                state.set(*dst, tuple_val.tuple_index(*index));
            }

            // All other instructions that produce a value -> Top
            InstKind::VarLoad { dst, .. }
            | InstKind::ParamLoad { dst, .. }
            | InstKind::FieldGet { dst, .. }
            | InstKind::ObjectGet { dst, .. }
            | InstKind::FunctionCall { dst, .. }
            | InstKind::LoadFunction { dst, .. }
            | InstKind::MakeDeque { dst, .. }
            | InstKind::MakeObject { dst, .. }
            | InstKind::MakeRange { dst, .. }
            | InstKind::ListIndex { dst, .. }
            | InstKind::ListGet { dst, .. }
            | InstKind::ListSlice { dst, .. }
            | InstKind::MakeClosure { dst, .. }
            | InstKind::UnwrapVariant { dst, .. }
            | InstKind::TestListLen { dst, .. }
            | InstKind::TestObjectKey { dst, .. }
            | InstKind::Cast { dst, .. }
            | InstKind::Spawn { dst, .. }
            | InstKind::Eval { dst, .. }
            | InstKind::Poison { dst }
            | InstKind::Undef { dst } => {
                state.set(*dst, AbstractValue::Top);
            }

            InstKind::ListStep { dst, index_dst, .. } => {
                state.set(*dst, AbstractValue::Top);
                state.set(*index_dst, AbstractValue::Top);
            }

            InstKind::BinOp { dst, .. } => {
                // Other BinOps (Add, Sub, Eq, Lt, etc.) -> Top for now
                state.set(*dst, AbstractValue::Top);
            }

            InstKind::UnaryOp { dst, .. } => {
                // Other UnaryOps -> Top
                state.set(*dst, AbstractValue::Top);
            }

            // Instructions that don't produce values
            InstKind::VarStore { .. }
            | InstKind::ContextStore { .. }
            | InstKind::Return(_)
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::BlockLabel { .. }
            | InstKind::Nop => {}
        }
    }
}
