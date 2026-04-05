//! Token liveness analysis — backward dataflow over the CFG.
//!
//! Computes which token QualifiedRefs are live at each program point.
//! A token is live at a point if there exists a path forward from that point
//! to an instruction that uses the token (FunctionCall/Spawn/Eval with that token
//! in its effect).
//!
//! This is used by the reorder pass to determine whether two instructions
//! sharing a token can be reordered relative to each other.

use rustc_hash::FxHashSet;

use crate::analysis::dataflow::{DataflowAnalysis, DataflowState, backward_analysis};
use crate::analysis::domain::SemiLattice;
use crate::cfg::{BlockIdx, CfgBody};
use crate::graph::{FnMetadata, QualifiedRef};
use crate::ir::{Callee, Inst, InstKind, ValueId};
use crate::ty::{Effect, EffectTarget, Ty};
use rustc_hash::FxHashMap;

// ── Domain ──────────────────────────────────────────────────────────

/// Token liveness domain: a token is either Live or Dead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenLiveness {
    Dead,
    Live,
}

impl SemiLattice for TokenLiveness {
    fn bottom() -> Self {
        TokenLiveness::Dead
    }

    fn join_mut(&mut self, other: &Self) -> bool {
        match (*self, *other) {
            (TokenLiveness::Live, _) => false,
            (TokenLiveness::Dead, TokenLiveness::Live) => {
                *self = TokenLiveness::Live;
                true
            }
            (TokenLiveness::Dead, TokenLiveness::Dead) => false,
        }
    }
}

// ── Analysis ────────────────────────────────────────────────────────

struct TokenLivenessAnalysis<'a> {
    fn_metadata: &'a FxHashMap<QualifiedRef, FnMetadata>,
    val_types: &'a FxHashMap<ValueId, Ty>,
}

/// Extract token QualifiedRefs from an instruction's effect.
fn token_ids_of(
    kind: &InstKind,
    fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>,
    val_types: &FxHashMap<ValueId, Ty>,
) -> smallvec::SmallVec<[QualifiedRef; 2]> {
    let effect_set = match kind {
        InstKind::FunctionCall {
            callee: Callee::Direct(qref),
            ..
        }
        | InstKind::Spawn {
            callee: Callee::Direct(qref),
            ..
        } => fn_metadata.get(qref).and_then(|m| match &m.ty {
            Ty::Fn {
                effect: Effect::Resolved(eff),
                ..
            } => Some(eff),
            _ => None,
        }),
        InstKind::Eval { src, .. } => val_types.get(src).and_then(|ty| match ty {
            Ty::Handle(_, Effect::Resolved(eff)) => Some(eff),
            _ => None,
        }),
        _ => None,
    };

    let Some(eff) = effect_set else {
        return smallvec::SmallVec::new();
    };

    eff.reads
        .iter()
        .chain(eff.writes.iter())
        .filter_map(|target| match target {
            EffectTarget::Token(tid) => Some(*tid),
            _ => None,
        })
        .collect()
}

impl<'a> DataflowAnalysis for TokenLivenessAnalysis<'a> {
    type Key = QualifiedRef;
    type Domain = TokenLiveness;

    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<QualifiedRef, TokenLiveness>) {
        // Gen: if this instruction uses a token, mark it live.
        for tid in token_ids_of(&inst.kind, self.fn_metadata, self.val_types) {
            state.set(tid, TokenLiveness::Live);
        }
        // No kill: tokens don't have defs/redefinitions.
    }

    fn propagate_forward(
        &self,
        _source_exit: &DataflowState<QualifiedRef, TokenLiveness>,
        _params: &[ValueId],
        _args: &[ValueId],
        _target_entry: &mut DataflowState<QualifiedRef, TokenLiveness>,
    ) -> bool {
        unreachable!("token liveness is backward-only")
    }

    fn propagate_backward(
        &self,
        succ_entry: &DataflowState<QualifiedRef, TokenLiveness>,
        _succ_params: &[ValueId],
        _term_args: &[ValueId],
        exit_state: &mut DataflowState<QualifiedRef, TokenLiveness>,
    ) {
        // Tokens are not SSA — no param/arg mapping. Pure join.
        exit_state.join_from(succ_entry);
    }
}

// ── Result ──────────────────────────────────────────────────────────

/// Per-block token liveness sets.
pub struct TokenLivenessResult {
    pub live_in: Vec<FxHashSet<QualifiedRef>>,
    pub live_out: Vec<FxHashSet<QualifiedRef>>,
}

impl TokenLivenessResult {
    /// Is `token` live at the entry of `block`?
    pub fn is_live_in(&self, block: BlockIdx, token: QualifiedRef) -> bool {
        self.live_in
            .get(block.0)
            .is_some_and(|set| set.contains(&token))
    }

    /// Is `token` live at the exit of `block`?
    pub fn is_live_out(&self, block: BlockIdx, token: QualifiedRef) -> bool {
        self.live_out
            .get(block.0)
            .is_some_and(|set| set.contains(&token))
    }
}

/// Run token liveness analysis on a CfgBody.
pub fn analyze(cfg: &CfgBody, fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>) -> TokenLivenessResult {
    if cfg.blocks.is_empty() {
        return TokenLivenessResult {
            live_in: vec![],
            live_out: vec![],
        };
    }

    let analysis = TokenLivenessAnalysis {
        fn_metadata,
        val_types: &cfg.val_types,
    };
    let result = backward_analysis(cfg, &analysis);

    let extract = |state: &DataflowState<QualifiedRef, TokenLiveness>| -> FxHashSet<QualifiedRef> {
        state
            .values
            .iter()
            .filter_map(|(&tid, &liveness)| match liveness {
                TokenLiveness::Live => Some(tid),
                TokenLiveness::Dead => None,
            })
            .collect()
    };

    let live_in = result.block_entry.iter().map(extract).collect();
    let live_out = result.block_exit.iter().map(extract).collect();

    TokenLivenessResult { live_in, live_out }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::*;
    use crate::ty::{Effect, EffectSet, EffectTarget};
    use acvus_utils::{LocalFactory, LocalIdOps};
    use std::collections::BTreeSet;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..val_count {
            factory.next();
        }
        promote(MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn io_fn_type_with_token(
        i: &acvus_utils::Interner,
        name: &str,
        tid: QualifiedRef,
    ) -> (QualifiedRef, FnMetadata) {
        let qref = QualifiedRef::root(i.intern(name));
        let mut reads = BTreeSet::new();
        reads.insert(EffectTarget::Token(tid));
        (
            qref,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: Effect::Resolved(EffectSet {
                        reads,
                        writes: BTreeSet::new(),
                        self_modifying: false,
                    }),
                },
                hint: None,
            },
        )
    }

    fn pure_fn_type(i: &acvus_utils::Interner, name: &str) -> (QualifiedRef, FnMetadata) {
        let qref = QualifiedRef::root(i.intern(name));
        (
            qref,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: Effect::pure(),
                },
                hint: None,
            },
        )
    }

    #[test]
    fn single_block_token_live() {
        let i = acvus_utils::Interner::new();
        let tid = QualifiedRef::root(i.intern("net"));
        let (qref, ty) = io_fn_type_with_token(&i, "io_fn", tid);
        let fn_metadata: FxHashMap<QualifiedRef, FnMetadata> = FxHashMap::from_iter([(qref, ty)]);

        let cfg = make_cfg(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(qref),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(0)),
            ],
            5,
        );

        let result = analyze(&cfg, &fn_metadata);
        assert!(result.is_live_in(BlockIdx(0), tid));
    }

    #[test]
    fn pure_call_no_token() {
        let i = acvus_utils::Interner::new();
        let tid = QualifiedRef::root(i.intern("net"));
        let (qref, ty) = pure_fn_type(&i, "pure_fn");
        let fn_metadata: FxHashMap<QualifiedRef, FnMetadata> = FxHashMap::from_iter([(qref, ty)]);

        let cfg = make_cfg(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(qref),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(0)),
            ],
            5,
        );

        let result = analyze(&cfg, &fn_metadata);
        assert!(!result.is_live_in(BlockIdx(0), tid));
    }

    #[test]
    fn token_live_across_blocks() {
        let i = acvus_utils::Interner::new();
        let tid = QualifiedRef::root(i.intern("net"));
        let (qref, ty) = io_fn_type_with_token(&i, "io_fn", tid);
        let fn_metadata: FxHashMap<QualifiedRef, FnMetadata> = FxHashMap::from_iter([(qref, ty)]);

        let cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Jump {
                    label: Label(0),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(qref),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            5,
        );

        let result = analyze(&cfg, &fn_metadata);
        assert!(result.is_live_out(BlockIdx(0), tid));
        assert!(result.is_live_in(BlockIdx(1), tid));
    }

    #[test]
    fn two_tokens_independent() {
        let i = acvus_utils::Interner::new();
        let tid1 = QualifiedRef::root(i.intern("net"));
        let tid2 = QualifiedRef::root(i.intern("fs"));
        let (qref1, ty1) = io_fn_type_with_token(&i, "io1", tid1);
        let (qref2, ty2) = io_fn_type_with_token(&i, "io2", tid2);
        let fn_metadata: FxHashMap<QualifiedRef, FnMetadata> =
            FxHashMap::from_iter([(qref1, ty1), (qref2, ty2)]);

        let cfg = make_cfg(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(qref1),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Jump {
                    label: Label(0),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(qref2),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            5,
        );

        let result = analyze(&cfg, &fn_metadata);
        assert!(result.is_live_in(BlockIdx(0), tid1));
        assert!(result.is_live_out(BlockIdx(0), tid2));
        assert!(result.is_live_in(BlockIdx(1), tid2));
        assert!(!result.is_live_in(BlockIdx(1), tid1));
    }

    #[test]
    fn diamond_token_both_arms() {
        let i = acvus_utils::Interner::new();
        let tid = QualifiedRef::root(i.intern("net"));
        let (qref, ty) = io_fn_type_with_token(&i, "io_fn", tid);
        let fn_metadata: FxHashMap<QualifiedRef, FnMetadata> = FxHashMap::from_iter([(qref, ty)]);

        let cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(qref),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(1)],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::FunctionCall {
                    dst: v(2),
                    callee: Callee::Direct(qref),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(2)],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Return(v(3)),
            ],
            5,
        );

        let result = analyze(&cfg, &fn_metadata);
        assert!(result.is_live_out(BlockIdx(0), tid));
        assert!(!result.is_live_in(BlockIdx(3), tid));
    }
}
