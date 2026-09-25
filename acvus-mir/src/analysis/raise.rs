//! A pass removes an instruction whose value nothing reads only where
//! `Raising::can_raise` says it cannot raise, because a trap the language
//! defines stays where control puts it (RFC-0048 rule 8).
//!
//! An overflowing `+`, `-`, `*`, negation or shift can raise nothing here, as
//! a decision: overflow is undefined (RFC-0037 rule 3), so no run of a
//! program that has a meaning raises it, and that rule lets a pass drop such
//! an operation where its value reaches nothing. Whether an instruction may
//! move is the separate question `inst_info::cannot_end_run` answers, and
//! there an overflow's trap counts.

use std::cell::OnceCell;

use rustc_hash::FxHashSet;

use crate::analysis::interval::{InstAt, untrapping};
use crate::cfg::{self, CfgBody};
use crate::graph::QualifiedRef;
use crate::ir::{BinOp, Callee, IndexBound, InstKind, MirBody};
use crate::laws::LawTable;
use crate::optimize::ssa_pass;
use crate::ty::Ty;

pub struct Raising<'a> {
    cfg: &'a CfgBody,
    laws: &'a LawTable,
    functions: &'a UntrappingFunctions,
    untrapping: OnceCell<FxHashSet<InstAt>>,
}

impl<'a> Raising<'a> {
    pub fn of(cfg: &'a CfgBody, laws: &'a LawTable, functions: &'a UntrappingFunctions) -> Self {
        Raising {
            cfg,
            laws,
            functions,
            untrapping: OnceCell::new(),
        }
    }

    pub fn can_raise(&self, at: InstAt, kind: &InstKind) -> bool {
        match kind {
            InstKind::BinOp {
                op: BinOp::Div | BinOp::Mod,
                left,
                ..
            } => matches!(self.cfg.val_types.get(left), Some(Ty::Int(_))) && !self.ruled_out(at),
            InstKind::Index {
                bound: IndexBound::Checked,
                ..
            }
            | InstKind::IndexSet {
                bound: IndexBound::Checked,
                ..
            } => !self.ruled_out(at),
            InstKind::FunctionCall {
                callee, callee_ty, ..
            }
            | InstKind::Spawn {
                callee, callee_ty, ..
            } => self.call_can_raise(callee, callee_ty),
            InstKind::Check { .. }
            | InstKind::CheckSteps { .. }
            | InstKind::Fetch { .. }
            | InstKind::Commit { .. }
            | InstKind::Eval { .. } => true,
            _ => false,
        }
    }

    fn ruled_out(&self, at: InstAt) -> bool {
        self.untrapping
            .get_or_init(|| untrapping(self.cfg, self.laws))
            .contains(&at)
    }

    fn call_can_raise(&self, callee: &Callee, callee_ty: &Ty) -> bool {
        if matches!(callee_ty, Ty::Fn { ret, .. } if matches!(**ret, Ty::Never)) {
            return true;
        }
        match callee {
            Callee::Extern { .. } => !self.laws.returns_of(callee).never_traps(),
            Callee::Direct(function) => !self.functions.contains(function),
            Callee::Indirect(_) => true,
        }
    }
}

#[derive(Debug, Default)]
pub struct UntrappingFunctions {
    functions: FxHashSet<QualifiedRef>,
}

impl UntrappingFunctions {
    pub fn unknown() -> Self {
        Self::default()
    }

    pub fn of<'m>(
        callees_first: impl IntoIterator<Item = (QualifiedRef, &'m MirBody)>,
        laws: &LawTable,
    ) -> Self {
        let mut this = Self::unknown();
        for (function, main) in callees_first {
            this.offer(function, main, laws);
        }
        this
    }

    fn contains(&self, function: &QualifiedRef) -> bool {
        self.functions.contains(function)
    }

    fn offer(&mut self, function: QualifiedRef, main: &MirBody, laws: &LawTable) {
        let mut cfg = cfg::promote(main.clone());
        ssa_pass::run(&mut cfg);
        let raising = Raising::of(&cfg, laws, self);
        let raises = cfg.blocks.iter().enumerate().any(|(b, block)| {
            block.insts.iter().enumerate().any(|(at, inst)| {
                let at = InstAt {
                    block: cfg::BlockIdx(b),
                    at,
                };
                raising.can_raise(at, &inst.kind)
            })
        });
        if !raises {
            self.functions.insert(function);
        }
    }
}
