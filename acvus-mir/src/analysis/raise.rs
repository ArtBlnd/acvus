//! A pass removes an instruction whose value nothing reads only where
//! `Removal::stays_unused` says it neither can raise nor may run forever:
//! a trap the language defines stays where control puts it (RFC-0048 rule
//! 8), and no effect-free loop is assumed to end (RFC-0088, Rejected), so a
//! call that may not finish is not removed as it is not run ahead of an exit
//! (RFC-0089 rule 5).
//!
//! An overflowing `+`, `-`, `*`, negation or shift can raise nothing here, as
//! a decision: overflow is undefined (RFC-0037 rule 3), so no run of a
//! program that has a meaning raises it, and that rule lets a pass drop such
//! an operation where its value reaches nothing. Whether an instruction may
//! move is the separate question `inst_info::cannot_end_run` answers, and
//! there an overflow's trap counts.

use std::cell::OnceCell;

use rustc_hash::FxHashSet;

use crate::analysis::domtree::DomTree;
use crate::analysis::interval::{InstAt, untrapping};
use crate::analysis::loops::natural_loops_innermost_first;
use crate::cfg::{self, CfgBody};
use crate::graph::QualifiedRef;
use crate::ir::{BinOp, Callee, IndexBound, InstKind, MirBody};
use crate::laws::LawTable;
use crate::optimize::ssa_pass;
use crate::ty::Ty;

pub struct Removal<'a> {
    cfg: &'a CfgBody,
    laws: &'a LawTable,
    functions: &'a FunctionSummary,
    untrapping: OnceCell<FxHashSet<InstAt>>,
}

impl<'a> Removal<'a> {
    pub fn of(cfg: &'a CfgBody, laws: &'a LawTable, functions: &'a FunctionSummary) -> Self {
        Removal {
            cfg,
            laws,
            functions,
            untrapping: OnceCell::new(),
        }
    }

    pub fn stays_unused(&self, at: InstAt, kind: &InstKind) -> bool {
        self.can_raise(at, kind) || !finishes(kind, self.laws, self.functions)
    }

    fn can_raise(&self, at: InstAt, kind: &InstKind) -> bool {
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
            Callee::Direct(function) => !self.functions.cannot_raise(function),
            Callee::Indirect(_) => true,
        }
    }
}

/// `RunAhead` asks this with `FunctionSummary::unknown()`: `validate`
/// checks RFC-0089 rule 5 from one body alone (rule 3), so run-ahead knows
/// nothing of a callee's body.
pub fn finishes(kind: &InstKind, laws: &LawTable, functions: &FunctionSummary) -> bool {
    match kind {
        InstKind::FunctionCall { callee, .. } | InstKind::Spawn { callee, .. } => match callee {
            Callee::Extern { .. } => laws.returns_of(callee).returns_or_traps(),
            Callee::Direct(function) => functions.finishes(function),
            Callee::Indirect(_) => false,
        },
        InstKind::Eval { .. } => false,
        _ => true,
    }
}

#[derive(Debug, Default)]
pub struct FunctionSummary {
    finishing: FxHashSet<QualifiedRef>,
    untrapping: FxHashSet<QualifiedRef>,
}

impl FunctionSummary {
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

    fn finishes(&self, function: &QualifiedRef) -> bool {
        self.finishing.contains(function)
    }

    fn cannot_raise(&self, function: &QualifiedRef) -> bool {
        self.untrapping.contains(function)
    }

    /// `back_edges` refuses irreducible control flow, so every cycle of the
    /// body is one of the natural loops read here.
    fn offer(&mut self, function: QualifiedRef, main: &MirBody, laws: &LawTable) {
        let mut cfg = cfg::promote(main.clone());
        ssa_pass::run(&mut cfg);
        let removal = Removal::of(&cfg, laws, self);
        let insts = || {
            cfg.blocks.iter().enumerate().flat_map(|(b, block)| {
                block.insts.iter().enumerate().map(move |(at, inst)| {
                    let at = InstAt {
                        block: cfg::BlockIdx(b),
                        at,
                    };
                    (at, &inst.kind)
                })
            })
        };
        let raises = insts().any(|(at, kind)| removal.can_raise(at, kind));
        let finishes = natural_loops_innermost_first(&cfg, &DomTree::build(&cfg))
            .iter()
            .all(|loop_| !loop_.is_while(&cfg))
            && insts().all(|(_, kind)| finishes(kind, laws, self));
        if !raises {
            self.untrapping.insert(function);
        }
        if finishes {
            self.finishing.insert(function);
        }
    }
}
