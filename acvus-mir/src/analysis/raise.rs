//! A pass removes an instruction whose value nothing reads only where
//! `Raising::can_raise` says it cannot raise, because a trap the language
//! defines stays where control puts it (RFC-0048 rule 8). An integer `/` or
//! `%` panics on a zero divisor and at `MIN / -1` (RFC-0037 rule 2) unless
//! the interval domain of RFC-0047 rule 7 shows its operands keep clear of
//! both, and a call typed `!` ends the run (RFC-0038).
//!
//! An overflowing `+`, `-`, `*`, negation or shift can raise nothing here, as
//! a decision: overflow is undefined (RFC-0037 rule 3), so no run of a
//! program that has a meaning raises it, and that rule lets a pass drop such
//! an operation where its value reaches nothing. Whether an instruction may
//! move is the separate question `inst_info::cannot_end_run` answers, and
//! there an overflow's trap counts.

use rustc_hash::FxHashSet;

use crate::analysis::interval::{InstAt, unfailing_divisions};
use crate::cfg::CfgBody;
use crate::ir::{BinOp, InstKind};
use crate::laws::LawTable;
use crate::ty::Ty;

pub struct Raising<'a> {
    cfg: &'a CfgBody,
    unfailing_divisions: FxHashSet<InstAt>,
}

impl<'a> Raising<'a> {
    pub fn of(cfg: &'a CfgBody, laws: &LawTable) -> Self {
        let unfailing_divisions = match holds_integer_division(cfg) {
            true => unfailing_divisions(cfg, laws),
            false => FxHashSet::default(),
        };
        Raising {
            cfg,
            unfailing_divisions,
        }
    }

    pub fn can_raise(&self, at: InstAt, kind: &InstKind) -> bool {
        match kind {
            InstKind::BinOp {
                op: BinOp::Div | BinOp::Mod,
                left,
                ..
            } => is_integer(self.cfg, left) && !self.unfailing_divisions.contains(&at),
            InstKind::FunctionCall { callee_ty, .. } => {
                matches!(callee_ty, Ty::Fn { ret, .. } if matches!(**ret, Ty::Never))
            }
            _ => false,
        }
    }
}

fn holds_integer_division(cfg: &CfgBody) -> bool {
    cfg.blocks.iter().flat_map(|block| &block.insts).any(|inst| {
        matches!(
            &inst.kind,
            InstKind::BinOp { op: BinOp::Div | BinOp::Mod, left, .. } if is_integer(cfg, left)
        )
    })
}

fn is_integer(cfg: &CfgBody, value: &crate::ir::ValueId) -> bool {
    matches!(cfg.val_types.get(value), Some(Ty::Int(_)))
}
