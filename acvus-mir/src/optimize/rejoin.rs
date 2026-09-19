//! A demoted diamond returns.
//!
//! `optimize::sroa` dissolves the block a `Diamond`'s arms met at, and where
//! its threading scatters the paths through that block the branch is demoted
//! to a `JumpIf` (`cfg::demote_diamond`) -- correct at that instant, because
//! at that instant the arms do not rejoin. `optimize::forward` and
//! `optimize::dce` then bring the arms back to one block, and the terminator
//! the source wrote is a `Diamond` again.
//!
//! # Why it runs last, and once
//!
//! The demotion is one pass and the arms' return is two others, so there is
//! no single edge rewrite to hang the restore on. Asked after every pass has
//! run, the question needs no history: the arms either meet again or they do
//! not, and `ir::meets_again` is what `lower`'s `close_diamond` asked of the
//! same arms before any pass moved them.

use crate::ir::{Branch, Inst, InstKind, Label, MirBody, demoted_branches, meets_again, two_way};

pub fn run(body: &mut MirBody) {
    let restored: Vec<Restored> = demoted_branches(&body.insts, &body.demoted_diamonds)
        .filter_map(|branch| Restored::of(&body.insts, branch))
        .collect();
    for restored in restored {
        body.insts[restored.at].kind = restored.diamond;
        body.demoted_diamonds.remove(&restored.block);
    }
}

/// A demoted branch whose arms meet again, and the terminator that restores
/// it.
struct Restored {
    at: usize,
    block: Label,
    diamond: InstKind,
}

impl Restored {
    fn of(insts: &[Inst], branch: Branch) -> Option<Self> {
        let join = meets_again(insts, branch.at)?;
        let demoted = two_way(&insts[branch.at].kind)?;
        Some(Self {
            at: branch.at,
            block: branch.block,
            diamond: InstKind::Diamond {
                cond: demoted.cond,
                then_label: demoted.then_label,
                then_args: demoted.then_args.to_vec(),
                else_label: demoted.else_label,
                else_args: demoted.else_args.to_vec(),
                join,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ValueId;
    use acvus_utils::LocalFactory;

    fn inst(kind: InstKind) -> Inst {
        Inst {
            span: acvus_ast::Span::ZERO,
            kind,
        }
    }

    const LOOP_TEST: usize = 2;
    const MARKED_BRANCH: usize = 4;
    const CONTINUES: Label = Label(5);
    const LEAVES: Label = Label(9);

    fn loop_with_a_marked_branch(else_label: Label) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let test = factory.next();
        let inner = factory.next();
        let result = factory.next();
        let block = |label: u32| {
            inst(InstKind::BlockLabel {
                label: Label(label),
                params: vec![],
            })
        };
        let jump = |label: u32| {
            inst(InstKind::Jump {
                label: Label(label),
                args: vec![],
            })
        };
        let mut body = MirBody::new();
        body.insts = vec![
            jump(0),
            block(0),
            inst(InstKind::JumpIf {
                cond: test,
                then_label: Label(1),
                then_args: vec![],
                else_label: LEAVES,
                else_args: vec![],
            }),
            block(1),
            inst(InstKind::JumpIf {
                cond: inner,
                then_label: Label(2),
                then_args: vec![],
                else_label,
                else_args: vec![],
            }),
            block(2),
            jump(CONTINUES.0),
            block(CONTINUES.0),
            jump(0),
            block(LEAVES.0),
            inst(InstKind::Return {
                value: result,
                order: None,
            }),
        ];
        body.val_factory = factory;
        body.label_count = 10;
        body.demoted_diamonds.insert(Label(1));
        body
    }

    fn branch_at(body: &MirBody, at: usize) -> &InstKind {
        &body.insts[at].kind
    }

    #[test]
    fn a_demoted_branch_whose_arms_meet_again_becomes_a_diamond() {
        let mut body = loop_with_a_marked_branch(CONTINUES);
        run(&mut body);
        assert!(
            matches!(
                branch_at(&body, MARKED_BRANCH),
                InstKind::Diamond { join, .. } if *join == CONTINUES
            ),
            "{:?}",
            branch_at(&body, MARKED_BRANCH)
        );
        assert!(body.demoted_diamonds.is_empty());
    }

    #[test]
    fn a_demoted_branch_whose_arm_leaves_the_loop_stays_a_jump_if() {
        let mut body = loop_with_a_marked_branch(LEAVES);
        run(&mut body);
        assert!(
            matches!(branch_at(&body, MARKED_BRANCH), InstKind::JumpIf { .. }),
            "{:?}",
            branch_at(&body, MARKED_BRANCH)
        );
        assert_eq!(body.demoted_diamonds.len(), 1);
    }

    #[test]
    fn a_branch_no_pass_demoted_is_left_alone_though_its_arms_meet() {
        let mut body = loop_with_a_marked_branch(CONTINUES);
        body.demoted_diamonds.clear();
        run(&mut body);
        assert!(
            matches!(branch_at(&body, MARKED_BRANCH), InstKind::JumpIf { .. }),
            "{:?}",
            branch_at(&body, MARKED_BRANCH)
        );
    }

    #[test]
    fn the_loops_own_test_is_untouched() {
        let mut body = loop_with_a_marked_branch(CONTINUES);
        run(&mut body);
        assert!(
            matches!(branch_at(&body, LOOP_TEST), InstKind::JumpIf { .. }),
            "{:?}",
            branch_at(&body, LOOP_TEST)
        );
    }
}
