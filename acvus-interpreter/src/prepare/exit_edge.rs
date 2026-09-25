use acvus_mir::ir::InstKind;

use super::Prepare;
use crate::code::{Op, chain};
use crate::ops::control;

pub(super) struct ExitEdge(Box<dyn Op>);

impl ExitEdge {
    pub(super) fn of_while(prep: &mut Prepare<'_>, test: usize) -> ExitEdge {
        let InstKind::JumpIf {
            else_label,
            else_args,
            ..
        } = &prep.body.insts[test].kind
        else {
            panic!("instruction {test} is not a `while`'s test")
        };
        let (label, args) = (*else_label, else_args.clone());
        ExitEdge(chain(prep.move_ops(&label, &args), Box::new(control::Yield)))
    }

    pub(super) fn of_for(prep: &mut Prepare<'_>, terminator: usize) -> ExitEdge {
        ExitEdge(chain(prep.exit_moves(terminator), Box::new(control::Yield)))
    }

    pub(super) fn into_op(self) -> Box<dyn Op> {
        self.0
    }
}
