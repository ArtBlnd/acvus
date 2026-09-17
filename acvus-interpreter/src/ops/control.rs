//! Control flow: the jumps and their parallel moves, the return, and the
//! instructions that produce no control at all.

use crate::code::{Flow, Op, Payload, SlotMove};
use crate::error::RuntimeError;
use crate::machine::Machine;
use crate::ops::payload;
use crate::value::Value;

/// A jump's arguments move into the block's parameters: a move-only
/// argument leaves its register, so one owner remains.
#[inline]
fn move_all(machine: &mut Machine<'_>, moves: &[SlotMove]) {
    for m in moves {
        let value = machine.use_val(m.from);
        machine.set(m.to, value);
    }
}

pub fn jump(machine: &mut Machine<'_>, op: &Op) -> Flow {
    move_all(machine, payload!(machine, op, Moves));
    Flow::Jump(op.b)
}

/// `p` holds the moves of the `then` side, `d` the index of the `else`
/// side's.
pub fn jump_if(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let taken = machine.reg(op.a).as_bool();
    let (target, at) = if taken {
        (op.b, op.p)
    } else {
        (op.c, op.d as usize)
    };
    let moves = match &machine.code().payloads[at] {
        Payload::Moves(moves) => moves,
        other => panic!(
            "a conditional jump wants a Moves payload, found {}",
            crate::code::payload_name(other)
        ),
    };
    move_all(machine, moves);
    Flow::Jump(target)
}

pub fn ret(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = machine.take(op.a);
    machine.finish(value)
}

pub fn diverge(machine: &mut Machine<'_>, _: &Op) -> Flow {
    machine.fail(RuntimeError::internal(
        "a call typed `!` returned: its handler must trap",
    ))
}

pub fn merge(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.set(op.a, Value::unit());
    Flow::Next
}

pub fn undef(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.set(op.a, Value::Undef);
    Flow::Next
}

pub fn nop(_: &mut Machine<'_>, _: &Op) -> Flow {
    Flow::Next
}

pub fn drop_value(machine: &mut Machine<'_>, op: &Op) -> Flow {
    drop(machine.take(op.a));
    Flow::Next
}

pub fn poison(_: &mut Machine<'_>, _: &Op) -> Flow {
    panic!("reached poison instruction")
}

pub fn load_function(_: &mut Machine<'_>, _: &Op) -> Flow {
    todo!("LoadFunction: graph-level function references not yet supported at runtime")
}
