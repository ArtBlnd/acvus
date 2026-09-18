//! Control flow: the jumps and their parallel moves, the return, and the
//! instructions that produce no control at all.

use crate::code::{BasicBlock, Flow, LoopBody, Op, Payload, SlotMove};
use crate::machine::Machine;
use crate::ops::payload;
use crate::value::Value;

/// A jump's arguments move into the block's parameters: a move-only
/// argument leaves its register, so one owner remains. An extern call's
/// window is filled the same way.
#[inline]
pub fn move_all(machine: &mut Machine<'_>, moves: &[SlotMove]) {
    for m in moves {
        let value = machine.use_val(m.from);
        machine.define(m.to, value);
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

#[inline]
fn run_block(machine: &mut Machine<'_>, block: &BasicBlock) -> Flow {
    for op in block.iter() {
        match (op.f)(machine, op) {
            Flow::Next => {}
            Flow::Return => return Flow::Return,
            Flow::Jump(_) | Flow::Await(_) => panic!(
                "an operation inside a loop transferred control: the recognizer admits \
                 only operations that return Next or Return"
            ),
        }
    }
    Flow::Next
}

pub fn while_loop(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let Payload::Loop(body) = machine.payload(op) else {
        panic!(
            "a loop operation wants a Loop payload, found {}",
            crate::code::payload_name(machine.payload(op))
        )
    };
    let LoopBody {
        enter,
        head,
        cond_slot,
        into_body,
        body: block,
        back,
        exit,
    } = body;

    move_all(machine, enter);
    loop {
        match run_block(machine, head) {
            Flow::Next => {}
            stop => return stop,
        }
        if !machine.reg(*cond_slot).as_bool() {
            move_all(machine, exit);
            return Flow::Next;
        }

        move_all(machine, into_body);
        match run_block(machine, block) {
            Flow::Next => {}
            stop => return stop,
        }
        move_all(machine, back);
    }
}

pub fn ret(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let value = machine.take(op.a);
    machine.finish(value)
}

pub fn diverge(_: &mut Machine<'_>, _: &Op) -> Flow {
    panic!("a call typed `!` returned: its handler must panic")
}

pub fn merge(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.define(op.a, Value::unit());
    Flow::Next
}

pub fn undef(machine: &mut Machine<'_>, op: &Op) -> Flow {
    machine.define(op.a, Value::UNDEF);
    Flow::Next
}

pub fn nop(_: &mut Machine<'_>, _: &Op) -> Flow {
    Flow::Next
}

/// Release whatever the register still owns.
///
/// Taking the register is also the double-drop check. It is sound because
/// `acvus_mir`'s drop insertion emits no drop for a storage it saw
/// emptied, and a take of a flat option's payload is one such emptying:
/// the payload is the option's whole value (RFC-0022). A drop that
/// arrives at an empty register is therefore a defect in the lowering,
/// and this assert is where it surfaces.
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
