# RFC-0003: Interpreter direction

Status: Accepted
Date: 2026-03-31
Supersedes: none

## Ruling

The interpreter reaches JIT-class performance without a JIT. Because the type
system fixes every type at compile time, values carry no runtime tag. On top
of that: fixed registers held as host-language locals so the host compiler
promotes them to physical registers; short instructions whose operands are
encoded in the opcode; bytecode fusion that matches a run of instructions as
one integer; and dispatch in fixed-size chunks. Each step depends on the one
before it.

The interpreter mediates every execution. Nothing is lowered to native code.

## Rationale

A JIT pays in complexity, executable memory, and warm-up to recover at runtime
the type and scheduling information that acvus never lost. Keeping that
information lets the same specializations happen ahead of time.

Mediating every execution is also the precondition for verification. Only an
interpreter that owns scheduling can run a script under an adversarial
schedule and observe whether a declared ordering freedom is honest. Native code
hands scheduling to the OS and the hardware, and that control is gone.

## Not built

- No JIT, for the reasons in RFC-0002 and because it would end the
  interpreter's control of scheduling.
- No runtime type tags on values. Erasure is justified by the enrichment in
  RFC-0001.

## Consequences

- The value representation at execution is a fixed-size slot; the MIR, not the
  value, knows the type.
- Instruction encoding must keep operands inside the opcode so that fusion can
  be an integer comparison.
- A verification mode of the interpreter may reorder any operations the program
  has declared order-irrelevant, and a difference in result is reported as a
  wrong declaration.

## Open questions

- Which fusion patterns to derive from the grammar, and how register pressure
  is handled when the fixed register set is exceeded.
