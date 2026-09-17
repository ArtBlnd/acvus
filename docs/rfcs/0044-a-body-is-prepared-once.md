# RFC-0044: A body is prepared once

Status: Accepted
Date: 2026-09-17
Extends: RFC-0007, RFC-0018, RFC-0020, RFC-0040

## Ruling

A `MirBody` is not what the interpreter runs. At module load each body,
`main` and every closure, is prepared once into a `Code`: a linear array
of fixed-size operations whose every static fact — the operand slots,
the type an arithmetic runs at, the literal's word, the label's target
index, the extern instance's handler — is resolved at that moment and
never looked up again. A `Code` is shared by `Arc`: a `MakeClosure`
copies a pointer, a closure call runs the `Code` its `FnValue` holds,
and nothing about a body is rebuilt per call or per closure creation.

The machine that runs a `Code` is synchronous. Its loop is

```rust
loop {
    match (code.ops[pc].f)(&mut machine, &code.ops[pc]) {
        Flow::Next => pc += 1,
        Flow::Jump(target) => pc = target,
        Flow::Return => break,
        Flow::Await(pending) => return Step::Await(pc, pending),
    }
}
```

and it leaves the loop only to return or to hand a future up. The one
`async fn` is the driver around it: it awaits the pending future, stores
the result in the slot the operation named, and re-enters the loop at
the next operation. An operation that is synchronous costs one indirect
call; no operation is an `async fn`, no `.await` sits between two
synchronous operations, and no future is boxed for a body that never
suspends.

An operation is `Op { f: OpFn, a: u32, b: u32, c: u32, d: u32, p: usize }`
— 32 bytes on a 64-bit target, 24 on wasm32 — with
`type OpFn = fn(&mut Machine, &Op) -> Flow`. `a..d` are register slots
or small immediates; `p` is an immediate word (an `f64`'s bits, an
integer) or an index into the `Code`'s payload table, where an operation
that needs more than the inline words keeps its rest: a `Vec<PathSeg>`,
an argument slot list, an `ExternHandler`, a `Ty` an extern boundary
needs, a `Label` list. The payload table is owned by the `Code`; an
operation reads it by index, so no pointer into it is held outside the
`Code`'s lifetime. Spans live in a parallel `Box<[Span]>` read only when
an error is raised.

Specialization at prepare time is the interpreter's instance selection
(RFC-0020, RFC-0040): the `InstKind::BinOp` at `Ty::Float` with
`BinOp::Add` prepares to the operation `add_f64`, at `Ty::Int(I64)` to
`add_i64`; `Take { Through }` of a word to `load_word_through`, of a
`String` to `clone_string_through`; `Const` to `const_word` with the
bits in `p`; `FunctionCall { Extern }` to `call_extern_sync` or
`call_extern_async` with the handler already in the payload and the
`AcvusRuntime` held by the machine, not made per call. What
`execute_inst` decided per execution by matching on a `Ty` the type
checker had fixed, the preparation decides once.

Three consequences the format fixes:

- **A jump is a parallel move.** Its arguments and the block's parameters
  may overlap (`i = i + 1` at a loop head reads the slot it writes). The
  preparation orders the moves so every source is read before it is
  overwritten and breaks a cycle with one scratch slot appended to the
  frame; the ordered pairs are the jump's payload, and nothing is
  collected at run time.
- **A pending future is `'static`.** An asynchronous extern's handler is
  an `Arc` and its arguments are moved; `Eval` clones the `Arc<dyn
  Executor>` into the future it hands up. `Spawn` is synchronous: it
  makes the handle and continues.
- **`val_types` is not read at run time.** Every use the interpreter had
  for it is a preparation-time choice: the literal's width, whether a
  `Take` clones a `String`, whether a `MakeVariant` is an `Option`, a
  `Result` or a variant, the width a `TestLiteral` compares at, whether
  a concatenated part or an indirect callee is a reference.

Operations are plain functions, one per operation, `fn(&mut Machine,
&Op) -> Flow`, with a macro for operand access and no trait. The
preparation is one exhaustive `match` over `InstKind` and the operand
types, so an instruction kind added to the IR fails to compile until it
is prepared; that `match` is where an instance is chosen, and where a
superinstruction is recognized in a later stage.

Slots in this RFC are the body's `ValueId`s as they stand: the frame is
one `Value` per `ValueId`, sized by the body's `val_factory`, as today.
Register selection, superinstructions (a single-exit loop or an
`if`/`else` diamond as one operation), and a synchronous call path for
pure closures are stages after this one, each an amendment to this RFC
with the measurement that motivated it. They presuppose the operation
format and the shared `Code` this RFC establishes; none of them exists
without it.

`execute_inst`, `run_loop`, `Frame::jump*`, `build_label_map*` and the
per-call clones they imply are removed, not kept beside the machine.
The whole test suite runs on the machine; the two loops never coexist.

## Problem

The interpreter at `5f9cb4f` walks `MirBody.insts` directly, and every
static fact is recovered at every execution:

- `MakeClosure` deep-clones the closure's `MirBody`
  (`Arc::new(closure_body.clone())`, interpreter.rs:642), once per
  closure creation — inside a loop, once per iteration.
- A closure call rebuilds the label map (`build_label_map_from_insts`,
  interpreter.rs:954) and allocates a frame.
- An extern call clones the `InterpreterContext` to make an
  `AcvusRuntime` (`ctx.shared.runtime()`, interpreter.rs:794): an
  `Interner` `Arc`, three `Freeze`s and two `Arc`s, per call; looks the
  handler up in a hash map and clones its `Arc`; collects the arguments
  into a fresh `Vec<Value>`.
- Twelve instruction arms look the operand's `Ty` up in
  `val_types: FxHashMap<ValueId, Ty>` before matching on it.
- A jump collects its arguments into a `Vec<Value>` before binding the
  block parameters (`bind_block_params`, interpreter.rs:238).
- `execute_inst` is an `async fn` awaited per instruction, and every
  `run_loop` boxes a future.

Measured before this RFC on the dot-product attention script
(`acvus-interpreter-test/benches/attention.rs`, bench profile: `opt-level
= 3`, `lto = "fat"`, `codegen-units = 1`; AMD Ryzen 9 9950X; medians,
first repetition discarded; 2026-09-17, tree `5f9cb4f` + the bench):

| (n, d) | compile | setup | execute | Rust f64 | execute / Rust |
|--------|---------|-------|---------|----------|----------------|
| (2, 2) | 2326 µs | 7.1 µs | 37.4 µs | 0.04 µs | 934 |
| (64, 64) | 2322 µs | 12.6 µs | 4090 µs | 5.3 µs | 770 |
| (256, 128) | 2334 µs | 15.1 µs | 32538 µs | 53.5 µs | 608 |

After stage 1, the same bench on the same machine (2026-09-17; the
machine runs `Code`, one operation per instruction, slots still the
body's `ValueId`s):

| (n, d) | compile | setup | execute | Rust f64 | execute / Rust |
|--------|---------|-------|---------|----------|----------------|
| (2, 2) | 2343 µs | 20.3 µs | 10.9 µs | 0.04 µs | 272 |
| (64, 64) | 2318 µs | 16.1 µs | 753 µs | 5.2 µs | 144 |
| (256, 128) | 2346 µs | 18.3 µs | 6219 µs | 53.6 µs | 116 |

The baseline row was re-measured beside it on the day: 38.0 µs, 4079 µs
and 32022 µs, so the execute column falls by 3.5×, 5.4× and 5.1×. The
setup column carries the preparation of every body and rose by 3 µs at
(64, 64).

The compile is constant in the input: the script is the same text at
every size. The execute ratio falls with size because the fixed cost of
a run (frames, closures made once) is amortized; the per-operation cost
is what the larger sizes show.

## Cost

- A preparation pass per body at module load: linear in the body, run
  once. Its correctness is the machine's correctness; it is tested at
  the contract the interpreter tests already state (the whole suite),
  not per operation.
- An operation's operands are limited to four slots and one word
  inline; the rest goes through one index into the payload table.
- A `Direct` or `Indirect` call in this stage runs the callee's driver
  recursively, which boxes one future per call; the synchronous path for
  pure bodies is a later stage.

## Rejected

- **Patching the per-call clones in `execute_inst`.** Each is a symptom
  of the body being consulted at execution; a `Code` shared once makes
  them inexpressible, and a patch would leave the shape that produced
  them.
- **Registers as `u64` in this stage.** An extern reads a lent value
  through `&Value` (`Cross::deref`, runtime.rs:184); a `u64` register
  file would materialize a `Value` at every lend, the most frequent act
  in the attention script. Revisited after the stages that follow, if
  the measurement then names register width.
- **Keeping `execute_inst` beside the machine during the change.** Two
  semantics under one test suite say nothing about either.
- **Words packed as three `usize`.** Two slot indexes per word on a
  64-bit target and one on wasm32 would make an operation's inline
  capacity platform-dependent; four `u32` slots and one word are the
  same shape everywhere.
