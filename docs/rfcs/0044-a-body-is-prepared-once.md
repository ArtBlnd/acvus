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

A call into a body runs to its result inside the calling operation when
the callee's prepared `Code` cannot suspend. `Code` carries that fact as
`may_suspend`, set while its operations are built: true if any of them is
an asynchronous extern, an `Eval`, or a call into another body. A call
site reads it off the callee it has in hand — the `Prepared` a `Direct`
call names, the `FnValue` an `Indirect` call holds — and either runs the
body on the caller's frames or hands a future up as before. At the extern
boundary the same fact is `Runtime::call_is_sync`, asked once per closure
when `Fn0`/`Fn1`/`Fn2`/`Fn3` is built and not again per element;
`Runtime::call_now` is the call that follows from a true answer.

A failure at run time is a Rust `panic!`. A division by zero, a `MIN /
-1`, an index out of range, a `Diverge` reached, a broken contract an
extern raises — each panics where
it happens, with the message the same operation writes in Rust, and
leaves through the unwinder. There is no failure channel at all: a
handler's return is `SyncFn(&R, &mut [Value]) -> Value` and
`AsyncFn(R, &mut [Value]) -> BoxFuture<'static, Value>`, `Flow::Return`
carries the body's value out, and no operation, stage, consumer or
boundary tests whether the one below it failed. A host that wants to
survive a failing script catches: `acvus-cli` wraps its `block_on` in
`catch_unwind` and prints the message, a test is
`#[should_panic(expected = …)]`, and a spawned task's panic comes back
through the `JoinHandle` and is resumed on the awaiting run's thread, so
`Eval` sees it as its own. `Runtime` has no `Error` associated type, no
`trap`, and no `empty`/`is_empty`.

A panic message names the operation, not the source position. Carrying
the failing operation's span would need the span, or the `pc` that
indexes it, in a place the unwinder can read — a store per operation in
the loop body, which is what `attach_span` was. Priced by a probe one
variable apart, restoring only that store: accum int `while` 9.2 to
10.1 ns per iteration and mandelbrot 37.7 to 39.6-41.4 ns. A span in a
message costs about a tenth of every loop iteration in the language, so
it is not kept.

Whether a lazy pipeline can suspend is settled when the pipeline is
built, once per stage. `Iter` holds `enum Stages { Sync(Box<dyn
SyncStage>), Async(Box<dyn AsyncStage>) }`; `SyncStage::next(&mut self,
&Rt) -> Option<Value>` is a plain function and `AsyncStage::next` returns
a future. The sources — `range`, `as_iter`, `into_iter`, `Generate` — are
`Sync`. An adaptor over a `Sync` source whose closure answers
`Fn1::is_sync` is `Sync`; otherwise it is `Async`, and a `Sync` source
feeding it is lifted once at construction rather than per element. A
consumer branches on the variant outside its loop and runs one of two
loops. `None` is the end of the source or a trap, and the consumer does
not need to tell them apart: the extern boundary above it reads the slot.

The effect in the type does not decide this. An extern's handler may be
asynchronous while its declared effect is `Pure`: `iter::sum`,
`iter::fold` and `iter::collect` are `async fn` at `effect = E`, and `E`
instantiates to `Pure` for a pure pipeline, so a body that calls `sum` is
typed pure and contains an asynchronous extern operation. Asynchrony is a
mechanism — awaiting the element closures — and purity is a semantics;
neither implies the other.

A synchronous call's registers live on the Rust stack of the call, as a
Rust function's locals do: `Registers::Stack([Value; 16])`, or a heap
array for a body wider than sixteen registers. `storage::ref_var` puts a
raw pointer to a register in a `Value::Ref`, so a frame may not move
while the body that owns it runs; a stack array does not. An
asynchronous body's registers are a `Vec` owned by its future, which
lives across its awaits. No frame is kept between calls and no
thread-local is read: an extern's callback enters a body exactly as a
call operation does. A `Machine` borrows its registers, its runtime and
its page rather than owning them, so entering a body costs no atomic.

An operation is `Op { f: OpFn, a: u32, b: u32, c: u32, d: u32, p: usize }`
— 32 bytes on a 64-bit target, 24 on wasm32 — with
`type OpFn = fn(&mut Machine, &Op) -> Flow`. `a..d` are register slots
or small immediates; `p` is an immediate word (an `f64`'s bits, an
integer) or an index into the `Code`'s payload table, where an operation
that needs more than the inline words keeps its rest: a `Vec<PathSeg>`,
an argument slot list, an `ExternHandler`, a `Ty` an extern boundary
needs, a `Label` list. The payload table is owned by the `Code`; an
operation reads it by index, so no pointer into it is held outside the
`Code`'s lifetime. Spans live in a parallel `Box<[Span]>` on the `Code`,
read only to name a closure body in an ICE; a `BasicBlock` inside a loop
carries none.

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

A `while` whose head and body transfer no control is one operation. The
preparation recognizes, in the linear `insts`, the shape the lowering
gives a `while` — an entering `Jump H` or a fall-through, `BlockLabel H`,
the head, `JumpIf { cond, then: B, else: X }`, `BlockLabel B`, the body,
`Jump H`, `BlockLabel X` — when `H` is named only by that entry and that
back edge, `B` only by the `JumpIf`, no jump from outside names a label
between them, and every operation in the head and the body is
straight-line: it returns `Flow::Next` or raises. A nested loop already
prepared as one operation is straight-line, so recognition runs
inner-first. The four edges become parallel-move lists the operation
holds directly, the head and the body become `BasicBlock`s, and the
instructions the operation covers leave `code.ops`; the jumps around them
are re-indexed. `control::while_loop` runs the whole loop in one Rust
`loop` without returning to the machine's dispatch loop. An operation
that raises inside gives the error the span of the instruction it came
from, not the loop's.

Operations are plain functions, one per operation, `fn(&mut Machine,
&Op) -> Flow`, with a macro for operand access and no trait. The
preparation is one exhaustive `match` over `InstKind` and the operand
types, so an instruction kind added to the IR fails to compile until it
is prepared; that `match` is where an instance is chosen, and where a
superinstruction is recognized in a later stage.

Slots are selected, not taken from the `ValueId`. The preparation runs
one assignment over a body before it prepares a single operation, and
every `slot(id)` reads it. Liveness is a backward dataflow over the
linear `insts` with the jumps as its edges; two values may share a slot
unless one is live where the other is written. On that relation the
assignment does three things. A jump argument and the block parameter it
feeds become one class when they do not interfere, so the parallel move
is a self-move and `order_moves` drops it. Every extern call site gets a
contiguous window of `arity` registers, and an argument whose last use is
that call is allocated *into* its window slot, so nothing is staged; an
argument that is live past the call is copied into its slot by a
`SlotMove` the call carries. Everything else takes the lowest slot free
over its live range. The extern ABI is a borrow of the window —
`SyncFn(&R, &mut [Value])`, `AsyncFn(R, &mut [Value])`, `Runtime::call_n`
the same — and the handler `mem::take`s each argument out of the register
it was lent, an asynchronous one before it builds its future, because the
future is `'static` and cannot hold the lent slice.

Two values a `ValueId` cannot speak for keep one slot for the whole body:
a storage a place names directly, because `storage::ref_var` builds a
pointer into its register and the reference's own live range governs how
long that pointer is read, and because a write through a path reads the
storage it writes while `inst_info` reports it as a definition alone. A
`Direct` or `Indirect` call keeps its argument `Vec`: its callee is a
body that owns its parameters, not a handler that borrows them.

Superinstructions (an `if`/`else` diamond as one operation) and a
synchronous call path for pure closures are stages after this one, each
an amendment to this RFC with the measurement that motivated it. They
presuppose the operation format and the shared `Code` this RFC
establishes; none of them exists without it.

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

After stage 3, the `while` as one operation (2026-09-18; interleaved A/B
against the stage-1 binaries, three repetitions, medians; all six `while`s
in the two benchmarks are recognized):

| bench | stage 1 | stage 3 |
|-------|---------|---------|
| attention (64, 64) execute | 742.8 µs | 721.2 µs |
| attention (256, 128) execute | 6406.7 µs | 5840.8 µs |
| accum int `while` | 16.9 ns/iteration | 13.0 ns/iteration |
| accum float `while` | 28.4 ns/iteration | 26.1 ns/iteration |
| accum `range \| sum` | 10.4 ns/iteration | 10.3 ns/iteration |

The stage was designed expecting attention to reach 600–680 µs on the
reading that dispatch was about a quarter of it. `perf`, one variable
apart on the execute-only mode at (256, 128), refutes that reading:
`Machine::run` is 1.28 % of the sampled process before the change and
`control::while_loop` 2.28 % after, while `ops::call::arg_values` is
25.4 % in both. The dispatch loop was never the cost of attention; the
argument `Vec` of every extern call is, which is what register selection
addresses. In accum, where no extern sits in the int loop, the four
removed dispatches — `nop`, `nop`, `jump_if`, `jump` — are 8.8 % of the
sampled process at stage 1 and gone at stage 3, and the iteration falls
23 %. What remains on top there is `control::move_all` at 33 %: the
parallel moves the loop still performs on its two edges are now the
per-iteration cost, and they are a register-selection problem too.

After stage 2b, register selection (2026-09-18; interleaved A/B against
stage-3 binaries built from the same tree, three repetitions, medians;
load average 2.3–3.9):

| bench | stage 3 | stage 2b |
|-------|---------|----------|
| attention (64, 64) execute | 719.7–729.2 µs | 408.0–418.1 µs |
| attention (256, 128) execute | 5746–6030 µs | 3181–3258 µs |
| attention (64, 64) setup | 23.3–24.9 µs | 114.1–118.3 µs |
| accum int `while` | 12.9–13.4 ns/iteration | 10.1–10.4 ns/iteration |
| accum float `while` | 25.4–26.3 ns/iteration | 15.9–16.1 ns/iteration |
| accum `range \| sum` | 10.4–10.7 ns/iteration | 10.6–10.9 ns/iteration |

Attention's main body is 31 registers where it was 197, with 24 argument
windows in it. `perf` on the execute-only mode at (64, 64), one variable
apart: `ops::call::arg_values` is 4.89 % of the sampled process before
and does not appear at all after, and the generated `array::get` handler
falls from 5.72 % to below the 1.5 % cut. In accum, `arg_values` goes
20.61 % → nothing and `control::move_all` 32.83 % → 26.27 % of a process
that is itself a quarter faster.

The stage was designed expecting attention (64, 64) at 540–620 µs and the
accum int `while` at 6–9 ns. Attention is faster than that band and accum
int is slower, and the same fact explains both. Every one of the six
loops keeps exactly one back-edge move, because the lowering computes the
next iteration's values in the *head* block, above the `JumpIf`: at the
definition of `acc + i` the parameter `acc` is still live, since the exit
edge leaves the loop below it and the block after reads `acc`. The two
inner loops of attention additionally keep two entering moves each, paid
once per outer iteration. Sinking a loop head's computation below its
test is a lowering change, not a register-selection one. Where the design
was underestimated is the window: of the six arguments the three extern
calls in an attention inner loop pass, four are allocated into their
window and cost nothing at all, and the two that are moved are moved for
one reason, that the value is live past the call.

The setup column is the assignment: five times the preparation cost at
(64, 64), 1.79 % of the sampled process, paid once per body at module
load against a per-iteration saving.

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
- A `Direct` or `Indirect` call whose callee can suspend still runs the
  callee's driver recursively and boxes one future per call.
- A body that contains any call into another body is `may_suspend`,
  because the callee is not known until the call runs. A closure that
  calls a pure closure is therefore never itself synchronous, and a
  `while` whose body calls a closure is not one operation: the loop
  recognizer needs a static answer and there is none. Resolving an
  `Indirect` callee to the `MakeClosure` that produced it would give one;
  it is not built.
- The slot assignment is a pass per body at module load, linear in the
  body except for the liveness fixed point and the pairwise interference
  test a coalescing candidate runs over two classes. It multiplies the
  attention benchmark's setup by five, 24 µs to 118 µs.
- A storage keeps its slot for the whole body, so a body that takes many
  addresses reuses few registers. The assignment reads `ValueId`
  liveness, not `analysis::loans`; reading loans is what would narrow
  this, and it is not built.
- A `while` whose head or body can suspend — an asynchronous extern, a
  call into another body, an `Eval` — is not one operation. It prepares
  as the separate operations it was before, and stays that way until the
  synchronous call path admits a closure call.
- An `if`/`else` inside a loop body is a control transfer, so the loop
  around it is not recognized either. The diamond is its own
  superinstruction, not part of this one.

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

- **Choosing the synchronous call path from the callee's effect.** Stage 4
  set out to prepare a call site synchronous when `callee_ty`'s effect is
  `Pure`, and to assert on the callee's `may_suspend` as an ICE. The
  assert fired on the first run of the test suite, in
  `attention_shape`: `let dot = |k| -> as_iter(k).map(|x| -> *x).sum()`
  is typed pure and its body calls `iter::sum`, whose handler is an
  `async fn`. It is not a typeck defect — `sum` is pure, and it is
  asynchronous because it awaits the element closures. A static effect
  cannot answer a question about a mechanism, so the decision moved to
  `Code::may_suspend`, read off the callee, where no ICE is expressible.

- **One contiguous register stack with the callee's frame as a window at
  the top.** A frame would then be `regs[base..base + frame_len]` and
  every register access one add. It is unsound: `storage::ref_var` puts a
  raw pointer to a register in a `Value::Ref`, and pushing a callee's
  frame can reallocate the buffer and move every frame below it. A stack
  array per call keeps a frame fixed while its body runs and leaves
  register access at exactly the cost it had.
- **A free list of register files, and a thread-local one for an
  extern's callback.** A call took a `Vec<Value>` from the list and gave
  it back, and `call_now` moved the whole list out of and back into a
  `RefCell` in a `thread_local!` around every closure call. `perf` put
  74 % of `call_now`'s samples on the load and store beside that
  thread-local read; a stack array in its place took `map(|x| -> x) |
  sum` from 23.2 to 12.6 ns per element and `map(|x| -> x + 1) | sum`
  from 21.3 to 15.6 (measured 2026-09-18, after `Value` became a scalar
  pair). The array's width is the remaining cost — 8: 10.1, 12: 11.5,
  16: 12.6, 32: 20.0 ns per element for `map(|x| -> x)` — and a
  frameless body (`Code::Expr`, stage 3) is what removes it.
- **An owned argument buffer across the handler boundary
  (`Args = SmallVec<[Value; 4]>` by value).** Stage 2a set out to delete
  the `Vec<Value>` that `arg_values` allocates for every extern call.
  Passing an inline-capacity buffer by value instead raised attention
  (64, 64) execute from 728 µs to 1163 µs and the accum float `while`
  loop from 28.4 ns to 41.8 ns per iteration — +60 % and +45 %, against
  an expected fall. `perf`, one variable apart: `malloc` + `cfree`
  barely moved (7.79 % → 6.82 % of the sampled process) while
  `arg_values` rose 8.87 % → 13.02 % and the generated handler closure
  1.75 % → 5.29 %. The allocation was never the cost; moving an 80-byte
  buffer across a `dyn Fn` boundary and again through `into_iter` is
  dearer than a 24-byte `Vec` plus one `malloc`/`free`. A four-element
  inline buffer is not the wrong size either: a two-element one
  (40 bytes) recovers about a tenth of the gap, and a `Vec` register
  file against a `SmallVec<[Value; 16]>` one measured 1157 µs against
  1163 µs — the frame is not where this sits. What it lacked is the
  window: a buffer of any ownership is built per call, and a window is
  chosen once at preparation and is already where the arguments live.

- **Arguments lent across the handler boundary (`&mut [Value]` over a
  stage the caller owns).** The reading that followed from the buffer
  measurement — that the cost is the transfer of ownership, not the
  allocation — is also refuted. With the handler ABI, `Runtime::call_n`
  and the macro's generated closures all taking `&mut [Value]`, and the
  caller staging into a local `SmallVec<[Value; 4]>`, attention (64, 64)
  execute rose from 735 µs to 850 µs and the accum float `while` loop
  from 28.2 ns to 33.2 ns per iteration (interleaved A/B, three
  repetitions). The allocation did go: `malloc` + `cfree` fell 8.49 % →
  5.58 %. It is `arg_values` (7.29 % → 10.48 %) and
  `ops::storage::take_through::<false>` (4.37 % → 6.51 %) that rose
  further, so building and dropping the stage in the calling frame costs
  more than the allocation it replaces. Two designs one variable apart
  from each other both measure worse than the `Vec`: a 64-byte stage of
  `Value`s built and dropped per call is the cost, not its ownership.
  What it lacked is the window: lending a stage still builds the stage,
  and stage 2b lends the registers themselves, which measures a 43 % fall
  at attention (64, 64) with the same `&mut [Value]` ABI this entry
  rejected.

After stage 4, the synchronous call path (2026-09-18; interleaved A/B
against stage-2b binaries built from the same tree with the two new accum
cases in them, three repetitions, medians; load average 1.8-2.5):

| bench | stage 2b | stage 4 |
|-------|----------|---------|
| accum `map(\|x\| -> x) \| sum` | 93.9-99.6 ns/iteration | 66.1-66.9 ns/iteration |
| accum `map(\|x\| -> x + 1) \| sum` | 86.1-87.1 ns/iteration | 71.4-72.0 ns/iteration |
| accum int `while` | 10.0-10.3 ns/iteration | 10.1-10.3 ns/iteration |
| accum float `while` | 15.9-16.1 ns/iteration | 15.9-16.2 ns/iteration |
| accum `range \| sum` | 10.5-10.6 ns/iteration | 10.8 ns/iteration |
| attention (64, 64) execute | 404.7-414.5 µs | 412.8-417.0 µs |
| attention (256, 128) execute | 3133-3240 µs | 3092-3231 µs |
| attention (64, 64) setup | 113.9-121.5 µs | 116.1-121.2 µs |

The stage was designed expecting `map | sum` at 22-32 ns, on the reading
that what a closure call costs is the boxed future, the callee's frame
allocation and the driver's poll. `perf`, one variable apart on the accum
benchmark: before, `drop_glue::<Machine>` is 16.17 % of the sampled
process, `machine::fn_value_call` 11.20 % and `machine::drive` 8.74 %;
after, all three are gone and `machine::enter` (14.68 %) and
`Runtime::call_now` (7.73 %) stand where they were, in a process a
quarter faster. What the band missed is the stage: `Map::next` boxes a
future per element whether or not the closure it calls suspends, and it
rises from 13.78 % to 19.04 % to become the largest single item. The
pipeline's own dynamic dispatch, not the call, is what remains.

A Rust-only microbench in `acvus-ext` (`benches/iter_cost.rs`) prices
that dispatch against the value's tag, by running one `Iter` pipeline
over two runtimes one variable apart — a value that is the bare word, and
a value that is a tag plus a word with a box for what does not fit, the
shape `acvus-interpreter`'s `Value` has:

| case | ns/element |
|------|------------|
| a Rust iterator over `i64` | 0.18-0.19 |
| `Iter<Erased<Word, i64>>` | 6.16-6.27 |
| `Iter<Erased<TaggedWord, i64>>` | 7.10-7.15 |

The stage expected the opposite split, dynamic dispatch under 1 ns and
the tag 8-9 ns of the 10.4. The dynamic stage and its boxed future are
6.2 ns per element and the tag is 0.9 ns on top of it. The value
representation is not where the pipeline's cost is; the boxed future per
element per stage is.

- **Deciding ready-or-later per element (`Pull { Ready(Option<Value>),
  Later(BoxFuture<..>) }`).** Stage 5 set out to let a stage answer
  without a future by returning an enum the consumer matches on. Measured,
  `range | sum` went from 10.6 ns to 26.8 ns and then 23.8 ns per element
  with `#[inline]` probes — 2.2x slower than the base it was meant to
  beat. Two things were wrong at once. The decision belongs at
  construction, once per stage, not at every element: a stage that can
  never suspend should be a plain function, not an enum a caller
  discriminates a million times. And the payload was
  `Result<Option<Value>, RuntimeError>` at 72 bytes, returned through
  memory per element. The first is replaced by `Stages { Sync, Async }`
  above; the second goes with the failure channel itself.

- **A trap channel beside the value, in either shape.** Stage 6 built
  one: handlers returning `Value` with a `Runtime::trap` side call, a
  `Cell<bool>` flag plus a cold boxed `RuntimeError` in a thread-local,
  and one flag read per extern call at `call_extern_sync`. The pipeline
  half of that stage was right and stands; the channel was not. Measured
  against `a84267f`, every loop with an extern call in it rose — accum
  float `while` 17.6 to 21.0 ns per iteration, attention (64, 64) 414 to
  492 us, attention (256, 128) 3206 to 3863 us — while the loops without
  one did not move. Three probes one variable apart placed 0.7 ns of the
  3.4 ns per call in the flag read and none of it in the handler, and
  left 2.7 ns unattributed. The `Result<Value, RuntimeError>` ABI it
  replaced is refused for the same reason it was: 72 bytes returned
  through memory per call. The thread-local is also refused outright on
  `wasm32`. Panicking removes both halves, and the same benchmarks fall
  below the base they were measured against.

After the panic (2026-09-18; interleaved A/B against binaries built from
`a84267f` with nothing changed, three repetitions, medians,
`n = 1_000_000`):

| bench | base `a84267f` | stage 6 | panic |
|-------|----------------|---------|-------|
| accum `range \| sum` | 10.9 ns/iteration | 6.7 | **5.1** |
| accum `map(\|x\| -> x) \| sum` | 67.8 ns/iteration | 41.1 | **36.9** |
| accum `map(\|x\| -> x + 1) \| sum` | 73.7 ns/iteration | 44.7 | **40.5** |
| accum int `while` | 10.2 ns/iteration | 10.4 | **9.4** |
| accum float `while` | 17.4 ns/iteration | 21.0 | **16.8** |
| mandelbrot 200x100x200 | 40.3 ns/iteration | 40.3 | **37.7** |
| `iter_cost` `Words` | 6.22 ns/element | 1.30 | **1.30** |
| `iter_cost` `Tags` | 7.22 ns/element | 2.12 | **2.04** |
| attention (64, 64) execute | 413.7 us | 491.5 | **396.4** |
| attention (256, 128) execute | 3206 us | 3863 | **3113** |

The extern call is where stage 6's rise was and where it goes: accum
float `while` is one `to_float` call per iteration, 21.0 back to 16.8,
which is below the base's 17.4 because the handler's 72-byte return went
with the channel. attention is `n*d` synchronous `get` calls in its inner
loops and follows, 492 to 396 us against a base of 414. `call_extern_sync`
is 172 disassembly lines against the base's 195, and its hot path is the
indirect handler call, the 16-byte store into the destination register,
and `ret` — no thread-relative load, no branch on a returned tag.

The loops with no extern call in them moved too, which the band did not
expect: int `while` 10.2 to 9.4 and mandelbrot 40.3 to 37.7. Two things
left the loop body at once — `ops::arith::binary` no longer builds a
`Result<Value, RuntimeError>` per operation, and `run_block` no longer
reads a span per operation. The span probe above separates them: putting
the span store back alone returns int `while` to 10.1 and mandelbrot to
39.6-41.4, so the span read is most of it and the `Result` is the rest.

After stage 2a, the value as a scalar pair (2026-09-18; interleaved A/B
against binaries built from `28b6033` with nothing changed, three
repetitions, medians, `n = 1_000_000`). `Value` stopped being a Rust enum
whose first word holds two bytes — the discriminant at offset 0 and the
`Tag` at offset 1, which is not one scalar, so the whole aggregate went
through memory — and became `#[repr(C)] struct Value { kind: Kind, word:
u64 }` with `Kind` merging the discriminant and the `Tag`. The layout is
`docs/runtime-value.md`.

| bench | base `28b6033` | stage 2a |
|-------|----------------|----------|
| accum `range \| sum` | 5.1 ns/iteration | **1.7** |
| accum `map(\|x\| -> x) \| sum` | 36.6 ns/iteration | **23.0** |
| accum `map(\|x\| -> x + 1) \| sum` | 40.4 ns/iteration | **20.8** |
| accum int `while` | 9.1 ns/iteration | **8.7** |
| accum float `while` | 16.7 ns/iteration | **13.3** |
| mandelbrot 200x100x200 | 37.4 ns/iteration | **35.2** |
| `iter_cost` `iter<word>` | 1.29 ns/element | **1.30** |
| `iter_cost` `iter<tagged>` | 1.97 ns/element | **1.66** |
| attention (64, 64) execute | 391.2 us | **323.8** |
| attention (256, 128) execute | 3018 us | **2591** |

The pipeline benches fell about twice as far as a count of `Value` moves
predicts, and the probe names why: `Option<Value>` is a `ScalarPair` too,
because the niche the option's discriminant needs is a spare `Kind`. In
the base, `probe_some_i64` is `mov %rdi,%rax; movw $0x302,(%rdi); mov
%rdx,0x8(%rdi); ret` — an sret buffer; after, the linker folds it onto
`probe_erase_i64`, `mov %rsi,%rdx; mov $0x7,%al; ret`, because `Some(v)`
and `v` compile to the same code. Every `Stage::next -> Option<Value>`,
once per element per stage, stopped going through memory.

`call_extern_sync`'s hot path is now the indirect handler call with no
sret pointer set up, the returned kind and word taken out of `rax` and
`rdx`, and two stores into the destination register — against the base's
sret buffer and its 1 + 4 + 4 + 8 reassembly, which the stage-6 and panic
reports both measured and neither could remove.

mandelbrot is the one bench that fell less than the stage expected
(35.2 against a 30-34 band). It has no extern call and no `Option<Value>`,
so its whole gain is in the arithmetic operations, and the disassembly of
`mul_f64` shows both halves of it: each operand's kind test went from two
branches (`test %al,%al` for `Empty`, `cmp $0x2,%al` for `Small`) to one
(`cmpb $0x0`), and the destination store went from `movw` plus `mov` to
`movb` plus `movsd`. Against that, LLVM now spills the product to the
stack across the destination's drop check and reloads it, where the base
kept it in a callee-saved GPR because the 16-byte store already needed it
there. That spill is a register-allocation consequence, not a rule this
stage could state differently, and it is what the remaining 1-2 ns per
iteration is.

- **Words packed as three `usize`.** Two slot indexes per word on a
  64-bit target and one on wasm32 would make an operation's inline
  capacity platform-dependent; four `u32` slots and one word are the
  same shape everywhere.
