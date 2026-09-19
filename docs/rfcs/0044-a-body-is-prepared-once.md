# RFC-0044: A body is prepared once

Status: Accepted; the machine representation is superseded by RFC-0052 —
`Op`, `OpFn`, `Flow`, `Payload` and the payload table are gone, and a body
is chains of `Box<dyn Op>`, each operation holding its successor, entered at
its joints. Everything else here stands: prepare-once, the recognizers, the
slot selection, and what the stages produce.
Date: 2026-09-17
Extends: RFC-0007, RFC-0018, RFC-0020, RFC-0040

## Problem

The interpreter at `5f9cb4f` walked `MirBody.insts` directly, and recovered
every static fact at every execution:

- `MakeClosure` deep-cloned the closure's `MirBody`, once per closure
  creation — inside a loop, once per iteration.
- A closure call rebuilt the label map and allocated a frame.
- An extern call cloned the `InterpreterContext` to make an `AcvusRuntime`
  (an `Interner` `Arc`, three `Freeze`s and two `Arc`s, per call), looked the
  handler up in a hash map and cloned its `Arc`, and collected the arguments
  into a fresh `Vec<Value>`.
- Twelve instruction arms looked the operand's `Ty` up in
  `val_types: FxHashMap<ValueId, Ty>` before matching on it.
- A jump collected its arguments into a `Vec<Value>` before binding the
  block parameters.
- `execute_inst` was an `async fn` awaited per instruction, and every
  `run_loop` boxed a future.

Measured on the dot-product attention script
(`acvus-interpreter-test/benches/attention.rs`, bench profile: `opt-level =
3`, `lto = "fat"`, `codegen-units = 1`; AMD Ryzen 9 9950X; medians, first
repetition discarded; 2026-09-17, tree `5f9cb4f` + the bench):

| (n, d) | compile | setup | execute | Rust f64 | execute / Rust |
|--------|---------|-------|---------|----------|----------------|
| (2, 2) | 2326 µs | 7.1 µs | 37.4 µs | 0.04 µs | 934 |
| (64, 64) | 2322 µs | 12.6 µs | 4090 µs | 5.3 µs | 770 |
| (256, 128) | 2334 µs | 15.1 µs | 32538 µs | 53.5 µs | 608 |

The compile is constant in the input — the script is the same text at every
size — and the execute ratio falls with size because a run's fixed cost is
amortized, so the per-operation cost is what the larger sizes show.

## Ruling

A `MirBody` is not what the interpreter runs. At module load each body,
`main` and every closure, is prepared once into a `Code`: operations whose
every static fact — the operand slots, the type an arithmetic runs at, the
literal's word, the branch target, the extern instance's handler — is
resolved at that moment and never looked up again. A `Code` is shared by
`Arc`: a `MakeClosure` copies a pointer, a closure call runs the `Code` its
`FnValue` holds, and nothing about a body is rebuilt per call or per closure
creation.

The machine that runs a `Code` is synchronous, and it leaves its loop only
to return or to hand a future up. The one `async fn` is the driver around
it: it awaits the pending future, stores the result in the slot the
operation named, and re-enters at the operation after it. No operation is an
`async fn`, no `.await` sits between two synchronous operations, and no
future is boxed for a body that never suspends.

A call into a body runs to its result inside the calling operation when the
callee cannot suspend. Whether it can is the callee's task, settled by the
checker (RFC-0046) and carried as `Code::may_suspend`; a call site reads it
off the callee it has in hand — the `Prepared` a `Direct` call names, the
`FnValue` an `Indirect` call holds. At the extern boundary the same fact is
`Runtime::call_is_sync`, asked once per closure when `Fn0`/`Fn1`/`Fn2`/`Fn3`
is built and not again per element; `Runtime::call_now` is the call that
follows from a true answer.

**A failure at run time is a Rust `panic!`.** A division by zero, a
`MIN / -1`, an index out of range, a `Diverge` reached, a broken contract an
extern raises — each panics where it happens, with the message the same
operation writes in Rust, and leaves through the unwinder. There is no
failure channel: a handler's return is `Value` at every arity, the body's
value leaves through its return, and no operation, stage, consumer or
boundary tests whether the one below it failed. A host that wants to survive
a failing script catches: `acvus-cli` wraps its `block_on` in
`catch_unwind`, a test is `#[should_panic(expected = …)]`, and a spawned
task's panic comes back through the `JoinHandle` and is resumed on the
awaiting run's thread, so `Eval` sees it as its own. `Runtime` has no
`Error` associated type, no `trap`, and no `empty`/`is_empty`.

**A panic message names the operation, not the source position.** Carrying
the failing operation's span would need the span, or the index that reaches
it, in a place the unwinder can read — one store per operation in the loop
body. Priced by a probe one variable apart, restoring only that store: accum
int `while` 9.2 → 10.1 ns per iteration, mandelbrot 37.7 → 39.6–41.4. A span
in a message costs about a tenth of every loop iteration in the language, so
it is not kept. Spans live in a parallel array on the `Code`, read only to
name a closure body in an ICE.

**Whether a lazy pipeline can suspend is settled when the pipeline is
built**, once per stage. `Iter` holds `enum Stages { Sync(Box<dyn
SyncStage>), Async(Box<dyn AsyncStage>) }`; `SyncStage::next` is a plain
function and `AsyncStage::next` returns a future. The sources — `range`,
`as_iter`, `into_iter`, `Generate` — are `Sync`; an adaptor over a `Sync`
source whose closure answers `Fn1::is_sync` is `Sync`, and otherwise the
source is lifted once at construction rather than per element. A consumer
branches on the variant outside its loop and runs one of two loops. That
`Option<Value>` is Rust's, internal to the pipeline; where the *language*
sees an option the boundary builds the form of RFC-0039.

The effect in the type does not decide this. An extern's handler may be
asynchronous while its declared effect is `Pure`: `iter::sum`, `iter::fold`
and `iter::collect` are `async fn` at `effect = E`, and `E` instantiates to
`Pure` for a pure pipeline, so a body that calls `sum` is typed pure and
contains an asynchronous extern operation. Asynchrony is a mechanism —
awaiting the element closures — and purity is a semantics.

**Specialization at prepare time is the interpreter's instance selection**
(RFC-0020, RFC-0040): `InstKind::BinOp` at `Ty::Float` with `BinOp::Add`
prepares to the `f64` addition, at `Ty::Int(I64)` to the `i64` one;
`Take { Through }` of a word to a word load, of a `String` to a clone;
`Const` to a word constant with the bits inline; an extern call to its
handler's arity form, with the handler already in hand and the
`AcvusRuntime` held by the machine rather than made per call. What
`execute_inst` decided per execution by matching on a `Ty` the type checker
had fixed, the preparation decides once.

**`val_types` is not read at run time.** Every use the interpreter had for
it is a preparation-time choice: the literal's width, whether a `Take`
clones a `String`, whether a `MakeVariant`, a `TestVariant` or an
`UnwrapVariant` is an `Option`, a `Result` or a variant, the width a
`TestLiteral` compares at, whether a concatenated part or an indirect callee
is a reference, and which of three a path's payload step reads — an option's
it drops, because a `Some` is its payload's own value unless the payload
type is itself an option.

**Operations are written one per operation, and the preparation is one
exhaustive `match`** over `InstKind` and the operand types, so an
instruction kind added to the IR fails to compile until it is prepared. That
`match` is where an instance is chosen and where a superinstruction is
recognized.

### The recognizers

**A `while` whose head and body transfer no control is one operation.** The
preparation recognizes, in the linear `insts`, the shape the lowering gives
a `while` — an entering `Jump H` or a fall-through, `BlockLabel H`, the
head, `JumpIf { cond, then: B, else: X }`, `BlockLabel B`, the body,
`Jump H`, `BlockLabel X` — when `H` is named only by that entry and that
back edge, `B` only by the `JumpIf`, no jump from outside names a label
between them, and every operation in the head and the body is straight-line.
A nested region already prepared as one operation is straight-line, so
recognition runs inner-first. The loop then runs in one Rust `loop` without
returning to the machine's dispatch. An operation that raises inside gives
the error the span of the instruction it came from, not the loop's.

**An `if`/`else` whose arms transfer no control is one operation.** The
diamond is `JumpIf { cond, then: T, else: E }`, one arm's block directly
after the test, that arm's straight run, its `Jump J`, then either the other
arm's block and its own `Jump J` or nothing where the other edge already
names `J`, then `BlockLabel J` — when each arm label is named only by the
test, `J` only by the two edges the operation absorbs, and no jump from
outside names a label between them. `if` and `&&` put the `then` arm
directly after the test and `||` puts the `else` arm there, so the
recognizer takes whichever is there and assigns the sides afterwards. A
recognized diamond is straight-line, so the two recognizers interleave: a
`while` whose head or body holds a branch is one loop operation again.

**A run of extern calls and its deref is one operation.** `@keys.get(t)
.get(i)` is a call, a call reading the first's result, and a `Take` reading
the word through the second's; the two intermediates are references that
live in a register for exactly one instruction. A maximal run of synchronous
by-value extern calls in one block, where each call's result is used exactly
once and by the next call, optionally closed by one `Take` through the last
result, prepares as one operation. The handlers are the ones a lone call
reaches; what disappears is the register round trip and the dispatch between
them, and an intermediate is a Rust local moved into its consumer. A call
carrying an RFC-0007 order edge stays unfused, as does a window-form or
asynchronous call, and a result read a second time. The literals the
arguments read do not break the run: a one-word literal whose every reader
takes it out of a fixed register is lifted to an entry constant, and a
fusable call's argument is such a reader.

**An arithmetic chain is one operation, and a body that is one chain has no
frame.** A maximal run of arithmetic over registers of one inline numeric
type, whose intermediate results are each used once and by the next
operation of the run, prepares to one operation; a chain may end in one
comparison, so a loop test and a loop body are each one dispatch, and
intermediates never touch a register. A numeric literal every reader of
which is an arithmetic or comparison instruction becomes a register the
frame's entry fills, and emits no operation — so a chain leaf is always a
register, and a leaf is one pre-multiplied byte displacement read with no
bounds test and no kind test, on the preparation's proof. A chain is at most
three nodes: `Shape` names the eight binary trees of one to three nodes, and
a longer run is split from the root into pieces joined through registers.
The operator alphabet is `{Add, Mul}`.

**`Code::Expr`.** A closure body that is exactly `params -> return` or
`params -> one chain -> return` prepares to `Code::Expr`, which is evaluated
with no frame, no machine and no operation loop. Its operand space is the
arguments followed by the chain's constants, built on the Rust stack at the
call, so one evaluator serves a body chain and an expression chain. A body
needing more than `ExprChain::MAX_OPERANDS` operands keeps its frame, and a
body with captures is not one: a capture arrives as a reference and a chain
leaf reads an inline word.

### Slots are selected, not taken from the `ValueId`

The preparation runs one assignment over a body before it prepares a single
operation, and every slot lookup reads it. Liveness is a backward dataflow
over the linear `insts` with the jumps as its edges; two values may share a
slot unless one is live where the other is written. On that relation the
assignment does three things:

- A jump argument and the block parameter it feeds become one class where
  they do not interfere, so the move on that edge is a self-move and drops.
- A call site whose handler takes a window gets a contiguous run of `arity`
  registers, and an argument whose last use is that call is allocated
  *into* its window slot, so nothing is staged; an argument live past the
  call is copied into its slot.
- Everything else takes the lowest slot free over its live range.

Two values a `ValueId` cannot speak for keep one slot for the whole body: a
storage a place names directly, because a reference holds a pointer into its
register and the reference's own live range governs how long that pointer is
read, and because a write through a path reads the storage it writes while
`inst_info` reports it as a definition alone.

### A definition does not drop what it overwrites; an assignment does

There are two kinds of register write and the preparation tells them apart,
never the run.

A **definition** is the destination of an SSA instruction — an arithmetic, a
constant, a `Take`, a call's `dst` and its order, a chain's `dst`, a block
parameter a jump's move fills, a parameter or capture the entry writes into
a fresh frame. Its slot holds no live value: the selector gives two values
one slot only where neither is live where the other is written, and
`acvus_mir`'s drop insertion (RFC-0018) has already emitted the `Drop` where
a move-only value's life ended, so what the definition overwrites is a word
or an uninitialized slot. A definition is therefore one store, with a
`debug_assert!` against the two guarantors.

An **assignment** writes a storage that may hold a live value: an `Assign`
to a variable or through a reference or a path, a `FieldSet`, a `Commit`. It
drops what it overwrites. A storage is pinned to one slot for the whole
body, so one operation writes a register by slot and the rest write through
the place they name.

### A synchronous handler takes its arguments by value, up to three

A `Value` is a scalar pair, so an arity-2 handler is six scalars under the
`rust-call` ABI and every argument crosses in a register; the operation
reads each out of the register its own word names. Such a call site
constrains no run of registers, so it reports no window to the selector. A
window is what the four-or-more form, every asynchronous handler and every
spawn keep: the handler takes each argument out of the register it was lent
— an asynchronous one before it builds its future, because the future is
`'static` and cannot hold the lent slice, and a spawn's work owns its
arguments past the frame.

The cut is at three because the fourth argument does not fit: measured on
the emitted code, an arity-2 handler passes both arguments in registers,
while an arity-3 handler runs the SysV integer registers out at seven and
pushes the third argument's two words. Two pushes is still less than a
window — a contiguous run the selector must find, a move per live argument,
a slice borrow, and a take per argument — so three stays by value.

## What it costs

- A preparation pass per body at module load, linear in the body, run once.
  Its correctness is the machine's correctness, and it is tested at the
  contract the interpreter tests already state.
- The slot assignment is a second pass per body, linear except for the
  liveness fixed point and the pairwise interference test a coalescing
  candidate runs over two classes. It multiplied attention's setup by five,
  24 µs to 118 µs at (64, 64), against a per-iteration saving.
- A storage keeps its slot for the whole body, so a body that takes many
  addresses reuses few registers. The assignment reads `ValueId` liveness,
  not `analysis::loans`; reading loans is what would narrow this, and it is
  not built.
- A `while` or a diamond whose head, body or arm can suspend is not one
  operation. It prepares as the blocks it was. A branch the lowering emits
  in another order than the one above — an arm whose block does not sit
  directly after the test, a join some third jump also names — stops
  matching in the same way.
- A chain longer than three nodes is split into pieces that meet in
  registers.

## Rejected

- **A postfix stack of micro-operations for a chain.** The first attempt
  packed four-byte micro-operations two to a `u64` and ran them on a fixed
  stack of Rust locals indexed by `sp`. That evaluator is refused on
  sight: for two to four operators the arm is hand-written straight-line
  Rust with `let` locals, not a run-time stack. Measured afterwards on the
  same machine, it was also slower than the base at mandelbrot (44.2 ns
  against 35.1) and slower than the tree form at every bench.
- **One dispatch for a run of several chain pieces (`ChainSeq`).** One outer
  dispatch, k inner indirect calls through the payload. Both forms were
  built one variable apart on mandelbrot's four-node expressions: separate
  20.4 ns per iteration, one dispatch 23.4. The outer dispatch it saves is
  an indirect call the machine's loop predicts well; the inner call it adds
  is a function pointer loaded per piece.
- **An operator alphabet of `{Add, Sub, Mul}`.** 356 instances per type per
  entry-point set against 156, 1.65 MB of instance text against 0.90, and
  mandelbrot 20.7 ns against 20.6 — inside the spread. The marginal text
  buys no measured time.
- **Four-node trees.** Twenty-two shapes, and a chain's operators as data
  read per node. Three nodes is eight shapes, which is what makes an
  instance per operator slot affordable; mandelbrot pays two more dispatches
  for it and is 4.5 ns per iteration faster.
- **A chain leaf that is either an operand or a constant.** The leaf carried
  a tag and the arm branched on it, then bounds-checked the index: nine
  instructions and three branches to deliver one `movsd`, 45 instructions to
  fetch five doubles. Two repairs inside that shape were measured and both
  lost — constants as a boxed array the leaf selects (46.6 ns against 25.5,
  the box's pointer and length reaching the critical path), and hoisting the
  leaf array out of the per-leaf loads (26.1–26.6 against 25.3). The shape
  was the defect: a constant is a register, and then a leaf is one
  displacement.
- **Choosing the synchronous call path from the callee's effect.** Preparing
  a call site synchronous where `callee_ty`'s effect is `Pure`, with the
  callee's `may_suspend` as an ICE. The assert fired on the first run of the
  test suite, in `attention_shape`: `let dot = |k| -> as_iter(k).map(|x| ->
  *x).sum()` is typed pure and its body calls `iter::sum`, whose handler is
  an `async fn`. It is not a typeck defect — `sum` is pure, and it is
  asynchronous because it awaits the element closures. A static effect
  cannot answer a question about a mechanism, so the decision is the
  callee's own, where no ICE is expressible. RFC-0046 then made the
  mechanism an effect of its own axis.
- **An owned argument buffer across the handler boundary
  (`Args = SmallVec<[Value; 4]>` by value).** Passing an inline-capacity
  buffer by value raised attention (64, 64) execute from 728 µs to 1163 µs
  and the accum float `while` loop from 28.4 ns to 41.8 ns per iteration.
  `perf`, one variable apart: `malloc` + `cfree` barely moved (7.79 % →
  6.82 % of the sampled process) while the argument collection rose 8.87 % →
  13.02 %. The allocation was never the cost; moving an 80-byte buffer
  across a `dyn Fn` boundary and again through `into_iter` is dearer than a
  24-byte `Vec` plus one `malloc`/`free`. A two-element inline buffer
  recovers about a tenth of the gap.
- **Arguments lent across the handler boundary (`&mut [Value]` over a stage
  the caller owns).** With the caller staging into a local `SmallVec`,
  attention (64, 64) rose from 735 µs to 850 µs and the accum float `while`
  loop from 28.2 ns to 33.2 ns. The allocation did go — `malloc` + `cfree`
  8.49 % → 5.58 % — and the staging and the read through it rose further.
  Building and dropping a 64-byte stage in the calling frame costs more than
  the allocation it replaces. What both entries lacked is the window: the
  registers themselves are where the arguments already live.
- **A free list of register files, and a thread-local one for an extern's
  callback.** A call took a `Vec<Value>` from the list and gave it back, and
  `call_now` moved the whole list out of and back into a `RefCell` in a
  `thread_local!` around every closure call. `perf` put 74 % of `call_now`'s
  samples on the load and store beside that thread-local read; a stack array
  in its place took `map(|x| -> x) | sum` from 23.2 to 12.6 ns per element
  and `map(|x| -> x + 1) | sum` from 21.3 to 15.6 (2026-09-18). The array's
  width was the remaining cost — 8: 10.1, 12: 11.5, 16: 12.6, 32: 20.0 ns
  per element for `map(|x| -> x)` — and a frameless body removes it.
- **One contiguous register stack with the callee's frame as a window at the
  top**, in the form where the whole stack is one growable buffer: a
  reference holds a raw pointer to a register, and pushing a callee's frame
  can reallocate the buffer and move every frame below it. (RFC-0052 §6's
  window keeps the frame fixed while its body runs, and roots a new store
  rather than growing one that is borrowed.)
- **Registers as `u64`.** An extern reads a lent value through `&Value`, so
  a `u64` register file would materialize a `Value` at every lend, the most
  frequent act in the attention script.
- **Words packed as three `usize`.** Two slot indexes per word on a 64-bit
  target and one on wasm32 would make an operation's inline capacity
  platform-dependent.
- **Deciding ready-or-later per element (`Pull { Ready, Later }`).** Letting
  a stage answer without a future by returning an enum the consumer matches
  on measured `range | sum` at 26.8 ns per element, and 23.8 with `#[inline]`
  probes, against a base of 10.6 — 2.2× slower than what it was meant to
  beat. Two things were wrong at once: the decision belongs at construction,
  once per stage, not at every element; and the payload was
  `Result<Option<Value>, RuntimeError>` at 72 bytes, returned through memory
  per element. `Stages { Sync, Async }` replaces the first, and the panic
  rule the second.
- **A trap channel beside the value, in either shape** — handlers returning
  `Value` with a `Runtime::trap` side call, a `Cell<bool>` flag plus a cold
  boxed error in a thread-local, and one flag read per extern call.
  Measured against `a84267f`, every loop with an extern call in it rose —
  accum float `while` 17.6 → 21.0 ns per iteration, attention (64, 64) 414 →
  492 µs, attention (256, 128) 3206 → 3863 µs — while the loops without one
  did not move. Three probes one variable apart placed 0.7 ns of the 3.4 ns
  per call in the flag read and none in the handler. The
  `Result<Value, RuntimeError>` ABI it replaced is refused for the same
  reason: 72 bytes returned through memory per call. The thread-local is
  also refused outright on `wasm32`.
- **Patching the per-call clones in `execute_inst`.** Each is a symptom of
  the body being consulted at execution; a `Code` shared once makes them
  inexpressible.
- **Keeping `execute_inst` beside the machine during the change.** Two
  semantics under one test suite say nothing about either. `execute_inst`,
  `run_loop`, `Frame::jump*`, `build_label_map*` and the per-call clones
  they imply are removed, not kept beside the machine.

## Consequences

- **The panic rule is worth more than the channel it replaced.** Against
  `a84267f`, interleaved A/B, three repetitions, medians, `n = 1_000_000`:
  accum `range | sum` 10.9 → 5.1 ns/iteration, `map(|x| -> x) | sum` 67.8 →
  36.9, `map(|x| -> x + 1) | sum` 73.7 → 40.5, int `while` 10.2 → 9.4, float
  `while` 17.4 → 16.8, mandelbrot 40.3 → 37.7, attention (64, 64) 413.7 →
  396.4 µs, (256, 128) 3206 → 3113 µs. The extern call is where it lands:
  float `while` is one call per iteration, and the handler's 72-byte return
  went with the channel. The loops with no extern call moved too, because
  two things left the loop body at once — the arithmetic no longer builds a
  `Result` per operation, and the loop no longer reads a span per operation.
  The span probe above separates them: putting the span store back alone
  returns int `while` to 10.1 and mandelbrot to 39.6–41.4.
- **A definition that does not drop is worth 0.5 to 0.8 ns per register
  write.** Against `master` 050b3657, interleaved A/B, three repetitions,
  medians: mandelbrot 18.8 → 14.1 ns/iteration, accum int `while` 7.4 → 4.9,
  float `while` 10.9 → 7.7, extern `while` 10.8 → 7.6, branch `while` 21.7 →
  18.6, option `while` 28.2 → 24.6, attention (64, 64) 230.3 → 191.4 µs. The
  op listing counts the writes: the int `while` loop writes 3 registers per
  iteration and gains 2.5 ns, float `while` 4 and gains 3.2, mandelbrot's
  inner loop 9 and gains 4.7. What each write no longer pays is a load of
  the old kind byte, a compare, a branch, and the spill and call boundary
  they forced. 64 of the 65 register writes in the interpreter are
  definitions.
- **The diamond is what makes the loops recognizable.** RFC-0020 made every
  `&&`, every `||` and every multi-part pattern conjunction a branch diamond
  in the MIR, and the loop recognizer stops at a `JumpIf` that is not the
  loop's own exit test, so mandelbrot's three `while`s were no operation at
  all and collatz- and grade-shaped loops never had been. Against `master`
  3589d1b7, seven repetitions, medians: mandelbrot 27.0 → 16.2 ns/iteration,
  branch `while` 19.3 → 10.1, option `while` 25.1 → 14.4, collatz `while`
  23.1 → 12.6, grade `while` 27.4 → 16.2, and every case with no diamond
  flat to within 2 %. The gain is per dispatch removed: mandelbrot's
  innermost iteration ran fifteen operations and runs eight.
- **A dispatch removed from a hot loop is worth about half a nanosecond.**
  attention's 65 536 inner iterations lose three operations each and
  256×128 loses 86.7 µs: 0.44 ns per removed dispatch, roughly two cycles.
  The indirect call is well predicted and the out-of-order engine hides it
  behind the handler's work, so what a three-operation loop's nanoseconds
  measure is the register traffic and the arithmetic, which fusing does not
  remove.
- **A fused operation must be specialized on the run's shape.** The first
  form iterated a boxed argument list per call and held the intermediate in
  an `Option<Value>`: it ran attention 5 % *slower* than the base, with the
  same instruction count and the same branch count as the three separate
  operations, because a payload walked at run time costs what a dispatch
  costs. With the call count and the tail as const parameters and the
  arguments a fixed array of slots, the same benchmark runs 0.941. The shape
  is a prepared fact like every other one in this RFC.
- **A `Value` that is a scalar pair is what makes the option form free.**
  `Value` stopped being a Rust enum whose first word held two bytes — which
  is not one scalar, so the whole aggregate went through memory — and became
  `#[repr(C)] struct Value { kind: Kind, word: u64 }`. Against `28b6033`,
  three repetitions, medians, `n = 1_000_000`: accum `range | sum` 5.1 →
  1.7 ns/iteration, `map(|x| -> x) | sum` 36.6 → 23.0, `map(|x| -> x + 1) |
  sum` 40.4 → 20.8, float `while` 16.7 → 13.3, mandelbrot 37.4 → 35.2,
  attention (64, 64) 391.2 → 323.8 µs, (256, 128) 3018 → 2591 µs. The
  pipeline benches fell about twice as far as a count of `Value` moves
  predicts, because `Option<Value>` is a `ScalarPair` too — the niche its
  discriminant needs is a spare `Kind` — so `Some(v)` and `v` compile to the
  same code and every `Stage::next -> Option<Value>` stopped going through
  memory. The layout is `docs/runtime-value.md`.
- **The option in the value costs no allocation.** Against `master`, ten
  alternating repetitions, medians, `n = 1e6`: accum `option while` 90.2 →
  59.2 ns/iteration, every other case flat to within 4 %. Two controls
  decompose it — `acc = acc + f(i)` isolates one extern call at 8.4 ns over
  the bare loop, and `if g(i) { acc = acc + i; }` adds an unpredictable
  script-level branch at 7.5 ns more — leaving the option itself at 52.4 ns
  before and 20.6 after. What went is the allocation and the vtable lookup
  the box needed.
- The machine as it stands, and the numbers for it, are RFC-0052's.
