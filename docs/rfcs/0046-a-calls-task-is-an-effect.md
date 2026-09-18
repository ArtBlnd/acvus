# RFC-0046: a call's task is an effect — `Task::{Sync, Async, Heavy}`

Status: Accepted (owner and coordinator, 2026-09-18)
Date: 2026-09-18
Extends: RFC-0007 (order), RFC-0013 (effects), RFC-0044 (`Code.may_suspend`)

## Problem

Whether a call may suspend is decided in two places that do not talk:

- The interpreter decides it **statically per extern handler**
  (`extern_is_sync(id, instance)` = the handler is `SyncHandler`), and
  per closure body **at preparation** (`Code.may_suspend`, set when the
  body holds a `Direct` call, a `Spawn`/`Eval`, or an async extern call).
- An `Iter`'s stages decide it **at construction, at run time**
  (`Stages::{Sync, Async}`, RFC-0044 stage 6), from `Fn1::is_sync`.

So `next` is one `async fn` extern for every `Iter`, and `while let
Some(x) = next(&mut it)` — the language's iteration idiom — is never a
`Loop` operation and pays a boxed future per element even over a `Vec`
(measured: 11 dispatches + one async call per element; 13 of 13 `while
let` sites in the corpus). The type knows nothing the interpreter could
use, and the run time knows what the type should have said.

## Decision

Synchrony is an effect, as a lattice the owner fixed (2026-09-18):

```
Task::Sync  <  Task::Async  <  Task::Heavy        join = max
```

`Effect` gains `task: Task`. A body's task is the join of the tasks of
what it does; nothing else raises it — that is the soundness claim, and
the set is the one `prepare` marks today as `may_suspend`
(`prepare.rs:1099–1111`), moved to where the type is decided.

| what a body does | task |
|---|---|
| calls a plain `fn` extern, arithmetic, moves | `Sync` |
| calls an `async fn` extern; holds a `Spawn` or an `Eval`; calls a function value whose effect is `Async` | `Async` |
| calls an extern marked `#[extern_fn(heavy)]`, or a function value whose effect is `Heavy` | `Heavy` |

- **`Task` and purity are independent.** A `heavy` pure extern — a regex
  match, a hash of a large buffer — is `Pure` with `task = Heavy`: it
  commutes, it is outside RFC-0007's order chain, and it runs on the
  blocking thread pool (`Executor::spawn_blocking`) and is awaited. So
  `Heavy` is above `Async`: whatever is Heavy is also awaited, and the
  runtime may offload it as a unit.
- **An extern instance declares its task.** `fn` → `Sync`; `async fn` →
  `Async`; `heavy` → `Heavy`. An extern generic in its effect (`effect =
  E`: `next`, `map`, `sum`) takes its task from `E` — the macro's
  `async fn` glue is a mechanism, not an effect (stage 4's finding).
- **A closure's effect is its body's**, computed by the checker where
  the closure is made (the capture site): `MakeClosure` types the value
  `Fn(args) -> R / E` with `E.task` the body's join. A lambda parameter
  with an unknown body carries an effect variable, resolved by its uses
  as effect variables are today.
- **Demotion is the join.** A `Sync` function value passed where the
  parameter's effect is `Async` or `Heavy` joins into it; a value whose
  task is above what the parameter admits is a type error: `a function
  whose task is Async where Sync is required`.
- **`Iter<T, E, I>` carries the task in `E`**: the join over the source
  and every stage's closure. `next<T, E, I>` has instances chosen by the
  solver from `E.task`, as instances are chosen by bound today: `Sync` →
  a `fn` reading `Stages::Sync` (the other arm unreachable by type, and
  saying so); `Async`/`Heavy` → the `async fn`. The run-time `Stages`
  enum remains the representation; the type says which arm is live.
- **The interpreter reads the type.** `Code.may_suspend` becomes
  `E.task > Sync`, handed from the checker; `prepare` keeps its own
  computation only as a `debug_assert_eq!` against it — where they
  disagree the checker is wrong (the interpreter's is the mechanism, the
  checker's the claim). `extern_is_sync(id, instance)` is unchanged: the
  instance the solver chose is the sync one. A `Heavy` call is
  `spawn_blocking` + await; a `Heavy` body may later be offloaded whole,
  which is what the third rung is for. A `while let` over a `Sync`
  iterator is a straight head and a `Loop` operation.

## What it costs

- `Effect` grows a field; every `Effect::join`, the printer, the
  `EffectTerm` unifier and the RFC-0013 commutation table carry it. The
  task does not change what a call reads or writes, so commutation is
  untouched.
- Externs generic in `E` that today have one `async fn` body and need a
  sync instance (`next`, `fold`, `sum`, `find`, `collect`…) get a second
  body or a shared body with the `Stages` match; the macro learns to
  emit both from one declaration where the body is `it.next(rt).await`
  shaped — or acvus-ext writes them twice. Count them in the first
  brief.
- A program that passed a spawning lambda to `map` still passes (join);
  a program that passes one where a sync closure is required now fails
  at the checker instead of running through the async path — that is
  the point, and RFC-0007's order chain already refuses the shapes that
  would have been wrong.

## Rejected

- **A separate synchrony axis with `SyncFn`/`AsyncFn` types** (the first
  shape considered): two function types, a coercion between them, and
  every container type gaining a second variable. Folding into the
  effect gives the same statics with the machinery that exists (effect
  variables, bounds, instance selection by effect), because the only
  ways to suspend are already effects RFC-0007 orders.
- **A dual handler that tries sync and falls back to async at run time.**
  The loop recognizer needs a static answer; a head that might suspend
  cannot be a `Loop` operation, and a `Loop` that resumes mid-head is
  machinery for a fact the type could have stated.
- **`Pure ⇒ Sync` as a theorem without the field.** False by design: a
  `heavy` pure extern is offloaded and awaited. The interpreter would be
  unsound on a claim no one checks; the field makes it explicit and
  checked.
- **A single `can_async` bit.** It cannot tell an awaited IO call from
  offloaded pure work, and the runtime wants to treat the second as a
  unit; two rungs above `Sync` cost one enum where a bit was.

## Consequences

- `while let Some(x) = next(&mut it)` over a `Vec`, `Deque`, `range` or
  any pipeline of synchronous stages is one `Loop` operation with a
  synchronous `next`: from 11 dispatches + a boxed future to the
  head/body counts the other loops have.
- `Fn1::is_sync` (run-time) becomes an assertion of what the type said.
- The task, purity and contexts are three independent parts of one
  `Effect`; a printer shows `Pure/Heavy` for a regex stage and
  `Opaque/Async` for a fetch.
- `heavy` is the owner's marker for "worth another thread": the checker
  propagates `Heavy`, the interpreter offloads it, and a loop over it is
  asynchronous — by type, not by discovery.
- kovac inherits a static answer for every call site.

## What landed in the compiler

`Effect` carries `task: Task`; `at_most`, `join` and `meet` order it, the
free effect variable's ceiling is `Effect::TOP` (`Opaque`, `Heavy`), and the
printer leaves `Sync` out exactly as it leaves `Pure` out, so a type shows
`with Opaque/Async` and an `Iter` shows `Pure/Heavy`. The macro reads
`asyncness` and the new `heavy` marker and writes the task onto a declared
level; `effect = E` adds no task, and `heavy` on an `async fn` or on an
effect variable is a macro error. A closure's effect is its body's at the
capture site, which was already true of the effect variable `check_lambda`
builds - the task rides in it with no new machinery. Demotion is the join
that `EffectRelation::AtMost` already took; a value whose task exceeds a
fixed one is `MirErrorKind::TaskTooHigh`, carried out of the failing join as
`MismatchReason::TaskTooHigh` and out of the conversion decision as
`Unsettled::TaskTooHigh`. `ty::Instances` holds `InstanceSig { ty, admits }`,
and `solve` closes an instance decision the types left tied by taking the
tightest ceiling that admits the call's task - in the same phase as the
other least elements, because the task a call runs with is only complete
once the argument decisions have settled. `MirBody` carries `task` for the
main body and for every closure body, so the interpreter reads it instead of
recomputing it.

`Spawn` and `Eval` are not instructions the checker sees: `spawn_split`
builds them from IO `FunctionCall`s after typecheck. Their task therefore
comes from the callee's declaration, which is where the RFC's table already
put it.

The table was incomplete there, and the runtime half measured it: see
**A third source of suspension** below.

## What the second brief owes

Every consumer in `acvus-ext` that is `#[extern_fn(effect = E)] async fn`
is `Async` at run time only because its glue awaits: 19 of them (`all`,
`any`, `collect`, `contains`, `count`, `find`, `fold`, `join`, `last`,
`max`, `max_by_key`, `min`, `min_by_key`, `next`, `nth`, `position`,
`product`, `reduce`, `sum`). Each needs a second instance whose signature is
the same and whose `admits` is `Task::Sync`, with a plain `fn` body that
takes the `Stages::Sync` arm of `Iter::stages_mut` and calls
`SyncStage::next` without awaiting - the `drain!` macro already writes both
arms, so the sync body is the arm that exists. The interpreter's
`Code::may_suspend` becomes `MirBody::task > Task::Sync` with `prepare`'s own
computation kept as a `debug_assert_eq!`.


## What landed in the runtime

**The nineteen instances.** `#[extern_fn(..., sync = <fn>)]` names the
plain `fn` that runs an `async fn` declaration at `Task::Sync`. The macro
builds both handlers from the one signature and emits, per member type,
the `Task::Sync` instance first and the declared one after it, with
`admits: Task::Heavy` — a ceiling, "at most", which `InstanceSig` now
says. `ExternHandler` gained a `Heavy` arm, so the enum has one variant
per rung of `Task` and `is_sync` is false for two of them.

The sync bodies are the nineteen consumers written once more with
`drain_now!` in place of `drain!` and `call_now` in place of `call`. The
arm itself is written **once**, in `Stages::sync_mut`, whose `Async` arm
is the `unreachable!` that states the guarantee; `drain_now!` and
`Iter::next_now` both go through it, and no consumer matches on `Stages`
by hand.

**The shape.** `while let Some(x) = next(&mut it)` over a `Vec` is one
`Loop` operation: 12 dispatches and a boxed future per element became 8
and no future, and over `range | map` 11 became 7. Measured, medians of
three alternating runs, ns per element:

| case | n | before | after |
|---|---|---|---|
| `while let` over a `Vec` | 100 000 | 43.3 | 16.4 |
| `while let` over a `Vec` | 1 000 000 | 43.7 | 16.6 |
| `while let` over `range \| map` | 100 000 | 44.1 | 17.8 |
| `while let` over `range \| map` | 1 000 000 | 45.2 | 18.0 |

A synchronous consumer is also cheaper than the same loop inside a
generator frame: `map id | sum` 4.0 → 3.6, `map add | sum` 6.3 → 5.8,
`map cap | sum` 14.1 → 13.6. `range | sum`, attention and mandelbrot are
unchanged.

**`heavy`.** `ExternHandler::Heavy` is prepared as `call_extern_heavy`,
which takes the window (the work outlives the frame, so it owns its
arguments), hands the closure to `Executor::spawn_blocking` and awaits
the handle. `SequentialExecutor` has no pool: it defers the closure and
runs it inline at `eval`, so the await is real and the offload is not.
A `spawn` of a `heavy` extern is `spawn_extern_sync`, which was already
`spawn_blocking`.

**A third source of suspension.** `optimize::spawn_split` rewrites every
call whose effect is not Pure into a `Spawn` and an `Eval`, and an `Eval`
awaits. A plain `fn` extern declared `opaque` or `idempotent` therefore
suspends at its call site, which the table's first row denied. Two
corrections, both measured by the interpreter's assertion: the macro
gives a non-pure declaration `Task::Async`, and `spawn_split` raises the
task of any body it splits. An `async fn` generic in its effect is now a
macro error unless it names its `sync` twin, because its glue awaits for
every effect the variable takes.

**`Code::may_suspend` is the claim.** It is `MirBody::task > Task::Sync`
alone, and `prepare`'s own reading is kept as a `debug_assert!` that the
reading implies the claim — an operation that awaits under a body typed as
one that does not is the direction that would be unsound. Equality, which
this RFC first asked for, is refuted by demotion: in
`as_iter(&@items) | map(|x| -> fetch_by(*x)) | fold(@sum, |a, b| -> a + b)`
the fold's closure is typed `Async` because the parameter's effect is,
while its operations only add, and that body holds no call for
`suspends_at` to read. It is the only body that differs in the four
crates' suites (`io_in_iteration`,
`acvus-interpreter-test/tests/extern_fn.rs`).

**The hole the runtime half measured, and where it lived.** `find` and
`last` over a suspending pipeline took their asynchronous instance while
their call froze to `Pure` — not `Pure/Sync`: the whole effect was lost,
reissue and contexts with the task. The origin is not in those consumers
and not in the task. A name only one declaration owns never showed it,
because there the call type *is* the instantiated declaration and carries
its effect term; an **overloaded** name went through a
`Decision::Signature` opened on a call type holding a fresh effect variable
of its own. That variable was related to the instance's effect in one
direction only — the value's at most what the position allows, which is
RFC-0017's demotion — so nothing held it up and it froze at its lower
bound. `find`, `last` and
`contains` share their bare names with `str::find`, `vec::last` and
`str::contains`; the other sixteen consumers do not. That, and not the
`Fn1` parameter or the `Option<T>` return, is the whole of the split —
`nth` and `reduce` have the same shapes as `find` and are unaffected, and
a two-declaration extern with no iterator in it reproduces the loss on its
own.

The rule: **a call has no effect of its own.** A signature decision is
opened on a `CallShape` — the arguments the call passes and where its result
goes — which has no field for an effect and is not a function type: until
the decision settles there is no function type anywhere in it, so there is
none for a reader to find and none to record. At the settle the instance's
effect term makes one, that type is joined with the instance, and the callee
is recorded then and only then, as the instance's type. The effect the
enclosing body is raised by is the same term, raised by the decision,
because the checker has nothing to raise it by at the call. There is no
second term for one effect, so there is nothing left to freeze below the
instance and no join is asked to keep the two in step.

`TermStore::join` is therefore untouched: its function arm relates effects
in the one direction RFC-0017 gives it, and a `Sync` closure passed where
the parameter is `Async` still joins in. Making that join symmetric at a
decision's own `Position::Value` was tried and is wrong, because such a join
is not only "call meets instance" — a closure argument's function type
reaches it too, and the symmetry refused
`a_closure_writing_a_lent_context_is_rejected_at_the_call` and
`a_context_read_after_a_call_whose_closure_writes_it_is_fetched_again`.

Every consumer's call now carries the task of the instance the solver chose,
and the entry body's task is exactly the join over its calls and its
`Spawn`/`Eval`s; a closure body's is that join or the demotion above it.
Measured in `acvus-mir-test/tests/consumer_task.rs`, where
`a_calls_effect_is_the_term_of_its_instance` states the structure rather
than the outcome: two declarations of one name running different effects,
and the call carries the settled one's, never the other and never `Pure`.

**`Fn1::is_sync`.** Every `call_now` asserts it, which is the type's claim
checked against the run-time answer. It does not fire anywhere in the
four crates' suites.
