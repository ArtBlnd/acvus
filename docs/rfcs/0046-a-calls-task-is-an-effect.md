# RFC-0046: a call's task is an effect — `Task::{Sync, Async, Heavy}`

Status: Accepted (owner and coordinator, 2026-09-18)
Date: 2026-09-18
Extends: RFC-0007 (order), RFC-0013 (effects), RFC-0044 (`Code.may_suspend`)

## Problem

Whether a call may suspend was decided in two places that do not talk. The
interpreter decided it statically per extern handler and per closure body at
preparation; an `Iter`'s stages decided it at construction, at run time,
from `Fn1::is_sync`.

So `next` was one `async fn` extern for every `Iter`, and `while let Some(x)
= next(&mut it)` — the language's iteration idiom — was never a `Loop`
operation and paid a boxed future per element even over a `Vec`: eleven
dispatches and one async call per element, at all thirteen `while let` sites
in the corpus. The type knew nothing the interpreter could use, and the run
time knew what the type should have said.

## Decision

Synchrony is an effect, on a lattice the owner fixed (2026-09-18):

```
Task::Sync  <  Task::Async  <  Task::Heavy        join = max
```

`Effect` carries `task: Task`. A body's task is the join of the tasks of
what it does, and nothing else raises it — that is the soundness claim.

| what a body does | task |
|---|---|
| calls a plain `fn` extern, arithmetic, moves | `Sync` |
| calls an `async fn` extern; holds a `Spawn` or an `Eval`; calls a function value whose effect is `Async` | `Async` |
| calls an extern marked `#[extern_fn(heavy)]`, or a function value whose effect is `Heavy` | `Heavy` |

A plain `fn` extern declared `opaque` or `idempotent` is `Async`, not
`Sync`: `optimize::spawn_split` rewrites every call that `runs_apart` into a
`Spawn` and an `Eval`, and an `Eval` awaits. The task a body is raised by
therefore includes what the split will do to it, and `spawn_split` raises
the task of any body it splits.

- **Task and purity are independent.** A `heavy` pure extern — a regex
  match, a hash of a large buffer — is `Pure` with `task = Heavy`: it
  commutes, it stands outside RFC-0007's order chain, and it runs on the
  blocking pool and is awaited. `Heavy` is above `Async` because whatever is
  heavy is also awaited, and the runtime may offload it as a unit.
- **An extern instance declares its task.** `fn` → `Sync`; `async fn` →
  `Async`; `heavy` → `Heavy`. An extern generic in its effect (`effect = E`:
  `next`, `map`, `sum`) takes its task from `E`, because the macro's `async
  fn` glue is a mechanism and not an effect. Such a declaration must name
  its synchronous twin — `#[extern_fn(..., sync = <fn>)]` — or the macro
  refuses it, since its glue awaits for every effect the variable takes.
- **A closure's effect is its body's**, computed where the closure is made:
  `MakeClosure` types the value `Fn(args) -> R / E` with `E.task` the body's
  join. A lambda parameter with an unknown body carries an effect variable,
  resolved by its uses.
- **Demotion is the join.** A `Sync` function value passed where the
  parameter's effect is `Async` or `Heavy` joins into it; a value whose task
  is above what the parameter admits is `MirErrorKind::TaskTooHigh`.
- **`Iter<T, E, I>` carries the task in `E`** — the join over the source and
  every stage's closure. `next<T, E, I>` has instances chosen from `E.task`:
  `Sync` reaches a plain `fn` that reads `Stages::Sync`, `Async`/`Heavy` the
  `async fn`. The run-time `Stages` enum stays the representation; the type
  says which arm is live, and `Stages::sync_mut` is the one place that arm
  is taken — its `Async` arm is the `unreachable!` that states the
  guarantee.
- **The interpreter reads the type.** `Code::may_suspend` is
  `MirBody::task > Task::Sync`, and `prepare`'s own reading is kept as a
  `debug_assert!` that the reading implies the claim — an operation that
  awaits inside a body typed as one that does not is the unsound direction.
  Equality does not hold, because of demotion: in `as_iter(&@items) |
  map(|x| -> fetch_by(*x)) | fold(@sum, |a, b| -> a + b)` the fold's closure
  is typed `Async` because the parameter's effect is, while its operations
  only add.

**A call has no effect of its own.** A signature decision (RFC-0043) is
opened on a `CallShape` — the arguments the call passes and where its result
goes — which has no field for an effect and is not a function type. Until
the decision settles there is no function type in it for a reader to find,
and none to record. At the settle, the instance's effect term makes one,
that type is joined with the instance, and the callee is recorded then, as
the instance's type. The effect the enclosing body is raised by is that same
term, so there is no second term for one effect and nothing left to freeze
below the instance.

## What it costs

- `Effect` carries one more field; `join`, `meet`, `at_most`, the printer,
  the `EffectTerm` unifier and RFC-0013's commutation table all carry it.
  The task does not change what a call reads or writes, so commutation is
  untouched. The free effect variable's ceiling is `Effect::TOP` (`Opaque`,
  `Heavy`), and the printer leaves `Sync` out exactly as it leaves `Pure`
  out, so a type shows `with Opaque/Async` and an `Iter` shows `Pure/Heavy`.
- Nineteen consumers in `acvus-ext` — `all`, `any`, `collect`, `contains`,
  `count`, `find`, `fold`, `join`, `last`, `max`, `max_by_key`, `min`,
  `min_by_key`, `next`, `nth`, `position`, `product`, `reduce`, `sum` —
  carry a second, synchronous instance: the same signature with `admits:
  Task::Sync`, written with `drain_now!` in place of `drain!` and `call_now`
  in place of `call`. The macro emits the `Task::Sync` instance first and
  the declared one after it with `admits: Task::Heavy`, a ceiling.
- A program that passes a spawning lambda to `map` still compiles, by the
  join; one that passes such a lambda where a synchronous closure is
  required now fails at the checker rather than running through the async
  path.
- `solve` closes an instance decision the types left tied by taking the
  tightest ceiling that admits the call's task, in the same phase as the
  other least elements, because the task a call runs with is complete only
  once the argument decisions have settled.

## Rejected

- **A separate synchrony axis with `SyncFn`/`AsyncFn` types.** Two function
  types, a coercion between them, and a second variable on every container
  type. Folding it into the effect gives the same statics with the machinery
  that already exists — effect variables, bounds, instance selection by
  effect — because the ways to suspend are already effects RFC-0007 orders.
- **A dual handler that tries sync and falls back to async at run time.**
  The loop recognizer needs a static answer: a head that might suspend
  cannot be a `Loop` operation, and a `Loop` that resumes mid-head is
  machinery for a fact the type could have stated.
- **`Pure ⇒ Sync` as a theorem without the field.** False by design: a
  `heavy` pure extern is offloaded and awaited. The interpreter would rest
  on a claim no one checks.
- **A single `can_async` bit.** It cannot tell an awaited IO call from
  offloaded pure work, and the runtime treats the second as a unit.
- **Symmetry in the join at a decision's `Position::Value`.** Tried, and
  wrong: such a join is not only "call meets instance" — a closure
  argument's function type reaches it too, and the symmetry refused
  `a_closure_writing_a_lent_context_is_rejected_at_the_call` and
  `a_context_read_after_a_call_whose_closure_writes_it_is_fetched_again`.
  `TermStore::join` relates effects in the one direction RFC-0017 gives it,
  and a `Sync` closure passed where the parameter is `Async` still joins in.

## Consequences

- `while let Some(x) = next(&mut it)` over a `Vec`, `Deque`, `range` or any
  pipeline of synchronous stages is one `Loop` operation with a synchronous
  `next`. Measured, medians of three alternating runs, ns per element:

  | case | n | before | after |
  |---|---|---|---|
  | `while let` over a `Vec` | 100 000 | 43.3 | 16.4 |
  | `while let` over a `Vec` | 1 000 000 | 43.7 | 16.6 |
  | `while let` over `range \| map` | 100 000 | 44.1 | 17.8 |
  | `while let` over `range \| map` | 1 000 000 | 45.2 | 18.0 |

  Twelve dispatches and a boxed future per element became eight and no
  future over a `Vec`, and eleven became seven over `range | map`. A
  synchronous consumer is also cheaper than the same loop inside a generator
  frame: `map id | sum` 4.0 → 3.6, `map add | sum` 6.3 → 5.8, `map cap |
  sum` 14.1 → 13.6. `range | sum`, attention and mandelbrot are unchanged.
- A `heavy` extern is the `CallHeavy` operation: it owns its arguments,
  because the work outlives the frame, hands the closure to
  `Executor::spawn_blocking`, and awaits the handle.
  `SequentialExecutor` has no pool — it defers the closure and runs it
  inline at `eval`, so the await is real and the offload is not.
- A pure call's `Spawn` takes no `Order` and its `Eval` yields none: the
  chain is built only from the `OrderEdge` the call carried (RFC-0007),
  which a pure call does not, so independent pure `Heavy` calls hoist
  without being declared `commutative`. Measured on
  `acvus-interpreter-test/benches/spawn.rs` (32 cores, `TokioExecutor`,
  medians of three runs): `straight-8 heavy/pure` 337.8 µs → 44.4 µs at
  33 µs of work per call, and 2889.6 µs → 353.3 µs at 330 µs; the listing
  goes from eight `CallHeavy` to eight `Spawn`s before the first `Eval`,
  with no `Merge` between them. `chain-8` has no parallelism to find and
  pays the pair's fixed cost instead of `CallHeavy`'s: 1.31× the sequential
  Rust twin → 1.20×.
- The task, purity and contexts are three independent parts of one `Effect`.
  `heavy` is the owner's marker for "worth another thread": the checker
  propagates it, the interpreter offloads it, and a loop over it is
  asynchronous — by type, not by discovery.
- `Fn1::is_sync` is asserted at every `call_now`: the type's claim, checked
  against the run-time answer.
- `MirBody` carries `task` for the main body and for every closure body, so
  the interpreter reads it instead of recomputing it.
- kovac inherits a static answer for every call site.
