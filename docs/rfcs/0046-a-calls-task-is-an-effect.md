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
