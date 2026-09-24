# Effects and order

What a call's effect records and how the order of effectful calls is kept.
An `Effect` is independent axes — reissue, commutation, task, and the
contexts it reads and writes (RFC-0025 rule 4) — each joined on its own. IO runs in
source order unless the author declares otherwise, and the IR carries that
order as a value.

## RFC-0007: IO runs in source order; `anyorder` declares a region where order is irrelevant

Status: Accepted

1. The intent behind an IO call is not inferred. Effectful calls execute in
   source order.
2. `anyorder { … }` declares order irrelevant for its whole region,
   including every iteration of a loop inside it. Its boundary is a join;
   outside it, source order resumes. It prescribes no execution strategy:
   parallel, concurrent and sequential each satisfy it.
3. A wrong declaration is a wrong program. An executor may verify one by
   running the block under an adversarial schedule and reporting a
   difference.
4. An ExternFn is effectful unless its author declares otherwise (RFC-0013).
5. An iterator is lazy and move-only; there is no memoized iterator, and
   cloning one is a type error.
6. **Order is a value.** The IR has a type `Order` no script can name. A
   call whose effect is not Pure takes an `Order` and yields one; a Pure call
   takes none. Sequential code is a chain. A local function or lambda whose
   effect is not Pure takes an `Order` first and yields one last; the
   lowering adds it.
7. **`merge(o..) -> o`** is the one join of orders, associative and
   commutative, meaning "after all of these". `anyorder` lowers to a fan-out
   from the block's entry order and one `merge` at its exit; a loop inside
   accumulates through a loop phi on `Order` and a `merge` per iteration.
8. A call split into spawn and evaluation: the spawn takes the `Order`, the
   evaluation yields it. An executor holds no `Order` values; it tracks which
   calls have not reached the merge that awaits them.
9. Code motion moves an instruction only between control-equivalent blocks,
   and never deeper in the loop nest than it started.
10. A call's issue is not hoisted above a branch: `Order` says "after", not
    "only if". The one exception is a commutative call whose block
    post-dominates the block of the call it follows; it joins that call's
    run (RFC-0013).

**Why.** Whether two calls may be reordered is the script author's intent,
which the ExternFn author cannot know; the ExternFn author declares the fact
(this does IO), the script author the intent (order does not matter here). A
sequential default with an explicit opening is sound, and one rule per region
does not grow with the square of the calls. Order as an SSA value lets every
dependency-driven pass respect it with no new concept.
**Rejected.**
- Inferring order from types, identities or consistency declarations — the
  intent is not in the types.
- Parallel by default — an unsound default.
- Bracket instructions opening and closing a region — crossing brackets have
  no inside, and every pass would have to preserve nesting; a `merge` is only
  a dependency.
- A linear or region-typed `Order` — only the block lowering fans out, so the
  value needs no linearity.
- Per-call or per-pair ordering relations — quadratic.

## RFC-0013: An effect is three declared axes — reissue, commutation, task

Status: Accepted

1. **Reissue** is a chain `Pure < Idempotent < Opaque`. A Pure call has no
   effect and stands nowhere in the order of a run. An Idempotent call keeps
   its order, and issuing it twice is the same as once. An Opaque call keeps
   its order and must not be issued twice. The default for an undeclared
   ExternFn is Opaque.
2. The chain decides whether a failed run may be re-run by the host: a run
   whose effect is Idempotent or Pure may; an Opaque one may not. Passes that
   reorder calls treat Idempotent as Opaque.
3. **Commutation** is `commutes: bool`, independent of the chain. Pure
   commutes; any other level may be declared commutative; the default is
   not. Two calls of commutative functions in either order are the same
   program.
4. **Task** is the axis of RFC-0046. It does not enter commutation.
5. A function's effect is the join of its calls, per axis: the higher level,
   the higher task, commutative only if both are.
6. Commutation is read only by the lowering of RFC-0007: a maximal run of
   commutative calls that are neighbours on the `Order` chain lowers as an
   `anyorder` block around the run. A Pure call neither joins nor breaks a
   run. A run admits a call only when no intervening load or store names a
   context in its summary (RFC-0025 rule 4). A commutative call is not reordered
   across a non-commutative one, and is not thereby re-issuable.
7. The declarations are the author's facts about the outside world; the
   compiler does not verify them. An executor may search for a
   counterexample by reordering a run and issuing an Idempotent call twice,
   comparing outcomes up to a renaming of fresh sources.

**Why.** Pure-but-not-re-issuable is not a real combination, so reissue is a
chain that cannot write it; its middle level is named for its cause. Each of
the four combinations of reissue and commutation is real (a counter increment
commutes and is not re-issuable; a put to one key is re-issuable and does not
commute), so commutation is a separate axis. The author of an ExternFn knows
what it is, and a wrong declaration lands where the knowledge is.
**Rejected.**
- Two independent bits, pure and re-issuable — admit the unreal
  combination.
- Inferring commutativity from a fresh identity in the return — one more
  derived fact a reader must know to read a signature.
- Per-pair or per-argument commutativity — a fact about arguments, which is
  what `anyorder` is for.

## RFC-0046: A call's task is an effect — `Sync < Async < Heavy`

Status: Accepted

1. `Effect` carries `task` on `Sync < Async < Heavy`, join = max. A body's
   task is the join of the tasks of what it does, and nothing else raises it.
2. A plain `fn` extern, arithmetic and moves are `Sync`. An `async fn`
   extern, a `Spawn` or `Eval`, or a call of a function value whose task is
   `Async` is `Async`. An extern marked `heavy`, or a function value whose
   task is `Heavy`, is `Heavy`.
3. A plain `fn` extern that is not Pure is `Async`, because the spawn split
   turns it into a `Spawn` and an awaiting `Eval` where no argument has a
   position (RFC-0079 rule 9).
4. Task and purity are independent: a `heavy` Pure extern commutes, stands
   outside the order chain, runs on a blocking pool and is awaited.
   Independent Pure `Heavy` calls therefore hoist without `commutative`.
5. An extern generic in its effect (`effect = E`) takes its task from `E`,
   and must name its synchronous twin (`sync = <fn>`) or is refused, unless
   `E: Suspends` (RFC-0011 rule 5): no site of it is `Sync`, so no site
   takes a twin.
6. A closure's effect is its body's, computed where the closure is made. A
   lambda parameter with an unknown body carries an effect variable resolved
   by its uses.
7. Demotion is the join: a function value of lower task passed where the
   parameter admits more joins into it; one above what the parameter admits
   is `TaskTooHigh`.
8. `Iter<T, E, I>` carries the task in `E`, the join over the source and
   every stage; `next` and the other consumers have instances chosen by
   `E.task`.
9. A call has no effect of its own: the callee's effect is the settled
   instance's. An instance decision the types leave tied closes on the
   tightest task ceiling that admits the call (RFC-0042).
10. The interpreter reads a body's task from the type; its own reading of
    the operations is checked to imply it.

**Why.** The ways to suspend are already effects RFC-0007 orders, so folding
synchrony into the effect gives it effect variables, bounds and instance
selection by effect with no new machinery, and a loop over a synchronous
iterator is a synchronous loop by type, not by discovery.
**Rejected.**
- Separate `SyncFn`/`AsyncFn` types — two function types, a coercion, and a
  second variable on every container type.
- A handler that tries sync and falls back to async at run time — the loop
  recognizer needs a static answer.
- `Pure ⇒ Sync` without the field — false: a `heavy` pure extern is
  offloaded and awaited.
- A single `can_async` bit — cannot tell an awaited IO call from offloaded
  pure work.
- A symmetric join at a decision's value position — a closure argument's
  function type reaches it too; effects relate in the one direction rule 7
  gives.
