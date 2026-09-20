# RFC-0065: An iterator is the types its elements pass through

Status: Rejected (2026-09-20) — built and measured, then withdrawn

## Why it is rejected

The design was built whole on a worktree (the uniform-per-length payload of
§3, adaptors at eight input lengths and consumers at nine, 35 shared
signatures with 292 registry names, `iter.rs` 622 lines and `iterator.rs`
2613) and passed every gate but `asm_probe`. Two facts came out of that
build. The probe's 366 refusals were not the payload: with the body boxed to
one word the same 366 bodies failed, and the objdump names the escape as the
`&mut Rt::Frame<'_>` slot lent per element from a drain loop LLVM inlined
into `Op::run` — a fact about handler shape, not about this design. And the
design specializes at the wrong layer. It puts the pipeline's shape into one
Rust type per length so the stage dispatch becomes a `match`, which is the
whole of what it buys per element; the cost that dominates a `map` row, the
re-entry into the interpreter for each closure call, it cannot reach from
Rust at all, and it pays for the `match` with a bound of eight, a length
recursion (`Len`), a stage enum, and a declaration layer of per-length
macro copies.

What replaces it is the mechanism RFC-0019 already has. `iter::next<I, T>
(&mut I) -> Option<T>` is a shared signature; an adaptor is an extension
type `Map<I, U>` with a `next` instance declared at that pattern, whose body
calls the inner `next` through the typed handle the glue fills at the site
(`Fn1`, the same handle a lambda crosses on); a consumer is a handler over
`I: HasInstance<next>`. That needs two extensions of RFC-0019 — an instance
at a pattern type whose other variables are bound by matching the instance's
signature, and a required instance callable inside the handler — and no
length bound, no `Len`, about forty declarations. Its per-element cost is
one call per stage, the dyn chain's; the per-element win lives above it,
where the consumer's call site knows the whole pipeline type and the machine
or the MIR can lower the pipeline to a loop the closure bodies inline into,
which is RFC-0066's loop. The extensions come after the type-helper pass
over `acvus-extern`, since they land on `Fn1`, the per-site glue and the
registry's instance matching, which that pass reshapes.

The build is kept as a patch beside the worktree
(`.claude/worktrees/machine2/scratchpad/runA/fixed-stack.tracked.patch`)
and nothing of it is on master; `payload_per_instantiation` (`04b79af6`)
and `effect = E` on signatures stay, being facts of the derive and the
macro rather than of this design.

## Problem

The `Iterator` extension type (`acvus-ext/src/iter.rs`) is a chain of
`Box<dyn SyncStage>`: every adaptor (`map`, `filter`, `take`, …) boxes the
stage before it and answers `next` through a virtual call, and every
consumer pulls one `Option<Value>` per element through the whole chain.
The box was chosen so the pipeline could cross the extern boundary as one
erased value — a payload that named an element type would be two Rust
types on the two sides of the crossing (the module's own note).

The cost, measured under the pinned-load-base protocol (`accum`, n = 1e6,
`b40f957e`): `range | sum` 1.8 ns per element with no closure at all —
the pull chain alone — against `for range` at 0.9; `map(|x| -> x) | sum`
3.3, so one stage and one closure call add 1.5. Per pipeline, one heap
box per adaptor on top of the `Iter` value itself, which the real scripts
(`examples/`: `.iter().map(f).collect()` 136 times, `.filter()` 40,
`.count()` 39, `.any()` 23, `.sum()` 21) pay inside loops over small
collections.

## Decision

1. **The pipeline's type is the list of element types it passes through.**
   `Iter<(T_k, …, T_1), O>` is a pipeline whose source yields `T_1`, whose
   stages take `T_1 → T_2 → … → T_k → O`, and whose output is `O`. The
   tuple holds types only; which adaptor made each step is not in the
   type. A source is `Iter<(), T>`. An adaptor prepends:

   ```
   map<T, U, Ts, E, I, Rt>(it: Iter<Ts, T, E, I, Rt>, f: Fn1<T, U, E, Rt>)
       -> Iter<(T, ..Ts), U, E, I, Rt>
   ```

   `(T, ..Ts)` is a type-level cons over a tuple of type variables,
   spelled as **nested pairs**: the empty list is `()`, one element is
   `(A, ())`, two are `(B, (A, ()))`. A 2-tuple is already a `TyArg` and
   the checker already unifies a tuple type argument structurally and
   selects an `instance_of` instance by that structure (probed on
   `d22cb966`: four `depth` instances at lengths 0–3 answer by length; a
   destructuring parameter `(T, Ts)` binds the head; two lists of one
   length run one Rust instance). The checker infers every element of the
   tuple while typing the pipeline; no syntax reaches the author.

2. **Length, not shape, is what Rust sees, and Rust sees the list.** Every
   `T` in the tuple is a `TyVar`, which the extern crate erases to the
   runtime's value; in an `instance_of` instance the tuple's *structure*
   survives and only its leaves erase (`(Owned, (Owned, ()))`), so the
   pipeline's Rust type carries the list. Every adaptor **and every
   consumer** is a shared signature with `instance_of` instances for the
   lengths it admits (adaptors 0..=7 on their input, consumers 0..=8);
   **the length is bounded at 8** and the ninth adaptor is refused by the
   checker with the declared-bound message. A shared signature declares
   its call effect by naming it — `extern_signature! { ns: "iter",
   effect = E, fn fold<S, A, F, E, Rt>(…) … }` emits `E` as the
   signature's call effect, and a signature that names none stays
   `Known(PURE)` — so a consumer whose closure carries `E` is
   instanceable at `effect = E`. Naming is required rather than read off
   the generics because a declaration's effect variable is as often a
   type argument as its call effect: `sig::into_iter`'s and
   `sig::as_iter`'s `E` is the effect the returned `Iter` carries, and
   taking it for the call's own effect makes `as_iter` `Async` over an
   effectful pipeline (`acvus-interpreter-test/tests/extern_fn.rs`
   `io_in_iteration` and `io_inside_iterator_pipeline` are refused with
   "a function whose task is Async where Sync is required"). **A
   signature names its argument**: the type the per-length instances are
   matched by is read at the position of the signature's first variable
   inside the first parameter that reaches it, so a consumer with no
   closure at all is written `fn collect<Ts, O, E, I, Rt>(it: Iter<Ts,
   O, E, I, Rt>)` and its `E` is the pipeline's. One macro in
   `acvus-ext` emits the instances.

3. **The payload is uniform per length; the element types live in the
   acvus type.** At the glue every type variable erases to `Owned<Rt>`
   (`runtime_stand_in`), and an `instance_of` instance is selected by the
   list's length, so the pipeline's Rust payload must be one type per
   length: `Stages<Stage<Rt>, Stages<…, Source<Rt>>>` with `Owned<Rt>` at
   every element position — a fixed-size stage stack, no `dyn`, no box
   per stage. The element types `T_k … T_1` and `O` are carried by the
   acvus type `Iter<(T, Ts), O, …>` alone; a consumer whose declared
   element is ground (`sum` over `Erased<Rt, i64>`, a producer's `char`)
   reads it through `FromValue` — identity for `Owned`, a checked
   downcast for `Erased` — as today's stages do. `Stage<Rt>` is one enum
   (`Map(Fn1<Owned, Owned, E, Rt>)`, `Filter`, `Take`, `Skip`, `StepBy`,
   `TakeWhile`, `SkipWhile`, `Dedup`, `Flatten`, `Chunks`, `FlatMap`), a
   stage step that needs the element's Rust type (`Dedup`'s equality,
   `Flatten`/`Chunks`' container) a function pointer the adaptor's
   instance supplies over `Erased`. A closure is called through the typed
   `Fn1::call_now(rt, frame, (x,))` with `x: Owned<Rt>` — safe Rust: the
   raw-value entries `call_value_now`/`call_value` are removed, and a
   handler author's generic `Fn1<T, U>` cannot be fed a `V` because `T`
   and `U` are the author's own type variables. No `unsafe` in the
   pipeline's code; the one trust point is the `ExternType` materialize
   every extern value has (`payload_per_instantiation`: the checker
   selects the instance whose length matches the value's Rust type).

   Measured on the way (2026-09-20): a payload whose Rust type is a
   function of the element types (`TypeList::Body<O, E>` threading `O`)
   crosses only where every element type in a declaration is a bare
   variable — 18 of 41 iterator declarations name a ground or container
   element, and a per-element `Stage<In, Out>` with a `Same<In, Out>`
   witness is inert because every leaf is `Owned` at every instantiation.

4. **A consumer pulls through the typed stack.** `collect`, `sum`, `count`,
   `fold`, `any`, `all`, `find`, `join`, `contains`, `reduce` and `next`
   each pull one element at a time through the list trait's `pull`,
   which recurses outward-in over the stage tuple: typed, one Rust
   instantiation per length, no `dyn` and no boxed stage. A pull returns
   `Option<O>` per element; that `Option` is the cost of letting a
   consumer `break` early and `.await` inside its loop, which the
   existing consumer bodies do and a push sink would not admit
   (`drain!`/`drain_now!` keep their interface over `pull`). **A consumer
   matches the stage tuple once**, on entry, and runs a fused loop for
   the shapes it names (`map → collect`, `filter → collect`, `filter →
   count`, `map → sum`, `filter → any`, and the others the scripts use);
   every other shape runs the generic pull. The shapes are values, so an
   unnamed shape is the generic loop and never a refusal.

5. **Two sources are a source, not a stage.** `chain(a, b)` combines
   pipelines; the result is a new source (`Iter<(), T>`) and stages after
   it prepend as usual. An `instance_of` instance is keyed by the first
   variable's type, so `chain` has one instance per length of `a`, and
   **`b` is a source: `Iter<(), T>`** — a right operand with stages is
   finished first (`b.collect()` and a fresh source). No script or example
   calls `chain` today; `chain_all`/`pchain` follow the same rule. A
   chained part is held as one `Pull<T, Rt>` trait object (a sync and an
   async arm) — one `dyn` per part, never per stage. `flatten` over a
   pipeline of containers is a stage.

6. **Effects and identity are unchanged.** `E` and `I` ride on `Iter` as
   today; a stage whose closure is `Async` makes the consumer's loop
   `await` the call, as the `Stages::Async` half does now. The `Sync`/
   `Async` split of the chain disappears; it is the closure's effect,
   read per stage.

7. **The crossing is unchanged.** `Iter` is one `ExternType` value, one
   heap allocation per pipeline; adaptors mutate it in place (prepend a
   stage into the array) and return it, which is one allocation where
   today there are one plus one per adaptor.

## What it costs

- `acvus-extern-macro`: `#[derive(ExternType)]` refused a payload naming any
  type, effect or length parameter, so that every instantiation of an
  extension type shares one payload and `Value::materialize`'s vtable
  `type_id` check holds whatever the type arguments are. A payload that is
  the stage stack its element-type list names cannot share, so the derive
  takes `#[extern_type(payload_per_instantiation)]`, which lifts the refusal
  for that one declaration and moves the obligation to the declaring crate:
  the checker selects the instance whose list length matches the value's
  Rust type, and a selection that got it wrong arrives as that
  `debug_assert_eq!` rather than as silence. The refusal stands for every
  other extension type.
- `acvus-extern-macro`: `extern_signature!` takes `effect = E` and emits
  that variable as the call effect instead of `Known(PURE)`, so a
  consumer can be instanced. Instance admission needs no effect rule of
  its own: `matches_pattern`'s `effect_matches` under `Unknowns::Fixed`
  already admits `(_, Var)` — an `E` instance and a pure instance of an
  `E`-signature both match — and refuses `(Var, Known)`, which is the
  refusal an `E` instance of a pure signature keeps. What `add_instance`
  did need is the `Sync`/`Async` pair: a declaration carrying `sync =
  <fn>` contributes two handlers at one type, where before an
  `instance_of` declaration could contribute only a single generic one.
  No checker change: nested pairs are plain
  2-tuples, which the checker already unifies structurally. The ninth
  adaptor is refused today by the instance list ("outside the declared
  bound one of …"); naming the bound of 8 in that message is a
  diagnostics change in `acvus-mir/src/error.rs`.
- `acvus-extern`: `registry.rs::instance_type` walks the first parameter
  to the position of the signature's first variable instead of demanding
  a bare `Var(0)` or `&Var(0)`, which is what lets a consumer with **no
  closure argument** (`count`, `sum`, `last`, `collect`, `next`) name the
  pipeline in its parameter and take the call's task from it. The bare
  form still hides the task: nothing in the instantiated signature
  relates the call's effect to the argument, so
  `Solver::tightest_admitting` (`acvus-mir/src/solver.rs`) closes the
  instance decision with the call `Sync`. The two forms are measured one
  variable apart in `acvus-interpreter-test/tests/signature_effect.rs`
  (`a_signature_that_names_the_pipeline_is_async_over_an_async_stage`
  against `a_signature_that_hides_the_pipeline_leaves_its_call_sync`).
  The walk descends a reference, an extern type's type arguments and a
  tuple's positions; a parameter that reaches its variable through an
  array, an option, a slice or a function type is refused with
  `InstanceMismatch`, as it was before the walk.
- The pipeline's Rust type changes at every adaptor, so the `Large` box
  holding it is re-made per adaptor: one allocation per adaptor, as
  today, with no `dyn` and no `unsafe`. Zero allocations per adaptor
  would need an in-place fixed buffer and `unsafe`; `unsafe`-free is the
  choice (decided 2026-09-20).
- Instances: 13 adaptors × 8 + 19 consumers × 9 = 275, one Rust
  instantiation each; the `.text` delta is stated at merge.
- `acvus-ext/src/{iter.rs, iterator.rs}` are rewritten: one `Stage` enum,
  one `Iter` struct, consumers as push loops, `next` as pull over the
  same array; the trait pair `SyncStage`/`AsyncStage` goes.
- Consumers are emitted for nine lengths; the binary grows by that
  factor over the consumer set (about ten consumers), which `asm_probe`
  does not count (they are not `Op::run` bodies) and the `.text` delta
  is stated at merge.
- Nothing in the machine or the MIR lowering moves; the pipeline is
  still an extern value and its calls are still `CallExtern*`.

## Rejected

- **Stage kinds in the type** (`Iter<(Range, Map, Filter), O>`): every
  pipeline shape is its own Rust instantiation, so consumers exist only
  for shapes registered ahead of time and an unregistered shape is a
  refusal or a silent fallback to the chain. The element-type tuple is
  finite by length alone.
- **Fusing closures into one stage** (extern fn fusion): a different
  lever, kept as such; this RFC leaves the closure call per stage per
  element as the one remaining cost.
- **A per-element typed stage stack** (`Stage<In, Out>` with a `Same<In,
  Out>` witness, the payload a function of the element types): measured
  inert — every leaf is `Owned` at every instantiation — and unable to
  carry a declaration whose element is ground or a container, because the
  instance is selected by length while the payload type would be selected
  by element type. The list carries length; the element types are the
  acvus type's, and the guarantee against a handler author is the
  removal of the raw-value closure entry, not a Rust-level witness.
- **Keeping the dyn chain and only flattening it** into a `Vec<Box<dyn
  Stage>>`: removes the nesting, keeps a box and a virtual call per stage
  and the `Option` per boundary.

## Order of work

1. `extern_signature!` takes `effect = E` and declares it; a test that an
   `effect = E` declaration is an instance of a shared signature, at two
   lengths and with a pure instance beside them.
2. `Stage<In, Out>`, `Same<In, Out>`, `Stages<Ts, O>`, adaptors and
   consumers as per-length instances, push consumers, pull `next`; the
   raw-value closure entry removed; every existing iterator test green;
   examples byte-identical.
3. Fused arms; `accum` rows `range | sum`, `map * | sum`, `while let map`
   and `shapes` measured under `setarch -R` against the base; the
   allocation difference; `.text`.
