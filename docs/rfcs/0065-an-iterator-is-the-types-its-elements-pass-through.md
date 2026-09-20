# RFC-0065: An iterator is the types its elements pass through

Status: Accepted (2026-09-20)

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
   its call effect from the declaration's own effect variable, so an
   `effect = E` consumer is instanceable (decided 2026-09-20: the macro's
   `Known(PURE)` literal becomes the declared effect). One macro in
   `acvus-ext` emits the instances.

3. **A stage is typed, and the pipeline is a typed list of stages.**
   `Stage<In, Out>` is one enum: `Map(Fn1<In, Out>)` calls the closure
   **typed** (`call_now(rt, frame, (x,))` with `x: In`); `Filter`,
   `Take`, `Skip`, `StepBy`, `TakeWhile`, `SkipWhile`, `Dedup` carry a
   `Same<In, Out>` — a type-equality witness constructible only at
   `Same<T, T>`, the identity as a value, which is how "specialize when
   `In = Out`" is written without specialization; `Flatten` and `Chunks`
   carry the witness for `Out = Vec<In>`'s inverse and `Vec<In>`. The
   pipeline holds `Stages<Ts, O>`: for `Ts = (T, Ts')`, `(Stage<T, O>,
   Stages<Ts', T>)`, recursion by one trait over the list. At the glue
   every leaf is `Owned`, so a length is one Rust instantiation and the
   kind stays in the enum. **No raw-value closure entry exists**: a
   closure receives what its type says, or nothing; `Fn1::call_value_now`
   and `call_value` are removed. The pipeline's internals hold no
   `unsafe`; the one trust point is the `ExternType` materialize every
   extern value already has (the checker selects the instance whose
   length matches the value's Rust type).

4. **A consumer pushes.** `collect`, `sum`, `count`, `fold`, `any`, `all`,
   `find`, `join`, `contains`, `reduce` run one loop: for each source
   element, through the stages in order, into the sink. No `Option` per
   stage boundary, no virtual `next`. `next` (the `while let` pull) is
   the one pull consumer: the `Iter` holds a cursor into its source and a
   buffer for a many-out stage, and pulls one output through the same
   stage array. **A consumer matches the stage array once**, on entry, and
   runs a fused loop for the shapes it names (`map → collect`, `filter →
   collect`, `filter → count`, `map → sum`, `filter → any`, and the others
   the scripts use); every other shape runs the generic loop, which
   matches each stage per element. The shapes are values, so an unnamed
   shape is the generic loop and never a refusal.

5. **Two sources are a source, not a stage.** `chain(a, b)`, `chain_all`,
   `zip` combine pipelines; the result is a new source (`Iter<(), T>`)
   whose value holds the two finished pipelines, and stages after it
   prepend as usual. `flatten` over a pipeline of containers is a stage
   (`FlatMap` with the identity).

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

- `acvus-extern-macro`: `extern_signature!` declares the call effect from
  the declaration's effect variable instead of `Known(PURE)`, so a
  consumer can be instanced. No checker change: nested pairs are plain
  2-tuples, which the checker already unifies structurally. The ninth
  adaptor is refused today by the instance list ("outside the declared
  bound one of …"); naming the bound of 8 in that message is a
  diagnostics change in `acvus-mir/src/error.rs`.
- The pipeline's Rust type changes at every adaptor, so the `Large` box
  holding it is re-made per adaptor: one allocation per adaptor, as
  today, with no `dyn` and no `unsafe`. Zero allocations per adaptor
  would need an in-place fixed buffer and `unsafe`; `unsafe`-free is the
  choice (decided 2026-09-20).
- Instances: (13 adaptors × 8) + (consumers × 9), one Rust instantiation
  each at `Owned` leaves; the `.text` delta is stated at merge.
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
- **An erased stage array** (`[Stage<Owned, Owned>; 8]` with the list only
  in the acvus type): one allocation per pipeline, but every closure call
  inside is `Fn1<Owned, Owned>` on a raw value — a lambda typed `T → U`
  can be handed any value, and the only fence is `unsafe` discharged by
  citing the checker. Rejected for the typed list (decided 2026-09-20).
- **Keeping the dyn chain and only flattening it** into a `Vec<Box<dyn
  Stage>>`: removes the nesting, keeps a box and a virtual call per stage
  and the `Option` per boundary.

## Order of work

1. `extern_signature!` declares its effect variable; a test that an
   `effect = E` declaration is an instance of a shared signature.
2. `Stage<In, Out>`, `Same<In, Out>`, `Stages<Ts, O>`, adaptors and
   consumers as per-length instances, push consumers, pull `next`; the
   raw-value closure entry removed; every existing iterator test green;
   examples byte-identical.
3. Fused arms; `accum` rows `range | sum`, `map * | sum`, `while let map`
   and `shapes` measured under `setarch -R` against the base; the
   allocation difference; `.text`.
