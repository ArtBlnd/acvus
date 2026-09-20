# RFC-0065: An iterator is the types its elements pass through

Status: Draft (2026-09-20)

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

   `(T, ..Ts)` is a type-level cons over a tuple of type variables. The
   checker infers every element of the tuple while typing the pipeline
   (it already names each intermediate type); no syntax reaches the
   author.

2. **Length, not shape, is what Rust sees.** Every `T` in the tuple is a
   `TyVar`, which the extern crate erases to the runtime's value; two
   tuples of the same length are one Rust instantiation. A consumer is
   therefore one instance per tuple length, and **the length is bounded
   at 8**: a pipeline of more than eight stages is refused by the checker
   at the ninth adaptor, naming the bound. The macro emits the cons and
   the consumers for lengths 0..=8.

3. **A stage is a value: one element in, zero or more out.** The runtime
   shape is one struct, `Iter { source, stages: [Stage; N] }`, `N` the
   tuple length. A stage is `Map(Fn1)` (one out), `Filter(Fn1)` (zero or
   one), `Take`/`Skip`/`StepBy`/`TakeWhile`/`SkipWhile` (one or zero, with
   state), `Enumerate` (one, `T → (u64, T)`), `FlatMap(Fn1)` (many),
   `Chunks(n)` (one per `n`), `Dedup` (zero or one, with state). Each is
   an arm of one enum; the type tuple records only its output type.

4. **A consumer pushes.** `collect`, `sum`, `count`, `fold`, `any`, `all`,
   `find`, `join`, `contains`, `reduce` run one loop: for each source
   element, through the stages in order, into the sink. No `Option` per
   stage boundary, no virtual `next`. `next` (the `while let` pull) is
   the one pull consumer: the `Iter` holds a cursor into its source and a
   buffer for a many-out stage, and pulls one output through the same
   stage array.

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

- `acvus-extern` gains a type-level tuple of type variables with a cons
  (`(T, ..Ts)` in `extern_signature!`/`#[extern_fn]`), and the checker
  gains the corresponding unification of a tuple type argument with a
  cons pattern. `Ty::Tuple` exists; what is new is the pattern.
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
- **Keeping the dyn chain and only flattening it** into a `Vec<Box<dyn
  Stage>>`: removes the nesting, keeps a box and a virtual call per stage
  and the `Option` per boundary.

## Order of work

1. The type-level cons in `acvus-extern` and the checker's unification of
   it; a test that `range(0,n) | map(f) | filter(g)` types as `Iter<(U, T),
   U>` and the ninth adaptor is refused.
2. `Stage`, `Iter { source, stages }`, push consumers, pull `next`; every
   existing iterator test green; `accum` rows `range | sum`, `map * | sum`,
   `while let map` and `shapes`/`examples` measured under `setarch -R`
   against the base.
