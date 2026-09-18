# RFC-0053: an aggregate that does not escape never exists

Status: Accepted — owner and coordinator, 2026-09-19
Extends: RFC-0018 (storage and references), RFC-0024 (paths),
RFC-0039 (an option is its payload), RFC-0041 (drop insertion),
RFC-0048 (ownership is the machine's), RFC-0051 (a `match` is one
dispatch, and it is exhaustive)

## Problem

`benches/shapes.rs`, this machine, master `a6f1d50d`, median of three
alternating runs of pinned binaries:

| case | ns/iteration | × Rust |
|---|---:|---:|
| construct | **37.2** | 177 |
| enum match | 27.2 | 74 |
| field read | 9.2 | 25 |
| field write | 8.6 | 46 |

`construct` is the largest ratio in the suite, and its body is seven
dispatched operations:

```
add::<i64>, make_object, assign_var, read_path::<false>, drop_value,
add::<i64>, add::<i64>
```

Seven operations at `int while`'s measured rate — 2.8 ns for three — is
about 7 ns. The other 30 ns is what the aggregate costs in memory.
`MakeObject` boxes an `FxHashMap<Astr, Value>`
(`acvus-interpreter/src/ops/composite.rs:62-71`,
`acvus-interpreter/src/value.rs:660-662`), a field read is
`FxHashMap::get` on an interned key
(`acvus-interpreter/src/ops/storage.rs:132-139`), and the release frees
both. Per iteration, in a loop.

The object in `construct` has three uses: it is assigned to `q`, one
field is read out of it, and `q` is dropped. Nothing outside the body
ever sees it. The same is true of `field read`, of `field write`, and of
the enum in `enum match`. **These aggregates are built, queried and freed
without ever being observed by anything but the instructions that build,
query and free them.**

`ssa_pass` cannot reach them: it promotes a storage read and written
whole, and refuses one touched by a path or a reference
(`acvus-mir/src/optimize/ssa_pass.rs:309-324`). That rule is
conservative, not wrong — it is the line this RFC widens.

## Decision

A storage slot of an object or enum type that **does not escape its
body** is replaced by one register per field, or by a `(tag, payload)`
pair. No `MakeObject`, no `MakeVariant`, no hash lookup, no drop of a
shell.

### 1. One escape predicate, complete, in `analysis::escape`

A value escapes when it is an argument of a `FunctionCall` or `Spawn`,
an indirect callee, a `MakeClosure` capture, the value a `Return`
carries, an element of `MakeArray`/`MakeTuple`/`StringConcat`, a field of
`MakeObject` or `FieldSet`, a `MakeVariant` payload, the value of an
`IndexSet`, `Commit` or `Assign`, or the container of `AsSlice`,
`Index` or `Eval`. A storage escapes when it, or a value naming it,
escapes; a value names a storage when it is the `dst` of a whole `Ref`
or `Take` of it. A `Ref` *under a path* hands out the address of a part,
and the storage that holds that part escapes outright.

The arms are enumerated over the whole of `InstKind`, so a new
instruction breaks the build rather than reading as non-escaping.

`validate::exhaustive` had its own version of this question that counted
only call arguments, spawn arguments and closure captures
(`exhaustive.rs:221-242` at `a6f1d50d`); this RFC made the two call one
function. Since 2026-09-20 `exhaustive` asks no escape question at all:
a `match` is closed by the scrutinee's settled `Ty::Enum` — the union of
every construction the value can flow from (RFC-0041) — wherever the
value came from (`exhaustive.rs::known_variants`; tests
`a_match_on_a_value_from_outside_is_closed_by_its_type`,
`a_match_on_a_container_element_needs_no_catch_all`). `escape`'s one
caller is `optimize::sroa`.

### 2. The same SSA builder, over a wider key

The pass runs immediately before `ssa_pass` in pass 2
(`graph/optimize.rs`) — after inlining, so a callee's object is visible
at its caller; before `dce`, which sweeps the constructor left with no
reader and the dead block parameter that carried the aggregate; before
`code_motion`, so the now-scalar loop invariants hoist; and before
`drop_insertion`, which then never sees the aggregate.

`SsaVar` gains one variant, `Part(ValueId, Part)`, where

```rust
pub enum Part { Field(Astr), Tag, Payload }
```

and every mechanism the Braun et al. builder already has — the pending
phi at an unsealed block, the trivial-phi removal, the `finish`
fixpoint, `patch_instructions` — applies to a field unchanged. A field
written under a branch gets its block parameter from the machinery a
whole variable uses. There is no second phi implementation.

A tag is not a `PathSeg`, because no place names one: it is which
variant a value holds, and it exists only as a register this pass
creates.

### 3. The instruction rule

| before | after |
|---|---|
| `MakeObject`/`MakeVariant` whose only reader is a replaced `Assign` | left in place; `dce` sweeps it |
| `Assign { Var(s), [], v }`, `v` from `MakeObject` | one define per field |
| `Assign { Var(s), [], v }`, `v` from `MakeVariant` | a tag define and a payload define |
| `Assign { Var(s), [], v }`, `v` a block parameter | each predecessor takes its own constructor apart; the parts get the phi |
| `Assign { Var(s), [Field(f)], x }` | a define of `f` |
| `Take { d, Var(s), [Field(f)] }` | `d` stands for the field's register |
| `Take { d, .., [Payload] }`, `UnwrapVariant { d, .. }` | `d` stands for the payload's register |
| `Ref { d, Var(s), [] }` | gone; the reads through `d` are the slot's reads |
| `TestVariant { d, .., tag }` | `d = tag_register == tag's number` |

Any use no row covers refuses that slot, and refusing a slot restarts
the walk. There is no partial escape: one escaping use anywhere sinks
the whole body's replacement for that slot.

`Ref { d, Var(s), [] }` is in the table and not refused, which is where
this RFC departs from the plan it was written against. The `match`
lowering takes a reference to a place scrutinee unconditionally
(`lower.rs:906-917`, "a place is lent for the read, as `TestVariant`
takes it"), so `enum match` carries `r17 = ref &e` and would be refused
by a rule that read "an `InstKind::Ref` names it". The reference is
never stored, never passed and never returned: it is read through and
let go, which the escape predicate above already decides. Treating it as
an alias rather than as an escape is strictly more precise, and it costs
nothing, because a `Ref` whose result does escape is still an escape.

An enum keeps its aggregate when a variant carries nothing — the payload
register would be undefined on that arm — or when two variants disagree
on the payload type, because then the merge has no type. This is the
same condition Graal's partial escape analysis states for a merge
(Stadler et al., CGO '14).

The table has no `Drop` row. `drop_insertion` is the only writer of
`InstKind::Drop` and it runs at the end of pass 2, so a slot this pass
sees has no `Drop` yet; the drops a replaced aggregate's move-only
fields need are the ones `drop_insertion` later places on the part
registers, each at its own last use.

## What it costs

- One pass file, `acvus-mir/src/optimize/sroa.rs`, and one analysis
  module, `acvus-mir/src/analysis/escape.rs`.
- One variant on `SsaVar` and one new enum, `Part`.
- `exhaustive` no longer asks an escape question (see §1): the
  settled type closes every `match`, so a value from a context, a
  container, a parameter or a return needs no catch-all.
- One existing test changed its assertion, not its claim:
  `a_match_over_a_locally_closed_enum_needs_no_catch_all` counted
  `is A`/`is B` in the optimized listing to mean "two arms cost one tag
  test". There is no variant left to test the tag of, so it now counts
  the branches: one chooses the constructor, one tests the tag.

## Rejected

**A scalar-replaced representation.** Keeping the aggregate but laying
it out as a shape table plus a `Box<[Value]>` — RFC-0050 — leaves one
allocation and one query per access, because a structural type's field
offset is not fixed by the type: two objects that meet join to the union
of their fields, so one field sits at different positions in different
values of one type. It
is the right answer for an aggregate that genuinely lives past the body
that built it; it is the wrong answer for one that does not live at all.
The two do not overlap: of the six `shapes` cases, three are reached by
this RFC alone, one by RFC-0050 alone (`vec of objects`, whose object is
returned from a closure and then stored in a `Vec`), one by both
(`enum match`), one by neither (`option match`, which has no aggregate).

**Partial escape.** Materializing the aggregate at the point it escapes,
so the fast path stays scalar, is what Graal's PEA buys over the simple
form. Every non-escaping aggregate in the six cases is non-escaping on
every path, and the only escaping one escapes on every path, so it buys
nothing here. The design extends to it without restructuring — the
per-slot map becomes per-block state and an escaping use emits a
`MakeObject` from the recorded parts — and that is a second increment.

**Reusing `exhaustive`'s predicate as it stood.** It counted three of
the escape kinds. Sound where it refuses an exhaustiveness claim; a
miscompile here. Completing it was the precondition for every other
rule in this RFC, not an improvement to it.

## Consequences

`benches/shapes.rs` and `benches/accum.rs`, median of three alternating
runs of pinned binaries, base `a6f1d50d`, n = 1e5 / 1e6:

| case | base ns | after ns | × Rust, base → after |
|---|---:|---:|---|
| construct | 37.2 / 37.2 | **2.8 / 2.8** | 177 → 13.6 / 13.4 |
| field write | 8.6 / 8.6 | **2.9 / 2.9** | 46 → 15.7 / 15.9 |
| field read | 9.1 / 9.3 | **3.1 / 3.2** | 25 → 8.4 / 8.8 |
| enum match | 27.1 / 27.3 | **13.6 / 13.6** | 74 → 36.9 / 36.9 |
| option match | 7.1 / 7.1 | 7.1 / 7.1 | unchanged |
| vec of objects | 9.5 / 9.1 | 9.5 / 9.1 | unchanged |

Every case in `benches/accum.rs` is unchanged: over five alternating
runs each case's after-samples overlap its base-samples, and the largest
median move is 6.5 % on a case whose own base samples span 2.8 to 3.7 ns
(`int while`, n = 1e6). At 3 ns the bench's 0.1 ns print resolution is
already 3 %.

`construct` is now `int while`'s body exactly — `add, add` under `lt` —
and `int while` measures 2.8 ns on the same runs. The remaining 13×
against Rust is RFC-0052's seven-instruction dispatch, which is a
different problem.

`field read` lands at 3.1 rather than the arithmetic band's 2.8 because
`acc + p.x + p.y` becomes `acc + 1 + 2` and nothing folds it: there is
no constant-folding pass (`const_dedup` deduplicates literals and
`code_motion` hoists, neither folds). Four operations, not three.

`enum match` landed at the top of its band on those runs, because the
compare against the threaded tag still ran at the merge. The section below
is where that compare goes.

## Jump threading through the tag

The tag this pass writes is a constant on each incoming edge, so the
compare at the merge asks a question each edge has already answered.
`reaching_tags` settles the tag at the end of every block by a forward
walk over the plan's own `DefineVariant` actions, and `thread` rewrites
the graph before the SSA builder reads it: a dispatch whose own tag is
settled becomes `Terminator::Jump` (RFC-0051 rule 4), and otherwise each
incoming edge whose tag is settled leaves for its arm directly, carrying
the arguments it had handed the dispatch block's parameters. The
builder then places each arm's payload phi against the edges the arm
actually has, so no tag register and no compare survive. `dispatch_for`
accepts a dispatch of any width, because a threaded edge needs no
compare; a dispatch that keeps an edge whose tag is not settled still
needs one, and the chain for three or more edges is not built.

Threading runs in this pass rather than one of its own because after
scalar replacement a tag is a numeric register, and no IR form hands a
numeric tag phi to a later pass.

A two-armed `enum match` and a three-armed one both lose their
`MakeVariant`, their tag and their dispatch. Threading alone does not
pay, though: what it leaves is an arm block that `code_motion` then
empties, because the sink lifts the arm's `acc + payload` into the branch
that selected it, and the arm is left holding `jump merge(value)` over
one `Nop` — reachable, so `prune` leaves it; a block, so `dce`, which
sweeps instructions and block parameters, leaves it too.
`prepare::recognize_diamond` needs both arms to jump to one join laid out
next to them, and a forwarder between arm and merge breaks that, taking
the enclosing `Loop` with it: measured, threading alone runs `enum match`
at **16.8 ns** against 9.9 at `c2116d3f`, as flat block dispatch.

`optimize::forward` is the other half, and the two land as one mechanism:
a block with no parameters whose instructions are all `Nop` and whose
terminator is an unconditional `Jump` is its target, so every predecessor
takes that edge with the arguments the block carried and the block is
removed, to a fixed point. `recognize_diamond` already admits an arm that
is the test's own edge into the join, which is exactly the shape the
collapse produces, so nothing in `prepare` is taught anything. The blocks
a loop is made of — the header, the block entering it, the blocks it
leaves for — are held: `lsr::Frame::of` writes a reduction into the
entering block and `recognize_loop` matches all three against the layout.

Median of three alternating runs of pinned binaries against `c2116d3f`,
n = 1e5 / 1e6:

| case | base ns | threading only | with the collapse |
|---|---:|---:|---:|
| enum match | 9.9 / 9.9 | 16.8 / 16.8 | **6.0 / 6.1** |
| enum match three | 33.0 / 33.2 | 21.6 / 21.6 | **20.2 / 20.2** |

`enum match` is now below its own base by 39 %: the body holds four
operations and one `Diamond` where the base held six and two, because the
constructor's diamond and the tag test's diamond have become the one
diamond that decides the arm. `enum match three` keeps the flat block
dispatch it had at base — its `match i % 3` is an `if`/`else if` whose far
arm is itself a `JumpIf`, and `recognize_diamond` needs a `Jump` there —
so its 39 % is the removed aggregate alone. The five other `shapes` cases
and the four `accum` cases whose medians moved at all have prepared
listings byte-identical to base, and in those four the pass removes no
block; what moves is the host binary's own layout, not the program.
