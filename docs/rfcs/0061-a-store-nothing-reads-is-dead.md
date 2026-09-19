# RFC-0061: a store nothing reads is dead

Status: Accepted — 2026-09-19
Extends: RFC-0018 (references, loans), RFC-0041 (drop insertion), RFC-0045
(an assign releases what the slot held), RFC-0048 (ownership is the
machine's), RFC-0060 (a small pure closure called where it was made is its
body)

## Problem

`dce::is_root` answered `true` for every `InstKind::Assign`, so a store
into a local slot was unconditionally live and no pass in the tree removed
one that nothing reads. Two costs were visible:

- A value stored and never read outlived the body: `let s = "ab" + "cd";
  1` kept both literal slots, both borrows, the concatenation, the store
  and the drop of the slot, for a body that returns `1`.
- RFC-0060's inliner had to remove the closure's own `assign f = r0`,
  `ref &f` and drop itself, because a `MakeClosure` whose only reader is a
  store outlived every call spliced from it. That RFC's first open question
  is this one.

## Decision

**A store into a local slot is live only where a live instruction reads
that slot before the next store into the same slot.** Three rules.

1. **The reader decides.** An `Assign` whose target is `Var`/`Param` with
   an empty path is a conditional root: `dce` keeps it when some
   instruction it has already marked live reads that slot on a path out of
   the store, up to the next whole store into it or the body's end. A read
   is what `loans` counts as one — a `Take`, a `Ref` and every use through
   that reference, a field read, a value lent to a call, anything the
   terminator carries the slot's loan out through. The path argument is
   `dse`'s: every path, over the CFG. Because a store's reader may itself
   be dead — the `ref &f` an inlined call leaves behind — the mark phase
   alternates between tracing operands and pulling in the stores its newly
   live instructions read, to one fixpoint. A store through a reference or
   into a path stays an unconditional root.

2. **The old occupant's release does not move.** An assign releases what
   the slot held (RFC-0045), so removing a store whose slot holds a value
   would move that release to the next store or to the body's end, which an
   extern's `Drop` sees. Rule 1 therefore applies only where the slot holds
   nothing on every path into the store: a forward may-hold analysis whose
   entry holds every parameter and capture, in which a whole store fills a
   slot and a take of a move-only slot empties it. Where the slot may hold
   a value, the store stays. The exception is a slot whose type is a word:
   its assign performs no release at all (RFC-0048 §4), so no release can
   move and the store is removable whatever the slot holds.

3. **The stored value is released once, and by `drop_insertion`.** With
   the store gone its value has no reader. A pure producer — a
   `MakeClosure`, a `string_clone`, arithmetic — goes with it in the same
   backward walk, and the value is never made. A producer that is a root —
   a `Fetch`, an opaque call — stands, and its result is a value the body
   still owns; `graph::optimize::run_pass2` runs `drop_insertion` after
   `dce`, and that pass places the release at the end of the value's live
   range, which is now the producer itself. `dce` emits no drop of its own.

## What it costs

Two analyses per body: the may-hold fixpoint of rule 2, and, per candidate
store, a forward walk of the points its slot may be read at. No compile-time
measurement has been taken.

Listings, `compile_script_optimized`, base `ef619b66`:

| source | before | after |
|---|---|---|
| `let s = "ab" + "cd"; 1` | 2 stores, 2 borrows, a concat, a drop | `return 1` |
| `@out = @other; @out = @out + ")"; 1` | `assign @out = r1` | `drop r1` |
| `let s = @text; s = @other; @out = s + ")"; 1` | 2 stores, 2 clones | 1 store, 1 clone |
| `@out = @out + ")"; 1` | unchanged | unchanged |

The `Large` a store leaves without a reader is released the same number of
times as before: `acvus-interpreter-test/tests/unread_store_drop.rs` counts
three releases for `let t = held(3); 7` and three for `let t = held(3);
rank(&t) + 4`, and the same two counts on the base pass.

Rule 2's word exception does not fire in the pipeline's own listings,
because `ssa_pass` promotes a word slot to a register before `dce` sees a
store into one; `let a = 1; let b = 2; a + a` and `let x = 1; x = 2; x`
hold no store at all at pass 2. The rule is read at the pass's own contract
in `dce`'s unit tests.

## Rejected

- **`dce` emitting the drop of the orphaned value.** There is one writer of
  `InstKind::Drop`, and it already runs after this pass; a second drop
  mechanism would be two places that decide one release.
- **Deleting RFC-0060's residue removal.** The inliner replaces the
  `MakeClosure` with the locals its captures bind to, so the residue's
  `assign f = r0` names a value that has no definition once the removal is
  off. Four `inline.rs` snapshots, whose listing is the inliner's output
  before any of pass 2, then change;
  `a_large_capture_is_inlined_as_the_local_it_was` carries
  `assign f = r2` with nothing defining `r2`. A use without a definition is
  what `graph::optimize`'s `debug_validate` refuses. Rule 1 does remove the
  chain once `dce` reaches it, but the eight passes of pass 2 above `dce`
  read the body first, so the residue stays the inliner's.

## Consequences

- `let s = <pure expression>; …` with `s` never read makes no value: the
  producer goes with the store. Where the producer is effectful it still
  runs, and its result is dropped where it is made.
- A test whose witness is a literal in a listing needs the binding read:
  `acvus-mir-test/tests/literal.rs`'s `b""` case reads `s`.
- Two `optimized.rs` snapshots lose the context-variable prologue's store
  and gain the drop of the value it held.
