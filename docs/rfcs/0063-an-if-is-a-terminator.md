# RFC-0063: An `if` is a terminator

Status: Accepted (2026-09-20)

## Problem

The lowering writes every two-way branch as `JumpIf { cond, then_label,
then_args, else_label, else_args }` (`lower.rs`, 17 emission sites — `if`,
`&&`, `||`, `?`, `while`'s test, a `match` arm's guard, a `for`'s older
shape). The shape the source had — two arms that rejoin at one block — is
gone the moment it is written, and two readers rebuild it:

- `prepare::recognize_diamond` (`prepare.rs:1432`) scans forward from a
  `JumpIf` for the near arm's block, its jump, the far arm, and a common
  join, checking `references(label)` at each step, to decide whether the
  branch is a `Diamond` region or two joints; `select_shape` then asks the
  region whether one arm computes and the other passes through (`Select`,
  RFC-0052). A lowering that emits a different order stops matching and
  the branch runs as joints — a performance cliff no test states.
- RFC-0057 deleted `merge_of`, a field that recorded the join no pass
  decided on; the recognizer is the same fact rediscovered from the
  other side.

RFC-0051 (`Switch`) and RFC-0057 (`For`) stated the rule: **a terminator
keeps the control shape the source wrote.** `if` is its third instance.

## Decision

1. **The terminator.** A two-way branch whose arms rejoin ends its block
   in

   ```
   Terminator::Diamond {
       cond: ValueId,
       then: Label,  then_args: Vec<ValueId>,
       else: Label,  else_args: Vec<ValueId>,
       join: Label,
   }
   ```

   `join` is the block both arms jump to; each arm's last block ends in
   `Jump { label: join, args }` (the `if`-expression's value is the
   join's parameter, as today). The lowering emits `Diamond` for `if`
   (statement and expression), `&&` and `||`, which rejoin by
   construction; `?` does not — its fail arm returns — and keeps
   `JumpIf` (corrected 2026-09-20 by the compiler half's measurement). A branch whose arm leaves the enclosing loop (`break`,
   `continue`) does not rejoin: the lowering knows this at emission (the
   statement is syntactic) and emits `JumpIf` for it, as RFC-0057
   Decision 4 says; the machine runs that branch as joints.

2. **`JumpIf` remains for the shapes that are not a diamond**: the
   `while` test (its exit is the loop's, not a join), a `match` arm's
   guard (its else is the next arm), and an arm that leaves a loop. A
   later RFC may give `while` its own terminator; this one does not.

3. **What the terminator makes free.**
   - `recognize_diamond` is deleted: `prepare` reads `join` and the two
     arms' extents from the terminator and the block table; the
     `Diamond<C>` region op is built from that, and `select_shape` is
     asked of the terminator (one arm computing, the other passing
     through — read from the arms' instruction ranges as today).
   - The join is one fact in one place: liveness, `assign_slots`'s
     edge moves and `spawn_split`'s split rule read `join` rather than
     re-deriving it.
   - Exhaustiveness of the two arms is the terminator's: both labels are
     present or the terminator is malformed (validate refuses).

4. **Passes.** Every consumer of `JumpIf` (81 sites in `optimize/`,
   `analysis/`, `graph/`) gains a `Diamond` arm with the same edge
   semantics (two successors, arguments per edge) and one more fact
   (`join`); a pass that rewrites edges rewrites `join` alongside
   (`inliner`, `cfg`). `Edges::successors` in `prepare` gains the arm the
   `For` run found missing for its terminator — the same class of hole.

## What it costs

- 81 match arms in the compiler and the lowering's 17 sites, of which
  the four diamond-shaped ones move; `Terminator` gains a variant.
- Two terminators for two-way branches (`Diamond`, `JumpIf`) until
  `while` and the guard get their own; a pass that treats them alike
  says so in one helper (`two_way(&Terminator) -> Option<(cond, then,
  else)>`).

## Rejected

- **`JumpIf` with an optional `join` field.** The field is set by the
  lowering and never decided on by a pass — `merge_of`, which RFC-0057
  removed for that reason. A terminator that *is* the shape has no
  optional part.
- **Recognizing the diamond in a MIR pass** (a `DiamondRegion` side
  table). The same rediscovery moved one stage earlier; the lowering
  already knows.

## Consequences

The compiler half has landed: `InstKind::Diamond` and
`cfg::Terminator::Diamond`, nine lowering sites, the consumer arms,
`validate`, and the printer's `if cond -> L1 else L2 join L3`. `prepare`
reads a `Diamond` exactly as it read a `JumpIf` — the four terminator
enumerations, the block emitter and `recognize_diamond` accept it — so the
machine's shapes are the ones it recognized before.

Three corrections Decision 1 needs, and one obligation it puts on every pass,
found in building it.

**`?` is not a diamond.** Its failure arm rebuilds the operand's failure at
the function's return type and *returns* (RFC-0038), so the two paths never
meet. `?` keeps `JumpIf`. Decision 1's list of four is a list of three:
`if`, `&&`, `||`.

**A payload pattern's test is a diamond.** `Some(1) = e` tests the tag, and
on a match tests the payload; on a failure it takes `false` to a join of one
`Bool` parameter. That is `&&` written by hand, it rejoins by construction,
and it is a `Diamond`. Decision 1 did not name it.

**An arm that does not rejoin is not only one that leaves a loop.** An arm
whose tail is typed `!` ends in `Diverge` and reaches no join either
(RFC-0038). The criterion is therefore reachability of `join` from each arm
over the instructions the branch's arms occupy — `ir::reaches`, which the
lowering asks of the arms it has just written and `validate` asks of a whole
body. It is exact where "an arm holds a `break`" would not be: a `break` of
a loop written *inside* an arm leaves that loop, and the branch still
rejoins.

**A pass that dissolves the join maintains it.** `optimize::sroa`'s
threading leaves the block a dispatch used to be reachable by no path; where
every path through it ended at one block, that block is where the arms now
meet, and where they scatter the branch is demoted (`cfg::demote_diamond`).
`optimize::forward::collapse` maps `join` through the forwarders it removes.
`graph::inliner` and `cfg::promote`/`demote` carry it. This is the
prerequisite the machine half needs: a stale `join` is unreadable, and
`validate` refuses one (`DiamondArmMissesJoin`).

**The demotion is not final.** `sroa` demotes from what it knows at its own
moment, and `forward` and `dce` then bring the scattered arms back to one
block; nothing in either of them is the place to decide a branch's shape.
`optimize::rejoin` is: it runs once, on the `MirBody` `cfg::demote` produces,
after every other pass and before `validate`. A branch it restores is one
`cfg::demote_diamond` recorded — the lowering's fact, kept in
`MirBody::demoted_diamonds`, since Decision 1's rejected `JumpIf` field would
be a shape a pass decides. The criterion is the lowering's own, `ir::reaches`
over the instructions between the branch and the candidate join, and
`ir::meets_again` is the one function both `rejoin` and `validate` ask. What
`rejoin` leaves demoted and `validate` then finds meeting again is a pass
that ran after it and did not restore the terminator
(`DemotedDiamondMeetsAgain`).

Waiting: the machine half — `prepare` builds `Diamond<C>`/`Select` from the
terminator and `recognize_diamond` and its straight-run scan go.
`spawn_split` over a body with an `if` reads `join`.

## Order of work

1. Compiler: the variant, the lowering, the passes, `validate`; `acvus
   mir` prints `if cond -> L1 else L2 join L3`; every existing test's
   pass/fail set unchanged; the listing tests that pinned `jump_if` for
   an `if` re-pinned.
2. Machine: `prepare` reads the terminator; `recognize_diamond` and its
   straight-run scan go; `select_shape` asked of the terminator; the
   `Diamond<C>`/`Select` ops unchanged; benches (`branch while`, `option
   while`, `grade while`) ±1 %.
