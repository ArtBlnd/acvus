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

Waiting: the compiler half (`Terminator::Diamond`, the four lowering
sites, the 81 arms, `validate`), then the machine half (`prepare` builds
`Diamond<C>`/`Select` from the terminator; `recognize_diamond` deleted).
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
