# RFC-0066: A loop is analyzed, and the lowerer decides

Status: Draft (2026-09-20) — a design refined over several passes before
any code run; each phase of the order of work is its own run.

## Problem

Every number the machine produces today is one thread. `spawn_split`
divides a loop of independent `Heavy` calls (RFC-0046, RFC-0057) and gains
5.8–7.5× on eight of them, and nothing else divides anything: a loop of
`Sync` work over a large iteration space — `logs inline`, two microseconds
a line over a hundred thousand lines, no carried state — runs to the end on
one core with thirty-one idle. The language's stance is that the author
writes the loop and the system finds the parallelism; the author never
writes it by hand.

Finding it needs facts a compiler classically obtains by *transforming*
the IR — unrolling to expose the iteration space, blocking to shape the
memory pattern, scalar-evolution to know the trip count — and every one of
those transformations is a guess about what the lowerer will do with the
result. MIR is not the place for that guess: it states what the program
means, and a transformation there pre-empts a decision the lowerer has
better information to make (the target, the runtime, the actual `n`).

## Decision

The work is analysis, written into tables the lowerer reads. MIR is not
rewritten by any layer. Five layers, each grounded in the one below it;
no layer holds a number nobody measured.

0. **A cost table is measured, not written.** At first compilation on a
   target, the runtime measures the cost of every operation kind it can
   emit — each `Op` family at its register forms, a spawn, a join, a
   frame bind — and stores the table for that target. Every cost any
   later layer computes is a sum of rows of this table, in the table's
   own unit; no layer knows the unit and no source file carries a cycle
   count. The table is cached per target and invalidated when the
   runtime binary changes. A target with no table decides nothing: the
   loop runs as the plain function it already is, on one thread, until a
   table exists.

1. **An invariant table over SSA chains, and its bound.** For every
   chain the escape and loan analyses already walk, the table records
   what is known at entry and at exit and what the chain costs. **The
   induction variable is `for`'s and only `for`'s**: entry, exit and
   step are read off RFC-0057's terminator, never recovered from a
   recurrence, and a `while` loop has no induction variable, a trip
   count of one (a lower bound) and is never divided. A value is
   invariant across the loop when it is defined outside it and the
   loop's header carries no phi for it — SSA's own fact; a storage is
   unwritten when the loans' element-write split says so. The trip
   count is the symbolic `(exit − entry) / step` over invariant values;
   nested `for`s multiply; the expression alphabet is constants,
   invariant values, `+`, `×`, `min`, `max`, and nothing else — the
   expression is not simplified, it is evaluated at layer 4's points.
   An index is understood only as the induction variable plus a
   constant; a bound check is absorbed where the invariants imply it
   (`i` in `0..n`, `n == len(xs)`, `xs` unwritten), as a row of the
   table, not a rewrite of the check; any other index is neither
   absorbed nor divisible. The cost is the sum of the chain's operations
   in table-0 rows times the trip-count expression — a lower bound,
   never an estimate.

   This is the whole of what this RFC takes from scalar evolution: an
   affine variable the syntax already owns, a symbolic count over it,
   and invariance from SSA. Non-affine recurrences, induction-variable
   canonicalization, expression simplification, overflow reasoning and
   address evolution are not built. A `while` whose body has the shape
   of a `for` — a counter compared against an invariant and stepped by
   a constant — may **later** be promoted to `for` by a recognizer that
   rewrites the terminator; if no such promotion is exact, the `while`
   stays undivided, and nothing else is done for it.

2. **Regions.** Over the table, a region is a set of chains that can be
   evaluated as one unit without crossing a jump the analysis cannot
   see through. The rules that bound a region: a jump's target must be
   inside it; a diamond inside it must have both arms free of effects
   that carry order (RFC-0013's commutative effects and RFC-0046's tasks
   are the type-level facts read here — an arm with an ordered effect
   ends the region at the branch); a storage written inside it is
   written only through the induction variable's index (RFC-0057's
   non-overlap). A loop whose body is not one region is not divisible,
   and the analysis says so rather than dividing part of it.

3. **A region carries its invariants.** Entry and exit values of every
   variable that crosses its boundary, its cost per iteration, its
   effect class, and whether its carried state is a reduction the
   machine can recombine exactly (integer sums, counts, min/max; a
   floating-point reduction is not exact under reassociation and is
   marked so). Only now do *spawn* and *split* exist as concepts: a
   region whose iteration space can be partitioned and whose carried
   state recombines exactly is splittable, and the table says into what.

4. **The lowerer decides.** `prepare` reads the region and its
   invariants and chooses the machine form: a split region whose
   chunks run as `Heavy` spawns and join through the exact reduction; a
   region evaluated in place as one operation; or a region unrolled at
   the operation level. The choice is one comparison of table-0 sums —
   the region's cost per iteration times the iteration count, against
   the spawn and join rows times the chunk count. Where the count is
   known only at run time, the lowerer inserts **one `if`** ahead of the
   loop whose condition is that comparison with the induction variable's
   exit value substituted for the count: an abstract interpretation of
   the invariant table at a handful of concrete points of `n`, folded to
   a threshold the machine tests once per loop entry. The program is
   otherwise unchanged; the `if` is the whole of the run-time decision,
   and in the machine it is **one dispatch point**: an operation that
   reads `n`, compares, and continues into one of two chains. The split
   chain — chunk, `Heavy` spawns, join — is the cold side, laid out of
   line as RFC-0052 lays any cold path; the other side is the region
   evaluated in place, spawning nothing, and is the hot side. A loop
   that is never worth splitting therefore costs exactly one compare
   over what a plain call of the same body costs today, and the two
   sides are the same body under two operations, not two programs.

   The consequence this buys: a parser — a loop over a byte or token
   array with a cursor as its induction variable — is divisible by the
   same rule, because `[]` is a native primitive with a table-0 row, the
   region's cost per element is a table-0 sum, and spawn and eval have
   rows too. Nothing about the loop being "a parser" is known or needed;
   the region and its invariants are.

## What it costs

- A measurement step at first compilation, per target, and the storage
  of its result; a runtime without one runs single-threaded until it has
  one.
- Two side tables (invariants per chain, regions per body) computed in
  the pass-0 position of `graph::optimize`, where the borrow check
  already runs in dependency order; neither is an IR field, for the
  reason RFC-0064's summaries are not.
- A new family of machine operations for a split region (chunk, spawn,
  join-with-reduction) and the lowerer's decision procedure over them.
- The analyses of RFC-0057 (induction variable, non-overlap, order
  token) and RFC-0064 (loans, summaries) become inputs and must stay
  stable in what they promise.

## Rejected

- **Transforming MIR** — unrolling, blocking, permutation in the IR: a
  guess about the lowerer written into the program's meaning; undone by
  nothing when the guess is wrong for a target.
- **Constant costs in source** (a cycle count per operation, a spawn
  cost in microseconds): true on one machine on one day; the cost table
  is measured where it is used.
- **Scalar evolution alone**: it answers "what is the trip count" and
  fails otherwise; the invariant table answers a lower bound on cost and
  what is unwritten, which is what a split needs, and a failed trip
  count is still one iteration of cost.
- **Recognizing recurrences in `while` loops** as induction variables:
  the door to general scalar evolution. A `while` is either promoted to
  `for` when its shape is exactly `for`'s, or left alone.
- **Cache blocking and loop permutation**: shape the memory pattern of
  nested loops; a later RFC, once regions exist to be permuted.
- **An explicit `par for`**: the author naming the parallelism. Kept out
  because the facts the split needs are the same facts the checker
  already establishes; where they are not established the loop is not
  split, and the author is told why, not asked to assert it.

## Where this is hard

- An ordered effect inside the loop body — I/O in the middle of a parse
  — ends the region at that point, and the loop is not divided. This is
  the rule working, not a case to rescue: the effect system already
  separates it in SSA, and the analysis reads that separation.
- The region rules of layer 2 are not settled. If they reduce to effect
  rules alone — a region is bounded exactly where an ordered effect
  appears — the analysis is the effect system read at loop granularity
  and needs no separate notion; whether every jump-boundary condition is
  expressible as an effect is the open question, and the rule set is
  refined before layer 2 is briefed.
- Layer 0's measurement must be reproducible enough to decide with:
  the load-base and layout effects RFC-0052 records (a docs-only commit
  moving a row ±6 %) bound what one row can mean; the table records
  ranges and the lowerer compares against the conservative end.
- A floating-point reduction: the region is splittable except for its
  carried state; whether the machine offers an explicitly reassociable
  form is a language decision, not this RFC's.
- Chunk ownership: a chunk's frame holds `Kind::Ref`s into the caller's
  storage and owned values it made; what a failed join drops, and in
  what order, is RFC-0057's unanswered question and is answered in
  layer 4's operation family.
- Nested regions: an inner splittable region inside an outer one; the
  lowerer splits at one level, and which level is a table-0 comparison.

## Order of work

1. Layer 0: the measurement, its storage and invalidation, and a target
   with no table running as today.
2. Layer 1: the invariant table over chains, read from RFC-0057's
   terminators and RFC-0064's loans; bound-check absorption as a row.
3. Layer 2–3: regions and their invariants; the divisible/indivisible
   verdict with its reason; `logs inline` as the first program whose
   loop is a region.
4. Layer 4: the split operation family and the lowerer's decision;
   `logs inline` divided; the spawn bench extended with a `Sync` split
   row and a Rust `rayon` twin.
5. The floating-point reduction decision; nested regions.
