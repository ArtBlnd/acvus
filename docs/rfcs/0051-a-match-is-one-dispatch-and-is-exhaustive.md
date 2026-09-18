# RFC-0051: a `match` is one dispatch, and it is exhaustive

Status: Accepted — owner and coordinator, 2026-09-19
Extends: RFC-0036 (variants), RFC-0039 (an option is its payload),
RFC-0043 (types as written), RFC-0044 (a body is prepared once),
RFC-0045 (one statement grammar)

## Problem

The language has no `match`. A value with several variants is tested
with the `MatchBind` statement, `Pattern = Expr { body };`, one per
variant:

```acvus
E::A(v) = e { acc = acc + v; };
E::B(v) = e { acc = acc + v; };
```

Measured 2026-09-18 (`benches/shapes.rs`, `enum match`): 39.4 ns per
iteration, 106× the Rust `match` — two `test_variant`, three `diamond`,
a `ref_var` per arm, and every tag tested every iteration. And the form
is not a match at all: nothing says the arms are exhaustive (a value no
arm takes passes silently), and the statement unifies the pattern's
variant set with the source's, so `E::A(v) = e` on `e: E{B(i64)}` is a
**type error** (`pattern type E{A(!)} incompatible with source type
E{B(i64)}`), not an untaken arm. The owner (21:20): the form is wrong;
replace it with `match`, and check exhaustiveness in `validate`.

Exhaustiveness is hard here for one reason: the language's enums are
structural and unify by union (memory `structural-types-unify-by-union`,
2026-09-16), so a value's variant set can grow from anywhere the value
flows, and a check at the `match` site does not in general see the
closed set. The full answer — the closed type after solving, Maranget's
matrix over nested patterns — is a later stage. This RFC fixes the
**boundary** of what is decided now, and leaves the place where the
rest goes.

## Decision

1. **Syntax.** Rust's: `match e { P1 => e1, P2 => { stmts; tail }, _ =>
   e3 }` is an expression; every arm has the type of the whole, `!`
   admitted. Arms are patterns the grammar already has (`Pattern`:
   variant, literal, binding, tuple, object, list) plus `_`. The
   `MatchBind` statement is **removed** from the grammar
   (`grammar.lalrpop:149`, `expr_to_pattern`, `Stmt::MatchBind`, its
   typeck and lowering, Grammar.md line 81); every use in the tree moves
   to `match` or `if let`. `if let P = e { .. } else { .. }` stays as
   the two-arm sugar it already is.
2. **An arm contributes no variant.** The scrutinee's type is its own;
   a pattern naming a variant the scrutinee cannot hold is an
   **unreachable arm** and is refused (the language has no warning
   axis; owner 21:25): `unreachable pattern: `E::C(_)` is not a variant
   of `E{A(i64), B(i64)}`` — types as written (RFC-0043). This replaces
   today's "pattern type incompatible" unification error.
3. **Exhaustiveness is decided in `validate`, on the MIR, and only
   where the variant set is known.** A `match` is exhaustive when one
   of:
   - (a) it has a catch-all arm (`_` or a bare binding);
   - (b) the scrutinee is an `Option` or a `Result` — two variants,
     always known (RFC-0039's in-value option included);
   - (c) the scrutinee's enum is **locally closed**: the scrutinee is a
     storage of this body (or an expression of such) whose every
     definition in this body is a `MakeVariant` (or a join of them),
     and which **does not come from or go outside** — not a parameter,
     a capture, a context (`@x`), an extern call's result, a field read
     of a value from outside; and not captured by a closure or passed
     as an argument before the match. Then the variant set is the union
     of the local constructors and the arms' top-level tags must cover
     it.
   Otherwise the `match` is refused: `non-exhaustive match: the variants
   of this value are not known in this function; add a `_` arm`.
   Nested patterns: each position is asked the same question; a
   position whose set is not known needs `_` there.
4. **The place left for the rest.** The pass asks one function,
   `known_variants(scrutinee) -> Known::{Closed(set), Open}`; today (b)
   and (c) answer `Closed`, everything else `Open`. The close-phase
   answer (the solver's closed type after `closed_or_reported`) and
   Maranget's matrix over nested positions widen `Closed` later and
   change neither the rule's shape nor its place. Not built now: the
   owner (21:30) — doing the outside-flowing and nested cases now
   costs more time than the stage is worth.
5. **MIR and machine.** `Terminator::Switch { tag: ValueId, arms:
   Vec<(Tag, Label, Vec<ValueId>)>, default: Option<(Label,
   Vec<ValueId>)> }` — the tag read once; `default` present exactly when
   an arm is a catch-all. A flat `Option`/`Result` scrutinee keeps
   `TestVariant` (its tag is the value's kind, RFC-0039); a boxed
   variant gets `Switch`. The interpreter prepares `Switch` as one
   `switch` operation: read the tag, jump through a table (`br_table`
   shape; the kovac reflection of 2026-09-18). Arm bindings are
   `UnwrapVariant` in the arm, as today.

## What it costs

- A grammar removal: every `MatchBind` in tests, goldens, scripts,
  docs migrates. `if let` covers the one-arm uses; `match` the rest.
- A `Switch` terminator through every CFG consumer (`cfg.rs`,
  `code_motion`, `ssa`, `drop_insertion`, the printer, `prepare`).
- A validate pass with a def-use walk for (c). Its refusals name what
  they checked; a `_` arm is always the way through.
- Programs that matched a value from outside without a catch-all are
  refused until the later stage — by design: a `_` states the intent
  the checker cannot yet verify.

## Rejected

- **Folding `MatchBind` chains into `Switch` without exhaustiveness**:
  one dispatch, but the silent fall-through and the unification error
  stay — the owner refused the form itself.
- **Exhaustiveness in `typeck`**: the union is open while checking; a
  decision there is either unsound (assumes closed) or refuses
  everything (assumes open). `validate` runs on lowered, typed MIR and
  can see the body's definitions; the closed-type answer comes from the
  solver's close phase later — also not typeck.
- **Full Maranget + close-phase now**: the right end state, deferred for
  time (owner, 21:30); the `known_variants` seam is where it lands.
- **Arms contributing variants** (the scrutinee's type widened by the
  patterns): makes every match exhaustive by construction and
  exhaustiveness meaningless; Rust's rule kept.
- **Warning for an unreachable arm**: the language has no warning
  axis; a refusal states the defect where a warning would be ignored.

## Consequences

- RFC-0050 (object/enum layout) builds on `Switch`: a `Variant` with a
  tag word and a payload slot is what `switch` reads.
- The later exhaustiveness stage is one function's widening.

### What landed

**The compiler.** `match` is in the grammar and `MatchBind` is gone, with
every use in the tree moved to `match` or `if let`. The arm rule (§2) is in
`typeck`; the exhaustiveness pass (§3-§4) is `validate::exhaustive`, run in
pass 0 of `graph::optimize`, where it reads `InstKind::Switch` before any
pass has moved it. `lower_match_expr` writes the `Switch` and no chain: the
chain the tag form used to emit does not name a `match` — it is the shape an
`if let` without an `else` writes, and a pass that refused on it would refuse
every `if let`.

**The machine.** `acvus-interpreter/src/ops/switch.rs` holds three
operations, and `prepare::switch_op` chooses between them by the
scrutinee's form. Each is a terminator: it returns the `BlockId` the machine
enters and holds no successor (RFC-0052 §1).

- A flat `Option` or `Result` carries its tag in the value's own kind
  (RFC-0039), so `SwitchOption` and `SwitchResult` *are* that one test, with
  each side's block — the catch-all included — resolved at preparation.
- A boxed variant's tag is an `Astr`, and `Switch` reads it once and scans
  the arms.

`default` is always a real edge: the catch-all where the `match` wrote one,
and otherwise the last arm, which is then the one tag left untested. That is
sound exactly because `validate::exhaustive` decided some arm holds, and it
is why no `run` decides that no arm holds and none reads a variant count.

**Not the table §5 names.** A value's tag today is an interned name, not an
ordinal, so nothing can be indexed by the tag itself; the nearest form is a
hashed table whose lookup is a multiply and a dependent load. Built and
measured at the seven arms of `bf table`, it cost **24.4 ns** a step against
the scan's **23.3** (three pinned reps, ranges apart), and it had no other
site in the tree — so it was cut. The table is RFC-0050's, where a tag word
gives an ordinal to index by and the variant count gives the bound.

**RFC-0053 owns the enum that never exists.** `optimize::sroa` replaces a
locally built enum with a tag register and a payload register, and a
`Switch` reading such a slot has no value left to read a tag from.
`sroa::dispatch_for` covers the two-edged dispatch — the compare against the
number the arm's tag stands for, which is what the pass already makes of a
`TestVariant` — and refuses the slot for a dispatch of three or more edges,
which keeps the enum built and hands the machine's `Switch` its tag.

**Measured** (2026-09-19, one pinned core, three alternating reps against
`b8b9e845`, medians):

| case | n | before | after |
| --- | --- | --- | --- |
| `bf table` | 1e6 | 34.6 ns/step | **22.9** |
| `bf table` | 5e6 | 35.3 | **23.4** |
| `bf scan` | 1e6 | 55.7 | **36.5** |
| `bf call` | 1e6 | 39.8 | **27.8** |
| `enum match` | 1e5 | 9.8 ns/iter | 9.8 |
| `enum match` | 1e6 | 9.9 | 9.9 |

`bf table`'s seven `TestVariant` + seven `JumpIf` (blocks 4–10 of the base
listing) are one `Switch<true>` in block 4, and the body falls from 21
prepared blocks to 15. Per step at n=1e6: branches 90.1 → 75.9, mispredicts
0.354 → 0.319, instructions 603 → 489, cycles 230 → 158 — the program is
periodic, so the dispatch was predicted before and is predicted now, and
what fell is the work rather than the misses. `bf scan` and `bf call` move
for the same reason. `field read`, `field write`, `construct`, `option match`,
`vec of objects` and every case of `benches/accum.rs` — `grade`, `branch`,
`collatz` among them — are unchanged.

`enum match` does not move, and its prepared listing is byte-identical to
`b8b9e845`'s: RFC-0053 had already removed the enum, so no `Switch` reaches
the machine there. What is left is the compare at the merge that RFC-0053's
own Consequences names — the tag is a constant on each incoming edge, and
reaching the arm directly on each edge is jump threading through the phi. No
`dce` or `fold` rule does that today, and it is the next to-be, not this
one.

**Not decided here**: a `match` whose arms are not one dispatch over a tag
(a literal arm, a nested refutable payload) has no `Switch` to read, so
`typeck` refuses it outright unless it has a `_` arm — the same sentence the
`Open` case writes. Nested positions are not asked separately; the `_` is
asked for at the `match`.
