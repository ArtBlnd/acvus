# RFC-0051: a `match` is one dispatch, and it is exhaustive

Status: Draft — owner and coordinator, 2026-09-18
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

- `match` is in the grammar and `MatchBind` is gone, with every use in the
  tree moved to `match` or `if let`. The arm rule (§2) is in `typeck`; the
  exhaustiveness pass (§3–§4) is `validate::exhaustive`, run in pass 0 of
  `graph::optimize`, where it reads `InstKind::Switch`.
- **The lowering emits `Switch`, and one pass expands it.** The chain the
  tag form used to emit does not name a `match`: it is the same shape an
  `if let` without an `else` writes, and a pass that refused on it would
  refuse every `if let`. So `lower_match_expr` writes the `Switch` — the one
  shape that names a dispatch — and `optimize::switch_expand`, the first
  step of pass 1, replaces every one with the `TestVariant` + `JumpIf` chain
  before any other pass or the interpreter sees it.
- **Measured** (`benches/shapes.rs`, `enum match`, n = 1e5 and 1e6): 39.4 ns
  per iteration before, **36.5–36.8 ns** now, 99× Rust. The chain is one
  `test_variant` per arm but the last, which is the chain's else, so the two
  arms of `E{A, B}` cost one test where the tag form cost two, and one
  diamond where it cost two.
- A `match` whose arms are not one dispatch over a tag — a literal arm, a
  nested refutable payload — has no `Switch` to read, so `typeck` refuses it
  outright unless it has a `_` arm, which is the same sentence the `Open`
  case writes. Nested positions are not asked separately; the `_` is asked
  for at the `match`.
- RFC-0050 (object and enum layout) builds on `Switch`: a variant with a tag
  word and a payload slot is what `switch` reads.
- The later exhaustiveness stage is one function's widening.

## What is left

The machine's `switch` operation: delete the `optimize::switch_expand::run`
call at the top of `graph::optimize::run_pass1_body`, delete the pass, and
add the `switch` handler in `acvus-interpreter/src/prepare.rs` — the `todo!`
arm in `Prepare::op`, plus `is_straight_line`, which already answers
`false`. The two artifacts move together.
