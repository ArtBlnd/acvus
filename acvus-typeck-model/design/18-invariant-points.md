# 18 Own relations first, invariant points last

Rests on the spec's l.74 (variance), l.86 (join), l.169 and l.176 (LitVars), l.218
(decided), l.226–238 (meeting, boundary, recording), l.240–271 (dependency),
l.273–293 (value), l.295–301 (fixpoint, fallback, acceptance) and l.335–344
(overload resolution); on `holes.md`, H3, H7 and O2; and on notes 08, 10, 14 and 15.
Its direction was agreed with 준량 on 2026-10-21: a type a declaration writes reaches
a class backward only where nothing else types it, since otherwise "함수 하나를
추가하면 완전히 다른 타입이 돼"; where it then leaves several types, it is refused,
and where none, the class falls back; a LitVar takes its family's default where that
is among several (choice (a): "Literal은 솔직히 편하게 가려면 이게 맞아"). Its open
questions were answered by 준량 on 2026-10-21: what reaches a class the rule typed
is "결정된 타입 그 자체 혹은 !"; an ambiguous class is refused once the rule types
nothing more, "일관성과 비순서 보장때문에"; the default comes after that, and a
decided node whose type the default overrides is left as it is. It is the reading
`InvariantPoints::Last`, beside the spec as stated, `InvariantPoints::AsStated`,
which stays the default. The rule is `src/invariant.rs`; where the other rows
change the code is under Code below.

## Meaning

The spec records every relation as a source as soon as it is given (l.228). A type
a declaration writes for a parameter then reaches the class of the argument passed
into it backward, through an invariant part (l.289) and through a LitVar (l.291),
and the equation it gives decides that class (l.279) even against what the text
gives it, which a conversion then reaches (Example 3). So every node at whose
position a class lies may change what another node reads there: two uses that read
one class depend on each other, and l.269 has each read without the other (H7;
Example 12's `x + y` and `y + 1`); and the order in which they are decided decides
the outcome wherever a reading leaves such a node out (`n(x); m(x)` under the
removed `Recording::Relation`, below).

Read by 준량's rule, a class takes its type from its own relations first. What a
declaration writes reaches it only where its own relations leave it without a type,
once nothing else can be solved, and then only as the one type that fits.

## Definitions

- An **invariant point** of a class `α` is an equation that an invariant part of a
  boundary `β ⇝ U` gives between `α`, a part of `β`, and the part of `U` there,
  where that part is closed, a type the text or a declaration writes in full, with
  no unknown in it (l.218, l.289); and `?N ⇝ U`, at the boundary or at any part of
  it, where `?N` is `α`'s LitVar and `U` is closed (l.291). A use decided as one
  candidate gives the invariant points of that candidate; one that is not, those
  of each of its candidates, all of them while it is undecided and the remaining
  ones where it is left unreached (note 08, reading 13). Where `α` takes its
  constructor from its own relations, an invariant point of it relates its parts
  instead, as an equation does (l.165), each again an invariant point, the parts
  of a closed type being closed.
- A class's **own relations** are every relation into it but its invariant points:
  its lower bounds, decided ones among them (a use's result, a value the text
  writes), and what their parts give when split (l.284–289); the text's equations
  (l.236), and those an invariant part of a boundary gives with an unknown
  (`push`'s `&mut Vec⟨T⟩`); kinds; and LitVars.
- A class is **own-solved** where its own relations give it its constructor
  (l.275).
- A node **waits for** a class that row 3 may still type, one not own-solved, not
  refused, and with an invariant point whose type fits it, where what the node
  reads there does not decide it: a use whose candidates remain several and differ
  at that class's position, and a lend, a projection or a binding where its value
  read decides neither by a constructor, nor by a LitVar, nor by a use's record at
  once (l.252) whether the class is a reference (l.267).

**Closed, where the draft said decided.** A type a declaration writes with a
variable in it, `push`'s `Vec⟨T⟩`, is decided as l.218 has it, and the draft's
own relations counted the equation it gives as own. Two of them, `Vec⟨T₁⟩` and
`Vec⟨T₂⟩` of two `push`es into one buffer whose own relations give it no
constructor, are not two types to choose between, but row 3's steps would count
them as two and refuse. Read as closed, an equation with a type that has an
unknown in it is own whole, the closed parts of that type with it, so in
`fill(&mut Vec⟨(u8, T)⟩)` the `u8` types such a buffer's element's first component
as its own relations do. This reading is the model's, taken from the draft's
`push` example, and is to be confirmed by 준량.

**Read as the rule reads it.** Whether row 3 may still type a class, its own
relations so far, its invariant points, and which of their types fit it, are read
in the value read leaving out nothing: each class solved by its sources whatever
its dependencies, as a node's value read solves it (l.267). In the state itself,
`X` of `x = 0` holds its LitVar only once `X` is solved (l.176), which a
dependency may hold back while a node's value read already has it. Run first with
whether a node waits read in its own value read, and row 3 in the state, H7 left
every node undecided: the lend `&x` waited for `X`, its value read giving `X` the
LitVar and `probe`'s invariant points, while in the state `X`, held back by its
dependency on `x + y`'s result, held no LitVar, so row 3 never took it. Two
changes followed together, that a LitVar decides that a class is no reference, and
the reading above. Run with the reading above and without the first, H7 reaches
one outcome, refused, every other test keeping its outcome; so the first is
needed with the second. Whether the first alone closes H7 is not run.

## Rows

1. **Recording.** Every relation is recorded as it is given (l.228). An invariant
   point is a check, not a source: it gives no class a value, and its equation is
   not recorded, until row 3 applies it.
2. **The own fixpoint.** Classes are solved, and nodes decided, as the spec states
   (l.240–293), by own relations alone. A class depends on a node at whose position
   it lies (l.244 (b)), and a node on the nodes that record into the classes it
   reads (l.247, l.250), only where that node may record an own relation into it.
   Joins raise by casts (l.86), and a coercion, a conversion among them (l.293),
   reaches only a class that is own-solved. A node that waits for a class is not
   decided. Where several candidates remain only because of a LitVar whose class
   has no invariant point, l.344 holds as stated.
3. **The invariant rule.** At each binding group's fixpoint, every class that is
   not own-solved, has invariant points, and has no unbroken dependency, in the
   graph of row 2 with an edge from each node to each class it waits for, is taken
   at once, from one state. Its candidates are the types its invariant points
   give, and of them it takes those that fit it, by its kind and its LitVar's
   family and values (l.169). Of these steps, the first that applies to some class
   is taken, and row 2 resumes:
   1. each class that exactly one candidate fits is unified with it;
   2. each class with a LitVar that several fit, its family's default among them,
      takes the default (l.344, moved here);
   3. each class that several fit is refused there, ambiguous.

   A class that no candidate fits is not taken, and is left to fallback. A class
   with an unbroken dependency on a universal, which depends on the use that makes
   an instance (l.246), is not taken: it stays in C, and is taken in each instance
   (l.219, l.222).
4. **Fallback.** At the run only, once row 3 takes nothing, as l.299 states; then
   rows 2 and 3 resume.
5. **What reaches a class row 3 typed.** Its type itself, or `!`: a value that
   enters it afterwards reaches it as it is, or is `!` (l.112), and no other row of
   the Coercion table reaches it.

Every invariant point is checked once everything is solved. Where its class was
own-solved, it is checked as part of the boundary it lies in, by coercion (l.293).

What each row excludes:

- Row 1: a declaration's written type giving a class a value before row 3.
- Row 2: a node decided before row 3 types a class it waits for; a node depending
  on another that records only invariant points into what it reads; a coercion
  into a class row 3 typed.
- Row 3: a choice among several by order; a class typed by its invariant points
  while a node that may still record an own relation into it waits on something
  other than it (an edge out of its component); a class with an unbroken
  dependency on a universal typed in its body's group; a refusal while steps 1 and 2 still type a
  class, and a default while step 1 still does. A decided node's invariant point
  at a class is among its candidates, so where that node's type fits, no other
  fitting type is chosen but by step 2: the choice is that type, the default, or a
  refusal. Where it is the default, the decided node's check fails, and the
  program is refused, as it is where no type holds both; I\* reports the decided
  node (P9 below).

The edge from a node to a class it waits for exists only for row 3: the node's own
waiting is not broken by l.269. In Example 13, `parse` waits for `R`, and `R`
depends on `parse`, whose result is an own relation of it; each is in the other's
component, so `R` has no unbroken dependency and row 3 types it `Vec⟨u8⟩`, and
`parse` then reads that. Were `R` typed while `parse` waited on another class
instead, `R` would take a declaration's type before its own.

Steps 1 and 2 are apart so that a use that still gives all its candidates at one
class narrows once another class is typed: in `let x = 0; let w = 0; g(x, w);`
with `g(u8, u8)` and `g(i32, &str)`, `W`'s candidates are `u8` and `&str`, of which
only `u8` fits its LitVar, and `X`'s are `u8` and `i32`, both fitting. Step 1
types `W` `u8`, `g` is then decided as `g(u8, u8)`, and `X` is `u8`. Were the
default taken with `W`, `X` would be `i32`, and `g` refused.

## Invariants

- **I2.** One outcome per program, whatever the order: row 3 and fallback take every
  class they apply to at once.
- **I\*** (note 15). A node decided before row 3 types a class it reads would not
  be decided otherwise reading that type: a node waits where what it reads there
  is not decided, and a use decided as one candidate gives row 3 that candidate's
  invariant points, so row 3 gives the class that type or refuses it; but where
  step 2 gives it the default instead, as row 3's exclusions state.
- **준량's rule.** A declaration's type decides a class only where its own relations
  give it none, and only as the one type that fits. Adding a declaration changes a
  type row 3 gives only into a refusal, but where the candidate added is a LitVar
  family's default (P7 below).
- **Soundness.** Every invariant point is checked (l.293), and every program accepted
  has every class typed and every node decided (l.301).

## What it omits

- **An equation that wins** (Example 3). A class its own relations type is not
  retyped by a declaration, so `push(&mut buf, 1u16); consume(buf)` with
  `consume(Vec⟨u8⟩)` is refused, though `u16 → u8` is declared. Borne by the
  program's author, who writes the type on the binding (`let mut buf: Vec⟨u8⟩`),
  which is the text's own; `1u16` then enters it by the conversion.
- **A LitVar's default added** (P7). Adding `f(i32)` to `f(u8)` turns `let x = 0;
  f(x)` from `x: u8` into `x: i32`. Only for a LitVar's class, and only where the
  candidate added is its family's default. Borne by whoever adds the overload, and
  agreed with 준량 as choice (a).

## What it changes in the spec

- l.228, l.279, l.289, l.291: an invariant point is no source until row 3.
- l.244 (b), l.247, l.250: the nodes that record into a class are those that may
  record an own relation into it.
- l.293: a coercion reaches only an own-solved class.
- l.297–299: row 3 at every group's fixpoint, before fallback.
- l.344: a LitVar's default among several candidates is taken in row 3.
- Example 3: refused with the conversion as without it.
- `holes.md`, H3: `take(&v)` with `v: Vec⟨u8⟩` is accepted, the boundary
  `&Vec⟨u8⟩ ⇝ &[u8]` reached by the view cast. `V` is own-solved by `make()`, so the
  target's equation `V = [u8]` is a check, not the value the spec gives.

## What it changes elsewhere

- `holes.md`, H7 and Example 12: what `probe` records into `X`, and what `x + y`
  records into `Y`, are invariant points, so neither `x + y` and `probe`, nor
  `x + y` and `y + 1`, depend on each other. A node that reads its own result
  through a class, `x + y` through `X`, still leaves its own relations out (l.267).
- The reading `Recording::Relation`, removed, approximated row 2's dependency
  under the spec as stated, but left out what l.291 makes a boundary record into a
  LitVar's class: `let x = 0; n(x); m(x);` with `n(u8)`, `n(u16)`, `m(u16)`,
  `m(&str)` reached two outcomes under it, `n(u16)` accepted and `n` refused, and
  one by position, `n` refused, I\* reporting completeness at `n`.
- Note 08, readings 11, 15 and 18; note 10, its rows of l.244 (b) and l.247–250.
- Note 14: an invariant point is a check of its boundary, which the checker
  checks as it checks every boundary; row 5 is a row of its contract, a value
  passing into a class row 3 typed holding only as that class's value or `!`
  (`DecidedBy::Rule`), which class row 3 typed being taken from the certificate.
  A row for row 3's choice, that each class it typed has, among the types its
  invariant points give at the end, exactly that type fitting it, or several of a
  LitVar's family with that type its default, is not added (Open).
- Note 15: "what a node records into" reads l.250 by relation; under this reading
  a use records into its result and into what its candidates would equate with an
  unknown, as row 2 has it, and an invariant point records into nothing. Row 3
  decides classes, which I\* does not decide again (note 15, Not covered), so I\*
  reporting nothing is weaker evidence here than for nodes.

## Code

- `Readings::invariant_points` (`src/program.rs`); `Relation::Invariant`, the
  pending check, and `Recorder::Invariant`, row 3's, which no value read leaves
  out (`src/relation.rs`); `ClassName::InvariantRule`, the class row 3 makes for
  the type it gives; `Refusal::Ambiguous`.
- Row 1: `State::split_parts` records an invariant point at an invariant part of
  a boundary, and at the reborrow row's targets, where the part of `U` holds a
  closed type (`State::closed`, the classes `instantiate`, `State::closed` and
  the text's closed classes make); `State::solve` records no l.291 equation into
  such a class; `relate_invariant_parts` relates an invariant point's parts once
  its class takes a constructor.
- Row 2: `may_record_own` and `would_record_own_by` (`src/node.rs`) give the
  dependency's edges of l.244 (b) and l.247–250 (`src/dep.rs`); `eligible` keeps
  out a node that waits (`decision_or_waits`, `waits`); `narrow` takes no default
  where a LitVar it reads lies in a class with invariant points; `has_work` counts
  no LitVar boundary into a closed type.
- Row 3: `State::apply_invariant_rule` (`src/invariant.rs`), taken where no step
  is left, in `fixpoints`, `explore_with`, `sample_with` and `State::run`, and,
  in the entry, before fallback. A universal depends on the use outside
  (`Edge::Chosen`), never decided in its group, so it stays unsolved there, and a
  class with an unbroken dependency on it is not unblocked, as in row 2 (P6).
- Row 5: `State::checks` checks a value entering a class row 3 typed by
  `as_it_is`.
- I\*: `records` and `into` (`src/istar.rs`) as note 15 above.

## Expected results

Derived before running, under `InvariantPoints::Last`; under
`InvariantPoints::AsStated` every current test keeps its outcome.

| Program | Expected |
|---|---|
| P1 `let x = 0; let y = x + 1;`, `core::add` for `u8`, `u16`, `i32` | one outcome, `x: i32`, `add(i32)`: row 3 takes the default |
| P2 `let x = 0; takes_u8(x);` | `x: u8` |
| P3 `let x = 0; m(x);`, `m(u16)`, `m(&str)` | `x: u16` |
| P4 `let x = 0; n(x);`, `n(u8)`, `n(u16)` | refused, ambiguous at `X` |
| P5 `let x = 0; n(x); m(x);` | refused, ambiguous at `X`, in every order |
| P6 `fn g(a) { let mut o = Some(a); fill(&mut o); }`, `fill(&mut Option⟨u8⟩)`, called `g(1u8)` or `g(1u16)` | `O`'s payload, whose lower bound `S` lies on the universal `A`, stays in `g`'s C, row 3 not taking it; `g(1u8)` accepted, `g(1u16)` refused at the instance's `&mut o ⇝ fill.Param(0)`; with `Some(0)` for `Some(a)`, `O: Option⟨u8⟩` by row 3 in `g`'s C, both accepted (added after the rest of the table was run; its control first derived as `Some(?s)`, below) |
| P7 `let x = 0; f(x);`, `f(u8)`, `f(i32)` | `x: i32` |
| P8 `let x = 0; let w = 0; g(x, w);`, `g(u8, u8)`, `g(i32, &str)` | `x: u8`, `w: u8`, `g(u8, u8)`: `W` by step 1, then `X` |
| P9 `let x = 0; m(x); f(x);`, `m(u16)`, `m(&str)`, `f(u8)`, `f(i32)` | refused: `X` takes `i32`, and `m`, decided, fails; I\* reports `m` |
| Example 2 | `a: Vec⟨u8⟩`, by row 3 at its element |
| Example 2 with `a.push(0u16)` before `fill_all` | refused at `fill_all`: `&mut Vec⟨u16⟩ ⇝ &mut Vec⟨u8⟩` |
| Example 3 | refused, with the conversion and without |
| Example 4 | as under the spec, `x: Option⟨!⟩`: a covariant part's boundary is no invariant point |
| Example 12 | one outcome, every class `u16`; `?b` by row 3; `x + y` and `y + 1` do not depend on each other |
| Example 13 | line 1 `parse → Vec⟨u8⟩` once row 3 types `R`; line 2 refused, its `R` having no invariant point |
| Example 14 | to run, sampled |
| H3 | accepted, `v: Vec⟨u8⟩`, the boundary by the view cast |
| H4 | `v: Vec⟨u8⟩`, by row 3 at its element |
| H7 | one outcome, accepted, `probe(u16)` |
| the other current examples | as under the spec |

In each, the certificate of every final state holds, and I\* reports nothing but
`m` in P9.

## Results

Run, every row of the table holds as derived, each a test in `examples.rs`: P1
`a_literal_added_to_a_literal`, P2 `a_literal_passed_to_one_parameter`, P3
`a_literal_passed_to_overloads_of_which_one_fits`, P4
`a_literal_passed_to_overloads_that_both_fit`, P5
`a_literal_read_by_two_uses_that_leave_two_types`, P6
`a_class_on_a_universal_is_typed_in_each_instance`, P7
`a_literal_passed_to_overloads_with_the_default`, P8
`a_literal_typed_first_narrows_the_use_another_reads`, P9
`a_default_where_a_decided_use_gives_another_type`, and the examples' and holes'
own tests, each with a half read this way (Example 2 with the push,
`example_2_with_a_push_before_fill_all`). That Example 12's `x + y` and `y + 1` do
not depend on each other is not asserted; its outcome is. Example 14, sampled,
reaches the spec's outcome.

Three results the table did not give:

- **O4 under the spec's captures** (`holes.md`, O4) reaches four outcomes, where the
  spec reaches three: `X` depends on `probe` only where `probe` may record an own
  relation into it, and a boundary into `A` or `B` is none, so `X` may be solved
  before `probe` and before the call's instance, and `probe`, decided after the
  instance, reads `B`: `probe(B)`, refused with `x: A`. The table's "the other
  current examples, as under the spec" was wrong there. By the captured
  environment, one outcome, as under the spec.
- **P4, first run**, had I\* report `n`: decided unreached once row 3 refused `X`,
  it gave no invariant point at the end, so I\* read `X` with none and took the
  default. A use left unreached gives its remaining candidates' points (note 08,
  reading 13), as I\* already reads what it records into; so read, I\* reports
  nothing.
- **A lower bound with no source.** `let mut x = None; fill(&mut x);` with
  `fill(&mut Option⟨u8⟩)`, accepted under the spec with `x: Option⟨u8⟩`, is
  refused (`a_none_lent_to_a_filled_option`): `X`'s payload has the invariant
  point `u8` and the lower bound `?k`, which has no source and no invariant
  point. The payload depends on `?k` (l.244), unsolved, so row 3 does not take
  it, and takes nothing; fallback then gives `?k` `!` (l.299), the payload is `!`
  by its own relations, and the boundary fails. P6's control as first derived,
  `Some(?s)` in `g`, `?s` with no source, was refused at the same boundary, in
  the instance, `S` and `O` left without a value in `g`'s C; that the payload is
  `!` there was not shown, the instance's values not being read. With the LitVar
  of `Some(0)` there, settled (l.176), the one difference, row 3 takes the
  payload. So
  fallback reaches a class row 3 would type, through its lower bound, before
  row 3 does, against the order of rows 3 and 4.

With the default swapped to `InvariantPoints::Last`, run before P6 and the lower
bound with no source were added, 104 of 108 tests pass; the four failing are the
spec's halves of Example 3, H3 and H7, and O4's under the spec's captures, each
the change above.

## Open

- **Closed or decided** (Definitions): to be confirmed by 준량.
- **A certificate row for row 3's choice** (note 14): the invariant points of a
  class row 3 typed lie at parts, `A`'s element in Example 2, which the checker,
  on closed values, does not name. Checking the choice needs the checker to name
  classes at parts, which changes what note 14 keeps it from; to be agreed.
- **O4 under the spec's captures**, one outcome more (Results).
- **A lower bound with no source** (Results): row 3 waits on a class only
  fallback can type, so fallback comes first there; to be decided with 준량.
- **Not covered.** O1 (a binding's path), O3 (a reference's mutability), O2 beyond
  the cycles row 2 no longer makes, and O5 are not changed here.
