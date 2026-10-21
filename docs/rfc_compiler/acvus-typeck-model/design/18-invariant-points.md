# 18 Own relations first, invariant points last (draft)

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
decided node whose type the default overrides is left as it is. It is to be a
reading, `InvariantPoints::Last`, beside the spec as stated,
`InvariantPoints::AsStated`.

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
  where that part is decided, a type the text or a declaration writes (l.218,
  l.289); and `?N ⇝ U`, at the boundary or at any part of it, where `?N` is `α`'s
  LitVar and `U` is decided (l.291). A decided use gives the invariant points of
  its candidate, and an undecided one those of each of its candidates (note 08,
  reading 13). Where `α` takes its constructor from its own relations, an invariant
  point of it relates its parts instead, as an equation does (l.165): an own
  equation where the part of `U` is an unknown, an invariant point where it is
  decided.
- A class's **own relations** are every relation into it but its invariant points:
  its lower bounds, decided ones among them (a use's result, a value the text
  writes), and what their parts give when split (l.284–289); the text's equations
  (l.236), and those an invariant part of a boundary gives with an unknown
  (`push`'s `&mut Vec⟨T⟩`); kinds; and LitVars.
- A class is **own-solved** where its own relations give it its constructor
  (l.275).
- A node **waits for** a class that is not own-solved, has invariant points, and
  has not yet been through row 3, where what it reads there is not decided by
  then: a use whose read leaves several candidates because of that class's
  position, and a lend, a projection or a binding where it reads whether the class
  is a reference (l.267), unless a use has recorded that at once (l.252).

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
   with a dependency on a universal is not taken: it stays in C, and is taken in
   each instance (l.219, l.222).
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
  other than it (an edge out of its component); a class with a universal's
  dependency typed in its body's group; a refusal while steps 1 and 2 still type a
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
- Note 14: an invariant point is a check of its boundary, and a row for row 3: each
  class row 3 typed has, among the types its invariant points give at the end,
  exactly that type fitting it, or several of a LitVar's family with that type its
  default.
- Note 15: "what a node records into" reads l.250 by relation; under this reading
  it is own relations, as row 2 has it. Row 3 decides classes, which
  I\* does not decide again (note 15, Not covered), so I\* reporting nothing is
  weaker evidence here than for nodes.

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

In each, I\* reports nothing and the certificate holds.

## Open

- **Not covered.** O1 (a binding's path), O2 beyond the cycles row 2 no longer
  makes, and O5 are not changed here.
