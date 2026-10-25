# 18 Own relations first, invariant points last (draft, superseded)

**Status.** The reading below was not implemented. A type a declaration writes is a
part toward which a decomposition records a coercion constraint, and a coercion constraint decides no value: what
this draft calls an invariant point is such a coercion constraint, a check by coercion, and no
source. An equivalence class nothing enters takes the one type its upper
bounds give, has no value where they give two, and otherwise falls back. A fill is a
step like a solve, taken once its upper bounds have values, not a row after an own
fixpoint. Where its upper bounds give two types, the class has no value: no candidate is
chosen by what fits it, nor a LitVar's default among several (row 3, steps 1 and 2),
and a class only LitVars enter is unified by a decomposition's `?N ⇝ U` instead. P1–P9 have
no case. The rest of the note is the draft as written.

A type a declaration writes reaches a class backward only where nothing else types
it; where it then leaves several types, it is refused, and where none, the class
falls back; a LitVar takes its family's default where that is among several. An
ambiguous class is refused once the rule types nothing more; the default comes after
that, and a decided disjunctive constraint whose type the default overrides is left as it is. It is to
be a reading, `InvariantPoints::Last`, beside `InvariantPoints::AsStated`.

## Meaning

A class takes its type from its own relations first. What a declaration writes
reaches it only where its own relations leave it without a type, once nothing else
can be solved, and then only as the one type that fits.

## Definitions

- An **invariant point** of a class `α` is an equation that an invariant part of a
  coercion constraint `β ⇝ U` gives between `α`, a part of `β`, and the part of `U` there,
  where that part is decided, a type the text or a declaration writes; and `?N ⇝ U`, at
  the coercion constraint or at any part of it, where `?N` is `α`'s
  LitVar and `U` is decided. A decided use gives the invariant points of
  its candidate, and an undecided one those of each of its candidates. Where `α` takes
  its constructor from its own relations, an invariant
  point of it relates its parts instead, as an equation does: an own
  equation where the part of `U` is an unknown, an invariant point where it is
  decided.
- A class's **own relations** are every relation into it but its invariant points:
  its lower bounds, decided ones among them (a use's result, a value the text
  writes), and what their parts give when decomposed; the text's equations,
  and those an invariant part of a coercion constraint gives with an unknown
  (`push`'s `&mut Vec⟨T⟩`); structures; and LitVars.
- A class is **own-solved** where its own relations give it its constructor.
- A disjunctive constraint **waits for** a class that is not own-solved, has invariant points, and
  has not yet been through row 3, where what it reads there is not decided by
  then: a use whose read leaves several candidates because of that class's
  position, and a lend, a projection or a binding where it reads whether the class
  is a reference, unless a generalization has recorded that at once.

## Rows

1. **Recording.** Every relation is recorded as it is given. An invariant
   point is a check, not a source: it gives no class a value, and its equation is
   not recorded, until row 3 applies it.
2. **The own fixpoint.** Classes are solved, and disjunctive constraints decided, by own relations
   alone. A class depends on a disjunctive constraint at whose position it lies, and a disjunctive constraint on the
   disjunctive constraints that record into the classes it reads, only where that disjunctive constraint may record an
   own relation into it. Joins raise by casts, and a coercion, a conversion among
   them, reaches only a class that is own-solved. A disjunctive constraint that waits for a class is
   not decided.
3. **The invariant rule.** At each binding group's fixpoint, every class that is
   not own-solved, has invariant points, and has no unbroken dependency, in the
   graph of row 2 with an edge from each disjunctive constraint to each class it waits for, is taken
   at once, from one state. Its candidates are the types its invariant points
   give, and of them it takes those that fit it, by its structure and its LitVar's
   family and values. Of these steps, the first that applies to some class
   is taken, and row 2 resumes:
   1. each class that exactly one candidate fits is unified with it;
   2. each class with a LitVar that several fit, its family's default among them,
      takes the default;
   3. each class that several fit is refused there, ambiguous.

   A class that no candidate fits is not taken, and is left to fallback. A class
   with a dependency on a universal is not taken: it stays in C, and is taken in
   each instance.
4. **Fallback.** At the run only, once row 3 takes nothing; then rows 2 and 3 resume.
5. **What reaches a class row 3 typed.** Its type itself, or `!`: a value that
   enters it afterwards reaches it as it is, or is `!`, and no other row of
   the Coercion table reaches it.

Every invariant point is checked once everything is solved. Where its class was
own-solved, it is checked as part of the coercion constraint it lies in, by coercion.

What each row excludes:

- Row 1: a declaration's written type giving a class a value before row 3.
- Row 2: a disjunctive constraint decided before row 3 types a class it waits for; a disjunctive constraint depending
  on another that records only invariant points into what it reads; a coercion
  into a class row 3 typed.
- Row 3: a choice among several by order; a class typed by its invariant points
  while a disjunctive constraint that may still record an own relation into it waits on something
  other than it (an edge out of its component); a class with a universal's
  dependency typed in its body's group; a refusal while steps 1 and 2 still type a
  class, and a default while step 1 still does. A decided disjunctive constraint's invariant point
  at a class is among its candidates, so where that disjunctive constraint's type fits, no other
  fitting type is chosen but by step 2: the choice is that type, the default, or a
  refusal. Where it is the default, the decided disjunctive constraint's check fails, and the
  program is refused, as it is where no type holds both; I\* reports the decided
  disjunctive constraint (P9 below).

The edge from a disjunctive constraint to a class it waits for exists only for row 3.

Steps 1 and 2 are apart so that a use that still gives all its candidates at one
class narrows once another class is typed: in `let x = 0; let w = 0; g(x, w);`
with `g(u8, u8)` and `g(i32, &str)`, `W`'s candidates are `u8` and `&str`, of which
only `u8` fits its LitVar, and `X`'s are `u8` and `i32`, both fitting. Step 1
types `W` `u8`, `g` is then decided as `g(u8, u8)`, and `X` is `u8`. Were the
default taken with `W`, `X` would be `i32`, and `g` refused.

## Invariants

- One outcome per program, whatever the order: row 3 and fallback take every class
  they apply to at once.
- **I\***. A disjunctive constraint decided before row 3 types a class it reads would not
  be decided otherwise reading that type: a disjunctive constraint waits where what it reads there
  is not decided, and a use decided as one candidate gives row 3 that candidate's
  invariant points, so row 3 gives the class that type or refuses it; but where
  step 2 gives it the default instead, as row 3's exclusions state.
- A declaration's type decides a class only where its own relations give it none, and
  only as the one type that fits. Adding a declaration changes a type row 3 gives
  only into a refusal, but where the candidate added is a LitVar family's default (P7
  below).
- **Soundness.** Every invariant point is checked, and every program accepted
  has every class typed and every disjunctive constraint decided.

## What it omits

- **An equation that wins**. A class its own relations type is not
  retyped by a declaration, so `push(&mut buf, 1u16); consume(buf)` with
  `consume(Vec⟨u8⟩)` is refused, though `u16 → u8` is declared. Borne by the
  program's author, who writes the type on the binding (`let mut buf: Vec⟨u8⟩`),
  which is the text's own; `1u16` then enters it by the conversion.
- **A LitVar's default added** (P7). Adding `f(i32)` to `f(u8)` turns `let x = 0;
  f(x)` from `x: u8` into `x: i32`. Only for a LitVar's class, and only where the
  candidate added is its family's default. Borne by whoever adds the overload.

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

In each, I\* reports nothing and the certificate holds.
