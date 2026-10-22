# 17 The captured environment

Rests on the spec's l.188 (the captured environment), l.198 (a call of a function
value), l.205–217 (binding groups, C, instances), l.221 (meeting), l.240 (b) (what a
class depends on), l.263, l.282 and l.292; on `docs/rfc_compiler/functions.md` §2 (a function value bound by `let`
generalizes what its type leaves unconstrained, "except what the enclosing scope or
the captured environment holds: those belong to where they were written"; its
environment is captured once, where it is written) and `types.md` §1 (every slot
has one type; values meeting through a chain of slots are of one type); and on
`holes.md`, H5 and O4. Its direction and its rows were agreed with 준량 on
2026-10-21, as a reading, `Captures::Environment`, beside the spec as stated then,
`Captures::AsStated`. The text took them at 8978eb70, as 준량 decided (l.23, l.188,
l.198, l.211, l.212, l.240), and both readings are removed with it: the
environment is cut always. The code
is `Program::cut_environments` and `Program::owner`, `Program::item_group` and
`Program::is_universal` in `src/program.rs`, `State::positions` and
`State::environment_of` in `src/node/position.rs`, and the certificate's contexts and
`call_value` in `src/certificate.rs`.

## Meaning

Before 8978eb70 the spec solved a lambda's group before its enclosing one, and
copied all of its C, the captures shared, at each call of its value (l.215, l.217).
What the lambda recorded into a capture was then recorded only by the calls'
instances, at none of their positions, so the capture and what reads it were
decided before or after it by order (O4). A node of the lambda that read a capture
read it before the enclosing group recorded into it, and where that decided
nothing it was decided in each copy (H5, which is not a hole).

By l.188, as functions §2 has it, the part of a lambda's text that its captured environment
holds is not generalized: it belongs to the enclosing group and is solved there,
once. Only the rest is the lambda's binding group, generalized and copied at each
call.

## Definitions

- A **capture** of a lambda is a class of an enclosing group that its text names
  (l.212).
- Two classes of a lambda's text are **joined** where one item of the text names
  both, or one is a part of a kind the text writes for the other. The
  **environment** of a lambda is every class joined, directly or through others, to
  a capture, and every item that names one of them. A lambda inside it records its
  own environment into it first, as items of its text.
- An item **names** the classes it relates. A use of a function of an enclosing
  group names that function's parameter and result types as well, which it takes
  as they are (note 09, reading 4), and a binding names the classes of the
  projections on its path (l.308).
- The **generic part** of a lambda is the rest of its text, its parameters among it
  where they are not in the environment.

## Rows

1. **The environment is the enclosing group's.** Its classes are classes of that
   group, each an unknown of it unless the text writes it (l.213). Its items are
   recorded there, and its nodes made there, once, when that group's solving
   starts, whether or not a call reaches the lambda.
2. **The lambda's binding group is its generic part** (l.205). Its C is the generic
   part's, copied at each instance (l.214, l.217). It names no capture, so no
   instance records into one.
3. **A call of a function value meets the environment where it is.** Each argument
   meets the copy of its candidate lambda's generic parameter, or that lambda's
   parameter in the environment, an unknown of the call's group; the call's result
   takes the copy of a generic result type, or the environment's. Within the depth
   its candidates reach (l.240 (b)), the call's positions include the parameters
   and the result type in the environment of each lambda its callee's Values name
   at the moment.

What each row excludes:

- Row 1: the environment generalized or copied, and a capture taking a relation
  from an instance. A lambda no call reaches still has its environment recorded,
  and the nodes there are decided (l.292 holds of them).
- Row 2: a class of an enclosing group in a lambda's C.
- Row 3: a parameter in the environment solved before a call whose Values name its
  lambda is decided.

## Invariants

- **I\* at a capture.** No relation into a capture is recorded after a decision that
  reads it, outside a cycle (l.265; types §6: "A decision reads a type only once
  nothing can change it").
- **I2 at the witnesses.** H5 and O4 reach one outcome each.
- **Generalization** leaves out the environment (functions §2), and every capture
  has one type (types §1).
- **Soundness.** Every item of a lambda's text is recorded, in the environment once
  or in each instance, and checked as before (l.282).
- **Conservative refinement.** A later, finer environment, and the phases agreed
  first as the fall-back (O4's capture closed before the instance, its relations
  checked), are judged by whether every program this reading accepts keeps its
  typing.

## What it omits

- **A parameter the environment holds is not polymorphic.** Every call shares its
  one type. What such a parameter is given reaches a capture's one type anyway
  (types §1), so the loss is only where calls would type it differently below that
  type: `|x| { last = Some(x); probe(x) }`, called with `A` and with `B` under a cast
  `A → B`, decides `probe(B)` once instead of `probe(A)` and `probe(B)`. Borne by the
  program's author, who passes the capture as a parameter where a per-call typing
  is wanted.
- **Joining is coarser than what flows.** It joins the classes an item names,
  whatever the direction, so a parameter can be held where none of its information
  reaches a capture: `|x| { let y = cap; y = x; }`. Reading the direction is a later
  refinement, under the conservative criterion.

## What it changes elsewhere, adopted

- Note 09, readings 7–9: a lambda's group, meeting and boundary, and a call of a
  function value, by rows 1–3.
- Note 10: a call's positions, by row 3.
- Note 14: the environment's items are checked in the enclosing group's context.
- Note 15: the environment's nodes are the enclosing group's, and the check's own
  instance copies the generic part.

## Expected results

Derived before running, under `Captures::Environment`, before the text took it;
under `Captures::AsStated` every test of then keeps its outcome.

| Program | Expected |
|---|---|
| H5 | one outcome, accepted, `probe(A)` |
| O4 | one outcome, accepted, `x: B`, `probe(B)` |
| O4 without `probe` | one outcome, accepted, `x: B` |
| Example 6 | one outcome, `last: Option⟨u8⟩` |
| Example 6 with `remember(1u8); remember(2u16)` | refused: `u8 ⊔ u16` has no join, as under the spec |
| Example 6 through a reference, `let r = &mut last; let remember = \|x\| *r = Some(x);` | one outcome, `last: Option⟨u8⟩` |
| `let x = 0; let f = \|\| takes_u8(x); f();` | one outcome, `x: u8` |
| `let mut buf = Vec::new(); let fill = \|\| fill_all(&mut buf); fill();` | one outcome, `buf: Vec⟨u8⟩` |
| `let x = "s"; let f = \|\| takes_u8(x);`, never called | refused at `takes_u8`, which the spec leaves in C and accepts |
| `let mut acc = 0; let f = \|\| acc + 1; acc = f();` | one outcome, `acc: i32` |
| `a_call_left_out_leaves_out_its_instance_relations` | `choose(B)` in every order: `make_b() ≤ X` is the environment's, recorded where it is written, so no read leaves it out with the call |
| the other current examples | as under the spec |

In each, I\* reports nothing and the certificate holds.

## Run

Every row of the table held as derived under `Captures::Environment`, each
reaching one outcome in every order, I\* reporting nothing, and the certificate
holding of each accepted final state (a refused one has nothing to check, note 14).

Since 8978eb70 the environment is cut always. The cases under
`cases/environment/` that had a twin of one name asserting the spec's outcome of
then, under `cases/spec/`, `cases/holes/` and `cases/istar/`, are in that twin's
place, the twin removed: H5, both of O4, Example 6 and Example 6 called with two
types, the lambda never called, and the call left out under `Deferral::Reading`.
Each keeps its outcome. Five cases written by the text of then changed:

- `rules/classes/a-a-capture-shared-by-two-instances` and
  `rules/solving/a-two-instances-meet-a-capture-written-by-name`, Example 6 called
  with `1u8` and `2u16`, were refused at `x`'s type, which both calls share, and no
  longer at `last`'s payload (What it omits). Being then one program with one
  derivation with `spec/example-06-called-with-two-types`, they are merged into
  it, which asserts the refusal at `x`'s type.
- `rules/solving/a-write-through-a-captured-reference-no-coercion-reaches` is
  refused at the same boundary, now the enclosing group's.
- `rules/classes/b-a-lambda-calling-a-captured-lambda` and
  `spec/example-07-a-write-through-a-captured-reference` keep their outcomes, their
  calls and writes decided and checked in the enclosing group.

## Open

- A function value that leaves the group its lambda lies in, returned from a
  function, reaches calls whose group copies the environment with that function's
  C. Function values crossing groups are not yet encoded (note 09).
- A call's candidates are the Values of the moment. Whether Values can still grow
  after a call is decided is not shown: in every witness the lambda reaches the
  callee's class by a pass recorded when the group's solving starts, so no call's
  Values grow after it. The certificate reports a call whose callee's Values name
  a lambda it does not call (note 14), so a final state where they grew fails
  there; no case shows one yet.
- Row 3 makes a node's result reach what it reads through a call. In
  `acc = f()` with `f = || acc + 1`, `acc + 1`'s result `R` reaches `ACC`, which it
  reads, by `R ≤ λ result ≤ f() ≤ ACC`. Probed in every order by changing the
  model, `f()` is decided first, no node breaks a dependency on another (l.265),
  nothing is deferred, and `acc + 1`'s read leaves out its own relations alone
  (l.263). Whether row 3 makes a cycle between distinct nodes, which l.265
  breaks, no witness yet shows.
