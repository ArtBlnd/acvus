# 17 The captured environment

Rests on the spec's l.186 (the captured environment), l.196 (a call of a function
value), l.203–215 (binding groups, C, instances), l.219 (meeting), l.238 (b) (what a
class depends on), l.261, l.280 and l.290; on `docs/rfc_compiler/functions.md` §2 (a function value bound by `let`
generalizes what its type leaves unconstrained, "except what the enclosing scope or
the captured environment holds: those belong to where they were written"; its
environment is captured once, where it is written) and `types.md` §1 (every slot
has one type; values meeting through a chain of slots are of one type); and on
`holes.md`, H5 and O4. Its direction and its rows were agreed with 준량 on
2026-10-21, as a reading, `Captures::Environment`, beside the spec as stated then,
`Captures::AsStated`. The text took them at 8978eb70, as 준량 decided (l.23, l.186,
l.196, l.209, l.210, l.238), and both readings are removed with it: the
environment is cut always. Since G4 of the review of 2026-10-22, as 준량
decided, a use gives a constraint for each set of positions its callee ties,
and the environment is cut by constraints, not items (Definitions, Rows 4–5);
a function of another group ties a use's positions by the parts of its C, so
the cut is made as the groups are solved (Row 6). The code is `Cut::of`,
`Cut::owner`, `Cut::part`, `Cut::is_universal` and `constraints` in
`src/cut.rs`, `State::record_closed` in `src/node/common.rs`,
`State::components` and `Instance::part` in `src/instance.rs`, `State::positions` and
`State::environment_of` in `src/node/position.rs`, and the certificate's contexts and
`call_value` in `src/certificate.rs`.

## Meaning

Before 8978eb70 the spec solved a lambda's group before its enclosing one, and
copied all of its C, the captures shared, at each call of its value (l.213, l.215).
What the lambda recorded into a capture was then recorded only by the calls'
instances, at none of their positions, so the capture and what reads it were
decided before or after it by order (O4). A node of the lambda that read a capture
read it before the enclosing group recorded into it, and where that decided
nothing it was decided in each copy (H5, which is not a hole).

By l.186, as functions §2 has it, the part of a lambda's text that its captured environment
holds is not generalized: it belongs to the enclosing group and is solved there,
once. Only the rest is the lambda's binding group, generalized and copied at each
call.

## Definitions

- A **capture** of a lambda is a class of an enclosing group that its text names
  (l.210).
- A **constraint** of the text is an item other than a use, on the classes it
  relates, a binding on those of the projections on its path as well (l.306);
  or one a use gives for each set of positions its callee ties (l.186). Of
  declarations: a position where every candidate writes one closed type, alone;
  with one candidate, the positions that write one of its variables, together;
  with several, the rest, together with `e?`'s exit, the choice of a candidate
  deciding them. Of a function of its own or an enclosing group: each position
  alone, on that function's parameter or result type as well, which it takes as
  they are (note 09, reading 4). Of another group: the positions whose types lie
  in one part of that group's C, together, and each whose type lies outside C,
  alone.
- The **parts of C** (`State::components`): two of C's classes lie in one part
  where one of C's relations is on both, a node left undecided names both, a
  binding's node and the projections on its path name them, one is a part of the
  other's kind, or one holds a function value whose Values name a lambda placed
  in the group and the other lies in that lambda's environment. Each is what an
  instance makes anew together: copying a relation, a node, a kind, or a lambda
  with its environment (l.215).
- Two classes of a lambda's text are **joined** where one constraint is on both,
  or one is a part of a kind the text writes for the other. The **environment**
  of a lambda is every class joined, directly or through others, to a capture,
  and every constraint on one of them, which is on them alone, its classes being
  joined. A lambda inside it records its own environment into it first, as
  constraints of its text.
- The **generic part** of a lambda is the rest of its text, its parameters among it
  where they are not in the environment.

## Rows

1. **The environment is the enclosing group's.** Its classes are classes of that
   group, each an unknown of it unless the text writes it (l.211). Its
   constraints are recorded there, and its nodes made there, once, when that group's solving
   starts, whether or not a call reaches the lambda.
2. **The lambda's binding group is its generic part** (l.203). Its C is the generic
   part's, copied at each instance (l.212, l.215). It names no capture, so no
   instance records into one.
3. **A call of a function value meets the environment where it is.** Each argument
   meets the copy of its candidate lambda's generic parameter, or that lambda's
   parameter in the environment, an unknown of the call's group; the call's result
   takes the copy of a generic result type, or the environment's. Within the depth
   its candidates reach (l.238 (b)), the call's positions include the parameters
   and the result type in the environment of each lambda its callee's Values name
   at the moment.
4. **A use whose constraints two groups record has a part in each.** Each group
   records the constraints at its positions (`Cut::part`). Of declarations of
   one candidate, each group records it at its positions at once, with no node
   (`State::record_one`); of several, the node is where the choice is, the
   positions where every candidate writes one closed type recorded where they lie
   by the common part (`State::record_closed`, l.246). So a node is one group's.
5. **A call of another group's function makes, in each group that records some
   of its constraints, an instance of the parts of C at those positions**
   (`Instance::part`, l.215). Where one group records every position, the
   instance is of all of C, as before.
6. **The cut is made as the groups are solved** (`Cut::of`). A lambda is cut
   once the groups it makes instances of are solved, before it is solved itself
   (l.213), so each group's start, and the certificate, takes the cut of the
   groups solved up to it, in that order, the lambdas inside a group first.

What each row excludes:

- Row 1: the environment generalized or copied, and a capture taking a relation
  from an instance. A lambda no call reaches still has its environment recorded,
  and the nodes there are decided (l.290 holds of them).
- Row 2: a class of an enclosing group in a lambda's C.
- Row 3: a parameter in the environment solved before a call whose Values name its
  lambda is decided.
- Row 4: a constraint recorded in a group that holds none of its positions, and a
  constraint on a capture recorded only where a call reaches the lambda, which no
  whole-use placement avoids: placing a use whose constraints are on a capture and
  on the lambda's own classes in the environment names an instance's class there,
  and placing it in the generic part records its constraint on the capture in each
  instance, or in none (`spec/a-lambda-never-called`, H5).
- Row 5: an instance in one group of a part of C at a position another group
  records. The cut groups the positions by the same parts, so no part lies at
  positions of both. Where some of C, or a node left in it, lies at no position of
  a use whose constraints two groups record, which no rule places, the model
  panics.
- Row 6: a lambda cut by the text of the groups it makes instances of, which ties
  positions through classes those groups decide, `a` and `b` of
  `fn g(a, b) { let z: u8 = a; z = b; }` (derived, not run), and a cut made
  before those groups are solved.

## Invariants

- **I\* at a capture.** No relation into a capture is recorded after a decision that
  reads it, outside a cycle (l.263; types §6: "A decision reads a type only once
  nothing can change it").
- **I2 at the witnesses.** H5 and O4 reach one outcome each.
- **Generalization** leaves out the environment (functions §2), and every capture
  has one type (types §1).
- **Soundness.** Every constraint of a lambda's text is recorded, in the
  environment once or in each instance, and checked as before (l.280), each in
  the context of the group that records it.
- **Conservative refinement**, superseded by G4. It judged a finer environment by
  whether every program this reading accepted kept its typing. G4's, which 준량
  decided, does not keep it: `|x| write(&mut out, x)` called with an `A` and a
  `B`, under `write(&mut Buf, A)`, `write(&mut Buf, B)` and a cast `A → B`, was
  accepted with `x: B` and `write` decided once as the second, and is accepted with
  `write` decided at each call, the first and then the second, run before and
  after (`spec/l186-a-lambda-writes-two-types-a-cast-joins`).

## What it omits

- **A parameter the environment holds is not polymorphic.** Every call shares its
  one type. What such a parameter is given reaches a capture's one type anyway
  (types §1), so the loss is only where calls would type it differently below that
  type: `|x| { last = Some(x); probe(x) }`, called with `A` and with `B` under a cast
  `A → B`, decides `probe(B)` once instead of `probe(A)` and `probe(B)`. Borne by the
  program's author, who passes the capture as a parameter where a per-call typing
  is wanted.
- **Joining is coarser than what flows.** It joins the classes a constraint is on,
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
types, the lambda never called, and the call left out (then under a reading of
l.263 since removed).
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

## G4: the environment cut by constraints

G4 of the review of 2026-10-22: a use joined every class it named, so
`let emit = |x| write(&mut out, x); emit(1u8); emit("s");` held `x` in the
environment, `write`'s two candidates both writing `&mut Buf` at `out`, and was
refused at `x`. 준량 decided that a use joins only the positions its callee ties
(Definitions). The cut placing whole items was then found to exclude both ways of
placing a use whose constraints are on a capture and on the lambda's own classes
(Row 4, What each row excludes), and the environment is cut by constraints (Rows
4–5). Derived before running:

| Program | Expected |
|---|---|
| `emit` above | accepted; `&mut out ~> write.CommonParam(0)` once in the enclosing group; `write#0` and `write#1`, one at each call |
| `let f = \|x\| push(&mut v, x); f(1u8); f("s");`, `push<T>(&mut Vec<T>, T)` | refused, `no_join X`: `T` ties `x` to the capture |
| `fn put(b, v) { write(b, v) }; let emit = \|x\| put(&mut out, x);`, called as `emit` | accepted; the enclosing group's instance of `put` holds `b`, each call's `v` and `write` |
| `let x = 1u8; let f = \|y\| pair(x, y); f(2u8); f("s");`, `pair<A, B>(A, B)` | accepted; `pair`'s constraint at `x` the enclosing group's, at `y` each call's |
| `fn put2(a, b) { takes_u8(a) }; let x = 0; let f = \|y\| put2(x, y);`, never called | accepted, `x: u8`: the enclosing group's instance of `put2` holds `a` |
| `\|x\| write(&mut out, x)` called with an `A` and a `B`, `write(&mut Buf, A)`, `write(&mut Buf, B)`, a cast `A → B` | accepted, `write` decided at each call as the first and the second; before G4, `x: B` and `write` once as the second |
| the cases above under `cases/environment/`, H5 and the lambda never called | their values and decisions kept; each use's constraint at its result, which writes no variable with its arguments, the lambda's |

Each held as derived, in every order, I\* reporting nothing and the certificate
holding (`cases/spec/l186-*`); every other case kept its outcome.

The cut then tied a function's positions by that function's text, and its
instance made what C tied to them, two computations a panicking check held
alike. Both left out that a lambda placed in the function is made anew with its
environment and the function value naming it (l.215), so
`fn adder(k) { |x| x + k }; let h = || adder(cap); let f = h(); f(2u8)` split
`adder(cap)` and panicked copying the closure, as derived before it was run.
준량 decided that a function ties a use's positions by its C alone, the parts of
C with the lambda's tie among them (Definitions, Rows 5–6), and the cut is made as
the groups are solved. Derived before running and held, in every order: every
case kept its outcome, and the program above is accepted, `adder(cap)` one
constraint, the environment's, `f(2u8)` `u8`
(`spec/l186-a-lambda-calls-a-function-that-returns-a-closure`). Of the four
cases whose result constraint moved, the checks of their results are each call's
instance's, and `phases/node-what-every-candidate-shares` records `bool`, which
both candidates write at the result, with the text, not in the common part's
phase.

## Open

- Where a call of another group's function has constraints two groups record,
  some of C, or a node left in it, lying at no position of it panics
  (`Instance::part`, Row 5): no witness shows one, and the text places none.
- Whether a function group reaches more than one fixpoint, whose C's parts
  differ, which would cut a lambda that calls it differently along two orders
  (Row 6), no witness shows.

- A function value that leaves the group its lambda lies in, returned from a
  function, is encoded since the text's l.215 makes the lambda anew at each
  instance of that group (`State::lambdas`, `LambdaCopy`): a class whose Values
  name a lambda whose environment is the group's stays in C, and the instance
  names the copy, whose environment is its copies, so each call of the returned
  value reaches the environment of the instance it came from
  (`cases/spec/l215-*`). A function value that reaches a group its environment is
  in neither as the original nor as a copy still panics (`State::environment_of`).
- A call's candidates are the Values of the moment. Whether Values can still grow
  after a call is decided is not shown: in every witness the lambda reaches the
  callee's class by a pass recorded when the group's solving starts, so no call's
  Values grow after it. The certificate reports a call whose callee's Values name
  a lambda it does not call (note 14), so a final state where they grew fails
  there; no case shows one yet.
- Row 3 makes a node's result reach what it reads through a call. In
  `acc = f()` with `f = || acc + 1`, `acc + 1`'s result `R` reaches `ACC`, which it
  reads, by `R ≤ λ result ≤ f() ≤ ACC`. Probed in every order by changing the
  model, `f()` is decided first, no node breaks a dependency on another (l.263),
  nothing is deferred, and `acc + 1`'s read leaves out its own relations alone
  (l.261). Whether row 3 makes a cycle between distinct nodes, which l.263
  breaks, no witness yet shows.
