# 17 The captured environment

Rests on the spec's l.203 (a call of a function value), l.210–222 (binding groups,
C, instances), l.226 (meeting), l.244 (b) (what a class depends on), l.267, l.293
and l.301; on `docs/rfc_compiler/functions.md` §2 (a function value bound by `let`
generalizes what its type leaves unconstrained, "except what the enclosing scope or
the captured environment holds: those belong to where they were written"; its
environment is captured once, where it is written) and `types.md` §1 (every slot
has one type; values meeting through a chain of slots are of one type); and on
`holes.md`, H5 and O4. Its direction and its rows were agreed with 준량 on
2026-10-21. It is a reading, `Captures::Environment`, beside the spec as stated,
`Captures::AsStated`, the default, so that both run on the same witnesses. The code
is `Program::cut_environments` and `Program::owner`, `Program::item_group` and
`Program::is_universal` in `src/program.rs`, `State::positions` and
`State::environment_of` in `src/node.rs`, and the certificate's contexts and
`call_value` in `src/certificate.rs`.

## Meaning

The spec solves a lambda's group before its enclosing one, and copies all of its C,
the captures shared, at each call of its value (l.220, l.222). What the lambda
records into a capture is then recorded only by the calls' instances, at none of
their positions, so the capture and what reads it are decided before or after it by
order (O4), and a node of the lambda that reads a capture is decided before the
enclosing group records into it (H5).

Read by functions §2, the part of a lambda's text that its captured environment
holds is not generalized: it belongs to the enclosing group and is solved there,
once. Only the rest is the lambda's binding group, generalized and copied at each
call.

## Definitions

- A **capture** of a lambda is a class of an enclosing group that its text names
  (l.217).
- Two classes of a lambda's text are **joined** where one item of the text names
  both, or one is a part of a kind the text writes for the other. The
  **environment** of a lambda is every class joined, directly or through others, to
  a capture, and every item that names one of them. A lambda inside it records its
  own environment into it first, as items of its text.
- An item **names** the classes it relates. A use of a function of an enclosing
  group names that function's parameter and result types as well, which it takes
  as they are (note 09, reading 4), and a binding names the classes of the
  projections on its path (l.317).
- The **generic part** of a lambda is the rest of its text, its parameters among it
  where they are not in the environment.

## Rows

1. **The environment is the enclosing group's.** Its classes are classes of that
   group, each an unknown of it unless the text writes it (l.218). Its items are
   recorded there, and its nodes made there, once, when that group's solving
   starts, whether or not a call reaches the lambda.
2. **The lambda's binding group is its generic part** (l.210). Its C is the generic
   part's, copied at each instance (l.219, l.222). It names no capture, so no
   instance records into one.
3. **A call of a function value meets the environment where it is.** Each argument
   meets the copy of its candidate lambda's generic parameter, or that lambda's
   parameter in the environment, an unknown of the call's group; the call's result
   takes the copy of a generic result type, or the environment's. Within the depth
   its candidates reach (l.244 (b)), the call's positions include the parameters
   and the result type in the environment of each lambda its callee's Values name
   at the moment.

What each row excludes:

- Row 1: the environment generalized or copied, and a capture taking a relation
  from an instance. A lambda no call reaches still has its environment recorded,
  and the nodes there are decided (l.301 holds of them).
- Row 2: a class of an enclosing group in a lambda's C.
- Row 3: a parameter in the environment solved before a call whose Values name its
  lambda is decided.

## Invariants

- **I\* at a capture.** No relation into a capture is recorded after a decision that
  reads it, outside a cycle (l.269; types §6: "A decision reads a type only once
  nothing can change it").
- **I2 at the witnesses.** H5 and O4 reach one outcome each.
- **Generalization** leaves out the environment (functions §2), and every capture
  has one type (types §1).
- **Soundness.** Every item of a lambda's text is recorded, in the environment once
  or in each instance, and checked as before (l.293).
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

Derived before running, under `Captures::Environment`; under `Captures::AsStated`
every current test keeps its outcome.

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

Every row of the table holds as derived, each asserted under
`Captures::Environment`, and under `Captures::AsStated` where the table states it
(`examples.rs`: `hole_5_a_lambda_reads_a_capture`,
`hole_o4_a_capture_written_after_a_use_reads_it`,
`example_6_a_capture_written_by_name`, `example_6_called_with_two_types`,
`example_6_through_a_reference`, `a_literal_capture_passed_to_a_parameter`,
`a_capture_lent_in_a_lambda`, `a_lambda_never_called_and_its_environment`,
`a_capture_written_by_the_lambdas_result`,
`a_call_left_out_leaves_out_its_instance_relations`). Under
`Captures::Environment` each reaches one outcome in every order, I\* reports
nothing, and the certificate holds of each accepted final state (a refused one has
nothing to check, note 14).

With the default changed to `Captures::Environment`, 95 of the 99 tests pass; the
four that fail are those whose first half asserts the spec's outcome through
`Readings::default()`: H5, O4, Example 6's `S` and the lambda never called. That
was run by changing the model, and no test keeps it.

## Open

- A function value that leaves the group its lambda lies in, returned from a
  function, reaches calls whose group copies the environment with that function's
  C. Function values crossing groups are not yet encoded (note 09).
- A call's candidates are the Values of the moment. Whether Values can still grow
  after a call is decided is not shown: in every witness the lambda reaches the
  callee's class by a pass recorded when the group's solving starts, so no call's
  Values grow after it.
- Row 3 makes a node's result reach what it reads through a call. In
  `acc = f()` with `f = || acc + 1`, `acc + 1`'s result `R` reaches `ACC`, which it
  reads, by `R ≤ λ result ≤ f() ≤ ACC`. Probed in every order by changing the
  model, `f()` is decided first, no node breaks a dependency on another (l.269),
  nothing is deferred, and `acc + 1`'s read leaves out its own relations alone
  (l.267). Whether row 3 makes a cycle between distinct nodes, which l.269
  breaks, no witness yet shows.
