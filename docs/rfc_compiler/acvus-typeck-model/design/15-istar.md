# 15 I\*: a decision sees what could change it

Rests on `docs/notes/typeck-pivot.md` §1–2 (I\*: a decision is made only where no
relation not yet recorded could change it), and on the spec's l.263 (the value a
node reads), l.265 (breaking), l.240 (b), l.244 and l.246 (which nodes record into
a class) and l.248. The code is `src/istar.rs`.

## Meaning

The dependency table (l.236–267) is the spec's means to I\*: a node waits for every
node that could record into what it reads, except within a cycle, where each reads
without the others. This check tests the means against the end it serves. Once
solving ends, each decided node is decided again, reading the relations recorded by
then, leaving out those recorded by itself and by the nodes that transitively
depend on it (l.263), the dependency taken from what the nodes record by then. A
decision that differs from the one made was made without a relation that could
change it, outside a cycle.

## Definitions

- **The end state** of a binding group: the entry's final state, after fallback,
  and another group's state at its fixpoint (l.288).
- **What a node records into** at the end, read by the relations it records, not
  by position as the solver reads l.246 (note 08, reading 18). A use, the classes its candidate would record a lower
  bound or an equation into by the splits of l.280 (note 08, reading 13), those
  of its remaining candidates where it is not decided as one, and its result where it is decided `Never`: the classes a use makes anew
  are its own, and no other node reads them. Another decided node, the classes its
  own relations record into, those of the instances it made among them: the upper
  side of a lower bound or a part lower bound, the type passed into at a
  boundary, both sides of an
  equation, and the class a kind, a fact, a LitVar or a lambda is given to. A node
  not decided, the classes at its positions. For a read of whether a class is a
  reference, a use leaves out the classes it recorded that of at once (l.244), a
  lend its place, a binding its position, and a projection of a field, a
  component, an index or a variant everything (note 08, readings 9 and 12).
- **Reach at the end.** A class reaches each class it has as a lower bound or a
  part lower bound, along every one recorded by the end, a split's among them. By the end every
  class has taken its constructor, so the rows of l.240 (a) and l.241, which wait
  for a class to take one, hold of none; what a class's value comes from then is
  its sources (l.271–278), and what its taking a constructor recorded into its
  parts is in the log as splits.
- **Dependency at the end.** A node depends on each node that records into a class
  reached from the classes at the positions it reads (l.243, l.246), and a binding
  on the projections on its path (l.308).
- **The exclusion of `n`**: `n` and every node that transitively depends on `n` in
  that graph.
- **The decision again**: the decision `n`'s rule (note 08) gives reading the end
  state's relations less those its exclusion's nodes recorded, the relations of the
  instances a call made counted as the call's. A projection in a binding's
  exclusion counts as undecided for its path.

A first encoding read reach with the rows of l.240 (a) and l.241 as well, at the
kinds of the end, and a lend, a binding, a projection and a call as recording into
every class at their positions. Kinds the decisions gave then made cycles that
held at no moment of solving: in Example 9, `&r`'s target is `V`, so `lend v`
depended on `lend r`, and `lend r` would read `R` without `&mut v`, against the
example; in O3, `first`'s result took a target that lies within the lower bound
`R ≤ r`, so the lend would read `r` without `first`.

## Domain

Every decided node, in each context its decision holds for:

- A node of the entry, in the entry's end state.
- A node decided in a group other than the entry, in that group's end state.
- A node decided in a lambda's group, its generic part (l.188), also in each
  instance of that group's C that was made, under that instance's map, and,
  through an instance made for the check, in the enclosing group's end state, so
  that a lambda never called is checked as well. A lambda inside a function's body
  is so checked in each instance of that function. The environment's nodes are the
  enclosing group's, decided and checked in its end state.
- A copy of a node left in C, in the state of its instance, as a node of that
  state.

## Contract

- **I\* holds** of a node where the decision again equals the decision made.
- **A completeness violation**: it does not hold, and the program is refused. The
  refusal may come of the decision made.
- **A soundness violation**: it does not hold, and the program is accepted, with a
  typing other than the one I\* gives.

Each violation names the node, the group of the state it is decided again in, the
two decisions, and the exclusion the decision again leaves out.

## Interface

`fn check(program: &Program, end: &Final) -> Vec<Violation>`, `Final` being a final
state with the groups solved before it, as exploration hands it (note 13). It
returns rather than panics, as note 14's check does.

It needs of the solver, besides note 14's additions: a node's decision under a
given exclusion (`State::decision_under`), where the solver's own reads take the
exclusion from the graph of the moment; what one candidate of a use would record
into (`State::would_record_by`); and an instance the check makes of a lambda's C
in a copy of the enclosing group's end state (`Recorder::Check`), which solving
never records.

## What it shares, and what it presupposes

- It shares with the solver the value read (`read_state`), each node's rule
  (`narrow`, the lends', projections' and bindings'), the positions a node reads,
  and what a candidate would record. An error there is shared, and the check does
  not see it.
- It computes on its own the dependency at the end, the exclusions and the
  comparison, never through `graph()`, which is the means under test.
- It reads what a node records into by relation, where the solver reads it by
  position, and reports H7 for that reason. A boundary out of a class that holds a
  LitVar alone is left out of what a node records into, though l.280 makes such a
  boundary change the class (`holes.md`, H7): in such a cycle the check reads
  relations the solver's read leaves out.
- It presupposes that a read leaves out the relations of the instances a call of a
  function value made, when the read leaves out the call (note 07, reading 3). An
  earlier encoding did not, naming an instance's relations by the instance alone:
  `let f = if c { put } else { choose(x) }; f(x);` with `put = |w| x = make_b()`,
  `choose(A)`, `choose(B)` and `make_a() ≤ X` reached `choose(A)` and `choose(B)`
  by order under `Deferral::Reading`, and then reached `choose(A)` in every order.
  Since 8978eb70 `put`'s write is its environment, the enclosing group's (l.188),
  so `make_b() ≤ X` is recorded where it is written, no read leaves it out with
  the call, and the program reaches `choose(B)` in every order. Since 80b33ad9 no
  declaration gives a function value it was not given (l.186), and `choose` is
  `choose(A, T) -> T`, `choose(B, T) -> T`, called as `choose(x, put)`
  (`cases/istar/a-call-left-out-leaves-out-its-instance-relations.case`).

## Expected results

Before the check runs, as its acceptance criteria:

| Program | Expected |
|---|---|
| H7 | a completeness violation at `probe`, in every order |
| H5 | a completeness violation at the lambda's `probe`, in the instance and in the entry's end state |
| O1 under `AsStated`, O3, O4 | a violation in exactly the orders that make the early decision, `x: u8`, the lend's `NoResult` and `probe(A)`; none in the others |
| the program above | none |
| Example 12, l.261 with `pick(u8) -> &u8`, the other current examples | none |

Run, the check reports these and no others: each witness of `holes.md` asserts,
for the outcome of each final state, the nodes it reports there and how, and every
other example asserts it reports none (`design/19-cases.md`: a case's `assert istar`,
none where it writes none, of every final state). Before 8978eb70, O4's `probe(A)`
was a soundness violation where the program was accepted and a completeness one in
its third outcome, refused at `put(..)#λ/V ≤ X`; since then O4 has one outcome and
reports none (`holes.md`, O4). Example 14 reports none over its sampled orders.

The table is as derived before that run. Since a node whose read decides nothing is
no longer decided (note 08, reading 19), its H5 and O3 rows no longer hold, and both
report none: H5's `probe` stays in C and is decided in the copy, as the check
decided it, and O3's lend waits for `first`, so no order makes the early decision
the table names.

## Not covered

- A class's solving, `α ≐ ⊔S` (l.286), is the decision the pivot's I\* is first
  stated of, and this check decides only nodes again. Before 8978eb70, O4's third
  outcome was such a decision: `X` was solved as `A` before the call's instance
  recorded `B ≤ X`, and the check reported it only through `probe`, which reads
  `X`. Since then `X` waits for the call, and O4 has one outcome.
