# 15 I\*: a decision sees what could change it

The code is `src/istar.rs`.

## Meaning

The dependency table is the means to I\*: a node waits for every
node that could record into what it reads, except within a cycle, where each reads
without the others. This check tests the means against the end it serves. Once
solving ends, each decided node is decided again, reading the relations recorded by
then, leaving out those recorded by itself and by the nodes that transitively
depend on it, the dependency taken from what the nodes record by then. A
decision that differs from the one made was made without a relation that could
change it, outside a cycle.

## Definitions

- **The end state** of a binding group: the entry's final state, after fallback,
  and another group's state at its fixpoint.
- **What a node records into** at the end, read by the relations it records, not
  by position as the solver reads. A use, the classes its candidate would record a
  lower bound or an equation into by the splits, those
  of its remaining candidates where it is not decided as one, and its result where it is decided `Never`: the classes a use makes anew
  are its own, and no other node reads them. Another decided node, the classes its
  own relations record into, those of the instances it made among them: the upper
  side of a lower bound or a part lower bound, the type passed into at a
  boundary, both sides of an
  equation, and the class a kind, a fact, a LitVar or a lambda is given to. A node
  not decided, the classes at its positions. For a read of whether a class is a
  reference, a use leaves out the classes it recorded that of at once, a
  lend its place, a binding its position, and a projection of a field, a
  component, an index or a variant everything.
- **Reach at the end.** A class reaches each class it has as a lower bound or a
  part lower bound, along every one recorded by the end, a split's among them. By the end every
  class has taken its constructor; what a class's value comes from then is
  its sources, and what its taking a constructor recorded into its
  parts is in the log as splits.
- **Dependency at the end.** A node depends on each node that records into a class
  reached from the classes at the positions it reads, and a binding
  on the projections on its path.
- **The exclusion of `n`**: `n` and every node that transitively depends on `n` in
  that graph.
- **The decision again**: the decision `n`'s rule gives reading the end
  state's relations less those its exclusion's nodes recorded, the relations of the
  instances a call made counted as the call's. A projection in a binding's
  exclusion counts as undecided for its path.

## Domain

Every decided node, in each context its decision holds for:

- A node of the entry, in the entry's end state.
- A node decided in a group other than the entry, in that group's end state.
- A node decided in a lambda's group, its generic part, also in each
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
state with the groups solved before it, as exploration hands it. It
returns rather than panics.

It needs of the solver: a node's decision under a
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
  position. A boundary out of a class that holds a LitVar alone is left out of what
  a node records into: in such a cycle the check reads relations the solver's read
  leaves out.
- It presupposes that a read leaves out the relations of the instances a call of a
  function value made, when the read leaves out the call.

## Not covered

- A class's solving, `α ≐ ⊔S`, is the decision I\* is first
  stated of, and this check decides only nodes again.
