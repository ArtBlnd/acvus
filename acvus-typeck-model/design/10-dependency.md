# 10 The dependency graph

Rests on the spec's l.240–271. The code is `src/dep.rs`.

## Meaning

The graph's vertices are the equivalence classes of the moment and the undecided
nodes. A class depends on its unsolved lower bounds, on each unsolved class whose
taking a constructor will record into it, and on each undecided node at whose
position it lies. A node depends on the undecided nodes that record into the
classes at the positions it reads, which are the nodes those classes, and the
classes they reach along dependencies between classes, depend on (l.250). Each
row is of the moment: a dependency on a solved class or a decided node is gone.

## Readings

1. **Breaking.** A vertex breaks its dependency on every vertex that transitively
   depends on it (l.269), so the dependencies within one strongly connected
   component are broken and those between components are not (l.271).
2. **Within, at any depth.** A class lies within the `τ` of `τ ≤ α` or the `U` of
   `α ⇝ U` at any depth of their kinds (l.244 (a)). Read as the direct parts only,
   a part of a part would take its value before `α`'s split records the equation an
   invariant part gives.
3. **A LitVar lower bound is not waited for.** A lower bound holding only a LitVar
   and waiting for nothing gives its LitVar, which the join unifies (l.176), so the
   class it bounds may be solved. Read as an unsolved lower bound, which waits,
   every current example reaches the outcome it does: run by changing the model,
   Example 12's `X` waits for `?a` and takes `u16` all the same, from the equation
   the decided `x + y` records. No example yet tells the two readings apart.
4. **The part row.** A part depends on the classes it is a part of taking a
   constructor (l.245). A class holding a kind is solved (note 06, reading 4), so
   the row holds of every part already.

## Left open for change

Each rule of the table that the ledger finds open is one function, so that a
reading changes one place:

- `State::reads`: what a node reads, and the projections on a binding's path it
  depends on (`holes.md`, O1).
- `State::records_into`: which nodes at a class's position may change what a node
  reads there (l.248, l.250; `holes.md`, H7).
- `State::chosen_outside`: which classes depend on the use that makes their
  group's instance (l.246; `holes.md`, H5).
