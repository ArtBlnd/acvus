# 10 The dependency graph

Rests on the spec's l.236–267. The code is `src/dep.rs`.

## Meaning

The graph's vertices are the equivalence classes of the moment and the nodes. A
class depends on its unsolved lower bounds and part lower bounds, on each unsolved
class whose taking a constructor will record into it, on each unsolved class it is
a part of, and on each node at whose position it lies that has not recorded its
relations. A dependency on a lower bound or a part lower bound nothing enters is
met (l.240, l.284). A node depends on the nodes that record into the classes at
the positions it reads, which are the nodes those classes, and the classes they
reach along dependencies between classes, depend on (l.246). Each row is of the
moment: a dependency on a solved class is gone, and a node's on a decided node is
met.

## Readings

1. **Breaking.** A vertex breaks its dependency on every vertex that transitively
   depends on it (l.265), so the dependencies within one strongly connected
   component are broken and those between components are not (l.267).
2. **Within, at any depth.** A class lies within the `τ` of `τ ≤ α` or the `U` of
   `α ⇝ U` at any depth of their kinds (l.240 (a)). Read as the direct parts only,
   a part of a part would take its value before `α`'s split records the equation an
   invariant part gives.
3. **A LitVar lower bound is not waited for.** A lower bound holding only a LitVar
   and waiting for nothing gives its LitVar, which the join unifies (l.171), so the
   class it bounds may be solved. Read as an unsolved lower bound, which waits,
   every current example reaches the outcome it does: run by changing the model,
   Example 12's `X` waits for `?a` and takes `u16` all the same, from the equation
   the decided `x + y` records. No example yet tells the two readings apart.
4. **The part row.** A part depends on the classes it is a part of taking a
   constructor (l.241). A class holding a kind is solved (note 06, reading 4), so
   the row holds of every part of a kind already; a class a kind known in part
   names (l.133) depends on the unsolved class holding that kind (`Edge::Part`).
   Without it, the target a lend makes for a place read as a reference, which
   nothing enters, was filled before its place took the constructor that equates
   it with the place's own target, and the outcome depended on the order
   (`rules/place/a-a-lend-of-a-reference-place-targets-its-target`).
5. **A node decided whose relations wait.** A node decided while undecided nodes
   broke their dependency on it records once they are decided (note 08, reading
   17). Until then the classes at its positions depend on it, as on the relations
   not yet recorded into them (l.240 (b)); a node's dependency on it is met, it
   being decided (l.243). Without it, a class at its position, which nothing
   entered until it recorded, met its dependents' dependencies on it, and the
   outcome depended on the order (`holes.md`, H7).
6. **A call of a function value's positions** include the parameters and the
   result type in the environment of each lambda its callee's Values name at the
   moment (l.240 (b); note 17, row 3), so a class the call meets there waits for
   it.
7. **`e?`'s positions** include the class of its exit's value (note 08, reading
   20), so that the innermost body's result, which that class passes into, does
   not take its value before the exit is recorded. Without it, the exit would
   record into a class no node lies at, as O4's instances once did.

## Left open for change

Each rule of the table that the ledger finds open is one function, so that a
reading changes one place:

- `State::reads`: what a node reads, and the projections on a binding's path it
  depends on (`holes.md`, O1).
- `State::records_into`: which nodes at a class's position may change what a node
  reads there (l.244, l.246; `holes.md`, H7).
