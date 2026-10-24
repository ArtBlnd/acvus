# 10 The dependency graph

The code is `src/dep.rs`.

## Meaning

The graph's vertices are the equivalence classes of the moment and the nodes. A
class depends on its unsolved lower bounds and part lower bounds, on each unsolved
class whose taking a constructor will record into it, on each unsolved class it is
a part of, and on each node at whose position it lies that has not recorded its
relations. A dependency on a lower bound or a part lower bound nothing enters is
met. A node depends on the nodes that record into the classes at
the positions it reads, which are the nodes those classes, and the classes they
reach along dependencies between classes, depend on. Each row is of the
moment: a dependency on a solved class is gone, and a node's on a decided node is
met.

## Readings

1. **Breaking.** A vertex breaks its dependency on every vertex that transitively
   depends on it, so the dependencies within one strongly connected
   component are broken and those between components are not.
2. **Within, at any depth.** A class lies within the `τ` of `τ ≤ α` or the `U` of
   `α ⇝ U` at any depth of their kinds. Read as the direct parts only,
   a part of a part would take its value before `α`'s split records the equation an
   invariant part gives.
3. **A LitVar lower bound is not waited for.** A lower bound holding only a LitVar
   and waiting for nothing gives its LitVar, which the join unifies, so the
   class it bounds may be solved.
4. **The part row.** A part depends on the classes it is a part of taking a
   constructor. A class holding a kind is solved, so
   the row holds of every part of a kind already; a class a kind known in part
   names depends on the unsolved class holding that kind (`Edge::Part`).
   Without it, the target a lend makes for a place read as a reference, which
   nothing enters, was filled before its place took the constructor that equates
   it with the place's own target, and the outcome depended on the order
   (`rules/place/a-a-lend-of-a-reference-place-targets-its-target`).
5. **A node decided whose relations wait.** A node decided while undecided nodes
   broke their dependency on it records once they are decided. Until then the classes
   at its positions depend on it, as on the relations
   not yet recorded into them; a node's dependency on it is met, it
   being decided. Without it, a class at its position, which nothing
   entered until it recorded, met its dependents' dependencies on it, and the
   outcome depended on the order.
6. **A call of a function value's positions** include the parameters and the
   result type in the environment of each lambda its callee's Values name at the
   moment, so a class the call meets there waits for
   it.
7. **`e?`'s positions** include the class of its exit's value, so that the innermost
   body's result, which that class passes into, does
   not take its value before the exit is recorded. Without it, the exit would
   record into a class no node lies at.
8. **A node that records only a boundary out of a class**: a use at an argument each
   candidate writes a type for, and within the
   depth those types reach, not at its result, its exit or an argument a candidate
   writes a variable for (`State::passed_out_only`). Such a boundary changes the
   class only by the fill of a class nothing enters or LitVars alone enter, so the
   class waits for the node only while it is such a class, judged on
   the graph without those dependencies (`State::filled_by_places_in`). Where the
   class has a source still to come, a use that only reads it waits for that source
   and is not waited for. Four cases refused at an undecided use now give the
   classes at its arguments their values, the refusal unchanged.
9. **What a split not yet made will relate**.
   One rule for every relation not yet split: the side that will take a relation
   waits for the class whose taking a constructor splits it (`Edge::Taking`): a
   part of `τ` in `τ ≤ α`, a part of `U` in `α ⇝ U`, a part at a parameter of a
   function value, and the class `e?`'s wrapping passes into. And the nodes
   recording into a class are reached along the parts of the two sides such a
   split will link, among those they hold now by a kind, a kind known in part
   or a function value's environment (`State::split_links`), so a pattern on a
   variant's payload waits for the lend that makes it a reference, and a binding for
   its path's projections. The exclusion of a field's, a component's, an index's and a
   variant's
   projection from a read of whether a class is a reference is of those whose
   operand is the class read (`State::projects_from`).
10. **A lambda's environment and a function type**. A class holding a function value
   is taken as holding the parameter and result types in its lambdas' environments as
   the parts at a function type's parameters and result: they lie within the depth of
   a candidate writing that function type (`State::environment_within`), the
   parameters as classes it records into and the result types as ones it passes out
   of. And each environment parameter of a lambda whose value reaches a class passing
   by a boundary into a function type waits for that class to take its constructor,
   the boundary's split recording into it (`Edge::Taking`). Without the second, the
   use was decided at once, the split waited for the function value's class, and the
   parameter was filled first.

11. **What a broken dependency defers**. The relations of a node a broken dependency
   pointed to wait for the node that broke it only where that node reads them: into a
   class at a position it reads, or a class those reach by lower bounds and part
   lower bounds, as I\* takes what a node records; the rest are recorded as the node
   is decided (`State::reading_what_it_records`). In `even(10u8)` with `even` and
   `odd` calling each other, `n - 1` in odd's copy, decided by its result, records
   into its parameters' types, which `n == 0` does not read; deferred whole, its
   relations waited for `n == 0`, which they alone decide, so the order decided
   whether the two had broken their dependencies before, refused, or not, accepted.

## Left open for change

- `State::reads`: what a node reads, and the projections on a binding's path it
  depends on.
- `State::records_into`: which nodes at a class's position may change what a node
  reads there.
