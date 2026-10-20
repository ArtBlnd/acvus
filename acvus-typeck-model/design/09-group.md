# 09 Binding groups, universals, C and instances

Rests on the spec's l.195–226 (instances, binding groups, meeting and boundary),
l.246 (a universal's dependency) and l.297–301 (fixpoint, fallback, acceptance).
The code is `src/program.rs` (groups and functions), `src/instance.rs`,
`State::of_group` and `State::call` in `src/solve.rs`, and `src/explore.rs`.
Passing a function value across a boundary into a given function type (l.204),
the facts other than Values, and a receiver's adjustments (l.333) are not yet
encoded.

## Meaning

Each binding group other than the run's entry is solved on its own to its
fixpoint, in every order, before the groups whose uses make instances of it
(l.220). Its functions' parameters' types are universal: each depends on the use
that chooses it, which lies outside the group and is never decided there, so what
reaches a universal stays in C. At the fixpoint, C is the group's classes whose
values are not closed, its universals, its undecided nodes and every relation on
them (l.219). A use in another group makes C anew and records it there; a use in
the group takes the body's own types.

## Readings

1. **The run.** The entry's inputs are the types the embedding declares, so the
   entry has no universal (l.206), and the run's instance of it is the entry's own
   solving continued: copying all of C into a group that adds nothing changes
   nothing. Fallback happens there alone. A group other than the entry has none
   (l.297), and what it leaves unsolved reaches fallback only as a copy (l.299),
   so a fallback after a group's C was copied out does not change that copy.
2. **A universal's dependency** (l.246) is a vertex its group never decides. A
   class with a lower bound on a universal, and a node that reads one, stay in C
   (Example 14, step 4, narrows `n == 0` in the instance).
3. **What C holds.** A class whose value is not closed at the fixpoint is of C,
   a solved one among them where a part is not (`Some(x)` in Example 6, l.447), as
   is each universal. A copy is made per root, so the equations among C's classes
   hold in it, and a kind the group solved a class to is recorded in its copy with
   that source, so that "with an equation" (l.279) stays as it was. A class
   outside C enters a copy as its closed type, written. Every relation of the
   group's log naming a class of C is copied, the splits it derived included,
   recorded by the use that made the instance or by the copy of the undecided
   node that recorded it.
4. **A use of a function in its own group** passes each argument into the
   parameter's universal type, a boundary (Example 10, l.516), and the body's
   result type enters the use's result as a lower bound, as a use's result does
   (note 07, reading 2).
5. **A group's outcome.** A group's text classes are reported with the values it
   solved them to, its decided nodes with their decisions, its checks where both
   sides are closed there, and its refusals. What it leaves in C needs neither a
   value nor a decision (l.301); its copies are checked where they are recorded.
6. **Several fixpoints.** A group whose solving reaches distinct fixpoints in
   different orders gives each to the groups after it, so an order dependence
   inside it shows in their outcomes.
7. **A lambda's group** lies inside its enclosing body's and is solved before it
   (l.220). The classes of enclosing groups its text names are shared: no unknowns
   of it (l.218), never solved or fallen back there, so its relations on them stay
   in its C, and a copy shares them (l.222). A lambda's use that reads one is
   `holes.md`, H5.
8. **Meeting and boundary** (l.226). A value passing into an unknown of the
   place's own group meets it. One passing into a binding of an enclosing group
   reached by name meets it where that binding's type is an unknown of its own
   group, judged by its text: a type the text writes, and a universal, are not
   (l.218). One passing through a reference from a binding of an enclosing group
   is a boundary, whatever class the place's last projection names, since that
   class is the lambda's own until the reference's target is known (Example 7).
9. **A call of a function value** reads the value's Values: the lambdas of the
   function values met in its class by equations or lower bounds (l.180, l.188).
   Decided, it makes an instance of each lambda's C, its arguments meeting the
   copies of the lambda's parameters' types, and the copy of its result type
   entering the call's result (l.203). A value whose Values names no lambda gives
   no result.
10. **A receiver's `T̄`** is universal (l.325), and the receiver's class holds the
   kind the text writes, so it is decided and universal at once: an argument
   passes into its copy by a boundary (Example 14, step 2). A node that reads
   whether a receiver is a reference depends on the use that chooses it (l.246)
   and is decided in the copies, although the kind the text writes already
   decides it.
