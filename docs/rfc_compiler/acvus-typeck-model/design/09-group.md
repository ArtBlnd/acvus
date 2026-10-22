# 09 Binding groups, universals, C and instances

Rests on the spec's l.190–221 (instances, binding groups, meeting and boundary),
l.242 (a universal's dependency) and l.288–292 (fixpoint, fallback, acceptance).
The code is `src/program.rs` (groups and functions), `src/instance.rs`,
`State::of_group` in `src/solve.rs`, `State::call` in `src/record.rs`, and
`src/explore.rs`.
Passing a function value across a boundary into a given function type (l.199),
the facts other than Values, and a receiver's adjustments (l.324) are not yet
encoded.

## Meaning

Each binding group other than the run's entry is solved on its own to its
fixpoint, in every order, before the groups whose uses make instances of it
(l.215). Its functions' parameters' types are universal: each depends on the use
that chooses it, which lies outside the group and is never decided there, so what
reaches a universal stays in C. At the fixpoint, C is the group's classes whose
values are not closed, its universals, its undecided nodes and every relation on
them (l.214). A use in another group makes C anew and records it there; a use in
the group takes the body's own types.

## Readings

1. **The run.** The entry's inputs are the types the embedding declares, so the
   entry has no universal (l.201), and the run's instance of it is the entry's own
   solving continued: copying all of C into a group that adds nothing changes
   nothing. Fallback happens there alone. A group other than the entry has none
   (l.288), and what it leaves unsolved reaches fallback only as a copy (l.290),
   so a fallback after a group's C was copied out does not change that copy.
2. **A universal's dependency** (l.242) is a vertex its group never decides. A
   class with a lower bound on a universal, and a node that reads one, stay in C
   (Example 14, step 4, narrows `n == 0` in the instance).
3. **What C holds.** A class whose value is not closed at the fixpoint is of C,
   a solved one among them where a part is not (`Some(x)` in Example 6, l.441), as
   is each universal. A copy is made per root, so the equations among C's classes
   hold in it, and a kind the group solved a class to is recorded in its copy with
   that source, so that "with an equation" (l.275) stays as it was. A class
   outside C enters a copy as its closed type, written. Every relation of the
   group's log naming a class of C is copied, the splits it derived included,
   recorded by the use that made the instance or by the copy of the undecided
   node that recorded it. A relation the group split stays split in the copy:
   what its split recorded is copied with it, and no class of the copy takes
   there a constructor it did not hold in the group, so the split is not made
   again where a copied universal is an unknown (l.217, l.280; Example 14,
   l.588–590).
4. **A use of a function in its own group** passes each argument into the
   parameter's universal type, a boundary (Example 10, l.510), and the body's
   result type enters the use's result as a lower bound, as a use's result does
   (note 07, reading 2).
5. **A group's outcome.** A group's text classes are reported with the values it
   solved them to, its decided nodes with their decisions, its checks where both
   sides are closed there, and its refusals. What it leaves in C needs neither a
   value nor a decision (l.292); its copies are checked where they are recorded.
6. **Several fixpoints.** A group whose solving reaches distinct fixpoints in
   different orders gives each to the groups after it, so an order dependence
   inside it shows in their outcomes.
7. **A lambda's group** is its generic part (l.23, l.188): the classes joined to a
   capture and the constraints on them, its environment, are the enclosing
   group's, recorded there once whether or not a call reaches the lambda (note
   17). The rest lies inside its enclosing body's group and is solved before it
   (l.215). It names no class of an enclosing group, so its C holds none, and no
   copy records into one (l.217). A block expression is such a lambda, called
   where it lies (l.343).
8. **Meeting and boundary** (l.221). A value passing into an unknown of the
   place's own group meets it. One passing into a binding of an enclosing group
   reached by name meets it where that binding's type is an unknown of its own
   group, judged by its text: a type the text writes, and a universal, are not
   (l.213). One passing through a reference from a binding of an enclosing group
   is a boundary, whatever class the place's last projection names, since that
   class is the lambda's own until the reference's target is known (Example 7).
   A constraint of a lambda's environment keeps the place its text gives, and is
   recorded in the enclosing group (l.188).
9. **A call of a function value** reads the value's Values: the lambdas of the
   function values met in its class by equations or lower bounds (l.175, l.183).
   Decided, it makes an instance of each lambda's C, each argument passing into
   the copy of the generic part's parameter, or into the environment's parameter
   itself, and the copy of the generic part's result type, or the environment's,
   entering the call's result as a lower bound (l.198; note 17, row 3). A call
   whose callee's Values name no lambda decides nothing and is not decided then
   (note 08, reading 19).
10. **A receiver's `T̄`** is universal (l.316), and the receiver's class holds the
   kind the text writes, so it is decided and universal at once: an argument
   passes into its copy by a boundary (Example 14, step 2). A node that reads
   whether a receiver is a reference depends on the use that chooses it (l.242)
   and is decided in the copies, although the kind the text writes already
   decides it.
