# 08 Uses, lends, projections and bindings

The code is `src/node/`: what each node is, its decision and
what it records once decided in `mod.rs`, its positions and what it reads in
`position.rs`, places in `place.rs`, overload resolution in `overload.rs`, and the
generalization in `generalization.rs`.

## Meaning

A use calls the declarations its name reaches. It is decided as one candidate by
the value it reads at the positions where its candidates differ, narrowed argument
by argument, then by its result, each step keeping the candidates whose passing
goes by a coercion of the lowest rank. Decided, it records its candidate's
signature: each argument passing into its parameter, and the result type entering
its result. A lend reads whether its place is a reference, and gives a reference
to the place, or to the place's target at the lesser mutability. A projection
gives the class it applies to a structure constraint, naming a part's class: its
operand, or its operand's target where the operand reads as a reference. A binding
reads its position, or lends it where its path passes a reference.

## Readings

1. **One candidate reads nothing.** A use's candidates are the declarations its name
   reaches with as many parameters as it has arguments. It reads the
   positions where they differ, so a use with one candidate reads nothing:
   its generalization is that candidate, its type parameters made anew, recorded as its
   text is, by the generalization (`State::record_one`), and
   nothing is left to decide, so it has no node. A use with none is never decided.
2. **A candidate's variables are made at the decision.** They are unknowns of the
   use's binding group. Only the decided candidate records, so only its
   variables are made.
3. **Meeting or boundary at a parameter.** An argument passing into a parameter the
   signature writes as a variable passes into that unknown, a meeting; one passing
   into a written type passes into a type a declaration gives, decided, a
   boundary.
4. **The positions of a node**. A use's are its arguments and result,
   and the classes inside them down to where a candidate's signature writes a
   variable; a lend's are its place and the reference it gives.
5. **A lend through a reference.** Where the place reads as `&ₙ U`, the place records
   that it is a reference to a class made anew, which the lend's reference targets
   at the lesser mutability. Where the place reads otherwise, or nothing,
   the target is the place.
6. **A Deref records at once.** A Deref has one structure constraint to give, "a reference to `τ`",
   whatever its operand reads as, so it is decided as it is recorded, as a
   use with one candidate records its candidate, and its structure constraint is a
   source of whether its operand is a reference from the start.
7. **Auto-deref.** A projection other than a Deref reads whether its operand is a
   reference. Where the operand reads as `&ₙ U`, it records that it is a
   reference to a class made anew, and that class takes the projection's structure constraint;
   where it reads otherwise, or nothing, the operand takes it.
8. **The positions of a projection and a binding**. A projection's are
   its operand, the class it names, and the target its operand is known to be a
   reference to, where auto-deref records; a binding's are its position and its own
   type.
9. **What a read of whether a class is a reference leaves out.** For a lend, a
   projection and a binding, the read and the dependency both leave
   out every projection other than a Deref.
10. **A binding.** "Its equivalence class" is its position, which a
   binding that lends reads as a lend reads its place; so "the position's
   own" mutability is that lend's `n`. A read of the position passes its
   value into the binding's type, and a lend passes its reference into it, both
   meetings. The references a path passes are those its projections applied
   through by auto-deref after its last Deref, since `&π` matches a reference
   itself, as in Rust. The binding learns its path's projections'
   decisions by waiting for them, since the nodes recording into its position are
   reached along the links of the decompositions not yet made.
11. **What a use reads.** At each position where its candidates differ in a
   constructor or a member, the class there and the classes within the depth its
   candidates reach there, since the value read there is compared whole.
   A type parameter is an unknown made anew at the use,
   which decides nothing, so a position where the candidates differ in their
   type parameters alone reaches every candidate whatever is read there, and is not
   read (`State::differing`, which compares each candidate's type there as the
   narrowing reads it).
12. **Which nodes a reference-ness read waits for**. Whether a class is a
   reference is decided by its lower bounds and equations, so a node found
   at a class is left out where it cannot decide it: a use where no
   candidate would record a lower bound or an equation into that class, or where
   it has recorded whether the class is a reference; a lend at its place and a
   binding at its position, which record there only that it is a reference, and
   only where it already reads as one; and a call of a function value at
   its arguments. The second keeps two lends
   of one place from depending on each other.
13. **What candidates would record**. Once the generalization is recorded, a
   candidate records its result type at the result's unknowns, into each of them,
   and an argument the generalization does not hold whole passes into its parameter's
   type; before, its result type into the use's result. An argument passes into its parameter's type, a boundary that
   records into that type alone, and inside the argument's structure as it is at the
   moment each part passes into the parameter's part by the decomposition's rule:
   where the parameter writes a variable, an unknown, an equation records into the
   argument's part unless it is a structural type's, whose part lower bound
   records into the variable alone; where it writes a type, a boundary records
   into that type, and the part's parts pass on into its parts. Two constructors
   are linked as a decomposition links them, through a reborrow's targets and the first
   cast's parameter. A variable is an unknown, neither a reference nor not one, so
   a candidate records whether a class is a reference at the result's unknowns
   alone (`State::would_record_by`).
14. **The generalization**. Recorded
   once for each undecided node (`State::record_generalizations`, `Recorder::Generalization`):
   - a use of several candidates records the anti-unification of their
     signatures (`Generalization::of`): one constructor and members kept and its parts
     generalized in turn, an unknown made anew (`NodeAt::Generalized(k)`) where they
     differ, one for each tuple of the types they write there, and a reference
     not split at the top of a parameter or of the result, a cast going between
     two references of one constructor. Each unknown is recorded a
     reference, or not, where every candidate agrees there. Arguments pass into
     the generalization's parameters and result into the use's result, as a
     decided candidate's are recorded. Where every candidate writes one closed
     type, the use's constraint there is one alone, recorded with the
     text where the group that holds it lies (`State::record_closed`), the
     recorder still `Recorder::Generalization`; the node holds the other positions, which
     the choice decides (`NodeKind::Use`, none at those it does not hold).
     Decided, the use records an argument the generalization does not hold whole
     passing whole into the candidate's parameter,
     and the candidate's type at each of the result's unknowns entering it
     (`NodeAt::Chosen(k)`); its positions are those, so its result no longer
     waits for it but its unknowns do (`State::left_to_record`). Two arguments of
     one unknown meet in it: the operands of an overload set whose candidates
     type them alike are one type, and a cast between them is the user's to call.
   - a lend records that its result is a reference; its mutability and target it
     records when decided, so it still records into what a read of whether its
     result is a reference reads.
   - a projection or a binding has no generalization.
   A value read leaves out only what a node records once decided: every
   generalization is in every read, the node's own included. The model's earlier
   records, of the positions every candidate writes alike and of whether a
   class is a reference, are removed into it.
15. **An argument of `!`**. An argument position the use reads, read as
   `!`, decides it with the result `!`: the result takes `!` as a lower bound, and
   nothing else is recorded. A position the use does not read is not seen.
16. **A reference of undecided mutability.** A class read as a reference only by a
   generalization's record has a mutability the value read does not decide; a
   projection applies to its target all the same, and a lend or a binding whose
   lesser mutability is then undecided, a `&mut` one, decides nothing
   until the mutability is read.
17. **A broken dependency**. A node decided while undecided nodes broke
   their dependency on it records once they are decided, in the state;
   the reading of their reads only is removed.
18. **Which nodes record into a class**. The nodes that record
   relations into a class are those at whose positions it lies. For a use's read,
19. **A read that decides nothing**. A node is decided only by a
   value read that decides it: a use where exactly one candidate remains or an
   argument it reads is `!`, a lend and a binding where their lesser mutability is
   read, a call of a function value where its Values name a lambda; a projection,
   and a binding whose path passes no reference, always are. A node whose read decides nothing
   is not decided then and waits, undecided, for relations that make it decide.
   One undecided at its group's fixpoint stays in C, and one undecided once the
   run ends is refused.
20. **`e?`** is a use of the language's two candidates, `?(Option⟨T⟩) -> T`
   and `?(Result⟨T, E⟩) -> T`, which every program's declarations hold
   (`decl::TRY`), so it reads its operand's outer constructor alone, where they
   differ. Its exit's value has a class (`Item::Use`'s `exit`),
   which passes into the innermost body's result, and lies at
   the use's position, so that it, and the result it passes
   into, wait for the use. Decided, the
   use records into it the value `Option⟨K⟩`, or `Result⟨O, X⟩` with the
   candidate's `E` wrapped into `X` (`Relation::Wrap`), `K`
   and `O` made anew; decided as `!`, it records no exit, being not run.
21. **A use's result into a meeting**. Where the use's result
   class has no source in the value read and passes into unknowns, the result step
   narrows by the join: a candidate's result type reaches by `≤` its join with each
   such upper bound's other sources, as a meeting takes every value entering it; into a
   boundary, by a coercion as before (`State::meeting_places`, `rank_into_places`). A
   use whose result meets a class others enter is no longer refused against that
   class's value without it.
22. **A LitVar's default narrows at a fixpoint**. Where several candidates remain
   and a value read holds a LitVar, the narrowing again with the LitVar taken as its
   family's
   default is a step only where no other step is left (`State::steps`,
   `LitDefaults`), as a class falls back to its default only then: before, a use
   took the default at once, before a node that could unify its LitVar with a type
   was decided. I\*'s decision again
   is taken at the end, a fixpoint, and takes it.

## Left open for change

- `State::record_generalizations` and `Generalization::of`: the generalization of all an
  undecided node may become.
- `State::defer`: when a decided node's relations are recorded, and which reads
  leave them out.
