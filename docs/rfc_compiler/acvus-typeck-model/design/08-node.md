# 08 Uses, lends, projections and bindings

Rests on the spec's l.292–334 (place, lends, patterns, structs, overload
resolution), with l.142–154 (projections), l.261 (what a node reads), l.246 (the
common part of all an undecided node may become) and l.263 (what a broken
dependency defers). The code is `src/node/`: what each node is, its decision and
what it records once decided in `mod.rs`, its positions and what it reads in
`position.rs`, places in `place.rs`, overload resolution in `overload.rs`, and the
common part in `common.rs`.

## Meaning

A use calls the declarations its name reaches. It is decided as one candidate by
the value it reads at the positions where its candidates differ, narrowed argument
by argument, then by its result, each step keeping the candidates whose passing
goes by a coercion of the lowest rank. Decided, it records its candidate's
signature: each argument passing into its parameter, and the result type entering
its result. A lend reads whether its place is a reference, and gives a reference
to the place, or to the place's target at the lesser mutability. A projection
gives the class it applies to a kind known in part, naming a part's class: its
operand, or its operand's target where the operand reads as a reference. A binding
reads its position, or lends it where its path passes a reference.

## Readings

1. **One candidate reads nothing.** A use's candidates are the declarations its name
   reaches with as many parameters as it has arguments (l.320). It reads the
   positions where they differ (l.261), so a use with one candidate reads nothing:
   it is decided as that candidate as it is recorded, with no common part. A use with none is never decided (l.333; reading 19).
2. **A candidate's variables are made at the decision.** They are unknowns of the
   use's binding group (l.320). Only the decided candidate records, so only its
   variables are made.
3. **Meeting or boundary at a parameter.** An argument passing into a parameter the
   signature writes as a variable passes into that unknown, a meeting; one passing
   into a written type passes into a type a declaration gives, decided (l.211), a
   boundary (Example 2).
4. **The positions of a node** (l.238 (b)). A use's are its arguments and result,
   and the classes inside them down to where a candidate's signature writes a
   variable; a lend's are its place and the reference it gives.
5. **A lend through a reference.** Where the place reads as `&ₙ U`, the place records
   that it is a reference to a class made anew, which the lend's reference targets
   at the lesser mutability (l.296). Where the place reads otherwise, or nothing,
   the target is the place.
6. **A Deref records at once.** A Deref has one kind to give, "a reference to `τ`"
   (l.148), whatever its operand reads as, so it is decided as it is recorded, as a
   use with one candidate is (reading 1), and its kind is a source of whether its
   operand is a reference (l.298) from the start.
7. **Auto-deref.** A projection other than a Deref reads whether its operand is a
   reference (l.261). Where the operand reads as `&ₙ U`, it records that it is a
   reference to a class made anew, and that class takes the projection's kind
   (l.298); where it reads otherwise, or nothing, the operand takes it. Example 8's
   `a.x` reads `A` as nothing and gives `A` the kind.
8. **The positions of a projection and a binding** (l.238 (b)). A projection's are
   its operand, the class it names, and the target its operand is known to be a
   reference to, where auto-deref records; a binding's are its position and its own
   type.
9. **What a read of whether a class is a reference leaves out.** For a lend, a
   projection and a binding, the read (l.261) and the dependency (l.242) both leave
   out every projection other than a Deref.
10. **A binding.** "Its equivalence class" (l.261) is its position (l.294), which a
   binding that lends reads as a lend reads its place (l.296); so "the position's
   own" mutability (l.306) is that lend's `n`. A read of the position passes its
   value into the binding's type, and a lend passes its reference into it, both
   meetings. The references a path passes are those its projections applied
   through by auto-deref after its last Deref, since `&π` matches a reference
   itself, as in Rust (patterns §1). The binding learns its path's projections'
   decisions by waiting for them, since the nodes recording into its position are
   reached along the links of the splits not yet made (l.244; `holes.md`, O1).
11. **What a use reads.** At each position where its candidates differ, the class
   there and the classes within the depth its candidates reach there (l.238 (b)),
   since the value read there is compared whole (l.326). Example 12 rests on it:
   `x + y` reads `Y`, the target of `&y`, and so depends on `y + 1`.
12. **Which nodes a reference-ness read waits for** (l.242). Whether a class is a
   reference is decided by its lower bounds and equations (l.298), so a node found
   at a class (l.244) is left out where it cannot decide it: a use where no
   candidate would record a lower bound or an equation into that class, or where
   it has recorded whether the class is a reference; a lend at its place and a
   binding at its position, which record there only that it is a reference, and
   only where it already reads as one (l.296); and a call of a function value at
   its arguments. The pick example (l.257) needs the first: `pick` lies at `A`, its
   argument, which its candidates record nothing into. The second keeps two lends
   of one place from depending on each other (Example 12's two `&y`; `holes.md`,
   O5).
13. **What candidates would record** (l.246). Once the common part is recorded, a
   candidate records its result type at the result's unknowns, into each of them,
   and an argument the common part does not hold whole passes into its parameter's
   type; before, its result type into the use's result. An argument passes into its parameter's type, a boundary that
   records into that type alone, and inside the argument's kind as it is at the
   moment each part passes into the parameter's part by the split's rule (l.278):
   where the parameter writes a variable, an unknown, an equation records into the
   argument's part unless it is a structural type's, whose part lower bound
   records into the variable alone; where it writes a type, a boundary records
   into that type, and the part's parts pass on into its parts. Two constructors
   are linked as a split links them, through a reborrow's targets and the first
   cast's parameter. A variable is an unknown, neither a reference nor not one, so
   a candidate records whether a class is a reference at the result's unknowns
   alone (`State::would_record_by`).
14. **The common part** (l.246, since the text's change of 2026-10-22). Recorded
   once for each undecided node (`State::record_at_once`, `Recorder::Common`):
   - a use of several candidates records the anti-unification of their
     signatures (`CommonPart::of`): one constructor and members kept and its parts
     generalized in turn, an unknown made anew (`NodeAt::Common(k)`) where they
     differ, one for each tuple of the types they write there, and a reference
     not split at the top of a parameter or of the result, a cast going between
     two references of one constructor (l.66, l.78). Each unknown is recorded a
     reference, or not, where every candidate agrees there. Arguments pass into
     the common parameters and the common result into the use's result, as a
     decided candidate's are recorded. Where every candidate writes one closed
     type, the use's constraint there is one alone (l.186, G4), recorded with the
     text where the group that holds it lies (`State::record_closed`), the
     recorder still `Recorder::Common`; the node holds the other positions, which
     the choice decides (`NodeKind::Use`, none at those it does not hold).
     Decided, the use records an argument the common part does not hold whole
     passing whole into the candidate's parameter,
     and the candidate's type at each of the result's unknowns entering it
     (`NodeAt::Chosen(k)`); its positions are those, so its result no longer
     waits for it but its unknowns do (`State::left_to_record`). Two arguments of
     one unknown meet in it: the operands of an overload set whose candidates
     type them alike are one type, and a cast between them is the user's to call.
   - a lend records that its result is a reference; its mutability and target it
     records when decided, so it still records into what a read of whether its
     result is a reference reads.
   - a projection or a binding has no common part.
   A value read leaves out only what a node records once decided (l.261): every
   common part is in every read, the node's own included. The model's earlier
   records, of the positions every candidate writes alike (l.324) and of whether a
   class is a reference (l.246), are removed into it.
15. **An argument of `!`** (l.333). An argument position the use reads, read as
   `!`, decides it with the result `!`: the result takes `!` as a lower bound, and
   nothing else is recorded. A position the use does not read is not seen.
16. **A reference of undecided mutability.** A class read as a reference only by a
   common part's record (l.246) has a mutability the value read does not decide; a
   projection applies to its target all the same, and a lend or a binding whose
   lesser mutability (l.296) is then undecided, a `&mut` one, decides nothing
   until the mutability is read (reading 19; `holes.md`, O3).
17. **A broken dependency** (l.263). A node decided while undecided nodes broke
   their dependency on it records once they are decided, in the state, as l.263
   states (`holes.md`, O2); the reading of their reads only is removed.
18. **Which nodes record into a class** (l.238 (b), l.244). The nodes that record
   relations into a class are those at whose positions it lies. For a use's read,
   which is of a value, the spec makes no narrowing such as l.242 makes for a
   reference read. A reading that kept only the nodes that would record a lower
   bound, an equation or a kind into the class left out what l.278 makes a boundary
   record into a class holding a LitVar alone, and made the outcome depend on the
   order; it is removed (`holes.md`, H7).
19. **A read that decides nothing** (l.261, l.333). A node is decided only by a
   value read that decides it: a use where exactly one candidate remains or an
   argument it reads is `!`, a lend and a binding where their lesser mutability is
   read, a call of a function value where its Values name a lambda; a projection,
   and a binding whose path passes no reference, always are. A node whose read decides nothing
   is not decided then and waits, undecided, for relations that make it decide.
   One undecided at its group's fixpoint stays in C, and one undecided once the
   run ends is refused (l.290), as the pick example's `pick` is at its call
   (l.259) and Example 13's `parse` (l.561).
20. **`e?`** (l.337) is a use of the language's two candidates, `?(Option⟨T⟩) -> T`
   and `?(Result⟨T, E⟩) -> T`, which every program's declarations hold
   (`decl::TRY`), so it reads its operand's outer constructor alone, where they
   differ. Its exit's value has a class the text gives (`Item::Use`'s `exit`),
   which passes into the innermost body's result as the text records it (l.219,
   l.223), and lies at the use's position, so that it, and the result it passes
   into, wait for the use (l.238 (b)), as the text has it since d2d9903b. Decided, the
   use records into it the value `Option⟨K⟩`, or `Result⟨O, X⟩` with the
   candidate's `E` wrapped into `X` (`Relation::Wrap`, note 11, reading 12), `K`
   and `O` made anew; decided as `!`, it records no exit, being not run (l.333).
21. **A use's result into a meeting** (l.326, since c30509e3). Where the use's result
   class has no source in the value read and passes into unknowns, the result step
   narrows by the join: a candidate's result type reaches by `≤` its join with each
   such place's other sources, as a meeting takes every value entering it; into a
   boundary, by a coercion as before (`State::meeting_places`, `rank_into_places`). A
   use whose result meets a class others enter is no longer refused against that
   class's value without it (G2 of the review of 2026-10-22).

## Left open for change

- `State::record_at_once` and `CommonPart::of`: the common part of all an
  undecided node may become (l.246).
- `State::defer`: when a decided node's relations are recorded, and which reads
  leave them out (l.263; `holes.md`, O2).
