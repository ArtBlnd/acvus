# 08 Uses, lends, projections and bindings

Rests on the spec's l.303–345 (place, lends, patterns, structs, overload
resolution), with l.267 (what a node reads) and l.252 (a use's record of whether a
class is a reference). Uses and lends are encoded; projections, bindings and l.252
are not yet. The code is `src/node.rs`.

## Meaning

A use calls the declarations its name reaches. It is decided as one candidate by
the value it reads at the positions where its candidates differ, narrowed argument
by argument, then by its result, each step keeping the candidates whose passing
goes by a coercion of the lowest rank. Decided, it records its candidate's
signature: each argument passing into its parameter, and the result type entering
its result. A lend reads whether its place is a reference, and gives a reference
to the place, or to the place's target at the lesser mutability.

## Readings

1. **One candidate reads nothing.** A use reads the positions where its candidates
   differ (l.267), so a use with one candidate reads nothing, depends on nothing,
   and is decided as that candidate.
2. **A candidate's variables are made at the decision.** They are unknowns of the
   use's binding group (l.331). Only the decided candidate records, so only its
   variables are made.
3. **Meeting or boundary at a parameter.** An argument passing into a parameter the
   signature writes as a variable passes into that unknown, a meeting; one passing
   into a written type passes into a type a declaration gives, decided (l.218), a
   boundary (Example 2).
4. **The positions of a node** (l.244 (b)). A use's are its arguments and result,
   and the classes inside them down to where a candidate's signature writes a
   variable; a lend's are its place and the reference it gives.
5. **A lend through a reference.** Where the place reads as `&ₙ U`, the place records
   that it is a reference to a class made anew, which the lend's reference targets
   at the lesser mutability (l.307). Where the place reads otherwise, or nothing,
   the target is the place.
