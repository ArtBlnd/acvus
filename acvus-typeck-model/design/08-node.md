# 08 Uses, lends, projections and bindings

Rests on the spec's l.303–345 (place, lends, patterns, structs, overload
resolution), with l.149–161 (projections), l.267 (what a node reads) and l.252 (a
use's record of whether a class is a reference). Uses, lends, projections and
bindings are encoded; l.252 is not yet. The code is `src/node.rs`.

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
6. **A Deref records at once.** A Deref has one kind to give, "a reference to `τ`"
   (l.155), whatever its operand reads as, so it is decided as it is recorded, as a
   use with one candidate is (reading 1), and its kind is a source of whether its
   operand is a reference (l.309) from the start.
7. **Auto-deref.** A projection other than a Deref reads whether its operand is a
   reference (l.267). Where the operand reads as `&ₙ U`, it records that it is a
   reference to a class made anew, and that class takes the projection's kind
   (l.309); where it reads otherwise, or nothing, the operand takes it. Example 8's
   `a.x` reads `A` as nothing and gives `A` the kind.
8. **The positions of a projection and a binding** (l.244 (b)). A projection's are
   its operand, the class it names, and the target its operand is known to be a
   reference to, where auto-deref records; a binding's are its position and its own
   type.
9. **What a read of whether a class is a reference leaves out.** For a lend, a
   projection and a binding, the read (l.267) and the dependency (l.248) both leave
   out every projection other than a Deref. Line 252 and the clause of l.248 about
   it are not yet encoded, so such a node still depends on every use that records
   into its place.
10. **A binding.** "Its equivalence class" (l.267) is its position (l.305), which a
   binding that lends reads as a lend reads its place (l.307); so "the position's
   own" mutability (l.317) is that lend's `n`. A read of the position passes its
   value into the binding's type, and a lend passes its reference into it, both
   meetings. The references a path passes are those its projections applied
   through by auto-deref after its last Deref, since `&π` matches a reference
   itself, as in Rust (patterns §1). How the binding learns its path's projections'
   decisions is `holes.md`, H5, and the model takes it as a reading
   (`BindingPath`).
