# 08 Uses, lends, projections and bindings

Rests on the spec's l.303–345 (place, lends, patterns, structs, overload
resolution), with l.149–161 (projections), l.267 (what a node reads) and l.252 (a
use's record of whether a class is a reference), l.269 (what a broken dependency
defers) and l.335 (what all candidates share). The code is `src/node.rs`.

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
   reaches with as many parameters as it has arguments (l.331). It reads the
   positions where they differ (l.267), so a use with one candidate reads nothing,
   and all of it is shared (l.335): it is decided as that candidate as it is
   recorded. A use with none is refused as it is recorded.
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
   out every projection other than a Deref.
10. **A binding.** "Its equivalence class" (l.267) is its position (l.305), which a
   binding that lends reads as a lend reads its place (l.307); so "the position's
   own" mutability (l.317) is that lend's `n`. A read of the position passes its
   value into the binding's type, and a lend passes its reference into it, both
   meetings. The references a path passes are those its projections applied
   through by auto-deref after its last Deref, since `&π` matches a reference
   itself, as in Rust (patterns §1). How the binding learns its path's projections'
   decisions is `holes.md`, O1, and the model takes it as a reading
   (`BindingPath`).
11. **What a use reads.** At each position where its candidates differ, the class
   there and the classes within the depth its candidates reach there (l.244 (b)),
   since the value read there is compared whole (l.337). Example 12 rests on it:
   `x + y` reads `Y`, the target of `&y`, and so depends on `y + 1`.
12. **Which uses a reference-ness read waits for** (l.248). Whether a class is a
   reference is decided by its lower bounds and equations (l.309), so a use found
   at a class (l.250) is left out where no candidate would record a lower bound or
   an equation into that class, or where it has recorded whether the class is a
   reference. The pick example (l.263) needs the first: `pick` lies at `A`, its
   argument, which its candidates record nothing into.
13. **What candidates would record** (l.252). A candidate records its result type
   into its result, and, inside each argument's kind as it is at the moment, the
   equations its parameter's invariant parts give (l.286), each part beneath an
   equated one equated too. A variable there is an unknown, neither a reference
   nor not one, so a candidate that writes one agrees with none.
14. **What all candidates share** (l.335). Each argument and the result where every
   candidate's signature writes one type with no variable, recorded at once as the
   decided candidate would record it. Line 252 is the reference-ness part of it.
15. **An argument of `!`** (l.344). An argument position the use reads, read as
   `!`, decides it with the result `!`: the result takes `!` as a lower bound, and
   nothing else is recorded. A position the use does not read is not seen.
16. **A reference of undecided mutability.** A class read as a reference only by a
   use's record (l.252) has a mutability the value read does not decide; a
   projection applies to its target all the same, and a lend or a binding whose
   reference's mutability is then undecided gives no result (`holes.md`, O3).
17. **A broken dependency** (l.269). A node decided while undecided nodes broke
   their dependency on it records once they are decided; whether that holds of
   the state or of their reads only is `holes.md`, O2, and the model takes it as a
   reading (`Deferral`).
