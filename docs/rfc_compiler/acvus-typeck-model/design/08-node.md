# 08 Uses, lends, projections and bindings

Rests on the spec's l.289–331 (place, lends, patterns, structs, overload
resolution), with l.142–154 (projections), l.260 (what a node reads) and l.245 (a
use's record of whether a class is a reference), l.262 (what a broken dependency
defers) and l.321 (what all candidates share). The code is `src/node/`: what each
node is, its decision and what it records once decided in `mod.rs`, its positions
and what it reads in `position.rs`, places in `place.rs`, overload resolution in
`overload.rs`, and what an undecided use's candidates share in `shared.rs`.

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
   reaches with as many parameters as it has arguments (l.317). It reads the
   positions where they differ (l.260), so a use with one candidate reads nothing,
   and all of it is shared (l.321): it is decided as that candidate as it is
   recorded. A use with none is never decided (l.330; reading 19).
2. **A candidate's variables are made at the decision.** They are unknowns of the
   use's binding group (l.317). Only the decided candidate records, so only its
   variables are made.
3. **Meeting or boundary at a parameter.** An argument passing into a parameter the
   signature writes as a variable passes into that unknown, a meeting; one passing
   into a written type passes into a type a declaration gives, decided (l.211), a
   boundary (Example 2).
4. **The positions of a node** (l.237 (b)). A use's are its arguments and result,
   and the classes inside them down to where a candidate's signature writes a
   variable; a lend's are its place and the reference it gives.
5. **A lend through a reference.** Where the place reads as `&ₙ U`, the place records
   that it is a reference to a class made anew, which the lend's reference targets
   at the lesser mutability (l.293). Where the place reads otherwise, or nothing,
   the target is the place.
6. **A Deref records at once.** A Deref has one kind to give, "a reference to `τ`"
   (l.148), whatever its operand reads as, so it is decided as it is recorded, as a
   use with one candidate is (reading 1), and its kind is a source of whether its
   operand is a reference (l.295) from the start.
7. **Auto-deref.** A projection other than a Deref reads whether its operand is a
   reference (l.260). Where the operand reads as `&ₙ U`, it records that it is a
   reference to a class made anew, and that class takes the projection's kind
   (l.295); where it reads otherwise, or nothing, the operand takes it. Example 8's
   `a.x` reads `A` as nothing and gives `A` the kind.
8. **The positions of a projection and a binding** (l.237 (b)). A projection's are
   its operand, the class it names, and the target its operand is known to be a
   reference to, where auto-deref records; a binding's are its position and its own
   type.
9. **What a read of whether a class is a reference leaves out.** For a lend, a
   projection and a binding, the read (l.260) and the dependency (l.241) both leave
   out every projection other than a Deref.
10. **A binding.** "Its equivalence class" (l.260) is its position (l.291), which a
   binding that lends reads as a lend reads its place (l.293); so "the position's
   own" mutability (l.303) is that lend's `n`. A read of the position passes its
   value into the binding's type, and a lend passes its reference into it, both
   meetings. The references a path passes are those its projections applied
   through by auto-deref after its last Deref, since `&π` matches a reference
   itself, as in Rust (patterns §1). How the binding learns its path's projections'
   decisions is `holes.md`, O1, and the model takes it as a reading
   (`BindingPath`).
11. **What a use reads.** At each position where its candidates differ, the class
   there and the classes within the depth its candidates reach there (l.237 (b)),
   since the value read there is compared whole (l.323). Example 12 rests on it:
   `x + y` reads `Y`, the target of `&y`, and so depends on `y + 1`.
12. **Which nodes a reference-ness read waits for** (l.241). Whether a class is a
   reference is decided by its lower bounds and equations (l.295), so a node found
   at a class (l.243) is left out where it cannot decide it: a use where no
   candidate would record a lower bound or an equation into that class, or where
   it has recorded whether the class is a reference; a lend at its place and a
   binding at its position, which record there only that it is a reference, and
   only where it already reads as one (l.293); and a call of a function value at
   its arguments. The pick example (l.256) needs the first: `pick` lies at `A`, its
   argument, which its candidates record nothing into. The second keeps two lends
   of one place from depending on each other (Example 12's two `&y`; `holes.md`,
   O5).
13. **What candidates would record** (l.245). A candidate records its result type
   into its result, and, inside each argument's kind as it is at the moment, the
   equations its parameter's invariant parts give (l.277), each part beneath an
   equated one equated too. A variable there is an unknown, neither a reference
   nor not one, so a candidate that writes one agrees with none.
14. **What all candidates share** (l.321). Each argument and the result where every
   candidate's signature writes one type with no variable, recorded at once as the
   decided candidate would record it. Line 245 is the reference-ness part of it.
15. **An argument of `!`** (l.330). An argument position the use reads, read as
   `!`, decides it with the result `!`: the result takes `!` as a lower bound, and
   nothing else is recorded. A position the use does not read is not seen.
16. **A reference of undecided mutability.** A class read as a reference only by a
   use's record (l.245) has a mutability the value read does not decide; a
   projection applies to its target all the same, and a lend or a binding whose
   lesser mutability (l.293) is then undecided, a `&mut` one, decides nothing
   until the mutability is read (reading 19; `holes.md`, O3).
17. **A broken dependency** (l.262). A node decided while undecided nodes broke
   their dependency on it records once they are decided; whether that holds of
   the state or of their reads only is `holes.md`, O2, and the model takes it as a
   reading (`Deferral`).
18. **Which nodes record into a class** (l.237 (b), l.243). The nodes that record
   relations into a class are those at whose positions it lies. For a use's read,
   which is of a value, the spec makes no narrowing such as l.241 makes for a
   reference read. A reading that kept only the nodes that would record a lower
   bound, an equation or a kind into the class left out what l.277 makes a boundary
   record into a class holding a LitVar alone, and made the outcome depend on the
   order; it is removed (`holes.md`, H7).
19. **A read that decides nothing** (l.260, l.330). A node is decided only by a
   value read that decides it: a use where exactly one candidate remains or an
   argument it reads is `!`, a lend and a binding where their lesser mutability is
   read, a call of a function value where its Values name a lambda; a projection,
   and a binding whose path passes no reference, always are. A node whose read decides nothing
   is not decided then and waits, undecided, for relations that make it decide.
   One undecided at its group's fixpoint stays in C, and one undecided once the
   run ends is refused (l.287), as the pick example's `pick` is at its call
   (l.258) and Example 13's `parse` (l.553).

## Left open for change

- `State::record_at_once`: what a use records before it is decided, the positions
  its candidates write alike (l.321) and whether a class is a reference where they
  agree (l.245).
- `State::defer`: when a decided node's relations are recorded, and which reads
  leave them out (l.262; `holes.md`, O2).
