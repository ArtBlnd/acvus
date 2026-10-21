# 05 Coercion and rank

Rests on the spec's l.93–111 (coercion and rank), l.282 (the steps of `≤`), l.263
and l.335 (the value a use reads, and the positions that decide nothing in it), and
l.129, l.164–165 and l.290 (a LitVar, what it admits and its default), with types §3
(a structured value coerces by its kind's shape, part by part) and §7 (a
conversion, exactly one). The code is `src/lit.rs`, `src/read.rs` and
`src/coerce.rs`, and `src/coerce/tests.rs` exercises their contract.

## Meaning

A coercion changes a value into a value of another type by the rows of the
Coercion table, composed `reborrow? ; cast* ; structural?`, or by `! ⇝ U`, or by
exactly one declared conversion where no composition reaches. Each part of a
structural stage is itself a coercion. The rank of a composition is the stages it
takes and its number of casts, counted down into its parts. Along the join's `≤`,
a value is changed by the same rows at each step, but an object gains fields
instead of dropping them, and at a `Result`'s error part elements stay at their
nodes. A value a use reads is a type some of whose positions are undecided, and a
position holding a LitVar decides nothing within its family.

## Not owned here

- Which value a use reads, and where narrowing compares ranks, are notes 08 and
  11's. Narrowing takes no conversion (l.335), so `coerce_read` takes none.
- When a coercion is checked, and where its functions are called (l.280, l.282),
  are note 11's.
- The fields an object gains along `≤` are written before the value is used whole
  (l.282, ownership §4); typeck records them as added and changes none.

## Readings

1. **One structural stage changes members and parts at once.** The object and enum
   rows change members and keep parts, and the part row keeps members and changes
   parts (l.97–106), and `structural?` takes at most one of them (l.109). Read so,
   `{x: A, y: u8} ⇝ {x: B}` reaches by no composition. Types §3 coerces a structured
   value by its kind's shape and part by part at once, and l.111 counts the stages
   of the parts' coercions inside the structural stage. So the structural stage at
   one constructor drops the fields the target does not name, adds the variants it
   names besides, and coerces each part both have (`holes.md`, T3).
2. **A part's coercion is any coercion.** The part row's `Tᵢ ⇝ Uᵢ` is the coercion
   relation, so a part may coerce by `!` or by a declared conversion:
   `Option⟨!⟩ ⇝ Option⟨u8⟩`, which `None` passed to a declaration needs.
3. **`!` and undecided positions take no stage.** `!` decides nothing (l.335), so
   `! ⇝ U` adds nothing to a rank, and a shape takes the structural stage only
   where a member changes or a part's coercion takes a stage. `Option⟨!⟩ ⇝
   Option⟨u8⟩` has rank `empty`. A coercion with a declared conversion in it has no
   rank (l.109).
4. **Comparing ranks.** By the highest stage, then the number of casts, then the
   lower stages from the highest down, where taking a stage ranks above not taking
   it.
5. **A wrap is exact.** `E ⇝ Error{N̄(Ē)}` where `E` is one of `Ē` (l.102): after
   the casts, the value's type is the variant's payload. Where the casts pass two
   types that are variants, two compositions reach (`holes.md`, H2).
6. **The steps of `≤`.** `S ≤ T` joins to `T`, so the steps are those of the join
   (l.86–88): a reborrow where an exclusive reference meets a shared one, the casts
   up to `T`'s node, then members gained, fields unchanged and variants added, and
   each part along `≤`; at a `Result`'s error part, by the error join (l.94–95),
   only within one node, and a type wrapped in the variant of `Error` of its node.
   Over the universe of note 04, `along_le` reaches exactly where `le` holds.
7. **A LitVar's position.** A LitVar reaches a LitVar of its family, and a scalar
   of its family that holds every value it holds (l.164), and nothing else. An
   integer literal is unsigned, a sign being an operator, and `isize` and `usize`
   are 64 bits wide, which the spec does not state.
8. **A candidate's type variables decide nothing.** A candidate's type parameters
   are unknowns made anew at the use (l.322), into which only the use itself
   records, and the value the use reads leaves out the relations it records itself
   (l.263). So a candidate's type, read, is undecided at each of its variables.

## Left open for change

- `HasTerm` is what closed types and values a use reads share, so matching a
  signature, stepping up the tree and raising take either.
- A coercion keeps the casts, the members dropped and added, and each part's
  coercion, so the outcome can name every function a check calls (I3).
