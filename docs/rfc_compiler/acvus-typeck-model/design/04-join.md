# 04 Join and `≤`

The code is `src/join.rs`, and `src/join/tests.rs` exercises its contract.

## Meaning

The join of two closed types is the least type both are below, where the table
gives one: `!` is below every type, an exclusive reference joins a shared one as
shared, and two types of other nodes are each raised by casts to their nearest
common ancestor node, then their members united, the parts of a structural type
joined by `⊔ₚ`, and every other part equal. `⊔ₚ` raises no part and takes no
reference as shared: two parts of one constructor are combined as `⊔` combines
them, `!` joins every part, and two parts of two constructors have no join.
`S ≤ T` where `S ⊔ T = T`.

## Not owned here

- The join of a LitVar unifies equivalence classes.
- An equivalence class's value joins the types its sources give at its own
  constructor and members, linking parts through the casts' signatures.
- Facts are joined beside a type.
- `Error`'s variants are made by `?` alone. Two `Error`s are joined as two enums of
  one name are: their variants united, each payload by `⊔ₚ`. `Error⟨E⟩`
  (`Term::ErrorOf`) is an `Error` too, so structural, its one part joined by `⊔ₚ`;
  its constructor is its own, so it has no join with an `Error` with variants.
  `Error` is structural so that what it holds joins part by part.

## Readings

1. **Invariant parts of one name are equal.** The relation between invariant parts is
   an equation. So `Box⟨A⟩ ⊔ Box⟨C⟩` has no join, nor has `&Box⟨A⟩ ⊔ &Box⟨B⟩`, a
   reference's target being invariant.
2. **The rows do not overlap.** `!` is taken first, and the reference row before
   the common ancestor row. A cast keeps a reference's mutability, so an
   exclusive and a shared reference never have a common ancestor node, and the
   common ancestor row never applies to them.
3. **No cast inside.** `⊔ₚ` is defined at one constructor and at `!` alone,
   so the parts of a structural type are never raised: `Option⟨A⟩ ⊔
   Option⟨C⟩` has no join though `A ⊔ C = B`, nor has `Option⟨&mut A⟩ ⊔
   Option⟨&A⟩`. A part of a part is joined by `⊔ₚ` again where it is a
   structural type's, and is equal otherwise: `Option⟨Option⟨!⟩⟩ ⊔
   Option⟨Option⟨A⟩⟩` is `Option⟨Option⟨A⟩⟩`, and `Option⟨Box⟨A⟩⟩ ⊔
   Option⟨Box⟨C⟩⟩` has no join.

## Evidence

Over a universe of 47 types, built from `!`, a number, three extern-defined types two
of which step up to the third, references to them, and one constructor of each kind
over those, the join is commutative, associative and idempotent, being undefined on
one side exactly where it is on the other. So the order `≤` the join defines is a
partial order there.

## Left open for change

- `join` returns the joined type alone. The casts a raise takes are in
  `Decls::raise`.
