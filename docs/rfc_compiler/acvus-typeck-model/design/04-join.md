# 04 Join and `≤`

Rests on the spec's l.80–91 (the join, the join of parts `⊔ₚ`, and `Error`), with
l.74 (the parts of a structural type are joined without casts, and an invariant
part's relation is an equation). The code is `src/join.rs`, and
`src/join/tests.rs` exercises its contract.

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

- The join of a LitVar (l.169) unifies equivalence classes, and is note 06's.
- An equivalence class's value joins the types its sources give at its own
  constructor and members, linking parts through the casts' signatures (l.274,
  l.278), and is note 11's. So is the value of an equivalence class whose
  sources carry two identities (l.351): `F⟨ι₁⟩ ⊔ F⟨ι₂⟩` has no join here, and the
  equivalence class takes their nearest common ancestor that carries no identity.
- How a value is changed along `≤`, and the fields an object gains (l.280), are
  note 05's.
- Facts are joined beside a type (l.173), from note 06 on.
- `Error`'s variants are made by `?` alone (l.91, l.337; note 08, reading 20). Two `Error`s are joined as two enums of one name are: their variants
  united, each payload by `⊔ₚ`. `Error⟨E⟩` (`Term::ErrorOf`) is an `Error` too, so
  structural, its one part joined by `⊔ₚ`; its constructor is its own, so it has no
  join with an `Error` with variants (l.91), as 준량 confirmed: `Error` is
  structural so that what it holds joins part by part.

## Readings

1. **Invariant parts of one name are equal.** l.86 needs every part that is no
   structural type's equal, and the relation between invariant parts is an
   equation (l.74). So `Box⟨A⟩ ⊔ Box⟨C⟩` has no join, nor has `&Box⟨A⟩ ⊔
   &Box⟨B⟩`, a reference's target being invariant.
2. **The rows do not overlap.** `!` is taken first, and the reference row before
   the common ancestor row. A cast keeps a reference's mutability (l.78), so an
   exclusive and a shared reference never have a common ancestor node, and the
   common ancestor row never applies to them.
3. **No cast inside.** `⊔ₚ` is defined at one constructor and at `!` alone
   (l.89), so the parts of a structural type are never raised: `Option⟨A⟩ ⊔
   Option⟨C⟩` has no join though `A ⊔ C = B`, nor has `Option⟨&mut A⟩ ⊔
   Option⟨&A⟩`. A part of a part is joined by `⊔ₚ` again where it is a
   structural type's, and is equal otherwise: `Option⟨Option⟨!⟩⟩ ⊔
   Option⟨Option⟨A⟩⟩` is `Option⟨Option⟨A⟩⟩`, and `Option⟨Box⟨A⟩⟩ ⊔
   Option⟨Box⟨C⟩⟩` has no join.

## Evidence

Over a universe of 47 types, built from `!`, a number, three extern-defined types
two of which step up to the third, references to them, and one constructor of each
kind over those, the join is commutative, associative and idempotent, being
undefined on one side exactly where it is on the other. So the order `≤` the join
defines is a partial order there. Without note 03's reading 2, associativity fails
(note 03).

## Left open for change

- `join` returns the joined type alone. The casts a raise takes are in
  `Decls::raise`, for note 05 to name the functions a step of `≤` calls.
