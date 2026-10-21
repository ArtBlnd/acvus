# 04 Join and `≤`

Rests on the spec's l.80–91 (the join, the error join and `Error`), with l.74 (an
invariant part's relation is an equation). The code is `src/join.rs`, and
`src/join/tests.rs` exercises its contract.

## Meaning

The join of two closed types is the least type both are below, where the table
gives one: `!` is below every type, an exclusive reference joins a shared one as
shared, and two types of other nodes are each raised by casts to their nearest
common ancestor node, then their members united and their parts joined. At a
`Result`'s error part, the error join takes the place of the join: each side is
taken as a set of error types, `Error`'s variants flattened, elements of one node
are joined, and what remains is `!`, one type, or `Error` with a variant named
after each element. `S ≤ T` where `S ⊔ T = T`.

## Not owned here

- The join of a LitVar (l.171) unifies equivalence classes, and is note 06's.
- An equivalence class's value joins the types its sources give at its own
  constructor and members, linking parts through the casts' signatures (l.276,
  l.280, l.280), and is note 11's. So is the value of an equivalence class whose
  sources carry two identities (l.353): `F⟨ι₁⟩ ⊔ F⟨ι₂⟩` has no join here, and the
  equivalence class takes their nearest common ancestor that carries no identity.
- How a value is changed along `≤`, and the fields an object gains (l.282), are
  note 05's.
- Facts are joined beside a type (l.175), from note 06 on.

## Readings

1. **Invariant parts of one name are equal.** The join row names members and
   covariant parts only (l.86), and the relation between invariant parts is an
   equation (l.74). So two types whose invariant parts of one name differ have no
   join: `Box⟨A⟩ ⊔ Box⟨C⟩`, and `&Box⟨A⟩ ⊔ &Box⟨B⟩`, a reference's target being
   invariant.
2. **The rows do not overlap.** `!` is taken first, and the reference row before
   the common ancestor row. A cast keeps a reference's mutability (l.78), so an
   exclusive and a shared reference never have a common ancestor node, and the
   common ancestor row never applies to them.
3. **A type of a name.** An enum's name, a struct's module and an extern-defined
   type's name are the names that make an `Error`'s variants (l.95), so the
   elements are of distinct names when those names differ as text. An enum `A`
   and an extern-defined type `A` are not, and their error join has no result.
   Elements of one node are joined by `⊔` and never raised across nodes: `A` and
   `C` stay two variants though both step up to `B`.

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
