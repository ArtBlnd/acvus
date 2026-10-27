# 05 Coercion and rank

The code is `src/lit.rs` and `src/coerce.rs`, and `src/coerce/tests.rs`
exercises their contract.

## Meaning

A coercion changes a whole value into a value of another type by the rows of the
Coercion table, composed `reborrow? ; cast* ; structural?`, or by `! ⇝ U`, or by
exactly one declared conversion where no composition reaches. Each part of a
structural stage is changed by its part's coercion `⇝ₚ`, which composes the
object, enum and part rows and `! ⇝ U` alone, so no reborrow, cast or
conversion is taken at a part. The rank of a composition is the stages it takes
and its number of casts, its parts' coercions adding the structural stage alone.
Along the join's `≤`, a value is changed by the same rows at each step, but an
object gains fields instead of dropping them.

## Not owned here

- The fields an object gains along `≤` are written before the value is used whole;
  typeck records them as added and changes none.

## Readings

1. **One structural stage changes members and parts at once.** The object and enum
   rows change members and keep parts, and the part row keeps members and changes
   parts, and `structural?` takes at most one of them. Read so,
   `{x: E{V(!)}, y: u8} ⇝ {x: E{V(!), W(!)}}` reaches by no composition.
   The structural stage at one constructor drops the fields the target does not
   name, adds the variants it names besides, and changes each part both have by
   its `⇝ₚ`.
2. **A part's coercion is `⇝ₚ`.**
   `Option⟨!⟩ ⇝ Option⟨u8⟩`, which `None` passed to a declaration needs, reaches,
   and `Option⟨u16⟩ ⇝ Option⟨u8⟩` does not, though a conversion `u16 → u8` is
   declared.
3. **`!` takes no stage.** `!` decides nothing, so
   `! ⇝ U` adds nothing to a rank, and a shape takes the structural stage only
   where a member changes or a part's coercion takes a stage. `Option⟨!⟩ ⇝
   Option⟨u8⟩` has rank `empty`. A declared conversion has no rank.
4. **Comparing ranks.** By the highest stage, then the number of casts, then the
   lower stages from the highest down, where taking a stage ranks above not taking
   it. A part takes the structural stage alone, so a structural stage with a cast
   below it is a whole value's: a cast `A → E{V(!)}` then widening to `E{V(!),
   W(!)}`.
5. **No row wraps a value in an `Error`.** An `Error`, which `?` alone makes, is
   written only where a value leaves the program, and
   `?` wraps by a decomposition, so the Coercion table has no wrap row and
   no value of another type reaches an `Error` by a coercion;
   an `Error` reaches one by the enum row, its variants added.
6. **The steps of `≤`.** `S ≤ T` joins to `T`, so the steps are those of the join:
   a reborrow where an exclusive reference passes into a shared one, the casts
   up to `T`'s node, then members gained, fields unchanged and variants added, and
   each part of a structural type along `≤ₚ`, by the same rows and no cast.
7. **A LitVar.** A LitVar admits a scalar of its family that holds every value
   it holds, and nothing else. An integer literal of the model is unsigned: a
   negative literal, which is a literal once `core::neg` is gone, is not
   encoded.

## Left open for change

- `HasTerm` is what closed types and the certificate's values share, so
  matching a signature, stepping up the tree and raising take either.
- A coercion keeps the casts, the members dropped and added, and each part's
  coercion, so the outcome can name every function a check calls.
