# 16 Inputs up to a size (draft)

A draft: its open questions are to be agreed before its contract is.

## Meaning

Every program up to a given size is generated, in order of size, and each is
explored in every order and given to the checks. Enumerated in
order of size, the first program a check fails is one of the least size that
fails it, so no shrinking is needed: the witness is already minimal by size.

## Open questions

1. **What is generated.** The model takes items, the constraints a text gives.
   Items generated freely include programs no text gives, an `Equation` between any
   two classes, a `Bind` whose path cannot occur, and a hole found there would be
   the generator's.
2. **The fragment.** `let`, assignment, a call, `&`, `&mut`, `*`, a field, `if let`
   with one pattern, `while`, a lambda with its captures, and a call of a function
   value are proposed.
3. **The declarations.** A fixed menu, along the axes: whether
   a type is a reference, mutability, the literal families,
   the stages of rank, casts and conversions, a
   view, `!`, and overloads that differ at one position or at
   several. The menu is a choice, and what it leaves out no enumeration reaches.
4. **The size.** The number of statements and of expressions in them is proposed.
   The number of states exploration visits grows fast with it. The size that can be
   reached is not yet
   measured.
