# 16 Inputs up to a size (draft)

Rests on the crate's second end ("over every input up to a given size", README),
and on the spec's l.221–233 (what a text records) and l.294–312 (places and
patterns). A draft: its open questions are to be agreed before its contract is.

## Meaning

Every program up to a given size is generated, in order of size, and each is
explored in every order and given to the checks of notes 13–15. Enumerated in
order of size, the first program a check fails is one of the least size that
fails it, so no shrinking is needed: the witness is already minimal by size.

## Open questions

1. **What is generated.** The model takes items, the constraints a text gives.
   Items generated freely include programs no text gives, an `Equation` between any
   two classes, a `Bind` whose path cannot occur, and a hole found there would be
   the generator's. Two ways to exclude them:
   - a small surface fragment, generated, and a lowering of it to items that
     encodes l.221–233 and l.294–312, a primitive of its own with its own note,
     and a new place for errors;
   - items, generated under a predicate saying which items some text gives, which
     states the same rules as the lowering, from the other side.
   The first is proposed: the rules are stated once, in the direction the spec
   states them.
2. **The fragment.** `let`, assignment, a call, `&`, `&mut`, `*`, a field, `if let`
   with one pattern, `while`, a lambda with its captures, and a call of a function
   value are proposed, as what the holes so far needed.
3. **The declarations.** A fixed menu, along the axes the spec decides by: whether
   a type is a reference (l.248), mutability (l.298), the literal families
   (l.129), the stages of rank (l.109–111), casts and conversions (l.78, l.107), a
   view (l.156), `!` (l.335), and overloads that differ at one position or at
   several. The menu is a choice, and what it leaves out no enumeration reaches.
4. **The size.** The number of statements and of expressions in them is proposed.
   The number of states exploration visits grows fast with it: Example 14 needed
   sampling (note 13, reading 2). The size that can be reached is not yet measured.
