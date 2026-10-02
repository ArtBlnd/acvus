# crossing

How values and references cross between a run and an extern fn.

## 1. One entry

Each call has one entry. It opens the call's arguments and the impls its site chose, each once, and hands them to the call's body inside a closure, so nothing it opens outlives the call. Nothing sound rests on a guard's drop: forgetting one is safe.

## 2. Forms

A declared type and the Rust type written for it correspond one to one, wherever it is written: a parameter, a result, or a function type's parameter or result.

- A value crosses by value.
- A reference names storage of the Rust type it is to: an extern-defined value, a primitive, text, a held value, or a slice.
- A value of a transparent structured type (compiler: types §1) is lent as the structure of references to its parts, the value `as_ref` or `as_mut` makes of it (compiler: ownership §5); a reference to the value itself is no form, except to an array, which is lent as a slice.
- A sequence is given as a `Vec` of held values, one to one.
- A function value is held at its function type.

## 3. Lending

A reference crosses only while its storage is live, whole, and excluded from every other access, for the whole call. Every call lends as Rust lends, a handler's call of a callback or an impl included, returning or awaited: the lender's borrow spans the call, and every run that may read what was lent is held by the call (§5), so none outlives the borrow.

## 4. Results

A call's result borrows only what that call lent, and no longer than that loan. The bound is a type.

## 5. Completion

A run is held by the future that calls it. Dropping a future ends the runs it holds, innermost first, and a run's storage is released only after what it holds has ended (runtime contract §4).

## 6. The host

A host passes a run its inputs and reads its result and its contexts through the crossing a handler call uses, at the types the checker settled. The host's API exposes no runtime value.
