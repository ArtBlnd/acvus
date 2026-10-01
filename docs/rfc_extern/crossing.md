# crossing

How values and references cross between a run and an extern fn.

## 1. One entry

Each call has one entry. The entry opens the call's guards and hands them to the call's body inside a closure, so no guard outlives the call. Forgetting a guard is safe: nothing sound rests on a guard's drop, so a loan's end action is a representation's upkeep, never a fact soundness needs. Everything after the entry is safe.

## 2. Projections

A reference to a value of a transparent structured type (compiler: types §1) crosses as its projection: a value of the same structure in which each part that is not itself one of those types is a reference to that part's storage, shared or mutable as the reference is. An extern-defined part is lent as itself. The slice is the array's projection, so an extern fn receives a program's array only as a slice, and returns a sequence as a `Vec`. That `Vec` is a sequence (compiler: operators §3) of its element type: each element is the runtime's uniform value at that type, which the crossing holds and gives back as one value of it, one to one. A projection is a contract operation. Through a mutable projection only its parts are written; an enum's variant, an `Option`'s presence and an object whole change by value.

*Why.* The layout of such a value is the backend's, so no Rust reference to it exists. The registry refuses a declaration carrying an array type, whoever built it.

## 3. The crossing rule

A reference crosses only when three facts hold of its storage: it is live for the whole call, it is whole, and it is excluded from every other access. Where one does not hold, the crossing remedies it: the value moves, the value is copied, or the access itself is handed over.

- An awaited call of a callback takes its arguments by move. A handler lends none of its storage to it, shared or mutable. Each argument is placed in the callback run's frame, and a by-reference parameter of the callback is a reference to it there.
- A returning call of a callback takes plain references; it hands no work to an executor.

*Why.* Rust has no asynchronous drop: a call future dropped while it awaits cannot wait for the callback run's work still aloft, which would then read or write storage the handler has released or reused.

*Rejected.* A move made by the boundary behind a reference parameter: the declaration would not show it. Tracking in the glue whether a dropped call finished: it can only turn the violation into death, at a state and a check per call. Lending from storage the run keeps: a second mechanism for what a move already gives.

## 4. Results

The result of a returning call of a callback borrows only what that call lent, and no longer than that loan; the result of an awaited call borrows nothing the handler holds. The bound is a type.

## 5. Completion

An awaited entry completes only after every callback run it started has no work aloft, including runs whose call future was dropped.

## 6. The host

A host passes a run its inputs and reads its result and its contexts through the crossing a handler call uses, at the types the checker settled; it has no access path of its own. The host's API exposes no runtime value.

*Why.* The handler crossing is sound; a second crossing needs a second proof.
