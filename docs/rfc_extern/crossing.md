# crossing

How values and references cross between a run and an extern fn.

## 1. One entry

Each call has one entry. The entry opens the call's guards and hands them to the call's body inside a closure, so no guard escapes it and none can be forgotten. Everything after the entry is safe.

## 2. Slices, not arrays

An extern fn receives a program's array only as a slice, shared or mutable, and returns a sequence as a `Vec`. The view of an array as a slice is a contract operation. The registry refuses a declaration carrying an array type, whoever built it.

## 3. The crossing rule

A reference crosses only when three facts hold of its storage: it is live for the whole call, it is whole, and it is excluded from every other access. Where one does not hold, the crossing remedies it: the value moves, the value is copied, or the access itself is handed over.

- An awaited callback receives, at every by-reference position, only a reference the boundary builds from storage the run keeps: an in-place lend of storage the run keeps alive, a per-call move into the callback's frame, or a value kept until the run's last work. Otherwise it receives the value by move.
- Every `&mut` a callback receives while awaited crosses by move: the handler's storage moves into the callback run and moves back when the callback returns and its work has ended.
- A handler that lends writable storage to an awaited callback hands over the access itself, and gets it back only with the call's result. Dropping the call drops it.
- A synchronous call keeps plain references; it hands no work to an executor.

## 4. Results

A callback's result borrows only what was lent in place, and no longer than that loan. The bound is a type.

## 5. Completion

An awaited entry completes only after every callback run it started has no work aloft, including runs whose call future was dropped.
