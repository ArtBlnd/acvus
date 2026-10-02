# runtime contract

The seam between extern fns and whatever runs them. Every extern fn is callable through the contract alone, whatever implements it.

## 1. Values

A value's representation is not part of the contract. The extern side reaches values only through the contract's operations, generic over the runtime and dispatched statically.

Nothing turns a Rust value into a runtime value except a crossing at the type the checker settled for its site. A value made at one type is never read at another: a path from a Rust type to a runtime value would make a function written for one type callable with a value made from another.

A runtime value is copyable data and owns nothing. The extern side owns one only through its owning holder, indexed by the value's type and releasing it on drop. A type variable's value is held so, indexed by the variable; a Rust value of a type that crosses may be held so at its own type.

An extern-defined generic type is stored as one Rust value whose type variables' parts are held values, indexed by phantom type parameters. Its instantiations that differ only in those indices are read as one another, so they are layout- and drop-equivalent: each index is parametric, and no projection through it determines a layout.

## 2. Operations

- **Entry.** Each call has one entry (crossing §1).
- **Kinds.** Values are made, taken and lent by kind: an extern-defined value, a primitive, text, an option, a result, a tuple, a reference. A type's crossing composes its kind's operations over its parts'; the runtime sees no structure beyond the kind.
- **Typed reads.** A value is read at a type only by the runtime (§3).
- **Guards.** Storage is lent to the extern side only under a guard: shared, exclusive, or taken.
- **References.** A reference value is made from a Rust borrow the call holds, writable only from an exclusive one, and read back at its target. It names its target until the checker's loan it is labelled with ends.
- **Views.** A program's array is lent as a slice, shared or mutable.
- **Function values.** A function value is called returning or awaited. A returning call of one that may suspend is driven to its end where it is made. An awaited call's run is held by its call future, and ends when the future ends or is dropped. What made a function value is not the caller's to know.
- **Impls.** An impl is registered, and its value, a function value, is made where a site chooses it (impls §2). A requirer that keeps the impl with its receiver is given a value of its own.
- **Names.** A name is interned in the interner the contract gives.
- **Time.** `sleep`.

## 3. What a caller owes

A typed read states its obligation over the checker's typing of the read path, never over how the value was made: a value reached through the runtime's operations along a path the checker typed `T` is taken at `T`. The extern side discharges it from the checker. It reinterprets a part the runtime handed it only by a layout-only cast under a layout witness, which exists only where the layout identity is signed (safety §2).

## 4. What the runtime signs

- **S1**, on storage a shared guard lends: it keeps its address, and is written only under an exclusive loan, until the checker's loan ends, after the guard drops as well.
- **S2**, on a value made as a reference: it names its target, or its written-back copy, until the checker's loan ends.
- **S3**, on the runtime: storage lent to a run, and a kept value, are released only after every work that may read them has ended or been dropped.

Nothing else is signed.

## 5. What the contract does not state

No word encoding, frame layout or register. No default or vacant value. No array type. No spawning: a handler that wants threads brings its own pool and joins it before it returns.

## 6. Cost

- A call reaches its handler by static dispatch. A returning call performs no dynamic dispatch and no allocation beyond what its declared types require.
- Where a call through the boundary is in tail position, the boundary executes nothing after it, and no address of the caller's frame is live across it.
- A loan's end action is a constant of the lent type and upkeep only. A loan whose end has no action compiles to no code.
