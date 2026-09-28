# runtime contract

The runtime contract is the seam between extern fns and whatever runs them. Every extern fn is callable through the contract alone, whatever implements it: this interpreter today, a JIT or another executor later. It is defined well enough to give that freedom.

## 1. Values

A value's representation is not part of the contract. The contract holds even when the runtime's value is a trivial safe enum of any size, so a safe, slow runtime implementing it runs every extern fn. The extern side reaches values only through the contract's operations, generic over the runtime and dispatched statically, so a runtime pays nothing for the generality.

**Nothing anywhere makes a runtime value from a Rust value.** No function, trait or conversion turns a `T` into a `Value`. A value of the runtime exists only where the crossing made it, at the type the checker settled for that site. The reason is variance: a path from `T` to `Value` makes `fn(Value)` usable as `fn(T)`, and a function parameter is contravariant, so a function written for one type would be called with a value made from another. That is undefined behaviour, and no bound or `unsafe` marker on the path prevents it; the path does not exist.

The runtime's value crosses to the extern side as the extern side's own owning holder. A runtime keeps an extern-defined value so that it can lend one: what `&S` points at is the runtime's, and the contract says only that it is a shared reference to an `S`.

## 2. Operations

- **Typed reads.** Reading a value back at a type is the runtime's operation, and so are its reads at a stored form and its guards. Erasing a Rust value into a runtime value happens only inside the crossing, at the site's settled type (§1).
- **Guards.** A shared guard, an exclusive guard, and a taken value. The extern side is lent storage only under a guard.
- **The entry.** Each call has one entry, which the runtime's boundary calls with the call's values (crossing §1).
- **Views.** A script array's slice view, shared or mutable, is a contract operation.
- **References.** The runtime makes a reference value from a target, a writable one only from an exclusive borrow of it.
- **Aggregates.** A tuple, an object and a variant are the runtime's associated types. Their bodies are safe traits, bounded `Send + Sync + 'static`, whose parts are owning holders.
- **Instances.** The runtime makes a value from an instance entry, reads an entry back at a signature family and task the caller names, and reads an instance value's task (instances).
- **Callbacks.** Calling a script closure synchronously and awaited.
- **Rust functions.** An extern may return a Rust function as a script function value of its declared function type. The script calls it as it calls a lambda, and each call crosses its arguments at that call's settled types, so no value is made from a Rust value (§1).
- **Names.** The extern side interns a name in the interner the contract returns; a runtime maps no name itself.
- **Time.** `sleep`.

## 3. What a caller owes

Every typed read states its caller's obligation over the checker's typing of the read path, never over how the value was made: this value, reached through this runtime's own operations along a path the checker typed `T`, is taken at `T`. The extern side discharges it from the checker. A runtime that checks its reads turns a breach into a panic; a runtime that trusts them trusts its own reads.

The extern side reinterprets a part the runtime handed it only by a layout-only cast under a layout witness, and every later typed read of that part goes back to the runtime.

The entry is safe. The axiom enters only where a runtime trusts its own typed reads: such a runtime owes that each value is live at the type the checker settled for the site (safety §3). A runtime that checks its reads owes nothing, so a safe runtime needs no `unsafe` of its own.

## 4. What the runtime signs

A runtime signs, by `unsafe` trait bounds, the three facts no type can carry:

- **S1**, on the storage a shared guard lends: it keeps its address, and is written only under an exclusive loan, until the checker's loan ends, after the guard drops as well. The checker's loan is not a Rust lifetime: a stored source owns all it holds and spans calls.
- **S2**, on the runtime's value: a value made as a reference to a target names that target, or its written-back copy, until the checker's loan ends. The value is copyable data with no lifetime.
- **S3**, on the runtime: storage lent in place to another run, and a kept value, are released only after the last work that may read them has ended or been dropped. Only a count per loan could carry it as a type, and that is a release cost.

Nothing else is signed. Every other fact is a type, a bound, the Rust validity of what the runtime returns, or the runtime's own business.

## 5. What the contract does not state

No word encoding, frame layout or register. No default or vacant value, and no destination filled ahead of its writer. No array type. No spawning of any kind: a handler that wants threads brings its own pool and joins it before it returns.
