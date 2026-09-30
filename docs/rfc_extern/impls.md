# impls

## 1. Generic functions

A generic function is a name with a polymorphic type and no body, with one synchronous and one awaited function type.

## 2. Impls

An impl entry is typed by its generic function's family and its task. The family is a nominal marker per generic function with no marker arguments, so one family serves every impl and every requirer. The entry holds the glue as a function pointer already typed at the family's function type for its task, and the required impls' values. Only the boundary constructs one.

There is one generic impl entry: one generic item, monomorphised per family, task and handler. At registration it gives a builder whose type does not mention the family; preparation calls it, and the runtime makes a value from the typed entry.

An impl is reached only through its generic function.

## 3. Requirements

A requirement is held in one of two forms, decided by how the generic function takes its receiver. Where it takes the receiver by value or by `&mut`, the requirer owns the receiver and the impl value chosen for it as one value: only the crossing makes it, from the argument and the impl the checker chose at its site, it calls that receiver only, and the two are never separated. Where it takes the receiver by `&`, the requirer holds a handle at the type and applies it to any value of that type. The other pairing is a type error.

*Why.* A handle pairs with a value by Rust type only where that Rust type determines one implementation. Storage erases the phantom index (runtime-contract §1), so values of different implementations share one Rust type, and nothing refuses applying one value's impl to another; owning both makes the pairing a fact of construction. A function of a type's values is chosen at the type, and applies to every value of it.

*Rejected.* The receiver and a handle side by side at one Rust type.

A requirer holds a typed handle over a value. At the call, the runtime reads the entry back at the requirer's family and task, a typed read like any other: a synchronous requirement reads at the returning task, an awaited one first reads the value's task. The result returns through the runtime's typed read at the requirer's type. No value is taken at a type other than the one it was made at, except by the runtime's own reads, so making a handle is safe.
