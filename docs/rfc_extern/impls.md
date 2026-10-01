# impls

## 1. Generic functions

A generic function is a name with a polymorphic type and no body, with one synchronous and one awaited function type.

## 2. Impls

An impl is a declaration that states the generic function it is an impl of. Its family is a nominal marker per generic function with no marker arguments, so one family serves every impl and every requirer; the family names the generic function, how it takes its receiver, and its function type at the requirer's type arguments.

An impl's value is a function value: the impl's dispatch, with the values of the impls it requires itself as its captures, each fixed where the requirer's site chose it. Its call is a function value's call (runtime contract §2, Callbacks). The runtime makes the value from the registered impl; how is its own.

An impl is reached only through its generic function.

*Rejected.* An impl entry of its own kind, read back by a family and a task, beside the function value: a second calling mechanism for what a function value's call already is, and a count of required impls carried at run time where each impl's is fixed.

## 3. Requirements

A requirement is held in one of two forms, decided by how the generic function takes its receiver. Where it takes the receiver by value or by `&mut`, the requirer owns the receiver and the impl value chosen for it as one value: only the crossing makes it, from the argument and the impl the checker chose at its site, it calls that receiver only, and the two are never separated. Where it takes the receiver by `&`, the requirer holds a handle at the type and applies it to any value of that type. The other pairing is a type error.

*Why.* A handle pairs with a value by Rust type only where that Rust type determines one implementation. Storage erases the phantom index (runtime-contract §1), so values of different implementations share one Rust type, and nothing refuses applying one value's impl to another; owning both makes the pairing a fact of construction. A function of a type's values is chosen at the type, and applies to every value of it.

*Rejected.* The receiver and a handle side by side at one Rust type.

A requirer holds a typed handle over a value, and calls it as a function value is called. A returning body calls it returning, and an impl that may suspend is driven to its end there (runtime contract §2, Callbacks); an awaited body calls it awaited, and which body of the impl then runs is the runtime's to choose from the value: its returning body where it has one, which is faster, its awaited body otherwise. The result returns through the runtime's typed read at the requirer's type. No value is taken at a type other than the one it was made at, except by the runtime's own reads, so making a handle is safe.
