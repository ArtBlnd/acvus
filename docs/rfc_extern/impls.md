# impls

## 1. Generic functions

A generic function is a name with a polymorphic type and no body, with one synchronous and one awaited function type.

## 2. Impls

An impl entry is typed by its generic function's family and its task. The family is a nominal marker per generic function with no marker arguments, so one family serves every impl and every requirer. The entry holds the glue as a function pointer already typed at the family's function type for its task, and the required impls' values. Only the boundary constructs one.

There is one generic impl entry: one generic item, monomorphised per family, task and handler. At registration it gives a builder whose type does not mention the family; preparation calls it, and the runtime makes a value from the typed entry.

## 3. Requirements

A requirer holds a typed handle over a value. At the call, the runtime reads the entry back at the requirer's family and task, a typed read like any other: a synchronous requirement reads at the returning task, an awaited one first reads the value's task. The result returns through the runtime's typed read at the requirer's type. No value is taken at a type other than the one it was made at, except by the runtime's own reads, so making a handle is safe.
