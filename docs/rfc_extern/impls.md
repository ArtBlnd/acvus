# impls

## 1. Generic functions

A generic function is a name with a signature and no body (compiler: extern fn §6). Its family names it, and states, at each choice of its type variables, its signature and how it takes its receiver. One family serves every impl and every requirer.

## 2. Impls

An impl states its generic function and the choice its signature is at, and Rust holds its signature to the family's at that choice. Its value is a function value: its dispatch, with the values of the impls it requires as its captures, each fixed where its site chose it. It takes no requirement as a parameter. It is reached only through its generic function.

## 3. Requirements

A requirement takes one of two forms, decided by how the generic function takes its receiver:

- By value or by `&mut`: the requirer owns the receiver and the impl value chosen for it as one value, an instance. Only the entry makes one, from the argument and the impl the site chose. It calls that receiver only, and may be kept.
- By `&`: the requirer is lent a handle at the type for the call, and applies it to any value of that type.

The other pairing is a type error. Values of different impls share one Rust type (runtime contract §1), so a receiver is paired with its impl by ownership, and a handle is not kept past the call that lent it.

A requirer calls an instance or a handle as a function value is called (runtime contract §2).
