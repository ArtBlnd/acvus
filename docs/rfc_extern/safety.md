# safety

## 1. No unsafe for the author

The crossing traits are sealed. The boundary and the derive implement them; an author implements none. The extern side's `unsafe` is the entry, the boundary's layout casts, each under a witness, and the structural signed implementations.

## 2. What the author signs

A fact about a type's own structure that no stable Rust bound states is signed at the defining type, and the derive emits it after reading the definition. There are four kinds:

- **L**: layout identity between two Rust types, a type and its payload, or a type and its canonical form.
- **P**: the type's layout and carriers reach its parameters only by holding them.
- **M**: no interior mutability over a lifetime or type parameter.
- **G**: the generic parameter list stated for the checker's flows.

Thread safety, whether a loan lends a word, and a type's lifetime family are types, not signatures.

## 3. The axiom

The checker is right. The one unsafe entry trusts the call-site shape built from the checker's decision. Debug assertions cross-check it as aids, not as the guarantee.
