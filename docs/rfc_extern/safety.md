# safety

## 1. No unsafe for the author

The crossing traits are sealed. The boundary and the derive implement them; an author implements none. The extern side's `unsafe` is the boundary's layout casts, each under a witness, and the structural signed implementations; the entry is safe.

The boundary is one set of modules in the extern crate, and it is the only place `unsafe` is written: every other module of every crate forbids it. Each fact the boundary's `unsafe` relies on is established at exactly one place in it, and everything else consumes it through a type; changing the assumption there changes, or breaks at compile time, every dependent.

Every fact on which `unsafe` code relies is carried by a type, a bound, a sealed trait, a private field or a private constructor, or is asserted by an `unsafe` token where it is asserted. A debug assertion and documentation carry none. A method's precondition is the type of a parameter; no trait declares an `unsafe fn`.

*Why.* When `unsafe` code trusts a safe trait or a public field, safe code violates the trust without writing `unsafe`.

## 2. What the author signs

A fact about a type's own structure that no stable Rust bound states is signed at the defining type, and the derive emits it after reading the definition. There are four kinds:

- **L**: layout identity between two Rust types, a type and its payload, or a type and its canonical form.
- **P**: the type's layout and carriers reach its parameters only by holding them.
- **M**: no interior mutability over a lifetime or type parameter.
- **G**: the generic parameter list stated for the checker's flows.

Thread safety, whether a loan lends a word, and a type's lifetime family are types, not signatures.

## 3. The axiom

The checker is right. A runtime's unchecked typed read trusts the call-site shape built from the checker's decision; it is the axiom's one entry. Debug assertions cross-check it as aids, not as the guarantee.
