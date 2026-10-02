# safety

## 1. No unsafe for the author

The crossing is sealed: one `unsafe` trait per meaning (macros §4), implemented by the boundary and the macros. An author implements none and writes no `unsafe`.

`unsafe` is written only in the boundary, one set of modules in the extern crate; every other module forbids it. Each fact it relies on is established at one place, and is carried by a type, a bound, a sealed trait, a private field or a private constructor, or asserted by an `unsafe` token where it is asserted. Documentation and debug assertions carry none. No trait declares an `unsafe fn`.

## 2. What the author signs

A fact about a type's own structure that no stable Rust bound states is signed at the defining type; the derive emits it after reading the definition.

- **L**: layout identity between two Rust types: a type and its payload, or a type and its canonical form.
- **P**: the type's layout and carriers reach its parameters only by holding them.
- **M**: no interior mutability over a lifetime or type parameter.
- **G**: the generic parameter list stated for the checker's flows.

## 3. The axiom

The checker is right. A runtime's unchecked typed read trusts the site's shape built from the checker's decision; it is the axiom's one entry.
