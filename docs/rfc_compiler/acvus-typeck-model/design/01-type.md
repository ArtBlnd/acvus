# 01 Types

The code is `src/ty.rs`; its types, signatures and docs carry the interface and the
contract, and `src/ty/tests.rs` exercises the contract. This note keeps what the code
cannot carry.

## Meaning

A type is a finite term: one constructor, its members and its parts, each part
again a type. `Ty` is a closed type, one that holds no equivalence class.
`Term<P>` is the shape of a term whose parts are `P`s, which closed types and structures
share.

## Not owned here

- `Term<P>` checks nothing about its `P`s.
- Facts are carried beside a type. A fact is neither a part nor a member, and two
  types are equal by their constructors, members and parts alone.
- The specialization mark has no representation, since a type carries none.

## Readings

1. **The type of a function value.** A function value's type is the constructor
   `Function(n)` with no parts, `Term::FunctionValue`, and `Term::Function` with its
   parts is the type a declaration writes. Function values come only from lambdas, so
   the latter appears only in signatures.
2. **An identity as a part.** An identity is a part of an extern-defined type,
   and a part is again a type. So an identity is a type whose
   constructor is its name and which has no parts, and it appears as a part only at
   an extern-defined type's identity. Two identities are then equal exactly when
   their names are, and two different identities are never made one.
3. **The payload of `None`.** `Option` has one payload. `None` written
   without a payload is `None(_)` in a pattern, and the payload that `None`
   makes anew lies in `Option`'s part. So `None`'s payload is that
   one part, `Payload(Some)`.

## Left open for change

- `Term<P>` is shared by closed types and structures. A constructor added later is added
  in one place, and the compiler then lists every operation that must handle it.
  `parts` and `map` are the only two functions that take a term apart part by
  part, `part` is made from `parts`, and a test checks that `parts` and `map` name
  each part alike.
- The whole constructor table is present from the start, enums, results and
  identities included.
- `Name` is one string. Interning, or provenance, changes only what is inside it.
- `HasTerm` is what a closed type shares with a value a use reads, whose
  positions may be undecided, and `check_term` is the one check of a term's own
  parts, which `Ty::new` makes.
