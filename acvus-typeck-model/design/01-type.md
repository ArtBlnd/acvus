# 01 Types

Rests on the spec's l.39–70 (Types) and l.72–74 (Variance), with l.165 (parts of
one name), l.191–193 (function values), l.318 and l.327 (a variant written without
a payload) and l.352–354 (identity). The code is `src/ty.rs`; its types, signatures
and docs carry the interface and the contract, and `src/ty/tests.rs` exercises the
contract. This note keeps what the code cannot carry.

## Meaning

A type is a finite term: one constructor, its members and its parts, each part
again a type (l.41). `Ty` is a closed type, one that holds no equivalence class.
`Term<P>` is the shape of a term whose parts are `P`s, which closed types and kinds
(note 06) share.

## Not owned here

- Agreement with the declarations, such as a struct's number of fields, or an
  extern-defined type's number of arguments and whether it carries an identity,
  is note 02's.
- Join is note 04's, and coercion and rank are note 05's.
- What holds of a kind, a `Term` whose parts are equivalence classes, is note 06's.
  `Term<P>` checks nothing about its `P`s.
- Facts are carried beside a type, from note 06 on. A fact is neither a part nor a
  member (l.180), and two types are equal by their constructors, members and parts
  alone (l.66).
- The specialization mark has no representation, since a type carries none (l.70).

## Readings

The spec leaves three questions open at this level. These readings were agreed on.

1. **The type of a function value.** The type table gives `Fn(P̄ → R)` the
   parameters and the result as parts (l.51), while an equivalence class that holds
   a function value does not carry them as parts and names its lambdas by Values
   (l.193). Such a class still takes a constructor (l.275). So a function value's
   type is the constructor `Function(n)` with no parts, `Term::FunctionValue`, and
   `Term::Function` with its parts is the type a declaration writes. Function values
   come only from lambdas (l.191), so the latter appears only in signatures.
2. **An identity as a part.** An identity is a part of an extern-defined type
   (l.55, l.352), and a part is again a type (l.41). So an identity is a type whose
   constructor is its name and which has no parts, and it appears as a part only at
   an extern-defined type's identity. Two identities are then equal exactly when
   their names are, and two different identities are never made one (l.354).
3. **The payload of `None`.** `Option` has one payload (l.49). `None` written
   without a payload is `None(_)` in a pattern (l.318), and the payload that `None`
   makes anew lies in `Option`'s part (l.327, Example 1). So `None`'s payload is that
   one part, `Payload(Some)`.

## Left open for change

- `Term<P>` is shared by closed types and kinds. A constructor added later is added
  in one place, and the compiler then lists every operation that must handle it.
  `parts` and `map` are the only two functions that take a term apart part by
  part, `part` is made from `parts`, and a test checks that `parts` and `map` name
  each part alike.
- The whole constructor table is present from the start, enums, results and
  identities included, although the first examples need none of them, so encoding
  more of the spec later does not reshape types.
- `Name` is one string. Interning, or provenance, changes only what is inside it.
