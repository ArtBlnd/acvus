# RFC-0020: An operator is a shared signature and borrows its operands

Status: Proposed
Date: 2026-09-15
Supersedes: none

## Ruling

Every operator is a shared signature (RFC-0019) with a fixed name:
`a == b` is `core::eq(&a, &b)`, `a < b` is `core::lt(&a, &b)`, `a + b` is
`core::add(&a, &b)`, `!a` is `core::not(&a)`, `-a` is `core::neg(&a)`. The
signatures take their operands by reference and return a fresh value.

An operator borrows its operands: an operand that is a value is lent for
the expression, an operand that is already a reference is passed as it
is, and a literal or any other temporary lives until the expression ends
so that it can be lent. This is the operator's own rule and reaches no
function call: `f(x)` with `f: Fn(&T)` stays a type error (RFC-0018).

A type that has no instance of an operator's signature cannot be used
with that operator; the error is at the expression.

## Rationale

With `String` on the heap and only a primitive copying (RFC-0018), `s ==
"x"` with `s: &String` had no meaning: `*s` reads only a primitive, and
`&String` does not unify with `String`. Comparing is a function of two
references — as `PartialEq::eq(&self, &other)` is — and once it is a
function it is a shared signature, so `Regex` compares as `Regex` by its
own instance. Arithmetic follows the same rule so that there is one rule:
`a + b` leaves `a` usable, and a `String` concatenation makes a new
`String`.

Borrowing the operands is what a reader expects of `a == b`, and making
it the operator's rule rather than a coercion keeps calls exact: the
operator's mode is fixed by the language, never spelled, so nothing is
hidden at the expression.

## Not built

- No operator on a value that is not a place except through the
  temporary rule. The temporary is a storage the expression owns.
- No user-declared operators. The operator names are fixed; a type joins
  one by declaring an instance.
- No special case for primitives at the language level. The interpreter
  may run `Int + Int` as a word operation; the IR's `BinOp` on primitive
  words remains its fast path, and a `&P` operand is read with `Load`.

## Consequences

- Type checking a binary or unary expression resolves the operator's
  signature and unifies each operand with `&T`, borrowing a value operand
  and passing a reference operand through.
- Lowering emits `Ref` of each operand (a temporary is first stored in a
  register the expression owns) and a call to the signature's function;
  for primitive operands it emits the existing `BinOp` / `UnaryOp` on
  words, reading a `&P` operand with `Load`.
- The first operator wired to a signature is `==` (and `!=` as its
  negation) through `core::eq` (RFC-0019). The other operators keep
  their primitive-only typing until their signatures are introduced;
  each introduction is one signature declared in the standard registry.

## Open questions

- Whether arithmetic follows the same rule — `add(&T, &T) -> T`, a
  `String` concatenation making a new `String` — or consumes its
  operands as `Fn(T, T) -> T`. The ruling above takes the first; it is
  the one point of this RFC the owner has not yet confirmed.
