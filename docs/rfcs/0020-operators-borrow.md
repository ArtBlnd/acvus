# RFC-0020: Operators on primitives are the type's operation; on everything else, a shared signature

Status: Accepted
Date: 2026-09-15
Supersedes: none

## Ruling

The compiler holds one operator table. Each operator has two entries:

- **Primitive operands** (`Int`, `Float`, `Bool`, `Byte`, `Unit`): the
  operator is an IR `BinOp`/`UnaryOp` on words, defined by this RFC and
  implemented once per host. No shared signature, no instance, no call.
- **Any other operand type**: the operator is a call of a shared signature
  (RFC-0019) with a fixed name, and the operands are borrowed for the
  expression. Today only `==` and `!=` have a signature, `core::eq<T>(&T,
  &T) -> Bool`; `!=` is the negation of `eq`. An operator with no
  signature entry, or a signature with no instance for the operand type,
  is a type error at the expression.

The meaning on primitives:

- `==` / `!=` is identity of the representation: the two words are equal.
  On `Float` this is bit equality: `NaN == NaN`, `0.0 != -0.0`.
- `<`, `<=`, `>`, `>=` is the total order of the representation: signed
  for `Int`, unsigned for `Byte`, `false < true`, and `f64::total_cmp`
  for `Float`, which agrees with bit equality.
- `+`, `-`, `*` on `Int` are checked: overflow is a runtime error, as
  division by zero is. On `Float` they are IEEE arithmetic; `NaN` is a
  value, not an error.
- `/`, `%` on `Int` are a runtime error at zero.

Every other meaning is a named function in a registry, called by name:
`ieee_eq`, `ieee_lt`, `wrapping_add`, `saturating_add`, `concat`. A
primitive has no instance of any shared signature; `eq(&1, &2)` by name
is a type error, `1 == 2` is the operator.

Borrowing the operands: an operand that is a place is lent; an operand
that is already a reference is passed as it is; any other operand is a
temporary the expression owns and lends. This is the operator's rule and
reaches no function call: `f(x)` with `f: Fn(&T)` stays a type error
(RFC-0018).

## Rationale

`==` is the one operation a tagless word has a meaning for without any
definition: the two words are the same. Everything that needs a
definition — an IEEE comparison, a saturating sum — is a definition, and
a definition has a name. Giving the operator the tagless meaning and the
definitions their names keeps each meaning in exactly one place; the
prior state, where the interpreter compared `Float` by IEEE and wrapped
`Int` on overflow while the language said nothing, was two unnamed
definitions hiding behind one symbol.

Defining primitive operators in the compiler rather than as `core::add`
instances that a host inlines keeps one definition per operation. A Rust
body that is the specification plus a fast path that reimplements it is
two definitions of one thing, and the fast path wins silently when they
drift.

For a non-primitive, comparing is a function of two references, as
`PartialEq::eq(&self, &other)` is, and once it is a function it is a
shared signature: `String` compares by its own instance, `Regex` by its
own. Borrowing the operands is what a reader expects of `a == b`, and
making it the operator's rule rather than a coercion keeps calls exact.

## Not built

- No instance of a shared signature for a primitive.
- No signature for ordering or arithmetic yet. `a + b` on `String` is a
  type error; `concat(&a, &b)` is the function. When a composite type
  needs an operator, one signature (`core::cmp`, `core::add`) is
  declared and the primitive entry of the table does not change.
- No user-declared operators. The names are fixed; a type joins one by
  declaring an instance.
- No coercion of a `&P` operand: `*r == 1`, not `r == 1`.

## Consequences

- The type checker, at `==`/`!=`, unifies the operand types; a primitive
  result types as `BinOp`, anything else resolves `core::eq`,
  instantiates it at the operand type, and records the call on the
  expression (`TypeResolution::operator_calls`).
- Lowering emits `BinOp` for a primitive operand pair, and for a recorded
  call a `Ref` of each operand (a temporary first assigned to a register
  the expression owns) and the call; `!=` adds `UnaryOp::Not`.
- `core::eq` and `core::clone` are declared in `acvus_extern::core`, with
  the `String` instances; `Externs::combine` always includes that
  registry.
- The interpreter's `Int` arithmetic is checked (`IntegerOverflow`), its
  `Float` comparisons are bit equality and `total_cmp`, and it has no
  `String` operator arm.
