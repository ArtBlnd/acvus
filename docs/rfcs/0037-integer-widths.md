# RFC-0037: Integers have a width, and a literal takes the width its use demands

Status: Accepted
Date: 2026-09-16
Extends: RFC-0022, RFC-0032

## Ruling

The language has eight integer types, `i8`, `i16`, `i32`, `i64`, `u8`,
`u16`, `u32`, `u64`, and they are Rust's. `Int` and `Byte` are gone:
`Int` was `i64` and `Byte` was `u8`, and those are their names now. A
Rust struct field of any of these types crosses as that type (RFC-0032).

Arithmetic, comparison, and the bit operators take two operands of one
width and produce that width; nothing widens or narrows on its own. Each
operation is what the same Rust operator does in a release build, at that
width: `+`, `-`, `*` and negation wrap, a shift takes its amount modulo
the width, and `/` and `%` panic on a zero divisor and at `MIN / -1`,
with Rust's own texts — `attempt to divide by zero`, `attempt to
calculate the remainder with a divisor of zero`, `attempt to divide with
overflow`, `attempt to calculate the remainder with overflow`. Those two
are the only integer operations that can fail, so RFC-0007's constraint
— an operation that can raise runs only where the program wrote it, and
no pass may move it onto a path the program did not take — binds `/` and
`%` and nothing else. Negation takes a signed integer or a `Float`.

An integer literal has no width of its own. Its type is a variable that
only an integer type can fill; the use decides which — `@b + 1` with `@b`
a `u8` makes `1` a `u8` — and where nothing decides, the literal is an
`i64`. A negated literal's variable takes signed types only. Once the
width is known, the literal's value is checked against it: `@b + 300`
with `@b` a `u8` is a compile error at the literal, `literal 300 does
not fit u8`.

At run time an integer of any width is the word it always was: its
two's-complement bits in the low bytes, read at the width the type
names. A space (RFC-0033) lays a `u16` out in two bytes, an `i64` in
eight. JSON reads a number as an `i64`, or as a `u64` when it is too
large for one.

## Rationale

Every wire schema counts something in `u32` or `u64` — tokens, retries,
offsets — and a struct that carried them could not be one struct for the
wire and the language while the language had one integer. Widening them
to `i64` in the struct would have said the wire allows negative counts;
it does not. Giving the language Rust's widths lets the struct say what
the wire says, and the derive projects it without a conversion.

A literal without a width of its own is what makes the widths usable
from a script: `max_tokens: 1024` reads as a number, and the field's
type decides what number. The alternative, a suffix on every literal
that is not an `i64`, puts the type in the script where the struct
already states it. Checking the value after the width is known, rather
than at the literal, is what lets `300` be fine as an `i64` and an error
as a `u8` with one rule.

## Not built

- No conversion between widths in the language; `to_int` still reads
  every width into an `i64`, and `to_string` prints every width. Named
  conversions (`u32::from_str`, narrowing with a range check) come with
  the standard library work.
- No `f32`.
- No integer suffix on literals.

## Consequences

- `acvus-mir`: `IntTy`, `TyTerm::Int(IntTy)`, `TyVarBound::Integer`,
  `Solver::fresh_int_var` and `require_signed`, the literal range check
  in `verify_bounds`, `MirErrorKind::IntegerLiteralOutOfRange`; the IR
  validator accepts an integer constant at any width.
- `acvus-ast`: `Literal::Int(i128)`, `Token::IntLit(i128)`;
  `Literal::Byte` is gone.
- `acvus-interpreter`: `int_binop` at the operand width, negation
  checked, `TestLiteral` reads at the source's width, `layout` writes
  `IntTy::bytes()` bytes.
- `acvus-extern`: `TyArg` and `Cross` for the eight widths.
- `acvus-ext`: `to_string` instances for the eight widths.
- `acvus`: JSON in and out by width.
- Type names in diagnostics, snapshots, and the LSP read `i64` where they
  read `Int`.
