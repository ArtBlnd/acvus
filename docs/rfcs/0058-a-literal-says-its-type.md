# RFC-0058: A literal says its type — `10u64`, `'c'`, `b"…"`, and `char`

Status: Accepted — 2026-09-20
Extends: RFC-0037 (integers have a width), RFC-0049 (`expr as T`),
RFC-0055 (a constant expression folds), RFC-0012 (the array literal)

## Problem

RFC-0037 gave an integer literal no width of its own — "the use decides" —
and listed "No integer suffix on literals" under Not built. RFC-0049 left
the suffix out again: "`3 as u64` is how a `u64` constant is written". Both
were right about the common case, `max_tokens: 1024`, and both cost the
uncommon one.

**A constant the use cannot decide.** `a[i]` takes a `u64` (RFC-0047), and
nothing in `a[1]` narrows the literal to one: the `logs` bench wrote
`plen / plen` to obtain the `u64` one, and after RFC-0049 `1 as u64`, which
is a cast written where a constant was meant.

**No character.** A `String` is UTF-8 and the language had nothing a scalar
value could be. `string::char_at` returned a one-character `String` — a heap
allocation per character — `string::chars` yielded those, and
`core::char_to_int` / `core::int_to_char` converted between a
one-character `String` and an `i64`, each returning a `Result` for the case
where the `String` was not one character. A character had no type, so it
had a representation that could be wrong.

**No bytes.** A byte string had to be written as a list of numbers: the
`logs` bench spells `GET` as `Vec<i64>` of byte values, and the reader has
to decode it.

## Decision

1. **A suffixed integer literal has that width.** `10u64`, `-1i8`, `255u8`,
   where the suffix is one of RFC-0037's eight names. An unsuffixed literal
   keeps RFC-0037's rule exactly: a variable only an integer can fill, the
   use decides, `i64` where nothing does. A suffixed literal has nothing for
   a use to decide, so its value is checked against its width at the
   literal, with RFC-0037's own message — `literal 300 does not fit u8` —
   and not wrapped. There are no underscore separators: the lexer had none
   and this RFC adds none.

2. **`char` is a type: one Unicode scalar value.** Rust's `char`, and it
   crosses the boundary as one. A literal is `'c'` with Rust's escapes —
   `'\n'`, `'\r'`, `'\t'`, `'\0'`, `'\''`, `'\\'`, `'\xNN'` up to `\x7F`,
   `'\u{1F600}'` — and a `'…'` holding no scalar value or more than one is a
   compile error. A `char` compares with `==` and `<`, by its scalar value,
   which is Rust's order. It casts by RFC-0049's rules and Rust's
   admissions: `char as T` for every integer width, `u8 as char` and no
   other cast into a `char`, and no `char as f64`.

3. **`b"…"` is a byte-string literal of type `Array<u8, N>`** — RFC-0012's
   array literal with its identity, not a new container. Its text is ASCII
   and Rust's escapes, `\xNN` reaching `0xFF`; a non-ASCII character inside
   is a compile error naming it, because `b"é"` would otherwise silently be
   two bytes. `b'x'` is a `u8` literal.

4. **The stdlib's character boundary is typed with `char`.**
   `string::char_at` returns a `char` and `string::chars` yields one.
   `core::char_to_int` is deleted: it was `c as u32` with a `Result` for the
   case the type now excludes. `core::int_to_char` stays, retyped to
   `Result<char, CharError>` and with its `NotOneChar` variant gone: it is
   the one `char` conversion `as` does not reach, because Rust admits only
   `u8 as char` and that stops at U+00FF. `core::to_string` gains a `char`
   instance.

5. **Nothing new runs.** The fold (RFC-0055) reads a suffixed constant at
   its width and wraps there; a cast of a `char` constant folds like any
   other. The cast (RFC-0049) reads both as leaves. No MIR instruction and
   no interpreter operation is added by this RFC.

## What it costs

**One `Kind`, and it is a `char`.** `Kind::Char` joins
`acvus_extern::for_each_inline!`, whose list is the one place the inline
types are named: `Inline for char`, `Kind::Char`, and the Rust `char`
crossing come from that single line. Reusing `Kind::U32` was the
alternative and it cannot be taken: `char_at` returns a Rust `char`, which
needs `Cross<Rt>` for `char`, which needs a `Kind` of its own for the
value's record of the type it was erased from. The word is the scalar
value's `u32` bits, its `Release` is nothing, and `Value` is still sixteen
bytes with a spare `Kind` for `Option<Value>`'s niche.

**No cast instances.** A `char` has no word of its own —
`CastTy::word` maps it to `u32` — and every cast Rust admits at either end
of a `char` has the value of the same cast through `u32`: `u8 as char`
zero-extends, `char as T` truncates or extends from 32 bits. So RFC-0049's
`Cast<From, To, Src, Dst>` family stays at 324 instances, and `==` and `<`
on `char` are the `u32` operations `arith::int_binop` already had. The
chain recognizer does not know the type, so a `char` never enters a chain
and `Chain{1,2,3}<T>` is untouched.

**`NumTy` is `CastTy`, and `WordTy` is new.** The enum RFC-0049 called
`NumTy` is what an `as` names at either end, and a `char` is not a number,
so it is `CastTy` now with a `Char` variant; `WordTy` is the two the machine
reads, an integer width and `f64`. The rename is 43 sites and no behaviour.
Making `CastTy::Char` unreachable in the machine by a comment was the
alternative; a second type that cannot hold it is what the compiler checks.

**Two deleted conversions and their tests.** `char_to_int` is gone;
`int_to_char`'s `NotOneChar` is gone with the `String` it reported on. Every
script that read a character as a `String` is written with `char`, and
`{{ c }}` in a template is `{{ c.to_string() }}` — emit takes a `String`
and this RFC does not widen it.

**One escape table, and it is not the string literal's.** `"…"` decodes
`\n`, `\t`, `\\` and `\"` and passes any other escape through as two
characters. `'c'`, `b'x'` and `b"…"` use Rust's table, where an unknown
escape is an error. The two differ, and this RFC does not change `"…"`:
a script that relies on `"\d"` being two characters keeps working, and no
new literal inherits a rule Rust does not have.

## Rejected

**`char` as a one-character `String`.** It is what the language had. A
character was a heap allocation, every read of one could be zero or two
characters instead, and each conversion carried a `Result` for a case the
type should have excluded. A scalar value is a word; making it one is the
whole point.

**`b"…"` as a `Vec<u8>`.** A literal's length is known at the literal, and
RFC-0012's array literal already carries a known length with an identity. A
`Vec` would allocate at every evaluation and lose the length from the type,
for no expression the array cannot write.

**A wrap on an oversize literal.** `300u8` as `44` would make the suffix a
truncation operator. RFC-0037 already refuses `@b + 300` at a `u8`; the
suffix is the same rule with the width named rather than inferred, so it
gets the same refusal and the same message.

**`u32 as char`.** It is not a Rust `as`, because most `u32`s are not scalar
values. Admitting it would mean a check, a `Result`, or a value that is not
a `char`; the check is `int_to_char` and it already exists.

**A suffix on a float literal.** RFC-0037 has no `f32` and RFC-0049 added
none, so `f64` is the only float and a suffix would name the only choice.

## Consequences

- `acvus-ast`: `literal.rs` — `IntWidth`, `SuffixedInt`, `LiteralErrorKind`
  and the three `decode_*` functions; `Token::{IntLitOf, CharLit, ByteLit,
  ByteStrLit}`; `Literal::{IntOf, Char, Bytes}` and `Literal::desugared`;
  `ParseErrorKind::BadLiteral`; `lexer::char_literal_end`, so a character
  literal holding a `"` or a `}` does not end a `{{ }}` tag.
- `acvus-mir`: `TyTerm::Char`, `CastTy` (was `NumTy`) with its `Char`
  variant, `WordTy`, `CastTy::{word, admits, NAMES}`, `From<IntWidth> for
  IntTy`, `Checker::suffixed_int_literal`, `CastSite`'s `to` and
  `target_span`, `MirErrorKind::{CastOfWhatDoesNotCast, CastNotAdmitted}`
  in place of `CastOfNonNumber`, and `fold::cast_result`'s `Char` ends.
- `acvus-interpreter`: `Kind::Char`, `Value::{char_, as_char}`, the `Char`
  arms of `layout::{encode, decode}`, `prepare`'s constant, pattern and
  word-kind arms, and `ops::cast`'s `Conversion` at `WordTy`.
- `acvus-extern`: `for_each_inline!` gains `Char: char`;
  `cross_as_stored!(char)`; `TyArg for char`.
- `acvus-ext`: `string::{char_at, chars}` at `char`; `conversion` loses
  `char_to_int` and `CharError::NotOneChar` and gains `to_string_char`.
- `acvus`: JSON reads a `char` out as the one-character string it is.
- The lowerer is the only producer of a MIR literal and it desugars one:
  `Literal::IntOf` reaches the IR as its value and `Literal::Bytes` as the
  list of its bytes, so the width and the `u8` live in the value's type,
  where RFC-0037 puts them, and no MIR reader has two spellings of one
  constant to agree about. `Literal::Char` has no older form and stays.
- Tests: `acvus-ast`'s `literal::every_escape_is_rusts` and the parser's
  literal tests, `acvus-mir-test/tests/literal.rs` for what is refused and
  what the listing shows, `acvus-interpreter-test/tests/literal.rs` for
  the values, and the `char` cases added to
  `acvus-mir-test/tests/cast.rs`.
