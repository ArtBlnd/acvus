# RFC-0062: a string slice is a register pair

Status: Draft — 2026-09-20
Extends: RFC-0047 (a slice is the one thing the machine indexes; two
registers; the `&v` coercion and the `Slice<T, Rt>` parameter), RFC-0018
(references and loans), RFC-0012 (every user-defined value moves; a copy is
written), RFC-0058 (`char`), RFC-0059 (a crossing says its width)

## Problem

A `String` is the only string. Every operation that yields part of one
allocates a new one: `substring`, `trim`, `split_str`, `find_all`, a regex
match's text. A line of a log parsed into five fields is five heap
allocations and five releases.

A string literal is a `String` constant: the listing shows `assign v13 =
T0` and `ref &v13` where the literal is used, one `String` per use of a
constant that never changes.

An extern that reads a string takes `&String` and unwraps it to Rust's
`&str` itself; an extern cannot take a part of a string without a copy,
and cannot return one.

RFC-0047 solved the same three problems for `Vec<T>`: `&[T]` is two
registers, `&v` coerces to it at a parameter, an extern takes it as a
`Slice<T, Rt>` view and returns one as a projection of its parameter. The
string has no such type.

## Decision

1. **`&str` is a type**: `Ref(Shared, Str)`, a shared borrow of a `String`'s
   bytes. There is no `&mut str` (a write through it could break UTF-8)
   and no owned `str` value. In the machine it is a `SlicePair` — `ptr`,
   `len` in two adjacent registers (RFC-0047 rule 6 as amended), the
   length in bytes.

2. **What produces one.**
   - A string literal is a `&'static str`: two constant words, no
     `String`, no slot. A literal that must be a `String` is written
     `"…".to_string()`.
   - `substring(&s, start: u64, end: u64) -> &str` — byte offsets, the
     one run-time check being `is_char_boundary` at both ends (RFC-0037's
     raise); `trim`, `trim_start`, `trim_end` return a view of their
     argument; `split(&s, sep) -> Iter<&str>` and a regex `find` yield
     views into the searched text.
   - An extern declared `-> &str` returns a projection of one `&str`
     or `&String` parameter (RFC-0047 §3's rule for a slice return).

3. **What consumes one.** A `&String` argument at a `&str` parameter
   coerces (`CastKind::Str`, the same site as `CastKind::Slice`: the
   argument becomes the whole-string view). A `&str` at a `&String`
   parameter does not — the callee may need the owned form and the
   copy is written: `to_string(&str) -> String`. `len(&str) -> u64` is
   bytes; `chars(&str)`, `char_at`, `==`, `<`, `starts_with`, `contains`
   take `&str` and therefore also `&String` by the coercion. The
   `string` module's functions move to `&str` parameters where they read;
   `String` where they consume.

4. **Crossing.** A `&str` parameter is `WIDTH = 2` and the handler
   receives Rust's `&str` for the call — no view type, the pair *is*
   `(ptr, len)` of UTF-8 bytes the checker guaranteed (RFC-0059 `Arg`
   with `Form = Pair`). A `&str` return is `Elements`-shaped: two words
   in `rax`/`rdx` stored to the pair.

5. **A view is frame-bound.** As every reference (RFC-0018), a `&str` is
   not stored in an object, a container or a context; it does not
   outlive the `String` it borrows, and a shape write to that `String`
   under a live view is refused by the loans. A `Match` of a regex holds
   `start`/`end` (byte offsets), and its text is `substring(&text, m.start,
   m.end)`.

## What it costs

- `Ty::Str` (the pointee of the reference) in every pass and reader of
  `Ty`; `Kind` unchanged (a view is two words, no `Value` of its own).
- The `string` module's parameters change type; every script that
  passes `&s` is unchanged by the coercion; a script that passed `s` by
  value to a reading function is refused and rewritten with `&s`.
- Byte offsets and `char` indices coexist: `substring`/`len`/`find` in
  bytes, `char_at`/`chars` in scalars — stated per function.

## Rejected

- **`&[u8]` for a string view**: loses the UTF-8 invariant the type
  carries; every consumer would re-validate.
- **`&mut str`**: no operation needs it that `String` does not give.
- **An owned `str` / a small-string value**: a value type with a
  representation of its own; the view plus `to_string()` covers the
  uses.
- **A `&str` stored in an object**: a borrow escaping its frame,
  refused by RFC-0018.

## Consequences

Expected, to be replaced by measurement: the `logs` bench's line
parsing without an allocation per field; string literals without a slot;
`regex` and `string` externs taking `&str`.

## Order of work

`Ty::Str` and the coercion in the checker/lowering → the literal as a
pair constant → the `string` module on `&str` → the machine (`SlicePair`
for `&str`, `is_char_boundary` at `substring`) → the extern crossing
(`Arg`/`Ret` at `Form = Pair` for `&str`) → regex on `&str`.
