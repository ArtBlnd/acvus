# RFC-0062: a string slice is a register pair

Status: Accepted — 2026-09-20
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

Decisions 2 and 3 hold for the literal and for everything that reads text.
A string literal is `Ref(Shared, Str)`, lowered to `InstKind::ConstStr`,
which the printer writes `const_str T0` with the text in the literals table.
Measured, `let s = "abc"; len(&s)` — six prepared operations before, three
after:

```
ConstLarge  AssignVar  MakeRef  AsSlice  CallWindow  DropValue
Const       Const      CallWindow
```

The `String` allocation, its slot, the `core::as_str` crossing and the
release are all gone; the two `Const`s are the pair's `ptr` and `len` words.
A literal used directly, `len("abc")`, reaches no slot at all.

The bytes live in the module. `prepare` copies each distinct text of a module
once into a `Literals` table, and every `Body` holds an `Arc` of it, so a run
outlives both the MIR module the text came from and the interner it was read
from: the compiler's `Checked::prepare` drops its modules before the machine
runs, which every script with a literal exercises.

A string pattern is unchanged in form and reads either representation:
`match s { "a" => … }` keeps `InstKind::TestLiteral`, which holds the text as
an immediate and compares the scrutinee's bytes, so the pattern's type check
admits a `String` or a `&str` scrutinee against one `&str` literal pattern.

`StringConcat` and `StringEq` read a part or an operand through either
representation, so `"a" + "b"`, `s + "a"` and `s == "a"` all hold with no
copy, and a template's `Node::Text` lowers to `ConstStr` and joins the other
parts in one concatenation. Where an operand of `+` is still an open type
variable the operand bound stays `String`: a value of type `str` does not
exist, so `|k| -> k + "a"` fixes nothing and is refused.

`core::to_string` takes `T = Str` as an instance, and `"x".to_string()` is
the spelling wherever an owned string is wanted: at a `String` parameter, in
a list, an object, a tuple or a context, at a capture, and as a body's
result. A body does not return a `&str` — the result leaves in the one
register a caller reads, and a host that declares `!` reads it by kind
(RFC-0054), which a pair has none of. The wider rule, decided with this
one and not yet enforced for the other references: a body's result is not
a reference of any kind until a reference's extent can be stated in a
signature (a later RFC on loans as lifetimes); today a `&String` or `&T`
result prints nothing at the host, and the checker will refuse it as it
refuses the `&str`.

The `string` module's reading half takes `&str` and its producers return
`String`: `len`, `is_empty`, `concat`, `contains`, `starts_with_str`,
`ends_with_str`, `find`, `rfind`, `char_at`, `chars`, `bytes`, `lines`,
`split_whitespace`, `trim`, `trim_start`, `trim_end`, `upper`, `lower`,
`capitalize`, `substring`, `split_str`, `split_once`, `strip_prefix`,
`strip_suffix`, `repeat_str`, `replace_str`, `eq_ignore_case`, `pad_start`,
`pad_end`. `to_bytes` keeps `String`, which it consumes. Two units coexist
and each function states its own: `len`, `find`, `rfind` and `substring` are
in bytes, `char_at`, `chars` and the `pad_*` width in Unicode scalar values.
`substring(s: &str, start: u64, end: u64)` refuses an inverted range, an
offset past the length and an offset inside a character, rather than clamping
as it did.

`as_str` keeps `&String`, being the coercion's own declaration, and is
declared in `core` rather than in `string`: the instruction it lowers to is
the language's one crossing (RFC-0039), so a `&str` parameter is reachable
wherever the checker runs. `regex`'s `text` is `&str` in every synchronous
entry; `regex::replace_with` and `replace_all_with` keep `&String` because
`AsyncGlue` admits `Arg<Form = Pair>` at no arity, so a `&str` parameter of
an asynchronous handler does not compile.

`contains` no longer keeps `&String`. Its bare name is shared with
`iter::contains`, whose first parameter is a value, and with the literal a
`&str` the admission order of RFC-0043 reaches the view in every form:
`contains(&s, "x")`, `contains("abc", "b")`, `s.contains("x")` and
`"abc".contains("b")` all settle on `string::contains`.

The cost of the literal's type, counted over the corpus this landed with:
274 sites in scripts, templates and examples. 217 took `.to_string()` at a
`String` parameter or a `String`-typed binding, branch or body result; 38 in
a list, an object, a tuple or a context, which hold no reference; 8 where an
operand of `+` or `==` was still an open type variable; 5 at a lambda
capture; 5 at `clone`, `hash` or `eq`, which have no instance at `&str`; and
1 where `&x` over a `&str` became `x`. Seven calls that read two texts —
`contains`, `find` — needed no rewrite at all: the `string` module's move to
`&str` is what admits them. Four run-time assertions moved number with the
unit change, all over `"héllo"`: `len` 5 to 6, `find(…, "l")` 2 to 3,
`rfind(…, "l")` 3 to 4. One allocation count moved for a different reason —
`split_str("a,b,c", ",")` unboxes 0 arguments where it unboxed 2, a `&str`
parameter being a borrow rather than a value materialized out of its box.

What waits is the `&str` return: an extern declared `-> &str` and a
`substring`, `trim` or `split` that returns a view of its argument need the
pair-wide `CallShape` Decision 4 describes, and until then every producer
returns a `String`.

## Order of work

`Ty::Str` and the coercion in the checker/lowering → the admission order of
RFC-0043 → the literal as a pair constant → the rest of the `string` module
on `&str` → the machine (`SlicePair` for `&str`, `is_char_boundary` at
`substring`) → the extern crossing (`Arg`/`Ret` at `Form = Pair` for `&str`)
→ regex on `&str`.
