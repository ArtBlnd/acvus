# `char`

Rust's `char` methods, the pure part. A `char` is one Unicode scalar value
and an inline word (RFC-0058), so every function takes and returns it by
value. The receiver form and the namespace form are the same call:
`c.is_alphabetic()` is `is_alphabetic(c)`, and an associated function is
written under the type, `char::from_digit(11u32, 16u32)`.

`to_digit` and `from_digit` are Rust's two radix-taking functions. Rust
panics on a radix above 36 and so does this module: a radix is a program's
own constant far more often than a datum, and folding it into the `None`
that already means "not a digit" would hide a program's mistake behind a
value.

| name | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `is_alphabetic` | `(char) -> bool` | `char::is_alphabetic` | none |
| `is_numeric` | `(char) -> bool` | `char::is_numeric` | none |
| `is_alphanumeric` | `(char) -> bool` | `char::is_alphanumeric` | none |
| `is_whitespace` | `(char) -> bool` | `char::is_whitespace` | none |
| `is_uppercase` | `(char) -> bool` | `char::is_uppercase` | none |
| `is_lowercase` | `(char) -> bool` | `char::is_lowercase` | none |
| `is_ascii` | `(char) -> bool` | `char::is_ascii` | none |
| `is_ascii_digit` | `(char) -> bool` | `char::is_ascii_digit` | none |
| `is_ascii_alphabetic` | `(char) -> bool` | `char::is_ascii_alphabetic` | none |
| `to_ascii_uppercase` | `(char) -> char` | `char::to_ascii_uppercase` | none |
| `to_ascii_lowercase` | `(char) -> char` | `char::to_ascii_lowercase` | none |
| `to_digit` | `(char, u32) -> Option<u32>` | `char::to_digit` | a radix above 36 is refused, as in Rust |
| `from_digit` | `(u32, u32) -> Option<char>` | `char::from_digit` | a radix above 36 is refused, as in Rust |
| `len_utf8` | `(char) -> u64` | `char::len_utf8` | `u64` where Rust has `usize` |

## Written elsewhere

| Rust | how a script writes it |
| --- | --- |
| `c.to_string()` | `c.to_string()`, the generic `to_string` over the `core::display` instance for `char` |
| `c as u32` | `c as u32`, a cast and not a call (RFC-0049) |
| `char::from_u32(n)` | `int_to_char(n)`, in `conversion`, which returns `Result<char, CharError>` |

## Not offered, and why

- **`to_uppercase` and `to_lowercase` on a `char`.** Rust returns an
  iterator, because one scalar value can fold into several — `'ß'` uppercases
  to two characters. The ASCII pair is here; the Unicode pair belongs on a
  string, where `string::upper` and `string::lower` already do it.
- **`is_control`, `is_ascii_punctuation` and the rest of the ASCII
  predicates.** Rust's names, no caller yet.
