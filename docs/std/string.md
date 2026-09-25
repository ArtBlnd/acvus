# `string`

Rust's `str` surface, the pure part, under Rust's names. A text-reading
parameter is `&str`, which is what a literal already is, so a call needs no
copy: `regex("[0-9]+")`, not `regex("[0-9]+".to_string())`.

Two units run through the module and every row says which it uses. Bytes are
what `len`, `find`, `rfind`, `substring`, `is_char_boundary`, `char_indices`
and `match_indices` speak in; Unicode scalar values are what `char_at`,
`chars` and the `pad_*` width count.

A function that walks the text answers a pipeline, and a pipeline's type is
the stage itself. Every one here builds `Items<T>`, the owned source, so the
whole text is walked before the value is returned and the pipeline borrows
nothing. Adaptors and consumers are reached from it as from any other
source: `s.chars() | count`.

Where Rust's method panics on a byte offset that is not a character
boundary, this module refuses the run rather than returning an `Option`. An
offset comes from `find`, `rfind` or a regex match, all of which report
boundaries, so a non-boundary offset is a program that computed one, not a
value a caller can branch on.

The `<` operator does not reach text. `cmp`, `lt`, `le`, `gt` and `ge` are
the ordering, bytewise as Rust's `Ord for str`; an operator instance for
`String` waits for RFC-0067's `ord<T>`. A comparison operator written on
text is refused by name: ``` `<` is not defined on String; use `string::cmp`,
`lt`, `le`, `gt` or `ge` ```.

| name | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `len` | `(&str) -> u64` | `str::len` | none |
| `is_empty` | `(&str) -> bool` | `str::is_empty` | none |
| `concat` | `(&str, &str) -> String` | `String::push_str` / `+` | Rust's `[&str]::concat` joins a slice; this one joins two |
| `trim` | `(&str) -> &str` | `str::trim` | none |
| `trim_start` | `(&str) -> &str` | `str::trim_start` | none |
| `trim_end` | `(&str) -> &str` | `str::trim_end` | none |
| `trim_matches` | `(&str, &str) -> &str` | `str::trim_matches` | Rust's pattern must be double-ended, so `&str` is not admitted there; this one strips the whole `&str` from both ends. An empty pattern strips nothing |
| `trim_start_matches` | `(&str, &str) -> &str` | `str::trim_start_matches` | an empty pattern strips nothing |
| `trim_end_matches` | `(&str, &str) -> &str` | `str::trim_end_matches` | an empty pattern strips nothing |
| `upper` | `(&str) -> String` | `str::to_uppercase` | name |
| `lower` | `(&str) -> String` | `str::to_lowercase` | name |
| `to_ascii_uppercase` | `(&str) -> String` | `str::to_ascii_uppercase` | none |
| `to_ascii_lowercase` | `(&str) -> String` | `str::to_ascii_lowercase` | none |
| `is_ascii` | `(&str) -> bool` | `str::is_ascii` | none |
| `contains` | `(&str, &str) -> bool` | `str::contains` | none |
| `starts_with` | `(&str, &str) -> bool` | `str::starts_with` | none |
| `ends_with` | `(&str, &str) -> bool` | `str::ends_with` | none |
| `replace` | `(&str, &str, &str) -> String` | `str::replace` | none |
| `replacen` | `(&str, &str, &str, u64) -> String` | `str::replacen` | count is `u64`; one above the address space is refused |
| `repeat` | `(&str, u64) -> String` | `str::repeat` | count is `u64`; one above the address space is refused |
| `is_char_boundary` | `(&str, u64) -> bool` | `str::is_char_boundary` | none |
| `substring` | `(&str, u64, u64) -> &str` | `str::get(a..b)` | Rust returns `Option`; this refuses an inverted range, an offset past the end, or one inside a character |
| `to_bytes` | `(String) -> Vec<u8>` | `String::into_bytes` | Rust's `str::as_bytes` borrows; the boundary returns no borrowed byte slice, so this one consumes the `String` |
| `to_utf8` | `(Vec<u8>) -> Option<String>` | `String::from_utf8` | `Option` where Rust has `Result`; the error carries only the bytes back |
| `to_utf8_lossy` | `(Vec<u8>) -> String` | `String::from_utf8_lossy` | always owned, never borrowed |
| `cmp` | `(&str, &str) -> i64` | `Ord::cmp for str` | `-1`, `0`, `1` where Rust has `Ordering` |
| `lt` | `(&str, &str) -> bool` | `PartialOrd::lt for str` | a function, because `<` has no instance for text |
| `le` | `(&str, &str) -> bool` | `PartialOrd::le for str` | as `lt` |
| `gt` | `(&str, &str) -> bool` | `PartialOrd::gt for str` | as `lt` |
| `ge` | `(&str, &str) -> bool` | `PartialOrd::ge for str` | as `lt` |
| `char_at` | `(&str, i64) -> char` | `str::chars().nth(i)` | counts scalar values; an index past the end is refused, where Rust gives `None` |
| `chars` | `(&str) -> Items<char>` | `str::chars` | the whole text is walked before the iterator is returned |
| `char_indices` | `(&str) -> Items<CharIndex>` | `str::char_indices` | yields `{ index: u64, ch: char }`, because an extern function returns no tuple |
| `lines` | `(&str) -> Items<String>` | `str::lines` | elements are owned |
| `bytes` | `(&str) -> Items<i64>` | `str::bytes` | element is `i64`, not `u8` |
| `split_whitespace` | `(&str) -> Items<String>` | `str::split_whitespace` | elements are owned |
| `split` | `(&str, &str) -> Items<String>` | `str::split` | elements are owned |
| `rsplit` | `(&str, &str) -> Items<String>` | `str::rsplit` | elements are owned |
| `splitn` | `(&str, u64, &str) -> Items<String>` | `str::splitn` | count is `u64`; one above the address space is refused |
| `rsplitn` | `(&str, u64, &str) -> Items<String>` | `str::rsplitn` | as `splitn` |
| `split_terminator` | `(&str, &str) -> Items<String>` | `str::split_terminator` | elements are owned |
| `matches` | `(&str, &str) -> Items<String>` | `str::matches` | elements are owned |
| `match_indices` | `(&str, &str) -> Items<MatchIndex>` | `str::match_indices` | yields `{ index: u64, text: String }`, because an extern function returns no tuple |
| `find` | `(&str, &str) -> Option<i64>` | `str::find` | byte offset as `i64` |
| `rfind` | `(&str, &str) -> Option<i64>` | `str::rfind` | byte offset as `i64` |
| `pad_start` | `(&str, i64, &str) -> String` | none | JS `padStart`; width counts scalar values |
| `pad_end` | `(&str, i64, &str) -> String` | none | JS `padEnd`; width counts scalar values |
| `strip_prefix` | `(&str, &str) -> Option<String>` | `str::strip_prefix` | owned, where Rust borrows |
| `strip_suffix` | `(&str, &str) -> Option<String>` | `str::strip_suffix` | owned, where Rust borrows |
| `split_once` | `(&str, &str) -> Option<Vec<String>>` | `str::split_once` | a two-element `Vec`, because an extern function returns no tuple |
| `eq_ignore_ascii_case` | `(&str, &str) -> bool` | `str::eq_ignore_ascii_case` | none |
| `eq_ignore_case` | `(&str, &str) -> bool` | none | the full Unicode fold, which `str` has no method for |
| `capitalize` | `(&str) -> String` | none | the first scalar value uppercased |

## Written elsewhere

| Rust | how a script writes it |
| --- | --- |
| `s.to_owned()`, `s.to_string()` | `s.to_string()`: `string::to_string(a: &str) -> String`, at a `&str` or a `String` lent as one |
| `s.parse::<i64>()` | `i64::parse(s)` or `i64::from_str(s)`, one pair per integer width |
| `s.chars().count()` | `s.chars() \| count` |
| `s.chars().rev()` | `rev_iter(s.chars() \| collect)` — there is no `rev` adaptor |
| `parts.join(sep)`, `parts.concat()` | `parts \| join(sep)`, in `iterator` |
| `&s[a..b]` | `substring(s, a, b)` |
| `s.as_bytes()` | `to_bytes(s)`, which consumes the `String` |

## Not offered, and why

- **Pattern types beyond `&str`.** Rust's `Pattern` admits a `char`, a
  `&[char]` and a closure. One extern name holds one handler, so a second
  pattern type is a shared signature (RFC-0019) with an instance per
  pattern type — buildable, and not built here, because every function in
  the table would need its own signature and the `&str` form already spells
  every case a script has written.
- **`split_inclusive`, `split_ascii_whitespace`, `trim_ascii*`.** Rust's
  names, no caller yet.
- **`str::get(a..b)` as an `Option`.** `substring` refuses instead; the
  reason is above.
