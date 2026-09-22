# `regex`

The `regex` crate's surface as the `Regex` and `Captures` extension types,
registered by `regex_registry` rather than by `std_registries`. Every offset
is a byte offset into the searched text. `docs/regex.md` is the prose on how
the module is used; this is the name table.

A pattern, a group name and a replacement are read and not kept, so all
three are `&str`: `regex("[0-9]+")`, `fields.named("stamp")`,
`re.replace_all(t, "~")`. A pattern that a call computes is bound first,
because a `String` result does not reach a `&str` parameter:
`let p = escape("a.c"); regex(&p)`.

`find_all`, `captures_all`, `split` and `split_n` answer a pipeline, and a
pipeline's type is the stage itself: each builds `Items<T>`, the owned
source, so the search finishes before the value is returned and the pipeline
borrows neither the `Regex` nor the text.

| name | signature | Rust `regex` twin | difference |
| --- | --- | --- | --- |
| `regex` | `(&str) -> Result<Regex, RegexError>` | `Regex::new` | the error carries the pattern and the crate's message |
| `regex_with` | `(&str, RegexFlags) -> Result<Regex, RegexError>` | `RegexBuilder` | all six switches at once, as a record |
| `regex_flags` | `() -> RegexFlags` | `RegexBuilder::new` defaults | none |
| `escape` | `(&str) -> String` | `regex::escape` | none |
| `is_match` | `(&Regex, &str) -> bool` | `Regex::is_match` | none |
| `is_match_at` | `(&Regex, &str, u64) -> bool` | `Regex::is_match_at` | a start past the end or inside a character matches nothing |
| `find` | `(&Regex, &str) -> Option<Match>` | `Regex::find` | `Match` is owned: `{ start: u64, end: u64, text: String }` |
| `find_at` | `(&Regex, &str, u64) -> Option<Match>` | `Regex::find_at` | a start past the end or inside a character gives `None` |
| `find_all` | `(&Regex, &str) -> Items<Match>` | `Regex::find_iter` | the search finishes before the iterator is returned |
| `shortest_match` | `(&Regex, &str) -> Option<u64>` | `Regex::shortest_match` | none |
| `captures` | `(&Regex, &str) -> Option<Captures>` | `Regex::captures` | `Captures` is owned, because `regex::Captures` borrows the text |
| `captures_all` | `(&Regex, &str) -> Items<Captures>` | `Regex::captures_iter` | the search finishes before the iterator is returned |
| `group` | `(&Captures, u64) -> Option<Match>` | `Captures::get` | none |
| `named` | `(&Captures, &str) -> Option<Match>` | `Captures::name` | none |
| `group_count` | `(&Regex) -> u64` | `Regex::captures_len` | group 0 is not counted |
| `group_names` | `(&Regex) -> Vec<Option<String>>` | `Regex::capture_names` | a `Vec`, not an iterator |
| `replace` | `(&Regex, &str, &str) -> String` | `Regex::replace` | always owned |
| `replace_all` | `(&Regex, &str, &str) -> String` | `Regex::replace_all` | always owned |
| `replace_n` | `(&Regex, &str, u64, &str) -> String` | `Regex::replacen` | an `n` of 0 replaces nothing |
| `replace_with` | `(&Regex, &String, Closure<(Match,), String>) -> String` | `Regex::replace_all` with a closure | no `$1` expansion; the closure returns the replacement itself |
| `split` | `(&Regex, &str) -> Items<String>` | `Regex::split` | elements are owned |
| `split_n` | `(&Regex, &str, u64) -> Items<String>` | `Regex::splitn` | an `n` of 0 gives no piece |

`RegexError` is `Invalid { pattern, message }`: the crate reports a syntax
error and a compiled-size overflow through one `Error` whose distinction is
in its text, so a script acts on the message.
