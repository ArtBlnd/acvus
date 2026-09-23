# Regular expressions

`acvus-ext`'s `regex_registry` puts the `regex` crate under `std`. A pattern
is compiled once into a `Regex` and every search borrows it, so one compiled
regex serves any number of searches:

```
if let Ok(re) = regex("[0-9]+") {
  let a = is_match(&re, &first);
  let b = is_match(&re, &second);
  ...
}
```

The method form is the same call with the regex as the receiver:
`re.is_match(&text)` is `is_match(&re, &text)`. Three functions here share a
bare name with a function elsewhere in the standard library — `find`,
`replace` and `split` — and the receiver type is what picks the one meant:
`find(&text, "42")` is `string::find` and reports a byte offset, while
`find(&re, &text)` is this module's and reports a `Match`.

## Offsets are bytes

Every offset a regex function takes or reports is a byte offset into the
searched text, the unit `string::substring` cuts with. `string::find` and
`string::char_at` count characters instead, so an offset from one family
does not index the other. In `"héllo 42"` the digits begin at byte 7 and at
character 6.

## The functions

| function | signature | contract |
| --- | --- | --- |
| `regex` | `(pattern: String) -> Result<Regex, RegexError>` | compiles under the flags `regex_flags()` names |
| `regex_with` | `(pattern: String, flags: RegexFlags) -> Result<Regex, RegexError>` | compiles under `flags` |
| `regex_flags` | `() -> RegexFlags` | the flag set `regex` itself compiles with |
| `escape` | `(text: String) -> String` | a pattern matching `text` literally |
| `is_match` | `(&Regex, text: &String) -> Bool` | whether the pattern matches anywhere |
| `is_match_at` | `(&Regex, text: &String, start: u64) -> Bool` | as `is_match`, from byte `start`; `^` still anchors to byte 0 |
| `find` | `(&Regex, text: &String) -> Option<Match>` | the leftmost match; `None` where the pattern does not match |
| `find_at` | `(&Regex, text: &String, start: u64) -> Option<Match>` | as `find`, from byte `start`; a `start` past the end or inside a character gives `None` |
| `find_all` | `(&Regex, text: &String) -> Iter<Match>` | every non-overlapping match, left to right |
| `shortest_match` | `(&Regex, text: &String) -> Option<u64>` | the byte end of the shortest match at the leftmost matching position |
| `captures` | `(&Regex, text: &String) -> Option<Captures>` | the groups of the leftmost match |
| `captures_all` | `(&Regex, text: &String) -> Iter<Captures>` | the groups of every non-overlapping match |
| `group` | `(&Captures, i: u64) -> Option<Match>` | group `i`, group 0 being the whole match; `None` where the pattern has no such group or the group did not participate |
| `named` | `(&Captures, name: String) -> Option<Match>` | as `group`, by group name |
| `group_count` | `(&Regex) -> u64` | how many groups the pattern declares, not counting group 0 |
| `group_names` | `(&Regex) -> Vec<Option<String>>` | each group's name at its own index, group 0 first and always unnamed |
| `replace` | `(&Regex, text: &String, with: String) -> String` | the leftmost match replaced |
| `replace_all` | `(&Regex, text: &String, with: String) -> String` | every non-overlapping match replaced |
| `replace_n` | `(&Regex, text: &String, n: u64, with: String) -> String` | the first `n` matches replaced; an `n` of 0 replaces nothing |
| `replace_with` | `(&Regex, text: &String, f: Fn(Match) -> String) -> String` | every match replaced by what `f` returns for it |
| `split` | `(&Regex, text: &String) -> Iter<String>` | the pieces between matches; a match at either end gives an empty piece there |
| `split_n` | `(&Regex, text: &String, n: u64) -> Iter<String>` | at most `n` pieces, the last holding the rest of the text |

`find_all`, `captures_all`, `split` and `split_n` finish the search before
they return. An `Iter` stage outlives the call that built it and a borrow
does not (RFC-0018), so a lazy stage could hold neither the regex nor the
text.

## Match and Captures

`Match` is an object with three fields, read as `m.start`, `m.end` and
`m.text`: the byte offsets bounding the match as `[start, end)`, and the
text it covers.

`Captures` is an extension type holding one match's groups. It is read
through `group` and `named`, not by field: the groups are owned copies,
because `regex::Captures` borrows the searched text and a borrow cannot
cross into the language.

Group 1 of every match is `captures_all` and `map`:

```
captures_all(&re, &text)
  | map(|c| -> if let Some(m) = group(&c, 1u64) { m.text } else { "" })
```

## Replacement syntax

`replace`, `replace_all` and `replace_n` expand the replacement text as the
`regex` crate does: `$1` and `${1}` are the group of that number, `$name`
and `${name}` the group of that name, and `$$` is one dollar sign. A name
runs to the first character that is neither a letter, a digit nor an
underscore, so `${name}` is how a name is written next to text that would
otherwise continue it.

`replace_with` does no expansion. Its closure receives each `Match` and
returns the replacement itself.

## Flags

`RegexFlags` is an object of six switches, every one a `Bool`:
`case_insensitive`, `multi_line`, `dot_matches_new_line`,
`ignore_whitespace`, `unicode`, `swap_greed`. They are `RegexBuilder`'s
switches of the same names.

`RegexFlags` declares those six fields and a value of it has exactly
them, so a literal that leaves one out is refused at the call by the
field's name (RFC-0042 rule 1). Write all six, or start from `regex_flags()`,
which is Unicode on and every other switch off — the set `regex` itself
compiles with.

```
let flags = { case_insensitive: true, multi_line: false,
              dot_matches_new_line: false, ignore_whitespace: false,
              unicode: true, swap_greed: false, };
regex_with("abc", flags)
```

## Errors

A pattern that is not a regular expression gives
`RegexError::Invalid { pattern, message }`. Syntax and compiled-size
overflow are not separate variants: the `regex` crate reports both through
one `Error` whose distinction is in its text alone, so a script that wants
to tell them apart reads `message`.
