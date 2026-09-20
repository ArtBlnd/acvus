# `Result`

Rust's `Result` methods, the pure ones, in the `result` namespace. `Ok` and
`Err` are the language's constructors and `?` and `match` are the language's;
these are the methods around them.

The namespace is `result` and not `std` because `Option` already holds
`unwrap`, `map`, `and_then` and eight more names in `std`. A call resolves by
its argument's type across namespaces — the same way `string::len` and
`vec::len` both answer to `len` — so a script writes `r.unwrap()` and
`o.unwrap()` and never the namespace.

Two differences hold for every row and are not repeated in the table.

**The receiver crosses by value.** Rust's `Result` methods take `&self`;
these take it by value.

**The trap text names the method and the arm.** Rust renders the other
side's payload through `Debug`; a language value has no `Debug`, so
`unwrap` on an `Err` says `unwrap: called on Err` with nothing of the
payload in it.

A closure parameter carries the handler's effect, as in `Option` (RFC-0050
rule 6).

| name | signature | Rust `std` twin | semantic difference |
| --- | --- | --- | --- |
| `is_ok` | `(Result<T, Er>) -> Bool` | `Result::is_ok` | none |
| `is_err` | `(Result<T, Er>) -> Bool` | `Result::is_err` | none |
| `is_ok_and` | `(Result<T, Er>, Fn(T) -> Bool) -> Bool` | `Result::is_ok_and` | none |
| `is_err_and` | `(Result<T, Er>, Fn(Er) -> Bool) -> Bool` | `Result::is_err_and` | none |
| `ok` | `(Result<T, Er>) -> Option<T>` | `Result::ok` | none |
| `err` | `(Result<T, Er>) -> Option<Er>` | `Result::err` | none |
| `unwrap` | `(Result<T, Er>) -> T` | `Result::unwrap` | traps with `unwrap: called on Err` |
| `unwrap_err` | `(Result<T, Er>) -> Er` | `Result::unwrap_err` | traps with `unwrap_err: called on Ok` |
| `unwrap_or` | `(Result<T, Er>, T) -> T` | `Result::unwrap_or` | none |
| `unwrap_or_else` | `(Result<T, Er>, Fn(Er) -> T) -> T` | `Result::unwrap_or_else` | none |
| `expect` | `(Result<T, Er>, String) -> T` | `Result::expect` | the trap text is the message alone |
| `expect_err` | `(Result<T, Er>, String) -> Er` | `Result::expect_err` | the trap text is the message alone |
| `map_err` | `(Result<T, Er>, Fn(Er) -> F) -> Result<T, F>` | `Result::map_err` | none |
| `map_or` | `(Result<T, Er>, U, Fn(T) -> U) -> U` | `Result::map_or` | none |
| `map_or_else` | `(Result<T, Er>, Fn(Er) -> U, Fn(T) -> U) -> U` | `Result::map_or_else` | none |
| `and` | `(Result<T, Er>, Result<U, Er>) -> Result<U, Er>` | `Result::and` | none |
| `and_then` | `(Result<T, Er>, Fn(T) -> Result<U, Er>) -> Result<U, Er>` | `Result::and_then` | none |
| `or` | `(Result<T, Er>, Result<T, F>) -> Result<T, F>` | `Result::or` | none |
| `or_else` | `(Result<T, Er>, Fn(Er) -> Result<T, F>) -> Result<T, F>` | `Result::or_else` | none |
| `flatten` | `(Result<Result<T, Er>, Er>) -> Result<T, Er>` | `Result::flatten` | none |
| `transpose` | `(Result<Option<T>, Er>) -> Option<Result<T, Er>>` | `Result::transpose` | none |
| `into_iter` | `(Result<T, Er>) -> Iter<T>` | `IntoIterator for Result<T, E>` | none; an instance of the shared `iter::into_iter` signature |

## What Rust has and this does not

`map` is absent for the reason `docs/std/option.md` gives: a method whose
name `Iter` also carries and which takes a closure cannot be a second entry.
`Result` has no `filter`, so that row's loss does not reach here.

`unwrap_or_default` needs a generic `default<T>`, which nothing declares.

`copied` and `cloned` need a handler to require `core::clone` of its own
type parameter, which a handler cannot state.

`unwrap_unchecked` and `unwrap_err_unchecked` are not built: they trade the
trap for undefined behavior, and a script has no way to discharge the
obligation.
