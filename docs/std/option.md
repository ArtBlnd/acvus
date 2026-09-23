# `Option`

Rust's `Option` methods, the pure ones, in the `std` namespace. A script
writes them as Rust writes them: `o.map(|x| -> x * 2).unwrap_or(0)`.

Two differences hold for every row and are not repeated in the table.

**The receiver crosses by value.** Rust's `Option` methods take `&self` or
`&mut self` and leave the option in place; these take it by value. An
`Option` is the value's own kind (RFC-0039), so a local of that type is
stored as the runtime's value and not as a Rust `Option<T>`, and
`Ref<Option<T>, M, Rt>` has nothing to dereference.

**A closure parameter carries the handler's effect.** Every method that
takes a closure is declared `effect = E` over that closure's effect, so a
pure closure keeps the call pure and an effectful one propagates
(RFC-0023 rule 8).

A zero-parameter closure is written `| |`, with a space: `||` lexes as the
or operator.

| name | signature | Rust `std` twin | semantic difference |
| --- | --- | --- | --- |
| `unwrap` | `(Option<T>) -> T` | `Option::unwrap` | traps with `unwrap: called on None`; Rust's text names the type |
| `expect` | `(Option<T>, String) -> T` | `Option::expect` | none; the trap text is the message |
| `unwrap_or` | `(Option<T>, T) -> T` | `Option::unwrap_or` | none |
| `unwrap_or_else` | `(Option<T>, Fn() -> T) -> T` | `Option::unwrap_or_else` | none |
| `is_some` | `(Option<T>) -> Bool` | `Option::is_some` | none |
| `is_none` | `(Option<T>) -> Bool` | `Option::is_none` | none |
| `is_some_and` | `(Option<T>, Fn(T) -> Bool) -> Bool` | `Option::is_some_and` | none |
| `is_none_or` | `(Option<T>, Fn(T) -> Bool) -> Bool` | `Option::is_none_or` | none |
| `map_or` | `(Option<T>, U, Fn(T) -> U) -> U` | `Option::map_or` | the default is evaluated at the call, as in Rust |
| `map_or_else` | `(Option<T>, Fn() -> U, Fn(T) -> U) -> U` | `Option::map_or_else` | none |
| `and` | `(Option<T>, Option<U>) -> Option<U>` | `Option::and` | none |
| `and_then` | `(Option<T>, Fn(T) -> Option<U>) -> Option<U>` | `Option::and_then` | none |
| `or` | `(Option<T>, Option<T>) -> Option<T>` | `Option::or` | none |
| `or_else` | `(Option<T>, Fn() -> Option<T>) -> Option<T>` | `Option::or_else` | none |
| `xor` | `(Option<T>, Option<T>) -> Option<T>` | `Option::xor` | none |
| `flatten` | `(Option<Option<T>>) -> Option<T>` | `Option::flatten` | none |
| `ok_or` | `(Option<T>, Er) -> Result<T, Er>` | `Option::ok_or` | none |
| `ok_or_else` | `(Option<T>, Fn() -> Er) -> Result<T, Er>` | `Option::ok_or_else` | none |
| `into_iter` | `(Option<T>) -> Items<T>` | `IntoIterator for Option<T>` | an instance of the shared `iter::into_iter` signature; `Items<T>` is the owned source, of one element or none |

## What Rust has and this does not

`map` and `filter` **wait on `step_signature`** in
`acvus-mir/src/solver.rs`, here and in `result`. Both are written and both
register; what stops them is the solver, and each was measured one variable
apart. With `option::map` declared beside `iter::map`,
`acvus-ext/tests/e2e.rs` and `examples/grades` are refused at
`ps.as_iter().map(|p| -> p.x)`; with `option::filter` declared beside
`iter::filter`, `acvus-interpreter-test/tests/extern_call_forms.rs` is
refused at `a | filter(|x| -> x > 1) | count`. Unregistering that one name
alone makes that one test pass again. `step_signature`
narrows an overloaded call by the candidates the call shape still takes; a
receiver whose own type is not yet settled leaves two closure-taking
candidates open, the closure is then typed from its body alone, and the
type that yields belongs to neither. `flatten`, which takes no closure,
coexists, and `acvus-interpreter-test/tests/option_methods.rs` pins that.
Meanwhile a mapped option is `o.map_or(default, f)` and a filtered one is
`if o.is_some_and(p) { o } else { None }`.

`take`, `replace`, `insert`, `get_or_insert`, `get_or_insert_with` and
`as_ref` each need a place of type `Option<T>` to lend. The runtime stores
an option as its own value, so no storage anywhere is shaped like Rust's
`Option<T>`, and a reference has nothing to name.

`zip` and `unzip` yield a tuple. A tuple has `TyArg` only and no `Cross`
impl, so it is neither an argument nor a result at the boundary.

`unwrap_or_default` needs a generic `default<T>`. Nothing declares one; the
row returns when something does.

`copied` and `cloned` need a handler to require `core::clone` of its own
type parameter. A handler states bounds on the Rust types it crosses, not on
the language's signatures, so it cannot say this today.

`unwrap_unchecked` is not built: it trades the trap for undefined behavior,
and a script has no way to discharge the obligation.
