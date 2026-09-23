# `Vec<T>`

Every function `vec` offers, the Rust name it carries, and where the
language's answer is not Rust's. The operations over a container's
*elements* are `docs/std/slice.md`'s, reached through the `as_slice` view.

A `Vec<T>` is Rust's `Vec<T>`. Indices are `u64`; where Rust panics on an
index, the language traps with Rust's own message. Integer overflow is what
the language does everywhere else: `i64` arithmetic wraps, as `num::abs`
and `num::pow` do (`acvus-ext/src/num.rs`, RFC-0058).

`as_slice` and `as_slice_mut` are the view. They are the coercion the
compiler inserts behind `&v` at a `&[T]` parameter, behind `a[i]` and behind
`for x in &v` (RFC-0047 rules 3 and 5), and a script may also write the call itself,
which resolves to the container's own instance. Where the target asks for no
view, `&v` stays a `&Vec<T>`.

A `&Vec<T>` and a `&Array<T, N>` reach the `slice` surface through that same
view, so the element operations — `sort`, `contains`, `binary_search`,
`is_sorted`, `to_vec`, `starts_with`, `ends_with`, `repeat`, `min`, `max`,
`rotate_left`, `rotate_right` — live once, over `&[T]`, and are tabulated in
`docs/std/slice.md`.

## `vec`

| language | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `vec` | `vec(items: C) -> Vec<T>` | `Vec::from` | a shared signature; any container demotes to a vec (RFC-0019) |
| `filled` | `filled(n: u64, x: T) -> Vec<T>` | `vec![x; n]` | instances for `i64`, `f64`, `bool`, `String` |
| `with_capacity` | `with_capacity(n: u64) -> Vec<T>` | `Vec::with_capacity` | none |
| `len` | `len(c: &Vec<T>) -> u64` | `Vec::len` | `u64`, not `usize` |
| `is_empty` | `is_empty(c: &Vec<T>) -> bool` | `Vec::is_empty` | none |
| `capacity` | `capacity(c: &Vec<T>) -> u64` | `Vec::capacity` | `u64`, not `usize` |
| `shrink_to_fit` | `shrink_to_fit(c: &mut Vec<T>)` | `Vec::shrink_to_fit` | none |
| `as_slice` | `as_slice(c: &Vec<T>) -> &[T]` | `Vec::as_slice` | none |
| `as_slice_mut` | `as_slice_mut(c: &mut Vec<T>) -> &mut [T]` | `Vec::as_mut_slice` | Rust's name is `as_mut_slice` |
| `first` | `first(c: &Vec<T>) -> Option<&T>` | `slice::first` | none |
| `last` | `last(c: &Vec<T>) -> Option<&T>` | `slice::last` | the name is also `iter::last`, `array::last`, `deque::last` and `slice::last`; where a call does not settle it is written `vec::last(&v)` |
| `get` | `get(c: &Vec<T>, at: u64) -> Option<&T>` | `slice::get` | `u64` index; an index past the address space is `None` |
| `push` | `push(c: &mut Vec<T>, x: T)` | `Vec::push` | none |
| `pop` | `pop(c: &mut Vec<T>) -> Option<T>` | `Vec::pop` | none |
| `insert` | `insert(c: &mut Vec<T>, at: u64, x: T)` | `Vec::insert` | traps past the length, with Rust's message |
| `remove` | `remove(c: &mut Vec<T>, at: u64) -> T` | `Vec::remove` | traps past the last element, with Rust's message |
| `clear` | `clear(c: &mut Vec<T>)` | `Vec::clear` | none |
| `truncate` | `truncate(c: &mut Vec<T>, n: u64)` | `Vec::truncate` | none |
| `extend` | `extend(c: &mut Vec<T>, items: Vec<T>)` | `Vec::append` | takes the other vec by value |
| `split_off` | `split_off(c: &mut Vec<T>, at: u64) -> Vec<T>` | `Vec::split_off` | traps past the length, with Rust's message |
| `swap` | `swap(c: &mut Vec<T>, i: u64, j: u64)` | `slice::swap` | none; `slice::swap` is the same operation over the view |
| `reverse` | `reverse(items: Vec<T>) -> Vec<T>` | `slice::reverse` | consumes and returns, where Rust's reverses in place |
| `sort_by` | `sort_by(c: &mut Vec<T>, f: \|&T, &T\| -> i64)` | `slice::sort_by` | stable; the closure answers −1/0/1, the protocol `string::cmp` speaks, because the language has no `Ordering` |
| `sort_by_key` | `sort_by_key(c: &mut Vec<T>, f: \|&T\| -> i64)` | `slice::sort_by_cached_key` | stable; the key is `i64` and nothing else, as `iter::min_by_key`'s is — a `Closure` has no specialized representation for a `Monomorphize` member to cross at. Rust's `sort_by_key` calls its closure once per *comparison*; this one calls it once per *element*, as `sort_by_cached_key` does, because a closure that crosses the boundary is dear. For a key that answers the same twice, the order is the same |

`sort_by` and `sort_by_key` stay the vec's own. Each is generic in `T` and
reads no element at a type — it hands the closure a `&T` and moves the
elements the verdicts order — and the namespace a registry carries is the
name a script writes.

## Walking a vec

There is no conversion from a container to a pipeline: `v | fold(..)` does
not compile. A script names the source. `into_iter(v)` consumes the vec and
yields its elements by value; `as_iter(&v)` yields references and leaves the
vec usable, so a closure over it receives references. Both spellings take
the method form as well — `v.into_iter() | max()`,
`ps.as_iter().map(|p| -> p.x)`.

`as_iter(&v) | max` is refused. `iter::max` bounds its element to `i64` or
`f64` and takes it by value, and `as_iter` yields references, so the
aggregate over a borrowing view is outside that bound. `slice::max(&v)`,
`v.max()` and `into_iter(v) | max` all answer.

`retain` is `into_iter(v) | filter(f) | collect`, and `drain` is
`into_iter(v)`; `chunks`, `join`, `sum`, `product` and `position` are the
iterator's, over `into_iter` or `as_iter`.

A consumer takes the pipeline's own type and its element, not one iterator
type: `fold` is `Fn(I, U, Fn(U, T) -> U) -> U`, where `I` is any pipeline
whose element is a `T`. The instance of `iter::next` the pipeline answers to
is required by the consumer and passed at the call; a script writes none of
it. `next(&mut it)` is that signature called directly, so
`while let Some(x) = next(&mut it) { .. }` walks a pipeline without a
consumer.

`dedup` is one stage over any element type that has an instance of
`core::eq`, which it requires and the call site decides (RFC-0070 rule 5):
`into_iter([1, 1, 2]) | dedup` collapses to `[1, 2]`. The stage holds the
element it last drew rather than a copy of it, so it requires no
`core::clone` and runs one draw behind its source; what it yields is
unchanged by that.

`Vec<T>` has an instance of each of `core::eq`, `core::clone`, `core::cmp`
and `core::hash`, and each requires the same signature at `T`
(RFC-0070 rule 5). So `clone(&v)` copies a `Vec<Vec<String>>` through the
`String` instance at the bottom, `cmp` is lexicographic with the length
deciding a tie between a prefix and what extends it, and `hash` folds the
element digests in the order the elements are in. An element type with no
instance of the signature refuses the call and names itself; that is what a
`Vec` of objects meets.

`pchain` is not built. It was refused by a rule that no longer holds — a
requirement's variable had to be a parameter's own type — and nobody has
since checked whether a `Vec<I>` of pipelines resolves, so this is an
absence rather than a bound.

## Not here

The element operations are not declared over `Vec<T>`. `sort`, `contains`,
`binary_search`, `is_sorted`, `to_vec`, `starts_with`, `ends_with`,
`repeat`, `min`, `max`, `rotate_left` and `rotate_right` each read or order
an element at its own type; each is declared once over `&[T]`, and a vec
reaches it through `as_slice` — `v.sort()` and `v.max()` in the method form,
`slice::contains(&v, &x)` where the bare name does not settle.

`dedup`, `fill` and `resize` are a bound rather than an omission. A
container's storage is a `Vec<Owned<Rt>>`, so a handler reaches its elements
at their own type only as a run of fixed length whose elements it may not
drop; the first two change the length and the third overwrites an owned
value. A script fills with `filled(n, x)`.

`get_mut`, `first_mut` and `last_mut` would each answer `Option<&mut T>`,
and the language does not carry a `&mut` inside an `Option` — a store
through what such a call binds is refused with "cannot store through &_:
not a `&mut`". An element is written through `v[i] = x`.

`split_at` is not here: one call returns one pair (RFC-0062), and two views
are two pairs.

`windows(n)`, `concat`, `join` over a `Vec<Vec<T>>`, `sort_unstable` and
`iter_mut` are not built; `docs/std/slice.md` gives the reason for each.
