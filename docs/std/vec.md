# `Vec<T>` and `&[T]`

Every function `vec` and `slice` offer, the Rust name it carries, and where
the language's answer is not Rust's.

A `Vec<T>` is Rust's `Vec<T>`. Indices are `u64`; where Rust panics on an
index, the language traps with Rust's own message. Integer overflow is what
the language does everywhere else: `i64` arithmetic wraps, as `num::abs`
and `num::pow` do (`acvus-ext/src/num.rs`, RFC-0058).

`as_slice` and `as_slice_mut` are the view. They are still the coercion the
compiler inserts behind `&v` at a `&[T]` parameter, behind `a[i]` and behind
`for x in &v` (RFC-0047 §5), and a script may also write the call itself,
which resolves to the container's own instance. Where the target asks for no
view, `&v` stays a `&Vec<T>`.

## `vec`

| language | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `vec` | `vec(items: C) -> Vec<T>` | `Vec::from` | a shared signature; any container demotes to a vec (RFC-0027) |
| `filled` | `filled(n: u64, x: T) -> Vec<T>` | `vec![x; n]` | instances for `i64`, `f64`, `bool`, `String` |
| `with_capacity` | `with_capacity(n: u64) -> Vec<T>` | `Vec::with_capacity` | none |
| `len` | `len(c: &Vec<T>) -> u64` | `Vec::len` | `u64`, not `usize` |
| `is_empty` | `is_empty(c: &Vec<T>) -> bool` | `Vec::is_empty` | none |
| `capacity` | `capacity(c: &Vec<T>) -> u64` | `Vec::capacity` | `u64`, not `usize` |
| `shrink_to_fit` | `shrink_to_fit(c: &mut Vec<T>)` | `Vec::shrink_to_fit` | none |
| `as_slice` | `as_slice(c: &Vec<T>) -> &[T]` | `Vec::as_slice` | none |
| `as_slice_mut` | `as_slice_mut(c: &mut Vec<T>) -> &mut [T]` | `Vec::as_mut_slice` | Rust's name is `as_mut_slice` |
| `first` | `first(c: &Vec<T>) -> Option<&T>` | `slice::first` | none |
| `last` | `last(c: &Vec<T>) -> Option<&T>` | `slice::last` | ambiguous as a bare name against `iter::last`; written `vec::last(&v)` |
| `get` | `get(c: &Vec<T>, at: u64) -> Option<&T>` | `slice::get` | `u64` index; an index past the address space is `None` |
| `push` | `push(c: &mut Vec<T>, x: T)` | `Vec::push` | none |
| `pop` | `pop(c: &mut Vec<T>) -> Option<T>` | `Vec::pop` | none |
| `insert` | `insert(c: &mut Vec<T>, at: u64, x: T)` | `Vec::insert` | traps past the length, with Rust's message |
| `remove` | `remove(c: &mut Vec<T>, at: u64) -> T` | `Vec::remove` | traps past the last element, with Rust's message |
| `clear` | `clear(c: &mut Vec<T>)` | `Vec::clear` | none |
| `truncate` | `truncate(c: &mut Vec<T>, n: u64)` | `Vec::truncate` | none |
| `extend` | `extend(c: &mut Vec<T>, items: Vec<T>)` | `Vec::append` | takes the other vec by value |
| `split_off` | `split_off(c: &mut Vec<T>, at: u64) -> Vec<T>` | `Vec::split_off` | traps past the length, with Rust's message |
| `swap` | `swap(c: &mut Vec<T>, i: u64, j: u64)` | `slice::swap` | none |
| `reverse` | `reverse(items: Vec<T>) -> Vec<T>` | `slice::reverse` | consumes and returns, where Rust's reverses in place |
| `sort` | `sort(c: &mut Vec<T>)` | `slice::sort` | stable, as Rust's; instances for `i64`, `u64`, `f64`, `bool`, `String`; `f64` orders by `total_cmp`, which is what Rust's own float slices sort by |
| `sort_by` | `sort_by(c: &mut Vec<T>, f: \|&T, &T\| -> i64)` | `slice::sort_by` | stable; the closure answers −1/0/1, the protocol `string::cmp` speaks, because the language has no `Ordering` |
| `sort_by_key` | `sort_by_key(c: &mut Vec<T>, f: \|&T\| -> i64)` | `slice::sort_by_cached_key` | stable; the key is `i64` and nothing else, as `iter::min_by_key`'s is — a `Closure` has no specialized representation for a `Monomorphize` member to cross at. Rust's `sort_by_key` calls its closure once per *comparison*; this one calls it once per *element*, as `sort_by_cached_key` does, because a closure that crosses the boundary is dear. For a key that answers the same twice, the order is the same |
| `is_sorted` | `is_sorted(c: &Vec<T>) -> bool` | `slice::is_sorted` | same instance set as `sort` |
| `min` | `min(c: &Vec<T>) -> Option<T>` | `slice::iter().min()` | answers the element, where Rust's yields `Option<&T>`: a shared signature's result is an owned value, and `iter::min` already spells the language's `min` that way. First of equal elements, as Rust's; same instance set as `sort`. Written `min(&v)`: the method form `v.min()` waits on the receiver-mode rule, as `contains` does |
| `max` | `max(c: &Vec<T>) -> Option<T>` | `slice::iter().max()` | as `min`, and last of equal elements, as Rust's |
| `contains` | `contains(c: &Vec<T>, x: &T) -> bool` | `slice::contains` | ambiguous as a bare name against `iter::contains`; written `vec::contains(&v, &x)`. The argument must be a variable — the language has no `&literal` |
| `binary_search` | `binary_search(c: &Vec<T>, x: &T) -> Option<u64>` | `slice::binary_search` | Rust's `Err(at)` carries where the element would go; the language has no `Result<u64, u64>` to tell two indices of one meaning apart, so a miss is `None` |
| `to_vec` | `to_vec(c: &Vec<T>) -> Vec<T>` | `slice::to_vec` | same instance set as `sort` |
| `starts_with` | `starts_with(c: &Vec<T>, prefix: &Vec<T>) -> bool` | `slice::starts_with` | the prefix is a `&Vec<T>`, not a `&[T]`; same instance set as `sort` |
| `ends_with` | `ends_with(c: &Vec<T>, suffix: &Vec<T>) -> bool` | `slice::ends_with` | the suffix is a `&Vec<T>`, not a `&[T]`; same instance set as `sort` |
| `repeat` | `repeat(c: &Vec<T>, times: u64) -> Vec<T>` | `slice::repeat` | `u64` count; a total length past the address space traps with "capacity overflow" |

## `slice`

The view's own surface. Each takes a `&[T]` or a `&mut [T]`, so a script
reaches it through `v.as_slice()`, `a.as_slice()` or a `&[T]` parameter. A
method receiver does not coerce to a slice parameter, so `v.reverse()` is
`vec::reverse`, not this one.

| language | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `len` | `len(s: &[T]) -> u64` | `slice::len` | `u64`, not `usize` |
| `is_empty` | `is_empty(s: &[T]) -> bool` | `slice::is_empty` | none |
| `get` | `get(s: &[T], at: u64) -> Option<&T>` | `slice::get` | `u64` index; an index past the address space is `None` |
| `first` | `first(s: &[T]) -> Option<&T>` | `slice::first` | none |
| `last` | `last(s: &[T]) -> Option<&T>` | `slice::last` | none |
| `swap` | `swap(s: &mut [T], i: u64, j: u64)` | `slice::swap` | none |
| `rotate_left` | `rotate_left(s: &mut [T], mid: u64)` | `slice::rotate_left` | traps unless `mid <= len`, as Rust does |
| `rotate_right` | `rotate_right(s: &mut [T], k: u64)` | `slice::rotate_right` | traps unless `k <= len`, as Rust does |

A named `&[T]` is not a `for` source and is not indexable: `for` traverses
`&v`, `&mut v`, an array or a range, and `a[i]` takes its own view of a
container. A view bound to a name is walked through `len` and `get`.

## Not here

`dedup`, `fill` and `resize` are a bound rather than an omission. A
container's storage is a `Vec<Owned<Rt>>`, so a handler reaches its elements
at their own type only as a run of fixed length whose elements it may not
drop; the first two change the length and the third overwrites an owned
value. A script dedups with `into_iter(v) | dedup | collect` and fills with
`filled(n, x)`.

`get_mut`, `first_mut` and `last_mut` would each answer
`Option<&mut T>`, and the language does not carry a `&mut` inside an
`Option` — a store through what such a call binds is refused with "cannot
store through &_: not a `&mut`". An element is written through `v[i] = x`.

`as_iter(&v) | max` is still refused: `iter::max` takes its element by
value and has no instance at a reference element, so the aggregate over a
borrowing view is "`&i64` is outside the declared bound". `max(&v)` and
`into_iter(v) | max` both answer.

There is still no `slice::reverse`, and that is a collision rather than a
bound. `vec::reverse` holds the bare name with a by-value
`Vec<T> -> Vec<T>` signature, and a second declaration stops a call that
used to settle: `reverse(x)` over a `Vec<#Float>` is refused with "no
`reverse` takes a call of type Fn(Vec<#Float>) -> Vec<_>". Measured one
variable apart — registering it fails
`acvus-mir-test/tests/regression_0041.rs`, unregistering it passes. A view
is reversed by `swap` over its halves until `vec::reverse` takes Rust's
in-place shape or RFC-0043 settles the pair.

A method receiver written `v.f()` does not settle between a candidate that
lends it and one that consumes it, so `v.max()`, `v.min()` and
`v.contains(&x)` are each "`f` is declared by iter::f and vec::f" while
`max(&v)`, `min(&v)` and `contains(&v, &x)` settle. Measured one variable
apart in `acvus-ext/tests/vec_ops.rs`. The call form is the answer until
RFC-0043 gains a mode rule for method receivers.

`split_at` is not here: one call returns one pair (RFC-0062), and two views
are two pairs.

`retain` is `into_iter(v) | filter(f) | collect`, and `drain` is
`into_iter(v)`; `chunks`, `join`, `sum`, `product` and `position` are the
iterator's, over `into_iter` or `as_iter`.

`windows(n)` is not built. Rust's yields views, and a view cannot be an
iterator's element: a slice crosses as a register pair where an `Iter`'s
element is one value, so `Iter<&[T]>` has no crossing. `chunks` is the
iterator's for the same reason it can be — it yields a `Vec`, not a view.
A `windows` returning `Vec<Vec<T>>` would build, at the cost of copying
every window, and whether that is worth having under Rust's name is a
decision not yet made.

`concat` and `join` for a `Vec<Vec<T>>` are not built. The outer run's
elements are each a whole container erased into one value, and reading one
back at its own type is the `NO_VEC_STORAGE` trap again — a nested run
needs a crossing `acvus-extern` does not offer. `join` is additionally the
iterator's name.

`sort_unstable` is not declared: both of this file's sorts are stable, the
language has no unstable one to distinguish, and an alias that promises
less than it delivers is a name with no meaning.

`iter_mut` is not built: the iterator has no `&mut` element form, for the
same reason `get_mut` is absent.
