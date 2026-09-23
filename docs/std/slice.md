# `&[T]` — namespace `slice`

Every operation over a container's *elements* is declared once here, over
`&[T]` and `&mut [T]`. A `Vec<T>` and an `Array<T, N>` reach it through
their own `as_slice` / `as_slice_mut` view, the way Rust's `[T]` serves
`Vec<T>` and `[T; N]` through deref.

The view is a call a script may write (`v.as_slice()`), and it is also the
coercion the compiler inserts behind `&v` at a `&[T]` parameter, behind
`a[i]` and behind `for x in &v` (RFC-0047 rules 3 and 5). Where the target asks for no
view, `&v` stays a `&Vec<T>`.

Indices are `u64`; where Rust panics on an index, the language traps with
Rust's own message.

The surface has two halves. The first is generic in the element and only
moves it: `len`, `is_empty`, `get`, `first`, `last`, `swap`, `rotate_left`,
`rotate_right`. The second *reads* an element at its own type, so each is a
shared signature (RFC-0019) with one instance per element type — `i64`,
`u64`, `f64`, `bool`, `String` — and each instance reads the element in
place through the crossing's own `Erased::as_ref` (RFC-0041), so no element
is read back at a type the checker did not settle.

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
| `sort` | `sort(s: &mut [T])` | `slice::sort` | stable, as Rust's; `f64` orders by `total_cmp`, which is what Rust's own float slices sort by |
| `contains` | `contains(s: &[T], x: &T) -> bool` | `slice::contains` | the argument must be a variable — the language has no `&literal` |
| `binary_search` | `binary_search(s: &[T], x: &T) -> Option<u64>` | `slice::binary_search` | Rust's `Err(at)` carries where the element would go; the language has no `Result<u64, u64>` to tell two indices of one meaning apart, so a miss is `None` |
| `is_sorted` | `is_sorted(s: &[T]) -> bool` | `slice::is_sorted` | none |
| `to_vec` | `to_vec(s: &[T]) -> Vec<T>` | `slice::to_vec` | none |
| `starts_with` | `starts_with(s: &[T], prefix: &[T]) -> bool` | `slice::starts_with` | none |
| `ends_with` | `ends_with(s: &[T], suffix: &[T]) -> bool` | `slice::ends_with` | none |
| `repeat` | `repeat(s: &[T], times: u64) -> Vec<T>` | `slice::repeat` | `u64` count; a total length past the address space traps with "capacity overflow" |
| `min` | `min(s: &[T]) -> Option<T>` | `slice::iter().min()` | answers the element, where Rust's yields `Option<&T>`: a shared signature's result is an owned value. First of equal elements, as Rust's |
| `max` | `max(s: &[T]) -> Option<T>` | `slice::iter().max()` | as `min`, and last of equal elements, as Rust's |

## Reaching it from a container

The method form carries a `Vec` or an `Array` receiver onto the view:
`v.sort()`, `v.is_sorted()`, `v.binary_search(&x)`, `v.to_vec()`,
`v.max()`, `a.is_sorted()`, `a.binary_search(&x)`. A named view takes the
same methods — `s.len()`, `s.first()`, `s.get(1u64)`,
`v.as_slice_mut().swap(0, 2)`, `v.as_slice_mut().rotate_left(2)`.
`acvus-ext/tests/vec_ops.rs` carries each of these programs.

Several of these names are declared by more than one namespace: `last` by
`iter`, `vec`, `array` and `deque`; `contains` by `iter`, `set` and
`string`; `min` and `max` by `num` and `iter`; `starts_with`, `ends_with`
and `repeat` by `string`. Where a call does not settle, the script writes
the namespace — `slice::contains(&v, &x)`, `slice::last(s)`,
`slice::min(&v)`, `slice::max(&v)`, `slice::starts_with(&v, &p)`,
`slice::ends_with(&v, &s)`, `slice::repeat(&v, 3u64)`, which is how the
tests spell them.

A named `&[T]` is not a `for` source and is not indexable: `for` traverses
`&v`, `&mut v`, an array or a range, and `a[i]` takes its own view of a
container. A view bound to a name is walked through `len` and `get`.

## Not here

`dedup`, `fill` and `resize` are a bound rather than an omission. A
container's storage is a `Vec<Owned<Rt>>`, so a handler reaches its elements
at their own type only as a run of fixed length whose elements it may not
drop; the first two change the length and the third overwrites an owned
value. A script fills with `filled(n, x)`.

`get_mut`, `first_mut` and `last_mut` would each answer `Option<&mut T>`,
and the language does not carry a `&mut` inside an `Option` — a store
through what such a call binds is refused with "cannot store through &_:
not a `&mut`". An element is written through `v[i] = x`, which is the
indexing instruction and keeps the exclusive loan at the place (RFC-0047).

There is no `slice::reverse`, and that is a collision rather than a bound.
`vec::reverse` holds the bare name with a by-value `Vec<T> -> Vec<T>`
signature, and a second declaration stops a call that settles without it:
`reverse(x)` over a `Vec<#Float>` is refused with "no `reverse` takes a call
of type Fn(Vec<#Float>) -> Vec<_>". Measured one variable apart —
registering it fails `acvus-mir-test/tests/regression_0041.rs`'s
`a_specialized_local_pays_one_erase_per_generic_consumer_and_none_at_its_member`,
unregistering it passes. A view is reversed by `swap` over its halves until
`vec::reverse` takes Rust's in-place shape or RFC-0043 settles the pair.

`split_at` is not here: one call returns one pair (RFC-0062), and two views
are two pairs.

`windows(n)` is not built. Rust's yields views, and a view cannot be a
pipeline's element: a slice crosses as a register pair where a stage's
element is one value. `iter::chunks` is built for the same reason it can be
— it yields a `Vec`, not a view. A `windows` returning `Vec<Vec<T>>` would
build, at the cost of copying every window, and whether that is worth having
under Rust's name is a decision not yet made.

`concat` and `join` for a `Vec<Vec<T>>` are not built. The outer run's
elements are each a whole container erased into one value, and reading one
back at its own type is the `NO_VEC_STORAGE` trap again — a nested run needs
a crossing `acvus-extern` does not offer. `join` is additionally the
iterator's name.

`sort_unstable` is not declared: `sort` here and `vec::sort_by` are both
stable, the language has no unstable one to distinguish, and an alias that
promises less than it delivers is a name with no meaning.

`iter_mut` is not built: no stage has a `&mut` element form, for the same
reason `get_mut` is absent.
