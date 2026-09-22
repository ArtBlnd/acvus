# `Array<T, N>`

The fixed-length sequence. `N` is the language's length variable, so every
function here is generic in it and none changes the length.

An array's view is `as_slice` / `as_slice_mut`, which a script may write as
a call as well as receive as the coercion behind `&a` at a `&[T]` parameter,
behind `a[i]` and behind `for x in &a` (RFC-0047 §5).
Through it an array reaches everything `docs/std/slice.md` tabulates. A
`&Vec<T>` and a `&Array<T, N>` reach that surface through the same
`as_slice` view, so the element operations live once, over `&[T]`.

| language | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `len` | `len(c: &Array<T, N>) -> u64` | `slice::len` | `u64`, not `usize` |
| `is_empty` | `is_empty(c: &Array<T, N>) -> bool` | `slice::is_empty` | none |
| `as_slice` | `as_slice(c: &Array<T, N>) -> &[T]` | `array::as_slice` | none |
| `as_slice_mut` | `as_slice_mut(c: &mut Array<T, N>) -> &mut [T]` | `array::as_mut_slice` | Rust's name is `as_mut_slice` |
| `first` | `first(c: &Array<T, N>) -> Option<&T>` | `slice::first` | none |
| `last` | `last(c: &Array<T, N>) -> Option<&T>` | `slice::last` | the name is also `iter::last`, `vec::last`, `deque::last` and `slice::last`; where a call does not settle it is written `array::last(&a)` |
| `get` | `get(c: &Array<T, N>, at: u64) -> Option<&T>` | `slice::get` | `u64` index; an index past the address space is `None` |

## The element surface is the view's

`sort`, `contains`, `binary_search`, `is_sorted`, `to_vec`, `starts_with`,
`ends_with`, `repeat`, `min` and `max` each read an element at its own type,
and every such operation is declared once over `&[T]` in
`docs/std/slice.md`. An array reaches them through `as_slice`: the method
form carries the receiver onto the view — `a.is_sorted()`,
`a.binary_search(&x)`, `a.get(2u64)` — and where the bare name does not
settle the script writes the namespace, `slice::contains(&a, &x)`.
`acvus-ext/tests/vec_ops.rs` carries these programs, and one of them settles
`is_sorted` at an array and at a vec in the same script.

`vec(a)` is the declared cast and answers the array's elements as a `Vec`.

## Not here

A fixed length admits no `push`, `pop` or `resize`, and none is declared.

`sort` is not the array's either. It is declared over `&mut [T]` in
`docs/std/slice.md`, which an array reaches through `as_slice_mut` — the
order of a run of fixed length, not a change to its length.
