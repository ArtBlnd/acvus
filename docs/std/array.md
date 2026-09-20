# `Array<T, N>`

The fixed-length sequence. `N` is the language's length variable, so every
function here is generic in it and none changes the length.

An array's view is `as_slice` / `as_slice_mut`, which a script may write as
a call as well as receive as the coercion behind `&a` at a `&[T]` parameter,
behind `a[i]` and behind `for x in &a` (RFC-0047 §5).
Through it an array reaches everything `docs/std/vec.md`'s `slice` table
offers.

| language | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `len` | `len(c: &Array<T, N>) -> u64` | `slice::len` | `u64`, not `usize` |
| `is_empty` | `is_empty(c: &Array<T, N>) -> bool` | `slice::is_empty` | none |
| `as_slice` | `as_slice(c: &Array<T, N>) -> &[T]` | `array::as_slice` | none |
| `as_slice_mut` | `as_slice_mut(c: &mut Array<T, N>) -> &mut [T]` | `array::as_mut_slice` | Rust's name is `as_mut_slice` |
| `first` | `first(c: &Array<T, N>) -> Option<&T>` | `slice::first` | none |
| `last` | `last(c: &Array<T, N>) -> Option<&T>` | `slice::last` | ambiguous as a bare name against `iter::last`; written `array::last(&a)` |
| `get` | `get(c: &Array<T, N>, at: u64) -> Option<&T>` | `slice::get` | `u64` index; an index past the address space is `None` |
| `contains` | `contains(c: &Array<T, N>, x: &T) -> bool` | `slice::contains` | ambiguous as a bare name against `iter::contains`; written `array::contains(&a, &x)`. The argument must be a variable — the language has no `&literal`. Instances for `i64`, `u64`, `f64`, `bool`, `String` |
| `binary_search` | `binary_search(c: &Array<T, N>, x: &T) -> Option<u64>` | `slice::binary_search` | a miss is `None`, not `Err(at)`; same instance set as `contains` |

## Not here

A fixed length admits no `sort`, `push`, `pop` or `resize`, and none is
declared.

`array::is_sorted` and `array::to_vec` are absent because they cannot be
registered, and the cause is known. `Externs::combine` gives an instance
its type through `instance_at_first_var`
(`acvus-extern/src/registry.rs:594`), which walks the signature looking for
`PolyTy::Var(0)`; it has arms for `Var`, `Ref`, `UserDefined` and `Tuple`,
and **none for `PolyTy::Array`**, which is what `Arr<T, N>` lowers to
(`acvus-extern/src/len.rs:65`). A signature whose only mention of the
element type sits inside the array therefore yields no instance type and is
refused with "`array::…` does not have the type of `array::…`".

`array::contains` and `array::binary_search` register only because their
second parameter, `x: &T`, mentions the element type outside the array, so
`find_map` skips the array parameter and settles on that one. Measured one
variable apart: two signatures identical but for a second `x: &T`
parameter — the one with it registers, the one without it is refused.
`vec::is_sorted` and `vec::to_vec` are unaffected because `Vec<T>` is a
`PolyTy::UserDefined`, which that walk handles.

The fix is one arm in `instance_at_first_var`, in `acvus-extern`. Until
then an array is ordered by `vec(a).is_sorted()` and copied by `vec(a)`,
which is the declared cast (RFC-0043) and Rust's `Vec::from(a)`.

`iter`, `first`, `last` and the rest of the read-only surface are reachable
on the view: `a.as_slice()` and then `docs/std/vec.md`'s `slice` table.
