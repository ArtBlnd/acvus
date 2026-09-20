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
| `is_sorted` | `is_sorted(c: &Array<T, N>) -> bool` | `slice::is_sorted` | same instance set as `contains` |
| `to_vec` | `to_vec(c: &Array<T, N>) -> Vec<T>` | `slice::to_vec` | **waits on `vec::to_vec`'s escaping iterator** — see below. `vec(a)` is the declared cast and answers the same vec |

## Not here

A fixed length admits no `sort`, `push`, `pop` or `resize`, and none is
declared.

`array::to_vec` registers — `instance_at_first_var` now descends an array —
but it cannot be kept, and the defect is `vec::to_vec`'s body rather than
anything an array does. That body is
`elements(run).cloned().collect()`, whose `Cloned<Map<Range, closure>>` is a
stack local the address of which reaches `Vec::<String>::from_iter`; with
one caller LLVM inlines `from_iter` and nothing escapes, and with a second
caller — `array::to_vec`, which shares `vec::elements`'s closure and so the
same `from_iter` monomorphization — it outlines it, and LLVM's sibling-call
rule then refuses a tail call out of every `Op::run` that reaches the
handler. Measured one variable apart with `cargo bench --bench asm_probe`:
with `array::to_vec` registered, five bodies land with a call
(`CallExtern1` x3 and `CallWindow` x2 over `vec::__extern_fn_to_vec_str`);
with it alone unregistered, the probe is green at 4873 / 49 / 16 / 16.
Naming the copy in one shared `pub(crate)` function did not move it — LLVM
inlines that function into both operations too. The row returns when
`vec::to_vec` fills its result without an intermediate iterator object;
that is a change to `vec::to_vec`, not to this module.

`iter`, `first`, `last` and the rest of the read-only surface are reachable
on the view: `a.as_slice()` and then `docs/std/vec.md`'s `slice` table.
