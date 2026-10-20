# Experiment: the runtime contract as kinds, nested types as their layers composed

The evidence for runtime §2–§4 read as kinds: one associated type per kind,
each with operations of its own, and a kind that holds other values dealing
in their `Value`s alone, so that a nested type crosses as its layers' kinds
composed, the innermost first. `Option<(A, B)>` is the parts' values tied by
the tuple kind into one value, which the option kind holds;
`Option<Option<T>>` is one option kind's value held by another.

A value not yet opened is held in one of two forms: owned, a `Uniform`,
which releases its value on drop, or borrowed, a `Ref` or a `Mut`, a
reference value the runtime makes from a place, which has no drop, so that a
`Ref` copies (references §1). A value opens one layer at a time through its
type's `Open`, its parts left unopened: an option's or a result's lent in
Rust's shape with its payload uniform, a tuple's as its parts' references
(references §3, runtime §2). By value, a type crosses as deep as it is written, and a
`Uniform` written in it stops there.

`contract.rs` is the contract, `boundary.rs` the extern side (`Uniform`,
`Ref`, `Mut`, the sealed `TypeVar`, `Cross`, `Uniformly`, `Open`), and
`slots.rs` and `boxes.rs` runtimes over the one contract:

- `slots`: checked; every value in a slot, a reference value the slot's index marked shared or exclusive, a misuse a panic (references §5, first example). A view is no slot's: its start and length, marked as a text, a run or an exclusive run, the mark checked on use.
- `boxes`: unchecked; a scalar in the `Value`, every other value behind a pointer to a box, a shared reference value the place's value turned borrowed, an exclusive one the address of the place (references §5, second example). A `Value` is two words, sixteen bytes. A view is its fat pointer as it is, its start and its length, whose top bit is clear, since Rust keeps every slice within `isize::MAX` bytes; every other form sets that bit and states itself in the rest of the second word, which only the operations that take no type read (values §1). Where a box is placed is its own: each on the heap (`Heap`), or one after another in a frame reused batch after batch (`Frame`). So is whether a layer is held in its `Value` when it fits: with `Chained`, an option over a scalar, or over such an option, is a count of `Some` layers over a scalar or over `None`, and its projection lends the option it decodes, an exclusive one encoding it back.

An opaque value is put in, taken out and projected to `&` or `&mut`, as a
`Box<dyn Any>` is, but held untyped and read by layout, since one value is
read at its generic and its concrete instantiations. A declared object or enum is
held the same way, as its Rust type lays it out, with two differences:
every field is a `Uniform` or a variable, so a `Value`, and its derive states
a table (`laid.rs`) of each field's name and offset, read off the type by
`offset_of!`, by which the runtime reads and writes the fields. A field is
found by its name, so no order of the fields is assumed. An enum's author
writes `repr(u32)`, so the enum stores a `u32` tag at its start and each
variant's fields where a `repr(C)` struct of the tag and those fields has
them; the table names each variant with its tag. A value of a declaration's
type is made by an extern, or by the caller where it coerces a value of its
own into that type (compiler types §1, §3), by the table; the extern side
reads one as its Rust type.

A function value is acvus's alone, which the runtime holds as a program
over `Value`s (`Programs`); the boundary calls one through a shared
reference, an exclusive one, or by value, which consumes it, and never makes
or gives one back (functions). A closure parameter is what an entry writes:
a Rust closure over a reference to the function value, its arguments given
as values by their own kinds, a `&T` as a `Ref` and a `&mut T` as a `Mut`,
and its result taken back by its kind. A kept function is what
`#[fn_type(mode)]` would expand to: a marker carrying the effect variable,
and a projection in the author's crate calling it as its mode states.
An asynchronous call is orthogonal to the mode: the function kind's
`call_async_ref`, `call_async_mut` and `call_async_once` call through a
shared reference, an exclusive one, or by value, each a future that names no
lifetime and runs the call as it is polled. The scope that lends the
function and its arguments to the call, and gives back with its result what
the call did not consume, is the extern side's (`scope`, `scope_mut`,
`scope_once`, `Lend`), the loan a parameter of `Lend`, never of an
associated type. A by-value call's future owns the function until it ends
or is dropped; a callable an owning scope gave and no call took is released
with it.

A view of a text or a run is a reference value made from a Rust `&str`,
`&[T]` or `&mut [T]`, and taken back as one at the lifetime its taker
names, an entry its own (declarations §2); a run's elements are `Value`s,
so `T` is a variable or a `Uniform` (k12). It crosses by value by its own
kind, so it nests as any type does: an `Option<(&str, &str)>` result is the
views' values tied by the tuple kind and held by the option kind, and a
release ends nothing for them. A `&String` is lent by the string kind, so
`core::as_str` at `String`, the caller's coercion of a `&String` into a
`&str` (compiler: operators §2, §5), is an extern fn as any other: its
entry lends the string and gives the text back as a view into it, which
outlives the lending.

The kinds and their operations are the probe's means, not decided
primitives.

| Part | What it checks | Run |
|---|---|---|
| `behavior.rs` | on four runtimes, `slots` and `boxes` on the heap unchained, on the heap chained and in a frame chained: nested values round trip and leave nothing held; generic code crosses through its variables, one value read at the generic and a concrete instantiation; a value opens one layer at a time; an exclusive reference changes values in place; a reference owns nothing; an opaque value is read at each instantiation; an option held in its `Value` moves to a box and back as its payload changes; a declared object and enum the extern made are read and written by the runtime through their tables; the caller makes one by the table, fields named in no order, and the extern reads it; an object is read at the generic and a concrete instantiation; an enum changes its variant in place; closure parameters are called by their bounds, `Fn`, `FnMut` and `FnOnce`; generic code calls a closure over its variable; kept functions are called as their modes state, one of them kept in an extern-defined type that goes back to acvus and is called later; an asynchronous kept function is called in a scope through a shared reference, an exclusive one or by value: the shared and the exclusive scope give the function back with the arguments and the result, an exclusive call's change to the function seen by the next, and the owning scope gives back the arguments and the result alone, the function released when the call ends; a scope dropped while its call is suspended, or before its call is made, abandons the call and releases what it holds; an exclusive reborrow of a variable whose value is a shared reference gives that shared reference; `core::as_str` at `String` gives a view into the string it is lent, and externs return views borrowing their text or run argument, alone and nested in an option of a tuple, at the generic instantiation and read at a concrete one, shared and exclusive; a view's release, and that of a value holding views, ends nothing for them | `cargo test --manifest-path docs/rfc_extern/experiments/kind-composition/Cargo.toml` |
| the same, under Miri | no undefined behaviour and no leak on these runs | `cargo +nightly miri test --manifest-path docs/rfc_extern/experiments/kind-composition/Cargo.toml` |
| `probes/` | what rustc admits and refuses, each probe's first line stating the expected outcome; then a check that `boundary.rs` has no branch, forced unwrap, panic or cast | `docs/rfc_extern/experiments/kind-composition/run_probes.sh` |
| `bench.rs` | what opening a nested value costs, by projection and by take, beside native Rust, with layers in boxes or in their `Value`; each pattern a ring of 1024 values whose scalars name the next value to open | `cargo run --release --no-default-features --bin bench --manifest-path docs/rfc_extern/experiments/kind-composition/Cargo.toml` |

| Probe | What it checks | Expected |
|---|---|---|
| `k01_type_beside_every_variable.rs` | an author's type with its own crossing beside the impl over every variable, nested in an option and a tuple | compiles |
| `k02_generic_nested.rs` | generic code giving and taking back nestings of its variables beside concrete parts | compiles |
| `k03_projection_payload_not_uniform.rs` | a projection of an option whose payload is written as a tuple | E0277 |
| `k04_variable_outside_the_seal.rs` | an author's type made a `TypeVar` | E0277 |
| `k05_parts_are_given_up_unsafely.rs` | a tuple made from part values outside `unsafe` | E0133 |
| `k06_variable_does_not_open.rs` | generic code opening a variable's value | E0277 |
| `k07_shared_gives_no_exclusive.rs` | an exclusive reference made from a `Ref` | E0599 |
| `k08_part_outlives_its_exclusive_projection.rs` | a part's exclusive reference, reborrowed, kept past the projection of its tuple | E0521 |
| `k09_author_drop_on_a_structure.rs` | an author's `Drop` beside the empty one the derive writes for a declared structure | E0119 |
| `k10_fields_not_moved_out.rs` | a declared structure's fields moved out of it, which its `Drop` forbids | E0509 |
| `k11_function_value_given_back.rs` | a kept function value given back to acvus, which its type admits | compiles |
| `k12_native_run_is_no_view.rs` | a run of `i64` given as a view | E0277 |
| `k13_scope_loan_escapes.rs` | a scope's call giving its loan out as its result | lifetime may not live long enough |
| `k14_outer_loan_into_inner_call.rs` | an inner scope's call given the outer scope's loan | lifetime may not live long enough |
| `k15_scope_over_a_variable.rs` | a scope over a type variable's value, which carries no `'static` | compiles |
| `k16_exclusive_callable_escapes.rs` | an exclusive scope's callable given out as its call's result | lifetime may not live long enough |
| `k17_owned_callable_escapes.rs` | an owning scope's callable given out as its call's result | lifetime may not live long enough |
| `k18_owned_function_not_given_back.rs` | a function used after an owning scope consumed it | E0382 |

Every build output goes to `target/` here, which git ignores. The feature
`count`, on by default, makes the box runtime count its boxes for the tests'
leak checks; the bench runs without it.

## Result, 2026-10-06

With stable rustc 1.97.1, and Miri on nightly 1.99.0 (2026-07-20):

- The ninety-six behaviour tests, twenty-four on each of the four runtimes, pass, natively and under Miri, with no leak.
- The eighteen probes are as expected. k03 is refused at `(i64, i64): TypeVar<R>`, k04 at the seal, k05 at the call of `Tuples::give`, k06 at `T: Open<'_, R>`, k07 for want of `Ref::as_mut`, k08 as borrowed data escaping the closure, k09 as conflicting `Drop` impls, k10 as a move out of a type with a `Drop`, k12 at `&[i64]: Cross<R>`, whose impl for a run asks `T: TypeVar<R>`, k13 at the loan returned from the call's closure, k14 at the outer loan the inner call is given, k16 and k17 at the callable returned from the call's closure, k18 as a use of the moved function; k15 compiles. k11 compiles: a kept function value is a `Uniform` and crosses by the impl over every variable, so that an extern gives acvus no function value is kept where its declaration is registered, which the model does not do.
- `boundary.rs` has no branch, forced unwrap, panic or cast. The variants are told apart inside the option and result kinds, which take the payload's crossing as a bound, `Options::give<P: Cross<R>>`.

The bench, release build, on one machine, nanoseconds per value, the
frame's boxes. A box layer is an option, a result or a tuple held in a box;
with `Chained`, the option layers over a scalar are held in their `Value`.

| Pattern | box layers | project chain: native, in value, in boxes | take chain: native, in value, in boxes | round trip: in value, in boxes |
|---|---|---|---|---|
| `Option<i64>` | 1 | 1.07, 2.02, 2.38 | 1.10, 0.91, 2.20 | 0.76, 3.05 |
| `Option<Option<i64>>` | 2 | 1.09, 3.10, 3.96 | 1.11, 2.69, 5.31 | 1.28, 5.80 |
| `Option<Option<Option<i64>>>` | 3 | 1.22, 4.17, 4.86 | 1.11, 2.88, 7.68 | 1.91, 11.46 |
| `Option<(i64, i64)>` | 2 | 1.27, 4.30, 4.28 | 1.41, 4.11, 4.20 | 12.41, 12.47 |
| `Option<Option<(i64, i64)>>` | 3 | 1.43, 5.45, 5.50 | 1.44, 7.03, 7.02 | 9.26, 9.19 |
| `Result<Option<(i64, i64)>, i64>` | 3 | 1.47, 5.36, 5.35 | 1.54, 5.42, 5.33 | 10.42, 10.44 |
| `(i64, (i64, (i64, i64)))` | 3 | 3.02, 5.69, 5.69 | 2.00, 5.60, 5.62 | 22.42, 22.41 |

- A chain waits for each opening before the next, so it gives the latency of opening. In boxes, each box layer adds about 1 to 1.5 ns over native on a projection: one dependent load per layer, the payload's `Value` naming its content by pointer. A scalar payload sits in its `Value` and adds no load.
- Held in their `Value`, option layers over a scalar cost no load. Projected, each such layer costs about 1 ns over native, against about 1.3 ns in a box; taken by value, one layer costs what native does, and two or three about 1.6 to 1.8 ns over it.
- The patterns with a tuple have the same layers in either holding, since a tuple is always a box here, and cost the same in either.
- A `Value` here is two words, sixteen bytes, a view held as its fat pointer. Against the 24-byte `Value` of the run before: the project chains with a tuple, and those of options in boxes, cost the same or less, and those of two or three option layers in their `Value` 0.3 to 0.8 ns more; the take chains of nested options, in boxes or in their `Value`, cost 1.3 to 2.8 ns more, which was not examined further; the tuples' round trips cost about 10% less. These remain about twice those of the first version of this experiment, so their cost is not the `Value`'s size.
- These numbers are the model runtime's code generation as much as its layout. Its small helpers, called from the kinds' code monomorphized in the bench's crate, were inlined there only once marked `#[inline]`: unmarked, the one that holds a panic path was called, and every projection cost 4 to 10 times as much (12.86 ns for the project chain of `Option<i64>` in a value). A `value_as_ref` that followed a place recursively kept option layers held in their `Value` at 2 to 3 times this table's projection cost; it follows one step, since a place an exclusive reference names never holds another.
- Two runs of a bench like this differed by up to about 60% on some rows; there is no control for frequency scaling. The numbers show where the cost lies, not its size elsewhere.

## What it shows, and what it does not

- The checked runtime's `unsafe` blocks are those runtime §3 admits. In its safe kinds they are the option and result kinds' calls of the payload's `Cross::take` when taking a value back, and of `TypeVar::take` to make a projection's payload, each an `unsafe fn` whose premise passes through from the caller and which the slots then check on use. Its unsafe kinds, the opaque and view kinds, read by layout, unchecked.
- A view needs no mark of its own. The operations that know its kind read its `Value` as a fat pointer; only the release and the making of a reference value, which take no type, tell it from the other forms, by its length's top bit, which Rust leaves clear. The checked runtime holds it as a form of its own instead, and checks its mark on use.
- `core::as_str` at `String` gives a view into the text of the string its entry is lent, and the view outlives the lending because it names the string's own text. So the string kind lends the `String` where it is held: a runtime that held a string's text in its `Value` could lend only a decoded copy, which the view would outlive.
- An exclusive projection of an option held in its `Value`, or of an option or result in the checked runtime, moves the payload into the projection and back.
- The bench's values are hot in the first-level cache. A value whose layers are cold costs a miss per layer it does not share a cache line with; the frame places one value's layers next to each other, which this bench does not measure.
- A declared object's or enum's release releases its fields through its table, and nothing else of its Rust type runs, whoever made the value. So the derive writes an empty `Drop` for the type, and an author's own is refused (k09); the cost is that an extern body reads the fields in place and cannot move them out (k10).
- With `repr(u32)`, a declared enum stores its tag apart from its fields, so it takes no niche of them: by the `repr(C)` rule, the tag takes a word of its own before fields aligned to 8 bytes. This is derived from the layout rule, not measured here.
- A closure's argument `&T` crosses as a `Ref`, which is made from a place holding a `Value`, so `T` is a type variable or a `Uniform`: a `&i64` into the extern's own memory has no place to name. The closures here take `&Uniform<R, i64>` and `&T` over a variable.
- A scope's loan is a parameter of `Lend`, not of a generic associated type. A `Lent<'s>` that needs `where Self: 's`, as Rust requires of an associated type a method gives from `&'s mut self`, asks under the scope's `for<'s>` that the arguments outlive every loan, `'static`: k15, written so, is refused with E0310 at the scope over a variable's value, which carries no `'static` (variables §2). The async-scope experiment's `Lend` has that form; its a07 compiles a scope generic over its arguments but instantiates none at an argument that is not `'static`.
- The run a view names is a Rust `Vec` of `Uniform`s here, standing for the elements a container the runtime holds lays out.
- Not modelled: an array, held by the runtime, and its `core::as_slice`; a reference or a view held uniformly (`Uniform<R, Shared<'a, X>>`); a tuple held in its `Value`; and the interning of field names, which the model writes as `&'static str` where the compiler has `Astr`.
