# references

## 1. `Ref` and `Mut`

`Ref<'a, R, X>` and `Mut<'a, R, X>` are the shared and exclusive references to one value. Each holds a reference value, a `Value` the runtime makes from the place it borrows (runtime §2), and how a reference value names its referent is the runtime's. An owned value gives both, by `as_ref` and `as_mut`; a `Mut` gives both, reborrowing; a `Ref` gives `as_ref` alone. A `Ref` is `Copy`. A `Mut` is not, and a store or a replace through it changes the value in its place, so nothing is written back after a call. Neither has a drop, so each ends its borrow at its last use. Each is invariant in `X`, as every type constructor is (compiler: types §4), and covariant in `'a`, as `&'a X` and `&'a mut X` are. A reference to a reference is that reference (compiler: types §4): a shared one as it is, and an exclusive one reborrowed for the loan it is reached through, shared or exclusive as that loan is.

A reference value rests on the borrow it is made from: its place is not moved, nor changed while a `Ref` lives, nor reached otherwise while a `Mut` does. Rust's borrow keeps that where the place is Rust's, and the compiler's loan where it is acvus code's; the runtime relies on nothing more.

`Ref` and `Mut` are types of their own, not a mode of `Uniform`: a `Uniform` releases its value on drop, and a shared reference copies (compiler: ownership §1), which no type with a drop can.

## 2. Inputs and outputs

An input or an output is a value, a reference to one value, or a view of a run, the elements a slice holds in order (compiler: operators §3), or of a text. A reference to one value never crosses as a Rust reference: it crosses as a `Ref` or a `Mut`, made by `as_ref` or `as_mut`, and is opened by projection alone (§3). A view crosses as a Rust slice or `str`, taken apart into its data pointer and length and rebuilt through std, so no fat pointer's layout is relied on. Giving out a Rust reference costs one load at a place the signature fixes, and is no dynamic conversion.

A place a reference names holds a `Value`, and so does each element of a run, so native data is lent neither as one value nor as a run, and a run's element type is a variable or a `Uniform`. A `#` value holds no run of `Value`s, so a use that indexes or iterates it by reference takes it unmarked (compiler: types §8).

## 3. Projections

A reference is opened by projection, one layer at a time: its referent's kind projects it and lends the projection to a closure (runtime §2, §4). The layer's parts stay uniform, each projected on its own, so a body opens only what it reads, and no Rust reference to one value leaves the projection. A reference to a structure is the structure of its parts' references (compiler: ownership §5), and is projected as what it is. A part that is itself a reference is lent as that reference (§1). An exclusive projection reaches every part's place, so a store into it writes an enum's or an option's variant and payload together.

## 4. A reference held uniformly

A reference can be a type variable's value. A container of such values seen concretely must then differ from its generic instantiation only in a marker (variables §4), and a `Ref` differs from a `Uniform` as a type, so a reference so held is a `Uniform` under a reference marker: `Uniform<R, Shared<'a, X>>` holds a `&'a X`, made from a `Ref`, and `Uniform<R, Exclusive<'a, X>>` a `&'a mut X`, made from a `Mut`. A view is held so too, its referent unsized: `Uniform<R, Shared<'a, str>>` holds a `&'a str`, and a `Shared` or an `Exclusive` of `[T]` a view of a run, each giving the view back as a view crosses (§2). The markers are never values. Each names its loan's lifetime, so a declaration shows the loan, and nothing a `Uniform` holds is out of its sight. The `Value` held is a reference's, so its drop ends no owner, and a reference to a type variable's value is held without wrapping the value twice.

A held reference costs what `Uniform`'s invariance costs. It is invariant in `'a`, so a body that mixes lifetimes names each, and it holds its borrow until it drops, its referent outliving its container. Both bind bodies, never soundness.

## 5. Examples

These are examples only, not part of the contract: two ways a runtime may make a reference value (§1), one in a checked runtime and one in an optimized one (runtime §3).

A checked runtime holds every value in a slot it checks, and a `Value` is the slot's index. A reference value is the index marked shared or exclusive: a projection through it borrows the slot after checking the mark, a store through a `Mut` replaces what the slot holds, and its release ends nothing. A misuse, a write through a shared mark or a projection of a released slot, panics.

An optimized runtime holds a scalar in the `Value` itself, and every other value behind a pointer to storage that does not move while a reference to it lives. A `Ref` is the `Value` its place holds, an owning one turned borrowed, so no release ends an owner through it; a `Mut` is the address of the place, read back as a Rust reference, and a store through it writes the place. Both rest on the borrow they are made from (§1), which this runtime does not check.
