# references

## 1. `Ref` and `Mut`

`Ref<'a, R, X>` and `Mut<'a, R, X>` are the shared and exclusive references to one value. Each holds a reference value, a `Value` the runtime makes from the place it borrows, by `value_as_ref` from a shared borrow and by `value_as_mut` from an exclusive one; how a reference value names its referent is the runtime's. An owned value gives both, by `as_ref` and `as_mut`; a `Mut` gives both, reborrowing; a `Ref` gives `as_ref` alone, and is `Copy`. A `Mut` is not `Copy`, and a store or a replace through it changes the value in its place, so nothing is written back after a call. Neither has a drop, so each ends its borrow at its last use. Each is invariant in `X` and covariant in `'a`, as `&'a X` and `&'a mut X` are. A Rust `&T` or `&mut T` to its referent is lent to a closure by the concrete type `T`'s kind, so no Rust reference leaves it. A reference to a reference is that reference (compiler: types §4).

A reference value rests on the borrow it is made from: its place is not moved, and is not changed while a `Ref` lives or reached otherwise while a `Mut` does. Rust's borrow keeps that where the place is Rust's, and the compiler's loan where it is acvus code's; the runtime relies on nothing more.

`Ref` and `Mut` are types of their own, not a mode of `Uniform`: a `Uniform` releases its value on drop, and a shared reference copies (compiler: ownership §1), which no type with a drop can.

## 2. Inputs and outputs

Inputs are values or views; outputs are owned values or references. A reference to one value never crosses as a Rust reference: it crosses as a `Ref` or a `Mut`, made by `as_ref` or `as_mut`. A run crosses as a Rust slice and text as a Rust `str`, each taken apart into its data pointer and length and rebuilt through std, so no fat pointer's layout is relied on. Giving out a Rust reference costs one load at a place the signature fixes, and is no dynamic conversion. A place a reference names holds a `Value`, so native data is not lent as one value, and a `#` value, which holds no run of handles, is demoted where it is indexed or iterated by reference (compiler: types §8).

## 3. Structures

A reference to a structure is the structure of its parts' references (compiler: ownership §5), and is lent as what it is. An object or a tuple is lent as a structure whose parts are `Ref`s or `Mut`s, which a declaration writes as such, a structural type (types §2) or a tuple, and which crosses by value as any structure does. An enum is lent shared as the enum of its payload's `Ref`s; lent exclusively, it is a `Mut` of the enum value, whose `replace` writes the variant and the payload together, and the enum of its payload's `Mut`s reaches the payload alone. An option and a result are held in Rust's own shapes, so a view of one is a Rust reference to that shape, its payload uniform; an exclusive one reaches the variant's place, and `Option<&mut T>` the payload alone. No object, enum or tuple is lent as a Rust value: of the values a runtime holds, only an option, a result and an opaque value are.

## 4. A reference held uniformly

A reference can be a type variable's value. A container of such values seen concretely must then differ from its generic view only in a marker (variables §4), and a `Ref` differs from a `Uniform` as a type, so a reference so held is a `Uniform` under a reference marker. `Uniform<R, Shared<'a, X>>` holds a `&'a X`, made from a `Ref`, and `Uniform<R, Exclusive<'a, X>>` a `&'a mut X`, made from a `Mut`. The markers are never values, and each names its loan's lifetime, so a declaration shows the loan and nothing a `Uniform` holds is out of its sight. The `Value` held is a reference's, so its drop ends no owner, and a reference to a type variable's value is held without wrapping the value twice.

A held reference costs what `Uniform`'s invariance costs: it is invariant in `'a`, so a body that mixes lifetimes names each; and it holds its borrow until it drops, its referent outliving its container. Both bind bodies, never soundness.

*Rejected.* A reference marker without its lifetime: an owned container of references would hold a loan its declaration does not show. A reference to an object as the object of its parts' references: the view's type would hold its lifetime inside, which a view lent to a closure carries only through a lifetime-generic associated type. Assembling a Rust value where an object is lent and taking it apart after: a write-back after the call, and a conversion at every lend.

## 5. Examples

These are examples only, not part of the contract: two ways a runtime may make a reference value (§1), one with no `unsafe` and one with it.

A runtime with no `unsafe` holds every value in a slot it checks, and a `Value` is the slot's index. A reference value is the index marked shared or exclusive: a view through it borrows the slot after checking the mark, a store through a `Mut` replaces what the slot holds, and a release of it ends nothing. A misuse, a write through a shared mark or a view of a released slot, panics.

A runtime with `unsafe` holds a scalar in the `Value` itself and every other value behind a pointer to storage that does not move while a reference to it lives. A `Ref` is the `Value` its place holds, an owning one turned borrowed, so no release ends an owner through it; a `Mut` is the address of the place, read back as a Rust reference, and a store through it writes the place. Both rest on the borrow they are made from (§1), which this runtime does not check.
