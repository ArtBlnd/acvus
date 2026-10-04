# references

## 1. `Ref` and `Mut`

`Ref<'a, R, X>` and `Mut<'a, R, X>` are the shared and exclusive references to one value. A `Ref` is the `Value` its place holds, an owning one turned borrowed, so no release ends an owner through it; it is made once by `as_ref` and copied plainly after. A `Mut` is the address of the place, made by `as_mut`; it is not `Copy` and reborrows, and a store or a replace through it changes the `Value` in the place, so nothing is written back after a call. Neither has a drop, so each ends its borrow at its last use. Each is invariant in `X` and covariant in `'a`, as `&'a X` and `&'a mut X` are, and each is read only through the runtime, lending `&T` or `&mut T` to a closure, so no Rust reference leaves it. A reference to a reference is that reference (compiler: types §4).

A `Ref` rests on its place not changing while it lives and on the storage its `Value` names not moving; a `Mut` on its place not moving while it is lent.

`Ref` and `Mut` are types of their own, not a mode of `Uniform`: a `Uniform` releases its value on drop, and a shared reference copies (compiler: ownership §1), which no type with a drop can.

## 2. Inputs and outputs

Inputs are values or views; outputs are owned values or references. A reference to one value never crosses as a Rust reference: it crosses as a `Ref` or a `Mut`, made by `as_ref` or `as_mut`. A run crosses as a Rust slice and text as a Rust `str`, each taken apart into its data pointer and length and rebuilt through std, so no fat pointer's layout is relied on. Giving out a Rust reference costs one load at a place the signature fixes, and is no dynamic conversion. A place a reference names holds a `Value`, so native data is not lent as one value, and a `#` value, which holds no run of handles, is demoted where it is indexed or iterated by reference (compiler: types §8).

## 3. Structures

A reference to a structure is the structure of its parts' references (compiler: ownership §5), and is lent as what it is. An object or a tuple is lent as a structure whose parts are `Ref`s or `Mut`s, which a declaration writes as such, a structural type (types §2) or a tuple, and which crosses by value as any structure does. An enum is lent shared as the enum of its payload's `Ref`s; lent exclusively, it is a `Mut` of the enum value, whose `replace` writes the variant and the payload together, and the enum of its payload's `Mut`s reaches the payload alone. An option and a result are held in Rust's own shapes, so a view of one is a Rust reference to that shape, its payload uniform; an exclusive one reaches the variant's place, and `Option<&mut T>` the payload alone. No object, enum or tuple is lent as a Rust value: of the values a runtime holds, only an option, a result and an opaque value are.

## 4. A reference held uniformly

A reference can be a type variable's value. A container of such values seen concretely must then differ from its generic view only in a marker (variables §4), and a `Ref` differs from a `Uniform` as a type, so a reference so held is a `Uniform` under a reference marker. `Uniform<R, Shared<'a, X>>` holds a `&'a X`, made from a `Ref`, and `Uniform<R, Exclusive<'a, X>>` a `&'a mut X`, made from a `Mut`. The markers are never values, and each names its loan's lifetime, so a declaration shows the loan and nothing a `Uniform` holds is out of its sight. The `Value` held is a reference's, so its drop ends no owner, and a reference to a type variable's value is held without wrapping the value twice.

A held reference costs what `Uniform`'s invariance costs: it is invariant in `'a`, so a body that mixes lifetimes names each; and it holds its borrow until it drops, its referent outliving its container. Both bind bodies, never soundness.

*Rejected.* A reference marker without its lifetime: an owned container of references would hold a loan its declaration does not show. A `Ref` that copies the owning `Value` as it is: a generic drop of it would end its owner. A reference to an object as the object of its parts' references: the view's type would hold its lifetime inside, which a view lent to a closure carries only through a lifetime-generic associated type. Assembling a Rust value where an object is lent and taking it apart after: a write-back after the call, and a conversion at every lend.
