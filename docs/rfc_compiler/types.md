# types

## 1. Transparency

A type is widened, joined or unioned only where the compiler sees every use of it. A type defined outside the program, by an extern fn or by the host, is not transparent: it equals only itself and crosses exactly as declared. A type parameter of an extern fn is parametric, so a program's type passes through it transparently.

## 2. Join

Two object types join at the union of their fields. Two enums of one name join at the union of their variants; enums of two names do not join. Two function types join at the join of their effects, which form a lattice, so a joined function is never assumed to do less than either side; laws, `ensures` and `means` do not join, and a joined function carries none.

## 3. Subtyping

A declared extern cast from `S` to `T` makes `S` a subtype of `T`. Where values of different types meet in one slot, the slot takes their least upper bound, and each value is coerced to it by the declared cast. Two `Deque`s of different identities meet at `Deque` with the identity demoted; values of one identity keep it.

Type constructors stay invariant: subtyping applies to a value entering a slot, not to a constructor's argument.

## 4. Bottom and invariance

`!` is the bottom type, and every type constructor is invariant. A product with a part of type `!` is `!`.

## 5. Defaults

A type variable nothing constrains is `!`. A variable a bound constrains and no use answers is refused. An unsuffixed integer literal is `i64` where no use decides its width.

## 6. Deciding by type

What a `for` iterates, what `?` unwraps, and which declaration a name or a method call reaches are decided by a type, wherever in the program that type becomes known. The outcome does not depend on the order the program is checked in. A construct whose type never becomes known is refused.

## 7. Conversions

Where a value's type is not the type its slot asks for and no subtyping relates them, the value converts only by a declared conversion, and exactly one must apply. Otherwise the program is refused.
