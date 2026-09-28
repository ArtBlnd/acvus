# types

## 1. Transparency

A type is widened, joined or unioned only where the compiler sees every use of it. A type defined outside the program, by an extern fn or by the host, is not transparent: it equals only itself and crosses exactly as declared. A type parameter of an extern fn is parametric, so a program's type passes through it transparently.

## 2. Join

Two object types join at the union of their fields. Two enums of one name join at the union of their variants; enums of two names do not join. A variant is written with its enum's name, except `Some`, `None`, `Ok` and `Err`, the variants of `core`'s `Option` and `Result`. Two function types join at the join of their effects, which form a lattice, so a joined function is never assumed to do less than either side; laws, `ensures` and `means` do not join, and a joined function carries none.

## 3. Subtyping

A declared extern cast from `S` to `T` makes `S` a subtype of `T`. The declared casts and the views (operators §5) form a tree: every type has at most one next step up, so every order they give is total. Where values of different types meet in one slot, the slot takes their nearest common ancestor, and each value is coerced to it. Values of one identity keep it; two identities meet at the type without identity, as two `Deque`s meet at `Deque`.

Type constructors stay invariant: subtyping applies to a value entering a slot, not to a constructor's argument.

## 4. Bottom and invariance

`!` is the bottom type, and every type constructor is invariant. A product with a part of type `!` is `!`.

## 5. Defaults

Defaults apply only where nothing more can be decided. A type variable nothing constrains is `!`, and an effect nothing constrains is opaque. A variable a bound constrains and no use answers is refused. An unsuffixed integer literal is `i64` where no use decides its width.

## 6. Deciding by type

What a `for` iterates, what `?` unwraps, and which declaration a name or a method call reaches are decided by a type, wherever in the program that type becomes known. A decision reads a type only once nothing can change it, and what the decision leads back into that type must fit it unchanged. The outcome does not depend on the order the program is checked in. A construct whose type never becomes known is refused.

A call reaches by its arguments in order, and a tuple by its parts in order: at each one, stepping up from its exact type, the declarations that match at the first step where any match are kept. Exactly one must be left; otherwise the call is refused.

## 7. Conversions

Where a value's type is not the type its slot asks for and no subtyping relates them, the value converts only by a declared conversion, and exactly one must apply. Otherwise the program is refused.
