# operators and numbers

## 1. Numbers

The numbers are the integer types of 8 to 64 bits and `f64`. An unsuffixed integer literal defaults as types §5 states. Integer overflow is undefined. `f64` follows IEEE: division by zero gives an infinity, not a trap.

## 2. Operators are core signatures

`core` is declared inside the compiler and implements nothing. Its signatures mean something only through the extern fns that give them instances.

Each operator is a call of a `core` signature: `+` of `core::add`, `-` of `core::sub`, `*` of `core::mul`, `/` of `core::div`, `%` of `core::rem`, unary `-` of `core::neg`, `==` of `core::eq`, the orderings of `core::cmp`. The operands' types decide which instance answers. Where no instance answers, the program is refused. An extern type joins an operator by declaring an instance.

`core`'s signatures are Rust's, but for two:

- `clone<T>(&T) -> T`, `eq<T>(&T, &T) -> bool`.
- `cmp<T>(&T, &T) -> i64`, answering `-1`, `0` or `1`.
- `add`, `sub`, `mul`, `div`, `rem`: `<T, O>(&T, &T) -> O`, and `neg<T, O>(&T) -> O`; the instance chooses `O`.
- `display<T>(&T, out: &mut String)`, appending to `out`.
- `as_slice<C, T>(&C) -> &[T]`, `as_slice_mut<C, T>(&mut C) -> &mut [T]`, `as_str<S>(&S) -> &str`: the views.

`&&`, `||` and `!` on `bool`, `^` on `bool` and the integers, and `&`, `|`, `<<` and `>>` on the integers are the language's own operations, as Rust defines them, with overflow as §1 states. They are not calls.

## 3. Slices

`a[i]` reaches an element through its container's `core::as_slice`, or `core::as_slice_mut` where the place is written or lent mutably. A `for` over `&v` or `&mut v` reaches its elements through the same instance. Any type that gives these instances is indexed and iterated as a slice.

## 4. Equality and ordering

Equality and ordering mean what they mean in Rust, with one exception: `NaN == NaN` is true. A structural type compares part by part where its parts compare; any other type compares through its `core::eq` and `core::cmp` instances.

## 5. Casts and coercion

`as` casts between the numbers, `bool` and `char` as Rust casts them.

A coercion is applied without being written. A reference to a container reads as its view: `&Vec<T>` as `&[T]`, `&String` as `&str`. A declared extern cast coerces as types §3 and §7 state.

## 6. Deref

A read of `*r` is a take: it takes the value `r` names, by the ownership rules. `*r = v` stores `v` into what `r` names, and `r` must be a `&mut`.

## 7. Display and clone

`core::display` formats a value inside `{{ }}`.

`core::clone` is the compiler-level means to clone any extern type, generally.
