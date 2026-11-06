# operators and numbers

## 1. Numbers

The numbers are the integer types, those of 8 to 64 bits and `usize`, and the float types `f32` and `f64` ([the type system](../transparency-constraint-type-system.md), Types). `usize` is an integer type of its own, whose width the host gives: it is a different type from an integer of that width, and no coercion relates the two. An unsuffixed integer literal defaults as types §5 states. Integer overflow is undefined. Arithmetic that checks, wraps, saturates or reports overflow is written as the extern fns that state it, with Rust's meanings. Integer division or remainder by zero traps; a later build flag may choose otherwise. `f32` and `f64` follow IEEE: division by zero gives an infinity, not a trap.

A string literal is a `&str`, as in Rust.

## 2. Operators are lang items

`core` is declared inside the compiler and implements nothing. Its generic functions are the lang items, the ones the language's syntax lowers to; they mean something only through the extern fns that impl them.

Each operator is a call of a lang item: `+` of `core::add`, `-` of `core::sub`, `*` of `core::mul`, `/` of `core::div`, `%` of `core::rem`, unary `-` of `core::neg`, `==` of `core::eq`, the orderings of `core::cmp`. The call is a use whose candidates are the lang item's impls, decided as types §6 states: as the one impl that what is decided of its operands and its result reaches ([the type system](../transparency-constraint-type-system.md), O15). Where no impl answers, the program is refused. An extern type joins an operator by declaring an impl.

The lang items are Rust's, but for two:

- `clone<T>(&T) -> T`, `eq<T>(&T, &T) -> bool`.
- `default<T>() -> T`: the value a type starts from, as Rust's `Default`. `Option`'s is `None`; `Result` has none.
- `cmp<T>(&T, &T) -> i64`, answering `-1`, `0` or `1`.
- `add`, `sub`, `mul`, `div`, `rem`: `<T, O>(&T, &T) -> O`, and `neg<T, O>(&T) -> O`; the impl chooses `O`.
- `display<T>(&T, out: &mut String)`, appending to `out`.
- `as_slice<C, T>(&C) -> &[T]`, `as_slice_mut<C, T>(&mut C) -> &mut [T]`, `as_str<S>(&S) -> &str`: the views.
- `unwrap<W, T>(W) -> T`, at `Option<T>` and `Result<T, E>`. Its meaning is the language's, since both types are primitive: `Some(x)` and `Ok(x)` give `x`, and `None` and `Err(_)` trap.
- `as_ref`, `as_mut`: a `&S` to `S` with each part lent `&`, a `&mut S` to `S` with each part lent `&mut`, one level, for a transparent type `S` (types §1): `(&(a, b)).as_ref()` is `(&a, &b)`, `(&x).as_ref()` of an `Option<T>` is an `Option<&T>`. Their meaning is the language's (ownership §5): they are used, and no declaration gives them impls.

`&&`, `||` and `!` on `bool`, `^` on `bool` and the integers, and `&`, `<<` and `>>` on the integers are the language's own operations, as Rust defines them, with overflow as §1 states. They are not calls. `|` is the pipe (`x | f` is `f(x)`); bitwise or has no operator.

## 3. Slices

A value is a sequence of atom `X` when its parts have positions in a total order of their own, and each part corresponds one to one with a value of `X`, so that its parts in that order are a slice of `X`. An array and a slice are sequences of their elements. A type whose positions have no order of their own is no sequence, an `Option` among them; nor is text, whose atom is a measure's (its bytes, its characters or more), nor a map, whose order and atom a measure chooses (its keys, its values, its entries).

A type gives `core::as_slice` and `core::as_slice_mut` only where it is a sequence, and each gives that slice. What such an impl states is trusted, as a declared fact is (extern fn §3): an impl that gives another slice is its declarer's error.

`a[i]` reaches an element through its container's `core::as_slice`, or `core::as_slice_mut` where the place is written or lent mutably. A `for` over `&v` or `&mut v` reaches its elements through the same impl. Any type that gives these impls is indexed and iterated as a slice. The class of `i` in `a[i]` has the upper bound `usize` ([the type system](../transparency-constraint-type-system.md), Structural constraints). An index outside its container traps.

A sequence's length, `core::len`, is the length of that slice. No other value has a length.

## 4. Equality and ordering

Equality and ordering mean what they mean in Rust, with one exception: `NaN == NaN` is true. A transparent type (types §1) compares part by part where its parts compare. An enum, `Option` and `Result` have equality but no order: which variant comes first is not defined, so ordering one is refused. Any other type compares through its `core::eq` and `core::cmp` impls.

## 5. Casts and coercion

`as` casts between the numbers, `bool` and `char` as Rust casts them.

A coercion is applied without being written; the coercions are the language's (types §3). A reference to an extern-defined container reaches its view, `&Vec<T>` to `&[T]`, by the conversion its `core::as_slice` impl is, made at a boundary alone (types §7; O10). A built-in container has no such view: `&String` reads as `&str` only through `as_str()`.

## 6. Deref

A read of `*r` is a take: it takes the value `r` names, by the ownership rules. `*r = v` stores `v` into what `r` names, and `r` must be a `&mut`.

## 7. Display and clone

`core::display` formats a value inside `{{ }}`.

`core::clone` is the compiler-level means to clone any extern type, generally.
