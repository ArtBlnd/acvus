# values

## 1. `Value`

A `Value` is the runtime's value: `Copy`, laid out as its runtime chooses, and relied on by nothing outside it. It does not carry its type, and how it holds a value, or names one it references, is the runtime's. Only the runtime's operations that take no type, its release and its making of a reference value (references §1), read a `Value` without its type. Code that knows a value's type never reads its `Value` to learn how it is held: its type's kind decides, so a read chooses by a constant.

## 2. `Uniform`

`Uniform<R, X>` is a value the runtime holds, its type kept by the marker `X`. It is made by `Uniform::new`, which requires `X: 'static`, so it holds a value made as `X`, and its views and its drop are safe. It is projected to `&` or `&mut` only through the runtime and never taken out; a `Copy` value may be got. It is moved, popped and dropped whole, its drop releasing its value. It is invariant in `X`, as every type constructor is (compiler: types §4). Its layout is the `Value`'s for every `X`, with no niche of `X`, which would set `Option<Uniform<R, X>>` apart from one `X` to another.

A `Uniform` stands where a concrete type is needed in a uniform position: `Stack<Uniform<R, String>>` is an unmarked `Stack<String>` seen uniformly (types §3).

## 3. Crossing by value

A value crosses by value by its type's own kind: it is given to the runtime, and taken back whole on the premise that it was given as that type and that the taker owns it. Every type of the intersection crosses so, and an argument or a field is opened by its own kind, never by its name.
