# values

## 1. `Value`

A `Value` is the runtime's value. It is `Copy`, laid out as the runtime chooses, and relied on by nothing outside the runtime. It does not carry its type, and how it holds a value, or names one it references, is the runtime's. Code that knows a value's type never reads its `Value` to learn how it is held: the type's kind decides, so the choice is a constant. Only the runtime's operations that take no type, its release and its making of a reference value (runtime §2), read a `Value` without its type.

## 2. `Uniform`

`Uniform<R, X>` is a value the runtime holds, its type kept by the marker `X`. `Uniform::new` requires `X: 'static`, so a `Uniform` holds a value made as `X` (variables §2), and its projections and its drop are safe. It is projected to `&` or `&mut` only through the runtime, one layer at a time (references §3), and never taken out, though a `Copy` value may be got. It is moved, popped and dropped whole, and its drop releases its value. It is invariant in `X`, as every type constructor is (compiler: types §4). Its layout is the `Value`'s for every `X`, with no niche of `X`, which would set `Option<Uniform<R, X>>` apart from one `X` to another.

A `Uniform` stands where a uniform position needs a concrete type: `Stack<R, Uniform<R, String>>` is an unmarked `Stack<String>` seen uniformly (types §3).

## 3. Crossing by value

A value crosses by value through its type's own kind (runtime §2). It is given to the runtime, and taken back whole on the premise that it was given as that type, or as another instantiation of it (variables §3), and that the taker owns it. Every type of the intersection (boundary §2) crosses so, and each argument or field is taken back by its own kind, never by its name. A value is taken back as its type is written: a part written as a `Uniform` comes back as that `Uniform`, its value still held, so `Option<(i64, String)>` is taken back whole, and `Option<Uniform<R, (i64, String)>>` leaves its payload held.
