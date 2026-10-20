# impls

## 1. Generic functions

A generic function enters the compiler as one (compiler: extern fn §6), and an impl enters with the function and the choice it implements, which Rust checks against the function's signature. An impl written with `Uniform` is the one generic code reaches, and one written natively is its `#` member (types §3).

An impl whose signature is its generic function's, the variables only renamed, is the function's generic impl: there is at most one, and a call reaches it only where no other impl answers. Every other impl is concrete, an impl at a type constructor over variables among them. The two are told apart by the impl's declared types, never by its syntax.

## 2. Witnesses

A body that calls a generic function at its own type variable receives the impl in one of two ways. The compiler decides at each call which impl arrives, and the parameter that receives it states no condition on it, so its choice gives the generic function's effect variable the declaration's own (variables §1). An impl's facts are its own, and the compiler uses them where it knows that impl is the one reached (facts §1).

`InstanceOf` is the impl's witness apart from any value. It is given only by reference and so never kept: a type variable stands for every acvus type as one Rust type (variables §3), so a kept witness could meet a value of another type whose impl has the same signature. `Instance` is a value and its impl of one generic function, as one. Its mode, owned, shared or exclusive, is a marker read through Rust, as every position's type is (declarations §1), and the body receives `T`, `&T` or `&mut T`. Once received it may go anywhere, since its impl meets no other value and its flows are tracked (declarations §2).
