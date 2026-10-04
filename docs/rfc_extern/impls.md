# impls

## 1. Generic functions

A generic function enters the compiler as one (compiler: extern fn §6), and an impl enters with the function and the choice it implements, which Rust checks against the function's signature. An impl written with `Uniform` is the one generic code reaches, and one written natively is its `#` counterpart (types §3).

## 2. Witnesses

A body that calls a generic function at its own type variable receives the impl in one of two forms. Which impl arrives the compiler decides at each call, and the parameter that receives it states no condition on it; an impl's facts are its own, and the compiler uses them where it knows that impl is the one reached (facts §1). `InstanceOf` is the impl's witness apart from any value, given only by reference and so never kept: a type variable stands for every acvus type as one Rust type (variables §3), so a witness kept could meet a value of another type whose impl has the same signature. `Instance` is a value and its impl of one generic function, as one. Its mode, owned, shared or exclusive, is a marker the macro reads from the signature, and the body receives `T`, `&T` or `&mut T`. Once received it may go anywhere, since its impl meets no other value and its flows are tracked (declarations §2).
