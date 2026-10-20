# runtime

## 1. Role

A runtime lowers the IR and runs it, implementing the compiler's primitives as functions, so that nothing outside it needs `unsafe` beyond what safety §3 states. The extern side is generic over the runtime, so a call into it is resolved statically and inlined where the extern fn is compiled, and a runtime costs next to nothing from outside (§4). The compiler's correctness is its premise. How it holds a value is its own (values §1), and a reference value it makes holds for the borrow it is made from (references §1). It runs an asynchronous call as the future that awaits it is polled (functions §3).

## 2. Kinds

The contract states the minimal conditions for a safe runtime, and states them as kinds: one associated type of `Runtime` per kind, each with a trait of its own operations, no kind's operation serving another kind. The operations that take no type are the runtime's own (values §1): its release of a value, which ends its owner and every value that owner holds and ends nothing for a reference value, and its making of a reference value from a place, by `value_as_ref` from a shared borrow and by `value_as_mut` from an exclusive one, a place that holds a reference giving that reference (references §1).

A kind groups the type constructors the boundary treats alike: how a value is given and taken back, how a reference to one is projected, and whether it is `Copy`. Scalars, strings, tuples, options, results and functions form safe kinds. Opaque values, structural objects and enums, and views of runs and text form unsafe kinds. An opaque or a structural value is read at several instantiations of its Rust type (variables §3), so it is held untyped and read by layout; a view names memory outside its `Value`, and is read back from it as a Rust slice or `str`. A kind that holds other values deals in their `Value`s alone, each given and taken back by its own kind within the holding kind's operation, so a nested type is its layers' kinds composed, the innermost first.

A scalar or a string is given and taken back as its Rust value, and a reference to one is projected to a Rust reference to it; a scalar is `Copy`. A tuple is given as its parts' values and taken back as them, and a reference to one is projected to the tuple of its parts' references, a `Ref` or a `Mut` of each, or a part itself where it is a reference (references §3), which a declaration writes as such and which crosses by value as any structure does. An option or a result is given and taken back as its Rust value, its variants told apart inside its kind alone, and a reference to one is projected to a Rust reference to its shape, its payload uniform: an exclusive one reaches the variant's place, and `Option<&mut T>` the payload alone. An opaque value is given and taken back as its Rust value and projected to a Rust reference to it, read by layout at whichever instantiation its reader holds (variables §3, §4). A structural object or enum is held as an opaque value is, and the runtime reads and writes its fields by the table its derive states (types §2). A view is given as a Rust slice or `str` (references §2) and taken back as one at the lifetime its taker names, an entry its own (declarations §2); it owns nothing, so a release ends nothing for it, and a value holds it as it holds any other.

A function value is never given or taken back at the boundary, since an extern gives acvus no function value (functions). It is called through a shared borrow, an exclusive one, or by value, which consumes it, as its mode states (functions §2), each synchronously or asynchronously in place (functions §3). Its arguments are given as values by the side that holds them as their types (safety §4), and its result is taken back by its own kind.

## 3. Safety per kind

A safe kind is implementable with no unsafe operation but its calls of the crossings of the values it holds, whose premises (safety §3) pass through from its own caller. A checked runtime checks the premises in its operations' bodies, and panics on a misuse such as aliasing or a released value. An unsafe kind reads storage as a Rust type, in an implementation the extern side provides once and a runtime may replace; misuse there is undefined behaviour, and a value's type there is the compiler's premise. A runtime asserts the compiler's premise about the values it holds in two places alone: where it calls an extern fn with the call's arguments, and where it reads or writes a structural value's field by its table. A misuse panics in a checked runtime and is undefined behaviour in an optimized one.

## 4. Projections and enums

A projection is lent to a closure, so its lifetime is the closure's, and the contract carries no lifetime-generic associated type. Each kind lends its own projections of a reference value, each argument is taken back or lent by its own kind, and the body runs inside the closures its projections are lent to; that this costs nothing rests on their inlining, which is to be verified, not assumed. A layer may be held in its `Value` where it fits, since how a value is held is the runtime's (values §1): a projection of a layer held so lends the shape it decodes, and an exclusive one writes it back.

The runtime tells a structural enum's variants apart by its table (types §2), and an extern by matching its Rust value, which exhausts them, so no code branches on a tag it cannot exhaust. An option's and a result's variants are told apart inside their kinds (§2).

## 5. Release

A release ends a value's owner, and no value has a second owner at the boundary: asynchronous work takes what it uses by value (functions §3).
