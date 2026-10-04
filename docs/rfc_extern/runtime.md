# runtime

## 1. Role

A runtime lowers the IR and runs it, implementing the compiler's primitives as functions, so nothing outside it needs `unsafe` beyond what safety §3 states. It is generic over backends and zero cost from outside, and the compiler's correctness is its premise. A value it holds out of line does not move while a reference to it lives; the places that hold its `Value`s may. It runs an asynchronous call as the future that awaits it is polled (functions §3).

## 2. Kinds

The contract states only the minimal conditions for a safe runtime, as one associated type per kind, each with a trait of its own functions; no function serves every kind. A kind groups the type constructors the boundary treats alike: how a value is made and given back, how a reference to it opens, and whether it is `Copy`. Scalars, strings, tuples, objects, options, results, enums and functions form safe kinds. Opaque values and references to runs, text and places form unsafe kinds: an opaque value is seen as several Rust types, generic and uniform, so it is held untyped and read by layout, and a reference's address is read back from a `Value`. A `Ref` of one value is read by its referent's kind.

## 3. Safety per kind

Every kind's consuming functions are `unsafe fn` (safety §3). A safe kind is implementable with no unsafe operation: a safe runtime checks the premises in those functions' bodies, and panics on a misuse such as aliasing or a released value. An unsafe kind reads storage as a Rust type, in an implementation the extern side provides once and a runtime may replace; misuse there is undefined behaviour, and a value's type there is the compiler's premise. A runtime performs an unsafe operation in two places alone, each asserting the compiler's premise about the values it holds: where it calls an extern fn with the call's arguments, and where it builds an enum value by the constructor for its tag. Misuse panics in a safe runtime and is undefined behaviour in an optimized one.

## 4. Views and enums

A view is lent to a closure, so its lifetime is the closure's, and the contract carries no lifetime-generic associated type. Each kind lends its own views, each argument is opened by its own kind, and the body runs inside those closures; that this costs nothing rests on their inlining, which is to be verified, not assumed. An enum value is a tag and one payload. The runtime picks the variant's constructor by its tag, among those the enum's derive gives (types §2), so no code outside the runtime branches on a tag it cannot exhaust.

## 5. Release

A release ends a value's owner, and no value has a second owner at the boundary: asynchronous work takes what it uses by value (functions §3).
