# facts

## 1. Declared facts

A declaration states its effect, its ends, its casts and conversions, and the facts of compiler: extern fn §3, access, action, source and relation, each in exactly the forms the compiler defines. They are conveyed as declared, and the compiler trusts them: nothing checks them against the body, and the author carries the compiler's burden for them (compiler: extern fn). A fact names only what its declaration has: its parameters, the places they lend, its result, its closure parameters, the contexts its effect reads and writes, and other declared functions and generic functions. Rust checks what Rust can, the signature, its lifetimes and the choice an impl writes; what a fact says of the body is the author's.

## 2. Where a fact is written

A fact is written as an attribute, where what it speaks of is written. Each form of what §1 lists, as the compiler defines it, is one attribute, under the compiler's own name for it, so a fact is written in those forms and nothing else, and a form the compiler adds is an attribute here.

A fact of a place, a touch, a give, an update or an inverse, is written on the parameter whose storage the place is: the whole of it, or, by `at`, one element a key or an index parameter names, as in `#[store(value, at = key)] map: &mut Map<R, K, V>`. Once a touch is written anywhere in a signature, the touches written are the call's access, and a place left out is one the call does not touch.

A fact of the declaration as a whole is written on the function, as is the generic function an impl implements: its effect and its ends, its being a combine, an equivalence or a total order, and its being a cast or a conversion. So is a fact of its result or of several of its positions at once, a source or a relation, since a Rust signature gives its result no place of its own. An effect is a known one, a variable, or the two joined by `|`: an extern fn or an impl that consumes its effect variable (variables §1) writes it as its own effect, `effect = E`, or, where its own work has an effect too, `effect = E | opaque`. A stream made from another is written as a chain from the parameter it is made from, as Rust chains an iterator's adapters, and a consumption closes the chain.

A fact names a parameter by its Rust name, the result as `result`, a context by the name the host declares it under, and another declaration or generic function by its Rust path, which Rust resolves.
