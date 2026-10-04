# facts

## 1. Declared facts

A declaration states its effect, its ends, its casts and conversions, and the facts of compiler: extern fn §3, access, action, source and relation, each in exactly the forms the compiler defines. They are conveyed as declared, and the compiler trusts them: nothing checks them against the body, and the author carries the compiler's burden for them (compiler: extern fn). A fact names only what its declaration has: its parameters, the places they lend, its closure parameters, and other declared functions and generic functions. Rust checks what Rust can, the signature, its lifetimes and the choice an impl writes; what a fact says of the body is the author's.

## 2. Where a fact is written

A fact is written where what it speaks of is written. A fact of a place, an access or an update or an inverse of it, is written on the parameter whose storage the place is, the whole of it or one element a key or an index parameter names; once a touch is written anywhere in a signature, the touches written are the call's access, and a place left out is one the call does not touch. A fact of the declaration as a whole, its effect and its ends, its being a combine, an equivalence or a total order, a cast or a conversion, is written on the function; so is a fact of its result or of several of its positions at once, a source or a relation, since a Rust signature gives its result no place of its own to write on. A stream made from another is written as a chain from the parameter it is made from, as Rust chains an iterator's adapters, and a consumption closes the chain. A fact names a parameter by its Rust name, the result as `result`, and another declaration or generic function by its Rust path, which Rust resolves.
