# boundary

The extern side turns a Rust function into an extern fn. Its author writes a Rust signature and a body; the signature states the declaration the compiler receives (compiler: extern fn), and the compiler gives that declaration its meaning. The extern side gives its declarations through its registry (declarations §5), and the compiler only receives them.

## 1. Aim

Every ordinary Rust function signature is expressible, and every declaration form is writable from Rust and written correctly. Two things are left out: a requirement that names a fact its impl must state, which the extern side does not write, and a type variable a use need not settle, which is deferred (variables §1). A signature that cannot be expressed is a defect of the structure, and the structure is revisited.

Excluded are the type-level tricks in an author's signature, associated types, GATs, HKTs and upcasting; an HRTB trick played in acvus so that an extern fn accepts; lazy monomorphization; a function value an extern gives (functions); and `'static` (safety §2).

## 2. Generality

The boundary covers the intersection of Rust and acvus. What acvus defines and Rust expresses by what Rust guarantees is one concept, met by one rule over the whole of it: options and results, tuples of every arity Rust's means reach, objects and enums, primitives, strings, slices, references, functions and the passing modes. A type only Rust has is no part of acvus, and enters only as an extern-defined type (types §1). What Rust does not guarantee is not guaranteed.
