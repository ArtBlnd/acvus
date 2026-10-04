# boundary

The extern side turns a Rust function into an extern fn. Its author writes a Rust signature and a body, and the declaration that signature states is what the compiler receives (compiler: extern fn). The registry that gives declarations is the extern side's; the compiler only receives them, and the meaning of every declaration is the compiler's.

## 1. Aim

Every ordinary Rust function signature is expressible at the boundary, and every declaration form is writable from Rust and written correctly, but for a requirement that names a fact its impl must state, which the extern side does not write, and the bound of a type variable a use need not settle, which is deferred (variables §1). A signature that cannot be expressed is a defect of the structure, and the structure is revisited. Excluded are the type-level tricks in an author's signature, associated types, GATs, HKTs and upcasting; an HRTB trick played in acvus for an extern fn to accept; lazy monomorphization; and `'static` (safety §2).

## 2. Generality

Generality is the intersection of Rust and acvus. What acvus defines and Rust expresses by what Rust guarantees is one concept, met by one rule over the whole of it: options and results, tuples of every arity Rust's means reach, objects and enums, primitives, strings, slices, references, functions and the passing modes. A type only Rust has is no part of acvus, and enters only as an extern-defined type (types §1). What Rust does not guarantee is not guaranteed.

*Rejected.* A rule of its own for a type only Rust has, as a vector given its slice by the boundary: it makes a path Rust does not guard (safety §4).
