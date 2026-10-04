# declarations

## 1. Mirroring

A declaration mirrors the Rust types as written. A `Uniform` stands for its marker's acvus type held uniformly, a reference marker's `Uniform` for its reference held uniformly, a `#[fn_type]` marker for the function type its alias writes, and an extern-defined type for itself in one of its two forms (types §3). Each position's acvus type is collected through its Rust type's own kind, never from its name.

## 2. Flows

Flows are Rust's lifetime parameters (compiler: extern fn §4). rustc checks the body against them, and the compiler tracks liveness in the program, so references may sit inside values. A result position borrows from the argument positions that carry its lifetime. A signature writes no bound between lifetimes; what one outlives is what its types imply (compiler: ownership §5).

A value that holds a loan is kept with its type, its lifetimes included, erased, and an entry opens it again at one named lifetime of its own that every loan of the call takes: never `'static`, and never one the code around it can stretch, so the body only narrows what it holds. A lifetime inferred there would be unbounded (safety §4).

## 3. Collecting lifetimes

A position's lifetimes are those of its resolved type, while a signature's syntax shows its written type. An alias between them, a lifetime elided in a path, or a structural type whose fields' order parts from its parameters' would label positions wrongly, and wrong flows are undefined behaviour. So the lifetimes the syntax names are confirmed by Rust against the resolved types, each type stating the lifetimes it carries in the order of its own parameters, and a signature whose two accounts differ does not compile. The confirmation verifies and computes nothing. The type-level machinery it needs stays in generated code and in the extern side's own, and never appears in an author's signature. A type variable carries no lifetime there, so the confirmation puts every one as one placeholder; which variable stands where, and so the flows through a variable's part, is read from the declared type (§1), which Rust resolves, never from the syntax.

## 4. Effect and suspension

A declaration's effect, ends and facts are facts §1's. A body that awaits declares so, with an effect that suspends.

## 5. Registry

A registry is a runtime that takes what the extern side declares and gives the declarations to the compiler. Each extern fn and each generic function reaches it as a type stating its own declaration, never as an erased function, so a runtime builds the call of an extern fn, with its fixed parameter shape, around that type. A type an extern fn writes is declared through that fn, and its crossing runs inside the fn's entry. A generic function states a signature alone; `core`'s are the compiler's own (compiler: operators §2), and the extern side has them only for impls to name. A cast or a conversion is an extern fn like any other, whose declaration also states it as one, so the compiler coerces through it (compiler: extern fn §7).
