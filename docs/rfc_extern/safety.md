# safety

## 1. Invariants

A round trip holds at every position. Lifetimes are exact. A value that crossed is kept nowhere its declaration does not show. The boundary has no dynamic check and no dynamic conversion. These hold by the primitives of this set, with nothing added on top.

Boundary and author code have no branch on a type's form or kind, no forced unwrap, no cast, and no assert or type check outside the runtime.

## 2. `'static`

Only the runtime contract, `Uniform`'s boxing included (values §2), may require `'static`. A signature writes `'static` nowhere, neither as a bound nor as a lifetime, and the macro refuses every one.

## 3. Honest `unsafe`

The standard is a safe API, and `unsafe` is welcome where it is stated honestly and kept local: an explicit `unsafe` can be safer than a safe-looking API. A function that consumes a `Value` rests on premises the `Value` does not carry: that it was made as a type, that it is live, and that its holder owns it or holds it exclusively. Every such function is an `unsafe fn` stating them, and a function that only makes a `Value` is safe. Where a runtime asserts those premises is runtime §3.

## 4. Forbidden

- A `Value` in a contravariant position, never and nowhere. A position that takes a `T`, a parameter of a function, a closure or a function value, never takes a `Value` in its place: erased so, it takes a value made as any type, reads it as `T`, and the first of another type is undefined behaviour. A type is erased only where a value is given, by the side that holds it as its type, never where a value is taken.
- Instantiation: a named type or form treated apart, a marker, trait, wrapper or conversion for one case, a flag parting two meanings held in one place. One makes a path Rust does not guard; nothing built on it can then be guarded, and no further work repairs that.
- Reliance on what Rust does not guarantee, but for the one exception of variables §4.
- Lifetime detach: a lifetime made through `unsafe` that is not bound to its source, such as an unsafe GAT projection `'x` over `'t`. In generic code it is unbounded, and every borrow check around it passes vacuously. A transmute to `'static` shows where it is written; a detach hides behind a safe signature.
- A step the primitives do not derive. The work stops there, and the question goes to the owner of the structure.
