# variables

## 1. Markers

A generic parameter carries one marker bound and nothing else: `TypeVar`, `EffectVar` or `IdentityVar`, one per kind of variable (compiler: extern fn §1, §2; types §3). Carried through other types, each variable stays one type, as Rust keeps it (compiler: types §1). `TypeVar` gives `as_ref` and `as_mut`, the only way generic code makes a reference to a variable's value that it can give out. Every type variable a declaration writes is one the use must settle, never taken as `!` (compiler: extern fn §1).

## 2. A variable's value

A type variable's value is never concrete and never wrapped in a `Uniform`; it lives in containers, and a reference to one is held as references §4 holds it. It carries no `'static` (safety §2), so nothing keeps it out of sight and `Uniform::new` cannot wrap it. A value holding a variable or a loan crosses as its own type, where its declaration shows them.

## 3. Instantiation

A type variable is monomorphized as a `Uniform` of one fixed uninhabited marker, never as a `Value`. One value is then read under several views, generic code's over that marker and a concrete view such as `Stack<Uniform<R, String>>`, and goes from one view to another and back.

The views stay one. `TypeVar` holds for every `Uniform` and has no associated type and no supertrait, so an impl generic code reaches through its one bound covers every view alike, coherence refuses a second impl for one view, and an impl naming the marker itself is one generic code cannot reach. A `Uniform` kept anyway carries its release in its drop. The views' fields differ only in a zero-sized marker, which adds no niche and no field. This holds while Rust admits no specialization, default impls or negative impls, any of which would let an impl single out one view.

## 4. The layout exception

Instantiations of one type that differ only in a zero-sized marker share a layout within one compilation. Rust does not guarantee it: two instantiations are two types, and the Reference states no layout for `repr(Rust)` beyond soundness. It is rustc's behaviour, computing a layout from the fields' layouts and the representation, to which the marker adds nothing, and it is the one reliance on what Rust does not guarantee. A program is compiled whole, so it is needed within one compilation alone. No check exercises it; where a structure is reread in place, a check that sizes and alignments agree confirms that much.

*Rejected.* Monomorphizing a variable as a `Value`: a `Value` is `Copy` and has no drop, so generic code's moves and drops would carry no ownership. A marker named afresh in each compilation: coherence already keeps generic code and every view to one impl.
