# variables

## 1. Markers

A generic parameter that stands for a variable carries one marker bound and nothing else, one per kind of variable: `TypeVar`, `EffectVar` or `IdentityVar` (compiler: extern fn §1, §2; types §3). Carried through other types, a variable stays one type, as Rust keeps it (compiler: types §1). `TypeVar` gives `as_ref` and `as_mut`, the only way generic code makes a reference to a variable's value that it can give out. A type variable a declaration writes is one its use must settle, never taken as `!`; one a use need not settle (compiler: extern fn §1) is deferred.

An effect is what a call does, so only a function has one, and every function's is opaque, idempotent or pure. Which function a callable holds is unknown where it is written, and no fixed effect admits every function, so whatever is callable has an effect variable for its effect: a closure parameter, a kept function, and the impl a witness carries (functions §1, §2; impls §2). A type has an effect variable only where its values may hold functions: it carries their effect as that variable, which the declarations that write it fill with their own, and holds no effect of its own. A declaration whose result does not carry its effect variable consumes it as its own effect (compiler: extern fn §2), and writes it as facts §2 states.

## 2. A variable's value

A type variable's value is never concrete and never wrapped in a `Uniform`. It lives in containers, and a reference to one is held as references §4 holds it. It carries no `'static` (safety §2), so nothing keeps it out of sight, and `Uniform::new` cannot wrap it. A value that holds a variable or a loan crosses as its own type, where its declaration shows them.

## 3. Instantiation

A type variable is monomorphized as a `Uniform` of one fixed uninhabited marker, never as a `Value`. One value is then read at several instantiations of its type, generic code's at that marker and a concrete one such as `Stack<R, Uniform<R, String>>`, and passes from one to another and back.

The instantiations stay one. `TypeVar` holds for every `Uniform` and for nothing else: it is sealed by a supertrait that no other crate can name and that carries nothing, and it has no associated type and no other supertrait. So an impl generic code reaches through its one bound covers every instantiation alike, coherence refuses a second impl for one instantiation, an impl naming the marker itself is one generic code cannot reach, and no type but a `Uniform` is ever taken for a variable's value. Because coherence knows that no other type is a `TypeVar`, an impl over every variable stands beside each concrete type's own, and a structure that holds a variable beside concrete parts crosses part by part, each by its own kind (values §3). A `Uniform` kept anyway carries its release in its drop. The instantiations' fields differ only in a zero-sized marker, which adds no niche and no field. All of this holds while Rust admits no specialization, default impls or negative impls, any of which would let an impl single out one instantiation.

## 4. The layout exception

Instantiations of one type that differ only in a zero-sized marker share a layout within one compilation. Rust does not guarantee it: two instantiations are two types, and the Reference states no layout for `repr(Rust)` beyond soundness. It is rustc's behaviour, which computes a layout from the fields' layouts and the representation, to which the marker adds nothing, and it is the one reliance on what Rust does not guarantee. A Rust program is compiled whole, so the reliance is needed within one compilation alone. No check exercises it; where a structure is reread in place, a check that sizes and alignments agree confirms that much.
