# extern fn

An extern fn is a declaration. A program knows nothing of an extern fn beyond its declaration.

**The compiler trusts every part of a declaration, absolutely.** Its signature, effect, declared facts (§3), flows, impls and casts are axioms. Nothing checks them against the body behind them, and the compiler builds on them as far as they reach: it reorders, removes, fuses and proves by them. Any behaviour of the body that departs from its declaration, in any part, is undefined behaviour.

An extern fn author therefore carries the compiler's burden. We expect the author to understand the compiler well enough to have written it: what each part of a declaration licenses, and what the compiler will do on the strength of it.

The reason is the language's identity: it is a DSL for DSLs. It defines no domain of its own. A domain enters only through its extern fns, and their declarations are that domain's semantics. Whoever writes them is building a language on this one, and stands where a compiler writer stands.

## 1. Signature

Named parameters and a result. A parameter takes its value by value, by `&T` or by `&mut T`; a sequence arrives as a slice. No declaration carries an array type.

A type parameter is parametric, and carries one bound: any type; an integer type; or any type that the use must settle, never taken as `!`. Which shapes a function takes is said by its impls (§6), not by a bound.

## 2. Effect

A call's effect states whether reissuing it is pure, idempotent or opaque; whether it commutes with other calls; whether it may be split, and at what cost (effects §6); and which contexts it reads and writes. An opaque effect is always sound (effects).

## 3. Declared facts: access, action, source, relation

Beyond its signature and effect, a declaration states facts of four kinds. Each speaks of a place, meaning a parameter's storage whole or one element of it named by a key or an index, and of what a call does there.

- **Access**: which places a call touches, and how. It reads a place; it lends a view of it (`&` or `&mut`); it opens it (an absent entry becomes a stated default, a present one stays, and a `&mut` to it is given); or it stores into it (the place holds a stated value, or is emptied, whatever it held). A call touches no place its access does not state. Two keys name one place exactly when the key type's `eq` impl, declared an equivalence, calls them equal.
- **Action**: what a call's update of a place is. An update is the action of a combine, an associative operation on updates, with or without an identity, and commutative or not. An update may be declared the inverse of another at the same place: one after the other leaves the place as it was. A total order declared on a type yields the combines that keep its least and its greatest. An assignment is always the combine that keeps the later value.
- **Source**: how a stream yields its elements. Either it is splittable at any position, or it yields only step by step, and a relation may bound its count. A stream made from another yields each element through a closure argument, or keeps the elements a closure argument accepts, or yields one stream after another.
- **Relation**: `=`, `≤` or `<` between affine integer terms: sums and differences of atoms and their constant multiples, and constants. The atoms are the arguments, the result, their lengths, and a value as it stood when the call began.

The forms of each kind are exactly those listed here, and a declaration states nothing else. A form is added only in this section, together with its meaning, stated as an equality with the sequential program. A declaration's impls are held to that equality on the declaring side (extern: declared facts). What a transform needs beyond these facts, it derives.

## 4. Flows

A declaration states what a call reaches through its reference arguments (by default, everything they lend), what its result borrows from them, and how a call ends: unstated, returns or traps, or total.

## 5. Parameters and overloading

Several declarations may share a name. A call reaches the one its arguments' types decide (types §6).

## 6. Generic functions and impls

A generic function is a name with a polymorphic type and no body. A declaration may be an impl of one, choosing some of its type variables. A declaration may require impls of generic functions at its own type variables, and may name a declared fact (§3) such an impl must state. A required impl's types are smaller than the declaration's own, so requirements end.

## 7. Conversions

A declaration may be a cast from its parameter's type to its result's type, which is a step up in the tree of types §3, or a conversion, which is no part of that tree; a type may have several conversions (types §7).

## 8. Types an extern fn defines

Every type a declaration writes, except a type parameter, is extern-defined, with all its parts, and follows the transparency rule (types §1). An extern-defined type may declare lifetimes; each is a position of its values (ownership §5).
