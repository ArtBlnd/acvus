# extern fn

An extern fn is a declaration. A program knows nothing of an extern fn beyond its declaration.

**The compiler trusts every part of a declaration, absolutely.** Its signature, effect, declared facts (§3), flows, impls and casts are axioms. Nothing checks them against the body behind them, and the compiler builds on them as far as they reach: it reorders, removes, fuses and proves by them. Any behaviour of the body that departs from its declaration, in any part, is undefined behaviour.

An extern fn author therefore carries the compiler's burden. We expect the author to understand the compiler well enough to have written it: what each part of a declaration licenses, and what the compiler will do on the strength of it.

The reason is the language's identity: it is a DSL for DSLs. It defines no domain of its own. A domain enters only through its extern fns, and their declarations are that domain's semantics. Whoever writes them is building a language on this one, and stands where a compiler writer stands.

## 1. Signature

Named parameters and a result. A parameter takes its value by value, by `&T` or by `&mut T`; a sequence arrives as a slice. No declaration carries an array type.

A type parameter is parametric, and carries one bound: any type; an integer type; or any type that the use must settle, never taken as `!`. Which shapes a function takes is said by its impls (§6), not by a bound.

An extern fn is called, never taken as a value: a program names it only to call it, and where a function value is wanted it writes a lambda that calls it. So a declaration's flows (§4) are read only at its calls.

## 2. Effect

A call's effect states whether reissuing it is pure, idempotent or opaque, an order in which each is below the next, so a join takes the greater; whether it commutes with other calls; whether it may be split, and at what cost (effects §6); and which contexts it reads and writes. An opaque effect is always sound (effects).

A declaration may name effect variables, as it names type variables. A function type it writes carries an effect, and an extern-defined type may carry an effect (§8).

A call's effect is the join of what its declaration states for the call itself and of the effect variables of the work the call runs before its result is taken: a closure argument it calls, or work a value it is given holds. Work the call does not run leaves only in its result, whose type carries its effect.

Every effect variable a declaration takes from its parameters appears in the call's effect, in its result's type, or in both. A declaration where one appears in neither is refused: the work it names would run where no effect records it. An effect variable its parameters do not give names no work, and is pure.

## 3. Declared facts: access, action, source, relation

Beyond its signature and effect, a declaration states facts of four kinds. Each speaks of a place, meaning a parameter's storage whole or one element of it named by a key or an index, and of what a call does there.

- **Access**: which places a call touches, and how. It reads a place; it lends a view of it (`&` or `&mut`); it opens it (an absent entry becomes a stated default, a present one stays, and a `&mut` to it is given); or it stores into it (the place holds a stated value, or is emptied, whatever it held). A call's result may be stated as a declared pure function of what a place held when the call began, the identity function among them: a call that stores into a place may give back what it held. A call touches no place its access does not state. Two keys name one place exactly when the key type's `eq` impl, declared an equivalence, calls them equal.
- **Action**: what a call's update of a place is. A function may be declared a combine: an associative operation, with or without an identity, commutative or not, and with or without an absorbing element, which it returns whatever the other operand. An update of a place is the action of a combine. An update may be declared the inverse of another at the same place: one after the other leaves the place as it was. A total order declared on a type yields the combines that keep its least and its greatest. An assignment is always the combine that keeps the later value.
- **Source**: how a stream yields its elements, and how a call consumes one. A stream is splittable at any position, or yields only step by step, and a relation may bound its count. A stream made from another yields each element through a closure argument, keeps the elements a closure argument accepts, keeps them until the first one it does not accept, or yields one stream after another; these forms compose. A call consumes a stream by updating a state place with each element in order, from a stated initial value, and giving a stated function of the final state as its result. Where the update is a combine, the consumption may be split; where the state reaches the combine's absorbing element, the remaining elements are not consumed.
- **Relation**: `=`, `≤` or `<` between affine integer terms: sums and differences of atoms and their constant multiples, and constants. The atoms are the arguments, the result, their lengths, and a value as it stood when the call began.

The forms of each kind are exactly those listed here, and a declaration states nothing else. A form is added only in this section, together with its meaning, stated as an equality with the sequential program. A declaration's impls are held to that equality on the declaring side (extern: declared facts). What a transform needs beyond these facts, it derives.

## 4. Flows

A declaration states what a call reaches through its reference arguments (by default, everything they lend), what its result borrows from them, and how a call ends: unstated, returns or traps, or total.

A function value a call gives, as its result or a part of it, has the flows its declared function type states; a declaration states none for such a type, so it has every flow (types §5).

## 5. Parameters and overloading

Several declarations may share a name. A call reaches the one its arguments' types decide (types §6).

## 6. Generic functions and impls

A generic function is a name with a polymorphic type and no body. A declaration may be an impl of one, choosing some of its type variables. A declaration may require impls of generic functions at its own type variables, and may name a declared fact (§3) such an impl must state. A required impl's types are smaller than the declaration's own, so requirements end.

## 7. Conversions

A declaration may be a cast from its parameter's type to its result's type, which is a step up in the tree of types §3, or a conversion, which is no part of that tree; a type may have several conversions (types §7).

## 8. Types an extern fn defines

Every type a declaration writes, except a type parameter, is extern-defined, with all its parts, and follows the transparency rule (types §1). An extern-defined type may declare lifetimes; each is a position of its values (ownership §5).

An extern-defined type may carry one effect, apart from its type arguments: `T<…> with E`. It bounds from above the effects of the work its values hold, which runs later. A type is covariant in its effect: widening an upper bound moves toward the opaque effect, which is always sound (effects).

## 9. What a call is given

The compiler holds its side of the declaration as the extern holds its own. A call receives each argument as a valid value of its declared type, by the passing its declaration states. What a call is lent stays alive until the call's result is taken; nothing else writes what is lent `&`, and nothing else touches what is lent `&mut`. A closure argument is callable as its function type states, any number of times, for as long as its holds allow.
