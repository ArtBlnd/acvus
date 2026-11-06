# context

A context is a value, or an object, of a fixed type that is always filled whenever the program runs.

## 1. What "filled" means

A program may take a context's value and put a value back.

"Filled" is an invariant wherever the page is seen beyond the program: at every call whose effect may read or write the context, and on every path on which the run ends. Between those points a context may be empty after a take. What a run that traps or does not end leaves in a context is not stated.

## 2. Opacity

A body sees a context by its declaration, at the Opaque level, and a context's type is given by the host ([the type system](../transparency-constraint-type-system.md), Transparency). That type is external at every position it is used at (the type system, Type states).

## 3. Take, lend, and commit

A context is a place outside the body, on a page the host owns. What the page does beyond the program is unknown to the language, as a syscall's is. A context is close to a typed vIOMMU.

A program reaches the page directly. It reads a context of a copy type by copying it, takes its value out, lends it in place (`&` or `&mut`), and commits a value into it; a commit into a filled context drops what it held, as an assignment does. No context aliases another place, so only a lend of it is tracked as a loan. A call whose effect may read or write a context is an access of it and meets its loans as any access does (ownership §5); a call whose effect declares the contexts it touches is an access of those, and an opaque one of every context.

Between those calls, nothing beyond the program sees the page. A read, take, lend or commit there carries no effect of its own and is ordered only against work on the same place. Promoting a context to a plain value over such a stretch removes its page operations there; it is an optimization, never a requirement.

A context may be declared volatile. A volatile context is never promoted: each read, take, lend and commit of it is an effect, and is ordered as effects are (effects §2). Nothing merges or removes them, and against them only an effect, or what depends on their values, keeps its order. Pure work passes them freely.

A value stored into a context outlives the run, so its type has no positions (ownership §5). Its type may be branded, by a brand type fresh for each context: each context is storage of its own, apart from every other, so its brand type is never made one with another (types §3).
