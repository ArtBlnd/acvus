# macros

An extern author writes Rust and macros turn it into declarations and glue: an attribute on a function, an attribute on a generic function's signature, a derive on a type, and a shape. Nothing else reaches the crossing: the crossing traits are sealed, and only the boundary and the derive implement them. No `macro_rules!` emits a crossing implementation into a user crate.

## 1. The function attribute

The attribute reads the function's Rust signature and its arguments and emits the declaration (compiler: extern fn) and the glue.

- A parameter is taken by value, by `&T` or by `&mut T`. A sequence is taken as a slice, `&[T]` or `&mut [T]`, never as an array.
- A result is a value; a sequence is returned as a `Vec<T>`. A result that borrows borrows only what an argument lends, in place or through a loan it holds, and no longer than that loan; its lifetimes say which (compiler: extern fn §4). A signature writes no bound between lifetimes; what one outlives is what its types imply.
- A closure parameter is a callback, callable any number of times. A callback is called synchronously or awaited (crossing §3).
- A declaration has one handler with a returning body, an awaited body, or both. An `async fn` is the awaited body and names its returning body where it has one; the two have one signature. The declaration states whether it has an awaited body, and one with no returning body suspends (compiler: extern fn §2). Which body a call runs is the backend's choice (interpreter: tasks), except that a call whose effect suspends, through a callback or an impl it runs, runs the awaited body where the declaration has one. A returning body calls each callback and impl synchronously; one that may suspend is driven to its end there (runtime contract §2, Callbacks). An awaited body's future is `Send`, since a task may move between threads.
- The effect states whether a call may be split, and at what cost; the cost is the backend's (compiler: effects §6).
- A declaration's path is its function's Rust module path, under its type for a function in an `impl` block of an extern-defined type the crate defines. A name the attribute states replaces the function's own only to overload within one module. An impl of a generic function is reached only through its generic function and states no name.
- Declared facts are written in the attribute (declared facts). A declared fact about a place (compiler: extern fn §3, access and action) is written on the parameter whose storage it speaks of; a relation and a source are written on the function.
- A closure parameter's function type may hold any loan, so a callback may be any function value the program makes.

The glue it emits contains no `unsafe`.

## 2. The derive

The derive makes a Rust type an opaque registered type (types §2). It reads the definition and emits exactly the structural facts about it that no stable Rust bound can state (safety §2), and nothing else. It emits no projection types: an opaque type crosses as itself. Neither macro specializes a type argument: a specialization is a whole type a registry names (registry). An object or an enum an extern takes or gives needs no derive: a shape declares it, giving its structural type and the Rust views it crosses as (types §1).

## 3. What the macros emit, and what they do not

The macros' output is safe code plus the derive's structural signed implementations and the seal of what each macro implements: one seal per meaning, a type's crossing (the derive's) and a declaration's dispatch (the attribute's). A source scan counts production `unsafe` with generated items included; the count is those structural implementations and those seals.

## 4. Crossing is selected by types

How a type crosses the boundary is selected by trait resolution at the type's instantiation, never by inspecting tokens at macro expansion.

*Why.* Expansion sees a type's spelling, not the type: a selection made there differs for an alias, a type parameter or another macro's output, and is fixed at the generic definition rather than at each instantiation.
