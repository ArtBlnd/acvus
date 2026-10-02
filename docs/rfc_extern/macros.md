# macros

An extern author writes Rust; macros turn it into declarations and glue: an attribute on a function, an attribute on a generic function's signature, and a derive on a type. Only the boundary and the macros implement the crossing.

## 1. The function attribute

The attribute reads the function's signature and emits its declaration (compiler: extern fn) and its dispatch.

- A parameter is an argument, a capture, the body's executor or a requirement, as it is marked, and its Rust type is held to that role. An argument's Rust type states its declared type (crossing §2).
- A generic parameter is the runtime, a type variable, an effect variable or an identity variable, as its bound states.
- A result that borrows states by its lifetimes which arguments it borrows from (compiler: extern fn §4). A signature writes no bound between lifetimes.
- A declaration has a returning body, an awaited body, or both, under one signature, and states whether it has an awaited body; one with no returning body suspends (compiler: extern fn §2). Which body a call runs is the backend's, except that a call whose effect suspends runs the awaited body. A returning body calls every callback and impl returning (runtime contract §2). An awaited body's future is `Send`.
- A declaration's path is its function's module path; a stated name overloads within one module. An impl states its generic function and no name.
- Declared facts are written in the attribute, a fact of a place on the parameter whose storage it speaks of (declared facts).

## 2. The generic function attribute

The attribute reads a signature with no body and emits the generic function's declaration and its family (impls §1). Each parameter is an argument.

## 3. The derive

The derive makes a Rust type an opaque registered type (types §2). It emits the type's crossing and the structural facts about it that no stable Rust bound states (safety §2), and nothing else.

## 4. What the macros emit

Safe code, the derive's structural signed implementations, and one seal per meaning: a type's crossing, a declaration's dispatch, and a generic function's family.

## 5. Selection by types

How a type crosses is selected by trait resolution at its instantiation, never by reading its tokens at expansion: expansion sees a type's spelling, not the type.
