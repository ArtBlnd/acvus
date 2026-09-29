# macros

An extern author writes Rust and two macros turn it into declarations and glue: an attribute on a function, and a derive on a type. Nothing else reaches the crossing: the crossing traits are sealed, and only the boundary and the derive implement them. No `macro_rules!` emits a crossing implementation into a user crate.

## 1. The function attribute

The attribute reads the function's Rust signature and its arguments and emits the declaration (compiler: extern fn) and the glue.

- A parameter is taken by value, by `&T` or by `&mut T`. A sequence is taken as a slice, `&[T]` or `&mut [T]`, never as an array.
- A result is a value; a sequence is returned as a `Vec<T>`. A result that borrows borrows only what was lent in place (crossing §4).
- A closure parameter is a callback, callable any number of times. A callback is called synchronously or awaited (crossing §3).
- An `async fn` is an extern that suspends. An extern declared heavy runs off the script's thread where the interpreter places it (interpreter: tasks).
- Declared facts are written in the attribute (declared facts). A declared fact about a place (compiler: extern fn §3, access and action) is written on the parameter whose storage it speaks of; a relation and a source are written on the function.
- A closure parameter's function type may hold any loan, so a callback may be any function value the program makes. An author who keeps a callback past the call declares it holding none (limitations §3).

The glue it emits contains no `unsafe`.

## 2. The derive

The derive makes a Rust type an opaque registered type (types §2). It reads the definition and emits exactly the structural facts about it that no stable Rust bound can state (safety §2), and nothing else. It emits no projection types: an opaque type crosses as itself. An object or an enum an extern takes or gives needs no derive: its shape is written in the declaration, and the extern reads and builds it through the runtime's aggregates (types §1).

## 3. What the macros emit, and what they do not

The macros' output is safe code plus the derive's structural signed implementations. A source scan counts production `unsafe` with generated items included; the count is those structural implementations.
