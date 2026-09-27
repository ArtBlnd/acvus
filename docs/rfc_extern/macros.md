# macros

An extern author writes Rust and two macros turn it into declarations and glue: an attribute on a function, and a derive on a type. Nothing else reaches the crossing: the crossing traits are sealed, and only the boundary and the derive implement them. No `macro_rules!` emits a crossing implementation into a user crate.

## 1. The function attribute

The attribute reads the function's Rust signature and its arguments and emits the declaration (compiler: extern fn) and the glue.

- A parameter is taken by value, by `&T` or by `&mut T`. A sequence is taken as a slice, `&[T]` or `&mut [T]`, never as an array.
- A result is a value; a sequence is returned as a `Vec<T>`. A result that borrows borrows only what was lent in place (crossing §4).
- A closure parameter is a callback, callable any number of times. A callback is called synchronously or awaited (crossing §3).
- An `async fn` is an extern that suspends. An extern declared heavy runs off the script's thread where the interpreter places it (interpreter: tasks).
- Declared facts are written in the attribute (declared facts).

The glue it emits contains no `unsafe`.

## 2. The derive

The derive makes a Rust struct or enum an extern-defined type (types). It reads the definition and emits exactly the structural facts about it that no stable Rust bound can state (safety §2), and nothing else. It emits no projection types: an extern-defined type crosses as itself.

## 3. What the macros emit, and what they do not

The macros' output is safe code plus the derive's structural signed implementations. A source scan counts production `unsafe` with generated items included; the count is those structural implementations.
