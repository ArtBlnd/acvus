# programs

A program is written in one of three kinds of source.

The sources of a graph are compiled together. Each script and each template is one function, and each module declares functions; every one stands at a name the embedding gives it, a path of any depth. A unit reaches another only by that name, and nothing is named but by the embedding. A run starts at any function the embedding chooses, instantiated at the types the embedding declares for its inputs and result (host).

## 1. Template (`acvt`)

Text with values interpolated in `{{ }}`, each formatted by `core::display`, and control lines that begin with `%`. Lambdas are a script's. A template is one function. Its parameters are the inputs it reads, each `$name` a parameter of that name, which a caller passes by name; its result is the `String` it builds. So a `return` or a `?` that would leave the template itself is refused: what it leaves with is not that `String`. Inside a lambda, either leaves the lambda only. Its types are decided as any body's are, by [the type system](../transparency-constraint-type-system.md) (types §1): another body sees its text separately at each use, so each use of the template records an instance of its text, whose parameter and result classes are unified with the use's arguments and result (O26), and its text is recorded once more, with every class it names made anew, and settled to the fixpoint without occurrences and without fallback (O31).

```acvt
You talk with {{ $user }}.
% match $focus
% Focus::User =>
Keep to {{ $user }}.
% Focus::Char =>
Keep to your own character.
% Focus::Custom(name) =>
Keep to {{ name }}.
% end
```

## 2. Script (`acvus`)

Statements and a tail. Lambdas in full. A script is one function, as a template is (§1): its parameters are the inputs it reads, passed by name, and its result is its tail's value; its types are decided as a template's are (§1). A caller reads its result with its effect.

```acvus
@big = 0;
let s = 0;
for x in &@xs {
    s = s + *x;
    if *x > 4 {
        @big = @big + 1;
    };
}
(s, @big)
```

## 3. Module (`acvum`, to-be)

Declarations: `fn`, `struct` and `impl`. Lambdas only inside a `fn`. No concrete type is written, and a module does not refer to any context `@x`. Seen from outside the module, its struct is opaque in its parts: its fields are named, and its values made, only inside the module's `impl` blocks (the type system, Transparency and Types).

An `acvum` file holds modules. Each `struct` with its `impl` blocks is one module, and each free `fn` is one. A module is the only source that declares a `fn`.

No type is declared. A `struct` states only its shape, the names of its fields. Its type is `Self⟨T̄⟩`, with one type parameter for each field, in the order the struct declares them (the type system, Types). A `fn`'s types are decided as a template's are (§1).

Free fns and methods that reach one another by calls form one binding group, and a lambda is a binding group of its own, lying inside the binding group of its enclosing body (the type system, Compilation graph). A use that lies in the binding group of the body it reaches, or in an instance of that body, records no instance, and takes the classes of that body's text, or of the instance it lies in (O27); every other use of another body records an instance of it (O26). `Self` is the module's own `struct`, and a struct's module is in its constructor (the type system, Types).

`impl` blocks are Rust's. A method's receiver is written `self`, `&self` or `&mut self`, so it is `Self` or a reference to it from the start. A method call's receiver is adjusted as types §6 states.

The units of a graph are compiled together, so a use of a module's `fn` sees its text, separately at each use (the type system, Transparency).

```acvum
struct Point { a, b }

impl Point {
    fn add(&self, other) { }
}
```

## 4. Inputs

An input `$x` is a parameter of the function its source is (§1, §2). The program never assigns it.
