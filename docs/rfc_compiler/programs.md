# programs

A program is written in one of three kinds of source.

The sources of a graph are compiled together. Each script and each template is one function, and each module declares functions; every one stands at a name the embedding gives it, a path of any depth. A unit reaches another only by that name, and nothing is named but by the embedding. A run starts at any function the embedding chooses, instantiated at the types the embedding declares for its inputs and result (host).

## 1. Template (`acvt`)

Text with values interpolated in `{{ }}`, each formatted by `core::display`, and control lines that begin with `%`. Lambdas are a script's. A template is one function. Its parameters are the inputs it reads, each `$name` a parameter of that name, which a caller passes by name; its result is the `String` it builds. Its parameters' types are inferred within the template alone, by the transparency rule (types §1): nothing outside it takes part in that inference, and a call only instantiates what the template settled. So a `return` or a `?` that would leave the template itself is refused: what it leaves with is not that `String`. Inside a lambda, either leaves the lambda only.

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

Statements and a tail. Lambdas in full. A script is one function, as a template is (§1): its parameters are the inputs it reads, passed by name, and its result is its tail's value; its types are inferred within the script alone, and a call only instantiates what the script settled. A caller reads its result with its effect.

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

Declarations: `fn`, `struct` and `impl`. Lambdas only inside a `fn`. No concrete type is written, and a module does not refer to any context `@x`. Seen from outside the module, its structs are opaque: by the transparency rule, the type of `Self` does not change, and nothing outside adds a field to it.

An `acvum` file holds modules. Each `struct` with its `impl` blocks is one module, and each free `fn` is one. A module is the only source that declares a `fn`.

No type is declared. A `struct` states only its shape, the names of its fields. Its field types, and every `fn`'s scheme, are inferred within the module alone: nothing outside the module takes part in that inference, and a use outside only instantiates what the module settled.

A module's `fn`s are polymorphic. What a body leaves unconstrained is generalized, modules whose `fn`s call one another are generalized together, and each use instantiates the result. `Self` is not generalized: it is the module's own `struct`, one opaque type by the transparency rule.

`impl` blocks are Rust's. A method's receiver is written `self`, `&self` or `&mut self`, so it is `Self` or a reference to it from the start. These receivers anchor inference: a use that does not fit a method is refused where it meets that method's declared receiver.

The units of a graph are compiled together, so a use of a module's `fn` sees its scheme and its body.

```acvum
struct Point { a, b }

impl Point {
    fn add(&self, other) { }
}
```

## 4. Inputs

An input `$x` is a parameter of the function its source is (§1, §2). The program never assigns it.
