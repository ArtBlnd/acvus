# programs

A program is written in one of three kinds of source.

## 1. Template (`acvt`)

Text with values interpolated in `{{ }}`, each formatted by `core::display`, and control lines that begin with `%`. Lambdas are a script's. A template's result is the `String` it builds. So a `return` or a `?` that would leave the template itself is refused: what it leaves with is not that `String`. Inside a lambda, either leaves the lambda only.

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

Statements and a tail, whose value is the run's. Lambdas in full. The run's result enters the result type the embedding declares; where none is declared, it is inferred, and a caller in the graph reads it with the run's effect.

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

No type is declared. A `struct` states only its shape, the names of its fields. Its field types, and every `fn`'s scheme, are inferred within the module alone: nothing outside the module takes part in that inference, and a use outside only instantiates what the module settled.

A module's `fn`s are polymorphic. What a body leaves unconstrained is generalized, `fn`s that call one another are generalized together, and each use instantiates the result. `Self` is not generalized: it is the module's own `struct`, one opaque type by the transparency rule.

`impl` blocks are Rust's. A method's receiver is written `self`, `&self` or `&mut self`, so it is `Self` or a reference to it from the start. These receivers anchor inference: a use that does not fit a method is refused where it meets that method's declared receiver.

The units of a graph are compiled together, so a use of a module's `fn` sees its scheme and its body.

```acvum
struct Point { a, b }

impl Point {
    fn add(&self, other) { }
}
```

## 4. Inputs

An input `$x` is a value the host passes into the run. The program never assigns it.
