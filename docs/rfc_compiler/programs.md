# programs

A program is written in one of three kinds of source.

## 1. Template (`acvt`)

Text with values interpolated in `{{ }}`, each formatted by `core::display`, and control lines that begin with `%`. Lambdas are limited.

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

Statements and a tail, whose value is the run's. Lambdas in full.

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

## 3. Model (`acvum`, to-be)

Declarations: `fn`, `struct` and `impl`. Lambdas only inside a `fn`. No concrete type is written, and a model does not refer to any context `@x`. Seen from outside the model, its structs are opaque: by the transparency rule, the type of `Self` does not change, and nothing outside adds a field to it.

```acvum
struct Point { a, b }

impl Point {
    fn add(&self, other) { }
}
```

## 4. Inputs

An input `$x` is a value the host passes into the run. The program never assigns it.
