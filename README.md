# Acvus

A statically typed scripting language, embedded in Rust. Scripts,
templates and single expressions share one compiler: types are inferred
from use, host data enters as `@context` values, and functions the host
registers from Rust are called like any other. The compiler lowers to an
SSA form, optimizes, and validates the result before a register machine
runs it.

## Run

```sh
cargo run -p acvus-cli -- run   script.acvus  --context ctx.json
cargo run -p acvus-cli -- run   -e '@items | map(|x| -> x.name) | join(", ")' --context ctx.json
cargo run -p acvus-cli -- check script.acvus [--json]
cargo run -p acvus-cli -- mir   script.acvus [--json]
cargo run -p acvus-cli -- ops   script.acvus [--json]
cargo run -p acvus-cli -- space store/
```

`--context` is a JSON file whose keys are the script's `@names`. A `.acvus`
file is a script, a `.acvt` file a template; both go through the same
stages.

Each command stops where its job stops. `check` and `mir` run parse,
typecheck, lowering, optimization and validation: `check` prints nothing,
`mir` prints the optimized program. `ops` and `run` add the interpreter's
`prepare`: `ops` prints the operations it prepares, `main` and every closure
body, block by block, and `run` executes them. What only `prepare` refuses —
an `@name` no context declares — therefore reaches `ops` and `run` alone.

A diagnostic is one `error: <message>` line, then the file, line and column,
the source line and a caret under the span. A refusal whose story needs a
second place carries a label there: that line too, with `---` under the
span and the label's words after it, in source order, the lines between
elided with `...`. A label with no place of its own is a `= help:` line.
`--json` puts them on stdout instead, as an array of `{severity, message,
path, line, col, span, labels}`, each label `{line, col, span, text}` — the
span is `[start, end]` in bytes, and `line`, `col` and `span` are `null`
where the failing stage has no span to give. Under `--json` every byte on
stdout is JSON: `check` and `mir` print the array alone, `ops` prints its
listing where it has one.

Exit status: `0` success, `1` a diagnostic, `2` a refusal at run time, `64`
a usage error.

## Scripts

```
let count = 0;
let it = as_iter(&@items);
while let Some(item) = next(&mut it) {
    if item.active {
        count = count + 1;
    };
}
let label = if count > 10 { "many" } else { "few" };
label
```

- `let x = expr;` binds; `x = expr;` assigns an existing binding. A block
  ends with the value of its last expression.
- `if`, `if let`, `while`, `while let`, and `match`:

  ```
  match shape {
      Shape::Circle(r) => { 3.14 * r * r },
      Shape::Square(s) => { s * s },
  }
  ```

  A `match` must be exhaustive; `_` is the catch-all. Variants need no
  declaration — writing `Shape::Circle(r)` introduces it.
- `Option` and `Result`: `Some(x)` / `None`, `Ok(x)` / `Err(e)`, and `?`
  to return early on `None` or `Err`.
- Containers: `vec([])`, `push`, `pop`, `len`, `v[i]` (a `u64` index),
  `as_slice`. Iterators: `as_iter`, `next`, `map`, `filter`, `fold`,
  `sum`, `collect`, and the pipe form below.
- `expr as T` converts between numbers — the eight integer widths and
  `f64` — with Rust's `as` values: `i as f64`, `n as u8`, `x as i64`.
- A literal can say its own type: `10u64` at a width, `'c'` a `char` (one
  Unicode scalar value), `b"GET"` an `Array<u8, 3>`, `b'G'` a `u8`.
- References: `&x` and `&mut x` borrow; a value passed by value moves.
  Use after move is a compile error.

## Expressions and pipes

```
@items | filter(|x| -> x.active) | map(|x| -> x.name) | join(", ")
```

`a | f(b)` is `f(a, b)`. Lambdas are `|args| -> expr`. The same
expression language runs inside a template's `{{ }}`.

## Templates

```
Hello, {{ @name }}!

{{ "korean" = @language }}
한국어로 답변합니다.
{{ _ }}
Responding in English.
{{ / }}

{{ item in @items }}
- {{ item.name }}: {{ item.value | to_string }}
{{ / }}
```

`{{ pattern = value }} … {{ / }}` matches; `{{ x in list }} … {{ / }}`
iterates; `{{-- … --}}` is a comment.

## Types

Every value has a static type; none is written in a script. `@items`'s
type comes from the host's context, a lambda's from its use, a literal's
from itself. Objects are structural — `{ name: String, age: i64 }` is a
type, and any value with those fields has it. Effects are part of a
function's type: a call that reaches outside the program (IO, an LLM,
a heavy computation) is known to the compiler, and independent such
calls run concurrently without the script saying so.

## Extending from Rust

A Rust function becomes a script function with one attribute; a registry
groups them under a namespace.

```rust
use acvus_extern::{extern_fn, extern_registry, Registry};
use acvus_interpreter::AcvusRuntime;

#[extern_fn(effect = pure)]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

let registry: Registry<AcvusRuntime> = extern_registry! {
    ns: "math",
    fns: [add],
};
```

The script calls `add(10, 32)` or `math::add(10, 32)`. `effect` states
what the function does (`pure`, or an effect the compiler must order);
`heavy` marks a computation to run on a worker thread; `async fn` marks
IO. A `#[state]` parameter carries Rust state into the function.

## Crates

```
acvus-ast           parser: template, script, expression
acvus-mir           types, inference, SSA lowering, optimization, validation
acvus-interpreter   the register machine
acvus-extern        the Rust-side ABI: values, ownership, registries
acvus-ext           the standard library
acvus-ext-llm       OpenAI, Anthropic, Google providers
acvus-ext-net       HTTP
acvus-orchestration TOML specs compiled into the same graph
acvus-lsp           language server
acvus-cli           `acvus run | check | mir | ops | space`
```

`acvus-mir` knows nothing about the interpreter; `acvus-interpreter`
depends on it, never the reverse.

## Documents

Design decisions are RFCs under [`docs/rfcs/`](docs/rfcs/README.md).
Runnable projects are under [`examples/`](examples/).

## License

Free to use, copy, modify and distribute; not to be sold as a product or
service. See [LICENSE](LICENSE).
