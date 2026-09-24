# Acvus

A statically typed scripting language, embedded in Rust. Scripts,
templates and single expressions share one compiler: types are inferred
from use, host data enters as `@context` values, and functions the host
registers from Rust are called like any other. The compiler lowers to an
SSA form, optimizes, and validates the result before a register machine
runs it.

## Run

```sh
cargo run -p acvus-cli -- run   script.acvus  [name=literal]... [--opt none|full] [--time]
cargo run -p acvus-cli -- run   -e '[1, 2, 3] | map(|x| -> x * 2) | sum'
cargo run -p acvus-cli -- check script.acvus [--json] [--opt none|full] [--time]
cargo run -p acvus-cli -- mir   script.acvus [--json] [--opt none|full] [--time]
cargo run -p acvus-cli -- ops   script.acvus [--json] [--opt none|full] [--time]
```

A `.acvus` file is a script, a `.acvt` file a template; both go through the
same stages. `name=literal` binds the input `$name` to a value written in
the script's own syntax.

A script that names an `@context` runs in a space, which `acvus ctl` keeps:
the space holds its scripts, one init per context, and the contexts
themselves. A context's type is what the space's scripts and its init solve
it to; its first value is its init, run when a run fetches the context and
the space lacks it.

```sh
acvus ctl use work                          # a ctl context: names spaces
acvus ctl space add notes dir:notes         # map a space to a directory
acvus ctl space add-script notes turn.acvus # the space keeps the source
acvus ctl space init notes log -e 'deque()' # the init of @log
acvus run turn --space notes                # fills @log from its init, runs, commits
acvus ctl space fill notes                  # runs every init the space lacks
acvus ctl space ls notes                    # scripts, inits and contexts
```

`--space` names a space, or `acvus ctl space mark <space>` names it for
every command under the working directory. `acvus ctl` with no arguments
lists its commands.

Each command stops where its job stops. `check` and `mir` run parse,
typecheck, lowering, optimization and validation: `check` prints nothing,
`mir` prints the optimized program. `ops` and `run` add the interpreter's
`prepare`: `ops` prints the operations it prepares, `main` and every closure
body, block by block, and `run` executes them. A program `check` admits
reaches the machine: a field no path stores,
a result that is or holds a reference, and a type the solve leaves open are
all refused by `check`, with a diagnostic.

`--opt` picks how hard the compiler works. `full`, the default, runs every
pass; `none` runs only what a program needs to reach the machine at all, so
`mir --opt none` prints the program the source wrote -- a loop's invariants
still inside its body. Which programs are refused does not move with the
level: the move, borrow and exhaustiveness checks and the validator run at
both, and `acvus-interpreter-test/tests/differential.rs` runs the corpus at
both levels and compares the values.

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

`--time` adds, after everything else the command printed, one line per
stage it ran:

```
time: compile 1.284 ms at opt full (parse 0.112, typeck 0.731, lower 0.201, optimize 0.240)
time: prepare 0.318 ms
time: run     42.907 ms
```

`check` and `mir` report `compile` alone, `ops` adds `prepare`, `run` all
three. `compile` is the whole of `check`, of which parse, typechecking,
lowering and optimization are the named parts; `run` is the machine and
nothing around it, so a script's own `print` is inside it and reading or
committing the context file is not. The lines go to stderr; under `--json`
they are a trailing `{"time": …}` object on stdout instead, the same
milliseconds as numbers. Without the flag no clock is read.

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
- `//` comments to the end of the line, in a script and inside a `{{ }}`
  tag. There is no block comment.
- `print(s)` writes one line to stdout while the script runs; the result
  line follows everything it printed.
- Regular expressions and dates are in the set `acvus run` registers:
  `regex`, `is_match`, `find`, `captures`, `named`, `replace_all`, and
  `parse_date`, `format_date`, `timestamp`, `add_days`.

## Expressions and pipes

```
@items | filter(|x| -> x.active) | map(|x| -> x.name) | join(", ")
```

`a | f(b)` is `f(a, b)`. Lambdas are `|args| -> expr`. The same
expression language runs inside a template's `{{ }}`.

## Templates

A template is a script with one extra rule (RFC-0071): a line that does not
begin with `%` is text, appended to the result as written. A line whose
first non-blank character is `%` is one statement of the script grammar,
without its `;` and without block braces, and `% end` closes what it opened.

```
Hello, {{ &@name }}!

% if @language == "korean"
한국어로 답변합니다.
% else
Responding in English.
% end

% for item in &@items
- {{ &item.name }}: {{ item.value.to_string() }}
% end
```

A text line carries its newline; one ending in `\` does not, and one
beginning with `%%` writes a single `%`. `{{ expr }}` inside a text line is
the format string a script already writes, and `//` inside a tag comments
out the rest of its line. Inline branching is the expression grammar's:
`{{ if c { "a" } else { "b" } }}`.

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
acvus-ext-net       HTTP
pomollu-core        TOML specs compiled into the same graph
acvus-lsp           language server
acvus-cli           `acvus run | check | mir | ops | ctl | lsp`
```

`acvus-mir` knows nothing about the interpreter; `acvus-interpreter`
depends on it, never the reverse.

## Examples

Each directory under [`examples/`](examples/) holds `main.acvus`, the
init of each context it reads under `inits/`, and the `expected.txt` it
prints:

```sh
acvus ctl space add collatz dir:collatz-space
acvus ctl space add-script collatz examples/collatz/main.acvus
acvus ctl space init collatz start -f examples/collatz/inits/start.acvus
acvus run main --space collatz
```

```
collatz     the Collatz walk of a context number: `while`, and `if` as a value
grades      objects in a context array: `for s in &@students`, and `map | filter | fold`
word-count  `split_whitespace`, `lower`, counts in a `HashMap`, the top three by count then bytes
log-parse   log lines cut by one regex's named groups: a tally per level, and the span in seconds between the first and last timestamp
shapes      a structural enum, an exhaustive `match`, a `Result` per entry carried out by `?`
ledger      money as whole cents: a running balance and the largest debit, no float
queue       a `deque` as a work queue: `push_back`, `pop_front`, and the order out
```

`acvus-cli/tests/examples.rs` runs each of them and compares stdout to
`expected.txt` byte for byte.

## Documents

Design decisions are RFCs under [`docs/rfcs/`](docs/rfcs/README.md).

## License

Free to use, copy, modify and distribute; not to be sold as a product or
service. See [LICENSE](LICENSE).
