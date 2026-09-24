# Syntax

The surface a script and a template are written in. This document decides how
a call names its callee, how a statement binds or assigns, how a failure is
written and propagated, how a value is converted, how a literal states its
type, and what a template is.

## RFC-0030: A qualified call names a namespace; a method call is a call on its receiver

Status: Accepted

1. `a::b(args)` is a call of the function `b` in namespace `a` when the
   registries declare one (RFC-0021); otherwise it is the structural variant
   `b` of the enum `a` with `args` as its payload. `a::b` without arguments is
   a variant. A bare name resolves by RFC-0043.
2. `recv.f(args)` is the call `f(recv', args)`: `f` resolves as the bare name
   `f` would, and `recv'` is `&recv`, `&mut recv` or `recv` as the callee's
   first parameter asks (RFC-0043 chooses it per candidate); a borrowed
   receiver must be a place. The call is checked, lowered and borrow-checked
   as the call it stands for.
3. `recv.f` without parentheses is a field; a closure held in a field is
   called as `(recv.f)(args)`.

**Why.** Namespaces are closed when the registries are combined, so the
checker tells a namespace from an enum name by looking, and the expression
means one thing in the checker and in the lowering. The method form adds only
the reference the callee's own signature asks for.

**Rejected.**
- Method resolution by receiver type — `f` is found by name; the receiver
  picks an instance only through the call's type, as any argument does.
- Auto-deref chains — `r.f()` with `r: &T` and `f` taking `T` is a reference
  where a value is needed, reported as such.

## RFC-0045: `let` binds, `x = e;` assigns

Status: Accepted

There is one statement grammar, and it is the grammar of every block: a
script's top level, a lambda's block body, a `while`/`while let`/`for`/
`anyorder` body, an `if`/`if let`/`else` block, and a template's `%` lines
(RFC-0071).

- `let x = e;` introduces a binding. It shadows any outer `x` and ends with
  the block that introduced it.
- `let x;` introduces an uninitialized binding; the init check owns it.
- `x = e;` assigns the `x` in scope and never introduces one. The expression's
  type unifies with the binding's.
- An assignment to a name with no binding in scope is a compile error:

  ```
  cannot assign to `x`: no binding named `x` is in scope; `let x = ...;` binds it
  ```

- An assignment inside a lambda to a name bound outside it is a compile error,
  because captures are by value (RFC-0018):

  ```
  cannot assign to `x`: it is captured by the lambda, not bound in it
  ```

- A name assigned inside a nested body is the binding the enclosing block
  introduced; the join carries the new value as a block parameter.

Assignments to places — `@x = e;`, `a.b = e;`, `a[i] = e;`, `*r = e;` — are
statements of the same rule. There is one parser entry for scripts; what a
file is (script, template, expression) chooses the pipeline, not the grammar.

A statement ends in `;` or at the `}` of a block. `while`, `for` and
`anyorder` end at their `}`, and a `;` after it is refused. An `if` or a
`match` that begins a statement is the whole statement, as in Rust: it ends
at its `}`, and no operator continues it. Without a `;`, it is the body's
tail when the body's `}` or the script's end follows it, and a statement
whose value is dropped otherwise; with a `;` it is a statement.

**Why.** `let` is the only way to introduce a name, so a reader knows every
name a body introduces by reading its `let`s, and the writer of `x = e;` knows
the store lands where the visible `x` was bound.

**Rejected.**
- Bare `x = e;` binds where no `x` is in scope and assigns where one is — it
  is implicit: a typo becomes a new binding, and a `let` moved or removed
  silently turns a store into a shadow.
- A `;` required after an `if` or `match` statement — a statement that ends in
  a block read as an unfinished expression, as a Rust reader does not expect.
- A `;` refused after an `if` or `match`, as after a loop — a body could no
  longer end in one whose value it drops.
- A block as such a statement — a statement that begins with `{` reads on,
  and `{ [5, 6] }[1]` would become `[1]`.

## RFC-0038: `Result<T, E>` is a primitive, `?` widens the error, and a trap is not an error

Status: Accepted

1. `Result<T, E>` is a language type beside `Option<T>`. `Ok(x)` and `Err(e)`
   build one, the patterns `Ok(v)` and `Err(e)` match one, and it crosses the
   extern boundary as Rust's `Result` (RFC-0039). A script that builds a
   `Result` must let both sides be known.
2. `!` is the type with no value, below every type: a `Result<T, !>` goes
   wherever a `Result<T, E>` is expected. A type variable nothing constrained
   closes to `!` at the final freeze only; the freeze that asks whether a
   declared type is fully known keeps refusing an open variable. A slot
   declared `!` is RFC-0054 rule 5.
3. `x?` on `x: Result<T, E1>` in a function returning `Result<U, E2>` yields
   the `T` and unifies `Result<_, E1>` with the return type. With structural
   error enums, that unification is the solver's enum merge: the function's
   error enum grows the variants of every `?` in its body, and nothing
   declares that growth. `x?` on an `Option` in a function returning an
   `Option` propagates `None`.
4. A function is a script or a lambda, and the tail flows into its return type
   as `?` does. In a function whose return is neither a `Result` nor an
   `Option`, `?` is a type error; in a template, which has no function to
   return from, `?` is refused. An operand whose type nothing has fixed when
   `?` is applied is taken to be a `Result`.
5. A failure the program can act on is a value: an extern fn returns
   `Result<T, E>` with `E` its own error type, usually an enum of its failure
   modes, and the script matches or `?`s it. A failure the program cannot act
   on is a trap, and a trap is RFC-0044's run-time failure. Every `Result` an
   extern returns is the language's.

**Why.** Different failures have different next steps — a rate limit with a
retry-after, a context overflow, a refused schema — and one error type
flattens them into a message the program can only print. `?` with structural
merge writes the caller's error type as the union of what it called. A trap
is not a value because it has no reader: the script cannot proceed from a
contract the runtime broke.

**Rejected.**
- One error type (`String`) for every failure — flattens distinct next steps.

## RFC-0049: `expr as T` is Rust's `as`, and inside a chain it is a leaf

Status: Accepted

1. **Syntax.** `expr as T`, where `T` is an integer width (RFC-0037), `f64` or
   `char` (RFC-0058 rule 2). Postfix, left-associative, binding tighter
   than every binary operator and looser than the unary ones, as in Rust:
   `1.0 + 1 as f64` is `1.0 + (1 as f64)` and `-x as u8` is `(-x) as u8`.
2. **Semantics are Rust's `as`, exactly.** Integer to integer truncates or
   sign-extends; integer to `f64` rounds to nearest; `f64` to an integer
   saturates at the width's ends and maps `NaN` to zero; `f64` to `f64` is the
   identity. `as` is total: no cast traps, so no pass has to keep one on the
   path the program wrote.
3. **A cast is a leaf, never a node.** Inside an arithmetic chain a cast into
   the chain's type is absorbed into the read of the leaf below it and adds no
   operator node, so a chain stays at one type. Outside a chain a cast is one
   word operation. Whether a chain reads its leaves plainly or through casts
   is settled once per chain, at preparation.
4. **A numeric conversion is not a call.** No extern performs a conversion
   `as` performs.
5. **The fold applies the same semantics.** A cast of a constant folds with
   the same Rust `as` the machine runs (RFC-0055); a cast of a non-constant,
   including every `NaN`, stands.

The MIR instruction `Cast { dst, src, to }` names the target alone: `src`'s
type is the one every reader holds, and a second copy could disagree with it.
It is not a declared user-type coercion, which is a call.

**Why.** A conversion that is a call costs an argument window and a dispatch
per use and hides the chain's operands from the recognizer.
**Cost.** One test of the chain's read mode, settled at preparation so a chain
with no cast pays nothing per leaf.

**Rejected.**
- `as` as extern signatures per width — still a call, still opaque to the
  recognizer.
- `float` and `int` as spellings — two names for one type.
- A cast as a chain node — a chain's interior would no longer be one type, and
  the family would grow with every pair a program casts between.
- The leaf's type as a parameter of the chain operation — it multiplies the
  chain operation family.
- `f32` — the language has none.

## RFC-0058: A literal says its type

Status: Accepted

1. **A suffixed integer literal has that width.** `10u64`, `-1i8`, `255u8`,
   the suffix one of the eight width names. An unsuffixed literal keeps
   RFC-0037 rule 4: the use decides, `i64` where nothing does. A suffixed
   literal's value is checked against its width at the literal, with the
   message `literal 300 does not fit u8`, and never wrapped. There are no
   underscore separators.

   A `-` that touches an integer literal is part of it when the token before
   the `-` does not end a value, so every minimum is writable: `-128i8`,
   `-9223372036854775808`. `f(-1)`, `[1, -1]`, `x = -1` and `2 * -1` carry a
   negative literal; `- 128i8`, `-(128i8)`, `-x` and `-1.5` are negations, and
   `a -1` is a subtraction. The lexer decides it by one token of lookahead, and
   every token states whether it can end an expression.
2. **`char` is a type: one Unicode scalar value**, Rust's `char`, crossing the
   boundary as one. A literal is `'c'` with Rust's escapes; a `'…'` holding no
   scalar value or more than one is a compile error. A `char` compares with
   `==` and `<` by scalar value. `char as T` is admitted for every integer
   width, `u8 as char` is the one cast into a `char`, and there is no
   `char as f64`.
3. **`b"…"` is a byte-string literal of type `Array<u8, N>`**, RFC-0012's
   array literal. Its text is ASCII and Rust's escapes, `\xNN` reaching
   `0xFF`; a non-ASCII character inside is a compile error naming it. `b'x'`
   is a `u8` literal.
5. **Nothing new runs.** The fold reads a suffixed constant at its width, and
   the cast reads a `char` as a leaf; no MIR instruction or machine operation
   is added. The lowerer desugars a suffixed integer to its value and a byte
   string to its bytes, so the MIR has one spelling per constant.
6. **One escape table and one tokenizer.** `"…"`, `'c'`, `b'x'` and `b"…"`
   share Rust's escape table — `\n`, `\r`, `\t`, `\0`, `\\`, `\'`, `\"`,
   `\xNN` up to `\x7F` (`0xFF` in bytes), `\u{…}` — and an escape outside it
   is a compile error at the literal naming the escape. A `"…"` holding
   `{{ expr }}` is a format string: its text and its tags are joined by
   string `+`, so nothing is converted to text implicitly. A tag's content is
   tokenized by the same pass as any expression, so the sign of rule 1 folds
   inside it and a string or character literal inside it may hold `{{`, `}}`
   or `}`; a `{{` never closed is two characters.

There is no float suffix.

**Why.** A constant the use cannot decide, such as a `u64` index, needed a
cast written where a constant was meant. A character had no type, so it had a
representation that could be wrong.

**Rejected.**
- `char` as a one-character `String` — a heap allocation per character, and
  every conversion carries a `Result` for a case the type should exclude.
- `b"…"` as a `Vec<u8>` — allocates at every evaluation and loses the length
  from the type.
- Wrapping an oversize suffixed literal — it would make the suffix a
  truncation operator.
- `u32 as char` — not a Rust `as`: most `u32`s are not scalar values.
- The sign as a grammar production — 25 LALR ambiguities, because the parser
  must choose before any location can be read.
- The sign left to the checker — the value the checker reads would differ from
  the one the literal spells, and every reader below would reassemble the
  pair.
- A suffix on a float literal — `f64` is the only float.

## RFC-0071: A template is a script whose text lines are output

Status: Accepted

A template file (`.acvt`) is a script with one extra rule: a line that does
not begin with `%` is text, appended to the result as written. A line whose
first non-blank character is `%` is one statement of the script grammar,
without its trailing `;` and without the braces that open or close a block. A
block opened on a `%` line is closed by `% end`. The template's value is the
accumulated text, a `String`.

    You are a careful assistant for {{ $name }}.

    % if $mode == "review"
    Review the code below and list defects.
    % else
    Help with the code below.
    % end

    % for m in &$journal
    - {{ &m.role }}: {{ &m.content }}
    % end

    % match $mode
    % "review" =>
    Review the code.
    % _ =>
    Help.
    % end

    {{ rules() }}

1. **A `%` line is a statement and leaves nothing in the output**, its newline
   included. The statements are the script's: `let`, assignment and stores,
   `if` / `else if` / `else`, `if let`, `match` with one `% pattern =>` line
   per arm, `for` with RFC-0057's heads, `while`, `while let`, `anyorder`,
   `break`, `continue`, and an expression statement. `% //` is a comment. The
   `%` may be indented.
2. **A text line is appended as written**, with its newline. A text line that
   ends in `\` is appended without that newline. A text line that begins with
   `%%` is appended with one `%` in its place. There is no other whitespace
   rule.
3. **`{{ expr }}` in a text line is the format string of RFC-0058 rule 6.**
   The expression is a `String` or a `&str`; nothing is converted to text
   implicitly, and `{{ "{{" }}` writes a literal `{{`.
4. **`$name` is an input the host injects, shared by the whole graph.** It is
   not a parameter of the template: `{{ rules() }}` passes nothing, and `$lang`
   inside `rules` is the same injected value as in its caller: a call to a
   body whose reachable code reads `$lang` passes the caller's `$lang`, after
   the call's own arguments. `@name` is a context, as in a script. There is no include and no parameter declaration:
   the inputs a template requires are the `$` names its reachable code reads,
   and the analysis reports them at their types. Binding a `$` is a graph
   update that re-infers only the functions that read it.
5. **A bound `$` is a constant, and what it makes unreachable is `!`.**
   Binding `$mode = "review"` folds `% if $mode == "review"`; the arms it
   leaves behind are typed `!`, a `$` read only there closes to `!` at the
   freeze (RFC-0038), and a `$` typed `!` is not required. Binding one input
   therefore narrows the set still required and never widens it. A template
   whose value is `!` appends nothing and is absent from what its caller
   assembles.

A template has no function to return from: `return` and `?` are refused in
it. Inline branching is an expression, `{{ if c { "a" } else { "b" } }}`.

**Why.** A template is string composition, which the script already lowers,
analyzes and optimizes; with a text line lowered as an append, every analysis
of the script applies to a template unchanged, and a second grammar cannot
drift from the first. Control flow on its own line removes the whitespace
question: a `%` line owns its whole line and nothing else, and `\` covers the
one newline left. A match is a table, and a text line is already a row. `{{ }}`
is the format string a script already writes.

**Rejected.**
- Inline block bodies (`@if[c]{text}{text}`) — `{ }` would mean text in one
  position and a block or object in another, and `@` is the context sigil.
- Trim markers and trim options — a `%` line and `\` cover the need; a second
  mechanism reopens the question.
- Include or inheritance — a template is a function; composition is a call.
- Per-block closers (`endif`, `endfor`) — every block closes with `% end`.
- `${ }` for the inline expression — `{{ }}` is already the script's format
  string.
- Inline template forms (`{{ pattern = expr }}`, `{{_}}`, `{{/}}`,
  `{-{ }-}`) — `%` lines replace them.

## RFC-0087: A bound `$` holds any value a literal writes, and its uses decide its enums

Status: Proposed

1. **A binding's value is a literal's value**: an integer, `i64` or the width
   its suffix names (RFC-0058 rule 1), a float, a `bool`, a `char`, text, a
   byte string, `()`, and an array, a tuple, an object, a structural enum's
   variant, `Some` or `None`, `Ok` or `Err` holding any of these. An object
   holds each field once.
2. **Its type is the type its literal expression gets, with two
   differences.** Text is `&str` as the input itself, where it stands in
   place of `"…"`, and `String` inside a value, where a script writes
   `.to_string()`, since data holds no reference (RFC-0062 rule 5). Every
   structural enum in it is a variable while the body is checked, as the type
   of an unbound `$` is; after the body is checked, the variant the value
   holds joins that variable.
3. **A value with no type is refused when it is bound**, before any body is
   checked: array elements that have no one type, an object wider than an
   object type admits, a suffixed integer that does not fit its width. The
   bound value is typed in every body that reads it, and never fails there.
4. **A use the value cannot meet is refused at the body.** Where the uses
   typed an enum position while the body was checked — a branch it meets, a
   variable it is stored in — a variant or payload type the value contradicts
   is refused at the input's first read, naming the input and its value. A
   use the solve settles — a pattern's literal, an operator, a call — meets
   the value's type and is refused at that use, in the use's own words. A
   field the body reads and the object lacks is refused where the constant
   is written (RFC-0071 rule 5).
5. **The constant is the value's constructors**: arrays, tuples, objects and
   variants are built from the constants of their parts, at the types the body
   closed them to, as the lowering builds a literal.

**Why.** A literal `M::R` is the whole enum a script writes, but a bound `M::R`
is one value of an enum only the host knows. An arm naming a variant a known
enum lacks is refused (RFC-0051 rule 2), so giving the input the literal's
enum refuses the `% M::D =>` arm of the very dispatch the binding decides;
binding would widen what is refused, where it only narrows what is required
(RFC-0071 rule 5). Left to the uses, the enum is the type the unbound input
has, and the value is one value of it. Every other head comes from the value,
because a method, an index or a `for` needs the head while the body is
checked. A refusal at the binding reaches the host that built the value, once,
at the call.

**Cost.** Binding types the value once on its own, apart from any body. A
field read the object lacks is refused at the constant, after the body checked,
not at the read.

**Rejected.**
- `acvus_ast::Literal` extended with objects and variants — a constant
  instruction and every pass that reads one would take values no single
  instruction writes.
- The literal's own enum as the input's — its dispatch's other arms are
  refused rather than decided.
- Every position left to the uses — a method, an index or a `for` on the
  input is refused while its head is open.
- A value with no type refused in each body — one refusal per body that
  reads the input, at no span.
