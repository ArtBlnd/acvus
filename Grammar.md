# Acvus Grammar Reference

## Template Structure

A template is a script with one extra rule (RFC-0071): a line that does not
begin with `%` is text, appended to the result as written. The template's
value is the accumulated text, a `String`.

```
Template     = Line*
Line         = TextLine | StmtLine

StmtLine     = Blank* "%" ScriptLine        ← one statement, no `;`, no braces
TextLine     = "%%" Text?                   ← text holding one `%`
             | Text                          ← text as written
Text         = (TextRun | "{{" Expr "}}")* "\"?      ← a trailing "\" drops the newline
```

### `%` lines

A line whose first non-blank character is `%` is one statement of the script
grammar below, written **without its trailing `;`** and **without the braces**
that would open or close a block. It leaves nothing in the output, its
newline included. A block a `%` line opens is closed by `% end`.

```
% let x = e                    ← every statement of the script grammar
% x = e        % @c = e        ← assignment and stores
% a.b = e      % v[i] = e      % *r = e
% f(x)                         ← an expression statement
% break        % continue
% // a comment, as `//` is in a script

% if E                         % if let P = E
% else if E
% else
% end

% for x in ForHead             ← the four heads of RFC-0057
% while E                      % while let P = E
% anyorder
% end

% match E                      ← one `% pattern =>` line per arm
% P =>
% _ =>
% end
```

A `% match` body begins with its first `% pattern =>` arm; nothing stands
between the scrutinee line and that arm. The refusals are one per fault:
`% end` closes no block, block not closed expected `% end`, `% else` needs
an `% if` to belong to, this `% if` already has an `% else`, `% pattern =>`
needs a `% match` to belong to.

A `%` line the statement grammar does not admit is a parse error at that
line. It does not fall back to text.

### Text lines

A text line is appended as written, **with its newline**. There is no other
whitespace rule:

- A line beginning with `%%` is appended with one `%` in its place.
- A line ending in `\` is appended without its newline.

`{{ expr }}` inside a text line is the format string of RFC-0058 rule 6. The
expression is a `String` or a `&str`; nothing is converted to text
implicitly, so a number takes `| to_string` or `.to_string()`. The tag's
content is tokenized as an expression, so a string literal inside it may
hold `{{` or `}}`: `{{ "{{" }}` writes a literal `{{`. A tag does not span
lines.

Inline branching is the expression grammar's, not a template form, because
`if` and `match` are operands: `{{ if c { "a" } else { "b" } }}`.

### `$name` is an injected input

`$name` is a value the host injects, shared by every function of the graph
and typed by its use; `@name` is a context, as in a script. A call passes
nothing, `{{ rules() }}`, and the inputs a template requires are the `$`
names its reachable code reads (RFC-0071 rules 4 and 5). There is no
include and no parameter declaration.

### Example

```
You are a careful assistant for {{ &@name }}.
Answer in {{ if @lang == "ko" { "Korean" } else { "English" } }}.

% if @mode == "review"
Review the code below and list defects.
% else
Help with the code below.
% end

Recent turns:
% for m in &@journal
- {{ &m.role }}: {{ &m.content }}
% end
```

### Lowering

A text line and a `{{ }}` tag each lower to one append onto the template's
accumulator, and a block statement lowers through the same lowering the
script uses for it — an `% if` is the `Diamond` of RFC-0063, a `% for` the
`For` terminator of RFC-0057. The accumulator is never copied, so a loop's
cost is the text it writes.

---

## Scripts

A script is a sequence of semicolon-terminated statements with an optional
tail expression. There is one statement rule, and it is the rule of every
block: a script's top level, a lambda's block body, a `while`/`for`/
`anyorder` body, a `match` arm's block, an `if`/`else` block, and a
template's `%` lines.

```
Script       = Stmt* Expr?
```

### Statements

```
Stmt         = LetBind | LetUninit | Assign | Store | DerefStore
             | While | WhileLet | For | Break | Continue
             | Anyorder | ExprStmt

LetBind      = "let" IDENT "=" Expr ";"                 ← let x = 0;
LetUninit    = "let" IDENT ";"                          ← let x;
Assign       = IDENT "=" Expr ";"                       ← x = 0;
Store        = Place "=" Expr ";"                       ← a.x = 0; / v[i].f = 0;
DerefStore   = "*" Expr "=" Expr ";"                    ← *r = 0;
Place        = PlaceRoot (("." IDENT) | ("[" Expr "]"))*
PlaceRoot    = IDENT | "$" IDENT | "@" IDENT
While        = "while" Expr "{" Stmt* "}"
WhileLet     = "while" "let" Pattern "=" Expr "{" Stmt* "}"
For          = "for" IDENT "in" ForHead "{" Stmt* "}"
ForHead      = Expr | Expr ".." Expr
Break        = "break" ";"
Continue     = "continue" ";"
Anyorder     = "anyorder" "{" Stmt* "}"
ExprStmt     = Expr ";"
```

`while`, `for` and `anyorder` are statements, not expressions: each ends at
the `}` closing its block, and a `;` after that `}` is refused. Every other
statement ends in `;`, an `if` and a `match` statement included, so a
statement that begins with one is the expression statement
rule and the grammar has no decision to make -- there is no
"a block-like expression at statement start is a statement" rule, as Rust
has. `if c { … };` and `match e { … };` are `Expr ";"`, and the same
expression without the `;` is the block's tail.

### `for`

```
for x in &v { ... }        ← v: Vec<T> or Array<T, N>;  x: &T
for x in &mut v { ... }    ← v: Vec<T> or Array<T, N>;  x: &mut T
for x in a { ... }         ← a: Array<T, N>, consumed;  x: T
for i in lo..hi { ... }    ← lo, hi: one integer width; x: that width
```

Four heads and no other (RFC-0057). The element comes through the
container's own `as_slice` or `as_slice_mut`, the same instance an `a[i]`
of it settles, so `&v` holds `v` shared for the loop and `&mut v` holds it
exclusively; an array by value is consumed, and its elements are taken out
one at a time. A head of any other shape is refused:

```
a `for` traverses `&v`, `&mut v`, an array by value, or `lo..hi`; `i64` is none of them
a container is not consumed by a loop; write `&v` or `&mut v`
```

`lo..hi` is a `for` head and not a value: there is no `Range` type, so `..`
appears in no expression rule and `let r = 0..3;` does not parse. The two
bounds are one integer width:

```
a `for` over `u64..u32` needs one integer width at both bounds
```

`break` and `continue` name the innermost loop -- there are no labels -- and
they are admitted in `for`, `while` and `while let` alike. Outside every
loop each is refused:

```
`break` is only inside a loop
```

A `for` over an array may be left by `break`, `?` or `return` at every
element type: the elements it had taken are the binding's and are released
by their scopes, and the array's release on the leaving edge releases the
ones it had not taken.

### `return`

```
Return       = "return" Expr                            ← return e
```

`return e` leaves the enclosing body -- a script or a lambda -- with `e`,
from any depth, loops included. It is an expression and not a statement:
its type is `!`, so `let x = if c { return 1 } else { 2 };` reads the other
branch's type and `return e;` is the expression-statement rule. There is no
`return;`, because every body returns a value (RFC-0054):

```
expected an expression, found `;`
```

The value meets the body's return type the way the tail does -- the host's
declaration for a script, the inferred return for a lambda -- and a `return`
in a template, which returns nothing, is refused as a `?` there is:

```
`return` needs a function to return from; a template has none
```

A `return` leaves every enclosing loop at once, and each array traversal
among them releases the elements it had not taken, as on a `break`.

What the source wrote after a `return` lands in a block no jump reaches,
as it does after `break`, `continue` or an expression typed `!`. It is
still checked: this language has no warnings, and an unreachable statement
meets the body's return type like any other.

### `match`

```
MatchExpr    = "match" Expr "{" (MatchArm ",")* MatchArm? "}"
MatchArm     = ArmPattern "=>" "{" Stmt+ "}"            ← run for its effect
             | ArmPattern "=>" Expr                     ← including { s; tail }
ArmPattern   = "_" | Pattern
```

`match` is an expression (RFC-0051): the scrutinee is evaluated once, every
arm has the type of the whole (`!` admitted), and an arm with no tail is
`Unit`. An arm naming a variant the scrutinee cannot hold is refused --
the arms contribute no variant:

```
unreachable pattern: `E::C(_)` is not a variant of `E{A(i64), B(i64)}`
```

Exhaustiveness is decided in `validate`, on the MIR, and only where the
variant set is known: an `Option` or a `Result`, or an enum every
definition of which is in this body. Elsewhere a `_` arm is the way
through:

```
non-exhaustive match: the variants of this value are not known in this function; add a `_` arm
```

**`let` binds, `=` assigns.** `let x = e;` introduces a binding and shadows
any outer `x`; the binding ends with the block that introduced it. `x = e;`
stores into the `x` already in scope in the same body, and never introduces
one: where no binding of that name is in scope the compiler reports

```
cannot assign to `x`: no binding named `x` is in scope; `let x = ...;` binds it
```

A name bound outside a lambda and assigned inside it is a capture, and a
capture is by value (RFC-0018), so the store would write the lambda's copy:

```
cannot assign to `x`: it is captured by the lambda, not bound in it
```

A name assigned inside a `while`, `anyorder`, `if`, `if let` or `match` arm
body is the outer binding, so its new value is live after the body; the
join carries it.

**An assignment target is a place.** A place is a root -- a name, a
`$parameter` or an `@context` -- under any path of `.field` and `[index]`
steps: `x`, `@a.x.y`, `v[i]`, `v[0].f`, `o.g[0]`, `o.f[i].g`. The left of
`=` is read as one:
- a bare name → `Assign`, which is the form the checker reads against the
  bindings in scope: a name that is not bound, or one the enclosing lambda
  captured, is refused there
- `*Expr` → `DerefStore`, a store through a reference **value**
- any other place → `Store`, holding a `Place`: a `Root` -- a name, a
  `$parameter` or an `@context` -- under a path of `.field` steps over a
  base, where the base is that root or a `[index]` step whose container is
  the place expression the index machinery reads
- anything else → parser error (`InvalidAssignTarget`)

The parser is the one site that decides: `Place::of` is the conversion and
`InvalidAssignTarget` its refusal, so `f(x)[0] = 1` is refused there and
nothing below re-derives that a store's target is a place.

A `$parameter` is a place, and a store into one is refused by the checker:
`extern param `$p` is immutable and cannot be assigned`. A store into a
place through a shared reference is refused the same way a read of it as a
`&mut` would be: ``cannot store through &{f: i64}: not a `&mut```, and a
`v[i]` step of a place takes the container's `as_slice_mut`, so the loan the
write holds is exclusive.

**A mutable projection demands its object mutably** (RFC-0018). A `.field`
under a write or a `&mut` hands its object the same mutable demand, so every
`[index]` below the field takes `as_slice_mut` too: `v[i].h[j] = e` writes
into `v`, where a shared slice would have written into a copy of the element
and lost it. The same rule makes a mutable reach through a shared reference
a refusal rather than a silent copy: `r.g[i] = e` and `&mut r.g[i]` with
`r: &{g: [i64]}` are both ``cannot store through &{g: [i64]}: not a
`&mut```. A read borrows the object shared, as before.

The IR has one shape per write: `Assign { target, path }` with
`PathSeg::Field` steps for every place whose steps are fields, and
`IndexSet { slice, index, value }` where the last step is an index, because
`PathSeg::Index` carries a constant and `v[i]`'s index is a value.

`@a = 0;` is the place `@a` with no steps: the store writes the context
itself.


## Expression Grammar

LALRPOP-based. Operator precedence (low → high):

```
Expr         = LambdaExpr
             | "return" Expr                ← leaves the body; type `!`

LambdaExpr   = "|" CommaSep<Ident> "|" "->" Expr    ← right-associative
             | PipeExpr

PipeExpr     = PipeExpr "|" OrExpr         ← left-associative
             | OrExpr

OrExpr       = OrExpr "||" AndExpr        ← left-associative
             | AndExpr

AndExpr      = AndExpr "&&" CompExpr      ← left-associative
             | CompExpr

CompExpr     = CompExpr CompOp AddExpr     ← left-associative
             | AddExpr

CompOp       = "==" | "!=" | "<" | ">" | "<=" | ">="

AddExpr      = AddExpr ("+" | "-") MulExpr ← left-associative
             | MulExpr

MulExpr      = MulExpr ("*" | "/" | "%") CastExpr ← left-associative
             | CastExpr

CastExpr     = CastExpr "as" IDENT         ← left-associative (RFC-0049)
             | UnaryExpr

UnaryExpr    = "-" UnaryExpr
             | "!" UnaryExpr
             | "*" UnaryExpr               ← read through a reference
             | "&" "mut"? UnaryExpr        ← borrow a place (RFC-0018)
             | PostfixExpr

PostfixExpr  = PostfixExpr "." IDENT       ← field access
             | PostfixExpr "[" Expr "]"    ← index (RFC-0047)
             | PostfixExpr "?"             ← the payload, or an early return (RFC-0038)
             | PostfixExpr "." IDENT "(" CommaSep<Expr> ")"  ← method call
             | PostfixExpr "(" CommaSep<Expr> ")"            ← function call
             | PrimaryExpr
```

A qualified name is a primary (below), not a level of its own, so every
postfix follows a qualified call: `i64::from_str(s)?`, `E::f(x).g`,
`E::f(x)[0]`.

`a[i]` is a place, as `a.f` is: `&a[i]`, `a[i][j]`, `a[i].f`, `a[i]` as a
method receiver, and `a[i] = v` on the left of an assignment. The index is
a `u64` and the container is a `Vec`, an `Array`, or a reference to one;
any other container is refused. A statement that begins with `[` is an
array literal, and a postfix `[` binds to the expression before it — every
statement ends in `;` or in a block, so the two never meet.

A `-` that touches an integer literal and stands where no value ended is
part of that literal, not a negation: `-128i8` is the `i8` literal whose
value is `-128`, `-9223372036854775808` is the `i64` minimum, and the range
check sees the signed value (RFC-0058). The rule is the token pair's, so
`- 128i8` with a space, `-(128i8)`, `-x`, `-1.5` and `a -1` are what they
were — a negation or a subtraction. The tokenizer carries it, because a
grammar production for `"-" INT` beside `"-" UnaryExpr` is ambiguous: after
`- 1` with `+` ahead, both parse.

### Primary Expressions

```
PrimaryExpr  = IDENT                       ← identifier (value binding)
             | IDENT "::" IDENT            ← qualified name: `Enum::Tag`, or a
                                             call's callee — which of the two
                                             it is, the checker settles
             | "$" IDENT                   ← extern parameter (immutable, injected)
             | "@" IDENT                   ← context reference (mutable storage)
             | INT                         ← integer literal
             | INT_OF                      ← suffixed integer literal (RFC-0058)
             | FLOAT                       ← float literal
             | CHAR                        ← character literal (RFC-0058)
             | BYTE                        ← byte literal (RFC-0058)
             | BYTES                       ← byte-string literal (RFC-0058)
             | STRING                      ← string literal, a `&str` (RFC-0062)
             | FORMAT_STRING               ← format string (see below)
             | "true" | "false"            ← boolean literal
             | "Some" "(" Expr ")"         ← Some variant constructor
             | "None"                      ← None variant constructor
             | "(" CommaSep<TupleElem> ")" ← paren / tuple (see below)
             | "[" CommaSep<ListElem> "]"  ← list
             | "{" ScriptStmt+ Expr "}"    ← block expression
             | "{" Expr "}"               ← block expression (single expr)
             | "{" (ObjectField ",")* "}"  ← object literal
             | MatchExpr                   ← an operand, braces as delimiter
             | IfExpr                      ← an operand, braces as delimiter

TupleElem    = Expr | "_"
ListElem     = Expr | ".."
ObjectField  = IDENT ":" Expr             ← explicit key
             | IDENT                       ← shorthand { name } = { name: name }
             | "$" IDENT                   ← shorthand { $name } = { name: $name }
             | "@" IDENT                   ← shorthand { @name } = { name: @name }
```

**Format String**:
- `"hello {{ name }}!"` → `"hello " + name + "!"`
- Any expression inside `{{ }}`: `"sum: {{ a + b | to_string }}"`
- Grammar-level desugaring — converted to a `BinOp::Add` chain. No new AST variant.
- **String type only** — no auto `to_string`. Non-String expressions require `| to_string` pipe.
- Empty text segments (`""`) are excluded from the chain.

**`match` and `if` are operands.** Wherever an expression stands -- an
operator's side, a call argument, a parenthesized expression, a list
element, an object field value, a pipe stage's input, a method receiver, a
scrutinee -- a `match` or an `if` stands, with its braces as its delimiter
and no `;` inside the expression: `10 + match n { … }`, `f(if c { 1 } else
{ 2 })`, `(match n { … }) + 10`.

**An object literal's trailing comma is required** -- a decision, not a hole.
`{ g: 1, }` is the form and `{ g: 1 }` is refused with ``expected `,`,
found `}```. One field and one comma read the same as ten, and the comma is
what separates the literal from a block expression whose tail is a name
(`{ g }` is the object `{ g: g, }`, and `{ g, }` says so). Nothing about
lists, tuples or calls follows it: each of those admits a trailing comma
and does not require one.

**Tuple vs Paren**:
- 1 element (non-wildcard): `(expr)` → parenthesized group (Paren)
- 1 element (wildcard): `(_)` → 1-element tuple
- 2+ elements: `(a, b)` → tuple (Tuple)
- 0 elements: `()` → empty tuple

**Lambda Parameter**:
- In `|a, b| -> expr`, `|a, b|` forms the lambda parameter list.

**Block Expression**:
- `{ stmt; stmt; expr }` — statement sequence + tail expression.
- `{ expr }` — single expression block.

---

## Patterns

Converted from expression LHS via `expr_to_pattern`:

```
Pattern      = Binding | ContextBind | Literal | List | Object
             | Tuple | Variant

Binding      = IDENT                       ← variable capture
             | "$" IDENT                   ← extern parameter capture

ContextBind  = "@" IDENT                   ← context binding

Literal      = INT | INT_OF | FLOAT | CHAR | BYTE | BYTES | STRING
             | "true" | "false"

List         = "[" Pattern* "]"            ← exact match
             | "[" Pattern* ".." Pattern* "]"  ← rest pattern

Object       = "{" ObjectPatternField* "}" ← open matching

Tuple        = "(" TuplePatternElem ("," TuplePatternElem)* ")"
TuplePatternElem = Pattern | "_"           ← wildcard

Variant      = "Some" "(" Pattern ")"      ← Some variant
             | "None"                       ← None variant
             | IDENT "::" IDENT "(" Pattern ")"  ← qualified with payload
             | IDENT "::" IDENT             ← qualified without payload
```

**ObjectPatternField**: `{ key: pattern }` or shorthand `{ name }` / `{ $name }` / `{ @name }`.

**Wildcard `_` scope**: `_` stands in a tuple pattern and as a `match` arm's
pattern, `% _ =>` in a template included. It is not an expression.

**ContextBind in destructure**: a `@name` sub-pattern stores the matched value into the context `@name`. No two names ever denote one storage (RFC-0018): `{ @x, } = @a { body }` copies `@a.x` into `@x`, and `@x` inside the body is the context `@x`.

---

## Tokens

| Token | Example |
|-------|---------|
| `IDENT` | `name`, `user`, `x` |
| `$REF` | `$name`, `$user` |
| `@REF` | `@name`, `@user` |
| `INT` | `0`, `42`, `-1` — a `-` the digits touch is the literal's sign (RFC-0058) |
| `INT_OF` | `10u64`, `255u8`, `-128i8` — the suffix is one of RFC-0037's eight widths, and a value the width does not hold is refused |
| `FLOAT` | `3.14`, `0.0` |
| `CHAR` | `'x'`, `'\n'`, `'\u{1F600}'` — one Unicode scalar value, Rust's escapes |
| `BYTE` | `b'G'`, `b'\xFF'` — a `u8` |
| `BYTES` | `b"GET"`, `b"\xFF\x00"` — an `Array<u8, N>`, ASCII and `\xNN` |
| `STRING` | `"hello"`, `"a\tb\x41\u{1F600}"` — Rust's escapes, the same table `CHAR`, `BYTE` and `BYTES` use, and an escape outside it is refused |
| `FORMAT_STRING` | `"hello {{ name }}!"` (lexer splits into `FmtStringStart`/`Mid`/`End`) |
| `true` `false` | boolean literals |
| `Some` `None` | variant constructors |
| `_` | wildcard (inside tuple patterns) |
| `+` `-` `*` `/` `%` | arithmetic operators |
| `!` | logical negation |
| `&&` `\|\|` | logical AND / OR — short-circuiting: the right operand is evaluated only where the left does not decide (RFC-0020) |
| `==` `!=` `<` `>` `<=` `>=` | comparison operators |
| `=` | assignment (a statement) |
| `in` | a `for` head |
| `return` | leaves the enclosing body with the expression that follows |
| `->` | lambda arrow |
| `..` `..=` `=..` | range operators |
| `.` | field access |
| `[` `]` | index, and array literal |
| `\|` | pipe operator |
| `::` | qualified name separator |
| `:` | object field separator |
| `;` | statement terminator |
| `(` `)` `[` `]` `{` `}` | delimiters |
| `,` | separator |
| `//` | a comment to the end of the line, skipped where whitespace is, in a script and inside a `{{ }}` tag. A tag's extent is the scanner's, so a comment inside one ends at the line's end or the tag's `}}`, whichever is first. A `//` between the quotes of a string literal is text, because the string lexer owns the literal's extent. There is no block comment. |

### Parse errors

A parse error names what the grammar admitted at the span, in the language's
words: `expected an expression, found `;``. The set LALRPOP reports is the
grammar's terminal names, and a message never prints one. `Terminal` in
`acvus-ast/src/error.rs` carries one class per terminal — `int_of` is "a
number", `ident` "a name", `fmt_start` "a format string", and every operator
and delimiter its own text — and a set that covers a nonterminal's whole
opening stands for that nonterminal:

| Set | Message |
|-----|---------|
| every operand opening, with `\|` and the statement keywords | `a statement` |
| every operand opening, with or without the lambda's `\|` | `an expression` |
| every operand opening and `_` | `a pattern` |
| the seven literal forms | `a literal` |
| anything else | the terminals themselves: ``expected `)` or `,``` |

Terminals the covered nonterminal does not account for follow it:
``expected an expression, `..` or `]```.

The table is closed over the terminals `extern { enum Token { … } }`
declares: `Terminal::of_token` gives the compiler one half — a new `Token`
variant does not compile until it has a `Terminal` — and the test
`terminals_match_the_grammar` reads the names out of `grammar.lalrpop` and
holds them equal to the table's.

### Operator Precedence (low → high)

| Precedence | Operator | Associativity |
|-----------|----------|---------------|
| 1 | `->` (lambda) | right |
| 2 | `\|` (pipe) | left |
| 3 | `\|\|` (logical or) | left |
| 4 | `&&` (logical and) | left |
| 5 | `==` `!=` `<` `>` `<=` `>=` | left |
| 6 | `+` `-` | left |
| 7 | `*` `/` `%` | left |
| 8 | `as` (cast) | left |
| 9 | `-` `!` `*` `&` (unary) | prefix |
| 10 | `.` `[]` `?` `()` (postfix) | left |
