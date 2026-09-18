# Acvus Grammar Reference

## Template Structure

A template is a sequence of segments. The lexer classifies segments first, then the LALRPOP parser handles expression internals.

### Segments (Lexer Level)

```
Template     = Segment*
Segment      = Text | Comment | ExprTag | CatchAll | CloseBlock

Text         = (any text outside {{ }})
Comment      = "{{--" content "--}}"
ExprTag      = "{{" content "}}"
CatchAll     = "{{_}}"
CloseBlock   = "{{/}}" | "{{/+" DIGITS "}}" | "{{/-" DIGITS "}}"
```

- `Text`: All text outside `{{ }}`.
- `Comment`: Wrapped in `{{-- --}}`. Not included in output.
- `ExprTag`: Expression or binding inside `{{ }}`.
- `CatchAll`: `{{_}}` is detected at the lexer level. Completely separate from `_` inside expressions.
- `CloseBlock`: `{{/}}` closes a match block. `+N`/`-N` are indent modifiers.

### AST Construction (Parser Level)

The AST is built from the segment sequence:

```
Node         = Text | Comment | InlineExpr | MatchBlock | IterBlock

InlineExpr   = ExprTag where content has no "=" or "in"

MatchBlock   = ExprTag("pattern = expr") Body Arm* CatchAll? CloseBlock
IterBlock    = ExprTag("pattern in expr") Body CatchAll? CloseBlock

Arm          = ExprTag("pattern =") Body       ← multi-arm continuation
Body         = Node*
```

**MatchBlock vs IterBlock**:
- `=` (MatchBlock): Pattern matching against a single value. Matches the source value directly against the pattern without iteration.
- `in` (IterBlock): Converts the source to an iterator and executes the body for each element.

**Multi-arm detection**: When `{{ pattern = }}` appears inside a match block, it is treated as a continuation arm. If there is no expression after `=`, it is a continuation arm; if there is, it is a binding. Multi-arm is only available with `=` (MatchBlock).

**Variable binding**: In `{{ x = expr }}`, if the LHS is a simple variable (`Binding` pattern), it is body-less — no `{{/}}` needed.

**Iteration pattern**: The pattern in `{{ pattern in expr }}` must be irrefutable (variable, object destructuring, tuple destructuring, etc.). Literal patterns are not allowed.

---

## Scripts

A script is a sequence of semicolon-terminated statements with an optional
tail expression. There is one statement rule, and it is the rule of every
block: a script's top level, a lambda's block body, a `while`/`anyorder`
body, a tag-form match-bind body, an `if`/`else` block.

```
Script       = Stmt* ScriptExpr?
ScriptExpr   = IfExpr | Expr
```

### Statements

```
Stmt         = LetBind | LetUninit | Assign | ContextStore | VarFieldStore
             | DerefStore | While | WhileLet | Anyorder | MatchBind | ExprStmt

LetBind      = "let" IDENT "=" ScriptExpr ";"           ← let x = 0;
LetUninit    = "let" IDENT ";"                          ← let x;
Assign       = IDENT "=" ScriptExpr ";"                 ← x = 0;
ContextStore = "@" IDENT ("." IDENT)* "=" ScriptExpr ";" ← @a = 0; / @a.x.y = 0;
VarFieldStore= IDENT ("." IDENT)+ "=" ScriptExpr ";"    ← a.x = 0;
DerefStore   = "*" Expr "=" ScriptExpr ";"              ← *r = 0;
While        = "while" Expr "{" Stmt* "}"
WhileLet     = "while" "let" Pattern "=" Expr "{" Stmt* "}"
Anyorder     = "anyorder" "{" Stmt* "}" ";"?
MatchBind    = Pattern "=" Expr "{" Stmt* "}" ";"       ← if-let with body
ExprStmt     = Expr ";" | IfExpr ";"
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

A name assigned inside a `while`, `anyorder`, `if` or match-bind body is the
outer binding, so its new value is live after the body; the join carries it.

**Assignment LHS resolution**: The LHS FieldAccess chain is flattened to
determine the root:
- Root is `IDENT` with no path → `Assign`
- Root is `IDENT` with path → `VarFieldStore`
- Root is `@IDENT` → `ContextStore` (with or without path)
- Root is `*Expr` with no path → `DerefStore`
- Otherwise → parser error (`InvalidAssignTarget`)

**MatchBind versus Assign**: both start with `Expr "="`. The trailing
`"{" Stmt* "}" ";"` is what makes it the tag form; `x = { a, };` with an
object literal on the right is an assignment.

**A template binding is the template's.** `{{ x = expr }}` inside a template
is a `MatchBlock` with a `Binding` pattern (see *Template Structure* above),
not a statement: this rule does not reach it.

**ContextStore path**: In `@a.x.y = 0;`, path = `[x, y]`. Empty path means identity store (`@a = 0;`).


## Expression Grammar

LALRPOP-based. Operator precedence (low → high):

```
TagContent   = Expr "=" Expr        ← binding / pattern matching
             | Expr "="             ← continuation arm
             | Expr "in" Expr       ← iteration
             | Expr                  ← inline expression

Expr         = LambdaExpr

LambdaExpr   = "|" CommaSep<Ident> "|" "->" Expr    ← right-associative
             | PipeExpr

PipeExpr     = PipeExpr "|" OrExpr         ← left-associative
             | OrExpr

OrExpr       = OrExpr "||" AndExpr        ← left-associative
             | AndExpr

AndExpr      = AndExpr "&&" CompExpr      ← left-associative
             | CompExpr

CompExpr     = CompExpr CompOp RangeExpr   ← left-associative
             | RangeExpr

CompOp       = "==" | "!=" | "<" | ">" | "<=" | ">="

RangeExpr    = AddExpr ".." AddExpr        ← exclusive [start, end)
             | AddExpr "..=" AddExpr       ← inclusive end [start, end]
             | AddExpr "=.." AddExpr       ← exclusive start (start, end]
             | AddExpr

AddExpr      = AddExpr ("+" | "-") MulExpr ← left-associative
             | MulExpr

MulExpr      = MulExpr ("*" | "/" | "%") UnaryExpr ← left-associative
             | UnaryExpr

UnaryExpr    = "-" UnaryExpr
             | "!" UnaryExpr
             | QualifiedExpr

QualifiedExpr = IDENT "::" IDENT "(" Expr ")"  ← qualified variant with payload
              | IDENT "::" IDENT                ← qualified variant without payload
              | PostfixExpr

PostfixExpr  = PostfixExpr "." IDENT       ← field access
             | PostfixExpr "[" Expr "]"    ← index (RFC-0047)
             | PostfixExpr "(" CommaSep<Expr> ")"  ← function call
             | PrimaryExpr
```

`a[i]` is a place, as `a.f` is: `&a[i]`, `a[i][j]`, `a[i].f`, `a[i]` as a
method receiver, and `a[i] = v` on the left of an assignment. The index is
a `u64` and the container is a `Vec`, an `Array`, or a reference to one;
any other container is refused. A statement that begins with `[` is an
array literal, and a postfix `[` binds to the expression before it — every
statement ends in `;` or in a block, so the two never meet.

### Primary Expressions

```
PrimaryExpr  = IDENT                       ← identifier (value binding)
             | "$" IDENT                   ← extern parameter (immutable, injected)
             | "@" IDENT                   ← context reference (mutable storage)
             | INT                         ← integer literal
             | FLOAT                       ← float literal
             | STRING                      ← string literal
             | FORMAT_STRING               ← format string (see below)
             | "true" | "false"            ← boolean literal
             | "Some" "(" Expr ")"         ← Some variant constructor
             | "None"                      ← None variant constructor
             | "(" CommaSep<TupleElem> ")" ← paren / tuple (see below)
             | "[" CommaSep<ListElem> "]"  ← list
             | "{" ScriptStmt+ Expr "}"    ← block expression
             | "{" Expr "}"               ← block expression (single expr)
             | "{" (ObjectField ",")+ "}"  ← object literal

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
             | Range | Tuple | Variant

Binding      = IDENT                       ← variable capture
             | "$" IDENT                   ← extern parameter capture

ContextBind  = "@" IDENT                   ← context binding

Literal      = INT | FLOAT | STRING | "true" | "false"

List         = "[" Pattern* "]"            ← exact match
             | "[" Pattern* ".." Pattern* "]"  ← rest pattern

Object       = "{" ObjectPatternField* "}" ← open matching

Range        = Pattern ".." Pattern
             | Pattern "..=" Pattern
             | Pattern "=.." Pattern

Tuple        = "(" TuplePatternElem ("," TuplePatternElem)* ")"
TuplePatternElem = Pattern | "_"           ← wildcard

Variant      = "Some" "(" Pattern ")"      ← Some variant
             | "None"                       ← None variant
             | IDENT "::" IDENT "(" Pattern ")"  ← qualified with payload
             | IDENT "::" IDENT             ← qualified without payload
```

**ObjectPatternField**: `{ key: pattern }` or shorthand `{ name }` / `{ $name }` / `{ @name }`.

**Wildcard `_` scope**: `_` is only available inside tuple patterns (not in general expressions). Separate from the `{{_}}` catch-all, which is detected at the lexer level.

**ContextBind in destructure**: a `@name` sub-pattern stores the matched value into the context `@name`. No two names ever denote one storage (RFC-0015): `{ @x, } = @a { body }` copies `@a.x` into `@x`, and `@x` inside the body is the context `@x`.

---

## Tokens

| Token | Example |
|-------|---------|
| `IDENT` | `name`, `user`, `x` |
| `$REF` | `$name`, `$user` |
| `@REF` | `@name`, `@user` |
| `INT` | `0`, `42`, `-1` |
| `FLOAT` | `3.14`, `0.0` |
| `STRING` | `"hello"`, `"world"` |
| `FORMAT_STRING` | `"hello {{ name }}!"` (lexer splits into `FmtStringStart`/`Mid`/`End`) |
| `true` `false` | boolean literals |
| `Some` `None` | variant constructors |
| `_` | wildcard (inside tuple patterns) |
| `+` `-` `*` `/` `%` | arithmetic operators |
| `!` | logical negation |
| `&&` `\|\|` | logical AND / OR — short-circuiting: the right operand is evaluated only where the left does not decide (RFC-0020) |
| `==` `!=` `<` `>` `<=` `>=` | comparison operators |
| `=` | assignment (a statement), pattern match (a tag / a template) |
| `in` | iteration |
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

### Operator Precedence (low → high)

| Precedence | Operator | Associativity |
|-----------|----------|---------------|
| 1 | `->` (lambda) | right |
| 2 | `\|` (pipe) | left |
| 3 | `\|\|` (logical or) | left |
| 4 | `&&` (logical and) | left |
| 5 | `==` `!=` `<` `>` `<=` `>=` | left |
| 6 | `..` `..=` `=..` (range) | non-assoc |
| 7 | `+` `-` | left |
| 8 | `*` `/` `%` | left |
| 9 | `-` `!` (unary) | prefix |
| 10 | `::` (qualified) | — |
| 11 | `.` `()` (postfix) | left |
