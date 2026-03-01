# Acvus Grammar Reference

## Template Structure

템플릿은 segment의 시퀀스로 구성된다. Lexer가 먼저 segment를 분류하고, expression 내부는 LALRPOP 파서가 처리한다.

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

- `Text`: `{{ }}` 바깥의 모든 텍스트.
- `Comment`: `{{-- --}}`로 감싸진 주석. 출력에 포함되지 않는다.
- `ExprTag`: `{{ }}` 안의 expression 또는 binding.
- `CatchAll`: `{{_}}`는 lexer 레벨에서 감지된다. expression 내부의 `_`와 완전히 분리.
- `CloseBlock`: `{{/}}`로 match 블럭을 닫는다. `+N`/`-N`은 indent modifier.

### AST Construction (Parser Level)

Segment 시퀀스에서 AST를 구축한다:

```
Node         = Text | Comment | InlineExpr | MatchBlock | IterBlock

InlineExpr   = ExprTag where content has no "=" or "in"

MatchBlock   = ExprTag("pattern = expr") Body Arm* CatchAll? CloseBlock
IterBlock    = ExprTag("pattern in expr") Body CatchAll? CloseBlock

Arm          = ExprTag("pattern =") Body       ← multi-arm continuation
Body         = Node*
```

**MatchBlock vs IterBlock**:
- `=` (MatchBlock): 단일 값에 대한 패턴 매칭. 이터레이션 없이 source 값을 직접 패턴에 대조한다.
- `in` (IterBlock): source를 이터레이터로 변환하여, 각 원소에 대해 바디를 반복 실행한다.

**Multi-arm 감지**: match 블럭 내부에서 `{{ pattern = }}`이 나타나면 continuation arm으로 취급된다. `=` 뒤에 expression이 없으면 continuation arm, 있으면 binding이다. Multi-arm은 `=` (MatchBlock)에서만 사용 가능하다.

**변수 바인딩**: `{{ x = expr }}`에서 LHS가 단순 변수(`Binding` 패턴)이면 body-less — `{{/}}` 불필요.

**Iteration 패턴**: `{{ pattern in expr }}`의 패턴은 irrefutable해야 한다 (변수, 오브젝트 destructuring, 튜플 destructuring 등). 리터럴 패턴은 사용 불가.

---

## Expression Grammar

LALRPOP 기반. 연산자 우선순위 (낮은 → 높은):

```
TagContent   = Expr "=" Expr        ← binding / pattern matching
             | Expr "="             ← continuation arm
             | Expr "in" Expr       ← iteration
             | Expr                  ← inline expression

Expr         = LambdaExpr

LambdaExpr   = PipeExpr "->" LambdaExpr    ← right-associative
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

MulExpr      = MulExpr ("*" | "/") UnaryExpr ← left-associative
             | UnaryExpr

UnaryExpr    = "-" UnaryExpr
             | "!" UnaryExpr
             | PostfixExpr

PostfixExpr  = PostfixExpr "." IDENT       ← field access
             | PostfixExpr "(" CommaSep<Expr> ")"  ← function call
             | PrimaryExpr
```

### Primary Expressions

```
PrimaryExpr  = IDENT                       ← identifier
             | "$" IDENT                   ← storage reference
             | INT                         ← integer literal
             | FLOAT                       ← float literal
             | STRING                      ← string literal
             | FORMAT_STRING               ← format string (see below)
             | "true" | "false"            ← boolean literal
             | "(" CommaSep<TupleElem> ")" ← paren / tuple (see below)
             | "[" CommaSep<ListElem> "]"  ← list
             | "{" (ObjectField ",")+ "}"  ← object

TupleElem    = Expr | "_"
ListElem     = Expr | ".."
ObjectField  = IDENT | "$" IDENT
```

**Format String**:
- `"hello {{ name }}!"` → `"hello " + name + "!"`
- `{{ }}` 안에 임의의 expression 사용 가능: `"sum: {{ a + b | to_string }}"`
- Grammar-level desugaring — `BinOp::Add` 체인으로 변환. 새 AST variant 없음.
- **String 타입만 허용** — auto `to_string` 없음. 비-String 표현식은 `| to_string` 파이프 필요.
- 빈 텍스트 세그먼트(`""`)는 체인에서 제외.

**Tuple vs Paren 구분**:
- 1개 원소 (non-wildcard): `(expr)` → 괄호 그룹 (Paren)
- 1개 원소 (wildcard): `(_)` → 1-원소 튜플
- 2개 이상 원소: `(a, b)` → 튜플 (Tuple)
- 0개 원소: `()` → 빈 튜플

**Lambda Parameter**:
- `(a, b) -> expr`에서 `(a, b)`는 먼저 Tuple로 파싱된 후, `->` 앞에서 lambda parameter list로 변환된다.

---

## Patterns

Expression의 LHS에서 패턴으로 변환된다 (`expr_to_pattern`):

```
Pattern      = Binding | Literal | List | Object | Range | Tuple

Binding      = IDENT                       ← 변수 캡처
             | "$" IDENT                   ← 스토리지 레퍼런스

Literal      = INT | FLOAT | STRING | "true" | "false"

List         = "[" Pattern* "]"            ← exact match
             | "[" Pattern* ".." Pattern* "]"  ← rest pattern

Object       = "{" ObjectPatternField* "}" ← open matching

Range        = Pattern ".." Pattern
             | Pattern "..=" Pattern
             | Pattern "=.." Pattern

Tuple        = "(" TuplePatternElem ("," TuplePatternElem)* ")"
TuplePatternElem = Pattern | "_"           ← wildcard
```

**Wildcard `_` 스코프**: `_`는 tuple 패턴 내부에서만 사용 가능하다 (일반 expression에서는 사용 불가). `{{_}}`의 catch-all과는 별개 — catch-all은 lexer 레벨에서 감지된다.

---

## Tokens

| Token | 예시 |
|-------|------|
| `IDENT` | `name`, `user`, `x` |
| `$REF` | `$name`, `$user` |
| `INT` | `0`, `42`, `-1` |
| `FLOAT` | `3.14`, `0.0` |
| `STRING` | `"hello"`, `"world"` |
| `FORMAT_STRING` | `"hello {{ name }}!"` (lexer가 `FmtStringStart`/`Mid`/`End`로 분할) |
| `true` `false` | boolean literals |
| `_` | wildcard (tuple 패턴 내부) |
| `+` `-` `*` `/` | 산술 연산자 |
| `!` | 논리 부정 |
| `&&` `\|\|` | 논리 AND / OR |
| `==` `!=` `<` `>` `<=` `>=` | 비교 연산자 |
| `=` | 바인딩 (패턴 매칭) |
| `in` | 이터레이션 |
| `->` | 람다 화살표 |
| `..` `..=` `=..` | 레인지 연산자 |
| `.` | 필드 접근 |
| `\|` | 파이프 연산자 |
| `(` `)` `[` `]` `{` `}` | 괄호류 |
| `,` | 구분자 |

### Operator Precedence (낮은 → 높은)

| 우선순위 | 연산자 | 결합 방향 |
|---------|--------|-----------|
| 1 | `->` (lambda) | right |
| 2 | `\|` (pipe) | left |
| 3 | `\|\|` (logical or) | left |
| 4 | `&&` (logical and) | left |
| 5 | `==` `!=` `<` `>` `<=` `>=` | left |
| 6 | `..` `..=` `=..` (range) | non-assoc |
| 7 | `+` `-` | left |
| 8 | `*` `/` | left |
| 9 | `-` `!` (unary) | prefix |
| 10 | `.` `()` (postfix) | left |
