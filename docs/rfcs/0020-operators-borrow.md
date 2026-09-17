# RFC-0020: Operators on language-owned types are instructions; on extension types, a shared signature

Status: Accepted
Date: 2026-09-15
Supersedes: none

## Ruling

The compiler holds one operator table. Each operator has two entries:

- **Language-owned operands** — the word primitives (`Int`, `Float`,
  `Bool`, `Byte`, `Unit`) and `String`: the operator is an IR instruction
  defined by this RFC and implemented once per host. On words it is
  `BinOp`/`UnaryOp`; on `String` it is a named string instruction
  (`StringEq`, `StringConcat`), never an overloaded `BinOp`. No shared
  signature, no instance, no call.
- **Extension-type operands**: the operator is a call of a shared signature
  (RFC-0019) with a fixed name, and the operands are borrowed for the
  expression. Today only `==` and `!=` have a signature, `core::eq<T>(&T,
  &T) -> Bool`; `!=` is the negation of `eq`. An operator with no
  signature entry, or a signature with no instance for the operand type,
  is a type error at the expression.

The meaning on words:

- `==` / `!=` is identity of the representation: the two words are equal.
  On `Float` this is bit equality: `NaN == NaN`, `0.0 != -0.0`.
- `<`, `<=`, `>`, `>=` is the total order of the representation: signed
  for `Int`, unsigned for `Byte`, `false < true`, and `f64::total_cmp`
  for `Float`, which agrees with bit equality.
- `+`, `-`, `*` on an integer wrap at the width, as Rust's release build
  does (RFC-0037). On `Float` they are IEEE arithmetic; `NaN` is a value,
  not an error.
- `/`, `%` on an integer panic at zero and at `MIN / -1`, with Rust's
  texts.

The meaning on `String`: `==` is byte equality (`StringEq`), `+` is
concatenation (`StringConcat`), and a template's output is one
`StringConcat` of its parts. Both take their operands by reference: a
place is lent, a `&String` passes as it is, a temporary is owned by the
expression; a `&String` part of a template is read through.

Every other meaning is a named function in a registry, called by name:
`ieee_eq`, `ieee_lt`, `wrapping_add`, `saturating_add`. A language-owned
type has no instance of any shared signature; `eq(&1, &2)` and `eq(&s,
&t)` by name are type errors, `1 == 2` and `s == t` are the operators.

Borrowing the operands is the operator's rule and reaches no function
call: `f(x)` with `f: Fn(&T)` stays a type error (RFC-0018).

An operand that is a reference is read at what it names: `+`, `-`, `*`,
`/`, `%`, the comparisons and the bit operators read a `&word` through the
reference (RFC-0018), and `==`, `!=` and `+` compare or concatenate what a
reference names at any size, keeping the reference in the IR. A `&T` whose
`T` is not a word is not an operand of the first group.

An operand the checker has not resolved yet carries the operator's bound
away from the operator: `+`, `-`, `*`, `/`, `%` and the comparisons bound
it by the numbers — every integer width and `Float`, with `String` for `+`
— and the bit operators by the integer widths. The bound is the one an
integer literal already carries (RFC-0037), so two literals meet: `k + 7`
leaves `k` an integer and `k + 7.0` a `Float`. A type outside the bound
that later joins the variable is refused where the bound is verified, as
`type Array<Float, 2> is outside the declared bound one of ...`.

## Rationale

`==` is the one operation a tagless word has a meaning for without any
definition: the two words are the same. Everything that needs a
definition — an IEEE comparison, a saturating sum — is a definition, and
a definition has a name. Giving the operator the tagless meaning and the
definitions their names keeps each meaning in exactly one place; the
prior state, where the interpreter compared `Float` by IEEE and wrapped
`Int` on overflow while the language said nothing, was two unnamed
definitions hiding behind one symbol.

Defining the language's own operators in the compiler rather than as
`core::add` instances that a host inlines keeps one definition per
operation. A Rust body that is the specification plus a fast path that
reimplements it is two definitions of one thing, and the fast path wins
silently when they drift.

`String` is language-owned because its literals are: a type the compiler
constructs is the compiler's, and a language that hands half of its
string to a library keeps the other half anyway. Its operators are named
instructions rather than a `BinOp` that dispatches on the operand type,
so the IR says what it does and the interpreter has one arm per meaning.

For an extension type, comparing is a function of two references, as
`PartialEq::eq(&self, &other)` is, and once it is a function it is a
shared signature: `Regex` compares by its own instance. Borrowing the
operands is what a reader expects of `a == b`, and making it the
operator's rule rather than a coercion keeps calls exact.

## Not built

- No instance of a shared signature for a language-owned type.
- No signature for ordering or arithmetic yet. When an extension type
  needs one, a signature (`core::cmp`, `core::add`) is declared and the
  language-owned entries of the table do not change.
- No string view type. `&String` is the borrow; a sub-string is a new
  `String`. A fat-pointer view would change the host's value layout and
  is held until a need for it exists.
- No user-declared operators. The names are fixed; a type joins one by
  declaring an instance.
- No coercion between a value and a reference anywhere but at an
  operator's own operand, and no implicit width or `Int`/`Float`
  conversion: `1 + 2.0` is a type error.

## Consequences

- The type checker unifies the two operand types at every operator and,
  where the result is still a variable, meets that variable's bound with
  the operator's (`typeck::operand_bound`) and records it for
  verification. At `==`/`!=`/`+`, a word
  result types as `BinOp`, `String` as a string instruction, anything
  else resolves `core::eq`, instantiates it at the operand type, and
  records the call on the expression (`TypeResolution::operator_calls`).
- Lowering emits `BinOp` for a word pair; for `String` and for a recorded
  call, a `Ref` of each operand (a temporary first assigned to a register
  the expression owns) and the instruction or call; `!=` adds
  `UnaryOp::Not`. A template body lowers to one `StringConcat`.
- `core::eq` and `core::clone` are declared in `acvus_extern::core` with
  no instances; `Externs::combine` always includes that registry.
- The interpreter's `Int` arithmetic is checked (`IntegerOverflow`), its
  `Float` comparisons are bit equality and `total_cmp`, and `BinOp` has
  no `String` arm.
