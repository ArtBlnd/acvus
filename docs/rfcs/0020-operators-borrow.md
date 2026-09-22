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
- `&&` and `||` are not instructions. `a && b` is `if a { b } else
  { false }` and `a || b` is `if a { true } else { b }`: the left operand
  decides alone where it can, and the right operand — its reads, its
  calls, its effects — happens only on the path that evaluates it. The
  checker's rule is unchanged, both operands are `Bool` at the
  expression, and the operand bound of the two is empty as `==`'s is.

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
definition: the two words are the same. Everything else — an IEEE
comparison, a saturating sum — is a definition, and a definition has a name.
Giving the operator the tagless meaning and the definitions their names keeps
each meaning in one place; an operator that silently carried a definition
would be a definition the language does not name, and the definition the
interpreter chose would be the one that holds.

Defining the language's own operators in the compiler rather than as
`core::add` instances that a host inlines keeps one definition per
operation. A Rust body that is the specification plus a fast path that
reimplements it is two definitions of one operation, and the fast path is the
one that runs when they differ.

`String` is language-owned because its literals are: the compiler constructs
the value, so the compiler owns the type. Its operators are named
instructions rather than a `BinOp` that dispatches on the operand type, so
the IR names the operation and the interpreter has one arm per meaning.

For an extension type, comparing is a function of two references, as
`PartialEq::eq(&self, &other)` is, and once it is a function it is a shared
signature: `Regex` compares by its own instance. The borrow is the operator's
rule rather than a coercion, so `f(x)` with `f: Fn(&T)` stays an error.

## Amended by RFC-0070

- The comparisons join `==`/`!=` in the table's second entry:
  `<`, `<=`, `>`, `>=` on an extension type are a call of
  `core::cmp<T>(&T, &T) -> Int`, whose result is `-1`, `0` or `1`, read
  by its sign. The language declares no `Ordering` type; `-1`/`0`/`1` is
  the protocol `string::cmp`, `num::total_cmp` and the `sort_by`
  comparator already speak. The word entries of the table are unchanged.
- An instance of a shared signature for a language-owned type is
  declared where a requirement can reach that type (RFC-0067,
  RFC-0070): `core::eq` at `Int` exists so that a handler requiring `eq`
  at `T` runs at `T = Int`. The operator table does not call it — `==` on
  two words is still the instruction — and the instance's meaning is the
  instruction's, pinned per type by a test in the standard registry.

## Not built

- No signature for arithmetic. When an extension type needs one, a
  signature (`core::add`) is declared and the language-owned entries of
  the table do not change.
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
  At `<`/`<=`/`>`/`>=` the same, with `core::cmp` (RFC-0070); the
  recorded call names which signature it resolved.
- Lowering emits `BinOp` for a word pair; for `String` and for a recorded
  call, a `Ref` of each operand (a temporary first assigned to a register
  the expression owns) and the instruction or call; `!=` adds
  `UnaryOp::Not`, and a comparison adds the operator's own `BinOp`
  between `cmp`'s answer and `0`. A template body lowers to one
  `StringConcat`.
- `&&` and `||` lower to the diamond `if`/`else` lowers to, the value
  being the join block's parameter, so `BinOp::And` and `BinOp::Or` reach
  no IR instruction and the interpreter has no arm for them. The same
  diamond carries the conjunction a refutable pattern's parts make, so a
  part is tested only while the parts before it have matched.
- `core::eq` and `core::clone` are declared in `acvus_extern::core` with
  no instances; `Externs::combine` always includes that registry.
- The interpreter's integer `+`, `-`, `*` wrap and its `/` and `%` panic at
  zero and at `MIN / -1` with Rust's texts; its `Float` equality is bit
  equality and its `Float` ordering is `total_cmp`; its `BinOp` dispatch has
  no `String` arm and no `And`/`Or` arm.
