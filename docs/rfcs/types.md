# Types and the solver

What a type variable may become, how a call of a shared name is settled on
one function and one instance, what the language's own operators and
integers mean, and the one shape every open question of the checker takes:
a join for equality, a decision for choice.

## RFC-0011: A declaration bounds its type variables, and the bound is verified when the variable freezes

Status: Accepted

1. A declaration states, per type variable, by position beside its
   polymorphic type, what the variable may become: `Any` or `OneOf` a set
   of type schemes. It is the only channel by which a declaration
   constrains a variable; an ExternFn bounds the variables of its type, an
   extension type those of its parameters.
2. `OneOf` admits a type that has the shape of one of its schemes. Two
   bounded variables that unify share the bound whose schemes are the
   pairwise unifiers of theirs; nothing satisfying both is a type error.
3. Binding a variable keeps its bound. Freezing it verifies the bound
   against the resolved type, and the violation is reported at the use that
   instantiated the declaration. Nothing is checked earlier.
4. The checker itself bounds a variable at an integer literal (RFC-0037)
   and at an operator's operand (RFC-0020); every other constraint arises
   from unification.
5. An effect variable's declared bound is a floor on its task. The one a
   declaration states is `E: Suspends`: `E`'s task is `Async` or `Heavy`.
   It is verified when the variable freezes, as a type variable's bound is
   (rule 3), because an effect variable's lower end only rises: a floor
   asserted at instantiation would lift a `Sync` effect to `Async` rather
   than refuse it. A violation names the task the variable froze at.

**Why.** The solver already carries a finite-domain bound and meets bounds
when it unifies; declared polymorphism uses that mechanism instead of a
second one at the call site. A bound is checked when the variable freezes,
so it never has to decide before an argument's type is known.

**Rejected.**
- An overload set resolved by trial at each call — it must decide before
  the arguments resolve.
- Capability bounds on a variable (`Cloneable`) — a function that needs an
  ability of its variable takes the instance as a parameter (RFC-0067).

## RFC-0019: A shared signature is a name with one polymorphic type and no body, filled by instances the registries declare

Status: Accepted

A shared signature is a namespaced name with one polymorphic function type
and no body: `core::clone<T>(&T) -> T`, `core::eq<T>(&T, &T) -> Bool`. The
signatures the compiler names are `core`'s (RFC-0070 rule 5); every other
signature is a registry's, under its own namespace.

An instance is an ExternFn whose type is the signature's type at a type
scheme — one concrete `T`, or a shape such as `List<T>` for every `T`.
Whoever declares a type may declare its instances, in any registry.
Combining the registries collects every instance of a signature into that
one function, so a script sees one name. Two instances of one signature
whose types unify are refused, so the head constructor at the signature's
first type variable picks at most one instance; the instance then fixes the
signature's other variables (`into_iter<C, T>(C) -> Iter<T>` learns `T`
from the `List<T>` instance). An instance declared as a cast (RFC-0023)
registers a coercion from its parameter scheme to its return scheme,
settled through the signature as any call of it is.

A call of the signature is settled on one instance by the compiler
(RFC-0040). An ExternFn that needs a signature at its own type variable
takes the instance as a parameter the checker fills; that requirement, and
an instance's own requirements, are instances.md's (RFC-0067, RFC-0070).
The requirement belongs to the declaration, not to the variable: the
solver's variable carries only its bound, which combining meets with `OneOf`
the signature's instance types where the requirement stands at the bare
variable.

No signature has a body, no type an impl block, and no value a table: the
set of types that fill a signature closes when the registries combine.
There are no script-level generic functions and no `dyn`, no default or
fallback instance — a type with no instance of `eq` is a type error at the
call — and no inheritance between signatures. `Object` and `Enum` have no
instance of any `core` signature (RFC-0070 rule 5).

**Why.** Copying, comparing, printing and hashing are one name over many
types, and the type that knows how is declared elsewhere than the name.
A type's abilities are a fact of the registry, known before a script is
checked; the runtime never asks a value.

**Rejected.**
- Traits with impl blocks and dynamic dispatch — a value would carry what
  it can do.
- Overlapping instances with a specificity order (`List<i64>` beside
  `List<T>`), or an instance chosen by more than one variable — the choice
  is by exclusion on one variable only.
- Structural `eq` on objects — an object type is an open set with no
  definition of total equality; a script compares chosen fields, and a map
  over object keys takes its comparator (`hash_map_by`).

## RFC-0020: Operators on language-owned types are instructions; on extension types, a call of a `core` signature

Status: Accepted

The compiler holds one operator table with three entries per operator.

- **Language-owned operands** — the words (integers, `f64`, `Bool`, `char`,
  `Unit`) and text: the operator is an IR instruction. On words it is
  `BinOp`/`UnaryOp`; on text a named instruction, `StringEq` or
  `StringConcat`, never an overloaded `BinOp`.
- **Extension-type operands** — the operator is a call of a `core`
  signature with borrowed operands: `==` calls `core::eq`, `!=` is its
  negation, and `<`, `<=`, `>`, `>=` call `core::cmp<T>(&T, &T) -> i64`,
  whose `-1`/`0`/`1` is read by its sign. No `Ordering` type exists. `+`,
  `-`, `*`, `/` and `%` call `core::add`, `core::sub`, `core::mul`,
  `core::div` and `core::rem`, each `<T, O>(&T, &T) -> O`, and unary `-`
  calls `core::neg<T, O>(&T) -> O`; the instance fixes `O`, and its answer
  is the operator's value. An operator with no signature, or a signature
  with no instance at the operand type, is a type error at the expression.
- **Structural operands** — an object, a tuple, an array, an enum, an
  `Option` or a `Result`: `==` and `!=` take the language's structural
  instance of `core::eq`, which is the signature at each component. The
  ordering and arithmetic operators have none.

The meaning on words:

- `==`/`!=` is identity of the representation; on `f64`, bit equality
  (`NaN == NaN`, `0.0 != -0.0`).
- `<`, `<=`, `>`, `>=` is the representation's total order: signed or
  unsigned by the integer's type, `false < true`, `f64::total_cmp` on
  `f64`, which agrees with bit equality.
- Integer arithmetic is RFC-0037's; `f64` arithmetic is IEEE, `NaN` a value.
- `&&` and `||` are control flow, not instructions: `a && b` is
  `if a { b } else { false }`, `a || b` is `if a { true } else { b }`, so
  the right operand's reads, calls and effects happen only on the path that
  evaluates it.

On text, `==` is byte equality and `+` concatenation, reading `String` or
`&str` operands alike (RFC-0062); a template's output is one
`StringConcat` of its parts. Operands are taken by reference: a place is
lent, a reference passes as it is, a temporary is owned by the expression.

Every other meaning is a named registry function (`ieee_eq`,
`wrapping_add`, `saturating_add`). The instances of `core` signatures at
language-owned types exist for requirement sites (RFC-0070 rule 5); the
operator does not call them, and each is pinned to the instruction's
meaning by a test.

Borrowing the operands is the operator's rule and reaches no function
call: `f(x)` with `f: Fn(&T)` stays a type error (RFC-0018). A `&word`
operand is read through the reference; every operator that is a call of a
`core` signature, and `+` on text, reads what a reference names at any
size, because the call lends its operands.

`==`, `!=`, `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, `/`, `%` and unary `-` are
calls of their `core` signature at every operand type. At a word or text
the language's own instance is the operator's instruction, `StringEq` and
`StringConcat` at text; a registry's instance at a
language-owned type is kept for requirement sites and named calls, and an
operator does not reach it, so a type with no ordering (`Bool`, `Unit`,
text) has none at an operator. The language's arithmetic instances are at
every integer width and `f64`, each answering at its operand type, `+`'s
also at `String` and `str`, answering `String`, and negation's at the
signed widths and `f64`; `Bool` and `char` have none, so
`true + 1` and `'a' - 'b'` are type errors. An operand whose type is still
open leaves the choice of instance to the solve (RFC-0042 rule 2), and an
operand that settles to a reference is read through it (RFC-0029 rule 3).
An operand of the bit operators still unresolved carries the operator's
bound, the integer widths (RFC-0011 rule 4). Text's `==` and `+` read
`String` and `str` alike, so where one side of `==`, `!=` or `+` is text
and the other still a variable, the variable is bounded by `String` and
`str` and closes on `String` (RFC-0042 rule 3): `|x| -> x + "b"` applied
to a `String` concatenates. The two operands of an operator meet at one
type before the instance is chosen, so `k + 7` leaves `k` an integer and
`k + 7.0` an `f64`. Nothing converts implicitly: `1 + 2.0` and
`1u8 + 1u16` are type errors.

The structural instance answers `==`, `!=`, and a named `eq(&a, &b)` or
`clone(&a)`, at every structural type. Each component of the operand type
is decided as an instance of its own: at a word or text the language's,
at a structural type the structural instance again, at an extension type
the registry's. A component with no instance, a function or a reference,
refuses the whole operation with the no-instance error at that component's
type. `==` compares an object field by field, a tuple and an array
position by position, and an enum, an `Option` or a `Result` by its tag
and then by the payload at that tag's type; a word compares by its bits
and text by its bytes, as at the top. `clone` copies a word, clones text,
calls the registry's `clone` at an extension type and rebuilds each
aggregate, so a clone shares no storage with its source. The two operands
meet as every operator's do, and a meet grows an object to the union of
the two field sets; a value built without a field the other side has is
then refused where it is lent, since its construction never stored that
field (RFC-0042 rule 1), so two objects compare only at one field set. An
enum is one type across any variant sets, and its values compare by tag.
A component whose type nothing constrains holds no value and closes to
`!`, which has nothing to compare, so `None == None` is `true`. A
registry instance that requires `eq` or `clone` at a structural type, as
`eq` at `Vec<{a: i64}>` requires `eq` at `{a: i64}`, reaches only the
registry's instances and is refused. This is a decision, not a gap: an
extern that compares states the exact type it compares and reads it
through its projection, so a structural instance, which is the language's
own, never crosses into a handler.

**Why.** `==` is the one operation a tagless word means without a
definition; every other meaning is a definition, and a definition has a
name. Defining the language's operators in the compiler keeps one
definition per operation. Text is language-owned because its literals are.
A structure's equality and copy are its components', which its type
fixes, so they need no definition of their own; an ordering of a
structure would choose an order of its fields, which is a definition.

**Rejected.**
- Language operators as `core::add` instances a host inlines — two
  definitions of one operation, and the fast path is the one that runs
  when they differ.
- A `BinOp` that dispatches on text — the IR names the operation.
- An operator that silently carries an IEEE or saturating meaning — a
  definition the language does not name.
- User-declared operators — the names are fixed; a type joins one by
  declaring an instance.
- A registry instance per structural shape — the shapes are unbounded,
  and each would restate the component rule.
- An object `==` at the union of two field sets — the side that never
  stored a field would be read there.

## RFC-0037: Integers have Rust's widths, and a literal takes the width its use demands

Status: Accepted

1. The integer types are `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`,
   `u64`, Rust's. A Rust struct field of one crosses as that type
   (RFC-0039 rule 4).
2. Arithmetic, comparison and the bit operators take two operands of one
   width and produce it; nothing widens or narrows by itself (conversion is
   `as`, RFC-0049). A shift takes its amount modulo the width, and `/` and
   `%` panic on a zero divisor and at `MIN / -1` with Rust's texts.
   Negation takes a signed integer or `f64`.
3. **Overflow is undefined.** A `+`, `-`, `*` or negation whose exact
   result does not fit the width gives a program no meaning, and every
   analysis and pass assumes it does not happen. This implementation traps
   at the operation that overflows (RFC-0048 rule 8), so an overflowing run
   ends before its value reaches anything; a pass may move, merge or drop
   such an operation as if it could not trap. `/` and `%` are the only
   integer operations whose failure is defined, so RFC-0007's constraint on
   moving an operation that can raise binds them alone. A script that wants
   another behaviour names it: `wrapping_*`, `checked_*`, `saturating_*`,
   `overflowing_*`; this rule stands on those being there.
4. An unsuffixed integer literal's type is a variable only an integer type
   fills; the use decides the width (`@b + 1` with `@b: u8` makes `1` a
   `u8`), and where nothing decides it is `i64` (RFC-0042 rule 3). A negated
   literal takes signed types only. A suffix fixes the width (RFC-0058).
5. Once the width is known the value is range-checked against it, as a
   compile error at the literal: `literal 300 does not fit u8`.
6. At run time an integer is a word, its two's-complement bits read at the
   type's width; a space lays it out in the width's bytes (RFC-0033). A
   host reading JSON reads a number as `i64`, or `u64` when too large.
7. There is no `f32`.

**Why.** Wire schemas count in `u32`/`u64`, and one struct must state
both the wire and the language; widening to `i64` would claim negative
counts. A literal without a width of its own lets a script write
`max_tokens: 1024` and the field decide.

An overflow undefined in the language lets every analysis reason as
arithmetic does, `i + 1 > i` included, where wrapping would send each
interval that could reach the width's end to ⊤; trapping keeps the
implementation sound where a program breaks the rule.
**Rejected.**
- Wrapping arithmetic — every analysis pays for a behaviour no correct
  script wants.
- Overflow undefined with no trap — a bound an analysis derived could admit
  an access the overflow put out of range.
- A suffix on every literal that is not `i64` as the rule — puts in the
  script the type the struct already states (suffixes exist as an option,
  RFC-0058).
- Range-checking a literal before its width is known — would reject `300`
  as `i64` or accept it as `u8`.

## RFC-0040: The compiler chooses an ExternFn's instance, and the IR records it by number

Status: Accepted

An ExternFn has one numbered instance list: the concrete instances in
declaration order, then the generic instance, whose type is the function's
own, when there is one. A `Monomorphize<(T0, T1, ..)>` member (one handler
compiled per listed type), an instance of a shared signature, and the plain
body of a function are all entries of that list; the registry hands the
compiler the types and the runtime the handlers in the same order.

The checker settles every call on one number. A function with no concrete
instance is settled at its instantiation. A function with concrete
instances opens an instance decision (RFC-0042 rule 2) per call, settled by
exclusion: after each argument, checked left to right, an instance the
call's type already contradicts is dropped, and the last one left is taken
— which is what lets a lambda later in the arguments see the element type
the container fixed. The generic instance is taken when every concrete one
is excluded; beside a generic instance, a concrete one is taken only once
the call type joins it with no variable left open (RFC-0041). A call no
instance can match is a type error at the call; instances the types leave
tied are broken by the task (RFC-0046), and a tie that remains is an error
at the call.

The lowering writes the number: `Callee::Extern { id, instance }` is a call
of an ExternFn, `Callee::Direct(id)` a call of a function with a body, and
a cast carries its instance the same way. The runtime indexes the handler
list and runs; it matches no types. Two concrete instances of one function
whose types unify are refused when the registries combine.

**Why.** The instance is a compile-time fact; matching types at run time
recomputes it. A position in a list both sides hold identifies it, since
nothing links by name.

**Rejected.**
- Mangled names — there is no ABI and no linker; one process built both
  lists.
- The generic instance as a candidate — it overlaps every concrete one; it
  is the fallback when candidates are exhausted.
- Search with backtracking — a choice is settled by exclusion only.
- Run-time type matching of instances — recomputes a compile-time fact.

## RFC-0042: The solver separates equality from decision

Status: Accepted

1. Unification is the join of the type lattice, taken where it is
   asked. `!` is the bottom at a value position only; inside a constructor
   every argument is invariant. Two `Object`s join to the union of their
   fields and two `Enum`s to the union of their variants. A value the body
   constructs — an object literal, a structural variant — has a type
   variable of its own, and a flow or a decision that joins two such
   variables makes them one: a member one gains, the other has, and a
   construction is laid at the union with the members it did not write
   undefined (RFC-0050 rule 8). A type no variable of the body names — a
   context's, an extern's parameter or result, a declared struct — has a
   layout the body did not choose: a variable joined to one takes it as it
   is, and a join that would have it gain a member is refused by that member.
   A read's object and a pattern are lower bounds and are not refused for
   lacking one; a pattern may name fewer members than its source. A value may
   lack a field only while it moves between the body's storages and
   registers: a read of a field takes that field, and every other use takes
   the value whole, so an element, a payload, an argument, a return and a
   commit are whole (the definite-assignment check). An
   object type carries which field set it has: the fields a struct declares,
   under the struct's name; the fields an object literal wrote; or at least
   the fields a read or a pattern named. Two undeclared field sets join to
   their union. A declared field set is the field set of every value of that
   type, so an object that is one and lacks a declared field, or carries an
   undeclared one, is refused by that field's name, and what only asks an
   object for fields joins to the declared type. At a projection parameter the
   object is matched at least, not exactly (RFC-0050 rule 6).

2. A decision is a position with more than one admissible answer; it
   holds its answer set and only shrinks. The decisions are: integer width
   (RFC-0037), effect interval (RFC-0013), instance (RFC-0040), representation
   `ρ` (RFC-0041), conversion (rule 4), signature (RFC-0043), lend (RFC-0029),
   capture read (RFC-0018), and pattern mode (RFC-0024).

3. A decision settles when one answer remains; nothing left is an error
   naming the position. Open at the end, it takes its least element: `!` for a
   type variable, `i64` for an integer literal, `String` for a text operand
   (RFC-0020), the lower effect, `Uniform`, the generic instance, a value
   pattern mode, a plain reference for a lend, a word read for a capture, a
   source of its own for an identity; an instance decision the types leave
   tied takes the tightest task ceiling admitting the call (RFC-0046). A
   signature decision has no least element (RFC-0043). `settle` is one
   fixpoint, idempotent, run once per body before `freeze`.

4. A conversion is a decision between two types at a site, answered
   from the registry once both resolve: identity where they agree, else a
   declared rule. Nothing else converts.

The body is checked in one pass that only creates variables, joins and
opens decisions (`check`); decisions read resolved terms inside `settle`
(`query`); `solve` closes what remains and only then renders messages.

**Why.** Width, instance, effect, representation and conversion were each
answered at the join by a device of their own — snapshot/rollback pairs, a
polarity, a settlement mid-body — each guessing an answer the terms had not
resolved. One shape for a decision and one settlement remove the devices
and let a message name the final type instead of the type at the guess.

**Rejected.**
- A device per question (snapshot/rollback, polarity, mid-body settlement)
  — replaced by one decision shape.

## RFC-0043: A bare name is a set of signatures, settled by evidence

Status: Accepted

A bare name that several namespaces declare resolves to the set of their
signatures (amending RFC-0021's one-name rule). Candidates whose arity
differs from the call leave; one left is the call, as a qualified name is.
Several left form one call: a fresh function type whose parameters carry
`OneOf` of the union of the shapes the remaining candidates have there (a
candidate's own `OneOf` variables expanded, a bare variable making the
position `Any`), and whose return is the candidates' common return pattern
with fresh variables (`get`'s `&T`), or a bare variable where they share
none. That bound is the parameter's range and nothing more: no argument is
admitted by reading it.

An argument is taken by one of four admissions, asked of each candidate's
own parameter: `Direct` — the argument's type is one of the parameter's
shapes; `Converted` — one declared rule (RFC-0023) casts it there, through
a reference one rule each way; `Viewed` — the argument borrows a storage
and the parameter takes a view of it (`&v` at `&[T]`, RFC-0047; `&s` at
`&str`, RFC-0062); `Refused`, and the candidate leaves the set there. A set
an argument empties is `NoMatchingFunction` there.

1. **Direct first.** At an argument some candidate takes directly, every
   candidate that would take it only by conversion or view leaves the set.
   A weaker admission would change the value the callee sees.
2. **A weaker admission needs a resolved head.** An argument still a
   variable is admitted directly where the bounds intersect, and by nothing
   else. The wait outlives the settle: when the variable freezes to a head
   no remaining candidate takes directly, admission is asked again there and
   rule 1 applies. If the rest of the call leaves one candidate while the
   argument is still a variable, the settle does not join it — that would
   decide the head by the candidate's shape; the argument is settled after
   the solve against the parameter the decision chose, as the view where
   the head arrived and the parameter takes one, as the conversion where the
   settled candidate converts it, as unification where nothing named the
   head.
3. **Equal strength is told apart by the rest of the call.** Candidates
   that remain at the end are `AmbiguousFunction`.
4. **A receiver is an argument.** It is admitted by the same admissions and
   rules, in the mode each candidate sees it in (rule 6); a variable
   receiver is admitted directly where bounds intersect and by no view until
   it resolves. A local binding is a candidate like any other: where it
   takes an argument directly, a signature that would convert it leaves.
5. **The view is the checker's, the parameter's type is the callee's.** A
   settled viewed candidate records the view at the argument; the
   argument's own type is untouched, so a later use of the place sees what
   it was.
6. **Receiver mode, per candidate.** A candidate whose first parameter is a
   reference sees the receiver lent, at that mutability; one whose first
   parameter is a value or a variable, or whose head is still open, sees the
   place by value. The call takes the mode the remaining candidates see,
   told apart by the type each sees: modes yielding one type are one mode,
   the lend; candidates seeing different types are `AmbiguousFunction`, since
   choosing a lend over a move would be a default. A receiver that is not a
   place narrows no candidate: it is lent where every candidate's first
   parameter is a reference of one mutability, by value otherwise. A place
   whose head is still a variable is held, and its mode is part of each
   candidate's admission of it: every step of the decision admits a
   candidate at the type its own mode sees, the place or a reference to
   what a lend of the place names, and that type waits as rule 2 has it
   while its head is open. The settled candidate's mode lends or moves the
   place, as it would at a known head. Choosing the mode first would ask
   admission of a type no candidate sees: a `String` a lambda parameter
   turns out to be would be passed by value, and `s.contains("a")` refused
   where the known form lends it to `string::contains`. A pipe passes its
   left side as a value.

Where a remaining candidate takes an argument only by conversion, the
call's parameter stays open and one conversion decision is opened there; it
has no answer of its own and settles after the signature — identity where
the settled candidate took the argument directly, the rule where it
converted. A conversion decision one side of which is a `OneOf` variable
answers identity only where the other side could match a shape of the
bound. Where another remaining candidate takes the same argument by view,
or two take it by different views, no conversion decision is opened: a
conversion decision has no view to answer (rule 5). The argument is held,
the decision keeps each candidate there by its own admission, and once it
settles the argument is met as the settled candidate admits it, as rule 2
settles one: the view is the checker's at the argument, the reborrow is
the shared one it reaches the parameter as, and the conversion is the
decision the call would have opened there, opened now and answered by a
solve of its own. The conversion opened at the call stays where no
candidate views the argument, answered within the body's solve. On either
path a conversion through the reference takes the place out for the call
(RFC-0041), and the move check refuses a use of it among the call's later
arguments in the MIR both paths lower to, so in the same words on both.

The call opens a signature decision (RFC-0042 rule 2), stepped as an instance
decision is: a candidate stays while the call type would join its type on
a copy of the terms, each open parameter's bound meets the candidate's, and
each argument it converts is taken by one rule. One left settles: the
candidate is instantiated as a qualified call — its own instance decision
and bounds opened — and joined with the call type. None left is
`NoMatchingFunction`; several left when the body is solved is
`AmbiguousFunction`, listing them in sorted display order. A signature
decision has no least element; nothing is defaulted. The lowering reads the
callee through the settled answer, then its instance.

A binding of the name in scope is one more candidate, whatever its type's
head. Its parameters are `Any` and it contributes no return pattern; it
stays while the call type would unify with its type (a head still open
does, a non-`Fn` head does not), and settling on it fixes an open head to
the call's `Fn` and lowers to an indirect call. `AmbiguousFunction` names
it ``the binding `len` ``.

A report shows a type as written, every variable nothing resolved closed to
`!` whatever its bound; `<error>` names only a subexpression that already
failed. What the resolution carries into lowering is the stricter freeze,
which refuses a variable a bound left open.

A library function that exists per container is a plain function in that
type's namespace (RFC-0028); `core`'s signatures remain the one-signature,
instance-per-type mechanism (RFC-0019).

**Why.** A lambda's parameter has no type when a call inside it is checked
(`|c| -> contains(c, 1)`); a dispatch at the call would decide wrong or
refuse. A decision that shrinks as terms resolve is what the solver already
has for instances. Between signatures there is no fallback — they are
different functions — so a signature decision has no least element, unlike
the instance decision's generic fallback within one signature.

**Cost.** An argument no candidate takes is reported at the call; a
disagreement under a shape some candidate takes is reported when the
decision settles or fails. A receiver two candidates see as different types
is reported, not resolved: the call is written qualified or the binding
renamed. A lambda parameter used only as a container stays ambiguous among
the container namespaces. No chain of two conversions is searched.

**Rejected.**
- Unique suffixed names (`contains_iter`) — moves the decision from the
  checker to the reader.
- A qualified path at every use — makes the common call the verbose one;
  kept as the escape hatch.
- Dispatch on the first argument's syntax (`&x` picks reference candidates)
  — decides before a lambda parameter has a type.
- One combined `OneOf` signature with no decision — cannot name which
  function runs.
- Settling an argument's conversion to identity eagerly — would decide the
  signature by the argument's shape.
