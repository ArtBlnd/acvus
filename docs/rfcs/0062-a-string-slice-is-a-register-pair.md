# RFC-0062: a string slice is a register pair

Status: Accepted — 2026-09-20
Extends: RFC-0047 (a slice is the one thing the machine indexes; two
registers; the `&v` coercion and the `Slice<T, Rt>` parameter), RFC-0018
(references and loans), RFC-0012 (every user-defined value moves; a copy is
written), RFC-0058 (`char`), RFC-0059 (a crossing says its width)

## Problem

A `String` is the only string. Every operation that yields part of one
allocates a new one: `substring`, `trim`, `split_str`, `find_all`, a regex
match's text. A line of a log parsed into five fields is five heap
allocations and five releases.

A string literal is a `String` constant: the listing shows `assign v13 =
T0` and `ref &v13` where the literal is used, one `String` per use of a
constant that never changes.

An extern that reads a string takes `&String` and unwraps it to Rust's
`&str` itself; an extern cannot take a part of a string without a copy,
and cannot return one.

RFC-0047 solved the same three problems for `Vec<T>`: `&[T]` is two
registers, `&v` coerces to it at a parameter, an extern takes it as a
`Slice<T, Rt>` view and returns one as a projection of its parameter. The
string has no such type.

## Decision

1. **`&str` is a type**: `Ref(Shared, Str)`, a shared borrow of a `String`'s
   bytes. There is no `&mut str` (a write through it could break UTF-8)
   and no owned `str` value. In the machine it is a `SlicePair` — `ptr`,
   `len` in two adjacent registers (RFC-0047 rule 6 as amended), the
   length in bytes.

2. **What produces one.**
   - A string literal is a `&'static str`: two constant words, no
     `String`, no slot. A literal that must be a `String` is written
     `"…".to_string()`.
   - `substring(&s, start: u64, end: u64) -> &str` — byte offsets, the
     one run-time check being `is_char_boundary` at both ends (RFC-0037's
     raise); `trim`, `trim_start`, `trim_end` return a view of their
     argument; `split(&s, sep) -> Iter<&str>` and a regex `find` yield
     views into the searched text.
   - An extern declared `-> &str` returns a projection of one `&str`
     or `&String` parameter (RFC-0047 §3's rule for a slice return).

3. **What consumes one.** A `&String` argument at a `&str` parameter
   coerces (`CastKind::Str`, the same site as `CastKind::Slice`: the
   argument becomes the whole-string view). A `&str` at a `&String`
   parameter does not — the callee may need the owned form and the
   copy is written: `to_string(&str) -> String`. `len(&str) -> u64` is
   bytes; `chars(&str)`, `char_at`, `==`, `<`, `starts_with`, `contains`
   take `&str` and therefore also `&String` by the coercion. The
   `string` module's functions move to `&str` parameters where they read;
   `String` where they consume.

4. **Crossing.** A `&str` parameter is `WIDTH = 2` and the handler
   receives Rust's `&str` for the call — no view type, the pair *is*
   `(ptr, len)` of UTF-8 bytes the checker guaranteed (RFC-0059 `Arg`
   with `Form = Pair`). A `&str` return is `Elements`-shaped: two words
   in `rax`/`rdx` stored to the pair.

5. **A view is frame-bound.** As every reference (RFC-0018), a `&str` is
   not stored in an object, a container or a context; it does not
   outlive the `String` it borrows, and a shape write to that `String`
   under a live view is refused by the loans. A `Match` of a regex holds
   `start`/`end` (byte offsets), and its text is `substring(&text, m.start,
   m.end)`.

## What it costs

- `Ty::Str` (the pointee of the reference) in every pass and reader of
  `Ty`; `Kind` unchanged (a view is two words, no `Value` of its own).
- The `string` module's parameters change type; every script that
  passes `&s` is unchanged by the coercion; a script that passed `s` by
  value to a reading function is refused and rewritten with `&s`.
- Byte offsets and `char` indices coexist: `substring`/`len`/`find` in
  bytes, `char_at`/`chars` in scalars — stated per function.

## Rejected

- **`&[u8]` for a string view**: loses the UTF-8 invariant the type
  carries; every consumer would re-validate.
- **`&mut str`**: no operation needs it that `String` does not give.
- **An owned `str` / a small-string value**: a value type with a
  representation of its own; the view plus `to_string()` covers the
  uses.
- **A `&str` stored in an object**: a borrow escaping its frame,
  refused by RFC-0018.

## Consequences

Decisions 2 and 3 hold for the literal and for everything that reads text.
A string literal is `Ref(Shared, Str)`, lowered to `InstKind::ConstStr`,
which the printer writes `const_str T0` with the text in the literals table.
Measured, `let s = "abc"; len(&s)` — six prepared operations before, three
after:

```
ConstLarge  AssignVar  MakeRef  AsSlice  CallWindow  DropValue
Const       Const      CallWindow
```

The `String` allocation, its slot, the `core::as_str` crossing and the
release are all gone; the two `Const`s are the pair's `ptr` and `len` words.
A literal used directly, `len("abc")`, reaches no slot at all.

The bytes live in the module. `prepare` copies each distinct text of a module
once into a `Literals` table, and every `Body` holds an `Arc` of it, so a run
outlives both the MIR module the text came from and the interner it was read
from: the compiler's `Checked::prepare` drops its modules before the machine
runs, which every script with a literal exercises.

A string pattern is unchanged in form and reads either representation:
`match s { "a" => … }` keeps `InstKind::TestLiteral`, which holds the text as
an immediate and compares the scrutinee's bytes, so the pattern's type check
admits a `String` or a `&str` scrutinee against one `&str` literal pattern.

`StringConcat` and `StringEq` read a part or an operand through either
representation, so `"a" + "b"`, `s + "a"` and `s == "a"` all hold with no
copy, and a template's `Node::Text` lowers to `ConstStr` and joins the other
parts in one concatenation. Where an operand of `+` is still an open type
variable the operand bound stays `String`: a value of type `str` does not
exist, so `|k| -> k + "a"` fixes nothing and is refused.

`core::to_string` takes `T = Str` as an instance, and `"x".to_string()` is
the spelling wherever an owned string is wanted: at a `String` parameter, in
a list, an object, a tuple or a context, and at a capture. A body's result is
no longer one of them. A direct call's destination is the two adjacent
registers `assign_slots` gives every value of `SlotClass::Slice`, and
`control::Return<WORD, PAIR>` writes both words of the view into the exit the
caller reads, which is the same two-`Value` run an extern's pair result lands
in. `CallDirect<LARGE, WORD, PAIR>` and `CallDirectAsync<LARGE, PAIR>` carry
it; the asynchronous form suspends with `Pending::Pair`, whose future the
driver stores as two words. `fn my_trim(s: &str) -> &str { trim(s) }` and `fn
head(xs: &[i64]) -> &[i64] { xs }` compile and run at both optimization
levels, a view crosses two body calls and stands as an extern's argument
where it is produced, and `let v = my_trim(&@text); len(&v)` allocates 15 —
the same count `trim(&@text)` allocates with no body between.

Two readers still take one `Value`, and a view is refused where they are what
reads the result. `typeck::ResultCrossing` is the distinction: `Registers`
admits the pair, `OneValue` refuses it, and `check_script` takes which one
applies rather than deciding it. A lambda's result crosses as one at every
call (`acvus_extern::Runtime::call_now`,
`machine::Callable::call_in_window`), and the entry's crosses as one to the
host, which reads it by kind (RFC-0054) and a pair has none. Both raise
`MirErrorKind::ReferenceReturnedFromBody` at compile time.

Which body is the entry is a fact of the graph: `CompilationGraph::entry`
holds its `QualifiedRef`, every builder of a graph states it, and `None` says
the graph has no host. `infer` compares that qref to the body it is checking,
so no name is compared and an entry not called `main` is checked the same
way. `Interpreter::execute` keeps its assert as the machine's own statement
of the contract — reaching it means a program arrived without passing the
checker.

A reference *to* a view — `&&str` — is not the pair and stays refused:
`typeck::is_pair` is the predicate, one level of reference over `Str` or
`Slice`, and it is `prepare::is_slice` on a frozen `Ty`. `is_view` recurses
through a reference and `is_pair` does not, which is the whole difference
between a result the machine lays in two registers and one it lays in a
`Kind::Ref` word that reaches half a pair.

A body does return a bare reference, which is one register: RFC-0064 gives
its result a summary saying which parameters it borrows, and
`validate::borrow_check` refuses one naming a place the run is about to
leave. A body returning a view has the same summary and needed no change
there: writing the argument while the view is live is one conflict, and a
view of the body's own local is `ReferenceToLocalLeavesBody` at the local.

The `string` module's reading half takes `&str`. A producer whose result is
a run of its argument's own bytes returns `&str` — `trim`, `trim_start`,
`trim_end`, `substring` — and every producer that builds new bytes returns
`String`: `concat`, `upper`, `lower`, `capitalize`, `split_str`,
`split_once`, `strip_prefix`, `strip_suffix`, `repeat_str`, `replace_str`,
`pad_start`, `pad_end`. `to_bytes` keeps `String`, which it consumes. The
split family stays owned because its elements are owned (RFC-0047 §3 lends
one run, not a list of them). Two units coexist
and each function states its own: `len`, `find`, `rfind` and `substring` are
in bytes, `char_at`, `chars` and the `pad_*` width in Unicode scalar values.
`substring(s: &str, start: u64, end: u64)` refuses an inverted range, an
offset past the length and an offset inside a character, rather than clamping
as it did.

`as_str` keeps `&String`, being the coercion's own declaration, and is
declared in `core` rather than in `string`: the instruction it lowers to is
the language's one crossing (RFC-0039), so a `&str` parameter is reachable
wherever the checker runs. `regex`'s `text` is `&str` in every synchronous
entry; `regex::replace_with` and `replace_all_with` keep `&String` because
`AsyncGlue` admits `Arg<Form = Pair>` at no arity, so a `&str` parameter of
an asynchronous handler does not compile.

`contains` no longer keeps `&String`. Its bare name is shared with
`iter::contains`, whose first parameter is a value, and with the literal a
`&str` the admission order of RFC-0043 reaches the view in every form:
`contains(&s, "x")`, `contains("abc", "b")`, `s.contains("x")` and
`"abc".contains("b")` all settle on `string::contains`.

The cost of the literal's type, counted over the corpus this landed with:
274 sites in scripts, templates and examples. 217 took `.to_string()` at a
`String` parameter or a `String`-typed binding, branch or body result; 38 in
a list, an object, a tuple or a context, which hold no reference; 8 where an
operand of `+` or `==` was still an open type variable; 5 at a lambda
capture; 5 at `clone`, `hash` or `eq`, which have no instance at `&str`; and
1 where `&x` over a `&str` became `x`. Seven calls that read two texts —
`contains`, `find` — needed no rewrite at all: the `string` module's move to
`&str` is what admits them. Four run-time assertions moved number with the
unit change, all over `"héllo"`: `len` 5 to 6, `find(…, "l")` 2 to 3,
`rfind(…, "l")` 3 to 4. One allocation count moved for a different reason —
`split_str("a,b,c", ",")` unboxes 0 arguments where it unboxed 2, a `&str`
parameter being a borrow rather than a value materialized out of its box.

A `&str` **parameter** now costs two registers rather than a window. The
argument run's form is counted in the runtime's values, so
`len(s: &str)` takes `CallExtern2`, `contains(s: &str, pat: &str)` takes
`CallExtern4`, and only a run past four values is lent its window.
`REGISTER_FORM` moved from three to four for that reason: of the 248
declarations instantiated in the `asm_probe` bench binary, 30 are past the
register forms once a pair costs two values, and four values brings 27 of the
30 back — five brings 29 and six all 30, each at the price of one more
`Runtime::op_*_arguments` method. The operations do not grow: at a zero-sized
handler `CallExtern1` through `CallExtern4` all measure 48 bytes, the `Off`s
fitting inside the padding `Marked` and the take mask leave, and `ops/call.rs`
asserts that the widest one stays inside a cache line.

The `&str` **return** is landed. A declaration written `-> &str` crosses as
`RetStr`, whose `Ret::Of<'a>` is `&'a str`: `Ret::Of` is a generic
associated type over the lifetime the call's arguments were taken at,
because a view of a parameter has no other lifetime to name. A call whose
handler declares a result two values wide is placed into the two adjacent
registers `assign_slots` already gives every value of `SlotClass::Slice`,
which is every `&[T]` and every `&str`; `prepare::extern_call` reads
`Width::ret` and builds `CallPair1` through `CallPair4` or `CallPairWindow`,
and `CallShape::Slice`, `Runtime::op_slice` and the `AsSlice` operation are
gone — `InstKind::AsSlice` now builds `CallPair1`, which is the same
operation `as_str` took under its old name.

RFC-0047 §3's restriction of a pair result to a declaration of one parameter
is withdrawn. `TakenForm<Pair>` is implemented for `InRegisters<1>` through
`InRegisters<4>` and for `InWindow`, so a view crosses at any argument width;
the one run it has no impl for is `InRegisters<0>`, because a declaration
with no parameter has no storage the view could be a projection of. The macro
refuses that declaration ahead of the missing impl, and refuses a `-> &str`
on a `heavy` or `async fn` declaration, whose frame is gone when the call
resumes. The macro also admits a lifetime parameter and drops it: a
declaration whose result may borrow either of two parameters has to name one
lifetime for Rust.

Measured, `let s = "  ab ".to_string(); let v = trim(&s); addr_of(&v) -
addr_of(&s)` answers **2** — the view is the argument's own buffer, two bytes
in, and nothing is copied. Counting every allocation of the run alone, with
compilation and preparation outside the count and the least of three runs
taken, `let v = trim(&@text); len(&v)` allocates **15** against `upper`'s
**17** over the same nine bytes.

`Option<&str>` is admitted by the checker and cannot be held by the machine.
`reject_reference_in_data` refuses a view in an array, an object or a tuple
literal and is not asked at `Some`'s construction, while `MakeSome` and
`TakeVar` each move one of the runtime's values; a two-word payload therefore
loses its length word, which is read back from whatever register follows.
Measured on one program, `let o = Some("abcdefgh"); let p = "ij"; let q =
len(&p); ... s.len() * 100 + q`: 802 is right, `8e937131` answers 202 and the
tree that moved the cut answers 2 — the same defect reading a different
neighbour. Closing it is one call to `reject_reference_in_data` at `Some`,
which is where the type is admitted; the machine's one-value option is not
what is wrong.

## Order of work

`Ty::Str` and the coercion in the checker/lowering → the admission order of
RFC-0043 → the literal as a pair constant → the rest of the `string` module
on `&str` → the machine (`SlicePair` for `&str`, `is_char_boundary` at
`substring`) → the extern crossing (`Arg`/`Ret` at `Form = Pair` for `&str`)
→ regex on `&str`. All of it is landed, and so is the pair destination for a body's result.
What is not: `Option<&str>` above, a result wider than two values, which is
what an aggregate returned by an extern would need, and the checker's refusal
of a view result at the entry, which waits on the graph saying which function
the host calls.
