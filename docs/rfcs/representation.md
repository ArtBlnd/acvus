# Representation

How the machine lays out what the compiler hands it. A slice or a string view
is two registers and never a `Value`; an object, tuple or enum is its
components until it escapes, and the heap is the spill; a `match` is one
dispatch over a key read once, closed by the scrutinee's settled type.

## RFC-0047: The machine indexes one thing, a slice

Status: Accepted

The machine indexes exactly one thing, a **slice**: Rust's `&[T]` /
`&mut [T]`, a pointer and a length. A container that can be indexed says so by
implementing `core::as_slice` (and `core::as_slice_mut`), a container
signature (RFC-0028); the compiler takes the slice with `AsSlice` and indexes
it with `Index` / `IndexSet`. No instruction knows a container's layout:
`AsSlice` runs the container's own `as_slice` instance.

1. **One element width.** Every container the machine can slice stores
   `Vec<Value>` (the runtime's `Array` is `Arr<Value, ()>`, a `Vec<T>` reaches
   the store as `Vec<Value>`, RFC-0039). A converted container (`Vec<f64>` in
   a Rust signature) has no storage of its own and cannot be sliced. A slice
   is always `&[Value]`, element width 16, and the interpreter's one ABI fact
   is its own `Value`.
2. **Types.** `&[T]` and `&mut [T]` are reference types (RFC-0018): a slice
   holds its container's loan and never outlives it. `&mut [T]` is an
   exclusive take; a live `&[T]` refuses `as_slice_mut` and `IndexSet` on its
   container. An element write leaves a slice's pointer and length where they
   were.
3. **Signatures and `AsSlice`.** `core::as_slice<T>(c: &C) -> &[T]` and
   `core::as_slice_mut<T>(c: &mut C) -> &mut [T]` are bare names settled by
   the container's evidence (RFC-0043). `Vec<T>` and `Array<T, N>` implement
   them; `Deque<T>` and `Map` do not. They live in the `machine` table, which
   `resolve_fn` never reads, so a script cannot name them: a slice is made
   only by an `AsSlice` the lowering emits, at an index or at a `&v` coerced to
   a slice parameter (rule 6). The lowering emits `AsSlice { dst, container, mutability, instance }`,
   not a `FunctionCall`: the instruction kind states that it is a pure,
   infallible borrow projection of its container, so `code_motion` classifies
   it as a shared borrow structurally, as it does `Ref`. A pair result — a
   slice or view returned by an extern — is a projection of one of the
   declaration's parameters: it is admitted at any argument width and refused
   for a declaration of no parameter, which lends no storage to project.
4. **`Index { dst, slice, index, mode }`** reads element `index`. The index
   is `u64` and nothing else; the one check is `index < len`, panicking with
   Rust's text (`index out of bounds: the len is {len} but the index is
   {index}`). `mode` is decided statically by the checker from the element
   type: `Copy` for a word element (`dst` is the element), `Ref` otherwise
   (`dst` is a `Ref` into the slice's storage carrying its loan).
   `IndexSet { slice, index, value }` writes through `&mut [T]` with `assign`
   semantics. Each carries `bound`, `Checked` or `Proven`; the user cannot
   ask for `Proven`, only rule 7's pass writes it, and `prepare` emits the
   unchecked form for it alone. `InstKind::ArrayIndex` stays distinct: it
   moves an element out of an owned scrutinee at a constant position, which
   is destructuring, not indexing.
5. **Syntax and place semantics.** `a[i]` is a place, as in Rust. In value
   position its element must be a word, else the refusal is Rust's
   ``cannot move out of index of `Vec<T>` `` and the way is `clone(&a[i])`
   (RFC-0028). `&a[i]`, `&mut a[i]`, the inner level of `a[i][j]` and a
   method receiver's auto-borrow are `Index` in `Ref` mode. `a[i] = v` is
   `as_slice_mut` + `IndexSet`. A statement beginning with `[` is an array
   literal; a postfix `[` binds to the expression before it.
6. **Representation.** A slice is two adjacent registers — `ptr` then `len`
   (`SlicePair`) — given to one MIR value by `prepare`; the second register is
   fixed at prepare and never computed in a `run`. It is never a `Value`: not
   an element of a container, not a capture (the loans refuse a borrow that
   outlives its body). It copies like every reference (`&[T]` owns nothing),
   and both registers open as `Kind::U64`. `Index`/`IndexSet` read `ptr`,
   `len`, `index` and the element at `ptr + index * 16`: register loads, no
   box, no allocation, no drop. `&str` is the same pair (RFC-0062).

   The pair crosses an ExternFn boundary in both directions. A pair result
   comes back as `Elements` in two return registers and is stored to the
   pair. A parameter is `Slice<T, Rt>` or `SliceMut<T, Rt>` by value, two of
   the argument run (`Arg` with `Form = Pair`, RFC-0059). `&v` reaches such a
   parameter by a coercion at the argument: where the parameter is
   `Ref(m, Slice(E))` and the argument `Ref(m, C)` with `C` a container that
   declares `as_slice` at `m`, the checker records `CastKind::Slice` and the
   lowering emits the `AsSlice` there, so no copy of the container is made. A
   coercion whose container is still a variable waits for the body's solve. A
   `heavy` or `async` declaration refuses a slice parameter: the elements
   belong to the frame the caller laid the arguments on, which is gone when
   the call resumes.
7. **Bounds-check elimination is an interval domain and nothing more.** Each
   integer value carries `[lo, hi]`, each endpoint a constant, another SSA
   value, a slice's length or a container's element count, plus a constant.
   `± constant` is exact interval arithmetic at the program's `+` and `-`
   (RFC-0037), ⊤ where a pass's could wrap; φ is join, widened at a loop
   header; everything else is ⊤. The edge of a
   comparison that holds `i < n` sets `i.hi = n − 1`; a range `for` gives its
   counter `[at, hi − 1]`. A call's result reads its callee's postconditions
   (RFC-0082 rule 4): `ret = len(x)` makes it `x`'s length. A slice's length
   is fixed while the slice lives. A container's element count holds from
   the `len` that read it until an instruction that may write any storage —
   an assignment, a move out, an element write, any call but a pure extern
   of words and shared references — and an `AsSlice` of the same target and
   path while it holds makes it the slice's length. The write condition is
   the domain's own, coarser than `Loans::storage_effect`. `Index(s, i)` and
   `IndexSet(s, i, _)` are marked `Proven` when `i.hi`, followed through the
   upper bounds of the values it names, reaches `len(s) − k` with `k ≥ 1`.
   `validate::bounds` runs the same domain over every module the pipeline
   returns and refuses a mark it does not derive. `a[i + 1]`, `i ≤ n` and an
   index derived elsewhere stay checked.
8. **Motion.** `AsSlice` hoists as a shared borrow of its container, out of
   every loop that does not write the container; a checked `Index` does not,
   because it may panic (RFC-0007); an unchecked `Index` whose slice and index
   are invariant hoists as pure. Where a row is itself an `Index`
   (`a[li][i]` in an inner loop), `code_motion` makes the inner `Index` the one
   a dominating block already computed, and the row's `AsSlice` then rises.
   Where every touch of a storage in a loop goes through slices of it and one
   is exclusive, the loop takes one `as_slice_mut` in its preheader and reads
   through it; any other touch (a `push`, a call taking the container) leaves
   the slices in place. A shared borrow is keyed by storage, instruction kind,
   every operand and the reference type, so slices of `m[z]` and `m[one]` stay
   two.

A length is `u64`, meeting the index; conversions are `as` (RFC-0049).

**Why.** While the bound check lives inside an extern, the compiler can
neither eliminate it nor hoist the call (RFC-0007); a slice instruction puts
both facts in its kind. The interpreter must not know any container's layout,
so the one layout it learns is Rust's `&[Value]`.

**Cost.** `Deque` and `Map` keep `get` and its call, by their own choice. A
Rust body reading a slice parameter walks 16 bytes per element and extracts a
word from each: a slice parameter gives the script its container back, not a
Rust body a native slice.

**Rejected.**
- `core::index_k` externs — the check stays inside the call; it cannot be
  eliminated or hoisted.
- `Index` reading a container's layout — the interpreter does not know
  containers.
- `as_slice` as a `FunctionCall` hoisted by a `total` marker — a general
  mechanism for one fact the instruction kind carries.
- Folding `ArrayIndex` into `Index` as a `Move` mode — two representations in
  one instruction.
- An `i64` index — a negative check for nothing; `u64` is Rust's `usize`.
- Induction-variable or recurrence analysis for bounds — the interval domain
  with one symbolic endpoint covers `while i < len { a[i] … i = i + 1 }`.
- A fat-pointer `Value` (24 B) — widens every value for one kind.
- A one-box slice (`Slice { ptr, len }` erased as a `Large`) — every
  `AsSlice` allocates, every slice drops, every `Index` reads through the box.
- The length inside `Value`'s head word — every crossing pays the layout, and
  `Option<Value>` loses its niche (24 bytes, returned through `sret`).
- The length as a third operand of `Index` — every crossing would carry the
  length beside the pointer in the compiler's hands; the pair is one MIR value
  that `prepare` lays in two registers.
- Unchecked indexing as a language-level `unsafe` — only the compiler's proof
  emits the unchecked form.

## RFC-0050: An aggregate is its components until it escapes

Status: Accepted

1. **An aggregate is its components.** In a body an object is its fields, a
   tuple its positions, an enum or a `Result` its `(tag, payload)`. **No
   `Make` exists for a value that stays in its body.** The MIR's scalar
   replacement removes an aggregate whose storage does not escape (rules
   10–13), `prepare` gives an aggregate that needs an address a run of
   registers (rules 2–3), and only an aggregate that must outlive the frame is
   realized (rule 4).

2. **A frame is registers, and an aggregate that needs an address is a run of
   them.** A frame holds up to 320 registers, scalars below runs. The scalar
   colouring gives each value and storage the lowest register free over its
   live range; a storage lives until the last read of it or of a loan on it,
   and equal hoisted constants share one. Runs hold flat layouts (rule 8),
   placed after the scalars, deepest loop first, spilling past the bound to
   the heap (rule 4), shallowest loop and longest range first. Scalars past
   the bound, over 64 parameter registers or over 16 laid arguments are
   refused at compile time with the count. A mark word covers 64 registers;
   the sweep releases a run's `Large` fields as any register's.

3. **A reference to an aggregate is a projection, and an addressed aggregate
   has one home.** `&agg` / `&mut agg` is one word, `Kind::LargeRef`, pointing
   at the flat layout in a run or on the heap; field `i` is `base + off(i)` in
   both. The unit placed is the **SSA web** — the values joined by an
   `Assign`, a block argument or their construction — which takes one base for
   its whole live range, so a write through a projection is never stale
   against a register copy. A web is placed only where every mention of every
   member has a register form, so an instruction kind added to the IR takes
   its web to the heap. The loans keep a projection inside its frame.

4. **The heap is the spill.** An aggregate is realized — one `Large` holding
   the same flat layout behind its header — only where its value must outlive
   the frame: in a container or a realized aggregate, into `Commit`, captured
   by value, returned, kept by an extern, handed to `Spawn`, or in a web the
   emitter cannot lower. That realization is the only `Make`; `prepare` emits
   it, and the type system has no realized/unrealized distinction. The frame is preferred
   to the heap even where native code would run a little faster on it,
   because on wasm the allocation is the cost.

5. **An aggregate result is its components in the caller's destination.** An
   extern returning a struct receives `Out<'_, Rt>`, a `&mut [Value]` over the
   destination the caller placed, a run or a realized object's flat body,
   and writes the components there. A derived enum's result is one heap
   value.

6. **At a Rust boundary a borrowed aggregate crosses as a projection, and
   nothing is allocated.** `#[derive(TyArg)] #[projection]` emits, for a
   struct, `SRef<'a>` / `SMut<'a>` with one borrowing field per declared
   field, and for an enum `ERef<'a>`, `EArms<'a>` and `EMut<'a, Rt>`, whose
   `set` rewrites `[tag, payload…]` in place. The GATs are on `Borrowed`,
   which has no runtime parameter, so projections nest. A handler declares the
   projection; `&S` is a compile error naming it. A by-value `S` is matched
   exactly (RFC-0042); a projection is matched **at least**, borrowing only
   the fields it names, and an argument whose closed field set lacks one is
   refused at the call. Field positions come from the settled type as a
   per-site glue table built at `prepare`. A struct variant has no
   projection. No storage holds Rust's `Option<T>` or `Result<T, E>`; each
   is lent as `Option<T::Ref<'a>>` or `Result<T::Ref<'a>, E::Ref<'a>>`
   anywhere, arms derived projections included, tag read-only.

7. **A container's element is realized, and a container is never a component
   set.** `Vec`, arrays and deques hold heap aggregates (rule 4); an array is
   not an object with positional fields.

8. **An aggregate's layout is flat.** A layout is its fields' layouts
   concatenated: a word or `Large` field is one `Value`, a nested aggregate is
   its own layout inline, an enum is `[tag, payload]` with the payload at the
   widest variant, and `off(i)` is the prefix sum fixed by the settled type
   (RFC-0041), in a run and on the heap alike. **Every object type's field
   order is its field names sorted as strings**, declared structs included.
   A field the settled union has and a construction lacks is `Undef` at its
   offset. **The tag word is the variant name's interned number**
   (`Astr::bits`), written alike by a run and by a heap variant; the dispatch
   is a scan of tag words. `Result` is this layout with tags `Ok` and `Err`.

9. **`Option<Aggregate>` is flat over the run**: the first `Value` is
   `Kind::None` for `None` and the payload's first component otherwise —
   RFC-0039 rule 6 at the width of the payload.

10. **One escape predicate, complete.** `analysis::escape` decides whether a
    storage slot reaches anything outside its body, through any operand that
    hands a value on. A storage escapes when it or a value naming it escapes;
    under a path, a `Ref` escapes its storage outright. The arms enumerate
    all of `InstKind`, so a new instruction breaks the build. Its one caller
    is scalar replacement; run placement asks rule 3's web question, because
    an `Assign` between two web members is an escape here and a join there.

11. **Scalar replacement reuses the one SSA builder.** A slot of object or
    enum type that does not escape is replaced by one register per field, or
    a tag and a payload register, as `SsaVar::Part` through the Braun builder
    unchanged; no place names a tag. The pass runs
    after inlining and immediately before `ssa_pass`, so it never sees a
    `Drop`; `drop_insertion` later places the part registers' drops.

12. **Every use is covered, or the slot keeps its aggregate.** A whole `Ref`
    of the slot that is only read through is an alias, not an escape. Any use
    the pass has no rewrite for refuses the slot, and one escaping use
    anywhere keeps the aggregate for the whole body. An enum keeps its
    aggregate when a variant carries nothing or two variants disagree on the
    payload type.

13. **A tag settled on an edge is threaded, and an empty forwarder
    collapses.** No IR form hands a numeric tag phi to a later pass, so
    threading runs inside scalar replacement: a dispatch whose own tag is
    settled becomes a `Jump`, and otherwise each incoming edge whose tag is
    settled leaves for its arm directly. A dispatch that keeps an unsettled
    edge needs a compare, which exists only for two edges; a wider one keeps
    its aggregate. Threading maintains or demotes the dispatch's `Diamond`
    join (RFC-0063). `optimize::forward` removes, to a fixpoint, a
    block with no parameters, only `Nop`s and an unconditional `Jump`, holding
    a loop's header, entering block and exit blocks; without it an arm the
    sink emptied breaks region recognition.

A register holds one kind class (RFC-0052), so there is no conditional
`Drop`: a payload register whose variants disagree in class is whole-typed
and released by kind.

**Why.** Allocation and a hash lookup per access were what an aggregate cost;
a CPU keeps a value in registers until it must spill, and the frame does the
same. The interned name is the only tag
numbering a writer typed by its one variant and a reader typed by the union
agree on.

**Cost.** Two forms of every aggregate read, register and projection, chosen
once per web. The heap form still pays the allocation wherever rule 4
applies. A partial projection loans the whole storage.

**Rejected.**
- A realized/unrealized mark in the type system — where a value lives is the
  machine's fact.
- A universal `(tag, payload)` with conditional drops — a register is
  whole-typed and releases by kind.
- Realizing at construction with a shape table, or a hash-map layout — an
  allocation and a query per field access remain.
- Runs carved among the scalar registers, or coloured inside the scalar
  allocator — a contiguity constraint on a file the scalars already fill.
- A fixed 256-byte `Cell` cut into slots — four fields before a spill.
- A `Context` associated type on the runtime — two `&mut` into one frame.
- Return in the callee's own component registers — unreadable through a
  closure value.
- A per-site choice of register or projection — a write through the
  projection leaves a register copy stale.
- Nested aggregates realized as the outer's `Large` fields — boxes on every
  `&line`.
- A dense tag ordinal — the writer and reader disagree on the type that
  would number it, and a table of blocks measured slower than the scan.
- Interned-symbol order for fields — first interning differs between
  programs.
- Partial escape analysis — it buys nothing where every aggregate escapes on
  every path or none.
- `exhaustive`'s old three-kind predicate — a miscompile for scalar
  replacement.

## RFC-0051: A `match` is one dispatch, and it is exhaustive

Status: Accepted

1. **Syntax.** Rust's: `match e { P1 => e1, P2 => { stmts; tail }, _ => e3 }`
   is an expression; every arm has the type of the whole, `!` admitted. Arms
   are the grammar's patterns plus `_`. There is no `MatchBind` statement;
   `if let P = e { .. } else { .. }` is two-arm sugar.
2. **An arm contributes no variant.** A pattern naming a variant the
   scrutinee cannot hold is an unreachable arm and is refused, types as
   written (RFC-0043). Two arms naming one key — a repeated tag or literal —
   are refused, naming the key. The language has no warning axis.
3. **Exhaustiveness is decided in `validate`, on the MIR.** A `match` is
   closed by the scrutinee's settled type: an enum's variants are the union of
   every construction the value can flow from (RFC-0041), wherever the value
   came from; `Option` and `Result` have two. `Bool` is the one literal space
   arms close; integers, chars and strings are open and need a catch-all. A
   `match` whose arms are not one dispatch — a nested refutable payload, a
   tuple, a list — is refused by `typeck` unless it has a `_` arm.
4. **One seam.** Exhaustiveness asks one function, `known_variants(scrutinee)
   -> Known::{Closed, ClosedBuiltin, Open}`; the scrutinee's type is the
   answer. A tag of a type that names no variant set is `Open` and refused.
5. **MIR and machine.** `Switch` keys its arms by one kind of `SwitchKey` —
   `Tag`, `Int`, `Bool`, `Char`, `Str` — fixed by the scrutinee's type; there
   is no float key (float equality is not a jump) and no byte-string key (a
   byte string is a list), so such a `match` keeps its test chain. `Switch` is
   a terminator (RFC-0057); `default` is always a real edge — the catch-all,
   or else the last arm, untested — which is sound because rule 3 proved some arm
   holds, so no `run` decides that none does. The machine reads the key once
   and **scans** the arms: a tag scan over tag words (RFC-0050 rule 8), a
   word compare for integers and chars, a string scan in source order. A
   `Bool` dispatch is the machine's two-way branch. A flat `Option`'s tag is
   its kind (RFC-0039). A dispatch whose tag is one constant is threaded to a
   jump (RFC-0050 rule 13).
6. **A place lent to a terminator is live across it.** Drop insertion reads a
   terminator's uses through the loans, as it reads an instruction's, so a
   scrutinee lent to a `Switch` is not dropped before the dispatch reads it.

**Why.** A `match` is one question with one read; a chain of tests per arm
reads the tag every time and says nothing about exhaustiveness. Enums are
structural and unify by union, so the settled type — not the arms, not
`typeck`'s open union — is what closes the set.

**Rejected.**
- Folding `MatchBind` chains into `Switch` without exhaustiveness — the
  silent fall-through stays.
- Exhaustiveness in `typeck` — the union is open while checking: either
  unsound or refuse-all.
- Arms contributing variants — makes every match exhaustive by construction.
- A warning for an unreachable arm — there is no warning axis.
- A jump table over tags — a tag word is a sparse interned number, and the
  nearest form, a hashed table, measured slower than the scan.
- Sorted string keys and a binary search — a string compare starts with the
  length, and a `match` has a handful of arms.

## RFC-0062: A string slice is a register pair

Status: Accepted

1. **`&str` is a type**: `Ref(Shared, Str)`, a shared borrow of a `String`'s
   UTF-8 bytes. There is no `&mut str` and no owned `str`. In the machine it
   is the pair of RFC-0047 rule 6, the length in bytes. `&&str` is not a pair: the
   pair predicate looks through one level of reference only.
2. **What produces one.** A string literal is a `&'static str`: two constant
   words (`ConstStr`), its text in a per-module `Literals` table every body
   holds by `Arc`, so a run outlives the module and interner it came from. An
   owned string is written `"…".to_string()`. A producer whose result is a run
   of its argument's bytes returns `&str` — `trim`, `trim_start`, `trim_end`,
   `substring`, `split`'s elements, a regex `find` — and a producer that builds
   bytes returns `String`. An extern declared `-> &str` returns a projection
   of a parameter (RFC-0047 rule 3).
3. **What consumes one.** A `&String` argument at a `&str` parameter coerces
   (`CastKind::Str`, at the site of `CastKind::Slice`); a `&str` never coerces
   to `&String` — the copy is written. The `string` module takes `&str` where
   it reads and `String` where it consumes. Each function states its unit:
   `len`, `find`, `rfind`, `substring` in bytes; `char_at`, `chars`, the
   `pad_*` width in scalar values. `substring` refuses an inverted range, an
   offset past the length and an offset inside a character. A string pattern
   admits a `String` or `&str` scrutinee. An operand of `+` whose type is
   still open stays bound to `String`, since no value of type `str` exists.
4. **Crossing.** A `&str` parameter is two values wide and the handler
   receives Rust's `&str`; a `&str` return is two words (`RetStr`, whose
   `Ret::Of<'a>` is a GAT over the arguments' lifetime). A `heavy` or `async`
   declaration may neither take nor return `&str`: its frame is gone on
   resume.
5. **A view is frame-bound.** A `&str` is not stored in an object, a
   container, an option, result or enum payload, or a context, and does not
   outlive its `String`; its extent is its loans (RFC-0064). A payload may
   hold a plain reference (RFC-0079 rule 10), but it is one value, and a
   view is the pair. A regex `Match` holds byte
   offsets, and its text is `substring(&text, m.start, m.end)`.
6. **A body returns a view in the pair; a lambda does not.** A direct call's
   destination is the two registers of the pair, and a body's `Return` writes
   both words. A lambda's result crosses as one `Value` at every call, so a
   view there is refused at compile time (`ReferenceReturnedFromBody`); the
   entry's result is the host's (RFC-0054).

**Why.** Every operation that yielded part of a string allocated a new one,
and every literal use was a `String`; the slice's pair already solved this
for `Vec<T>`.

**Cost.** Byte offsets and scalar-value indices coexist, stated per function.

**Rejected.**
- `&[u8]` as the string view — drops the UTF-8 invariant; every consumer
  re-validates.
- `&mut str` — nothing needs it that `String` does not give.
- An owned `str` or small-string value type — a new representation; the view
  plus `to_string()` covers the uses.
