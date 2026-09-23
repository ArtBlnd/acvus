# acvus RFCs

An RFC is one design decision that the code cannot carry by itself: a
direction, a boundary, or a thing not to build. The decisions live in topic
documents in this directory; each is a section headed by its number.

Check the tree against these rules with:

```sh
cargo test --manifest-path docs/rfcs/lint/Cargo.toml
```

## Rules

Every rule here is one check of `lint/`.

- A topic document opens with its `# ` title. Its only `## ` headings are
  decisions: `## RFC-NNNN: <the ruling as one sentence>`.
- A number is one decision, is never reused, and appears in one heading. A
  number no longer in use is listed under Retired.
- The first line of a decision is `Status: Accepted` or `Status: Proposed`,
  and no other status line appears.
- A decision's rules are the lines that begin `N. ` at the start of a line,
  numbered upward. Nothing else in a document is numbered that way.
- A decision changes in place, in its own section. A document carries no
  dates, commit hashes, `file:line` references, tables, or revision notes
  (`Extends:`, `Supersedes:`, amended, superseded).
- A decision is at most 1500 words.
- Every citation in the tree names a decision that exists: `RFC-NNNN`, or
  `RFC-NNNN rule N` for a rule of it. A rule is cited in that one form.
- The Index below is the one the headings give.

Every sentence of a decision is the decision, its reason, its cost, or an
alternative it rejects. A decision another section owns is pointed to with
`(RFC-NNNN)`, not restated.

## Retired

- RFC-0005
- RFC-0006
- RFC-0008
- RFC-0009
- RFC-0010
- RFC-0015
- RFC-0016
- RFC-0017
- RFC-0022
- RFC-0026
- RFC-0027
- RFC-0032
- RFC-0034
- RFC-0035
- RFC-0036
- RFC-0053
- RFC-0065

## Index

### [context.md](context.md)

- RFC-0014: The run is the unit; a context is a static variable that outlives it
- RFC-0025: A context is a variable of the body that touches it, and a call is bracketed by its callee's summary
- RFC-0033: A space holds a context as its type lays it out and its ops change it

### [effects.md](effects.md)

- RFC-0007: IO runs in source order; `anyorder` declares a region where order is irrelevant
- RFC-0013: An effect is three declared axes — reissue, commutation, task
- RFC-0046: A call's task is an effect — `Sync < Async < Heavy`

### [extern.md](extern.md)

- RFC-0021: A registry is a manifest and a handler table, combined once
- RFC-0023: An ExternFn is declared once, as a Rust function under `#[extern_fn]`
- RFC-0028: An element read out of a container is a loan on it; a reference is one carrier
- RFC-0039: Every type that crosses the boundary says how, through one trait
- RFC-0041: `#τ` is the representation of a slot
- RFC-0054: The host declares what `main` returns
- RFC-0059: The macro emits only calls; the runtime owns the ABI
- RFC-0075: The contract gains `sleep` alone; a handler joins its concurrency in its own future (Proposed)
- RFC-0076: A box is keyed by its payload's canonical type, and an extension holds values through `Erased`
- RFC-0077: A converted `&place` argument is taken out of its slot for the call
- RFC-0080: A fact unsafe code relies on is held by a type or asserted with `unsafe`

### [identity.md](identity.md)

- RFC-0012: Identity is a parameter of a user-defined type

### [instances.md](instances.md)

- RFC-0067: The machine holds no generics; a required instance is one word beside the value
- RFC-0068: A value is read back at a type only where the checker decided
- RFC-0070: An instance requires what its own declaration says

### [machine.md](machine.md)

- RFC-0044: A body is prepared once into a `Code`, and the machine that runs it is synchronous
- RFC-0048: Ownership is the machine's — a value copies, a register is written once
- RFC-0052: An operation is a struct, and the machine calls it once
- RFC-0069: A closure is a code word beside its captures
- RFC-0073: A capture is read in place (Proposed)
- RFC-0074: A diamond of two pure arms is a select

### [mir-opt.md](mir-opt.md)

- RFC-0055: a binary operation on two constants is the constant
- RFC-0056: a loop's `i * k + x` becomes its own counter
- RFC-0057: a `for` loop is one terminator that is its own condition
- RFC-0060: a small pure closure called where it was made is its body
- RFC-0061: a store nothing reads is dead
- RFC-0063: an `if` whose arms rejoin is a `Diamond` terminator
- RFC-0066: a loop is analyzed and normalized in MIR, and the lowerer decides its shape (Proposed)

### [ownership.md](ownership.md)

- RFC-0018: A reference is a type, and only a word or a String copies
- RFC-0024: A pattern matched against a reference binds references
- RFC-0029: Exclusion is checked as the source wrote it; a reference to a reference is a reborrow
- RFC-0064: A reference's extent is its loans

### [positioning.md](positioning.md)

- RFC-0001: acvus is an enrichment pipeline; information is lost at the execution boundary alone
- RFC-0002: ExternFn is the only door to the outside world, and everything inside it is sound
- RFC-0003: The interpreter mediates every execution
- RFC-0004: The language has no map; a map is an extension type

### [representation.md](representation.md)

- RFC-0047: The machine indexes one thing, a slice
- RFC-0050: An aggregate is its components until it escapes
- RFC-0051: A `match` is one dispatch, and it is exhaustive
- RFC-0062: A string slice is a register pair

### [syntax.md](syntax.md)

- RFC-0030: A qualified call names a namespace; a method call is a call on its receiver
- RFC-0038: `Result<T, E>` is a primitive, `?` widens the error, and a trap is not an error
- RFC-0045: `let` binds, `x = e;` assigns
- RFC-0049: `expr as T` is Rust's `as`, and inside a chain it is a leaf
- RFC-0058: A literal says its type
- RFC-0071: A template is a script whose text lines are output

### [tooling.md](tooling.md)

- RFC-0031: `acvus` runs one file as one function, the result alone on stdout
- RFC-0078: A parse recovers past an error, and a tree that holds one cannot be lowered

### [types.md](types.md)

- RFC-0011: A declaration bounds its type variables, and the bound is verified when the variable freezes
- RFC-0019: A shared signature is a name with one polymorphic type and no body, filled by instances the registries declare
- RFC-0020: Operators on language-owned types are instructions; on extension types, a call of a `core` signature
- RFC-0037: Integers have Rust's widths, and a literal takes the width its use demands
- RFC-0040: The compiler chooses an ExternFn's instance, and the IR records it by number
- RFC-0042: The solver separates equality from decision
- RFC-0043: A bare name is a set of signatures, settled by evidence
