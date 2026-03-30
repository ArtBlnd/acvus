# Identity Type System

## What Identity Is

Every collection literal in acvus receives a unique **Identity** — a compile-time tag embedded in the type that tracks where the value came from. Two values with the same Identity are guaranteed to originate from the same source. Two values with different Identities are guaranteed to originate from different sources.

```
a = [1, 2, 3]       // Deque<Int, Identity(1)>
b = [4, 5, 6]       // Deque<Int, Identity(2)>
c = a                // Deque<Int, Identity(1)> — same source as a
```

Identity is not an annotation. It is not a runtime tag. It is a type-level invariant, enforced by unification and propagated automatically through the type system.

---

## Why Identity Exists

acvus has radically structural typing. `{ a: u64, b: u64 }` is not "a type defined somewhere" — it is a universal structure that exists globally. There is no declaration site. Any value with fields `a: u64` and `b: u64` has this type, everywhere, always.

This is more radical than HoTT's univalence axiom ("equivalent types are equal"). In acvus, structure doesn't merely *imply* equality — structure *is* the type, globally. There is no nominal escape hatch.

This creates a problem: structurally identical values from different sources are indistinguishable. An LLM response `{ role: String, content: String }` and a user input `{ role: String, content: String }` are the same type. There is no way to prevent accidentally mixing them.

Identity solves this. It is the mechanism by which structurally identical values can be distinguished — not by declaring separate types (nominal), but by tracking provenance through the type system.

```
Nominal typing:    distinction = free (every declaration creates a new type),
                   flexibility = expensive (explicit conversions required)

Structural typing: flexibility = free (same structure = same type),
                   distinction = impossible

acvus:             flexibility = default (structural),
                   distinction = opt-in (Identity),
                   erase       = coercion (one-way)
```

Identity exists because the type system is radically structural. If it were nominal, Identity would be unnecessary — declarations would provide distinction. But it would also lose the flexibility of structural subtyping. Identity gives both.

---

## Mechanics

### Two Kinds of Identity

**Concrete** — A fixed, unique identity assigned at a specific point in the program. Collection literals, ExternFn return values, and instantiated Fresh identities produce Concrete identities.

**Fresh** — A placeholder in function signatures. When the function is called (instantiated), each Fresh identity becomes a new Concrete identity. Multiple uses of the same Fresh ID within one signature share the same Concrete after instantiation.

```
// ExternFn signature:
split(s: String) -> (StringRef<Fresh(0)>, StringRef<Fresh(0)>)

// At call site, Fresh(0) → Concrete(42):
(a, b) = "hello,world" | split(",")
// a: StringRef<Identity(42)>
// b: StringRef<Identity(42)>  — same identity, same source
```

### Unification Rules

Identity unification is **invariant** — only identical identities unify:

- `Identity(N)` unifies with `Identity(N)` — same source, OK
- `Identity(N)` does NOT unify with `Identity(M)` where N != M — different sources, rejected
- When identity mismatch occurs, LUB (Least Upper Bound) erases the identity:
  - `Deque<T, Identity(1)>` vs `Deque<T, Identity(2)>` → LUB is `List<T>` (identity erased)
  - `Sequence<T, Identity(1), E>` vs `Sequence<T, Identity(2), E>` → LUB is `Iterator<T, E>` (identity erased)

### Identity on Structs

Identity can tag any structural type, not only collections:

```
{ role: String, content: String, _: Identity(1) }
```

This type cannot unify with `{ role: String, content: String }` (no identity) or `{ role: String, content: String, _: Identity(2) }` (different identity). To use the value in a context that expects the plain struct, the identity must be erased via coercion — a one-way operation.

---

## Properties That Fall Out

The following properties are not separate features. They are consequences of Identity's interaction with existing type system mechanisms (unification, LUB coercion, move checking). No additional analysis passes or runtime machinery is required.

### 1. Taint Tracking

If every IO boundary (ExternFn call) produces a new Identity, then the provenance of every value is tracked at the type level. Values from different sources carry different identities and cannot be mixed without explicit coercion.

```
llm_response:  Sequence<Message, Identity(42), IO>
user_input:    Deque<Message, Identity(43)>
tool_result:   Deque<Message, Identity(44)>

// Can't pass llm_response where user_input is expected — different identity.
// Must coerce to Iterator<Message> (identity erased) to combine them.
// That coercion point IS the "I accept data from mixed sources" declaration.
```

Traditional taint tracking requires either a separate static analysis pass (approximation, false positives) or runtime tagging (overhead). Identity provides taint tracking as a natural consequence of the type system, with zero runtime cost and no separate analysis.

### 2. Zero-Cost Rejoin

If a value is split into parts, the parts share the source's Identity. Rejoining parts with the same Identity is a guaranteed-safe operation that can be optimized to O(1) — the parts reference the same backing buffer.

```
str: String<Identity(42)>
(a, b) = str | split(",")
// a: StringRef<Identity(42)>
// b: StringRef<Identity(42)>

a ++ b  // Same identity → compiler knows: same buffer → slice adjustment, O(1)
a ++ c  // Different identity → actual copy, O(n)
```

This decision is made at **compile time**. No runtime identity check. No runtime branching. The type already carries the information.

This generalizes beyond strings: split a Deque, take a slice, window over a Sequence — any operation that produces parts of a whole preserves Identity, and any operation that recombines same-Identity parts can be optimized to avoid copies.

### 3. Static Copy-on-Write

Rust's `Cow<'a, B>` is a runtime enum (`Borrowed | Owned`) that branches on every mutation. Identity combined with move checking provides the same semantics at compile time:

- **Identity** answers: "where did this come from?" (provenance)
- **Move check** answers: "is the original still alive?" (ownership)

Together:

```
str: String<Identity(42)>
ref = str | slice(0, 5)   // StringRef<Identity(42)>

// If str has been moved (no other references to Identity(42)):
//   → ref is the sole owner → mutate in place, zero copy

// If str is still live (other references to Identity(42) exist):
//   → copy required
```

This is decided at compile time. No runtime tag. No runtime branch. Cow is free.

### 4. Structural-to-Nominal Bridge

Identity turns structural typing into optionally nominal typing:

- **Without Identity:** pure structural typing. `{ a: u64, b: u64 }` from any source is the same type. Maximum flexibility.
- **With Identity:** nominal-like safety. Same structure from different sources = different types. Cannot be mixed.
- **Erase Identity:** explicit coercion back to structural. One-way — once erased, cannot be recovered.

This differs from phantom types (Haskell, Rust `PhantomData<T>`):

| | Phantom Types | Identity |
|---|---|---|
| Safety | nominal separation | nominal separation |
| Conversion | manual functions required | LUB coerces automatically |
| Unaware code | must be generic over the marker | just works (accepts identity-free type) |
| Direction | both directions blocked | one-way (erase only) |
| Declaration | manual marker type definition | automatic allocation |

Phantom types make the **user** pay for safety (manual conversions, generic wrappers everywhere). Identity makes safety the **default** and flexibility the opt-in (via coercion).

### 5. Arena / Region Tracking

Same Identity = same physical memory region. This is arena tracking at the type level:

- Values with the same Identity can share a backing allocation
- The compiler can group same-Identity allocations into a single arena
- Deallocation of an Identity's arena frees all values from that source at once
- No per-object tracking needed — the type system already groups them

---

## The One-Way Valve

Identity erase is a **projection that loses the provenance distinction**. This is irreversible by design.

```
// Has Identity → can call functions that require provenance
fn needs_provenance(msg: { content: String, _: Identity }) -> ...

// No Identity → cannot call such functions
fn agnostic(msg: { content: String }) -> ...

msg_with_id: { content: String, _: Identity(42) }
msg_erased:  { content: String }

needs_provenance(msg_with_id)  // OK
needs_provenance(msg_erased)   // Type error — Identity cannot be recovered
agnostic(msg_with_id)          // OK — Identity present but not required
agnostic(msg_erased)           // OK
```

This creates a natural trust boundary. Code that handles raw, provenance-tracked data operates in the Identity-aware world. Code that has intentionally mixed sources operates in the erased world. The boundary between them is visible in the types and enforced by the compiler.

---

## Design Decisions

**Why invariant unification, not covariant?**
If Identity were covariant, `Deque<T, Identity(1)>` could silently unify with `Deque<T, Identity(2)>`. This would destroy the entire point — provenance would be lost silently. Invariant unification forces explicit coercion, making provenance loss visible in the code.

**Why automatic allocation, not user-declared?**
User-declared identities (like phantom type markers) add friction and require the user to understand the Identity system. Automatic allocation means Identity works without the user thinking about it — collection literals, IO results, and split operations just get identities. The system is opt-out (erase when you don't care), not opt-in.

**Why erase via LUB, not via explicit cast?**
LUB coercion is the existing mechanism for type mismatch resolution in acvus. Reusing it means Identity erase is a natural part of the type system, not a special-case operation. When two different-Identity values need to coexist, the type system automatically finds their common supertype (which lacks Identity). This is the same mechanism used for all other type coercions — no new concept for users to learn.

**Why not separate the provenance system from the type system?**
Separation would mean provenance information is invisible to unification, optimization, and move checking. The power of Identity comes precisely from its integration — the type system reasons about provenance the same way it reasons about element types and effects. A separate system would require a separate analysis pass and could not inform optimizations like zero-cost rejoin or static CoW.
