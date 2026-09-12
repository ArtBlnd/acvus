# Runtime and Value: the extraction baseline

A runtime supplies the language's values, and the language extracts typed
values back out. The baseline is one entry, `materialize::<T>`, dispatched by
the target type through `FromValue`; a runtime supplies the raw bits of a small
value and constructs values, and no more of the value taxonomy lives in the
contract.

## What the baseline is

- `Runtime::materialize::<T>(value)` is `<T as FromValue>::from_value(value)`.
  Extraction is driven by the target type, not chosen from a menu of typed
  accessors.
- The small values — unit, int, float, bool, byte — are folded into one
  accessor, `small_bits(value) -> u64`, and `FromValue` reinterprets those bits
  per type. A type mismatch here is a bug the type checker prevents, so it
  panics rather than returning an error.
- The `into_*` family for the small values is gone: extracting them no longer
  names their type in the runtime contract.
- The string and array composites are runtime associated types, not fixed Rust
  shapes: `type Str: AsRef<str>` and the GAT `type Array<T>: AsRef<[T]> +
  IntoIterator<Item = T> + FromIterator<T>`. The host chooses the layout;
  `AcvusRuntime` declares `Str = String`, `Array<T> = Vec<T>`. `into_string`
  became `into_str -> Self::Str`, and `array`/`into_array` speak
  `Self::Array<Self::Value>`.

## Why only materialize

The `into_*` menu names the language's value taxonomy in the runtime contract —
one method per built-in shape. That is the interpreter's own taxonomy leaking
into the contract. `materialize::<T>` lets the target type drive extraction, so
the contract is agnostic to the taxonomy: a new extractable type is a
`FromValue` impl, not a runtime method.

Materialize-only is the natural extraction for a uniform erased value: bits for
a small value, a cast for a boxed one — two cases, no per-shape menu. The menu
exists because the current value is a tagged enum whose per-shape variants force
per-shape extraction. "Only materialize" and "an erased representation" are the
same decision seen twice.

## Open decisions

- **The large `into_*`: string, array, tuple, object, option, extern.** They
  are the tagged representation's residue. Removing them against the tagged enum
  is lateral churn; they dissolve when the representation becomes erased — a
  small-bits word and a boxed pointer — where materialize's large case is a
  single cast. That representation change also requires the interpreter's
  tag-dispatching operations, equality and display among them, to become fully
  typed rather than reading the value's tag.

- **Object's layout.** String and array are now associated types (above);
  object is not. Object cannot simply be dropped from the contract — a Rust
  struct crossing the boundary maps to an acvus object through
  `#[derive(TyArg)]`, which emits `Runtime::object`/`into_object`, so the
  `FxHashMap<Astr, _>` signature has real consumers. Its layout freedom is a
  different question from string and array: acvus objects are fixed once
  formed — an unnamed struct — so the right shape is not a map at all but a
  projection over slots. That projection lives in the host, not the IR: the IR
  carries no layout or size, and even the field identifier `Astr` is a host
  handle (a `u64`) that a transpiling host like kovac would lower to its own
  index. So object projection is a concern of MIR-executing hosts, not the
  universal `Runtime` boundary the way string and array are. Deferred with
  key = `Astr`.

## The order these land in

Each change rests on the one before it:

1. String and array become runtime associated types, so their layout is the
   runtime's choice. **Done.** Object is held back: it needs projection, which
   is host-tier work, not a boundary associated type.
2. The Store trait fixes how it keeps the basic primitives directly, now that
   their representation is the runtime's own.
3. The large `into_*` convert; with the composites abstracted and the Store in
   place, each is a single cast rather than a per-shape accessor.

Object projection — an unnamed struct over slots, keyed today by `Astr` — is a
separate line, belonging to MIR-executing hosts rather than the boundary.
