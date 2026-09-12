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

- **String, Array, and Object as runtime associated types.** Their layout drives
  performance sharply, and fixing them to one Rust shape — `String`, `Vec`, a
  hash map — serves no workload well. A runtime should choose: a rope, a small
  inline vector, a sorted or perfect-hashed map. Making them associated types
  abstracts the composites the way the value itself is already abstract, and it
  subsumes the large `into_*` above, since those name exactly these composites.
  It is a large change: every operation on a string, an array, or an object goes
  through the runtime's type and the interface it exposes.

## The order these land in

Each change rests on the one before it:

1. String, Array, and Object become runtime associated types, so their layout
   is the runtime's choice.
2. The Store trait fixes how it keeps the basic primitives directly, now that
   their representation is the runtime's own.
3. The large `into_*` convert; with the composites abstracted and the Store in
   place, each is a single cast rather than a per-shape accessor.
