# RFC-0032: An object crosses the boundary as its fields

Status: Accepted; the representation superseded by RFC-0050 rules 4 and 8
Date: 2026-09-16
Extends: RFC-0022, RFC-0023

An object is no longer a map from field name to value. It is its type's field
names in RFC-0050 rule 8's order, shared by every object of the type, and one
value per field in that order. The crossing writes and reads those values by
position, so the field-by-field conversion below stands and the name lookup it
used to go through does not.

## Ruling

A Rust struct with `#[derive(TyArg)]` declares the object type its fields
spell, as before, and now also crosses as that object: an ExternFn takes
one as a parameter and returns one, and the script sees `{ x: 1, label:
"a" }` with every field reachable. The crossing converts, field by field:
into the language the struct becomes an object whose keys are the field
names and whose values are the fields crossed by their own types; out of
the language an object is read the same way. A container of such structs
— `List<T>`, `Array<T, N>`, `Option<T>` — converts each element.

A scalar — `Int`, `Float`, `Bool`, `Byte`, `String`, `Unit` — converts as
itself, so a container of scalars declared in Rust (`List<u8>`,
`Option<String>`) crosses as the runtime's container of values, which is
what a script's `List<Byte>` or `Option<String>` is.

The language's object is the extern contract's `Obj<V>` at `V = Value`,
as its array is `Arr<Value, ()>` (RFC-0022), so a handler that wants the
object as the runtime holds it can take `Obj<Rt::Value>` as it is.

The runtime contract gains one method, `symbol(&str) -> Astr`: the name a
field key is at run time. A crossing is chosen where the glue expands, as
RFC-0022 chose it: a type with a `Repr` crosses as its shape, a type that
converts crosses by conversion, any other crosses as it is. RFC-0039
replaced that choice with one `Cross` trait per type.

## Rationale

`#[derive(TyArg)]` declared an object type the checker accepted and the
runtime could not honor: a `Point` returned by a handler was erased as the
Rust value, and a script reading `.x` found no object. The LLM registry
already returns `ChatResponse { content: List<OutputMessage>, .. }` and no
script could read it.

The struct and the object are different layouts, so the conversion is
O(fields), paid once per crossing and never on access. A `Repr` states a
shared layout, which these two do not have. The third tier is what leaves
a value with the runtime's own shape crossing without conversion.

## Not built

- No enum crossing: a Rust enum is not an object (built in RFC-0036).
- No renamed or skipped fields: a field is a key of the same name.
- No object with a generic field: a structural object has no type
  parameters, as before.
- No `&T` / `&mut T` parameter of a converted type: a converted value has
  no storage of its own type to read through, so such a parameter is a
  runtime error at the crossing until the boundary can lend a converted
  view.

## Consequences

- `acvus-extern`: `Obj<V>`, the `Cross<Rt>` trait with impls for the
  scalars, `Option<T>`, and `Arr<T, N>`, `Crossing<T, Rt>` with the
  conversion tier between `Repr` and as-is, `Runtime::symbol`.
- `#[derive(TyArg)]` also derives `Cross<Rt>`; the macro's glue names the
  runtime in every `Crossing`.
- `acvus-ext`: `List<T>` converts when `T` does.
- The interpreter's `Object` is `Obj<Value>`; `AcvusRuntime::symbol`
  interns through the run's interner.
