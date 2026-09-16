# RFC-0036: A Rust enum crosses the boundary as the language's enum

Status: Accepted
Date: 2026-09-16
Extends: RFC-0032

## Ruling

`#[derive(TyArg)]` on a Rust enum declares the language's enum of the
same name and crosses as it. A unit variant has no payload; a tuple
variant has one field, and that field is the payload; a struct variant's
payload is the object its fields spell (RFC-0032). A script matches a
returned value with `Shape::Circle(r) = s` and builds one to pass in with
`Shape::Circle(3)`; an extern fn receives and returns the Rust enum.

    #[derive(TyArg)]
    enum Shape {
        Dot,
        Circle(i64),
        Rect { w: i64, h: i64 },
    }

The language's variant is the extern contract's `Variant<V>` at `V =
Value`, as its object is `Obj<V>`: the tag and an optional payload. A
field of an extension type (`Regex`, `Deque<T>`, `Decimal`) crosses as
itself, whether in a struct or a variant, as RFC-0022's third tier
already had it.

The derive reads no attribute of another derive. A struct or enum may
carry `serde` derives alongside, and what `serde` renames or tags is the
wire's business; the language sees the Rust names.

## Rationale

A provider's wire schema is a set of Rust structs and enums under `serde`
— content blocks tagged by `type`, an untagged text-or-blocks union, an
`other` variant for what the schema does not name. Every value an extern
fn hands the script has been parsed into one of those before it crosses,
so the script's type is the Rust type; what remained was for the Rust
enum to have a language type at all. Nothing is parsed at run time by a
language type: parsing is the extern fn's job, and the derive projects
the result.

## Not built

- No tuple variant with two or more fields: the language's variant has one
  payload, and a tuple of them is not yet a crossing.
- No unit variant built by a script: `Shape::Dot` is matched, not
  constructed; a variant expression takes a payload.
- No generic enum, as with objects.

## Consequences

- `acvus-extern`: `Variant<V>`, `take_payload`, `materialize_payload`.
- `acvus-extern-macro`: `#[derive(TyArg)]` accepts an enum; the struct
  path and the struct-variant path share `ObjectShape`.
- `acvus-interpreter`: `VariantValue` is `Variant<Value>`.
- `acvus-ext`: `Decimal`, the extension type over `rust_decimal::Decimal`,
  `serde`-transparent so a wire struct's field is the one type on both
  sides; `std::decimal(text)`, `to_string`, `decimal_to_float`, `core::eq`
  and `core::clone` instances.
