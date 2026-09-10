# RFC-0009: ExternFn declaration from a Rust function

Status: Accepted
Date: 2026-09-10
Supersedes: none

## Ruling

An ExternFn is declared once, as a Rust function. Its acvus type and its
runtime handler are both derived from that one signature; nothing about the
function is stated twice.

    #[extern_fn(effect = pure)]
    fn len_str(_: &Interner, s: String) -> i64

The mapping from the Rust signature to the acvus type is fixed:

- The first parameter is the interner: `&Interner` for a synchronous
  function, an owned `Interner` for an `async` one. It is not part of the
  acvus type.
- Every other parameter is an acvus parameter. Its Rust type names the
  acvus type through the `TyArg` trait, and its runtime conversion through
  `FromValue`.
- The return type names the acvus return type the same way, through
  `IntoValue`. A return type of `Result<R, RuntimeError>` means `R`.
- The acvus name is the Rust identifier unless `name = "..."` overrides it.
- The effect is `effect = pure | idempotent | <effect parameter>`; an
  undeclared effect is Opaque (RFC-0014).

Type-level variables are generic parameters of the Rust function, and each
kind has one bound that names it:

- `T: TyVar` is an acvus type variable.
- `E: EffectVar` is an effect variable. It stands in effect positions of
  other types and in `effect = E`.
- `N: LenVar` is an array-length variable. It stands only in `Arr<T, N>`.

A generic parameter with any other bound is an error. `TyVar` is the
runtime conversion pair, so a body written against it can move a value
across the boundary; at runtime every type variable is `Value` and every
effect or length variable is `()`. A Rust function is therefore compiled
once, and the acvus type system, not Rust, instantiates it.

Positions that acvus types have and Rust types do not are spelled by host
types: `Arr<T, N>` for an array of unknown length, `Fn0<R, E>`,
`Fn1<A, R, E>`, `Fn2<A, B, R, E>` for function parameters, and `Pure`,
`Idempotent`, `Opaque` for known effects in an effect position. A Rust
array `[T; 3]` is `Array<T, 3>`; a Rust tuple, `Option`, and the scalar
types are their acvus counterparts.

An extension type is declared once, as a Rust struct:

    #[derive(ExternType)]
    struct Regex(regex::Regex);

The struct's first field is the runtime payload, stored as it is; every
other field is `PhantomData`. The payload type therefore never mentions a
generic parameter, so every instantiation of the type shares one payload
type. Its `TyVar` generic parameters are the type's type parameters, in
order; its `EffectVar` generic parameters are its effect parameters. The
derive yields the acvus type, the declaration for the type registry, and
the two runtime conversions. The acvus name is the struct identifier unless
`name = "..."` overrides it. A type with an identity parameter refuses a payload that is
still shared when a handler takes it; any other type clones it.

A structural object is declared the same way:

    #[derive(TyArg)]
    struct Config { endpoint: String, model: String }

Each field is an object field of the field's acvus type.

A cast is an ExternFn of one parameter marked `#[extern_cast]`. Its cast rule
is its own signature: from the parameter type to the return type. A cast is
pure; anything else is an error.

A registry lists what it contributes and nothing else:

    extern_registry! {
        types: [List<_>, Iter<_, _>],
        fns: [len, reverse, list],
    }

Registering it adds the types and the cast rules to the type registry and
the functions to the compilation graph in one step. A type is registered
by the registry that declares it, never by a caller.

## Rationale

The previous declaration form stated every function three times: the
handler, its acvus type built by hand from type terms, and the registration
that joined them. The three could disagree, and the disagreement was only
found by a script at runtime. One signature cannot disagree with itself.

Type variables must be generic parameters rather than a special runtime
type, so that the Rust function reads as the polymorphic function it is.
They cannot be Rust-instantiated per script, because the script is compiled
after the Rust function; so the runtime instantiation is fixed at `Value`,
and the bound names the kind of variable rather than any runtime capability.

Length polymorphism is spelled by an explicit host type rather than a Rust
const generic, because a const generic is fixed when the Rust function is
compiled and an acvus length is fixed when the script is. The spelling is
rare by design: a function over a sequence of unknown length takes the
extension's dynamic-length type, and an array literal reaches it through
the registered cast.

The registry names its types with `_` in every argument position because
the type's declaration is what carries the arity; the registry states only
that the type belongs to it.

## Not built

- No `Vec<T>` mapping to a host-known list type. The dynamic-length
  sequence is an extension type like any other; the host knows no
  extension type by name.
- No runtime instantiation other than `Value`; a monomorphizing bound over
  a set of concrete types is not part of this ruling.
- No declaration of context reads or writes (RFC-0014).
- No inference-time handler returning a system-checked value; that mechanism
  was removed with the effect rewrite and is not carried here.

## Consequences

- A Rust type used in an ExternFn signature implements `TyArg` and the
  runtime conversion for its direction; a type that implements only one of
  them is rejected when the function is declared.
- `TyVar` is exactly the pair of runtime conversions, so `Value`, the
  compile-time stand-in, and every convertible type implement it.
  `EffectVar` and `LenVar` are implemented by every effect or length
  argument and by `()`. Nothing else implements them.
- A runtime extension value carries its acvus type name as a static string;
  identity needs no interner, so the conversions take none.
- A registry's `register` takes the type registry as well as the interner
  and returns functions and executables.

## Open questions

- Whether a function-typed parameter should carry the callback's own
  parameter names, or positional names are enough. Positional today.
