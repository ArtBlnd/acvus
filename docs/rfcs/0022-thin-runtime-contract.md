# RFC-0022: The runtime contract is erase, materialize, reference, and call

Status: Accepted
Date: 2026-09-15
Supersedes: RFC-0010, RFC-0016

## Ruling

A runtime is anything that implements `Runtime`:

    trait Runtime: Send + Sync + 'static {
        type Value: Send + Sync + 'static;
        type Error: From<ExternError> + Send + Sync + 'static;
        type CallFuture<'a>: Future<Output = Result<Self::Value, Self::Error>> + Send + 'a
        where Self: 'a;

        unsafe fn materialize<T: Send + Sync + 'static>(&self, v: Self::Value) -> T;
        unsafe fn erase<T: Send + Sync + 'static>(&self, t: T) -> Self::Value;
        unsafe fn deref<'a, T: Send + Sync + 'static>(&self, r: &'a Self::Value) -> &'a T;
        unsafe fn deref_mut<'a, T: Send + Sync + 'static>(&self, r: &'a Self::Value) -> &'a mut T;
        unsafe fn reference(&self, target: &Self::Value) -> Self::Value;

        fn call_0<'a>(&'a self, f: &'a Self::Value, _: CallToken) -> Self::CallFuture<'a>;
        fn call_1<'a>(&'a self, f: &'a Self::Value, a: Self::Value, _: CallToken) -> Self::CallFuture<'a>;
        fn call_n<'a>(&'a self, f: &'a Self::Value, args: Vec<Self::Value>, _: CallToken) -> Self::CallFuture<'a>;
    }

A value is opaque to the contract: it has no shape methods, no
constructors, no equality, and no copy. `erase` and `materialize` are the
whole crossing, by Rust type; with `T = Self::Value` they are the
identity. A type variable of an ExternFn is `Self::Value` at runtime and
a body never opens it. `deref` and `deref_mut` read the storage a
reference value names; `reference` makes such a value. `call_0/1/n` run a
closure value on owned arguments and are reachable only through
`Fn0..Fn3::call`, whose token is theirs to mint.

A container of a type variable — `List<T>`, `Arr<T, N>` — crosses whole
as the Rust type it is at the call's resolved `T`: a `Monomorphize`
instance per member type, and the runtime's value as the instance of last
resort. Nothing unifies layouts.

`TypesOnly` is the runtime with no values, for checking declarations where
nothing will run.

## Rationale

RFC-0010 gave the runtime a constructor and an opener per value shape and
a `Closure` type; RFC-0016 proposed views over the runtime's value to
make containers cross for free. Both put knowledge of shapes on the
boundary. Real monomorphization removed the reason for views: a
`List<String>` is a `Vec<String>` on both sides of the crossing, so the
crossing is a move of one Rust value and the contract needs only the pair
that moves it. References (RFC-0018) added the only other thing a body can
do with a value it does not own: read the storage it names.

The `unsafe` on the crossing is honest: a type mismatch is a compiler bug,
not a runtime condition, so it panics rather than returning `Result`.

## Not built

- No value taxonomy on the contract: no `into_string`, `Str`, `Array`,
  `Object` associated types; no `FromValue`/`IntoValue`.
- No equality on the contract: comparing is a shared signature
  (RFC-0019).
- No copy on the contract: a runtime copies a word and nothing else
  (RFC-0018).
- No `Pointee`-based unsized crossing until `core::ptr::Pointee` is
  stable.

## Consequences

- The interpreter's `Value` is `Empty | Undef | Small(u64) | Large(ptr)`,
  the vtable in the allocation header carries drop and debug only, and a
  reference is `Small(address of the target)`.
- The generated glue materializes value arguments, derefs reference
  arguments, erases the return, and never copies.
