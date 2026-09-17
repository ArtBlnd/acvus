# RFC-0028: A container is read through per-type functions; a reference is one carrier

Status: Accepted
Date: 2026-09-15, rewritten 2026-09-17
Extends: RFC-0018, RFC-0027, RFC-0043

## Ruling

Reading a container is a set of plain functions, one per container
namespace, under the bare names `len`, `is_empty`, `get`, `get_mut`,
`first`, and `last`:

```
vec::len<T>(c: &Vec<T>) -> Int              array::len<T, N>(c: &Array<T, N>) -> Int
vec::get<T>(c: &Vec<T>, i: Int) -> &T       deque::len<T>(c: &Deque<T>) -> Int
vec::get_mut<T>(c: &mut Vec<T>, i: Int) -> &mut T
vec::first<T>(c: &Vec<T>) -> Option<&T>     string::len(s: &String) -> Int
vec::last<T>(c: &Vec<T>) -> Option<&T>      string::is_empty(s: &String) -> Bool
                                            string::contains(s: &String, pat: String) -> Bool
```

and the same six for `array` and `deque`. A bare name is the set of these
signatures and is settled by the call's evidence (RFC-0043); the qualified
name picks one. `string::contains` and `string::find` share their bare
names with `iter::contains` and `iter::find`.

A container is read through a reference to it, and an element read out of
a borrowed container is a reference into it: the call's result holds the
container's loan (RFC-0018), so the container is neither moved nor
changed while the element is in use. `get` at an index outside the
container is an error at the call; `first` and `last` of an empty container
are `None`. No function reads an element out by value: a value of the
element's type is `clone(get(&c, i))`, and exists for the types that have
an instance of `core::clone`.

In an extern declaration, `Ref<T, Rt>` and `RefMut<T, Rt>` are the acvus
types `&T` and `&mut T` wherever they stand — a parameter, a return, a type
argument — and each is the reference value itself: a body reads through
it with `get` / `get_mut` and returns it as it is. A Rust parameter `&T` /
`&mut T` still declares the same acvus type and is read at entry. A return
of `Ref`, `RefMut`, or `Option` of either is passed to the runtime as the
reference it carries.

## Rationale

The first version of this RFC declared `container::{len, get, get_mut,
first, last}` as shared signatures with an instance per container. A
shared signature is one generic handler per instance (`add_instance`), so
a per-element `Monomorphize` function such as `contains` could not be an
instance, and a signature's name was one function to the resolver, so
`string::len` and the container `len` could not both exist. RFC-0043 made
a bare name a set of signatures; with it, a plain function per namespace
gives every container the same names, admits `Monomorphize` members, and
leaves `core::` — `clone`, `eq`, `hash`, `to_string`, `to_int` — as the
shared-signature mechanism it was built for: one operation every type
answers, that a generic function asks for by name.

`deque_get` once took the deque by value because an extern function had
never returned a reference, and a body cannot copy an erased element; one
element cost the whole container. The compiler ties a reference-typed
result to the reference arguments it was built from, so returning the
reference is the exact answer.

`Ref<T>` was a phantom that named the type, and `Lent<T, Rt>` was the same
type with the value attached; every use of the phantom stood where a
runtime was in scope, so one name carrying the value serves both.

## Not built

- No `get` by value, no `pop`/`remove` here: a change to a container is
  its own set.
- No slicing and no negative index.
- No `string::get`: a string is a value (RFC-0026), and a `char` has no
  acvus type to be referenced as; a character is read out by value with
  `string::char_at`.
- No `vec::contains`, `array::contains`, or `deque::contains`. A
  per-element comparison is a `Monomorphize` member over the element type,
  and a member's glue crosses every parameter naming the member through
  `CrossSpecialized`; no form of a container of `Erased<Rt, T>` has one
  that reads the storage — `Vec` and `Deque` cross whole, so the payload's
  `TypeId` is the container of values; `Arr` crosses per element, which an
  erased element cannot; `Ref<C, Rt>` has none. `Iter` has, as an extension
  type stored as its payload, so `iter::contains` exists and a container
  is searched as `into_iter(xs) | contains(x)`, which the checker settles
  through the declared cast (RFC-0043). A crossing for a container of an
  erased element is an `acvus-extern` change and is not made here.

## Consequences

- `acvus-ext`: an `Iter` is one of two pipelines, and which one is settled
  when the pipeline is built (RFC-0044). `Iter` holds `Stages { Sync, Async
  }`; a source is `Sync`, an adaptor is `Sync` when its source is and its
  closure answers `Fn1::is_sync`, and an `Async` adaptor lifts a `Sync`
  source once rather than per element. A consumer reads the variant once
  and runs one of two loops. A stage yields `Option<Rt::Value>` and no
  `Result`: `None` is the end of the source and nothing else, because a
  stage whose closure failed panicked and never returned (RFC-0038).
- `acvus-extern`: `Ref<T, Rt>` and `RefMut<T, Rt>` carry `Rt::Value`;
  `Lent` is gone. The macro treats both as carriers in every position and
  unwraps `Option` of a carrier on return.
- `acvus-ext`: the registries `vec`, `array`, `deque`, `string` each
  declare their reading functions; the `container` module and its shared
  signatures are gone; `std::len`, `deque_len`, `deque_get`, `len_str`,
  `contains_str`, and the iterator's `first` are gone.
- Scripts: `xs | len` becomes `len(&xs)`; a length of a pipeline's result
  binds the result first.
