# RFC-0028: A container is read through shared signatures; a reference is one carrier

Status: Accepted
Date: 2026-09-15
Extends: RFC-0018, RFC-0027

## Ruling

Reading a container is a set of shared signatures, `container::{len, get,
get_mut, first, last}`, and every container type declares its instances:

```
container::len<C>(c: &C) -> Int
container::get<C, T>(c: &C, i: Int) -> &T
container::get_mut<C, T>(c: &mut C, i: Int) -> &mut T
container::first<C, T>(c: &C) -> Option<&T>
container::last<C, T>(c: &C) -> Option<&T>
```

A container is read through a reference to it, and an element read out of
a borrowed container is a reference into it: the call's result holds the
container's loan (RFC-0018), so the container is neither moved nor
changed while the element is in use. `get` at an index outside the
container is an error at the call; `first` and `last` of an empty container
are `None`. No signature reads an element out by value: a value of the
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

`deque_get` took the deque by value because an extern function had never
returned a reference, and a body cannot copy an erased element; one element
cost the whole container. The compiler already ties a reference-typed
result to the reference arguments it was built from, so returning the
reference is the exact answer, and `as_iter` was already doing it one
element at a time.

`Ref<T>` was a phantom that named the type, and `Lent<T, Rt>` was the same
type with the value attached; every use of the phantom stood where a
runtime was in scope, so one name carrying the value serves both.

`len`, `get`, `first`, and `last` are the same fact once per container;
RFC-0027 lets each container declare them under one name, as it did for
`into_iter` and `as_iter`. The iterator's `first` and `last` are dropped:
no script used them, and a name is one signature.

## Not built

- No `get` by value, no `pop`/`remove` here: a change to a container is
  its own signature set.
- No slicing and no negative index.
- No instance for `String`: a string is a value (RFC-0026), not a
  container of characters.

## Consequences

- `acvus-extern`: `Ref<T, Rt>` and `RefMut<T, Rt>` carry `Rt::Value`;
  `Lent` is gone. The macro treats both as carriers in every position and
  unwraps `Option` of a carrier on return.
- `acvus-ext`: a `container` module declares the five signatures and the
  `List` and `Array` instances; `deque` declares the `Deque` instances.
  `std::len`, `deque_len`, `deque_get`, and the iterator's `first`/`last`
  are gone.
- Scripts: `xs | len` becomes `len(&xs)`; a length of a pipeline's result
  binds the result first.
