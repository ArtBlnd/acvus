# Extern boundary: the note

The single basis of the extern rework. Every step derives from it, and structure derives from it, never from instances; where a step does not, the work stops and the question goes to its owner. It is fixed until its owner changes it. The compiler's meaning is `docs/rfc_compiler`.

## Status (2026-10-04)

| Topic | Decision | State | Section |
|---|---|---|---|
| Honest `unsafe` | A function that consumes a `Value` is an `unsafe fn` stating its premises; one that only makes a `Value` is safe. A safe runtime writes `unsafe` in signatures, and performs an unsafe operation at two places only: an extern fn's dispatch and an enum constructor's call. | decided, implemented | Invariants and constraints; Runtime and registry |
| Enum constructors | `unsafe fn` pointers, which the runtime calls for the tag. | decided, implemented | Kinds |
| Derive output and author lints | Generated under `allow(unsafe_code)`: an author's `deny` stands, a `forbid` refuses it (E0453). | observed | Runtime and registry |
| Layout of views | Instantiations differing only in a marker inside `PhantomData` share a layout within one compilation. This is the one exception: it rests on rustc's algorithm, the Rust project declines to guarantee it, and no check exercises it. The views' parts differ only in `Uniform`, each exactly `R::Value`, so their padding and validity agree byte for byte. | decided, recorded | Type variables; Forbidden |
| Values | A `Value`'s layout is its runtime's, and the extern crate relies on none of it. A typed read knows a value's form from its type and runtime; the operations that take no type read it from the `Value` inside the runtime. The acvus scalars are the closed set a runtime may hold in place. The runtimes built here use sixteen bytes on every target, two 64-bit lanes. | decided | Values and references |
| References | A `Ref` is made once by `as_ref`, from the `Value` its place holds with an owning one turned borrowed, and is copied plainly after; a `Mut` is the place's address. Runs and text are held as their data pointer and length. One value never crosses as a Rust reference: `as_ref` and `as_mut` make one. Native data is not lent as one value. | decided; code not yet changed | Values and references |
| Fat pointers | Taken apart by `as_ptr` and `len` and rebuilt through std, so no fat-pointer layout is relied on. | decided; code not yet changed | Values and references; Forbidden |
| `Ref` and `Mut` as types | Types of their own beside `Uniform`, without a drop, their variance set by `fn` markers. | decided | Values and references |
| Crossing a Rust reference | One load per reference, at a place the signature fixes: not a dynamic conversion. | decided | Values and references |
| Reference held uniformly | A reference that is a type variable's value is held as `Uniform<R, Shared<'a, X>>` or `Uniform<R, Exclusive<'a, X>>`, its marker naming its lifetime, so a container of references seen concretely is one instantiation of `Uniform` apart. | decided, with an experiment; code not yet changed | A reference held uniformly |
| Closure parameters | An author writes Rust's own closure types, `impl Fn(&T) -> bool` and the like; the macro declares the function type and flows from them, and the entry hands the body a closure calling the acvus function value. A `Fn` bound is admitted: the excluded HRTB trick is one played in acvus for an extern fn to accept. | candidate, from probes | Closure parameters |
| Kept functions | A function kept past the call crosses as a function value, `Uniform<R, Pred<'f, T>>`, its type named once by `#[fn_type]` and viewed as Rust's own closure for a scope. No runtime primitive added, no `'static` runtime handle. | decided by the owner; names, an alias's effect and the `FnMut`/`FnOnce` views open | A function kept past the call |
| Async calls | Work that outlives the future of its call borrows only what it owns: a scope takes the callable and the arguments by value, lends them, and gives them back with the result; the runtime owns spawning and keeps what a scope holds until the call's frame ends. No second owner: `Runtime::share` withdrawn. | decided by the owner; names and the contract's spawn open; code not yet changed | Async calls |
| `'static` in a signature | Refused everywhere, as a bound or as a lifetime. | decided | Invariants and constraints |
| Structural types | An object or an enum a declaration writes is closed by the transparency rule, so it is held whole as its Rust type and lent whole, as `Option` and `Result` are, with no part projected; one acvus type is one Rust type; a field can be lent. | decided by the owner; code not yet changed | Primitives; Values and references; Kinds |
| Collecting a signature | Types are collected through a trait each type implements. Lifetimes are proposed by the macro from syntax and confirmed by a generated check, which asks rustc, through trait matching, whether each written type's lifetime sequence is its resolved type's; an alias that reorders or a lifetime hidden from the syntax is a compile error. | candidate, from probes | Collecting a signature |
| Open questions | Listed under Open. | open | Open |

## Aim

- Every ordinary Rust function signature is expressible at the extern boundary, strictly more than before. Excluded from signatures are type-level tricks, such as associated types, GATs, HKTs and upcasting; an HRTB trick played in acvus for an extern fn to accept; and lazy monomorphization. A `Fn` bound on a closure parameter is no such trick: an acvus closure is an ordinary lambda, whose body acvus checks and whose handling of a reference its flows state, and the higher-ranked bound says only what such a closure is (Closure parameters).
- Generality is the intersection of Rust and acvus. What acvus defines and Rust expresses by what Rust guarantees is a general case, met by one rule over the whole concept: `Option`, `Result`, tuples of every arity Rust's means reach, objects and enums as Rust structs and enums, lent whole, primitives, `String`, `str`, slices, references, functions and passing modes. A type only Rust has, such as `Vec`, is no part of acvus: an extern-defined type is a Rust type its author defines and marks `#[derive(Opaque)]`, so that what is opaque is written where the type is, and a type the author did not define enters wrapped in one, as `#[derive(Opaque)] struct Stack<T>(Vec<T>)`. What Rust does not guarantee is not guaranteed.
- A signature that cannot be expressed is a defect of the structure, and the structure is revisited.

## Invariants and constraints

1. A round trip holds at every position.
2. Lifetimes are exact.
3. A value that crossed is kept nowhere its declaration does not show.
4. The boundary has no dynamic check and no dynamic conversion.

These hold by the primitives below, with nothing added on top. The standard is a safe API, and `unsafe` is welcome where it is stated honestly and kept local: an explicit `unsafe` can be safer than a safe-looking API. A function that consumes a `Value` rests on premises the `Value` does not carry, that it was made as a type, that it is live, and that its holder owns it or holds it exclusively; every such function is an `unsafe fn` stating them, and a function that only makes a `Value` is safe.

Boundary and author code have no branch on a type's form or kind, no forced unwrap, no cast, and no assert or type check outside the runtime. Only the runtime contract, `Uniform`'s boxing included, may require `'static`; a signature writes `'static` nowhere, neither as a bound nor as a lifetime, and the macro refuses every one. Inputs are values or views; outputs are owned values or references (Values and references). Where attaching something to a structure or to its parts means different things, the difference is expressed without ambiguity. The registry is the extern side's; the compiler only receives declarations, and every declaration form is writable and written correctly.

## Primitives

| Primitive | Meaning | Guarded by |
|---|---|---|
| `Value` | The runtime's value: `Copy`, laid out as its runtime chooses. It does not carry its type: its bits tell its form alone, a scalar held in place, a reference, or a value held out of line, and only the runtime's operations that take no type read them (Values and references). Any code may hold one; only an `unsafe fn` consumes one. | `unsafe` at every consumer |
| `Uniform<R, X>` | A value the runtime holds, in place for a scalar and out of line otherwise, its type kept by the marker `X`. Made by `Uniform::new(rt, value)`, which requires `X: 'static`, or by `Crossing::take`, which is `unsafe`; a reference is held under a reference marker, made from a `Ref` or a `Mut` (A reference held uniformly). So a `Uniform` holds a value made as `X`, and its views and its drop are safe. Projected to `&` or `&mut` only through the runtime, never taken out; a `Copy` value may be got. Moved, popped and dropped whole. Invariant in `X` by `PhantomData<fn(X) -> X>` (types §4). Its layout is `R::Value`'s for every `X`: it carries no niche of `X`, which would make `Option<Uniform<R, X>>` differ from one `X` to another. Used where a concrete type is needed: `Stack<Uniform<R, String>>` is an unmarked `Stack<String>` seen uniformly. | marker, privacy, `'static` |
| `Ref<'a, R, X>`, `Mut<'a, R, X>` | The shared and exclusive references to one value. A `Ref` is made by `as_ref` from the `Value` its place holds, an owning one turned borrowed, and is `Copy`; a `Mut` is the address of that place, made by `as_mut`, is not `Copy`, and reborrows; a store or a replace through a `Mut` changes the `Value` in the place. Neither has a drop. A reference to a reference is that reference (types §4). Both are invariant in `X` and covariant in `'a` by `PhantomData<(fn(X) -> X, fn() -> &'a ())>`, as `&'a X` and `&'a mut X` are. Read only through the runtime, lending `&T` or `&mut T` in continuation-passing style alone. | lifetime, marker, borrow checker |
| `Shared<'a, X>`, `Exclusive<'a, X>` | The markers under which a `Uniform` holds a reference: `Uniform<R, Shared<'a, X>>` is a `&'a X` held uniformly, made from a `Ref`, and `Uniform<R, Exclusive<'a, X>>` a `&'a mut X`, made from a `Mut`. Markers only, never values; the lifetime they name is the loan's, written where the declaration shows it (A reference held uniformly). | lifetime, marker, coherence |
| `TypeVar`, `EffectVar`, `IdentityVar` | The one marker bound of a generic parameter, one per kind. Carried through other types, so Rust keeps each variable one type. `TypeVar` gives `as_ref` and `as_mut`, the only way generic code makes a reference to a variable's value that it can give out. | Rust's type checker |
| `InstanceOf` | An impl's witness apart from its value, given only by reference and so never kept: kept, it could meet another type of the same signature. | borrow checker |
| `Instance` | A value and its impl as one, of one generic function. Its mode, owned, shared or exclusive, is a marker the macro reads from the signature; the body receives `T`, `&T` or `&mut T`. Once received it may go anywhere, its flows tracked. | type, flows |
| `Crossing` | A type's crossing by value: `give` hands the runtime a value, and `take`, an `unsafe fn`, takes one back whole on the premise that it was given as the type and that the caller owns it. Every type of the intersection implements it, so an entry or a derive opens each argument or field by its own kind, never by its name. | Rust's type checker |
| `#[derive(Opaque)]` | An extern-defined type: a Rust type its author defines and marks, so that what is opaque is written where the type is; a type the author did not define enters wrapped in one. Its values are held by the Opaque kind with their type erased, and the derive states the type's own declaration. | Rust's type checker |
| `#[derive(Structural)]` | A Rust struct with named fields is an acvus object, and a Rust enum an acvus enum, one acvus type to one Rust type. By value, the derive gives each field by its own crossing and takes each back. A declaration's object or enum is closed by the transparency rule (types §1), so the whole is the type: it is held whole as its Rust type and lent whole, as `Option` and `Result` are, an exclusive view reaching the variant's place, and no part is projected; a field can be lent. Fields and variants take their places in their names' text order (types §2). A variant's payload is nothing for a unit variant, its field for one unnamed field, the tuple of its fields for several, and the object of its fields for named ones. | Rust's type checker |
| `#[fn_type]` | An alias naming a function type an author keeps, written as Rust's own closure type, `type Pred<'f, T> = dyn Fn(&T) -> bool + 'f`. The macro makes a marker, under which a `Uniform` holds the function value, and a trait that views it as Rust's own closure for a scope (A function kept past the call). | Rust's type checker, flows |
| A scope | Takes a callable and its arguments by value, lends them to an asynchronous call, and gives them back with its result, so work that outlives the caller's future borrows only what it owns (Async calls). | ownership |
| Flows | Rust lifetime parameters. rustc checks the body; the compiler tracks liveness in the program, so references may sit inside values. | rustc, compiler |
| Generic function, impl | A generic function enters the compiler as one; an impl enters with the function and the choice it implements. It pairs with `Instance` and `InstanceOf`. | compiler |
| `#`, demotion | A whole concrete type such as `Stack<String>` is a specialized opaque type, apart from the generic `Stack<T>` over uniform values. A value that passes a non-`#` function even once is unmarked from where it is made, as the compiler decides (types §8). | compiler |
| Optimization declarations | Effect, declared facts, ends, casts and conversions, conveyed as declared and trusted. | author |
| Runtime | Lowers the IR and runs it, implementing the compiler's primitives as functions so nothing else needs `unsafe`. Generic over backends and zero cost from outside. Its contract states only the minimal conditions for a safe runtime, with associated types; its safe kinds are implementable without `unsafe`. It may check internally: misuse panics in a safe runtime and is UB in an optimized one. A value it holds out of line does not move while a reference to it lives; the places that hold its `Value`s may. It owns spawning and outlives its tasks (Async calls). The compiler's correctness is its premise. | Rust |

## Type variables

A type variable's value is never concrete and never wrapped in `Uniform`; it lives in containers, and a reference to one is held as `Uniform<R, Shared<'a, T>>` (A reference held uniformly). It carries no `'static`: the macro sees every lifetime bound and refuses one. So nothing keeps it out of sight, and nothing wraps it. A value holding a variable or a loan crosses as its own type, where its declaration shows them.

It is monomorphized as `Uniform<R, Never>`, never as `Value`, where `Never` is one fixed uninhabited marker, apart from Rust's `!`, which stands for acvus's `!`. So one value is read under several views: generic code sees `Stack<Uniform<R, Never>>`, a concrete view `Stack<Uniform<R, String>>`, and a value goes from one to the other and back. The views differ only in `Uniform`'s marker, and three constraints keep them one layout:

- Associated types. `TypeVar` holds for every `Uniform<R, X>`, and has no associated type and no supertrait. So an impl that generic code reaches through its one bound covers every view alike, and coherence refuses a second impl for one view; an impl naming `Never` itself is one that generic code cannot reach.
- Storage. A type variable carries no `'static`, so generic code keeps no value of one anywhere. A `Uniform` kept anyway carries its release in its drop.
- Niche and layout. `Uniform<R, X>` is `R::Value` behind `repr(transparent)` for every `X`, its marker a zero-sized `PhantomData<fn(X) -> X>`. The views' fields differ only in a marker that adds no niche and no field, so the views agree in padding and validity byte for byte: once sizes and alignments are checked equal, the views differ in arrangement only if rustc lays out identical fields apart by a type's identity alone, which its algorithm has no input for.

The claim is this narrow one, and it is needed within one compilation alone. A program is compiled ahead of time and whole, so no layout crosses from one build to another, and every concern about layouts that separately built artifacts must agree on does not apply. Within one compilation one type has one layout in every crate, or the crates would not link together.

What the language guarantees is less than the claim. `Uniform<R, X>` has `R::Value`'s layout for every `X`, by `repr(transparent)` and `PhantomData` being a zero-sized, minimum-alignment type, and arrays and slices of types of one layout share one, by their stride. Two instantiations of a `repr(Rust)` type are two types, and nothing guarantees them one layout, even within one compilation: the Reference states no layout guarantee for `repr(Rust)` beyond soundness, and the Rust project declines one for instantiations (UCG #581, 2025-09: "Type layout currently makes no guarantee that different instantiations of a type get a similar layout in any sense"; UCG #604, open, on instantiations differing only in `PhantomData`). That `X<T>` and `X<U>` share a layout whenever `T` and `U` do is therefore not claimed, and rustc's `-Zrandomize-layout` breaks it, laying out `S<u64>` and `S<i64>` apart within one compilation.

That instantiations differing only in a marker inside `PhantomData` share a layout within one compilation is rustc's behaviour: it computes a `repr(Rust)` layout from the fields' layouts and the representation, to which the marker adds nothing. It is the one exception under Forbidden. No check exercises it. `-Zrandomize-layout` derives a type's seed from its fields' seeds, and `PhantomData` has no fields, so every view of one value receives one seed and one layout by construction; and Miri sees only the layout the current compiler computes. Where the extern crate rereads a structure in place, an `Option`'s or a `Result`'s payloads, a const check confirms the sizes and alignments agree; an opaque value's views have no such check, since the type it was made at is erased.

The argument assumes Rust admits no specialization, no default impls and no negative impls, any of which would let an impl single out one view. The owner's assessment is that none will arrive in a form that reaches this: negative impls are all but ruled out, and specialization and default impls remain unfinished and, under the next-generation trait solver, held back by lifetime coercion. Should one arrive, this section is revisited.

## Values and references

A `Value`'s layout is its runtime's. The extern crate relies on none of it: `Uniform<R, X>` is `R::Value` behind `repr(transparent)`, a typed read reaches a value through its type's kind, a run or text crosses through `Stores` as its data pointer and length, and the operations that take no type, `release` and a type variable's `as_ref`, are the runtime's own. A runtime is correct at any size that holds what its kinds put in a `Value`; its size decides its speed.

What a `Value` holds follows from its acvus type. The values a runtime may hold in place form a closed set, the acvus scalars: `i8` to `i64`, `u8` to `u64`, `f64`, `bool`, `char` and `()`. A runtime holds in place those its layout has room for, a reference as its pointer where it fits, and every other value out of line, in storage that does not move while a reference to it lives.

Code that knows a value's type never reads its `Value` to learn its form. Whether a value is held in place is a fact of its type and its runtime, both of which every read site knows at monomorphization: a read chooses in place or out of line by a constant, and inlining removes the choice. That this costs nothing is verified, not assumed. The operations that take no type read the form from the `Value` itself, inside the runtime. A generic drop is one of them: a type variable's value drops as `Uniform<R, Never>`, whose type does not say whether it holds a scalar, and a `Drop` impl cannot carry a bound its type does not (E0367, observed with rustc 1.97.1), so its `release` reads the `Value`. A runtime's layout therefore leaves every form it holds recoverable from the bits, by what Rust guarantees and never by a layout Rust leaves open.

A `Ref` is made once, by `as_ref`, from the `Value` its place holds: a `Value` that owns storage is turned borrowed, so no `release` ends an owner through a `Ref`, and the `Ref` is copied plainly from then on. This matters where a `Ref` is kept as a type variable's value: a generic container of `T` standing for `&X` drops its elements as `Uniform<R, Never>`, and their `release` must find nothing to end. Generic code's `as_ref` does not know its value's form, so it is an operation that takes no type: it reads the `Value`, turns an owning one borrowed and leaves any other as it is, so a `Ref` of a `Ref` is that `Ref` (types §4).

A run or text is taken apart by `as_ptr` and `len` and rebuilt by `slice::from_raw_parts`, a `str` by `from_utf8_unchecked` over that, so no fat-pointer layout is relied on. Nothing lends a reference to a stored pointer itself, since a reference to a reference is that reference (types §4).

Inputs are values or views; outputs are owned values or references, a reference to one value being a `Ref` or a `Mut` and a reference to a run or to text a fat pointer. A view of a structure is a Rust reference to the structure as the runtime holds it, its parts uniform, and an exclusive view reaches the variant's place, so a store through it writes the variant and the payload together (ownership §5); `Option<&T>` reaches the payload alone. A tuple is the exception: Rust guarantees no tuple layout and a runtime holds a tuple of any arity as a run of values, so a reference to a tuple is the tuple of its parts' references. An object or an enum is held whole as its Rust type and lent whole (`#[derive(Structural)]`).

A reference's form follows from its type, `&` or `&mut`, and its referent's acvus type tells how the `Value` it reaches reads; so no read branches on a reference's form.

| Reference | Form | Made by | Read | Written |
|---|---|---|---|---|
| `&X` to one value: `Ref<'a, R, X>` | the `Value` the place holds, an owning one turned borrowed by `as_ref`, copied plainly after | `as_ref` on the place; a Rust `&'a T` or `&'a Uniform<R, X>` given out, by one load | by the runtime, as `X`'s acvus type reads | never |
| `&mut X` to one value: `Mut<'a, R, X>` | the address of the place that holds the `Value`: a frame's register, a container's element, a runtime's slot | `as_mut` on the place; a Rust `&'a mut T` or `&'a mut Uniform<R, X>` is one already | by the runtime, through the place | a store or a replace changes the `Value` in the place; `Mut::replace(new) -> T` gives the old one back |
| `&[X]`, a slice | a run of owned handles, `[Uniform<R, X>]`, held as its data pointer and length | the array's view or the container's `as_slice`, once per loan | `Op::Index` addresses an element directly; an element's reference is a `Ref` or `Mut` of the element's place | through an element's `Mut` |
| `&str` | bytes, held as their data pointer and length | `as_str` | directly | never |

Rules:

- A reference to one value never crosses as a Rust reference. `Crossing` is implemented for `&str` and `&[Y]` alone, so a Rust reference to one value given out is a missing impl, refused by type checking, whose diagnostic points at `as_ref` and `as_mut`; `TypeVar` and `Uniform` give these two, and they are the only way generic code makes a reference it can give out. `Ref` and `Mut` lend `&T` and `&mut T` in continuation-passing style alone, so no Rust reference leaves them.
- A `Mut` names the place, so nothing is written back after a call: a generic body that replaces a handle replaces it in the place itself, and the `T` it takes out is again a `Value`.
- Native data is not lent as one value: a place a reference names holds a `Value`, of a `Uniform` or of a type variable. A value an extern lends is held as a `Uniform`. A `#` value holds no run of handles, so indexing or iterating one by reference demotes it (types §8).
- A reference to a reference is that reference (types §4): a `Ref` of a `Ref` is the inner `Ref`, and a `Mut` of a `Mut` reborrows it.

Premises and costs:

| What | Premise or cost |
|---|---|
| A `Ref` | the place does not change while it lives, by its shared loan (ownership §5), and the storage its `Value` names does not move |
| A `Mut` | the place does not move while it is lent: a frame's register by the interpreter, a container's element by Rust's borrow of the container, a runtime's slot by its block |
| A Rust reference given out | one load per crossing reference, at a place the signature fixes, and no more than making the reference cost; it is no dynamic conversion |
| Reading through a `Mut` | two steps for a value held out of line, the place then the storage; through a `Ref`, one |
| Indexing | one `as_slice` per loan, which the loan keeps from being written, moved or reallocated (ownership §5); then direct addressing, its bounds checks removable by length relations |

`Ref` and `Mut` are types of their own, not a mode parameter of `Uniform`. `Uniform` releases its value on drop, so it has a `Drop` impl, and one Rust type cannot be `Copy` and have one (E0184), nor have one for some of its instantiations alone (E0366). A `Ref` must be `Copy`, as a shared reference copies (ownership §1), and neither may have a drop: a type with a `Drop` impl holds its borrow until it drops (E0502, "borrow might be used here, when `r` is dropped"), where the same type without one ends its borrow at its last use, so a reference that dropped would hold every loan to the end of its scope. A mode that chooses a held field through an associated type compiles, but makes the reference invariant in its lifetime, which `&'a T` is not. `PhantomData<(fn(X) -> X, fn() -> &'a ())>` makes each invariant in `X` and covariant in `'a`. All were observed with rustc 1.97.1.

### The sixteen-byte layout

The runtimes built here lay a `Value` out in sixteen bytes on every target: two 64-bit lanes, `P` and `M`, which WebAssembly moves as one `v128`. A pointer or a length is zero-extended into its lane on a 32-bit target.

| Value | `P` | `M` | `release` |
|---|---|---|---|
| A scalar | 0 | its bits | nothing |
| A run or text: `&[X]`, `&mut [X]`, `&str` | its data pointer | its length | nothing |
| A `Mut` of one value | its place's address | the place marker | nothing |
| Any other value: `String`, an opaque value, a structure, a function | its storage's address | the storage marker | ends its owner |
| A `Ref` of a value held out of line | its storage's address | the borrowed marker | nothing |

A `Ref` of a scalar, a run or text takes its referent's row.

The lanes are told apart by what Rust guarantees. A data pointer, a place's address and a storage's address are never null, so `P` is 0 for a scalar alone. A run's total size is at most `isize::MAX` bytes and its elements are sixteen-byte handles or bytes, so a length is below `2^63` on every target, and every `M` from `2^63` up is free for the markers.

Sixteen bytes is the least power of two that holds every acvus scalar and every run's pointer and length in place, with room to tell them apart. Eight bytes is correct as well: a pointer carries its marker in the low bits its alignment leaves zero, and the 64-bit scalars and the runs, which leave no room, are held out of line, at an allocation for each `i64`, `u64` or `f64` made and each run or text lent. Thirty-two bytes is correct too, at twice the memory per handle. These costs are reasoned, not measured.

### A reference held uniformly

A reference can be a type variable's value. With `T` standing for `&X`, a uniform container's elements are references, and code that knows `X` sees the container with its element written concretely. Generic code sees the same value as `C<Uniform<R, Never>>`, so the concrete view must differ from that only in a marker inside `PhantomData`, the one layout exception (Type variables), or nothing admits it. `Ref` cannot be that element: `Ref` and `Uniform` are two structs (Values and references), so `C<Ref<'a, R, X>>` differs from `C<Uniform<R, Never>>` in more than a marker, and layout randomization lays the two apart.

So a reference held uniformly is a `Uniform` whose marker names the reference: `Uniform<R, Shared<'a, X>>` holds a `&'a X`, and `Uniform<R, Exclusive<'a, X>>` holds a `&'a mut X`. `Shared` and `Exclusive` are markers only, never values, each `PhantomData<(fn(X) -> X, fn() -> &'a ())>` inside. A marker without the lifetime would lose it: `fn first<'a>(v: &Vec<&'a X>) -> &'a X` could not be written, and an owned `Vec<&'a X>` would hold a loan its declaration does not show.

| Form | What it is | Made by | Gives |
|---|---|---|---|
| `Ref<'a, R, X>` | a shared reference standing alone: `Copy`, no drop, covariant in `'a` | `as_ref` | a view of `X`, through the runtime |
| `Uniform<R, Shared<'a, X>>` | a shared reference held uniformly: moved and dropped whole, its drop releasing nothing, invariant in `'a` | `Uniform::from_ref(Ref<'a, R, X>)` | `get() -> Ref<'a, R, X>` |
| `Mut<'a, R, X>` | an exclusive reference standing alone: not `Copy`, no drop, covariant in `'a` | `as_mut` | a view of `X`, through the runtime |
| `Uniform<R, Exclusive<'a, X>>` | an exclusive reference held uniformly, as the shared one is | `Uniform::from_mut(Mut<'a, R, X>)` | `as_mut(&mut self) -> Mut<'_, R, X>`, a reborrow |

What holds:

- Layout. `Uniform<R, Shared<'a, X>>` is the struct `Uniform<R, Never>` is, with another marker, so a container of held references falls under the one exception, as a container of `Uniform<R, String>` does. Lifetimes are erased before layout.
- Release. The `Value` a held reference holds is a `Ref`'s, carrying the borrowed marker, or a `Mut`'s, carrying the place marker, so its drop, and a type variable's drop of it, end no owner.
- Flows. `'a` is written in the type, so a declaration shows the loan, and the macro reads it as it reads any lifetime parameter.
- Impls per marker. Inherent impls on one marker's `Uniform` stand beside the value markers' once the value markers' impl carries its bound on the impl block, `impl<R, X: Lend<R>> Uniform<R, X>`; a bound on each method instead is a duplicate definition (E0592). No trait needs an impl per marker: `Crossing` and the drop treat every marker alike, the `Value` deciding the release. Generic code reaches no inherent impl of a marker, since it reaches a value only through `TypeVar`, which holds for every `Uniform` and so for the markers; an author's impl through that bound covers a held reference too, and a second impl for that view is refused (E0119). No other crate can implement the extern crate's traits for its markers (E0117).
- The `'static` gate. `Uniform::new` requires `X: 'static`, so neither a type variable's value nor a value holding a loan is wrapped (E0310). A reference is held uniformly only from a `Ref` or a `Mut`, its lifetime named in its marker, and that lifetime cannot be stretched: a held `Shared<'a, _>` gives no `Ref<'static, _>`, and is no `dyn Any`. So nothing a `Uniform` holds is out of a declaration's sight.
- Declaration. `Uniform<R, Shared<'a, X>>` stands for `&'a X` and `Uniform<R, Exclusive<'a, X>>` for `&'a mut X`, each held uniformly; a container written with one is in the uniform form, a `Uniform` at the root of every type argument.

What it costs:

- `Uniform` is invariant in its marker (types §4), so a held reference is invariant in `'a`: `&'x Vec<Uniform<R, Shared<'a, X>>>` does not shorten to `'x`, where `&'x Vec<Ref<'a, R, X>>` does. An entry opens every loan of a call at one `'call`, so every signature is met; a body that mixes lifetimes names each.
- A held reference keeps its borrow until it drops, as any type with a drop does, where a `Ref` ends its borrow at its last use. And `Uniform`'s drop has no `may_dangle`, so in a container of held references the referent outlives the container (E0597), where std's `Vec<&X>` lets it dangle at the drop. Both bind bodies only, never soundness.

A reference to a type variable's value is held as `Uniform<R, Shared<'a, T>>`. What it holds is a reference's `Value`, not `T`'s value, so nothing is wrapped twice; and `T` stays without `'static`, since the marker names `'a`, which cannot be stretched. This reads Type variables' "never wrapped in `Uniform`" as concerning a variable's value, not a reference to one, as its owner confirmed. The surface is the one the experiment exercised, `from_ref` and `get`, `from_mut` and `as_mut`, and grows only by agreement.

Experiment, 2026-10-03, in `docs/experiments/reference-markers`: a model runtime of sixteen-byte `Value`s with the markers of The sixteen-byte layout, and `Uniform`, `Ref`, `Mut`, `Shared`, `Exclusive` and `TypeVar` as stated here; its storage counts owners instead of freeing, so a wrong count is seen without undefined behaviour.

- Behaviour, with stable rustc 1.97.1 and under Miri on nightly 1.99.0 (2026-07-20), Stacked Borrows: five tests pass, and Miri reports no undefined behaviour, only warnings of the model's integer-to-pointer casts. Generic code that drops a container of held references, read as `Vec<Uniform<Never>>`, leaves the owner's count at 1, and the owner's own drop takes it to 0. The negative control, a `Ref` copying the owning `Value` bit for bit, takes the count to -1 in the same flow, before the owner drops. A `Ref`, its copy, the `Ref` a held reference gives back, and a type variable's `as_ref` of a held reference carry one `Value`. A scalar's `Ref` is its bits. A held exclusive reference writes through its place.
- Compilation, with stable rustc 1.97.1: eighteen probes, each as expected. A `Ref` ends its borrow at its last use, and a held reference holds it until it drops (E0502). `Copy` with a `Drop` impl (E0184); a `Drop` impl for one marker (E0366). A method-bound duplicate (E0592), while impl-block bounds compile. A view impl beside a `TypeVar` blanket (E0119); a foreign impl of a trait for a marker (E0117). `Ref` shortens and a held reference does not ("lifetime may not live long enough"). A body with its own lifetimes called at one `'call` compiles. A held reference gives no `Ref<'static, _>`, and is no `dyn Any`. std's `Vec<&i64>` lets the referent dangle at its drop, a container of held references does not (E0597). A generic `Uniform::new` (E0310). A generic body holds a reference to a type variable's value, and cannot give it `'static`.
- Layout, with nightly 1.99.0 under `-Zrandomize-layout` at seeds 1 to 32: the field offsets and size of an author-defined container, a struct of a `Vec`-like run, an `Option` and scalar fields, compared against the generic view. `Uniform<String>`, `Uniform<Shared<'static, i64>>` and `Uniform<Exclusive<'static, i64>>` matched it at all 32 seeds; `Ref<'static, i64>` differed at all 32 and `Mut<'static, i64>` at 31. Without randomization all matched. std was not rebuilt with the flag, so its own types were not randomized; this is consistent with the exception's mechanism and no proof of it.

## Functions

Function values cross both ways: acvus gives its closures to an extern, and an extern gives Rust functions to acvus (Kinds). An extern receives a function in one of two ways, lent for the call, as a closure parameter, or kept past it, as a function value, and writes either in one vocabulary, Rust's closure signatures, on a parameter or in an alias. A Rust function an extern gives to acvus is open (Open).

### Closure parameters (candidate)

An author writes a closure parameter as Rust writes one: `impl Fn(&T) -> bool`, `impl FnMut(&mut T)`, a result that borrows the call's argument by elision, `impl Fn(&T) -> Ref<'_, R, U>`, or arguments with lifetimes of their own, `impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, R, T>`. The macro reads the closure's signature as it is written and declares the parameter's function type and flows from it: the result above borrows argument 0 (extern-fn §4), and the closure's effect joins the call's, since the call runs it (extern-fn §2). The entry hands the body a Rust closure of its own, which calls the acvus function value through the runtime's Fn kind: an argument `&T` or `&mut T` crosses as a `Ref` or `Mut` by one load, as any Rust reference given out does (Values and references), and the result comes back as an owned value, a `Ref` or a `Mut`. A closure parameter is a parameter of its own kind, a function, beside the type, effect and identity variables; the macro takes it by its `Fn`, `FnMut` or `FnOnce` bound.

It adds no primitive: `Ref`, `Mut`, `TypeVar`, `Crossing` and the Fn kind suffice. Each argument's lifetime is exact, Rust's elision and binders stating it. And no projection sits under a binder, which is what failed in the per-call wrapper (Set aside).

A closure lent so is the call's alone: the entry's closure holds the call's borrow of the runtime, and one kept past the call is refused (E0310). A function kept past the call crosses instead as what it is in acvus, a function value, held uniformly.

Probes, 2026-10-03, with stable rustc 1.97.1, in `docs/experiments/closure-params`: the signatures above, generic over `T: TypeVar` alone, and an entry writing the closures it hands over, compile; a result that borrows an argument its signature does not name is refused (E0621); a closure lent for the call and boxed to be kept past it is refused (E0310).

### A function kept past the call

An author names a function type it keeps, once, where it writes it, as `#[derive(Opaque)]` names an opaque type there:

```rust
#[fn_type]
type Pred<'f, T> = dyn Fn(&T) -> bool + 'f;
```

The macro makes in the author's crate a marker, `Pred<'f, T>`, and a trait that views `Uniform<R, Pred<'f, T>>` as Rust's own closure for a scope, with the alias's signature: `fn as_fn<'s>(&'s self, rt: &'s R) -> impl Fn(&T) -> bool + 's`. The function crosses by value as a function value; an author keeps it as `Uniform<R, Pred<'f, T>>`, in a field of an opaque type as anywhere, and calls it through its view:

```rust
fn count<R, T: TypeVar<R>>(rt: &R, filter: &mut Filter<'_, T>, xs: &[T]) -> u64 {
    let pred = filter.pred.as_fn(rt);
    xs.iter().filter(|x| pred(x)).count() as u64
}
```

What holds:

- No runtime primitive is added, and the runtime contract gives no `'static` handle: the view borrows `rt` for its scope alone. The attribute is the one addition, of the kind `#[derive(Opaque)]` is.
- `'f`, the loans the function value holds, is named in its marker, as a reference marker names its lifetime, so a type that keeps the function declares that it holds them (extern-fn §4), and `Uniform::new`'s `'static` gate is untouched.
- Each argument's lifetime is exact, since the alias writes Rust's own signature.
- The views of a kept function differ only in its marker's type arguments, inside `PhantomData`, so a container of kept functions falls under the one exception.
- The view is a trait of the author's crate, since an inherent impl on `Uniform` there is refused (E0116). Its method returns `impl Fn(..)`, a return-position `impl Trait` in a trait; that stays inside generated code, and no author signature shows an associated type.

Probes, 2026-10-03, with stable rustc 1.97.1, in `docs/experiments/closure-params`: the expansion of `#[fn_type]`, written by hand, compiles, a function kept in an author's type and called later through its view, generic and at a concrete view, with arguments of their own lifetimes; its view's result borrows only the argument it names (E0621); and the view as an inherent method on `Uniform` is refused (E0116).

### Async calls

Work can outlive the future of the call that started it: an acvus function value an extern calls asynchronously, which the runtime may run as a task of its own, or any spawn. Such work must borrow nothing that future owns. Rust guarantees no destructor runs, `mem::forget` being safe, so a cancelled or forgotten future would leave the work holding a dangling borrow, and nothing may rest on a drop ending the work in time.

So the work borrows only what it owns. A scope takes the callable and the arguments by value, lends them to the call, and gives them back with its result:

```rust
let ((a1, a2), out) = f.scope(rt, (a1, a2), |callable, (r1, r2)| callable.call_async(rt, (r1, r2))).await;
```

The borrower carries its owner, so no borrow outlives its owner, whatever happens. A cancelled caller leaves the scope with the runtime, which releases what it holds when the call's frame ends; a forgotten one leaks it, which is safe. tokio-uring takes the same course for the same reason: an operation takes its buffer and gives it back with its result. In acvus the call takes its arguments by value and its result gives them back, so its flows are plain moves, no loan is extended, and what the work keeps shows in its declaration.

The runtime owns spawning. It runs the call, keeps what a scope holds and the storage the call borrows until the call's frame ends, and outlives its tasks. That guarantee stands where a thread pool would ask for `'static`, so a scope may hold a type variable's values and the loans its markers name. The obligation is the runtime's alone, and no author code reaches the runtime's tasks to forget one. No value needs a second owner: `Runtime::share` is withdrawn.

A function called asynchronously crosses as a function value, named by `#[fn_type]`. A closure lent for the call is the entry's synchronous closure, holding the call's borrow of the runtime, and cannot outlive a cancellation.

Within a run, a started call is taken on every path from it (effects §6), so a call is cancelled only by its run's end, a run that does not end (effects §4); a panic is death (effects §5).

## Collecting a signature (candidate)

A declaration needs each position's acvus type and, for its flows, which lifetime each position carries. Types are collected through a trait each Rust type implements, since collecting them from syntax would tell types apart by name. Lifetimes can be collected only from syntax, since Rust's trait selection does not see which lifetime a type carries. Two things stand between the syntax and the positions. The syntax shows the written type, not the resolved one, so an alias between them, `type Swap<'a, 'b> = Pair<'b, 'a>`, or a lifetime elided in a path would label positions silently wrong, and wrong flows are undefined behaviour. And positions do not follow the syntax's order: a `#[derive(Structural)] struct S<'a, 'b> { z: Ref<'a, ..>, a: Ref<'b, ..> }` lays its fields out in their names' text order (types §2), so its positions come as `'b`, `'a` while `S<'x, 'y>` names them `'x`, `'y`.

The macro proposes; rustc confirms. Each type states, as a type, the lifetimes it carries in the order of its own parameters: its own lifetime arguments first, then each type argument's, in order, as a list of lifetime markers. Its definer states it once, a derive for an author's type and the extern crate for its own. The macro reads the same sequence from a signature's syntax, in the order Rust's syntax fixes, naming each elided lifetime by Rust's elision rules and putting each type variable as a placeholder that carries none. It then generates a check, a function generic over the signature's lifetimes and over each closure parameter's binder, which asks of every written type whether its sequence from syntax and the sequence of the type rustc resolved are one type. The question is put through trait matching, `Same<A> for A`, which equates its two sides: their shapes must match, and borrowck then requires each pair of lifetimes to be one. An alias that reorders, or a lifetime the syntax does not show, is a compile error. rustc borrow-checks a function whether or not anything calls it, so the check costs compile time alone.

The query verifies and cannot compute: its answer is whether the crate compiles, and nothing branches on that. So the macro computes from syntax, where it can read names, and rustc checks that computation against the resolved types, where aliases are already seen through, with no access to the compiler's internals. The declaration then labels positions by consuming the confirmed sequence in the same order, each type's impl knowing which of its parameters stands at which acvus position, so a structural type whose fields its names' text orders apart from its parameters is labeled rightly. Flows then follow by name: a result position borrows from the argument positions that carry its lifetime, the outlives its types imply being the compiler's own (ownership §5).

What it adds: the trait, a type-level list of lifetime markers with its append, and the `Same` question. They are associated types and type-level computation, which stay in generated code and in the extern crate, and never appear in an author's signature. They are primitives to agree on.

Probes, 2026-10-04, with stable rustc 1.97.1, in `docs/experiments/lifetime-query`: a whole signature's queries compile, a slice, an opaque type over a held reference, a kept function, a type of two lifetimes and two type arguments, a structural type and a result among them; so do a closure parameter's own, its binder's lifetimes standing as the check's; and sixteen lifetimes through nested types compile in milliseconds. Refused: an alias reversing lifetimes, an alias reversing type arguments, and such an alias inside a closure's argument (each "lifetime may not live long enough"); a lifetime elided in a path, and a type variable left as itself rather than put as the placeholder (each E0277). The markers' variance does not carry the query: with covariant markers at one position and `'x: 'y` implied, where a subtyping check would admit one direction, both directions are refused, since trait matching equates.

Not yet built: the macro that generates the checks, and the declaration that consumes the confirmed sequence.

## Runtime and registry

Safety is decided per kind. Every kind's consuming functions are `unsafe fn`, since the contract's premises are ones an optimized runtime does not check. A safe kind is implementable with no unsafe operation: a safe runtime writes `unsafe` only in those signatures, checks the premises in their bodies, and panics on a misuse such as aliasing or a released value. An unsafe kind reads storage as a Rust type, in an implementation the extern crate provides once and Miri checks, and which a runtime may replace with its own; misuse within it is UB, and a value's type there is the compiler's premise.

A runtime performs an unsafe operation in two places alone, each asserting the compiler's premise about values it holds: where it calls an extern fn, `ExternFn::call`, with the call's arguments, and where it builds an enum value by the constructor for its tag. In the extern crate, storage is read as a Rust type in `repr` alone; elsewhere an `unsafe fn` states its premises and forwards them. This is a trade-off taken: `unsafe` appears outside one module, but each appearance states a premise where it is used, and none is a case treated apart. The derives' output implements `take` and the views as `unsafe fn` under an `allow(unsafe_code)`, so an author's crate may `deny(unsafe_code)` and still derive; one that `forbid`s it cannot, since `forbid` refuses the `allow` (E0453, observed with rustc 1.97.1).

A view is lent in continuation-passing style: the runtime calls a closure with it, so its lifetime is the closure's, and the contract carries no lifetime-generic associated type for it. Each kind lends its own views so, `Option` as `Option<&_>`, and each argument is opened by its own kind; the entry nests the body inside these closures and gives its result inside them. Its zero cost rests on the closures being inlined, which is to be verified, not assumed.

The unsafe kinds' implementations sit on `Stores`, which every runtime implements with no unsafe operation, its consuming functions `unsafe fn` as every kind's are: it holds the extern crate's `Boxed` behind a `Value`, lends it back shared or exclusive, and gives it back whole, and it holds the `Words` of a run or of text. `Boxed` owns one value of any type, `Send + Sync`, behind one pointer to a header and the value, the header holding an atomic count and the value's drop; it needs no `'static`, since it erases the type. `Words` holds a run's or text's data pointer and length. A `Boxed` is read back as a Rust type, and `Words` rebuilt into a reference through std, in the extern crate's `repr` alone, on the premise the `unsafe fn` around the read states, which is the compiler's.

`release` ends a value's owner, and no value has a second owner at the boundary: work that outlives the future of the call that started it takes what it uses by value, through a scope, and gives it back (Async calls), so the contract carries no `share`. Whether `Boxed` keeps its atomic count is open (Open).

A declaration mirrors the Rust types as written. `Uniform<R, X>` stands for `X`'s acvus type, held uniformly; `Uniform<R, Shared<'a, X>>` and `Uniform<R, Exclusive<'a, X>>` for `&'a X` and `&'a mut X`, held uniformly; and a `#[fn_type]` marker for the function type its alias writes. An extern-defined type is written in one of two forms: uniform, every type argument a `Uniform` or a type variable at its root, which is unmarked; or whole and native, with no `Uniform` and no variable anywhere in it, which is marked `#`. Any other form specializes at a type argument, which types §8 rejects, and is refused; a part a type always holds natively is a field of its own definition, not a type argument. So an impl or an overload written with `Uniform` is the one generic code reaches, and one written natively is its `#` counterpart, which the compiler picks between (types §8). A body that awaits declares `awaits` and an effect that suspends; the compiler is left as it is.

`Registry: Runtime` is a runtime that registers extern fns. An extern fn reaches it as a type implementing `ExternFn`, which states its own declaration and whose fixed parameter shape is an associated type: its `N` arguments are `Self::Args`, a `[Value; N]`, which `call` takes; the registry gives the declarations to the compiler. Because the runtime receives each extern fn as a type rather than as an erased function, it can build specialized operations around one: a natural closed loop with its `next` inlined differs greatly from one that calls through an erased function. The macros are what implement `ExternFn`.

## Kinds

The runtime contract gives one associated type per kind, each with a trait of its own functions; no function serves every kind. A kind groups the constructors of the compiler's `TyKind` that the boundary treats alike: how a value is made and given back, how a reference to it opens, and whether it is `Copy`. Each kind is safe or unsafe.

| Kind | `TyKind` | Safety |
|---|---|---|
| Scalar | `Int`, `Float`, `Char`, `Bool`, `Unit` | safe |
| String | `String` | safe |
| Tuple | `Tuple` | safe |
| Object | `Object` | safe |
| Option | `Option` | safe |
| Result | `Result` | safe |
| Enum | `Enum` | safe |
| Fn | `Fn`, `Closures` | safe |
| Opaque | `UserDefined` | unsafe |
| Ref | `Ref` to `Str` or `Slice`, and a `Mut` of one value, the address of its place | unsafe |

Opaque is unsafe because one value is seen as several Rust types, generic and uniform, so it is held untyped and read by layout. Ref is unsafe because an address is read back from a `Value`: a run's or text's data pointer and length, which `Stores` holds as `Words` and the extern crate rebuilds through std, or a `Mut`'s place; `Str` and `Slice` need no kind of their own. A `Ref` of one value needs no kind: it is the `Value` its place holds, an owning one turned borrowed, read by its referent's kind. An opaque value is `Send + Sync`, so that a task the runtime spawns may carry it.

`Never` has no value, `Order` and `Handle` are no program's, and `Array` is no declaration's. `Option` and `Result` are held in Rust's own shapes, so a view of one is a Rust reference to that shape, and its payload opens natively, as `Option<&_>` and `Result<&_, &_>`. Function values cross both ways: acvus gives its closures to an extern, and an extern gives Rust functions to acvus. Either is a Rust lambda, run on an acvus frame or as a Rust function.

Object and Enum hold a value whole as its Rust type and lend a view of the whole, as Option and Result lend theirs. How a runtime keeps such a value between uses is its own, the language stating no layout (undefined-behaviour §2): the owner expects scalar replacement to keep its parts in registers, and an assignment into the Rust type where it is lent or given, in place of a coercion; the Rust type may carry a `repr`. An enum value is a tag and one payload, `()` for a variant that carries none. The runtime picks a variant from a table of the derive's constructors by its tag, checking in a safe runtime, so that no code outside the runtime branches on a tag it cannot exhaust. A constructor is an `unsafe fn` taking the payload, since it takes it back as the variant's fields; calling it is the runtime's assertion that the payload is the one made for its tag.

## Lifetimes in the macro

A value that holds a loan is kept in a `Boxed` with its type, its lifetimes included, erased. An entry opens it again at one named lifetime of its own, `'call`, that every loan of the call takes: never `'static`, and never a lifetime the code around it can stretch, so the body, checked against its own signature's lifetimes, only narrows what it holds, and the loans' life is the flows' premise. A lifetime inferred there instead would be unbounded, and every borrow check around it would pass vacuously (Forbidden).

Branded lifetimes, as GhostCell's, serve well but sit uneasily with associated types, and Rust's trait solver is incomplete there: it can fail to prove what holds, and it can ICE. Where it stops the work, the cause is told apart first, with a minimal reproduction: a wrong structure Rust rightly refuses, the solver's incompleteness, or an ICE. A workaround is a layer added; it is brought back for agreement, never put in alone, and it applies to the whole concept, never to the instance that tripped the solver.

## Set aside

- The per-call wrapper for closures, a candidate before Closure parameters. A closure's signature was written in lifetime-less marker types, which its `call<'b>` mapped through `At<'b>`, a trait with a lifetime parameter, to `&'b T`, `&'b mut T` and so on, a fresh `'b` per call. A model of it, built on 2026-10-03 with stable rustc 1.97.1 and Miri on nightly 2026-07-20, showed a fresh lifetime per call sound: `for_each`, a `map_first` returning a borrow of its argument, a `find`, `retain_mut` and lending a local compiled generically and ran clean under Miri, and rustc refused every misuse probed (E0515, E0499, E0521, E0502). One lifetime `'x` for every call was sound but lost expressiveness: a local could not be lent (E0597, E0505), nor `retain_mut` pass its per-element borrows (E0521). A higher-ranked opener over the projection, `impl for<'b> FnOnce(<A as At<'b>>::Out)`, failed for a generic `T` (E0311), since `for<'b> At<'b>` asks `T` to outlive every `'b`. It was set aside for Closure parameters, which adds none of its primitives, the closure marker, the passing-mode markers, `At` and `call<'b>`, and keeps each argument's lifetime exact where the wrapper gave every reference argument of a call one `'b`.
- Lending a struct's projection in continuation-passing style. Through a closure, `impl for<'a> FnOnce(Self::Ref<'a>)` with `Self: 'a` on the GAT, rustc takes the bound to imply `'static` and refuses a generic or loan-holding struct, the solver's incompleteness; an implied bound through an extra `PhantomData<&'a Self>` argument does not lift it. An implied bound through a defaulted parameter of the family trait, `trait At<'b, Bound = &'b Self>`, does lift it for a generic or loan-holding `T` (observed by probe, stable rustc 1.97.1 and the next-generation solver alike, as lending-iterator and nougat use it), and a continuation trait whose method quantifies its own lifetime under `where S: 'a` passes every such case, nesting included. It was not taken: a declaration's struct or enum is closed by the transparency rule and lent whole, so no part is projected.

## Open

- A Rust function given to acvus. `FnKind::make` takes a Rust closure of raw `Value`s, called safely, which consumes its arguments on a premise its type does not state. Rust has no `unsafe` closure, so an honest body needs a type of its own with an `unsafe` call, a primitive to agree on. The per-call wrapper's model found how such a body opens its arguments: at a named call lifetime, `open_call<'call, T>(args: &'call [Value]) -> &'call T`, which compiles generically, as an entry's `'call` does, where a higher-ranked opener fails (Set aside). A generic body may hand acvus a Rust function that names `T` under `FnKind::make`'s `'static` bound when it captures nothing.
- What a Rust closure given to acvus may capture. A capture whose lifetime appears inside `T` can flow into a per-call result under a binder over `&'b T`, since the binder then ranges only over lifetimes `T` outlives (rust-lang/rust#106431, observed by probe); yoke found this unsound in its own setting. It matters once such a closure may capture a loan.
- Kept functions: the names, `#[fn_type]` and `as_fn`; how an alias states the function's effect; and the views of a kept `FnMut` and `FnOnce`, such as `as_fn_mut(&mut self, ..)` and `into_fn(self, ..)`.
- Async calls: the names, `scope` and `call_async`; how the contract states the runtime's spawn, which takes a future that is not `'static` under the obligation of Async calls; and whether an alias writes a suspending function as `dyn AsyncFn(..)`.
- Whether `Boxed` keeps its atomic count. It was there for `Runtime::share`, which is withdrawn; values that share storage by one identity end it with the last of them (ownership §6), which may or may not ask for a count.
- Agreement on the candidates: Closure parameters; and Collecting a signature, with its primitives, the trait, the list of lifetime markers and `Same`, and with how a declaration tells type variables apart, the check's placeholder being one for all.

## Forbidden

- Instantiation: a named type or form treated apart, a marker, trait, wrapper or conversion for one case, a flag parting two meanings held in one place. One makes a path Rust does not guard; nothing built on it can then be guarded, and no further work repairs that.
- Reliance on what Rust does not guarantee. One exception is admitted: instantiations of one type that differ only in a marker inside `PhantomData` sharing a layout within one compilation (Type variables). It rests on rustc's current layout algorithm alone, the Rust project declines to guarantee it, and no check exercises it. A fat pointer's layout is not relied on: a run or text is taken apart and rebuilt through std (Values and references).
- Lifetime detach: a lifetime made through `unsafe`, such as an unsafe GAT projection `'x` over `'t`, that is not bound to its source. In generic code it is unbounded, and every borrow check around it passes vacuously.
- A step the primitives do not derive: stop and ask.

## Verification

Each change to the extern crates passes, on the safe runtime:

- `cargo test -p acvus-extern -p acvus-extern-macro -p acvus-extern-test`
- `cargo +nightly miri test -p acvus-extern-test`
- `RUSTFLAGS="-Zrandomize-layout -Zlayout-seed=N" cargo +nightly test -p acvus-extern-test` for several `N`, in a target directory of its own. It catches a reliance on layouts other than the admitted exception, which it cannot exercise (Type variables).

The evidence for decisions taken by experiment is in `docs/experiments`, each with a README stating how to run it: `reference-markers` (A reference held uniformly), `closure-params` (Functions) and `lifetime-query` (Collecting a signature).
