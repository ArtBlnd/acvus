# Extern boundary

acvus does nothing by itself: a host supplies every capability as an ExternFn.
This topic decides how an ExternFn is declared, how registries combine into
one input for the compiler and one for the runtime, how a value crosses
between Rust and the language, the ABI a runtime implements to call a handler,
and what the host states about the value the entry returns.

## RFC-0021: A registry is a manifest and a handler table, combined once

Status: Accepted

1. A registry contributes a manifest — type declarations, shared signatures
   (RFC-0019) and function declarations, free of any runtime — and a handler
   table, one handler per declared function, for one runtime. A function
   declaration carries its name, its polymorphic type, the bound of each type
   variable, whether it is a cast or a view, the signature it is an instance
   of if any, and the instances it requires.
2. Every registry declares one namespace, and every name it declares lives
   under it: `core::eq`, `vec::push`. A name without a namespace is a
   script's own.
3. There is one way to declare a function, `#[extern_fn]` (RFC-0023). A
   function that needs state beyond its arguments — a client, a cache — takes
   it as a `#[state]` parameter; the registry lists the function with its
   state value (`greet(greeting)`) and the handler holds it.
4. All registries are combined once, before anything is checked or run,
   together with the `core` registry of the shared signatures the compiler
   names. Combining rejects a name declared twice and a second instance of
   one signature at one type, collects every instance of a signature into one
   function under the signature's name, registers cast rules and views, and
   yields the compiler's input — the functions and the type registry — and
   the runtime's — the handler table. Nothing is registered after combining:
   a program's externs are fixed before its first check.
5. A requirement stays on the declaration that takes it: the signature, its
   pattern at the declaration's own variables, and the highest task an
   instance it reaches may run at. The checker decides it as it decides a
   call of the signature (RFC-0068 rule 5). Where the requirement's receiver
   stands at one of the declaration's type variables, combining also meets
   that variable's declared bound with `OneOf` the types the signature has
   instances at; a signature's own receiver variable is bounded the same way.
6. A type's name is `ns::name`, for an extension type and for a derived
   struct or enum alike. The namespace is the root unless the declaration
   writes one: `#[extern_type(ns = "..")]`, `#[ty_arg(ns = "..")]`.
7. Combining holds type names and Rust types to a bijection: one name is one
   Rust type, and one Rust type is one name. A registry's `types` gives each
   extension type's name with its Rust type. A function or signature
   declaration gives the name and the Rust type of every type its Rust
   types reach, recorded as its acvus type is built, so the author of a
   derived type writes nothing for it. Combining refuses a name given two
   Rust types, as it refuses any name declared twice; refuses a Rust type
   given two names; refuses a declaration that reaches an extension type no
   registry's `types` lists; and takes two declarations of one Rust type
   under its one name as one.
8. The Rust type behind a name is its `TypeId` at its declaration form:
   the type with `()` for each of its variables and `TypesOnly` for its
   runtime, the form a registry's `types: [X<_>]` names. It is compared only
   with another declaration form's, never with a box's, since a box holds the
   type at its run-time instantiation.

**Why.** Joining a declaration to its handler in one value made a registry
generic over a runtime even where only its declarations were wanted, and
registering one registry at a time into a type registry left no place that
held all of them. Shared signatures need that place. A parameter marked as
state gives what a closure's capture gave, inside the one declaration form,
and the declaration's acvus type does not mention it.

The checker names a type by its acvus name and the runtime keys a box by the
Rust type (RFC-0076). With nothing tying the two, two Rust types declared
under one name were both admitted, and a box made as one was read as the
other. Held to a bijection where the registries combine, the acvus type
determines the box, and no second key is carried in the type.

**Cost.** A script's `A::B` names the enum `A` at the root, so a script
writes no literal of a derived enum declared under a namespace; it takes
such a value from a declaration. A registry that may be combined without
the registry declaring a type it names lists that type too, which combines
as one. An extension type
whose parameter carries a bound `()` does not meet has no declaration form,
and its derive does not compile.

**Rejected.**
- Declaration by closure, a handler trait over closure types, `with_effect` —
  a second form every consumer meets; state is a parameter and the effect is
  on the attribute.
- Registration one registry at a time, by side effect — no place holds every
  registry, and shared signatures need one.
- A Rust key in the acvus type, a `TypeId` at each leaf — a declaration
  form's `TypeId` is not the box's, and a leaf that holds a variable has
  none.
- A check at run time — the program was admitted; a name with two Rust types
  is refused before anything is checked.
- A derived type's module path as its namespace — automatic, but it moves
  the name whenever the type moves; the root is the default, and a clash is
  refused by the name it is on.

## RFC-0023: An ExternFn is declared once, as a Rust function under `#[extern_fn]`

Status: Accepted

    #[extern_fn(effect = pure)]
    fn len<T>(c: &Vec<T>) -> u64
    where
        T: Var<kind::Type>,

1. The acvus type and the handler both come from the one Rust signature.
2. A function that uses the runtime takes `ctx: &mut Ctx<'_, Rt>` first: the
   runtime and the window above the calling frame as one parameter (RFC-0067
   rule 7). A function that does not use it takes nothing for it. The runtime
   is first or absent, and is not part of the acvus type.
3. A `#[state]` parameter is taken by shared reference and is not part of the
   acvus type: its value is supplied when the registry is built (RFC-0021
   rule 3).
4. An `Instance<S, I, Rt, Task>` parameter declares a requirement (RFC-0067)
   and is not part of the acvus type.
5. Every other parameter is an acvus parameter, and its Rust spelling is its
   mode: `T` is taken by value; `&T` / `&mut T` declare `&T` / `&mut T`
   (RFC-0018) and are read at entry; `&str` declares the string view
   (RFC-0062); `&[T]` / `&mut [T]` declare the slice (RFC-0047); a type that
   carries a lifetime is a projection (RFC-0050 rule 6). `T` names the acvus
   type through `TyArg`. There is no `&mut str`.
6. The return type names the acvus return type. `Result<T, E>` is the
   language's `Result<T, E>` (RFC-0038) and crosses as one value. A returned
   borrow — `&T`, `&mut T`, `&[T]`, `&str`, `Option<&T>` — is a reference
   into storage a parameter lent: the declaration must take a parameter by
   reference and runs at `Task::Sync` (RFC-0047 rule 3).
7. The acvus name is the Rust identifier unless `name = "..."` says
   otherwise; the namespace is the registry's.
8. `effect = pure | idempotent | opaque | <effect variable>`; undeclared is
   opaque. `commutative` marks a commuting effect (RFC-0013); `heavy` and
   `sync = <fn>` name the task (RFC-0046); `instance_of = sig` declares an
   instance of a shared signature (RFC-0019). `#[extern_cast]` declares a
   cast rule from the one parameter's type to the return type, and a cast is
   pure; `#[extern_view]` declares a machine view (RFC-0047 rule 3, RFC-0062).
9. Generic parameters are the declaration's variables, one kind each:
   `Var<kind::Type>`, `Var<kind::Effect>`, `Var<kind::Length>`,
   `Var<kind::Identity>`, and at most one `Runtime`. A type variable may add
   `Monomorphize<(T0, ..)>` (RFC-0011, RFC-0041).
10. A body never opens a type variable: at run time it is the runtime's value
    (RFC-0039 rule 8), read back at a type only through the crossing
    (RFC-0068).
11. Positions acvus has and Rust does not are spelled by host types:
    `Arr<T, N>` for an array of variable length, `Closure<A, R, E, Rt>` for a
    function-typed parameter, `Ref<T, M, Rt>` for a reference the body keeps
    (RFC-0028 rule 5), and `Pure` / `Idempotent` / `Opaque` / `()` where an
    effect, length or identity argument is fixed.

**Why.** A declaration stated three times — handler, hand-built type,
registration — can disagree, and only a script finds it. A runtime parameter
every function must declare and most never read is the glue's convenience
written into every signature; the glue knows whether it passed one.

**Rejected.**
- A closure-declared ExternFn — state is a parameter, not a capture.
- A parameter mode written beside the type — the Rust spelling already is
  the mode.
- A runtime parameter every function takes — see Why.

## RFC-0028: An element read out of a container is a loan on it; a reference is one carrier

Status: Accepted

1. A container is read through plain functions, one set per container
   namespace (`vec`, `array`, `deque`, `string`), under shared bare names —
   `len`, `is_empty`, `get`, `first`, `last`. A bare name is the set of these
   signatures, settled by the call's evidence (RFC-0043); the qualified name
   picks one. `core::` shared signatures are kept for operations every type
   answers: `clone`, `eq`, `cmp`, `hash`, `to_string`.
2. A container is read through a reference to it, and an element read out of
   a borrowed container is a reference into it: the result holds the
   container's loan (RFC-0018), so the container is neither moved nor changed
   while the element is in use. No function reads an element out by value; a
   value of the element type is a `clone` of the reference, for the types
   with a `core::clone` instance.
3. A `Vec<T>` is changed through `&mut Vec<T>` under Rust's own names —
   `push`, `pop`, `insert`, `remove`, `clear`, `truncate`, `extend`, `swap`
   and their neighbours. Each takes the vec first, so `v.push(x)` is the call
   written as a method on a place (RFC-0030), and each panics where Rust's
   does, with Rust's message.
4. `vec::filled` is a shared signature with one instance per element type,
   not one generic function: `Runtime` offers no clone of a value, so the
   copies are made in Rust by an instance that knows the element type.
5. In an extern declaration, `Ref<T, Shared, Rt>` and `Ref<T, Mut, Rt>` are
   the acvus types `&T` and `&mut T` wherever they stand — a parameter, a
   result, a type argument — and each is the reference value itself, for a
   body that keeps it, returns it, or hands it to a closure. A Rust `&T` /
   `&mut T` parameter declares the same type and is read at entry. A returned
   `Ref`, or `Option` of one, is the reference it carries, and `None` is
   `Runtime::none()` (RFC-0039 rule 6). A parameter `&Option<T>` or
   `&mut Option<T>` is refused where it is declared: no storage is shaped like
   an `Option<T>`; take `Option<&T>` or the option by value.

**Why.** A shared signature is one generic handler per instance, so a
per-element `Monomorphize` function cannot be one of its instances, and a
signature's name is one function to the resolver, so `string::len` and a
container `len` could not both exist; a bare name as a set of signatures
(RFC-0043) gives every container the same names and admits both. An element
is returned as a reference because a body cannot copy an erased element, and
taking the container by value to read one element costs the whole container.

**Rejected.**
- Container reads as shared signatures under one `container::` name — see
  Why.
- An element read out by value, the container taken by value — costs the
  container for one element.
- A phantom `Ref<T>` naming the type beside a `Lent<T, Rt>` carrying the
  value — every use of the phantom stood where a runtime was in scope, so one
  carrier serves both.

## RFC-0039: Every type that crosses the boundary says how, through one trait

Status: Accepted

1. Every type an ExternFn takes or returns implements `Cross`, and the glue
   calls it and nothing else. No crossing is chosen where a macro expands:
   the type says how it crosses. The crossing's width, its one-value half and
   the argument modes are RFC-0059 rules 1–3.
2. A scalar is stored as itself.
3. An extension type — `#[derive(ExternType)]` — is stored as its payload,
   the first field, and is `#[repr(transparent)]` over it, so a reference to
   the payload is a reference to the type; the derive requires the attribute.
   Its phantom parameters never reach the store: the checker's types live
   only in the name.
4. `#[derive(TyArg)]` on a Rust struct declares the object type its fields
   spell, and on a Rust enum the language's enum of the same name, and the
   value crosses as that object or enum: an ExternFn takes and returns the
   Rust type, and a script reads every field and matches every variant. A
   unit variant has no payload, a one-field tuple variant's payload is that
   field, and a struct variant's payload is the object its fields spell. Each
   field crosses by its own `Cross` — a struct inside a `Vec` inside a struct
   is rebuilt at every level, an extension-typed field crosses as itself. The
   derive reads no attribute of another derive: what `serde` renames or tags
   is the wire's, and the language sees the Rust names. It refuses generic
   parameters and a tuple variant of two or more fields, since a variant has
   one payload. The runtime layout is RFC-0050 rules 4 and 8. A derived value
   has no storage of its own type: a borrowed aggregate crosses as its
   projection (RFC-0050 rule 6), a `#[projection]` type borrowed whole is a
   compile error, and any other derived type read through a reference panics
   at the crossing.
5. A container crosses each element by the element's own crossing. The
   dynamic-length sequence is `Vec<T>` in the language and
   `std::vec::Vec<T>` in Rust, with no newtype; its `TyArg`, `Cross` and
   `ExternTypeDecl` impls live in `acvus-extern`, the crate that owns the
   traits, and its functions in the registries. A `Vec<T>` is stored as the
   runtime's `Vec<Owned<Rt>>` — the whole buffer when its element is the
   runtime's value in place, element by element when the element converts.
   `Arr<T, N>` crosses per element; `Result<T, E>` is one runtime value
   holding each side by its own crossing. A container is read through a
   reference only when its element is the runtime's value: a converted
   container has no storage of its element type.
6. An `Option<T>` is its payload's value with no shape of its own:
   `Runtime::none()` is `None`, `Runtime::some(v)` is `Some(v)`, and
   `is_none` and `unwrap_some` read them back. It adds no allocation and no
   indirection, and its `Cross` is those four calls over `T`'s own crossing
   and nothing else — no static type, no marker, no `TypeId` — so a handler
   at `T = Rt::Value` crosses it by the same calls. The runtime holds the two
   apart however it likes; the interpreter gives a `None` a word that counts
   the `Some`s around it, so `Some(Some(None))` is one word at depth two and
   `Some(v)` for any other `v` is `v`. No storage is shaped like an
   `Option<T>`, so `&Option<T>` and `&mut Option<T>` are refused where an
   extern declares them.
7. A carrier — `Ref<T, M, Rt>`, `Closure<A, R, E, Rt>` — is the runtime value
   it holds, and is made only by the crossing (RFC-0068 rule 1). A slice and a
   string view are the register pair (RFC-0047 rule 6, RFC-0062).
8. The runtime's own value crosses as itself. A type variable of an ExternFn
   is that value at run time, so a `Vec<T>` parameter reaches the store as
   `Vec<Value>` in one erase. The ABI passes `Rt::Value` and owes nothing; a
   Rust store that owns a runtime value holds `Owned<Rt>`, converted at the
   glue (RFC-0048 rules 1 and 7).

**Why.** A crossing chosen by autoref specialization where the glue expands
resolves at the generic definition, not at the instantiation: a derived
struct nested in another crossed as an opaque Rust value, and
`Result<Regex, E>` fell to the as-is tier as a whole. One trait every
boundary type implements makes an element's crossing a bound the compiler
checks and puts the choice where the type is declared. `repr(transparent)` is
the one fact that lets a reference cross between the checker's phantoms and
the store's payload. A struct and an object are different layouts, so the
conversion is O(fields), paid once per crossing and never on access. `Vec<T>`
under Rust's own name lets a wire struct and its language object be one
struct.

**Rejected.**
- Crossing tiers chosen at the expansion site — see Why; a nested type
  silently crosses opaque.
- A `List<T>` newtype over `Vec<T>` — every function on it wraps and
  unwraps, and a struct declared for the boundary has to name `List<T>` where
  the rest of its code says `Vec<T>`.
- A shared layout between a derived struct and its object — they have none.
- Parsing a wire format at run time through a language type — parsing is the
  extern fn's job; the derive projects its result.
- A write through an option in Rust storage (`take`, `replace`, an
  `&mut Option<T>` parameter) — there is no such storage: a
  `Vec<Option<i64>>` crosses to flat elements, and a pattern through `&mut`
  binds `&T`, through which the checker refuses a store.

## RFC-0041: `#τ` is the representation of a slot

Status: Accepted

A value of type `τ` has a representation: uniform, the runtime's `Value`,
which every polymorphic position holds; or specialized, the Rust type `τ`
itself. `#τ` names the specialized representation and is a fact about a
**slot** — a type argument or an effect argument of a user-defined type, or
the target of a reference — never about a bare type: there is no `##τ`. A
user-defined type declares per type parameter whether its slot can
specialize (`specializable`); a slot that cannot is uniform. Every effect
slot can. `Fn`, `Object`, `Enum` and handles have no representation the
language defines and carry no `#`.

**`#` is marked per part.** A `#` argument is a tree that follows its type
(`TypeArg`, `HeldTy`): a `#` tuple, option, result or array marks each of its
parts again, a part whose Rust type is itself is `#`, and a part a type
variable's run-time instantiation fills (`Owned`, `Erased`) is uniform.
`Bag<(i64, U)>` is `Bag<#(#i64, U)>`; any other head is one `#` leaf. A `#`
array states its Rust head as well: the language's array, whose Rust type is
`Arr` with its elements in a buffer, is `#Array<#i64, 2>`, and a Rust
`[i64; 2]` with its elements in place is `#[#i64; 2]`. They are one type and
two trees. `Held(T)` is never a Rust array, since `T` with every part `#`
writes an array as the language's. Two
types meet only where the marks agree at every part: uniform with uniform,
`#τ` with `#τ`; uniform against `#` is a mismatch. A variable inside a `#`
composite is uniform at its own position and never binds across a `#` leaf.
`#T` over a variable in a pattern is `Held(T)`: `T` with every part `#`.

**An argument written concrete is held.** A map, a set, a deque and a
derived extension type are each kept as one box of the Rust type their
arguments name, and the glue fills a variable with its run-time
instantiation — a type variable with `Owned<R>`, an effect variable with
`()`. A declaration that writes an argument concrete — `Deque<i64>`,
`HashMap<K, V, Pure, R>`, `Keys<K, V, Pure, I, R>` — names another box, so
that argument is held: `#i64`, `#pure`. A variable stays uniform, and a
`Monomorphize` member is `#` as everywhere. `TyArg::held` gives the tree per
Rust type, and `held_effect` an effect's one mark, at every crossing: by
value, `&` and `&mut`. A value a generic constructor made meets
a held parameter as a type mismatch that names both types; a writer and a
reader that write the same concrete argument meet at the held slot.

Only a `Monomorphize` member and a held argument make `#`. `#[extern_fn] fn reverse<T:
Monomorphize<(f64,)>>(Vec<T>) -> Vec<T>` has the concrete instance
`reverse@#f64 : Vec<#f64> -> Vec<#f64>` and, when `T` has no other bound, the
generic instance `Vec<ρT> -> Vec<ρT>`. A specializing slot is the held tree
when it holds a member part, `ρ` over the whole argument when it holds a
variable part, and uniform otherwise; a declaration has one `ρ` per slot
type, and a `ρ` binds the whole tree it meets. A plain concrete signature
(`-> Vec<String>`) stays uniform. The compiler chooses the instance by type
(RFC-0040); a member's glue crosses the family whole
(`OneValue<Rt, Specialized>`: one box, O(1)); every family a member names
declares its two casts `F<#T> -> F<T>` (erase) and `F<T> -> F<#T>`
(materialize), one generic fn each with concrete instances, merged across
registries by type. A member is a Rust type written out, with no part a
variable or `Erased` fills; the registry refuses one that has such a part
(`FamilyMemberNotWritten`), since `#T` would call that part `#`.

**An instance may choose a signature's variable.** `extern_signature!` alone
takes the bound `Ts: Var<kind::Type> + Chosen`, a variable each instance
fills with a Rust type of its own. Its slot is a `ρ`; `combine` binds it to
the instance's tree at that slot, and `Ts` to the tree's type, then matches
the rest exactly. An undeclared `ρ` still stands for uniform. The reason is a
payload that projects through the variable, whose box is its own Rust type
per instance (RFC-0076).

A signature's `ρ` is bound only by a decision — the instance choice, or
`solve`'s default `Uniform` — never by a value flow. A flow whose only
disagreement with its target is such an open `ρ` against a fixed
representation is a conversion decision at that site (RFC-0042): a call
argument, a store into a typed place, a return, a pattern's source, an `else`
branch. It is answered by identity when the decision agrees and by the
family's cast when it does not. A conversion consumes the value it converts.
A converted `&place` argument is taken out of its slot for the call
(RFC-0077).

An extension holds a uniform value through `Erased` (RFC-0076).

**Why.** Two representations of one type need one rule for where they meet.
Putting `#` on the slot keeps it structural and lets the solver treat it as
one more component of a type; making it only by `Monomorphize` leaves a
program that asks for no native layout unchanged and confines the second box
to fns that ask for it (`&[f64]`, `Vec<T>` by value into a Rust API). Deciding
`ρ` by instance choice rather than by flow is what takes a `#` value to a
generic-only fn through one erase instead of a mismatch, and a uniform value
to a member through one materialize. Holding an argument
written concrete makes the Rust type of a box a fact of its acvus type, so
a box of one instantiation never reaches a declaration of another, and the
refusal is the checker's, where it names the two types. The mark is per part because a box's Rust
type differs part by part and the runtime reads a held value by its parts:
a handler taking `Bag<(i64, U)>` reads `.0` of a `Vec<(i64, Owned)>`, and
`Vec<(i64, i64)>` handed to it panicked in debug and crashed in release. The
array's head is in the tree for the same reason: a `Bag<[i64; 2]>` handed to
a declaration of `Bag<Arr<i64, N>>` panicked in debug and aborted in release.

**Rejected.**
- One mark per argument, a composite taking the strongest of its parts — it
  admitted `Vec<(i64, i64)>` where `Vec<(i64, Owned)>` was declared. One
  more mark, "mixed", admits `Bag<(i64, U)>` against `Bag<(V, i64)>`.
- A Rust array as one `#` leaf of its type — a leaf keeps no marks of its
  parts, so `[Arr<i64, N>; 2]` and `[[i64; 2]; 2]` would be one leaf, and a
  leaf's type is written back as the tree `#` builds, so the next
  substitution turns it into the language's array.
- A cast-only matching mode where a pattern's `ρ` matches any tree —
  `Held(T)` states the family's pattern under the one matching rule.
- Refusing a concrete argument at the declaration, as a bound asked of the
  variable (`InPlaceElement` of a deque's element, an effect bound of a
  map's effect) — it covered a borrow only, left the by-value crossing and a
  derived type's effect argument to read a box of another type, and refused
  a writer and a reader that name the same concrete argument, which agree.
- `ρ` on a plain concrete signature and on `let` bindings, so that a value no
  `#` consumer touches is uniform from birth without a cast node — the
  checker is conservative: a value keeps the representation it is born with,
  and `#` arises only by conversion at a site whose instance demands it. The
  cost is a copy at that site; moving a representation beyond it is the
  optimizer's.

## RFC-0054: The host declares what `main` returns

Status: Accepted

**The host declares the entry's return type at compile time, and the
compilation holds the body to it.**

1. **A compilation declares the entry's return type.** The declaration is a
   `Ty`, and it enters through the slot the graph already has for one: the
   entry `Function`'s `PolyTy::Fn { ret }`. A declared entry is an ordinary
   declared function. A `CompilationGraph` carries its entries, each a
   `QualifiedRef` with its own declaration, and a graph with none has no
   host. A template's entry declares `String` itself, since its tail is
   text.
2. **The body joins the declaration on the shared path.** `check_script`
   sets the body's own return variable from the declaration, so the tail and
   every `?`'s early return join it through one path; a body with no tail
   joins `Unit`. There is no entry-only branch in the solver.
3. **`validate` receives the declaration by structure.** `MirModule` carries
   `ret: Ty` — not `Option<Ty>` — and every pass carries it through, so no
   module has an unchecked `main`.
4. **`validate` holds the pipeline, not the source.** A source-level mismatch
   is refused by the checker and produces no module; `validate` checks what
   the passes left against the declaration.
5. **`!` declares that the host states no return type.** A host that cannot
   name a type — one that prints whatever the file returns — declares `!`
   (RFC-0038). Under it every `Return` is accepted, the checker holds the
   tail to nothing, and the host reads the value it gets back by the kind the
   runtime carries with the value. It is spelled by the host, never
   defaulted. `types_match` reads a declared `!` on the expected side; a `!`
   value satisfying any slot is a separate site.
6. **A compilation declares the entry's inputs as it declares its
   return.** The declared inputs are the entry's parameters, and a body reads
   a `$` only as one of them or as a binding (RFC-0071 rule 5). A
   declaration of none declares none. Taking the inputs from the body's `$`
   reads instead is spelled by the host, as `!` is: it serves an analysis that
   reports what a body requires, and an entry compiled that way runs only
   once a binding fixes every `$` it reads.

What the entry's result is at run time — one runtime value — is RFC-0062 and
RFC-0064.

**Why.** An entry return type inferred from the body is a summary of the
body's returns, so every return agrees with it by construction and nothing is
refused. The declaration is the contract the host reads the value with.

**Cost.** Every host names a type.

**Rejected.**
- Inferring `main`'s return type and checking the body against it —
  circular; see Why.
- A run-time kind check on the host side — a convention every host must
  remember, it fires after the run, and it says nothing at the contract the
  script was compiled against.
- An explicit-any `Ty` beside `!`, or an `Error` token that unifies with
  everything — a second spelling of "I state no type" that every site reading
  a declaration would have to keep in step with the first.
- A `--returns` flag at the CLI — moves the declaration to the person running
  the file, who has no more to say about it than the CLI.
- An entry-only branch in the checker — leaves `?` unheld.

## RFC-0059: The macro emits only calls; the runtime owns the ABI

Status: Accepted

`#[extern_fn]`'s output for a handler is a call to a library constructor over
typed markers, and every ABI fact is a constant or an associated type of the
crossing, never read from tokens.

1. **A crossing is the run of values it occupies.** `Cross<Rt>` carries
   `type Form: Form`, `unsafe fn from_run(rt, run) -> Self` and
   `fn into_run(self, rt, out)`, and at the return position `type ReturnForm:
   Returned` and `into_return_run(self, rt, out) -> Verdict`, which has no
   default: a crossing whose result may be absent answers its verdict from
   the value. A form is the width as a type — `One`; `Pair`, the register
   pair of a slice or a string view (RFC-0047 rule 6); `Run<W>`, an
   aggregate's `W` components (RFC-0050 rules 5, 6, 8); `Nothing`, a
   parameter that takes no argument value; and at a result `OptionOf<One>`.
   `Form::WIDTH` is the only place a width is a number, and `Form::KIND`
   (`Value`, `View`, `Components`) tells apart forms of one width. A
   declaration's argument width is the sum of its parameters' widths, added
   in the library from the types. A slice or a view crosses through
   `Runtime::slice_into_run` / `slice_from_run`, because only a runtime knows
   what one of its values is made of. A `#[derive(TyArg)]` struct is one heap
   value as a field, an element or a by-value parameter, and its own
   components as a result: its `ReturnForm` is `Run<W>`, with `W` the field
   count the derive writes as a literal.
2. **Two crossings, and no blanket.** `OneValue<Rt, Rep>` is the crossing
   that is one of the runtime's values — `erase`, `materialize`,
   `STORED_AS_VALUE` — at `Rep = Uniform` or `Specialized` (RFC-0041). Every bound that needs a value says `OneValue`: an object's
   field, a container's element, a closure's argument and result, a
   parameter taken by reference. A slice and a view implement `Cross` alone,
   so a slice in any of those positions is a compile error. A parameter by value is built by
   its `Cross` at that crossing's width, which is what admits a slice
   parameter. There is no `impl<T: OneValue> Cross for T`: coherence cannot
   admit it beside `impl Cross for Slice`, so each one-value type states both
   impls, the `Cross` half through `cross_one_value!`. `Cross` is not a
   supertrait of `OneValue`: `Option<T>`'s `Cross` holds only where `T`
   crosses uniformly, and a supertrait would demand that of the specialized
   impl too.
3. **Argument modes are one trait.** `Arg<'a, Rt>` carries `type Out`,
   `type Form` and `unsafe fn take(rt, run, site) -> Out`. What a parameter
   needs from its call site — nothing for most; an object projection's field
   positions, an enum projection's tag words, a required instance's entry —
   is `Sited<Rt>::Site`, on a lifetime-free trait so that a site table built
   once is one type per parameter and not one per lifetime it is read at. The
   macro names the marker from the Rust parameter's spelling (RFC-0023 rule
   5): `ByValue<T, C>`, `ByRef<T, M, C>` with `M` = `Shared` | `Mut` and `C`
   = `Uniform` | `Specialized`, `ByStr`, `BySlice<T, M>`, `ByProjection<P>`,
   `Required<S, I, Task, N>`. The borrow modes require `Borrowable<Rt>` (or
   `BorrowableSpecialized<Rt>`), whose `on_unimplemented` text is the
   refusal of `&Option<T>` and of a whole borrowed aggregate: a trait
   error, not a name check. The in-place read, `deref` and `deref_mut`, is
   a method of those two traits, so no reader reaches a storage as a `Self`
   without the bound; `Loan::borrow` and the `Restore*` positions, written
   once over the representation, ask it through `Lends<T, Rt>`, which
   `Uniform` and `Specialized` implement under each one's bound. A `Vec`
   or an array is borrowable only at an `InPlaceElement` element, which only
   `Owned<Rt>` is: its storage is the runtime's `Vec<Owned<Rt>>` whatever
   the element type. A map, a set, a deque and a derived extension type are
   each one box of the Rust type their arguments name, and each argument is
   held (RFC-0041), so each is borrowable at every argument. A result is `Ret<Rt>` — `Val<T, C>`,
   `RetStr`, `RetLent<L>` for a returned borrow — and `Ret::Of<'a>` is a
   generic associated type, so a result may borrow what the arguments lent.
4. **A handler is the operation's type parameter.** The registry holds
   object-safe factories: a `HandlerFactory<Rt>`, sited into an `AtSite<Rt>`
   that makes the operation; `ExternHandler` is `Sync`, `Heavy` or `Async`
   over them. The operation's trait is not object-safe:
   `Handler<Rt>` carries `type Args: ArgRun`, `type Ret: Returned`,
   `const WIDTH: Width` and one call,

       unsafe fn call(
           &self,
           ctx: &mut Ctx<'_, Rt>,
           run: <Self::Args as ArgRun>::Run<'_, Rt>,
           out: <Self::Ret as Returned>::Out<'_, Rt>,
       ) -> <Self::Ret as Returned>::Verdict;

   where `run` is the argument run — an array at a register form, the window
   otherwise — and `out` the destination run; the verdict is `()`, or
   `bool` for a result that may be absent. `Runtime::op<H>`, `fused<H>` and
   `async_extern_op<H>` take the handler by value under its own type, and the
   host builds an operation holding it as a type parameter, so a call is a
   static call and the handler's body is what the operation runs. The one
   `dyn` on the synchronous path is taken at preparation, in `into_op`.
   Which form an operation takes is a fact of the handler's type: `ArgRun`
   (`InRegisters<0>` … `InRegisters<4>`, `InWindow`) and `Returned` each name
   their form, and `select` hands the handler to the one method of the host's
   `CallForms` / `RetForms` that form names, so a handler instantiates its own
   form and no other.

   The library implements the handler for closures: `Glue<Rt, F, A, R, S,
   E>` over a tuple `A` of `Arg` markers and a result marker `R`, one impl
   per arity through a `macro_rules!` over tuples, with the `Width` sum
   written once there. `Handler` is implemented for the sited glue alone, so
   an operation cannot hold a glue whose site table was never filled;
   `Unsited` is what the registry holds. `#[state]` is a capture: the macro
   emits a closure over an `Arc<(T0, …)>` — typed, never `Any`, never
   downcast. `ExternHandler::heavy` takes `ValuesOnly` handlers — every
   parameter one that survives the caller suspending, the result one
   register — because a call the caller
   waits for outlives the frame its arguments were lent from.

   A call that crosses a thread — heavy, awaited, spawned — keeps a shared
   `dyn`: it is sent to a pool rather than run in the caller's frame, and the
   send, not the call, is its cost. A fused run keeps its `dyn` over nodes
   rather than handlers, because its calls reach different declarations
   (RFC-0044). A host with no registers to lay a call in implements the three
   entries with `direct_call_forms!` and gets `DirectOp`: the handler behind a
   closure that takes the argument run as it comes.
5. **Object, enum and transparent glue are library functions.**
   `#[derive(TyArg)]` calls `acvus_extern::derive::object` for a struct and
   `derive::variant` for an enum, so the layout (RFC-0050 rules 4 and 8)
   lives in those files alone and a layout change touches no macro.
   `#[derive(ExternType)]`'s pointer cast is `derive::transparent::{erase,
   materialize, deref, deref_mut}`, guarded by `unsafe trait
   Transparent<P>`, which the derive implements only for a
   `#[repr(transparent)]` struct. A family's
   specialization casts are glues like every other handler: argument
   `ByValue<T, Specialized>` and result `Val<T, Uniform>`, or the reverse,
   over the identity body — the two markers are the whole conversion.
6. **Async stays boxed.** `AsyncCall::call(&self, rt, run) ->
   BoxFuture<'static, Rt::Value>` is one `Pin<Box<dyn Future>>` per call,
   owning the runtime and a copy of the argument run so the future outlives
   the frame. Storing the future in the operation needs an `async` block's
   type named in an associated type, which is `impl_trait_in_assoc_type`,
   unstable on the pinned toolchain.
7. **The host reads the handler's width.** `prepare` reads
   `HandlerFactory::width()`, sites the factory with each argument's settled
   type, and builds the `CallShape` the form names; it counts nothing of its
   own. Both halves count the runtime's values, not parameters: a
   `&str` or slice parameter is two, so a declaration whose arguments total
   `REGISTER_FORM` — four — values or fewer takes a register form with one
   register per value, and a wider one is lent its window.
   `HandlerFactory::arity` counts parameters, which is what a site table is
   indexed by. `Width::absent` is the one fact about the result a caller
   holding no handler type needs: whether the call answers a verdict.

**Why.** A macro that decides ABI facts by counting tokens puts every layout
change in a proc macro rather than the library, and answers wrongly for an
alias or a type parameter. A handler behind a `dyn` in the operation is an
indirect call and a body the operation cannot inline; state behind
`Arc<dyn Any>` pays a downcast on every call for a type known when the
registry was built.

**Cost.** One operation instance per handler and form, so the binary grows
with the declarations. A handler called from one operation is inlined there,
and a body that takes a local's address across a callee costs that operation
its tail call.

**Rejected.**
- `Box<dyn Handler>` or `Arc<dyn Handler>` in the operation — one indirect
  call per call and a Rust body the operation cannot inline.
- Every form instantiated for every handler — the forms a width does not
  name are dead code that still costs size and tail calls; forms as types
  make the dead instance unwritable.
- `fn` pointers with `Arc<dyn Any>` state — a `fn` pointer cannot close over
  typed state, so the state is erased and re-checked per call, and every
  statefulness × arity pair is a form enumerated by hand.
- Token detection (`returns_slice`, `names_option`, `by_value_variant`) —
  see Why.
- A panicking one-value crossing for a slice — "has none" is a missing impl,
  and rule 2's split makes the same mistake a compile error.
- A future stored in the operation — rule 6.

## RFC-0075: The contract gains `sleep` alone; a handler joins its concurrency in its own future

Status: Proposed

1. **`Runtime` carries `sleep`.**

       fn sleep(&self, d: Duration)
           -> impl Future<Output = ()> + Send + use<Self>;

   The `use<Self>` bound keeps `&self` out of the future: in edition 2024 a
   return-position `impl Trait` in a trait captures every lifetime in scope
   unless it lists the ones it captures, and an impl repeats its own list.
   `Send` is stated because a handler's future is `Send` and a generic
   caller sees only the bounds the trait names. There is no default: a timer
   is the host's, and a `wasm32` host has no thread to sleep.
2. **Concurrency inside a handler is joined in the handler's own future.**
   An async handler that calls a closure more than once at a time roots one
   frame per call (`Runtime::rooted`, `Runtime::ctx_of`) and polls the calls
   together, with `join_all` or `FuturesUnordered`. Nothing leaves the
   handler's future, so every borrow is Rust's, and a run dropped while it
   waits drops the calls with it.

   `unordered` is such a stage. It takes a `map` stage, `Map`, and is the
   script author's statement that the order of the map's calls and of the
   draws below it is irrelevant, as `anyorder` states it for a region
   (RFC-0007): no declaration of the closure is asked to commute. Its first
   pull draws the whole input below the map in order, calls the closure on
   every element with the calls joined, and keeps the results; each pull
   hands out the next one in input order. Every call the pipeline makes
   after the first pull comes after every call of the closure. It stands at
   an effect `E: Suspends` (RFC-0011 rule 5) that the map's input, its
   closure and its `next` share, so it is refused on a pipeline that cannot
   suspend, where it would join nothing. It takes a `Map` by type, because
   only a map separates drawing an element from the work done on it: a
   stage's `next` holds its input exclusively and draws one element at a
   time, so an `unordered` over any other stage could not overlap anything.

3. **Parallelism is the handler's own.** A handler that wants threads brings
   its own pool and joins it before it returns, as `rayon::scope` does: the
   call is synchronous, so the borrows keep their Rust lifetimes and there is
   no await at which a run could be dropped. It is declared `heavy`, so it
   runs on the blocking pool rather than on the thread that drives the run.
4. **In the interpreter, `sleep` is the executor's.** `TokioExecutor` uses
   tokio's timer. `SequentialExecutor`, which runs nothing else while a run
   waits, sleeps the thread.

**Why.** The one thing a handler cannot build without the host is a timer.
Concurrency over awaited calls needs no second task, and a second task that
borrows is what makes a spawn unsound when a run is dropped mid-call.

**Cost.**
- `unordered` holds every result of its input at once, and its closure runs
  for elements a later `take` or `find` never reads.
- `unordered` stands after a `map` and nowhere else.
- `SequentialExecutor`'s `sleep` blocks the thread it runs on, including
  other tasks a host runs there.
- Every `Runtime` implementation, the test runtimes included, implements
  `sleep`.

**Rejected.**
- `pmap`, a map stage at an effect bounded to commute: the order a
  script may ignore is the script author's intent (RFC-0007 rule 1), and an
  extern that does not declare itself commutative, such as an `idempotent`
  HTTP `get`, could not be mapped concurrently at all.
- `unordered` before the `map` it affects, marking the pipeline below: a
  map would change behavior by the type of its input, and a stage between
  the two would drop the mark with no error.
- Yielding in completion order: with every call finished at the first
  pull it saves no time, and streaming them needs calls held across pulls,
  borrowing the stage that holds them and polled by no one while the loop
  body runs.
- `spawn` and `spawn_blocking` on the contract as `'static` tasks: a
  closure whose capture references a frame (RFC-0064 rule 5) is then read by
  a task after a dropped run has freed that frame, unless the checker
  refuses every such closure.
- The same two confined to a scope the async glue owns, whose `Drop` waits
  for the tasks it started: sound, but it needs one `unsafe` that erases the
  scope's lifetime, a gate on every poll, and a thread blocked in `Drop` when
  a run is cancelled. That buys parallelism a handler already has through
  its own pool (rule 3) and concurrency it already has through a join
  (rule 2).
- A scope the handler makes and owns, as a local or around an `async`
  closure: whatever a handler owns it can `mem::forget`, and its tasks then
  run past the call.

## RFC-0076: A box is keyed by its payload's canonical type, and an extension holds values through `Erased`

Status: Accepted

An extension reads and edits uniform values in place through `Erased<R, T>`:
`repr(transparent)` over the runtime's value, with `T` only in
`PhantomData<fn() -> T>`. It is made by `Erased::new(rt, T)` and read by
`as_ref(&self, rt)` / `as_mut`, and for an `Inline` type (one that fits the
value word) by `get`, `get_ref` and `get_mut` with no runtime in hand. Each
is an inherent method bounded on what it reads: `T` is `Stored` there, since
a type converted on the way in has no `T` in storage to read. An extension
type is `Stored` at its payload. `Vec<Erased<R, T>>` is the runtime's
`Vec<Value>` and crosses whole.

`Owned<R>` is `Erased<R, Never>`, where `Never` is Rust's `!`, named through
`fn() -> !` until `never_type` stabilizes and then written `!`. Nothing
implements a trait on it. The language's `!` and the uninhabited field of
the compile-time stand-ins is `Bottom`. `Owned` alone is made from a bare
value (`vacant`, `from_value`) and written through (`value_mut`) as
inherent methods, so no safe code breaks an `Erased`'s `T`.

1. At a runtime that makes values, `Erased<R, T>` has no trait impl whose
   existence or items depend on `T`, with two exceptions: `Within`'s impl
   exists only where `T` is `Within` at every lifetime, holding no
   carrier, or is `Never`; and `Owned`'s `Deref` (RFC-0068 rule 2). A
   carrier, and a type that holds one, has no `Within` there, so an
   `Erased` at it reaches no handler (RFC-0079 rule 6). Every other impl
   (`OneValue`, `Cross`, `Stored`, …) bounds neither `T` nor `Self`, so no
   item of it answers by `T`. acvus-extern's `tests/erased_impls.rs` reads
   every trait impl on `Erased` or `Owned` the workspace writes and refuses
   any other. What reads `T` for the checker
   (`TyArg`) holds only at `Erased<TypesOnly, T>`, through `HoldsNoValues`,
   a sealed trait no other crate can name; the macro builds every
   declaration's checker-side types, its own parameters included, at
   `TypesOnly`.
2. A box is keyed by its payload's canonical type (`Canonical`): the payload
   with every `Erased<R, X>` at a uniform part taken to `Owned<R>`, following
   the per-part representation (RFC-0041). `erase`, `materialize` and every
   read in place key on it. Each stored type states its canonical form: the
   derive writes it for a derived type, and `Vec`, `Deque`, `HashMap`,
   `HashSet` and `Arr` write it by hand. A `Chosen` or specialized part is
   not canonicalized; its Rust type is its own.
3. Every read of a box at a type other than its canonical one carries an
   inline `const` assert that the two `Layout`s agree in size and alignment,
   and a `SAFETY` note. An instantiation where they differ does not compile.
4. A derived payload that names a uniform type parameter is proved
   `UniformPayload<M>`, the unsafe marker of a type whose layout reaches its
   type parameters only by holding them. acvus-extern implements it for the
   primitives, `String`, `Vec`, `VecDeque`, `vec::IntoIter`, `Option`,
   `Result`, `Box`, tuples, arrays, `PhantomData` and the runtime's own
   types; `#[derive(Payload)]` implements it for a struct or enum by
   bounding each field type that names a type parameter, and in the same
   expansion `Within` (RFC-0079 rule 6), so a payload's author writes one
   derive; `#[derive(ExternType)]`
   implements it for the extension type by bounding its payload. The derive
   proves the payload in a generated `fn` generic over a marker `__M` that
   only it names, under the struct's own predicates, with each type variable
   assumed `UniformPayload<__M>`: a variable counts as held, and what it
   holds is its own `Canonical`'s obligation. No bound the struct writes and
   no associated type a trait declares names that marker, so a projection
   hidden behind an alias is refused as a visible one is, and every other
   impl the proof selects holds at every argument the struct admits. A
   payload that fails is refused with the field type that is not
   `UniformPayload`.
5. `#[extern_type(unsafe(uniform_payload))]` skips that proof for the whole
   payload: the author asserts it where the marker does not reach, for
   another crate's generic type or a projection through a `Chosen`
   parameter (`O` in `<Ts as TypeList>::Body<O, E>`). A payload that visibly
   projects through a uniform type parameter, `<T as Tr>::A`, is refused
   with or without the attribute. A derived type may bound a parameter by
   `Chosen`: it is never canonicalized, and a projection through it is
   admitted. `Pipe<Ts, …>` holds `<Ts as TypeList>::Body`, so each
   instance's box is its own Rust type, and its parts derive `Within`
   alone. A payload that names no uniform
   type parameter is read at its own type and is not proved.

A read at a type other than the canonical one rests on three layers.
Release: `Drop` cannot be implemented with bounds narrower than the type's
(E0367), so every `Erased<R, X>`, and every type built over one, releases
the same way for every `X`. Size and alignment: checked per instantiation by
rule 3's assert. Field order within one size and alignment: argued, not
checked. It rests on rule 1 within acvus-extern, rule 4's proof or rule 5's
assertion,
`repr(transparent)`, `PhantomData` having size 0, alignment 1 and auto
traits that do not follow `X`, and `Never` having no value. It does not rest
on a promise that two instantiations of one `repr(Rust)` type share a
layout; Rust makes none. `std::mem::TransmuteFrom` (unstable,
`transmutability`) replaces this layer with a bound once stable.

**Why.** The checker types a generic constructor's `Bag<Owned>` and a
declaration's `Bag<Erased<R, i64>>` alike, `Bag<i64>` with a uniform part,
and their Rust types differ only in `Erased`'s parameter. Keyed by the raw
type, one read the other's box: the debug `TypeId` check panicked, and
release read right only because the layouts happened to coincide. One
canonical key makes them one box, and the layers are what make reading it
at either name sound. `X` can reach a layout only through a trait, so the
rule keeps every `T`-dependent impl off a value-making `Erased`; a derived
payload is the one place a downstream trait can reach `X`, which rule 4
proves it does not, at a marker no downstream trait can name. An inherent method takes part in no specialization or projection,
so `Owned`'s construction and mutable access keep rule 1. Rule 1 guards
against a read of a box, a layout or a run-time dispatch answering by `T`;
`Within` does none of these (it has no item), and without its exception a
derived payload holding `Vec<Erased<R, T>>` kept a `Ref<'static, …>` past
the call.

**Cost.** Rule 1 binds acvus-extern only: the orphan rule lets another
crate implement its own trait for `Erased<R, i64>` with a bound on `T`,
and a payload under rule 5 that reads `X` through one breaks the
obligation its author asserted. `erased_impls.rs` scans only the
workspace. An `Erased`
whose `T` holds a carrier is not `Within`, so it crosses at no glue. A type
stored as itself states its canonical form, so a concrete type under
`cross_as_stored!` writes its `Var` and `Canonical` impls.

**Rejected.**
- `Default` and `DerefMut` on every `Erased` — safe code could make an
  `Erased<R, String>` holding the default value, or write another value
  into one, and `as_ref` then reads a box that does not exist.
- A trait impl on `Erased<R, Never>` alone for `Owned`'s construction and
  mutable access — an impl whose existence depends on `T`, against rule 1.
- `Within` for `Erased` wherever `T` is `Within` at the same lifetime — it
  admits an `Erased` of a carrier at the call's lifetime, which no read can
  use (`as_ref` asks `Stored`). Refusing admits fewer types and closes the
  same path.
- `Within` implemented on `Never` — an impl on its alias conflicts (E0119)
  with every other crate's `Within` impl. `Never` is admitted through
  `fn() -> !` instead, a sealed trait's impl disjoint from the one at
  `fn() -> T` for a `T` that is `Within` at every lifetime.
- Refusing, in the derive, a type variable named inside `Erased` in a
  payload unless it holds no carrier — it reads names, which RFC-0059
  withdrew `returns_slice` for.
- Requiring every extension type's type variable to hold no carrier — it
  refuses `Refs<'a, HashSet<'a, …>>` (`as_iter_set`), and `Refs` would have
  to be reshaped.
- The language's `!` as the `Never` alias — each trait the language's `!`
  needs, written on the alias, conflicts (E0119) with every other crate's
  impl of that trait.
- Bounding the runtime's `erase` and reads on `T: Canonical<Canon = T>` —
  a derived payload is a foreign or user type, and the orphan rule keeps
  acvus-extern from implementing its trait for one.
- Refusing every projection in a payload — it refuses `Pipe`, whose
  projection is through a `Chosen` parameter.
- Proving the payload in the generated impls' own `where` clauses — a bound
  the struct puts on a uniform parameter, whose trait declares an
  associated type `UniformPayload`, lets an alias hide a projection the
  proof then accepts.
- Proving it with each uniform parameter bounded by `Var<kind::Type>`
  alone — no user trait reaches the parameter, but a payload of a type that
  bounds its parameter by more, as each iterator stage's body does, is then
  refused.

## RFC-0077: A converted `&place` argument is taken out of its slot for the call

Status: Accepted

A conversion decision (RFC-0041) consumes the value it converts. At a
`&place` argument that is the place's value: taken out of its slot and cast
into a temporary of the referent type the parameter names, which the
call borrows; after the call the temporary is cast back and assigned to the
place, for `&` and `&mut` alike, so a slot never holds a value of a type
other than its own. The place is one the body owns directly — a local, an
input or a context, or a field of one; a place reached through a reference
or an element, and a reference that is not a borrow of a place, are errors
naming both types. A borrowed value that is no place is cast into the
temporary and not cast back, since nothing reads it after the call. The MIR
marks the take-out's `Take` and the `Assign` that restores it, and the move
rule (RFC-0029 rule 2) reads the marks: from the take-out to its restore
the place is moved, a word's as any other's. A read of it is a use of a
moved place; a store into a place overlapping it is refused at the store
and revives nothing; the restore is the one store that gives it a value
again. Inside the call's later arguments, and every call among them, a shared lend of the
place through the same cast lends the same temporary, so the place is cast
once and restored once, after the call that took it out; a lend at another
type or through `&mut` lends the place itself and is refused as a use of
it. A take while a reference is live is a borrow error.

**Why.** Taking a converted place out of its slot keeps every slot at its
own type. The exclusion for the call is the move rule, with the take-out
marked where the lowering emits it, so RFC-0018's rules keep one
implementation, the MIR's, and it holds whichever path settled the
conversion; a store is refused rather than let revive the place because the
restore would overwrite it. Shared lends at one type share the temporary
where the one cast is emitted, in the lowering. Storage the body owns is the
one kind of place a value can be taken out of and put back into.

**Rejected.**
- Holding a converted place at the parameter's type for the call — the
  cast value stored back into the place's slot, every later lend inside the
  call lending the held type, and the place restored after it. The slot
  held a value of another type than its own, which the MIR validator
  refuses, and a held argument settled after the solve recorded no hold, so
  a later argument met the place at the held type on the known path and at
  its own on the held one.
- An unmarked move of the converted place, excluded by the move check as
  any move is — a store revives a moved place, so a write inside the call
  was admitted and overwritten by the restore, and a word's copy is no
  move, so a word place was not excluded at all.
- A loan for the call kept by the checker over the names each argument
  uses — a second implementation of RFC-0018 rules 7 and 8, over the AST,
  beside the MIR's.

## RFC-0080: A fact unsafe code relies on is held by a type or asserted with `unsafe`

Status: Accepted

1. **One form of guarantee.** A fact that unsafe code relies on is a bound,
   a sealed trait, a private field or a private constructor, or else an
   `unsafe` token where it is asserted. A debug assert or a doc sentence is
   neither.
2. **The surfaces it seals.** `Inline` and `Monomorphize` are sealed to
   their lists. `CallSite`'s fields are private, and its one constructor of
   requirement words is `unsafe`, `prepare` being its caller. `Ctx`'s frame
   is private, and `Ctx::new`, `Ctx::frame_mut` and `Runtime::ctx_of` are
   `unsafe`, since two `Ctx`s in safe hands swap frames. A handler that
   needs cells of its own calls `Closure::call_rooted`. `Owned::from_value`
   and `Owned::value_mut` are `unsafe`: the runtime's word is `Copy`, so a
   word read out of a live holder and made an `Owned`, or written into
   one, is released twice, and one kept past the call it was lent to
   comes back holding a loan that ended. Both also take the runtime's
   `Holding`, as `Owned::erased` takes the glue's `Crossing` (RFC-0068
   rule 1), and the space's `Decode` and `Visit` hand over and lend `Owned`
   holders, not words. A trait whose methods receive a `Crossing` or a
   `Holding` (`Cross`, `OneValue`, `Passed`, `CallArgs`, `Takes`, `Gives`,
   `Parameters`, `IntoRun`, `CrossesRest`, the three `Restore*`, `Stored`,
   `InPlaceElement`, and a signature's `Returned`) is
   `unsafe` to implement, since the capability crosses any word at any
   type: its `# Safety` is that the word an impl hands back or reads is
   the value of `Self` at the type the checker settled, that it crosses
   nothing else with the capability, and that it keeps none. The derives,
   `#[extern_fn]`, `extern_signature!` and the library's macros emit
   `unsafe impl` with the `SAFETY` that discharges it, so their users
   write no `unsafe`; a crossing written by hand says `unsafe impl`.
3. **A keep is refused by Rust.** A type variable is not `'static` and a
   carrier is `Within` the call (RFC-0079 rules 6 to 8), so a handler that
   keeps either past the call does not compile, and no handler asserts it
   keeps none. `Within` and `UniformPayload` state in their `# Safety`
   that no part a region or type parameter reaches sits behind an
   `UnsafeCell`.
4. **Effect declarations are outside it.** `pure`, `idempotent` and
   `commutative` are the extern author's own promise, which the checker
   trusts (RFC-0013). An extension library is written for the users of its
   own language, and its author answers for its soundness.

**Why.** A safe trait or a public field that unsafe code trusts lets safe
code break the trust: an `Inline` impl outside the list, a `Ctx` frame
swapped, an `Instance` forged from a word each reached undefined behaviour
with no `unsafe` in the author's code. One form of guarantee makes each
such fact visible where it is made.
**Cost.** A runtime writes `unsafe` where it builds a `Ctx` or a call site,
and where it makes an `Owned` of a word, and says `unsafe impl` for its own
value's crossing.
**Rejected.**
- `ctx_of` on a runtime-only trait — the glue calls it with `Rt: Runtime`
  alone, so the trait is `Runtime`'s supertrait and a handler reaches it
  through the same bound; `unsafe` is what keeps it from a handler.
- Debug asserts on the trusted facts — they vanish in the build that runs.
- A lent variable asserted with `unsafe(lent(T))` — an assertion nothing
  checks at each of two hundred declarations; a variable that is not
  `'static` lets Rust refuse the keep.
- Refusing a spawn of a loan — the spawn is the optimizer's split, so a
  refusal would differ between optimization levels.

## RFC-0082: An extern states its laws and its postconditions in a closed vocabulary, and a pass reads each

Status: Proposed

1. **A declaration has a reader.** An extern states only what a pass reads.
   A law or a postcondition no pass reads is not written, and a form is
   added to the vocabulary with the pass that reads it.
2. **Laws of a binary extern.** `#[extern_fn(law(associative, commutative,
   identity = e))]` on `f(a: T, b: T) -> T` states that `f` is associative,
   that it is commutative, and that `e` is its identity. Any nonempty subset
   may be stated. `e` written as a literal is a constant, held to `T`; `e`
   written as a path is a registered extern of no argument returning `T`,
   `g` naming it in the declaration's own namespace and `ns::g` in `ns`.
   A law is a declaration's, so each instance of a shared signature states
   its own: `num::min` over an integer width is associative and commutative
   with that width's `MAX` as identity, and over `f64` it states none.
3. **Laws of a storage write.** `#[extern_fn(law(fold(combine = g,
   identity = e)))]` on `f(s: &mut S, x: X)` returning nothing states that
   a run of `f` over `s` equals `g` applied to the states that runs over
   its parts reach, each part started from `e`. `commutative` written
   beside `fold` states that `g` commutes; without it, `g` keeps the parts
   in order. `g` is a registered `g(s: &mut S, part: S)` and `e` a
   registered `e() -> S`, named as rule 2 names an extern.
   `associative` and `identity` beside `fold` are refused: they are a
   binary function's. `#[extern_fn(law(inverse = g))]` on `f(s: &mut S) ->
   Option<X>` states that after `g(s, x)`, `f(s)` gives `Some(x)` and
   leaves `s` as before `g`, and that `g(s, x)` after `f(s)` gave `Some(x)`
   leaves `s` as before `f`: the two are one cell, which RFC-0089 rule 4
   promotes across a loop.
4. **Postconditions.** `#[extern_fn(ensures(t1 rel t2, ..))]` relates two
   terms by `=`, `≤` or `<`. A term is RFC-0066 rule 3's: a constant, a
   parameter, the result `ret`, `len(x)` of a parameter or of `ret`, and
   `+`, `−`, `×` and `max` of terms. `len(x)` is the element count of a
   slice or a container; a parameter or `ret` read as a number is an
   integer. There is no quantifier, no condition and no function of the
   author's. Rust's lexer refuses `≤`, `−` and `×` before a macro reads
   them, so a declaration writes `<=`, `-` and `*`.
5. **Both are the author's promise.** The checker and every pass trust a
   law and a postcondition as they trust an effect (RFC-0080 rule 3), and
   an extern that breaks one answers for what a pass does with it. A debug
   build of the extension evaluates each postcondition at the function's
   return, inside its own body, and a relation that fails panics there
   with the relation and both sides' values; a law is sampled only by
   tests. `#[extern_fn]` refuses a law on a signature it is not stated
   over, a word outside the vocabulary, and a term naming no parameter; a
   `len` of something neither a slice nor a container fails the bound the
   evaluation names. Combining the registries refuses an identity or a
   combine that names no registered extern or one of the wrong type.
6. **The readers.** `analysis::carried` reads a law through a call's
   callee. A header parameter `p` whose back edges send `f(p, x)` for an
   associative `f`, or `f(x, p)` when `f` also commutes, where the body
   reads `p` only as that operand, is a `Merge` (RFC-0066 rule 5) on `f`,
   exact. A storage `s` every write of which in the loop is a call of one
   instance of an extern with a `fold` law, lending `s` through its first
   argument and no other, and which the loop reads only to lend it to those
   calls, is a merge through storage: it does not make the loop strong,
   and it is carried state. `analysis::interval` reads a postcondition
   through a call's callee (RFC-0047 rule 7). The merge names the extern
   instance, and its identity and `combine` are for the split of RFC-0066
   rule 10,
   which is not built; until it is, no pass reads them, and rule 1 holds
   them only by that reader to come.

**Why.** RFC-0066 rule 6 leaves what merge a storage write is to the
extern, and `min`, `max`, `&&` and `||` reach MIR as calls whose laws no
pass can see. The author of an extension is the one who knows them, as
with effects. A vocabulary sized to what a pass reads keeps every
declaration meaningful and keeps the promise small enough to state.
**Cost.** A second kind of trusted promise beside effects, and a debug
evaluation of postconditions at every return.
**Rejected.**
- Quantifiers, conditional relations and functions of the author's — the
  language becomes a proof assistant, and no pass reads what it states.
- Preconditions a checker proves at each call — a refinement type system,
  far beyond what any pass needs.
- Laws inferred from the handler's body — the body is Rust, opaque to the
  checker.

## RFC-0090: A host reads and writes values through the Rust types it declares, and never names a runtime value

Status: Proposed

A host runs programs, reads their results, and reads and writes the
contexts that persist between runs. A host is an extern turned around. The
entry's result crosses from the script into Rust, as an extern's argument
does. A value the host puts into a context crosses into the script, as an
extern's return does. Each crossing is the glue's, at the type the checker
settled, as the rule at the top of `acvus-extern` holds for an extern.

1. **A context's type is the graph's.**
   - The host does not declare a context's type, and neither does data the
     host reads.
   - A context enters the graph as a variable (RFC-0025). Its type is solved
     with the rest of the graph from every body that stores or reads it, and
     structural types meet as they do anywhere else.
   - A store is always admitted where the solved type holds it. Initializing
     a context is a script that stores it (`@log = [];`), compiled into the
     same graph as the scripts that read it.
   - A context whose type the graph leaves open closes to `!` at the freeze
     (RFC-0038). A `Vec<!>` holds nothing, and that is sound.
   - A value no script names is not a context. The host keeps it itself.

2. **The entry's inputs and result are declared by Rust types.**
   - A compilation takes the entry's inputs as a Rust type `I`: a derived
     struct whose fields name the `$` inputs the entry reads (RFC-0071 rule
     4) and give their types, in the order the entry takes them, or `()` for
     none (RFC-0054 rule 6). A `$` the entry reads that `I` does not name is
     refused at compile, and so is a field a binding already fixes. A run
     takes an `I`, and it crosses as an extern's returned value does.
   - A compilation takes the entry's return type as a Rust type `R`. Its
     `Ty` is read by the derive an extern's parameter uses, and it is the
     entry's declaration (RFC-0054 rule 1). The declaration and `R` cannot
     differ.
   - A script that only stores returns `Unit`.
   - A host that names no type declares `!` (RFC-0054 rule 5), and rule 6
     covers it.

3. **A host reads and edits as a handler does.** The host lends a value to a
   closure, and the closure's parameters cross exactly as an extern
   handler's do, through the glue the macro emits for a handler parameter:
   `&T`, `&mut T`, `&str`, a slice of `Erased<Rt, T>`, a derive's
   projection, and a `Ctx` that carries the runtime, written first as a
   handler writes it (RFC-0023 rule 2). There is no second
   crossing and no host-only view.
   - Running the entry gives an `Output<R>`. It owns the value, releases it
     when dropped, and offers `with(|p| …)` and `with_mut(|p| …)`.
   - A page (RFC-0033) is built from a compilation's solved context types.
     It offers `with(key, |p| …)` and `with_mut(key, |p| …)`, and
     `insert::<T>(key, value: T)`, which moves `value` in through the glue
     an extern's return uses.
   - A type a handler parameter cannot take, a host cannot take either, and
     the gap is closed on the extern side.

4. **The type is checked before the closure runs.**
   - A page compares the closure parameter's acvus type with the solved
     type of `key`, and `Output` compares it with `R`. A mismatch, a key the
     graph does not have, or a key that holds no value yet is an error
     before any value is touched.
   - The comparison is of types the compilation settled. It reads no tag on
     the value, which an untagged runtime does not have.
   - The contexts a run wrote are read the same way, after the run.
   - A stored value whose type differs from the solved one, as after a
     script changed, is a mismatch when the page opens. What to do with it
     is the host's.

5. **A lent value lives for the closure.**
   - The borrow is the closure's, as a handler's is the call's
     (`Within<'s>`, RFC-0079 rule 6), and `with_mut` holds the value
     exclusively. Nothing the closure is lent outlives it.
   - A value the host keeps is a copy the host makes in Rust inside the
     closure, as `&str` to `String`.
   - No lent value holds a loan into a run: the checker refuses an entry
     result or a context write that may hold one (RFC-0079 rule 9).

6. **The host's surface names no runtime value.**
   - The surface is the entry declaration, `Output`, the page's lending
     methods and `insert`.
   - The value word, the construction of an `Owned`, the accessors that
     read a word at a kind, and a run's raw writes belong to the runtime and
     the glue. Raw writes are reached only under the runtime's `tooling`
     feature, which a host turns on only by naming it.
   - A storage behind a page (`RuntimeContext`) moves whole holders and
     never reads inside one.
   - The CLI, which declares `!` and prints whatever a file returns, is the
     runtime's own tooling. It reads by the settled `Ty` inside the
     workspace. No public reader by `Ty` is offered.

**Why.** The burden falls on the language's developers first, then on the
authors of externs and hosts, who take care but meet no trap. The script's
user takes on nothing. A host that reads the value word writes, again, the
walk only the runtime can check, and a reinterpretation at a wrong type is a
transmute. A context's type written in data is a second statement of what
the scripts already say, and two statements can disagree. Solved in the
graph, the type has one source, and the stores that initialize it are
checked like every other store. An extern handler already crosses the
boundary soundly with values lent and not kept, and a host that lends into a
closure needs nothing more, so one crossing serves both and a gap in one is a
gap in the other. Anything the host keeps is copied in Rust, where Rust
checks it.
**Cost.**
- A host names a Rust type for its entry, and for each context it reads or
  inserts.
- A host can take what a handler parameter can take, and no more.
- A page compares two types on each call.

**Rejected.**
- A context type declared by the host or by data such as a manifest — a
  second source that can disagree with the scripts.
- An initializer that returns the context's value — its result type would be
  inferred from its own body with nothing to refuse it (RFC-0054). A store
  is checked against the type the whole graph solves.
- `as_typed::<T>()` on a result value — it is a `Value -> T`, and it checks
  a kind after the run (RFC-0054).
- Returning the value word with documented accessors — every host rewrites
  an unchecked walk, which is a trap at the host's tier.
- A public reader over a `(Ty, value)` pair — every host that used it would
  rebuild a type system on the runtime's layout.
