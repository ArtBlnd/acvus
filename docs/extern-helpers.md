# The extern framework's helpers

An **ExternFn** is a Rust function declared once and reached from a script:
`#[extern_fn]` on the Rust body, a registry that carries the declaration to
the compiler and the handler to the runtime, and a crossing at the boundary
between them. The checker settles every type at a call site before the
handler runs; the machine hands the handler a run of the runtime's values
and a place to write the result. Everything in `acvus-extern` exists to
carry one of those three things: what a value becomes at the boundary, what
runs, and what the compiler was told.

The principle the framework is built on is RFC-0067's: **the machine holds
no generics.** Every type at a call is ground. An instance of a shared
signature is a value — the entry pointer of one concrete handler. No
function type crosses the boundary as a type-level list, and there is no
position in the machine where a type is computed. A Rust generic parameter
of a declaration is therefore not a Rust type variable that survives the
call: it is a *variable of a kind* (`Var<K>`), filled by a stand-in while
the declaration's polymorphic type is built and by whatever the runtime
carries a value in at run time.

The helpers sit on seven axes.

- **crossing** — what a Rust value becomes when it passes the boundary, and
  what the runtime holds it as.
- **borrowing** — what a reference, a view or a slice is at the boundary:
  the strength of a borrow, where a projection reaches, and the carriers
  that hold a borrow rather than a value.
- **calling** — what runs: the handler, its factory and glue, the task it
  runs at, and the closure a body calls back into.
- **typing** — what a declaration's type is: the four kinds, their
  variables and terms, and the Rust types that name acvus types.
- **registry** — what is declared and to whom: manifests, signatures,
  instances, the combination of every registry, and the space hooks a
  declared type carries.
- **forms** — how wide a crossing is and which call entry that width picks.
- **sites** — what one call site contributes: the settled types a crossing
  reads once, when the handler is placed at that site.

## What this page counts

The rows are the `pub use` list of `acvus-extern/src/lib.rs` at `0d308c5e`,
lines 30–66: **135 names, 73 atoms and 62 derived.**

| axis | rows | atoms | derived |
|---|---|---|---|
| crossing | 19 | 14 | 5 |
| borrowing | 22 | 10 | 12 |
| calling | 34 | 14 | 20 |
| typing | 16 | 9 | 7 |
| registry | 26 | 19 | 7 |
| forms | 11 | 5 | 6 |
| sites | 7 | 2 | 5 |
| **total** | **135** | **73** | **62** |

An **atom** names a fact only it carries. A **derived** row is reached from
another row in the same list and keeps a name for a reason the row states.
No row is kept for compatibility.

Three things in `lib.rs` are not rows, and each is left out for a stated
reason.

- Lines 68–76 re-export 28 names from `acvus-mir`, `acvus-utils`, `futures`
  and `rustc_hash` (`Ty`, `PolyTy`, `Interner`, `QualifiedRef`, `Task`,
  `BoxFuture`, …). They are the compiler's and the host's vocabulary passed
  through so that a declaration needs one dependency; none of them is a
  helper of this framework.
- `pub mod core` holds the shared signatures the compiler always has —
  `clone`, `eq`, `hash` (RFC-0019) and `as_str`, the machine name the slice
  coercion resolves (RFC-0062) — and the registry that carries them. They
  are declarations, not helpers.
- `pub mod derive` holds the library half of what `acvus-extern-macro`
  expands to. Its own decision is that nothing there is exported from the
  root: each item has one caller, which names it by path, and at the root
  the same name would read as a step a hand-written crossing is invited to
  take. `derive::transparent::Transparent` is the one item the root
  re-exports, because a hand-written extension type names it.

Two groups sit on an axis by a placement rule rather than by obvious fit.
Ownership at the boundary (`Owned`, `Release`, `lend_run`) is on
**crossing**: the ABI passes `Rt::Value` and owes nothing, and the
conversion to the holder that owes a release happens at the glue's crossing
and nowhere else (RFC-0048 §1). The journaling surface (`Journaled` and its
seven neighbours) is on **registry**: a space never sees a type, it sees
the hooks a registry contributed for it (`Contribution::space`,
`ExternTypeDecl::space`, RFC-0033).

## Axis 1 — crossing

`obj.rs`, `owned.rs`, `erased.rs`, `derive/transparent.rs`, `handler.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `Cross` | what a type occupies at the boundary: a run of the runtime's values | atom — the storage is `Rt::Value`, never a Rust `T`, and the width is the run's |
| `OneValue` | a crossing that is exactly one of the runtime's values | atom — `erase`/`materialize` are one value's, and `deref` reads a `Self` through a reference, which only a type stored as itself can answer |
| `Uniform` | the representation every slot takes unless a member says otherwise (RFC-0040) | atom — the tag that picks the crossing |
| `Specialized` | the representation of a `Monomorphize` member's slot | atom — the other representation; the run is the value at every impl |
| `Stored` | a type the runtime keeps as itself | atom — the runtime's box holds the Rust value, which is what makes `value_as_ref::<T>` sound |
| `TransparentOver` | a name for the runtime's value with its layout | atom — the unsafe promise that a run of values is a run of `Self` in place |
| `Inline` | a stored type that lives in the value word | atom — read with no runtime in hand |
| `FromValue` | the `Value -> Self` step a body takes outside the glue | atom — the recursion through a container, ending in a checked materialize |
| `Owned` | the one holder that owes a release | atom — every Rust store that owns a runtime value is one of these |
| `Release` | what a value owes when its holder drops | atom — the obligation itself |
| `Transparent` | an extension type stored as its payload | atom — it licenses the `repr(transparent)` pointer cast, and only the derive implements it |
| `Obj` | an object as the runtime holds it: one shared shape and a flat run of field values | atom — rules 4 and 8's layout |
| `ObjectShape` | an object type's field names in rule 8's order, shared by every object of the type | atom — the order the interpreter lays the same object in |
| `Variant` | a variant as the runtime holds it: a tag register and a payload register | atom — unlike an object it carries no shape, because a tag word carries the name it stands for |
| `FieldAt` | which of a type's fields, in rule 8's order | derived from `ObjectShape` — named so that a field position and a frame register stay two spaces of small numbers |
| `Erased` | a runtime value with the Rust type it was erased from remembered | derived from `Stored` — named because it reads and edits that type in place, which a bare value cannot |
| `expect_type` | the refusal when a value was erased from another type | derived from `Runtime::type_of` — the panic names both types |
| `materialize_checked` | a materialize that checks first | derived from `expect_type` + `Runtime::materialize` |
| `lend_run` | the runtime's values behind a run of holders | derived from `Owned`'s `repr(transparent)` — for a caller filling a destination it owns (RFC-0050 rule 6) |

## Axis 2 — borrowing

`loan.rs`, `reference.rs`, `slice.rs`, `str.rs`, `projection.rs`,
`handler.rs`, `obj.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `Loan` | the strength of a borrow | atom — it carries the Rust borrow, the runtime calls that open it, and the acvus mutability, which is what lets one body serve both strengths |
| `Shared` | a shared borrow | derived from `Loan` — one of its two inhabitants, and what a declaration writes |
| `Mut` | an exclusive borrow | derived from `Loan` — the other, and the only one the checker admits a second live name of nothing else |
| `Ref` | the acvus `&T` / `&mut T`, carried as the reference value itself | atom — for a body that keeps the reference, returns it, or takes it into a lambda; its region is the caller's |
| `Slice` | the acvus `&[T]` / `&mut [T]`, the one thing the machine indexes | atom — it crosses as a register pair and is no value of the language |
| `Words` | a run as the machine holds it: a pointer and a length in two registers | atom — the spelling only the runtime writes into a value |
| `Elements` | a run of the runtime's values in a storage | derived from `Words` — the same pair at the value type, constructible only from a Rust slice, so a stray pointer and length cannot be named |
| `StrView` | the language's `&str`: a view of a `String`'s UTF-8 bytes | derived from `Words` — the same pair over bytes rather than values; the encoding is the checker's obligation, which is why it has its own name |
| `Borrowable` | a type whose values are places the language names | atom — the missing impl is the refusal, and its message is the whole of it |
| `BorrowableSpecialized` | the same at a `Monomorphize` member | derived from `Borrowable` — a second trait because a member's storage holds a `Self` only where its specialized crossing wrote one |
| `BorrowedWhole` | the refusal of `&S` on an aggregate | atom — a trait with no impl anywhere; the diagnostic text is what it does |
| `Borrowed` | the two projection types an aggregate has | atom — named without a runtime, because a projection struct has a lifetime and no runtime parameter |
| `Project` | a field's borrow, built over the one value a field occupies | atom — the obligation that a field is one value is the interpreter's |
| `Projected` | the projection a handler's signature names, at the call's own lifetime | atom — the derive writes one impl per strength, so the choice is the projection's own |
| `Reach` | where a projection reaches the aggregate it borrows | atom — a lent reference and a nested value are opened by two different runtime calls |
| `Lent` | through the reference the caller handed in | derived from `Reach` — one of its two inhabitants |
| `Nested` | in the value a field or a payload holds | derived from `Reach` — the other |
| `Fields` | an object's fields by position, at a strength | derived from `Obj` + `Loan` — named because the exclusive form must take its positions disjointly |
| `ObjectAt` | where each field a projection names sits at this site, with each field's own site | derived from `object_fields_at` — the table a glue holds and every call reads |
| `VariantAt` | the tag word of each variant a projection names, with each payload's site | derived from `variant_tags_at` — the same table at an enum |
| `object` | the object a projection borrows | derived from `Reach::at` at `Obj<Owned<Rt>>` |
| `variant` | the variant a projection borrows | derived from `Reach::at` at `Variant<Owned<Rt>>` |

## Axis 3 — calling

`runtime.rs`, `handler.rs`, `func.rs`, `str.rs`, `projection.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `Runtime` | the contract a host signs to run every declared ExternFn | atom — the host owns its value representation, its frame and its call entries, and nothing here names a runtime but this |
| `TypesOnly` | a host that holds no values | derived from `Runtime` — for registering declarations where nothing will run |
| `Handler` | a declaration compiled to a Rust body, with the crossing on both sides | atom — the call entries the machine reaches a body through, and the one `WIDTH` they all read |
| `HandlerFactory` | one declared instance in the module table, its type erased | atom — the one `dyn` on the path, taken once at preparation |
| `AtSite` | that instance with its site table filled | atom — the only way to reach one, so an operation cannot hold a handler whose table was never filled |
| `Glue` | the Rust closure with its parameter modes, its result and its site table | atom — `Handler` is implemented for the sited glue alone |
| `glue` | the glue's constructor | derived from `Glue` — it exists rather than a `Glue::new` because a closure is inferred higher-ranked only where the `Fn` bound is in scope at its own site |
| `AsyncCall` | a body that hands the runtime a future outliving the call's frame | derived from `Handler` — a separate trait because the future is boxed |
| `AsyncFactory` | its factory | derived from `HandlerFactory` |
| `AsyncAtSite` | its sited form | derived from `AtSite` |
| `AsyncGlue` | its glue, the closure shared rather than owned | derived from `Glue` — the future outlives the call that made it |
| `async_glue` | its constructor | derived from `glue` |
| `ExternHandler` | one handler per rung of `Task`: run now, offloaded, or awaited | atom — the declaration's task is the ceiling of its handler's |
| `DeclaredInstance` | one concrete signature with its handler and the task it admits | atom — the greatest task the instance runs, not the instance's own |
| `Instances` | a declaration's instances: the concrete list and the generic one | derived from `DeclaredInstance` — named because its order *is* the numbering the compiler puts in a call's callee |
| `ValuesOnly` | a handler no crossing of which borrows the caller's frame | derived from `HandlerFactory` + `ValueParameters` — what a task above `Sync` admits |
| `DirectOp` | what a host with no register machine runs a call as | derived from `Handler` — the handler behind a closure that takes the run as it comes |
| `Arg` | how one Rust parameter takes its argument out of the call's argument run | atom — the mode is written in Rust and read by the macro |
| `ByValue` | a parameter taken by value | derived from `Arg` — the mode the macro picks from a Rust parameter written `T` |
| `ByRef` | a parameter taken by reference, at a strength | derived from `Arg` — from `&T` and `&mut T`, one name for both since the loan is a parameter |
| `ByProjection` | a parameter declared as a projection | derived from `Arg` — a projection is a type with a lifetime argument, neither a value nor a borrow of one, so the macro picks it by that shape |
| `ByStr` | a parameter written `&str` | derived from `Arg` — its run is the pair a view occupies where every `ByRef` is one value |
| `Ret` | how a result is written into the destination run the caller lent | atom — the caller's registers or a heap object's body, written the same way |
| `Val` | a result crossing as itself | derived from `Ret` |
| `RetStr` | a result written `&str` | derived from `Ret` — the view's pair rather than an owned value |
| `Parameters` | the run a declaration's parameter list makes, its site table and the tuple the body is handed | derived from `Arg` — the fold over the list |
| `ValueParameters` | a parameter list every member of which is one value | derived from `Parameters` — what a suspending call admits, a pair being a borrow of the frame |
| `IntoRun` | a closure call's arguments written into the callee's parameter registers | atom — the argument side of what `Ret` does for a result |
| `Closure` | the runtime's closure value at the types its declaration names | atom — it holds the value and the one answer to "does a call suspend" |
| `ClosureFn` | calling one | atom — the one place a body crosses back into the runtime |
| `Args` | a closure's parameter tuple, with the erased tuple it has at run time | atom — arity is a tuple, not a family of types |
| `CallArgs` | the same tuple as a call needs it | derived from `Args` — the runtime entry that takes this many arguments |
| `ArgTypes` | the same tuple as the declared type needs it | derived from `Args` — the parameter terms of the closure's acvus type |
| `CallToken` | proof that a call came through `Closure` | atom — only that module mints one, so a handler cannot reach `Runtime::call_*` directly |

## Axis 4 — typing

`ty_arg.rs`, `effect.rs`, `len.rs`, `identity.rs`, `vec.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `Kind` | one of the four kinds a declaration's variables have | atom — it carries the term a variable of that kind takes in a polymorphic type |
| `kind` | the four kinds: type, effect, length, identity | derived from `Kind` — each is uninhabited, because a kind names a class of variables and is never a value |
| `Var` | the bound of a declaration's generic parameter that is a variable of kind `K` | atom — no kind has an impl over an unbounded parameter, so a type the language does not know cannot reach a declaration by satisfying `Send + Sync` |
| `Term` | a Rust type that names a term of a kind | atom — a known level, or a declaration's own variable |
| `Nth` | the `N`-th variable of a kind, as a Rust type | derived from `Kind` + `Term` — the stand-in that fills the parameter while the declaration's type is built; uninhabited, so it names a variable and is never a value |
| `PolyVars` | the variables a declaration ranges over, by kind and position | atom — built once per declaration; how many of each it has is the length of its vector |
| `TyArg` | a Rust type that names an acvus type | atom — it takes the interner and carries `SLOT`, which `Term` takes and carries neither of |
| `SlotRepr` | the representation a specializing slot gives its argument | atom — a composite takes the strongest of its parts |
| `Spec` | the member stand-in in a member instance's type | derived from `TyArg` — its parameter is the member and its `SLOT` is `Member`; that is the whole difference from `Nth<kind::Type, N>` |
| `Monomorphize` | a type variable ranging over a finite set of concrete types | atom — the one specialization the machine keeps, the handler compiled once per member |
| `Never` | the language's `!` | atom — a declaration returning it panics instead of returning, so its call is typed `!` |
| `Pure` | the pure effect level | derived from `Term<kind::Effect>` — a level a declaration names, and one a closure parameter may stand at |
| `Idempotent` | the idempotent effect level | derived from `Term<kind::Effect>` — as `Pure`, at the middle level |
| `Opaque` | the opaque effect level | derived from `Term<kind::Effect>` — as `Pure`, at the top |
| `Arr` | an array whose length is a variable of the length kind | atom — the language's array, holding the elements at run time |
| `vec_ty` | `Vec<elem>` as a settled type | derived from `TyArg for Vec<T>` — for a context declared outside a script, where there is no Rust element type to read it off |

## Axis 5 — registry

`registry.rs`, `space.rs`, `acvus-extern-macro/src/lib.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `Registry` | a contribution not yet built | atom — it is a function of the interner, so names are interned once, where every registry meets |
| `Contribution` | what one registry contributes: a manifest, its instances, its space hooks | atom |
| `Manifest` | the declarations one registry contributes | atom — nothing in it names a runtime |
| `FnDecl` | a function as the compiler sees it | atom — its type, the bound of each type variable, whether it is a cast, and the signature it instantiates |
| `SignatureDecl` | a shared signature: a name, a polymorphic type and no body | atom — a name the compiler resolves to one instance per ground type |
| `SharedSignature` | the marker `extern_signature!` declares | derived from `SignatureDecl` — it has a name so that a declaration names its signature as a Rust path rather than a string |
| `ExternFn` | one declared function with its instances | derived from `FnDecl` + `Instances` — exactly their pair, as `#[extern_fn]` produces it |
| `ExternTypeDecl` | a type a registry declares, and its space hooks | atom — a type without hooks cannot be a context a space holds |
| `MemberType` | a member signature's type at both representations, with the handlers converting between them | atom |
| `family_casts` | the two casts a family type declares at a member | derived from `MemberType` — `erase` from the specialized type to the uniform one and `materialize` back |
| `Externs` | every registry combined: the compiler's functions and types, and the runtime's handlers | atom |
| `Handlers` | the handler table by name | derived from `ExternHandler` — the map the runtime is handed |
| `CombineError` | why two registries do not combine | atom — a duplicate name, an unknown or mismatched signature, a cast that is not one, a handler above its declaration's task |
| `Journaled` | an extension type a space can hold: its bytes, its ops, its replay, its children | atom — the type never sees the space |
| `SpaceHooks` | that same contract over the runtime's erased value | derived from `Journaled` — `SpaceHooks::of::<J>` is the whole of it |
| `NodeHash` | a content address: the hash of a node's canonical bytes | atom |
| `SpaceError` | why a space operation failed | atom |
| `SpaceResult` | its result | derived from `SpaceError` |
| `Encode` | writing one element as canonical bytes | atom — the space provides it, and it turns a nested journaled value into its head hash, which the type cannot do |
| `Decode` | reading one element back | atom — the space's half again, in the other direction |
| `Visit` | visiting a nested value the space commits before its parent | atom |
| `extern_fn` | a Rust function declares an ExternFn | atom — `name`, `instance_of`, `effect`, `commutative`, `heavy`, `sync` |
| `ExternType` | a `repr(transparent)` Rust struct declares an extension type | atom — the helper attribute `extern_type` takes `name`, `ns`, `payload_per_instantiation` |
| `TyArg` (derive) | a Rust struct declares an object type, a Rust enum the language's enum of the same name | atom — the helper attribute `projection` is a bare presence test and adds the borrow types beside the aggregate |
| `extern_signature` | a shared signature with no body | atom — it declares the marker type a declaration's `instance_of` names |
| `extern_registry` | the items one registry contributes | derived from `Contribution` + `Manifest` — the list written out, so the map types stay in the library |

## Axis 6 — forms

`obj.rs`, `handler.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `Form` | the run of the runtime's values a crossing occupies, as a type | atom — the width is read off the type, which is the only place a width is a number |
| `FormKind` | which of the three landing rules a width takes: a value, a view, an aggregate's components | derived from `Form` — it exists because two forms share a width and land by different rules, and this is what a caller matches on |
| `One` | one of the runtime's values | derived from `Form` — the run of every crossing but a view and an aggregate's components |
| `Pair` | the two registers a slice or a view occupies | derived from `Form` — the borrow of the caller's frame, which is why a suspending call admits none |
| `Run` | an aggregate's `W` components, written where the caller placed its destination | derived from `Form` — its fold over `W` register steps is deliberately unwritten: no `Arg` impl names it, because a by-value aggregate crosses as the one value rule 4 realizes it into |
| `Width` | how many values a call's arguments and its result occupy, and which rule the result lands by | atom — summed once in the `Handler` impl, so `prepare` reads the answer rather than counting |
| `REGISTER_FORM` | where the register forms stop: four of the runtime's values | atom — a measured number, four being where the handlers run out, not where the operations do |
| `ArgRun` | an argument run, and the run one more parameter makes of it | atom — the type-level fold that counts values and not parameters |
| `InRegisters` | an argument run of `N` values, one per register | derived from `ArgRun` — one of the two shapes it answers |
| `InWindow` | an argument run the register forms do not cover, lent the window it sits in | derived from `ArgRun` — the other |
| `TakenForm` | which `Rt::op_*` and which `Rt::fused_*` a run of this shape and a result of this form name | atom — a trait impl and not a branch, because a branch drags every operation into the monomorphization whichever way the constant goes |

## Axis 7 — sites

`handler.rs`, `projection.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `ArgAt` | one argument of one call site as the checker settled it | atom — the type and the interner that resolves the names in it |
| `Sited` | what a parameter's crossing needs from the call site | atom — a lifetime-free trait, so a site table is one type per parameter rather than one per lifetime the parameter is read at |
| `Unsited` | the site table of a glue that is at no site | derived from `Sited` — a distinct type rather than `()`, so that `Handler` cannot be implemented for a glue whose table was never filled |
| `SitesNoParameterReads` | a call site of `n` arguments typed `Unit` | derived from `ArgAt` — for a test whose parameters read no site; a projection parameter reaching one panics |
| `object_fields_at` | where each field a projection names sits in the object the caller lent | derived from `ObjectShape` + `ArgAt` — the positions are rule 8's, not the projection's own field order, since a projection may name a subset |
| `variant_tags_at` | the tag word of each variant a projection names, beside that payload's settled type | derived from `ArgAt` — the type is read so a payload's own crossing has a site, and so a wrongly typed argument is refused at preparation |
| `payload_at` | the payload a projection borrows | derived from `variant_tags_at` — the refusal when the variant carries none at this site |

## Axis 8 — requiring

`instance.rs`, `handler.rs`, `registry.rs`.

| name | the acvus concept it stands for | atom, or derived from |
|---|---|---|
| `Carrier` | a bounded type variable's filling: the value one argument passed, with the entry of each instance the declaration required beside it | atom — `#[extern_fn]` writes one per bounded variable, and its `entries` and the `FnDecl::requires` list are the two halves of one order; built at each call and never stored |
| `Signature` | a shared signature as a Rust caller of one of its instances sees it: the shape of a call | atom — `extern_signature!` writes the impl, so a requiring handler restates no mode and no width |
| `Signature::Recv` | how the first parameter takes its carrier: `&This`, `&mut This`, or `This` | derived from the declaration's mode — the mode reaches a requiring handler here and nowhere else, so the handler's own `I::call` is where a wrong one is refused |
| `Signature::as_this` | the carrier a receiver in any of the three modes stands at | derived from `Recv` — the entry lies in the carrier, and an opaque `Recv` is the only handle a generic caller holds |
| `one_value` | one of the runtime's values holding what a parameter crosses as | derived from `Cross<Form = One>` — the argument run an entry is called with is built out of these |
| `InstanceOf` | a type with an instance of a shared signature, called where it stands | atom — the handle the deleted marker bound `HasInstance<Sig>` never had |
| `InstanceOfAsync` | the same predicate at the async task | derived from `InstanceOf` — it drives a node of either form, awaiting the `Await` arm and handing the `Sync` arm back as a ready future, and there is no trait the other way |
| `EntryFn` | a resolved instance's handler as a plain function | derived from `Handler::call` — without the `&self`, and with the window lent rather than moved |
| `AsyncEntryFn` | the same for an instance whose body is an `async fn` | derived from `AsyncGlue`'s closure — the values ABI's run, and the future the body is; `AsyncCall`'s own `'static` form is a claim an entry cannot keep, its receiver being a reference into the calling handler's storage |
| `EntryRun` | the task an instance's body runs at, as the function that runs it | derived from `Task` — one node kind with two arms, because which arm a node has is the registry's answer at the ground type and the site's word is untyped |
| `EntryNode` | one instance as a site resolved it: its `EntryRun`, and the entry of each of its own bounds in `requires` order | atom — the structural recursion of Decision 2, as a value the program owns |
| `Entry` | a resolved instance, as the one word `Kind::Entry` carries | derived from `EntryNode` — the address of one, so a run carries a whole tree in one value |
| `Bounds` | the entries one declaration's bounds resolved to | derived from `Entry` — a node's children, read by position in `requires` order |
| `NodeArena` | where the nodes a site resolved live | atom — owned by `InstanceTable` and held by every site table that read one, which is what makes an `Entry` an address worth reading |
| `Held` | a bounded variable's value where a value keeps it | derived from `Owned` — the pointer-free form an `ExternType` payload holds, so the handler that wrote it and the instance that reads it back name one Rust type |
| `HeldMut` | a carrier built over a `Held` for one call | derived from `Held` + `Carrier` — a `&mut` receiver may write its own value word, and the guard puts it back where it came from; the carrier is a `ManuallyDrop`, so a guard with no carrier is not a state the type has |
| `Bound` | the entries a `Held` is read at | derived from `Bounds` — what `#[extern_fn]` appends to a declaration whose parameter holds a bounded variable rather than taking it |
| `AtEntry` | a declaration's entry as a type | derived from `EntryFn` — named where the glue's type is named, so the glue itself stays the closure |
| `SitesAtEntry` | a parameter list whose site table a call through an entry can rebuild | derived from `Parameters` — the glue over such a list can be an entry, because its site holds nothing the site alone knew |
| `SitedAtEntry` | one parameter of such a list | derived from `Sited` — the missing impl is the refusal: a parameter resolved at the argument's settled type is reachable from a call site and not from an entry |
| `InstanceEntries` | where a site table finds the instance a bound requires | atom — keyed by signature and by the ground type an instance stands at |
| `NoInstances` | the registry of a site built where no declaration requires an instance | derived from `InstanceEntries` — named explicitly at every such site, so no default hides one that should have had a registry |
| `InstanceTable` | every shared signature's instances, as `Externs::combine` collected them | derived from `InstanceAt` — the registry half of `InstanceEntries` |
| `InstanceAt` | one instance as a site table needs it: the pattern it stands at, the plain function that runs it, and what its own bounds require | atom — the entry is absent where the instance has none |
| `BoundAt` | one bound of an instance's own declaration | `acvus_mir::ty::InnerBound` under this crate's name — the checker decides the same recursion the site table walks, so the two read one type |
| `Requirement` | one `InstanceOf<sig::S<..>>` or `InstanceOfAsync<sig::S<..>>` bound of a declaration | atom — which variable carries it, which signature it names, and the highest task an instance it reaches may run at; the order is the order of the carrier's entries |
| `ByBound` | a parameter whose type is a bounded variable's carrier | derived from `Carrier` — the site holds the entries, so the argument run is no wider than it would be without the bound |
| `AtBound` | a parameter whose type *holds* a bounded variable's value | derived from `ByBound` — the parameter crosses as it would without the bound, and the site resolves this instance's own entry beside it |
| `AtBounds` | what a site resolved for one such parameter | derived from `Bounds` — with the arena where the entries came from the site table, without it where they came through the run |
| `SiteAtBound` | one parameter's site table where the parameter holds a bounded variable | derived from `Sited::Site` + `AtBounds` — exactly their pair |
| `ArgAtBound` | what the body is handed for such a parameter | derived from `Arg::Out` + `Bound` — exactly their pair |

## What the pass folded

A reader of an older RFC or of `scratchpad/type-helpers/inventory.md` meets
names that are gone. This is the map; it is the one place they appear.

Batch A — a variable has a kind (`69cbd6e7`):

- `TyVar`, `EffectVar`, `LenVar`, `IdentityVar` → `Var<kind::Type>`,
  `Var<kind::Effect>`, `Var<kind::Length>`, `Var<kind::Identity>`.
- `EffectArg`, `LenArg`, `IdentityArg` → `Term<K>`.
- `Typeck<N>`, `Eff<K>`, `Len<K>`, `Idn<K>` → `Nth<K, N>`.
- `VarCounts` → `PolyVars`; its four counts are `PolyVars::fresh`'s
  arguments.
- `HasInstance<Sig>` → nothing. A required instance is a function pointer
  the site fills (RFC-0067), not a marker on a type parameter.
- `Instance<S, Rt>` → `InstanceOf<S, Rt>`, with `InstanceOfAsync<S, Rt>`
  beside it. The name `Instance` is reserved for the parameter form a
  container's element will take.

Batch B — a crossing is one trait (`3c2b4dec`):

- `CrossSpecialized` → `OneValue<Rt, Specialized>`.
- `Returned` → `Cross::ReturnForm`, and `returned_as_crossed!` with it.
- `downcast` → `materialize_checked`.
- `one_from_run`, `one_into_run` → the `OneValue::from_run` and
  `OneValue::into_run` defaults.
- `VARIANT_WIDTH` → `Variant::WIDTH`.
- `erase_field`, `materialize_field`, `take_payload`,
  `materialize_payload`, and the `transparent`/`object`/`variant` function
  groups → `acvus_extern::derive::*`, off the crate root.

Batch C — a borrow has one shape (`0d308c5e`):

- `RefMut<T, Rt>`, `SliceMut<T, Rt>`, `ByRefMut<T, C>`, `FieldsMut<'a, Rt>`
  → `Ref<T, Mut, Rt>`, `Slice<T, Mut, Rt>`, `ByRef<T, Mut, C>`,
  `Fields<'a, Mut, Rt>`; the shared twins take `Shared` in the same place.
- `object_of`, `object_of_mut`, `object_in`, `object_in_mut`, `variant_of`,
  `variant_of_mut`, `variant_in`, `variant_in_mut` → `object` and `variant`
  over `Reach` (`Lent`, `Nested`) and `Loan`.
- `Fn0`, `Fn1`, `Fn2`, `Fn3` → `Closure<A, R, E, Rt>`, arity being a tuple;
  `Fn1::call_value_now` and `call_value` → `Closure::erased` with
  `ClosureFn`.
- `glue0`…`glue8` and `async_glue0`…`async_glue8` → `glue` and
  `async_glue`.

## What is still open

Facts, each with the line that states it. Every path is relative to the
repository root at `0d308c5e`.

1. **The frame is a lent handle.** A handler names it `&mut Rt::Frame<'_>`,
   which is two references for a host whose handle is itself a reference.
   The one-reference form — making `Frame` the state itself — was built and
   does not compile: the frame below owns the state and keeps it for its
   next call, so every site that hands the window out can only lend it.
   `acvus-extern/src/runtime.rs:25` (the obligation), `:37` (the type).
2. **`Monomorphize`'s members are enumerated here.** The reach of the
   bound is a list in this crate — `i64`, `f64`, `bool`, `u8`, `String`,
   and the runtime's own carrier — with no blanket impl, so a member type
   not named there is refused at the declaration.
   `acvus-extern/src/ty_arg.rs:391` (the trait), `:408`–`:412` (the list),
   `:416` (the carrier).
3. **The sync and async twin traits exist because
   `impl_trait_in_assoc_type` is unstable** on the pinned toolchain.
   Naming an `async` block's type in an associated type needs it; without
   it the future is a `BoxFuture` and its size is not a constant any impl
   can state. `acvus-extern/src/handler.rs:929` (the reason), `:933` (the
   trait).
4. **`ByStr` and `ByProjection` are separate `Arg` impls by parameter
   shape.** The macro picks a parameter's mode from the Rust type's shape,
   and neither of these is a value or a borrow of one: a `&str` is the pair
   a view occupies where every `ByRef` is one value, and a projection is a
   type with a lifetime argument. `acvus-extern/src/str.rs:117`,
   `acvus-extern/src/projection.rs:419`.
5. **`BorrowedWhole` is a trait that exists to be unimplemented.** Its
   absence is the refusal of `&S` on an aggregate, and its
   `on_unimplemented` text is the whole of what it does; the derive names
   it in the `where` clause of the `Borrowable` impl it emits.
   `acvus-extern/src/projection.rs:453` (why), `:468` (the trait).
6. **`Stored` and `FromValue` each carry a fact `OneValue` lacks.**
   `Stored` says the runtime's box holds the Rust value itself, which is
   what makes `Runtime::value_as_ref::<T>` sound and is exactly what
   `OneValue` does not say. `FromValue` is not `materialize_checked` under
   another name: its impl for the runtime's own value is the identity, and
   its impl for `Vec<E>` calls the checked materialize for the buffer and
   then its own element step. `acvus-extern/src/obj.rs:424` and `:428`;
   `:452` and `:460`.

Nothing in the export list is unplaced: every name on it stands for a
concept the language already has a phrase for, and every name is an atom or
is reached from another row in the list.
