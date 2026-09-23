# Required instances

How a Rust handler generic over a type variable calls a shared signature
(RFC-0019) at that variable's type. The checker chooses every instance;
the machine receives the choice as one word and computes no type. This
document decides what that word is, where it lives, how a handler
declares and calls it, which crossings may read a value back at a type,
and which signatures `core` holds.

## RFC-0067: The machine holds no generics; a required instance is one word beside the value

Status: Accepted

Every type at a call is ground. Whatever a type variable ranges over is
settled by the checker and arrives at the machine as a word; no function
type crosses the extern boundary as a type-level list, and no position in
the machine computes a type. The runtime shape is a value with a function
pointer beside it; the checker picks the pointer, the loans analysis
governs the borrow, the effect system says whether the call may suspend.

1. **A requirement is declared by taking it, and it is one value.**
   A handler requires signature `S` at its own type variable `I` by taking a
   parameter `Instance<S, I, Rt, T>`; `#[extern_fn]` refuses an `I` that is not
   one of the declaration's own `Var<kind::Type>` parameters and records the
   requirement on `FnDecl::requires` (its form is RFC-0068 rule 5). `Instance` is
   exactly one of the runtime's values (`ONE_VALUE`). A handler requiring two
   signatures takes two `Instance` parameters: one value per signature, never
   a bundle, so a requirer of `A` and `B` calling a requirer of `A` passes the
   word and converts nothing. An instance is not a stamp in the value's header
   or vtable. What the word addresses is RFC-0070 rule 2.

2. **An instance is matched by the signature's argument.** An
   instance may stand at a pattern type (`next` at `Map<I, U>`). The type an
   instance is matched by is read at the position of the signature's first
   type variable inside the first parameter that reaches it, through
   references, arrays, slices, extension-type arguments and tuple positions;
   a signature naming that variable only under another head is refused at
   `combine`. `combine` meets the required variable's bound with the types
   the signature's instances stand at; which instance a call takes is the
   checker's decision (RFC-0068 rule 5). A shared signature names its call effect
   (`extern_signature! { effect = E, … }`, one of its own effect variables);
   one that names none is `Known(PURE)`.

3. **The word lives in the site table.** `prepare` places the
   chosen word in the call site's table once per site; `Required<S, I, T, N>`
   has `ARGUMENTS = 0` and copies it out. A requirement costs zero ABI words,
   no move before the call, and nothing rebuilt per call.

4. **Rust pairs an instance with its variable.** `I` is the
   requiring handler's own variable, so `call` compiles only with a receiver
   at `I`, and a mispairing is a Rust type error. A value that must reach its
   own `S` later stores the inner value and its `Instance` side by side,
   typed at the same `I` and `E` (RFC-0068 rule 1), laid once at
   construction: a stage `Map<I, F>` holds `inner: I` and `next` at `I`. An
   instance's own requirements are its declaration's, not its value's
   (RFC-0070 rule 1).

5. **`Now` and `Later`.** `Instance::into_async` retypes `Now` to
   `Later`, one way; there is nothing the other way, since a sync caller has
   nowhere to suspend to. `call_await` exists only on `Later`. `Later::call`
   serves a `sync =` twin, the body a `Sync` site runs where the checker
   settled on an instance that returns.

6. **The call.** The receiver does not cross as an argument:
   `Instance::call` names it in `Ctx` and the mono glue reads it back at the
   signature's declared mode (RFC-0070 rule 4). A rest parameter concrete in the
   signature crosses as typed; one at a signature type variable crosses as
   the caller's own value (`&Rt::Value`, `&mut Rt::Value`, `Owned<Rt>` by
   mode), and the glue restores the instance's type from it. Each signature
   has one `Now` and one `Later` `fn` type, and the word is transmuted to that
   type and no other. The indirect call is the last expression of `call`: no
   write-back, drop or marshalling follows, so the caller's tail position is
   the glue's.

7. **`Ctx` is owned by the machine.** A handler takes
   `ctx: &mut Ctx<'_, Rt>`, lent by the `Machine` beside its registers and
   owned by an async glue's rooted store. A handler parameter `rt: &Rt` is
   refused by name.

8. **A required instance has a mono glue.** An instance is called
   through a mono glue, a plain `fn` of the call's own arguments. A
   declaration written `heavy`, one holding a `#[state]` value, and one taking
   a two-word parameter (`&str`, a slice) or a projection has none, and
   `combine` refuses a requirement that reaches it
   (`RequiredInstanceWithoutGlue`).

**Why.** A handler generic over `I` needs something to call at `I`, and the
checker already knows which instance that is. A word the checker chose is
the cheapest carrier of that knowledge and keeps generics out of the
machine.
**Cost.** One store of the receiver into `ctx`, one load of the word, one
indirect call, one load on the far side; a parameter at a signature
variable adds one reference value and one borrow on the far side.
**Rejected.**
- Entry tree carried in the value (a node arena, a carrier struct per
  bounded variable) — rebuilds per element what frame and site already
  hold; measured 1.2–1.8× slower than the `dyn` chain it was to replace.
- Stamp in spare bytes, or a bound bundle — a requirement becomes a
  position in a type-level list, so `A + B → A` needs a coercion; scalars
  have nowhere to hold a stamp.
- Interfaces in the `Large` header's vtable — scalars have no header,
  several acvus types share one Rust payload, and a `&'static` table cannot
  take another crate's registration.
- `Ctx` built at the call op — its address escapes into the handler and the
  op loses its tail jump.
- The word as an argument of the call's window — it has no `ValueId`.
- A run-time `TypeId → handler` table — a lookup where the compiler knows
  the answer.
- A `Box<dyn>` per stage — a box and a virtual call per stage, and the
  checker's knowledge thrown away at the boundary.
- A marker bound with no handle (`T: HasInstance<S>`) — a requirement the
  body cannot call.
- An iterator typed by the element types it passes through
  (`Iter<(T_k, …, T_1), O>` as nested pairs, one uniform payload per
  pipeline length, adaptors and consumers as per-length instances bounded
  at eight) — built and withdrawn: it specializes at the wrong layer. It
  buys a `match` for stage dispatch at the price of a length bound, a
  length recursion and per-length declarations, and cannot reach the cost
  that dominates, the closure re-entry per element. The per-element win
  belongs to loop lowering (RFC-0066).
- Stage kinds in the type (`Iter<(Range, Map, Filter), O>`) — every
  pipeline shape its own instantiation.
- Flattening a `dyn` chain into `Vec<Box<dyn Stage>>` — keeps a box and a
  virtual call per stage.

## RFC-0068: A value is read back at a type only where the checker decided

Status: Accepted

`X → T` is one function and sound; `T → X` has as many candidates as there
are `X` and is sound only where something outside Rust chose. The one
ground for `Rt::Value → T` is the crossing from the machine into a handler,
written by the macro from the type the checker settled at the site. Every
other reading back is cut.

1. **Three rules, held by review, not at run time.**
   - Only `T → Value` is a public operation of `acvus-extern`; `Value → T`
     exists only in code `#[extern_fn]` and `extern_signature!` write. No
     extension body writes `erase`, `materialize`, `Owned::from_value` or a
     `Lent`; `Rt::Value` enters a body only as the fill of a
     `Var<kind::Type>` parameter.
   - A handler's type variable is stored and passed at the variable: a
     payload holding a value at `I` holds it as `I`, and an `Instance` or
     `Closure` beside it is typed at the same `I` and `E`. The glue fills
     every such `I` with `Owned<Rt>` and every `E` with `()`, so a
     declaration is one Rust type however its payload is spelled.
   - `#T → T` always exists. An instance is resolved at the ground type
     including its representation, and a requirement at a variable is met
     only by the uniform instance.

   Every check inside the crossing is a `debug_assert!`; the machine restates
   none of the checker's proofs in release.

2. **`Instance` is closed.** `Instance::at` is `#[doc(hidden)] pub unsafe`;
   its contract is that the word was made by `Runtime::instance_value` from an
   entry of an instance of `S` at the type `I` is filled with. Its callers are
   `Required::site` and the glue the macro writes. `InstanceRun`'s fields are
   private and its one constructor, `from_glue`, is `#[doc(hidden)] pub
   unsafe`, called by the macro with the typed glue. `Instance::call` and
   `call_await` are safe. `into_value` stays: it is the forgetting direction.

3. **A payload names its variables.** A derived `ExternType` payload may
   be or mention a type variable. What that once guarded is kept by the
   type: an argument of a derived extension type that names no variable is
   the slot's specialized representation (`Items<#String>`); one naming a
   variable, or an `Erased<Rt, _>`, is uniform. `X<#T>` and `X<T>` do not
   join, so a generic declaration over `X<T>` refuses an `X<#T>` at compile
   time.

4. **The crossing is the only reader.** `Runtime::materialize`,
   `value_as_ref/mut`, `inline_ref/mut`, `deref/mut`, `OneValue::materialize`,
   `FromValue::from_value`, `Loan::borrow`, `Ctx::receiver` and the `Restore*`
   traits are `unsafe`, `#[doc(hidden)]`, and called only by generated code.
   A handler has a value at a type from its parameters, its payload, or an
   instance's typed return.

5. **A requirement is a call the checker decides.** A requirement is
   `Requirement { signature, pattern, calls }`: `pattern` is the signature's
   type with its variables replaced by what the marker names, and `calls` is
   the task it calls at. Instantiating the declaration's scheme opens one
   `Decision::Instance` per requirement, its call type the pattern at the
   same fresh variables as the handler's type. That decision narrows by shape
   and by task (a candidate carries its body's task; only instances at or
   below `calls` are admitted), binds the signature's other variables from
   the chosen instance (`T := i64` from `next` at `NRange`), and refuses with
   the sentence naming the signature, the call it could not place, and the
   instances it could have reached. The instances an instance itself
   requires are decided the same way, nested (RFC-0070 rule 3).

6. **A result at a signature variable crosses as the runtime's value.**
   A signature whose result mentions one of its type variables crosses that
   result as `Rt::Value`: `Returned<Rt>::cross` erases what the instance
   returned and `restore` materializes at the requirer's own result type,
   which the checker unified with the instance's. A concrete result crosses as
   typed. `Signature::call_later` returns `impl Future`, adding no box.

8. **`X<#T>` reaches `X<T>` only through the family's declared cast.**
   Under a constructor the two representations are invariant in the slot, and
   only the family's `#[extern_cast]` (RFC-0041) converts between them.
   - With no declared cast, `X<#T>` where `X<T>` is asked is a type mismatch.
   - With one, the checker inserts it at the conversion site.
   - A value whose path would go `#T → T → #T` or `T → #T → T` is uniform
     from where it is made; the cast runs once, at the producer.
   - `X<#T> → X<T>` is the declared direction; nothing asks for the other.

   A requirement adds a conversion site: the argument standing at the
   required variable, where no instance stands at the settled `X<#T>` and the
   cast reaches an `X<T>` that has one. A source that wants to skip the cast
   declares its own instance at `X<#T>`.

**Why.** The checker's decision is the only ground a reverse crossing has;
an `unsafe` whose invariant lives in another compiler is a sentence nothing
checks, so it belongs to generated code alone.
**Rejected.**
- A checked `materialize` (`TypeId` at run time) — makes the bypass safe to
  write and restates a proof the checker holds.
- `Instance::at` exposed with a longer safety contract for handler authors
  — its reader cannot check it; only the constructing crossing knows the
  fact.
- A `Value`-typed payload with a runtime tag — moves the checker's decision
  into the machine.
- A bundle per handler naming several signatures (`InstanceOf<S>` for
  receiver-less functions such as `clone`, `eq`, `hash` at `T`) — one value
  per signature holds (RFC-0067 rule 1); a receiver-less requirement is
  an `Instance` like any other (RFC-0070 rule 4).

## RFC-0070: An instance requires what its own declaration says

Status: Accepted

1. **An instance declaration takes what it requires.** Beside the
   signature's parameters, an instance declaration may take `Required`
   parameters (zero-width, `ARGUMENTS = 0`) naming what the instance itself
   requires: `clone_vec<T>(a: &Vec<T>, elem: Instance<core::clone<T, Rt>, T,
   Rt>)` is `impl<T: Clone> Clone for Vec<T>`. The requirement belongs to the
   instance, not to the value. The checker settles a requirement's variable by
   unifying its pattern with the declaration's type, so `T` settles from
   inside `&Vec<T>`, and `hash_map()`'s `K` from its return type at the first
   `insert`; no parameter has to stand at the variable itself.

2. **The word addresses an entry.** `Instance` stays one `Rt::Value`. The
   word addresses an `InstanceEntry { run, requires }` owned by `Prepared`:
   `run` is the glue and its task, `requires` one word per `Required`
   parameter of the instance's declaration, in declaration order, each
   addressing an entry. `prepare` builds one entry per distinct (signature,
   chosen tree), address-stable and shared across the sites that chose it.
   The glue takes the entry and binds its `Required` parameters from
   `entry.requires`, where an ordinary handler binds them from the site
   table's; the macro chooses by declaration kind. `Required::site` reads the
   site table and no registry; the registry answers only `glue(signature,
   instance)`.

3. **The checker decides the tree, and the IR carries it.** An
   `InstanceSig` carries its declaration's requirements, written at its own
   variables. When `Decision::Instance` settles on a candidate with
   requirements, the solver instantiates the candidate's type and its
   requirement patterns with one variable map and opens one nested
   `Decision::Instance` per requirement, reaching the signature's instances
   through a map lent at construction (the requirement graph is cyclic, so
   candidates cannot embed their children). A child that reaches no instance
   fails as `NoInstance` naming the signature and the type. The IR carries
   the tree: `Callee::Extern { id, instance, required: Vec<Chosen> }`, with
   `Chosen { signature, instance, required }` self-describing.

4. **`Instance::call` takes the receiver the signature declared.** The
   receiver is `S::Recv`: `&I` or `&mut I` by the signature's first
   parameter, checked where the call is written through `Receiver<Rt>`, a
   bound on the method rather than on the `Signature` impl. A signature
   taking its receiver by value has no impl, and the missing impl is the
   refusal. `Ctx::recv` is a `*const Rt::Value`; the glue opens what it names
   at the declared loan, and no `&mut` to the receiver word is made.

5. **The core signatures.** `core` holds the signatures the compiler
   names, and nothing else:

   - `eq<T>(a: &T, b: &T) -> bool` — named at `==`, `!=` on an extension
     type (RFC-0020).
   - `cmp<T>(a: &T, b: &T) -> i64`, answering `-1`/`0`/`1` — named at `<`,
     `<=`, `>`, `>=` on an extension type (RFC-0020).
   - `clone<T>(a: &T) -> T` — named at the explicit copy (RFC-0018).
   - `hash<T>(a: &T) -> u64` — named at a map's keying.
   - `to_string<T>(a: &T) -> String` — named at interpolation, once it
     lowers to a call.

   `cmp` answers an `i64` because the language has no `Ordering`, and a core
   type beside the signature is not worth it. `hash` answers `u64` because a
   hash is a bit pattern, not a number. An instance of `hash` at `T` is
   admitted only where an instance of `eq` at `T` is declared (`combine`
   refuses `HashWithoutEq`); that `eq`-equal values hash equal is pinned by a
   test per type. `Object` and `Enum` have no instance of any of the five
   (RFC-0019).

   The standard registry declares `eq`, `clone`, `cmp` and `hash` at `i64`,
   `f64`, `bool`, `u8`, `char` and `String`, each pinned by a test to the
   operator instruction it stands beside (RFC-0020); `Vec<T>` has each,
   requiring the same signature at `T` (`cmp` lexicographic). `hash_map()`
   requires `hash` and `eq` at `K`; `hash_map_by(hash, eq)` is the closure
   form, where an object key goes. `dedup` requires `eq` at `T`.

**Why.** A container is not built with its element's instances beside it,
and must not be: `clone` at `Vec<T>` needs `clone` at `T` because of what
the instance is, not what the value holds. Deciding the tree in the checker
keeps the machine from re-proving anything.
**Cost.** One more load per instance call (entry, then glue).
`Callee::Extern` widens by a `Vec<Chosen>`, empty for a declaration with no
requirements.
**Rejected.**
- A uniform structural walker as the fallback instance (`eq`/`clone` over
  `Value` by vtable) — an implicit path where none was registered, deciding
  structural equality by accident.
- `T::clone` as a static call in the handler — a uniform handler is one
  Rust body for every `T`; a `#T` member already has Rust's `Clone`.
- Storing a container's requirement beside the value — right for a stage
  (RFC-0067 rule 4), wrong for a container.
- `prepare` re-resolving inner requirements from settled types — the
  machine checks nothing the compiler proved.
- A single overload position over the tree — a cyclic requirement graph
  has no finite radix.
- Folding `Instance` into a bound — one value per signature holds.
