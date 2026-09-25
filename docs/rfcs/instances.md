# Required instances

How a Rust handler generic over a type variable calls a shared signature
(RFC-0019) at that variable's type. The checker chooses every instance;
the machine receives the choice as one word and computes no type. This
document decides what that word is, where it lives, how a handler
declares and calls it, which crossings may read a value back at a type,
and which signatures `core` holds.

## RFC-0067: The machine holds no generics; an instance that steps a value owns it

Status: Accepted

Every type at a call is ground. Whatever a type variable ranges over is
settled by the checker and arrives at the machine as a word; no function
type crosses the extern boundary as a type-level list, and no position in
the machine computes a type. The runtime shape is a value with a function
pointer with it; the checker picks the pointer, the loans analysis
governs the borrow, the effect system says whether the call may suspend.

1. **A requirement is declared by taking it, in one of two forms the
   signature's receiver decides.** `#[extern_fn]` records either form on
   `FnDecl::requires` (its form is RFC-0068 rule 5) and refuses a variable
   that is not one of the declaration's own `Var<kind::Type>` parameters.
   What the word addresses is RFC-0070 rule 2.
   - **A receiver the call steps or consumes: `Instance<S, I, Rt, T>` owns
     it.** A signature whose first parameter takes its receiver `&mut I` or
     `I` (`iter::next`) is required by taking the receiver and its instance
     as one parameter, `it: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>`.
     It holds the receiver and the word, both private. Only the glue makes
     one, from the argument and the word the checker chose for its type at
     that site (rule 3). Nothing hands out the word apart from its receiver,
     and `into_inner` drops the word with it. A requirer lent its receiver
     `&mut` takes `Instance<S, &mut I, …>`, which owns that loan.
   - **A function of a type's values: `InstanceOf<S, I, Rt, T>` stands at
     the type**, `S`'s receiver type being `I` by a bound of the type. A signature whose receiver is `&I` (`core::eq`,
     `core::hash`, `core::cmp`, `core::clone`, `core::display`) is required
     by an `InstanceOf`, exactly one of the runtime's values (`ONE_VALUE`).
     Its call takes the receiver, since a requirer applies it to many values
     of the type (a map's keys, both sides of `==`). Its ground is RFC-0068
     rule 1: an instance is resolved at the ground type.
   - The other pairing is a type error of the glue's crossing, naming this
     rule: an `Instance` at a signature whose receiver is `&I`, and an
     `InstanceOf` at one whose receiver is `&mut I` or `I`.
   - Two signatures take two parameters. Two `Instance`s cannot own one
     receiver, so two receiver-owning requirements at one variable are
     refused; no declaration asks for it.

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
   one that names none is `Known(PURE)`. A signature whose parameter type
   projects through a variable bounds it `Chosen` (RFC-0041):
   `tally<Ts: Var<kind::Type> + Chosen, O, E, I, Rt>(it: Pipe<Ts, O, E, I, Rt>)`,
   whose per-length instances fill `Ts` with their own stage list, `()` the
   empty one.

3. **The word lives in the site table.** `prepare` places the
   chosen word in the call site's table once per site; `Required<S, I, T, N>`
   has `ARGUMENTS = 0` and copies it out: into an `InstanceOf` as it stands,
   and into an `Instance` together with the argument it owns. A requirement
   costs zero ABI words, no move before the call, and nothing rebuilt per
   call.

4. **The pairing is built where the checker chose it, not asserted by a
   type.** An `Instance` calls its own receiver and no other, so no Rust
   code can hand one value to the instance chosen for another. A stage keeps
   the pipeline below it as one `Instance`: `Map<I, F>` holds
   `inner: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>` and the closure.
   The pairing is a fact of construction, and no premise that one Rust type
   at `I` has one implementation is needed. An instance's own requirements
   are its declaration's, not its value's (RFC-0070 rule 1).

5. **`Now` and `Later`.** `Instance::into_async` retypes `Now` to
   `Later`, one way; there is nothing the other way, since a sync caller has
   nowhere to suspend to. `call_await` exists only on `Later`. `Later::call`
   serves a `sync =` twin, the body a `Sync` site runs where the checker
   settled on an instance that returns.

6. **The call.** The receiver does not cross as an argument:
   `Instance::call` names its own receiver in `Ctx`, lent at the signature's
   mode (`&mut self` for `&mut I`, `self` for `I`), and `InstanceOf::call`
   names the receiver it is given (`&I`). The mono glue reads it back at the
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
machine. The word and the value it was chosen for are one fact, so they
travel as one: kept apart, they are safe only while every value at a Rust
`I` has the word's implementation, which no local code holds, since the
glue fills every `I` with `Owned<Rt>` (RFC-0068 rule 1).
**Cost.** One store of the receiver into `ctx`, one load of the word, one
indirect call, one load on the far side; a parameter at a signature
variable adds one reference value and one borrow on the far side. An
`Instance` is its receiver and one word.
**Rejected.**
- Entry tree carried in the value (a node arena, a carrier struct per
  bounded variable) — rebuilds per element what frame and site already
  hold; measured 1.2–1.8× slower than the `dyn` chain it was to replace.
- Stamp in spare bytes — scalars have nowhere to hold one.
- The receiver and its `Instance` side by side, typed at one `I` (this
  rule's former form) — a Rust type error refuses a mispairing only while
  one Rust type at `I` has one implementation, so nothing local refuses
  a value handed to another value's instance.
- One bundle of several signatures at one receiver — `A + B → A` needs a
  coercion no declaration needs.
- Every requirement as an owning `Instance` — a container would hold a
  word per element, the carrier measured slower above.
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
   - `T → Value` and `Value → T` exist only through the glue's and the
     runtime's capability. Every function that crosses between a Rust type
     and the runtime's value (`OneValue`, `Cross`, `Passed`, `Arg::take`,
     `IntoRun`, the derive's helpers) takes a `Crossing`, and every one that
     makes, empties or retypes a holder of a bare value (`Owned::from_value`,
     `vacant`, `value_mut`, `erased`, `lend_run`, `Erased::into_value`,
     `InPlaceElement`, `Stored::from_payload`) takes a `Crossing` or a
     `Holding`. Both constructors are `unsafe`; the glue `#[extern_fn]` and
     `extern_signature!` write makes a `Crossing` from its `Ctx`'s runtime,
     and the runtime makes its own. A handler is called with its parameters
     and a `Ctx`, neither of which holds one.
   - A handler's type variable is stored and passed at the variable: a
     payload holding a value at `I` holds it as `I` or inside the `Instance`
     that owns it, and a `Closure` beside it is typed at the same `E`. The glue fills
     every such `I` with `Owned<Rt>` and every `E` with `()`, so a
     declaration is one Rust type however its payload is spelled.
   - `#T → T` always exists. An instance is resolved at the ground type
     including its representation, and a requirement at a variable is met
     only by the uniform instance.

   Every check inside the crossing is a `debug_assert!`; the machine restates
   none of the checker's proofs in release.

2. **`Instance` and `InstanceOf` are closed.** Their constructors are
   crate-private: `InstanceOf::at(word)` and `Instance::own(receiver, word)`,
   whose contract is that the word was made by `Runtime::instance_value` from
   an entry of an instance of `S` at the type of that receiver, as the
   checker chose it at the site the receiver was passed to. The glue and the
   runtime reach them through `Crossing::instance` and
   `Crossing::instance_owning`, from `Required::site` and an `Owning`
   parameter's site.
   `InstanceRun`'s fields are private and its one constructor, `from_glue`, is
   `#[doc(hidden)] pub unsafe`, called by the macro with the typed glue.
   `call` and `call_await` are safe. The receiver is the `Instance`'s own
   `I`, or the `&I` an `InstanceOf` is given, with
   `I: Deref<Target = Rt::Value>`; the far side's mono glue
   reads that value as the instance's literal receiver type, which is the
   same crossing as any other parameter's. A later argument standing at a
   signature variable is passed at that variable, `&T` where the receiver is
   `&T`, and the glue crosses it (`CrossesRest`) under the same bound.
   `into_value` stays, behind the runtime's `Holding`: it is the forgetting
   direction.

3. **A payload names its variables.** A derived `ExternType` payload may
   be or mention a type variable. What that once guarded is kept by the
   type: an argument of a derived extension type that names no variable is
   the slot's specialized representation (`Items<#String>`); one naming a
   variable, or an `Erased<Rt, _>`, is uniform. `X<#T>` and `X<T>` do not
   join, so a generic declaration over `X<T>` refuses an `X<#T>` at compile
   time.

4. **The crossing is the only reader.** `Runtime::materialize`,
   `value_as_ref/mut`, `inline_ref/mut`, `deref/mut`, `OneValue::materialize`,
   `Loan::borrow`, `Ctx::receiver` and the `Restore*` traits are `unsafe`,
   `#[doc(hidden)]`, and called only by generated code.
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

7. **`T` stays `T`.** What a handler may do with a `T` is what its bounds
   enable; nothing is stored at the type and nothing is asserted about a
   value.

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
- A bundle per handler naming several signatures — one value per
  signature holds (RFC-0067 rule 1).

## RFC-0070: An instance requires what its own declaration says

Status: Accepted

1. **An instance declaration takes what it requires.** Beside the
   signature's parameters, an instance declaration may take `Required`
   parameters (zero-width, `ARGUMENTS = 0`) naming what the instance itself
   requires: `clone_vec<T>(a: &Vec<T>, elem: InstanceOf<core::clone<T, Rt>, T,
   Rt>)` is `impl<T: Clone> Clone for Vec<T>`. The requirement belongs to the
   instance, not to the value. The checker settles a requirement's variable by
   unifying its pattern with the declaration's type, so `T` settles from
   inside `&Vec<T>`, and `hash_map()`'s `K` from its return type at the first
   `insert`; no parameter has to stand at the variable itself.

2. **The word addresses an entry.** An `InstanceOf` is one `Rt::Value`, and
   an `Instance` is its receiver and one. The
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

4. **The call takes the receiver the signature declared.** An `Instance`
   lends its own receiver at the signature's first parameter's mode:
   `call(&mut self)` for `&mut I`, `Consume::call(self)` for `I`. An
   `InstanceOf` takes `&I`. The mode is `Signature::Mode` (`Shared`, `Mut`,
   `Moved`), read as a bound on the method where the call is written. `Ctx::recv` is a `*const Rt::Value`; the glue opens what it names
   at the declared loan, and no `&mut` to the receiver word is made.

5. **The core signatures.** `core` holds the signatures the compiler
   names, and nothing else:

   - `eq<T>(a: &T, b: &T) -> bool` — named at `==`, `!=` on an extension
     type (RFC-0020).
   - `cmp<T>(a: &T, b: &T) -> i64`, answering `-1`/`0`/`1` — named at `<`,
     `<=`, `>`, `>=` on an extension type (RFC-0020).
   - `clone<T>(a: &T) -> T` — named at the explicit copy (RFC-0018).
   - `hash<T>(a: &T) -> u64` — named at a map's keying.
   - `display<T>(a: &T, out: &mut String)`, appending `a`'s text to `out`
     — named at a template's `{{ x }}` whose `x` is not a `String` or a
     `&str` (RFC-0071 rule 3), with the template's text as `out`, so a tag
     allocates no string of its own. The standard registry's generic
     `to_string<T>(a: &T) -> String` requires `display` at `T` (rule 1)
     and appends to an empty `String`, so every type with a `display`
     instance has `.to_string()`, and its text has one source. `display`
     stands at no `str` (a two-word receiver has no mono glue, RFC-0067
     rule 8) and at no `String`, whose text is itself. The owned copy of
     text, `"…".to_string()` and `s.to_string()` (RFC-0062 rule 2), is
     `string::to_string(a: &str) -> String`, which `copies(a)`; a `String`
     reaches it as a view. The generic's `T` ranges over the types
     `display` stands at (RFC-0067 rule 2), which hold neither, so a text
     receiver leaves it (RFC-0043).

   `display` writes into its caller's `String` rather than returning one,
   so a text built of many parts grows one buffer.
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
- Storing a container's requirement with the value — right for a stage,
  whose `Instance` owns the pipeline below it (RFC-0067 rule 4), and wrong
  for a container, whose elements' functions stand at the type
  (`InstanceOf`).
- `prepare` re-resolving inner requirements from settled types — the
  machine checks nothing the compiler proved.
- A single overload position over the tree — a cyclic requirement graph
  has no finite radix.
- Folding `Instance` into a bound — one value per signature holds.
