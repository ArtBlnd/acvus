# RFC-0067: A required instance is a parameter the site fills

Status: Draft (2026-09-20) — the design the owner and coordinator settle
before any code run; the iterator is its first customer and the machine's
later specialization its reason.

## Problem

A handler can require nothing of a type variable but its kind. RFC-0019
gave the language shared signatures — `core::clone<T>(&T) -> T` with one
instance per concrete type, resolved at a call by the call's ground type —
and stopped there: "no generic functions in the language, and no `dyn`";
"nothing new is inferred or dispatched". A handler generic over `T` cannot
call `clone` on its `T`, because the requirement a Rust bound stated
(`T: HasInstance<clone>`) only narrowed which `T` the checker admitted and
gave the body no way to reach the instance. The type-helper pass of
2026-09-20 deleted that marker for exactly that reason: a requirement no
body can use is a convention, not a guarantee.

The iterator showed what the gap costs. Without a way to say "`I` is a type
with a `next`" and call it, every design put the whole pipeline inside one
extension type: a `Box<dyn>` per stage (the design on master), or one Rust
type per pipeline length with a bound of eight and 292 registry names
(RFC-0065, built, measured, rejected). Both encode in the payload what the
types already know at every call site, and neither can reach the cost
that dominates a `map` row — the re-entry into the interpreter for each
closure call — because the pipeline's shape is not visible where a lowering
could use it.

## What exists

- A shared signature is a name with one polymorphic type and no body
  (`SignatureDecl`); an instance is an `ExternFn` declared `instance_of =
  sig` at a concrete type; `Externs::combine` collects instances under the
  signature and refuses two whose types unify (`DuplicateInstance` through
  `unify_patterns`) — RFC-0019.
- A call to a signature resolves like a `Monomorphize` function: the
  instance whose type matches the call's resolved type runs — RFC-0011,
  RFC-0019.
- A member instance may stand at a pattern (`at_a_member`: the declaration
  with one variable replaced by a member type its `OneOf` bound names) —
  the first pattern instance, admitted for `Monomorphize` only.
- Per-site glue: `HandlerFactory::at_site(args: &[ArgAt])` fills a
  declaration's site table from the call site's settled types and hands
  back the `AtSite` of that one call — RFC-0059. The site knows every
  argument's ground type.
- A closure crosses as `Fn<Args, R, E, Rt>`, a typed callable the glue
  fills from a closure value and the body calls at its declared types —
  RFC-0050 rule 6. `Fn::call_now` re-enters the interpreter.
- A type variable's kind is `Var<K>`; the fills of `kind::Type` are the
  types that name an acvus type (`TyArg: Var<kind::Type>`) and the
  runtime's carrier — batch A of the type-helper pass.

## Decision

1. **A requirement is the signature itself.** A handler requires an
   instance by naming the signature's type at its own variables:
   `next<I, T, E>` is the constraint, not a marker on `I`. What the
   instance's other variables are (the element `T` of an `I`) is read off
   the instance's signature by unification, the way a call already reads
   it. No associated type, no functional dependency: the signature's own
   variables carry the relation.

2. **An instance may stand at a pattern.** `fn next<I, T, E, Rt>(it:
   &mut Map<I, T, E, Rt>, …)` is an instance of `next` at the pattern
   `Map<I, T, E>`; it may itself require `next<I, U, E>` of its inner `I`.
   Two instances whose patterns unify are refused at combine, as two
   concrete instances at one type are today. The set of types with an
   instance is no longer a closed list: a bound is a predicate — "an
   instance whose pattern unifies with this ground type exists, and its
   requirements hold" — checked on ground types by structural recursion,
   which terminates because types are finite trees.

3. **The requirement is a parameter, and the site fills it.** A handler
   that requires `next<I, T, E>` takes it: `inner: Instance<sig::next<I, T,
   E, Rt>>` (the name is settled in the type-helper pass; the shape is a
   typed callable, as `Fn` is). The parameter *is* the requirement — the
   macro reads `requires` from it, and a declaration that requires without
   taking, or takes without requiring, cannot be written. At a call site
   the glue knows the ground `I`; it resolves the instance and every
   instance that instance requires, recursively, into a tree of handler
   entries — the dictionary — built once at prepare and owned by the site.
   A handle is `(entry, &dict)`. The body calls `inner.call(rt, frame,
   &mut it.inner)`: a direct call through a function pointer to a handler
   whose own dictionary rides beside it. No value carries a table
   (RFC-0019's rule stands): resolution is by type, at the site, never
   asked of a value.

4. **The inner `I` inside a value is the runtime's carrier.** `Map<I, T,
   E, Rt>` holds its inner iterator as `Owned<Rt>`; Rust cannot
   monomorphize on an acvus `I`. The call to the inner `next` hands
   `&mut Rt::Value` to the inner instance's glue, which reads it through
   `deref_mut` at its own concrete type — one pointer, no allocation. An
   element `Option<T>` arrives as `Option<Owned<Rt>>`; a closure `Fn<(T,),
   U>` is called with the carrier at `T`, as RFC-0065's accepted leaf
   already did.

### What this buys

- `iter::next` is one signature; an adaptor is an extension type with one
  `next` instance at its pattern; a consumer is a handler requiring `next`
  of its argument. About forty declarations, no length bound, no per-length
  copies.
- Per element: one direct call per stage plus one `deref`, against the
  dyn chain's vtable call and box hop per stage. Cheaper, and not the point.
- The point: the pipeline's shape is a **static fact at the call site** —
  the dictionary tree — where RFC-0066's lowerer and the machine can read
  it. A consumer's site that knows `sum(Map<Filter<Range>>)` whole can be
  lowered to a loop the closure bodies inline into. That lowering is not
  this RFC; this RFC makes it possible by refusing to hide the shape in a
  value.

## What it costs

- The solver gains a deferred constraint: a requirement fires when its
  `I` becomes ground and unifies the other variables. `map(range(0, 10),
  |x| x + 1)` types the lambda only after `next<Range, T>` yields `T`. The
  `Decision`-waiting machinery the solver has for captures is the model.
- `Externs::combine` matches instance patterns, not only concrete types;
  `at_a_member` generalizes.
- Per-site glue grows a resolver that builds the dictionary tree from the
  registry at prepare, and a handle type the body calls through. Prepare
  cost is per site, once.
- Effects: `Map<I, T, E>`'s `E` is the join of the inner `E` and the
  closure's, carried in the type as `Iter<…, E, …>` carried it.
- The `&mut Rt::Frame<'_>` double reference is paid per stage per element
  until the frame's form is settled (type-helper pass, batch C).

## Rejected

- **A marker bound with no handle** (`T: HasInstance<sig>`): a requirement
  the body cannot use; deleted in batch A.
- **The instance handle routed through `Fn::call_now`**: that path
  re-enters the interpreter with a `CallToken` and a frame; an instance is
  a Rust handler and is called directly. The two handle types share a
  shape and differ in the call path; whether they are one type with two
  fills is the type-helper pass's question, not this RFC's.
- **The dictionary stored in the value** (`Map` holding its inner `next`
  pointer): RFC-0019's "no value carries a table", and it hides the shape
  the lowerer needs.
- **Rust-side specialization** (one type per pipeline shape, RFC-0065):
  measured; buys a `match` for a vtable and cannot reach the closure
  re-entry.
- **A signature variable** (`S: Signature, T: HasInstance<S>`): not needed
  by any customer; the signature is always named.

## Where this is hard

- Inference order: a lambda argument's parameter type depends on the
  requirement firing; if `I` never grounds, the lambda is unchecked —
  the language's rule "the later check catches the path's facts" applies,
  and the check must report which requirement never fired.
- Overlap: `Chain<A, B>` requiring `next` of both is two requirements of
  one instance; the dictionary is a tree, not a list.
- Recursion in patterns: `Map<Map<Range>>` resolves by structural
  recursion; a pattern instance requiring itself at the same type is a
  cycle and is refused at combine (the pattern would unify with its own
  requirement).
- The lambda inside a `Map`: its `Fn<(T,), U>` is filled by the glue at
  the *adaptor's* site (`map(it, f)`), stored in the value as today, and
  called from `Map::next` at `Owned<Rt>` — the closure's site and the
  consumer's site are different sites, and the closure travels in the
  value while the instance dictionary does not. State this distinction in
  the handle's doc: a closure is a value; an instance is a fact of the
  types.

## Order of work

1. Type-helper pass batches A–C land (they reshape `Fn`, the glue and the
   registry this RFC builds on). Done for A and B; C running.
2. Pattern instances and the predicate bound in `Externs::combine` and the
   solver's deferred requirement; `clone<T>` as the first requirement
   (`same<T>(a: T, b: T, clone: Instance<clone<T>>)` — the test batch A
   deleted returns as a body that calls what it requires).
3. The dictionary resolver in per-site glue and the handle's call path;
   measured against a direct extern call.
4. `iter::next`, sources, adaptors, consumers on it; the dyn chain and
   `Fn::call_value*` (if still present) removed; numbers against the dyn
   chain under `benches/README.md`'s protocol.
5. Hand the dictionary tree to RFC-0066's lowerer as a loop customer.
