# Holes

What encoding the spec found, each with the lines it rests on and the evidence for
it. Three kinds of entry:

- **H**, a hole: the spec gives no result where its own text, its examples or the
  RFCs it cites want one, so l.301 refuses there.
- **T**, a text: the spec's two texts disagree, or the spec and an RFC it cites do.
- **O**, an order: the outcome depends on the order of solving.

A reading the model takes where the text leaves one open is stated in the note of
the primitive it belongs to, not here.

## H1. A cast narrower than its node

A cast's node is its parameter's constructor (l.78), and nothing requires its
parameter to match every type of that node, nor its result to write only variables
its parameter writes: `Vec<u8> → Bytes`, or `A → Box<T>`. The join raises each type
by casts to the nearest common ancestor node (l.86), and a type of that node the
cast does not match, `Vec<u16>`, or one whose step up is not determined, `A`, can
be raised by none, though its node has an ancestor. Types §3 states the tree per
type ("every type has at most one next step up") and the spec per node. The spec
gives no join there, so a relation that needs one is refused. Requiring a cast's
parameter to be its node at distinct variables, and its result to write only those,
would close the hole at the declaration.

## H2. A value whose path up meets two variants of `Error`

The error join keeps elements of different nodes apart even where one steps up to
the other (l.95), so `Error{B(B), D(D)}` is a type where a cast goes from `B` to
`D`. A value of `A`, with casts `A → B → D`, then reaches it by two compositions:
`A → B` then the wrap in `B`, and `A → B → D` then the wrap in `D` (l.108, l.114).
Line 116 states that the composition toward a given type is unique, and the two
call different functions. The model refuses there (`CoerceError::Compositions`);
taking the composition of the lowest rank would choose `B`. Whether a program
reaches it is not yet shown. A candidate is a value entering by a boundary whose
part is not an unknown of the binding group the place lies in (l.286), such as a
reference reaching a capture whose error part the enclosing body joined from `B`
and `D`.

## H3. A view at a boundary

A view impl is a cast from a reference to a container to a reference to a slice or
`str` (l.78), and a reference's node is its mutability and its target's constructor
(l.78). A reference's constructor is its mutability alone (l.66), so at a boundary
`&V ⇝ &[u8]` the two sides have one constructor, and the target, an invariant part,
gives the equation `V = [u8]` (l.289); the casts link parts only where constructors
differ (l.291). The view is never taken, and `make() ≤ V` with `make() -> Vec⟨u8⟩`
fails. Witness, refused by the model (`examples.rs`,
`hole_3_a_view_at_a_boundary_equates_the_targets`):

```
let v = make();          // make() -> Vec⟨u8⟩; the view as_slice(&Vec⟨T⟩) -> &[T]
take(&v);                // take(&[u8])
```

The join of two references reaches the same place: `&A ⊔ &C` is `&B` by the casts
`&A → &B` and `&C → &B` (l.86), while two lower bounds `&A ≤ R` and `&C ≤ R` equate
their targets.

## H4. A reborrow at a split

Where `&mut X` meets `&Y` at a lower bound or a boundary, the constructors differ
and no cast links them (l.291), so nothing relates `X` and `Y`. In
`look(&mut v)` with `look(&Vec⟨u8⟩)` and `v = Vec::new()`, `v`'s argument has no
relation and falls back to `!`, and `&mut Vec⟨!⟩ ⇝ &Vec⟨u8⟩` fails. The model
relates the targets by an equation, as the reborrow row `&mut T ⇝ &T` keeps `T`
(`design/11-value.md`, reading 2), and accepts it (`examples.rs`,
`hole_4_a_reborrow_at_a_boundary_equates_the_targets`).

## H5. What a binding reads of its path

A binding reads its position where its path passes no reference, and lends it where
the path passes one (l.317). Whether the path passes one is what the projections on
it became: each applies to its operand's target where the operand is a reference
(l.309). But a binding reads "whether its equivalence class is a reference" (l.267),
and depends on the nodes that record into its place, other than projections of a
field, a component, an index or a variant (l.248), which are the projections on
its path. So nothing makes a binding wait for them, and nothing it reads tells it
what they became. Under the dependencies l.248 states, the outcome depends on the
order: a binding decided before its path's projection reads its position, and one
decided after lends it. Witness, two outcomes in the model, `x: u8` and `x: &u8`,
both accepted (`examples.rs`, `a_binding_behind_a_reference_lends_its_position`,
`BindingPath::AsStated`):

```
let o = Some(1u8);
if let Some(x) = &o { }
```

Making the binding depend on the projections on its path gives one outcome, `x:
&u8`, as in Rust (`BindingPath::AfterPath`). The exclusion in l.248 is about whether
a class is a reference, which those projections do not decide; what a path passes
is a different question, which l.248 does not ask.

## H6. What a broken dependency's relations wait for

A lend reads whether its place is a reference (l.267) and depends on the nodes
that record into its place (l.248), which are the nodes at whose positions the
place lies (l.244 (b), l.250); a lend's place is at its position. So two lends of
one place depend on each other, each breaks its dependency on the other (l.269),
and "the relations recorded by a node that a broken dependency pointed to are
recorded once the node that broke it is decided" (l.269). Until then the
reference the first lend gives has no kind, and a use that reads it in between
reads nothing there. In Example 12, `&y` and the `&y` of `y + 1` are such a pair:
`y + 1` decided between them reads its first argument as nothing and its second as
`?{1}`, takes the default `i32` (l.344), and the program is refused, `1u16 ≤ Y` not
reaching `i32`. Witness: Example 12 itself, which reaches an accepted and a
refused outcome (`examples.rs`, `example_12_accumulation_that_starts_from_a_literal`,
`Deferral::Recording`).

Read so that a broken dependency's relations are recorded when their node is
decided, and only the value read by the node that broke it leaves them out
(`Deferral::Reading`), Example 12 reaches the spec's outcome in every order. That
reading keeps what l.269 is for, that each node of a cycle reads without the
others (Example 12, step 3), without holding the relations from third nodes.

## H7. A reference recorded at once has no mutability

Where every candidate records a reference into a class, a use records at once that
the class is a reference (l.252), and a lend that reads it no longer depends on the
use (l.248). The record names no mutability, and a lend's mutability is the lesser
of its own and its place's (l.307), so a `&mut` lend decided before the use has no
result. Witness, two outcomes in the model, one where the lend gives no result
(`examples.rs`, `hole_7_a_reference_recorded_at_once_has_no_mutability`):

```
let r = first(x);        // x: A; first(A) -> &u8, first(B) -> &mut u8
let s = &mut r;
```

Where the candidates agree on the mutability, recording the reference's
constructor, which is its mutability (l.66), would close it there; where they
differ, as here, the lend would have to keep depending on the use.

## T1. A type parameter's bound

Extern fn §1 gives a type parameter one bound, any type or any type the use must
settle, never taken as `!`. The spec states no bound, and its fallback gives `!` to
a class nothing decides (l.299). The model follows the spec, so a declaration's
type parameter that no use settles becomes `!` where extern fn §1 would refuse.

## T2. Which side of a boundary records into the parts

The Korean l.284 records "하한과 경계의 part 사이 관계" when a class takes a
constructor, naming no side of the boundary; the English l.284 records "the
relations between the parts of its lower bounds and boundaries". Line 244 (a) names
only the class `α` of `α ⇝ U` taking a constructor. Read as the English text and
l.244 (a) suggest, a boundary `α ⇝ U` whose `U` takes its constructor after `α`
relates no parts, and the outcome then depends on the order of solving (review 2,
P4: `fn g(a, b) { g(b, b); 0u8 }` with `g(Some(f1()), Some(f2()))`). The model reads
the Korean text: the parts are related once both sides have taken a constructor.

## T3. A structural coercion that changes members and parts

The spec composes at most one structural row (l.114), and its object and enum rows
keep the parts while its part row keeps the members (l.103–111), so `{x: A, y: u8}
⇝ {x: B}` reaches by no composition. Types §3 coerces a structured value by its
kind's shape and part by part at once: the object narrows to the fields the slot
names, and each remaining part coerces as its type does. The model reads types §3
(`design/05-coercion.md`, reading 1).

## T4. Which scalars are numbers

The spec's scalars are Rust's (l.56). Operators §1 makes the numbers the integer
types of 8 to 64 bits and `f64`, so `i128`, `u128`, `isize`, `usize` and `f32` are
scalars of the spec and no number of the RFC. The model follows the spec.

## T5. Which members a host's type keeps from growing

A kind gives a member only where the member is visible: an object's fields and an
enum's variants are visible since the body's text writes them, and the members of a
type the host gives are not (l.282). Read by the member, the field `extra` in
`a.extra` is written by the text and visible wherever `a`'s type came from, so the
clause about the host's type changes no outcome: a class holding the host's type
itself has an equation, which a kind cannot grow (Example 8). Patterns §2 has an
enum grow only "where every use of the enum is visible", and an enum not visible
there does not grow; read so, a class with a lower bound the host gives grows by no
kind. Witness, accepted read by the member with `a: {extra: i32, name: String}`,
refused read by the sources (`examples.rs`,
`hole_t5_a_field_on_a_class_the_host_bounds`):

```
let a = @cfg;            // @cfg is { name: String }, given by the host
a.extra = 1;
```

Read by the sources, the clause holds of a direct lower bound only: through `let b
= a; b.extra = 1;` the class `b` has no lower bound the host gives, and grows.

## T6. A pattern on a scrutinee of type `!`

Patterns §4 has a scrutinee of type `!` reach no arm, each slot of its pattern
deciding nothing of its type. In the spec, a projection of the pattern gives the
scrutinee's class a kind known in part (l.311–315), and its value must meet that
kind (l.282). `!` has no part, so it meets none, and the program is refused; the
spec states no exception for `!`. Witness, refused by the model (`examples.rs`,
`hole_t6_a_pattern_on_a_scrutinee_of_type_never`):

```
let n = never();         // never() -> !
if let Some(x) = n { }
```
