# Holes

What encoding the spec found, each with the lines it rests on and the evidence for
it. Three kinds of entry:

- **H**, a hole: the spec gives no result where its own text, its examples or the
  RFCs it cites want one, so l.292 refuses there.
- **T**, a text: the spec's two texts disagree, or the spec and an RFC it cites do.
- **O**, an order: the outcome depends on the order of solving.

Within each kind, the entries are numbered in the order encoding found them. An
entry encoding found where the model departed from the text, and the text does not
have, keeps its number and says so ("Not a hole", "Not an order"), as does an entry
the text has since closed.

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

`?` makes a variant of `Error` for each error type's name (l.339), and two `Error`s
join by uniting their variants, raising no part (l.89), so `Error{B(B), D(D)}` is a
type where a cast goes from `B` to `D`. A value of `A`, with casts `A → B → D`, then
reaches it by two compositions: `A → B` then the wrap in `B`, and `A → B → D` then
the wrap in `D` (l.104, l.111). Line 111 states that the composition toward a given
type is unique, and the two call different functions. The model refuses there
(`CoerceError::Compositions`); taking the composition of the lowest rank would
choose `B`. Whether a program reaches it is not yet shown. A candidate is a whole
value passing into such an `Error`, a part taking no wrap (l.111), by a boundary
toward a type a declaration writes.

## H3. A view at a boundary

A view impl is a cast from a reference to a container to a reference to a slice or
`str` (l.78), and a reference's node is its mutability and its target's constructor
(l.78). A reference's constructor is its mutability alone (l.66), so where two
references meet, they have one constructor, and their targets are related by the
split (l.280). At a boundary this is no hole since 13c726db: `&V ⇝ &[u8]` records
`V ⇝ [u8]` toward a type the declaration writes, which decides no value, so `V`
takes `Vec⟨u8⟩` from `make()`, and the whole value reaches `&[u8]` by the view.
Witness, accepted by the model
(`cases/holes/h3-a-view-at-a-boundary-is-taken-whole.case`):

```
let v = make();          // make() -> Vec⟨u8⟩; the view as_slice(&Vec⟨T⟩) -> &[T]
take(&v);                // take(&[u8])
```

Where two references meet, the hole stays: `&A ⊔ &C` is `&B` by the casts
`&A → &B` and `&C → &B` (l.86), while two lower bounds `&A ≤ R` and `&C ≤ R` give
`R` the constructor `&` and equate its target, an unknown, with `A` and `C`
(l.280).

## H4. A reborrow at a split

Not a hole since 13c726db, which links the targets where `&mut T` passes into `&U`
(l.280). Where `&mut X` met `&Y` at a lower bound or a boundary, the constructors
differed and no cast linked them, so nothing related `X` and `Y`: in `look(&mut v)`
with `look(&Vec⟨u8⟩)` and `v = Vec::new()`, `v`'s argument had no relation and fell
back to `!`, and `&mut Vec⟨!⟩ ⇝ &Vec⟨u8⟩` failed. The text now records `V ⇝
Vec⟨u8⟩` there, whose split records `T ⇝ u8` for `new`'s `T`, and `T`, which
nothing enters, is filled with `u8` (l.284). The model accepts it
(`cases/holes/h4-a-reborrow-at-a-boundary-links-the-targets.case`).

## H5. A lambda's node that reads a capture

Not a hole. A lambda is a binding group of its own (l.23), solved before the group
in which a use records its C (l.215), which for a lambda is its enclosing body's. A
capture's type is a class of the enclosing group (l.212), which the lambda shares
and does not solve (l.213, l.217), and no row makes anything wait for what the
enclosing group records into a shared class (l.240–246). So a lambda's use that
reads a capture reads it as nothing when the lambda is solved. The position then
decides nothing, every candidate remains, and the use is not decided (l.263,
l.335): it stays in C (l.214) and is decided in the copy, where `X` is `A`. Encoding
found it as a hole while the model decided every node it read, one whose read
decides nothing included, and refused there. One outcome, accepted, in every order
(`cases/holes/h5-a-lambda-reads-a-capture.case`):

```
let x = make();          // make() -> A
let f = || probe(x);     // probe(A), probe(B)
f();
```

Read by functions §2, what a lambda's captured environment holds is not
generalized and belongs to where it was written (`design/17-environment.md`,
`Captures::Environment`): `probe` names the capture `X`, so it is the enclosing
group's, decided there once, and reads `X` as `A`. One outcome, accepted, in every
order (`cases/environment/h5-a-lambda-reads-a-capture.case`).

## H6. Example 14's `q` has no value

Not a hole since 13c726db. Example 14 has `q` be `Pair⟨&str, u8⟩`, and the copy of
C of `nest` refused at the recursive call (l.590). The split of the recursive
call's boundary into the receiver equated `Self { l: self, r: 0u8 }`'s first part
`P` with `nest`'s `A`, an invariant part, and with `P ≥ Pair⟨A, B⟩` (l.588) the
copy of `A` in `q.nest(3)`'s instance, `q`'s first part, held `Pair⟨A', B'⟩` and
reached a part of its own (l.271), so `q` had no value. The split now records the
boundaries `P ⇝ A` and `Q ⇝ B` toward the universals, which C keeps (l.588), and
the copy keeps that relation split (`design/09-group.md`, reading 3), so `q` is
`Pair⟨&str, u8⟩`, and the copy of the recursive call's argument,
`Pair⟨Pair⟨&str, u8⟩, u8⟩`, reaches `Pair⟨&str, u8⟩` by no coercion: the model
refuses that whole boundary, where the text names its part `Pair⟨&str, u8⟩ ⇝ &str`,
which the model checks as a part of the whole (`design/12-solving.md`, reading 3).
Witness:
Example 14 itself (`cases/spec/example-14-a-struct-and-recursion-that-grows.case`,
sampled orders).

## H7. Two uses that read one class

A use depends on the nodes that record relations into the classes at the positions
it reads (l.243), and the nodes that record relations into a class are those the
class depends on (l.246): by l.240 (b), the relations not yet recorded into it by a
node at whose position it lies. So two uses that both read a class depend on each
other, each breaks its dependency on the other (l.265), and each reads the class
without the other's relations. Witness, refused in every order
(`cases/holes/h7-two-uses-that-read-one-class.case`):

```
let mut x = 0;
let mut y = 1u16;
while c {
    x = x + y;
    y = y + 1;
}
probe(x);                // probe(u16), probe(i32)
```

`probe` passes `x` into a parameter its declarations write, a boundary out of `X`
(l.213), and `x + y` depends on it, so `probe` reads `X` without `x + y`'s
relations, as `?a` alone, takes the default `i32` (l.335), and `X ⇝ i32` fails once
`X` is `u16`: `x + y`'s relations wait for `probe` (l.265), and `R1`, at `x + y`'s
position, depends on those it has not yet recorded (l.240 (b)), so `X` waits for
`R1`'s `u16` in every order, and is not one only LitVars enter (l.284). The
dependency is not one to leave out: while only LitVars enter `X`, `?N ⇝ U` unifies
it with the parameter (l.280), so `probe` can change what `x + y` reads. A reading
that left out of a node's dependencies the nodes that would record no lower bound,
equation or kind into the class it reads made the outcome depend on the order:
`let x = 0; n(x); m(x);` with `n(u8)`, `n(u16)`, `m(u16)` and `m(&str)` reached
`n(u16)` accepted where `m` was decided first, and `n` refused where `n` was. It is
removed. `design/18-invariant-points.md` drafted an order of solving under which
what `probe` records into `X` is no source until `X`'s own relations leave it
without a type, so that `x + y` does not depend on `probe`; the text's fill
(l.284), which that draft's direction became, leaves this dependency as it is.

## T1. A type parameter's bound

Extern fn §1 gives a type parameter one bound, any type or any type the use must
settle, never taken as `!`. The spec states no bound, and its fallback gives `!` to
a class nothing decides (l.290). The model follows the spec, so a declaration's
type parameter that no use settles becomes `!` where extern fn §1 would refuse.

## T2. Which side of a boundary records into the parts

Before 13c726db, the Korean l.280 recorded "하한과 경계의 part 사이 관계" when a class
took a constructor, naming no side of the boundary, and the English l.280 recorded
"the relations between the parts of its lower bounds and boundaries". Line 240 (a)
names only the class `α` of `α ⇝ U` taking a constructor. Read as the English text
and l.240 (a) suggested, a boundary `α ⇝ U` whose `U` takes its constructor after
`α` relates no parts, and the outcome then depends on the order of solving (review
2, P4: `fn g(a, b) { g(b, b); 0u8 }` with `g(Some(f1()), Some(f2()))`). Both texts
now record at each part of the two sides ("두 쪽의 part마다", "each part of the two
sides"), and the model relates the parts once both sides have taken a constructor.

## T3. A structural coercion that changes members and parts

The spec composes at most one structural row (l.111), and its object and enum rows
keep the parts while its part row keeps the members (l.101–103), so `{x: E{V(!)},
y: u8} ⇝ {x: E{V(!), W(!)}}` reaches by no composition. Types §3 coerces a
structured value by its kind's shape and part by part at once: the object narrows
to the fields the slot names, and each remaining part coerces as its type does,
here by `⇝ₚ` (l.111). The model reads types §3 (`design/05-coercion.md`,
reading 1).

## T4. Which scalars are numbers

The spec's scalars are Rust's (l.56). Operators §1 makes the numbers the integer
types of 8 to 64 bits and `f64`, so `i128`, `u128`, `isize`, `usize` and `f32` are
scalars of the spec and no number of the RFC. The model follows the spec.

## T5. Which members a host's type keeps from growing

A kind gives a member only where the member is visible: an object's fields and an
enum's variants are visible since the body's text writes them, and the members of a
type the host gives are not (l.278). Read by the member, the field `extra` in
`a.extra` is written by the text and visible wherever `a`'s type came from, so the
clause about the host's type changes no outcome: a class holding the host's type
itself has an equation, which a kind cannot grow (Example 8). Patterns §2 has an
enum grow only "where every use of the enum is visible", and an enum not visible
there does not grow; read so, a class with a lower bound the host gives grows by no
kind. Witness, accepted read by the member with `a: {extra: i32, name: String}`,
refused read by the sources (`cases/holes/t5-a-field-on-a-class-the-host-bounds.case`,
`cases/holes/t5-read-by-the-sources.case`):

```
let a = @cfg;            // @cfg is { name: String }, given by the host
a.extra = 1;
```

Read by the sources, the clause holds of a direct lower bound only: through `let b
= a; b.extra = 1;` the class `b` has no lower bound the host gives, and grows.

## T6. A pattern on a scrutinee of type `!`

Patterns §4 has a scrutinee of type `!` reach no arm, each slot of its pattern
deciding nothing of its type. In the spec, a projection of the pattern gives the
scrutinee's class a kind known in part (l.302–306), and its value must meet that
kind (l.278). `!` has no part, so it meets none, and the program is refused; the
spec states no exception for `!`. Witness, refused by the model
(`cases/holes/t6-a-pattern-on-a-scrutinee-of-type-never.case`):

```
let n = never();         // never() -> !
if let Some(x) = n { }
```

## O1. What a binding reads of its path

A binding reads its position where its path passes no reference, and lends it where
the path passes one (l.308). Whether the path passes one is what the projections on
it became: each applies to its operand's target where the operand is a reference
(l.300). But a binding reads "whether its equivalence class is a reference" (l.263),
and depends on the nodes that record into its place, other than projections of a
field, a component, an index or a variant (l.244), which are the projections on
its path. So nothing makes a binding wait for them, and nothing it reads tells it
what they became. Under the dependencies l.244 states, the outcome depends on the
order: a binding decided before its path's projection reads its position, and one
decided after lends it. Witness, two outcomes in the model, `x: u8` and `x: &u8`,
both accepted (`cases/holes/o1-a-binding-behind-a-reference-as-stated.case`):

```
let o = Some(1u8);
if let Some(x) = &o { }
```

Making the binding depend on the projections on its path gives one outcome, `x:
&u8`, as in Rust (`BindingPath::AfterPath`). The exclusion in l.244 is about whether
a class is a reference, which those projections do not decide; what a path passes
is a different question, which l.244 does not ask.

## O2. What a broken dependency's relations wait for

"The relations recorded by a node that a broken dependency pointed to are recorded
once the node that broke it is decided" (l.265). Until then, a third node that no
longer waits for the first, it being decided, reads without them. The witness was
the program of H7 under the reading H7 removes, where `probe` lay outside the cycle
of `x + y` and `y + 1`: decided after `x + y` and before `y + 1`, it read `X` as
`?a` alone and took `i32`, and decided after both, `u16`. Under the spec as stated
`probe` is in that cycle, and a witness whose third node lies outside a cycle is
not yet found.

Read so that a broken dependency's relations are recorded when their node is
decided, and only the value read by the node that broke it leaves them out
(`Deferral::Reading`), no third node reads without them. That reading keeps what
l.265 is for, that each node of a cycle reads without the others (Example 12, step
3), without holding the relations from third nodes.

## O3. A reference recorded at once has no mutability

Not an order. Where every candidate records a reference into a class, a use records
at once that the class is a reference (l.248), and a lend that reads it no longer
depends on the use (l.244). The record names no mutability, and a lend's
mutability is the lesser of its own and its place's (l.298), so before the use is
decided a `&mut` lend's read decides nothing, and the lend is not decided until it
does (l.263). Encoding found it as an order while the model decided such a lend
as giving no result. One outcome, `&mut r` lent `&` through `&`, accepted
(`cases/holes/o3-a-reference-recorded-at-once-has-no-mutability.case`):

```
let r = first(x);        // x: A; first(A) -> &u8, first(B) -> &mut u8
let s = &mut r;
```

## O4. A capture written after a use reads it

A lambda's write to a capture reached by name is a meeting, and stays in the
lambda's C (Example 6, l.440); it is recorded into the capture's class only when a
call of the lambda's value is decided and makes its instance (l.198). Until then
nothing at the capture's class waits for that call: the class lies at no position
of it (l.240 (b)). So the capture's class may be solved before the call or after
it, and so may a use reading it. Witness, three outcomes in the model
(`cases/holes/o4-a-capture-written-after-a-use-it-reads.case`): `probe` decided
as `probe(B)`, accepted; as `probe(A)`, accepted with `x: B`, `B ⇝ A` being the
conversion; and as `probe(A)`, refused with `x: A`, at the instance's `V ≤ X`
(l.292), `X` solved before the instance recorded it:

```
let mut x = make_a();    // make_a() -> A, make_b() -> B; a cast A → B, a conversion B → A
let put = |v| x = v;
probe(x);                // probe(A), probe(B)
put(make_b());
```

Without `probe(x)`, two outcomes: `x: B` accepted, and `x: A` refused at `V ≤ X`.

The invariant broken is that a node recording into a class is seen as the class's
dependency before it records and stays after: a lambda's instance writes classes
its call does not lie at.

Read by the captured environment (`design/17-environment.md`,
`Captures::Environment`), `V` is joined to the capture `X` by `x = v`, so `V ≤ X`
is the enclosing group's, recorded once, and `V` lies at the call's positions:
`X` and `probe` wait for the call. With `probe(x)`, one outcome, `x: B` and
`probe(B)`, accepted; without it, one outcome, `x: B`, accepted (same test).

## O5. Two lends of one place

A lend depends on the nodes that record into its place's class, other than
projections of a field, a component, an index or a variant, and uses that have
already recorded whether that class is a reference (l.244); the nodes that record
into a class include those at whose position it lies (l.246). Two lends of one place
each lie at it, so each depends on the other, and each breaks its dependency on the
other (l.265). Example 12's `&y` and `&y 2` are two such lends. Read as l.244
states, the model reaches two outcomes there: `u16` everywhere, accepted, and
`i32` everywhere, refused at `1u16 ≤ Y`. That was run by changing the model, and no
test keeps it.

A lend records into its place only that it is a reference, and only where it
already reads as one (l.298), so it cannot decide whether its place is one. The
model leaves out of a read of whether a class is a reference every node that cannot
decide it: a use whose candidates would record no lower bound or equation into the
class, which the spec's own example needs (l.259); a lend at its place and a binding
at its position; and a call of a function value at its arguments
(`design/08-node.md`, reading 12). Read so, Example 12 reaches its outcome in every
order. The model takes this with no reading of its own: l.244 as stated is not
encoded.
