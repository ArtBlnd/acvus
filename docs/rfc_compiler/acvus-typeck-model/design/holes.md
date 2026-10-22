# Holes

What encoding the spec found, each with the lines it rests on and the evidence for
it. Three kinds of entry:

- **H**, a hole: the spec gives no result where its own text, its examples or the
  RFCs it cites want one, so l.290 refuses there.
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

Not a hole since 17d0be51. `?` makes a variant of `Error` for each error type's name
(l.337), and two `Error`s join by uniting their variants, raising no part (l.89), so
`Error{B(B), D(D)}` was a type where a cast goes from `B` to `D`. A value of `A`,
with casts `A → B → D`, then reached it by two compositions: `A → B` then the wrap
in `B`, and `A → B → D` then the wrap in `D`, by the wrap rows of the Coercion
table, though the composition toward a given type is unique (l.109). Since
17d0be51 an `Error`, which `?` alone makes, is written only where a value leaves the
program (l.91), and `?` wraps by a split (l.278, l.337), so the wrap rows are removed
and no value of another type reaches an `Error` by a coercion: what enters a written
`Error`, a parameter's or the embedding's result's, is an `Error` already.

## H3. A view at a boundary

A view impl is a cast from a reference to a container to a reference to a slice or
`str` (l.78), and a reference's node is its mutability and its target's constructor
(l.78). A reference's constructor is its mutability alone (l.66), so where two
references meet, they have one constructor, and their targets are related by the
split (l.278). At a boundary this is no hole since 13c726db: `&V ⇝ &[u8]` records
`V ⇝ [u8]` toward a type the declaration writes, which decides no value, so `V`
takes `Vec⟨u8⟩` from `make()`, and the whole value reaches `&[u8]` by the view.
Witness, accepted by the model
(`cases/holes/h3-a-view-at-a-boundary-is-taken-whole.case`):

```
let v = make();          // make() -> Vec⟨u8⟩; the view as_slice(&Vec⟨T⟩) -> &[T]
take(&v);                // take(&[u8])
```

Where two references meet, `&A ⊔ &C` is `&B` by the casts `&A → &B` and
`&C → &B` (l.86), while two lower bounds `&A ≤ R` and `&C ≤ R` give `R` the
constructor `&` and equate its target, an unknown, with `A` and `C` (l.278), so
they have no value. 준량 judged this the intended reading on 2026-10-22: a
reference is no structural type, so its target is invariant where references
meet, and the two are left so.

## H4. A reborrow at a split

Not a hole since 13c726db, which links the targets where `&mut T` passes into `&U`
(l.278). Where `&mut X` met `&Y` at a lower bound or a boundary, the constructors
differed and no cast linked them, so nothing related `X` and `Y`: in `look(&mut v)`
with `look(&Vec⟨u8⟩)` and `v = Vec::new()`, `v`'s argument had no relation and fell
back to `!`, and `&mut Vec⟨!⟩ ⇝ &Vec⟨u8⟩` failed. The text now records `V ⇝
Vec⟨u8⟩` there, whose split records `T ⇝ u8` for `new`'s `T`, and `T`, which
nothing enters, is filled with `u8` (l.282). The model accepts it
(`cases/holes/h4-a-reborrow-at-a-boundary-links-the-targets.case`).

## H5. A lambda's node that reads a capture

Not a hole. Before 8978eb70 a lambda was a binding group of its own (l.23), solved
before the group in which a use records its C (l.213), which for a lambda is its
enclosing body's. A capture's type is a class of the enclosing group (l.210), which
the lambda shared and did not solve (l.211, l.215), and no row made anything wait
for what the enclosing group records into a shared class (l.238–244). So a
lambda's use that read a capture read it as nothing when the lambda was solved.
The position then decided nothing, every candidate remained, and the use was not
decided (l.261, l.333): it stayed in C (l.212) and was decided in the copy, where
`X` is `A`. Encoding found it as a hole while the model decided every node it read,
one whose read decides nothing included, and refused there.

```
let x = make();          // make() -> A
let f = || probe(x);     // probe(A), probe(B)
f();
```

Since 8978eb70 a lambda's captured environment is the enclosing group's (l.186):
`probe` names the capture `X`, so it is the enclosing group's, decided there once,
and reads `X` as `A`. One outcome, accepted, in every order
(`cases/holes/h5-a-lambda-reads-a-capture.case`).

## H6. Example 14's `q` has no value

Not a hole since 13c726db. Example 14 has `q` be `Pair⟨&str, u8⟩`, and the copy of
C of `nest` refused at the recursive call (l.588). The split of the recursive
call's boundary into the receiver equated `Self { l: self, r: 0u8 }`'s first part
`P` with `nest`'s `A`, an invariant part, and with `P ≥ Pair⟨A, B⟩` (l.586) the
copy of `A` in `q.nest(3)`'s instance, `q`'s first part, held `Pair⟨A', B'⟩` and
reached a part of its own (l.269), so `q` had no value. The split now records the
boundaries `P ⇝ A` and `Q ⇝ B` toward the universals, which C keeps (l.586), and
the copy keeps that relation split (`design/09-group.md`, reading 3), so `q` is
`Pair⟨&str, u8⟩`, and the copy of the recursive call's argument,
`Pair⟨Pair⟨&str, u8⟩, u8⟩`, reaches `Pair⟨&str, u8⟩` by no coercion: the model
refuses that whole boundary, where the text names its part `Pair⟨&str, u8⟩ ⇝ &str`,
which the model checks as a part of the whole (`design/12-solving.md`, reading 3).
Witness:
Example 14 itself (`cases/spec/example-14-a-struct-and-recursion-that-grows.case`,
sampled orders).

## H7. Two uses that read one class

Not a hole since 09aef04e. Before it, a use depended on the nodes that record
relations into the classes at the positions it reads (l.241), and a class on every
node at whose position it lay (l.238 (b)). So two uses that both read a class
depended on each other, each broke its dependency on the other (l.263), and each
read the class without the other's relations:

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
(l.211), and `x + y` depended on it, since while only LitVars enter `X`, `?N ⇝ U`
unifies it with the parameter (l.278). So `probe` read `X` without `x + y`'s
relations, as `?a` alone, took the default `i32` (l.333), and `X ⇝ i32` failed once
`X` was `u16`: refused in every order. A reading that left that dependency out
whatever entered `X` made `let x = 0; n(x); m(x);` depend on the order.

Since 09aef04e a class waits for a node whose only relation into it is a boundary
out of it only while only LitVars enter it, judged as whether nothing enters it is
(l.238 (b)), the condition of the fill that boundary would take part in (l.282).
`X` has the lower bound `R1`, which waits for `x + y`, so it does not wait for
`probe`; `probe` waits for `x + y` and reads `u16`: accepted in every order,
`probe(u16)` (`cases/holes/h7-two-uses-that-read-one-class.case`). `n(x); m(x)`,
where only LitVars enter `X`, keeps its dependencies and its one refusal
(`cases/spec/l240-two-uses-that-fill-one-literal.case`).

## T1. A type parameter's bound

Extern fn §1 gives a type parameter one bound, any type or any type the use must
settle, never taken as `!`. The spec states no bound, and its fallback gives `!` to
a class nothing decides (l.288). The model follows the spec, so a declaration's
type parameter that no use settles becomes `!` where extern fn §1 would refuse.

## T2. Which side of a boundary records into the parts

Before 13c726db, the Korean l.278 recorded "하한과 경계의 part 사이 관계" when a class
took a constructor, naming no side of the boundary, and the English l.278 recorded
"the relations between the parts of its lower bounds and boundaries". Line 238 (a)
names only the class `α` of `α ⇝ U` taking a constructor. Read as the English text
and l.238 (a) suggested, a boundary `α ⇝ U` whose `U` takes its constructor after
`α` relates no parts, and the outcome then depends on the order of solving (review
2, P4: `fn g(a, b) { g(b, b); 0u8 }` with `g(Some(f1()), Some(f2()))`). Both texts
now record at each part of the two sides ("두 쪽의 part마다", "each part of the two
sides"), and the model relates the parts once both sides have taken a constructor.

## T3. A structural coercion that changes members and parts

The spec composes at most one structural row (l.109), and its object and enum rows
keep the parts while its part row keeps the members (l.101–103), so `{x: E{V(!)},
y: u8} ⇝ {x: E{V(!), W(!)}}` reaches by no composition. Types §3 coerces a
structured value by its kind's shape and part by part at once: the object narrows
to the fields the slot names, and each remaining part coerces as its type does,
here by `⇝ₚ` (l.109). The model reads types §3 (`design/05-coercion.md`,
reading 1).

## T4. Which scalars are numbers

The spec's scalars are Rust's (l.56). Operators §1 makes the numbers the integer
types of 8 to 64 bits and `f64`, so `i128`, `u128`, `isize`, `usize` and `f32` are
scalars of the spec and no number of the RFC. The model follows the spec.

## T5. Which members a host's type keeps from growing

A kind gives a member only where the member is visible: an object's fields and an
enum's variants are visible since the body's text writes them, and the members of a
type the host gives are not (l.276). Read by the member, the field `extra` in
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

Not a text since 87efde71. Patterns §4 has a scrutinee of type `!` reach no arm,
each slot of its pattern deciding nothing of its type. In the spec, a projection of
the pattern gives the scrutinee's class a kind known in part (l.300–304), and its
value must meet that kind (l.276). `!` has no part, so it met none, and the program
was refused; the spec stated no exception for `!`. Since 87efde71 `!`, which has
no value, meets every kind, the parts that kind names being `!` (l.276, since
27a9a960), so the program is accepted with `n: !` and `x: !`
(`cases/holes/t6-a-pattern-on-a-scrutinee-of-type-never.case`). The parts being
`!` keep a binding whose fallback is `!` from passing on what was written into it:
`let t; t.0 = 1u8; takes_u8(t);` is refused where `1u8` enters `t.0`
(`cases/spec/never-a-written-binding-of-no-source.case`):

```
let n = never();         // never() -> !
if let Some(x) = n { }
```

## O1. What a binding reads of its path

Not an order since c30509e3. A binding reads its position where its path passes no
reference, and lends it where the path passes one (l.306). Whether the path passes
one is what the projections on it became: each applies to its operand's target
where the operand is a reference (l.298). A binding depended on the nodes that
record into its place, other than projections of a field, a component, an index or
a variant (l.242), which are the projections on its path, so a binding decided
before its path's projection read its position, and one decided after lent it: two
outcomes, `x: u8` and `x: &u8`, which a reading of the model took apart by making
the binding wait for its path (`BindingPath::AfterPath`, removed).

```
let o = Some(1u8);
if let Some(x) = &o { }
```

Since c30509e3 the exclusion in l.242 and l.261 is of the projections whose operand
is the class read, and the nodes recording into a class are reached along the
parts of the two sides that a relation not yet split will link (l.244). The
binding's position, the payload of `&o`'s target, is so linked to `O`'s payload,
and the binding waits for its path's projection: one outcome, `x: &u8`
(`cases/holes/o1-a-binding-behind-a-reference.case`).

## O2. What a broken dependency's relations wait for

"The relations recorded by a node that a broken dependency pointed to are recorded
once the node that broke it is decided" (l.263). Until then, a third node that no
longer waits for the first, it being decided, reads without them. The witness was
the program of H7 under the reading H7 removes, where `probe` lay outside the cycle
of `x + y` and `y + 1`: decided after `x + y` and before `y + 1`, it read `X` as
`?a` alone and took `i32`, and decided after both, `u16`. Under the spec as stated
`probe` is in that cycle, and a witness whose third node lies outside a cycle is
not yet found.

Read so that a broken dependency's relations are recorded when their node is
decided, and only the value read by the node that broke it leaves them out
(`Deferral::Reading`), no third node reads without them. That reading keeps what
l.263 is for, that each node of a cycle reads without the others (Example 12, step
3), without holding the relations from third nodes.

## O3. A reference recorded at once has no mutability

Not an order. Where every candidate records a reference into a class, a use's
common part records at once that the class is a reference (l.246), and a lend that
reads it no longer depends on the use (l.242). The record names no mutability, and a lend's
mutability is the lesser of its own and its place's (l.296), so before the use is
decided a `&mut` lend's read decides nothing, and the lend is not decided until it
does (l.261). Encoding found it as an order while the model decided such a lend
as giving no result. One outcome, `&mut r` lent `&` through `&`, accepted
(`cases/holes/o3-a-reference-recorded-at-once-has-no-mutability.case`):

```
let r = first(x);        // x: A; first(A) -> &u8, first(B) -> &mut u8
let s = &mut r;
```

## O4. A capture written after a use reads it

Not an order since 8978eb70, where the text takes a lambda's captured environment
as the enclosing binding group's (l.186). Before it, a lambda's write to a capture
reached by name was a meeting, and stayed in the lambda's C (Example 6, l.438); it
was recorded into the capture's class only when a call of the lambda's value was
decided and made its instance (l.196). Until then nothing at the capture's class
waited for that call: the class lay at no position of it (l.238 (b)). So the
capture's class could be solved before the call or after it, and so could a use
reading it. The model gave three outcomes: `probe` decided as `probe(B)`,
accepted; as `probe(A)`, accepted with `x: B`, `B ⇝ A` being the conversion; and
as `probe(A)`, refused with `x: A`, at the instance's `V ≤ X` (l.290), `X` solved
before the instance recorded it:

```
let mut x = make_a();    // make_a() -> A, make_b() -> B; a cast A → B, a conversion B → A
let put = |v| x = v;
probe(x);                // probe(A), probe(B)
put(make_b());
```

Without `probe(x)`, two outcomes: `x: B` accepted, and `x: A` refused at `V ≤ X`.

The invariant broken was that a node recording into a class is seen as the class's
dependency before it records and stays after: a lambda's instance wrote classes
its call does not lie at.

Now `V` is joined to the capture `X` by `x = v`, so `V ≤ X` is the environment's,
recorded once in the enclosing group, and `V` lies at the call's positions (l.238
(b)): `X` and `probe` wait for the call. With `probe(x)`, one outcome, `x: B` and
`probe(B)`, accepted (`cases/holes/o4-a-capture-written-after-a-use-it-reads.case`);
without it, one outcome, `x: B`, accepted
(`cases/holes/o4-a-capture-written-after-a-use.case`). A block writing a capture,
`{ x = make_b(); 0u8 }`, is such a lambda called where it lies (l.341), and has
one outcome too (`cases/spec/block-a-write-to-a-capture.case`).

## O5. Two lends of one place

A lend depends on the nodes that record into its place's class, other than
projections of a field, a component, an index or a variant, and uses that have
already recorded whether that class is a reference (l.242); the nodes that record
into a class include those at whose position it lies (l.244). Two lends of one place
each lie at it, so each depends on the other, and each breaks its dependency on the
other (l.263). Example 12's `&y` and `&y 2` are two such lends. Read as l.242
states, the model reaches two outcomes there: `u16` everywhere, accepted, and
`i32` everywhere, refused at `1u16 ≤ Y`. That was run by changing the model, and no
test keeps it.

A lend records into its place only that it is a reference, and only where it
already reads as one (l.296), so it cannot decide whether its place is one. The
model leaves out of a read of whether a class is a reference every node that cannot
decide it: a use whose candidates would record no lower bound or equation into the
class, which the spec's own example needs (l.257); a lend at its place and a binding
at its position; and a call of a function value at its arguments
(`design/08-node.md`, reading 12). Read so, Example 12 reaches its outcome in every
order. The model takes this with no reading of its own: l.242 as stated is not
encoded.
