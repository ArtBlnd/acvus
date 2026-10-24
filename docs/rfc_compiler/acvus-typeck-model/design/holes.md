# Holes

## A cast narrower than its node

A declared cast's parameter writes its node with distinct type parameters as its
parts, and its result writes those alone, so every type of a node goes up by its cast
(`TreeError::NarrowerThanNode`, `ResultUnbound`). Before, nothing required it:
`Vec<u8> → Bytes` made `Bytes` the node above `Vec`, but `Vec<u16>` went up by no
cast, and `A → Box<T>` decided no `T`, so a join the tree gave had no result. A
narrower step is a conversion, which is no part of the tree.

## A value whose path up meets two variants of `Error`

`?` makes a variant of `Error` for each error type's name, and two `Error`s join by
uniting their variants, raising no part, so `Error{B(B), D(D)}` was a type where a
cast goes from `B` to `D`. A value of `A`, with casts `A → B → D`, then reached it by
two compositions: `A → B` then the wrap in `B`, and `A → B → D` then the wrap in `D`,
by the wrap rows of the Coercion table, though the composition toward a given type is
unique. An `Error`, which `?` alone makes, is written only where a value leaves the
program, and `?` wraps by a split, so the wrap rows are removed and no value of
another type reaches an `Error` by a coercion: what enters a written `Error`, a
parameter's or the embedding's result's, is an `Error` already.

## A view at a boundary

A view impl is a cast from a reference to a container to a reference to a slice or
`str`, and a reference's node is its mutability and its target's constructor. A
reference's constructor is its mutability alone, so where two references meet, they
have one constructor, and their targets are related by the split. At a boundary `&V ⇝
&[u8]` records `V ⇝ [u8]` toward a type the declaration writes, which decides no
value, so `V` takes `Vec⟨u8⟩` from `make()`, and the whole value reaches `&[u8]` by
the view. Witness, accepted by the model
(`cases/holes/a-view-at-a-boundary-is-taken-whole.case`):

```
let v = make();          // make() -> Vec⟨u8⟩; the view as_slice(&Vec⟨T⟩) -> &[T]
take(&v);                // take(&[u8])
```

Where two references meet, `&A ⊔ &C` is `&B` by the casts `&A → &B` and `&C → &B`,
while two lower bounds `&A ≤ R` and `&C ≤ R` give `R` the constructor `&` and equate
its target, an unknown, with `A` and `C`, so they have no value. A reference is no
structural type, so its target is invariant where references meet, and the two are
left so.

## A reborrow at a split

Where `&mut X` met `&Y` at a lower bound or a boundary, the constructors differed and
no cast linked them, so nothing related `X` and `Y`: in `look(&mut v)` with
`look(&Vec⟨u8⟩)` and `v = Vec::new()`, `v`'s argument had no relation and fell back
to `!`, and `&mut Vec⟨!⟩ ⇝ &Vec⟨u8⟩` failed. The model accepts it
(`cases/holes/a-reborrow-at-a-boundary-links-the-targets.case`).

## A lambda's node that reads a capture

A lambda was a binding group of its own, solved before the group in which a use
records its C, which for a lambda is its enclosing body's. A capture's type is a
class of the enclosing group, which the lambda shared and did not solve, and no row
made anything wait for what the enclosing group records into a shared class. So a
lambda's use that read a capture read it as nothing when the lambda was solved. The
position then decided nothing, every candidate remained, and the use was not decided:
it stayed in C and was decided in the copy, where `X` is `A`.

```
let x = make();          // make() -> A
let f = || probe(x);     // probe(A), probe(B)
f();
```

A lambda's captured environment is the enclosing group's: `probe` names the capture
`X`, so it is the enclosing group's, decided there once, and reads `X` as `A`. One
outcome, accepted, in every order (`cases/holes/a-lambda-reads-a-capture.case`).

## Two uses that read one class

A use depended on the nodes that record relations into the classes at the positions
it reads, and a class on every node at whose position it lay. So two uses that both
read a class depended on each other, each broke its dependency on the other, and each
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

`probe` passes `x` into a parameter its declarations write, a boundary out of `X`,
and `x + y` depended on it, since while only LitVars enter `X`, `?N ⇝ U`
unifies it with the parameter. So `probe` read `X` without `x + y`'s
relations, as `?a` alone, took the default `i32`, and `X ⇝ i32` failed once
`X` was `u16`: refused in every order. A reading that left that dependency out
whatever entered `X` made `let x = 0; n(x); m(x);` depend on the order.

A class waits for a node whose only relation into it is a boundary out of it only
while only LitVars enter it, judged as whether nothing enters it is, the condition of
the fill that boundary would take part in. `X` has the lower bound `R1`, which waits
for `x + y`, so it does not wait for `probe`; `probe` waits for `x + y` and reads
`u16`: accepted in every order, `probe(u16)`
(`cases/holes/two-uses-that-read-one-class.case`). `n(x); m(x)`, where only LitVars
enter `X`, keeps its dependencies and its one refusal
(`cases/spec/two-uses-that-fill-one-literal.case`).

## A structural coercion that changes members and parts

`{x: E{V(!)}, y: u8} ⇝ {x: E{V(!), W(!)}}` reaches by one structural stage.

## Which scalars are numbers

The scalars are the integers of 8 to 64 bits, `f32`, `f64`, `bool` and `char`.

## Which members a host's type keeps from growing

A kind adds a member to a type another source gives only where that member is visible
to this body, and a field the text writes is visible, so a class a type the host
gives is a lower bound of grows by it. The class holding the host's type itself has
an equation, which a kind cannot grow. Accepted with `a: {extra: i32, name: String}`
(`cases/holes/a-field-on-a-class-the-host-bounds.case`):

```
let a = @cfg;            // @cfg is { name: String }, given by the host
a.extra = 1;
```

The reading by the sources, under which the host's type grew by no kind, is
removed with its case.

## A pattern on a scrutinee of type `!`

A projection of the pattern gives the scrutinee's class a kind known in part, and its
value must meet that kind. `!` has no part, so it met none, and the program was
refused. `!`, which has no value, meets every kind, the parts that kind names being
`!`, so the program is accepted with `n: !` and `x: !`
(`cases/holes/a-pattern-on-a-scrutinee-of-type-never.case`). The parts being `!` keep
a binding whose fallback is `!` from passing on what was written into it: `let t; t.0
= 1u8; takes_u8(t);` is refused where `1u8` enters `t.0`
(`cases/spec/never-a-written-binding-of-no-source.case`):

```
let n = never();         // never() -> !
if let Some(x) = n { }
```

## What a binding reads of its path

A binding reads its position where its path passes no reference, and lends it where
the path passes one. Whether the path passes one is what the projections on it
became: each applies to its operand's target where the operand is a reference. A
binding depended on the nodes that record into its place, other than projections of a
field, a component, an index or a variant, which are the projections on its path, so
a binding decided before its path's projection read its position, and one decided
after lent it: two outcomes, `x: u8` and `x: &u8`, which a reading of the model took
apart by making the binding wait for its path (`BindingPath::AfterPath`, removed).

```
let o = Some(1u8);
if let Some(x) = &o { }
```

The exclusion is of the projections whose operand is the class read, and the nodes
recording into a class are reached along the parts of the two sides that a relation
not yet split will link. The binding's position, the payload of `&o`'s target, is so
linked to `O`'s payload, and the binding waits for its path's projection: one
outcome, `x: &u8` (`cases/holes/a-binding-behind-a-reference.case`).

## What a broken dependency's relations wait for

Only the relations recorded into what the node that broke the dependency reads wait
for it, as I\* takes what a node records, and a LitVar's default narrows only where
no other step is left. The witness is `even(10u8)` with `even` and `odd` calling each
other, two outcomes, and five with an `i32` candidate of each, now one, accepted.

## A reference recorded at once has no mutability

Where every candidate records a reference into a class, a use's common part records
at once that the class is a reference, and a lend that reads it no longer depends on
the use. The record names no mutability, and a lend's mutability is the lesser of its
own and its place's, so before the use is decided a `&mut` lend's read decides
nothing, and the lend is not decided until it does. One outcome, `&mut r` lent `&`
through `&`, accepted
(`cases/holes/a-reference-recorded-at-once-has-no-mutability.case`):

```
let r = first(x);        // x: A; first(A) -> &u8, first(B) -> &mut u8
let s = &mut r;
```

## A capture written after a use reads it

A lambda's write to a capture reached by name was a meeting, and stayed in the
lambda's C; it was recorded into the capture's class only when a call of the lambda's
value was decided and made its instance. Until then nothing at the capture's class
waited for that call: the class lay at no position of it. So the capture's class
could be solved before the call or after it, and so could a use reading it. The model
gave three outcomes: `probe` decided as `probe(B)`, accepted; as `probe(A)`, accepted
with `x: B`, `B ⇝ A` being the conversion; and as `probe(A)`, refused with `x: A`, at
the instance's `V ≤ X`, `X` solved before the instance recorded it:

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
recorded once in the enclosing group, and `V` lies at the call's positions: `X` and
`probe` wait for the call. With `probe(x)`, one outcome, `x: B` and
`probe(B)`, accepted (`cases/holes/a-capture-written-after-a-use-it-reads.case`);
without it, one outcome, `x: B`, accepted
(`cases/holes/a-capture-written-after-a-use.case`). A block writing a capture,
`{ x = make_b(); 0u8 }`, is such a lambda called where it lies, and has
one outcome too (`cases/spec/block-a-write-to-a-capture.case`).

## Two lends of one place

A lend depends on the nodes that record into its place's class, other than
projections of a field, a component, an index or a variant, and uses that have
already recorded whether that class is a reference; the nodes that record into a
class include those at whose position it lies. Two lends of one place each lie at it,
so each depends on the other, and each breaks its dependency on the other.

A lend records into its place only that it is a reference, and only where it already
reads as one, so it cannot decide whether its place is one. The model leaves out of a
read of whether a class is a reference every node that cannot decide it: a use whose
candidates would record no lower bound or equation into the class; a lend at its
place and a binding at its position; and a call of a function value at its arguments.
A lend, a projection or a binding waits for the nodes that can decide whether its
class is a reference.
