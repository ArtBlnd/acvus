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
