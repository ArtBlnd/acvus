# ownership

## 1. Copy

A value copies when its type is `!`, a primitive (the integer types, `f64`, `bool`, `char`, `()`), a shared reference `&T`, or a structural type (a tuple, an array, an object, an `Option`, a `Result`) whose every part copies.

## 2. Clone and move

A value that does not copy follows the move rule: a use moves it. A clone is written, except where section 3 makes it implicit.

## 3. String

A `String` does not copy. For convenience, a use of a `String` that is not its last is an implicit clone, and its last use moves it.

## 4. Initialization

A binding declared without a value is built field by field: a store into a field writes that field, nested fields included. A field is read only where every path to the read has written it, and the value is used whole only where every path to the use has written every field of its type. This departs from Rust, which refuses a partly assigned binding.

```acvus
let a;
if blackbox {
    a.x = 0;
    f(a); // f takes { x }: admitted
}
```

On the one path that reaches `f(a)`, every field of `a`'s type is written before the whole use, so the program compiles.

## 5. Borrows

A reference lends its storage: `&` shared, `&mut` exclusive. A value holds loans only at the positions of its type: each reference, each function value (what it captured), and each lifetime an extern-defined type declares. A structural type holds its parts' positions, and all elements of a sequence share one. A loan lives while a value holding it is live. While an exclusive loan lives, its storage is reached only through it; while a shared one lives, its storage is not written, moved or lent exclusively, and a shared reborrow of a `&mut` keeps its storage so excluded. A `&mut` does not copy; each use reborrows it implicitly.

A call moves loans only as its function type's flows state (extern-fn §4); a lambda's flows are what its body does. A body's result holds no loan on the body's own storage. A run's result has a type with no positions.

## 6. Where a value ends

A value ends at exactly one place. Unlike Rust, which drops at the end of a scope, a value the program holds is dropped as soon as its liveness ends. A value taken by another holder is dropped by that holder, not by the program. A part moved out of a value is ended by its holder; the value's other parts each end where their own liveness ends, and the value is not used whole again.
