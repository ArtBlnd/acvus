# ownership

## 1. Copy

A value copies when its type is a primitive (the integer types, `f64`, `bool`, `char`, `()`), a shared reference `&T`, or a structural type (a tuple, an array, an object) whose every part copies.

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

A `&mut T` does not copy. Each use reborrows it implicitly.

## 6. Where a value ends

A value ends at exactly one place. Unlike Rust, which drops at the end of a scope, a value the program holds is dropped as soon as its liveness ends. A value taken by another holder is dropped by that holder, not by the program.
