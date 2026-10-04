# ownership

## 1. Copy

A value copies when its type is `!`, a primitive (the integer types, `f64`, `bool`, `char`, `()`), a shared reference `&T`, or a transparent structured type (types §1) whose every part copies.

## 2. Clone and move

A value that does not copy follows the move rule: a use moves it. A clone is written, except where section 3 makes it implicit. A value that copies clones by its copy.

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

Storage outside the body, a context and what a reference the body was given or captured lends, is whole whenever the body returns and wherever it may suspend: a part taken out of it through the body is written back on every path to the return and to each call that may suspend, since what awaits a suspended body may abandon it there. A part taken out of the body's own storage through a reference is taken from the place the reference lends, and where that place is not one on every path, it is not taken. What a shared reference lends is never taken from (§5).

## 5. Borrows

A reference lends its storage: `&` shared, `&mut` exclusive. A value holds loans only at the positions of its type: each reference, each function value (what it captured), and each lifetime an extern-defined type declares. A transparent structured type holds its parts' positions, and all elements of a sequence share one. A loan lives while a value holding it is live. A reference lives no longer than any loan its target holds, as Rust's well-formedness requires of a reference's lifetime. So what a path of references reaches, as auto-deref reaches it, is held by the first reference on the path, and is exclusive only where every reference on it is. A reference to a value of a transparent structured type (types §1) is the structure of references to its parts, each of the reference's mutability and held by its loan, down to the parts that are not one: `&(A, B)` is `(&A, &B)`, and `&Option<T>` is `Option<&T>`. A sum's parts include the place of its variant, so a store through a `&mut` to it writes its variant and its payload together. While an exclusive loan lives, its storage is reached only through it; while a shared one lives, its storage is not written, moved or lent exclusively. A reborrow, a reference made through another, shared or exclusive, is a loan of its own on what it reaches, within the loans of the reference it is made through: it lives no longer than they do, and while it lives it excludes as any loan does, the reference it is made through included. A lend of a place whose type is a reference reborrows its target, since a reference's target is never a reference (types §4). A `&mut` does not copy; each use reborrows it implicitly. A loan an argument makes, by a lend or a reborrow, is made where its call starts (effects §6), once every argument is evaluated, so evaluating the other arguments does not meet it.

A call moves loans only as its function type's flows state (extern-fn §4); a lambda's flows are what its body does. Nothing that outlives a body holds a loan on the body's own storage: not its result, and not what it stores into storage outside it. A run's result has a type with no positions.

## 6. Where a value ends

A value ends at exactly one place. Unlike Rust, which drops at the end of a scope, a value the program holds is dropped as soon as its liveness ends. A value taken by another holder is dropped by that holder, not by the program. A part moved out of a value is ended by its holder; the value's other parts each end where their own liveness ends, and the value is not used whole again.

Values carrying one identity share the storage it names (types §3). Each ends as any value does, and the storage ends where the last of them ends.
