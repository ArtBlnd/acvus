# extern types

A type an extern fn defines is extern-defined: an opaque registered type. An object or an enum an extern takes or gives is the language's own, written in its declaration in the shape it has. Each follows the transparency rule (compiler: types §1), and crosses by value or by reference exactly as declared, containers included.

## 1. Objects and enums

A declaration writes an object by its fields and an enum by its name and variants, the language's own shapes (compiler: types §2), closed as written: a value crosses only as the declared shape, and nothing is widened. `Option` and `Result` are the language's primitive types (compiler: types §2).

The language type is the shape, structural and closed as written. The extern holds such a value as the Rust views its shape gives: by value, a struct or an enum of owned fields, each crossing by its own type; by reference, its projection (crossing §2). No view is the value's layout, which is the backend's, and no view is a nominal type to the checker, so a shape a program writes and a shape an extern gives are one kind of value.

## 2. Opaque types

An opaque registered type is known only by its name and the extern fns that take or give it. A type that must be told apart from others of its shape, or that carries an identity, an effect or lifetimes, is opaque (compiler: types §3, extern fn §8).

An effect is a function's. A type carries one only from the work its values hold, and that work is a function value a declaration was given: the type is tainted by that function's effect, which is one of the declaration's effect variables, and the declaration states its own effect as that variable where it runs the work. So a type's effect is always an effect variable; no type is written with a stated effect.

## 3. Conversions

A declared conversion may name any extern-defined type on either side. It is the one way a program's value meets an extern-defined type, and the declared casts give the subtyping of compiler: types §3.
