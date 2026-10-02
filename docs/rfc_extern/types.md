# extern types

A type an extern fn defines is extern-defined: an opaque registered type. An object or an enum an extern takes or gives is the language's own. Each follows the transparency rule (compiler: types §1), and crosses exactly as declared.

## 1. Objects and enums

A declaration writes an object by its fields and an enum by its name and variants (compiler: types §2), closed as written: a value crosses only as the declared shape, and nothing is widened. The extern holds one by value as a struct or an enum of its parts, each crossing by its own type, and a reference to one as its parts' references (crossing §2). No view is the value's layout, which is the backend's.

## 2. Opaque types

An opaque registered type is known only by its name and the extern fns that take or give it. A type that must be told apart from others of its shape, or that carries an identity, an effect or lifetimes, is opaque (compiler: types §3, extern fn §8). Its type parameters are type variables (runtime contract §1).

A type carries an effect only from the work its values hold, a function value a declaration was given. Its effect is that function's, one of the declaration's effect variables; no type is written with a stated effect.

## 3. Conversions

A declared conversion may name any extern-defined type on either side. It is the one way a program's value meets an extern-defined type; the declared casts give the subtyping of compiler: types §3.
