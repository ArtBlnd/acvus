# extern types

A type an extern fn defines is extern-defined: a derived struct, an extern enum, or an opaque registered type. It follows the transparency rule (compiler: types §1), and crosses by value or by reference exactly as declared, containers included.

## 1. Derived structs

A derived struct is extern-defined. Its declared fields may be read, and written at their declared types, which widens nothing.

## 2. Extern enums

An extern enum is closed. A script path naming one of its variants builds a value of that extern type, exactly as declared, and a struct variant's payload is part of it. `Option` and `Result` are the language's primitive types (compiler: types §2).

## 3. Opaque types

An opaque registered type is known only by its name and the extern fns that take or give it.

## 4. Conversions

A declared conversion may name any extern-defined type on either side. It is the one way a program's value meets an extern-defined type, and the declared casts give the subtyping of compiler: types §3.
