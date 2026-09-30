# registry

A registry is the declarations one crate contributes, with their handlers. A declaration and its handler come from one signature, and a declaration's fields are private.

A registry's generic declarations are made concrete at the runtime's uniform value: a type parameter stands for any value, held one way. A registry may also name whole concrete types to specialize, such as `Vec<f64>`. Each generic declaration a named type fits is made concrete once more at it, under the same path: the value's own type in place of the uniform one, and the marked type (compiler: types §8) where the declaration writes the named type, as the result as well as a parameter. A type parameter a bound excludes from the named type's argument is not made concrete at it. Such a declaration is the same function made concrete at other types, so it means what the uniform one means. An impl of a generic function is made concrete the same way.

All registries are combined once into the compiler's input and the runtime's input. Combining refuses a declaration carrying an array type, and two declarations of one name whose generic function disagrees.
