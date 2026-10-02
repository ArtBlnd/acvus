# registry

A registry is the declarations one crate contributes, with their handlers. A declaration and its handler come from one signature.

A handler may own captures given when its registry is built. They are typed and `Send + Sync`, absent from the declared type, and state they reach is part of the declaration's effect.

A generic declaration is made concrete at held values. A registry may name whole concrete types to specialize (compiler: types §8): each generic declaration a named type fits is made concrete once more at it, under the same path, with the marked type where the declaration writes the named type. A type variable whose bound excludes the named type's argument is not made concrete at it. An impl is made concrete the same way.

All registries are combined once into the compiler's input and the runtime's. Combining refuses a declaration with an array type, two declarations of one name whose generic function disagrees, and a type name given two Rust types or a Rust type given two names: the checker tells types apart by name, and the runtime by Rust type.
