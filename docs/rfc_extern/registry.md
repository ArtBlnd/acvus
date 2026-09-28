# registry

A registry is the declarations one crate contributes, with their handlers. A declaration and its handler come from one signature, and a declaration's fields are private.

All registries are combined once into the compiler's input and the runtime's input. Combining refuses a declaration carrying an array type, and two declarations of one name whose generic function disagrees.
