# registry

A registry is the declarations one crate contributes, with their handlers. A declaration's handler is the Rust function the runtime calls for each call of the declaration. A declaration and its handler come from one signature, and a declaration's fields are private.

A handler may be a Rust closure whose captures are given when its registry is built. The captures are typed and `Send + Sync`, the handler owns them for the lifetime of the combined registries, and they do not appear in the declared type. A read or write of state reachable through the captures is part of the declaration's effect.

*Rejected.* Captures reached through `Any` and a downcast: the type known when the registry was built is lost and re-checked at each call. A separate declaration form for handlers with captures.

A registry's generic declarations are made concrete at the runtime's uniform value: a type parameter stands for any value, held one way. A registry may also name whole concrete types to specialize, such as `Vec<f64>`. Each generic declaration a named type fits is made concrete once more at it, under the same path: the value's own type in place of the uniform one, and the marked type (compiler: types §8) where the declaration writes the named type, as the result as well as a parameter. A type parameter a bound excludes from the named type's argument is not made concrete at it. Such a declaration is the same function made concrete at other types, so it means what the uniform one means. An impl of a generic function is made concrete the same way.

All registries are combined once into the compiler's input and the runtime's input. Combining refuses a declaration carrying an array type, and two declarations of one name whose generic function disagrees.

Across the combined registries, type names and the Rust types behind them are in bijection. Combining refuses a name given two Rust types and a Rust type given two names; two declarations of one pair declare one type.

*Why.* The checker distinguishes types by name and the runtime distinguishes stored values by Rust type; without a bijection, a value stored as one Rust type is read as another under one name.
