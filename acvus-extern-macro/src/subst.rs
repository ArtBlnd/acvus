//! Type substitution over a `syn::Type` tree.

use syn::{GenericArgument, Ident, PathArguments, Type};

/// Replace every bare single-segment path whose identifier `lookup` maps,
/// recursing into generic arguments, tuples, arrays, and references.
pub fn substitute(ty: &Type, lookup: &dyn Fn(&Ident) -> Option<Type>) -> Type {
    match ty {
        Type::Path(type_path) => {
            if type_path.qself.is_none()
                && type_path.path.segments.len() == 1
                && let seg = &type_path.path.segments[0]
                && matches!(seg.arguments, PathArguments::None)
                && let Some(replacement) = lookup(&seg.ident)
            {
                return replacement;
            }
            let mut new_path = type_path.clone();
            for seg in &mut new_path.path.segments {
                if let PathArguments::AngleBracketed(args) = &mut seg.arguments {
                    for arg in &mut args.args {
                        if let GenericArgument::Type(inner) = arg {
                            *inner = substitute(inner, lookup);
                        }
                    }
                }
            }
            Type::Path(new_path)
        }
        Type::Tuple(tuple) => {
            let mut new_tuple = tuple.clone();
            for elem in &mut new_tuple.elems {
                *elem = substitute(elem, lookup);
            }
            Type::Tuple(new_tuple)
        }
        Type::Array(array) => {
            let mut new_array = array.clone();
            new_array.elem = Box::new(substitute(&array.elem, lookup));
            Type::Array(new_array)
        }
        Type::Reference(reference) => {
            let mut new_ref = reference.clone();
            new_ref.elem = Box::new(substitute(&reference.elem, lookup));
            Type::Reference(new_ref)
        }
        other => other.clone(),
    }
}

/// `List<_>` -> `List<()>`: an inferred argument in a registry's type list
/// stands for any instantiation.
pub fn infer_to_unit(ty: &Type) -> Type {
    match ty {
        Type::Infer(_) => syn::parse_quote! { () },
        Type::Path(type_path) => {
            let mut new_path = type_path.clone();
            for seg in &mut new_path.path.segments {
                if let PathArguments::AngleBracketed(args) = &mut seg.arguments {
                    for arg in &mut args.args {
                        if let GenericArgument::Type(inner) = arg {
                            *inner = infer_to_unit(inner);
                        }
                    }
                }
            }
            Type::Path(new_path)
        }
        other => other.clone(),
    }
}
