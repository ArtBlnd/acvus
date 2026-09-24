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

/// `ty` with every lifetime it names at `'static`: the form a marker names a
/// type at, while the glue hands the handler the type it wrote at the call's
/// lifetime (RFC-0079 rule 6).
pub fn at_static(ty: &Type) -> Type {
    let mut ty = ty.clone();
    syn::visit_mut::VisitMut::visit_type_mut(&mut Static, &mut ty);
    ty
}

/// As `at_static`, for a predicate a `'static` impl restates.
pub fn predicate_at_static(predicate: &syn::WherePredicate) -> syn::WherePredicate {
    let mut predicate = predicate.clone();
    syn::visit_mut::VisitMut::visit_where_predicate_mut(&mut Static, &mut predicate);
    predicate
}

struct Static;

impl syn::visit_mut::VisitMut for Static {
    fn visit_lifetime_mut(&mut self, lifetime: &mut syn::Lifetime) {
        *lifetime = syn::Lifetime::new("'static", lifetime.apostrophe);
    }
}

/// `Vec<_>` -> `Vec<()>`: an inferred argument in a registry's type list
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
