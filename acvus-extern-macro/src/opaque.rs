//! `#[derive(Opaque)]`: the author's mark that a Rust type is an
//! extern-defined type, held whole by the runtime's Opaque kind, whatever its
//! shape. It crosses by value through `Crossing`, is lent as itself through
//! `Lend`, and declares itself through `Declared`: its type parameters are
//! its type arguments, except one bounded by `EffectVar`, its effect, and one
//! bounded by `IdentityVar`, its identity (rfc_compiler extern-fn §8).

use proc_macro2::TokenStream;
use quote::quote;
use syn::ext::IdentExt;
use syn::{DeriveInput, GenericParam, Type, TypeParam, TypeParamBound, WherePredicate, parse_quote};

/// What a type parameter of an extern-defined type stands for.
enum Role {
    TypeArg,
    Effect,
    Identity,
}

/// The marker a bound names, by its last path segment; `None` for any other
/// bound.
fn marked_role(bound: &TypeParamBound) -> Option<Role> {
    let TypeParamBound::Trait(bound) = bound else {
        return None;
    };
    match bound.path.segments.last()?.ident.to_string().as_str() {
        "EffectVar" => Some(Role::Effect),
        "IdentityVar" => Some(Role::Identity),
        _ => None,
    }
}

/// The role of `param`, read off its inline bounds and the where clause.
fn role(input: &DeriveInput, param: &TypeParam) -> Role {
    let bounds_this = |ty: &Type| matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident(&param.ident));
    let in_where = input
        .generics
        .where_clause
        .iter()
        .flat_map(|clause| &clause.predicates)
        .filter_map(|predicate| match predicate {
            WherePredicate::Type(typed) if bounds_this(&typed.bounded_ty) => Some(&typed.bounds),
            _ => None,
        })
        .flatten();
    param.bounds.iter().chain(in_where).find_map(marked_role).unwrap_or(Role::TypeArg)
}

pub fn derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let label = name.unraw().to_string();
    let mut type_args = Vec::new();
    let mut effect = None;
    let mut identity = None;
    let mut lifetimes = 0_usize;
    for param in &input.generics.params {
        match param {
            GenericParam::Lifetime(_) => lifetimes += 1,
            GenericParam::Const(param) => {
                return Err(syn::Error::new_spanned(param, "an extern-defined type takes types, effects, identities and lifetimes, not constants"));
            }
            GenericParam::Type(param) => {
                let slot = match role(input, param) {
                    Role::TypeArg => {
                        type_args.push(&param.ident);
                        continue;
                    }
                    Role::Effect => &mut effect,
                    Role::Identity => &mut identity,
                };
                if slot.replace(&param.ident).is_some() {
                    return Err(syn::Error::new_spanned(param, "an extern-defined type carries at most one effect and one identity"));
                }
            }
        }
    }

    let mut generics = input.generics.clone();
    generics.params.insert(0, parse_quote!(__R));
    let predicates = &mut generics.make_where_clause().predicates;
    predicates.push(parse_quote!(__R: ::acvus_extern::Runtime));
    let (_, ty_generics, _) = input.generics.split_for_impl();
    predicates.push(parse_quote!(#name #ty_generics: ::core::marker::Send + ::core::marker::Sync));
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let opaque = quote!(<<__R as ::acvus_extern::Runtime>::Opaque as ::acvus_extern::OpaqueKind<__R>>);
    let value = quote!(<__R as ::acvus_extern::Runtime>::Value);

    let mut declared_generics = input.generics.clone();
    let declared_predicates = &mut declared_generics.make_where_clause().predicates;
    for arg in &type_args {
        declared_predicates.push(parse_quote!(#arg: ::acvus_extern::Declared));
    }
    let effect_term = match effect {
        Some(effect) => {
            declared_predicates.push(parse_quote!(#effect: ::acvus_extern::DeclaredEffect));
            quote!(::core::option::Option::Some(<#effect as ::acvus_extern::DeclaredEffect>::effect()))
        }
        None => quote!(::core::option::Option::None),
    };
    let identity_term = match identity {
        Some(identity) => {
            declared_predicates.push(parse_quote!(#identity: ::acvus_extern::DeclaredIdentity));
            quote!(::core::option::Option::Some(<#identity as ::acvus_extern::DeclaredIdentity>::identity()))
        }
        None => quote!(::core::option::Option::None),
    };
    let (declared_impl, declared_ty, declared_where) = declared_generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics ::acvus_extern::Crossing<__R> for #name #ty_generics #where_clause {
            fn give(self, __rt: &__R) -> #value {
                #opaque::make(__rt, self)
            }

            fn take(__rt: &__R, __value: #value) -> Self {
                #opaque::take(__rt, __value)
            }
        }

        impl #impl_generics ::acvus_extern::Lend<__R> for #name #ty_generics #where_clause {
            type Viewed = Self;

            fn view<__O>(__rt: &__R, __value: &#value, __lent: impl ::core::ops::FnOnce(&Self) -> __O) -> __O {
                #opaque::view(__rt, __value, __lent)
            }

            fn view_mut<__O>(__rt: &__R, __value: &mut #value, __lent: impl ::core::ops::FnOnce(&mut Self) -> __O) -> __O {
                #opaque::view_mut(__rt, __value, __lent)
            }
        }

        impl #declared_impl ::acvus_extern::Declared for #name #declared_ty #declared_where {
            const UNIFORM: bool = false #(|| <#type_args as ::acvus_extern::Declared>::UNIFORM)*;

            fn declare(__cx: &mut ::acvus_extern::Declaring<'_>) -> ::acvus_extern::Ty {
                let __type_args = [#(<#type_args as ::acvus_extern::Declared>::declare(__cx)),*];
                __cx.opaque(::acvus_extern::OpaqueDecl {
                    name: #label,
                    type_args: &__type_args,
                    lifetimes: #lifetimes,
                    effect: #effect_term,
                    identity: #identity_term,
                    uniform: <Self as ::acvus_extern::Declared>::UNIFORM,
                })
            }
        }
    })
}
