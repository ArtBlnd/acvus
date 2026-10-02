//! What `#[generic_fn]` emits for a generic function's signature (rfc_extern
//! impls §1-§2, compiler: extern fn §6): its family marker, its declaration,
//! a name with a signature and no body, how it takes its receiver, and its
//! signature at each choice of its variables as keys, which a requirer's
//! call is made by and an impl's is checked against. It emits no dispatch: a
//! call of a generic function is a call of the impl it reaches.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ForeignItemFn, LitStr, Type};

use crate::attr::ExternFnAttr;
use crate::glue::{Signature, signature, signature_checks};
use crate::sig::{Generic, Kind, Role, Sig, at_lifetime};

pub fn generic_fn(attr: ExternFnAttr, mut func: ForeignItemFn) -> syn::Result<TokenStream> {
    refuse_body_keys(&attr, &func)?;
    let sig = Sig::read(&mut func.sig)?;
    if sig.awaited {
        return Err(syn::Error::new_spanned(
            func.sig.asyncness,
            "a generic function has no body: its synchronous and awaited function types are its one signature's",
        ));
    }
    if let Some(param) = sig.params.iter().find(|p| !matches!(p.role, Role::Argument)) {
        return Err(syn::Error::new(
            param.name.span(),
            "a generic function has no body, so no capture, executor or requirement: its parameters are its arguments",
        ));
    }
    if let Some(param) = sig.params.iter().find(|p| !p.place.is_empty()) {
        return Err(syn::Error::new(
            param.name.span(),
            "a generic function states no fact of a place: its impls state their own (compiler: extern fn §6)",
        ));
    }
    let marker = func.sig.ident.clone();
    let vis = &func.vis;
    let name = attr.name.clone().unwrap_or_else(|| LitStr::new(&marker.to_string(), marker.span()));
    let runtime = format_ident!("__R");
    let [types, effects, identities] = sig.counts;
    let types = types as usize;
    let Signature { params, result, lifetimes } = signature(&sig, &runtime);

    // Its signature at its variables, each the Rust type a value of it is,
    // named as written; the runtime is the function's own where it names
    // one.
    let own_runtime = sig.generics.iter().find_map(|generic| match generic {
        Generic::Runtime { ident, .. } => Some(ident.clone()),
        Generic::Var { .. } => None,
    });
    let at_runtime = own_runtime.unwrap_or_else(|| runtime.clone());
    let variables: Vec<_> = sig
        .generics
        .iter()
        .filter_map(|generic| match generic {
            Generic::Var { ident, .. } => Some(ident),
            Generic::Runtime { .. } => None,
        })
        .collect();
    let type_variables = sig.generics.iter().filter_map(|generic| match generic {
        Generic::Var { ident, kind: Kind::Type, .. } => Some(ident),
        _ => None,
    });
    // Each variable at its kind's bound: a type variable is chosen as any
    // type its signature's places admit, a value type or a variable of the
    // requirer's; an effect's and an identity's as a variable of the
    // requirer's.
    let bounded = sig.generics.iter().filter_map(|generic| match generic {
        Generic::Var { ident, kind: Kind::Type, .. } => Some(quote!(#ident: ::acvus_extern::decl::DeclType)),
        Generic::Var { ident, kind: Kind::Effect, .. } => Some(quote!(#ident: ::acvus_extern::decl::EffectVar)),
        Generic::Var { ident, kind: Kind::Identity, .. } => Some(quote!(#ident: ::acvus_extern::decl::IdentityVar)),
        Generic::Runtime { .. } => None,
    });
    // Each part's key, read off the signature at one loan.
    let loan = syn::Lifetime::new("'__p", proc_macro2::Span::call_site());
    let key = |ty: &Type| {
        let ty = at_lifetime(ty, &loan);
        quote!(<#ty as ::acvus_extern::runtime::Place<#loan, #at_runtime>>::Key)
    };
    let arguments: Vec<&Type> = sig.params.iter().map(|p| &p.ty).collect();
    let keys: Vec<TokenStream> = arguments.iter().map(|ty| key(ty)).collect();
    // Each part names a key at the loan: a choice its places do not admit
    // has no signature.
    let places: Vec<TokenStream> = arguments
        .iter()
        .copied()
        .chain([&sig.written_output])
        .map(|ty| {
            let ty = at_lifetime(ty, &loan);
            quote!(#ty: ::acvus_extern::runtime::Place<#loan, #at_runtime>)
        })
        .collect();
    let result_key = key(&sig.written_output);
    let rest = &keys[keys.len().min(1)..];
    let (takes, held) = match arguments.first() {
        None => (quote!(None), quote!(())),
        Some(Type::Reference(reference)) if reference.mutability.is_some() => {
            let held = &reference.elem;
            (quote!(Mut), quote!(#held))
        }
        Some(Type::Reference(reference)) => {
            let held = &reference.elem;
            (quote!(Shared), quote!(#held))
        }
        Some(ty) => (quote!(Value), quote!(#ty)),
    };
    let outlive = sig.generics.iter().filter_map(|generic| match generic {
        Generic::Var { ident, kind: Kind::Type, .. } => Some(quote!(#ident: #loan)),
        _ => None,
    });
    let bounded: Vec<TokenStream> = bounded.collect();

    let checks = signature_checks(&func.sig, None);

    Ok(quote! {
        #checks

        /// The generic function's family marker (impls §1).
        #[allow(non_camel_case_types)]
        #vis struct #marker {}

        // SAFETY: `#[generic_fn]` wrote this family below from the one
        // signature it declares: its path and how it takes its receiver.
        unsafe impl ::acvus_extern::seal::Families for #marker {}

        impl<#runtime: ::acvus_extern::runtime::Runtime> ::acvus_extern::register::GenericDeclared<#runtime> for #marker {
            /// The generic function's Rust module path and its name.
            fn path(names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::QualifiedRef {
                let module: ::std::vec::Vec<_> =
                    ::core::module_path!().split("::").map(|segment| names.intern(segment)).collect();
                ::acvus_extern::decl::QualifiedRef::qualified(names.namespace(&module), names.intern(#name))
            }

            fn declaration(names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::GenericDeclaration {
                ::acvus_extern::decl::GenericDeclaration {
                    path: <Self as ::acvus_extern::register::GenericDeclared<#runtime>>::path(names),
                    binders: ::acvus_extern::decl::Binders {
                        types: ::std::vec![::acvus_extern::decl::TyVarBound::Any; #types],
                        effects: #effects,
                        identities: #identities,
                    },
                    params: ::std::vec![#(#params),*],
                    result: #result,
                    lifetimes: #lifetimes,
                }
            }
        }

        impl ::acvus_extern::instance::GenericFunction for #marker {
            type Takes = ::acvus_extern::instance::receiver::#takes;
        }

        // SAFETY: `#[generic_fn]` read these keys off the one signature it
        // declares, at every choice of its variables, each at its kind's
        // bound, and at every loan.
        unsafe impl<#loan, #at_runtime: ::acvus_extern::runtime::Runtime, #(#bounded),*> ::acvus_extern::instance::FamilyAt<#loan, #at_runtime, (#(#variables,)*)> for #marker
        where
            #(#outlive,)*
            #(#places,)*
        {
            type Params = (#(#keys,)*);
            type Result = #result_key;
            type Rest = (#(#rest,)*);
        }

        // SAFETY: as above, of the same signature: its receiver, and its
        // type variables in order.
        unsafe impl<#at_runtime: ::acvus_extern::runtime::Runtime, #(#bounded),*> ::acvus_extern::instance::Holds<#at_runtime, (#(#variables,)*)> for #marker {
            type Held = #held;

            fn pattern(names: &::acvus_extern::decl::Interner) -> ::std::vec::Vec<::acvus_extern::decl::DeclTy> {
                ::std::vec![#(<#type_variables as ::acvus_extern::decl::DeclType>::decl_ty(names)),*]
            }
        }
    })
}

/// A generic function states nothing a body states (compiler: extern fn §6),
/// so of the attribute's keys it takes its name alone.
fn refuse_body_keys(attr: &ExternFnAttr, func: &ForeignItemFn) -> syn::Result<()> {
    let stated = [
        ("effect", attr.effect.is_some()),
        ("ends", attr.ends.is_some()),
        ("relation", !attr.relations.is_empty()),
        ("combine", attr.combine.is_some()),
        ("sync", attr.sync.is_some()),
        ("impl_of", attr.impl_of.is_some()),
        ("generic", attr.generic.is_some()),
    ];
    match stated.iter().find(|(_, stated)| *stated) {
        Some((key, _)) => Err(syn::Error::new_spanned(
            &func.sig.ident,
            format!("a generic function has no body, so it states no `{key}`: its impls state their own"),
        )),
        None => Ok(()),
    }
}

