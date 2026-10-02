//! What `#[generic_fn]` emits for a generic function's signature (rfc_extern
//! impls §1-§2, compiler: extern fn §6): its family marker, its declaration,
//! a name with a signature and no body, and its signature at each choice of
//! its variables, which an impl's is checked against. It emits no dispatch:
//! a call of a generic function is a call of the impl it reaches.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ForeignItemFn, LitStr};

use crate::attr::ExternFnAttr;
use crate::glue::{Signature, fn_pointer, signature};
use crate::sig::{Generic, Role, Sig};

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
            "a generic function has no body, so no capture and no executor: its parameters are its arguments",
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

    // Its signature at its variables, each named as written and held as a
    // value of it is, `Uniform<R, T>`; the runtime is the function's own
    // where it names one.
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
    let pointer = fn_pointer(&sig, &func.sig.generics, &at_runtime);

    Ok(quote! {
        /// The generic function's family marker (impls §2).
        #[allow(non_camel_case_types)]
        #vis struct #marker {}

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

        impl<#at_runtime: ::acvus_extern::runtime::Runtime, #(#variables),*>
            ::acvus_extern::register::FamilyAt<#at_runtime, (#(#variables,)*)> for #marker
        {
            type Signature = #pointer;
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
