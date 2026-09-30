//! What `#[extern_fn]` emits beside the function: its marker, the seal, its
//! declaration and its dispatch. The glue holds no `unsafe` (macros §1): the
//! seal is the one `unsafe impl`, the attribute's assertion that the dispatch
//! opens each argument by its parameter's own mode.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, ItemFn, LitStr};

use crate::attr::{Effect, ExternFnAttr};
use crate::sig::{Output, Role, Sig};

pub fn extern_fn(attr: ExternFnAttr, mut func: ItemFn) -> syn::Result<TokenStream> {
    let sig = Sig::read(&mut func)?;
    if let Some(Effect::Var(var)) = &attr.effect
        && !sig.names_effect_variable(var)
    {
        return Err(syn::Error::new(
            var.span(),
            "an effect is `pure`, `idempotent`, `opaque`, or one of the function's `Var<kind::Effect>` parameters",
        ));
    }
    let marker = func.sig.ident.clone();
    let vis = &func.vis;
    let runtime = format_ident!("__R");

    let captures: Vec<(&Ident, &syn::Type)> = sig
        .params
        .iter()
        .filter_map(|p| match &p.role {
            Role::Capture { held } => Some((&p.name, &**held)),
            _ => None,
        })
        .collect();
    let fields = captures.iter().map(|(name, held)| quote!(#name: #held));
    let constructor = if captures.is_empty() {
        quote! {
            /// The function's marker, which a runtime registers.
            #vis fn new() -> Self {
                #marker {}
            }
        }
    } else {
        let args = captures.iter().map(|(name, held)| quote!(#name: #held));
        let names = captures.iter().map(|(name, _)| name);
        quote! {
            /// The function's marker holding its captures, which a runtime
            /// registers; each call is lent them (registry).
            #vis fn with(#(#args),*) -> Self {
                #marker { #(#names),* }
            }
        }
    };

    let declared = declared(&attr, &sig, &marker, &runtime);
    let dispatch = dispatch(&sig, &marker, &runtime);

    Ok(quote! {
        #func

        #[allow(non_camel_case_types)]
        #vis struct #marker {
            #(#fields,)*
        }

        impl #marker {
            #constructor
        }

        // SAFETY: `#[extern_fn]` wrote the dispatch below from the function's
        // signature: it opens each argument by the mode its parameter's Rust
        // type picks (`ParamMode`), at that type.
        unsafe impl ::acvus_extern::seal::ExternFns for #marker {}

        #declared

        #dispatch
    })
}

fn declared(attr: &ExternFnAttr, sig: &Sig, marker: &Ident, runtime: &Ident) -> TokenStream {
    let name = attr.name.clone().unwrap_or_else(|| LitStr::new(&marker.to_string(), marker.span()));
    // A call whose effect is an effect variable's states nothing of its own:
    // its effect is the work it runs, which every effect variable is taken to
    // be (`decl::declaration`).
    let effect = match attr.effect.clone().unwrap_or(Effect::Opaque) {
        Effect::Pure | Effect::Var(_) => quote!(Pure),
        Effect::Idempotent => quote!(Idempotent),
        Effect::Opaque => quote!(Opaque),
    };
    let [types, effects, identities] = sig.counts;
    let params = sig.params.iter().filter(|p| matches!(p.role, Role::Argument)).map(|p| {
        let name = LitStr::new(&p.name.to_string(), p.name.span());
        let ty = sig.at_stand_ins(&p.ty, runtime);
        quote!((#name, <#ty as ::acvus_extern::decl::ParamMode>::decl(&mut __cx)))
    });
    let result = match &sig.output {
        Output::Never => quote!(::acvus_extern::decl::DeclTy::Never),
        Output::Value(_) | Output::Borrowed { .. } => {
            let ty = sig.at_stand_ins(&sig.written_output, runtime);
            quote!(<#ty as ::acvus_extern::decl::DeclType>::decl_ty(&mut __cx))
        }
    };
    quote! {
        impl<#runtime: ::acvus_extern::runtime::Runtime> ::acvus_extern::register::Declared<#runtime> for #marker {
            fn path(&self, names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::QualifiedRef {
                ::acvus_extern::decl::path(names, ::core::module_path!(), #name)
            }

            fn declaration(&self, names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::Declaration {
                let mut __cx = ::acvus_extern::decl::DeclCx { names };
                let params = ::std::vec![#(#params),*];
                let result = #result;
                ::acvus_extern::decl::declaration(
                    names,
                    <Self as ::acvus_extern::register::Declared<#runtime>>::path(self, names),
                    [#types, #effects, #identities],
                    params,
                    result,
                    ::acvus_extern::decl::Stated::#effect,
                )
            }
        }
    }
}

fn dispatch(sig: &Sig, marker: &Ident, runtime: &Ident) -> TokenStream {
    let arguments: Vec<(usize, Ident, syn::Type)> = sig
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.role, Role::Argument))
        .map(|(i, p)| (i, format_ident!("__a{}", i), sig.at_stand_ins(&p.ty, runtime)))
        .collect();
    let modes: Vec<TokenStream> = arguments
        .iter()
        .map(|(_, _, ty)| quote!(<#ty as ::acvus_extern::decl::ParamMode>::Mode<#runtime>))
        .collect();
    let opened = arguments.iter().map(|(_, a, _)| a);
    let lent = match sig.output {
        Output::Borrowed { from, .. } => Some(from),
        Output::Value(_) | Output::Never => None,
    };

    // Each argument but the lent one is made by its mode, kept where lent, in
    // the order the parameters are written.
    let mut made = Vec::new();
    for ((i, a, _), mode) in arguments.iter().zip(&modes) {
        if Some(*i) == lent {
            continue;
        }
        let keep = format_ident!("__k{}", i);
        made.push(quote! {
            let mut #keep = ::core::option::Option::None;
            let #a = <#mode as ::acvus_extern::Mode<#runtime>>::arg(#a, &mut #keep);
        });
    }
    let call_args = sig.params.iter().enumerate().map(|(i, p)| {
        let name = &p.name;
        match &p.role {
            Role::Capture { .. } => quote!(&self.#name),
            Role::Context => quote!(&__cx),
            Role::Argument if Some(i) == lent => quote!(__lent),
            Role::Argument => {
                let a = format_ident!("__a{}", i);
                quote!(#a)
            }
        }
    });
    let function = &sig.ident;
    let type_args = sig.type_args(runtime);
    let call = quote!(#function #type_args(#(#call_args),*));

    let bounds = sig.runtime_bounds();
    let bounds = quote!(#(+ #bounds)*);
    let args_tuple = quote!((#(#modes,)*));
    let opened_tuple = quote!((#(#opened,)*));

    let awaited_finish = match sig.output {
        Output::Never => quote!(match #call.await {}),
        Output::Value(_) | Output::Borrowed { .. } => quote! {
            let __result = #call.await;
            __cx.finish(__result)
        },
    };
    if sig.awaited {
        return quote! {
            impl<#runtime: ::acvus_extern::runtime::Runtime #bounds> ::acvus_extern::register::AwaitedFn<#runtime> for #marker {
                fn call<'c>(
                    &'c self,
                    runtime: &'c #runtime,
                    args: &'c [<#runtime as ::acvus_extern::runtime::Runtime>::Value],
                    site: &'c ::acvus_extern::SiteShape<#runtime>,
                ) -> impl ::core::future::Future<Output = ::acvus_extern::done::Done<#runtime>> + ::core::marker::Send + 'c {
                    ::acvus_extern::Entry::new(runtime, args, site)
                        .open_awaited::<#args_tuple, (), _>(move |__cx, #opened_tuple, ()| async move {
                            #(#made)*
                            #awaited_finish
                        })
                }
            }
        };
    }

    let finish = match sig.output {
        Output::Borrowed { from, mutable } => {
            let a = format_ident!("__a{}", from);
            let finish = if mutable { quote!(finish_borrowed_mut) } else { quote!(finish_borrowed) };
            let out = sig.at_stand_ins(&sig.written_output, runtime);
            quote!(__cx.#finish::<_, <#out as ::acvus_extern::borrowed::Borrows>::Form>(#a, move |__lent| #call))
        }
        Output::Never => quote!(match #call {}),
        Output::Value(_) => quote! {
            let __result = #call;
            __cx.finish(__result)
        },
    };
    quote! {
        impl<#runtime: ::acvus_extern::runtime::Runtime #bounds> ::acvus_extern::register::ExternFn<#runtime> for #marker {
            fn call(
                &self,
                runtime: &#runtime,
                args: &[<#runtime as ::acvus_extern::runtime::Runtime>::Value],
                site: &::acvus_extern::SiteShape<#runtime>,
            ) -> ::acvus_extern::done::Done<#runtime> {
                ::acvus_extern::Entry::new(runtime, args, site).open::<#args_tuple, ()>(|__cx, #opened_tuple, ()| {
                    #(#made)*
                    #finish
                })
            }
        }
    }
}
