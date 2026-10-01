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
    // The call's own effect. Unstated, it is opaque, the effect that is
    // always sound; a call whose effect is an effect variable's states
    // nothing of its own beyond the work it runs (compiler: extern fn §2).
    let effect = match attr.effect.clone().unwrap_or(Effect::Opaque) {
        Effect::Pure | Effect::Var(_) => quote!(::acvus_extern::decl::Effect::pure()),
        Effect::Opaque => quote!(::acvus_extern::decl::Effect::opaque()),
        Effect::Idempotent => quote! {
            ::acvus_extern::decl::Effect::new(
                ::acvus_extern::decl::Reissue::Idempotent,
                false,
                ::acvus_extern::decl::Contexts::new(),
                ::acvus_extern::decl::Contexts::new(),
                ::acvus_extern::decl::SplitCost::Unavailable,
            )
        },
    };
    let [types, effects, identities] = sig.counts;
    let types = types as usize;
    let params = sig.params.iter().filter(|p| matches!(p.role, Role::Argument)).map(|p| {
        let name = LitStr::new(&p.name.to_string(), p.name.span());
        let ty = sig.at_stand_ins(&p.ty, runtime);
        quote! {{
            let ::acvus_extern::decl::ParamDecl::Argument(ty) =
                <#ty as ::acvus_extern::decl::ParamMode>::decl(&mut __cx);
            (names.intern(#name), ty)
        }}
    });
    let result = match &sig.output {
        Output::Never => quote!(::acvus_extern::decl::DeclTy::Never),
        Output::Value | Output::Borrowed { .. } => {
            let ty = sig.at_stand_ins(&sig.written_output, runtime);
            quote!(<#ty as ::acvus_extern::decl::DeclType>::decl_ty(&mut __cx))
        }
    };
    quote! {
        impl<#runtime: ::acvus_extern::runtime::Runtime> ::acvus_extern::register::Declared<#runtime> for #marker {
            /// The function's Rust module path and its name (macros §1).
            fn path(&self, names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::QualifiedRef {
                let module: ::std::vec::Vec<_> =
                    ::core::module_path!().split("::").map(|segment| names.intern(segment)).collect();
                ::acvus_extern::decl::QualifiedRef::qualified(names.namespace(&module), names.intern(#name))
            }

            fn declaration(&self, names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::Declaration {
                let mut __cx = ::acvus_extern::decl::DeclCx { names };
                ::acvus_extern::decl::Declaration {
                    path: <Self as ::acvus_extern::register::Declared<#runtime>>::path(self, names),
                    binders: ::acvus_extern::decl::Binders {
                        types: ::std::vec![::acvus_extern::decl::TyVarBound::Any; #types],
                        effects: #effects,
                        identities: #identities,
                    },
                    params: ::std::vec![#(#params),*],
                    result: #result,
                    effect: #effect,
                    // Every effect variable is taken to be run by the call,
                    // the bound that is always sound: a call may run each
                    // closure it is given.
                    runs: (0..#effects).collect(),
                    facts: ::acvus_extern::decl::DeclaredFacts::default(),
                    requires: ::std::vec::Vec::new(),
                    role: ::acvus_extern::decl::Role::Function,
                }
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
        Output::Value | Output::Never => None,
    };

    // Each argument but the lent one is made by its mode, in the order the
    // parameters are written.
    let mut made = Vec::new();
    for ((i, a, _), mode) in arguments.iter().zip(&modes) {
        if Some(*i) == lent {
            continue;
        }
        made.push(quote! {
            let #a = <#mode as ::acvus_extern::Mode<#runtime>>::arg(#a);
        });
    }
    let call_args = sig.params.iter().enumerate().map(|(i, p)| {
        let name = &p.name;
        match &p.role {
            Role::Capture { .. } => quote!(&self.#name),
            Role::Context => quote!(__context),
            Role::Argument if Some(i) == lent => quote!(__lent),
            Role::Argument => {
                let a = format_ident!("__a{}", i);
                quote!(#a)
            }
        }
    });
    // The context the body is given, where it takes one: borrowed from the
    // body's own, or, finishing with a borrow, the one the finish lends.
    let takes_context = sig.params.iter().any(|p| matches!(p.role, Role::Context));
    let context = takes_context.then(|| quote!(let __context = &__cx;));
    let function = &sig.ident;
    let type_args = sig.type_args(runtime);
    let call = quote!(#function #type_args(#(#call_args),*));

    let bounds = sig.runtime_bounds();
    let bounds = quote!(#(+ #bounds)*);
    let args_tuple = quote!((#(#modes,)*));
    let arity = arguments.len();
    // The attribute states no requirement yet: a site chooses no impl.
    let requirements = 0usize;
    let opened_tuple = quote!((#(#opened,)*));

    let awaited_finish = match sig.output {
        Output::Never => quote! {
            #context
            match #call.await {}
        },
        Output::Value | Output::Borrowed { .. } => quote! {
            #context
            let __result = #call.await;
            __cx.finish(__result)
        },
    };
    if sig.awaited {
        return quote! {
            impl<#runtime: ::acvus_extern::runtime::Runtime #bounds> ::acvus_extern::register::AwaitedFn<#runtime, #arity, #requirements> for #marker {
                fn call<'c>(
                    &'c self,
                    runtime: &'c #runtime,
                    args: [<#runtime as ::acvus_extern::runtime::Runtime>::Value; #arity],
                    site: &'c ::acvus_extern::SiteShape<#runtime, #requirements>,
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
            let context = if takes_context { quote!(__context) } else { quote!(_) };
            quote! {
                __cx.#finish(#a, move |#context, __lent, __sink| {
                    ::acvus_extern::borrowed::Borrows::made(#call, &__sink)
                })
            }
        }
        Output::Never => quote! {
            #context
            match #call {}
        },
        Output::Value => quote! {
            #context
            let __result = #call;
            __cx.finish(__result)
        },
    };
    quote! {
        impl<#runtime: ::acvus_extern::runtime::Runtime #bounds> ::acvus_extern::register::ExternFn<#runtime, #arity, #requirements> for #marker {
            fn call(
                &self,
                runtime: &#runtime,
                args: [<#runtime as ::acvus_extern::runtime::Runtime>::Value; #arity],
                site: &::acvus_extern::SiteShape<#runtime, #requirements>,
            ) -> ::acvus_extern::done::Done<#runtime> {
                const {
                    ::core::assert!(
                        !<#args_tuple as ::acvus_extern::Args<#runtime, #arity>>::CALLS,
                        "an extern fn that runs a callback has an awaited body (macros §1): declare it `async fn`",
                    )
                };
                ::acvus_extern::Entry::new(runtime, args, site).open::<#args_tuple, ()>(|__cx, #opened_tuple, ()| {
                    #(#made)*
                    #finish
                })
            }
        }
    }
}
