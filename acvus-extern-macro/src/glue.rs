//! What `#[extern_fn]` emits beside the function: its marker, the seal, its
//! declaration and its dispatch. The glue holds no `unsafe` (macros §1): the
//! seal is the one `unsafe impl`, the attribute's assertion that the dispatch
//! opens each argument by its parameter's own mode.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, ItemFn, LitStr};

use crate::attr::{Effect, ExternFnAttr, ImplOf};
use crate::facts;
use crate::lifetimes::Names;
use crate::sig::{Generic, Output, Role, Sig};

pub fn extern_fn(attr: ExternFnAttr, mut func: ItemFn) -> syn::Result<TokenStream> {
    let sig = Sig::read(&mut func.sig)?;
    if let Some(Effect::Var(var)) = &attr.effect
        && !sig.names_effect_variable(var)
    {
        return Err(syn::Error::new(
            var.span(),
            "an effect is `pure`, `idempotent`, `opaque`, or one of the function's `Var<kind::Effect>` parameters",
        ));
    }
    if let (Some(generic), None) = (&attr.generic, &attr.impl_of) {
        return Err(syn::Error::new(generic.span(), "`generic` marks an impl's kind: it is stated with `impl_of`"));
    }
    if let Some(body) = &attr.sync
        && !sig.awaited
    {
        return Err(syn::Error::new_spanned(body, "`sync` names an `async fn`'s returning body; this function is one"));
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

    let declared = declared(&attr, &sig, &marker, &runtime)?;
    let dispatch = dispatch(&sig, attr.sync.as_ref(), &marker, &runtime);
    let impl_of = attr.impl_of.as_ref().map(|impl_of| signature_is_at_choice(impl_of, &sig, &func.sig.generics, &runtime));

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

        #impl_of
    })
}

fn declared(attr: &ExternFnAttr, sig: &Sig, marker: &Ident, runtime: &Ident) -> syn::Result<TokenStream> {
    let name = attr.name.clone().unwrap_or_else(|| LitStr::new(&marker.to_string(), marker.span()));
    // The call's own effect. Unstated, its reissue is opaque, which is
    // always sound; a call whose effect is an effect variable's states
    // nothing of its own beyond the work it runs. It suspends where the
    // declaration has an awaited body and no returning one (compiler:
    // extern fn §2).
    let (reissue, commutes) = match attr.effect.clone().unwrap_or(Effect::Opaque) {
        Effect::Pure | Effect::Var(_) => (quote!(Pure), true),
        Effect::Idempotent => (quote!(Idempotent), false),
        Effect::Opaque => (quote!(Opaque), false),
    };
    let awaits = sig.awaited;
    let suspends = sig.awaited && attr.sync.is_none();
    let effect = quote! {
        ::acvus_extern::decl::Effect::new(
            ::acvus_extern::decl::Reissue::#reissue,
            #commutes,
            ::acvus_extern::decl::Contexts::new(),
            ::acvus_extern::decl::Contexts::new(),
            ::acvus_extern::decl::SplitCost::Unavailable,
            #suspends,
        )
    };
    let [types, effects, identities] = sig.counts;
    let types = types as usize;
    let Signature { params, result, lifetimes } = signature(sig, runtime);
    let role = match &attr.impl_of {
        None => quote!(::acvus_extern::decl::Role::Function),
        Some(ImplOf { generic_function, .. }) => {
            let kind = match attr.generic {
                Some(_) => quote!(Generic),
                None => quote!(Concrete),
            };
            quote! {
                ::acvus_extern::decl::Role::ImplOf {
                    generic_function: <#generic_function as ::acvus_extern::register::GenericDeclared<#runtime>>::path(names),
                    kind: ::acvus_extern::decl::ImplKind::#kind,
                }
            }
        }
    };
    // The facts the attribute states (compiler: extern fn §3-§4), each of
    // the arguments by their names.
    let argument_names: Vec<&Ident> =
        sig.params.iter().filter(|p| matches!(p.role, Role::Argument)).map(|p| &p.name).collect();
    let ending = facts::ends(attr.ends);
    let relations = facts::relations(&attr.relations, &argument_names)?;
    let places: Vec<&facts::PlaceFacts> =
        sig.params.iter().filter(|p| matches!(p.role, Role::Argument)).map(|p| &p.place).collect();
    let access = facts::access(&places, &argument_names, runtime)?;
    let action = facts::actions(attr.combine.as_ref(), &places, &argument_names, runtime)?;
    Ok(quote! {
        impl<#runtime: ::acvus_extern::runtime::Runtime> ::acvus_extern::register::Declared<#runtime> for #marker {
            /// The function's Rust module path and its name (macros §1).
            fn path(names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::QualifiedRef {
                let module: ::std::vec::Vec<_> =
                    ::core::module_path!().split("::").map(|segment| names.intern(segment)).collect();
                ::acvus_extern::decl::QualifiedRef::qualified(names.namespace(&module), names.intern(#name))
            }

            fn declaration(&self, names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::Declaration {
                ::acvus_extern::decl::Declaration {
                    path: <Self as ::acvus_extern::register::Declared<#runtime>>::path(names),
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
                    awaits: #awaits,
                    facts: ::acvus_extern::decl::DeclaredFacts {
                        lifetimes: ::core::option::Option::Some(#lifetimes),
                        access: ::std::boxed::Box::new([#(#access),*]),
                        action: ::std::boxed::Box::new([#(#action),*]),
                        ends: #ending,
                        relations: ::std::boxed::Box::new([#(#relations),*]),
                        ..::acvus_extern::decl::DeclaredFacts::default()
                    },
                    requires: ::std::vec::Vec::new(),
                    role: #role,
                }
            }
        }
    })
}

/// A signature as a declaration states it: each argument's name and
/// declared type, the result's, and the lifetimes the signature names, each
/// end with its names (ownership §5, extern-fn §4).
pub struct Signature {
    pub params: Vec<TokenStream>,
    pub result: TokenStream,
    pub lifetimes: TokenStream,
}

pub fn signature(sig: &Sig, runtime: &Ident) -> Signature {
    let params = sig.params.iter().filter(|p| matches!(p.role, Role::Argument)).map(|p| {
        let name = LitStr::new(&p.name.to_string(), p.name.span());
        let ty = sig.at_stand_ins(&p.ty, runtime);
        quote!((names.intern(#name), <#ty as ::acvus_extern::decl::ParamMode>::decl(names)))
    }).collect();
    let result = match &sig.output {
        Output::Never => quote!(::acvus_extern::decl::DeclTy::Never),
        Output::Value | Output::Borrowed => {
            let ty = sig.at_stand_ins(&sig.written_output, runtime);
            quote!(<#ty as ::acvus_extern::decl::DeclType>::decl_ty(names))
        }
    };
    // Each end with the lifetime names it writes, read by the declaration's
    // labels (ownership §5, extern-fn §4).
    let names = |names: &Names| {
        let outer = match &names.outer {
            Some(outer) => quote!(::core::option::Option::Some(#outer)),
            None => quote!(::core::option::Option::None),
        };
        let inner = &names.inner;
        (outer, quote!(&[#(#inner),*]))
    };
    let mut ends: Vec<TokenStream> = sig
        .params
        .iter()
        .filter(|p| matches!(p.role, Role::Argument))
        .enumerate()
        .map(|(at, p)| {
            let ty = sig.at_stand_ins(&p.ty, runtime);
            let (outer, inner) = names(&p.names);
            quote! {
                ::acvus_extern::decl::EndNames {
                    at: ::acvus_extern::decl::At::Param(#at),
                    ty: <#ty as ::acvus_extern::decl::ParamMode>::decl(names),
                    outer: #outer,
                    inner: #inner,
                }
            }
        })
        .collect();
    let (outer, inner) = names(&sig.result_names);
    ends.push(quote! {
        ::acvus_extern::decl::EndNames { at: ::acvus_extern::decl::At::Result, ty: #result, outer: #outer, inner: #inner }
    });
    let lifetimes = quote! {
        ::acvus_extern::decl::labelled(&[#(#ends),*])
    };
    Signature { params, result, lifetimes }
}

/// An impl's signature is its generic function's at the choice it writes
/// (extern fn §6), checked by Rust: the function pointer type of its
/// arguments and result is `FamilyAt`'s at that choice, or the impl does not
/// compile. Nothing is checked at run time.
fn signature_is_at_choice(impl_of: &ImplOf, sig: &Sig, generics: &syn::Generics, runtime: &Ident) -> TokenStream {
    let ImplOf { generic_function, choice } = impl_of;
    let own_runtime = sig.generics.iter().find_map(|generic| match generic {
        Generic::Runtime { ident, .. } => Some(ident.clone()),
        Generic::Var { .. } => None,
    });
    let (at_runtime, added) = match own_runtime {
        Some(ident) => (ident, None),
        None => (runtime.clone(), Some(quote!(#runtime: ::acvus_extern::runtime::Runtime))),
    };
    let params = generics.type_params();
    let predicates = generics.where_clause.iter().flat_map(|clause| &clause.predicates);
    let pointer = fn_pointer(sig, generics, &at_runtime);
    quote! {
        const _: () = {
            #[allow(dead_code, non_snake_case)]
            fn __signature_is_at_choice<#(#params,)* #added>()
            where
                #(#predicates,)*
            {
                let _: ::core::marker::PhantomData<
                    <#generic_function as ::acvus_extern::register::FamilyAt<#at_runtime, (#(#choice,)*)>>::Signature,
                > = ::core::marker::PhantomData::<#pointer>;
            }
        };
    }
}

/// The function pointer type of `sig`'s arguments and its result, each
/// type variable as the value it stands for is held, `Uniform<runtime, T>`,
/// and each lifetime `generics` names bound by it: as `FamilyAt` states a
/// generic function's signature, and an impl's is compared with it
/// (extern fn §6).
pub fn fn_pointer(sig: &Sig, generics: &syn::Generics, runtime: &Ident) -> TokenStream {
    let lifetimes = generics.lifetimes().map(|param| &param.lifetime);
    let arguments = sig
        .params
        .iter()
        .filter(|p| matches!(p.role, Role::Argument))
        .map(|p| sig.held(&p.ty, runtime));
    let output = sig.held(&sig.written_output, runtime);
    quote!(for<#(#lifetimes),*> fn(#(#arguments),*) -> #output)
}

/// The dispatch of each body the function has: an awaited body's, with its
/// returning body's where `sync` names one, or a returning body's. Both open
/// one signature's arguments and call a function of it.
fn dispatch(sig: &Sig, sync: Option<&syn::Path>, marker: &Ident, runtime: &Ident) -> TokenStream {
    let arguments: Vec<(usize, Ident, syn::Type)> = sig
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.role, Role::Argument))
        .map(|(i, p)| (i, format_ident!("__a{}", i), sig.at_stand_ins(&p.ty, runtime)))
        .collect();
    let modes: Vec<TokenStream> = arguments
        .iter()
        .map(|(_, _, ty)| quote!(<#ty as ::acvus_extern::decl::ParamMode>::Mode))
        .collect();
    let opened = arguments.iter().map(|(_, a, _)| a);
    let call_args: Vec<TokenStream> = sig
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let name = &p.name;
            match &p.role {
                Role::Capture { .. } => quote!(&self.#name),
                Role::Executor => quote!(__executor),
                Role::Argument => {
                    let a = format_ident!("__a{}", i);
                    quote!(#a)
                }
            }
        })
        .collect();
    // The executor the body is given, where it takes one, borrowed from the
    // body's own.
    let takes_executor = sig.params.iter().any(|p| matches!(p.role, Role::Executor));
    let executor = takes_executor.then(|| quote!(let __executor = &__cx;));
    let type_args = sig.type_args(runtime);
    let function = &sig.ident;
    let call_of = |body: TokenStream| quote!(#body #type_args(#(#call_args),*));

    let bounds = sig.runtime_bounds();
    let bounds = quote!(#(+ #bounds)*);
    let args_tuple = quote!((#(#modes,)*));
    let arity = arguments.len();
    // The attribute states no requirement yet: a site chooses no impl.
    let requirements = 0usize;
    let opened_tuple = quote!((#(#opened,)*));

    let returning = |call: TokenStream| {
        let finish = match sig.output {
            Output::Borrowed => quote! {
                #executor
                let __result = #call;
                __cx.finish_borrowed(__result)
            },
            Output::Never => quote! {
                #executor
                match #call {}
            },
            Output::Value => quote! {
                #executor
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
                    ::acvus_extern::Entry::new(runtime, args, site).open::<#args_tuple, ()>(|__cx, #opened_tuple, ()| {
                        #finish
                    })
                }
            }
        }
    };
    if !sig.awaited {
        return returning(call_of(quote!(#function)));
    }

    let call = call_of(quote!(#function));
    let awaited_finish = match sig.output {
        Output::Never => quote! {
            #executor
            match #call.await {}
        },
        Output::Value | Output::Borrowed => quote! {
            #executor
            let __result = #call.await;
            __cx.finish(__result)
        },
    };
    let returning = sync.map(|body| returning(call_of(quote!(#body))));
    quote! {
        impl<#runtime: ::acvus_extern::runtime::Runtime #bounds> ::acvus_extern::register::AwaitedFn<#runtime, #arity, #requirements> for #marker {
            fn call<'c>(
                &'c self,
                runtime: &'c #runtime,
                args: [<#runtime as ::acvus_extern::runtime::Runtime>::Value; #arity],
                site: &'c ::acvus_extern::SiteShape<#runtime, #requirements>,
            ) -> impl ::core::future::Future<Output = ::acvus_extern::done::Done<#runtime>> + ::core::marker::Send + 'c {
                ::acvus_extern::Entry::new(runtime, args, site)
                    .open_awaited::<#args_tuple, (), _>(move |__cx, #opened_tuple, ()| async move {
                        #awaited_finish
                    })
            }
        }

        #returning
    }
}
