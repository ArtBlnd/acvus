//! What `#[extern_fn]` emits beside the function: its marker, the seal, its
//! declaration and its dispatch. The glue holds no `unsafe` (macros §4): the
//! seal is the one `unsafe impl`, the attribute's assertion that the dispatch
//! opens each argument at its parameter's own Rust type.

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
            "an effect is `pure`, `idempotent`, `opaque`, or one of the function's `EffectVar` parameters",
        ));
    }
    if let (Some(_), Some(param)) = (&attr.impl_of, sig.params.iter().find(|p| p.role.takes_impl())) {
        return Err(syn::Error::new(
            param.name.span(),
            "an impl takes no requirement as a parameter: the impls it requires are captured in its value where its site chooses it (impls §2)",
        ));
    }
    if let (Some(_), Some(name)) = (&attr.impl_of, &attr.name) {
        return Err(syn::Error::new(name.span(), "an impl states no name: it is reached only through its generic function (macros §1)"));
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

    let checks = signature_checks(&func.sig, attr.sync.as_ref());

    Ok(quote! {
        #func

        #checks

        #[allow(non_camel_case_types)]
        #vis struct #marker {
            #(#fields,)*
        }

        impl #marker {
            #constructor
        }

        // SAFETY: `#[extern_fn]` wrote the dispatch below from the function's
        // signature: it opens each argument once, at the Rust type its
        // parameter writes (`Open`) or by its requirement's mode
        // (`RequiresParam`, `InstanceParam`), and each impl its site chose
        // with the parameter that states the requirement.
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
    let argument_names: Vec<&Ident> = sig.params.iter().filter(|p| p.role.takes_argument()).map(|p| &p.name).collect();
    let ending = facts::ends(attr.ends);
    let relations = facts::relations(&attr.relations, &argument_names)?;
    let places: Vec<&facts::PlaceFacts> = sig.params.iter().filter(|p| p.role.takes_argument()).map(|p| &p.place).collect();
    // Each requirement, in the order of the parameters that state one: the
    // order the site's impls are in (impls §3).
    let requires = sig.params.iter().filter(|p| p.role.takes_impl()).map(|p| {
        let ty = sig.at_stand_ins(&p.ty, runtime);
        match p.role {
            Role::Instance => quote!(<#ty as ::acvus_extern::decl::InstanceParam<#runtime>>::requirement(names)),
            _ => quote!(<#ty as ::acvus_extern::decl::RequiresParam<#runtime>>::requirement(names)),
        }
    });
    let access = facts::access(&places, &argument_names, runtime)?;
    let action = facts::actions(attr.combine.as_ref(), &places, &argument_names, runtime)?;
    // The declaration is the handler's at each runtime it runs at: the
    // bounds it states of its runtime, as its dispatch is.
    let bounds = sig.runtime_bounds();
    Ok(quote! {
        impl<#runtime: ::acvus_extern::runtime::Runtime #(+ #bounds)*> ::acvus_extern::register::Declared<#runtime> for #marker {
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
                    requires: ::std::vec![#(#requires),*],
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
    // Each argument's declared type: its Rust type's form's, a receiver's
    // as its instance states it.
    let declared = |p: &crate::sig::Param| {
        let ty = sig.at_stand_ins(&p.ty, runtime);
        match p.role {
            Role::Instance => quote!(<#ty as ::acvus_extern::decl::InstanceParam<#runtime>>::decl(names)),
            _ => quote!(::acvus_extern::decl::decl_in::<#runtime, #ty>(names)),
        }
    };
    let params = sig
        .params
        .iter()
        .filter(|p| p.role.takes_argument())
        .map(|p| {
            let name = LitStr::new(&p.name.to_string(), p.name.span());
            let ty = declared(p);
            quote!((names.intern(#name), #ty))
        })
        .collect();
    let result = match &sig.output {
        Output::Never => quote!(::acvus_extern::decl::DeclTy::Never),
        Output::Value => {
            let ty = sig.at_stand_ins(&sig.written_output, runtime);
            quote!(::acvus_extern::decl::decl_out::<#runtime, #ty>(names))
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
        .filter(|p| p.role.takes_argument())
        .enumerate()
        .map(|(at, p)| {
            let ty = declared(p);
            let (outer, inner) = names(&p.names);
            quote! {
                ::acvus_extern::decl::EndNames {
                    at: ::acvus_extern::decl::At::Param(#at),
                    ty: #ty,
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
/// (extern fn §6), checked by Rust: its parameters' and its result's keys,
/// read at one loan, are `FamilyAt`'s at that choice, or the impl does not
/// compile. Each chosen
/// type is the Rust type a value of the variable is: a variable of the
/// impl's own, or `Uniform<R, X>`. Nothing is checked at run time.
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
    let loan = syn::Lifetime::new("'__p", proc_macro2::Span::call_site());
    let lifetimes = generics.lifetimes();
    let params: Vec<_> = generics.type_params().collect();
    let outlive = params.iter().map(|param| {
        let ident = &param.ident;
        quote!(#ident: #loan)
    });
    let predicates = generics.where_clause.iter().flat_map(|clause| &clause.predicates);
    let key = |ty: &syn::Type| {
        let ty = crate::sig::at_lifetime(ty, &loan);
        quote!(<#ty as ::acvus_extern::runtime::Place<#loan, #at_runtime>>::Key)
    };
    let arguments = sig.params.iter().filter(|p| matches!(p.role, Role::Argument)).map(|p| key(&p.ty));
    let result = key(&sig.written_output);
    quote! {
        const _: () = {
            #[allow(dead_code, non_snake_case)]
            fn __signature_is_at_choice<#loan, #(#lifetimes,)* #(#params,)* #added>()
            where
                #(#outlive,)*
                #(#predicates,)*
            {
                ::acvus_extern::instance::is_the_signature::<#loan, #at_runtime, #generic_function, (#(#choice,)*), (#(#arguments,)*), #result>();
            }
        };
    }
}

/// Checks Rust makes of the signature as written, at compile time only: no
/// lifetime is elided in a path, so every lifetime the declaration's labels
/// read is written (lifetimes); and where `sync` names a returning body, it
/// has the awaited body's signature, lifetimes included (macros §1).
pub fn signature_checks(signature: &syn::Signature, sync: Option<&syn::Path>) -> TokenStream {
    let generics = &signature.generics;
    let where_clause = &generics.where_clause;
    let inputs: Vec<&syn::Type> = signature
        .inputs
        .iter()
        .filter_map(|input| match input {
            syn::FnArg::Typed(typed) => Some(&*typed.ty),
            syn::FnArg::Receiver(_) => None,
        })
        .collect();
    let output = match &signature.output {
        syn::ReturnType::Type(_, ty) => quote!(#ty),
        syn::ReturnType::Default => quote!(()),
    };
    let sync_check = sync.map(|body| {
        let lifetimes = generics.lifetimes().map(|param| &param.lifetime);
        let type_params: Vec<_> = generics.type_params().collect();
        let type_args = type_params.iter().map(|param| &param.ident);
        quote! {
            #[allow(dead_code)]
            fn __sync_is_the_signature<#(#type_params),*>() #where_clause {
                let _: for<#(#lifetimes),*> fn(#(#inputs),*) -> #output = #body::<#(#type_args),*>;
            }
        }
    });
    quote! {
        const _: () = {
            #[deny(elided_lifetimes_in_paths)]
            #[allow(dead_code, unused_variables, clippy::all)]
            fn __signature #generics (#(_: #inputs),*) -> #output #where_clause {
                loop {}
            }

            #sync_check
        };
    }
}

/// The dispatch of each body the function has: an awaited body's, with its
/// returning body's where `sync` names one, or a returning body's. Both open
/// one signature's arguments and call a function of it.
fn dispatch(sig: &Sig, sync: Option<&syn::Path>, marker: &Ident, runtime: &Ident) -> TokenStream {
    // The values the entry hands the body: one per parameter taking an
    // argument, in the arguments' order, and one per parameter taking an
    // impl, in the requirements' order.
    let given: Vec<Ident> =
        sig.params.iter().enumerate().filter(|(_, p)| p.role.takes_argument()).map(|(i, _)| format_ident!("__given{}", i)).collect();
    let chosen: Vec<Ident> =
        sig.params.iter().enumerate().filter(|(_, p)| p.role.takes_impl()).map(|(i, _)| format_ident!("__chosen{}", i)).collect();
    // Each parameter opened once, from its values, at its Rust type or by the
    // mode its requirement's Rust type picks for its role.
    let opens: Vec<TokenStream> = sig
        .params
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let ty = sig.at_stand_ins(&p.ty, runtime);
            let (a, given, chosen) = (format_ident!("__a{}", i), format_ident!("__given{}", i), format_ident!("__chosen{}", i));
            match p.role {
                Role::Argument => Some(quote!(let #a = #given.open::<#ty>();)),
                Role::Requires => Some(quote! {
                    let #a = #chosen.open::<<#ty as ::acvus_extern::decl::RequiresParam<#runtime>>::Mode>();
                }),
                Role::Instance => Some(quote! {
                    let #a = #given.open_with::<<#ty as ::acvus_extern::decl::InstanceParam<#runtime>>::Mode>(#chosen);
                }),
                Role::Capture { .. } | Role::Executor => None,
            }
        })
        .collect();
    let call_args: Vec<TokenStream> = sig
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let name = &p.name;
            match &p.role {
                Role::Capture { .. } => quote!(&self.#name),
                Role::Executor => quote!(__executor),
                Role::Argument | Role::Requires | Role::Instance => {
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
    let arity = given.len();
    let requirements = chosen.len();

    let returning = |call: TokenStream| {
        let finish = match sig.output {
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
                    ::acvus_extern::Entry::new(runtime, args, site).open(|__cx, [#(#given),*], [#(#chosen),*]| {
                        #(#opens)*
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
        Output::Value => quote! {
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
                    .open_awaited(move |__cx, [#(#given),*], [#(#chosen),*]| {
                        #(#opens)*
                        async move {
                            #awaited_finish
                        }
                    })
            }
        }

        #returning
    }
}
