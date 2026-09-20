//! Proc macros for acvus-extern. See RFC-0009.
//!
//! - `#[extern_fn]`: a Rust function declares an ExternFn.
//! - `#[derive(ExternType)]`: a Rust struct declares an extension type.
//! - `#[derive(TyArg)]`: a Rust struct declares a structural object type; a
//!   Rust enum declares the language's enum of the same name.
//! - `extern_registry!`: the items one registry contributes.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, DeriveInput, FnArg, GenericParam, Ident, ItemFn, LitStr, Pat, Path, ReturnType,
    Token, Type, TypeParamBound, parse_macro_input,
};

mod generics;
mod subst;

use generics::{VarKind, Vars};

// -- #[extern_fn] ----------------------------------------------------

#[proc_macro_attribute]
pub fn extern_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as ExternFnAttr);
    let mut func = parse_macro_input!(item as ItemFn);
    match generate_extern_fn(attr, &mut func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// `#[extern_fn(name = "...", ns = "...", effect = pure | idempotent | E)]`.
struct ExternFnAttr {
    name: Option<LitStr>,
    /// The shared signature this function is an instance of (RFC-0019).
    instance_of: Option<Path>,
    effect: Option<Ident>,
    /// `commutative`: two calls of this function in either order are the
    /// same program (RFC-0013).
    commutative: bool,
    /// `heavy`: the call is offloaded to a blocking pool and awaited
    /// (RFC-0046).
    heavy: bool,
    /// `sync = f`: the plain `fn` that runs this declaration when the
    /// call's task is `Sync`, declared beside the `async fn` at the same
    /// signature (RFC-0046).
    sync: Option<Ident>,
}

impl Parse for ExternFnAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut out = ExternFnAttr {
            name: None,
            instance_of: None,
            effect: None,
            commutative: false,
            heavy: false,
            sync: None,
        };
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key == "commutative" {
                out.commutative = true;
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            if key == "heavy" {
                out.heavy = true;
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            input.parse::<Token![=]>()?;
            if key == "name" {
                out.name = Some(input.parse()?);
            } else if key == "instance_of" {
                out.instance_of = Some(input.parse()?);
            } else if key == "effect" {
                out.effect = Some(input.parse()?);
            } else if key == "sync" {
                out.sync = Some(input.parse()?);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    "expected `name`, `instance_of`, `effect`, `commutative`, `heavy`, or `sync`",
                ));
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(out)
    }
}

/// An acvus parameter of the declared function.
struct ExternParam {
    name: String,
    /// The acvus type: for `&T` and `&mut T`, the `T`.
    ty: Type,
    mode: Mode,
}

/// How the Rust parameter takes its argument (RFC-0015).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Value,
    Borrow,
    BorrowMut,
    Str,
    /// `SRef<'_>` / `SMut<'_>`, which `#[derive(TyArg)]` emits beside the
    /// struct (RFC-0050 rule 6).
    Projection,
}

impl Mode {
    fn lends_its_storage(self) -> bool {
        match self {
            Mode::Borrow | Mode::BorrowMut | Mode::Str | Mode::Projection => true,
            Mode::Value => false,
        }
    }

    /// The acvus type of a parameter whose Rust type is `ty` under this
    /// mode, with `rt` as the runtime a reference carrier names.
    fn acvus_ty(self, ty: &Type, rt: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Mode::Value => quote! { #ty },
            Mode::Borrow => quote! { ::acvus_extern::Ref<#ty, #rt> },
            Mode::BorrowMut => quote! { ::acvus_extern::RefMut<#ty, #rt> },
            Mode::Str => quote! { ::acvus_extern::StrView },
            Mode::Projection => {
                let at_static = at_static(ty);
                quote! { #at_static }
            }
        }
    }
}

fn is_str(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident("str"))
}

/// How the Rust result crosses back, which is the `Mode` of the return
/// position.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Returning {
    Value,
    Str,
}

impl Returning {
    fn of(ty: &Type) -> Self {
        match ty {
            Type::Reference(r) if is_str(&r.elem) => Returning::Str,
            _ => Returning::Value,
        }
    }

    /// The acvus type of a result whose Rust type crosses as `owned`.
    fn acvus_ty(self, owned: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Returning::Value => quote! { #owned },
            Returning::Str => quote! { ::acvus_extern::StrView },
        }
    }

    /// The `Ret` marker the glue is built with, where `owned` is the `Val`
    /// a result that crosses as itself takes.
    fn marker(self, owned: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Returning::Value => quote! { #owned },
            Returning::Str => quote! { ::acvus_extern::RetStr },
        }
    }
}

/// The test is the lifetime and not the type's name because reading a
/// crossing out of a path's last segment is exactly what RFC-0050 rule 6
/// withdrew `returns_slice` for: an alias defeated it.
fn borrows_caller(ty: &Type) -> bool {
    let Type::Path(p) = ty else {
        return false;
    };
    p.path.segments.iter().any(|segment| {
        let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
            return false;
        };
        args.args
            .iter()
            .any(|arg| matches!(arg, syn::GenericArgument::Lifetime(_)))
    })
}

/// `TyArg` and an `Arg` marker are both `'static`, so a projection reaches
/// them with its lifetimes at `'static`, and `Projected::At<'a>` hands the
/// handler's body the same projection at the call's own lifetime.
fn at_static(ty: &Type) -> Type {
    struct Static;

    impl syn::visit_mut::VisitMut for Static {
        fn visit_lifetime_mut(&mut self, lifetime: &mut syn::Lifetime) {
            *lifetime = syn::Lifetime::new("'static", lifetime.apostrophe);
        }
    }

    let mut ty = ty.clone();
    syn::visit_mut::VisitMut::visit_type_mut(&mut Static, &mut ty);
    ty
}

/// A parameter marked `#[state]`: supplied when the registry is built,
/// held by the handler, and absent from the acvus type (RFC-0021).
struct StateParam {
    ident: Ident,
    ty: Type,
}

/// One Rust parameter after the runtime, in declaration order.
enum RustParam {
    Acvus(ExternParam),
    State(StateParam),
}

fn generate_extern_fn(
    attr: ExternFnAttr,
    func: &mut ItemFn,
) -> syn::Result<proc_macro2::TokenStream> {
    let is_cast = take_marker_attr(&mut func.attrs, "extern_cast");
    let is_async = func.sig.asyncness.is_some();
    let vars = Vars::from_generics(&func.sig.generics)?;
    let Signature {
        takes_runtime,
        takes_frame,
        params: rust_params,
    } = parse_params(&mut func.sig, vars.runtime_ident())?;
    let params: Vec<&ExternParam> = rust_params
        .iter()
        .filter_map(|p| match p {
            RustParam::Acvus(a) => Some(a),
            RustParam::State(_) => None,
        })
        .collect();
    let states: Vec<&StateParam> = rust_params
        .iter()
        .filter_map(|p| match p {
            RustParam::State(st) => Some(st),
            RustParam::Acvus(_) => None,
        })
        .collect();
    let ret = parse_return(&func.sig.output);
    let returning = Returning::of(&ret);
    if returning == Returning::Str {
        if !params.iter().any(|p| p.mode.lends_its_storage()) {
            return Err(syn::Error::new_spanned(
                &ret,
                "a declaration returning the language's `&str` has no parameter the view \
                 can be a projection of: every parameter is taken by value, so the bytes \
                 the result names belong to no storage the caller kept (RFC-0047 §3). \
                 Take one parameter by reference, or return `String`.",
            ));
        }
        if is_async || attr.heavy {
            return Err(syn::Error::new(
                func.sig.ident.span(),
                "a declaration returning the language's `&str` runs at `Task::Sync`: the \
                 view borrows the frame the call laid its arguments on, and that frame is \
                 gone by the time an awaited or offloaded call resumes (RFC-0047 §3). \
                 Return `String`, or declare this at `Sync`.",
            ));
        }
    }

    let fn_ident = &func.sig.ident;
    let vis = &func.vis;
    let acvus_name = attr
        .name
        .as_ref()
        .map(LitStr::value)
        .unwrap_or_else(|| fn_ident.to_string());
    let qref = qref_expr(&acvus_name);
    let decl_ident = format_ident!("__extern_fn_{}", fn_ident);
    let instance_of = match &attr.instance_of {
        Some(sig) => quote! {
            ::core::option::Option::Some(<#sig as ::acvus_extern::SharedSignature>::qref(__i))
        },
        None => quote! { ::core::option::Option::None },
    };

    if attr.heavy && is_async {
        return Err(syn::Error::new(
            fn_ident.span(),
            "`heavy` and `async fn` are two tasks; declare one",
        ));
    }
    if attr.sync.is_some() && !is_async {
        return Err(syn::Error::new(
            fn_ident.span(),
            "`sync` names the plain `fn` that runs this declaration at `Task::Sync`; \
             a declaration that is not an `async fn` already is that instance",
        ));
    }
    let commutes = if attr.commutative {
        quote! { .commutative() }
    } else {
        quote! {}
    };
    let task = match (attr.heavy, is_async) {
        (true, _) => quote! { ::acvus_extern::Task::Heavy },
        (false, true) => quote! { ::acvus_extern::Task::Async },
        (false, false) => quote! { ::acvus_extern::Task::Sync },
    };
    let at_task = quote! { .at_task(#task) };
    // `optimize::spawn_split` rewrites every call whose effect is not Pure
    // into a `Spawn` and an `Eval`, and an `Eval` awaits. Such a call runs
    // at `Async` however synchronous its Rust body is, so a declaration
    // that is not pure says so (RFC-0046).
    let spawned = quote! { .at_task(#task.join(::acvus_extern::Task::Async)) };
    let effect = match &attr.effect {
        None => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::OPAQUE #commutes #spawned) }
        }
        Some(e) if e == "pure" => {
            if attr.commutative {
                return Err(syn::Error::new(
                    e.span(),
                    "a pure function commutes by definition; drop `commutative`",
                ));
            }
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::PURE #at_task) }
        }
        Some(e) if e == "idempotent" => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::IDEMPOTENT #commutes #spawned) }
        }
        Some(e) if e == "opaque" => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::OPAQUE #commutes #spawned) }
        }
        Some(e) => match vars.lookup(e) {
            Some((VarKind::Effect, k)) => {
                if attr.commutative {
                    return Err(syn::Error::new(
                        e.span(),
                        "`commutative` cannot be declared on an effect variable",
                    ));
                }
                if attr.heavy {
                    return Err(syn::Error::new(
                        e.span(),
                        "`heavy` cannot be declared on an effect variable: the task is the variable's",
                    ));
                }
                if is_async && attr.sync.is_none() {
                    return Err(syn::Error::new(
                        e.span(),
                        "an `async fn` generic in its effect takes its task from the variable, \
                         so it needs the plain `fn` that runs it at `Task::Sync`: declare \
                         `sync = <fn>`. Without one the glue awaits for every effect the \
                         variable takes, and the declared task is a claim nothing keeps \
                         (RFC-0046).",
                    ));
                }
                quote! { __vars.effects[#k].clone() }
            }
            _ => {
                return Err(syn::Error::new(
                    e.span(),
                    "effect must be `pure`, `idempotent`, `opaque`, or an `EffectVar` parameter",
                ));
            }
        },
    };

    if is_cast {
        if params.len() != 1 {
            return Err(syn::Error::new(
                fn_ident.span(),
                "an extern_cast takes exactly one parameter",
            ));
        }
        if !matches!(&attr.effect, Some(e) if e == "pure") {
            return Err(syn::Error::new(
                fn_ident.span(),
                "an extern_cast declares `effect = pure`",
            ));
        }
    }

    let signature = |member: Option<&Type>| -> proc_macro2::TokenStream {
        let param_terms = params.iter().map(|p| {
            let name = &p.name;
            let comp_ty = p.mode.acvus_ty(
                &vars.to_compile_time_instance(&p.ty, member),
                &quote! { __R },
            );
            quote! {
                ::acvus_extern::ParamTerm::<::acvus_extern::Poly>::new(
                    __i.intern(#name),
                    <#comp_ty as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
                )
            }
        });
        let comp_ret = vars.to_compile_time_instance(&ret, member);
        let comp_ret = returning.acvus_ty(&quote! { #comp_ret });
        quote! {
            ::acvus_extern::PolyTy::Fn {
                params: vec![#(#param_terms),*],
                ret: Box::new(<#comp_ret as ::acvus_extern::TyArg>::poly_ty(__i, &__vars)),
                captures: vec![],
                effect: #effect,
            }
        }
    };

    let arg_idents: Vec<Ident> = (0..params.len()).map(|i| format_ident!("__a{i}")).collect();
    let crossing = |ty: &Type, member: Option<&Type>| -> proc_macro2::TokenStream {
        if member.is_some() && vars.mentions_mono(ty) {
            quote! { ::acvus_extern::Specialized }
        } else {
            quote! { ::acvus_extern::Uniform }
        }
    };
    let sync_variant = if attr.heavy {
        quote! { heavy }
    } else {
        quote! { sync }
    };
    let arity = params.len();
    let state_tys: Vec<&Type> = states.iter().map(|st| &st.ty).collect();
    let glue = |member: Option<&Type>, callee: &Ident, awaits: bool| -> proc_macro2::TokenStream {
        let rt_tys: Vec<Type> = params
            .iter()
            .map(|p| vars.to_runtime_instance(&p.ty, member))
            .collect();
        let rt_ret = vars.to_runtime_instance(&ret, member);
        let ret_marker = {
            let c = crossing(&ret, member);
            returning.marker(&quote! { ::acvus_extern::Val<#rt_ret, #c> })
        };
        let arg_markers: Vec<proc_macro2::TokenStream> = params
            .iter()
            .zip(&rt_tys)
            .map(|(p, ty)| {
                let c = crossing(&p.ty, member);
                match p.mode {
                    Mode::Value => quote! { ::acvus_extern::ByValue<#ty, #c> },
                    Mode::Borrow => quote! { ::acvus_extern::ByRef<#ty, #c> },
                    Mode::BorrowMut => quote! { ::acvus_extern::ByRefMut<#ty, #c> },
                    Mode::Str => quote! { ::acvus_extern::ByStr },
                    Mode::Projection => {
                        let at_static = at_static(ty);
                        quote! { ::acvus_extern::ByProjection<#at_static> }
                    }
                }
            })
            .collect();
        let turbofish = vars.runtime_turbofish_instance(member);
        let mut acvus_args = arg_idents.iter();
        let mut state_at = 0usize;
        let passed: Vec<proc_macro2::TokenStream> = rust_params
            .iter()
            .map(|p| match p {
                RustParam::Acvus(_) => {
                    let a = acvus_args.next().expect("one ident per acvus parameter");
                    quote! { #a }
                }
                RustParam::State(_) => {
                    let at = proc_macro2::Literal::usize_unsuffixed(state_at);
                    state_at += 1;
                    quote! { &__state.#at }
                }
            })
            .collect();
        let capture_state = (!states.is_empty()).then(|| {
            quote! { let __state = ::std::sync::Arc::clone(&__state); }
        });
        let rt_arg = takes_runtime.then(|| quote! { __rt, });
        // A synchronous handler is handed the window by value and lends it
        // onward; an `async` one is handed a borrow, because the future it
        // returns is what holds that borrow.
        let frame_arg = takes_frame.then(|| match awaits {
            true => quote! { __frame, },
            false => quote! { &mut __frame, },
        });
        let frame_param = match (takes_frame, awaits) {
            (true, false) => quote! { mut __frame },
            _ => quote! { __frame },
        };
        let call = quote! { #callee #turbofish (#rt_arg #frame_arg #(#passed),*) };
        if awaits {
            let builder = format_ident!("async_glue{arity}");
            quote! {
                ::acvus_extern::ExternHandler::awaited({
                    #capture_state
                    ::acvus_extern::#builder::<__R, _, #(#arg_markers,)*>(
                        move |__rt: &__R, #frame_param #(, #arg_idents)*| {
                            #capture_state
                            ::std::boxed::Box::pin(async move {
                                let __r = (#call).await;
                                <#rt_ret as ::acvus_extern::OneValue<__R>>::erase(__r, __rt)
                            })
                        }
                    )
                })
            }
        } else {
            let builder = format_ident!("glue{arity}");
            quote! {
                ::acvus_extern::ExternHandler::#sync_variant({
                    #capture_state
                    ::acvus_extern::#builder::<__R, _, #(#arg_markers,)* #ret_marker>(
                        move |__rt: &__R, #frame_param #(, #arg_idents)*| #call
                    )
                })
            }
        }
    };

    let family_casts = |member: &Type| -> Vec<proc_macro2::TokenStream> {
        let mut seen = std::collections::HashSet::new();
        params
            .iter()
            .map(|p| &p.ty)
            .chain(std::iter::once(&ret))
            .filter(|ty| vars.mentions_mono(ty))
            .filter(|ty| seen.insert(quote! { #ty }.to_string()))
            .map(|ty| {
                let specialized = vars.to_compile_time_instance(ty, Some(member));
                let uniform = vars.to_compile_time_uniform(ty, member);
                let rt_ty = vars.to_runtime_instance(ty, Some(member));
                let cast_glue = |from: proc_macro2::TokenStream,
                                 into: proc_macro2::TokenStream|
                 -> proc_macro2::TokenStream {
                    quote! {
                        ::acvus_extern::ExternHandler::sync(
                            ::acvus_extern::glue1::<
                                __R,
                                _,
                                ::acvus_extern::ByValue<#rt_ty, #from>,
                                ::acvus_extern::Val<#rt_ty, #into>,
                            >(|_, _, __v| __v)
                        )
                    }
                };
                let erase = cast_glue(
                    quote! { ::acvus_extern::Specialized },
                    quote! { ::acvus_extern::Uniform },
                );
                let materialize = cast_glue(
                    quote! { ::acvus_extern::Uniform },
                    quote! { ::acvus_extern::Specialized },
                );
                quote! {
                    __casts.extend(::acvus_extern::family_casts::<__R>(
                        __i,
                        ::acvus_extern::MemberType {
                            specialized: <#specialized as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
                            uniform: <#uniform as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
                            erase: #erase,
                            materialize: #materialize,
                        },
                    ));
                }
            })
            .collect()
    };

    // The instances one member of the declaration contributes, in the
    // order `Instances::into_handlers` indexes them.
    let at_member = |member: Option<&Type>| -> Vec<proc_macro2::TokenStream> {
        let declared_sig = signature(member);
        let declared_handler = glue(member, fn_ident, is_async);
        let declared = quote! {
            ::acvus_extern::Instance {
                signature: #declared_sig,
                handler: #declared_handler,
                admits: ::acvus_extern::Task::Heavy,
            }
        };
        let Some(sync_fn) = &attr.sync else {
            return vec![declared];
        };
        let sync_sig = signature(member);
        let sync_handler = glue(member, sync_fn, false);
        vec![
            quote! {
                ::acvus_extern::Instance {
                    signature: #sync_sig,
                    handler: #sync_handler,
                    admits: ::acvus_extern::Task::Sync,
                }
            },
            declared,
        ]
    };

    let mut casts: Vec<proc_macro2::TokenStream> = Vec::new();
    let instances = match vars.mono_var() {
        None => match &attr.sync {
            None => {
                let generic = glue(None, fn_ident, is_async);
                quote! { ::acvus_extern::Instances::generic(#generic) }
            }
            Some(_) => {
                let concrete = at_member(None);
                quote! {
                    ::acvus_extern::Instances {
                        concrete: vec![#(#concrete),*],
                        generic: ::core::option::Option::None,
                    }
                }
            }
        },
        Some(mono) => {
            let members = mono.mono.as_ref().expect("mono_var has members");
            let concrete: Vec<proc_macro2::TokenStream> = members
                .iter()
                .flat_map(|member| at_member(Some(member)))
                .collect();
            let generic = if mono.mono_fallback {
                let handler = glue(None, fn_ident, is_async);
                quote! { ::core::option::Option::Some(#handler) }
            } else {
                quote! { ::core::option::Option::None }
            };
            casts = members.iter().flat_map(|m| family_casts(m)).collect();
            quote! {
                ::acvus_extern::Instances {
                    concrete: vec![#(#concrete),*],
                    generic: #generic,
                }
            }
        }
    };
    let bounds = vars.bound_exprs();
    let requires = vars.requires_exprs();
    let declared_ty = signature(None);

    let counts = vars.counts_expr();
    let rt_bounds = quote! { __R: ::acvus_extern::Runtime, };
    let state_idents: Vec<&Ident> = states.iter().map(|st| &st.ident).collect();
    let state_arc = (!states.is_empty()).then(|| {
        quote! {
            let __state: ::std::sync::Arc<(#(#state_tys,)*)> =
                ::std::sync::Arc::new((#(#state_idents,)*));
        }
    });
    Ok(quote! {
        #func

        #[doc(hidden)]
        #vis fn #decl_ident<__R>(
            __i: &::acvus_extern::Interner,
            __ns: ::core::option::Option<&str>,
            #(#state_idents: #state_tys,)*
        ) -> ::std::vec::Vec<::acvus_extern::ExternFn<__R>>
        where
            #rt_bounds
            #(#state_tys: ::core::marker::Send + ::core::marker::Sync + 'static,)*
        {
            let __vars = ::acvus_extern::PolyVars::fresh(#counts);
            #state_arc
            let mut __casts: ::std::vec::Vec<::acvus_extern::ExternFn<__R>> = ::std::vec::Vec::new();
            #(#casts)*
            let __declared = ::acvus_extern::ExternFn {
                decl: ::acvus_extern::FnDecl {
                    qref: #qref,
                    ty: #declared_ty,
                    bounds: vec![#(#bounds),*],
                    cast: #is_cast,
                    instance_of: #instance_of,
                    requires: vec![#(#requires),*],
                },
                instances: #instances,
            };
            ::core::iter::once(__declared).chain(__casts).collect()
        }
    })
}

fn take_marker_attr(attrs: &mut Vec<Attribute>, name: &str) -> bool {
    let before = attrs.len();
    attrs.retain(|a| !a.path().is_ident(name));
    attrs.len() != before
}

/// What a declaration's Rust signature carries in front of its acvus
/// parameters: `&R` with `R` the parameter bounded by `Runtime`, then the
/// frame a closure call runs in. A function that uses neither takes neither.
struct Signature {
    takes_runtime: bool,
    takes_frame: bool,
    params: Vec<RustParam>,
}

fn parse_params(sig: &mut syn::Signature, runtime: Option<&Ident>) -> syn::Result<Signature> {
    let mut inputs = sig.inputs.iter_mut().peekable();
    let takes_runtime = match (runtime, inputs.peek()) {
        (Some(runtime), Some(first)) => is_runtime_param(first, runtime),
        _ => false,
    };
    if takes_runtime {
        inputs.next();
    }
    let takes_frame = inputs.peek().is_some_and(|next| is_frame_param(next));
    if takes_frame {
        inputs.next();
    }

    let mut params = Vec::new();
    for (i, arg) in inputs.enumerate() {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "an extern_fn has no self parameter",
            ));
        };
        let is_state = take_marker_attr(&mut pat_type.attrs, "state");
        let ident = match pat_type.pat.as_ref() {
            Pat::Ident(p) => p.ident.clone(),
            _ => format_ident!("_{i}"),
        };
        if is_state {
            let Type::Reference(r) = pat_type.ty.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &pat_type.ty,
                    "a `#[state]` parameter is taken by shared reference",
                ));
            };
            params.push(RustParam::State(StateParam {
                ident,
                ty: (*r.elem).clone(),
            }));
            continue;
        }
        let (ty, mode) = match pat_type.ty.as_ref() {
            Type::Reference(r) if r.mutability.is_some() && is_str(&r.elem) => {
                return Err(syn::Error::new_spanned(
                    &pat_type.ty,
                    "there is no `&mut str`: a write through one could leave the bytes \
                     invalid UTF-8. Take `&str` to read, or `String` to own (RFC-0062).",
                ));
            }
            Type::Reference(r) if r.mutability.is_some() => ((*r.elem).clone(), Mode::BorrowMut),
            Type::Reference(r) if is_str(&r.elem) => ((*r.elem).clone(), Mode::Str),
            Type::Reference(r) => ((*r.elem).clone(), Mode::Borrow),
            ty if borrows_caller(ty) => (ty.clone(), Mode::Projection),
            ty => (ty.clone(), Mode::Value),
        };
        params.push(RustParam::Acvus(ExternParam {
            name: ident.to_string(),
            ty,
            mode,
        }));
    }
    Ok(Signature {
        takes_runtime,
        takes_frame,
        params,
    })
}

fn is_runtime_param(arg: &FnArg, runtime: &Ident) -> bool {
    let FnArg::Typed(pat_type) = arg else {
        return false;
    };
    let Type::Reference(r) = pat_type.ty.as_ref() else {
        return false;
    };
    r.mutability.is_none() && matches!(r.elem.as_ref(), Type::Path(p) if p.path.is_ident(runtime))
}

/// The frame is read by its type and not by its position, because the runtime
/// parameter in front of it is optional: a declaration that takes the frame
/// alone has it first, and a first acvus parameter taken `&mut T` sits in the
/// same place.
fn is_frame_param(arg: &FnArg) -> bool {
    let FnArg::Typed(pat_type) = arg else {
        return false;
    };
    let Type::Reference(r) = pat_type.ty.as_ref() else {
        return false;
    };
    let Type::Path(p) = r.elem.as_ref() else {
        return false;
    };
    let named_frame = p
        .path
        .segments
        .last()
        .is_some_and(|last| last.ident == "Frame");
    r.mutability.is_some() && named_frame
}

fn parse_return(output: &ReturnType) -> Type {
    match output {
        ReturnType::Default => syn::parse_quote! { () },
        ReturnType::Type(_, ty) => (**ty).clone(),
    }
}

/// The name under the namespace the registry passes as `__ns`.
fn qref_expr(name: &str) -> proc_macro2::TokenStream {
    quote! {
        ::acvus_extern::QualifiedRef {
            namespace: __ns.map(|__n| __i.intern(__n)),
            name: __i.intern(#name),
        }
    }
}

/// The name under a namespace fixed at the declaration.
fn qref_expr_in(ns: Option<&str>, name: &str) -> proc_macro2::TokenStream {
    match ns {
        Some(ns) => quote! {
            ::acvus_extern::QualifiedRef::qualified(__i.intern(#ns), __i.intern(#name))
        },
        None => quote! { ::acvus_extern::QualifiedRef::root(__i.intern(#name)) },
    }
}

// -- #[derive(ExternType)] -------------------------------------------

/// `#[extern_type(name = "...", ns = "...")]`.
struct ExternTypeAttr {
    name: Option<String>,
    ns: Option<String>,
}

/// Whether the struct carries `#[repr(transparent)]`.
fn is_repr_transparent(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("repr")
            && a.parse_args::<Ident>()
                .is_ok_and(|ident| ident == "transparent")
    })
}

fn parse_extern_type_attr(attrs: &[Attribute]) -> syn::Result<ExternTypeAttr> {
    let mut out = ExternTypeAttr {
        name: None,
        ns: None,
    };
    for attr in attrs {
        if !attr.path().is_ident("extern_type") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                meta.input.parse::<Token![=]>()?;
                out.name = Some(meta.input.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("ns") {
                meta.input.parse::<Token![=]>()?;
                out.ns = Some(meta.input.parse::<LitStr>()?.value());
            } else {
                return Err(meta.error("expected `name` or `ns`"));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

#[proc_macro_derive(ExternType, attributes(extern_type))]
pub fn derive_extern_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match generate_extern_type(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_extern_type(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = &input.ident;
    let attr = parse_extern_type_attr(&input.attrs)?;
    let name = attr.name.clone().unwrap_or_else(|| ident.to_string());
    let vars = Vars::from_generics(&input.generics)?;
    if vars.has_len_vars() {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type has no length parameters",
        ));
    }

    let syn::Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            ident.span(),
            "ExternType is derived on a struct",
        ));
    };
    let syn::Fields::Unnamed(fields) = &data.fields else {
        return Err(syn::Error::new(
            ident.span(),
            "ExternType is derived on a tuple struct: payload first, then PhantomData",
        ));
    };
    let mut field_iter = fields.unnamed.iter();
    let Some(payload) = field_iter.next() else {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type has a payload field",
        ));
    };
    let payload_ty = &payload.ty;
    if !is_repr_transparent(&input.attrs) {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type is `#[repr(transparent)]`: it is stored as its payload and read back through it",
        ));
    }
    for extra in field_iter {
        let is_phantom = match &extra.ty {
            Type::Path(p) => p
                .path
                .segments
                .last()
                .is_some_and(|s| s.ident == "PhantomData"),
            _ => false,
        };
        if !is_phantom {
            return Err(syn::Error::new_spanned(
                &extra.ty,
                "every field after the payload is PhantomData",
            ));
        }
    }

    if vars.mentions_var(payload_ty) {
        return Err(syn::Error::new_spanned(
            payload_ty,
            "the payload type names no type, effect, or length parameter; every instantiation shares one payload",
        ));
    }
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let arg_impl_generics = vars.arg_impl_generics();
    let impl_params = {
        let params = &input.generics.params;
        if params.is_empty() {
            quote! {}
        } else {
            quote! { #params, }
        }
    };
    let where_predicates = input
        .generics
        .where_clause
        .as_ref()
        .map(|w| {
            let preds = &w.predicates;
            quote! { #preds }
        })
        .unwrap_or_default();
    let type_arg_exprs = vars.type_arg_exprs();
    let effect_arg_exprs = vars.effect_arg_exprs();
    let identity_arg_exprs = vars.identity_arg_exprs();
    let n_tys = vars.count(VarKind::Ty);
    let n_effects = vars.count(VarKind::Effect);
    let n_identities = vars.count(VarKind::Identity);
    if n_identities > 1 {
        return Err(syn::Error::new(
            ident.span(),
            "an extension type has at most one identity parameter; a value is one source",
        ));
    }

    let qref = qref_expr_in(attr.ns.as_deref(), &name);
    let one_value_run = one_value_run();
    let returned_as_one_value = returned_as_one_value();
    let payload_crossing = quote! {
        fn erase(self, __rt: &__R) -> <__R as ::acvus_extern::Runtime>::Value {
            ::acvus_extern::transparent::erase::<Self, #payload_ty, __R>(self, __rt)
        }

        unsafe fn materialize(__rt: &__R, __value: <__R as ::acvus_extern::Runtime>::Value) -> Self {
            // SAFETY: the caller's contract, and `erase` is `transparent::erase`.
            unsafe {
                ::acvus_extern::transparent::materialize::<Self, #payload_ty, __R>(__rt, __value)
            }
        }

        unsafe fn deref<'__a>(
            __rt: &__R,
            __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
        ) -> &'__a Self {
            // SAFETY: the caller's contract: a live storage of the payload.
            unsafe { ::acvus_extern::transparent::deref::<Self, #payload_ty, __R>(__rt, __reference) }
        }

        unsafe fn deref_mut<'__a>(
            __rt: &__R,
            __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
        ) -> &'__a mut Self {
            // SAFETY: as in `deref`, exclusively.
            unsafe {
                ::acvus_extern::transparent::deref_mut::<Self, #payload_ty, __R>(__rt, __reference)
            }
        }
    };

    Ok(quote! {
        impl #arg_impl_generics ::acvus_extern::TyArg for #ident #ty_generics #where_clause {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                ::acvus_extern::PolyTy::UserDefined {
                    id: #qref,
                    type_args: vec![#(::acvus_extern::TypeArg::uniform(#type_arg_exprs)),*],
                    effect_args: vec![#(#effect_arg_exprs),*],
                    identity_args: vec![#(#identity_arg_exprs),*],
                }
            }
        }

        impl<#impl_params __R> ::acvus_extern::Cross<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #one_value_run
        }

        impl<#impl_params __R> ::acvus_extern::OneValue<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #payload_crossing
        }

        impl<#impl_params __R> ::acvus_extern::Returned<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #returned_as_one_value
        }

        impl<#impl_params __R> ::acvus_extern::CrossSpecialized<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
            #payload_crossing
        }

        // SAFETY: the struct is `#[repr(transparent)]`, checked above, with the
        // payload as its one non-zero-sized field: every other field is
        // `PhantomData`, also checked above.
        unsafe impl<#impl_params> ::acvus_extern::Transparent<#payload_ty> for #ident #ty_generics
        where
            #where_predicates
        {
        }

        impl<#impl_params __R> ::acvus_extern::Borrowable<__R> for #ident #ty_generics
        where
            __R: ::acvus_extern::Runtime,
            #where_predicates
        {
        }

        impl #impl_generics ::acvus_extern::ExternTypeDecl for #ident #ty_generics #where_clause {
            fn type_decl(__i: &::acvus_extern::Interner) -> ::acvus_extern::UserDefinedDecl {
                ::acvus_extern::UserDefinedDecl {
                    qref: #qref,
                    type_params: vec![::acvus_extern::TyVarBound::Any; #n_tys],
                    effect_params: #n_effects,
                    identity_params: #n_identities,
                    specializable: vec![false; #n_tys],
                }
            }
        }
    })
}

// -- #[derive(TyArg)] ------------------------------------------------

#[proc_macro_derive(TyArg, attributes(projection))]
pub fn derive_ty_arg(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match generate_ty_arg(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_ty_arg(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            ident.span(),
            "a structural type has no generic parameters",
        ));
    }
    match &input.data {
        syn::Data::Struct(data) => {
            let syn::Fields::Named(fields) = &data.fields else {
                return Err(syn::Error::new(
                    ident.span(),
                    "TyArg is derived on a struct with named fields, one per object field",
                ));
            };
            let shape = ObjectShape::of(fields);
            let ty = shape.declared_poly_ty(&ident.to_string());
            let erase = shape.erase(quote! { self });
            let materialize = shape.materialize(quote! { __value }, quote! { Self });
            let projected = input.attrs.iter().any(|a| a.path().is_ident("projection"));
            let borrowing = match projected {
                true => Borrowing::AsProjection,
                false => Borrowing::Whole,
            };
            let cross = cross_impl(
                ident,
                Crossing {
                    ty,
                    erase,
                    materialize,
                    borrowing,
                    returned: shape.returned(),
                },
            );
            let projection = projected.then(|| shape.projection(ident));
            Ok(quote! { #cross #projection })
        }
        syn::Data::Enum(data) => generate_enum_ty_arg(ident, data),
        syn::Data::Union(_) => Err(syn::Error::new(
            ident.span(),
            "TyArg is derived on a struct or an enum",
        )),
    }
}

#[derive(Clone, Copy)]
enum Borrowing {
    Whole,
    AsProjection,
}

impl Borrowing {
    fn bound(self) -> proc_macro2::TokenStream {
        match self {
            Borrowing::Whole => quote! {},
            Borrowing::AsProjection => {
                quote! { Self: ::acvus_extern::BorrowedWhole<__R>, }
            }
        }
    }
}

/// How a derived type crosses: the acvus type it names, and the two
/// directions of the crossing itself.
struct Crossing {
    ty: proc_macro2::TokenStream,
    erase: proc_macro2::TokenStream,
    materialize: proc_macro2::TokenStream,
    borrowing: Borrowing,
    returned: proc_macro2::TokenStream,
}

/// The `TyArg` and `Cross` impls of a derived type.
fn cross_impl(ident: &Ident, crossing: Crossing) -> proc_macro2::TokenStream {
    let Crossing {
        ty,
        erase,
        materialize,
        borrowing,
        returned,
    } = crossing;
    let one_value_run = one_value_run();
    let borrowable_bound = borrowing.bound();
    quote! {
        impl ::acvus_extern::TyArg for #ident {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                #ty
            }
        }

        impl<__R> ::acvus_extern::Borrowable<__R> for #ident
        where
            __R: ::acvus_extern::Runtime,
            #borrowable_bound
        {
        }

        impl<__R> ::acvus_extern::Cross<__R> for #ident
        where
            __R: ::acvus_extern::Runtime,
        {
            #one_value_run
        }

        impl<__R> ::acvus_extern::OneValue<__R> for #ident
        where
            __R: ::acvus_extern::Runtime,
        {
            fn erase(self, __rt: &__R) -> <__R as ::acvus_extern::Runtime>::Value {
                #erase
            }

            unsafe fn materialize(__rt: &__R, __value: <__R as ::acvus_extern::Runtime>::Value) -> Self {
                #materialize
            }
        }

        impl<__R> ::acvus_extern::Returned<__R> for #ident
        where
            __R: ::acvus_extern::Runtime,
        {
            #returned
        }
    }
}

/// A result that crosses back as the one heap value `OneValue::erase`
/// builds: every derived type but a struct, whose components go to the
/// caller's destination run instead.
fn returned_as_one_value() -> proc_macro2::TokenStream {
    quote! {
        type Form = ::acvus_extern::One;

        fn into_run(self, __rt: &__R, __out: &mut [<__R as ::acvus_extern::Runtime>::Value]) {
            ::acvus_extern::one_into_run(self, __rt, __out)
        }
    }
}

/// The `Cross` of a one-value crossing, as the library writes it: `Form` and
/// the two forwards to `OneValue`, with `__R` the runtime parameter both
/// derives already name.
fn one_value_run() -> proc_macro2::TokenStream {
    quote! {
        type Form = ::acvus_extern::One;

        unsafe fn from_run(
            __rt: &__R,
            __run: &[<__R as ::acvus_extern::Runtime>::Value],
        ) -> Self {
            // SAFETY: the caller's contract, at one value.
            unsafe { ::acvus_extern::one_from_run(__rt, __run) }
        }

        fn into_run(self, __rt: &__R, __out: &mut [<__R as ::acvus_extern::Runtime>::Value]) {
            ::acvus_extern::one_into_run(self, __rt, __out)
        }
    }
}

/// Named fields as an object: the struct's, or a struct variant's.
struct ObjectShape<'a> {
    idents: Vec<&'a Ident>,
    names: Vec<String>,
    tys: Vec<&'a Type>,
}

impl<'a> ObjectShape<'a> {
    /// The fields in RFC-0050 rule 8's order — ascending by the field name —
    /// and not in declaration order. Rule 8's order is what a run's layout,
    /// a committed object's canonical bytes and this table all have to agree
    /// on, and it is the field names sorted as strings: `ObjectTy` carries no
    /// declaration order for a `Declared` struct to be laid by. The comparison
    /// here is on the same strings `acvus_extern::Shape::of` resolves and
    /// compares, and `acvus-extern/tests/owned_holders.rs::
    /// a_derived_structs_field_table_is_the_shape_order` pins the two equal.
    fn of(fields: &'a syn::FieldsNamed) -> Self {
        let mut sorted: Vec<(String, &'a Ident, &'a syn::Type)> = fields
            .named
            .iter()
            .map(|f| {
                let ident = f.ident.as_ref().expect("named");
                (ident.to_string(), ident, &f.ty)
            })
            .collect();
        sorted.sort_by(|(a, ..), (b, ..)| a.cmp(b));
        Self {
            idents: sorted.iter().map(|(_, ident, _)| *ident).collect(),
            names: sorted.iter().map(|(name, ..)| name.clone()).collect(),
            tys: sorted.iter().map(|(.., ty)| *ty).collect(),
        }
    }

    fn fields(&self) -> proc_macro2::TokenStream {
        let (names, tys) = (&self.names, &self.tys);
        quote! {
            [#((
                __i.intern(#names),
                <#tys as ::acvus_extern::TyArg>::poly_ty(__i, __vars),
            )),*]
            .into_iter()
            .collect()
        }
    }

    /// The type of the struct `name` declares: these fields and no others,
    /// so an object reaching it has all of them (RFC-0042).
    fn declared_poly_ty(&self, name: &str) -> proc_macro2::TokenStream {
        let fields = self.fields();
        quote! {
            ::acvus_extern::PolyTy::Object(
                ::acvus_extern::ObjectTy::declared(__i.intern(#name), #fields),
            )
        }
    }

    /// The type of a struct variant's payload, which an enum declares as
    /// the object its fields spell.
    fn written_poly_ty(&self) -> proc_macro2::TokenStream {
        let fields = self.fields();
        quote! {
            ::acvus_extern::PolyTy::Object(::acvus_extern::ObjectTy::written(#fields))
        }
    }

    /// Erases the fields reached as `#owner.field`; `acvus_extern::object`
    /// owns the layout they go into.
    fn erase(&self, owner: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {
            ::acvus_extern::object::object_in_order::<__R, #width>(
                __rt,
                [#(#names),*],
                [#(::acvus_extern::erase_field::<#tys, __R>(__rt, #owner.#idents)),*],
            )
        }
    }

    /// The struct's components, written into the caller's destination run in
    /// the same order `erase` writes them, which is rule 8's.
    ///
    /// The width is a literal because the derive counts the fields here.
    /// `Run<W>` reached through a type parameter's associated constant would
    /// need `generic_const_exprs`, which is unstable on the pinned
    /// toolchain.
    fn returned(&self) -> proc_macro2::TokenStream {
        let (idents, tys) = (&self.idents, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {
            type Form = ::acvus_extern::Run<#width>;

            fn into_run(self, __rt: &__R, __out: &mut [<__R as ::acvus_extern::Runtime>::Value]) {
                ::acvus_extern::object::fields_into_run::<__R, #width>(
                    [#(::acvus_extern::erase_field::<#tys, __R>(__rt, self.#idents)),*],
                    __out,
                )
            }
        }
    }

    /// Erases fields already bound to their own idents (a matched variant).
    fn erase_bound(&self) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {
            ::acvus_extern::object::object_in_order::<__R, #width>(
                __rt,
                [#(#names),*],
                [#(::acvus_extern::erase_field::<#tys, __R>(__rt, #idents)),*],
            )
        }
    }

    /// `SRef<'a>` and `SMut<'a>`: one field per declared field, each a
    /// borrow of the value in the object's own storage, and the impls that
    /// build them out of an object the caller lent (RFC-0050 rule 6).
    ///
    /// The acvus type of a projection is `&S` over an **at-least** field set,
    /// where the struct's own is `Declared`: a projection naming a subset of
    /// the object's fields borrows those alone, which `ObjectTy::meet`
    /// already admits and `acvus-mir-test/tests/projection_parameter.rs`
    /// pins.
    ///
    /// Rule 6 gives every derived struct a projection and `#[projection]`
    /// asks for one, which is less. A projection type carries a lifetime and
    /// no runtime, so each of its fields must name a borrow that does not
    /// mention the runtime either — `&'a String`, `&'a i64`, a nested
    /// projection. A container field refutes that: the language stores a
    /// `Vec<T>` as `Vec<Owned<Rt>>`, so the borrow of such a field is a
    /// borrow of the runtime's own values and only `SRef<'a, Rt>` could name
    /// it. Until the projection carries the runtime, a struct with a
    /// container field has no projection to emit.
    fn projection(&self, owner: &Ident) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        let shared = format_ident!("{owner}Ref");
        let exclusive = format_ident!("{owner}Mut");
        let fields = self.fields();
        let width = syn::Index::from(idents.len());
        let doc_shared = format!("A shared projection of [`{owner}`] (RFC-0050 rule 6).");
        let doc_exclusive = format!("An exclusive projection of [`{owner}`] (RFC-0050 rule 6).");
        let at_least = quote! {
            ::acvus_extern::PolyTy::Object(::acvus_extern::ObjectTy::at_least(#fields))
        };
        let ats: Vec<syn::Index> = (0..idents.len()).map(syn::Index::from).collect();
        let table_ty = quote! {
            ::acvus_extern::ObjectAt<
                #width,
                (#(<#tys as ::acvus_extern::Project<__R>>::Table,)*),
            >
        };
        let table_of = quote! {
            let [#(#idents),*] = ::acvus_extern::object_fields_at(__at, [#(#names),*]);
            ::acvus_extern::ObjectAt {
                at: [#(#idents.0),*],
                fields: (#(
                    <#tys as ::acvus_extern::Project<__R>>::table(#idents.1),
                )*),
            }
        };
        quote! {
            #[doc = #doc_shared]
            pub struct #shared<'__a> {
                #(pub #idents: <#tys as ::acvus_extern::Borrowed>::Ref<'__a>,)*
            }

            #[doc = #doc_exclusive]
            pub struct #exclusive<'__a> {
                #(pub #idents: <#tys as ::acvus_extern::Borrowed>::Mut<'__a>,)*
            }

            impl ::acvus_extern::Borrowed for #owner {
                type Ref<'__a> = #shared<'__a> where Self: '__a;
                type Mut<'__a> = #exclusive<'__a> where Self: '__a;
            }

            impl<'__a> #shared<'__a> {
                /// # Safety
                /// `obj` holds what the owner's crossing wrote and is live
                /// for `'__a`.
                pub unsafe fn over<__R>(
                    __rt: &'__a __R,
                    __obj: &'__a ::acvus_extern::Obj<::acvus_extern::Owned<__R>>,
                    __table: &#table_ty,
                ) -> Self
                where
                    __R: ::acvus_extern::Runtime,
                {
                    let __fields = ::acvus_extern::Fields::of(__rt, __obj);
                    let [#(#idents),*] = __table.at;
                    Self {
                        #(#idents: {
                            // SAFETY: the caller's contract, and the field
                            // at that position holds what this field's own
                            // crossing wrote.
                            unsafe {
                                <#tys as ::acvus_extern::Project<__R>>::project(
                                    __rt,
                                    __fields.field(#idents),
                                    &__table.fields.#ats,
                                )
                            }
                        },)*
                    }
                }
            }

            impl<'__a> #exclusive<'__a> {
                /// # Safety
                /// As the shared projection's `over`, and `obj` is
                /// exclusively named for `'__a`.
                pub unsafe fn over<__R>(
                    __rt: &'__a __R,
                    __obj: &'__a mut ::acvus_extern::Obj<::acvus_extern::Owned<__R>>,
                    __table: &#table_ty,
                ) -> Self
                where
                    __R: ::acvus_extern::Runtime,
                {
                    let __fields = ::acvus_extern::FieldsMut::of(__rt, __obj);
                    let [#(#idents),*] = __fields.disjoint::<#width>(__table.at);
                    Self {
                        // SAFETY: as the shared projection's, exclusively.
                        #(#idents: unsafe {
                            <#tys as ::acvus_extern::Project<__R>>::project_mut(
                                __rt,
                                #idents,
                                &__table.fields.#ats,
                            )
                        },)*
                    }
                }
            }

            impl<__R> ::acvus_extern::Project<__R> for #owner
            where
                __R: ::acvus_extern::Runtime,
            {
                type Table = #table_ty;

                fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                    #table_of
                }

                unsafe fn project<'__a>(
                    __rt: &'__a __R,
                    __value: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #shared<'__a> {
                    // SAFETY: the caller's contract: a nested aggregate field
                    // holds its own object.
                    unsafe {
                        #shared::over(
                            __rt,
                            ::acvus_extern::object_in(__rt, __value),
                            __table,
                        )
                    }
                }

                unsafe fn project_mut<'__a>(
                    __rt: &'__a __R,
                    __value: &'__a mut <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #exclusive<'__a> {
                    // SAFETY: as `project`, with the caller's exclusive loan.
                    unsafe {
                        #exclusive::over(
                            __rt,
                            ::acvus_extern::object_in_mut(__rt, __value),
                            __table,
                        )
                    }
                }
            }

            impl<'__x, __R> ::acvus_extern::Projected<__R> for #shared<'__x>
            where
                __R: ::acvus_extern::Runtime,
            {
                type At<'__a> = #shared<'__a>;
                type Table = #table_ty;

                fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                    #table_of
                }

                unsafe fn of<'__a>(
                    __rt: &'__a __R,
                    __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #shared<'__a> {
                    // SAFETY: the caller's contract: a live object storage.
                    unsafe {
                        #shared::over(
                            __rt,
                            ::acvus_extern::object_of(__rt, __reference),
                            __table,
                        )
                    }
                }
            }

            impl<'__x, __R> ::acvus_extern::Projected<__R> for #exclusive<'__x>
            where
                __R: ::acvus_extern::Runtime,
            {
                type At<'__a> = #exclusive<'__a>;
                type Table = #table_ty;

                fn table(__at: ::acvus_extern::ArgAt<'_>) -> Self::Table {
                    #table_of
                }

                unsafe fn of<'__a>(
                    __rt: &'__a __R,
                    __reference: &'__a <__R as ::acvus_extern::Runtime>::Value,
                    __table: &Self::Table,
                ) -> #exclusive<'__a> {
                    // SAFETY: as the shared projection's, exclusively.
                    unsafe {
                        #exclusive::over(
                            __rt,
                            ::acvus_extern::object_of_mut(__rt, __reference),
                            __table,
                        )
                    }
                }
            }

            impl ::acvus_extern::TyArg for #shared<'static> {
                fn poly_ty(
                    __i: &::acvus_extern::Interner,
                    __vars: &::acvus_extern::PolyVars,
                ) -> ::acvus_extern::PolyTy {
                    ::acvus_extern::PolyTy::Ref(
                        ::acvus_extern::Mutability::Shared,
                        ::std::boxed::Box::new(::acvus_extern::TypeArg::uniform(#at_least)),
                    )
                }
            }

            impl ::acvus_extern::TyArg for #exclusive<'static> {
                fn poly_ty(
                    __i: &::acvus_extern::Interner,
                    __vars: &::acvus_extern::PolyVars,
                ) -> ::acvus_extern::PolyTy {
                    ::acvus_extern::PolyTy::Ref(
                        ::acvus_extern::Mutability::Mut,
                        ::std::boxed::Box::new(::acvus_extern::TypeArg::uniform(#at_least)),
                    )
                }
            }
        }
    }

    /// Materializes `#value` into `#path { fields }`.
    fn materialize(
        &self,
        value: proc_macro2::TokenStream,
        path: proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let (idents, tys) = (&self.idents, &self.tys);
        let width = syn::Index::from(idents.len());
        quote! {{
            // SAFETY: the caller's contract, and `erase` built this object, so
            // its width is this table's and the destructuring is exhaustive.
            let [#(#idents),*] = unsafe {
                ::acvus_extern::object::open_in_order::<__R, #width>(__rt, #value)
            };
            #path {
                // SAFETY: the caller's contract, forwarded: `erase` erased each
                // field from its declared type, at this position.
                #(#idents: unsafe {
                    ::acvus_extern::materialize_field::<#tys, __R>(__rt, #idents)
                },)*
            }
        }}
    }
}

/// A Rust enum is the language's enum of the same name: a unit variant has
/// no payload, a one-field tuple variant's payload is that field, and a
/// struct variant's payload is the object its fields spell.
fn generate_enum_ty_arg(
    ident: &Ident,
    data: &syn::DataEnum,
) -> syn::Result<proc_macro2::TokenStream> {
    let name = ident.to_string();
    let Some(last) = data.variants.len().checked_sub(1) else {
        return Err(syn::Error::new(
            ident.span(),
            "an enum declares at least one variant: a type with no value is `Never`",
        ));
    };
    let mut tags = Vec::new();
    let mut variant_tys = Vec::new();
    let mut erase_arms = Vec::new();
    let mut materialize_arms = Vec::new();
    for (at, variant) in data.variants.iter().enumerate() {
        let v = &variant.ident;
        let tag = v.to_string();
        tags.push(tag.clone());
        // The last position absorbs the index, which `variant::opened`
        // already proved to be one of `tags`.
        let matched = match at == last {
            true => quote! { _ },
            false => {
                let at = proc_macro2::Literal::usize_unsuffixed(at);
                quote! { #at }
            }
        };
        match &variant.fields {
            syn::Fields::Unit => {
                variant_tys.push(quote! { (__i.intern(#tag), ::core::option::Option::None) });
                erase_arms.push(quote! { Self::#v => (#tag, ::core::option::Option::None) });
                materialize_arms.push(quote! { #matched => Self::#v });
            }
            syn::Fields::Unnamed(fields) => {
                let mut tys = fields.unnamed.iter().map(|f| &f.ty);
                let (Some(ty), None) = (tys.next(), tys.next()) else {
                    return Err(syn::Error::new_spanned(
                        &variant.fields,
                        "a tuple variant has one field, the variant's payload",
                    ));
                };
                variant_tys.push(quote! {
                    (
                        __i.intern(#tag),
                        ::core::option::Option::Some(::std::boxed::Box::new(
                            <#ty as ::acvus_extern::TyArg>::poly_ty(__i, __vars),
                        )),
                    )
                });
                erase_arms.push(quote! {
                    Self::#v(__payload) => (
                        #tag,
                        ::core::option::Option::Some(
                            ::acvus_extern::erase_field::<#ty, __R>(__rt, __payload),
                        ),
                    )
                });
                materialize_arms.push(quote! {
                    #matched => {
                        // SAFETY: the caller's contract, forwarded: `erase` erased
                        // this variant's payload from its declared type.
                        Self::#v(unsafe { ::acvus_extern::materialize_payload::<#ty, __R>(
                            __rt, __payload, #tag,
                        ) })
                    }
                });
            }
            syn::Fields::Named(fields) => {
                let shape = ObjectShape::of(fields);
                let idents = &shape.idents;
                let ty = shape.written_poly_ty();
                let erase = shape.erase_bound();
                let materialize = shape.materialize(
                    quote! { ::acvus_extern::take_payload(__payload, #tag).into_value() },
                    quote! { Self::#v },
                );
                variant_tys.push(quote! {
                    (__i.intern(#tag), ::core::option::Option::Some(::std::boxed::Box::new(#ty)))
                });
                erase_arms.push(quote! {
                    Self::#v { #(#idents),* } => (
                        #tag,
                        ::core::option::Option::Some(
                            ::acvus_extern::Owned::from_value(#erase),
                        ),
                    )
                });
                materialize_arms.push(quote! { #matched => #materialize });
            }
        }
    }

    let ty = quote! {
        ::acvus_extern::PolyTy::Enum {
            name: __i.intern(#name),
            variants: [#(#variant_tys),*].into_iter().collect(),
        }
    };
    let erase = quote! {{
        let (__tag, __payload) = match self { #(#erase_arms,)* };
        ::acvus_extern::variant::erase(__rt, __tag, __payload)
    }};
    let materialize = quote! {{
        // SAFETY: the caller's contract, and `erase` wrote this variant.
        let ::acvus_extern::variant::Opened { at: __at, payload: __payload } = unsafe {
            ::acvus_extern::variant::opened::<__R>(__rt, __value, #name, &[#(#tags),*])
        };
        match __at { #(#materialize_arms,)* }
    }};
    Ok(cross_impl(
        ident,
        Crossing {
            ty,
            erase,
            materialize,
            borrowing: Borrowing::Whole,
            returned: returned_as_one_value(),
        },
    ))
}

// -- extern_registry! ------------------------------------------------

/// `extern_registry! { ns: "core", types: [Vec<_>], signatures: [eq],
/// fns: [len, chat(client)] }` (RFC-0021).
struct RegistryInput {
    ns: LitStr,
    types: Vec<Type>,
    signatures: Vec<Path>,
    fns: Vec<RegistryFn>,
}

/// One listed function and the state values its declaration takes.
struct RegistryFn {
    path: Path,
    state: Vec<syn::Expr>,
}

impl Parse for RegistryFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let state = if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            content
                .parse_terminated(syn::Expr::parse, Token![,])?
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self { path, state })
    }
}

impl Parse for RegistryInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut ns = None;
        let mut types = Vec::new();
        let mut signatures = Vec::new();
        let mut fns = Vec::new();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            if key == "ns" {
                ns = Some(input.parse::<LitStr>()?);
            } else {
                let content;
                syn::bracketed!(content in input);
                if key == "types" {
                    let list: Punctuated<Type, Token![,]> =
                        content.parse_terminated(Type::parse, Token![,])?;
                    types.extend(list);
                } else if key == "signatures" {
                    let list: Punctuated<Path, Token![,]> =
                        content.parse_terminated(Path::parse, Token![,])?;
                    signatures.extend(list);
                } else if key == "fns" {
                    let list: Punctuated<RegistryFn, Token![,]> =
                        content.parse_terminated(RegistryFn::parse, Token![,])?;
                    fns.extend(list);
                } else {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `ns`, `types`, `signatures`, or `fns`",
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        let Some(ns) = ns else {
            return Err(syn::Error::new(
                Span::call_site(),
                "a registry declares its namespace: `ns: \"...\"`",
            ));
        };
        Ok(Self {
            ns,
            types,
            signatures,
            fns,
        })
    }
}

#[proc_macro]
pub fn extern_registry(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as RegistryInput);
    let ns = input.ns;
    let types: Vec<Type> = input.types.iter().map(subst::infer_to_unit).collect();
    let signatures = input.signatures;
    let fns = input.fns.into_iter().map(|f| {
        let mut path = f.path;
        let last = path.segments.last_mut().expect("path has a segment");
        last.ident = format_ident!("__extern_fn_{}", last.ident);
        let state = f.state;
        quote! { #path(__i, __ns, #(#state),*) }
    });
    quote! {
        ::acvus_extern::Registry::new(move |__i: &::acvus_extern::Interner| {
            let __ns: ::core::option::Option<&str> = ::core::option::Option::Some(#ns);
            let mut __contribution = ::acvus_extern::Contribution::of(::acvus_extern::Manifest {
                types: vec![#(<#types as ::acvus_extern::ExternTypeDecl>::type_decl(__i)),*],
                signatures: vec![#(
                    <#signatures as ::acvus_extern::SharedSignature>::signature_decl(__i)
                ),*],
                fns: ::std::vec::Vec::new(),
            });
            #(
                if let ::core::option::Option::Some(__hooks) =
                    <#types as ::acvus_extern::ExternTypeDecl>::space()
                {
                    __contribution.register_space(
                        <#types as ::acvus_extern::ExternTypeDecl>::type_decl(__i).qref,
                        __hooks,
                    );
                }
            )*
            for __f in ::std::vec::Vec::<::std::vec::Vec<::acvus_extern::ExternFn<_>>>::from([#(#fns),*])
                .into_iter()
                .flatten()
            {
                __contribution.declare(__f);
            }
            __contribution
        })
    }
    .into()
}

// -- extern_signature! -----------------------------------------------

/// `extern_signature! { ns: "core", fn eq<T>(a: &T, b: &T) -> bool where T: TyVar; }`
/// declares a shared signature (RFC-0019) and a marker type named after it.
struct SignatureInput {
    ns: LitStr,
    sig: syn::Signature,
}

impl Parse for SignatureInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        if key != "ns" {
            return Err(syn::Error::new(key.span(), "expected `ns`"));
        }
        input.parse::<Token![:]>()?;
        let ns: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let sig: syn::Signature = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
        }
        Ok(Self { ns, sig })
    }
}

#[proc_macro]
pub fn extern_signature(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as SignatureInput);
    match generate_signature(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_signature(input: SignatureInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut sig = input.sig;
    let ident = sig.ident.clone();
    let vars = Vars::from_generics(&sig.generics)?;
    let Signature {
        params: rust_params,
        ..
    } = parse_params(&mut sig, None)?;
    let mut params = Vec::new();
    for p in rust_params {
        match p {
            RustParam::Acvus(a) => params.push(a),
            RustParam::State(st) => {
                return Err(syn::Error::new(
                    st.ident.span(),
                    "a signature declares no state",
                ));
            }
        }
    }
    let ret = parse_return(&sig.output);
    let name = ident.to_string();
    let qref = qref_expr_in(Some(&input.ns.value()), &name);
    let types_only = |ty: &Type| {
        subst::substitute(ty, &|ident| {
            (ident == "__R").then(|| syn::parse_quote! { ::acvus_extern::TypesOnly })
        })
    };
    let param_terms = params.iter().map(|p| {
        let pname = &p.name;
        let comp_ty = p.mode.acvus_ty(
            &types_only(&vars.to_compile_time_instance(&p.ty, None)),
            &quote! { ::acvus_extern::TypesOnly },
        );
        quote! {
            ::acvus_extern::ParamTerm::<::acvus_extern::Poly>::new(
                __i.intern(#pname),
                <#comp_ty as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
            )
        }
    });
    let comp_ret = types_only(&vars.to_compile_time_instance(&ret, None));
    let bounds = vars.bound_exprs();
    let counts = vars.counts_expr();
    Ok(quote! {
        #[allow(non_camel_case_types)]
        pub struct #ident;

        impl ::acvus_extern::SharedSignature for #ident {
            fn qref(__i: &::acvus_extern::Interner) -> ::acvus_extern::QualifiedRef {
                #qref
            }
            fn signature_decl(__i: &::acvus_extern::Interner) -> ::acvus_extern::SignatureDecl {
                let __vars = ::acvus_extern::PolyVars::fresh(#counts);
                ::acvus_extern::SignatureDecl {
                    qref: #qref,
                    ty: ::acvus_extern::PolyTy::Fn {
                        params: vec![#(#param_terms),*],
                        ret: Box::new(<#comp_ret as ::acvus_extern::TyArg>::poly_ty(__i, &__vars)),
                        captures: vec![],
                        effect: ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::PURE),
                    },
                    bounds: vec![#(#bounds),*],
                }
            }
        }
    })
}

// -- shared helpers --------------------------------------------------

fn bound_ident(bound: &TypeParamBound) -> Option<&Ident> {
    match bound {
        TypeParamBound::Trait(t) => t.path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}

fn span_of(param: &GenericParam) -> Span {
    match param {
        GenericParam::Type(t) => t.ident.span(),
        GenericParam::Lifetime(l) => l.lifetime.span(),
        GenericParam::Const(c) => c.ident.span(),
    }
}
