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
}

impl Parse for ExternFnAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut out = ExternFnAttr {
            name: None,
            instance_of: None,
            effect: None,
            commutative: false,
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
            input.parse::<Token![=]>()?;
            if key == "name" {
                out.name = Some(input.parse()?);
            } else if key == "instance_of" {
                out.instance_of = Some(input.parse()?);
            } else if key == "effect" {
                out.effect = Some(input.parse()?);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    "expected `name`, `instance_of`, `effect`, or `commutative`",
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
}

impl Mode {
    /// The acvus type of a parameter whose Rust type is `ty` under this
    /// mode, with `rt` as the runtime a reference carrier names.
    fn acvus_ty(self, ty: &Type, rt: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Mode::Value => quote! { #ty },
            Mode::Borrow => quote! { ::acvus_extern::Ref<#ty, #rt> },
            Mode::BorrowMut => quote! { ::acvus_extern::RefMut<#ty, #rt> },
        }
    }
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

/// The declared return: the acvus type, and whether the Rust function wraps
/// it in `Result`.
struct ExternReturn {
    ty: Type,
    is_result: bool,
}

fn generate_extern_fn(
    attr: ExternFnAttr,
    func: &mut ItemFn,
) -> syn::Result<proc_macro2::TokenStream> {
    let is_cast = take_marker_attr(&mut func.attrs, "extern_cast");
    let is_async = func.sig.asyncness.is_some();
    let vars = Vars::from_generics(&func.sig.generics)?;
    let (has_runtime, rust_params) = parse_params(&mut func.sig, vars.runtime_ident())?;
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

    let commutes = if attr.commutative {
        quote! { .commutative() }
    } else {
        quote! {}
    };
    let effect = match &attr.effect {
        None => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::OPAQUE #commutes) }
        }
        Some(e) if e == "pure" => {
            if attr.commutative {
                return Err(syn::Error::new(
                    e.span(),
                    "a pure function commutes by definition; drop `commutative`",
                ));
            }
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::PURE) }
        }
        Some(e) if e == "idempotent" => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::IDEMPOTENT #commutes) }
        }
        Some(e) if e == "opaque" => {
            quote! { ::acvus_extern::EffectTerm::Known(::acvus_extern::Effect::OPAQUE #commutes) }
        }
        Some(e) => match vars.lookup(e) {
            Some((VarKind::Effect, k)) => {
                if attr.commutative {
                    return Err(syn::Error::new(
                        e.span(),
                        "`commutative` cannot be declared on an effect variable",
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
        let comp_ret = vars.to_compile_time_instance(&ret.ty, member);
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
    let glue = |member: Option<&Type>| -> proc_macro2::TokenStream {
        let rt_tys: Vec<Type> = params
            .iter()
            .map(|p| vars.to_runtime_instance(&p.ty, member))
            .collect();
        let rt_ret = vars.to_runtime_instance(&ret.ty, member);
        let turbofish = vars.runtime_turbofish_instance(member);
        let error_ty = quote! { <__R as ::acvus_extern::Runtime>::Error };
        let unpack_stmts: Vec<proc_macro2::TokenStream> = params
            .iter()
            .zip(&arg_idents)
            .zip(&rt_tys)
            .map(|((p, a), ty)| {
                let next = quote! { __args.next().expect("arity checked by typeck") };
                let lent = format_ident!("{a}_lent");
                match p.mode {
                    Mode::Value if is_carrier(ty) => quote! { let #a = <#ty>::new(#next); },
                    Mode::Value => quote! { let #a = unsafe { (&::acvus_extern::Crossing::<#ty, __R>::new()).materialize(__rt, #next) }; },
                    Mode::Borrow => quote! {
                        let #lent = #next;
                        let #a: &#ty = unsafe { (&::acvus_extern::Crossing::<#ty, __R>::new()).deref(__rt, &#lent) };
                    },
                    Mode::BorrowMut => quote! {
                        let #lent = #next;
                        let #a: &mut #ty = unsafe { (&::acvus_extern::Crossing::<#ty, __R>::new()).deref_mut(__rt, &#lent) };
                    },
                }
            })
            .collect();
        let unpack = quote! {
            #[allow(unused_imports)]
            use ::acvus_extern::{AsCross as _, AsIs as _};
            let mut __args = __args.into_iter();
            #(#unpack_stmts)*
            debug_assert!(__args.next().is_none(), "arity checked by typeck");
        };
        let mut acvus_args = arg_idents.iter();
        let passed: Vec<proc_macro2::TokenStream> = rust_params
            .iter()
            .map(|p| match p {
                RustParam::Acvus(_) => {
                    let a = acvus_args.next().expect("one ident per acvus parameter");
                    quote! { #a }
                }
                RustParam::State(st) => {
                    let ident = &st.ident;
                    quote! { &*#ident }
                }
            })
            .collect();
        let state_idents: Vec<&Ident> = states.iter().map(|st| &st.ident).collect();
        let hold_state = quote! {
            #(let #state_idents = ::std::sync::Arc::clone(&#state_idents);)*
        };
        let ret_value = if is_carrier(&rt_ret) {
            quote! { __r.into_value() }
        } else if is_option_of_carrier(&rt_ret) {
            quote! { ::acvus_extern::Carried::into_value(__r, __rt) }
        } else {
            quote! { unsafe { (&::acvus_extern::Crossing::<#rt_ret, __R>::new()).erase(__rt, __r) } }
        };
        let returned = quote! {
            ::core::result::Result::<_, #error_ty>::Ok(#ret_value)
        };
        let rt_arg = has_runtime.then(|| quote! { __rt, });
        if is_async {
            let call = quote! { #fn_ident #turbofish (#rt_arg #(#passed),*) };
            let awaited = if ret.is_result {
                quote! { (#call).await.map_err(::core::convert::Into::<#error_ty>::into)? }
            } else {
                quote! { (#call).await }
            };
            quote! {{
                #hold_state
                ::acvus_extern::ExternHandler::Async(::std::sync::Arc::new(
                    move |__rt: __R, __args: ::std::vec::Vec<<__R as ::acvus_extern::Runtime>::Value>| {
                        #hold_state
                        ::std::boxed::Box::pin(async move {
                            let __rt = &__rt;
                            #unpack
                            let __r = #awaited;
                            #returned
                        })
                    }
                ))
            }}
        } else {
            let call = quote! { #fn_ident #turbofish (#rt_arg #(#passed),*) };
            let result = if ret.is_result {
                quote! { (#call).map_err(::core::convert::Into::<#error_ty>::into)? }
            } else {
                quote! { #call }
            };
            quote! {{
                #hold_state
                ::acvus_extern::ExternHandler::Sync(::std::sync::Arc::new(
                    move |__rt: &__R, __args: ::std::vec::Vec<<__R as ::acvus_extern::Runtime>::Value>| {
                        #unpack
                        let __r = #result;
                        #returned
                    }
                ))
            }}
        }
    };

    let handler = match vars.mono_var() {
        None => {
            let single = glue(None);
            quote! { ::acvus_extern::ExternEntry::Single(#single) }
        }
        Some(mono) => {
            let members = mono.mono.as_ref().expect("mono_var has members");
            let instances = members.iter().map(|member| {
                let handler = glue(Some(member));
                let signature = signature(Some(member));
                quote! {
                    ::acvus_extern::MonoInstance {
                        signature: #signature,
                        handler: #handler,
                    }
                }
            });
            let fallback = if mono.mono_fallback {
                let fallback_signature = signature(None);
                let fallback_handler = glue(None);
                quote! {
                    ::acvus_extern::MonoInstance {
                        signature: #fallback_signature,
                        handler: #fallback_handler,
                    },
                }
            } else {
                quote! {}
            };
            quote! {
                ::acvus_extern::ExternEntry::Mono(::acvus_extern::MonoHandler {
                    instances: vec![
                        #(#instances,)*
                        #fallback
                    ],
                })
            }
        }
    };
    let bounds = vars.bound_exprs();
    let requires = vars.requires_exprs();
    let declared_ty = signature(None);

    let counts = vars.counts_expr();
    let rt_bounds = quote! { __R: ::acvus_extern::Runtime, };
    let state_idents: Vec<&Ident> = states.iter().map(|st| &st.ident).collect();
    let state_tys: Vec<&Type> = states.iter().map(|st| &st.ty).collect();
    Ok(quote! {
        #func

        #[doc(hidden)]
        #vis fn #decl_ident<__R>(
            __i: &::acvus_extern::Interner,
            __ns: ::core::option::Option<&str>,
            #(#state_idents: #state_tys,)*
        ) -> ::acvus_extern::ExternFn<__R>
        where
            #rt_bounds
            #(#state_tys: ::core::marker::Send + ::core::marker::Sync + 'static,)*
        {
            let __vars = ::acvus_extern::PolyVars::fresh(#counts);
            #(let #state_idents = ::std::sync::Arc::new(#state_idents);)*
            ::acvus_extern::ExternFn {
                decl: ::acvus_extern::FnDecl {
                    qref: #qref,
                    ty: #declared_ty,
                    bounds: vec![#(#bounds),*],
                    cast: #is_cast,
                    instance_of: #instance_of,
                    requires: vec![#(#requires),*],
                },
                handler: #handler,
            }
        }
    })
}

/// `Fn0`, `Fn1`, `Fn2` carry a closure value by name; they wrap it rather
/// than materialize it.
fn is_carrier(ty: &Type) -> bool {
    let Type::Path(p) = ty else {
        return false;
    };
    p.path.segments.last().is_some_and(|s| {
        matches!(
            s.ident.to_string().as_str(),
            "Fn0" | "Fn1" | "Fn2" | "Fn3" | "Ref" | "RefMut"
        )
    })
}

fn is_option_of_carrier(ty: &Type) -> bool {
    let Type::Path(p) = ty else {
        return false;
    };
    let Some(last) = p.path.segments.last() else {
        return false;
    };
    if last.ident != "Option" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return false;
    };
    matches!(args.args.first(), Some(syn::GenericArgument::Type(inner)) if is_carrier(inner))
}

/// Remove `#[name]` from the attribute list; report whether it was there.
fn take_marker_attr(attrs: &mut Vec<Attribute>, name: &str) -> bool {
    let before = attrs.len();
    attrs.retain(|a| !a.path().is_ident(name));
    attrs.len() != before
}

/// The parameters of a signature, `#[state]` markers taken off, and whether
/// the first one was the runtime: `&R` with `R` the parameter bounded by
/// `Runtime`. A function that does not use its runtime does not take it.
fn parse_params(
    sig: &mut syn::Signature,
    runtime: Option<&Ident>,
) -> syn::Result<(bool, Vec<RustParam>)> {
    let mut inputs = sig.inputs.iter_mut().peekable();
    let has_runtime = match (runtime, inputs.peek()) {
        (Some(runtime), Some(first)) => is_runtime_param(first, runtime),
        _ => false,
    };
    if has_runtime {
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
            Type::Reference(r) if r.mutability.is_some() => ((*r.elem).clone(), Mode::BorrowMut),
            Type::Reference(r) => ((*r.elem).clone(), Mode::Borrow),
            ty => (ty.clone(), Mode::Value),
        };
        params.push(RustParam::Acvus(ExternParam {
            name: ident.to_string(),
            ty,
            mode,
        }));
    }
    Ok((has_runtime, params))
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

fn parse_return(output: &ReturnType) -> ExternReturn {
    match output {
        ReturnType::Default => ExternReturn {
            ty: syn::parse_quote! { () },
            is_result: false,
        },
        ReturnType::Type(_, ty) => {
            if let Type::Path(p) = ty.as_ref()
                && let Some(seg) = p.path.segments.last()
                && seg.ident == "Result"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(ok)) = args.args.first()
            {
                return ExternReturn {
                    ty: ok.clone(),
                    is_result: true,
                };
            }
            ExternReturn {
                ty: (**ty).clone(),
                is_result: false,
            }
        }
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

    Ok(quote! {
        impl #arg_impl_generics ::acvus_extern::TyArg for #ident #ty_generics #where_clause {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                ::acvus_extern::PolyTy::UserDefined {
                    id: #qref,
                    type_args: vec![#(#type_arg_exprs),*],
                    effect_args: vec![#(#effect_arg_exprs),*],
                    identity_args: vec![#(#identity_arg_exprs),*],
                }
            }
        }

        impl #impl_generics ::acvus_extern::ExternTypeDecl for #ident #ty_generics #where_clause {
            fn type_decl(__i: &::acvus_extern::Interner) -> ::acvus_extern::UserDefinedDecl {
                ::acvus_extern::UserDefinedDecl {
                    qref: #qref,
                    type_params: vec![::acvus_extern::TyVarBound::Any; #n_tys],
                    effect_params: #n_effects,
                    identity_params: #n_identities,
                }
            }
        }
    })
}

// -- #[derive(TyArg)] ------------------------------------------------

#[proc_macro_derive(TyArg)]
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
            let ty = shape.poly_ty();
            let erase = shape.erase(quote! { self });
            let materialize = shape.materialize(quote! { __value }, quote! { Self });
            Ok(cross_impl(ident, ty, erase, materialize))
        }
        syn::Data::Enum(data) => generate_enum_ty_arg(ident, data),
        syn::Data::Union(_) => Err(syn::Error::new(
            ident.span(),
            "TyArg is derived on a struct or an enum",
        )),
    }
}

/// The `TyArg` and `Cross` impls of a derived type, given its poly type
/// and its two crossings.
fn cross_impl(
    ident: &Ident,
    ty: proc_macro2::TokenStream,
    erase: proc_macro2::TokenStream,
    materialize: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    quote! {
        impl ::acvus_extern::TyArg for #ident {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                #ty
            }
        }

        impl<__R> ::acvus_extern::Cross<__R> for #ident
        where
            __R: ::acvus_extern::Runtime,
        {
            fn erase(self, __rt: &__R) -> <__R as ::acvus_extern::Runtime>::Value {
                #erase
            }

            fn materialize(__rt: &__R, __value: <__R as ::acvus_extern::Runtime>::Value) -> Self {
                #materialize
            }
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
    fn of(fields: &'a syn::FieldsNamed) -> Self {
        let idents: Vec<&Ident> = fields
            .named
            .iter()
            .map(|f| f.ident.as_ref().expect("named"))
            .collect();
        let names = idents.iter().map(|f| f.to_string()).collect();
        let tys = fields.named.iter().map(|f| &f.ty).collect();
        Self { idents, names, tys }
    }

    fn poly_ty(&self) -> proc_macro2::TokenStream {
        let (names, tys) = (&self.names, &self.tys);
        quote! {
            ::acvus_extern::PolyTy::Object(
                [#((
                    __i.intern(#names),
                    <#tys as ::acvus_extern::TyArg>::poly_ty(__i, __vars),
                )),*]
                .into_iter()
                .collect(),
            )
        }
    }

    /// Erases the fields reached as `#owner.field` into the runtime's object.
    fn erase(&self, owner: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        quote! {{
            let mut __fields = ::acvus_extern::FxHashMap::default();
            #(
                __fields.insert(
                    __rt.symbol(#names),
                    ::acvus_extern::erase_field::<#tys, __R>(__rt, #owner.#idents),
                );
            )*
            // SAFETY: the language's object is `Obj<Value>` (RFC-0032).
            unsafe {
                __rt.erase::<::acvus_extern::Obj<<__R as ::acvus_extern::Runtime>::Value>>(
                    ::acvus_extern::Obj(__fields),
                )
            }
        }}
    }

    /// Erases fields already bound to their own idents (a matched variant).
    fn erase_bound(&self) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        quote! {{
            let mut __fields = ::acvus_extern::FxHashMap::default();
            #(
                __fields.insert(
                    __rt.symbol(#names),
                    ::acvus_extern::erase_field::<#tys, __R>(__rt, #idents),
                );
            )*
            // SAFETY: the language's object is `Obj<Value>` (RFC-0032).
            unsafe {
                __rt.erase::<::acvus_extern::Obj<<__R as ::acvus_extern::Runtime>::Value>>(
                    ::acvus_extern::Obj(__fields),
                )
            }
        }}
    }

    /// Materializes `#value` into `#path { fields }`.
    fn materialize(
        &self,
        value: proc_macro2::TokenStream,
        path: proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let (idents, names, tys) = (&self.idents, &self.names, &self.tys);
        quote! {{
            // SAFETY: as in `erase`.
            let ::acvus_extern::Obj(mut __fields) = unsafe {
                __rt.materialize::<::acvus_extern::Obj<<__R as ::acvus_extern::Runtime>::Value>>(#value)
            };
            #path {
                #(#idents: ::acvus_extern::materialize_field::<#tys, __R>(
                    __rt, &mut __fields, #names,
                ),)*
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
    let mut variant_tys = Vec::new();
    let mut erase_arms = Vec::new();
    let mut materialize_arms = Vec::new();
    for variant in &data.variants {
        let v = &variant.ident;
        let tag = v.to_string();
        match &variant.fields {
            syn::Fields::Unit => {
                variant_tys.push(quote! { (__i.intern(#tag), ::core::option::Option::None) });
                erase_arms
                    .push(quote! { Self::#v => (__rt.symbol(#tag), ::core::option::Option::None) });
                materialize_arms.push(quote! { if __tag == __rt.symbol(#tag) { Self::#v } });
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
                        __rt.symbol(#tag),
                        ::core::option::Option::Some(::std::boxed::Box::new(
                            ::acvus_extern::erase_field::<#ty, __R>(__rt, __payload),
                        )),
                    )
                });
                materialize_arms.push(quote! {
                    if __tag == __rt.symbol(#tag) {
                        Self::#v(::acvus_extern::materialize_payload::<#ty, __R>(
                            __rt, __payload, #tag,
                        ))
                    }
                });
            }
            syn::Fields::Named(fields) => {
                let shape = ObjectShape::of(fields);
                let idents = &shape.idents;
                let ty = shape.poly_ty();
                let erase = shape.erase_bound();
                let materialize = shape.materialize(
                    quote! { ::acvus_extern::take_payload(__payload, #tag) },
                    quote! { Self::#v },
                );
                variant_tys.push(quote! {
                    (__i.intern(#tag), ::core::option::Option::Some(::std::boxed::Box::new(#ty)))
                });
                erase_arms.push(quote! {
                    Self::#v { #(#idents),* } => (
                        __rt.symbol(#tag),
                        ::core::option::Option::Some(::std::boxed::Box::new(#erase)),
                    )
                });
                materialize_arms.push(quote! { if __tag == __rt.symbol(#tag) { #materialize } });
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
        // SAFETY: the language's variant is `Variant<Value>`.
        unsafe {
            __rt.erase::<::acvus_extern::Variant<<__R as ::acvus_extern::Runtime>::Value>>(
                ::acvus_extern::Variant { tag: __tag, payload: __payload },
            )
        }
    }};
    let materialize = quote! {{
        // SAFETY: as in `erase`.
        let ::acvus_extern::Variant { tag: __tag, payload: __payload } = unsafe {
            __rt.materialize::<::acvus_extern::Variant<<__R as ::acvus_extern::Runtime>::Value>>(__value)
        };
        #(#materialize_arms else)*
        {
            panic!(
                "a variant not in enum `{}`: the checker admits only variants of the declared enum",
                #name,
            )
        }
    }};
    Ok(cross_impl(ident, ty, erase, materialize))
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
            let mut __fns: ::std::vec::Vec<::acvus_extern::FnDecl> = ::std::vec::Vec::new();
            let mut __handlers: ::acvus_extern::Handlers<_> = ::acvus_extern::FxHashMap::default();
            for __f in ::std::vec::Vec::<::acvus_extern::ExternFn<_>>::from([#(#fns),*]) {
                __handlers.insert(__f.decl.qref, __f.handler);
                __fns.push(__f.decl);
            }
            let mut __space = ::acvus_extern::FxHashMap::default();
            #(
                if let ::core::option::Option::Some(__hooks) =
                    <#types as ::acvus_extern::ExternTypeDecl>::space()
                {
                    __space.insert(
                        <#types as ::acvus_extern::ExternTypeDecl>::type_decl(__i).qref,
                        __hooks,
                    );
                }
            )*
            ::acvus_extern::Contribution {
                manifest: ::acvus_extern::Manifest {
                    types: vec![#(<#types as ::acvus_extern::ExternTypeDecl>::type_decl(__i)),*],
                    signatures: vec![#(
                        <#signatures as ::acvus_extern::SharedSignature>::signature_decl(__i)
                    ),*],
                    fns: __fns,
                },
                handlers: __handlers,
                space: __space,
            }
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
    let (_, rust_params) = parse_params(&mut sig, None)?;
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
    let comp_ret = types_only(&vars.to_compile_time_instance(&ret.ty, None));
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
