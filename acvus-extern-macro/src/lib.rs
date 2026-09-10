//! Proc macros for acvus-extern. See RFC-0009.
//!
//! - `#[extern_fn]`: a Rust function declares an ExternFn.
//! - `#[derive(ExternType)]`: a Rust struct declares an extension type.
//! - `#[derive(TyArg)]`: a Rust struct declares a structural object type.
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
    ns: Option<LitStr>,
    effect: Option<Ident>,
    /// `commutative`: two calls of this function in either order are the
    /// same program (RFC-0013).
    commutative: bool,
}

impl Parse for ExternFnAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut out = ExternFnAttr {
            name: None,
            ns: None,
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
            } else if key == "ns" {
                out.ns = Some(input.parse()?);
            } else if key == "effect" {
                out.effect = Some(input.parse()?);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    "expected `name`, `ns`, `effect`, or `commutative`",
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
    fn acvus(self) -> proc_macro2::TokenStream {
        match self {
            Mode::Value => quote! { ::acvus_extern::ParamMode::Value },
            Mode::Borrow => quote! { ::acvus_extern::ParamMode::Borrow },
            Mode::BorrowMut => quote! { ::acvus_extern::ParamMode::BorrowMut },
        }
    }
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
    let params = parse_params(func, is_async)?;
    let ret = parse_return(&func.sig.output);

    let fn_ident = &func.sig.ident;
    let vis = &func.vis;
    let acvus_name = attr
        .name
        .as_ref()
        .map(LitStr::value)
        .unwrap_or_else(|| fn_ident.to_string());
    let qref = qref_expr(attr.ns.as_ref().map(LitStr::value).as_deref(), &acvus_name);
    let decl_ident = format_ident!("__extern_fn_{}", fn_ident);

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
                quote! { __vars.effects[#k] }
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
            let comp_ty = vars.to_compile_time_instance(&p.ty, member);
            let mode = p.mode.acvus();
            quote! {
                ::acvus_extern::ParamTerm::<::acvus_extern::Poly>::new(
                    __i.intern(#name),
                    <#comp_ty as ::acvus_extern::TyArg>::poly_ty(__i, &__vars),
                )
                .with_mode(#mode)
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
        let rt_tys_lent: Vec<&Type> = params
            .iter()
            .zip(&rt_tys)
            .filter(|(p, _)| p.mode != Mode::Value)
            .map(|(_, t)| t)
            .collect();
        let rt_ret = vars.to_runtime_instance(&ret.ty, member);
        let turbofish = vars.runtime_turbofish_instance(member);
        let passed: Vec<proc_macro2::TokenStream> = params
            .iter()
            .zip(&arg_idents)
            .map(|(p, a)| match p.mode {
                Mode::Value => quote! { #a },
                Mode::Borrow => quote! { &#a },
                Mode::BorrowMut => quote! { &mut #a },
            })
            .collect();
        let lent: Vec<&Ident> = params
            .iter()
            .zip(&arg_idents)
            .filter(|(p, _)| p.mode != Mode::Value)
            .map(|(_, a)| a)
            .collect();
        let binds: Vec<proc_macro2::TokenStream> = params
            .iter()
            .zip(&arg_idents)
            .map(|(p, a)| match p.mode {
                Mode::BorrowMut => quote! { mut #a },
                _ => quote! { #a },
            })
            .collect();
        let call = quote! { #fn_ident #turbofish (__i, #(#passed),*) };
        if !lent.is_empty() {
            let give_back = quote! { (#(#lent,)*) };
            return match (is_async, ret.is_result) {
                (false, true) => quote! {
                    ::acvus_extern::into_sync_lending_handler::<__R, _, _, _, _>(
                        |__i: &::acvus_extern::Interner, (#(#binds,)*): (#(#rt_tys,)*)|
                            -> ::core::result::Result<(#rt_ret, (#(#rt_tys_lent,)*)), <__R as ::acvus_extern::Runtime>::Error> {
                                let __r = (#call).map_err(::core::convert::Into::<<__R as ::acvus_extern::Runtime>::Error>::into)?;
                                Ok((__r, #give_back))
                            }
                    )
                },
                (false, false) => quote! {
                    ::acvus_extern::into_sync_lending_handler::<__R, _, _, _, _>(
                        |__i: &::acvus_extern::Interner, (#(#binds,)*): (#(#rt_tys,)*)|
                            -> ::core::result::Result<(#rt_ret, (#(#rt_tys_lent,)*)), <__R as ::acvus_extern::Runtime>::Error> {
                                let __r = #call;
                                Ok((__r, #give_back))
                            }
                    )
                },
                (true, true) => quote! {
                    ::acvus_extern::into_async_lending_handler::<__R, _, _, _, _, _>(
                        |__i: ::acvus_extern::Interner, (#(#binds,)*): (#(#rt_tys,)*)| async move {
                            let __r = (#call).await.map_err(::core::convert::Into::<<__R as ::acvus_extern::Runtime>::Error>::into)?;
                            ::core::result::Result::<_, <__R as ::acvus_extern::Runtime>::Error>::Ok((__r, #give_back))
                        }
                    )
                },
                (true, false) => quote! {
                    ::acvus_extern::into_async_lending_handler::<__R, _, _, _, _, _>(
                        |__i: ::acvus_extern::Interner, (#(#binds,)*): (#(#rt_tys,)*)| async move {
                            let __r = #call.await;
                            ::core::result::Result::<_, <__R as ::acvus_extern::Runtime>::Error>::Ok((__r, #give_back))
                        }
                    )
                },
            };
        }
        match (is_async, ret.is_result) {
            (false, true) => quote! {
                ::acvus_extern::into_sync_extern_handler::<__R, _, _, _>(
                    |__i: &::acvus_extern::Interner, (#(#arg_idents,)*): (#(#rt_tys,)*)|
                        -> ::core::result::Result<#rt_ret, <__R as ::acvus_extern::Runtime>::Error> {
                            (#call).map_err(::core::convert::Into::into)
                        }
                )
            },
            (false, false) => quote! {
                ::acvus_extern::into_sync_extern_handler::<__R, _, _, _>(
                    |__i: &::acvus_extern::Interner, (#(#arg_idents,)*): (#(#rt_tys,)*)|
                        -> ::core::result::Result<#rt_ret, <__R as ::acvus_extern::Runtime>::Error> { Ok(#call) }
                )
            },
            (true, true) => quote! {
                ::acvus_extern::into_async_extern_handler::<__R, _, _, _, _>(
                    |__i: ::acvus_extern::Interner, (#(#arg_idents,)*): (#(#rt_tys,)*)| async move {
                        (#call).await.map_err(::core::convert::Into::into)
                    }
                )
            },
            (true, false) => quote! {
                ::acvus_extern::into_async_extern_handler::<__R, _, _, _, _>(
                    |__i: ::acvus_extern::Interner, (#(#arg_idents,)*): (#(#rt_tys,)*)| async move {
                        ::core::result::Result::<#rt_ret, <__R as ::acvus_extern::Runtime>::Error>::Ok(#call.await)
                    }
                )
            },
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
            quote! {
                ::acvus_extern::ExternEntry::Mono(::acvus_extern::MonoHandler {
                    instances: vec![#(#instances),*],
                })
            }
        }
    };
    let bounds = vars.bound_exprs();
    let declared_ty = signature(None);

    let counts = vars.counts_expr();
    Ok(quote! {
        #func

        #[doc(hidden)]
        #vis fn #decl_ident<__R>(__i: &::acvus_extern::Interner) -> ::acvus_extern::ExternFn<__R>
        where
            __R: ::acvus_extern::Runtime,
        {
            let __vars = ::acvus_extern::PolyVars::fresh(#counts);
            ::acvus_extern::ExternFn {
                qref: #qref,
                ty: #declared_ty,
                bounds: vec![#(#bounds),*],
                handler: #handler,
                cast: #is_cast,
            }
        }
    })
}

/// Remove `#[name]` from the attribute list; report whether it was there.
fn take_marker_attr(attrs: &mut Vec<Attribute>, name: &str) -> bool {
    let before = attrs.len();
    attrs.retain(|a| !a.path().is_ident(name));
    attrs.len() != before
}

fn parse_params(func: &ItemFn, is_async: bool) -> syn::Result<Vec<ExternParam>> {
    let mut inputs = func.sig.inputs.iter();
    let Some(first) = inputs.next() else {
        return Err(syn::Error::new(
            func.sig.ident.span(),
            "an extern_fn takes the interner as its first parameter",
        ));
    };
    check_interner_param(first, is_async)?;

    let mut params = Vec::new();
    for (i, arg) in inputs.enumerate() {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "an extern_fn has no self parameter",
            ));
        };
        let name = match pat_type.pat.as_ref() {
            Pat::Ident(p) => p.ident.to_string(),
            _ => format!("_{i}"),
        };
        let (ty, mode) = match pat_type.ty.as_ref() {
            Type::Reference(r) if r.mutability.is_some() => ((*r.elem).clone(), Mode::BorrowMut),
            Type::Reference(r) => ((*r.elem).clone(), Mode::Borrow),
            ty => (ty.clone(), Mode::Value),
        };
        params.push(ExternParam { name, ty, mode });
    }
    Ok(params)
}

fn check_interner_param(arg: &FnArg, is_async: bool) -> syn::Result<()> {
    let FnArg::Typed(pat_type) = arg else {
        return Err(syn::Error::new_spanned(
            arg,
            "an extern_fn has no self parameter",
        ));
    };
    let is_interner_path = |ty: &Type| match ty {
        Type::Path(p) => p
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "Interner"),
        _ => false,
    };
    let ok = match (pat_type.ty.as_ref(), is_async) {
        (Type::Reference(r), false) => r.mutability.is_none() && is_interner_path(&r.elem),
        (ty, true) => is_interner_path(ty),
        _ => false,
    };
    if ok {
        Ok(())
    } else if is_async {
        Err(syn::Error::new_spanned(
            &pat_type.ty,
            "an async extern_fn takes an owned `Interner` first",
        ))
    } else {
        Err(syn::Error::new_spanned(
            &pat_type.ty,
            "an extern_fn takes `&Interner` first",
        ))
    }
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

fn qref_expr(ns: Option<&str>, name: &str) -> proc_macro2::TokenStream {
    match ns {
        Some(ns) => quote! {
            ::acvus_extern::QualifiedRef::qualified(__i.intern(#ns), __i.intern(#name))
        },
        None => quote! { ::acvus_extern::QualifiedRef::root(__i.intern(#name)) },
    }
}

fn type_name_expr(ns: Option<&str>, name: &str) -> proc_macro2::TokenStream {
    let ns = match ns {
        Some(ns) => quote! { ::core::option::Option::Some(#ns) },
        None => quote! { ::core::option::Option::None },
    };
    quote! { ::acvus_extern::ExternTypeName { ns: #ns, name: #name } }
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
    let mut phantoms = 0usize;
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
        phantoms += 1;
    }
    let phantom_inits = (0..phantoms).map(|_| quote! { ::core::marker::PhantomData });

    if vars.mentions_var(payload_ty) {
        return Err(syn::Error::new_spanned(
            payload_ty,
            "the payload type names no type, effect, or length parameter; every instantiation shares one payload",
        ));
    }
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let arg_impl_generics = vars.arg_impl_generics();
    let runtime: Ident = match vars.runtime_param() {
        Some(rt) => rt.clone(),
        None => format_ident!("__R"),
    };
    let conv_impl_generics = match vars.runtime_param() {
        Some(_) => quote! { #impl_generics },
        None => {
            let params = input.generics.params.iter();
            quote! { <__R: ::acvus_extern::Runtime, #(#params),*> }
        }
    };
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
    let move_only = n_identities > 0;

    let qref = qref_expr(attr.ns.as_deref(), &name);
    let type_name = type_name_expr(attr.ns.as_deref(), &name);
    let take_payload = if move_only {
        quote! {
            match __o.into_owned::<#payload_ty>() {
                Ok(p) => p,
                Err(::acvus_extern::PayloadMismatch::Shared(_)) => {
                    return Err(::acvus_extern::ExternError::SharedMoveOnly { type_name: Self::TYPE_NAME }.into());
                }
                Err(::acvus_extern::PayloadMismatch::OtherType) => {
                    return Err(::acvus_extern::ExternError::internal(
                        ::std::format!("payload of {} is not {}", Self::TYPE_NAME, stringify!(#payload_ty)),
                    ).into());
                }
            }
        }
    } else {
        quote! {
            match __o.into_cloned::<#payload_ty>() {
                Ok(p) => p,
                Err(_) => {
                    return Err(::acvus_extern::ExternError::internal(
                        ::std::format!("payload of {} is not {}", Self::TYPE_NAME, stringify!(#payload_ty)),
                    ).into());
                }
            }
        }
    };

    Ok(quote! {
        impl #impl_generics #ident #ty_generics #where_clause {
            pub const TYPE_NAME: ::acvus_extern::ExternTypeName = #type_name;
        }

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

        impl #conv_impl_generics ::acvus_extern::FromValue<#runtime> for #ident #ty_generics #where_clause {
            fn from_value(
                __value: <#runtime as ::acvus_extern::Runtime>::Value,
                _: &::acvus_extern::Interner,
            ) -> ::core::result::Result<Self, <#runtime as ::acvus_extern::Runtime>::Error> {
                let __o = <#runtime as ::acvus_extern::Runtime>::into_extern(__value)?;
                if __o.type_name != Self::TYPE_NAME {
                    return Err(::acvus_extern::ExternError::UnexpectedExtern {
                        expected: Self::TYPE_NAME,
                        got: __o.type_name,
                    }
                    .into());
                }
                let __payload = #take_payload;
                Ok(Self(__payload #(, #phantom_inits)*))
            }
        }

        impl #conv_impl_generics ::acvus_extern::IntoValue<#runtime> for #ident #ty_generics #where_clause {
            fn into_value(self, _: &::acvus_extern::Interner) -> <#runtime as ::acvus_extern::Runtime>::Value {
                <#runtime as ::acvus_extern::Runtime>::extern_value(
                    ::acvus_extern::ExternValue::new(Self::TYPE_NAME, self.0),
                )
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
            "a structural object has no generic parameters",
        ));
    }
    let syn::Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            ident.span(),
            "TyArg is derived on a struct",
        ));
    };
    let syn::Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new(
            ident.span(),
            "TyArg is derived on a struct with named fields, one per object field",
        ));
    };
    let field_idents: Vec<&Ident> = fields
        .named
        .iter()
        .map(|f| f.ident.as_ref().expect("named"))
        .collect();
    let field_names: Vec<String> = field_idents.iter().map(|f| f.to_string()).collect();
    let field_tys: Vec<&Type> = fields.named.iter().map(|f| &f.ty).collect();

    Ok(quote! {
        impl ::acvus_extern::TyArg for #ident {
            fn poly_ty(
                __i: &::acvus_extern::Interner,
                __vars: &::acvus_extern::PolyVars,
            ) -> ::acvus_extern::PolyTy {
                ::acvus_extern::PolyTy::Object(
                    [#((
                        __i.intern(#field_names),
                        <#field_tys as ::acvus_extern::TyArg>::poly_ty(__i, __vars),
                    )),*]
                    .into_iter()
                    .collect(),
                )
            }
        }

        impl<__R: ::acvus_extern::Runtime> ::acvus_extern::FromValue<__R> for #ident {
            fn from_value(
                __value: <__R as ::acvus_extern::Runtime>::Value,
                __i: &::acvus_extern::Interner,
            ) -> ::core::result::Result<Self, <__R as ::acvus_extern::Runtime>::Error> {
                let mut __fields = <__R as ::acvus_extern::Runtime>::into_object(__value)?;
                Ok(Self {
                    #(#field_idents: match __fields.remove(&__i.intern(#field_names)) {
                        Some(__v) => <#field_tys as ::acvus_extern::FromValue<__R>>::from_value(__v, __i)?,
                        None => return Err(::acvus_extern::ExternError::MissingField {
                            field: #field_names.to_owned(),
                        }
                        .into()),
                    },)*
                })
            }
        }

        impl<__R: ::acvus_extern::Runtime> ::acvus_extern::IntoValue<__R> for #ident {
            fn into_value(self, __i: &::acvus_extern::Interner) -> <__R as ::acvus_extern::Runtime>::Value {
                <__R as ::acvus_extern::Runtime>::object(
                    [#((
                        __i.intern(#field_names),
                        <#field_tys as ::acvus_extern::IntoValue<__R>>::into_value(self.#field_idents, __i),
                    )),*]
                    .into_iter()
                    .collect(),
                )
            }
        }
    })
}

// -- extern_registry! ------------------------------------------------

/// `extern_registry! { types: [List<_>], fns: [len, reverse] }`.
struct RegistryInput {
    types: Vec<Type>,
    fns: Vec<Path>,
}

impl Parse for RegistryInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut types = Vec::new();
        let mut fns = Vec::new();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let content;
            syn::bracketed!(content in input);
            if key == "types" {
                let list: Punctuated<Type, Token![,]> =
                    content.parse_terminated(Type::parse, Token![,])?;
                types.extend(list);
            } else if key == "fns" {
                let list: Punctuated<Path, Token![,]> =
                    content.parse_terminated(Path::parse, Token![,])?;
                fns.extend(list);
            } else {
                return Err(syn::Error::new(key.span(), "expected `types` or `fns`"));
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(Self { types, fns })
    }
}

#[proc_macro]
pub fn extern_registry(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as RegistryInput);
    let types: Vec<Type> = input.types.iter().map(subst::infer_to_unit).collect();
    let fns: Vec<Path> = input
        .fns
        .into_iter()
        .map(|mut path| {
            let last = path.segments.last_mut().expect("path has a segment");
            last.ident = format_ident!("__extern_fn_{}", last.ident);
            path
        })
        .collect();
    quote! {
        ::acvus_extern::ExternRegistry::new(|__i: &::acvus_extern::Interner| {
            ::acvus_extern::ExternItems {
                types: vec![#(<#types as ::acvus_extern::ExternTypeDecl>::type_decl(__i)),*],
                fns: vec![#(#fns(__i)),*],
            }
        })
    }
    .into()
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
