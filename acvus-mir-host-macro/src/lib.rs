//! Proc macros for acvus-mir-host.
//!
//! - `#[extern_fn]`: ExternFn handler — struct + ExternFnDef + call wrapper + FnConstraint.
//! - `#[derive(ExternType)]`: UserDefined type — ITy impl generating Ty::UserDefined.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, DeriveInput};

mod parse;

// ── ExternType derive ──────────────────────────────────────────────

/// Derive `ITy` for user-defined types that map to `Ty::UserDefined`.
///
/// ```ignore
/// #[derive(ExternType)]
/// #[extern_type(name = "Iterator", effects(E))]
/// struct AcvusIter<T, E>(PhantomData<(T, E)>);
/// ```
///
/// - Generic params listed in `effects(...)` → `effect_args` (bound: `EffectParam`)
/// - All other generic params → `type_args` (bound: `ITy`)
#[proc_macro_derive(ExternType, attributes(extern_type))]
pub fn derive_extern_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match generate_extern_type(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate_extern_type(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;

    // Parse #[extern_type(name = "...", ns = "...", effects(E1, E2))] attribute.
    let attr = parse::parse_extern_type_attr(&input.attrs)?;
    let ty_name = &attr.name;
    let effect_names = &attr.effects;

    // Classify generic params: effect vs type.
    let mut type_params = Vec::new();
    let mut effect_params = Vec::new();

    for param in &input.generics.params {
        if let syn::GenericParam::Type(tp) = param {
            if effect_names.contains(&tp.ident) {
                effect_params.push(tp.ident.clone());
            } else {
                type_params.push(tp.ident.clone());
            }
        }
    }

    // Build generic params for the impl.
    let all_params: Vec<_> = input.generics.params.iter().collect();
    let type_bounds: Vec<_> = type_params.iter().map(|name| {
        quote! { #name: ::acvus_mir_host::ITy }
    }).collect();
    let effect_bounds: Vec<_> = effect_params.iter().map(|name| {
        quote! { #name: ::acvus_mir_host::EffectParam }
    }).collect();
    let all_bounds: Vec<_> = type_bounds.iter().chain(effect_bounds.iter()).collect();

    // Build type_args and effect_args expressions.
    let type_arg_exprs: Vec<_> = type_params.iter().map(|name| {
        quote! { <#name as ::acvus_mir_host::ITy>::ty(__i, __tv, __ev) }
    }).collect();
    let effect_arg_exprs: Vec<_> = effect_params.iter().map(|name| {
        quote! { <#name as ::acvus_mir_host::EffectParam>::effect(__ev) }
    }).collect();

    // Generic param names for the impl header.
    let param_names: Vec<_> = input.generics.params.iter().map(|p| {
        match p {
            syn::GenericParam::Type(tp) => {
                let name = &tp.ident;
                quote! { #name }
            }
            _ => quote! {},
        }
    }).collect();

    let n_type_params = type_params.len();
    let n_effect_params = effect_params.len();

    // Build QualifiedRef expression with optional namespace.
    let qref_expr = make_qref_expr(attr.ns.as_deref(), ty_name);

    Ok(quote! {
        impl<#(#all_params),*> ::acvus_mir_host::ITy for #struct_name<#(#param_names),*>
        where #(#all_bounds),*
        {
            fn ty(
                __i: &::acvus_mir_host::Interner,
                __tv: &[::acvus_mir_host::Ty],
                __ev: &[::acvus_mir_host::Effect],
            ) -> ::acvus_mir_host::Ty {
                ::acvus_mir_host::Ty::UserDefined {
                    id: #qref_expr,
                    type_args: vec![#(#type_arg_exprs),*],
                    effect_args: vec![#(#effect_arg_exprs),*],
                }
            }
        }

        impl<#(#all_params),*> #struct_name<#(#param_names),*>
        where #(#all_bounds),*
        {
            /// Build the UserDefinedDecl for registering in TypeRegistry.
            pub fn type_decl(__i: &::acvus_mir_host::Interner) -> ::acvus_mir_host::UserDefinedDecl {
                ::acvus_mir_host::UserDefinedDecl {
                    qref: #qref_expr,
                    type_params: vec![None; #n_type_params],
                    effect_params: vec![None; #n_effect_params],
                }
            }
        }
    })
}

/// `#[extern_fn(name = "len_str", LenStrFn)]`
#[proc_macro_attribute]
pub fn extern_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(attr as parse::ExternFnAttr);
    let input_fn = parse_macro_input!(item as ItemFn);

    match generate(attr_args, input_fn) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn generate(attr: parse::ExternFnAttr, input_fn: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &attr.struct_name;
    let fn_name = &input_fn.sig.ident;
    let vis = &input_fn.vis;

    let gen_result = parse::parse_generics(&input_fn.sig)?;
    let generics = &gen_result.generics;
    let params = parse::parse_params(&input_fn.sig, gen_result.scope_param.as_ref())?;
    let ret = parse::parse_return(&input_fn.sig)?;
    let original_fn = &input_fn;

    // Constrained monomorphization: detect Monomorphize bounds in generics.
    let mono_params: Vec<_> = generics.iter().filter(|g| g.monomorphize.is_some()).collect();
    if mono_params.len() > 1 {
        return Err(syn::Error::new(
            mono_params[1].name.span(),
            "multiple Monomorphize constraints not yet supported (requires N-dim cartesian product)",
        ));
    }
    if mono_params.len() == 1 {
        return generate_monomorphized(&attr, original_fn, fn_name, vis, &generics, &params, &ret);
    }

    // Infer handler: special path. Concrete only — no generics allowed.
    if attr.infer {
        if !generics.is_empty() {
            return Err(syn::Error::new(
                fn_name.span(),
                "infer handlers must be concrete — no generic parameters allowed",
            ));
        }
        return generate_infer(original_fn, fn_name, vis, &attr.struct_name, &params);
    }

    // Normal path: single struct.
    let reg_calls = params.registration_tokens(&generics);
    let ret_reg_calls = ret.registration_tokens(&generics);
    let is_async = input_fn.sig.asyncness.is_some();
    let call_wrapper = generate_call_wrapper(fn_name, &generics, &params, &ret, is_async)?;
    let constraint_fn = generate_constraint(&generics, &params, &ret)?;

    let extern_name = &attr.name;
    let fn_qref = make_qref_expr(attr.ns.as_ref().map(|s| s.value()).as_deref(), &extern_name.value());

    Ok(quote! {
        #original_fn

        #vis struct #struct_name;

        impl ::acvus_mir_host::ExternFnDef for #struct_name {
            fn register(reg: &mut impl ::acvus_mir_host::Registrar) {
                #reg_calls
                #ret_reg_calls
            }
        }

        impl #struct_name {
            #call_wrapper
            #constraint_fn

            /// Build the Function for CompilationGraph registration.
            pub fn function(__i: &::acvus_mir_host::Interner) -> ::acvus_mir_host::Function {
                ::acvus_mir_host::Function {
                    qref: #fn_qref,
                    kind: ::acvus_mir_host::FnKind::Extern,
                    constraint: Self::constraint(__i),
                }
            }
        }
    })
}

/// Generate N concrete structs for constrained monomorphization.
///
/// A generic param with `Monomorphize<(i64, f64)>` bound generates
/// AddFn0 (for i64) and AddFn1 (for f64), plus a type alias
/// `type AddFn = (AddFn0, AddFn1);`
fn generate_monomorphized(
    attr: &parse::ExternFnAttr,
    original_fn: &ItemFn,
    fn_name: &syn::Ident,
    vis: &syn::Visibility,
    generics: &[parse::GenericInfo],
    params: &parse::ParsedParams,
    ret: &parse::ReturnInfo,
) -> syn::Result<proc_macro2::TokenStream> {
    // Find the single Monomorphize-constrained param.
    let constrained_generic = generics.iter()
        .find(|g| g.monomorphize.is_some())
        .unwrap(); // guaranteed by caller check

    let mono_types = constrained_generic.monomorphize.as_ref().unwrap();
    let struct_name = &attr.struct_name;
    let mut structs = proc_macro2::TokenStream::new();
    let mut struct_names = Vec::new();

    for (i, concrete_ty) in mono_types.iter().enumerate() {
        let variant_name = syn::Ident::new(
            &format!("{struct_name}{i}"),
            struct_name.span(),
        );
        struct_names.push(variant_name.clone());

        // Build a "fake" generics where the constrained param is removed
        // (it's now concrete). For constraint + call wrapper generation,
        // substitute the constrained param name with the concrete type.
        let mono_generics: Vec<_> = generics.iter()
            .filter(|g| g.name != constrained_generic.name)
            .cloned()
            .collect();

        // For type substitution: replace the constrained param with concrete type.
        let substitute_constrained = |ty: &syn::Type| -> syn::Type {
            substitute_name(ty, &constrained_generic.name, concrete_ty)
        };

        // Build concrete params.
        let mono_params = parse::ParsedParams {
            params: params.params.iter().map(|p| {
                parse::ParsedParam {
                    name: p.name.clone(),
                    ty: substitute_constrained(&p.ty),
                }
            }).collect(),
            has_scope: params.has_scope,
        };

        let mono_ret = parse::ReturnInfo {
            types: ret.types.iter().map(|ty| substitute_constrained(ty)).collect(),
        };

        let reg_calls = mono_params.registration_tokens(&mono_generics);
        let ret_reg_calls = mono_ret.registration_tokens(&mono_generics);
        let is_async = original_fn.sig.asyncness.is_some();
        let call_wrapper = generate_call_wrapper(fn_name, &mono_generics, &mono_params, &mono_ret, is_async)?;
        let constraint_fn = generate_constraint(&mono_generics, &mono_params, &mono_ret)?;

        structs.extend(quote! {
            #vis struct #variant_name;

            impl ::acvus_mir_host::ExternFnDef for #variant_name {
                fn register(reg: &mut impl ::acvus_mir_host::Registrar) {
                    #reg_calls
                    #ret_reg_calls
                }
            }

            impl #variant_name {
                #call_wrapper
                #constraint_fn
            }
        });
    }

    // Type alias: AddFn = (AddFn0, AddFn1)
    Ok(quote! {
        #original_fn

        #structs

        #vis type #struct_name = (#(#struct_names,)*);
    })
}

/// Replace a specific identifier with a concrete type throughout a type AST.
fn substitute_name(ty: &syn::Type, name: &syn::Ident, replacement: &syn::Type) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
                let seg = &type_path.path.segments[0];
                if matches!(seg.arguments, syn::PathArguments::None) && seg.ident == *name {
                    return replacement.clone();
                }
            }
            let mut new_path = type_path.clone();
            for seg in &mut new_path.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments {
                    for arg in &mut args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            *inner = substitute_name(inner, name, replacement);
                        }
                    }
                }
            }
            syn::Type::Path(new_path)
        }
        syn::Type::Tuple(tuple) => {
            let elems: Vec<_> = tuple.elems.iter()
                .map(|e| substitute_name(e, name, replacement))
                .collect();
            syn::parse_quote! { (#(#elems),*) }
        }
        other => other.clone(),
    }
}

/// Generate call wrapper: take args from uses → call handler → store results to defs.
///
/// For generic handlers, type params are substituted with `__S::Owned`.
/// Sync handlers → ExternFnFuture::Ready. Async handlers → ExternFnFuture::Async.
fn generate_call_wrapper(
    fn_name: &syn::Ident,
    generics: &[parse::GenericInfo],
    params: &parse::ParsedParams,
    ret: &parse::ReturnInfo,
    is_async: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let has_generics = !generics.is_empty();

    // Substitute generic type names → __S::Owned for runtime.
    let runtime_type = |ty: &syn::Type| -> syn::Type {
        if has_generics { substitute_owned(ty, generics) } else { ty.clone() }
    };

    // Extract args from uses.
    let mut extractions = Vec::new();
    let mut arg_names = Vec::new();

    for (i, p) in params.params.iter().enumerate() {
        let name = &p.name;
        let rt_ty = runtime_type(&p.ty);
        let idx = syn::Index::from(i);
        extractions.push(quote! {
            let #name: #rt_ty = ::acvus_mir_host::Scope::take(__scope, __uses[#idx]);
        });
        arg_names.push(name.clone());
    }

    // Build turbofish for generic handler call.
    // S: Scope → __S, Type params → __S::Owned, Effect params → Eff<N>.
    let turbofish = if !generics.is_empty() || params.has_scope {
        let mut turbofish_params: Vec<proc_macro2::TokenStream> = Vec::new();
        // If scoped, S is first generic param → __S.
        if params.has_scope {
            turbofish_params.push(quote! { __S });
        }
        // Remaining generic params.
        for g in generics.iter() {
            if g.is_type {
                turbofish_params.push(quote! { __S::Owned });
            } else {
                let idx = g.index;
                turbofish_params.push(quote! { ::acvus_mir_host::Eff<#idx> });
            }
        }
        quote! { ::<#(#turbofish_params),*> }
    } else {
        quote! {}
    };

    // Build handler arguments. If scoped, prepend __scope.
    let handler_args: Vec<proc_macro2::TokenStream> = if params.has_scope {
        std::iter::once(quote! { __scope })
            .chain(arg_names.iter().map(|n| quote! { #n }))
            .collect()
    } else {
        arg_names.iter().map(|n| quote! { #n }).collect()
    };

    // Call handler + destructure.
    let n_ret = ret.types.len();
    let call_expr = if n_ret == 0 {
        quote! { #fn_name #turbofish (#(#handler_args),*); }
    } else if n_ret == 1 {
        quote! { let (__r0,) = #fn_name #turbofish (#(#handler_args),*); }
    } else {
        let bindings: Vec<syn::Ident> = (0..n_ret)
            .map(|i| syn::Ident::new(&format!("__r{i}"), proc_macro2::Span::call_site()))
            .collect();
        quote! { let (#(#bindings),*) = #fn_name #turbofish (#(#handler_args),*); }
    };

    // Store results to defs.
    let store_stmts = if n_ret == 0 {
        quote! { ::acvus_mir_host::Scope::store(__scope, __defs[0], ()); }
    } else {
        let mut stmts = proc_macro2::TokenStream::new();
        for i in 0..n_ret {
            let idx = syn::Index::from(i);
            let binding = syn::Ident::new(&format!("__r{i}"), proc_macro2::Span::call_site());
            stmts.extend(quote! {
                ::acvus_mir_host::Scope::store(__scope, __defs[#idx], #binding);
            });
        }
        stmts
    };

    if is_async {
        // Async handler: extract args (sync), then wrap handler + store in BoxFuture.
        let async_call_expr = if n_ret == 0 {
            quote! { #fn_name #turbofish (#(#handler_args),*).await; }
        } else if n_ret == 1 {
            quote! { let (__r0,) = #fn_name #turbofish (#(#handler_args),*).await; }
        } else {
            let bindings: Vec<syn::Ident> = (0..n_ret)
                .map(|i| syn::Ident::new(&format!("__r{i}"), proc_macro2::Span::call_site()))
                .collect();
            quote! { let (#(#bindings),*) = #fn_name #turbofish (#(#handler_args),*).await; }
        };

        Ok(quote! {
            /// Call this ExternFn through a Scope. Returns ExternFnFuture::Async.
            pub fn call<'__scope, __S: ::acvus_mir_host::Scope>(
                __scope: &'__scope mut __S,
                __uses: &[__S::Repr],
                __defs: &[__S::Repr],
            ) -> ::acvus_mir_host::ExternFnFuture<'__scope> {
                #(#extractions)*
                let __defs: Vec<__S::Repr> = __defs.to_vec();
                ::acvus_mir_host::ExternFnFuture::Async(::std::boxed::Box::pin(async move {
                    #async_call_expr
                    #store_stmts
                    Ok(())
                }))
            }
        })
    } else {
        Ok(quote! {
            /// Call this ExternFn through a Scope. Returns ExternFnFuture::Ready.
            pub fn call<'__scope, __S: ::acvus_mir_host::Scope>(
                __scope: &'__scope mut __S,
                __uses: &[__S::Repr],
                __defs: &[__S::Repr],
            ) -> ::acvus_mir_host::ExternFnFuture<'__scope> {
                #(#extractions)*
                #call_expr
                #store_stmts
                ::acvus_mir_host::ExternFnFuture::Ready(Some(Ok(())))
            }
        })
    }
}

/// Build the Ty expression for a parameter type.
///
/// If the type is a bare generic name with a Callable bound, generates Ty::Fn directly.
/// Otherwise, substitutes generic names with Typeck<N> and calls ITy::ty().
fn build_param_ty(ty: &syn::Type, generics: &[parse::GenericInfo]) -> proc_macro2::TokenStream {
    // Check if this is a bare generic name that has a Callable bound.
    if let syn::Type::Path(type_path) = ty {
        if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
            let seg = &type_path.path.segments[0];
            if matches!(seg.arguments, syn::PathArguments::None) {
                for g in generics {
                    if g.is_type && seg.ident == g.name {
                        if let Some(callable) = &g.callable {
                            return build_callable_ty(callable, generics);
                        }
                    }
                }
            }
        }
    }
    // Default: substitute generics → Typeck<N>, use ITy::ty().
    let sub_ty = substitute_generics(ty, generics);
    quote! { <#sub_ty as ::acvus_mir_host::ITy>::ty(__interner, &__type_vars, &__effect_vars) }
}

/// Build Ty::Fn from a Callable<Args, Ret, E> bound.
///
/// Args tuple is decomposed into individual params.
/// E is substituted with Eff<N> and resolved via EffectParam::effect().
fn build_callable_ty(
    callable: &parse::CallableInfo,
    generics: &[parse::GenericInfo],
) -> proc_macro2::TokenStream {
    // Decompose Args tuple into individual types.
    let arg_types: Vec<syn::Type> = match &callable.args_ty {
        syn::Type::Tuple(tuple) => tuple.elems.iter().cloned().collect(),
        single => vec![single.clone()],
    };

    // Substitute generic names and build Param entries.
    let param_exprs: Vec<_> = arg_types.iter().enumerate().map(|(i, ty)| {
        let sub_ty = substitute_generics(ty, generics);
        let name_str = format!("_{i}");
        quote! {
            ::acvus_mir_host::Param::new(
                __interner.intern(#name_str),
                <#sub_ty as ::acvus_mir_host::ITy>::ty(__interner, &__type_vars, &__effect_vars),
            )
        }
    }).collect();

    // Return type.
    let sub_ret = substitute_generics(&callable.ret_ty, generics);
    let ret_expr = quote! { <#sub_ret as ::acvus_mir_host::ITy>::ty(__interner, &__type_vars, &__effect_vars) };

    // Effect: substitute generic names with Eff<N>, resolve via EffectParam::effect().
    let sub_effect = substitute_effect(&callable.effect_ty, generics);

    quote! {
        {
            ::acvus_mir_host::Ty::Fn {
                params: vec![#(#param_exprs),*],
                ret: Box::new(#ret_expr),
                captures: vec![],
                effect: #sub_effect,
            }
        }
    }
}

/// Resolve an effect type to an Effect expression.
///
/// If the type is a bare generic name with EffectParam bound → Eff<N>::effect(&effect_vars).
/// Otherwise it's concrete → Effect::pure() (shouldn't normally happen).
fn substitute_effect(
    ty: &syn::Type,
    generics: &[parse::GenericInfo],
) -> proc_macro2::TokenStream {
    if let syn::Type::Path(type_path) = ty {
        if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
            let seg = &type_path.path.segments[0];
            if matches!(seg.arguments, syn::PathArguments::None) {
                for g in generics {
                    if !g.is_type && seg.ident == g.name {
                        let idx = g.index;
                        return quote! {
                            <::acvus_mir_host::Eff<#idx> as ::acvus_mir_host::EffectParam>::effect(&__effect_vars)
                        };
                    }
                }
            }
        }
    }
    // Fallback: pure effect.
    quote! { ::acvus_mir_host::Effect::pure() }
}

/// Replace generic param names for runtime call wrapper.
/// Type params → `__S::Owned`. Effect params → `Eff<N>`.
fn substitute_owned(ty: &syn::Type, generics: &[parse::GenericInfo]) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
                let seg = &type_path.path.segments[0];
                if matches!(seg.arguments, syn::PathArguments::None) {
                    for g in generics {
                        if seg.ident == g.name {
                            if g.is_type {
                                return syn::parse_quote! { __S::Owned };
                            } else {
                                let idx = g.index;
                                return syn::parse_quote! { ::acvus_mir_host::Eff<#idx> };
                            }
                        }
                    }
                }
            }
            let mut new_path = type_path.clone();
            for seg in &mut new_path.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments {
                    for arg in &mut args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            *inner = substitute_owned(inner, generics);
                        }
                    }
                }
            }
            syn::Type::Path(new_path)
        }
        syn::Type::Tuple(tuple) => {
            let elems: Vec<_> = tuple.elems.iter()
                .map(|e| substitute_owned(e, generics))
                .collect();
            syn::parse_quote! { (#(#elems),*) }
        }
        other => other.clone(),
    }
}

/// Generate FnConstraint from the signature.
///
/// For concrete handlers: type_vars and effect_vars are empty.
/// For generic handlers: TySubst allocates Ty::Param / Effect::Var,
/// and generic type names in the signature are replaced with Typeck<N>.
fn generate_constraint(
    generics: &[parse::GenericInfo],
    params: &parse::ParsedParams,
    ret: &parse::ReturnInfo,
) -> syn::Result<proc_macro2::TokenStream> {
    let n_type_vars = generics.iter().filter(|g| g.is_type).count();
    let n_effect_vars = generics.iter().filter(|g| !g.is_type).count();

    // Generate TySubst allocation for type vars and effect vars.
    let type_var_allocs: Vec<_> = (0..n_type_vars).map(|_| {
        quote! { __subst.fresh_param() }
    }).collect();
    let effect_var_allocs: Vec<_> = (0..n_effect_vars).map(|_| {
        quote! { __subst.fresh_effect_var() }
    }).collect();

    // Build param types.
    // For each function param:
    //   - If its type is a Callable generic → generate Ty::Fn directly
    //   - Otherwise → substitute generics with Typeck<N> and use ITy::ty()
    let param_constructs: Vec<_> = params.params.iter().map(|p| {
        let name_str = p.name.to_string();
        let ty_expr = build_param_ty(&p.ty, generics);
        quote! {
            ::acvus_mir_host::Param::new(
                __interner.intern(#name_str),
                #ty_expr,
            )
        }
    }).collect();

    // Build return type.
    let ret_ty = if ret.types.is_empty() {
        quote! { ::acvus_mir_host::Ty::Unit }
    } else if ret.types.len() == 1 {
        build_param_ty(&ret.types[0], generics)
    } else {
        let tys: Vec<_> = ret.types.iter().map(|ty| build_param_ty(ty, generics)).collect();
        quote! { ::acvus_mir_host::Ty::Tuple(vec![#(#tys),*]) }
    };

    Ok(quote! {
        /// Build the FnConstraint for this ExternFn.
        pub fn constraint(__interner: &::acvus_mir_host::Interner) -> ::acvus_mir_host::FnConstraint {
            let mut __subst = ::acvus_mir_host::TySubst::new();
            let __type_vars: Vec<::acvus_mir_host::Ty> = vec![#(#type_var_allocs),*];
            let __effect_vars: Vec<::acvus_mir_host::Effect> = vec![#(#effect_var_allocs),*];
            let __params = vec![#(#param_constructs),*];
            ::acvus_mir_host::FnConstraint {
                signature: Some(::acvus_mir_host::Signature {
                    params: __params.clone(),
                }),
                output: ::acvus_mir_host::Constraint::Exact(::acvus_mir_host::Ty::Fn {
                    params: __params,
                    ret: Box::new(#ret_ty),
                    captures: vec![],
                    effect: ::acvus_mir_host::Effect::pure(),
                }),
                effect: None,
                hint: None,
            }
        }
    })
}

/// Generate an infer-tagged ExternFn. Concrete only — no generics, no scope.
///
/// Handler: `fn name(args..., inferred_ret_ty: Ty) -> Inferrable`
/// Call wrapper: extra `&Ty` parameter. FnConstraint output = Inferred.
fn generate_infer(
    original_fn: &ItemFn,
    fn_name: &syn::Ident,
    vis: &syn::Visibility,
    struct_name: &syn::Ident,
    params: &parse::ParsedParams,
) -> syn::Result<proc_macro2::TokenStream> {
    // Exclude last param (inferred_ret_ty: Ty) — system-provided.
    let user_params: Vec<_> = if !params.params.is_empty() {
        params.params[..params.params.len() - 1].to_vec()
    } else {
        vec![]
    };

    let user_parsed = parse::ParsedParams { params: user_params.clone(), has_scope: false };
    let reg_calls = user_parsed.registration_tokens(&[]);

    // Param types for FnConstraint.
    let param_constructs: Vec<_> = user_params.iter().map(|p| {
        let name_str = p.name.to_string();
        let ty = &p.ty;
        quote! {
            ::acvus_mir_host::Param::new(
                __interner.intern(#name_str),
                <#ty as ::acvus_mir_host::ITy>::ty(__interner, &[], &[]),
            )
        }
    }).collect();

    // Extractions: take from uses.
    let mut extractions = Vec::new();
    let mut handler_args = Vec::new();

    for (i, p) in user_params.iter().enumerate() {
        let name = &p.name;
        let ty = &p.ty;
        let idx = syn::Index::from(i);
        extractions.push(quote! {
            let #name: #ty = ::acvus_mir_host::Scope::take(__scope, __uses[#idx]);
        });
        handler_args.push(quote! { #name });
    }
    handler_args.push(quote! { __inferred_ret_ty.clone() });

    Ok(quote! {
        #original_fn

        #vis struct #struct_name;

        impl ::acvus_mir_host::ExternFnDef for #struct_name {
            fn register(reg: &mut impl ::acvus_mir_host::Registrar) {
                #reg_calls
            }
        }

        impl #struct_name {
            /// Call this infer-tagged ExternFn.
            pub fn call<'__scope, __S: ::acvus_mir_host::Scope>(
                __scope: &'__scope mut __S,
                __uses: &[__S::Repr],
                __defs: &[__S::Repr],
                __inferred_ret_ty: &::acvus_mir_host::Ty,
            ) -> ::acvus_mir_host::ExternFnFuture<'__scope> {
                #(#extractions)*
                let __inferrable = #fn_name(#(#handler_args),*);
                ::acvus_mir_host::Scope::store(__scope, __defs[0], __inferrable);
                ::acvus_mir_host::ExternFnFuture::Ready(Some(Ok(())))
            }

            /// Build the FnConstraint. Output = Inferred.
            pub fn constraint(__interner: &::acvus_mir_host::Interner) -> ::acvus_mir_host::FnConstraint {
                let __params = vec![#(#param_constructs),*];
                ::acvus_mir_host::FnConstraint {
                    signature: Some(::acvus_mir_host::Signature {
                        params: __params,
                    }),
                    output: ::acvus_mir_host::Constraint::Inferred,
                    effect: None,
                    hint: None,
                }
            }
        }
    })
}

/// Build a QualifiedRef expression with optional namespace.
fn make_qref_expr(ns: Option<&str>, name: &str) -> proc_macro2::TokenStream {
    match ns {
        Some(ns) => quote! {
            ::acvus_mir_host::QualifiedRef::qualified(__i.intern(#ns), __i.intern(#name))
        },
        None => quote! {
            ::acvus_mir_host::QualifiedRef::root(__i.intern(#name))
        },
    }
}

/// Replace generic parameter names with Typeck<N> / Eff<N> in a type AST.
///
/// Type params → Typeck<N>, Effect params → Eff<N>.
fn substitute_generics(ty: &syn::Type, generics: &[parse::GenericInfo]) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
                let seg = &type_path.path.segments[0];
                if matches!(seg.arguments, syn::PathArguments::None) {
                    for g in generics {
                        if seg.ident == g.name {
                            let idx = g.index;
                            if g.is_type {
                                return syn::parse_quote! { ::acvus_mir_host::Typeck<#idx> };
                            } else {
                                return syn::parse_quote! { ::acvus_mir_host::Eff<#idx> };
                            }
                        }
                    }
                }
            }
            // Recurse into generic arguments.
            let mut new_path = type_path.clone();
            for seg in &mut new_path.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments {
                    for arg in &mut args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            *inner = substitute_generics(inner, generics);
                        }
                    }
                }
            }
            syn::Type::Path(new_path)
        }
        syn::Type::Tuple(tuple) => {
            let elems: Vec<_> = tuple.elems.iter()
                .map(|e| substitute_generics(e, generics))
                .collect();
            syn::parse_quote! { (#(#elems),*) }
        }
        syn::Type::Reference(ref_ty) => {
            let inner = substitute_generics(&ref_ty.elem, generics);
            let mut new_ref = ref_ty.clone();
            new_ref.elem = Box::new(inner);
            syn::Type::Reference(new_ref)
        }
        other => other.clone(),
    }
}
