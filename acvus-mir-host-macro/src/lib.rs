//! Proc macros for acvus-mir-extern.
//!
//! - `#[extern_fn]`: ExternFn → struct + `constraint()` (PolyTy::Fn) + `function()`.
//! - `#[derive(ExternType)]`: UserDefined type → `ITy` impl (Ty::UserDefined).

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, DeriveInput};

mod parse;

// ── ExternType derive ──────────────────────────────────────────────

/// Derive `ITy` for user-defined types that map to `Ty::UserDefined`.
///
/// ```ignore
/// #[derive(ExternType)]
/// #[extern_type(name = "Iterator")]
/// struct AcvusIter<T>(PhantomData<T>);
/// ```
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

    let attr = parse::parse_extern_type_attr(&input.attrs)?;
    let ty_name = &attr.name;

    // All generic params → type params.
    let type_params: Vec<_> = input.generics.params.iter().filter_map(|p| {
        if let syn::GenericParam::Type(tp) = p { Some(tp.ident.clone()) } else { None }
    }).collect();

    let all_params: Vec<_> = input.generics.params.iter().collect();
    let type_bounds: Vec<_> = type_params.iter().map(|name| {
        quote! { #name: ::acvus_mir_host::ITy }
    }).collect();
    let param_names: Vec<_> = type_params.iter().map(|name| quote! { #name }).collect();

    let ty_arg_exprs: Vec<_> = type_params.iter().map(|name| {
        quote! { <#name as ::acvus_mir_host::ITy>::ty(__i, __tv) }
    }).collect();
    let poly_arg_exprs: Vec<_> = type_params.iter().map(|name| {
        quote! { <#name as ::acvus_mir_host::ITy>::poly_ty(__i, __tv) }
    }).collect();

    let n_type_params = type_params.len();
    let qref_expr = make_qref_expr(attr.ns.as_deref(), ty_name);

    Ok(quote! {
        impl<#(#all_params),*> ::acvus_mir_host::ITy for #struct_name<#(#param_names),*>
        where #(#type_bounds),*
        {
            fn ty(
                __i: &::acvus_mir_host::Interner,
                __tv: &[::acvus_mir_host::Ty],
            ) -> ::acvus_mir_host::Ty {
                ::acvus_mir_host::Ty::UserDefined {
                    id: #qref_expr,
                    type_args: vec![#(#ty_arg_exprs),*],
                }
            }

            fn poly_ty(
                __i: &::acvus_mir_host::Interner,
                __tv: &[::acvus_mir_host::PolyTy],
            ) -> ::acvus_mir_host::PolyTy {
                ::acvus_mir_host::PolyTy::UserDefined {
                    id: #qref_expr,
                    type_args: vec![#(#poly_arg_exprs),*],
                }
            }
        }

        impl<#(#all_params),*> #struct_name<#(#param_names),*>
        where #(#type_bounds),*
        {
            /// Build the UserDefinedDecl for registering in TypeRegistry.
            pub fn type_decl(__i: &::acvus_mir_host::Interner) -> ::acvus_mir_host::UserDefinedDecl {
                ::acvus_mir_host::UserDefinedDecl {
                    qref: #qref_expr,
                    type_params: vec![None; #n_type_params],
                }
            }
        }
    })
}

// ── extern_fn ──────────────────────────────────────────────────────

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
        return generate_monomorphized(&attr, original_fn, fn_name, vis, generics, &params, &ret);
    }

    // Infer handler: concrete only — no generics allowed.
    if attr.infer {
        if !generics.is_empty() {
            return Err(syn::Error::new(
                fn_name.span(),
                "infer handlers must be concrete — no generic parameters allowed",
            ));
        }
        return generate_infer(original_fn, fn_name, vis, &attr.struct_name, &params);
    }

    // Normal path: single struct. Solver-facing constraint only.
    let constraint_fn = generate_constraint(generics, &params, &ret)?;
    let fn_qref = make_qref_expr(attr.ns.as_ref().map(|s| s.value()).as_deref(), &attr.name.value());

    Ok(quote! {
        #original_fn

        #vis struct #struct_name;

        impl #struct_name {
            #constraint_fn

            /// Build the Function for CompilationGraph registration.
            pub fn function(__i: &::acvus_mir_host::Interner) -> ::acvus_mir_host::Function {
                ::acvus_mir_host::Function {
                    qref: #fn_qref,
                    kind: ::acvus_mir_host::FnKind::Extern,
                    ty: Self::constraint(__i),
                }
            }
        }
    })
}

/// Generate N concrete structs for constrained monomorphization.
///
/// `Monomorphize<(i64, f64)>` → AddFn0 (i64) + AddFn1 (f64) + `type AddFn = (AddFn0, AddFn1);`
fn generate_monomorphized(
    attr: &parse::ExternFnAttr,
    original_fn: &ItemFn,
    _fn_name: &syn::Ident,
    vis: &syn::Visibility,
    generics: &[parse::GenericInfo],
    params: &parse::ParsedParams,
    ret: &parse::ReturnInfo,
) -> syn::Result<proc_macro2::TokenStream> {
    let constrained_generic = generics.iter()
        .find(|g| g.monomorphize.is_some())
        .unwrap(); // guaranteed by caller check

    let mono_types = constrained_generic.monomorphize.as_ref().unwrap();
    let struct_name = &attr.struct_name;
    let mut structs = proc_macro2::TokenStream::new();
    let mut struct_names = Vec::new();

    for (i, concrete_ty) in mono_types.iter().enumerate() {
        let variant_name = syn::Ident::new(&format!("{struct_name}{i}"), struct_name.span());
        struct_names.push(variant_name.clone());

        // Drop the constrained param (now concrete) and substitute it in types.
        let mono_generics: Vec<_> = generics.iter()
            .filter(|g| g.name != constrained_generic.name)
            .cloned()
            .collect();

        let substitute_constrained = |ty: &syn::Type| -> syn::Type {
            substitute_name(ty, &constrained_generic.name, concrete_ty)
        };

        let mono_params = parse::ParsedParams {
            params: params.params.iter().map(|p| parse::ParsedParam {
                name: p.name.clone(),
                ty: substitute_constrained(&p.ty),
            }).collect(),
            has_scope: params.has_scope,
        };

        let mono_ret = parse::ReturnInfo {
            types: ret.types.iter().map(|ty| substitute_constrained(ty)).collect(),
        };

        let constraint_fn = generate_constraint(&mono_generics, &mono_params, &mono_ret)?;

        structs.extend(quote! {
            #vis struct #variant_name;

            impl #variant_name {
                #constraint_fn
            }
        });
    }

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

// ── Constraint (PolyTy::Fn) generation ─────────────────────────────

/// Build the PolyTy expression for a parameter type (poly phase).
///
/// Bare generic with a Callable bound → PolyTy::Fn directly.
/// Otherwise → substitute generics with Typeck<N> and call ITy::poly_ty().
fn build_poly_param_ty(ty: &syn::Type, generics: &[parse::GenericInfo]) -> proc_macro2::TokenStream {
    if let syn::Type::Path(type_path) = ty {
        if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
            let seg = &type_path.path.segments[0];
            if matches!(seg.arguments, syn::PathArguments::None) {
                for g in generics {
                    if seg.ident == g.name {
                        if let Some(callable) = &g.callable {
                            return build_poly_callable_ty(callable, generics);
                        }
                    }
                }
            }
        }
    }
    let sub_ty = substitute_generics(ty, generics);
    quote! { <#sub_ty as ::acvus_mir_host::ITy>::poly_ty(__interner, &__type_vars) }
}

/// Build PolyTy::Fn from a Callable<Args, Ret> bound (poly phase).
fn build_poly_callable_ty(
    callable: &parse::CallableInfo,
    generics: &[parse::GenericInfo],
) -> proc_macro2::TokenStream {
    let arg_types: Vec<syn::Type> = match &callable.args_ty {
        syn::Type::Tuple(tuple) => tuple.elems.iter().cloned().collect(),
        single => vec![single.clone()],
    };

    let param_exprs: Vec<_> = arg_types.iter().enumerate().map(|(i, ty)| {
        let sub_ty = substitute_generics(ty, generics);
        let name_str = format!("_{i}");
        quote! {
            ::acvus_mir_host::ParamTerm::<::acvus_mir_host::Poly>::new(
                __interner.intern(#name_str),
                <#sub_ty as ::acvus_mir_host::ITy>::poly_ty(__interner, &__type_vars),
            )
        }
    }).collect();

    let sub_ret = substitute_generics(&callable.ret_ty, generics);
    let ret_expr = quote! { <#sub_ret as ::acvus_mir_host::ITy>::poly_ty(__interner, &__type_vars) };

    quote! {
        {
            ::acvus_mir_host::PolyTy::Fn {
                params: vec![#(#param_exprs),*],
                ret: Box::new(#ret_expr),
                captures: vec![],
                hint: None,
            }
        }
    }
}

/// Generate `constraint()` → PolyTy::Fn from the signature.
///
/// Concrete handlers: type_vars empty. Generic handlers: PolyBuilder allocates
/// PolyTy::Var, and generic names in the signature become Typeck<N>.
fn generate_constraint(
    generics: &[parse::GenericInfo],
    params: &parse::ParsedParams,
    ret: &parse::ReturnInfo,
) -> syn::Result<proc_macro2::TokenStream> {
    let n_type_vars = generics.len();

    let type_var_allocs: Vec<_> = (0..n_type_vars).map(|_| {
        quote! { __builder.fresh_ty_var() }
    }).collect();

    let param_constructs: Vec<_> = params.params.iter().map(|p| {
        let name_str = p.name.to_string();
        let ty_expr = build_poly_param_ty(&p.ty, generics);
        quote! {
            ::acvus_mir_host::ParamTerm::<::acvus_mir_host::Poly>::new(
                __interner.intern(#name_str),
                #ty_expr,
            )
        }
    }).collect();

    let ret_ty = if ret.types.is_empty() {
        quote! { ::acvus_mir_host::PolyTy::Unit }
    } else if ret.types.len() == 1 {
        build_poly_param_ty(&ret.types[0], generics)
    } else {
        let tys: Vec<_> = ret.types.iter().map(|ty| build_poly_param_ty(ty, generics)).collect();
        quote! { ::acvus_mir_host::PolyTy::Tuple(vec![#(#tys),*]) }
    };

    Ok(quote! {
        /// Build the PolyTy for this ExternFn (always a TyTerm::Fn).
        pub fn constraint(__interner: &::acvus_mir_host::Interner) -> ::acvus_mir_host::PolyTy {
            let mut __builder = ::acvus_mir_host::PolyBuilder::new();
            let __type_vars: Vec<::acvus_mir_host::PolyTy> = vec![#(#type_var_allocs),*];
            let __params = vec![#(#param_constructs),*];
            ::acvus_mir_host::PolyTy::Fn {
                params: __params,
                ret: Box::new(#ret_ty),
                captures: vec![],
                hint: None,
            }
        }
    })
}

/// Generate an infer-tagged ExternFn. Concrete only — no generics, no scope.
///
/// Handler: `fn name(args..., inferred_ret_ty: Ty) -> Inferrable`.
/// Constraint return type = fresh Var (inferred). No call wrapper.
fn generate_infer(
    original_fn: &ItemFn,
    _fn_name: &syn::Ident,
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

    let param_constructs: Vec<_> = user_params.iter().map(|p| {
        let name_str = p.name.to_string();
        let ty = &p.ty;
        quote! {
            ::acvus_mir_host::ParamTerm::<::acvus_mir_host::Poly>::new(
                __interner.intern(#name_str),
                <#ty as ::acvus_mir_host::ITy>::poly_ty(__interner, &[]),
            )
        }
    }).collect();

    Ok(quote! {
        #original_fn

        #vis struct #struct_name;

        impl #struct_name {
            /// Build the PolyTy. Return type = fresh Var (inferred).
            pub fn constraint(__interner: &::acvus_mir_host::Interner) -> ::acvus_mir_host::PolyTy {
                let mut __builder = ::acvus_mir_host::PolyBuilder::new();
                let __params = vec![#(#param_constructs),*];
                ::acvus_mir_host::PolyTy::Fn {
                    params: __params,
                    ret: Box::new(__builder.fresh_ty_var()),
                    captures: vec![],
                    hint: None,
                }
            }
        }
    })
}

// ── Helpers ────────────────────────────────────────────────────────

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

/// Replace generic type-param names with Typeck<N> in a type AST.
fn substitute_generics(ty: &syn::Type, generics: &[parse::GenericInfo]) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
                let seg = &type_path.path.segments[0];
                if matches!(seg.arguments, syn::PathArguments::None) {
                    for g in generics {
                        if seg.ident == g.name {
                            let idx = g.index;
                            return syn::parse_quote! { ::acvus_mir_host::Typeck<#idx> };
                        }
                    }
                }
            }
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
