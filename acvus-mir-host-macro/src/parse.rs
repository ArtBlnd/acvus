//! Parsing logic for #[extern_fn] and #[derive(ExternType)] attributes.

use syn::parse::{Parse, ParseStream};
use syn::{Attribute, FnArg, GenericParam, Ident, LitStr, Pat, ReturnType, Signature, Token, Type, TypeParamBound};

// ── Attribute parsing ──────────────────────────────────────────────

/// Parsed `#[extern_fn(name = "...", StructName, ns = "...", infer)]`.
pub struct ExternFnAttr {
    pub name: LitStr,
    pub struct_name: Ident,
    /// Namespace for QualifiedRef. None = root.
    pub ns: Option<LitStr>,
    /// Handler returns Inferrable; output = Constraint::Inferred.
    pub infer: bool,
}

impl Parse for ExternFnAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name_ident: Ident = input.parse()?;
        if name_ident != "name" {
            return Err(syn::Error::new(name_ident.span(), "expected `name`"));
        }
        let _eq: Token![=] = input.parse()?;
        let name: LitStr = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let struct_name: Ident = input.parse()?;

        let mut ns = None;
        let mut infer = false;
        while input.peek(Token![,]) {
            let _comma: Token![,] = input.parse()?;
            if input.is_empty() { break; }
            let key: Ident = input.parse()?;
            if key == "infer" {
                infer = true;
            } else if key == "ns" {
                let _eq: Token![=] = input.parse()?;
                ns = Some(input.parse::<LitStr>()?);
            } else {
                return Err(syn::Error::new(key.span(), "expected `infer` or `ns`"));
            }
        }

        Ok(ExternFnAttr { name, struct_name, ns, infer })
    }
}

// ── Parameter classification ───────────────────────────────────────

#[derive(Clone)]
pub struct ParsedParam {
    pub name: Ident,
    pub ty: Type,
}

pub struct ParsedParams {
    pub params: Vec<ParsedParam>,
    /// Handler receives &mut Scope as first param (detected from S: Scope generic).
    pub has_scope: bool,
}

pub struct ReturnInfo {
    /// Concrete return types (from the tuple).
    pub types: Vec<Type>,
}

// ── Generic parameter info ──────────────────────────────────────────

/// A generic parameter mapped to a Typeck<N> index.
#[derive(Clone)]
pub struct GenericInfo {
    pub name: Ident,
    pub index: usize,
    /// If this param has a Callable<Args, Ret, E> bound.
    pub callable: Option<CallableInfo>,
    /// If this param has a Monomorphize<(T1, T2, ...)> bound.
    pub monomorphize: Option<Vec<Type>>,
}

/// Extracted Callable<Args, Ret> bound info.
#[derive(Clone)]
pub struct CallableInfo {
    pub args_ty: Type,
    pub ret_ty: Type,
}

/// Parse generic type parameters from the function signature.
/// Skips `S: Scope` (interpreter parameter, not a type variable).
/// Returns the list of type variables with Typeck indices.
/// Result of parsing generics — includes the Scope param name if present.
pub struct GenericsResult {
    pub generics: Vec<GenericInfo>,
    /// Name of the Scope type param (e.g., "S"), if present.
    pub scope_param: Option<Ident>,
}

pub fn parse_generics(sig: &Signature) -> syn::Result<GenericsResult> {
    let mut generics = Vec::new();
    let mut type_index = 0usize;
    let mut scope_param = None;

    for param in &sig.generics.params {
        if let GenericParam::Type(type_param) = param {
            let has_bound = |name: &str| {
                type_param.bounds.iter().any(|bound| {
                    if let TypeParamBound::Trait(trait_bound) = bound {
                        trait_bound.path.segments.last()
                            .map_or(false, |seg| seg.ident == name)
                    } else {
                        false
                    }
                })
            };

            if has_bound("Scope") {
                scope_param = Some(type_param.ident.clone());
                continue;
            }

            let callable = extract_callable_bound(&type_param.bounds);
            let monomorphize = extract_monomorphize_bound(&type_param.bounds);
            generics.push(GenericInfo {
                name: type_param.ident.clone(),
                index: type_index,
                callable,
                monomorphize,
            });
            type_index += 1;
        }
    }

    Ok(GenericsResult { generics, scope_param })
}

/// Extract concrete types from a `Monomorphize<(T1, T2, ...)>` bound, if present.
fn extract_monomorphize_bound(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
) -> Option<Vec<Type>> {
    for bound in bounds {
        if let TypeParamBound::Trait(trait_bound) = bound {
            let seg = trait_bound.path.segments.last()?;
            if seg.ident != "Monomorphize" {
                continue;
            }
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                if let Some(syn::GenericArgument::Type(Type::Tuple(tuple))) = args.args.first() {
                    return Some(tuple.elems.iter().cloned().collect());
                }
            }
        }
    }
    None
}

/// Extract Args and Ret from a `Callable<Args, Ret>` bound, if present.
fn extract_callable_bound(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
) -> Option<CallableInfo> {
    for bound in bounds {
        if let TypeParamBound::Trait(trait_bound) = bound {
            let seg = trait_bound.path.segments.last()?;
            if seg.ident != "Callable" {
                continue;
            }
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                let mut iter = args.args.iter();
                let args_ty = match iter.next() {
                    Some(syn::GenericArgument::Type(ty)) => ty.clone(),
                    _ => continue,
                };
                let ret_ty = match iter.next() {
                    Some(syn::GenericArgument::Type(ty)) => ty.clone(),
                    _ => continue,
                };
                return Some(CallableInfo { args_ty, ret_ty });
            }
        }
    }
    None
}

// ── Parsing ────────────────────────────────────────────────────────

/// Parse function params. If `scope_param` is set, skip any `&mut S` parameter
/// (it receives __scope directly, not from a slot).
pub fn parse_params(sig: &Signature, scope_param: Option<&Ident>) -> syn::Result<ParsedParams> {
    let mut params = Vec::new();
    let mut has_scope = false;

    for arg in &sig.inputs {
        match arg {
            FnArg::Typed(pat_type) => {
                let name = match pat_type.pat.as_ref() {
                    Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                    _ => return Err(syn::Error::new_spanned(&pat_type.pat, "expected identifier")),
                };

                // Check if this is `&mut S` where S is the Scope type param.
                if let Some(scope_name) = scope_param {
                    if is_mut_ref_to(&pat_type.ty, scope_name) {
                        has_scope = true;
                        continue; // skip — __scope is passed directly
                    }
                }

                params.push(ParsedParam { name, ty: pat_type.ty.as_ref().clone() });
            }
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(arg, "extern_fn cannot have self parameter"));
            }
        }
    }
    Ok(ParsedParams { params, has_scope })
}

/// Check if a type is `&mut Name` where Name matches the given ident.
fn is_mut_ref_to(ty: &Type, name: &Ident) -> bool {
    if let Type::Reference(ref_ty) = ty {
        if ref_ty.mutability.is_some() {
            if let Type::Path(path) = ref_ty.elem.as_ref() {
                if path.qself.is_none() && path.path.segments.len() == 1 {
                    return path.path.segments[0].ident == *name;
                }
            }
        }
    }
    false
}

pub fn parse_return(sig: &Signature) -> syn::Result<ReturnInfo> {
    match &sig.output {
        ReturnType::Default => Ok(ReturnInfo { types: vec![] }),
        ReturnType::Type(_, ty) => {
            match ty.as_ref() {
                Type::Tuple(tuple) => {
                    Ok(ReturnInfo { types: tuple.elems.iter().cloned().collect() })
                }
                // Bare return type (not tuple) — treat as single return.
                other => Ok(ReturnInfo { types: vec![other.clone()] }),
            }
        }
    }
}

// ── ExternType attribute parsing ───────────────────────────────────

/// Parsed ExternType attribute info.
pub struct ExternTypeAttr {
    pub name: String,
    pub ns: Option<String>,
}

/// Parse `#[extern_type(name = "Iterator", ns = "core")]` from derive attributes.
pub fn parse_extern_type_attr(attrs: &[Attribute]) -> syn::Result<ExternTypeAttr> {
    for attr in attrs {
        if !attr.path().is_ident("extern_type") {
            continue;
        }
        let mut name = None;
        let mut ns = None;

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let _eq: Token![=] = meta.input.parse()?;
                let lit: LitStr = meta.input.parse()?;
                name = Some(lit.value());
            } else if meta.path.is_ident("ns") {
                let _eq: Token![=] = meta.input.parse()?;
                let lit: LitStr = meta.input.parse()?;
                ns = Some(lit.value());
            } else {
                return Err(meta.error("expected `name` or `ns`"));
            }
            Ok(())
        })?;

        let name = name.ok_or_else(|| syn::Error::new_spanned(attr, "missing `name` in #[extern_type]"))?;
        return Ok(ExternTypeAttr { name, ns });
    }
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "missing #[extern_type(name = \"...\")] attribute",
    ))
}
