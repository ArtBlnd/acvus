//! Classification of generic parameters into the three variable kinds.

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{GenericParam, Generics, Ident, Type, TypeParam, TypeParamBound, WherePredicate};

use crate::{bound_ident, span_of, subst};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    Ty,
    Effect,
    Len,
    Identity,
    /// The runtime parameter: at most one, bounded by `Runtime`.
    Runtime,
}

/// One generic parameter, its kind, and its index among that kind. A type
/// variable bounded by `Monomorphize<(..)>` also carries the member types
/// its handler is compiled for, and whether the runtime's value can stand in
/// for it: it can unless the parameter carries a trait bound the erased
/// value cannot satisfy.
pub struct Var {
    pub ident: Ident,
    pub kind: VarKind,
    pub index: usize,
    pub mono: Option<Vec<Type>>,
    pub mono_fallback: bool,
    /// The shared signature a `HasInstance<sig>` bound requires (RFC-0019).
    pub requires: Option<syn::Path>,
}

pub struct Vars(Vec<Var>);

/// The bounds on `tp`, written inline or in the where clause.
fn bounds_of<'a>(
    generics: &'a Generics,
    tp: &'a TypeParam,
) -> impl Iterator<Item = &'a TypeParamBound> {
    let from_where = generics
        .where_clause
        .iter()
        .flat_map(|w| w.predicates.iter())
        .filter_map(|p| match p {
            WherePredicate::Type(pt) => Some(pt),
            _ => None,
        })
        .filter(move |pt| match &pt.bounded_ty {
            Type::Path(path) => path.qself.is_none() && path.path.is_ident(&tp.ident),
            _ => false,
        })
        .flat_map(|pt| pt.bounds.iter());
    tp.bounds.iter().chain(from_where)
}

/// The signature of a `HasInstance<sig>` bound, if present.
fn required_signature<'a>(
    bounds: impl Iterator<Item = &'a TypeParamBound>,
) -> syn::Result<Option<syn::Path>> {
    for bound in bounds {
        let TypeParamBound::Trait(t) = bound else {
            continue;
        };
        let Some(seg) = t.path.segments.last() else {
            continue;
        };
        if seg.ident != "HasInstance" {
            continue;
        }
        let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
            return Err(syn::Error::new_spanned(seg, "HasInstance takes a signature"));
        };
        let Some(syn::GenericArgument::Type(Type::Path(sig))) = args.args.first() else {
            return Err(syn::Error::new_spanned(seg, "HasInstance takes a signature"));
        };
        return Ok(Some(sig.path.clone()));
    }
    Ok(None)
}

/// The member types of a `Monomorphize<(T0, T1, ..)>` bound, if present.
fn mono_members<'a>(
    bounds: impl Iterator<Item = &'a TypeParamBound>,
) -> syn::Result<Option<Vec<Type>>> {
    for bound in bounds {
        let TypeParamBound::Trait(t) = bound else {
            continue;
        };
        let Some(seg) = t.path.segments.last() else {
            continue;
        };
        if seg.ident != "Monomorphize" {
            continue;
        }
        let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
            return Err(syn::Error::new_spanned(
                seg,
                "Monomorphize takes a tuple of types",
            ));
        };
        let Some(syn::GenericArgument::Type(Type::Tuple(members))) = args.args.first() else {
            return Err(syn::Error::new_spanned(
                seg,
                "Monomorphize takes a tuple of types",
            ));
        };
        if members.elems.is_empty() {
            return Err(syn::Error::new_spanned(
                seg,
                "Monomorphize needs at least one type",
            ));
        }
        return Ok(Some(members.elems.iter().cloned().collect()));
    }
    Ok(None)
}

impl Vars {
    pub fn from_generics(generics: &Generics) -> syn::Result<Self> {
        let mut vars = Vec::new();
        let mut counts = [0usize; 5];
        for param in &generics.params {
            let GenericParam::Type(tp) = param else {
                return Err(syn::Error::new(
                    span_of(param),
                    "only type parameters bounded by TyVar, EffectVar, LenVar, IdentityVar, Monomorphize, or Runtime are allowed",
                ));
            };
            let kinds: Vec<VarKind> = bounds_of(generics, tp)
                .filter_map(bound_ident)
                .filter_map(|b| match b.to_string().as_str() {
                    "TyVar" => Some(VarKind::Ty),
                    "EffectVar" => Some(VarKind::Effect),
                    "LenVar" => Some(VarKind::Len),
                    "IdentityVar" => Some(VarKind::Identity),
                    "Runtime" => Some(VarKind::Runtime),
                    _ => None,
                })
                .collect();
            let mono = mono_members(bounds_of(generics, tp))?;
            let requires = required_signature(bounds_of(generics, tp))?;
            let has_extra_bounds = bounds_of(generics, tp).any(|b| {
                matches!(b, TypeParamBound::Trait(_))
                    && !bound_ident(b).is_some_and(|i| {
                        matches!(
                            i.to_string().as_str(),
                            "TyVar"
                                | "EffectVar"
                                | "LenVar"
                                | "IdentityVar"
                                | "Runtime"
                                | "Monomorphize"
                                | "HasInstance"
                        )
                    })
            });
            let kind = match (kinds.as_slice(), &mono, &requires) {
                ([kind], _, _) => *kind,
                ([], Some(_), _) | ([], None, Some(_)) => VarKind::Ty,
                _ => {
                    return Err(syn::Error::new(
                        tp.ident.span(),
                        "a generic parameter has exactly one of the bounds TyVar, EffectVar, LenVar, IdentityVar, Runtime, Monomorphize",
                    ));
                }
            };
            if mono.is_some() && kind != VarKind::Ty {
                return Err(syn::Error::new(
                    tp.ident.span(),
                    "Monomorphize is a type-variable bound",
                ));
            }
            let slot = kind as usize;
            let mono_fallback = mono.is_some() && !has_extra_bounds;
            vars.push(Var {
                ident: tp.ident.clone(),
                kind,
                index: counts[slot],
                mono,
                mono_fallback,
                requires,
            });
            counts[slot] += 1;
        }
        if vars.iter().filter(|v| v.mono.is_some()).count() > 1 {
            return Err(syn::Error::new(
                generics.params.span(),
                "at most one generic parameter is bounded by Monomorphize",
            ));
        }
        if counts[VarKind::Runtime as usize] > 1 {
            return Err(syn::Error::new(
                generics.params.span(),
                "at most one generic parameter is bounded by Runtime",
            ));
        }
        Ok(Self(vars))
    }

    pub fn lookup(&self, ident: &Ident) -> Option<(VarKind, usize)> {
        self.0
            .iter()
            .find(|v| v.ident == *ident)
            .map(|v| (v.kind, v.index))
    }

    /// `VarCounts { .. }` for this declaration, as an expression.
    pub fn counts_expr(&self) -> TokenStream {
        let count = |k| self.0.iter().filter(|v| v.kind == k).count();
        let (tys, effects, lens, identities) = (
            count(VarKind::Ty),
            count(VarKind::Effect),
            count(VarKind::Len),
            count(VarKind::Identity),
        );
        quote! {
            ::acvus_extern::VarCounts {
                tys: #tys,
                effects: #effects,
                lens: #lens,
                identities: #identities,
            }
        }
    }

    pub fn count(&self, kind: VarKind) -> usize {
        self.0.iter().filter(|v| v.kind == kind).count()
    }

    pub fn has_len_vars(&self) -> bool {
        self.0.iter().any(|v| v.kind == VarKind::Len)
    }

    /// Whether `ty` mentions a type, effect, or length variable.
    pub fn mentions_var(&self, ty: &Type) -> bool {
        let found = std::cell::Cell::new(false);
        subst::substitute(ty, &|ident| {
            if self
                .0
                .iter()
                .any(|v| v.kind != VarKind::Runtime && v.ident == *ident)
            {
                found.set(true);
            }
            None
        });
        found.get()
    }

    fn compile_time_stand_in(v: &Var) -> Type {
        let k = v.index;
        match v.kind {
            VarKind::Ty => syn::parse_quote! { ::acvus_extern::Typeck<#k> },
            VarKind::Effect => syn::parse_quote! { ::acvus_extern::Eff<#k> },
            VarKind::Len => syn::parse_quote! { ::acvus_extern::Len<#k> },
            VarKind::Identity => syn::parse_quote! { ::acvus_extern::Idn<#k> },
            VarKind::Runtime => syn::parse_quote! { __R },
        }
    }

    fn runtime_stand_in(v: &Var) -> Type {
        match v.kind {
            VarKind::Ty => syn::parse_quote! { <__R as ::acvus_extern::Runtime>::Value },
            VarKind::Effect | VarKind::Len | VarKind::Identity => syn::parse_quote! { () },
            VarKind::Runtime => syn::parse_quote! { __R },
        }
    }

    /// Compile-time substitution with the Monomorphize variable set to `member`.
    pub fn to_compile_time_instance(&self, ty: &Type, member: Option<&Type>) -> Type {
        subst::substitute(ty, &|ident| {
            let v = self.0.iter().find(|v| v.ident == *ident)?;
            match (&v.mono, member) {
                (Some(_), Some(m)) => Some(m.clone()),
                _ => Some(Self::compile_time_stand_in(v)),
            }
        })
    }

    /// Runtime substitution with the Monomorphize variable set to `member`.
    pub fn to_runtime_instance(&self, ty: &Type, member: Option<&Type>) -> Type {
        subst::substitute(ty, &|ident| {
            let v = self.0.iter().find(|v| v.ident == *ident)?;
            match (&v.mono, member) {
                (Some(_), Some(m)) => Some(m.clone()),
                _ => Some(Self::runtime_stand_in(v)),
            }
        })
    }

    /// The Monomorphize variable, if any.
    pub fn mono_var(&self) -> Option<&Var> {
        self.0.iter().find(|v| v.mono.is_some())
    }

    /// The required signature of every type variable, by position.
    pub fn requires_exprs(&self) -> Vec<TokenStream> {
        self.0
            .iter()
            .filter(|v| v.kind == VarKind::Ty)
            .map(|v| match &v.requires {
                None => quote! { ::core::option::Option::None },
                Some(sig) => quote! {
                    ::core::option::Option::Some(
                        <#sig as ::acvus_extern::SharedSignature>::qref(__i),
                    )
                },
            })
            .collect()
    }

    /// `TyVarBound` of every type variable, by position.
    pub fn bound_exprs(&self) -> Vec<TokenStream> {
        self.0
            .iter()
            .filter(|v| v.kind == VarKind::Ty)
            .map(|v| match &v.mono {
                None => quote! { ::acvus_extern::TyVarBound::Any },
                Some(members) => quote! {
                    ::acvus_extern::TyVarBound::OneOf(vec![#(
                        <#members as ::acvus_extern::TyArg>::poly_ty(__i, &__vars)
                    ),*])
                },
            })
            .collect()
    }

    /// `::<<__R as Runtime>::Value, (), (), __R>` in declaration order, with the
    /// Monomorphize variable set to `member`; empty when there are no generic
    /// parameters.
    pub fn runtime_turbofish_instance(&self, member: Option<&Type>) -> TokenStream {
        if self.0.is_empty() {
            return TokenStream::new();
        }
        let args = self.0.iter().map(|v| match (&v.mono, member) {
            (Some(_), Some(m)) => m.clone(),
            _ => Self::runtime_stand_in(v),
        });
        quote! { ::<#(#args),*> }
    }

    /// Impl generics with the argument bounds: `<T: TyArg, E: EffectArg>`.
    pub fn arg_impl_generics(&self) -> TokenStream {
        if self.0.is_empty() {
            return TokenStream::new();
        }
        let params = self.0.iter().map(|v| {
            let ident = &v.ident;
            match v.kind {
                VarKind::Ty => quote! { #ident: ::acvus_extern::TyArg + ::acvus_extern::TyVar },
                VarKind::Effect => quote! { #ident: ::acvus_extern::EffectArg },
                VarKind::Len => quote! { #ident: ::acvus_extern::LenArg },
                VarKind::Identity => quote! { #ident: ::acvus_extern::IdentityArg },
                VarKind::Runtime => quote! { #ident: ::acvus_extern::Runtime },
            }
        });
        quote! { <#(#params),*> }
    }

    pub fn type_arg_exprs(&self) -> Vec<TokenStream> {
        self.0
            .iter()
            .filter(|v| v.kind == VarKind::Ty)
            .map(|v| {
                let ident = &v.ident;
                quote! { <#ident as ::acvus_extern::TyArg>::poly_ty(__i, __vars) }
            })
            .collect()
    }

    pub fn identity_arg_exprs(&self) -> Vec<TokenStream> {
        self.0
            .iter()
            .filter(|v| v.kind == VarKind::Identity)
            .map(|v| {
                let ident = &v.ident;
                quote! { <#ident as ::acvus_extern::IdentityArg>::poly_identity(__vars) }
            })
            .collect()
    }

    pub fn effect_arg_exprs(&self) -> Vec<TokenStream> {
        self.0
            .iter()
            .filter(|v| v.kind == VarKind::Effect)
            .map(|v| {
                let ident = &v.ident;
                quote! { <#ident as ::acvus_extern::EffectArg>::poly_effect(__vars) }
            })
            .collect()
    }
}
