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
    /// The runtime parameter: at most one, bounded by `Runtime`.
    Runtime,
}

/// One generic parameter, its kind, and its index among that kind.
pub struct Var {
    pub ident: Ident,
    pub kind: VarKind,
    pub index: usize,
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

impl Vars {
    pub fn from_generics(generics: &Generics) -> syn::Result<Self> {
        let mut vars = Vec::new();
        let mut counts = [0usize; 4];
        for param in &generics.params {
            let GenericParam::Type(tp) = param else {
                return Err(syn::Error::new(
                    span_of(param),
                    "only type parameters bounded by TyVar, EffectVar, LenVar, or Runtime are allowed",
                ));
            };
            let kinds: Vec<VarKind> = bounds_of(generics, tp)
                .filter_map(bound_ident)
                .filter_map(|b| match b.to_string().as_str() {
                    "TyVar" => Some(VarKind::Ty),
                    "EffectVar" => Some(VarKind::Effect),
                    "LenVar" => Some(VarKind::Len),
                    "Runtime" => Some(VarKind::Runtime),
                    _ => None,
                })
                .collect();
            let [kind] = kinds.as_slice() else {
                return Err(syn::Error::new(
                    tp.ident.span(),
                    "a generic parameter has exactly one of the bounds TyVar, EffectVar, LenVar, Runtime",
                ));
            };
            let slot = *kind as usize;
            vars.push(Var {
                ident: tp.ident.clone(),
                kind: *kind,
                index: counts[slot],
            });
            counts[slot] += 1;
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

    pub fn counts(&self) -> (usize, usize, usize) {
        let count = |k| self.0.iter().filter(|v| v.kind == k).count();
        (
            count(VarKind::Ty),
            count(VarKind::Effect),
            count(VarKind::Len),
        )
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

    /// The declared runtime parameter, if the item names one.
    pub fn runtime_param(&self) -> Option<&Ident> {
        self.0
            .iter()
            .find(|v| v.kind == VarKind::Runtime)
            .map(|v| &v.ident)
    }

    fn compile_time_stand_in(v: &Var) -> Type {
        let k = v.index;
        match v.kind {
            VarKind::Ty => syn::parse_quote! { ::acvus_extern::Typeck<#k> },
            VarKind::Effect => syn::parse_quote! { ::acvus_extern::Eff<#k> },
            VarKind::Len => syn::parse_quote! { ::acvus_extern::Len<#k> },
            VarKind::Runtime => syn::parse_quote! { __R },
        }
    }

    fn runtime_stand_in(v: &Var) -> Type {
        match v.kind {
            VarKind::Ty => syn::parse_quote! { <__R as ::acvus_extern::Runtime>::Value },
            VarKind::Effect | VarKind::Len => syn::parse_quote! { () },
            VarKind::Runtime => syn::parse_quote! { __R },
        }
    }

    pub fn to_compile_time(&self, ty: &Type) -> Type {
        subst::substitute(ty, &|ident| {
            self.0
                .iter()
                .find(|v| v.ident == *ident)
                .map(Self::compile_time_stand_in)
        })
    }

    pub fn to_runtime(&self, ty: &Type) -> Type {
        subst::substitute(ty, &|ident| {
            self.0
                .iter()
                .find(|v| v.ident == *ident)
                .map(Self::runtime_stand_in)
        })
    }

    /// `::<<__R as Runtime>::Value, (), (), __R>` in declaration order; empty
    /// when there are no generic parameters.
    pub fn runtime_turbofish(&self) -> TokenStream {
        if self.0.is_empty() {
            return TokenStream::new();
        }
        let args = self.0.iter().map(Self::runtime_stand_in);
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
