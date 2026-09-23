//! Classification of a declaration's generic parameters by kind.
//!
//! A parameter's kind is the argument of its `Var<K>` bound; the runtime
//! parameter is bounded by `Runtime`, which is a contract, not a kind.

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{GenericParam, Generics, Ident, Type, TypeParam, TypeParamBound, WherePredicate};

use crate::{bound_ident, span_of, subst};

/// The bounds a declaration's generic parameter may carry, by name.
const KIND_BOUNDS: &str = "Var<kind::Type>, Var<kind::Effect>, Var<kind::Length>, \
                           Var<kind::Identity>, Monomorphize, or Runtime";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    Ty,
    Effect,
    Len,
    Identity,
    /// The runtime parameter: at most one, bounded by `Runtime`.
    Runtime,
}

impl VarKind {
    /// The `kind::` marker naming this kind, as a type.
    fn marker(self) -> Type {
        match self {
            VarKind::Ty => syn::parse_quote! { ::acvus_extern::kind::Type },
            VarKind::Effect => syn::parse_quote! { ::acvus_extern::kind::Effect },
            VarKind::Len => syn::parse_quote! { ::acvus_extern::kind::Length },
            VarKind::Identity => syn::parse_quote! { ::acvus_extern::kind::Identity },
            VarKind::Runtime => unreachable!("the runtime parameter has no kind marker"),
        }
    }
}

/// The kind a `Var<K>` bound names, by `K`'s last path segment.
fn kind_of(bound: &TypeParamBound) -> Option<syn::Result<VarKind>> {
    let TypeParamBound::Trait(t) = bound else {
        return None;
    };
    let seg = t.path.segments.last()?;
    if seg.ident == "Runtime" {
        return Some(Ok(VarKind::Runtime));
    }
    if seg.ident != "Var" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Some(Err(syn::Error::new_spanned(seg, "Var takes a kind")));
    };
    let Some(syn::GenericArgument::Type(Type::Path(k))) = args.args.first() else {
        return Some(Err(syn::Error::new_spanned(seg, "Var takes a kind")));
    };
    let Some(name) = k.path.segments.last() else {
        return Some(Err(syn::Error::new_spanned(seg, "Var takes a kind")));
    };
    Some(match name.ident.to_string().as_str() {
        "Type" => Ok(VarKind::Ty),
        "Effect" => Ok(VarKind::Effect),
        "Length" => Ok(VarKind::Len),
        "Identity" => Ok(VarKind::Identity),
        _ => Err(syn::Error::new_spanned(
            name,
            "a kind is Type, Effect, Length, or Identity",
        )),
    })
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

/// The signature a required instance names, as a path with its arguments
/// dropped: `sig::eq<T, Rt>` is `sig::eq`, whose defaulted parameters make
/// it the same marker type the `instance_of` attribute names.
pub fn signature_path(sig: &Type) -> syn::Result<syn::Path> {
    let Type::Path(p) = sig else {
        return Err(syn::Error::new_spanned(sig, "a signature is a path"));
    };
    let mut path = p.path.clone();
    for segment in &mut path.segments {
        segment.arguments = syn::PathArguments::None;
    }
    Ok(path)
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
            // A lifetime parameter is admitted and dropped, which is a
            // decision: a declaration whose result may borrow either of two
            // parameters has to name one lifetime for Rust, and that
            // lifetime is the call's own rather than an acvus type variable.
            if matches!(param, GenericParam::Lifetime(_)) {
                continue;
            }
            let GenericParam::Type(tp) = param else {
                return Err(syn::Error::new(
                    span_of(param),
                    format!(
                        "only lifetimes and type parameters bounded by {KIND_BOUNDS} are allowed"
                    ),
                ));
            };
            let kinds: Vec<VarKind> = bounds_of(generics, tp)
                .filter_map(kind_of)
                .collect::<syn::Result<_>>()?;
            let mono = mono_members(bounds_of(generics, tp))?;
            let has_extra_bounds = bounds_of(generics, tp).any(|b| {
                matches!(b, TypeParamBound::Trait(_))
                    && kind_of(b).is_none()
                    && !bound_ident(b).is_some_and(|i| i == "Monomorphize")
            });
            let kind = match (kinds.as_slice(), &mono) {
                ([kind], _) => *kind,
                ([], Some(_)) => VarKind::Ty,
                _ => {
                    return Err(syn::Error::new(
                        tp.ident.span(),
                        format!("a generic parameter has exactly one of the bounds {KIND_BOUNDS}"),
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

    /// Every generic parameter, in declaration order.
    pub fn idents(&self) -> Vec<&Ident> {
        self.0.iter().map(|v| &v.ident).collect()
    }

    /// The first type variable, which is the one an instance of a signature
    /// is matched by (RFC-0067 rule 2).
    pub fn first_ty(&self) -> Option<&Ident> {
        self.0
            .iter()
            .find(|v| v.kind == VarKind::Ty)
            .map(|v| &v.ident)
    }

    /// Every type variable, in declaration order.
    pub fn type_vars(&self) -> impl Iterator<Item = &Ident> {
        self.0
            .iter()
            .filter(|v| v.kind == VarKind::Ty)
            .map(|v| &v.ident)
    }

    /// The generic parameter bounded by `Runtime`, when the declaration has one.
    pub fn runtime_ident(&self) -> Option<&Ident> {
        self.0
            .iter()
            .find(|v| v.kind == VarKind::Runtime)
            .map(|v| &v.ident)
    }

    pub fn lookup(&self, ident: &Ident) -> Option<(VarKind, usize)> {
        self.0
            .iter()
            .find(|v| v.ident == *ident)
            .map(|v| (v.kind, v.index))
    }

    /// This declaration's `PolyVars`, as an expression.
    pub fn fresh_vars_expr(&self) -> TokenStream {
        let count = |k| self.0.iter().filter(|v| v.kind == k).count();
        let (tys, effects, lens, identities) = (
            count(VarKind::Ty),
            count(VarKind::Effect),
            count(VarKind::Len),
            count(VarKind::Identity),
        );
        quote! {
            ::acvus_extern::PolyVars::fresh(#tys, #effects, #lens, #identities)
        }
    }

    pub fn count(&self, kind: VarKind) -> usize {
        self.0.iter().filter(|v| v.kind == kind).count()
    }

    pub fn has_len_vars(&self) -> bool {
        self.0.iter().any(|v| v.kind == VarKind::Len)
    }

    fn compile_time_stand_in(v: &Var) -> Type {
        let k = v.index;
        match v.kind {
            VarKind::Runtime => syn::parse_quote! { __R },
            kind => {
                let marker = kind.marker();
                syn::parse_quote! { ::acvus_extern::Nth<#marker, #k> }
            }
        }
    }

    /// The Rust type a variable is instantiated at when the glue calls the
    /// handler. Obligation across crates: a runtime declares the Rust type
    /// it erases a container through (`acvus-interpreter`'s `value::Array`
    /// is `Arr<Owned<AcvusRuntime>, ()>`), and this substitution is what
    /// makes a handler's `Arr<T, N>` that same type. Change one and the
    /// other must follow; both compile either way.
    fn runtime_stand_in(v: &Var) -> Type {
        match v.kind {
            VarKind::Ty => syn::parse_quote! { ::acvus_extern::Owned<__R> },
            VarKind::Effect | VarKind::Len | VarKind::Identity => syn::parse_quote! { () },
            VarKind::Runtime => syn::parse_quote! { __R },
        }
    }

    /// Whether `ty` mentions a type variable.
    pub fn mentions_ty_var(&self, ty: &Type) -> bool {
        let found = std::cell::Cell::new(false);
        subst::substitute(ty, &|ident| {
            if self.0.iter().any(|v| v.kind == VarKind::Ty && v.ident == *ident) {
                found.set(true);
            }
            None
        });
        found.get()
    }

    /// Whether `ty` mentions the Monomorphize variable.
    pub fn mentions_mono(&self, ty: &Type) -> bool {
        let found = std::cell::Cell::new(false);
        subst::substitute(ty, &|ident| {
            if self.0.iter().any(|v| v.mono.is_some() && v.ident == *ident) {
                found.set(true);
            }
            None
        });
        found.get()
    }

    /// Compile-time substitution with the Monomorphize variable set to
    /// `Spec<member>`, the form whose slots are `#`.
    pub fn to_compile_time_instance(&self, ty: &Type, member: Option<&Type>) -> Type {
        subst::substitute(ty, &|ident| {
            let v = self.0.iter().find(|v| v.ident == *ident)?;
            match (&v.mono, member) {
                (Some(_), Some(m)) => Some(syn::parse_quote! { ::acvus_extern::Spec<#m> }),
                _ => Some(Self::compile_time_stand_in(v)),
            }
        })
    }

    /// Compile-time substitution with the Monomorphize variable set to
    /// `member` itself, the form whose slots are uniform.
    pub fn to_compile_time_uniform(&self, ty: &Type, member: &Type) -> Type {
        subst::substitute(ty, &|ident| {
            let v = self.0.iter().find(|v| v.ident == *ident)?;
            match &v.mono {
                Some(_) => Some(member.clone()),
                None => Some(Self::compile_time_stand_in(v)),
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

    /// The kind bound of every variable, as a where predicate: what a
    /// declaration wrote in its own `where` clause, restated where a
    /// generated impl names a type built out of these variables.
    pub fn kind_predicates(&self) -> Vec<TokenStream> {
        self.0
            .iter()
            .map(|v| {
                let ident = &v.ident;
                match v.kind {
                    VarKind::Runtime => quote! { #ident: ::acvus_extern::Runtime },
                    kind => {
                        let marker = kind.marker();
                        quote! { #ident: ::acvus_extern::Var<#marker> }
                    }
                }
            })
            .collect()
    }

    /// Whether `ty` is exactly the variable `ident` and nothing else.
    pub fn is_exactly(ty: &Type, ident: &Ident) -> bool {
        matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident(ident))
    }

    /// The Monomorphize variable, if any.
    pub fn mono_var(&self) -> Option<&Var> {
        self.0.iter().find(|v| v.mono.is_some())
    }

    /// `TyVarBound` of every type variable, by position.
    pub fn bound_exprs(&self) -> Vec<TokenStream> {
        self.0
            .iter()
            .filter(|v| v.kind == VarKind::Ty)
            .map(|v| match &v.mono {
                None => quote! { ::acvus_extern::TyVarBound::Any },
                Some(members) => quote! {
                    ::acvus_extern::TyVarBound::one_of(vec![#(
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

    /// Impl generics with the bounds that name a term: `<T: TyArg + Var<kind::Type>, E: Term<kind::Effect>>`.
    pub fn arg_impl_generics(&self) -> TokenStream {
        if self.0.is_empty() {
            return TokenStream::new();
        }
        let params = self.0.iter().map(|v| {
            let ident = &v.ident;
            match v.kind {
                VarKind::Ty => {
                    let marker = VarKind::Ty.marker();
                    quote! {
                        #ident: ::acvus_extern::TyArg + ::acvus_extern::Var<#marker>
                    }
                }
                VarKind::Runtime => quote! { #ident: ::acvus_extern::Runtime },
                kind => {
                    let marker = kind.marker();
                    quote! { #ident: ::acvus_extern::Term<#marker> }
                }
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
                quote! { <#ident as ::acvus_extern::TyArg>::held(__i, __vars) }
            })
            .collect()
    }

    pub fn identity_arg_exprs(&self) -> Vec<TokenStream> {
        self.term_exprs(VarKind::Identity)
    }

    pub fn effect_arg_exprs(&self) -> Vec<TokenStream> {
        self.term_exprs(VarKind::Effect)
    }

    /// `<X as Term<K>>::poly(__vars)` for every variable of kind `kind`.
    fn term_exprs(&self, kind: VarKind) -> Vec<TokenStream> {
        let marker = kind.marker();
        self.0
            .iter()
            .filter(|v| v.kind == kind)
            .map(|v| {
                let ident = &v.ident;
                quote! { <#ident as ::acvus_extern::Term<#marker>>::poly(__vars) }
            })
            .collect()
    }
}
