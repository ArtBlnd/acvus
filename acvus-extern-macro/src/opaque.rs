//! `#[derive(Opaque)]` (rfc_extern macros §3): an extern-defined opaque type,
//! held by the runtime as the Rust value itself. The derive reads the
//! definition and emits the type's crossing, its declared type and the
//! structural facts no stable bound states (safety §2), and nothing else.
//!
//! A generic parameter is a lifetime, the runtime (`R: Runtime`), the effect
//! the type carries (`E: EffectVar`), its identity (`I: IdentityVar`), or a
//! type variable (`T: TypeVar<R>`): parametric, its value the runtime's
//! uniform one, so a type parameter crosses and is declared only at
//! `Uniform<R, X>` (runtime contract §1). A field naming a
//! lifetime or a type parameter is `Structural`, which carries what P and M
//! say of them (safety §2): a field whose type projects through one or holds
//! it under interior mutability has no `Structural` and is refused by its
//! bound. The runtime, the effect and the identity are no layout's: a field
//! naming only them, a marker, is free.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Lifetime, Type, TypeParamBound, WherePredicate};

/// What a generic parameter of the type is.
enum Param {
    Lifetime(Lifetime),
    Runtime { ident: Ident, bounds: Vec<TypeParamBound> },
    Type { ident: Ident, bounds: Vec<TypeParamBound> },
    Effect { ident: Ident, bounds: Vec<TypeParamBound> },
    Identity { ident: Ident, bounds: Vec<TypeParamBound> },
}

pub fn derive(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let params = params(&input)?;
    let fields = field_types(&input)?;

    let lifetimes: Vec<&Lifetime> = params.iter().filter_map(|p| match p { Param::Lifetime(l) => Some(l), _ => None }).collect();
    let types: Vec<&Ident> = params.iter().filter_map(|p| match p { Param::Type { ident, .. } => Some(ident), _ => None }).collect();
    let runtime = params.iter().find_map(|p| match p { Param::Runtime { ident, .. } => Some(ident), _ => None });

    // Each parameter as written, with the bounds the definition gives it.
    let declared: Vec<TokenStream> = params.iter().map(|p| match p {
        Param::Lifetime(l) => quote!(#l),
        Param::Type { ident, bounds }
        | Param::Runtime { ident, bounds }
        | Param::Effect { ident, bounds }
        | Param::Identity { ident, bounds } => {
            quote!(#ident: #(#bounds)+*)
        }
    }).collect();
    let written: Vec<TokenStream> = params.iter().map(|p| match p {
        Param::Lifetime(l) => quote!(#l),
        Param::Type { ident: t, .. } | Param::Runtime { ident: t, .. } | Param::Effect { ident: t, .. } | Param::Identity { ident: t, .. } => {
            quote!(#t)
        }
    }).collect();

    // The fields that name a lifetime or a type parameter, each `Structural`
    // (P, M).
    let named: Vec<&Type> = fields.iter().copied().filter(|ty| names_a_parameter(ty, &params)).collect();
    let canon: Vec<TokenStream> = params.iter().map(|p| match p {
        Param::Lifetime(_) => quote!('static),
        Param::Type { ident: t, .. } => quote!(<#t as ::acvus_extern::Structural>::Canon),
        Param::Runtime { ident, .. } | Param::Effect { ident, .. } | Param::Identity { ident, .. } => quote!(#ident),
    }).collect();
    let statics: Vec<TokenStream> = params.iter().filter_map(|p| match p {
        Param::Effect { ident, .. } | Param::Identity { ident, .. } => Some(quote!(#ident: 'static)),
        _ => None,
    }).collect();
    let layout = match (lifetimes.is_empty(), types.is_empty()) {
        (false, _) => quote!(::acvus_extern::LayoutFact::CanonicalOf),
        (true, true) => quote!(::acvus_extern::LayoutFact::OwnCanonical),
        (true, false) => quote!(::acvus_extern::LayoutFact::of_parts(&[#(<#types as ::acvus_extern::Structural>::FACTS.layout),*])),
    };
    let generic_facts = params.iter().filter_map(|p| match p {
        Param::Lifetime(_) => Some(quote!(::acvus_extern::GenericParam::Lifetime)),
        Param::Type { .. } => Some(quote!(::acvus_extern::GenericParam::Type)),
        _ => None,
    });

    // The runtime the type crosses at: its own, or a fresh one.
    let at_runtime = runtime.cloned().unwrap_or_else(|| Ident::new("__R", Span::call_site()));
    let at_generics: Vec<TokenStream> = declared
        .iter()
        .cloned()
        .chain(runtime.is_none().then(|| quote!(#at_runtime: ::acvus_extern::runtime::Runtime)))
        .collect();
    // Its declared type: its name, each type parameter's declared type, its
    // effect and its identity variables' positions, and its lifetimes.
    let effect = match params.iter().find_map(|p| match p { Param::Effect { ident, .. } => Some(ident), _ => None }) {
        Some(e) => quote!(::core::option::Option::Some(::acvus_extern::decl::DeclEffect::Var(<#e as ::acvus_extern::decl::EffectVar>::INDEX))),
        None => quote!(::core::option::Option::None),
    };
    let identity = match params.iter().find_map(|p| match p { Param::Identity { ident, .. } => Some(ident), _ => None }) {
        Some(i) => quote!(::core::option::Option::Some(<#i as ::acvus_extern::decl::IdentityVar>::INDEX)),
        None => quote!(::core::option::Option::None),
    };
    let lifetime_count = lifetimes.len() as u32;
    let type_name = name.to_string();
    let runtime_value = quote!(<#at_runtime as ::acvus_extern::runtime::Runtime>::Value);
    // The type at one loan: each of its lifetimes `'__l`, the rest as
    // written; and the generics that leaves.
    let at_loan: Vec<TokenStream> = params.iter().map(|p| match p {
        Param::Lifetime(_) => quote!('__l),
        Param::Type { ident: t, .. } | Param::Runtime { ident: t, .. } | Param::Effect { ident: t, .. } | Param::Identity { ident: t, .. } => {
            quote!(#t)
        }
    }).collect();
    let loan_generics: Vec<TokenStream> = std::iter::once(quote!('__l))
        .chain(params.iter().filter_map(|p| match p {
            Param::Lifetime(_) => None,
            Param::Type { ident, bounds }
            | Param::Runtime { ident, bounds }
            | Param::Effect { ident, bounds }
            | Param::Identity { ident, bounds } => Some(quote!(#ident: #(#bounds)+*)),
        }))
        .chain(runtime.is_none().then(|| quote!(#at_runtime: ::acvus_extern::runtime::Runtime)))
        .collect();
    let key_generics: Vec<TokenStream> = std::iter::once(quote!('__l)).chain(at_generics.iter().cloned()).collect();
    let opaque_kind = quote!(<<#at_runtime as ::acvus_extern::runtime::Runtime>::Opaque as ::acvus_extern::runtime::OpaqueKind<#at_runtime>>);

    Ok(quote! {
        // SAFETY: `#[derive(Opaque)]` wrote the crossing below from the
        // definition: it gives, takes and lends the type as an opaque value,
        // by the runtime's opaque kind, at itself.
        #[allow(unsafe_code)]
        unsafe impl<#(#declared),*> ::acvus_extern::seal::Crossing for #name<#(#written),*> {}

        // Every field naming a parameter is `Structural`, at every
        // instantiation whose type parameters are: what the facts below rest
        // on (P, M).
        const _: () = {
            fn is_structural<S: ?::core::marker::Sized + ::acvus_extern::Structural>() {}
            #[allow(dead_code)]
            fn fields_are_structural<#(#declared),*>()
            where
                #(#types: ::acvus_extern::Structural,)*
            {
                #(is_structural::<#named>();)*
            }
        };

        // SAFETY: `#[derive(Opaque)]` read the definition. `Canon` is the type
        // with each type parameter at its canonical form and each lifetime at
        // `'static`; a type parameter's index has no projection, and a
        // lifetime no layout, so the two have one size, alignment, field
        // offsets and release. Every field naming a parameter is
        // `Structural` (checked above), so it reaches the parameter only by
        // holding it and has no interior mutability over it (P, M). `FACTS`
        // lists the lifetimes and type parameters in order (G).
        #[allow(unsafe_code)]
        unsafe impl<#(#declared),*> ::acvus_extern::Structural for #name<#(#written),*>
        where
            #(#types: ::acvus_extern::Structural,)*
            #(#statics,)*
        {
            type Canon = #name<#(#canon),*>;

            const FACTS: ::acvus_extern::StructuralFacts = ::acvus_extern::StructuralFacts {
                layout: #layout,
                generics: &[#(#generic_facts),*],
            };
        }

        impl<#(#declared),*> ::acvus_extern::runtime::Opaque for #name<#(#written),*>
        where
            Self: ::core::marker::Send + ::core::marker::Sync + ::acvus_extern::Structural,
        {
        }

        impl<#(#declared),*> ::acvus_extern::decl::DeclType for #name<#(#written),*> {
            fn decl_ty(names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::DeclTy {
                let module: ::std::vec::Vec<_> =
                    ::core::module_path!().split("::").map(|segment| names.intern(segment)).collect();
                ::acvus_extern::decl::DeclTy::Extern {
                    name: ::acvus_extern::decl::QualifiedRef::qualified(names.namespace(&module), names.intern(#type_name)),
                    args: ::std::vec![#(<#types as ::acvus_extern::decl::DeclType>::decl_ty(names)),*],
                    effect: #effect,
                    identity: #identity,
                    lifetimes: #lifetime_count,
                }
            }
        }

        // The type is held in place by the opaque kind, so it is boxed,
        // viewed and taken as every extern-defined value is. At a loan it
        // comes in as itself, every lifetime of it that loan; as a part of a
        // function type it is named with any lifetime, given boxed and opened
        // as itself at the call's loan; and written in a signature it is its
        // own key, viewed as itself and held boxed.
        impl<#(#loan_generics),*> ::acvus_extern::runtime::Open<'__l, #at_runtime> for #name<#(#at_loan),*>
        where
            Self: ::acvus_extern::runtime::Opaque,
        {
            fn decl(names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::DeclTy {
                <Self as ::acvus_extern::decl::DeclType>::decl_ty(names)
            }

            fn open(rt: &'__l #at_runtime, v: #runtime_value, at: ::acvus_extern::Settled<'__l>) -> Self {
                #opaque_kind::take(rt, v, at)
            }
        }

        impl<#(#key_generics),*> ::acvus_extern::runtime::Key<'__l, #at_runtime> for #name<#(#written),*>
        where
            #name<#(#at_loan),*>: ::acvus_extern::runtime::Open<'__l, #at_runtime> + ::acvus_extern::runtime::Opaque,
        {
            type Give = ::acvus_extern::Uniform<#at_runtime, #name<#(#at_loan),*>>;
            type Open = #name<#(#at_loan),*>;
        }

        impl<#(#at_generics),*> ::acvus_extern::Index<#at_runtime> for #name<#(#written),*>
        where
            Self: ::acvus_extern::runtime::Opaque,
        {
            type View = Self;
            type Key = Self;
        }

        impl<#(#at_generics),*> ::acvus_extern::runtime::Viewed<#at_runtime> for #name<#(#written),*>
        where
            Self: ::acvus_extern::runtime::Opaque,
        {
            type View = Self;
        }

        impl<#(#at_generics),*> ::acvus_extern::runtime::Repr<#at_runtime> for #name<#(#written),*>
        where
            Self: ::acvus_extern::runtime::Opaque,
        {
            type Held = ::acvus_extern::Uniform<#at_runtime, Self>;
        }

        impl<#(#loan_generics),*> ::acvus_extern::runtime::Place<'__l, #at_runtime> for #name<#(#at_loan),*>
        where
            Self: ::acvus_extern::runtime::Key<'__l, #at_runtime>,
        {
            type Key = Self;
        }
    })
}

/// The type's generic parameters, each by what its bounds say it is.
fn params(input: &DeriveInput) -> syn::Result<Vec<Param>> {
    let mut params = Vec::new();
    for param in &input.generics.params {
        match param {
            GenericParam::Const(c) => {
                return Err(syn::Error::new(c.span(), "an opaque type has no const parameter: a declaration states no length"));
            }
            GenericParam::Lifetime(l) => {
                if !l.bounds.is_empty() {
                    return Err(syn::Error::new(l.span(), "an opaque type's lifetime is unbounded"));
                }
                params.push(Param::Lifetime(l.lifetime.clone()));
            }
            GenericParam::Type(t) => {
                let mut bounds: Vec<TypeParamBound> = t.bounds.iter().cloned().collect();
                if let Some(clause) = &input.generics.where_clause {
                    for predicate in &clause.predicates {
                        if let WherePredicate::Type(p) = predicate
                            && matches!(&p.bounded_ty, Type::Path(path) if path.path.is_ident(&t.ident))
                        {
                            bounds.extend(p.bounds.iter().cloned());
                        }
                    }
                }
                let ident = t.ident.clone();
                params.push(match bound_kind(&bounds) {
                    Some(Kind::Type) => Param::Type { ident, bounds },
                    Some(Kind::Runtime) => Param::Runtime { ident, bounds },
                    Some(Kind::Effect) => Param::Effect { ident, bounds },
                    Some(Kind::Identity) => Param::Identity { ident, bounds },
                    None => {
                        return Err(syn::Error::new(
                            t.span(),
                            "an opaque type's parameter is its runtime (`R: Runtime`), a type variable \
                             (`T: TypeVar<R>`), an `EffectVar` or an `IdentityVar`",
                        ));
                    }
                });
            }
        }
    }
    if let Some(clause) = &input.generics.where_clause {
        for predicate in &clause.predicates {
            let bounds_a_param = matches!(predicate, WherePredicate::Type(p)
                if matches!(&p.bounded_ty, Type::Path(path) if params.iter().any(|q| matches!(q,
                    Param::Runtime { ident, .. } | Param::Type { ident, .. } | Param::Effect { ident, .. } | Param::Identity { ident, .. }
                        if path.path.is_ident(ident)))));
            if !bounds_a_param {
                return Err(syn::Error::new(predicate.span(), "an opaque type's `where` clause bounds only its runtime, type variables, effect and identity"));
            }
        }
    }
    let once = |kind: fn(&Param) -> bool, what: &str| -> syn::Result<()> {
        match params.iter().filter(|p| kind(p)).count() {
            0 | 1 => Ok(()),
            _ => Err(syn::Error::new(input.ident.span(), format!("an opaque type has at most one {what}"))),
        }
    };
    once(|p| matches!(p, Param::Runtime { .. }), "runtime")?;
    once(|p| matches!(p, Param::Effect { .. }), "effect (extern-fn §8)")?;
    once(|p| matches!(p, Param::Identity { .. }), "identity (types §3)")?;
    Ok(params)
}

enum Kind {
    Type,
    Runtime,
    Effect,
    Identity,
}

/// What a type parameter's bounds say it is: one `Runtime`, one
/// `TypeVar<R>`, one `EffectVar` or one `IdentityVar`.
fn bound_kind(bounds: &[TypeParamBound]) -> Option<Kind> {
    let [bound] = bounds else {
        return None;
    };
    let TypeParamBound::Trait(trait_bound) = bound else {
        return None;
    };
    let last = trait_bound.path.segments.last()?;
    if last.ident == "Runtime" && last.arguments.is_none() {
        return Some(Kind::Runtime);
    }
    match (last.ident.to_string().as_str(), last.arguments.is_none()) {
        ("TypeVar", false) => Some(Kind::Type),
        ("EffectVar", true) => Some(Kind::Effect),
        ("IdentityVar", true) => Some(Kind::Identity),
        _ => None,
    }
}

/// Every field's type: a struct's, or each variant's of an enum.
fn field_types(input: &DeriveInput) -> syn::Result<Vec<&Type>> {
    fn types(fields: &Fields) -> impl Iterator<Item = &Type> {
        fields.iter().map(|field| &field.ty)
    }
    match &input.data {
        Data::Struct(data) => Ok(types(&data.fields).collect()),
        Data::Enum(data) => Ok(data.variants.iter().flat_map(|variant| types(&variant.fields)).collect()),
        Data::Union(data) => Err(syn::Error::new(data.union_token.span(), "an opaque type is a struct or an enum")),
    }
}

/// Whether `ty` names one of the type's lifetimes or type parameters.
fn names_a_parameter(ty: &Type, params: &[Param]) -> bool {
    struct Names<'p> {
        params: &'p [Param],
        found: bool,
    }
    impl<'a> Visit<'a> for Names<'_> {
        fn visit_lifetime(&mut self, lifetime: &'a Lifetime) {
            self.found |= self.params.iter().any(|p| matches!(p, Param::Lifetime(l) if l == lifetime));
        }
        fn visit_ident(&mut self, ident: &'a Ident) {
            self.found |= self.params.iter().any(|p| matches!(p, Param::Type { ident: i, .. } if i == ident));
        }
    }
    let mut names = Names { params, found: false };
    names.visit_type(ty);
    names.found
}
