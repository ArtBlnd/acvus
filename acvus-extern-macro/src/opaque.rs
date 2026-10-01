//! `#[derive(Opaque)]` (rfc_extern macros §2): an extern-defined opaque type,
//! held by the runtime as the Rust value itself. The derive reads the
//! definition and emits the type's crossing, its declared type and the
//! structural facts no stable bound states (safety §2), and nothing else.
//!
//! A generic parameter is a lifetime, the runtime (`R: Runtime`), the effect
//! the type carries (`E: Var<kind::Effect>`), its identity
//! (`I: Var<kind::Identity>`), or a type parameter, unbounded: parametric, its
//! value the runtime's uniform one, so a type parameter crosses and is
//! declared only at `Uniform<R, X>` (runtime contract §1). A field naming a
//! lifetime or a type parameter is `Structural`, which carries what P and M
//! say of them (safety §2): a field whose type projects through one or holds
//! it under interior mutability has no `Structural` and is refused by its
//! bound. The runtime, the effect and the identity are no layout's: a field
//! naming only them, a marker, is free.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Lifetime, PathArguments, Type, TypeParamBound, WherePredicate};

/// What a generic parameter of the type is.
enum Param {
    Lifetime(Lifetime),
    Runtime { ident: Ident, bounds: Vec<TypeParamBound> },
    Type(Ident),
    Effect { ident: Ident, bounds: Vec<TypeParamBound> },
    Identity { ident: Ident, bounds: Vec<TypeParamBound> },
}

pub fn derive(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let params = params(&input)?;
    let fields = field_types(&input)?;

    let lifetimes: Vec<&Lifetime> = params.iter().filter_map(|p| match p { Param::Lifetime(l) => Some(l), _ => None }).collect();
    let types: Vec<&Ident> = params.iter().filter_map(|p| match p { Param::Type(t) => Some(t), _ => None }).collect();
    let runtime = params.iter().find_map(|p| match p { Param::Runtime { ident, .. } => Some(ident), _ => None });

    // Each parameter as written, with the bounds the definition gives it.
    let declared: Vec<TokenStream> = params.iter().map(|p| match p {
        Param::Lifetime(l) => quote!(#l),
        Param::Type(t) => quote!(#t),
        Param::Runtime { ident, bounds } | Param::Effect { ident, bounds } | Param::Identity { ident, bounds } => {
            quote!(#ident: #(#bounds)+*)
        }
    }).collect();
    let written: Vec<TokenStream> = params.iter().map(|p| match p {
        Param::Lifetime(l) => quote!(#l),
        Param::Type(t) | Param::Runtime { ident: t, .. } | Param::Effect { ident: t, .. } | Param::Identity { ident: t, .. } => {
            quote!(#t)
        }
    }).collect();

    // The fields that name a lifetime or a type parameter, each `Structural`
    // (P, M).
    let named: Vec<&Type> = fields.iter().copied().filter(|ty| names_a_parameter(ty, &params)).collect();
    let canon: Vec<TokenStream> = params.iter().map(|p| match p {
        Param::Lifetime(_) => quote!('static),
        Param::Type(t) => quote!(<#t as ::acvus_extern::Structural>::Canon),
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
        Param::Type(_) => Some(quote!(::acvus_extern::GenericParam::Type)),
        _ => None,
    });

    // The type at the runtime it crosses at: each type parameter its uniform
    // value at an index, the runtime its own or a fresh one.
    let at_runtime = runtime.cloned().unwrap_or_else(|| Ident::new("__R", Span::call_site()));
    let indices: Vec<Ident> = (0..types.len()).map(|at| format_ident!("__X{}", at)).collect();
    let crossing_args = at_indices(&params, &at_runtime, |effect| quote!(#effect), |identity| quote!(#identity));
    let crossing_generics: Vec<TokenStream> = lifetimes
        .iter()
        .map(|l| quote!(#l))
        .chain(std::iter::once(quote!(#at_runtime: ::acvus_extern::runtime::Runtime)))
        .chain(indices.iter().map(|x| quote!(#x)))
        .chain(params.iter().filter_map(|p| match p {
            Param::Effect { ident, bounds } | Param::Identity { ident, bounds } => Some(quote!(#ident: #(#bounds)+*)),
            _ => None,
        }))
        .collect();

    // The type as a declaration writes it: each type parameter at a declared
    // type, its effect and identity at the declaration's variables.
    let declared_args = at_indices(
        &params,
        &at_runtime,
        |_| quote!(::acvus_extern::decl::EffectNth<__EFFECT>),
        |_| quote!(::acvus_extern::decl::IdentityNth<__IDENTITY>),
    );
    let declared_generics: Vec<TokenStream> = lifetimes
        .iter()
        .map(|l| quote!(#l))
        .chain((runtime.is_some() || !types.is_empty()).then(|| quote!(#at_runtime: ::acvus_extern::runtime::Runtime)))
        .chain(indices.iter().map(|x| quote!(#x: ::acvus_extern::decl::DeclType)))
        .chain(params.iter().filter_map(|p| match p {
            Param::Effect { .. } => Some(quote!(const __EFFECT: u32)),
            Param::Identity { .. } => Some(quote!(const __IDENTITY: u32)),
            _ => None,
        }))
        .collect();
    let effect = match params.iter().any(|p| matches!(p, Param::Effect { .. })) {
        true => quote!(::core::option::Option::Some(::acvus_extern::decl::DeclEffect::Var(__EFFECT))),
        false => quote!(::core::option::Option::None),
    };
    let identity = match params.iter().any(|p| matches!(p, Param::Identity { .. })) {
        true => quote!(::core::option::Option::Some(__IDENTITY)),
        false => quote!(::core::option::Option::None),
    };
    let lifetime_count = lifetimes.len() as u32;
    let type_name = name.to_string();

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

        impl<#(#declared),*> ::acvus_extern::runtime::Referent for #name<#(#written),*> {}

        impl<#(#declared),*> ::acvus_extern::runtime::Opaque for #name<#(#written),*>
        where
            Self: ::core::marker::Send + ::core::marker::Sync + ::acvus_extern::Structural,
        {
        }

        impl<#(#crossing_generics),*> ::acvus_extern::runtime::Crosses<#at_runtime> for #name<#(#crossing_args),*>
        where
            Self: ::acvus_extern::runtime::Opaque,
        {
            type Open<'__a>
                = &'__a Self
            where
                Self: '__a,
                #at_runtime: '__a;
            type OpenMut<'__a>
                = &'__a mut Self
            where
                Self: '__a,
                #at_runtime: '__a;

            fn give(self, rt: &#at_runtime, at: ::acvus_extern::Settled<'_>) -> <#at_runtime as ::acvus_extern::runtime::Runtime>::Value {
                <<#at_runtime as ::acvus_extern::runtime::Runtime>::Opaque as ::acvus_extern::runtime::OpaqueKind<#at_runtime>>::give(rt, self, at)
            }

            fn take(rt: &#at_runtime, v: <#at_runtime as ::acvus_extern::runtime::Runtime>::Value, at: ::acvus_extern::Settled<'_>) -> Self {
                <<#at_runtime as ::acvus_extern::runtime::Runtime>::Opaque as ::acvus_extern::runtime::OpaqueKind<#at_runtime>>::take(rt, v, at)
            }

            fn open<'__a>(
                rt: &'__a #at_runtime,
                place: &'__a <#at_runtime as ::acvus_extern::runtime::Runtime>::Value,
                at: ::acvus_extern::Settled<'__a>,
            ) -> Self::Open<'__a>
            where
                Self: '__a,
            {
                <<#at_runtime as ::acvus_extern::runtime::Runtime>::Opaque as ::acvus_extern::runtime::OpaqueKind<#at_runtime>>::shared(rt, place, at)
            }

            fn open_mut<'__a>(
                rt: &'__a #at_runtime,
                place: &'__a mut <#at_runtime as ::acvus_extern::runtime::Runtime>::Value,
                at: ::acvus_extern::Settled<'__a>,
            ) -> Self::OpenMut<'__a>
            where
                Self: '__a,
            {
                <<#at_runtime as ::acvus_extern::runtime::Runtime>::Opaque as ::acvus_extern::runtime::OpaqueKind<#at_runtime>>::exclusive(rt, place, at)
            }

            fn open_ref<'__a>(
                rt: &'__a #at_runtime,
                reference: <#at_runtime as ::acvus_extern::runtime::Runtime>::Value,
                at: ::acvus_extern::Settled<'__a>,
            ) -> Self::Open<'__a>
            where
                Self: '__a,
            {
                <<#at_runtime as ::acvus_extern::runtime::Runtime>::Ref as ::acvus_extern::runtime::RefKind<#at_runtime>>::target(rt, reference, at)
            }

            fn open_ref_mut<'__a>(
                rt: &'__a #at_runtime,
                reference: <#at_runtime as ::acvus_extern::runtime::Runtime>::Value,
                at: ::acvus_extern::Settled<'__a>,
            ) -> Self::OpenMut<'__a>
            where
                Self: '__a,
            {
                <<#at_runtime as ::acvus_extern::runtime::Runtime>::Ref as ::acvus_extern::runtime::RefKind<#at_runtime>>::target_mut(rt, reference, at)
            }
        }

        impl<#(#declared_generics),*> ::acvus_extern::decl::DeclType for #name<#(#declared_args),*> {
            fn decl_ty(names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::DeclTy {
                let module: ::std::vec::Vec<_> =
                    ::core::module_path!().split("::").map(|segment| names.intern(segment)).collect();
                ::acvus_extern::decl::DeclTy::Extern {
                    name: ::acvus_extern::decl::QualifiedRef::qualified(names.namespace(&module), names.intern(#type_name)),
                    args: ::std::vec![#(<#indices as ::acvus_extern::decl::DeclType>::decl_ty(names)),*],
                    effect: #effect,
                    identity: #identity,
                    lifetimes: #lifetime_count,
                }
            }
        }

        impl<#(#declared_generics),*> ::acvus_extern::decl::Part for #name<#(#declared_args),*> {
            type Class = ::acvus_extern::decl::part::Value;
            type Of = Self;
        }

        impl<#(#declared_generics),*> ::acvus_extern::decl::ParamMode for #name<#(#declared_args),*> {
            type Mode = ::acvus_extern::mode::ByValue<Self>;

            fn decl(names: &::acvus_extern::decl::Interner) -> ::acvus_extern::decl::DeclTy {
                <Self as ::acvus_extern::decl::DeclType>::decl_ty(names)
            }
        }
    })
}

/// The type's arguments with each type parameter its uniform value at its own
/// index, `__X0`, `__X1`, .. in order, each lifetime and the runtime as
/// written, and the effect and the identity as `effect` and `identity` write
/// them.
fn at_indices(
    params: &[Param],
    runtime: &Ident,
    effect: impl Fn(&Ident) -> TokenStream,
    identity: impl Fn(&Ident) -> TokenStream,
) -> Vec<TokenStream> {
    let mut args = Vec::with_capacity(params.len());
    let mut next = 0usize;
    for param in params {
        let arg = match param {
            Param::Lifetime(l) => quote!(#l),
            Param::Runtime { ident, .. } => quote!(#ident),
            Param::Type(_) => {
                let index = format_ident!("__X{}", next);
                next += 1;
                quote!(::acvus_extern::Uniform<#runtime, #index>)
            }
            Param::Effect { ident, .. } => effect(ident),
            Param::Identity { ident, .. } => identity(ident),
        };
        args.push(arg);
    }
    args
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
                    Some(Kind::Unbounded) => Param::Type(ident),
                    Some(Kind::Runtime) => Param::Runtime { ident, bounds },
                    Some(Kind::Effect) => Param::Effect { ident, bounds },
                    Some(Kind::Identity) => Param::Identity { ident, bounds },
                    None => {
                        return Err(syn::Error::new(
                            t.span(),
                            "an opaque type's parameter is unbounded (a type, parametric), `Runtime`, \
                             `Var<kind::Effect>` or `Var<kind::Identity>`",
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
                    Param::Runtime { ident, .. } | Param::Effect { ident, .. } | Param::Identity { ident, .. }
                        if path.path.is_ident(ident)))));
            if !bounds_a_param {
                return Err(syn::Error::new(predicate.span(), "an opaque type's `where` clause bounds only its runtime, effect and identity"));
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
    Unbounded,
    Runtime,
    Effect,
    Identity,
}

/// What a type parameter's bounds say it is: none, one `Runtime`, or one
/// `Var` of the effect or the identity kind.
fn bound_kind(bounds: &[TypeParamBound]) -> Option<Kind> {
    let [bound] = bounds else {
        return bounds.is_empty().then_some(Kind::Unbounded);
    };
    let TypeParamBound::Trait(trait_bound) = bound else {
        return None;
    };
    let last = trait_bound.path.segments.last()?;
    if last.ident == "Runtime" && last.arguments.is_none() {
        return Some(Kind::Runtime);
    }
    if last.ident != "Var" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    let [syn::GenericArgument::Type(Type::Path(kind))] = args.args.iter().collect::<Vec<_>>().as_slice() else {
        return None;
    };
    match kind.path.segments.last()?.ident.to_string().as_str() {
        "Effect" => Some(Kind::Effect),
        "Identity" => Some(Kind::Identity),
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
            self.found |= self.params.iter().any(|p| matches!(p, Param::Type(i) if i == ident));
        }
    }
    let mut names = Names { params, found: false };
    names.visit_type(ty);
    names.found
}
