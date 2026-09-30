//! What the attribute reads off the function's signature: its generic
//! parameters by kind, its parameters by role, and how its result is made.
//! How a type crosses is never read here (macros §4); only where the
//! signature's own syntax is the fact: which parameter is a capture or the
//! context, which generic parameter is the runtime or a variable of which
//! kind, and which lent parameter a borrowed result's lifetime names.

use proc_macro2::Span;
use quote::quote;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{
    FnArg, GenericArgument, GenericParam, Ident, ItemFn, Lifetime, Pat, PathArguments, ReturnType, Signature, Type,
    TypeParamBound, TypeReference, WherePredicate,
};

/// The kinds a generic parameter may be (compiler: extern fn §1-§2).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Type,
    Effect,
    Identity,
}

impl Kind {
    fn index(self) -> usize {
        match self {
            Kind::Type => 0,
            Kind::Effect => 1,
            Kind::Identity => 2,
        }
    }
}

/// A generic type parameter: the runtime, or a variable of a kind and its
/// position among that kind's.
pub enum Generic {
    Runtime { ident: Ident, bounds: Vec<TypeParamBound> },
    Var { ident: Ident, kind: Kind, index: u32 },
}

/// What a parameter is to the call.
pub enum Role {
    /// An argument position, opened by its mode.
    Argument,
    /// A capture the marker holds, lent to every call: `#[capture] name: &T`
    /// holds a `T`.
    Capture { held: Box<Type> },
    /// The body's context: `#[context] cx: &Returning<'_, R>`, or
    /// `&Awaited<'_, R>` for an awaited body.
    Context,
}

pub struct Param {
    pub name: Ident,
    pub ty: Type,
    pub role: Role,
}

/// How the result is made.
pub enum Output {
    /// A value that crosses.
    Value(Box<Type>),
    /// A value whose every leaf is a reference to a part of what the argument
    /// at `from` lent, shared or exclusively as `mutable` says (crossing §4).
    Borrowed { from: usize, mutable: bool },
    /// No value: the call never returns.
    Never,
}

pub struct Sig {
    /// The function's name.
    pub ident: Ident,
    /// The result type as written, `()` where none is.
    pub written_output: Type,
    pub generics: Vec<Generic>,
    /// How many variables of each kind: types, effects, identities.
    pub counts: [u32; 3],
    pub params: Vec<Param>,
    pub output: Output,
    pub awaited: bool,
}

impl Sig {
    /// Reads `func`'s signature, and strips the attribute's own marks from
    /// its parameters so the function compiles as written.
    pub fn read(func: &mut ItemFn) -> syn::Result<Sig> {
        let sig = &mut func.sig;
        refuse_unsupported(sig)?;
        let (generics, counts) = generics(sig)?;
        let params = params(sig)?;
        let output = output(&sig.output, &params)?;
        let written_output = match &sig.output {
            ReturnType::Type(_, ty) => (**ty).clone(),
            ReturnType::Default => syn::parse_quote!(()),
        };
        let awaited = sig.asyncness.is_some();
        if matches!(output, Output::Borrowed { .. }) && params.iter().any(|p| matches!(p.role, Role::Context)) {
            return Err(syn::Error::new(
                sig.output.span(),
                "a body finishing with a borrow takes no context: finishing it consumes the context",
            ));
        }
        if awaited && matches!(output, Output::Borrowed { .. }) {
            return Err(syn::Error::new(
                sig.output.span(),
                "an awaited body's result borrows nothing: it moves its arguments (crossing §3-§4)",
            ));
        }
        Ok(Sig { ident: sig.ident.clone(), written_output, generics, counts, params, output, awaited })
    }

    /// `ty` as the dispatch and the declaration name it: the runtime as
    /// `runtime`, each variable as its stand-in, each lifetime inferred.
    pub fn at_stand_ins(&self, ty: &Type, runtime: &Ident) -> Type {
        let mut ty = ty.clone();
        StandIns { generics: &self.generics, runtime }.visit_type_mut(&mut ty);
        ty
    }

    /// The function's type arguments at the stand-ins, in order, for a
    /// turbofish; none where it has no generic type parameter.
    /// Whether `ident` is one of the function's effect variables.
    pub fn names_effect_variable(&self, ident: &Ident) -> bool {
        self.generics
            .iter()
            .any(|g| matches!(g, Generic::Var { ident: v, kind: Kind::Effect, .. } if v == ident))
    }

    /// The bounds the function states of its runtime, which its dispatch is
    /// implemented under.
    pub fn runtime_bounds(&self) -> Vec<TypeParamBound> {
        self.generics
            .iter()
            .find_map(|generic| match generic {
                Generic::Runtime { bounds, .. } => Some(bounds.clone()),
                Generic::Var { .. } => None,
            })
            .unwrap_or_default()
    }

    pub fn type_args(&self, runtime: &Ident) -> Option<proc_macro2::TokenStream> {
        if self.generics.is_empty() {
            return None;
        }
        let args = self.generics.iter().map(|generic| match generic {
            Generic::Runtime { .. } => quote!(#runtime),
            Generic::Var { kind, index, .. } => stand_in(*kind, *index, runtime),
        });
        Some(quote!(::<#(#args),*>))
    }
}

/// A variable's stand-in: a type variable is its uniform value at its index.
fn stand_in(kind: Kind, index: u32, runtime: &Ident) -> proc_macro2::TokenStream {
    match kind {
        Kind::Type => quote!(::acvus_extern::Uniform<#runtime, ::acvus_extern::decl::Nth<#index>>),
        Kind::Effect => quote!(::acvus_extern::decl::EffectNth<#index>),
        Kind::Identity => quote!(::acvus_extern::decl::IdentityNth<#index>),
    }
}

fn refuse_unsupported(sig: &Signature) -> syn::Result<()> {
    if let Some(variadic) = &sig.variadic {
        return Err(syn::Error::new(variadic.span(), "an extern fn is not variadic"));
    }
    if let Some(unsafety) = &sig.unsafety {
        return Err(syn::Error::new(unsafety.span(), "an extern fn's body is safe code"));
    }
    if let Some(constness) = &sig.constness {
        return Err(syn::Error::new(constness.span(), "an extern fn is not `const`"));
    }
    if let Some(abi) = &sig.abi {
        return Err(syn::Error::new(abi.span(), "an extern fn has Rust's ABI"));
    }
    Ok(())
}

/// Each generic type parameter is a variable, bounded by one `Var<kind::K>`
/// and nothing else, or the runtime: the one parameter that is not a
/// variable, whose bounds (`Runtime`, `CallsBack`, ..) the dispatch keeps. A
/// lifetime parameter is the signature's own; a const parameter is refused.
fn generics(sig: &Signature) -> syn::Result<(Vec<Generic>, [u32; 3])> {
    let mut generics = Vec::new();
    let mut counts = [0u32; 3];
    let mut runtime = false;
    for param in &sig.generics.params {
        let ty = match param {
            GenericParam::Lifetime(_) => continue,
            GenericParam::Const(c) => {
                return Err(syn::Error::new(c.span(), "an extern fn's generic parameters are types"));
            }
            GenericParam::Type(ty) => ty,
        };
        let mut bounds: Vec<TypeParamBound> = ty.bounds.iter().cloned().collect();
        if let Some(clause) = &sig.generics.where_clause {
            for predicate in &clause.predicates {
                if let WherePredicate::Type(p) = predicate
                    && matches!(&p.bounded_ty, Type::Path(path) if path.path.is_ident(&ty.ident))
                {
                    bounds.extend(p.bounds.iter().cloned());
                }
            }
        }
        let kinds: Vec<Option<Kind>> = bounds.iter().map(var_kind).collect();
        match kinds.as_slice() {
            [Some(kind)] => {
                let index = counts[kind.index()];
                counts[kind.index()] += 1;
                generics.push(Generic::Var { ident: ty.ident.clone(), kind: *kind, index });
            }
            kinds if kinds.iter().all(Option::is_none) => {
                if std::mem::replace(&mut runtime, true) {
                    return Err(syn::Error::new(
                        ty.ident.span(),
                        "an extern fn names one runtime; every other generic parameter is a `Var<kind::..>`",
                    ));
                }
                generics.push(Generic::Runtime { ident: ty.ident.clone(), bounds });
            }
            _ => {
                return Err(syn::Error::new(
                    ty.ident.span(),
                    "a variable is bounded by one `Var<kind::..>` and nothing else",
                ));
            }
        }
    }
    Ok((generics, counts))
}

/// The kind `K` of a `Var<kind::K>` bound; none for any other bound.
fn var_kind(bound: &TypeParamBound) -> Option<Kind> {
    let TypeParamBound::Trait(tr) = bound else {
        return None;
    };
    let last = tr.path.segments.last()?;
    if last.ident != "Var" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    let [GenericArgument::Type(Type::Path(kind))] = args.args.iter().collect::<Vec<_>>().as_slice() else {
        return None;
    };
    match kind.path.segments.last()?.ident.to_string().as_str() {
        "Type" => Some(Kind::Type),
        "Effect" => Some(Kind::Effect),
        "Identity" => Some(Kind::Identity),
        _ => None,
    }
}

fn params(sig: &mut Signature) -> syn::Result<Vec<Param>> {
    let mut params = Vec::new();
    for input in &mut sig.inputs {
        let FnArg::Typed(typed) = input else {
            return Err(syn::Error::new(input.span(), "an extern fn takes no `self`"));
        };
        let Pat::Ident(pat) = &*typed.pat else {
            return Err(syn::Error::new(typed.pat.span(), "an extern fn's parameter is a name"));
        };
        let name = pat.ident.clone();
        let mut role = Role::Argument;
        let mut marked = None;
        typed.attrs.retain(|attr| {
            let which = if attr.path().is_ident("capture") {
                Some("capture")
            } else if attr.path().is_ident("context") {
                Some("context")
            } else {
                None
            };
            match which {
                Some(which) => {
                    marked = Some((which, attr.span(), marked.is_some()));
                    false
                }
                None => true,
            }
        });
        if let Some((which, span, twice)) = marked {
            if twice {
                return Err(syn::Error::new(span, "a parameter is a capture or the context, not both"));
            }
            role = match which {
                "capture" => {
                    let Type::Reference(TypeReference { mutability: None, elem, .. }) = &*typed.ty else {
                        return Err(syn::Error::new(
                            typed.ty.span(),
                            "a capture is lent to every call as `&T`; the marker holds the `T`",
                        ));
                    };
                    let mut held = (**elem).clone();
                    Statics.visit_type_mut(&mut held);
                    Role::Capture { held: Box::new(held) }
                }
                _ => Role::Context,
            };
        }
        params.push(Param { name, ty: (*typed.ty).clone(), role });
    }
    if params.iter().filter(|p| matches!(p.role, Role::Context)).count() > 1 {
        return Err(syn::Error::new(sig.inputs.span(), "an extern fn takes its context once"));
    }
    Ok(params)
}

fn output(output: &ReturnType, params: &[Param]) -> syn::Result<Output> {
    let ReturnType::Type(_, ty) = output else {
        return Ok(Output::Value(Box::new(syn::parse_quote!(()))));
    };
    if let Type::Never(_) = &**ty {
        return Ok(Output::Never);
    }
    if !borrows(ty) {
        return Ok(Output::Value(ty.clone()));
    }
    let refused = || {
        syn::Error::new(
            ty.span(),
            "a result that borrows borrows only from the one lent parameter its lifetime names, \
             every reference in it at that lifetime (crossing §4)",
        )
    };
    let lent: Vec<(usize, &TypeReference)> = params
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.role, Role::Argument))
        .filter_map(|(i, p)| match &p.ty {
            Type::Reference(r) => Some((i, r)),
            _ => None,
        })
        .collect();
    let lifetimes = result_lifetimes(ty);
    let from: Vec<&(usize, &TypeReference)> = match lifetimes.as_slice() {
        [] => lent.iter().collect(),
        [only] => lent.iter().filter(|(_, r)| r.lifetime.as_ref() == Some(only)).collect(),
        _ => return Err(refused()),
    };
    let [(from, lender)] = from.as_slice() else {
        return Err(refused());
    };
    Ok(Output::Borrowed { from: *from, mutable: lender.mutability.is_some() })
}

/// The lifetimes a result's references are written at, each once; an
/// elided one is none. A result mixing an elided reference and a named one is
/// refused by leaving two entries.
fn result_lifetimes(ty: &Type) -> Vec<Lifetime> {
    struct Lifetimes(Vec<Lifetime>, bool);
    impl<'a> Visit<'a> for Lifetimes {
        fn visit_type_reference(&mut self, r: &'a TypeReference) {
            match &r.lifetime {
                Some(l) if l.ident != "_" => {
                    if !self.0.contains(l) {
                        self.0.push(l.clone());
                    }
                }
                _ => self.1 = true,
            }
            syn::visit::visit_type(self, &r.elem);
        }
        fn visit_lifetime(&mut self, l: &'a Lifetime) {
            if !self.0.contains(l) {
                self.0.push(l.clone());
            }
        }
    }
    let mut found = Lifetimes(Vec::new(), false);
    found.visit_type(ty);
    if found.1 && !found.0.is_empty() {
        found.0.push(Lifetime::new("'_", Span::call_site()));
    }
    found.0
}

/// Whether `ty` holds a reference or names a lifetime.
fn borrows(ty: &Type) -> bool {
    struct Finds(bool);
    impl<'a> Visit<'a> for Finds {
        fn visit_type_reference(&mut self, _: &'a TypeReference) {
            self.0 = true;
        }
        fn visit_lifetime(&mut self, _: &'a Lifetime) {
            self.0 = true;
        }
    }
    let mut finds = Finds(false);
    finds.visit_type(ty);
    finds.0
}

/// The runtime named `runtime`, each variable its stand-in, each lifetime
/// inferred.
struct StandIns<'g> {
    generics: &'g [Generic],
    runtime: &'g Ident,
}

impl VisitMut for StandIns<'_> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        if let Type::Path(path) = ty
            && path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
        {
            for generic in self.generics {
                match generic {
                    Generic::Runtime { ident: r, .. } if r == ident => {
                        let runtime = self.runtime;
                        *ty = syn::parse_quote!(#runtime);
                        return;
                    }
                    Generic::Var { ident: v, kind, index } if v == ident => {
                        let stand_in = stand_in(*kind, *index, self.runtime);
                        *ty = syn::parse_quote!(#stand_in);
                        return;
                    }
                    _ => {}
                }
            }
        }
        syn::visit_mut::visit_type_mut(self, ty);
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        *lifetime = Lifetime::new("'_", lifetime.span());
    }
}

/// Every lifetime `'static`: a held capture's type names none of the
/// signature's.
struct Statics;

impl VisitMut for Statics {
    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        *lifetime = Lifetime::new("'static", Span::call_site());
    }
}
