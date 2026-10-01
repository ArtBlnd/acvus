//! What the attribute reads off the function's signature: its generic
//! parameters by kind, its parameters by role, and how its result is made.
//! How a type crosses is never read here (macros §4); only where the
//! signature's own syntax is the fact: which parameter is a capture or the
//! executor, which generic parameter is the runtime or a variable of which
//! kind, and the lifetime names each end writes (`lifetimes`).

use proc_macro2::Span;
use quote::quote;
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{
    FnArg, GenericArgument, GenericParam, Ident, ItemFn, Lifetime, Pat, PathArguments, ReturnType, Signature, Type,
    TypeParamBound, TypeReference, WherePredicate,
};

use crate::facts;
use crate::lifetimes::{self, Bound, Names};

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
    /// The body's executor: `#[executor] cx: &Executor<'_, R>`, for either
    /// body.
    Executor,
}

pub struct Param {
    pub name: Ident,
    pub ty: Type,
    pub role: Role,
    /// The lifetime names its type writes.
    pub names: Names,
    /// The facts written on it about its place (compiler: extern fn §3).
    pub place: facts::PlaceFacts,
}

/// How the result is made.
pub enum Output {
    /// A value that crosses: the result as written.
    Value,
    /// A value whose every leaf is a reference, each holding a loan the result's
    /// lifetimes name (runtime contract §2, References).
    Borrowed,
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
    /// The lifetime names the result writes.
    pub result_names: Names,
    /// The bounds written between lifetimes.
    pub written: Vec<Bound>,
    pub awaited: bool,
}

impl Sig {
    /// Reads `func`'s signature, and strips the attribute's own marks from
    /// its parameters so the function compiles as written.
    pub fn read(func: &mut ItemFn) -> syn::Result<Sig> {
        let sig = &mut func.sig;
        refuse_unsupported(sig)?;
        let (generics, counts) = generics(sig)?;
        lifetimes::refuse_static(sig)?;
        let params = params(sig)?;
        let inputs: Vec<&Names> = params.iter().map(|p| &p.names).collect();
        let arguments: Vec<&Names> =
            params.iter().filter(|p| matches!(p.role, Role::Argument)).map(|p| &p.names).collect();
        let result_names = lifetimes::result(&sig.output, &inputs, &arguments)?;
        let written = lifetimes::written(sig);
        let output = output(&sig.output);
        let written_output = match &sig.output {
            ReturnType::Type(_, ty) => (**ty).clone(),
            ReturnType::Default => syn::parse_quote!(()),
        };
        let awaited = sig.asyncness.is_some();
        if awaited && matches!(output, Output::Borrowed) {
            return Err(syn::Error::new(
                sig.output.span(),
                "an awaited body's result borrows nothing: it moves its arguments (crossing §3-§4)",
            ));
        }
        Ok(Sig {
            ident: sig.ident.clone(),
            written_output,
            generics,
            counts,
            params,
            output,
            result_names,
            written,
            awaited,
        })
    }

    /// `ty` as the dispatch and the declaration name it: the runtime as
    /// `runtime`, each variable as its stand-in, each lifetime inferred.
    pub fn at_stand_ins(&self, ty: &Type, runtime: &Ident) -> Type {
        let mut ty = ty.clone();
        StandIns { generics: &self.generics, runtime }.visit_type_mut(&mut ty);
        ty
    }

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

    /// The function's type arguments at the stand-ins, in order, for a
    /// turbofish; none where it has no generic type parameter.
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
    let mut elided = lifetimes::Elided::new();
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
        // The facts written on the parameter about its place, read once its
        // role is known.
        let mut place_facts = Vec::new();
        typed.attrs.retain(|attr| {
            if ["touch", "gives", "update", "inverse_of"].iter().any(|fact| attr.path().is_ident(fact)) {
                place_facts.push(attr.clone());
                return false;
            }
            let which = if attr.path().is_ident("capture") {
                Some("capture")
            } else if attr.path().is_ident("executor") {
                Some("executor")
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
                return Err(syn::Error::new(span, "a parameter is a capture or the executor, not both"));
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
                _ => Role::Executor,
            };
        }
        if let (Some(attr), false) = (place_facts.first(), matches!(role, Role::Argument)) {
            return Err(syn::Error::new(attr.span(), "a fact of a place is of an argument's; a capture and the executor are none"));
        }
        let place = facts::PlaceFacts::read(&place_facts)?;
        let names = lifetimes::param(&typed.ty, &mut elided);
        params.push(Param { name, ty: (*typed.ty).clone(), role, names, place });
    }
    if params.iter().filter(|p| matches!(p.role, Role::Executor)).count() > 1 {
        return Err(syn::Error::new(sig.inputs.span(), "an extern fn takes its executor once"));
    }
    Ok(params)
}

fn output(output: &ReturnType) -> Output {
    let ReturnType::Type(_, ty) = output else {
        return Output::Value;
    };
    if let Type::Never(_) = &**ty {
        return Output::Never;
    }
    match holds_reference(ty) {
        true => Output::Borrowed,
        false => Output::Value,
    }
}

/// Whether `ty` holds a reference, a leaf the finish makes a reference value
/// of. A result holding what a carrier holds and no reference is a value.
fn holds_reference(ty: &Type) -> bool {
    struct Finds(bool);
    impl<'a> syn::visit::Visit<'a> for Finds {
        fn visit_type_reference(&mut self, _: &'a TypeReference) {
            self.0 = true;
        }
    }
    let mut finds = Finds(false);
    syn::visit::Visit::visit_type(&mut finds, ty);
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
