//! What the attribute reads off the function's signature: its generic
//! parameters by kind, its parameters by role, and how its result is made.
//! How a type crosses is never read here (macros §5); only where the
//! signature's own syntax is the fact: which parameter is a capture or the
//! executor, which generic parameter is the runtime or a variable of which
//! kind, and the lifetime names each end writes (`lifetimes`).

use proc_macro2::Span;
use quote::quote;
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{
    FnArg, GenericParam, Ident, Lifetime, Pat, PathArguments, ReturnType, Signature, Type,
    TypeParamBound, TypeReference, WherePredicate,
};

use crate::facts;
use crate::lifetimes::{self, Names};

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
    /// An argument position, opened at its Rust type.
    Argument,
    /// A capture the marker holds, lent to every call: `#[capture] name: &T`
    /// holds a `T`.
    Capture { held: Box<Type> },
    /// The body's executor: `#[executor] cx: &Executor<'_, R>`, for either
    /// body.
    Executor,
    /// A requirement held as a handle at a type: `#[requires] name:
    /// &InstanceOf<R, g, Ts>`, opened from the impl its site chose (impls §3).
    Requires,
    /// A requirement owned with its receiver: `#[instance] name:
    /// Instance<R, g, Ts>`, an argument, the receiver, opened with the impl
    /// its site chose (impls §3).
    Instance,
}

impl Role {
    /// Whether the parameter takes an argument of the call.
    pub fn takes_argument(&self) -> bool {
        matches!(self, Role::Argument | Role::Instance)
    }

    /// Whether the parameter takes an impl its call's site chose.
    pub fn takes_impl(&self) -> bool {
        matches!(self, Role::Requires | Role::Instance)
    }
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
    /// A value that crosses, at the form written: one that borrows holds a
    /// loan its lifetimes name (crossing §4).
    Value,
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
    pub awaited: bool,
}

impl Sig {
    /// Reads a function's signature, and strips the attribute's own marks
    /// from its parameters so the function compiles as written.
    pub fn read(sig: &mut Signature) -> syn::Result<Sig> {
        refuse_unsupported(sig)?;
        let (generics, counts) = generics(sig)?;
        lifetimes::refuse_static(sig)?;
        let params = params(sig)?;
        let inputs: Vec<&Names> = params.iter().map(|p| &p.names).collect();
        let arguments: Vec<&Names> =
            params.iter().filter(|p| p.role.takes_argument()).map(|p| &p.names).collect();
        let result_names = lifetimes::result(&sig.output, &inputs, &arguments)?;
        lifetimes::refuse_bounds(sig)?;
        let output = output(&sig.output);
        let written_output = match &sig.output {
            ReturnType::Type(_, ty) => (**ty).clone(),
            ReturnType::Default => syn::parse_quote!(()),
        };
        let awaited = sig.asyncness.is_some();
        Ok(Sig {
            ident: sig.ident.clone(),
            written_output,
            generics,
            counts,
            params,
            output,
            result_names,
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

/// Each generic type parameter is a variable, bounded by one of
/// `TypeVar<R>`, `EffectVar` and `IdentityVar` and nothing else, or the runtime: the one parameter that is not a
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
                        "an extern fn names one runtime; every other generic parameter is a `TypeVar<R>`, an `EffectVar` or an `IdentityVar`",
                    ));
                }
                generics.push(Generic::Runtime { ident: ty.ident.clone(), bounds });
            }
            _ => {
                return Err(syn::Error::new(
                    ty.ident.span(),
                    "a variable is bounded by one `TypeVar<R>`, `EffectVar` or `IdentityVar` and nothing else",
                ));
            }
        }
    }
    Ok((generics, counts))
}

/// The kind a variable's bound says: `TypeVar<R>`, `EffectVar` or
/// `IdentityVar`; none for any other bound.
fn var_kind(bound: &TypeParamBound) -> Option<Kind> {
    let TypeParamBound::Trait(tr) = bound else {
        return None;
    };
    let last = tr.path.segments.last()?;
    match (last.ident.to_string().as_str(), &last.arguments) {
        ("TypeVar", PathArguments::AngleBracketed(_)) => Some(Kind::Type),
        ("EffectVar", PathArguments::None) => Some(Kind::Effect),
        ("IdentityVar", PathArguments::None) => Some(Kind::Identity),
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
            let which = ["capture", "executor", "requires", "instance"].into_iter().find(|role| attr.path().is_ident(role));
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
                return Err(syn::Error::new(span, "a parameter has one role: a capture, the executor, `requires` or `instance`"));
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
                "requires" => Role::Requires,
                "instance" => Role::Instance,
                _ => Role::Executor,
            };
        }
        if let (Some(attr), false) = (place_facts.first(), matches!(role, Role::Argument)) {
            return Err(syn::Error::new(
                attr.span(),
                "a fact of a place is of an argument's; a capture, the executor and a requirement are none",
            ));
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
    match output {
        ReturnType::Type(_, ty) if matches!(&**ty, Type::Never(_)) => Output::Never,
        _ => Output::Value,
    }
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

/// `ty` with every lifetime, written or elided in a reference, `lifetime`:
/// a type read for its key at one loan (crossing §2).
pub fn at_lifetime(ty: &Type, lifetime: &Lifetime) -> Type {
    let mut ty = ty.clone();
    AtLifetime(lifetime.clone()).visit_type_mut(&mut ty);
    ty
}

struct AtLifetime(Lifetime);

impl VisitMut for AtLifetime {
    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        *lifetime = self.0.clone();
    }

    fn visit_type_reference_mut(&mut self, reference: &mut TypeReference) {
        reference.lifetime.get_or_insert_with(|| self.0.clone());
        syn::visit_mut::visit_type_reference_mut(self, reference);
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
