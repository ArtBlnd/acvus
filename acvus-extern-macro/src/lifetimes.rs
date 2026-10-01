//! The lifetime names a signature writes: for each end, the name its outermost
//! `&` is written at, where its type is one, and every other name written in
//! it; and the bounds written between names. This is all the attribute reads
//! of a signature's lifetimes. Where each name lies in a declared type is the
//! type's, read by the compiler (rfc_extern macros §4; acvus_extern::decl::labels).
//!
//! An elided lifetime is named here: in an argument, freshly, as Rust elides
//! it there; in the result, as the one lifetime the inputs name, as Rust does.
//! `'static` names no loan the checker tracks, and is refused.

use std::collections::BTreeSet;

use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{GenericParam, Lifetime, ReturnType, Signature, Type, TypeBareFn, TypeReference, WherePredicate};

/// The names one end writes.
pub struct Names {
    /// The name its outermost `&` is written at, where its type is a `&`.
    pub outer: Option<String>,
    /// Every other name written in it.
    pub inner: Vec<String>,
}

impl Names {
    pub fn all(&self) -> impl Iterator<Item = &String> {
        self.outer.iter().chain(&self.inner)
    }
}

/// A bound the signature writes, `longer: shorter`.
pub struct Bound {
    pub longer: String,
    pub shorter: String,
}

/// The names elided arguments are given, each its own.
pub struct Elided(u32);

impl Elided {
    pub fn new() -> Self {
        Elided(0)
    }

    fn next(&mut self) -> Option<String> {
        self.0 += 1;
        Some(format!("'__elided{}", self.0))
    }
}

/// The names a parameter's type writes, each elided one given its own.
pub fn param(ty: &Type, elided: &mut Elided) -> Names {
    names(ty, &mut || elided.next()).0
}

/// The names the result writes, each elided one the one lifetime the inputs
/// name (`inputs`, every parameter's), as Rust elides it. The result names
/// only what an argument (`arguments`) names: a capture and the context lend
/// it nothing.
pub fn result(output: &ReturnType, inputs: &[&Names], arguments: &[&Names]) -> syn::Result<Names> {
    let ReturnType::Type(_, ty) = output else {
        return Ok(Names { outer: None, inner: Vec::new() });
    };
    let named: BTreeSet<&String> = inputs.iter().flat_map(|names| names.all()).collect();
    let elision = match named.iter().collect::<Vec<_>>().as_slice() {
        [only] => Some((**only).clone()),
        _ => None,
    };
    let (result, unnamed) = names(ty, &mut || elision.clone());
    if unnamed {
        return Err(syn::Error::new(
            ty.span(),
            "a lifetime the result elides is the one lifetime its inputs name; name it where they name several",
        ));
    }
    let lent: BTreeSet<&String> = arguments.iter().flat_map(|names| names.all()).collect();
    if let Some(name) = result.all().find(|name| !lent.contains(name)) {
        return Err(syn::Error::new(
            ty.span(),
            format!("the result borrows only what an argument lends; {name} is a capture's or the context's alone"),
        ));
    }
    Ok(result)
}

/// The names `ty` writes, each elided one named by `elided`, and whether one
/// was elided that `elided` names none for.
fn names(ty: &Type, elided: &mut dyn FnMut() -> Option<String>) -> (Names, bool) {
    let mut inner = Inner { names: Vec::new(), elided, unnamed: false };
    let outer = match ty {
        Type::Reference(reference) => {
            let outer = inner.name(reference.lifetime.as_ref());
            inner.visit_type(&reference.elem);
            outer
        }
        _ => {
            inner.visit_type(ty);
            None
        }
    };
    let unnamed = inner.unnamed;
    (Names { outer, inner: inner.names }, unnamed)
}

struct Inner<'e> {
    names: Vec<String>,
    elided: &'e mut dyn FnMut() -> Option<String>,
    unnamed: bool,
}

impl Inner<'_> {
    fn name(&mut self, lifetime: Option<&Lifetime>) -> Option<String> {
        match lifetime {
            Some(lifetime) if lifetime.ident != "_" => Some(format!("'{}", lifetime.ident)),
            _ => {
                let name = (self.elided)();
                self.unnamed |= name.is_none();
                name
            }
        }
    }
}

impl<'a> Visit<'a> for Inner<'_> {
    fn visit_type_reference(&mut self, reference: &'a TypeReference) {
        if let Some(name) = self.name(reference.lifetime.as_ref()) {
            self.names.push(name);
        }
        self.visit_type(&reference.elem);
    }

    fn visit_lifetime(&mut self, lifetime: &'a Lifetime) {
        if let Some(name) = self.name(Some(lifetime)) {
            self.names.push(name);
        }
    }

    /// A function pointer's lifetimes are its own, bound by it.
    fn visit_type_bare_fn(&mut self, _: &'a TypeBareFn) {}
}

/// Refuses `'static` anywhere in the signature.
pub fn refuse_static(sig: &Signature) -> syn::Result<()> {
    struct Static(Option<Lifetime>);
    impl<'a> Visit<'a> for Static {
        fn visit_lifetime(&mut self, lifetime: &'a Lifetime) {
            if lifetime.ident == "static" && self.0.is_none() {
                self.0 = Some(lifetime.clone());
            }
        }
    }
    let mut found = Static(None);
    found.visit_signature(sig);
    match found.0 {
        Some(lifetime) => Err(syn::Error::new(
            lifetime.span(),
            "an extern fn names no `'static`: it is no loan a program's call can be tracked by",
        )),
        None => Ok(()),
    }
}

/// The bounds written between lifetimes: on a lifetime parameter, and in the
/// `where` clause.
pub fn written(sig: &Signature) -> Vec<Bound> {
    let mut bounds = Vec::new();
    for param in &sig.generics.params {
        if let GenericParam::Lifetime(param) = param {
            for shorter in &param.bounds {
                bounds.push(Bound { longer: format!("'{}", param.lifetime.ident), shorter: format!("'{}", shorter.ident) });
            }
        }
    }
    if let Some(clause) = &sig.generics.where_clause {
        for predicate in &clause.predicates {
            if let WherePredicate::Lifetime(predicate) = predicate {
                for shorter in &predicate.bounds {
                    bounds.push(Bound {
                        longer: format!("'{}", predicate.lifetime.ident),
                        shorter: format!("'{}", shorter.ident),
                    });
                }
            }
        }
    }
    bounds
}
