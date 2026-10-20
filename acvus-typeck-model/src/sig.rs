//! The types a declaration writes, over its own variables: its type
//! parameters and, where it names one, its identity (extern fn §1, §2).
//! A signature is a declaration's parameter types and result type.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ty::{Constructor, Name, PartKey, Term, Ty, TyError};

#[cfg(test)]
mod tests;

/// An extern-defined type's declaration: how many arguments it takes, and
/// whether it carries an identity (l.55).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExternDecl {
    pub arity: usize,
    pub identity: bool,
}

/// A module's struct: its fields, in the order it declares them (l.54).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StructDecl {
    pub fields: Vec<Name>,
}

/// The extern-defined types and the structs, each by its name.
#[derive(Clone, Default, Debug)]
pub struct DeclaredTypes {
    externs: BTreeMap<Name, ExternDecl>,
    structs: BTreeMap<Name, StructDecl>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeclError {
    Redeclared(Name),
    Undeclared(Constructor),
    Arity {
        constructor: Constructor,
        declared: usize,
        written: usize,
    },
    Identity {
        name: Name,
        declared: bool,
    },
}

impl DeclaredTypes {
    pub fn declare_extern(&mut self, name: Name, decl: ExternDecl) -> Result<(), DeclError> {
        if self.externs.contains_key(&name) {
            return Err(DeclError::Redeclared(name));
        }
        self.externs.insert(name, decl);
        Ok(())
    }

    pub fn declare_struct(&mut self, module: Name, decl: StructDecl) -> Result<(), DeclError> {
        if self.structs.contains_key(&module) {
            return Err(DeclError::Redeclared(module));
        }
        self.structs.insert(module, decl);
        Ok(())
    }

    pub fn extern_decl(&self, name: &Name) -> Option<&ExternDecl> {
        self.externs.get(name)
    }

    pub fn struct_decl(&self, module: &Name) -> Option<&StructDecl> {
        self.structs.get(module)
    }

    /// Whether the term's own constructor agrees with its declaration. Its parts
    /// are not looked at, so closed types, signatures and kinds share this check.
    pub fn check<P>(&self, term: &Term<P>) -> Result<(), DeclError> {
        match term {
            Term::Extern {
                name,
                args,
                identity,
            } => {
                let constructor = term.constructor();
                let decl = self
                    .externs
                    .get(name)
                    .ok_or_else(|| DeclError::Undeclared(constructor.clone()))?;
                if args.len() != decl.arity {
                    return Err(DeclError::Arity {
                        constructor,
                        declared: decl.arity,
                        written: args.len(),
                    });
                }
                if identity.is_some() != decl.identity {
                    return Err(DeclError::Identity {
                        name: name.clone(),
                        declared: decl.identity,
                    });
                }
                Ok(())
            }
            Term::Struct(module, args) => {
                let constructor = term.constructor();
                let decl = self
                    .structs
                    .get(module)
                    .ok_or_else(|| DeclError::Undeclared(constructor.clone()))?;
                if args.len() != decl.fields.len() {
                    return Err(DeclError::Arity {
                        constructor,
                        declared: decl.fields.len(),
                        written: args.len(),
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// `check` at every term of `ty`.
    pub fn check_ty(&self, ty: &Ty) -> Result<(), DeclError> {
        self.check(ty.term())?;
        for (_, part) in ty.term().parts() {
            self.check_ty(part)?;
        }
        Ok(())
    }
}

/// A variable of a signature.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SigVar {
    Type(usize),
    Identity,
}

/// A type a declaration writes.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SigTy {
    Var(SigVar),
    Term(Rc<Term<SigTy>>),
}

/// Closed types for some of a signature's variables.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SigArgs {
    pub types: Vec<Option<Ty>>,
    pub identity: Option<Ty>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SubstError {
    Unbound(SigVar),
    Ty(TyError),
}

impl SigArgs {
    pub fn get(&self, var: SigVar) -> Option<&Ty> {
        match var {
            SigVar::Type(i) => self.types.get(i).and_then(Option::as_ref),
            SigVar::Identity => self.identity.as_ref(),
        }
    }

    fn bind(&mut self, var: SigVar, ty: Ty) {
        match var {
            SigVar::Type(i) => self.types[i] = Some(ty),
            SigVar::Identity => self.identity = Some(ty),
        }
    }
}

impl SigTy {
    pub fn term(term: Term<SigTy>) -> SigTy {
        SigTy::Term(Rc::new(term))
    }

    pub fn mentions(&self, var: SigVar) -> bool {
        match self {
            SigTy::Var(own) => *own == var,
            SigTy::Term(term) => term.parts().into_iter().any(|(_, part)| part.mentions(var)),
        }
    }

    /// Each bound variable's closed type put in its place. Where that makes a
    /// reference's target a reference, the two are one reference to the inner
    /// target, exclusive only where both are (l.64).
    pub fn substitute(&self, args: &SigArgs) -> Result<Ty, SubstError> {
        match self {
            SigTy::Var(var) => args.get(*var).cloned().ok_or(SubstError::Unbound(*var)),
            SigTy::Term(term) => {
                let term = (**term).clone().try_map(|_, part| part.substitute(args))?;
                let term = match term {
                    Term::Ref(outer, target) => match target.term() {
                        Term::Ref(inner, inner_target) => {
                            Term::Ref(outer.min(*inner), inner_target.clone())
                        }
                        _ => Term::Ref(outer, target),
                    },
                    term => term,
                };
                Ty::new(term).map_err(SubstError::Ty)
            }
        }
    }

    /// Extends `args` so that `self` substituted by them is `ty`, and returns
    /// whether it could; where it could not, `args` is left as it was. A
    /// variable is bound to the closed type at its place.
    pub fn match_ty(&self, ty: &Ty, args: &mut SigArgs) -> bool {
        let mut found = args.clone();
        let matched = self.match_into(ty, &mut found);
        if matched {
            *args = found;
        }
        matched
    }

    fn match_into(&self, ty: &Ty, found: &mut SigArgs) -> bool {
        match self {
            SigTy::Var(var) => match found.get(*var) {
                Some(bound) => bound == ty,
                None => {
                    found.bind(*var, ty.clone());
                    true
                }
            },
            SigTy::Term(term) => {
                if term.constructor() != ty.term().constructor()
                    || term.members() != ty.term().members()
                {
                    return false;
                }
                term.parts()
                    .into_iter()
                    .zip(ty.term().parts())
                    .all(|((_, written), (_, part))| written.match_into(part, found))
            }
        }
    }
}

/// A declaration's parameter types and result type, over its type parameters
/// and, where it names one, its identity.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Signature {
    type_params: Vec<Name>,
    names_identity: bool,
    params: Vec<SigTy>,
    result: SigTy,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SigError {
    Decl(DeclError),
    VarOutOfRange(usize),
    /// The identity variable written where the signature names none.
    IdentityUnnamed,
    /// An identity anywhere but an extern-defined type's identity part, that
    /// part holding anything but the identity variable, or an identity a
    /// declaration makes its own. An identity is made only where a call or a
    /// body lies (l.354).
    IdentityPlace,
    /// A reference whose target is a reference (l.64).
    RefTarget,
}

impl Signature {
    pub fn new(
        type_params: Vec<Name>,
        names_identity: bool,
        params: Vec<SigTy>,
        result: SigTy,
        types: &DeclaredTypes,
    ) -> Result<Signature, SigError> {
        let signature = Signature {
            type_params,
            names_identity,
            params,
            result,
        };
        for ty in signature.params.iter().chain([&signature.result]) {
            signature.check(ty, types, false)?;
        }
        Ok(signature)
    }

    fn check(&self, ty: &SigTy, types: &DeclaredTypes, at_identity: bool) -> Result<(), SigError> {
        match ty {
            SigTy::Var(SigVar::Type(i)) => {
                if at_identity {
                    Err(SigError::IdentityPlace)
                } else if *i >= self.type_params.len() {
                    Err(SigError::VarOutOfRange(*i))
                } else {
                    Ok(())
                }
            }
            SigTy::Var(SigVar::Identity) => {
                if !self.names_identity {
                    Err(SigError::IdentityUnnamed)
                } else if !at_identity {
                    Err(SigError::IdentityPlace)
                } else {
                    Ok(())
                }
            }
            SigTy::Term(term) => {
                if at_identity || matches!(**term, Term::Identity(_)) {
                    return Err(SigError::IdentityPlace);
                }
                if let Term::Ref(_, SigTy::Term(target)) = &**term
                    && matches!(**target, Term::Ref(..))
                {
                    return Err(SigError::RefTarget);
                }
                types.check(term).map_err(SigError::Decl)?;
                for (key, part) in term.parts() {
                    self.check(part, types, key == PartKey::Identity)?;
                }
                Ok(())
            }
        }
    }

    pub fn type_params(&self) -> &[Name] {
        &self.type_params
    }

    pub fn names_identity(&self) -> bool {
        self.names_identity
    }

    pub fn params(&self) -> &[SigTy] {
        &self.params
    }

    pub fn result(&self) -> &SigTy {
        &self.result
    }

    /// The identity lies in the result and in no parameter's type, so a call
    /// makes it anew (l.354).
    pub fn existential_identity(&self) -> bool {
        self.names_identity
            && self.result.mentions(SigVar::Identity)
            && !self
                .params
                .iter()
                .any(|param| param.mentions(SigVar::Identity))
    }

    /// Bindings with no variable bound.
    pub fn unbound(&self) -> SigArgs {
        SigArgs {
            types: vec![None; self.type_params.len()],
            identity: None,
        }
    }
}
