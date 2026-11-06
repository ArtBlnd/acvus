//! The types a declaration writes, over its own variables: its type
//! parameters and, where it names one, its identity.
//! A signature is a declaration's parameter types and result type.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ty::{Constructor, HasTerm, Name, PartKey, Term, Ty, TyError};

/// An extern-defined type's declaration: how many arguments it takes, and
/// whether it carries an identity.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExternDecl {
    pub arity: usize,
    pub identity: bool,
}

/// A module's struct: its fields, in the order it declares them.
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
    /// are not looked at, so closed types, signatures and structures share this check.
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

/// Types for some of a signature's variables: closed types, or values a use
/// reads.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SigArgs<T = Ty> {
    pub types: Vec<Option<T>>,
    pub identity: Option<T>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SubstError {
    Unbound(SigVar),
    Ty(TyError),
}

impl<T> SigArgs<T> {
    pub fn get(&self, var: SigVar) -> Option<&T> {
        match var {
            SigVar::Type(i) => self.types.get(i).and_then(Option::as_ref),
            SigVar::Identity => self.identity.as_ref(),
        }
    }

    fn bind(&mut self, var: SigVar, ty: T) {
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

    /// Whether it writes no variable.
    pub fn is_closed(&self) -> bool {
        match self {
            SigTy::Var(_) => false,
            SigTy::Term(term) => term.parts().into_iter().all(|(_, part)| part.is_closed()),
        }
    }

    pub fn mentions(&self, var: SigVar) -> bool {
        match self {
            SigTy::Var(own) => *own == var,
            SigTy::Term(term) => term.parts().into_iter().any(|(_, part)| part.mentions(var)),
        }
    }

    /// Each bound variable's closed type put in its place. Where that makes a
    /// reference's target a reference, the two are one reference to the inner
    /// target, exclusive only where both are.
    pub fn substitute<T>(&self, args: &SigArgs<T>) -> Result<T, SubstError>
    where
        T: HasTerm,
    {
        match self {
            SigTy::Var(var) => args.get(*var).cloned().ok_or(SubstError::Unbound(*var)),
            SigTy::Term(term) => {
                let term = (**term).clone().try_map(|_, part| part.substitute(args))?;
                let term = match term {
                    Term::Ref(outer, target) => match target.decided() {
                        Some(Term::Ref(inner, inner_target)) => {
                            Term::Ref(outer.min(*inner), inner_target.clone())
                        }
                        _ => Term::Ref(outer, target),
                    },
                    term => term,
                };
                T::from_term(term).map_err(SubstError::Ty)
            }
        }
    }

    /// Extends `args` so that `self` substituted by them is `ty`, and returns
    /// whether it could; where it could not, `args` is left as it was. A
    /// variable is bound to the type at its place, and a term written where
    /// `ty`'s position is not decided does not match.
    pub fn match_ty<T>(&self, ty: &T, args: &mut SigArgs<T>) -> bool
    where
        T: HasTerm,
    {
        let mut found = args.clone();
        let matched = self.match_into(ty, &mut found);
        if matched {
            *args = found;
        }
        matched
    }

    fn match_into<T>(&self, ty: &T, found: &mut SigArgs<T>) -> bool
    where
        T: HasTerm,
    {
        match self {
            SigTy::Var(var) => match found.get(*var) {
                Some(bound) => bound == ty,
                None => {
                    found.bind(*var, ty.clone());
                    true
                }
            },
            SigTy::Term(term) => {
                let Some(decided) = ty.decided() else {
                    return false;
                };
                if term.constructor() != decided.constructor()
                    || term.members() != decided.members()
                {
                    return false;
                }
                term.parts()
                    .into_iter()
                    .zip(decided.parts())
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
    /// body lies.
    IdentityPlace,
    /// A reference whose target is a reference.
    RefTarget,
    /// A function value, which a declaration makes none of: it returns one
    /// only as a type variable takes it in.
    FunctionValue,
    /// An `Error` where a value enters the program, which `?` alone makes.
    Error,
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
        for ty in &signature.params {
            signature.check(ty, types, false, true)?;
        }
        signature.check(&signature.result, types, false, false)?;
        Ok(signature)
    }

    /// Whether the type is one a signature writes: `outward` where a value
    /// leaves the program by it, at a parameter, which alone may hold an
    /// `Error`, turned over at a function type's parameters.
    fn check(
        &self,
        ty: &SigTy,
        types: &DeclaredTypes,
        at_identity: bool,
        outward: bool,
    ) -> Result<(), SigError> {
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
                if let Term::FunctionValue { .. } = **term {
                    return Err(SigError::FunctionValue);
                }
                if crate::join::is_error(term) && !outward {
                    return Err(SigError::Error);
                }
                if let Term::Ref(_, SigTy::Term(target)) = &**term
                    && matches!(**target, Term::Ref(..))
                {
                    return Err(SigError::RefTarget);
                }
                types.check(term).map_err(SigError::Decl)?;
                for (key, part) in term.parts() {
                    let outward = match (&**term, &key) {
                        (Term::Function { .. }, PartKey::Param(_)) => !outward,
                        _ => outward,
                    };
                    self.check(part, types, key == PartKey::Identity, outward)?;
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
    /// makes it anew.
    pub fn existential_identity(&self) -> bool {
        self.names_identity
            && self.result.mentions(SigVar::Identity)
            && !self
                .params
                .iter()
                .any(|param| param.mentions(SigVar::Identity))
    }

    /// Bindings with no variable bound.
    pub fn unbound<T>(&self) -> SigArgs<T>
    where
        T: Clone,
    {
        SigArgs {
            types: vec![None; self.type_params.len()],
            identity: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{IntTy, Mutability, Scalar};

    fn name(text: &str) -> Name {
        Name::new(text)
    }

    fn ty(term: Term<Ty>) -> Ty {
        Ty::new(term).expect("the term is a closed type")
    }

    fn u8_() -> Ty {
        ty(Term::Scalar(Scalar::Int(IntTy::U8)))
    }

    fn u16_() -> Ty {
        ty(Term::Scalar(Scalar::Int(IntTy::U16)))
    }

    fn var(i: usize) -> SigTy {
        SigTy::Var(SigVar::Type(i))
    }

    fn sig_ref(mutability: Mutability, target: SigTy) -> SigTy {
        SigTy::term(Term::Ref(mutability, target))
    }

    fn sig_extern(text: &str, args: Vec<SigTy>, identity: Option<SigTy>) -> SigTy {
        SigTy::term(Term::Extern {
            name: name(text),
            args,
            identity,
        })
    }

    /// `Vec` of one argument and no identity, `F` of no argument and an identity,
    /// `Idx` of no argument and an identity, and the struct `Pair` of `l` and `r`.
    fn types() -> DeclaredTypes {
        let mut types = DeclaredTypes::default();
        types
            .declare_extern(
                name("Vec"),
                ExternDecl {
                    arity: 1,
                    identity: false,
                },
            )
            .unwrap();
        types
            .declare_extern(
                name("F"),
                ExternDecl {
                    arity: 0,
                    identity: true,
                },
            )
            .unwrap();
        types
            .declare_extern(
                name("Idx"),
                ExternDecl {
                    arity: 0,
                    identity: true,
                },
            )
            .unwrap();
        types
            .declare_struct(
                name("Pair"),
                StructDecl {
                    fields: vec![name("l"), name("r")],
                },
            )
            .unwrap();
        types
    }

    #[test]
    fn a_name_is_declared_once_in_each_kind() {
        let mut types = types();
        assert_eq!(
            types.declare_extern(
                name("Vec"),
                ExternDecl {
                    arity: 0,
                    identity: false
                }
            ),
            Err(DeclError::Redeclared(name("Vec")))
        );
        assert_eq!(
            types.declare_struct(name("Pair"), StructDecl { fields: vec![] }),
            Err(DeclError::Redeclared(name("Pair")))
        );
        assert!(
            types
                .declare_struct(name("Vec"), StructDecl { fields: vec![] })
                .is_ok()
        );
    }

    #[test]
    fn a_term_agrees_with_its_declaration() {
        let types = types();
        let vec_u8 = Term::Extern {
            name: name("Vec"),
            args: vec![u8_()],
            identity: None,
        };
        assert_eq!(types.check(&vec_u8), Ok(()));
        let bare_vec: Term<Ty> = Term::Extern {
            name: name("Vec"),
            args: vec![],
            identity: None,
        };
        assert_eq!(
            types.check(&bare_vec),
            Err(DeclError::Arity {
                constructor: Constructor::Extern(name("Vec")),
                declared: 1,
                written: 0,
            })
        );
        let f_without_identity: Term<Ty> = Term::Extern {
            name: name("F"),
            args: vec![],
            identity: None,
        };
        assert_eq!(
            types.check(&f_without_identity),
            Err(DeclError::Identity {
                name: name("F"),
                declared: true,
            })
        );
        let unknown: Term<Ty> = Term::Extern {
            name: name("Map"),
            args: vec![],
            identity: None,
        };
        assert_eq!(
            types.check(&unknown),
            Err(DeclError::Undeclared(Constructor::Extern(name("Map"))))
        );
        let pair = Term::Struct(name("Pair"), vec![u8_(), u16_()]);
        assert_eq!(types.check(&pair), Ok(()));
        let short_pair = Term::Struct(name("Pair"), vec![u8_()]);
        assert!(matches!(
            types.check(&short_pair),
            Err(DeclError::Arity { .. })
        ));
        assert_eq!(types.check(&Term::<Ty>::Unit), Ok(()));
    }

    #[test]
    fn check_ty_looks_at_every_term() {
        let types = types();
        let inner = ty(Term::Extern {
            name: name("Vec"),
            args: vec![],
            identity: None,
        });
        let outer = ty(Term::Option(inner));
        assert!(types.check(outer.term()).is_ok());
        assert!(matches!(
            types.check_ty(&outer),
            Err(DeclError::Arity { .. })
        ));
    }

    #[test]
    fn a_signature_is_checked_at_every_term_it_writes() {
        let types = types();
        let vec_t = sig_extern("Vec", vec![var(0)], None);
        let push = Signature::new(
            vec![name("T")],
            false,
            vec![sig_ref(Mutability::Exclusive, vec_t.clone()), var(0)],
            SigTy::term(Term::Unit),
            &types,
        );
        assert!(push.is_ok());
        assert_eq!(
            Signature::new(vec![], false, vec![var(0)], SigTy::term(Term::Unit), &types),
            Err(SigError::VarOutOfRange(0))
        );
        let f_p = sig_extern("F", vec![], Some(SigTy::Var(SigVar::Identity)));
        assert_eq!(
            Signature::new(vec![], false, vec![], f_p.clone(), &types),
            Err(SigError::IdentityUnnamed)
        );
        assert_eq!(
            Signature::new(
                vec![],
                true,
                vec![SigTy::Var(SigVar::Identity)],
                f_p.clone(),
                &types
            ),
            Err(SigError::IdentityPlace)
        );
        let f_t = sig_extern("F", vec![], Some(var(0)));
        assert_eq!(
            Signature::new(vec![name("T")], false, vec![], f_t, &types),
            Err(SigError::IdentityPlace)
        );
        let own_identity = sig_extern("F", vec![], Some(SigTy::term(Term::Identity(name("ι")))));
        assert_eq!(
            Signature::new(vec![], false, vec![], own_identity, &types),
            Err(SigError::IdentityPlace)
        );
        let ref_ref = sig_ref(Mutability::Shared, sig_ref(Mutability::Shared, var(0)));
        assert_eq!(
            Signature::new(
                vec![name("T")],
                false,
                vec![ref_ref],
                SigTy::term(Term::Unit),
                &types
            ),
            Err(SigError::RefTarget)
        );
        let function_value = SigTy::term(Term::FunctionValue { arity: 1 });
        assert_eq!(
            Signature::new(vec![], false, vec![], function_value.clone(), &types),
            Err(SigError::FunctionValue)
        );
        assert_eq!(
            Signature::new(
                vec![],
                false,
                vec![sig_ref(Mutability::Shared, function_value)],
                SigTy::term(Term::Unit),
                &types
            ),
            Err(SigError::FunctionValue)
        );
        let error = SigTy::term(Term::Enum(
            name(crate::join::ERROR),
            [(
                name("Vec"),
                sig_extern("Vec", vec![SigTy::term(Term::Unit)], None),
            )]
            .into(),
        ));
        let unit = || SigTy::term(Term::Unit);
        let function = |param: SigTy| {
            SigTy::term(Term::Function {
                params: vec![param],
                ret: unit(),
            })
        };
        assert!(
            Signature::new(vec![], false, vec![error.clone()], unit(), &types).is_ok(),
            "a parameter, where a value leaves the program, may hold an `Error`"
        );
        assert_eq!(
            Signature::new(vec![], false, vec![], error.clone(), &types),
            Err(SigError::Error),
            "a result, where a value enters it, holds none"
        );
        assert_eq!(
            Signature::new(vec![], false, vec![function(error.clone())], unit(), &types),
            Err(SigError::Error),
            "a function type's parameter turns the side over"
        );
        assert!(
            Signature::new(vec![], false, vec![], function(error), &types).is_ok(),
            "and turns it over again at a result's function type"
        );
        let bare_vec = sig_extern("Vec", vec![], None);
        assert!(matches!(
            Signature::new(
                vec![],
                false,
                vec![bare_vec],
                SigTy::term(Term::Unit),
                &types
            ),
            Err(SigError::Decl(DeclError::Arity { .. }))
        ));
    }

    #[test]
    fn an_identity_only_the_result_carries_is_existential() {
        let types = types();
        let f_p = sig_extern("F", vec![], Some(SigTy::Var(SigVar::Identity)));
        let idx_p = sig_extern("Idx", vec![], Some(SigTy::Var(SigVar::Identity)));
        let open = Signature::new(vec![], true, vec![], f_p.clone(), &types).unwrap();
        assert!(open.existential_identity());
        let index_of = Signature::new(
            vec![],
            true,
            vec![sig_ref(Mutability::Shared, f_p)],
            idx_p,
            &types,
        )
        .unwrap();
        assert!(!index_of.existential_identity());
        let unit = Signature::new(vec![], true, vec![], SigTy::term(Term::Unit), &types).unwrap();
        assert!(!unit.existential_identity());
    }

    #[test]
    fn substitution_puts_each_bound_variable_in_its_place() {
        let vec_t = sig_extern("Vec", vec![var(0)], None);
        let args = SigArgs {
            types: vec![Some(u8_())],
            identity: None,
        };
        let expected = ty(Term::Extern {
            name: name("Vec"),
            args: vec![u8_()],
            identity: None,
        });
        assert_eq!(vec_t.substitute(&args), Ok(expected));
        let unbound: SigArgs = SigArgs {
            types: vec![None],
            identity: None,
        };
        assert_eq!(
            vec_t.substitute(&unbound),
            Err(SubstError::Unbound(SigVar::Type(0)))
        );
    }

    #[test]
    fn a_reference_substituted_under_a_reference_is_one_reference_exclusive_only_where_both_are() {
        let target = u8_();
        let shared_u = ty(Term::Ref(Mutability::Shared, target.clone()));
        let exclusive_u = ty(Term::Ref(Mutability::Exclusive, target.clone()));
        let at = |t: &Ty| SigArgs {
            types: vec![Some(t.clone())],
            identity: None,
        };
        let shared_t = sig_ref(Mutability::Shared, var(0));
        let exclusive_t = sig_ref(Mutability::Exclusive, var(0));
        assert_eq!(shared_t.substitute(&at(&shared_u)), Ok(shared_u.clone()));
        assert_eq!(shared_t.substitute(&at(&exclusive_u)), Ok(shared_u.clone()));
        assert_eq!(exclusive_t.substitute(&at(&shared_u)), Ok(shared_u.clone()));
        assert_eq!(
            exclusive_t.substitute(&at(&exclusive_u)),
            Ok(exclusive_u.clone())
        );
    }

    #[test]
    fn matching_finds_what_substitution_gives_back() {
        let vec_t = sig_extern("Vec", vec![var(0)], None);
        let param = sig_ref(Mutability::Exclusive, vec_t);
        let closed = ty(Term::Ref(
            Mutability::Exclusive,
            ty(Term::Extern {
                name: name("Vec"),
                args: vec![u8_()],
                identity: None,
            }),
        ));
        let mut args = SigArgs {
            types: vec![None],
            identity: None,
        };
        assert!(param.match_ty(&closed, &mut args));
        assert_eq!(args.types, vec![Some(u8_())]);
        assert_eq!(param.substitute(&args), Ok(closed));
    }

    #[test]
    fn matching_binds_a_variable_under_a_reference_to_the_target() {
        let shared_t = sig_ref(Mutability::Shared, var(0));
        let closed = ty(Term::Ref(Mutability::Shared, u8_()));
        let mut args = SigArgs {
            types: vec![None],
            identity: None,
        };
        assert!(shared_t.match_ty(&closed, &mut args));
        assert_eq!(args.types, vec![Some(u8_())]);
    }

    #[test]
    fn matching_fails_at_another_constructor_or_a_variable_met_twice_and_keeps_the_bindings() {
        let pair_t = SigTy::term(Term::Tuple(vec![var(0), var(0)]));
        let mut args = SigArgs {
            types: vec![None],
            identity: None,
        };
        let same = ty(Term::Tuple(vec![u8_(), u8_()]));
        let different = ty(Term::Tuple(vec![u8_(), u16_()]));
        assert!(!pair_t.match_ty(&different, &mut args));
        assert_eq!(args.types, vec![None]);
        assert!(pair_t.match_ty(&same, &mut args));
        assert!(!pair_t.match_ty(&ty(Term::Option(u8_())), &mut args));
        assert_eq!(args.types, vec![Some(u8_())]);
        let shared_t = sig_ref(Mutability::Shared, var(0));
        let mut fresh = SigArgs {
            types: vec![None],
            identity: None,
        };
        assert!(!shared_t.match_ty(&ty(Term::Ref(Mutability::Exclusive, u8_())), &mut fresh));
    }

    #[test]
    fn matching_binds_the_identity_variable_to_an_identity() {
        let f_p = sig_extern("F", vec![], Some(SigTy::Var(SigVar::Identity)));
        let closed = ty(Term::Extern {
            name: name("F"),
            args: vec![],
            identity: Some(ty(Term::Identity(name("ι₀")))),
        });
        let mut args = SigArgs {
            types: vec![],
            identity: None,
        };
        assert!(f_p.match_ty(&closed, &mut args));
        assert_eq!(args.identity, Some(ty(Term::Identity(name("ι₀")))));
    }

    #[test]
    fn matching_compares_an_objects_fields() {
        let written = SigTy::term(Term::Object([(name("x"), var(0))].into()));
        let closed = ty(Term::Object(
            [(name("x"), u8_()), (name("y"), u8_())].into(),
        ));
        let mut args = SigArgs {
            types: vec![None],
            identity: None,
        };
        assert!(!written.match_ty(&closed, &mut args));
    }
}
