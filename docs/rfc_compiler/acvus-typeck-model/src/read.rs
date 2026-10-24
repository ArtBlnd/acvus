//! The value a use, a lend, a projection or a binding reads: a type
//! some of whose positions may be left undecided, a position holding a LitVar
//! deciding nothing within its family.

use std::fmt;
use std::rc::Rc;

use crate::lit::LitVar;
use crate::sig::{SigTy, SigVar};
use crate::ty::{HasTerm, Term, Ty, TyError, check_term};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReadValue {
    /// A position the value read does not decide.
    Undecided,
    Lit(LitVar),
    Term(Rc<Term<ReadValue>>),
}

impl ReadValue {
    pub fn of_ty(ty: &Ty) -> ReadValue {
        ReadValue::Term(Rc::new(
            ty.term().clone().map(|_, part| ReadValue::of_ty(&part)),
        ))
    }

    /// A candidate's type written by its signature. Each of its variables is an
    /// unknown made anew at the use, which only the use itself records
    /// into, so the value the use reads there decides nothing.
    pub fn of_sig(ty: &SigTy) -> ReadValue {
        match ty {
            SigTy::Var(SigVar::Type(_) | SigVar::Identity) => ReadValue::Undecided,
            SigTy::Term(term) => ReadValue::Term(Rc::new(
                (**term).clone().map(|_, part| ReadValue::of_sig(&part)),
            )),
        }
    }

    /// The closed type, where every position is decided by a term.
    pub fn to_ty(&self) -> Option<Ty> {
        let ReadValue::Term(term) = self else {
            return None;
        };
        let term = (**term)
            .clone()
            .try_map(|_, part| part.to_ty().ok_or(()))
            .ok()?;
        Some(Ty::new(term).expect("a value read is checked as a closed type is"))
    }

    /// Each LitVar taken as its family's default.
    pub fn with_defaults(&self) -> ReadValue {
        match self {
            ReadValue::Undecided => ReadValue::Undecided,
            ReadValue::Lit(lit) => ReadValue::Term(Rc::new(Term::Scalar(lit.default_scalar()))),
            ReadValue::Term(term) => ReadValue::Term(Rc::new(
                (**term).clone().map(|_, part| part.with_defaults()),
            )),
        }
    }

    pub fn has_lit(&self) -> bool {
        match self {
            ReadValue::Undecided => false,
            ReadValue::Lit(_) => true,
            ReadValue::Term(term) => term.parts().into_iter().any(|(_, part)| part.has_lit()),
        }
    }
}

impl HasTerm for ReadValue {
    fn decided(&self) -> Option<&Term<ReadValue>> {
        match self {
            ReadValue::Term(term) => Some(term),
            ReadValue::Undecided | ReadValue::Lit(_) => None,
        }
    }

    fn from_term(term: Term<ReadValue>) -> Result<ReadValue, TyError> {
        check_term(&term)?;
        Ok(ReadValue::Term(Rc::new(term)))
    }
}

/// A position not decided is written `_`, and a LitVar as `LitVar` writes it.
impl fmt::Display for ReadValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadValue::Undecided => f.write_str("_"),
            ReadValue::Lit(lit) => write!(f, "{lit}"),
            ReadValue::Term(term) => write!(f, "{term}"),
        }
    }
}

impl fmt::Debug for ReadValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
