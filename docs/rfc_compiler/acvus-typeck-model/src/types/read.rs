//! The value a use, a lend, a projection or a binding reads: a type
//! some of whose positions may be left undecided, a position holding only a
//! domain deciding nothing among the constructors it admits.

use std::fmt;
use std::rc::Rc;

use crate::domain::Domain;
use crate::sig::{SigTy, SigVar};
use crate::ty::{HasTerm, Term, Ty, TyError, check_term};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReadValue {
    /// A position the value read does not decide.
    Undecided,
    /// A position holding only a domain that admits no `!`.
    Domain(Domain),
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

}

impl HasTerm for ReadValue {
    fn decided(&self) -> Option<&Term<ReadValue>> {
        match self {
            ReadValue::Term(term) => Some(term),
            ReadValue::Undecided | ReadValue::Domain(_) => None,
        }
    }

    fn from_term(term: Term<ReadValue>) -> Result<ReadValue, TyError> {
        check_term(&term)?;
        Ok(ReadValue::Term(Rc::new(term)))
    }
}

/// A position not decided is written `_`, and a domain as `Domain` writes it.
impl fmt::Display for ReadValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadValue::Undecided => f.write_str("_"),
            ReadValue::Domain(domain) => write!(f, "{domain}"),
            ReadValue::Term(term) => write!(f, "{term}"),
        }
    }
}

impl fmt::Debug for ReadValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
