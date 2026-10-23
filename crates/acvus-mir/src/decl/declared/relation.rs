//! Relation (extern-fn §3): `=`, `≤` or `<` between two affine integer
//! terms over one call's atoms (D83). This is the syntax only; its meaning
//! is read by the affine value domain.

use super::Stated;
use crate::decl::Refused;
use crate::ty::{Mutability, Poly, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    pub left: Term,
    pub comparison: Comparison,
    pub right: Term,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// `=`
    Eq,
    /// `≤`
    Le,
    /// `<`
    Lt,
}

/// An affine term: sums and differences of atoms and of their constant
/// multiples, and constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Const(i128),
    Atom(Atom),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    Scale(i128, Box<Term>),
}

/// An integer one call determines. A parameter is numbered as a call's
/// arguments are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    /// An argument of an integer type, or the integer a reference argument
    /// lends when the call ends.
    Arg(usize),
    /// The result, of an integer type, or the integer a returned reference
    /// lends.
    Result,
    /// The element count of an argument or of the result.
    Len(Subject),
    /// What a `&mut` argument lent when the call began.
    Old(Before),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Subject {
    Arg(usize),
    Result,
}

/// What a `&mut` parameter lent when the call began.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Before {
    /// The integer it lent.
    Lent(usize),
    /// The element count of what it lent.
    Len(usize),
}

impl Term {
    /// Every atom the term reads, each where it is written.
    pub fn atoms(&self) -> Vec<Atom> {
        let mut atoms = Vec::new();
        let mut pending = vec![self];
        while let Some(term) = pending.pop() {
            match term {
                Term::Const(_) => {}
                Term::Atom(atom) => atoms.push(*atom),
                Term::Add(left, right) | Term::Sub(left, right) => {
                    pending.push(right);
                    pending.push(left);
                }
                Term::Scale(_, inner) => pending.push(inner),
            }
        }
        atoms
    }
}

/// Refuses each relation whose atoms are not integers at `stated`'s
/// signature.
pub(super) fn refuse_ill_formed(stated: &Stated<'_>, types: &Types<Poly>, relations: &[Relation], refused: &mut Vec<Refused>) {
    for relation in relations {
        for term in [&relation.left, &relation.right] {
            refuse_ill_formed_term(stated, types, term, refused);
        }
    }
}

pub(super) fn refuse_ill_formed_term(stated: &Stated<'_>, types: &Types<Poly>, term: &Term, refused: &mut Vec<Refused>) {
    for atom in term.atoms() {
        refuse_ill_formed_atom(stated, types, atom, refused);
    }
}

fn refuse_ill_formed_atom(stated: &Stated<'_>, types: &Types<Poly>, atom: Atom, refused: &mut Vec<Refused>) {
    let declared = stated.declared;
    let integer = |ty| stated.is_integer(types, ty) || stated.lent(types, ty).is_some_and(|(_, lent)| stated.is_integer(types, lent));
    match atom {
        Atom::Arg(at) => {
            if let Some(ty) = stated.param(at, refused)
                && !integer(ty)
            {
                refused.push(Refused::RelationOverNonInteger { declared, atom });
            }
        }
        Atom::Result => {
            if !integer(stated.ret) {
                refused.push(Refused::RelationOverNonInteger { declared, atom });
            }
        }
        Atom::Len(subject) => {
            let ty = match subject {
                Subject::Arg(at) => stated.param(at, refused),
                Subject::Result => Some(stated.ret),
            };
            if let Some(ty) = ty
                && !stated.has_length(types, ty)
            {
                refused.push(Refused::LengthOfNoSequence { declared, subject });
            }
        }
        Atom::Old(before) => {
            let param = match before {
                Before::Lent(at) | Before::Len(at) => at,
            };
            let Some(ty) = stated.param(param, refused) else {
                return;
            };
            let Some((Mutability::Mut, lent)) = stated.lent(types, ty) else {
                refused.push(Refused::OldOfNoMutableReference { declared, param });
                return;
            };
            match before {
                Before::Lent(_) if !stated.is_integer(types, lent) => {
                    refused.push(Refused::RelationOverNonInteger { declared, atom });
                }
                Before::Len(_) if !stated.has_length(types, ty) => {
                    refused.push(Refused::LengthOfNoSequence { declared, subject: Subject::Arg(param) });
                }
                Before::Lent(_) | Before::Len(_) => {}
            }
        }
    }
}
