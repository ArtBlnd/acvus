//! A declaration's scheme at one use (extern-fn §6).

use acvus_utils::QualifiedRef;

use super::{Infer, Term};
use crate::decl::{Declarations, Scheme};
use crate::laws::NamedLaw;

/// A requirement at the use's own terms: an instance of `signature` whose
/// type joins `pattern`, stating `law` where named.
#[derive(Debug, Clone, PartialEq)]
pub struct RequiredAt {
    pub signature: QualifiedRef,
    pub pattern: Term,
    pub law: Option<NamedLaw>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instantiated {
    /// The function type at this use.
    pub ty: Term,
    pub requirements: Vec<RequiredAt>,
}

impl Infer {
    /// `scheme` at fresh variables, each with its declared bound. Every
    /// structural node it builds is `Label::Opaque`; a type parameter's
    /// place takes the program's term as it is (types §1).
    pub fn instantiate(&mut self, decls: &Declarations, scheme: &Scheme) -> Instantiated {
        todo!("infer::instantiate")
    }
}
