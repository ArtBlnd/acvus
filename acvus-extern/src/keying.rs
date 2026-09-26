//! The keying marker a map or a set carries as its last type argument
//! (RFC-0098 rule 2). `acvus_mir::analysis::loop_deps` knows `Equiv` by the
//! declaration `Externs::combine` names from this Rust type, not by its
//! spelling.

use acvus_mir::ty::{PolyTy, TyTerm, UserDefinedDecl};
use acvus_utils::{Interner, QualifiedRef};

use crate::canonical::Canonical;
use crate::registry::ExternTypeDecl;
use crate::ty_arg::{PolyVars, TyArg, Var, kind};

/// A table whose keys meet by an `eq` requirement naming
/// `law::Equivalence`.
pub enum Equiv {}

pub enum Opaque {}

macro_rules! keying_marker {
    ($marker:ident, $name:literal) => {
        impl Var<kind::Type> for $marker {}

        // SAFETY: an uninhabited marker holds no `Erased`.
        unsafe impl Canonical<kind::Type> for $marker {
            type Canon = Self;
        }

        // SAFETY: an uninhabited marker holds no carrier.
        unsafe impl<'s> crate::Within<'s> for $marker {}

        impl TyArg for $marker {
            fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy {
                TyTerm::UserDefined {
                    id: vars.extension::<Self>(interner),
                    type_args: Vec::new(),
                    effect_args: Vec::new(),
                    identity_args: Vec::new(),
                    region_params: 0,
                }
            }
        }

        impl ExternTypeDecl for $marker {
            type DeclarationForm = Self;

            const REGION_PARAMS: usize = 0;

            fn type_decl(interner: &Interner) -> UserDefinedDecl {
                UserDefinedDecl {
                    qref: QualifiedRef::root(interner.intern($name)),
                    type_params: Vec::new(),
                    effect_params: 0,
                    identity_params: 0,
                    region_params: 0,
                    specializable: Vec::new(),
                    may_hold_a_function: false,
                }
            }
        }
    };
}

keying_marker!(Equiv, "Equiv");
keying_marker!(Opaque, "Opaque");
