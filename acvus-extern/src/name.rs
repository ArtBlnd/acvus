//! The Rust type behind each acvus type name, which `Externs::combine`
//! holds to a bijection so that a value's acvus type determines its box.

use std::any::TypeId;

use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::UserDefinedDecl;
use acvus_utils::Interner;

use crate::registry::ExternTypeDecl;

/// The `TypeId` of a Rust type at its declaration form: `()` for each of
/// its variables and `TypesOnly` for its runtime. A box holds the type at
/// its run-time instantiation, whose `TypeId` differs, so this is compared
/// only with another `DeclarationForm` and exposes no `TypeId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeclarationForm(TypeId);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameKind {
    Extension,
    Derived,
}

/// A name a declaration gives a Rust type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Named {
    pub qref: QualifiedRef,
    pub rust: DeclarationForm,
    pub rust_path: &'static str,
    pub kind: NameKind,
}

impl Named {
    pub fn extension<T>(interner: &Interner) -> Self
    where
        T: ExternTypeDecl + ?Sized,
    {
        Self {
            qref: T::type_decl(interner).qref,
            rust: DeclarationForm(TypeId::of::<T::DeclarationForm>()),
            rust_path: std::any::type_name::<T::DeclarationForm>(),
            kind: NameKind::Extension,
        }
    }

    /// `#[derive(TyArg)]` refuses a type with generic parameters, so `T`
    /// is its own declaration form.
    pub fn derived<T>(qref: QualifiedRef) -> Self
    where
        T: 'static,
    {
        Self {
            qref,
            rust: DeclarationForm(TypeId::of::<T>()),
            rust_path: std::any::type_name::<T>(),
            kind: NameKind::Derived,
        }
    }
}

/// An extension type's declaration with the Rust type behind its name.
pub struct DeclaredType {
    pub decl: UserDefinedDecl,
    pub named: Named,
}

impl DeclaredType {
    pub fn of<T>(interner: &Interner) -> Self
    where
        T: ExternTypeDecl + ?Sized,
    {
        Self {
            decl: T::type_decl(interner),
            named: Named::extension::<T>(interner),
        }
    }
}
