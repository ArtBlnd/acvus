//! Declarations: what a body sees opaquely (Transparency).

use std::collections::BTreeMap;

use super::Name;
use super::ty::Ty;

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Decls {
    pub extern_types: BTreeMap<Name, ExternType>,
    /// By module name.
    pub structs: BTreeMap<Name, StructDecl>,
    /// The declarations of one name are the candidates of a use of it
    /// (O13 to O15).
    pub fns: BTreeMap<Name, Vec<Signature>>,
    pub conversions: BTreeMap<Name, Conversion>,
    /// The context type of each `@x`, which has no hole (O13).
    pub contexts: BTreeMap<Name, Ty>,
}

/// `extern type N<T̄>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExternType {
    pub params: usize,
}

/// A module's struct: one type parameter for each field, in this order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StructDecl {
    pub fields: Vec<Name>,
}

/// A hole is marked by `Ty::Hole` alone.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Signature {
    pub params: Vec<Ty>,
    pub result: Ty,
}

/// A declared function `S → T` (Changes), of one parameter.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Conversion {
    pub from: Ty,
    pub to: Ty,
}
