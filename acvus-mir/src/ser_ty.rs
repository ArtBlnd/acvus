//! Serializable type representation.
//!
//! [`SerTy`] mirrors [`Ty`] but replaces [`Astr`] (interned strings) with
//! [`String`], making it `Serialize + Deserialize`. This is used at storage
//! boundaries (BlobStore, JSON export) where types must roundtrip through
//! serialization without an interner.
//!
//! Conversion:
//! - `Ty::to_ser(interner) -> SerTy` — resolve all Astr to String.
//! - `SerTy::to_ty(interner) -> Ty` — re-intern all String to Astr.

use std::collections::BTreeMap;

use acvus_utils::Interner;
use serde::{Deserialize, Serialize};

use crate::graph::QualifiedRef;
use acvus_utils::LocalIdOps;

use crate::ty::{IdentityId, Ty};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerQualifiedRef {
    pub namespace: Option<String>,
    pub name: String,
}

fn qref_to_ser(r: &QualifiedRef, interner: &Interner) -> SerQualifiedRef {
    SerQualifiedRef {
        namespace: r.namespace.map(|ns| interner.resolve(ns).to_string()),
        name: interner.resolve(r.name).to_string(),
    }
}

fn ser_to_qref(r: &SerQualifiedRef, interner: &Interner) -> QualifiedRef {
    QualifiedRef {
        namespace: r.namespace.as_ref().map(|ns| interner.intern(ns)),
        name: interner.intern(&r.name),
    }
}

/// Serializable mirror of [`IdentityId`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SerIdentity {
    pub id: u32,
}

/// Serializable mirror of [`Ty`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerTy {
    Int,
    Float,
    String,
    Bool,
    Unit,
    Byte,
    Error,
    List {
        elem: Box<SerTy>,
    },
    Object {
        fields: BTreeMap<std::string::String, SerTy>,
    },
    Tuple {
        elems: Vec<SerTy>,
    },
    Fn {
        params: Vec<SerTy>,
        ret: Box<SerTy>,
    },
    UserDefined {
        id: SerQualifiedRef,
        type_args: Vec<SerTy>,
    },
    Option {
        inner: Box<SerTy>,
    },
    Enum {
        name: std::string::String,
        variants: BTreeMap<std::string::String, Option<Box<SerTy>>>,
    },
    Identity(SerIdentity),
}

impl Ty {
    /// Convert to a serializable representation by resolving all interned strings.
    pub fn to_ser(&self, interner: &Interner) -> SerTy {
        match self {
            Ty::Int => SerTy::Int,
            Ty::Float => SerTy::Float,
            Ty::String => SerTy::String,
            Ty::Bool => SerTy::Bool,
            Ty::Unit => SerTy::Unit,
            Ty::Byte => SerTy::Byte,
            Ty::Error(_) => SerTy::Error,
            Ty::List(elem) => SerTy::List {
                elem: Box::new(elem.to_ser(interner)),
            },
            Ty::Object(fields) => SerTy::Object {
                fields: fields
                    .iter()
                    .map(|(k, v)| (interner.resolve(*k).to_string(), v.to_ser(interner)))
                    .collect(),
            },
            Ty::Tuple(elems) => SerTy::Tuple {
                elems: elems.iter().map(|e| e.to_ser(interner)).collect(),
            },
            Ty::Fn {
                params,
                ret,
                ..
            } => SerTy::Fn {
                params: params.iter().map(|p| p.ty.to_ser(interner)).collect(),
                ret: Box::new(ret.to_ser(interner)),
            },
            Ty::UserDefined {
                id,
                type_args,
            } => SerTy::UserDefined {
                id: qref_to_ser(id, interner),
                type_args: type_args.iter().map(|t| t.to_ser(interner)).collect(),
            },
            Ty::Option(inner) => SerTy::Option {
                inner: Box::new(inner.to_ser(interner)),
            },
            Ty::Enum { name, variants } => SerTy::Enum {
                name: interner.resolve(*name).to_string(),
                variants: variants
                    .iter()
                    .map(|(k, v)| {
                        (
                            interner.resolve(*k).to_string(),
                            v.as_ref().map(|t| Box::new(t.to_ser(interner))),
                        )
                    })
                    .collect(),
            },
            Ty::Identity(id) => SerTy::Identity(SerIdentity {
                id: id.to_raw() as u32,
            }),
            Ty::Handle(..) => todo!("Handle serialization not yet implemented"),
            Ty::Ref(..) => todo!("Ref serialization not yet implemented"),
            Ty::Var(v) => match *v {},
        }
    }
}

impl SerTy {
    /// Convert back to [`Ty`] by re-interning all strings.
    pub fn to_ty(&self, interner: &Interner) -> Ty {
        match self {
            SerTy::Int => Ty::Int,
            SerTy::Float => Ty::Float,
            SerTy::String => Ty::String,
            SerTy::Bool => Ty::Bool,
            SerTy::Unit => Ty::Unit,
            SerTy::Byte => Ty::Byte,
            SerTy::Error => Ty::error(),
            SerTy::List { elem } => Ty::List(Box::new(elem.to_ty(interner))),
            SerTy::Object { fields } => Ty::Object(
                fields
                    .iter()
                    .map(|(k, v)| (interner.intern(k), v.to_ty(interner)))
                    .collect(),
            ),
            SerTy::Tuple { elems } => Ty::Tuple(elems.iter().map(|e| e.to_ty(interner)).collect()),
            SerTy::Fn {
                params,
                ret,
            } => Ty::Fn {
                params: params
                    .iter()
                    .map(|p| crate::ty::Param::new(interner.intern("_"), p.to_ty(interner)))
                    .collect(),
                ret: Box::new(ret.to_ty(interner)),
                captures: vec![],
                hint: None,
            },
            SerTy::UserDefined {
                id,
                type_args,
            } => Ty::UserDefined {
                id: ser_to_qref(id, interner),
                type_args: type_args.iter().map(|t| t.to_ty(interner)).collect(),
            },
            SerTy::Option { inner } => Ty::Option(Box::new(inner.to_ty(interner))),
            SerTy::Enum { name, variants } => Ty::Enum {
                name: interner.intern(name),
                variants: variants
                    .iter()
                    .map(|(k, v)| {
                        (
                            interner.intern(k),
                            v.as_ref().map(|t| Box::new(t.to_ty(interner))),
                        )
                    })
                    .collect(),
            },
            SerTy::Identity(ser_id) => Ty::Identity(
                IdentityId::from_raw(ser_id.id as usize),
            ),
        }
    }
}
