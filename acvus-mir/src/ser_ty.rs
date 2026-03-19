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

use crate::ty::{Effect, FnKind, Origin, Ty, TyVar};

/// Serializable mirror of [`Ty`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerTy {
    Int,
    Float,
    String,
    Bool,
    Unit,
    Range,
    Byte,
    Infer,
    Error,
    List { elem: Box<SerTy> },
    Object { fields: BTreeMap<std::string::String, SerTy> },
    Tuple { elems: Vec<SerTy> },
    Fn {
        params: Vec<SerTy>,
        ret: Box<SerTy>,
        fn_kind: FnKind,
        effect: Effect,
    },
    Opaque { name: std::string::String },
    Option { inner: Box<SerTy> },
    Enum {
        name: std::string::String,
        variants: BTreeMap<std::string::String, Option<Box<SerTy>>>,
    },
    Iterator { elem: Box<SerTy>, effect: Effect },
    Sequence { elem: Box<SerTy>, origin: Origin, effect: Effect },
    Deque { elem: Box<SerTy>, origin: Origin },
    Var { id: u32 },
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
            Ty::Range => SerTy::Range,
            Ty::Byte => SerTy::Byte,
            Ty::Infer(_) => SerTy::Infer,
            Ty::Error(_) => SerTy::Error,
            Ty::List(elem) => SerTy::List { elem: Box::new(elem.to_ser(interner)) },
            Ty::Object(fields) => SerTy::Object {
                fields: fields
                    .iter()
                    .map(|(k, v)| (interner.resolve(*k).to_string(), v.to_ser(interner)))
                    .collect(),
            },
            Ty::Tuple(elems) => SerTy::Tuple {
                elems: elems.iter().map(|e| e.to_ser(interner)).collect(),
            },
            Ty::Fn { params, ret, kind, effect, .. } => SerTy::Fn {
                params: params.iter().map(|p| p.to_ser(interner)).collect(),
                ret: Box::new(ret.to_ser(interner)),
                fn_kind: *kind,
                effect: *effect,
            },
            Ty::Opaque(name) => SerTy::Opaque { name: name.clone() },
            Ty::Option(inner) => SerTy::Option { inner: Box::new(inner.to_ser(interner)) },
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
            Ty::Iterator(elem, effect) => SerTy::Iterator {
                elem: Box::new(elem.to_ser(interner)),
                effect: *effect,
            },
            Ty::Sequence(elem, origin, effect) => SerTy::Sequence {
                elem: Box::new(elem.to_ser(interner)),
                origin: *origin,
                effect: *effect,
            },
            Ty::Deque(elem, origin) => SerTy::Deque {
                elem: Box::new(elem.to_ser(interner)),
                origin: *origin,
            },
            Ty::Var(TyVar(id)) => SerTy::Var { id: *id },
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
            SerTy::Range => Ty::Range,
            SerTy::Byte => Ty::Byte,
            SerTy::Infer => Ty::infer(),
            SerTy::Error => Ty::error(),
            SerTy::List { elem } => Ty::List(Box::new(elem.to_ty(interner))),
            SerTy::Object { fields } => Ty::Object(
                fields
                    .iter()
                    .map(|(k, v)| (interner.intern(k), v.to_ty(interner)))
                    .collect(),
            ),
            SerTy::Tuple { elems } => Ty::Tuple(elems.iter().map(|e| e.to_ty(interner)).collect()),
            SerTy::Fn { params, ret, fn_kind, effect } => Ty::Fn {
                params: params.iter().map(|p| p.to_ty(interner)).collect(),
                ret: Box::new(ret.to_ty(interner)),
                kind: *fn_kind,
                captures: vec![],
                effect: *effect,
            },
            SerTy::Opaque { name } => Ty::Opaque(name.clone()),
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
            SerTy::Iterator { elem, effect } => Ty::Iterator(Box::new(elem.to_ty(interner)), *effect),
            SerTy::Sequence { elem, origin, effect } => {
                Ty::Sequence(Box::new(elem.to_ty(interner)), *origin, *effect)
            }
            SerTy::Deque { elem, origin } => Ty::Deque(Box::new(elem.to_ty(interner)), *origin),
            SerTy::Var { id } => Ty::Var(TyVar(*id)),
        }
    }
}
