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

use crate::ty::{Effect, EffectSet, EffectTarget, Identity, IdentityId, Ty};

// ── Serializable Effect (mirrors ty::Effect without Astr) ────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerEffect {
    Resolved(SerEffectSet),
    Var(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SerEffectSet {
    pub reads: Vec<SerEffectTarget>,
    pub writes: Vec<SerEffectTarget>,
    pub self_modifying: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerEffectTarget {
    Context(SerQualifiedRef),
    Token(SerQualifiedRef),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerQualifiedRef {
    pub namespace: Option<String>,
    pub name: String,
}

fn target_to_ser(t: &EffectTarget, interner: &Interner) -> SerEffectTarget {
    match t {
        EffectTarget::Context(qref) => SerEffectTarget::Context(qref_to_ser(qref, interner)),
        EffectTarget::Token(qref) => SerEffectTarget::Token(qref_to_ser(qref, interner)),
    }
}

fn ser_to_target(t: &SerEffectTarget, interner: &Interner) -> EffectTarget {
    match t {
        SerEffectTarget::Context(qref) => EffectTarget::Context(ser_to_qref(qref, interner)),
        SerEffectTarget::Token(qref) => EffectTarget::Token(ser_to_qref(qref, interner)),
    }
}

impl Effect {
    pub fn to_ser(&self, interner: &Interner) -> SerEffect {
        match self {
            Effect::Resolved(set) => SerEffect::Resolved(SerEffectSet {
                reads: set
                    .reads
                    .iter()
                    .map(|r| target_to_ser(r, interner))
                    .collect(),
                writes: set
                    .writes
                    .iter()
                    .map(|r| target_to_ser(r, interner))
                    .collect(),
                self_modifying: set.self_modifying,
            }),
            Effect::Var(v) => SerEffect::Var(*v),
        }
    }
}

impl SerEffect {
    pub fn to_effect(&self, interner: &Interner) -> Effect {
        match self {
            SerEffect::Resolved(set) => Effect::Resolved(EffectSet {
                reads: set
                    .reads
                    .iter()
                    .map(|r| ser_to_target(r, interner))
                    .collect(),
                writes: set
                    .writes
                    .iter()
                    .map(|r| ser_to_target(r, interner))
                    .collect(),
                self_modifying: set.self_modifying,
            }),
            SerEffect::Var(v) => Effect::Var(*v),
        }
    }
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

/// Serializable mirror of [`Identity`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerIdentity {
    Concrete { id: u32 },
    Fresh { id: u32 },
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
    Range,
    Byte,
    Error,
    /// Type parameter. Should not appear in persisted data; deserializes to Ty::error().
    Param {
        id: u32,
    },
    /// Legacy variant kept for backward-compatible deserialization only.
    Infer,
    /// Legacy variant kept for backward-compatible deserialization only.
    Var {
        id: u32,
    },
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
        effect: SerEffect,
    },
    UserDefined {
        id: SerQualifiedRef,
        type_args: Vec<SerTy>,
        effect_args: Vec<SerEffect>,
    },
    Option {
        inner: Box<SerTy>,
    },
    Enum {
        name: std::string::String,
        variants: BTreeMap<std::string::String, Option<Box<SerTy>>>,
    },
    Deque {
        elem: Box<SerTy>,
        identity: Box<SerTy>,
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
            Ty::Range => SerTy::Range,
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
                effect,
                ..
            } => SerTy::Fn {
                params: params.iter().map(|p| p.ty.to_ser(interner)).collect(),
                ret: Box::new(ret.to_ser(interner)),
                effect: effect.to_ser(interner),
            },
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
            } => SerTy::UserDefined {
                id: qref_to_ser(id, interner),
                type_args: type_args.iter().map(|t| t.to_ser(interner)).collect(),
                effect_args: effect_args.iter().map(|e| e.to_ser(interner)).collect(),
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
            Ty::Deque(elem, identity) => SerTy::Deque {
                elem: Box::new(elem.to_ser(interner)),
                identity: Box::new(identity.to_ser(interner)),
            },
            Ty::Identity(id) => SerTy::Identity(match id {
                Identity::Concrete(cid) => SerIdentity::Concrete {
                    id: cid.to_raw() as u32,
                },
                Identity::Fresh(fid) => SerIdentity::Fresh {
                    id: fid.to_raw() as u32,
                },
            }),
            Ty::Handle(..) => todo!("Handle serialization not yet implemented"),
            Ty::Ref(..) => todo!("Ref serialization not yet implemented"),
            Ty::Param { token: p, .. } => SerTy::Param { id: p.id() },
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
            SerTy::Error => Ty::error(),
            // Param / Infer / Var should never appear in persisted data.
            // Recover gracefully with poison type.
            SerTy::Param { .. } | SerTy::Infer | SerTy::Var { .. } => Ty::error(),
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
                effect,
            } => Ty::Fn {
                params: params
                    .iter()
                    .map(|p| crate::ty::Param::new(interner.intern("_"), p.to_ty(interner)))
                    .collect(),
                ret: Box::new(ret.to_ty(interner)),
                captures: vec![],
                effect: effect.to_effect(interner),
            },
            SerTy::UserDefined {
                id,
                type_args,
                effect_args,
            } => Ty::UserDefined {
                id: ser_to_qref(id, interner),
                type_args: type_args.iter().map(|t| t.to_ty(interner)).collect(),
                effect_args: effect_args.iter().map(|e| e.to_effect(interner)).collect(),
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
            SerTy::Deque { elem, identity } => Ty::Deque(
                Box::new(elem.to_ty(interner)),
                Box::new(identity.to_ty(interner)),
            ),
            SerTy::Identity(ser_id) => Ty::Identity(match ser_id {
                SerIdentity::Concrete { id } => {
                    Identity::Concrete(IdentityId::from_raw(*id as usize))
                }
                SerIdentity::Fresh { id } => Identity::Fresh(IdentityId::from_raw(*id as usize)),
            }),
        }
    }
}
