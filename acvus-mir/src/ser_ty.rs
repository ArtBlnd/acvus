//! Serializable type representation.
//!
//! [`SerTy`] mirrors [`Ty`] but replaces [`Astr`] (interned strings) with
//! [`String`], making it `Serialize + Deserialize`. This is used at storage
//! boundaries (BlobStore, JSON export) where types must roundtrip through
//! serialization without an interner.
//!
//! Conversion:
//! - `Ty::to_ser(interner) -> SerTy` - resolve all Astr to String.
//! - `SerTy::to_ty(interner) -> Ty` - re-intern all String to Astr.

use std::collections::BTreeMap;

use acvus_utils::Interner;
use serde::{Deserialize, Serialize};

use crate::graph::QualifiedRef;
use acvus_utils::LocalIdOps;

use crate::ty::{
    Concrete, Effect, EffectArg, EffectTerm, FieldSet, IdentityId, IdentityTerm, IntTy, LenTerm, ObjectTy,
    Reissue, Repr, Task, Ty, TypeArg,
};

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

/// An `Effect` with its context names written out.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerEffect {
    pub reissue: Reissue,
    #[serde(default)]
    pub task: Task,
    pub commutes: bool,
    pub reads: Vec<SerQualifiedRef>,
    pub writes: Vec<SerQualifiedRef>,
}

fn effect_to_ser(e: &Effect, interner: &Interner) -> SerEffect {
    SerEffect {
        reissue: e.reissue,
        task: e.task,
        commutes: e.commutes,
        reads: e.reads.iter().map(|q| qref_to_ser(q, interner)).collect(),
        writes: e.writes.iter().map(|q| qref_to_ser(q, interner)).collect(),
    }
}

fn ser_to_effect(e: &SerEffect, interner: &Interner) -> Effect {
    Effect::with_contexts(
        e.reissue,
        e.commutes,
        e.reads.iter().map(|q| ser_to_qref(q, interner)).collect(),
        e.writes.iter().map(|q| ser_to_qref(q, interner)).collect(),
    )
    .at_task(e.task)
}

/// A parameter of a serialized function type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerParam {
    pub ty: SerTy,
}

/// The representation of a slot's argument (hash-types.md).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SerRepr {
    Uniform,
    Specialized,
}

/// A type argument of a user-defined type, with its representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerTypeArg {
    pub repr: SerRepr,
    pub ty: SerTy,
}

/// An effect argument of a user-defined type, with its representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerEffectArg {
    pub repr: SerRepr,
    pub effect: SerEffect,
}

fn repr_to_ser(repr: Repr<Concrete>) -> SerRepr {
    match repr {
        Repr::Uniform => SerRepr::Uniform,
        Repr::Specialized => SerRepr::Specialized,
        Repr::Var(v) => match v {},
    }
}

fn ser_to_repr(repr: SerRepr) -> Repr<Concrete> {
    match repr {
        SerRepr::Uniform => Repr::Uniform,
        SerRepr::Specialized => Repr::Specialized,
    }
}

fn arg_to_ser(arg: &TypeArg<Concrete>, interner: &Interner) -> SerTypeArg {
    SerTypeArg {
        repr: repr_to_ser(arg.repr),
        ty: arg.ty.to_ser(interner),
    }
}

fn ser_to_arg(arg: &SerTypeArg, interner: &Interner) -> TypeArg<Concrete> {
    TypeArg {
        repr: ser_to_repr(arg.repr),
        ty: arg.ty.to_ty(interner),
    }
}

fn effect_arg_to_ser(arg: &EffectArg<Concrete>, interner: &Interner) -> SerEffectArg {
    SerEffectArg {
        repr: repr_to_ser(arg.repr),
        effect: effect_to_ser(arg.effect.get(), interner),
    }
}

fn ser_to_effect_arg(arg: &SerEffectArg, interner: &Interner) -> EffectArg<Concrete> {
    EffectArg {
        repr: ser_to_repr(arg.repr),
        effect: EffectTerm::Known(ser_to_effect(&arg.effect, interner)),
    }
}

/// An object type's field set (RFC-0042) as it is written down.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerFieldSet {
    #[default]
    Written,
    AtLeast,
    Declared {
        name: std::string::String,
    },
}

/// Serializable mirror of [`Ty`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerTy {
    Int {
        width: IntTy,
    },
    Order,
    Float,
    Char,
    String,
    Bool,
    Unit,
    Never,
    Error,
    Array {
        len: usize,
        elem: Box<SerTy>,
    },
    Object {
        /// A serialized object type without it is an object literal's, so a
        /// freeze written before a field set said which it is still reads.
        #[serde(default)]
        field_set: SerFieldSet,
        fields: BTreeMap<std::string::String, SerTy>,
    },
    Tuple {
        elems: Vec<SerTy>,
    },
    Fn {
        params: Vec<SerParam>,
        ret: Box<SerTy>,
        effect: SerEffect,
    },
    UserDefined {
        id: SerQualifiedRef,
        type_args: Vec<SerTypeArg>,
        effect_args: Vec<SerEffectArg>,
        identity_args: Vec<u32>,
    },
    Option {
        inner: Box<SerTy>,
    },
    Result {
        ok: Box<SerTy>,
        err: Box<SerTy>,
    },
    Enum {
        name: std::string::String,
        variants: BTreeMap<std::string::String, Option<Box<SerTy>>>,
    },
}

impl Ty {
    /// Convert to a serializable representation by resolving all interned strings.
    pub fn to_ser(&self, interner: &Interner) -> SerTy {
        match self {
            Ty::Int(k) => SerTy::Int { width: *k },
            Ty::Order => SerTy::Order,
            Ty::Float => SerTy::Float,
            Ty::Char => SerTy::Char,
            Ty::String => SerTy::String,
            Ty::Bool => SerTy::Bool,
            Ty::Unit => SerTy::Unit,
            Ty::Never => SerTy::Never,
            Ty::Error(_) => SerTy::Error,
            Ty::Array(elem, len) => SerTy::Array {
                len: len.get(),
                elem: Box::new(elem.to_ser(interner)),
            },
            Ty::Object(object) => SerTy::Object {
                field_set: match object.field_set() {
                    FieldSet::Declared(name) => SerFieldSet::Declared {
                        name: interner.resolve(name).to_string(),
                    },
                    FieldSet::Written => SerFieldSet::Written,
                    FieldSet::AtLeast => SerFieldSet::AtLeast,
                },
                fields: object
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
                params: params
                    .iter()
                    .map(|p| SerParam {
                        ty: p.ty.to_ser(interner),
                    })
                    .collect(),
                ret: Box::new(ret.to_ser(interner)),
                effect: effect_to_ser(effect.get(), interner),
            },
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => SerTy::UserDefined {
                id: qref_to_ser(id, interner),
                type_args: type_args.iter().map(|t| arg_to_ser(t, interner)).collect(),
                effect_args: effect_args
                    .iter()
                    .map(|e| effect_arg_to_ser(e, interner))
                    .collect(),
                identity_args: identity_args
                    .iter()
                    .map(|i| i.get().to_raw() as u32)
                    .collect(),
            },
            Ty::Option(inner) => SerTy::Option {
                inner: Box::new(inner.to_ser(interner)),
            },
            Ty::Result(ok, err) => SerTy::Result {
                ok: Box::new(ok.to_ser(interner)),
                err: Box::new(err.to_ser(interner)),
            },
            Ty::Enum { name, variants, .. } => SerTy::Enum {
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
            Ty::Handle(..) => todo!("Handle serialization not yet implemented"),
            Ty::Ref(..) | Ty::Slice(..) | Ty::Str => {
                todo!("Ref serialization not yet implemented")
            }
            Ty::Var(v) => match *v {},
        }
    }
}

impl SerTy {
    /// Convert back to [`Ty`] by re-interning all strings.
    pub fn to_ty(&self, interner: &Interner) -> Ty {
        match self {
            SerTy::Int { width } => Ty::Int(*width),
            SerTy::Order => Ty::Order,
            SerTy::Float => Ty::Float,
            SerTy::Char => Ty::Char,
            SerTy::String => Ty::String,
            SerTy::Bool => Ty::Bool,
            SerTy::Unit => Ty::Unit,
            SerTy::Never => Ty::Never,
            SerTy::Error => Ty::error(),
            SerTy::Array { len, elem } => {
                Ty::Array(Box::new(elem.to_ty(interner)), LenTerm::Known(*len))
            }
            SerTy::Object { field_set, fields } => {
                let fields = fields
                    .iter()
                    .map(|(k, v)| (interner.intern(k), v.to_ty(interner)))
                    .collect();
                Ty::Object(match field_set {
                    SerFieldSet::Declared { name } => {
                        ObjectTy::declared(interner.intern(name), fields)
                    }
                    SerFieldSet::Written => ObjectTy::written(fields),
                    SerFieldSet::AtLeast => ObjectTy::at_least(fields),
                })
            }
            SerTy::Tuple { elems } => Ty::Tuple(elems.iter().map(|e| e.to_ty(interner)).collect()),
            SerTy::Fn {
                params,
                ret,
                effect,
            } => Ty::Fn {
                params: params
                    .iter()
                    .map(|p| crate::ty::Param::new(interner.intern("_"), p.ty.to_ty(interner)))
                    .collect(),
                ret: Box::new(ret.to_ty(interner)),
                captures: vec![],
                effect: EffectTerm::Known(ser_to_effect(effect, interner)),
            },
            SerTy::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => Ty::UserDefined {
                id: ser_to_qref(id, interner),
                type_args: type_args.iter().map(|t| ser_to_arg(t, interner)).collect(),
                effect_args: effect_args
                    .iter()
                    .map(|e| ser_to_effect_arg(e, interner))
                    .collect(),
                identity_args: identity_args
                    .iter()
                    .map(|i| IdentityTerm::Known(IdentityId::from_raw(*i as usize)))
                    .collect(),
            },
            SerTy::Option { inner } => Ty::Option(Box::new(inner.to_ty(interner))),
            SerTy::Result { ok, err } => {
                Ty::Result(Box::new(ok.to_ty(interner)), Box::new(err.to_ty(interner)))
            }
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
                home: crate::ty::Home::NONE,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::QualifiedRef;

    #[test]
    fn effect_survives_serialization() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: Effect::IDEMPOTENT.into(),
        };
        assert_eq!(
            fn_ty.to_ser(&i).to_ty(&i).effect(),
            Some(Effect::IDEMPOTENT)
        );

        let ud = Ty::UserDefined {
            id: QualifiedRef::root(i.intern("Iterator")),
            type_args: vec![TypeArg::uniform(Ty::I64)],
            effect_args: vec![EffectArg::specialized(Effect::OPAQUE.into())],
            identity_args: vec![],
        };
        assert_eq!(ud.to_ser(&i).to_ty(&i), ud);
    }
}
