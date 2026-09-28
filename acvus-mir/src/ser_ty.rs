//! Serializable type representation.
//!
//! [`SerTy`] mirrors a concrete type but replaces [`Astr`] (interned
//! strings) with [`String`] and the type table's ids with nested values,
//! making it `Serialize + Deserialize`. This is used at storage boundaries
//! (BlobStore, JSON export) where types must roundtrip through
//! serialization without an interner or a table.
//!
//! Conversion:
//! - `to_ser(types, ty, interner) -> SerTy` - resolve all Astr to String.
//! - `SerTy::to_ty(types, interner) -> Ty` - re-intern all String to Astr,
//!   and every part into the table.
//!
//! [`Astr`]: acvus_utils::Astr

use std::collections::BTreeMap;

use acvus_utils::{Interner, LocalIdOps, QualifiedRef};
use serde::{Deserialize, Serialize};

use crate::ir::{known_effect, known_len};
use crate::ty::{
    Alignment, Calls, Concrete, Contexts, Declared, Effect, EffectArg, EffectTerm, Flow, FlowEnd, FlowTerm, Flows,
    Holds, IdentityId, IdentityTerm, IntTy, Labelled, LenTerm, ObjectTy, Reissue, SplitCost, Ty, TyKind,
    TypeArg, Types,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerQualifiedRef {
    pub namespace: Option<String>,
    pub name: String,
    /// The declaring script and the instance, for a script's `fn`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<(String, u32)>,
}

fn qref_to_ser(r: &QualifiedRef, interner: &Interner) -> SerQualifiedRef {
    SerQualifiedRef {
        namespace: r.namespace.map(|ns| interner.resolve(ns).to_string()),
        name: interner.resolve(r.name).to_string(),
        scope: r
            .scope
            .map(|scope| (interner.resolve(scope.script).to_string(), scope.instance)),
    }
}

fn ser_to_qref(r: &SerQualifiedRef, interner: &Interner) -> QualifiedRef {
    QualifiedRef {
        namespace: r.namespace.as_ref().map(|ns| interner.intern(ns)),
        name: interner.intern(&r.name),
        scope: r.scope.as_ref().map(|(script, instance)| acvus_utils::FnScope {
            script: interner.intern(script),
            instance: *instance,
        }),
    }
}

/// An `Effect` with its context names written out.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerEffect {
    pub reissue: Reissue,
    pub split: SplitCost,
    pub commutes: bool,
    pub reads: Vec<SerQualifiedRef>,
    pub writes: Vec<SerQualifiedRef>,
}

fn effect_to_ser(e: &Effect, interner: &Interner) -> SerEffect {
    SerEffect {
        reissue: e.reissue(),
        split: e.split(),
        commutes: e.commutes(),
        reads: e.reads().iter().map(|q| qref_to_ser(q, interner)).collect(),
        writes: e.writes().iter().map(|q| qref_to_ser(q, interner)).collect(),
    }
}

fn ser_to_effect(e: &SerEffect, interner: &Interner) -> Effect {
    let contexts = |refs: &[SerQualifiedRef]| -> Contexts { refs.iter().map(|q| ser_to_qref(q, interner)).collect() };
    Effect::new(e.reissue, e.commutes, contexts(&e.reads), contexts(&e.writes), e.split)
}

/// A function type's flows, written out.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerFlows {
    Every,
    Listed { flows: Vec<SerFlow> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerFlow {
    pub to: FlowEnd,
    pub from: FlowEnd,
    pub alignment: SerAlignment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerAlignment {
    Aligned,
    /// The declaration the ends are read off, and per output segment the
    /// input segments it takes.
    Labelled {
        function: SerQualifiedRef,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        r#impl: Option<usize>,
        map: Vec<Vec<usize>>,
    },
    Any,
}

fn flows_to_ser(flows: &Flows, interner: &Interner) -> SerFlows {
    match flows {
        Flows::Every => SerFlows::Every,
        Flows::Listed(listed) => SerFlows::Listed {
            flows: listed
                .iter()
                .map(|flow| SerFlow {
                    to: flow.to,
                    from: flow.from,
                    alignment: match &flow.alignment {
                        Alignment::Aligned => SerAlignment::Aligned,
                        Alignment::Any => SerAlignment::Any,
                        Alignment::Labelled(labelled) => SerAlignment::Labelled {
                            function: qref_to_ser(&labelled.declared.function, interner),
                            r#impl: labelled.declared.r#impl,
                            map: labelled.map.iter().map(|takes| takes.to_vec()).collect(),
                        },
                    },
                })
                .collect(),
        },
    }
}

fn ser_to_flows(flows: &SerFlows, interner: &Interner) -> Flows {
    match flows {
        SerFlows::Every => Flows::Every,
        SerFlows::Listed { flows } => Flows::of(flows.iter().map(|flow| Flow {
            to: flow.to,
            from: flow.from,
            alignment: match &flow.alignment {
                SerAlignment::Aligned => Alignment::Aligned,
                SerAlignment::Labelled {
                    function,
                    r#impl,
                    map,
                } => Alignment::Labelled(Box::new(Labelled {
                    declared: Declared {
                        function: ser_to_qref(function, interner),
                        r#impl: *r#impl,
                    },
                    map: map.iter().map(|takes| takes.clone().into_boxed_slice()).collect(),
                })),
                SerAlignment::Any => Alignment::Any,
            },
        })),
    }
}

/// A type argument of a user-defined type, with its mark (types §8).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "repr", rename_all = "camelCase")]
pub enum SerTypeArg {
    Uniform { ty: SerTy },
    Specialized { ty: SerTy },
}

/// A recorded identity argument, written as its index. A record whose index
/// is outside the identity space is refused when it is read, so a held
/// `SerIdentity` is always an [`IdentityId`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(try_from = "u32", into = "u32")]
pub struct SerIdentity(IdentityId);

impl SerIdentity {
    pub fn id(self) -> IdentityId {
        self.0
    }
}

impl From<IdentityId> for SerIdentity {
    fn from(id: IdentityId) -> Self {
        SerIdentity(id)
    }
}

impl TryFrom<u32> for SerIdentity {
    type Error = &'static str;

    fn try_from(index: u32) -> Result<Self, Self::Error> {
        IdentityId::try_from_raw(index as usize)
            .map(SerIdentity)
            .ok_or("an identity index is at most u32::MAX - 1")
    }
}

impl From<SerIdentity> for u32 {
    fn from(identity: SerIdentity) -> Self {
        // An `IdentityId` stores index + 1 as a `NonZero<u32>`, so its
        // index is below `u32::MAX`.
        identity.0.to_raw() as u32
    }
}

/// An effect argument of a user-defined type, with its representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "repr", rename_all = "camelCase")]
pub enum SerEffectArg {
    Uniform { effect: SerEffect },
    Specialized { effect: SerEffect },
}

fn arg_to_ser(types: &Types<Concrete>, arg: &TypeArg<Concrete>, interner: &Interner) -> SerTypeArg {
    match arg {
        TypeArg::Uniform(ty) => SerTypeArg::Uniform {
            ty: to_ser(types, *ty, interner),
        },
        TypeArg::Specialized(ty) => SerTypeArg::Specialized {
            ty: to_ser(types, *ty, interner),
        },
        TypeArg::Open(v, _) => match *v {},
    }
}

fn ser_to_arg(types: &mut Types<Concrete>, arg: &SerTypeArg, interner: &Interner) -> TypeArg<Concrete> {
    match arg {
        SerTypeArg::Uniform { ty } => TypeArg::Uniform(ty.to_ty(types, interner)),
        SerTypeArg::Specialized { ty } => TypeArg::Specialized(ty.to_ty(types, interner)),
    }
}

fn effect_arg_to_ser(arg: &EffectArg<Concrete>, interner: &Interner) -> SerEffectArg {
    let effect = effect_to_ser(known_effect(arg.effect()), interner);
    match arg {
        EffectArg::Uniform(_) => SerEffectArg::Uniform { effect },
        EffectArg::Specialized(_) => SerEffectArg::Specialized { effect },
    }
}

fn ser_to_effect_arg(arg: &SerEffectArg, interner: &Interner) -> EffectArg<Concrete> {
    match arg {
        SerEffectArg::Uniform { effect } => EffectArg::Uniform(EffectTerm::Known(ser_to_effect(effect, interner))),
        SerEffectArg::Specialized { effect } => {
            EffectArg::Specialized(EffectTerm::Known(ser_to_effect(effect, interner)))
        }
    }
}

/// Whether an object type is structural or names a declaration.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SerObjectKind {
    #[default]
    Structural,
    Declared {
        /// The declaration's namespace; absent at the root.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<std::string::String>,
        name: std::string::String,
    },
}

/// Serializable mirror of a concrete type.
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
    Array {
        len: usize,
        elem: Box<SerTy>,
    },
    Object {
        #[serde(default)]
        object: SerObjectKind,
        fields: BTreeMap<std::string::String, SerTy>,
    },
    Tuple {
        elems: Vec<SerTy>,
    },
    Fn {
        params: Vec<SerTy>,
        ret: Box<SerTy>,
        effect: SerEffect,
        flows: SerFlows,
        calls: SerCalls,
        holds: SerHolds,
    },
    UserDefined {
        id: SerQualifiedRef,
        type_args: Vec<SerTypeArg>,
        effect_args: Vec<SerEffectArg>,
        identity_args: Vec<SerIdentity>,
    },
    Option {
        inner: Box<SerTy>,
    },
    Result {
        ok: Box<SerTy>,
        err: Box<SerTy>,
    },
    Enum {
        /// The enum's namespace; absent at the root.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<std::string::String>,
        name: std::string::String,
        variants: BTreeMap<std::string::String, Option<Box<SerTy>>>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SerCalls {
    Once,
    Many,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SerHolds {
    Nothing,
    Shared,
    Mut,
}

/// `ty` as a serializable value, every interned string resolved.
pub fn to_ser(types: &Types<Concrete>, ty: Ty, interner: &Interner) -> SerTy {
    acvus_utils::grow(|| to_ser_level(types, ty, interner))
}

fn to_ser_level(types: &Types<Concrete>, ty: Ty, interner: &Interner) -> SerTy {
    let part = |ty: Ty| Box::new(to_ser(types, ty, interner));
    match types.kind(ty) {
        TyKind::Int(k) => SerTy::Int { width: *k },
        TyKind::Order => SerTy::Order,
        TyKind::Float => SerTy::Float,
        TyKind::Char => SerTy::Char,
        TyKind::String => SerTy::String,
        TyKind::Bool => SerTy::Bool,
        TyKind::Unit => SerTy::Unit,
        TyKind::Never => SerTy::Never,
        TyKind::Array(elem, len) => SerTy::Array {
            len: known_len(len),
            elem: part(*elem),
        },
        TyKind::Object(object) => SerTy::Object {
            object: match object {
                ObjectTy::Declared(name, _) => SerObjectKind::Declared {
                    namespace: name.namespace.map(|ns| interner.resolve(ns).to_string()),
                    name: interner.resolve(name.name).to_string(),
                },
                ObjectTy::Structural(_, ()) => SerObjectKind::Structural,
            },
            fields: object
                .fields()
                .iter()
                .map(|(k, v)| (interner.resolve(*k).to_string(), to_ser(types, *v, interner)))
                .collect(),
        },
        TyKind::Tuple(elems) => SerTy::Tuple {
            elems: elems.iter().map(|e| to_ser(types, *e, interner)).collect(),
        },
        TyKind::Fn {
            params,
            ret,
            effect,
            flows,
            calls,
            holds,
        } => SerTy::Fn {
            params: params
                .iter()
                .map(|param| to_ser(types, *param, interner))
                .collect(),
            ret: part(*ret),
            effect: effect_to_ser(known_effect(effect), interner),
            flows: flows_to_ser(
                match flows {
                    FlowTerm::Known(flows) => flows,
                    FlowTerm::Var(var) => match *var {},
                },
                interner,
            ),
            calls: match calls {
                Calls::Once => SerCalls::Once,
                Calls::Many => SerCalls::Many,
            },
            holds: match holds {
                Holds::Nothing => SerHolds::Nothing,
                Holds::Shared => SerHolds::Shared,
                Holds::Mut => SerHolds::Mut,
            },
        },
        TyKind::UserDefined {
            id,
            type_args,
            effect_args,
            identity_args,
        } => SerTy::UserDefined {
            id: qref_to_ser(id, interner),
            type_args: type_args.iter().map(|t| arg_to_ser(types, t, interner)).collect(),
            effect_args: effect_args.iter().map(|e| effect_arg_to_ser(e, interner)).collect(),
            identity_args: identity_args
                .iter()
                .map(|identity| match identity {
                    IdentityTerm::Known(id) => SerIdentity::from(*id),
                    IdentityTerm::Var(var) => match *var {},
                })
                .collect(),
        },
        TyKind::Option(inner) => SerTy::Option { inner: part(*inner) },
        TyKind::Result(ok, err) => SerTy::Result {
            ok: part(*ok),
            err: part(*err),
        },
        TyKind::Enum { name, variants, .. } => SerTy::Enum {
            namespace: name.namespace.map(|ns| interner.resolve(ns).to_string()),
            name: interner.resolve(name.name).to_string(),
            variants: variants
                .iter()
                .map(|(k, v)| (interner.resolve(*k).to_string(), v.map(part)))
                .collect(),
        },
        TyKind::Handle(..) => todo!("Handle serialization not yet implemented"),
        TyKind::Ref(..) | TyKind::Slice(..) | TyKind::Str => {
            todo!("Ref serialization not yet implemented")
        }
        TyKind::Var(v) => match *v {},
    }
}

impl SerTy {
    /// The type back, every string re-interned and every part interned
    /// into `types`.
    pub fn to_ty(&self, types: &mut Types<Concrete>, interner: &Interner) -> Ty {
        acvus_utils::grow(|| self.to_ty_level(types, interner))
    }

    fn to_ty_level(&self, types: &mut Types<Concrete>, interner: &Interner) -> Ty {
        let kind = match self {
            SerTy::Int { width } => TyKind::Int(*width),
            SerTy::Order => TyKind::Order,
            SerTy::Float => TyKind::Float,
            SerTy::Char => TyKind::Char,
            SerTy::String => TyKind::String,
            SerTy::Bool => TyKind::Bool,
            SerTy::Unit => TyKind::Unit,
            SerTy::Never => TyKind::Never,
            SerTy::Array { len, elem } => TyKind::Array(elem.to_ty(types, interner), LenTerm::Known(*len)),
            SerTy::Object { object, fields } => {
                let mut fields: Vec<_> = fields
                    .iter()
                    .map(|(k, v)| (interner.intern(k), v.to_ty(types, interner)))
                    .collect();
                fields.sort_by_key(|(name, _)| *name);
                let fields = fields.into_boxed_slice();
                TyKind::Object(match object {
                    SerObjectKind::Declared { namespace, name } => ObjectTy::Declared(
                        QualifiedRef {
                            namespace: namespace.as_deref().map(|ns| interner.intern(ns)),
                            name: interner.intern(name),
                            scope: None,
                        },
                        fields,
                    ),
                    SerObjectKind::Structural => ObjectTy::Structural(fields, ()),
                })
            }
            SerTy::Tuple { elems } => TyKind::Tuple(elems.iter().map(|e| e.to_ty(types, interner)).collect()),
            SerTy::Fn {
                params,
                ret,
                effect,
                flows,
                calls,
                holds,
            } => TyKind::Fn {
                params: params
                    .iter()
                    .map(|param| param.to_ty(types, interner))
                    .collect(),
                ret: ret.to_ty(types, interner),
                effect: EffectTerm::Known(ser_to_effect(effect, interner)),
                flows: FlowTerm::Known(ser_to_flows(flows, interner)),
                calls: match calls {
                    SerCalls::Once => Calls::Once,
                    SerCalls::Many => Calls::Many,
                },
                holds: match holds {
                    SerHolds::Nothing => Holds::Nothing,
                    SerHolds::Shared => Holds::Shared,
                    SerHolds::Mut => Holds::Mut,
                },
            },
            SerTy::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => TyKind::UserDefined {
                id: ser_to_qref(id, interner),
                type_args: type_args.iter().map(|t| ser_to_arg(types, t, interner)).collect(),
                effect_args: effect_args.iter().map(|e| ser_to_effect_arg(e, interner)).collect(),
                identity_args: identity_args.iter().map(|i| IdentityTerm::Known(i.id())).collect(),
            },
            SerTy::Option { inner } => TyKind::Option(inner.to_ty(types, interner)),
            SerTy::Result { ok, err } => TyKind::Result(ok.to_ty(types, interner), err.to_ty(types, interner)),
            SerTy::Enum {
                namespace,
                name,
                variants,
            } => {
                let mut variants: Vec<_> = variants
                    .iter()
                    .map(|(k, v)| (interner.intern(k), v.as_ref().map(|t| t.to_ty(types, interner))))
                    .collect();
                variants.sort_by_key(|(tag, _)| *tag);
                TyKind::Enum {
                    name: QualifiedRef {
                        namespace: namespace.as_deref().map(|ns| interner.intern(ns)),
                        name: interner.intern(name),
                        scope: None,
                    },
                    variants: variants.into_boxed_slice(),
                    label: (),
                }
            }
        };
        types.intern(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::QualifiedRef;

    #[test]
    fn effect_survives_serialization() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let idempotent = Effect::new(
            Reissue::Idempotent,
            true,
            Contexts::new(),
            Contexts::new(),
            SplitCost::Unavailable,
        );
        let fn_ty = types.intern(TyKind::Fn {
            params: Box::new([]),
            ret: i64_ty,
            effect: EffectTerm::Known(idempotent.clone()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });

        let round_tripped = to_ser(&types, fn_ty, &i).to_ty(&mut types, &i);
        assert_eq!(round_tripped, fn_ty);
        let TyKind::Fn { effect, .. } = types.kind(round_tripped) else {
            panic!("expected a function type");
        };
        assert_eq!(known_effect(effect), &idempotent);

        let ud = types.intern(TyKind::UserDefined {
            id: QualifiedRef::root(i.intern("Iterator")),
            type_args: Box::new([TypeArg::Uniform(i64_ty)]),
            effect_args: Box::new([EffectArg::Specialized(EffectTerm::Known(Effect::opaque()))]),
            identity_args: Box::new([]),
        });
        let round_tripped_ud = to_ser(&types, ud, &i).to_ty(&mut types, &i);
        assert_eq!(round_tripped_ud, ud);
    }
}
