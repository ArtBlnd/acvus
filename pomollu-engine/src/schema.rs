use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use tsify::Tsify;

use crate::error;

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub enum Mode {
    Template,
    Script,
}

#[derive(Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeOptions {
    pub source: String,
    pub mode: Mode,
    #[serde(default)]
    pub context_types: FxHashMap<String, TypeDesc>,
    #[serde(default)]
    pub expected_tail: Option<TypeDesc>,
    #[serde(default)]
    pub known_values: FxHashMap<String, String>,
}

#[derive(Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TypecheckOptions {
    pub source: String,
    pub mode: Mode,
    #[serde(default)]
    pub context_types: FxHashMap<String, TypeDesc>,
    #[serde(default)]
    pub expected_tail: Option<TypeDesc>,
}

#[derive(Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TypecheckNodesOptions {
    pub nodes: Vec<crate::convert::WebNode>,
    pub injected_types: FxHashMap<String, TypeDesc>,
}

#[derive(Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateOptions {
    pub source: String,
    pub mode: Mode,
    #[serde(default)]
    pub context: FxHashMap<String, JsConcreteValue>,
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeResult {
    pub ok: bool,
    pub errors: Vec<error::EngineError>,
    pub context_keys: Vec<ContextKey>,
    pub tail_type: TypeDesc,
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ContextKey {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: TypeDesc,
    pub status: ContextKeyStatus,
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub enum ContextKeyStatus {
    Eager,
    Lazy,
    Pruned,
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct CheckResult {
    pub ok: bool,
    pub errors: Vec<error::EngineError>,
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TypecheckNodesResult {
    pub env_errors: Vec<error::EngineError>,
    pub context_types: FxHashMap<String, TypeDesc>,
    pub node_locals: FxHashMap<String, NodeLocalTypes>,
    pub node_errors: FxHashMap<String, NodeErrors>,
}

impl TypecheckNodesResult {
    pub fn fail(errors: Vec<error::EngineError>) -> Self {
        Self {
            env_errors: errors,
            context_types: FxHashMap::default(),
            node_locals: FxHashMap::default(),
            node_errors: FxHashMap::default(),
        }
    }
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct NodeLocalTypes {
    pub raw: TypeDesc,
    #[serde(rename = "self")]
    pub self_ty: TypeDesc,
}

#[derive(Default, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct NodeErrors {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<error::EngineError>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub initial_value: Vec<error::EngineError>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bind: Vec<error::EngineError>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub if_modified_key: Vec<error::EngineError>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub assert: Vec<error::EngineError>,
    #[serde(skip_serializing_if = "FxHashMap::is_empty")]
    pub messages: FxHashMap<String, Vec<error::EngineError>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub expr_source: Vec<error::EngineError>,
}

impl NodeErrors {
    pub fn is_empty(&self) -> bool {
        self.env.is_empty()
            && self.initial_value.is_empty()
            && self.bind.is_empty()
            && self.if_modified_key.is_empty()
            && self.assert.is_empty()
            && self.messages.is_empty()
            && self.expr_source.is_empty()
    }
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateResult {
    pub ok: bool,
    pub errors: Vec<error::EngineError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<JsConcreteValue>,
}

// ---------------------------------------------------------------------------
// Tree / turn types
// ---------------------------------------------------------------------------

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TurnNode {
    pub uuid: String,
    pub parent: Option<String>,
    pub depth: usize,
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TurnResult {
    pub value: JsConcreteValue,
    pub turn: TurnNode,
}

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TreeView {
    pub nodes: Vec<TurnNode>,
    pub cursor: String,
}

// ---------------------------------------------------------------------------
// JsRenderedDisplayEntry — Tsify wrapper for RenderedDisplayEntry
// ---------------------------------------------------------------------------

#[derive(Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct JsRenderedDisplayEntry {
    pub name: String,
    pub content: String,
}

impl From<acvus_orchestration::RenderedDisplayEntry> for JsRenderedDisplayEntry {
    fn from(e: acvus_orchestration::RenderedDisplayEntry) -> Self {
        Self {
            name: e.name,
            content: e.content,
        }
    }
}

#[derive(Serialize, Tsify)]
pub struct DisplayRenderResult(pub Vec<JsRenderedDisplayEntry>);

// ---------------------------------------------------------------------------
// JsConcreteValue — Tsify wrapper for acvus_interpreter::ConcreteValue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[serde(tag = "t")]
pub enum JsConcreteValue {
    Int { v: i64 },
    Float { v: f64 },
    String { v: std::string::String },
    Bool { v: bool },
    Unit,
    Range { start: i64, end: i64, inclusive: bool },
    List { items: Vec<JsConcreteValue> },
    Object { fields: Vec<(std::string::String, JsConcreteValue)> },
    Tuple { items: Vec<JsConcreteValue> },
    Byte { v: u8 },
    Variant { tag: std::string::String, payload: Option<Box<JsConcreteValue>> },
}

impl From<acvus_interpreter::ConcreteValue> for JsConcreteValue {
    fn from(cv: acvus_interpreter::ConcreteValue) -> Self {
        use acvus_interpreter::ConcreteValue as CV;
        match cv {
            CV::Int { v } => Self::Int { v },
            CV::Float { v } => Self::Float { v },
            CV::String { v } => Self::String { v },
            CV::Bool { v } => Self::Bool { v },
            CV::Unit => Self::Unit,
            CV::Range { start, end, inclusive } => Self::Range { start, end, inclusive },
            CV::List { items } => Self::List {
                items: items.into_iter().map(Into::into).collect(),
            },
            CV::Object { fields } => Self::Object {
                fields: fields.into_iter().map(|(k, v)| (k, v.into())).collect(),
            },
            CV::Tuple { items } => Self::Tuple {
                items: items.into_iter().map(Into::into).collect(),
            },
            CV::Byte { v } => Self::Byte { v },
            CV::Variant { tag, payload } => Self::Variant {
                tag,
                payload: payload.map(|p| Box::new((*p).into())),
            },
        }
    }
}

impl From<JsConcreteValue> for acvus_interpreter::ConcreteValue {
    fn from(jcv: JsConcreteValue) -> Self {
        use JsConcreteValue as JCV;
        match jcv {
            JCV::Int { v } => Self::Int { v },
            JCV::Float { v } => Self::Float { v },
            JCV::String { v } => Self::String { v },
            JCV::Bool { v } => Self::Bool { v },
            JCV::Unit => Self::Unit,
            JCV::Range { start, end, inclusive } => Self::Range { start, end, inclusive },
            JCV::List { items } => Self::List {
                items: items.into_iter().map(Into::into).collect(),
            },
            JCV::Object { fields } => Self::Object {
                fields: fields.into_iter().map(|(k, v)| (k, v.into())).collect(),
            },
            JCV::Tuple { items } => Self::Tuple {
                items: items.into_iter().map(Into::into).collect(),
            },
            JCV::Byte { v } => Self::Byte { v },
            JCV::Variant { tag, payload } => Self::Variant {
                tag,
                payload: payload.map(|p| Box::new((*p).into())),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// TypeDesc — bridging Ty ↔ JS
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Tsify)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum TypeDesc {
    #[serde(rename = "primitive")]
    Primitive { name: String },
    #[serde(rename = "option")]
    Option { inner: Box<TypeDesc> },
    #[serde(rename = "object")]
    Object { fields: Vec<TypeDescField> },
    #[serde(rename = "list")]
    List { elem: Box<TypeDesc> },
    #[serde(rename = "deque")]
    Deque { elem: Box<TypeDesc>, origin: TypeDescOrigin },
    #[serde(rename = "enum")]
    Enum {
        name: String,
        variants: Vec<TypeDescVariant>,
    },
    #[serde(rename = "unsupported")]
    Unsupported { raw: String },
}

#[derive(Serialize, Deserialize, Clone, Tsify)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum TypeDescOrigin {
    Concrete { id: u32 },
    Var { id: u32 },
}

#[derive(Serialize, Deserialize, Clone, Tsify)]
pub struct TypeDescField {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: TypeDesc,
}

#[derive(Serialize, Deserialize, Clone, Tsify)]
pub struct TypeDescVariant {
    pub tag: String,
    #[serde(rename = "hasPayload")]
    pub has_payload: bool,
    #[serde(rename = "payloadType", skip_serializing_if = "Option::is_none")]
    pub payload_type: Option<Box<TypeDesc>>,
}

// ---------------------------------------------------------------------------
// TypeDesc ↔ Ty conversion
// ---------------------------------------------------------------------------

pub fn ty_to_desc(interner: &Interner, ty: &Ty) -> TypeDesc {
    match ty {
        Ty::Int => TypeDesc::Primitive { name: "int".into() },
        Ty::Float => TypeDesc::Primitive { name: "float".into() },
        Ty::String => TypeDesc::Primitive { name: "string".into() },
        Ty::Bool => TypeDesc::Primitive { name: "bool".into() },
        Ty::Unit => TypeDesc::Unsupported { raw: "Unit".into() },
        Ty::Range => TypeDesc::Unsupported { raw: "Range".into() },
        Ty::Byte => TypeDesc::Unsupported { raw: "Byte".into() },
        Ty::Option(inner) => TypeDesc::Option {
            inner: Box::new(ty_to_desc(interner, inner)),
        },
        Ty::Object(fields) => {
            let mut desc_fields: Vec<TypeDescField> = fields
                .iter()
                .map(|(k, v)| TypeDescField {
                    name: interner.resolve(*k).to_string(),
                    ty: ty_to_desc(interner, v),
                })
                .collect();
            desc_fields.sort_by(|a, b| a.name.cmp(&b.name));
            TypeDesc::Object { fields: desc_fields }
        }
        Ty::Enum { name, variants } => {
            let mut desc_variants: Vec<TypeDescVariant> = variants
                .iter()
                .map(|(tag, payload)| TypeDescVariant {
                    tag: interner.resolve(*tag).to_string(),
                    has_payload: payload.is_some(),
                    payload_type: payload.as_ref().map(|p| Box::new(ty_to_desc(interner, p))),
                })
                .collect();
            desc_variants.sort_by(|a, b| a.tag.cmp(&b.tag));
            TypeDesc::Enum {
                name: interner.resolve(*name).to_string(),
                variants: desc_variants,
            }
        }
        Ty::List(inner) => TypeDesc::List {
            elem: Box::new(ty_to_desc(interner, inner)),
        },
        Ty::Deque(inner, origin) => TypeDesc::Deque {
            elem: Box::new(ty_to_desc(interner, inner)),
            origin: match origin {
                acvus_mir::ty::Origin::Concrete(id) => TypeDescOrigin::Concrete { id: *id },
                acvus_mir::ty::Origin::Var(id) => TypeDescOrigin::Var { id: *id },
            },
        },
        Ty::Var(_) | Ty::Infer | Ty::Error => TypeDesc::Unsupported { raw: "?".into() },
        Ty::Fn { .. } => TypeDesc::Unsupported { raw: "Fn".into() },
        Ty::Opaque(_) => TypeDesc::Unsupported { raw: "Opaque".into() },
        Ty::Tuple(_) => TypeDesc::Unsupported { raw: "Tuple".into() },
        Ty::Iterator(..) => TypeDesc::Unsupported { raw: "Iterator".into() },
        Ty::Sequence(..) => TypeDesc::Unsupported { raw: "Sequence".into() },
    }
}

pub fn desc_to_ty(interner: &Interner, desc: &TypeDesc) -> Ty {
    match desc {
        TypeDesc::Primitive { name } => match name.as_str() {
            "int" => Ty::Int,
            "float" => Ty::Float,
            "string" => Ty::String,
            "bool" => Ty::Bool,
            _ => Ty::Infer,
        },
        TypeDesc::Option { inner } => Ty::Option(Box::new(desc_to_ty(interner, inner))),
        TypeDesc::List { elem } => Ty::List(Box::new(desc_to_ty(interner, elem))),
        TypeDesc::Deque { elem, origin } => {
            let o = match origin {
                TypeDescOrigin::Concrete { id } => acvus_mir::ty::Origin::Concrete(*id),
                TypeDescOrigin::Var { id } => acvus_mir::ty::Origin::Var(*id),
            };
            Ty::Deque(Box::new(desc_to_ty(interner, elem)), o)
        }
        TypeDesc::Object { fields } => {
            let ty_fields: FxHashMap<Astr, Ty> = fields
                .iter()
                .map(|f| (interner.intern(&f.name), desc_to_ty(interner, &f.ty)))
                .collect();
            Ty::Object(ty_fields)
        }
        TypeDesc::Enum { name, variants } => {
            let ty_variants: FxHashMap<Astr, Option<Box<Ty>>> = variants
                .iter()
                .map(|v| {
                    let tag = interner.intern(&v.tag);
                    let payload = if v.has_payload {
                        let ty = v.payload_type.as_ref()
                            .map(|pt| desc_to_ty(interner, pt))
                            .unwrap_or(Ty::Infer);
                        Some(Box::new(ty))
                    } else {
                        None
                    };
                    (tag, payload)
                })
                .collect();
            Ty::Enum {
                name: interner.intern(name),
                variants: ty_variants,
            }
        }
        TypeDesc::Unsupported { .. } => Ty::Infer,
    }
}

/// Parse a simple type string (e.g. "string", "int", "bool") into a Ty.
/// Falls back to Ty::Infer for unknown types.
pub fn parse_type_string(_interner: &Interner, s: &str) -> Ty {
    match s {
        "string" => Ty::String,
        "int" => Ty::Int,
        "float" => Ty::Float,
        "bool" => Ty::Bool,
        "byte" => Ty::Byte,
        _ => Ty::Infer,
    }
}
