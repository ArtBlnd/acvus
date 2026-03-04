use std::collections::HashMap;

use crate::ty::{Ty, TySubst};
use crate::user_type::UserTypeId;

/// Numeric identifier for a variant tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariantTagId(pub u16);

/// Payload specification for a variant constructor.
#[derive(Debug, Clone)]
pub enum VariantPayload {
    /// No payload (e.g. `None`).
    None,
    /// Payload type is the i-th type parameter of the parent enum.
    TypeParam(usize),
}

/// Definition of a single variant within an enum.
#[derive(Debug, Clone)]
pub struct VariantDef {
    pub tag: String,
    pub payload: VariantPayload,
}

/// Definition of an enum (tagged union) type.
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: String,
    pub type_param_count: usize,
    pub variants: Vec<VariantDef>,
}

/// Registry of all known enum types, indexed by variant tag.
#[derive(Debug, Clone)]
pub struct VariantRegistry {
    enums: Vec<EnumDef>,
    /// Flat tag → VariantTagId (unqualified lookup).
    tag_index: HashMap<String, VariantTagId>,
    /// (enum_name, tag) → VariantTagId (qualified lookup).
    enum_tag_index: HashMap<(String, String), VariantTagId>,
    /// VariantTagId(n) → (enum index, variant index).
    tag_id_index: Vec<(usize, usize)>,
}

impl Default for VariantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VariantRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            enums: Vec::new(),
            tag_index: HashMap::new(),
            enum_tag_index: HashMap::new(),
            tag_id_index: Vec::new(),
        };
        registry.register_builtins();
        registry
    }

    fn register_builtins(&mut self) {
        self.register(EnumDef {
            name: "Option".into(),
            type_param_count: 1,
            variants: vec![
                VariantDef {
                    tag: "Some".into(),
                    payload: VariantPayload::TypeParam(0),
                },
                VariantDef {
                    tag: "None".into(),
                    payload: VariantPayload::None,
                },
            ],
        });
    }

    pub fn register(&mut self, def: EnumDef) {
        let enum_idx = self.enums.len();
        for (var_idx, variant) in def.variants.iter().enumerate() {
            let tag_id = VariantTagId(self.tag_id_index.len() as u16);
            self.tag_id_index.push((enum_idx, var_idx));

            let prev = self.tag_index.insert(variant.tag.clone(), tag_id);
            assert!(prev.is_none(), "duplicate variant tag: {}", variant.tag);

            self.enum_tag_index
                .insert((def.name.clone(), variant.tag.clone()), tag_id);
        }
        self.enums.push(def);
    }

    /// Resolve a variant tag, optionally qualified by enum name.
    /// - `resolve_tag(None, "Some")` → flat lookup
    /// - `resolve_tag(Some("Color"), "Red")` → qualified lookup
    pub fn resolve_tag(&self, enum_name: Option<&str>, tag: &str) -> Option<VariantTagId> {
        match enum_name {
            Some(name) => self
                .enum_tag_index
                .get(&(name.to_string(), tag.to_string()))
                .copied(),
            None => self.tag_index.get(tag).copied(),
        }
    }

    /// Resolve a variant tag to its parent enum and variant definition.
    pub fn resolve(&self, tag: &str) -> Option<(&EnumDef, &VariantDef)> {
        let id = self.tag_index.get(tag)?;
        Some(self.get_tag_info(*id))
    }

    /// Get enum and variant definition from a VariantTagId.
    pub fn get_tag_info(&self, id: VariantTagId) -> (&EnumDef, &VariantDef) {
        let (enum_idx, var_idx) = self.tag_id_index[id.0 as usize];
        (&self.enums[enum_idx], &self.enums[enum_idx].variants[var_idx])
    }

    /// Get the tag name string from a VariantTagId.
    pub fn tag_name(&self, id: VariantTagId) -> &str {
        let (_, vdef) = self.get_tag_info(id);
        &vdef.tag
    }

    /// Build a tag name table for the MirModule: index `i` → tag name for `VariantTagId(i)`.
    pub fn build_tag_names(&self) -> Vec<String> {
        self.tag_id_index
            .iter()
            .map(|&(enum_idx, var_idx)| self.enums[enum_idx].variants[var_idx].tag.clone())
            .collect()
    }
}

/// Construct a `Ty` for the given enum with resolved type parameters.
pub fn make_enum_ty(
    enum_name: &str,
    type_params: &[Ty],
    subst: &TySubst,
    user_type_id: Option<UserTypeId>,
) -> Ty {
    match enum_name {
        "Option" => Ty::Option(Box::new(subst.resolve(&type_params[0]))),
        _ => match user_type_id {
            Some(id) => Ty::UserType(id),
            None => panic!("unknown enum type: {enum_name}"),
        },
    }
}
