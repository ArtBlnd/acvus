use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::variant::EnumDef;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UserTypeId(pub u32);

pub struct UserTypeRegistry {
    enums: Vec<EnumDef>,
    name_index: FxHashMap<Astr, UserTypeId>,
}

impl Default for UserTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl UserTypeRegistry {
    pub fn new() -> Self {
        Self {
            enums: Vec::new(),
            name_index: FxHashMap::default(),
        }
    }

    pub fn register(&mut self, def: EnumDef) -> UserTypeId {
        let id = UserTypeId(self.enums.len() as u32);
        assert!(
            !self.name_index.contains_key(&def.name),
            "duplicate user type: {:?}",
            def.name,
        );
        self.name_index.insert(def.name, id);
        self.enums.push(def);
        id
    }

    pub fn get(&self, id: UserTypeId) -> &EnumDef {
        &self.enums[id.0 as usize]
    }

    pub fn resolve_name(&self, name: Astr) -> Option<UserTypeId> {
        self.name_index.get(&name).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (UserTypeId, &EnumDef)> {
        self.enums
            .iter()
            .enumerate()
            .map(|(i, def)| (UserTypeId(i as u32), def))
    }
}
