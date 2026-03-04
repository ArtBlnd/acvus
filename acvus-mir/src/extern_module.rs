use std::collections::{HashMap, HashSet};

use crate::ty::Ty;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternFnId(pub u32);

/// Definition of a single external function.
#[derive(Debug, Clone)]
pub struct ExternFnDef {
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub effectful: bool,
}

/// A named collection of external function definitions.
#[derive(Debug, Clone)]
pub struct ExternModule {
    pub name: String,
    fns: HashMap<String, ExternFnDef>,
    opaque_types: HashSet<String>,
}

impl ExternModule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fns: HashMap::new(),
            opaque_types: HashSet::new(),
        }
    }

    pub fn add_opaque(&mut self, name: impl Into<String>) -> &mut Self {
        let name = name.into();
        assert!(
            self.opaque_types.insert(name.clone()),
            "duplicate opaque type in ExternModule '{}': {name}",
            self.name,
        );
        self
    }

    pub fn opaque_types(&self) -> &HashSet<String> {
        &self.opaque_types
    }

    pub fn add_fn(
        &mut self,
        name: impl Into<String>,
        params: Vec<Ty>,
        ret: Ty,
        effectful: bool,
    ) -> &mut Self {
        let name = name.into();
        assert!(
            !self.fns.contains_key(&name),
            "duplicate function in ExternModule '{}': {name}",
            self.name,
        );
        self.fns.insert(
            name,
            ExternFnDef {
                params,
                ret,
                effectful,
            },
        );
        self
    }

    pub fn fns(&self) -> &HashMap<String, ExternFnDef> {
        &self.fns
    }
}

/// Registry that merges multiple ExternModules.
/// Panics on duplicate function names across modules.
#[derive(Debug, Clone)]
pub struct ExternRegistry {
    opaque_types: HashSet<String>,
    /// ID-indexed storage: ExternFnId(n) → (name, def).
    fn_list: Vec<(String, ExternFnDef)>,
    /// Name → ExternFnId mapping.
    fn_id_index: HashMap<String, ExternFnId>,
}

impl Default for ExternRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternRegistry {
    pub fn new() -> Self {
        Self {
            opaque_types: HashSet::new(),
            fn_list: Vec::new(),
            fn_id_index: HashMap::new(),
        }
    }

    pub fn register(&mut self, module: &ExternModule) -> &mut Self {
        for name in &module.opaque_types {
            assert!(
                self.opaque_types.insert(name.clone()),
                "duplicate opaque type '{name}' (from module '{}')",
                module.name,
            );
        }
        for (name, def) in &module.fns {
            assert!(
                !self.fn_id_index.contains_key(name),
                "duplicate extern function '{name}' (from module '{}')",
                module.name,
            );
            let id = ExternFnId(self.fn_list.len() as u32);
            self.fn_list.push((name.clone(), def.clone()));
            self.fn_id_index.insert(name.clone(), id);
        }
        self
    }

    pub fn get(&self, name: &str) -> Option<&ExternFnDef> {
        let id = self.fn_id_index.get(name)?;
        Some(&self.fn_list[id.0 as usize].1)
    }

    pub fn resolve(&self, name: &str) -> Option<ExternFnId> {
        self.fn_id_index.get(name).copied()
    }

    pub fn get_by_id(&self, id: ExternFnId) -> &ExternFnDef {
        &self.fn_list[id.0 as usize].1
    }

    pub fn name_by_id(&self, id: ExternFnId) -> &str {
        &self.fn_list[id.0 as usize].0
    }

    /// Build a name table mapping ExternFnId → name for the MirModule.
    pub fn build_name_table(&self) -> HashMap<ExternFnId, String> {
        self.fn_list
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (ExternFnId(i as u32), name.clone()))
            .collect()
    }

    pub fn has_opaque(&self, name: &str) -> bool {
        self.opaque_types.contains(name)
    }

    pub fn fns(&self) -> impl Iterator<Item = (&str, &ExternFnDef)> {
        self.fn_list.iter().map(|(name, def)| (name.as_str(), def))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup() {
        let mut module = ExternModule::new("math");
        module.add_fn("abs", vec![Ty::Int], Ty::Int, false);

        let mut registry = ExternRegistry::new();
        registry.register(&module);

        let def = registry.get("abs").unwrap();
        assert_eq!(def.params, vec![Ty::Int]);
        assert_eq!(def.ret, Ty::Int);
        assert!(!def.effectful);
    }

    #[test]
    fn multiple_modules() {
        let mut math = ExternModule::new("math");
        math.add_fn("abs", vec![Ty::Int], Ty::Int, false);

        let mut io = ExternModule::new("io");
        io.add_fn("fetch", vec![Ty::String], Ty::String, true);

        let mut registry = ExternRegistry::new();
        registry.register(&math).register(&io);

        assert!(registry.get("abs").is_some());
        assert!(registry.get("fetch").is_some());
        assert!(registry.get("fetch").unwrap().effectful);
    }

    #[test]
    #[should_panic(expected = "duplicate extern function")]
    fn duplicate_across_modules_panics() {
        let mut a = ExternModule::new("a");
        a.add_fn("foo", vec![], Ty::Unit, false);

        let mut b = ExternModule::new("b");
        b.add_fn("foo", vec![], Ty::Int, false);

        let mut registry = ExternRegistry::new();
        registry.register(&a).register(&b);
    }

    #[test]
    #[should_panic(expected = "duplicate function in ExternModule")]
    fn duplicate_within_module_panics() {
        let mut module = ExternModule::new("test");
        module.add_fn("foo", vec![], Ty::Unit, false);
        module.add_fn("foo", vec![], Ty::Int, false);
    }
}
