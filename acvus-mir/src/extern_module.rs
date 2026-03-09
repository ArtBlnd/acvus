use std::collections::{HashMap, HashSet};

use acvus_utils::Astr;

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
    pub name: Astr,
    fns: HashMap<Astr, ExternFnDef>,
    opaque_types: HashSet<Astr>,
}

impl ExternModule {
    pub fn new(name: Astr) -> Self {
        Self {
            name,
            fns: HashMap::new(),
            opaque_types: HashSet::new(),
        }
    }

    pub fn add_opaque(&mut self, name: Astr) -> &mut Self {
        assert!(
            self.opaque_types.insert(name),
            "duplicate opaque type in ExternModule '{}': {name}",
            self.name,
        );
        self
    }

    pub fn opaque_types(&self) -> &HashSet<Astr> {
        &self.opaque_types
    }

    pub fn add_fn(
        &mut self,
        name: Astr,
        params: Vec<Ty>,
        ret: Ty,
        effectful: bool,
    ) -> &mut Self {
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

    pub fn fns(&self) -> &HashMap<Astr, ExternFnDef> {
        &self.fns
    }
}

/// Registry that merges multiple ExternModules.
/// Panics on duplicate function names across modules.
#[derive(Debug, Clone)]
pub struct ExternRegistry {
    opaque_types: HashSet<Astr>,
    /// ID-indexed storage: ExternFnId(n) -> (name, def).
    fn_list: Vec<(Astr, ExternFnDef)>,
    /// Name -> ExternFnId mapping.
    fn_id_index: HashMap<Astr, ExternFnId>,
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
                self.opaque_types.insert(*name),
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
            self.fn_list.push((*name, def.clone()));
            self.fn_id_index.insert(*name, id);
        }
        self
    }

    pub fn get(&self, name: Astr) -> Option<&ExternFnDef> {
        let id = self.fn_id_index.get(&name)?;
        Some(&self.fn_list[id.0 as usize].1)
    }

    pub fn resolve(&self, name: Astr) -> Option<ExternFnId> {
        self.fn_id_index.get(&name).copied()
    }

    pub fn get_by_id(&self, id: ExternFnId) -> &ExternFnDef {
        &self.fn_list[id.0 as usize].1
    }

    pub fn name_by_id(&self, id: ExternFnId) -> Astr {
        self.fn_list[id.0 as usize].0
    }

    /// Build a name table mapping ExternFnId -> name for the MirModule.
    pub fn build_name_table(&self) -> HashMap<ExternFnId, Astr> {
        self.fn_list
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (ExternFnId(i as u32), *name))
            .collect()
    }

    pub fn has_opaque(&self, name: Astr) -> bool {
        self.opaque_types.contains(&name)
    }

    pub fn fns(&self) -> impl Iterator<Item = (Astr, &ExternFnDef)> {
        self.fn_list.iter().map(|(name, def)| (*name, def))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    #[test]
    fn register_and_lookup() {
        let interner = Interner::new();
        let mut module = ExternModule::new(interner.intern("math"));
        module.add_fn(interner.intern("abs"), vec![Ty::Int], Ty::Int, false);

        let mut registry = ExternRegistry::new();
        registry.register(&module);

        let def = registry.get(interner.intern("abs")).unwrap();
        assert_eq!(def.params, vec![Ty::Int]);
        assert_eq!(def.ret, Ty::Int);
        assert!(!def.effectful);
    }

    #[test]
    fn multiple_modules() {
        let interner = Interner::new();
        let mut math = ExternModule::new(interner.intern("math"));
        math.add_fn(interner.intern("abs"), vec![Ty::Int], Ty::Int, false);

        let mut io = ExternModule::new(interner.intern("io"));
        io.add_fn(interner.intern("fetch"), vec![Ty::String], Ty::String, true);

        let mut registry = ExternRegistry::new();
        registry.register(&math).register(&io);

        assert!(registry.get(interner.intern("abs")).is_some());
        assert!(registry.get(interner.intern("fetch")).is_some());
        assert!(registry.get(interner.intern("fetch")).unwrap().effectful);
    }

    #[test]
    #[should_panic(expected = "duplicate extern function")]
    fn duplicate_across_modules_panics() {
        let interner = Interner::new();
        let mut a = ExternModule::new(interner.intern("a"));
        a.add_fn(interner.intern("foo"), vec![], Ty::Unit, false);

        let mut b = ExternModule::new(interner.intern("b"));
        b.add_fn(interner.intern("foo"), vec![], Ty::Int, false);

        let mut registry = ExternRegistry::new();
        registry.register(&a).register(&b);
    }

    #[test]
    #[should_panic(expected = "duplicate function in ExternModule")]
    fn duplicate_within_module_panics() {
        let interner = Interner::new();
        let mut module = ExternModule::new(interner.intern("test"));
        module.add_fn(interner.intern("foo"), vec![], Ty::Unit, false);
        module.add_fn(interner.intern("foo"), vec![], Ty::Int, false);
    }
}
