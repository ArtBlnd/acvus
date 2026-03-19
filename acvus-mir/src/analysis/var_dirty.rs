

use crate::hints::InstIdx;
use crate::ir::{InstKind, MirModule, ValueId};
use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::pass::AnalysisPass;
use crate::analysis::val_def::ValDefMap;

/// Whether all fields are dirty or a specific set is known.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldStatus {
    /// Every field is considered modified (conservative).
    AllDirty,
    /// Exactly these fields are dirty/clean.
    Fields {
        dirty: FxHashSet<Astr>,
        clean: FxHashSet<Astr>,
    },
}

/// One VarStore entry's dirty field information.
#[derive(Debug, Clone)]
pub struct VarDirtyEntry {
    pub name: Astr,
    pub status: FieldStatus,
}

/// Result of VarDirtyAnalysis -- keyed by the VarStore's instruction index.
#[derive(Debug, Clone)]
pub struct VarDirtyTable {
    pub entries: FxHashMap<InstIdx, VarDirtyEntry>,
}

pub struct VarDirtyAnalysis;

impl AnalysisPass for VarDirtyAnalysis {
    type Required<'a> = (&'a ValDefMap,);
    type Output = VarDirtyTable;

    fn run(&self, module: &MirModule, (val_def,): (&ValDefMap,)) -> VarDirtyTable {
        let insts = &module.main.insts;
        let mut entries = FxHashMap::default();

        for (idx, inst) in insts.iter().enumerate() {
            let InstKind::VarStore { name, src } = &inst.kind else {
                continue;
            };

            let status = analyze_store(insts, val_def, *name, *src);
            entries.insert(
                idx,
                VarDirtyEntry {
                    name: *name,
                    status,
                },
            );
        }

        VarDirtyTable { entries }
    }
}

fn analyze_store(
    insts: &[crate::ir::Inst],
    val_def: &ValDefMap,
    store_name: Astr,
    src: ValueId,
) -> FieldStatus {
    // src must be defined by MakeObject, otherwise conservative
    let Some(&src_idx) = val_def.0.get(&src) else {
        return FieldStatus::AllDirty;
    };

    let InstKind::MakeObject { fields, .. } = &insts[src_idx].kind else {
        return FieldStatus::AllDirty;
    };

    let mut dirty = FxHashSet::default();
    let mut clean = FxHashSet::default();

    for (field_name, field_val) in fields {
        if is_clean_field(insts, val_def, store_name, *field_name, *field_val) {
            clean.insert(*field_name);
        } else {
            dirty.insert(*field_name);
        }
    }

    FieldStatus::Fields { dirty, clean }
}

/// A field is "clean" if its value traces back to a FieldGet/ObjectGet of the same
/// field name from a VarLoad of the same variable name.
fn is_clean_field(
    insts: &[crate::ir::Inst],
    val_def: &ValDefMap,
    store_name: Astr,
    field_name: Astr,
    val: ValueId,
) -> bool {
    let Some(&def_idx) = val_def.0.get(&val) else {
        return false;
    };

    match &insts[def_idx].kind {
        // Direct field extraction: object.field or object["key"]
        InstKind::FieldGet {
            object, field: f, ..
        }
        | InstKind::ObjectGet { object, key: f, .. }
            if *f == field_name =>
        {
            // The object must come from VarLoad of the same name
            traces_to_var_load(insts, val_def, store_name, *object)
        }
        _ => false,
    }
}

/// Check if a Val traces back to a VarLoad of the given name.
fn traces_to_var_load(
    insts: &[crate::ir::Inst],
    val_def: &ValDefMap,
    expected_name: Astr,
    val: ValueId,
) -> bool {
    let Some(&def_idx) = val_def.0.get(&val) else {
        return false;
    };

    matches!(
        &insts[def_idx].kind,
        InstKind::VarLoad { name, .. } if *name == expected_name
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::Span;
    use crate::ir::{DebugInfo, Inst, MirBody};
    use acvus_utils::Interner;

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                debug: DebugInfo::new(),
                val_count: 0,
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    }

    fn inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::new(0, 0),
            kind,
        }
    }

    fn build_val_def(module: &MirModule) -> ValDefMap {
        use crate::pass::AnalysisPass;
        use crate::analysis::val_def::ValDefMapAnalysis;
        ValDefMapAnalysis.run(module, ())
    }

    /// VarStore of a scalar (non-object) -> AllDirty
    #[test]
    fn scalar_store_is_all_dirty() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::Const {
                dst: ValueId(0),
                value: acvus_ast::Literal::Int(42),
            }),
            inst(InstKind::VarStore {
                name: i.intern("x"),
                src: ValueId(0),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        assert_eq!(table.entries[&1].status, FieldStatus::AllDirty);
    }

    /// Full passthrough: VarLoad -> FieldGet each field -> MakeObject -> VarStore
    /// All fields should be clean.
    #[test]
    fn full_passthrough_all_clean() {
        let i = Interner::new();
        let module = make_module(vec![
            // v0 = VarLoad("user")
            inst(InstKind::VarLoad {
                dst: ValueId(0),
                name: i.intern("user"),
            }),
            // v1 = FieldGet(v0, "name")
            inst(InstKind::FieldGet {
                dst: ValueId(1),
                object: ValueId(0),
                field: i.intern("name"),
            }),
            // v2 = FieldGet(v0, "age")
            inst(InstKind::FieldGet {
                dst: ValueId(2),
                object: ValueId(0),
                field: i.intern("age"),
            }),
            // v3 = MakeObject { name: v1, age: v2 }
            inst(InstKind::MakeObject {
                dst: ValueId(3),
                fields: vec![
                    (i.intern("name"), ValueId(1)),
                    (i.intern("age"), ValueId(2)),
                ],
            }),
            // VarStore("user", v3)
            inst(InstKind::VarStore {
                name: i.intern("user"),
                src: ValueId(3),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&4];
        assert_eq!(entry.name, i.intern("user"));
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.is_empty());
                assert!(clean.contains(&i.intern("name")));
                assert!(clean.contains(&i.intern("age")));
            }
            _ => panic!("expected Fields"),
        }
    }

    /// Partial modification: one field changed, one passthrough.
    #[test]
    fn partial_modification_mixed() {
        let i = Interner::new();
        let module = make_module(vec![
            // v0 = VarLoad("user")
            inst(InstKind::VarLoad {
                dst: ValueId(0),
                name: i.intern("user"),
            }),
            // v1 = FieldGet(v0, "name")  -- passthrough
            inst(InstKind::FieldGet {
                dst: ValueId(1),
                object: ValueId(0),
                field: i.intern("name"),
            }),
            // v2 = Const("new_email")  -- new value
            inst(InstKind::Const {
                dst: ValueId(2),
                value: acvus_ast::Literal::String("new@email.com".into()),
            }),
            // v3 = MakeObject { name: v1, email: v2 }
            inst(InstKind::MakeObject {
                dst: ValueId(3),
                fields: vec![
                    (i.intern("name"), ValueId(1)),
                    (i.intern("email"), ValueId(2)),
                ],
            }),
            // VarStore("user", v3)
            inst(InstKind::VarStore {
                name: i.intern("user"),
                src: ValueId(3),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&4];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(clean.contains(&i.intern("name")));
                assert!(dirty.contains(&i.intern("email")));
                assert_eq!(dirty.len(), 1);
                assert_eq!(clean.len(), 1);
            }
            _ => panic!("expected Fields"),
        }
    }

    /// Field from a different variable -> dirty.
    #[test]
    fn field_from_different_variable_is_dirty() {
        let i = Interner::new();
        let module = make_module(vec![
            // v0 = VarLoad("other")
            inst(InstKind::VarLoad {
                dst: ValueId(0),
                name: i.intern("other"),
            }),
            // v1 = FieldGet(v0, "name")  -- from "other", not "user"
            inst(InstKind::FieldGet {
                dst: ValueId(1),
                object: ValueId(0),
                field: i.intern("name"),
            }),
            // v2 = MakeObject { name: v1 }
            inst(InstKind::MakeObject {
                dst: ValueId(2),
                fields: vec![(i.intern("name"), ValueId(1))],
            }),
            // VarStore("user", v2)
            inst(InstKind::VarStore {
                name: i.intern("user"),
                src: ValueId(2),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&3];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.contains(&i.intern("name")));
                assert!(clean.is_empty());
            }
            _ => panic!("expected Fields"),
        }
    }

    /// src is not MakeObject -> AllDirty.
    #[test]
    fn non_make_object_src_is_all_dirty() {
        let i = Interner::new();
        let module = make_module(vec![
            // v0 = ExternCall("make_user", [])
            inst(InstKind::ExternCall {
                dst: ValueId(0),
                name: i.intern("make_user"),
                args: vec![],
            }),
            // VarStore("user", v0)
            inst(InstKind::VarStore {
                name: i.intern("user"),
                src: ValueId(0),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        assert_eq!(table.entries[&1].status, FieldStatus::AllDirty);
    }

    /// ObjectGet (pattern matching variant) also counts as clean extraction.
    #[test]
    fn object_get_is_clean() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::VarLoad {
                dst: ValueId(0),
                name: i.intern("cfg"),
            }),
            inst(InstKind::ObjectGet {
                dst: ValueId(1),
                object: ValueId(0),
                key: i.intern("mode"),
            }),
            inst(InstKind::MakeObject {
                dst: ValueId(2),
                fields: vec![(i.intern("mode"), ValueId(1))],
            }),
            inst(InstKind::VarStore {
                name: i.intern("cfg"),
                src: ValueId(2),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&3];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.is_empty());
                assert!(clean.contains(&i.intern("mode")));
            }
            _ => panic!("expected Fields"),
        }
    }

    /// BlockLabel param (phi-like) -> conservative dirty.
    #[test]
    fn block_label_param_is_dirty() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::VarLoad {
                dst: ValueId(0),
                name: i.intern("data"),
            }),
            // v1 comes from a block param (phi node) -- could be anything
            inst(InstKind::BlockLabel {
                label: crate::ir::Label(0),
                params: vec![ValueId(1)],
                merge_of: None,
            }),
            inst(InstKind::MakeObject {
                dst: ValueId(2),
                fields: vec![(i.intern("value"), ValueId(1))],
            }),
            inst(InstKind::VarStore {
                name: i.intern("data"),
                src: ValueId(2),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&3];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.contains(&i.intern("value")));
                assert!(clean.is_empty());
            }
            _ => panic!("expected Fields"),
        }
    }

    /// Function call result used as field value -> dirty.
    #[test]
    fn call_result_is_dirty() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::VarLoad {
                dst: ValueId(0),
                name: i.intern("user"),
            }),
            inst(InstKind::FieldGet {
                dst: ValueId(1),
                object: ValueId(0),
                field: i.intern("name"),
            }),
            inst(InstKind::ExternCall {
                dst: ValueId(2),
                name: i.intern("transform"),
                args: vec![ValueId(1)],
            }),
            inst(InstKind::MakeObject {
                dst: ValueId(3),
                fields: vec![(i.intern("name"), ValueId(2))],
            }),
            inst(InstKind::VarStore {
                name: i.intern("user"),
                src: ValueId(3),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = VarDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&4];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.contains(&i.intern("name")));
                assert!(clean.is_empty());
            }
            _ => panic!("expected Fields"),
        }
    }
}
