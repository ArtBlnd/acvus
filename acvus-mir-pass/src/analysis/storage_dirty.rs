use std::collections::{HashMap, HashSet};

use acvus_mir::hints::InstIdx;
use acvus_mir::ir::{InstKind, MirModule, Val};

use crate::AnalysisPass;
use crate::analysis::val_def::ValDefMap;

/// Whether all fields are dirty or a specific set is known.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldStatus {
    /// Every field is considered modified (conservative).
    AllDirty,
    /// Exactly these fields are dirty/clean.
    Fields {
        dirty: HashSet<String>,
        clean: HashSet<String>,
    },
}

/// One StorageStore entry's dirty field information.
#[derive(Debug, Clone)]
pub struct StorageDirtyEntry {
    pub name: String,
    pub status: FieldStatus,
}

/// Result of StorageDirtyAnalysis — keyed by the StorageStore's instruction index.
#[derive(Debug, Clone)]
pub struct StorageDirtyTable {
    pub entries: HashMap<InstIdx, StorageDirtyEntry>,
}

pub struct StorageDirtyAnalysis;

impl AnalysisPass for StorageDirtyAnalysis {
    type Required<'a> = (&'a ValDefMap,);
    type Output = StorageDirtyTable;

    fn run(&self, module: &MirModule, (val_def,): (&ValDefMap,)) -> StorageDirtyTable {
        let insts = &module.main.insts;
        let mut entries = HashMap::new();

        for (idx, inst) in insts.iter().enumerate() {
            let InstKind::StorageStore { name, src } = &inst.kind else {
                continue;
            };

            let status = analyze_store(insts, &val_def, name, *src);
            entries.insert(
                idx,
                StorageDirtyEntry {
                    name: name.clone(),
                    status,
                },
            );
        }

        StorageDirtyTable { entries }
    }
}

fn analyze_store(
    insts: &[acvus_mir::ir::Inst],
    val_def: &ValDefMap,
    store_name: &str,
    src: Val,
) -> FieldStatus {
    // src must be defined by MakeObject, otherwise conservative
    let Some(&src_idx) = val_def.0.get(&src) else {
        return FieldStatus::AllDirty;
    };

    let InstKind::MakeObject { fields, .. } = &insts[src_idx].kind else {
        return FieldStatus::AllDirty;
    };

    let mut dirty = HashSet::new();
    let mut clean = HashSet::new();

    for (field_name, field_val) in fields {
        if is_clean_field(insts, val_def, store_name, field_name, *field_val) {
            clean.insert(field_name.clone());
        } else {
            dirty.insert(field_name.clone());
        }
    }

    FieldStatus::Fields { dirty, clean }
}

/// A field is "clean" if its value traces back to a FieldGet/ObjectGet of the same
/// field name from a StorageLoad of the same storage key.
fn is_clean_field(
    insts: &[acvus_mir::ir::Inst],
    val_def: &ValDefMap,
    store_name: &str,
    field_name: &str,
    val: Val,
) -> bool {
    let Some(&def_idx) = val_def.0.get(&val) else {
        return false;
    };

    match &insts[def_idx].kind {
        // Direct field extraction: object.field or object["key"]
        InstKind::FieldGet {
            object,
            field: f,
            ..
        }
        | InstKind::ObjectGet {
            object,
            key: f,
            ..
        } if f == field_name => {
            // The object must come from StorageLoad of the same key
            traces_to_storage_load(insts, val_def, store_name, *object)
        }
        _ => false,
    }
}

/// Check if a Val traces back to a StorageLoad of the given name.
fn traces_to_storage_load(
    insts: &[acvus_mir::ir::Inst],
    val_def: &ValDefMap,
    expected_name: &str,
    val: Val,
) -> bool {
    let Some(&def_idx) = val_def.0.get(&val) else {
        return false;
    };

    matches!(
        &insts[def_idx].kind,
        InstKind::StorageLoad { name, .. } if name == expected_name
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::Span;
    use acvus_mir::ir::{DebugInfo, Inst, MirBody};

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: HashMap::new(),
                debug: DebugInfo::new(),
                val_count: 0,
                label_count: 0,
            },
            closures: HashMap::new(),
            texts: vec![],
        }
    }

    fn inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::new(0, 0),
            kind,
        }
    }

    fn build_val_def(module: &MirModule) -> ValDefMap {
        use crate::analysis::val_def::ValDefMapAnalysis;
        use crate::AnalysisPass;
        ValDefMapAnalysis.run(module, ())
    }

    /// StorageStore of a scalar (non-object) → AllDirty
    #[test]
    fn scalar_store_is_all_dirty() {
        let module = make_module(vec![
            inst(InstKind::Const {
                dst: Val(0),
                value: acvus_ast::Literal::Int(42),
            }),
            inst(InstKind::StorageStore {
                name: "x".into(),
                src: Val(0),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        assert_eq!(table.entries[&1].status, FieldStatus::AllDirty);
    }

    /// Full passthrough: StorageLoad → FieldGet each field → MakeObject → StorageStore
    /// All fields should be clean.
    #[test]
    fn full_passthrough_all_clean() {
        let module = make_module(vec![
            // v0 = StorageLoad("user")
            inst(InstKind::StorageLoad {
                dst: Val(0),
                name: "user".into(),
            }),
            // v1 = FieldGet(v0, "name")
            inst(InstKind::FieldGet {
                dst: Val(1),
                object: Val(0),
                field: "name".into(),
            }),
            // v2 = FieldGet(v0, "age")
            inst(InstKind::FieldGet {
                dst: Val(2),
                object: Val(0),
                field: "age".into(),
            }),
            // v3 = MakeObject { name: v1, age: v2 }
            inst(InstKind::MakeObject {
                dst: Val(3),
                fields: vec![("name".into(), Val(1)), ("age".into(), Val(2))],
            }),
            // StorageStore("user", v3)
            inst(InstKind::StorageStore {
                name: "user".into(),
                src: Val(3),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&4];
        assert_eq!(entry.name, "user");
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.is_empty());
                assert!(clean.contains("name"));
                assert!(clean.contains("age"));
            }
            _ => panic!("expected Fields"),
        }
    }

    /// Partial modification: one field changed, one passthrough.
    #[test]
    fn partial_modification_mixed() {
        let module = make_module(vec![
            // v0 = StorageLoad("user")
            inst(InstKind::StorageLoad {
                dst: Val(0),
                name: "user".into(),
            }),
            // v1 = FieldGet(v0, "name")  — passthrough
            inst(InstKind::FieldGet {
                dst: Val(1),
                object: Val(0),
                field: "name".into(),
            }),
            // v2 = Const("new_email")  — new value
            inst(InstKind::Const {
                dst: Val(2),
                value: acvus_ast::Literal::String("new@email.com".into()),
            }),
            // v3 = MakeObject { name: v1, email: v2 }
            inst(InstKind::MakeObject {
                dst: Val(3),
                fields: vec![("name".into(), Val(1)), ("email".into(), Val(2))],
            }),
            // StorageStore("user", v3)
            inst(InstKind::StorageStore {
                name: "user".into(),
                src: Val(3),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&4];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(clean.contains("name"));
                assert!(dirty.contains("email"));
                assert_eq!(dirty.len(), 1);
                assert_eq!(clean.len(), 1);
            }
            _ => panic!("expected Fields"),
        }
    }

    /// Field from a different storage key → dirty.
    #[test]
    fn field_from_different_storage_is_dirty() {
        let module = make_module(vec![
            // v0 = StorageLoad("other")
            inst(InstKind::StorageLoad {
                dst: Val(0),
                name: "other".into(),
            }),
            // v1 = FieldGet(v0, "name")  — from "other", not "user"
            inst(InstKind::FieldGet {
                dst: Val(1),
                object: Val(0),
                field: "name".into(),
            }),
            // v2 = MakeObject { name: v1 }
            inst(InstKind::MakeObject {
                dst: Val(2),
                fields: vec![("name".into(), Val(1))],
            }),
            // StorageStore("user", v2)
            inst(InstKind::StorageStore {
                name: "user".into(),
                src: Val(2),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&3];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.contains("name"));
                assert!(clean.is_empty());
            }
            _ => panic!("expected Fields"),
        }
    }

    /// src is not MakeObject → AllDirty.
    #[test]
    fn non_make_object_src_is_all_dirty() {
        let module = make_module(vec![
            // v0 = Call("make_user", [])
            inst(InstKind::Call {
                dst: Val(0),
                func: "make_user".into(),
                args: vec![],
            }),
            // StorageStore("user", v0)
            inst(InstKind::StorageStore {
                name: "user".into(),
                src: Val(0),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        assert_eq!(table.entries[&1].status, FieldStatus::AllDirty);
    }

    /// ObjectGet (pattern matching variant) also counts as clean extraction.
    #[test]
    fn object_get_is_clean() {
        let module = make_module(vec![
            inst(InstKind::StorageLoad {
                dst: Val(0),
                name: "cfg".into(),
            }),
            inst(InstKind::ObjectGet {
                dst: Val(1),
                object: Val(0),
                key: "mode".into(),
            }),
            inst(InstKind::MakeObject {
                dst: Val(2),
                fields: vec![("mode".into(), Val(1))],
            }),
            inst(InstKind::StorageStore {
                name: "cfg".into(),
                src: Val(2),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&3];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.is_empty());
                assert!(clean.contains("mode"));
            }
            _ => panic!("expected Fields"),
        }
    }

    /// BlockLabel param (phi-like) → conservative dirty.
    #[test]
    fn block_label_param_is_dirty() {
        let module = make_module(vec![
            inst(InstKind::StorageLoad {
                dst: Val(0),
                name: "data".into(),
            }),
            // v1 comes from a block param (phi node) — could be anything
            inst(InstKind::BlockLabel {
                label: acvus_mir::ir::Label(0),
                params: vec![Val(1)],
            }),
            inst(InstKind::MakeObject {
                dst: Val(2),
                fields: vec![("value".into(), Val(1))],
            }),
            inst(InstKind::StorageStore {
                name: "data".into(),
                src: Val(2),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&3];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.contains("value"));
                assert!(clean.is_empty());
            }
            _ => panic!("expected Fields"),
        }
    }

    /// Function call result used as field value → dirty.
    #[test]
    fn call_result_is_dirty() {
        let module = make_module(vec![
            inst(InstKind::StorageLoad {
                dst: Val(0),
                name: "user".into(),
            }),
            inst(InstKind::FieldGet {
                dst: Val(1),
                object: Val(0),
                field: "name".into(),
            }),
            inst(InstKind::Call {
                dst: Val(2),
                func: "uppercase".into(),
                args: vec![Val(1)],
            }),
            inst(InstKind::MakeObject {
                dst: Val(3),
                fields: vec![("name".into(), Val(2))],
            }),
            inst(InstKind::StorageStore {
                name: "user".into(),
                src: Val(3),
            }),
        ]);
        let val_def = build_val_def(&module);
        let table = StorageDirtyAnalysis.run(&module, (&val_def,));
        let entry = &table.entries[&4];
        match &entry.status {
            FieldStatus::Fields { dirty, clean } => {
                assert!(dirty.contains("name"));
                assert!(clean.is_empty());
            }
            _ => panic!("expected Fields"),
        }
    }
}
