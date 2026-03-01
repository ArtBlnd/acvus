use std::collections::HashMap;

use acvus_mir::hints::InstIdx;
use acvus_mir::ir::{InstKind, MirModule, Val};

use crate::AnalysisPass;

/// Maps each Val to the instruction index that defines it.
#[derive(Debug, Clone)]
pub struct ValDefMap(pub HashMap<Val, InstIdx>);

pub struct ValDefMapAnalysis;

impl AnalysisPass for ValDefMapAnalysis {
    type Required<'a> = ();
    type Output = ValDefMap;

    fn run(&self, module: &MirModule, _: ()) -> ValDefMap {
        let mut map = HashMap::new();
        for (idx, inst) in module.main.insts.iter().enumerate() {
            if let Some(dst) = dst_of(&inst.kind) {
                map.insert(dst, idx);
            }
            // Some instructions define multiple vals
            for extra in extra_dsts(&inst.kind) {
                map.insert(extra, idx);
            }
        }
        ValDefMap(map)
    }
}

/// Primary destination Val of an instruction, if any.
fn dst_of(kind: &InstKind) -> Option<Val> {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::StorageLoad { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::Call { dst, .. }
        | InstKind::AsyncCall { dst, .. }
        | InstKind::Await { dst, .. }
        | InstKind::MakeList { dst, .. }
        | InstKind::MakeObject { dst, .. }
        | InstKind::MakeRange { dst, .. }
        | InstKind::MakeTuple { dst, .. }
        | InstKind::TupleIndex { dst, .. }
        | InstKind::TestLiteral { dst, .. }
        | InstKind::TestListLen { dst, .. }
        | InstKind::TestObjectKey { dst, .. }
        | InstKind::TestRange { dst, .. }
        | InstKind::ListIndex { dst, .. }
        | InstKind::ListSlice { dst, .. }
        | InstKind::ObjectGet { dst, .. }
        | InstKind::MakeClosure { dst, .. }
        | InstKind::CallClosure { dst, .. }
        | InstKind::IterInit { dst, .. } => Some(*dst),

        // IterNext defines dst_value as primary
        InstKind::IterNext { dst_value, .. } => Some(*dst_value),

        // These don't define a new Val
        InstKind::EmitText(_)
        | InstKind::EmitValue(_)
        | InstKind::StorageStore { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Return(_)
        | InstKind::Nop => None,

        // BlockLabel params are defined at the label site
        InstKind::BlockLabel { .. } => None,
    }
}

/// Additional destination Vals beyond the primary one.
fn extra_dsts(kind: &InstKind) -> Vec<Val> {
    match kind {
        InstKind::IterNext { dst_done, .. } => vec![*dst_done],
        InstKind::BlockLabel { params, .. } => params.clone(),
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::Span;
    use acvus_mir::ir::{Inst, MirBody};

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: HashMap::new(),
                debug: acvus_mir::ir::DebugInfo::new(),
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

    #[test]
    fn storage_load_mapped() {
        let module = make_module(vec![inst(InstKind::StorageLoad {
            dst: Val(0),
            name: "user".into(),
        })]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&Val(0)], 0);
    }

    #[test]
    fn multiple_defs() {
        let module = make_module(vec![
            inst(InstKind::Const {
                dst: Val(0),
                value: acvus_ast::Literal::Int(1),
            }),
            inst(InstKind::Const {
                dst: Val(1),
                value: acvus_ast::Literal::Int(2),
            }),
            inst(InstKind::BinOp {
                dst: Val(2),
                op: acvus_ast::BinOp::Add,
                left: Val(0),
                right: Val(1),
            }),
        ]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&Val(0)], 0);
        assert_eq!(result.0[&Val(1)], 1);
        assert_eq!(result.0[&Val(2)], 2);
    }

    #[test]
    fn iter_next_defines_two_vals() {
        let module = make_module(vec![inst(InstKind::IterNext {
            dst_value: Val(0),
            dst_done: Val(1),
            iter: Val(2),
        })]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&Val(0)], 0);
        assert_eq!(result.0[&Val(1)], 0);
    }

    #[test]
    fn block_label_params_mapped() {
        let module = make_module(vec![inst(InstKind::BlockLabel {
            label: acvus_mir::ir::Label(0),
            params: vec![Val(0), Val(1)],
        })]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&Val(0)], 0);
        assert_eq!(result.0[&Val(1)], 0);
    }

    #[test]
    fn non_defining_insts_skipped() {
        let module = make_module(vec![
            inst(InstKind::EmitText(0)),
            inst(InstKind::StorageStore {
                name: "x".into(),
                src: Val(0),
            }),
            inst(InstKind::Nop),
        ]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert!(result.0.is_empty());
    }
}
