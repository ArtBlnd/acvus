use crate::hints::InstIdx;
use crate::ir::{InstKind, MirModule, ValueId};
use rustc_hash::FxHashMap;

use crate::pass::AnalysisPass;

/// Maps each Val to the instruction index that defines it.
#[derive(Debug, Clone)]
pub struct ValDefMap(pub FxHashMap<ValueId, InstIdx>);

pub struct ValDefMapAnalysis;

impl AnalysisPass for ValDefMapAnalysis {
    type Required<'a> = ();
    type Output = ValDefMap;

    fn run(&self, module: &MirModule, _: ()) -> ValDefMap {
        let mut map = FxHashMap::default();
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
fn dst_of(kind: &InstKind) -> Option<ValueId> {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::ContextLoad { dst, .. }
        | InstKind::VarLoad { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::BuiltinCall { dst, .. }
        | InstKind::ExternCall { dst, .. }
        | InstKind::MakeDeque { dst, .. }
        | InstKind::MakeObject { dst, .. }
        | InstKind::MakeRange { dst, .. }
        | InstKind::MakeTuple { dst, .. }
        | InstKind::TupleIndex { dst, .. }
        | InstKind::TestLiteral { dst, .. }
        | InstKind::TestListLen { dst, .. }
        | InstKind::TestObjectKey { dst, .. }
        | InstKind::TestRange { dst, .. }
        | InstKind::ListIndex { dst, .. }
        | InstKind::ListGet { dst, .. }
        | InstKind::ListSlice { dst, .. }
        | InstKind::ObjectGet { dst, .. }
        | InstKind::MakeClosure { dst, .. }
        | InstKind::ClosureCall { dst, .. }
        | InstKind::MakeVariant { dst, .. }
        | InstKind::TestVariant { dst, .. }
        | InstKind::UnwrapVariant { dst, .. }
        | InstKind::Cast { dst, .. }
        | InstKind::IterStep { dst, .. }
        | InstKind::Poison { dst } => Some(*dst),

        // These don't define a new Val
        InstKind::Yield(_)
        | InstKind::VarStore { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Return(_)
        | InstKind::Nop => None,

        // BlockLabel params are defined at the label site
        InstKind::BlockLabel { .. } => None,
    }
}

/// Additional destination Vals beyond the primary one.
fn extra_dsts(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::BlockLabel { params, .. } => params.clone(),
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::Span;
    use crate::ir::{Inst, MirBody};
    use acvus_utils::Interner;

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                debug: crate::ir::DebugInfo::new(),
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

    #[test]
    fn context_load_mapped() {
        let i = Interner::new();
        let module = make_module(vec![inst(InstKind::ContextLoad {
            dst: ValueId(0),
            name: i.intern("user"),
        })]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&ValueId(0)], 0);
    }

    #[test]
    fn var_load_mapped() {
        let i = Interner::new();
        let module = make_module(vec![inst(InstKind::VarLoad {
            dst: ValueId(0),
            name: i.intern("count"),
        })]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&ValueId(0)], 0);
    }

    #[test]
    fn multiple_defs() {
        let module = make_module(vec![
            inst(InstKind::Const {
                dst: ValueId(0),
                value: acvus_ast::Literal::Int(1),
            }),
            inst(InstKind::Const {
                dst: ValueId(1),
                value: acvus_ast::Literal::Int(2),
            }),
            inst(InstKind::BinOp {
                dst: ValueId(2),
                op: acvus_ast::BinOp::Add,
                left: ValueId(0),
                right: ValueId(1),
            }),
        ]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&ValueId(0)], 0);
        assert_eq!(result.0[&ValueId(1)], 1);
        assert_eq!(result.0[&ValueId(2)], 2);
    }

    #[test]
    fn block_label_params_mapped() {
        let module = make_module(vec![inst(InstKind::BlockLabel {
            label: crate::ir::Label(0),
            params: vec![ValueId(0), ValueId(1)],
            merge_of: None,
        })]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert_eq!(result.0[&ValueId(0)], 0);
        assert_eq!(result.0[&ValueId(1)], 0);
    }

    #[test]
    fn non_defining_insts_skipped() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::Yield(ValueId(99))),
            inst(InstKind::VarStore {
                name: i.intern("x"),
                src: ValueId(0),
            }),
            inst(InstKind::Nop),
        ]);
        let result = ValDefMapAnalysis.run(&module, ());
        assert!(result.0.is_empty());
    }
}
