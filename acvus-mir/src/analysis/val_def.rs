use crate::hints::InstIdx;
use crate::ir::{InstKind, MirModule, ValueId};
use rustc_hash::FxHashMap;

/// Maps each Val to the instruction index that defines it.
#[derive(Debug, Clone)]
pub struct ValDefMap(pub FxHashMap<ValueId, InstIdx>);

pub fn build(module: &MirModule) -> ValDefMap {
    let mut map = FxHashMap::default();
    for (idx, inst) in module.main.insts.iter().enumerate() {
        if let Some(dst) = dst_of(&inst.kind) {
            map.insert(dst, idx);
        }
        for extra in extra_dsts(&inst.kind) {
            map.insert(extra, idx);
        }
    }
    ValDefMap(map)
}

/// Primary destination Val of an instruction, if any.
fn dst_of(kind: &InstKind) -> Option<ValueId> {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::ContextProject { dst, .. }
        | InstKind::ContextLoad { dst, .. }
        | InstKind::VarLoad { dst, .. }
        | InstKind::ParamLoad { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::FunctionCall { dst, .. }
        | InstKind::LoadFunction { dst, .. }
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
        | InstKind::MakeVariant { dst, .. }
        | InstKind::TestVariant { dst, .. }
        | InstKind::UnwrapVariant { dst, .. }
        | InstKind::Cast { dst, .. }
        | InstKind::ListStep { dst, .. }
        | InstKind::Spawn { dst, .. }
        | InstKind::Eval { dst, .. }
        | InstKind::Poison { dst }
        | InstKind::Undef { dst } => Some(*dst),

        // These don't define a new Val
        InstKind::VarStore { .. }
        | InstKind::ContextStore { .. }
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
        InstKind::ListStep { index_dst, .. } => vec![*index_dst],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::QualifiedRef;
    use crate::ir::{Inst, MirBody};
    use acvus_ast::Span;
    use acvus_utils::{Interner, LocalFactory};

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: Vec::new(),
                capture_regs: Vec::new(),
                debug: crate::ir::DebugInfo::new(),
                val_factory: LocalFactory::new(),
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
    fn context_project_mapped() {
        let i = Interner::new();
        let id0 = QualifiedRef::root(i.intern("x"));
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::ContextLoad { dst: v1, src: v0 , volatile: false }),
        ]);
        let result = super::build(&module);
        assert_eq!(result.0[&v0], 0); // ContextProject defines v0
        assert_eq!(result.0[&v1], 1); // ContextLoad defines v1
    }

    #[test]
    fn var_load_mapped() {
        let i = Interner::new();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let module = make_module(vec![inst(InstKind::VarLoad {
            dst: v0,
            name: i.intern("count"),
        })]);
        let result = super::build(&module);
        assert_eq!(result.0[&v0], 0);
    }

    #[test]
    fn multiple_defs() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let module = make_module(vec![
            inst(InstKind::Const {
                dst: v0,
                value: acvus_ast::Literal::Int(1),
            }),
            inst(InstKind::Const {
                dst: v1,
                value: acvus_ast::Literal::Int(2),
            }),
            inst(InstKind::BinOp {
                dst: v2,
                op: acvus_ast::BinOp::Add,
                left: v0,
                right: v1,
            }),
        ]);
        let result = super::build(&module);
        assert_eq!(result.0[&v0], 0);
        assert_eq!(result.0[&v1], 1);
        assert_eq!(result.0[&v2], 2);
    }

    #[test]
    fn block_label_params_mapped() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let module = make_module(vec![inst(InstKind::BlockLabel {
            label: crate::ir::Label(0),
            params: vec![v0, v1],
            merge_of: None,
        })]);
        let result = super::build(&module);
        assert_eq!(result.0[&v0], 0);
        assert_eq!(result.0[&v1], 0);
    }

    #[test]
    fn non_defining_insts_skipped() {
        let i = Interner::new();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        // Skip v1..v98 to get v99
        for _ in 1..99 {
            vf.next();
        }
        let v99 = vf.next();
        let module = make_module(vec![
            inst(InstKind::Return(v99)),
            inst(InstKind::VarStore {
                name: i.intern("x"),
                src: v0,
            }),
            inst(InstKind::Nop),
        ]);
        let result = super::build(&module);
        assert!(result.0.is_empty());
    }
}
