use crate::ir::{Callee, InstKind, MirBody, MirModule, ValueId};
use acvus_ast::Literal;
use rustc_hash::FxHashMap;

/// Hashable key for Literal values.
/// Wraps f64 via to_bits() to provide Hash + Eq.
#[derive(Hash, Eq, PartialEq)]
enum LiteralKey {
    Int(i64),
    Float(u64),
    String(String),
    Bool(bool),
    Byte(u8),
    List(Vec<LiteralKey>),
    Unit,
}

impl LiteralKey {
    fn from_literal(lit: &Literal) -> Self {
        match lit {
            Literal::Int(v) => LiteralKey::Int(*v),
            Literal::Float(v) => LiteralKey::Float(v.to_bits()),
            Literal::String(v) => LiteralKey::String(v.clone()),
            Literal::Bool(v) => LiteralKey::Bool(*v),
            Literal::Byte(v) => LiteralKey::Byte(*v),
            Literal::List(elems) => {
                LiteralKey::List(elems.iter().map(LiteralKey::from_literal).collect())
            }
            Literal::Unit => LiteralKey::Unit,
        }
    }
}

/// Deduplicate identical `Const` instructions across a MirModule.
/// Canonical constants are hoisted to the top of each body.
pub fn dedup(mut module: MirModule) -> MirModule {
    module.main = dedup_body(module.main);
    for closure in module.closures.values_mut() {
        *closure = dedup_body(std::mem::take(closure));
    }
    module
}

fn dedup_body(mut body: MirBody) -> MirBody {
    // 1. Scan Const instructions, build canonical map and remap table.
    let mut canonical: FxHashMap<LiteralKey, ValueId> = FxHashMap::default();
    let mut remap: FxHashMap<ValueId, ValueId> = FxHashMap::default();

    for inst in &body.insts {
        if let InstKind::Const { dst, ref value } = inst.kind {
            let key = LiteralKey::from_literal(value);
            match canonical.get(&key) {
                Some(&canon) => {
                    remap.insert(dst, canon);
                }
                None => {
                    canonical.insert(key, dst);
                }
            }
        }
    }

    if remap.is_empty() {
        return body;
    }

    // 2. Separate canonical Const instructions (for hoisting) from the rest.
    let canonical_set: std::collections::HashSet<ValueId> = canonical.values().copied().collect();

    let mut hoisted = Vec::new();
    let mut rest = Vec::new();

    for inst in body.insts {
        match &inst.kind {
            InstKind::Const { dst, .. } => {
                if canonical_set.contains(dst) {
                    hoisted.push(inst);
                }
                // else: duplicate, drop it
            }
            _ => {
                rest.push(inst);
            }
        }
    }

    // 3. Remap use-position ValueIds in remaining instructions.
    for inst in &mut rest {
        remap_uses(&mut inst.kind, &remap);
    }

    // 4. Combine: hoisted constants first, then the rest.
    hoisted.append(&mut rest);
    body.insts = hoisted;
    body
}

fn remap_val(v: &mut ValueId, remap: &FxHashMap<ValueId, ValueId>) {
    if let Some(&canon) = remap.get(v) {
        *v = canon;
    }
}

fn remap_vec(vs: &mut Vec<ValueId>, remap: &FxHashMap<ValueId, ValueId>) {
    for v in vs.iter_mut() {
        remap_val(v, remap);
    }
}

fn remap_uses(kind: &mut InstKind, remap: &FxHashMap<ValueId, ValueId>) {
    match kind {
        // No uses
        InstKind::Const { .. }
        | InstKind::Ref { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Nop
        | InstKind::Poison { .. }
        | InstKind::Undef { .. } => {}

        // Single use
        InstKind::Return(v) => remap_val(v, remap),

        // Load: src is a use.
        InstKind::Load { src, .. } => remap_val(src, remap),

        // Store: dst is a projection use, value is a value use.
        InstKind::Store { dst, value, .. } => {
            remap_val(dst, remap);
            remap_val(value, remap);
        }

        InstKind::UnaryOp { operand, .. } => remap_val(operand, remap),

        InstKind::FieldGet { object, .. } => remap_val(object, remap),
        InstKind::FieldSet { object, value, .. } => {
            remap_val(object, remap);
            remap_val(value, remap);
        }

        InstKind::TupleIndex { tuple, .. } => remap_val(tuple, remap),

        InstKind::TestLiteral { src, .. } => remap_val(src, remap),

        InstKind::TestListLen { src, .. } => remap_val(src, remap),

        InstKind::TestObjectKey { src, .. } => remap_val(src, remap),

        InstKind::TestRange { src, .. } => remap_val(src, remap),

        InstKind::ListIndex { list, .. } => remap_val(list, remap),

        InstKind::ListSlice { list, .. } => remap_val(list, remap),

        InstKind::ObjectGet { object, .. } => remap_val(object, remap),

        // Two uses
        InstKind::BinOp { left, right, .. } => {
            remap_val(left, remap);
            remap_val(right, remap);
        }

        InstKind::MakeRange { start, end, .. } => {
            remap_val(start, remap);
            remap_val(end, remap);
        }

        InstKind::ListGet { list, index, .. } => {
            remap_val(list, remap);
            remap_val(index, remap);
        }

        // Vec uses
        InstKind::FunctionCall { callee, args, .. } => {
            if let Callee::Indirect(val) = callee {
                remap_val(val, remap);
            }
            remap_vec(args, remap);
        }

        InstKind::MakeDeque { elements, .. } | InstKind::MakeTuple { elements, .. } => {
            remap_vec(elements, remap);
        }

        InstKind::MakeObject { fields, .. } => {
            for (_, v) in fields.iter_mut() {
                remap_val(v, remap);
            }
        }

        InstKind::MakeClosure { captures, .. } => {
            remap_vec(captures, remap);
        }

        InstKind::LoadFunction { .. } => {
            // No values to remap (id is a FunctionId, not a ValueId)
        }

        InstKind::Jump { args, .. } => {
            remap_vec(args, remap);
        }

        InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            remap_val(cond, remap);
            remap_vec(then_args, remap);
            remap_vec(else_args, remap);
        }

        InstKind::MakeVariant { payload, .. } => {
            if let Some(p) = payload {
                remap_val(p, remap);
            }
        }

        InstKind::TestVariant { src, .. } => remap_val(src, remap),

        InstKind::UnwrapVariant { src, .. } => remap_val(src, remap),

        InstKind::Cast { src, .. } => remap_val(src, remap),

        InstKind::Clone { src, .. } => remap_val(src, remap),

        InstKind::Drop { src } => remap_val(src, remap),

        InstKind::ListStep {
            list,
            index_src,
            done_args,
            ..
        } => {
            remap_val(list, remap);
            remap_val(index_src, remap);
            remap_vec(done_args, remap);
        }

        InstKind::Spawn { callee, args, .. } => {
            if let Callee::Indirect(val) = callee {
                remap_val(val, remap);
            }
            remap_vec(args, remap);
        }

        InstKind::Eval { src, .. } => remap_val(src, remap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DebugInfo, Inst, MirBody};
    use acvus_ast::Span;
    use acvus_utils::LocalFactory;

    fn span() -> Span {
        Span::ZERO
    }

    fn make_body(insts: Vec<Inst>) -> MirBody {
        MirBody {
            insts,
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: LocalFactory::new(),
            label_count: 0,
        }
    }

    #[test]
    fn dedup_removes_duplicate_int_consts() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let body = make_body(vec![
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v0,
                    value: Literal::Int(42),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v1,
                    value: Literal::Int(42),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v2,
                    value: Literal::Int(42),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::BinOp {
                    dst: v3,
                    op: acvus_ast::BinOp::Add,
                    left: v1,
                    right: v2,
                },
            },
        ]);

        let result = dedup_body(body);

        // Only 1 Const + 1 BinOp
        assert_eq!(result.insts.len(), 2);

        // BinOp uses should be remapped to canonical v0
        match &result.insts[1].kind {
            InstKind::BinOp { left, right, .. } => {
                assert_eq!(*left, v0);
                assert_eq!(*right, v0);
            }
            other => panic!("expected BinOp, got {other:?}"),
        }
    }

    #[test]
    fn dedup_keeps_distinct_values() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let body = make_body(vec![
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v0,
                    value: Literal::Int(1),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v1,
                    value: Literal::Int(2),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::BinOp {
                    dst: v2,
                    op: acvus_ast::BinOp::Add,
                    left: v0,
                    right: v1,
                },
            },
        ]);

        let result = dedup_body(body);
        assert_eq!(result.insts.len(), 3);
    }

    #[test]
    fn dedup_no_change_when_no_duplicates() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let body = make_body(vec![
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v0,
                    value: Literal::Int(1),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Return(v0),
            },
        ]);

        let result = dedup_body(body);
        assert_eq!(result.insts.len(), 2);
    }

    #[test]
    fn dedup_string_consts() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let body = make_body(vec![
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v0,
                    value: Literal::String("hello".into()),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v1,
                    value: Literal::String("hello".into()),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Return(v1),
            },
        ]);

        let result = dedup_body(body);
        assert_eq!(result.insts.len(), 2);
        match &result.insts[1].kind {
            InstKind::Return(v) => assert_eq!(*v, v0),
            other => panic!("expected Return, got {other:?}"),
        }
    }

    #[test]
    fn dedup_hoists_canonical_to_top() {
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let body = make_body(vec![
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v0,
                    value: Literal::Int(1),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Return(v0),
            },
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v1,
                    value: Literal::Int(256),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: v2,
                    value: Literal::Int(256),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Return(v2),
            },
        ]);

        let result = dedup_body(body);
        // 2 canonical Consts hoisted, then 2 Returns
        assert_eq!(result.insts.len(), 4);
        assert!(matches!(result.insts[0].kind, InstKind::Const { .. }));
        assert!(matches!(result.insts[1].kind, InstKind::Const { .. }));
    }
}
