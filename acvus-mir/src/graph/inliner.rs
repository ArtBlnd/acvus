//! Phase 4: Inliner
//!
//! Inlines all local function calls into a single flat MIR body.
//! After inlining, re-run SSABuilder to deduplicate context loads and
//! insert PHIs at merge points.
//!
//! What gets inlined:
//! - `FunctionCall` with `Callee::Direct(id)` where id is a local function
//!
//! What stays as a call:
//! - `Callee::Indirect` (closures, function-valued variables)
//! - `Callee::Direct` to extern/builtin functions (no body to inline)
//! - Recursive calls (detected via SCC — self-referencing or mutual recursion)

use acvus_utils::LocalIdOps;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::*;

use super::types::FunctionId;

/// Result of the inlining pass.
#[derive(Debug)]
pub struct InlineResult {
    /// Inlined MIR per top-level function.
    pub modules: FxHashMap<FunctionId, MirModule>,
}

/// Inline all local function calls within each top-level function's MIR.
///
/// `modules`: per-function MIR from the lower phase.
/// `recursive_fns`: functions that should NOT be inlined (recursive / mutual recursive).
pub fn inline(
    modules: &FxHashMap<FunctionId, MirModule>,
    recursive_fns: &FxHashSet<FunctionId>,
) -> InlineResult {
    let mut result = FxHashMap::default();

    for (&fn_id, module) in modules {
        let inlined = inline_module(module, modules, recursive_fns);
        result.insert(fn_id, inlined);
    }

    InlineResult { modules: result }
}

/// Inline all eligible calls within a single MirModule.
fn inline_module(
    module: &MirModule,
    all_modules: &FxHashMap<FunctionId, MirModule>,
    recursive_fns: &FxHashSet<FunctionId>,
) -> MirModule {
    let body = inline_body(&module.main, all_modules, recursive_fns);

    // Inline within closures too.
    let mut closures = FxHashMap::default();
    for (label, closure) in &module.closures {
        let mut inlined_body = inline_body(closure, all_modules, recursive_fns);
        inlined_body.capture_regs = remap_value_ids(&closure.capture_regs, &FxHashMap::default());
        inlined_body.param_regs = remap_value_ids(&closure.param_regs, &FxHashMap::default());
        closures.insert(*label, inlined_body);
    }

    MirModule {
        main: body,
        closures,
    }
}

/// Inline all eligible FunctionCall instructions within a MirBody.
/// Iterates until no more inlining opportunities remain (handles nested calls
/// where an inlined body itself contains calls to other local functions).
fn inline_body(
    body: &MirBody,
    all_modules: &FxHashMap<FunctionId, MirModule>,
    recursive_fns: &FxHashSet<FunctionId>,
) -> MirBody {
    let mut current = body.clone();

    loop {
        let mut changed = false;
        let mut new_insts = Vec::new();
        let mut val_remap: FxHashMap<ValueId, ValueId> = FxHashMap::default();

        for inst in &current.insts {
            match &inst.kind {
                InstKind::FunctionCall {
                    dst,
                    callee: Callee::Direct(callee_id),
                    args,
                    context_uses: _,
                    context_defs: _,
                } if !recursive_fns.contains(callee_id) && all_modules.contains_key(callee_id) => {
                    let callee_module = &all_modules[callee_id];
                    let callee_body = &callee_module.main;

                    // Build ValueId remap: callee's ids → fresh ids in caller.
                    let mut callee_remap: FxHashMap<ValueId, ValueId> = FxHashMap::default();
                    for i in 0..callee_body.val_factory.len() {
                        let old_id = ValueId::from_raw(i);
                        let new_id = current.val_factory.next();
                        callee_remap.insert(old_id, new_id);
                    }

                    // Build Label remap: callee's labels → fresh labels in caller.
                    let label_offset = current.label_count;
                    current.label_count += callee_body.label_count;

                    // Map callee params to caller args via VarLoad/VarStore pattern.
                    // The callee uses VarLoad { name } to read params.
                    // We find the first N VarLoads and map their dst to our args.
                    let mut param_idx = 0;
                    let mut param_map: FxHashMap<ValueId, ValueId> = FxHashMap::default();

                    // Copy callee's val_types (remapped).
                    for (&old_val, ty) in &callee_body.val_types {
                        if let Some(&new_val) = callee_remap.get(&old_val) {
                            current.val_types.insert(new_val, ty.clone());
                        }
                    }

                    // Copy callee's debug info (remapped).
                    for (&old_val, origin) in &callee_body.debug.val_origins {
                        if let Some(&new_val) = callee_remap.get(&old_val) {
                            current.debug.val_origins.insert(new_val, origin.clone());
                        }
                    }

                    // Emit callee instructions with remapped ids.
                    for callee_inst in &callee_body.insts {
                        match &callee_inst.kind {
                            // VarLoad at the start = parameter load.
                            // Map to the corresponding argument.
                            InstKind::VarLoad { dst, name: _ } if param_idx < args.len() => {
                                let remapped_dst = callee_remap[dst];
                                param_map.insert(remapped_dst, args[param_idx]);
                                // Don't emit the VarLoad — replace with a direct binding.
                                // Instead, remap all uses of remapped_dst to args[param_idx].
                                callee_remap.insert(*dst, args[param_idx]);
                                param_idx += 1;
                            }

                            // Return → map result to caller's dst.
                            InstKind::Return(val) => {
                                let remapped_val = remap_one(*val, &callee_remap);
                                val_remap.insert(*dst, remapped_val);
                                // Don't emit Return.
                            }

                            // All other instructions: remap and emit.
                            _ => {
                                let remapped =
                                    remap_inst(&callee_inst.kind, &callee_remap, label_offset);
                                new_insts.push(Inst {
                                    span: callee_inst.span,
                                    kind: remapped,
                                });
                            }
                        }
                    }

                    changed = true;
                }

                // Non-inlineable call or non-call instruction: emit as-is,
                // applying any val_remap from previous inlinings.
                _ => {
                    let remapped = remap_inst(&inst.kind, &val_remap, 0);
                    new_insts.push(Inst {
                        span: inst.span,
                        kind: remapped,
                    });
                }
            }
        }

        current.insts = new_insts;

        if !changed {
            break;
        }
    }

    current
}

/// Remap a single ValueId through a remap table. Returns original if not mapped.
fn remap_one(val: ValueId, remap: &FxHashMap<ValueId, ValueId>) -> ValueId {
    remap.get(&val).copied().unwrap_or(val)
}

/// Remap a vec of ValueIds.
fn remap_value_ids(vals: &[ValueId], remap: &FxHashMap<ValueId, ValueId>) -> Vec<ValueId> {
    vals.iter().map(|v| remap_one(*v, remap)).collect()
}

/// Remap a Label with an offset.
fn remap_label(label: Label, offset: u32) -> Label {
    if offset == 0 {
        label
    } else {
        Label(label.0 + offset)
    }
}

/// Remap all ValueIds and Labels in an instruction.
fn remap_inst(
    kind: &InstKind,
    val_remap: &FxHashMap<ValueId, ValueId>,
    label_offset: u32,
) -> InstKind {
    let r = |v: ValueId| -> ValueId { remap_one(v, val_remap) };
    let rl = |l: Label| -> Label { remap_label(l, label_offset) };
    let rv = |vals: &[ValueId]| -> Vec<ValueId> { vals.iter().map(|v| r(*v)).collect() };

    match kind {
        // Constants
        InstKind::Const { dst, value } => InstKind::Const {
            dst: r(*dst),
            value: value.clone(),
        },

        // Context
        InstKind::ContextProject { dst, ctx } => InstKind::ContextProject {
            dst: r(*dst),
            ctx: *ctx,
        },
        InstKind::ContextLoad { dst, src } => InstKind::ContextLoad {
            dst: r(*dst),
            src: r(*src),
        },
        InstKind::ContextStore { dst, value } => InstKind::ContextStore {
            dst: r(*dst),
            value: r(*value),
        },

        // Variables
        InstKind::VarLoad { dst, name } => InstKind::VarLoad {
            dst: r(*dst),
            name: *name,
        },
        InstKind::ParamLoad { dst, name } => InstKind::ParamLoad {
            dst: r(*dst),
            name: *name,
        },
        InstKind::VarStore { name, src } => InstKind::VarStore {
            name: *name,
            src: r(*src),
        },

        // Arithmetic
        InstKind::BinOp {
            dst,
            op,
            left,
            right,
        } => InstKind::BinOp {
            dst: r(*dst),
            op: *op,
            left: r(*left),
            right: r(*right),
        },
        InstKind::UnaryOp { dst, op, operand } => InstKind::UnaryOp {
            dst: r(*dst),
            op: *op,
            operand: r(*operand),
        },
        InstKind::FieldGet { dst, object, field } => InstKind::FieldGet {
            dst: r(*dst),
            object: r(*object),
            field: *field,
        },

        // Functions
        InstKind::LoadFunction { dst, id } => InstKind::LoadFunction {
            dst: r(*dst),
            id: *id,
        },
        InstKind::FunctionCall {
            dst,
            callee,
            args,
            context_uses,
            context_defs,
        } => {
            let callee = match callee {
                Callee::Direct(id) => Callee::Direct(*id),
                Callee::Indirect(v) => Callee::Indirect(r(*v)),
            };
            InstKind::FunctionCall {
                dst: r(*dst),
                callee,
                args: rv(args),
                context_uses: context_uses.iter().map(|(id, v)| (*id, r(*v))).collect(),
                context_defs: context_defs.iter().map(|(id, v)| (*id, r(*v))).collect(),
            }
        }
        InstKind::Spawn {
            dst,
            callee,
            args,
            context_uses,
        } => {
            let callee = match callee {
                Callee::Direct(id) => Callee::Direct(*id),
                Callee::Indirect(v) => Callee::Indirect(r(*v)),
            };
            let ctx = context_uses.iter().map(|(id, v)| (*id, r(*v))).collect();
            InstKind::Spawn {
                dst: r(*dst),
                callee,
                args: rv(args),
                context_uses: ctx,
            }
        }
        InstKind::Eval {
            dst,
            src,
            context_defs,
        } => {
            let ctx = context_defs.iter().map(|(id, v)| (*id, r(*v))).collect();
            InstKind::Eval {
                dst: r(*dst),
                src: r(*src),
                context_defs: ctx,
            }
        }

        // Composite constructors
        InstKind::MakeDeque { dst, elements } => InstKind::MakeDeque {
            dst: r(*dst),
            elements: rv(elements),
        },
        InstKind::MakeObject { dst, fields } => InstKind::MakeObject {
            dst: r(*dst),
            fields: fields.iter().map(|(name, v)| (*name, r(*v))).collect(),
        },
        InstKind::MakeRange {
            dst,
            start,
            end,
            kind,
        } => InstKind::MakeRange {
            dst: r(*dst),
            start: r(*start),
            end: r(*end),
            kind: *kind,
        },
        InstKind::MakeTuple { dst, elements } => InstKind::MakeTuple {
            dst: r(*dst),
            elements: rv(elements),
        },
        InstKind::TupleIndex { dst, tuple, index } => InstKind::TupleIndex {
            dst: r(*dst),
            tuple: r(*tuple),
            index: *index,
        },

        // Pattern matching
        InstKind::TestLiteral { dst, src, value } => InstKind::TestLiteral {
            dst: r(*dst),
            src: r(*src),
            value: value.clone(),
        },
        InstKind::TestListLen {
            dst,
            src,
            min_len,
            exact,
        } => InstKind::TestListLen {
            dst: r(*dst),
            src: r(*src),
            min_len: *min_len,
            exact: *exact,
        },
        InstKind::TestObjectKey { dst, src, key } => InstKind::TestObjectKey {
            dst: r(*dst),
            src: r(*src),
            key: *key,
        },
        InstKind::TestRange {
            dst,
            src,
            start,
            end,
            kind,
        } => InstKind::TestRange {
            dst: r(*dst),
            src: r(*src),
            start: *start,
            end: *end,
            kind: *kind,
        },
        InstKind::ListIndex { dst, list, index } => InstKind::ListIndex {
            dst: r(*dst),
            list: r(*list),
            index: *index,
        },
        InstKind::ListGet { dst, list, index } => InstKind::ListGet {
            dst: r(*dst),
            list: r(*list),
            index: r(*index),
        },
        InstKind::ListSlice {
            dst,
            list,
            skip_head,
            skip_tail,
        } => InstKind::ListSlice {
            dst: r(*dst),
            list: r(*list),
            skip_head: *skip_head,
            skip_tail: *skip_tail,
        },
        InstKind::ObjectGet { dst, object, key } => InstKind::ObjectGet {
            dst: r(*dst),
            object: r(*object),
            key: *key,
        },

        // Closures
        InstKind::MakeClosure {
            dst,
            body,
            captures,
        } => InstKind::MakeClosure {
            dst: r(*dst),
            body: rl(*body),
            captures: rv(captures),
        },

        // Iterator
        InstKind::IterStep {
            dst,
            iter_src,
            iter_dst,
            done,
            done_args,
        } => InstKind::IterStep {
            dst: r(*dst),
            iter_src: r(*iter_src),
            iter_dst: r(*iter_dst),
            done: rl(*done),
            done_args: done_args.iter().map(|v| r(*v)).collect(),
        },

        // Variant
        InstKind::MakeVariant { dst, tag, payload } => InstKind::MakeVariant {
            dst: r(*dst),
            tag: *tag,
            payload: payload.map(&r),
        },
        InstKind::TestVariant { dst, src, tag } => InstKind::TestVariant {
            dst: r(*dst),
            src: r(*src),
            tag: *tag,
        },
        InstKind::UnwrapVariant { dst, src } => InstKind::UnwrapVariant {
            dst: r(*dst),
            src: r(*src),
        },

        // Control flow
        InstKind::BlockLabel {
            label,
            params,
            merge_of,
        } => InstKind::BlockLabel {
            label: rl(*label),
            params: rv(params),
            merge_of: merge_of.map(&rl),
        },
        InstKind::Jump { label, args } => InstKind::Jump {
            label: rl(*label),
            args: rv(args),
        },
        InstKind::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } => InstKind::JumpIf {
            cond: r(*cond),
            then_label: rl(*then_label),
            then_args: rv(then_args),
            else_label: rl(*else_label),
            else_args: rv(else_args),
        },
        InstKind::Return(v) => InstKind::Return(r(*v)),
        InstKind::Nop => InstKind::Nop,

        // Cast
        InstKind::Cast { dst, src, kind } => InstKind::Cast {
            dst: r(*dst),
            src: r(*src),
            kind: *kind,
        },

        // Poison
        InstKind::Poison { dst } => InstKind::Poison { dst: r(*dst) },
    }
}

#[cfg(test)]
mod tests {
    use crate::ty::Ty;

    use super::*;
    use acvus_ast::Span;
    use acvus_utils::LocalFactory;

    fn make_inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::ZERO,
            kind,
        }
    }

    fn make_body(insts: Vec<InstKind>, val_count: usize) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for i in 0..val_count {
            let v = factory.next();
            val_types.insert(v, Ty::Int);
        }
        MirBody {
            insts: insts.into_iter().map(make_inst).collect(),
            val_types,
            param_regs: Vec::new(),
            capture_regs: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        }
    }

    fn make_module(body: MirBody) -> MirModule {
        MirModule {
            main: body,
            closures: FxHashMap::default(),
        }
    }

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    #[test]
    fn inline_simple_call() {
        // caller: r0 = const 1; r1 = call f(r0); yield r1
        // callee f: r0 = param; r1 = r0 + r0; return r1
        let callee_id = FunctionId::alloc();

        let callee_body = make_body(
            vec![
                InstKind::VarLoad {
                    dst: v(0),
                    name: acvus_utils::Interner::new().intern("_0"),
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return(v(1)),
            ],
            2,
        );

        let caller_body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(callee_id),
                    args: vec![v(0)],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            2,
        );

        let mut modules = FxHashMap::default();
        modules.insert(callee_id, make_module(callee_body));
        let caller_id = FunctionId::alloc();
        modules.insert(caller_id, make_module(caller_body));

        let result = inline(&modules, &FxHashSet::default());
        let inlined = &result.modules[&caller_id];

        // After inlining, there should be no FunctionCall instruction.
        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(!has_call, "FunctionCall should be inlined away");

        // Should have a BinOp (from callee) and a Yield.
        let has_binop = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::BinOp { .. }));
        assert!(has_binop, "callee's BinOp should be present after inlining");

        let has_yield = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::Return(_)));
        assert!(has_yield, "Yield should remain");
    }

    #[test]
    fn inline_preserves_extern_call() {
        // call to an extern function (not in modules) should stay.
        let extern_id = FunctionId::alloc();

        let caller_body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(extern_id),
                    args: vec![v(0)],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            2,
        );

        let caller_id = FunctionId::alloc();
        let mut modules = FxHashMap::default();
        modules.insert(caller_id, make_module(caller_body));
        // extern_id is NOT in modules → cannot be inlined.

        let result = inline(&modules, &FxHashSet::default());
        let inlined = &result.modules[&caller_id];

        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(has_call, "extern call should remain");
    }

    #[test]
    fn inline_skips_recursive() {
        let rec_id = FunctionId::alloc();

        let rec_body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Return(v(0)),
            ],
            1,
        );

        let caller_body = make_body(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(rec_id),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(0)),
            ],
            1,
        );

        let caller_id = FunctionId::alloc();
        let mut modules = FxHashMap::default();
        modules.insert(rec_id, make_module(rec_body));
        modules.insert(caller_id, make_module(caller_body));

        let mut recursive = FxHashSet::default();
        recursive.insert(rec_id);

        let result = inline(&modules, &recursive);
        let inlined = &result.modules[&caller_id];

        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(has_call, "recursive call should NOT be inlined");
    }

    #[test]
    fn inline_chain() {
        // g: return 42
        // f: return g()
        // main: yield f()
        // After inlining: main should have Const(42) + Yield, no calls.
        let g_id = FunctionId::alloc();
        let f_id = FunctionId::alloc();
        let main_id = FunctionId::alloc();

        let g_body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(42),
                },
                InstKind::Return(v(0)),
            ],
            1,
        );

        let f_body = make_body(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(g_id),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(0)),
            ],
            1,
        );

        let main_body = make_body(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(f_id),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(0)),
            ],
            1,
        );

        let mut modules = FxHashMap::default();
        modules.insert(g_id, make_module(g_body));
        modules.insert(f_id, make_module(f_body));
        modules.insert(main_id, make_module(main_body));

        let result = inline(&modules, &FxHashSet::default());
        let inlined = &result.modules[&main_id];

        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(!has_call, "all calls should be inlined in chain");

        let has_const = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::Const { .. }));
        assert!(has_const, "g's Const(42) should be present");
    }

    #[test]
    fn inline_indirect_call_preserved() {
        // Indirect call (closure) should never be inlined.
        let caller_body = make_body(
            vec![
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Indirect(v(0)),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            2,
        );

        let caller_id = FunctionId::alloc();
        let mut modules = FxHashMap::default();
        modules.insert(caller_id, make_module(caller_body));

        let result = inline(&modules, &FxHashSet::default());
        let inlined = &result.modules[&caller_id];

        let has_call = inlined.main.insts.iter().any(|i| {
            matches!(
                i.kind,
                InstKind::FunctionCall {
                    callee: Callee::Indirect(_),
                    ..
                }
            )
        });
        assert!(has_call, "indirect call should remain");
    }
}
