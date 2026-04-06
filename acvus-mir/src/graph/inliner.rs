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

use super::types::QualifiedRef;

/// Result of the inlining pass.
#[derive(Debug)]
pub struct InlineResult {
    /// Inlined MIR per top-level function.
    pub modules: FxHashMap<QualifiedRef, MirModule>,
}

/// Inline all local function calls within each top-level function's MIR.
///
/// `modules`: per-function MIR from the lower phase.
/// `recursive_fns`: functions that should NOT be inlined (recursive / mutual recursive).
pub fn inline(
    modules: &FxHashMap<QualifiedRef, MirModule>,
    recursive_fns: &FxHashSet<QualifiedRef>,
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
    all_modules: &FxHashMap<QualifiedRef, MirModule>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> MirModule {
    let body = inline_body(&module.main, all_modules, recursive_fns, &module.closures);

    // Inline within closures too.
    let mut closures = FxHashMap::default();
    for (label, closure) in &module.closures {
        let mut inlined_body = inline_body(closure, all_modules, recursive_fns, &module.closures);
        inlined_body.captures = closure.captures.clone();
        inlined_body.params = closure.params.clone();
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
///
/// Handles both:
/// - **Direct calls** to local functions (from `all_modules`)
/// - **Indirect calls** where the callee is a known MakeClosure (devirtualization)
fn inline_body(
    body: &MirBody,
    all_modules: &FxHashMap<QualifiedRef, MirModule>,
    recursive_fns: &FxHashSet<QualifiedRef>,
    closures: &FxHashMap<Label, MirBody>,
) -> MirBody {
    let mut current = body.clone();

    loop {
        let mut changed = false;
        let mut new_insts = Vec::new();
        let mut val_remap: FxHashMap<ValueId, ValueId> = FxHashMap::default();

        // Build def_map for devirtualization: ValueId → instruction index.
        let def_map: FxHashMap<ValueId, usize> = current
            .insts
            .iter()
            .enumerate()
            .flat_map(|(idx, inst)| {
                crate::analysis::inst_info::defs(&inst.kind)
                    .into_iter()
                    .map(move |d| (d, idx))
            })
            .collect();

        for inst in &current.insts {
            // Try to resolve callee body for inlining.
            let inline_target = match &inst.kind {
                // Direct call to a local function.
                InstKind::FunctionCall {
                    dst,
                    callee: Callee::Direct(callee_id),
                    args,
                    ..
                } if !recursive_fns.contains(callee_id) && all_modules.contains_key(callee_id) => {
                    let callee_body = &all_modules[callee_id].main;
                    Some((*dst, callee_body, args.clone(), Vec::new()))
                }

                // Indirect call — try devirtualization.
                InstKind::FunctionCall {
                    dst,
                    callee: Callee::Indirect(callee_val),
                    args,
                    ..
                } => {
                    let callee_val = remap_one(*callee_val, &val_remap);
                    try_devirt(&current.insts, &def_map, callee_val, closures)
                        .map(|(callee_body, captures)| (*dst, callee_body, args.clone(), captures))
                }

                _ => None,
            };

            if let Some((dst, callee_body, args, captures)) = inline_target {
                // Apply val_remap to args: earlier inlinings may have replaced
                // the original dst with a new value.
                let args: Vec<ValueId> = captures
                    .iter()
                    .chain(args.iter())
                    .map(|a| remap_one(*a, &val_remap))
                    .collect();

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

                // Map callee capture_regs and param_regs directly to caller args.
                // LLVM-style: direct SSA value substitution.
                let n_captures = callee_body.captures.len();
                let mut substituted_regs: FxHashSet<ValueId> = FxHashSet::default();
                for ((_, cap_reg), arg) in callee_body.captures.iter().zip(args.iter()) {
                    callee_remap.insert(*cap_reg, *arg);
                    substituted_regs.insert(*cap_reg);
                }
                for ((_, param_reg), arg) in callee_body.params.iter().zip(args[n_captures..].iter())
                {
                    callee_remap.insert(*param_reg, *arg);
                    substituted_regs.insert(*param_reg);
                }

                // Copy callee's val_types (remapped).
                // Skip types for substituted regs — caller already has types for those.
                for (&old_val, ty) in &callee_body.val_types {
                    if substituted_regs.contains(&old_val) {
                        continue;
                    }
                    if let Some(&new_val) = callee_remap.get(&old_val) {
                        current.val_types.insert(new_val, ty.clone());
                    }
                }

                // Copy callee's debug info (remapped).
                for (&old_val, origin) in &callee_body.debug.val_origins {
                    if substituted_regs.contains(&old_val) {
                        continue;
                    }
                    if let Some(&new_val) = callee_remap.get(&old_val) {
                        current.debug.val_origins.insert(new_val, origin.clone());
                    }
                }

                // Emit callee instructions with remapped ids.
                for callee_inst in &callee_body.insts {
                    match &callee_inst.kind {
                        InstKind::Return(val) => {
                            let remapped_val = remap_one(*val, &callee_remap);
                            val_remap.insert(dst, remapped_val);
                        }

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
            } else {
                // Non-inlineable: emit as-is, applying val_remap.
                let remapped = remap_inst(&inst.kind, &val_remap, 0);
                new_insts.push(Inst {
                    span: inst.span,
                    kind: remapped,
                });
            }
        }

        current.insts = new_insts;

        if !changed {
            break;
        }
    }

    current
}

/// Try to devirtualize an indirect call: if the callee ValueId is defined by
/// a single MakeClosure (not from a phi), return the closure's body and captures.
fn try_devirt<'a>(
    insts: &[Inst],
    def_map: &FxHashMap<ValueId, usize>,
    callee_val: ValueId,
    closures: &'a FxHashMap<Label, MirBody>,
) -> Option<(&'a MirBody, Vec<ValueId>)> {
    let &def_idx = def_map.get(&callee_val)?;
    let inst = &insts[def_idx];
    match &inst.kind {
        InstKind::MakeClosure { body, captures, .. } => {
            let closure_body = closures.get(body)?;
            Some((closure_body, captures.clone()))
        }
        _ => None,
    }
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

        // Projection
        InstKind::Ref { dst, target, path } => {
            let new_target = match target {
                crate::ir::RefTarget::Var(slot) => crate::ir::RefTarget::Var(r(*slot)),
                crate::ir::RefTarget::Param(slot) => crate::ir::RefTarget::Param(r(*slot)),
                crate::ir::RefTarget::Context(qref) => crate::ir::RefTarget::Context(*qref),
            };
            InstKind::Ref {
                dst: r(*dst),
                target: new_target,
                path: path.clone(),
            }
        }
        InstKind::Load { dst, src, volatile } => InstKind::Load {
            dst: r(*dst),
            src: r(*src),
            volatile: *volatile,
        },
        InstKind::Store {
            dst,
            value,
            volatile,
        } => InstKind::Store {
            dst: r(*dst),
            value: r(*value),
            volatile: *volatile,
        },

        // Scalar field access
        InstKind::FieldGet {
            dst,
            object,
            field,
            rest,
        } => InstKind::FieldGet {
            dst: r(*dst),
            object: r(*object),
            field: *field,
            rest: rest.clone(),
        },
        InstKind::FieldSet {
            dst,
            object,
            field,
            rest,
            value,
        } => InstKind::FieldSet {
            dst: r(*dst),
            object: r(*object),
            field: *field,
            rest: rest.clone(),
            value: r(*value),
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
        InstKind::ListStep {
            dst,
            list,
            index_src,
            index_dst,
            done,
            done_args,
        } => InstKind::ListStep {
            dst: r(*dst),
            list: r(*list),
            index_src: r(*index_src),
            index_dst: r(*index_dst),
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

        // Clone / Drop
        InstKind::Clone { dst, src } => InstKind::Clone {
            dst: r(*dst),
            src: r(*src),
        },
        InstKind::Drop { src } => InstKind::Drop { src: r(*src) },

        // Poison / Undef
        InstKind::Poison { dst } => InstKind::Poison { dst: r(*dst) },
        InstKind::Undef { dst } => InstKind::Undef { dst: r(*dst) },
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
        for _ in 0..val_count {
            let v = factory.next();
            val_types.insert(v, Ty::Int);
        }
        MirBody {
            insts: insts.into_iter().map(make_inst).collect(),
            val_types,
            params: Vec::new(),
            captures: Vec::new(),
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
        let i = acvus_utils::Interner::new();
        let callee_id = QualifiedRef::root(i.intern("callee"));

        let mut callee_body = make_body(
            vec![
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
        callee_body.params = vec![(i.intern("p0"), v(0))];

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
        let caller_id = QualifiedRef::root(i.intern("caller"));
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
        let i = acvus_utils::Interner::new();
        let extern_id = QualifiedRef::root(i.intern("ext"));

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

        let caller_id = QualifiedRef::root(i.intern("caller"));
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
        let i = acvus_utils::Interner::new();
        let rec_id = QualifiedRef::root(i.intern("rec"));

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

        let caller_id = QualifiedRef::root(i.intern("caller"));
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
        let i = acvus_utils::Interner::new();
        let g_id = QualifiedRef::root(i.intern("g"));
        let f_id = QualifiedRef::root(i.intern("f"));
        let main_id = QualifiedRef::root(i.intern("main"));

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

        let i = acvus_utils::Interner::new();
        let caller_id = QualifiedRef::root(i.intern("caller"));
        let mut modules = FxHashMap::default();
        modules.insert(caller_id, make_module(caller_body));

        let result = inline(&modules, &FxHashSet::default());
        let inlined = &result.modules[&caller_id];

        let has_call = inlined.main.insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::FunctionCall {
                    callee: Callee::Indirect(_),
                    ..
                }
            )
        });
        assert!(has_call, "indirect call should remain");
    }
}
