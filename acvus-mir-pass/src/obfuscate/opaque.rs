//! Opaque predicates: insert conditional branches where the condition always
//! evaluates to true, but is hard to prove statically.
//!
//! 4 opaque closure variants (all `(Int) -> Int`, always return 1):
//!   0. (x - x) + 1
//!   1. (x*x+1) / (x*x+1)
//!   2. (x ^ x) + 1
//!   3. (x*x) % 1 + 1
//!
//! The Int(1) result is stored in `__entangle` variable, which text decrypt
//! reads to derive dispatch indices. This creates a dependency chain:
//! removing opaque predicates breaks text decryption.
//!
//! Each predicate is a closure dispatched via a table. The dispatch index
//! is derived at runtime from a value (`rv % 4` or `rand_val % 4`),
//! so static analysis cannot determine which closure is called.

use std::sync::Arc;

use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::builtins::BuiltinId;
use acvus_mir::ir::{
    ClosureBody, DebugInfo, Inst, InstKind, Label, MirBody, MirModule, ValOrigin, ValueId,
};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rand::Rng;
use rand::rngs::StdRng;

use super::rewriter::PassState;

/// Number of opaque predicates to insert per body.
const PREDICATES_PER_BODY: usize = 1;

/// Labels for the 4 opaque closure bodies, allocated once per module.
pub struct OpaqueTable {
    pub labels: [Label; 4],
}

const OPAQUE_NAMES: [&str; 4] = [
    "opaque_sub_zero",
    "opaque_sq_nonneg",
    "opaque_xor_zero",
    "opaque_sq_plus_one",
];

/// Register 4 opaque closure bodies into the module.
pub fn register_opaque_closures(module: &mut MirModule, interner: &Interner) -> OpaqueTable {
    let base_label = module.closures.keys().map(|l| l.0).max().unwrap_or(0) + 1;
    let base_label = base_label.max(module.main.label_count);
    for closure in module.closures.values() {
        let _ = base_label.max(closure.body.label_count);
    }

    let mut labels = [Label(0); 4];
    for (i, _name) in OPAQUE_NAMES.iter().enumerate() {
        let label = Label(base_label + i as u32);
        labels[i] = label;
        let body = make_opaque_closure_body(i as u32, interner);
        module.closures.insert(label, Arc::new(body));
    }

    let max_label = base_label + 4;
    if module.main.label_count < max_label {
        module.main.label_count = max_label;
    }

    OpaqueTable { labels }
}

/// Build the MIR closure body for opaque variant `idx`.
///
/// Params: [x: Int (Val 0)]
/// Returns: Int (always 1)
fn make_opaque_closure_body(variant: u32, interner: &Interner) -> ClosureBody {
    let mut body = MirBody::new();
    let mut debug = DebugInfo::new();

    let v_x = ValueId(0);
    body.val_count = 1;
    body.val_types.insert(v_x, Ty::Int);
    debug.set(v_x, ValOrigin::Named(interner.intern("x")));

    let mut next_val = 1u32;
    let mut alloc = |ty: Ty| -> ValueId {
        let v = ValueId(next_val);
        next_val += 1;
        body.val_types.insert(v, ty);
        debug.set(v, ValOrigin::Expr);
        v
    };
    let span = Span { start: 0, end: 0 };

    let mut insts = Vec::new();

    let v_result = match variant {
        0 => {
            // (x - x) + 1 → always 1
            let v_sub = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_sub,
                    op: BinOp::Sub,
                    left: v_x,
                    right: v_x,
                },
            });
            let v_one = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_one,
                    value: Literal::Int(1),
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Add,
                    left: v_sub,
                    right: v_one,
                },
            });
            v_res
        }
        1 => {
            // (x*x+1) / (x*x+1) → always 1
            let v_sq = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_sq,
                    op: BinOp::Mul,
                    left: v_x,
                    right: v_x,
                },
            });
            let v_one = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_one,
                    value: Literal::Int(1),
                },
            });
            let v_sum = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_sum,
                    op: BinOp::Add,
                    left: v_sq,
                    right: v_one,
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Div,
                    left: v_sum,
                    right: v_sum,
                },
            });
            v_res
        }
        2 => {
            // (x ^ x) + 1 → always 1
            let v_xor = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_xor,
                    op: BinOp::Xor,
                    left: v_x,
                    right: v_x,
                },
            });
            let v_one = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_one,
                    value: Literal::Int(1),
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Add,
                    left: v_xor,
                    right: v_one,
                },
            });
            v_res
        }
        _ => {
            // (x*x) % 1 + 1 → always 1
            let v_sq = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_sq,
                    op: BinOp::Mul,
                    left: v_x,
                    right: v_x,
                },
            });
            let v_one = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_one,
                    value: Literal::Int(1),
                },
            });
            let v_mod = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mod,
                    op: BinOp::Mod,
                    left: v_sq,
                    right: v_one,
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Add,
                    left: v_mod,
                    right: v_one,
                },
            });
            v_res
        }
    };

    insts.push(Inst {
        span,
        kind: InstKind::Return(v_result),
    });

    body.insts = insts;
    body.val_count = next_val;
    body.label_count = 0;
    body.debug = debug;

    ClosureBody {
        capture_names: vec![],
        param_names: vec![interner.intern("x")],
        body,
    }
}

// ── Insertion ──────────────────────────────────────────────────

pub fn insert(
    insts: Vec<Inst>,
    ctx: &mut PassState,
    rng: &mut StdRng,
    decrypt_dispatch: Option<ValueId>,
) -> Vec<Inst> {
    insert_inner(insts, ctx, rng, decrypt_dispatch)
}

fn insert_inner(
    insts: Vec<Inst>,
    ctx: &mut PassState,
    rng: &mut StdRng,
    decrypt_dispatch: Option<ValueId>,
) -> Vec<Inst> {
    if insts.len() < 4 {
        return insts;
    }

    let mut result = insts;

    for _ in 0..PREDICATES_PER_BODY {
        if result.len() < 3 {
            break;
        }
        let pos = find_insertion_point(&result, rng);
        if let Some(pos) = pos {
            let span = result[pos].span;

            let runtime_val = find_runtime_int_val(&result[..pos], ctx);

            let predicate_insts = make_closure_predicate(ctx, rng, span, runtime_val);
            let dead = make_dead_block(ctx, rng, span, decrypt_dispatch);

            let mut new_result =
                Vec::with_capacity(result.len() + predicate_insts.len() + dead.len() + 4);

            new_result.extend(result[..pos].iter().cloned());

            let continue_label = ctx.alloc_label();
            let dead_label = ctx.alloc_label();

            new_result.extend(predicate_insts);

            let cond_val = find_last_defined_val(&new_result).unwrap();

            new_result.push(Inst {
                span,
                kind: InstKind::JumpIf {
                    cond: cond_val,
                    then_label: continue_label,
                    then_args: vec![],
                    else_label: dead_label,
                    else_args: vec![],
                },
            });

            new_result.push(Inst {
                span,
                kind: InstKind::BlockLabel {
                    label: dead_label,
                    params: vec![],
                    merge_of: None,
                },
            });
            new_result.extend(dead);
            new_result.push(Inst {
                span,
                kind: InstKind::Jump {
                    label: continue_label,
                    args: vec![],
                },
            });

            new_result.push(Inst {
                span,
                kind: InstKind::BlockLabel {
                    label: continue_label,
                    params: vec![],
                    merge_of: None,
                },
            });

            new_result.extend(result[pos..].iter().cloned());

            result = new_result;
        }
    }

    result
}

/// Generate opaque predicate instructions using 2-level factory dispatch.
///
/// The closure returns Int(1), which is stored in `__entangle` variable
/// (used by text decrypt to derive dispatch indices). Then Int→Bool
/// conversion (1 > 0 = true) produces the JumpIf condition.
///
/// 2-level dispatch:
///   v_factory = ListGet(__opaque_table, arg % 4)
///   v_inner   = CallClosure(v_factory, [seed])
///   v_fn      = ListGet(v_inner, inner_idx)      // inner_idx=0 (rotation=factory_idx)
///   v_entangle = CallClosure(v_fn, [arg])         // → Int(1)
///   VarStore("__entangle", v_entangle)
///   v_cond = v_entangle > 0                       // → Bool(true)
fn make_closure_predicate(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    runtime_val: Option<ValueId>,
) -> Vec<Inst> {
    let mut out = Vec::new();

    let v_arg = if let Some(rv) = runtime_val {
        rv
    } else {
        let rand_val: i64 = rng.random_range(1..=1_000_000);
        let v_x = ctx.alloc_val(Ty::Int);
        out.push(Inst {
            span,
            kind: InstKind::Const {
                dst: v_x,
                value: Literal::Int(rand_val),
            },
        });
        v_x
    };

    let inner_fn_ty = Ty::Fn {
        params: vec![Ty::Int],
        ret: Box::new(Ty::Int),
        is_extern: false,
    };
    let factory_fn_ty = Ty::Fn {
        params: vec![Ty::Int],
        ret: Box::new(Ty::List(Box::new(inner_fn_ty.clone()))),
        is_extern: false,
    };

    // Load opaque meta table from variable (stores factory dispatch table).
    let meta_ty = Ty::List(Box::new(factory_fn_ty.clone()));
    let v_meta = ctx.alloc_val(meta_ty);
    out.push(Inst {
        span,
        kind: InstKind::VarLoad {
            dst: v_meta,
            name: ctx.interner.intern("__opaque_table"),
        },
    });

    // Emit local Const(4) — cannot share across CFF blocks.
    let v_four = ctx.alloc_val(Ty::Int);
    out.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_four,
            value: Literal::Int(4),
        },
    });

    // factory_idx = v_arg % 4
    let v_factory_idx = ctx.alloc_val(Ty::Int);
    out.push(Inst {
        span,
        kind: InstKind::BinOp {
            dst: v_factory_idx,
            op: BinOp::Mod,
            left: v_arg,
            right: v_four,
        },
    });

    // v_factory = ListGet(meta_table, factory_idx)
    let v_factory = ctx.alloc_val(factory_fn_ty);
    out.push(Inst {
        span,
        kind: InstKind::ListGet {
            dst: v_factory,
            list: v_meta,
            index: v_factory_idx,
        },
    });

    // v_inner_table = CallClosure(v_factory, [seed=0])
    let v_seed = ctx.alloc_val(Ty::Int);
    out.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_seed,
            value: Literal::Int(0),
        },
    });
    let inner_table_ty = Ty::List(Box::new(inner_fn_ty.clone()));
    let v_inner_table = ctx.alloc_val(inner_table_ty);
    out.push(Inst {
        span,
        kind: InstKind::ClosureCall {
            dst: v_inner_table,
            closure: v_factory,
            args: vec![v_seed],
        },
    });

    // inner_idx = 0 (rotation == factory_idx, target == factory_idx, so (target - rotation + 4) % 4 = 0)
    let v_inner_idx = ctx.alloc_val(Ty::Int);
    out.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_inner_idx,
            value: Literal::Int(0),
        },
    });

    // v_fn = ListGet(v_inner_table, inner_idx)
    let v_fn = ctx.alloc_val(inner_fn_ty);
    out.push(Inst {
        span,
        kind: InstKind::ListGet {
            dst: v_fn,
            list: v_inner_table,
            index: v_inner_idx,
        },
    });

    // v_entangle = CallClosure(v_fn, [v_arg]) → Int(1)
    let v_entangle = ctx.alloc_val(Ty::Int);
    out.push(Inst {
        span,
        kind: InstKind::ClosureCall {
            dst: v_entangle,
            closure: v_fn,
            args: vec![v_arg],
        },
    });

    // Store entangle value for text decrypt dependency
    out.push(Inst {
        span,
        kind: InstKind::VarStore {
            name: ctx.interner.intern("__entangle"),
            src: v_entangle,
        },
    });

    // Int → Bool conversion: 1 > 0 = true
    let v_zero = ctx.alloc_val(Ty::Int);
    out.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_zero,
            value: Literal::Int(0),
        },
    });
    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst {
        span,
        kind: InstKind::BinOp {
            dst: v_cond,
            op: BinOp::Gt,
            left: v_entangle,
            right: v_zero,
        },
    });

    out
}

fn find_insertion_point(insts: &[Inst], rng: &mut StdRng) -> Option<usize> {
    let valid: Vec<usize> = (1..insts.len())
        .filter(|&i| {
            !matches!(
                insts[i].kind,
                InstKind::BlockLabel { .. }
                    | InstKind::Jump { .. }
                    | InstKind::JumpIf { .. }
                    | InstKind::Return(_)
            ) && !matches!(
                insts[i - 1].kind,
                InstKind::Jump { .. } | InstKind::JumpIf { .. } | InstKind::Return(_)
            )
        })
        .collect();

    if valid.is_empty() {
        None
    } else {
        Some(valid[rng.random_range(0..valid.len())])
    }
}

/// Scan backwards through instructions to find a ContextLoad/VarLoad with Int type
/// in the same basic block.
fn find_runtime_int_val(insts: &[Inst], ctx: &PassState) -> Option<ValueId> {
    for inst in insts.iter().rev() {
        match &inst.kind {
            InstKind::BlockLabel { .. }
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::Return(_) => return None,
            InstKind::ContextLoad { dst, .. } | InstKind::VarLoad { dst, .. } => {
                if ctx.val_types.get(dst) == Some(&Ty::Int) {
                    return Some(*dst);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_last_defined_val(insts: &[Inst]) -> Option<ValueId> {
    for inst in insts.iter().rev() {
        let val = match &inst.kind {
            InstKind::Const { dst, .. }
            | InstKind::BinOp { dst, .. }
            | InstKind::UnaryOp { dst, .. }
            | InstKind::BuiltinCall { dst, .. }
            | InstKind::ExternCall { dst, .. }
            | InstKind::ClosureCall { dst, .. } => Some(*dst),
            _ => None,
        };
        if val.is_some() {
            return val;
        }
    }
    None
}

/// Generate dead block contents: emit garbage strings so fake paths look like
/// real output paths. Uses 2-level factory dispatch pattern to match real code.
fn make_dead_block(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    meta_c_table: Option<ValueId>,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let count = rng.random_range(1..4);

    for _ in 0..count {
        if let Some(meta_c) = meta_c_table {
            // Emit a fake 2-level factory dispatch (mimics real decrypt pattern)
            let len = rng.random_range(4..20);
            let garbage: Vec<u8> = (0..len).map(|_| rng.random_range(0u8..=255)).collect();
            let key: i64 = rng.random_range(1..=i64::MAX);
            let factory_idx_val = rng.random_range(0i64..4);
            let inner_idx_val = rng.random_range(0i64..4);

            let v_bytes = ctx.alloc_val(Ty::bytes());
            out.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_bytes,
                    value: Literal::List(garbage.iter().map(|b| Literal::Byte(*b)).collect()),
                },
            });

            let v_key = ctx.alloc_val(Ty::Int);
            out.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_key,
                    value: Literal::Int(key),
                },
            });

            let v_factory_idx = ctx.alloc_val(Ty::Int);
            out.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_factory_idx,
                    value: Literal::Int(factory_idx_val),
                },
            });

            // 2-level dispatch: ListGet factory → CallClosure → ListGet inner → CallClosure
            let inner_fn_ty = Ty::Fn {
                params: vec![Ty::bytes(), Ty::Int],
                ret: Box::new(Ty::String),
                is_extern: false,
            };
            let factory_fn_ty = Ty::Fn {
                params: vec![Ty::Int],
                ret: Box::new(Ty::List(Box::new(inner_fn_ty.clone()))),
                is_extern: false,
            };

            let v_factory = ctx.alloc_val(factory_fn_ty);
            out.push(Inst {
                span,
                kind: InstKind::ListGet {
                    dst: v_factory,
                    list: meta_c,
                    index: v_factory_idx,
                },
            });

            let v_seed = ctx.alloc_val(Ty::Int);
            out.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_seed,
                    value: Literal::Int(0),
                },
            });

            let v_inner_table = ctx.alloc_val(Ty::List(Box::new(inner_fn_ty.clone())));
            out.push(Inst {
                span,
                kind: InstKind::ClosureCall {
                    dst: v_inner_table,
                    closure: v_factory,
                    args: vec![v_seed],
                },
            });

            let v_inner_idx = ctx.alloc_val(Ty::Int);
            out.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_inner_idx,
                    value: Literal::Int(inner_idx_val),
                },
            });

            let v_fn = ctx.alloc_val(inner_fn_ty);
            out.push(Inst {
                span,
                kind: InstKind::ListGet {
                    dst: v_fn,
                    list: v_inner_table,
                    index: v_inner_idx,
                },
            });

            let v_result = ctx.alloc_val(Ty::String);
            out.push(Inst {
                span,
                kind: InstKind::ClosureCall {
                    dst: v_result,
                    closure: v_fn,
                    args: vec![v_bytes, v_key],
                },
            });

            out.push(Inst {
                span,
                kind: InstKind::Yield(v_result),
            });
        } else {
            let len = rng.random_range(2..6);
            let mut v_accum: Option<ValueId> = None;

            for _ in 0..len {
                let code: i64 = rng.random_range(32..127);
                let key: i64 = rng.random_range(1..256);
                let encrypted = code ^ key;

                let v_enc = ctx.alloc_val(Ty::Int);
                out.push(Inst {
                    span,
                    kind: InstKind::Const {
                        dst: v_enc,
                        value: Literal::Int(encrypted),
                    },
                });

                let v_key = ctx.alloc_val(Ty::Int);
                out.push(Inst {
                    span,
                    kind: InstKind::Const {
                        dst: v_key,
                        value: Literal::Int(key),
                    },
                });

                let v_dec = ctx.alloc_val(Ty::Int);
                out.push(Inst {
                    span,
                    kind: InstKind::BinOp {
                        dst: v_dec,
                        op: BinOp::Xor,
                        left: v_enc,
                        right: v_key,
                    },
                });

                let v_char = ctx.alloc_val(Ty::String);
                out.push(Inst {
                    span,
                    kind: InstKind::BuiltinCall {
                        dst: v_char,
                        builtin: BuiltinId::IntToChar,
                        args: vec![v_dec],
                    },
                });

                v_accum = Some(match v_accum {
                    None => v_char,
                    Some(prev) => {
                        let v_concat = ctx.alloc_val(Ty::String);
                        out.push(Inst {
                            span,
                            kind: InstKind::BinOp {
                                dst: v_concat,
                                op: BinOp::Add,
                                left: prev,
                                right: v_char,
                            },
                        });
                        v_concat
                    }
                });
            }

            out.push(Inst {
                span,
                kind: InstKind::Yield(v_accum.unwrap()),
            });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rustc_hash::FxHashMap;

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn make_ctx() -> PassState {
        let interner = Interner::new();
        PassState {
            insts: Vec::new(),
            val_types: FxHashMap::default(),
            debug: DebugInfo::new(),
            next_val: 200,
            next_label: 200,
            interner,
        }
    }

    #[test]
    fn inserts_opaque_branches() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);

        let insts: Vec<Inst> = (0..10)
            .map(|i| Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: ValueId(i),
                    value: Literal::Int(i as i64),
                },
            })
            .collect();

        let result = insert(insts, &mut ctx, &mut rng, None);

        let jumpif_count = result
            .iter()
            .filter(|i| matches!(i.kind, InstKind::JumpIf { .. }))
            .count();
        assert!(jumpif_count >= 1, "expected opaque predicate branches");

        let label_count = result
            .iter()
            .filter(|i| matches!(i.kind, InstKind::BlockLabel { .. }))
            .count();
        assert!(label_count >= 2, "expected dead + continue labels");
    }

    #[test]
    fn closure_predicate_produces_var_load_and_call() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);

        let rv = ctx.alloc_val(Ty::Int);
        let pred = make_closure_predicate(&mut ctx, &mut rng, span(), Some(rv));

        let opaque_table_name = ctx.interner.intern("__opaque_table");
        let has_var_load = pred.iter().any(
            |i| matches!(&i.kind, InstKind::VarLoad { name, .. } if *name == opaque_table_name),
        );
        assert!(has_var_load, "expected VarLoad for __opaque_table");

        let has_call_closure = pred
            .iter()
            .any(|i| matches!(i.kind, InstKind::ClosureCall { .. }));
        assert!(has_call_closure, "expected CallClosure in predicate");

        // Entangle: VarStore __entangle should exist
        let entangle_name = ctx.interner.intern("__entangle");
        let has_entangle_store = pred
            .iter()
            .any(|i| matches!(&i.kind, InstKind::VarStore { name, .. } if *name == entangle_name));
        assert!(has_entangle_store, "expected VarStore for __entangle");

        // Last instruction is BinOp Gt (Int→Bool conversion)
        let last = pred.last().unwrap();
        if let InstKind::BinOp {
            dst, op: BinOp::Gt, ..
        } = &last.kind
        {
            assert_eq!(ctx.val_types[dst], Ty::Bool);
        } else {
            panic!("last instruction should be BinOp Gt (Int→Bool)");
        }
    }

    #[test]
    fn closure_predicate_without_runtime_val_emits_const() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);

        let pred = make_closure_predicate(&mut ctx, &mut rng, span(), None);

        let has_const = pred
            .iter()
            .any(|i| matches!(i.kind, InstKind::Const { .. }));
        assert!(has_const, "expected Const when no runtime val");
    }

    #[test]
    fn prefers_runtime_variant_when_context_load_available() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(99);

        let rv = ValueId(0);
        ctx.val_types.insert(rv, Ty::Int);

        let mut insts = vec![Inst {
            span: span(),
            kind: InstKind::ContextLoad {
                dst: rv,
                name: ctx.interner.intern("count"),
            },
        }];
        for i in 1..8u32 {
            let dst = ValueId(i);
            ctx.val_types.insert(dst, Ty::Int);
            insts.push(Inst {
                span: span(),
                kind: InstKind::Const {
                    dst,
                    value: Literal::Int(i as i64),
                },
            });
        }

        let result = insert(insts, &mut ctx, &mut rng, None);

        let has_jumpif = result
            .iter()
            .any(|i| matches!(i.kind, InstKind::JumpIf { .. }));
        assert!(has_jumpif, "expected opaque predicate branch");

        let uses_runtime = result.iter().any(|i| match &i.kind {
            InstKind::BinOp {
                left,
                right,
                op: BinOp::Mod,
                ..
            } => *left == rv || *right == rv,
            _ => false,
        });
        assert!(
            uses_runtime,
            "expected runtime val to be used in index derivation"
        );
    }
}
