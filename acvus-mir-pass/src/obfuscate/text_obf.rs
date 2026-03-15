use std::sync::Arc;

use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::builtins::BuiltinId;
use acvus_mir::ir::DebugInfo;
use acvus_mir::ir::{
    ClosureBody, Inst, InstKind, Label, MirBody, MirModule, ValOrigin, ValueId,
};
use acvus_mir::ty::{Effect, FnKind, Ty};
use acvus_utils::Interner;
use rand::Rng;
use rand::rngs::StdRng;

use super::rewriter::PassState;

// ── Multi-stage decrypt ──────────────────────────────────────────
//
// 3-stage pipeline: key → subkey (A) → combined (B) → string (C)
//
// Stage A: (Int) → Int         — 4 variants of key transform
// Stage B: (Int, Int) → Int    — 4 variants of key combine
// Stage C: (Bytes, Int) → String — 4 variants of final decrypt
//
// Each stage dispatched via closures, creating deep dependency chains.

/// Labels for all 3 stages of decrypt closures, 4 variants each.
pub struct MultiStageDecryptTable {
    pub stage_a: [Label; 4], // (Int) -> Int
    pub stage_b: [Label; 4], // (Int, Int) -> Int
    pub stage_c: [Label; 4], // (Bytes, Int) -> String
}

/// Register 12 multi-stage decrypt closure bodies into the module.
pub fn register_multistage_decrypt_closures(
    module: &mut MirModule,
    interner: &Interner,
) -> MultiStageDecryptTable {
    let base_label = module.closures.keys().map(|l| l.0).max().unwrap_or(0) + 1;
    let base_label = base_label.max(module.main.label_count);
    for closure in module.closures.values() {
        let _ = base_label.max(closure.body.label_count);
    }

    let mut stage_a = [Label(0); 4];
    let mut stage_b = [Label(0); 4];
    let mut stage_c = [Label(0); 4];

    for i in 0..4u32 {
        let label = Label(base_label + i);
        stage_a[i as usize] = label;
        module
            .closures
            .insert(label, Arc::new(make_stage_a_closure_body(i, interner)));
    }
    for i in 0..4u32 {
        let label = Label(base_label + 4 + i);
        stage_b[i as usize] = label;
        module
            .closures
            .insert(label, Arc::new(make_stage_b_closure_body(i, interner)));
    }
    for i in 0..4u32 {
        let label = Label(base_label + 8 + i);
        stage_c[i as usize] = label;
        module
            .closures
            .insert(label, Arc::new(make_stage_c_closure_body(i, interner)));
    }

    let max_label = base_label + 12;
    if module.main.label_count < max_label {
        module.main.label_count = max_label;
    }

    MultiStageDecryptTable {
        stage_a,
        stage_b,
        stage_c,
    }
}

/// Get all labels from the multi-stage table (for filtering user closures).
pub fn all_decrypt_labels(table: &MultiStageDecryptTable) -> Vec<Label> {
    let mut v = Vec::with_capacity(12);
    v.extend_from_slice(&table.stage_a);
    v.extend_from_slice(&table.stage_b);
    v.extend_from_slice(&table.stage_c);
    v
}

// ── Stage A: transform_key ───────────────────────────────────────
// (Int) → Int — key → subkey

fn make_stage_a_closure_body(variant: u32, interner: &Interner) -> ClosureBody {
    let mut body = MirBody::new();
    let mut debug = DebugInfo::new();

    let v_key = ValueId(0);
    body.val_count = 1;
    body.val_types.insert(v_key, Ty::Int);
    debug.set(v_key, ValOrigin::Named(interner.intern("key")));

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

    let v_65537 = alloc(Ty::Int);
    insts.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_65537,
            value: Literal::Int(65537),
        },
    });

    let v_result = match variant {
        0 => {
            // (key * 7 + 3) % 65537
            let v_7 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_7,
                    value: Literal::Int(7),
                },
            });
            let v_mul = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mul,
                    op: BinOp::Mul,
                    left: v_key,
                    right: v_7,
                },
            });
            let v_3 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_3,
                    value: Literal::Int(3),
                },
            });
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_mul,
                    right: v_3,
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Mod,
                    left: v_add,
                    right: v_65537,
                },
            });
            v_res
        }
        1 => {
            // ((key ^ 0xDEAD) * 11 + 17) % 65537
            let v_dead = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_dead,
                    value: Literal::Int(0xDEAD),
                },
            });
            let v_xor = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_xor,
                    op: BinOp::Xor,
                    left: v_key,
                    right: v_dead,
                },
            });
            let v_11 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_11,
                    value: Literal::Int(11),
                },
            });
            let v_mul = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mul,
                    op: BinOp::Mul,
                    left: v_xor,
                    right: v_11,
                },
            });
            let v_17 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_17,
                    value: Literal::Int(17),
                },
            });
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_mul,
                    right: v_17,
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Mod,
                    left: v_add,
                    right: v_65537,
                },
            });
            v_res
        }
        2 => {
            // ((key >> 3) * 13 + 7) % 65537
            let v_3 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_3,
                    value: Literal::Int(3),
                },
            });
            let v_shr = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_shr,
                    op: BinOp::Shr,
                    left: v_key,
                    right: v_3,
                },
            });
            let v_13 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_13,
                    value: Literal::Int(13),
                },
            });
            let v_mul = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mul,
                    op: BinOp::Mul,
                    left: v_shr,
                    right: v_13,
                },
            });
            let v_7 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_7,
                    value: Literal::Int(7),
                },
            });
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_mul,
                    right: v_7,
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Mod,
                    left: v_add,
                    right: v_65537,
                },
            });
            v_res
        }
        _ => {
            // ((key * key + 5) % 65537 + 65537) % 65537
            let v_sq = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_sq,
                    op: BinOp::Mul,
                    left: v_key,
                    right: v_key,
                },
            });
            let v_5 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_5,
                    value: Literal::Int(5),
                },
            });
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_sq,
                    right: v_5,
                },
            });
            let v_m1 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_m1,
                    op: BinOp::Mod,
                    left: v_add,
                    right: v_65537,
                },
            });
            let v_add2 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add2,
                    op: BinOp::Add,
                    left: v_m1,
                    right: v_65537,
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Mod,
                    left: v_add2,
                    right: v_65537,
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
        param_names: vec![interner.intern("key")],
        body,
    }
}

/// Compile-time stage A transform (mirrors closure logic).
pub fn stage_a_transform(variant: u32, key: i64) -> i64 {
    let m = 65537i64;
    match variant {
        0 => ((key.wrapping_mul(7).wrapping_add(3)) % m + m) % m,
        1 => (((key ^ 0xDEAD).wrapping_mul(11).wrapping_add(17)) % m + m) % m,
        2 => (((key >> 3).wrapping_mul(13).wrapping_add(7)) % m + m) % m,
        _ => {
            let sq = key.wrapping_mul(key);
            ((sq.wrapping_add(5) % m) + m) % m
        }
    }
}

// ── Stage B: combine_keys ────────────────────────────────────────
// (Int, Int) → Int — (subkey, key) → combined_key

fn make_stage_b_closure_body(variant: u32, interner: &Interner) -> ClosureBody {
    let mut body = MirBody::new();
    let mut debug = DebugInfo::new();

    let v_subkey = ValueId(0);
    let v_key = ValueId(1);
    body.val_count = 2;
    body.val_types.insert(v_subkey, Ty::Int);
    body.val_types.insert(v_key, Ty::Int);
    debug.set(v_subkey, ValOrigin::Named(interner.intern("subkey")));
    debug.set(v_key, ValOrigin::Named(interner.intern("key")));

    let mut next_val = 2u32;
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
            // (subkey ^ key) + 1
            let v_xor = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_xor,
                    op: BinOp::Xor,
                    left: v_subkey,
                    right: v_key,
                },
            });
            let v_1 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_1,
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
                    right: v_1,
                },
            });
            v_res
        }
        1 => {
            // (subkey * 3 + key * 7) % 65537
            let v_3 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_3,
                    value: Literal::Int(3),
                },
            });
            let v_mul1 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mul1,
                    op: BinOp::Mul,
                    left: v_subkey,
                    right: v_3,
                },
            });
            let v_7 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_7,
                    value: Literal::Int(7),
                },
            });
            let v_mul2 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mul2,
                    op: BinOp::Mul,
                    left: v_key,
                    right: v_7,
                },
            });
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_mul1,
                    right: v_mul2,
                },
            });
            let v_m = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_m,
                    value: Literal::Int(65537),
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Mod,
                    left: v_add,
                    right: v_m,
                },
            });
            v_res
        }
        2 => {
            // (subkey + key) ^ 0xBEEF
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_subkey,
                    right: v_key,
                },
            });
            let v_beef = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_beef,
                    value: Literal::Int(0xBEEF),
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Xor,
                    left: v_add,
                    right: v_beef,
                },
            });
            v_res
        }
        _ => {
            // ((subkey << 3) ^ key) % 65537
            let v_3 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_3,
                    value: Literal::Int(3),
                },
            });
            let v_shl = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_shl,
                    op: BinOp::Shl,
                    left: v_subkey,
                    right: v_3,
                },
            });
            let v_xor = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_xor,
                    op: BinOp::Xor,
                    left: v_shl,
                    right: v_key,
                },
            });
            let v_m = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_m,
                    value: Literal::Int(65537),
                },
            });
            let v_res = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_res,
                    op: BinOp::Mod,
                    left: v_xor,
                    right: v_m,
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
        param_names: vec![interner.intern("subkey"), interner.intern("key")],
        body,
    }
}

/// Compile-time stage B combine (mirrors closure logic).
pub fn stage_b_combine(variant: u32, subkey: i64, key: i64) -> i64 {
    match variant {
        0 => (subkey ^ key).wrapping_add(1),
        1 => {
            let m = 65537i64;
            ((subkey.wrapping_mul(3).wrapping_add(key.wrapping_mul(7))) % m + m) % m
        }
        2 => (subkey.wrapping_add(key)) ^ 0xBEEF,
        _ => {
            let m = 65537i64;
            (((subkey << 3) ^ key) % m + m) % m
        }
    }
}

// ── Stage C: decrypt_final ───────────────────────────────────────
// (List<Byte>, Int) → String — final decryption using combined_key

fn make_stage_c_closure_body(variant: u32, interner: &Interner) -> ClosureBody {
    let mut body = MirBody::new();
    let mut debug = DebugInfo::new();

    let v_bytes = ValueId(0);
    let v_key = ValueId(1); // combined_key
    body.val_count = 2;
    body.val_types.insert(v_bytes, Ty::bytes());
    body.val_types.insert(v_key, Ty::Int);
    debug.set(v_bytes, ValOrigin::Named(interner.intern("bytes")));
    debug.set(v_key, ValOrigin::Named(interner.intern("key")));

    let mut next_val = 2u32;
    let mut alloc = |ty: Ty| -> ValueId {
        let v = ValueId(next_val);
        next_val += 1;
        body.val_types.insert(v, ty);
        debug.set(v, ValOrigin::Expr);
        v
    };
    let span = Span { start: 0, end: 0 };

    let v_len = alloc(Ty::Int);
    let v_zero = alloc(Ty::Int);
    let v_empty = alloc(Ty::String);
    let v_one = alloc(Ty::Int);
    let v_256 = alloc(Ty::Int);

    let mut insts = Vec::new();

    insts.push(Inst {
        span,
        kind: InstKind::BuiltinCall {
            dst: v_len,
            builtin: BuiltinId::Len,
            args: vec![v_bytes],
        },
    });
    insts.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_zero,
            value: Literal::Int(0),
        },
    });
    insts.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_empty,
            value: Literal::String(String::new()),
        },
    });
    insts.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_one,
            value: Literal::Int(1),
        },
    });
    insts.push(Inst {
        span,
        kind: InstKind::Const {
            dst: v_256,
            value: Literal::Int(256),
        },
    });

    let l_header = Label(0);
    let l_body = Label(1);
    let l_exit = Label(2);
    body.label_count = 3;

    let v_i = alloc(Ty::Int);
    let v_accum = alloc(Ty::String);

    insts.push(Inst {
        span,
        kind: InstKind::Jump {
            label: l_header,
            args: vec![v_zero, v_empty],
        },
    });

    insts.push(Inst {
        span,
        kind: InstKind::BlockLabel {
            label: l_header,
            params: vec![v_i, v_accum],
            merge_of: None,
        },
    });

    let v_done = alloc(Ty::Bool);
    insts.push(Inst {
        span,
        kind: InstKind::BinOp {
            dst: v_done,
            op: BinOp::Gte,
            left: v_i,
            right: v_len,
        },
    });

    let v_result_param = alloc(Ty::String);
    insts.push(Inst {
        span,
        kind: InstKind::JumpIf {
            cond: v_done,
            then_label: l_exit,
            then_args: vec![v_accum],
            else_label: l_body,
            else_args: vec![v_i, v_accum],
        },
    });

    let v_body_i = alloc(Ty::Int);
    let v_body_accum = alloc(Ty::String);
    insts.push(Inst {
        span,
        kind: InstKind::BlockLabel {
            label: l_body,
            params: vec![v_body_i, v_body_accum],
            merge_of: None,
        },
    });

    let v_byte_raw = alloc(Ty::Byte);
    insts.push(Inst {
        span,
        kind: InstKind::ListGet {
            dst: v_byte_raw,
            list: v_bytes,
            index: v_body_i,
        },
    });
    let v_byte = alloc(Ty::Int);
    insts.push(Inst {
        span,
        kind: InstKind::BuiltinCall {
            dst: v_byte,
            builtin: BuiltinId::ToInt,
            args: vec![v_byte_raw],
        },
    });

    // Same decrypt algorithms as before, but now operating on combined_key
    let v_plain = match variant {
        0 => {
            // XOR: plain = enc ^ ((key + i) % 256)
            let v_ki = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_ki,
                    op: BinOp::Add,
                    left: v_key,
                    right: v_body_i,
                },
            });
            let v_mod = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mod,
                    op: BinOp::Mod,
                    left: v_ki,
                    right: v_256,
                },
            });
            let v_p = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_p,
                    op: BinOp::Xor,
                    left: v_byte,
                    right: v_mod,
                },
            });
            v_p
        }
        1 => {
            // Sub: plain = (enc + ((key * (i+1)) % 256)) % 256
            let v_i1 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_i1,
                    op: BinOp::Add,
                    left: v_body_i,
                    right: v_one,
                },
            });
            let v_ki = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_ki,
                    op: BinOp::Mul,
                    left: v_key,
                    right: v_i1,
                },
            });
            let v_mod = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mod,
                    op: BinOp::Mod,
                    left: v_ki,
                    right: v_256,
                },
            });
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_byte,
                    right: v_mod,
                },
            });
            let v_p = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_p,
                    op: BinOp::Mod,
                    left: v_add,
                    right: v_256,
                },
            });
            v_p
        }
        2 => {
            // Rot: plain = enc ^ ((key >> (i % 8)) % 256)
            let v_eight = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_eight,
                    value: Literal::Int(8),
                },
            });
            let v_im = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_im,
                    op: BinOp::Mod,
                    left: v_body_i,
                    right: v_eight,
                },
            });
            let v_shift = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_shift,
                    op: BinOp::Shr,
                    left: v_key,
                    right: v_im,
                },
            });
            let v_mod = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_mod,
                    op: BinOp::Mod,
                    left: v_shift,
                    right: v_256,
                },
            });
            let v_p = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_p,
                    op: BinOp::Xor,
                    left: v_byte,
                    right: v_mod,
                },
            });
            v_p
        }
        _ => {
            // Mul: plain = enc ^ (((key * (i+1)) % 251 + 251) % 251)
            let v_251 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::Const {
                    dst: v_251,
                    value: Literal::Int(251),
                },
            });
            let v_i1 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_i1,
                    op: BinOp::Add,
                    left: v_body_i,
                    right: v_one,
                },
            });
            let v_ki = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_ki,
                    op: BinOp::Mul,
                    left: v_key,
                    right: v_i1,
                },
            });
            let v_m1 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_m1,
                    op: BinOp::Mod,
                    left: v_ki,
                    right: v_251,
                },
            });
            let v_add = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_add,
                    op: BinOp::Add,
                    left: v_m1,
                    right: v_251,
                },
            });
            let v_m2 = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_m2,
                    op: BinOp::Mod,
                    left: v_add,
                    right: v_251,
                },
            });
            let v_p = alloc(Ty::Int);
            insts.push(Inst {
                span,
                kind: InstKind::BinOp {
                    dst: v_p,
                    op: BinOp::Xor,
                    left: v_byte,
                    right: v_m2,
                },
            });
            v_p
        }
    };

    let v_char = alloc(Ty::String);
    insts.push(Inst {
        span,
        kind: InstKind::BuiltinCall {
            dst: v_char,
            builtin: BuiltinId::IntToChar,
            args: vec![v_plain],
        },
    });

    let v_new_accum = alloc(Ty::String);
    insts.push(Inst {
        span,
        kind: InstKind::BinOp {
            dst: v_new_accum,
            op: BinOp::Add,
            left: v_body_accum,
            right: v_char,
        },
    });

    let v_next_i = alloc(Ty::Int);
    insts.push(Inst {
        span,
        kind: InstKind::BinOp {
            dst: v_next_i,
            op: BinOp::Add,
            left: v_body_i,
            right: v_one,
        },
    });

    insts.push(Inst {
        span,
        kind: InstKind::Jump {
            label: l_header,
            args: vec![v_next_i, v_new_accum],
        },
    });

    insts.push(Inst {
        span,
        kind: InstKind::BlockLabel {
            label: l_exit,
            params: vec![v_result_param],
            merge_of: None,
        },
    });
    insts.push(Inst {
        span,
        kind: InstKind::Return(v_result_param),
    });

    body.insts = insts;
    body.val_count = next_val;
    body.debug = debug;

    ClosureBody {
        capture_names: vec![],
        param_names: vec![interner.intern("bytes"), interner.intern("key")],
        body,
    }
}

// ── Encryption (compile-time) ────────────────────────────────────

/// Chunk size range for splitting texts. Each chunk gets its own
/// variant + key, so a static analyzer must solve N independent
/// decrypt algorithms per text.
const CHUNK_MIN: usize = 32;
const CHUNK_MAX: usize = 96;

pub struct EncryptedChunk {
    pub bytes: Vec<u8>,
    pub key: i64,
    // variant_a is encoded in key: (key + 1) % 4 == variant_a
    // variant_b = subkey % 4 (derived from stage A result)
    // variant_c = (key ^ subkey) % 4 (derived from both)
}

/// One text split into encrypted chunks.
pub struct EncryptedText {
    pub chunks: Vec<EncryptedChunk>,
}

/// Encrypt all text pool entries. Each text is split into chunks,
/// each chunk with a random variant + key.
pub fn encrypt_texts(texts: &[String], rng: &mut StdRng) -> Vec<EncryptedText> {
    texts.iter().map(|text| encrypt_single(text, rng)).collect()
}

fn encrypt_single(text: &str, rng: &mut StdRng) -> EncryptedText {
    let plain = text.as_bytes();
    if plain.is_empty() {
        return EncryptedText { chunks: vec![] };
    }

    let mut chunks = Vec::new();
    let mut offset = 0;
    while offset < plain.len() {
        let remaining = plain.len() - offset;
        let chunk_size = if remaining <= CHUNK_MAX {
            remaining
        } else {
            rng.random_range(CHUNK_MIN..=CHUNK_MAX)
        };

        // Pick variant_a randomly, then find key such that (key + 1) % 4 == variant_a
        let variant_a = rng.random_range(0u32..4);
        let key: i64 = loop {
            let k = rng.random_range(1i64..=1_000_000);
            if ((k + 1) as u32) % 4 == variant_a {
                break k;
            }
        };

        // Derive the 3-stage pipeline at compile time
        let subkey = stage_a_transform(variant_a, key);
        let variant_b = (subkey.unsigned_abs() as u32) % 4;
        let combined = stage_b_combine(variant_b, subkey, key);
        let variant_c = ((key ^ subkey).unsigned_abs() as u32) % 4;

        let chunk_plain = &plain[offset..offset + chunk_size];
        let encrypted: Vec<u8> = chunk_plain
            .iter()
            .enumerate()
            .map(|(i, &b)| encrypt_byte_staged(variant_c, b, combined, i))
            .collect();

        chunks.push(EncryptedChunk {
            bytes: encrypted,
            key,
        });
        offset += chunk_size;
    }

    EncryptedText { chunks }
}

/// Encrypt a single byte using stage C's inverse (same as decrypt since XOR/mod are self-inverse).
fn encrypt_byte_staged(variant_c: u32, plain: u8, combined_key: i64, i: usize) -> u8 {
    match variant_c {
        // Stage C variant 0: XOR — plain = enc ^ ((key + i) % 256) => enc = plain ^ ((key + i) % 256)
        0 => {
            let k = ((combined_key + i as i64) % 256).unsigned_abs() as u8;
            plain ^ k
        }
        // Stage C variant 1: Sub — plain = (enc + ((key * (i+1)) % 256)) % 256
        1 => {
            let k = (combined_key.wrapping_mul((i + 1) as i64) % 256 + 256) % 256;
            ((plain as i64 - k) % 256 + 256) as u8
        }
        // Stage C variant 2: Rot — plain = enc ^ ((key >> (i % 8)) % 256)
        2 => {
            let k = ((combined_key >> (i % 8)) % 256).unsigned_abs() as u8;
            plain ^ k
        }
        // Stage C variant 3: Mul — plain = enc ^ (((key * (i+1)) % 251 + 251) % 251)
        _ => {
            let k = ((combined_key.wrapping_mul((i + 1) as i64) % 251 + 251) % 251).unsigned_abs()
                as u8;
            plain ^ k
        }
    }
}

/// Decrypt a single byte (for testing). Mirrors stage C closure logic.
#[cfg(test)]
fn decrypt_byte_staged(variant_c: u32, enc: u8, combined_key: i64, i: usize) -> u8 {
    match variant_c {
        0 => {
            let k = ((combined_key + i as i64) % 256).unsigned_abs() as u8;
            enc ^ k
        }
        1 => {
            let k = (combined_key.wrapping_mul((i + 1) as i64) % 256 + 256) % 256;
            ((enc as i64 + k) % 256) as u8
        }
        2 => {
            let k = ((combined_key >> (i % 8)) % 256).unsigned_abs() as u8;
            enc ^ k
        }
        _ => {
            let k = ((combined_key.wrapping_mul((i + 1) as i64) % 251 + 251) % 251).unsigned_abs()
                as u8;
            enc ^ k
        }
    }
}

// ── Factory closures (rotation-based 2-level dispatch) ───────────

/// Labels for 4 factory closures (one per rotation).
pub struct FactoryTable {
    pub labels: [Label; 4],
}

/// Register 4 factory closures into the module. Each factory captures
/// 4 inner closures and returns them in a rotated order.
pub fn register_factory_closures(
    module: &mut MirModule,
    _inner_labels: &[Label; 4],
    inner_fn_ty: &Ty,
    name_prefix: &str,
    interner: &Interner,
) -> FactoryTable {
    let base_label = module.closures.keys().map(|l| l.0).max().unwrap_or(0) + 1;
    let base_label = base_label.max(module.main.label_count);
    for closure in module.closures.values() {
        let _ = base_label.max(closure.body.label_count);
    }

    let mut labels = [Label(0); 4];
    for rotation in 0..4u32 {
        let label = Label(base_label + rotation);
        labels[rotation as usize] = label;
        let body = make_factory_closure_body(rotation, inner_fn_ty, interner);
        module.closures.insert(label, Arc::new(body));
        let _ = name_prefix; // for future debug info
    }

    let max_label = base_label + 4;
    if module.main.label_count < max_label {
        module.main.label_count = max_label;
    }

    FactoryTable { labels }
}

/// Build a factory closure body that captures 4 inner closures and returns
/// them in rotated order.
///
/// Captures: [fn0 (Val 0), fn1 (Val 1), fn2 (Val 2), fn3 (Val 3)]
/// Params: [seed: Int (Val 4)]
/// Returns: List<inner_fn_ty>
fn make_factory_closure_body(rotation: u32, inner_fn_ty: &Ty, interner: &Interner) -> ClosureBody {
    let mut body = MirBody::new();
    let mut debug = DebugInfo::new();

    // 4 captures + 1 param = 5 vals
    for i in 0..4u32 {
        let v = ValueId(i);
        body.val_types.insert(v, inner_fn_ty.clone());
        debug.set(v, ValOrigin::Named(interner.intern(&format!("fn{i}"))));
    }
    let v_seed = ValueId(4);
    body.val_types.insert(v_seed, Ty::Int);
    debug.set(v_seed, ValOrigin::Named(interner.intern("seed")));
    body.val_count = 5;

    let mut next_val = 5u32;
    let mut alloc = |ty: Ty| -> ValueId {
        let v = ValueId(next_val);
        next_val += 1;
        body.val_types.insert(v, ty);
        debug.set(v, ValOrigin::Expr);
        v
    };
    let span = Span { start: 0, end: 0 };

    // Build rotated list: [fn_{(0+r)%4}, fn_{(1+r)%4}, fn_{(2+r)%4}, fn_{(3+r)%4}]
    let elements: Vec<ValueId> = (0..4).map(|i| ValueId((i + rotation) % 4)).collect();

    let list_ty = Ty::List(Box::new(inner_fn_ty.clone()));
    let v_list = alloc(list_ty);

    let mut insts = Vec::new();
    insts.push(Inst {
        span,
        kind: InstKind::MakeList {
            dst: v_list,
            elements,
        },
    });
    insts.push(Inst {
        span,
        kind: InstKind::Return(v_list),
    });

    body.insts = insts;
    body.val_count = next_val;
    body.label_count = 0;
    body.debug = debug;

    ClosureBody {
        capture_names: vec![
            interner.intern("fn0"),
            interner.intern("fn1"),
            interner.intern("fn2"),
            interner.intern("fn3"),
        ],
        param_names: vec![interner.intern("seed")],
        body,
    }
}

/// Get all factory labels for filtering.
pub fn all_factory_labels(ft: &FactoryTable) -> Vec<Label> {
    ft.labels.to_vec()
}

// ── Emit helpers (MIR generation) ────────────────────────────────

/// Emit the 2-level dispatch meta table for a single stage.
///
/// Creates: 4 inner MakeClosure + 4 factory MakeClosure (each capturing all 4 inner) + MakeList of factories.
/// Returns the meta table ValueId (List<factory_fn>).
pub fn emit_factory_dispatch_setup(
    ctx: &mut PassState,
    span: Span,
    inner_labels: &[Label; 4],
    factory_table: &FactoryTable,
    inner_fn_ty: &Ty,
) -> ValueId {
    // Make 4 inner closures
    let mut inner_vals = Vec::with_capacity(4);
    for &label in inner_labels {
        let v = ctx.alloc_val(inner_fn_ty.clone());
        ctx.emit(
            span,
            InstKind::MakeClosure {
                dst: v,
                body: label,
                captures: vec![],
            },
        );
        inner_vals.push(v);
    }

    // Factory fn type: (Int) → List<inner_fn_ty>
    let factory_fn_ty = Ty::Fn {
        params: vec![Ty::Int],
        ret: Box::new(Ty::List(Box::new(inner_fn_ty.clone()))),
        kind: FnKind::Lambda, effect: Effect::Pure,
        captures: vec![],
    };

    // Make 4 factory closures, each capturing all 4 inner closures
    let mut factory_vals = Vec::with_capacity(4);
    for &label in &factory_table.labels {
        let v = ctx.alloc_val(factory_fn_ty.clone());
        ctx.emit(
            span,
            InstKind::MakeClosure {
                dst: v,
                body: label,
                captures: inner_vals.clone(),
            },
        );
        factory_vals.push(v);
    }

    // Meta table: List of factory closures
    let meta_ty = Ty::List(Box::new(factory_fn_ty));
    let v_meta = ctx.alloc_val(meta_ty);
    ctx.emit(
        span,
        InstKind::MakeList {
            dst: v_meta,
            elements: factory_vals,
        },
    );

    v_meta
}

/// Emit 2-level dispatch: factory_idx → factory call → inner_idx → actual call.
///
/// Returns the result ValueId.
fn emit_two_level_dispatch(
    ctx: &mut PassState,
    span: Span,
    meta_table: ValueId,
    factory_idx: ValueId,
    seed: ValueId,
    inner_idx: ValueId,
    inner_fn_ty: &Ty,
    args: Vec<ValueId>,
    result_ty: &Ty,
) -> ValueId {
    // Factory fn type
    let factory_fn_ty = Ty::Fn {
        params: vec![Ty::Int],
        ret: Box::new(Ty::List(Box::new(inner_fn_ty.clone()))),
        kind: FnKind::Lambda, effect: Effect::Pure,
        captures: vec![],
    };

    // v_factory = ListGet(meta_table, factory_idx)
    let v_factory = ctx.alloc_val(factory_fn_ty);
    ctx.emit(
        span,
        InstKind::ListGet {
            dst: v_factory,
            list: meta_table,
            index: factory_idx,
        },
    );

    // v_inner_table = CallClosure(v_factory, [seed])
    let inner_table_ty = Ty::List(Box::new(inner_fn_ty.clone()));
    let v_inner_table = ctx.alloc_val(inner_table_ty);
    ctx.emit(
        span,
        InstKind::ClosureCall {
            dst: v_inner_table,
            closure: v_factory,
            args: vec![seed],
        },
    );

    // v_fn = ListGet(v_inner_table, inner_idx)
    let v_fn = ctx.alloc_val(inner_fn_ty.clone());
    ctx.emit(
        span,
        InstKind::ListGet {
            dst: v_fn,
            list: v_inner_table,
            index: inner_idx,
        },
    );

    // v_result = CallClosure(v_fn, args)
    let v_result = ctx.alloc_val(result_ty.clone());
    ctx.emit(
        span,
        InstKind::ClosureCall {
            dst: v_result,
            closure: v_fn,
            args,
        },
    );

    v_result
}

/// Emit the full 3-stage multi-stage decrypt call for one chunk.
///
/// Stage A: key → subkey (via factory dispatch on stage_a meta table)
/// Stage B: (subkey, key) → combined (via factory dispatch on stage_b meta table)
/// Stage C: (bytes, combined) → String (via factory dispatch on stage_c meta table)
///
/// Entanglement: uses `__entangle` VarLoad to adjust key for stage A dispatch index.
fn emit_multistage_decrypt_call(
    ctx: &mut PassState,
    span: Span,
    bytes: &[u8],
    key: i64,
    meta_a: ValueId,
    meta_b: ValueId,
    meta_c: ValueId,
    v_four: ValueId,
    use_entangle: bool,
) -> ValueId {
    let v_bytes = ctx.alloc_val(Ty::bytes());
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_bytes,
            value: Literal::List(bytes.iter().map(|b| Literal::Byte(*b)).collect()),
        },
    );

    let v_key = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_key,
            value: Literal::Int(key),
        },
    );

    // Seed for factory calls (constant, unused by current factories but required param)
    let v_seed = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_seed,
            value: Literal::Int(0),
        },
    );

    // ── Stage A dispatch index ──
    // With entanglement: idx_a = (key + __entangle) % 4  (entangle=1, so key+1)
    // Without: idx_a = (key + 1) % 4
    let v_key_adj = if use_entangle {
        let v_entangle = ctx.alloc_val(Ty::Int);
        ctx.emit(
            span,
            InstKind::VarLoad {
                dst: v_entangle,
                name: ctx.interner.intern("__entangle"),
            },
        );
        let v_adj = ctx.alloc_val(Ty::Int);
        ctx.emit(
            span,
            InstKind::BinOp {
                dst: v_adj,
                op: BinOp::Add,
                left: v_key,
                right: v_entangle,
            },
        );
        v_adj
    } else {
        let v_one = ctx.alloc_val(Ty::Int);
        ctx.emit(
            span,
            InstKind::Const {
                dst: v_one,
                value: Literal::Int(1),
            },
        );
        let v_adj = ctx.alloc_val(Ty::Int);
        ctx.emit(
            span,
            InstKind::BinOp {
                dst: v_adj,
                op: BinOp::Add,
                left: v_key,
                right: v_one,
            },
        );
        v_adj
    };

    let v_idx_a = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_idx_a,
            op: BinOp::Mod,
            left: v_key_adj,
            right: v_four,
        },
    );

    // Precompute variant_a, factory_idx_a, and inner_idx_a at compile time
    let variant_a = ((key + 1) as u32) % 4;
    // factory rotation == factory_idx == variant_a, so inner_idx = (variant_a - variant_a + 4) % 4 = 0
    let inner_idx_a = 0u32;

    // We need the inner_idx as a constant for rotation compensation
    let v_inner_a = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_inner_a,
            value: Literal::Int(inner_idx_a as i64),
        },
    );

    let stage_a_fn_ty = Ty::Fn {
        params: vec![Ty::Int],
        ret: Box::new(Ty::Int),
        kind: FnKind::Lambda, effect: Effect::Pure,
        captures: vec![],
    };
    let v_subkey = emit_two_level_dispatch(
        ctx,
        span,
        meta_a,
        v_idx_a,
        v_seed,
        v_inner_a,
        &stage_a_fn_ty,
        vec![v_key],
        &Ty::Int,
    );

    // ── Stage B dispatch index ──
    // idx_b = subkey % 4
    let v_idx_b = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_idx_b,
            op: BinOp::Mod,
            left: v_subkey,
            right: v_four,
        },
    );

    // Compile-time: compute inner_idx_b from rotation compensation
    let subkey = stage_a_transform(variant_a, key);
    let variant_b = (subkey.unsigned_abs() as u32) % 4;
    let factory_idx_b = variant_b; // subkey % 4 == variant_b
    let inner_idx_b = (variant_b as i32 - factory_idx_b as i32).rem_euclid(4) as u32; // always 0

    let v_inner_b = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_inner_b,
            value: Literal::Int(inner_idx_b as i64),
        },
    );

    let stage_b_fn_ty = Ty::Fn {
        params: vec![Ty::Int, Ty::Int],
        ret: Box::new(Ty::Int),
        kind: FnKind::Lambda, effect: Effect::Pure,
        captures: vec![],
    };
    let v_combined = emit_two_level_dispatch(
        ctx,
        span,
        meta_b,
        v_idx_b,
        v_seed,
        v_inner_b,
        &stage_b_fn_ty,
        vec![v_subkey, v_key],
        &Ty::Int,
    );

    // ── Stage C dispatch index ──
    // idx_c = (key ^ subkey) % 4  — but subkey is runtime, so we use BinOp
    let v_xor_ks = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_xor_ks,
            op: BinOp::Xor,
            left: v_key,
            right: v_subkey,
        },
    );
    let v_idx_c = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_idx_c,
            op: BinOp::Mod,
            left: v_xor_ks,
            right: v_four,
        },
    );

    // Compile-time rotation compensation for stage C
    let variant_c = ((key ^ subkey).unsigned_abs() as u32) % 4;
    let factory_idx_c = variant_c;
    let inner_idx_c = (variant_c as i32 - factory_idx_c as i32).rem_euclid(4) as u32; // always 0

    let v_inner_c = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_inner_c,
            value: Literal::Int(inner_idx_c as i64),
        },
    );

    let stage_c_fn_ty = Ty::Fn {
        params: vec![Ty::bytes(), Ty::Int],
        ret: Box::new(Ty::String),
        kind: FnKind::Lambda, effect: Effect::Pure,
        captures: vec![],
    };
    emit_two_level_dispatch(
        ctx,
        span,
        meta_c,
        v_idx_c,
        v_seed,
        v_inner_c,
        &stage_c_fn_ty,
        vec![v_bytes, v_combined],
        &Ty::String,
    )
}

/// Emit encrypted text decryption using 3-stage multi-stage pipeline.
pub fn emit_encrypted_text(
    ctx: &mut PassState,
    span: Span,
    enc: &EncryptedText,
    meta_a: ValueId,
    meta_b: ValueId,
    meta_c: ValueId,
    v_four: ValueId,
    use_entangle: bool,
) {
    if enc.chunks.is_empty() {
        return;
    }

    let mut v_accum: Option<ValueId> = None;
    for chunk in &enc.chunks {
        let v_part = emit_multistage_decrypt_call(
            ctx,
            span,
            &chunk.bytes,
            chunk.key,
            meta_a,
            meta_b,
            meta_c,
            v_four,
            use_entangle,
        );
        v_accum = Some(match v_accum {
            None => v_part,
            Some(prev) => {
                let v_concat = ctx.alloc_val(Ty::String);
                ctx.emit(
                    span,
                    InstKind::BinOp {
                        dst: v_concat,
                        op: BinOp::Add,
                        left: prev,
                        right: v_part,
                    },
                );
                v_concat
            }
        });
    }

    ctx.emit(span, InstKind::Yield(v_accum.unwrap()));
}

/// Emit hash-key-based text decryption with 3-stage multi-stage pipeline.
///
/// Uses `__obf_key` variable as the key.
pub fn emit_hashed_text(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    text: &str,
    compile_key: i64,
    meta_a: ValueId,
    meta_b: ValueId,
    meta_c: ValueId,
    v_four: ValueId,
    use_entangle: bool,
) {
    if text.is_empty() {
        return;
    }

    // For hashed text, key is loaded from __obf_key at runtime.
    // We know at compile time what compile_key is, so we can precompute the 3-stage pipeline.
    let variant_a = ((compile_key + 1) as u32) % 4;
    let subkey = stage_a_transform(variant_a, compile_key);
    let variant_b = (subkey.unsigned_abs() as u32) % 4;
    let combined = stage_b_combine(variant_b, subkey, compile_key);
    let variant_c = ((compile_key ^ subkey).unsigned_abs() as u32) % 4;

    let plain = text.as_bytes();
    let mut v_accum: Option<ValueId> = None;
    let mut offset = 0;

    while offset < plain.len() {
        let remaining = plain.len() - offset;
        let chunk_size = if remaining <= CHUNK_MAX {
            remaining
        } else {
            rng.random_range(CHUNK_MIN..=CHUNK_MAX)
        };

        let chunk_plain = &plain[offset..offset + chunk_size];
        let encrypted: Vec<u8> = chunk_plain
            .iter()
            .enumerate()
            .map(|(i, &b)| encrypt_byte_staged(variant_c, b, combined, i))
            .collect();

        // Use the multi-stage pipeline with __obf_key
        let v_part = emit_multistage_decrypt_call(
            ctx,
            span,
            &encrypted,
            compile_key,
            meta_a,
            meta_b,
            meta_c,
            v_four,
            use_entangle,
        );

        v_accum = Some(match v_accum {
            None => v_part,
            Some(prev) => {
                let v_concat = ctx.alloc_val(Ty::String);
                ctx.emit(
                    span,
                    InstKind::BinOp {
                        dst: v_concat,
                        op: BinOp::Add,
                        left: prev,
                        right: v_part,
                    },
                );
                v_concat
            }
        });

        offset += chunk_size;
    }

    ctx.emit(span, InstKind::Yield(v_accum.unwrap()));
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn stage_a_transform_all_variants() {
        for variant in 0..4u32 {
            let key = 12345i64;
            let result = stage_a_transform(variant, key);
            assert!(
                result >= 0 && result < 65537,
                "stage_a variant {variant} out of range: {result}"
            );
        }
    }

    #[test]
    fn stage_b_combine_all_variants() {
        for variant in 0..4u32 {
            let subkey = 42i64;
            let key = 12345i64;
            let _result = stage_b_combine(variant, subkey, key);
            // Just ensure no panics
        }
    }

    #[test]
    fn encrypt_decrypt_roundtrip_staged_all_variants() {
        for variant_c in 0..4u32 {
            let text = "hello world! 🌍";
            let combined_key: i64 = 12345;
            let plain = text.as_bytes();
            let encrypted: Vec<u8> = plain
                .iter()
                .enumerate()
                .map(|(i, &b)| encrypt_byte_staged(variant_c, b, combined_key, i))
                .collect();

            let decrypted: Vec<u8> = encrypted
                .iter()
                .enumerate()
                .map(|(i, &enc)| decrypt_byte_staged(variant_c, enc, combined_key, i))
                .collect();

            assert_eq!(
                plain,
                &decrypted[..],
                "stage_c variant {variant_c} roundtrip failed"
            );
        }
    }

    #[test]
    fn full_3stage_roundtrip() {
        // Simulate the full 3-stage pipeline
        let key = 777i64;
        let variant_a = ((key + 1) as u32) % 4;
        let subkey = stage_a_transform(variant_a, key);
        let variant_b = (subkey.unsigned_abs() as u32) % 4;
        let combined = stage_b_combine(variant_b, subkey, key);
        let variant_c = ((key ^ subkey).unsigned_abs() as u32) % 4;

        let text = "test 3-stage decrypt";
        let plain = text.as_bytes();
        let encrypted: Vec<u8> = plain
            .iter()
            .enumerate()
            .map(|(i, &b)| encrypt_byte_staged(variant_c, b, combined, i))
            .collect();

        let decrypted: Vec<u8> = encrypted
            .iter()
            .enumerate()
            .map(|(i, &enc)| decrypt_byte_staged(variant_c, enc, combined, i))
            .collect();

        assert_eq!(plain, &decrypted[..]);
    }

    #[test]
    fn encrypt_texts_produces_chunks() {
        let mut rng = StdRng::seed_from_u64(42);
        let texts = vec!["hello".to_string(), "world".to_string(), "".to_string()];
        let results = encrypt_texts(&texts, &mut rng);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].chunks.len(), 1);
        assert_eq!(results[0].chunks[0].bytes.len(), 5);
        assert_eq!(results[1].chunks.len(), 1);
        assert_eq!(results[1].chunks[0].bytes.len(), 5);
        assert!(results[2].chunks.is_empty());
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        let mut rng = StdRng::seed_from_u64(0);
        let results = encrypt_texts(&["".to_string()], &mut rng);
        assert!(results[0].chunks.is_empty());
    }

    #[test]
    fn long_text_produces_multiple_chunks() {
        let mut rng = StdRng::seed_from_u64(42);
        let text = "a".repeat(200);
        let results = encrypt_texts(&[text.clone()], &mut rng);
        assert!(
            results[0].chunks.len() >= 2,
            "200-byte text should produce multiple chunks"
        );
        let total: usize = results[0].chunks.iter().map(|c| c.bytes.len()).sum();
        assert_eq!(total, 200);
    }
}
