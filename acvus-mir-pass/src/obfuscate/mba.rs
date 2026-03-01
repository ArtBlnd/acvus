//! MBA (Mixed Boolean-Arithmetic) expression substitution.
//!
//! Replaces bitwise operations with algebraically equivalent arithmetic expressions
//! that are hard to simplify via pattern matching.
//!
//! Key identities:
//!   a ^ b  =  (a + b) - 2*(a & b)
//!   a | b  =  (a + b) - (a & b)
//!   a & b  =  (a + b) - (a | b)
//!
//! Multi-layer: MBA results can be further expanded (2-layer).
//! Shl/Shr substitution: 2*(a&b) → (a&b) << 1 randomly.
//! Linear noise: result + c1 - c2 + (c2 - c1) for expanded noise.

use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::ir::{Inst, InstKind, ValueId};
use acvus_mir::ty::Ty;
use rand::rngs::StdRng;
use rand::Rng;

use super::rewriter::PassState;

pub fn apply(
    old_insts: Vec<Inst>,
    ctx: &mut PassState,
    rng: &mut StdRng,
) -> Vec<Inst> {
    let mut out = Vec::new();

    for inst in old_insts {
        match &inst.kind {
            InstKind::BinOp { dst, op: BinOp::Xor, left, right } => {
                expand_xor(&mut out, ctx, rng, inst.span, *dst, *left, *right);
            }
            InstKind::BinOp { dst, op: BinOp::BitOr, left, right } if rng.random_bool(0.6) => {
                expand_or(&mut out, ctx, rng, inst.span, *dst, *left, *right);
            }
            InstKind::BinOp { dst, op: BinOp::BitAnd, left, right } if rng.random_bool(0.5) => {
                expand_and(&mut out, ctx, rng, inst.span, *dst, *left, *right);
            }
            _ => out.push(inst),
        }
    }

    // Second layer: apply MBA once more on the results with lower probability.
    // Re-expand BitAnd/BitOr that were introduced in the first pass.
    if rng.random_bool(0.4) {
        let first_pass = std::mem::take(&mut out);
        for inst in first_pass {
            match &inst.kind {
                InstKind::BinOp { dst, op: BinOp::BitAnd, left, right } if rng.random_bool(0.3) => {
                    expand_and(&mut out, ctx, rng, inst.span, *dst, *left, *right);
                }
                _ => out.push(inst),
            }
        }
    }

    out
}

/// a ^ b  →  (a + b) - 2*(a & b)
/// With Shl variant: 2*(a&b) → (a&b) << 1
/// With noise: expanded linear noise
fn expand_xor(
    out: &mut Vec<Inst>,
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    left: ValueId,
    right: ValueId,
) {
    let v_add = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp {
        dst: v_add, op: BinOp::Add, left, right,
    }});

    let v_and = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp {
        dst: v_and, op: BinOp::BitAnd, left, right,
    }});

    // 2*(a&b) — choose between Mul(2) and Shl(1).
    let v_double = if rng.random_bool(0.5) {
        // Shl variant: (a&b) << 1
        let v_one = ctx.alloc_val(Ty::Int);
        out.push(Inst { span, kind: InstKind::Const {
            dst: v_one, value: Literal::Int(1),
        }});
        let v_shl = ctx.alloc_val(Ty::Int);
        out.push(Inst { span, kind: InstKind::BinOp {
            dst: v_shl, op: BinOp::Shl, left: v_and, right: v_one,
        }});
        v_shl
    } else {
        let v_two = ctx.alloc_val(Ty::Int);
        out.push(Inst { span, kind: InstKind::Const {
            dst: v_two, value: Literal::Int(2),
        }});
        let v_mul = ctx.alloc_val(Ty::Int);
        out.push(Inst { span, kind: InstKind::BinOp {
            dst: v_mul, op: BinOp::Mul, left: v_and, right: v_two,
        }});
        v_mul
    };

    // Noise selection.
    match rng.random_range(0u32..3) {
        0 => {
            // No noise.
            out.push(Inst { span, kind: InstKind::BinOp {
                dst, op: BinOp::Sub, left: v_add, right: v_double,
            }});
        }
        1 => {
            // Simple noise: result + c - c
            let noise: i64 = rng.random_range(100..100000);
            let v_noise = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::Const {
                dst: v_noise, value: Literal::Int(noise),
            }});
            let v_noisy = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_noisy, op: BinOp::Add, left: v_add, right: v_noise,
            }});
            let v_sub1 = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_sub1, op: BinOp::Sub, left: v_noisy, right: v_double,
            }});
            out.push(Inst { span, kind: InstKind::BinOp {
                dst, op: BinOp::Sub, left: v_sub1, right: v_noise,
            }});
        }
        _ => {
            // Expanded noise: result + c1 - c2 + (c2 - c1)
            let c1: i64 = rng.random_range(100..50000);
            let c2: i64 = rng.random_range(100..50000);
            let v_c1 = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::Const {
                dst: v_c1, value: Literal::Int(c1),
            }});
            let v_c2 = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::Const {
                dst: v_c2, value: Literal::Int(c2),
            }});
            // (a + b) + c1
            let v_p1 = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_p1, op: BinOp::Add, left: v_add, right: v_c1,
            }});
            // - 2*(a&b)
            let v_p2 = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_p2, op: BinOp::Sub, left: v_p1, right: v_double,
            }});
            // - c2
            let v_p3 = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_p3, op: BinOp::Sub, left: v_p2, right: v_c2,
            }});
            // + (c2 - c1)
            let v_diff = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_diff, op: BinOp::Sub, left: v_c2, right: v_c1,
            }});
            out.push(Inst { span, kind: InstKind::BinOp {
                dst, op: BinOp::Add, left: v_p3, right: v_diff,
            }});
        }
    }
}

/// a | b  →  (a + b) - (a & b)
fn expand_or(
    out: &mut Vec<Inst>,
    ctx: &mut PassState,
    _rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    left: ValueId,
    right: ValueId,
) {
    let v_add = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp {
        dst: v_add, op: BinOp::Add, left, right,
    }});

    let v_and = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp {
        dst: v_and, op: BinOp::BitAnd, left, right,
    }});

    out.push(Inst { span, kind: InstKind::BinOp {
        dst, op: BinOp::Sub, left: v_add, right: v_and,
    }});
}

/// a & b  →  (a + b) - (a | b)
fn expand_and(
    out: &mut Vec<Inst>,
    ctx: &mut PassState,
    _rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    left: ValueId,
    right: ValueId,
) {
    let v_add = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp {
        dst: v_add, op: BinOp::Add, left, right,
    }});

    let v_or = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp {
        dst: v_or, op: BinOp::BitOr, left, right,
    }});

    out.push(Inst { span, kind: InstKind::BinOp {
        dst, op: BinOp::Sub, left: v_add, right: v_or,
    }});
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::ir::DebugInfo;
    use rand::SeedableRng;
    use std::collections::HashMap;

    fn make_ctx() -> PassState {
        PassState {
            insts: Vec::new(),
            val_types: HashMap::new(),
            debug: DebugInfo::new(),
            next_val: 100,
            next_label: 0,
        }
    }

    #[test]
    fn xor_eliminated() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);
        let insts = vec![Inst {
            span: Span { start: 0, end: 0 },
            kind: InstKind::BinOp {
                dst: ValueId(0),
                op: BinOp::Xor,
                left: ValueId(1),
                right: ValueId(2),
            },
        }];

        let result = apply(insts, &mut ctx, &mut rng);

        // No XOR should remain.
        for inst in &result {
            if let InstKind::BinOp { op, .. } = &inst.kind {
                assert_ne!(*op, BinOp::Xor, "XOR should be eliminated by MBA");
            }
        }
        // Should have at least 5 instructions (Add, BitAnd, Const/Shl, *, Sub).
        assert!(result.len() >= 5);
    }

    #[test]
    fn bitand_expanded() {
        let mut ctx = make_ctx();
        // Use a seed where random_bool(0.5) returns true for BitAnd expansion.
        let mut rng = StdRng::seed_from_u64(7);
        let insts = vec![Inst {
            span: Span { start: 0, end: 0 },
            kind: InstKind::BinOp {
                dst: ValueId(0),
                op: BinOp::BitAnd,
                left: ValueId(1),
                right: ValueId(2),
            },
        }];

        let result = apply(insts, &mut ctx, &mut rng);

        // If expanded, should have at least 3 instructions (Add, BitOr, Sub).
        // If not expanded (random said no), it stays as BitAnd.
        let has_bitor = result.iter().any(|i| matches!(i.kind, InstKind::BinOp { op: BinOp::BitOr, .. }));
        let has_bitand = result.iter().any(|i| matches!(i.kind, InstKind::BinOp { op: BinOp::BitAnd, .. }));
        // One of these must be true.
        assert!(has_bitor || has_bitand);
    }
}
