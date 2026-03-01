//! Opaque predicates: insert conditional branches where the condition always
//! evaluates to one specific value, but is hard to prove statically.
//!
//! Techniques:
//!   1. x*(x+1) % 2 == 0  — product of consecutive integers is always even.
//!   2. x*x >= 0 — square is non-negative (for practical ranges).
//!   3. (2x + 2y) % 2 == 0 — sum of even numbers is always even.
//!   4. (a^2 + b^2 + 1) > 0 — always true (multi-variable).
//!   5. (x | ~x) == -1 — always true (bit trick).
//!
//! Each opaque predicate guards a fake branch to a dead block that contains
//! garbage instructions.

use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::ir::{Inst, InstKind, ValueId};
use acvus_mir::ty::Ty;
use rand::rngs::StdRng;
use rand::Rng;

use super::rewriter::PassState;

/// Number of opaque predicates to insert per body.
const PREDICATES_PER_BODY: usize = 3;

pub fn insert(
    insts: Vec<Inst>,
    ctx: &mut PassState,
    rng: &mut StdRng,
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
            let (predicate_insts, dead_block) = make_opaque_branch(ctx, rng, span);

            let mut new_result = Vec::with_capacity(result.len() + predicate_insts.len() + dead_block.len() + 4);

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
                kind: InstKind::BlockLabel { label: dead_label, params: vec![] },
            });
            new_result.extend(dead_block);
            new_result.push(Inst {
                span,
                kind: InstKind::Jump { label: continue_label, args: vec![] },
            });

            new_result.push(Inst {
                span,
                kind: InstKind::BlockLabel { label: continue_label, params: vec![] },
            });

            new_result.extend(result[pos..].iter().cloned());

            result = new_result;
        }
    }

    result
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
                InstKind::Jump { .. }
                    | InstKind::JumpIf { .. }
                    | InstKind::Return(_)
            )
        })
        .collect();

    if valid.is_empty() {
        None
    } else {
        Some(valid[rng.random_range(0..valid.len())])
    }
}

/// Generate an opaque predicate that always evaluates to true.
fn make_opaque_branch(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> (Vec<Inst>, Vec<Inst>) {
    let variant = rng.random_range(0u32..7);
    let predicate_insts = match variant {
        0 => opaque_consecutive_product(ctx, rng, span),
        1 => opaque_square_nonneg(ctx, rng, span),
        2 => opaque_even_sum(ctx, rng, span),
        3 => opaque_sum_of_squares_positive(ctx, rng, span),
        4 => opaque_xor_identity(ctx, rng, span),
        5 => opaque_cubic_mod(ctx, rng, span),
        _ => opaque_square_mod4(ctx, rng, span),
    };

    let dead = make_dead_block(ctx, rng, span);

    (predicate_insts, dead)
}

/// x*(x+1) % 2 == 0 — always true.
fn opaque_consecutive_product(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let x_val: i64 = rng.random_range(1..100000);

    let v_x = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_x, value: Literal::Int(x_val) } });

    let v_one = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_one, value: Literal::Int(1) } });

    let v_x1 = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_x1, op: BinOp::Add, left: v_x, right: v_one } });

    let v_prod = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_prod, op: BinOp::Mul, left: v_x, right: v_x1 } });

    let v_two = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_two, value: Literal::Int(2) } });

    let v_mod = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_mod, op: BinOp::Mod, left: v_prod, right: v_two } });

    let v_zero = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_zero, value: Literal::Int(0) } });

    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_cond, op: BinOp::Eq, left: v_mod, right: v_zero } });

    out
}

/// x*x >= 0 — always true for any integer (ignoring overflow for practical values).
fn opaque_square_nonneg(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let x_val: i64 = rng.random_range(1..1000);

    let v_x = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_x, value: Literal::Int(x_val) } });

    let v_sq = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_sq, op: BinOp::Mul, left: v_x, right: v_x } });

    let v_zero = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_zero, value: Literal::Int(0) } });

    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_cond, op: BinOp::Gte, left: v_sq, right: v_zero } });

    out
}

/// (2*x + 2*y) % 2 == 0 — sum of even numbers is always even.
fn opaque_even_sum(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let x: i64 = rng.random_range(1..100000);
    let y: i64 = rng.random_range(1..100000);

    let v_x = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_x, value: Literal::Int(x) } });

    let v_y = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_y, value: Literal::Int(y) } });

    let v_two = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_two, value: Literal::Int(2) } });

    let v_2x = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_2x, op: BinOp::Mul, left: v_x, right: v_two } });

    let v_2y = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_2y, op: BinOp::Mul, left: v_y, right: v_two } });

    let v_sum = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_sum, op: BinOp::Add, left: v_2x, right: v_2y } });

    let v_mod_divisor = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_mod_divisor, value: Literal::Int(2) } });

    let v_mod = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_mod, op: BinOp::Mod, left: v_sum, right: v_mod_divisor } });

    let v_zero = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_zero, value: Literal::Int(0) } });

    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_cond, op: BinOp::Eq, left: v_mod, right: v_zero } });

    out
}

/// (a^2 + b^2 + 1) > 0 — always true (multi-variable).
fn opaque_sum_of_squares_positive(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let a: i64 = rng.random_range(1..1000);
    let b: i64 = rng.random_range(1..1000);

    let v_a = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_a, value: Literal::Int(a) } });

    let v_b = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_b, value: Literal::Int(b) } });

    let v_a2 = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_a2, op: BinOp::Mul, left: v_a, right: v_a } });

    let v_b2 = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_b2, op: BinOp::Mul, left: v_b, right: v_b } });

    let v_sum = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_sum, op: BinOp::Add, left: v_a2, right: v_b2 } });

    let v_one = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_one, value: Literal::Int(1) } });

    let v_total = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_total, op: BinOp::Add, left: v_sum, right: v_one } });

    let v_zero = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_zero, value: Literal::Int(0) } });

    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_cond, op: BinOp::Gt, left: v_total, right: v_zero } });

    out
}

/// (x ^ x) == 0 — always true (xor identity).
fn opaque_xor_identity(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let x: i64 = rng.random_range(1..100000);

    let v_x = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_x, value: Literal::Int(x) } });

    let v_xor = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_xor, op: BinOp::Xor, left: v_x, right: v_x } });

    let v_zero = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_zero, value: Literal::Int(0) } });

    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_cond, op: BinOp::Eq, left: v_xor, right: v_zero } });

    out
}

/// 3*n*(n+1) % 6 == 0 — always true (product of 3 consecutive factors includes 2 and 3).
fn opaque_cubic_mod(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let n: i64 = rng.random_range(1..10000);

    let v_n = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_n, value: Literal::Int(n) } });

    let v_one = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_one, value: Literal::Int(1) } });

    let v_n1 = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_n1, op: BinOp::Add, left: v_n, right: v_one } });

    let v_three = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_three, value: Literal::Int(3) } });

    let v_3n = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_3n, op: BinOp::Mul, left: v_three, right: v_n } });

    let v_prod = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_prod, op: BinOp::Mul, left: v_3n, right: v_n1 } });

    let v_six = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_six, value: Literal::Int(6) } });

    let v_mod = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_mod, op: BinOp::Mod, left: v_prod, right: v_six } });

    let v_zero = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_zero, value: Literal::Int(0) } });

    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_cond, op: BinOp::Eq, left: v_mod, right: v_zero } });

    out
}

/// n² % 4 != 3 — always true (squares mod 4 are 0 or 1, never 3).
fn opaque_square_mod4(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let n: i64 = rng.random_range(1..10000);

    let v_n = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_n, value: Literal::Int(n) } });

    let v_sq = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_sq, op: BinOp::Mul, left: v_n, right: v_n } });

    let v_four = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_four, value: Literal::Int(4) } });

    let v_mod = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_mod, op: BinOp::Mod, left: v_sq, right: v_four } });

    let v_three = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const { dst: v_three, value: Literal::Int(3) } });

    let v_cond = ctx.alloc_val(Ty::Bool);
    out.push(Inst { span, kind: InstKind::BinOp { dst: v_cond, op: BinOp::Neq, left: v_mod, right: v_three } });

    out
}

fn find_last_defined_val(insts: &[Inst]) -> Option<ValueId> {
    for inst in insts.iter().rev() {
        let val = match &inst.kind {
            InstKind::Const { dst, .. }
            | InstKind::BinOp { dst, .. }
            | InstKind::UnaryOp { dst, .. }
            | InstKind::Call { dst, .. } => Some(*dst),
            _ => None,
        };
        if val.is_some() {
            return val;
        }
    }
    None
}

/// Generate dead block contents: emit garbage strings so fake paths look like
/// real output paths. A static analyzer cannot distinguish these from genuine
/// emit instructions without proving the branch is unreachable.
fn make_dead_block(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
) -> Vec<Inst> {
    let mut out = Vec::new();
    let count = rng.random_range(1..4);

    for _ in 0..count {
        // Build a garbage string via XOR-encrypted char codes + int_to_char,
        // then emit it — mirrors real string-construction patterns.
        let len = rng.random_range(2..6);
        let mut v_accum: Option<ValueId> = None;

        for _ in 0..len {
            let code: i64 = rng.random_range(32..127); // printable ASCII
            let key: i64 = rng.random_range(1..256);
            let encrypted = code ^ key;

            let v_enc = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::Const { dst: v_enc, value: Literal::Int(encrypted) } });

            let v_key = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::Const { dst: v_key, value: Literal::Int(key) } });

            let v_dec = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_dec, op: BinOp::Xor, left: v_enc, right: v_key,
            }});

            let v_char = ctx.alloc_val(Ty::String);
            out.push(Inst { span, kind: InstKind::Call {
                dst: v_char, func: "int_to_char".into(), args: vec![v_dec],
            }});

            v_accum = Some(match v_accum {
                None => v_char,
                Some(prev) => {
                    let v_concat = ctx.alloc_val(Ty::String);
                    out.push(Inst { span, kind: InstKind::BinOp {
                        dst: v_concat, op: BinOp::Add, left: prev, right: v_char,
                    }});
                    v_concat
                }
            });
        }

        out.push(Inst { span, kind: InstKind::EmitValue(v_accum.unwrap()) });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::ir::DebugInfo;
    use rand::SeedableRng;
    use std::collections::HashMap;

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn make_ctx() -> PassState {
        PassState {
            insts: Vec::new(),
            val_types: HashMap::new(),
            debug: DebugInfo::new(),
            next_val: 200,
            next_label: 200,
        }
    }

    #[test]
    fn inserts_opaque_branches() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);

        let insts: Vec<Inst> = (0..10)
            .map(|i| Inst {
                span: span(),
                kind: InstKind::Const { dst: ValueId(i), value: Literal::Int(i as i64) },
            })
            .collect();

        let result = insert(insts, &mut ctx, &mut rng);

        let jumpif_count = result.iter().filter(|i| matches!(i.kind, InstKind::JumpIf { .. })).count();
        assert!(jumpif_count >= 1, "expected opaque predicate branches");

        let label_count = result.iter().filter(|i| matches!(i.kind, InstKind::BlockLabel { .. })).count();
        assert!(label_count >= 2, "expected dead + continue labels");
    }

    #[test]
    fn all_predicate_variants_produce_bool() {
        // Run enough times to cover all 5 variants.
        for seed in 0..20 {
            let mut ctx = make_ctx();
            let mut rng = StdRng::seed_from_u64(seed);
            let span = span();
            let (pred, _) = make_opaque_branch(&mut ctx, &mut rng, span);

            // Last instruction should define a Bool.
            let last = pred.last().unwrap();
            if let InstKind::BinOp { dst, .. } = &last.kind {
                assert_eq!(ctx.val_types[dst], Ty::Bool);
            } else {
                panic!("last predicate instruction should be BinOp");
            }
        }
    }
}
