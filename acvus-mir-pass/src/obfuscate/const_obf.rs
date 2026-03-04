use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::ir::{InstKind, ValueId};
use acvus_mir::ty::Ty;
use rand::Rng;
use rand::rngs::StdRng;

use super::rewriter::PassState;

// ── Hash predicate ──────────────────────────────────────────────

struct HashParams {
    p1: i64,
    p2: i64,
    p3: i64,
}

fn gen_hash_params(rng: &mut StdRng) -> HashParams {
    HashParams {
        p1: rng.random_range(1000i64..10000),
        p2: rng.random_range(1000i64..10000),
        p3: rng.random_range(10000i64..100000),
    }
}

/// Compile-time hash: non-negative ((x * p1) ^ p2) % p3.
fn compute_hash(x: i64, params: &HashParams) -> i64 {
    let raw = (x.wrapping_mul(params.p1)) ^ params.p2;
    ((raw % params.p3) + params.p3) % params.p3
}

/// Emit MIR that computes hash(src) at runtime. Returns the hash ValueId.
fn emit_hash(ctx: &mut PassState, span: Span, src: ValueId, params: &HashParams) -> ValueId {
    let v_p1 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_p1,
            value: Literal::Int(params.p1),
        },
    );

    let v_mul = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_mul,
            op: BinOp::Mul,
            left: src,
            right: v_p1,
        },
    );

    let v_p2 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_p2,
            value: Literal::Int(params.p2),
        },
    );

    let v_xor = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_xor,
            op: BinOp::Xor,
            left: v_mul,
            right: v_p2,
        },
    );

    let v_p3 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_p3,
            value: Literal::Int(params.p3),
        },
    );

    let v_mod1 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_mod1,
            op: BinOp::Mod,
            left: v_xor,
            right: v_p3,
        },
    );

    let v_add = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_add,
            op: BinOp::Add,
            left: v_mod1,
            right: v_p3,
        },
    );

    let v_hash = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_hash,
            op: BinOp::Mod,
            left: v_add,
            right: v_p3,
        },
    );

    v_hash
}

/// Replace TestLiteral(Int) with: hash(src) → VarStore + checksum comparison.
/// Returns compile_key = hash(literal_value) for text decryption.
///
/// The checksum result is written to `dst` so the original JumpIf still works.
/// This hides the literal value while preserving the branch structure.
pub fn hash_test_literal(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    src: ValueId,
    value: &Literal,
) -> i64 {
    let Literal::Int(n) = value else {
        unreachable!("hash_test_literal only supports Int");
    };

    let params = gen_hash_params(rng);
    let compile_key = compute_hash(*n, &params);

    let v_hash = emit_hash(ctx, span, src, &params);
    ctx.emit(
        span,
        InstKind::VarStore {
            name: "__obf_key".into(),
            src: v_hash,
        },
    );

    // Checksum: ((hash * C1 + C2) % C3 + C3) % C3 == expected
    let c1: i64 = rng.random_range(1000i64..10000);
    let c2: i64 = rng.random_range(1000i64..10000);
    let c3: i64 = rng.random_range(10000i64..100000);

    let expected = {
        let raw = compile_key.wrapping_mul(c1).wrapping_add(c2);
        ((raw % c3) + c3) % c3
    };

    let v_c1 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_c1,
            value: Literal::Int(c1),
        },
    );

    let v_mul = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_mul,
            op: BinOp::Mul,
            left: v_hash,
            right: v_c1,
        },
    );

    let v_c2 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_c2,
            value: Literal::Int(c2),
        },
    );

    let v_add = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_add,
            op: BinOp::Add,
            left: v_mul,
            right: v_c2,
        },
    );

    let v_c3 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_c3,
            value: Literal::Int(c3),
        },
    );

    let v_mod1 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_mod1,
            op: BinOp::Mod,
            left: v_add,
            right: v_c3,
        },
    );

    let v_add2 = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_add2,
            op: BinOp::Add,
            left: v_mod1,
            right: v_c3,
        },
    );

    let v_checksum = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::BinOp {
            dst: v_checksum,
            op: BinOp::Mod,
            left: v_add2,
            right: v_c3,
        },
    );

    let v_expected = ctx.alloc_val(Ty::Int);
    ctx.emit(
        span,
        InstKind::Const {
            dst: v_expected,
            value: Literal::Int(expected),
        },
    );

    ctx.emit(
        span,
        InstKind::BinOp {
            dst,
            op: BinOp::Eq,
            left: v_checksum,
            right: v_expected,
        },
    );

    compile_key
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
    fn hash_test_literal_emits_hash_var_store_and_checksum() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);
        let src = ValueId(0);
        let dst = ValueId(1);
        ctx.val_types.insert(src, Ty::Int);
        ctx.val_types.insert(dst, Ty::Bool);

        let key = hash_test_literal(
            &mut ctx,
            &mut rng,
            Span { start: 0, end: 0 },
            dst,
            src,
            &Literal::Int(1),
        );

        // VarStore __obf_key present.
        assert!(ctx.insts.iter().any(|i| matches!(
            &i.kind,
            InstKind::VarStore { name, .. } if name == "__obf_key"
        )));

        // Last instruction is Eq (checksum comparison) writing to dst.
        assert!(matches!(
            ctx.insts.last().unwrap().kind,
            InstKind::BinOp { dst: d, op: BinOp::Eq, .. } if d == dst
        ));

        // No literal value exposed (only hash params, checksum params).
        // The original literal (1) should NOT appear.
        let has_literal_1 = ctx.insts.iter().any(|i| {
            matches!(
                &i.kind,
                InstKind::Const {
                    value: Literal::Int(1),
                    ..
                }
            )
        });
        assert!(!has_literal_1);

        // compile_key is deterministic.
        assert_ne!(key, 0);
    }
}
