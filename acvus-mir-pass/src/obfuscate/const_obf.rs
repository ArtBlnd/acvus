use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::ir::{InstKind, ValueId};
use acvus_mir::ty::Ty;
use rand::rngs::StdRng;
use rand::Rng;

use super::config::ObfConfig;
use super::rewriter::PassState;

/// Encrypt a string constant: decompose into chunks of 4-8 chars, XOR each
/// char code with a chained key, then reconstruct via int_to_char + concat.
/// Each chunk uses a randomly chosen decryption pipeline:
///   0 = XOR only
///   1 = XOR then Add offset
///   2 = XOR then BitAnd mask then Add offset
pub fn obfuscate_string(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    value: &Literal,
) {
    let Literal::String(s) = value else {
        unreachable!()
    };

    if s.is_empty() {
        ctx.emit(span, InstKind::Const {
            dst,
            value: Literal::String(String::new()),
        });
        return;
    }

    let chars: Vec<char> = s.chars().collect();
    let chunk_size = rng.random_range(4usize..=8).min(chars.len());
    let chunks: Vec<&[char]> = chars.chunks(chunk_size).collect();

    let mut v_accum: Option<ValueId> = None;
    let mut current_key: i64 = rng.random_range(1..256);

    for chunk in &chunks {
        // Pick decryption variant for this chunk.
        let variant = rng.random_range(0u32..2);

        // Pre-emit the key const for this chunk.
        let v_key = ctx.alloc_val(Ty::Int);
        ctx.emit(span, InstKind::Const {
            dst: v_key,
            value: Literal::Int(current_key),
        });

        // Variant 1: emit offset const.
        let (offset_val, v_offset) = if variant == 1 {
            let offset: i64 = rng.random_range(1..256);
            let vo = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::Const { dst: vo, value: Literal::Int(offset) });
            (offset, Some(vo))
        } else {
            (0, None)
        };

        let mut v_chunk_accum: Option<ValueId> = None;
        let mut v_current_key = v_key;

        for &ch in *chunk {
            let code = ch as i64;
            // Encrypt: apply inverse of the decryption pipeline.
            let encrypted = if variant == 1 {
                // decrypt: (enc ^ key) + offset = code => enc = (code - offset) ^ key
                (code - offset_val) ^ current_key
            } else {
                // decrypt: enc ^ key = code => enc = code ^ key
                code ^ current_key
            };

            let v_enc = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::Const {
                dst: v_enc,
                value: Literal::Int(encrypted),
            });

            // Step 1: XOR with key
            let v_xored = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::BinOp {
                dst: v_xored,
                op: BinOp::Xor,
                left: v_enc,
                right: v_current_key,
            });

            let v_dec_int = if variant == 1 {
                // Step 2: Add offset
                let v_added = ctx.alloc_val(Ty::Int);
                ctx.emit(span, InstKind::BinOp {
                    dst: v_added,
                    op: BinOp::Add,
                    left: v_xored,
                    right: v_offset.unwrap(),
                });
                v_added
            } else {
                v_xored
            };

            // int_to_char
            let v_char = ctx.alloc_val(Ty::String);
            ctx.emit(span, InstKind::Call {
                dst: v_char,
                func: "int_to_char".into(),
                args: vec![v_dec_int],
            });

            v_chunk_accum = Some(match v_chunk_accum {
                None => v_char,
                Some(prev) => {
                    let v_concat = ctx.alloc_val(Ty::String);
                    ctx.emit(span, InstKind::BinOp {
                        dst: v_concat,
                        op: BinOp::Add,
                        left: prev,
                        right: v_char,
                    });
                    v_concat
                }
            });

            // Chain: next key = decrypted char code.
            // Emit a const for the next key value so the scheduler can reorder it.
            let next_key_val = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::Const {
                dst: next_key_val,
                value: Literal::Int(code),
            });
            v_current_key = next_key_val;
            current_key = code;
        }

        // Concat this chunk to the overall accumulator.
        v_accum = Some(match v_accum {
            None => v_chunk_accum.unwrap(),
            Some(prev) => {
                let v_concat = ctx.alloc_val(Ty::String);
                ctx.emit(span, InstKind::BinOp {
                    dst: v_concat,
                    op: BinOp::Add,
                    left: prev,
                    right: v_chunk_accum.unwrap(),
                });
                v_concat
            }
        });
    }

    // Alias final accumulator to dst.
    let v_empty = ctx.alloc_val(Ty::String);
    ctx.emit(span, InstKind::Const {
        dst: v_empty,
        value: Literal::String(String::new()),
    });
    ctx.emit(span, InstKind::BinOp {
        dst,
        op: BinOp::Add,
        left: v_accum.unwrap(),
        right: v_empty,
    });
}

/// Split an integer constant: n = a + b where a is random.
pub fn obfuscate_int(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    value: &Literal,
) {
    let Literal::Int(n) = value else {
        unreachable!()
    };

    let a: i64 = rng.random_range(-1000000..1000000);
    let b = n.wrapping_sub(a);

    let v_a = ctx.alloc_val(Ty::Int);
    ctx.emit(span, InstKind::Const {
        dst: v_a,
        value: Literal::Int(a),
    });

    let v_b = ctx.alloc_val(Ty::Int);
    ctx.emit(span, InstKind::Const {
        dst: v_b,
        value: Literal::Int(b),
    });

    ctx.emit(span, InstKind::BinOp {
        dst,
        op: BinOp::Add,
        left: v_a,
        right: v_b,
    });
}

/// Split a float constant: f = a + b where a is random.
pub fn obfuscate_float(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    value: &Literal,
) {
    let Literal::Float(f) = value else {
        unreachable!()
    };

    let a: f64 = rng.random_range(-1000000.0..1000000.0);
    let b = f - a;

    let v_a = ctx.alloc_val(Ty::Float);
    ctx.emit(span, InstKind::Const {
        dst: v_a,
        value: Literal::Float(a),
    });

    let v_b = ctx.alloc_val(Ty::Float);
    ctx.emit(span, InstKind::Const {
        dst: v_b,
        value: Literal::Float(b),
    });

    ctx.emit(span, InstKind::BinOp {
        dst,
        op: BinOp::Add,
        left: v_a,
        right: v_b,
    });
}

/// Bool obfuscation: true → (rand == rand), false → (x == y) where x ≠ y.
pub fn obfuscate_bool(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    value: &Literal,
) {
    let Literal::Bool(b) = value else {
        unreachable!()
    };

    let x: i64 = rng.random_range(0..1000000);
    let y = if *b {
        x
    } else {
        loop {
            let v: i64 = rng.random_range(0..1000000);
            if v != x {
                break v;
            }
        }
    };

    let v_x = ctx.alloc_val(Ty::Int);
    ctx.emit(span, InstKind::Const {
        dst: v_x,
        value: Literal::Int(x),
    });

    let v_y = ctx.alloc_val(Ty::Int);
    ctx.emit(span, InstKind::Const {
        dst: v_y,
        value: Literal::Int(y),
    });

    ctx.emit(span, InstKind::BinOp {
        dst,
        op: BinOp::Eq,
        left: v_x,
        right: v_y,
    });
}

/// Decompose TestLiteral { dst, src, value } into:
///   v_lit = (obfuscated const load)
///   dst   = BinOp(Eq, src, v_lit)
pub fn decompose_test_literal(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    dst: ValueId,
    src: ValueId,
    value: &Literal,
    config: &ObfConfig,
) {
    let ty = match value {
        Literal::String(_) => Ty::String,
        Literal::Int(_) => Ty::Int,
        Literal::Float(_) => Ty::Float,
        Literal::Bool(_) => Ty::Bool,
    };

    let v_lit = ctx.alloc_val(ty);

    match value {
        Literal::String(_) if config.string_encryption => {
            obfuscate_string(ctx, rng, span, v_lit, value);
        }
        Literal::Int(_) if config.numeric_split => {
            obfuscate_int(ctx, rng, span, v_lit, value);
        }
        Literal::Float(_) if config.numeric_split => {
            obfuscate_float(ctx, rng, span, v_lit, value);
        }
        Literal::Bool(_) if config.numeric_split => {
            obfuscate_bool(ctx, rng, span, v_lit, value);
        }
        _ => {
            ctx.emit(span, InstKind::Const {
                dst: v_lit,
                value: value.clone(),
            });
        }
    }

    ctx.emit(span, InstKind::BinOp {
        dst,
        op: BinOp::Eq,
        left: src,
        right: v_lit,
    });
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
    fn int_split_roundtrip() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);
        let dst = ValueId(0);
        ctx.val_types.insert(dst, Ty::Int);
        obfuscate_int(&mut ctx, &mut rng, Span { start: 0, end: 0 }, dst, &Literal::Int(42));

        // Should produce 3 instructions: Const(a), Const(b), BinOp(Add).
        assert_eq!(ctx.insts.len(), 3);
        assert!(matches!(ctx.insts[2].kind, InstKind::BinOp { op: BinOp::Add, .. }));
    }

    #[test]
    fn bool_true_produces_eq() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);
        let dst = ValueId(0);
        ctx.val_types.insert(dst, Ty::Bool);
        obfuscate_bool(&mut ctx, &mut rng, Span { start: 0, end: 0 }, dst, &Literal::Bool(true));

        assert_eq!(ctx.insts.len(), 3);
        if let (
            InstKind::Const { value: Literal::Int(a), .. },
            InstKind::Const { value: Literal::Int(b), .. },
        ) = (&ctx.insts[0].kind, &ctx.insts[1].kind)
        {
            assert_eq!(a, b);
        } else {
            panic!("expected two Int consts");
        }
    }

    #[test]
    fn bool_false_produces_neq() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);
        let dst = ValueId(0);
        ctx.val_types.insert(dst, Ty::Bool);
        obfuscate_bool(&mut ctx, &mut rng, Span { start: 0, end: 0 }, dst, &Literal::Bool(false));

        assert_eq!(ctx.insts.len(), 3);
        if let (
            InstKind::Const { value: Literal::Int(a), .. },
            InstKind::Const { value: Literal::Int(b), .. },
        ) = (&ctx.insts[0].kind, &ctx.insts[1].kind)
        {
            assert_ne!(a, b);
        } else {
            panic!("expected two Int consts");
        }
    }

    #[test]
    fn string_empty_passthrough() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);
        let dst = ValueId(0);
        ctx.val_types.insert(dst, Ty::String);
        obfuscate_string(&mut ctx, &mut rng, Span { start: 0, end: 0 }, dst, &Literal::String(String::new()));

        assert_eq!(ctx.insts.len(), 1);
        assert!(matches!(&ctx.insts[0].kind, InstKind::Const { value: Literal::String(s), .. } if s.is_empty()));
    }

    #[test]
    fn string_encryption_produces_xor_chain() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);
        let dst = ValueId(0);
        ctx.val_types.insert(dst, Ty::String);
        obfuscate_string(&mut ctx, &mut rng, Span { start: 0, end: 0 }, dst, &Literal::String("hi".into()));

        assert!(ctx.insts.len() > 5);

        // The original string "hi" should not appear in any instruction.
        for inst in &ctx.insts {
            if let InstKind::Const { value: Literal::String(s), .. } = &inst.kind {
                assert_ne!(s, "hi");
                assert_ne!(s, "h");
                assert_ne!(s, "i");
            }
        }
    }
}
