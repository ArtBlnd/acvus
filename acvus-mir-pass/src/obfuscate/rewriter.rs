use std::collections::HashMap;

use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::ir::{
    DebugInfo, Inst, InstKind, Label, MirBody, MirModule, ValOrigin, ValueId,
};
use acvus_mir::ty::Ty;
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::cff;
use super::config::ObfConfig;
use super::const_obf;
use super::mba;
use super::opaque;
use super::scheduler;
use super::text_obf;

pub fn obfuscate(mut module: MirModule, config: &ObfConfig) -> MirModule {
    let mut rng = StdRng::seed_from_u64(config.seed);

    let text_map = if config.text_encryption {
        Some(text_obf::encrypt_texts(&module.texts, &mut rng))
    } else {
        None
    };

    module.main = rewrite_body(module.main, config, &text_map, &mut rng);
    let closure_labels: Vec<Label> = module.closures.keys().copied().collect();
    for label in closure_labels {
        let mut closure = module.closures.remove(&label).unwrap();
        closure.body = rewrite_body(closure.body, config, &text_map, &mut rng);
        module.closures.insert(label, closure);
    }

    if config.text_encryption {
        module.texts.clear();
    }

    module
}

fn rewrite_body(
    body: MirBody,
    config: &ObfConfig,
    text_map: &Option<Vec<text_obf::EncryptedText>>,
    rng: &mut StdRng,
) -> MirBody {
    let mut ctx = PassState::from_body(&body);

    // Phase 1: Instruction-level transforms (const, string, text, TestLiteral).
    phase_instruction_transform(&mut ctx, &body.insts, config, text_map, rng);

    // Phase 2: MBA — replace XOR/BitAnd with algebraic equivalents.
    if config.mba {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = mba::apply(insts, &mut ctx, rng);
    }

    // Phase 3: Instruction scheduling — reorder within basic blocks.
    if config.scheduling {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = scheduler::reorder(insts, rng);
    }

    // Phase 4: Control flow flattening — split blocks + shuffle / dispatcher.
    if config.control_flow_flatten {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = cff::flatten(insts, &mut ctx, rng);
    }

    // Phase 5: Opaque predicates — fake branches with dead blocks.
    if config.opaque_predicates {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = opaque::insert(insts, &mut ctx, rng);
    }

    // Phase 6: Dead code — noise instructions.
    if config.dead_code {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = dead_code_insert(insts, &mut ctx, rng);
    }

    ctx.into_body()
}

fn phase_instruction_transform(
    ctx: &mut PassState,
    original: &[Inst],
    config: &ObfConfig,
    text_map: &Option<Vec<text_obf::EncryptedText>>,
    rng: &mut StdRng,
) {
    for inst in original {
        match &inst.kind {
            InstKind::Const { dst, value } => match value {
                Literal::String(_) if config.string_encryption => {
                    const_obf::obfuscate_string(ctx, rng, inst.span, *dst, value);
                    continue;
                }
                Literal::Int(_) if config.numeric_split => {
                    const_obf::obfuscate_int(ctx, rng, inst.span, *dst, value);
                    continue;
                }
                Literal::Float(_) if config.numeric_split => {
                    const_obf::obfuscate_float(ctx, rng, inst.span, *dst, value);
                    continue;
                }
                Literal::Bool(_) if config.numeric_split => {
                    const_obf::obfuscate_bool(ctx, rng, inst.span, *dst, value);
                    continue;
                }
                _ => {}
            },
            InstKind::EmitText(idx) => {
                if let Some(texts) = text_map {
                    text_obf::emit_encrypted_text(ctx, rng, inst.span, &texts[*idx]);
                    continue;
                }
            }
            InstKind::TestLiteral { dst, src, value } if config.test_literal_decompose => {
                const_obf::decompose_test_literal(ctx, rng, inst.span, *dst, *src, value, config);
                continue;
            }
            _ => {}
        }
        ctx.emit(inst.span, inst.kind.clone());
    }
}

fn dead_code_insert(
    mut insts: Vec<Inst>,
    ctx: &mut PassState,
    rng: &mut StdRng,
) -> Vec<Inst> {
    use rand::Rng;

    let count = rng.random_range(8..20);
    for _ in 0..count {
        if insts.is_empty() {
            break;
        }
        let pos = rng.random_range(0..insts.len());
        let span = insts[pos].span;

        match rng.random_range(0u32..6) {
            0 => {
                insts.insert(pos, Inst { span, kind: InstKind::Nop });
            }
            1 => {
                let v = ctx.alloc_val(Ty::Int);
                insts.insert(pos, Inst {
                    span,
                    kind: InstKind::Const { dst: v, value: Literal::Int(rng.random_range(-999999..999999)) },
                });
            }
            2 => {
                // Dead computation: a & b where a, b are random consts.
                let v_a = ctx.alloc_val(Ty::Int);
                let v_b = ctx.alloc_val(Ty::Int);
                let v_r = ctx.alloc_val(Ty::Int);
                let a: i64 = rng.random_range(1..999999);
                let b: i64 = rng.random_range(1..999999);
                insts.insert(pos, Inst { span, kind: InstKind::Const { dst: v_a, value: Literal::Int(a) } });
                insts.insert(pos + 1, Inst { span, kind: InstKind::Const { dst: v_b, value: Literal::Int(b) } });
                insts.insert(pos + 2, Inst {
                    span,
                    kind: InstKind::BinOp { dst: v_r, op: BinOp::BitAnd, left: v_a, right: v_b },
                });
            }
            3 => {
                // Dead MBA-like: (a + b) - (a + b) = 0, unused.
                let v_a = ctx.alloc_val(Ty::Int);
                let v_b = ctx.alloc_val(Ty::Int);
                let v_sum = ctx.alloc_val(Ty::Int);
                let v_r = ctx.alloc_val(Ty::Int);
                let a: i64 = rng.random_range(1..999999);
                let b: i64 = rng.random_range(1..999999);
                insts.insert(pos, Inst { span, kind: InstKind::Const { dst: v_a, value: Literal::Int(a) } });
                insts.insert(pos + 1, Inst { span, kind: InstKind::Const { dst: v_b, value: Literal::Int(b) } });
                insts.insert(pos + 2, Inst {
                    span,
                    kind: InstKind::BinOp { dst: v_sum, op: BinOp::Add, left: v_a, right: v_b },
                });
                insts.insert(pos + 3, Inst {
                    span,
                    kind: InstKind::BinOp { dst: v_r, op: BinOp::Sub, left: v_sum, right: v_sum },
                });
            }
            4 => {
                // Shl/Shr chain: a << s >> s (identity for small s).
                let v_a = ctx.alloc_val(Ty::Int);
                let v_s = ctx.alloc_val(Ty::Int);
                let v_shl = ctx.alloc_val(Ty::Int);
                let v_shr = ctx.alloc_val(Ty::Int);
                let a: i64 = rng.random_range(1..999999);
                let s: i64 = rng.random_range(1..8);
                insts.insert(pos, Inst { span, kind: InstKind::Const { dst: v_a, value: Literal::Int(a) } });
                insts.insert(pos + 1, Inst { span, kind: InstKind::Const { dst: v_s, value: Literal::Int(s) } });
                insts.insert(pos + 2, Inst {
                    span,
                    kind: InstKind::BinOp { dst: v_shl, op: BinOp::Shl, left: v_a, right: v_s },
                });
                insts.insert(pos + 3, Inst {
                    span,
                    kind: InstKind::BinOp { dst: v_shr, op: BinOp::Shr, left: v_shl, right: v_s },
                });
            }
            _ => {
                // Xor identity: a ^ a = 0, unused.
                let v_a = ctx.alloc_val(Ty::Int);
                let v_r = ctx.alloc_val(Ty::Int);
                let a: i64 = rng.random_range(1..999999);
                insts.insert(pos, Inst { span, kind: InstKind::Const { dst: v_a, value: Literal::Int(a) } });
                insts.insert(pos + 1, Inst {
                    span,
                    kind: InstKind::BinOp { dst: v_r, op: BinOp::Xor, left: v_a, right: v_a },
                });
            }
        }
    }
    insts
}

// ── Shared pass state ──────────────────────────────────────────

pub struct PassState {
    pub insts: Vec<Inst>,
    pub val_types: HashMap<ValueId, Ty>,
    pub debug: DebugInfo,
    pub next_val: u32,
    pub next_label: u32,
}

impl PassState {
    pub fn from_body(body: &MirBody) -> Self {
        Self {
            insts: Vec::new(),
            val_types: body.val_types.clone(),
            debug: body.debug.clone(),
            next_val: body.val_count,
            next_label: body.label_count,
        }
    }

    pub fn into_body(self) -> MirBody {
        MirBody {
            insts: self.insts,
            val_types: self.val_types,
            debug: self.debug,
            val_count: self.next_val,
            label_count: self.next_label,
        }
    }

    pub fn alloc_val(&mut self, ty: Ty) -> ValueId {
        let id = ValueId(self.next_val);
        self.next_val += 1;
        self.val_types.insert(id, ty);
        self.debug.set(id, ValOrigin::Expr);
        id
    }

    pub fn alloc_label(&mut self) -> Label {
        let id = Label(self.next_label);
        self.next_label += 1;
        id
    }

    pub fn emit(&mut self, span: Span, kind: InstKind) {
        self.insts.push(Inst { span, kind });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ObfConfig {
        ObfConfig {
            seed: 42,
            ..ObfConfig::default()
        }
    }

    fn make_module(insts: Vec<Inst>, texts: Vec<String>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: HashMap::new(),
                debug: DebugInfo::new(),
                val_count: 10,
                label_count: 0,
            },
            closures: HashMap::new(),
            texts,
        }
    }

    #[test]
    fn nop_passes_through() {
        let module = make_module(
            vec![Inst {
                span: Span { start: 0, end: 0 },
                kind: InstKind::Nop,
            }],
            vec![],
        );
        let result = obfuscate(module, &default_config());
        assert!(result.main.insts.iter().any(|i| matches!(i.kind, InstKind::Nop)));
    }
}
