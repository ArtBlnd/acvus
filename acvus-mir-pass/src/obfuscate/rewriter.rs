use std::collections::HashMap;

use acvus_ast::{Literal, Span};
use acvus_mir::ir::{
    DebugInfo, Inst, InstKind, Label, MirBody, MirModule, ValOrigin, ValueId,
};
use acvus_mir::ty::Ty;
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::cff;
use super::config::ObfConfig;
use super::const_obf;
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

    module.main = rewrite_body(module.main, config, &text_map, &module.texts, &mut rng);
    let closure_labels: Vec<Label> = module.closures.keys().copied().collect();
    for label in closure_labels {
        let mut closure = module.closures.remove(&label).unwrap();
        closure.body = rewrite_body(closure.body, config, &text_map, &module.texts, &mut rng);
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
    texts: &[String],
    rng: &mut StdRng,
) -> MirBody {
    let mut ctx = PassState::from_body(&body);

    // Phase 1: Instruction-level transforms (text, hash predicate).
    phase_instruction_transform(&mut ctx, &body.insts, config, text_map, texts, rng);

    // Phase 2: Instruction scheduling — reorder within basic blocks.
    if config.scheduling {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = scheduler::reorder(insts, rng);
    }

    // Phase 3: Control flow flattening — split blocks + shuffle / dispatcher.
    if config.control_flow_flatten {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = cff::flatten(insts, &mut ctx, rng);
    }

    // Phase 4: Opaque predicates — fake branches with garbage emit.
    if config.opaque_predicates {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = opaque::insert(insts, &mut ctx, rng);
    }

    ctx.into_body()
}

fn phase_instruction_transform(
    ctx: &mut PassState,
    original: &[Inst],
    config: &ObfConfig,
    text_map: &Option<Vec<text_obf::EncryptedText>>,
    texts: &[String],
    rng: &mut StdRng,
) {
    // Hash predicate state: tracks compile_key from most recent hash_test_literal.
    let mut active_hash_key: Option<i64> = None;
    let mut blocks_since_hash: u32 = 0;

    for inst in original {
        match &inst.kind {
            InstKind::BlockLabel { .. } => {
                blocks_since_hash += 1;
                if blocks_since_hash >= 2 {
                    active_hash_key = None;
                }
            }
            InstKind::EmitText(idx) => {
                if let Some(key) = active_hash_key.take() {
                    text_obf::emit_hashed_text(ctx, rng, inst.span, &texts[*idx], key);
                    continue;
                }
                if let Some(enc_texts) = text_map {
                    text_obf::emit_encrypted_text(ctx, rng, inst.span, &enc_texts[*idx]);
                    continue;
                }
            }
            InstKind::TestLiteral { dst, src, value }
                if config.hash_predicate && matches!(value, Literal::Int(_)) =>
            {
                let key = const_obf::hash_test_literal(
                    ctx, rng, inst.span, *dst, *src, value,
                );
                active_hash_key = Some(key);
                blocks_since_hash = 0;
                continue;
            }
            _ => {}
        }
        ctx.emit(inst.span, inst.kind.clone());
    }
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
