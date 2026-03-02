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

/// All registered (internal) labels that should not be rewritten.
struct RegisteredLabels {
    all: Vec<Label>,
}

pub fn obfuscate(mut module: MirModule, config: &ObfConfig) -> MirModule {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut registered = RegisteredLabels { all: Vec::new() };

    // Register multi-stage decrypt closures (12 closures: 4 per stage).
    let decrypt_table = if config.text_encryption {
        let table = text_obf::register_multistage_decrypt_closures(&mut module);
        registered.all.extend(text_obf::all_decrypt_labels(&table));
        Some(table)
    } else {
        None
    };

    // Register factory closures for each decrypt stage (4 factories × 3 stages = 12).
    let factory_tables = if config.text_encryption {
        let dt = decrypt_table.as_ref().unwrap();
        let stage_a_fn_ty = Ty::Fn { params: vec![Ty::Int], ret: Box::new(Ty::Int) };
        let stage_b_fn_ty = Ty::Fn { params: vec![Ty::Int, Ty::Int], ret: Box::new(Ty::Int) };
        let stage_c_fn_ty = Ty::Fn { params: vec![Ty::bytes(), Ty::Int], ret: Box::new(Ty::String) };

        let fa = text_obf::register_factory_closures(&mut module, &dt.stage_a, &stage_a_fn_ty, "factory_a");
        registered.all.extend(text_obf::all_factory_labels(&fa));
        let fb = text_obf::register_factory_closures(&mut module, &dt.stage_b, &stage_b_fn_ty, "factory_b");
        registered.all.extend(text_obf::all_factory_labels(&fb));
        let fc = text_obf::register_factory_closures(&mut module, &dt.stage_c, &stage_c_fn_ty, "factory_c");
        registered.all.extend(text_obf::all_factory_labels(&fc));

        Some((fa, fb, fc))
    } else {
        None
    };

    // Register opaque closures (4 closures).
    let opaque_table = if config.opaque_predicates {
        let table = opaque::register_opaque_closures(&mut module);
        registered.all.extend(table.labels.iter().copied());

        // Register factory closures for opaque dispatch (4 factories).
        let opaque_fn_ty = Ty::Fn { params: vec![Ty::Int], ret: Box::new(Ty::Int) };
        let opaque_factory = text_obf::register_factory_closures(
            &mut module, &table.labels, &opaque_fn_ty, "factory_opaque",
        );
        registered.all.extend(text_obf::all_factory_labels(&opaque_factory));

        Some((table, opaque_factory))
    } else {
        None
    };

    let text_map = if config.text_encryption {
        Some(text_obf::encrypt_texts(&module.texts, &mut rng))
    } else {
        None
    };

    module.main = rewrite_body(
        module.main, config, &text_map, &module.texts,
        &decrypt_table, &factory_tables, &opaque_table, &mut rng,
    );

    // User closures: skip meta table/entangle preamble (no text in closures).
    let closure_config = ObfConfig {
        seed: config.seed,
        text_encryption: false,
        scheduling: config.scheduling,
        control_flow_flatten: config.control_flow_flatten,
        opaque_predicates: config.opaque_predicates,
        hash_predicate: false,
    };
    let closure_labels: Vec<Label> = module.closures.keys()
        .copied()
        .filter(|l| !registered.all.contains(l))
        .collect();
    for label in closure_labels {
        let mut closure = module.closures.remove(&label).unwrap();
        closure.body = rewrite_body(
            closure.body, &closure_config, &None, &module.texts,
            &None, &None, &opaque_table, &mut rng,
        );
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
    decrypt_table: &Option<text_obf::MultiStageDecryptTable>,
    factory_tables: &Option<(text_obf::FactoryTable, text_obf::FactoryTable, text_obf::FactoryTable)>,
    opaque_table: &Option<(opaque::OpaqueTable, text_obf::FactoryTable)>,
    rng: &mut StdRng,
) -> MirBody {
    let mut ctx = PassState::from_body(&body);
    let span = body.insts.first().map(|i| i.span).unwrap_or(Span { start: 0, end: 0 });

    // Emit v_four constant at the top (used by text decryption dispatch).
    let v_four = if config.text_encryption {
        let v = ctx.alloc_val(Ty::Int);
        ctx.emit(span, InstKind::Const { dst: v, value: Literal::Int(4) });
        Some(v)
    } else {
        None
    };

    // Initialize __entangle = 1 (opaque predicates will overwrite, but text
    // decrypt may run before the first opaque predicate in a body).
    let use_entangle = config.opaque_predicates && config.text_encryption;
    if use_entangle {
        let v_init = ctx.alloc_val(Ty::Int);
        ctx.emit(span, InstKind::Const { dst: v_init, value: Literal::Int(1) });
        ctx.emit(span, InstKind::VarStore {
            name: "__entangle".into(),
            src: v_init,
        });
    }

    // Emit 3 meta tables for multi-stage decrypt (each is a factory dispatch setup).
    let meta_tables = if let (Some(dt), Some((fa, fb, fc))) = (decrypt_table, factory_tables) {
        let stage_a_fn_ty = Ty::Fn { params: vec![Ty::Int], ret: Box::new(Ty::Int) };
        let stage_b_fn_ty = Ty::Fn { params: vec![Ty::Int, Ty::Int], ret: Box::new(Ty::Int) };
        let stage_c_fn_ty = Ty::Fn { params: vec![Ty::bytes(), Ty::Int], ret: Box::new(Ty::String) };

        let meta_a = text_obf::emit_factory_dispatch_setup(&mut ctx, span, &dt.stage_a, fa, &stage_a_fn_ty);
        let meta_b = text_obf::emit_factory_dispatch_setup(&mut ctx, span, &dt.stage_b, fb, &stage_b_fn_ty);
        let meta_c = text_obf::emit_factory_dispatch_setup(&mut ctx, span, &dt.stage_c, fc, &stage_c_fn_ty);

        // Store meta tables in variables for CFF/scheduler block safety
        ctx.emit(span, InstKind::VarStore { name: "__decrypt_meta_a".into(), src: meta_a });
        ctx.emit(span, InstKind::VarStore { name: "__decrypt_meta_b".into(), src: meta_b });
        ctx.emit(span, InstKind::VarStore { name: "__decrypt_meta_c".into(), src: meta_c });

        Some((meta_a, meta_b, meta_c))
    } else {
        None
    };

    let has_opaque = opaque_table.is_some();

    // Phase 1: Instruction-level transforms (text, hash predicate).
    phase_instruction_transform(
        &mut ctx, &body.insts, config, text_map, texts,
        meta_tables, v_four, use_entangle, rng,
    );

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

    // Phase 4: Opaque predicates — 2-level factory-based dispatch.
    // Insert opaque predicates first (they use VarLoad __opaque_meta),
    // then prepend the table setup + VarStore so it always runs first.
    if config.opaque_predicates && has_opaque {
        let insts = std::mem::take(&mut ctx.insts);
        ctx.insts = opaque::insert(insts, &mut ctx, rng, meta_tables.map(|m| m.2));

        // Build preamble: opaque meta table (factory dispatch) + VarStore.
        let body_insts = std::mem::take(&mut ctx.insts);
        let (opaque_tbl, opaque_factory) = opaque_table.as_ref().unwrap();
        let opaque_fn_ty = Ty::Fn { params: vec![Ty::Int], ret: Box::new(Ty::Int) };
        let v_opaque_meta = text_obf::emit_factory_dispatch_setup(
            &mut ctx, span, &opaque_tbl.labels, opaque_factory, &opaque_fn_ty,
        );
        ctx.emit(span, InstKind::VarStore {
            name: "__opaque_table".into(),
            src: v_opaque_meta,
        });
        ctx.insts.extend(body_insts);
    }

    ctx.into_body()
}

fn phase_instruction_transform(
    ctx: &mut PassState,
    original: &[Inst],
    config: &ObfConfig,
    text_map: &Option<Vec<text_obf::EncryptedText>>,
    texts: &[String],
    meta_tables: Option<(ValueId, ValueId, ValueId)>,
    v_four: Option<ValueId>,
    use_entangle: bool,
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
                if let (Some((ma, mb, mc)), Some(_vf)) = (meta_tables, v_four) {
                    // Load meta tables from variables (safe across CFF blocks).
                    let meta_a = ctx.alloc_val(ctx.val_types[&ma].clone());
                    ctx.emit(inst.span, InstKind::VarLoad { dst: meta_a, name: "__decrypt_meta_a".into() });
                    let meta_b = ctx.alloc_val(ctx.val_types[&mb].clone());
                    ctx.emit(inst.span, InstKind::VarLoad { dst: meta_b, name: "__decrypt_meta_b".into() });
                    let meta_c = ctx.alloc_val(ctx.val_types[&mc].clone());
                    ctx.emit(inst.span, InstKind::VarLoad { dst: meta_c, name: "__decrypt_meta_c".into() });

                    // Need fresh v_four per usage site (CFF safety).
                    let v_four_local = ctx.alloc_val(Ty::Int);
                    ctx.emit(inst.span, InstKind::Const { dst: v_four_local, value: Literal::Int(4) });

                    if let Some(key) = active_hash_key.take() {
                        text_obf::emit_hashed_text(
                            ctx, rng, inst.span, &texts[*idx], key,
                            meta_a, meta_b, meta_c, v_four_local, use_entangle,
                        );
                        continue;
                    }
                    if let Some(enc_texts) = text_map {
                        text_obf::emit_encrypted_text(
                            ctx, inst.span, &enc_texts[*idx],
                            meta_a, meta_b, meta_c, v_four_local, use_entangle,
                        );
                        continue;
                    }
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
