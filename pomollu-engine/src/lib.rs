mod config;
mod convert;
pub mod error;
mod fetch;
mod history;
mod idb;
pub mod schema;
mod session;

#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn init() {
    console_error_panic_hook::set_once();
}

use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir_pass::analysis::reachable_context::KnownValue;
use acvus_mir::ty::Ty;
use acvus_orchestration::NodeSpec;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};
use tsify::Ts;
use tsify::Tsify;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;

use acvus_mir::context_registry::{ContextTypeRegistry, PartialContextTypeRegistry, RegistryConflictError};
use schema::*;

/// Compile-time context types for asset extern functions.
fn asset_context_types(interner: &Interner) -> FxHashMap<Astr, Ty> {
    let mut types = FxHashMap::default();
    types.insert(interner.intern("asset_url"), Ty::Fn {
        params: vec![Ty::String],
        ret: Box::new(Ty::Option(Box::new(Ty::String))),
        is_extern: true,
        captures: vec![],
    });
    types
}

/// Build a registry with extern fns, empty system, and user-provided types.
fn build_registry(
    interner: &Interner,
    user_types: FxHashMap<Astr, Ty>,
) -> Result<PartialContextTypeRegistry, RegistryConflictError> {
    let mut extern_fns = acvus_ext::regex_context_types(interner);
    extern_fns.extend(asset_context_types(interner));
    PartialContextTypeRegistry::new(extern_fns, FxHashMap::default(), user_types)
}


/// Try to compile a short script and extract a known value (literal or variant).
/// Returns None if the script is not a simple constant expression.
fn try_extract_known(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> Option<KnownValue> {
    if source.trim().is_empty() {
        return None;
    }
    let script = acvus_ast::parse_script(interner, source).ok()?;
    let (module, _, _) = acvus_mir::compile_script_analysis(interner, &script, registry).ok()?;
    // Look for a Const or MakeVariant instruction in the main body
    for inst in &module.main.insts {
        match &inst.kind {
            InstKind::Const { value, .. } => {
                return Some(KnownValue::Literal(value.clone()));
            }
            InstKind::MakeVariant { tag, payload: None, .. } => {
                return Some(KnownValue::Variant { tag: *tag, payload: None });
            }
            _ => {}
        }
    }
    None
}

fn extract_context_keys_with_types(
    interner: &Interner,
    module: &MirModule,
    known: &FxHashMap<Astr, KnownValue>,
) -> Vec<ContextKey> {
    use acvus_mir_pass::AnalysisPass;
    use acvus_mir_pass::analysis::val_def::ValDefMapAnalysis;
    use acvus_mir_pass::analysis::reachable_context::partition_context_keys;

    let val_def = ValDefMapAnalysis.run(module, ());
    let partition = partition_context_keys(module, known, &val_def);

    // Collect types from val_types for each ContextLoad
    let mut type_map = FxHashMap::<Astr, Ty>::default();
    let mut collect_types = |insts: &[acvus_mir::ir::Inst],
                             val_types: &FxHashMap<acvus_mir::ir::ValueId, Ty>| {
        for inst in insts {
            if let InstKind::ContextLoad { dst, name, .. } = &inst.kind {
                type_map
                    .entry(*name)
                    .or_insert_with(|| val_types.get(dst).cloned().unwrap_or(Ty::Infer));
            }
        }
    };
    collect_types(&module.main.insts, &module.main.val_types);
    for body in module.closures.values() {
        collect_types(&body.body.insts, &body.body.val_types);
    }

    let mut seen = FxHashSet::<Astr>::default();
    let mut keys = Vec::new();

    for name in &partition.eager {
        seen.insert(*name);
        let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
        keys.push(ContextKey {
            name: interner.resolve(*name).to_string(),
            ty: ty_to_desc(interner, &ty),
            status: ContextKeyStatus::Eager,
        });
    }
    for name in &partition.lazy {
        seen.insert(*name);
        let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
        keys.push(ContextKey {
            name: interner.resolve(*name).to_string(),
            ty: ty_to_desc(interner, &ty),
            status: ContextKeyStatus::Lazy,
        });
    }

    // Re-add known keys that appear on reachable paths.
    // Keys behind dead branches are NOT re-added — this is how
    // value-based pruning hides params from the UI.
    for name in &partition.reachable_known {
        if !seen.contains(name) {
            seen.insert(*name);
            let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
            keys.push(ContextKey {
                name: interner.resolve(*name).to_string(),
                ty: ty_to_desc(interner, &ty),
                status: ContextKeyStatus::Eager,
            });
        }
    }

    // Pruned keys: in dead branches, not needed at runtime, but the
    // typechecker compiles all branches and needs their types injected.
    // Reported so the caller can inject types without showing them in the UI.
    for name in &partition.pruned {
        if !seen.contains(name) {
            seen.insert(*name);
            let ty = type_map.get(name).cloned().unwrap_or(Ty::Infer);
            keys.push(ContextKey {
                name: interner.resolve(*name).to_string(),
                ty: ty_to_desc(interner, &ty),
                status: ContextKeyStatus::Pruned,
            });
        }
    }

    keys.sort_by(|a, b| a.name.cmp(&b.name));
    keys
}

/// Convert a map of `name -> TypeDesc` into `HashMap<Astr, Ty>`.
fn convert_context_types(interner: &Interner, raw: &FxHashMap<String, TypeDesc>) -> FxHashMap<Astr, Ty> {
    raw.iter()
        .map(|(k, v)| (interner.intern(k), desc_to_ty(interner, v)))
        .collect()
}

fn do_analyze(
    interner: &Interner,
    source: &str,
    mode: &Mode,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
    known: &FxHashMap<Astr, KnownValue>,
    partial: &PartialContextTypeRegistry,
) -> AnalyzeResult {
    use error::EngineError;

    match mode {
        Mode::Template => {
            let ast = match acvus_ast::parse(interner, source) {
                Ok(ast) => ast,
                Err(e) => {
                    return AnalyzeResult {
                        ok: false,
                        errors: vec![EngineError::from_parse(&e)],
                        context_keys: vec![],
                        tail_type: TypeDesc::Unsupported { raw: String::new() },
                    };
                }
            };
            let (module, _hints, errs) =
                acvus_mir::compile_analysis_partial(interner, &ast, registry);
            let keys = extract_context_keys_with_types(interner, &module, known)
                .into_iter()
                .filter(|k| partial.is_user_key(&interner.intern(&k.name)))
                .collect();
            let errors = EngineError::from_mir_errors(&errs, interner);
            AnalyzeResult {
                ok: errors.is_empty(),
                errors,
                context_keys: keys,
                tail_type: TypeDesc::Primitive { name: "string".into() },
            }
        }
        Mode::Script => {
            let script = match acvus_ast::parse_script(interner, source) {
                Ok(s) => s,
                Err(e) => {
                    return AnalyzeResult {
                        ok: false,
                        errors: vec![EngineError::from_parse(&e)],
                        context_keys: vec![],
                        tail_type: TypeDesc::Unsupported { raw: String::new() },
                    };
                }
            };
            let (module, _hints, tail_ty, errs) =
                acvus_mir::compile_script_analysis_with_tail_partial(
                    interner,
                    &script,
                    registry,
                    expected_tail,
                );
            let keys = extract_context_keys_with_types(interner, &module, known)
                .into_iter()
                .filter(|k| partial.is_user_key(&interner.intern(&k.name)))
                .collect();
            let errors = EngineError::from_mir_errors(&errs, interner);
            AnalyzeResult {
                ok: errors.is_empty(),
                errors,
                context_keys: keys,
                tail_type: ty_to_desc(interner, &tail_ty),
            }
        }
    }
}

#[wasm_bindgen]
pub fn analyze(options: Ts<AnalyzeOptions>) -> Result<Ts<AnalyzeResult>, JsError> {
    let options = options.to_rust()?;
    let interner = Interner::new();
    let user_types = convert_context_types(&interner, &options.context_types);
    let registry = match build_registry(&interner, user_types) {
        Ok(r) => r,
        Err(e) => {
            let key_name = interner.resolve(e.key);
            return Ok(AnalyzeResult {
                ok: false,
                errors: vec![error::EngineError::general(
                    error::ErrorCategory::Type,
                    format!("context type conflict: @{key_name} exists in both {} and {} tier", e.tier_a, e.tier_b),
                )],
                context_keys: vec![],
                tail_type: TypeDesc::Unsupported { raw: "error".into() },
            }.into_ts()?);
        }
    };
    let full_reg = registry.to_full();
    let expected_tail = options.expected_tail.as_ref().map(|d| desc_to_ty(&interner, d));

    let mut known = FxHashMap::default();
    for (name, script) in &options.known_values {
        if let Some(kv) = try_extract_known(&interner, script, &full_reg) {
            known.insert(interner.intern(name), kv);
        }
    }

    Ok(do_analyze(&interner, &options.source, &options.mode, &full_reg, expected_tail.as_ref(), &known, &registry).into_ts()?)
}

fn do_typecheck(
    interner: &Interner,
    source: &str,
    mode: &Mode,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> CheckResult {
    use error::EngineError;

    let result = match mode {
        Mode::Template => {
            let ast = match acvus_ast::parse(interner, source) {
                Ok(ast) => ast,
                Err(e) => {
                    return CheckResult {
                        ok: false,
                        errors: vec![EngineError::from_parse(&e)],
                    };
                }
            };
            acvus_mir::compile(interner, &ast, registry).map(|_| ())
        }
        Mode::Script => {
            let script = match acvus_ast::parse_script(interner, source) {
                Ok(s) => s,
                Err(e) => {
                    return CheckResult {
                        ok: false,
                        errors: vec![EngineError::from_parse(&e)],
                    };
                }
            };
            acvus_mir::compile_script_with_hint(
                interner,
                &script,
                registry,
                expected_tail,
            )
            .map(|_| ())
        }
    };

    match result {
        Ok(()) => CheckResult {
            ok: true,
            errors: vec![],
        },
        Err(errs) => CheckResult {
            ok: false,
            errors: EngineError::from_mir_errors(&errs, interner),
        },
    }
}

#[wasm_bindgen]
pub fn typecheck(options: Ts<TypecheckOptions>) -> Result<Ts<CheckResult>, JsError> {
    let options = options.to_rust()?;
    let interner = Interner::new();
    let user_types = convert_context_types(&interner, &options.context_types);
    let registry = match build_registry(&interner, user_types) {
        Ok(r) => r,
        Err(e) => {
            let key_name = interner.resolve(e.key);
            return Ok(CheckResult {
                ok: false,
                errors: vec![error::EngineError::general(
                    error::ErrorCategory::Type,
                    format!("context type conflict: @{key_name} exists in both {} and {} tier", e.tier_a, e.tier_b),
                )],
            }.into_ts()?);
        }
    };
    let full_reg = registry.to_full();
    let expected_tail = options.expected_tail.as_ref().map(|d| desc_to_ty(&interner, d));
    Ok(do_typecheck(
        &interner,
        &options.source,
        &options.mode,
        &full_reg,
        expected_tail.as_ref(),
    ).into_ts()?)
}

/// Typecheck all nodes at once using the full orchestration pipeline.
#[wasm_bindgen]
pub fn typecheck_nodes(
    options: Ts<TypecheckNodesOptions>,
) -> Result<Ts<TypecheckNodesResult>, JsError> {
    use error::{EngineError, ErrorCategory};

    let options = options.to_rust()?;
    let web_nodes = options.nodes;
    let raw_types = options.injected_types;

    let interner = Interner::new();
    let user_types = convert_context_types(&interner, &raw_types);
    let registry = match build_registry(&interner, user_types) {
        Ok(r) => r,
        Err(e) => {
            let key_name = interner.resolve(e.key);
            return Ok(TypecheckNodesResult::fail(vec![
                EngineError::general(ErrorCategory::Type,
                    format!("context type conflict: @{key_name} exists in both {} and {} tier", e.tier_a, e.tier_b)),
            ]).into_ts()?);
        }
    };

    let specs: Vec<NodeSpec> = match web_nodes
        .iter()
        .map(|w| w.into_node(&interner))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(specs) => specs,
        Err(e) => {
            return Ok(TypecheckNodesResult::fail(vec![
                EngineError::general(ErrorCategory::Parse, e),
            ]).into_ts()?);
        }
    };

    // 1. Compute external context env (context_types + per-node locals)
    let env = match acvus_orchestration::compute_external_context_env(
        &interner,
        &specs,
        registry,
    ) {
        Ok(env) => env,
        Err(errs) => {
            return Ok(TypecheckNodesResult::fail(
                EngineError::from_orch_errors(&errs, &interner),
            ).into_ts()?);
        }
    };

    let context_types_str: FxHashMap<String, TypeDesc> = env
        .registry.visible()
        .iter()
        .map(|(k, v)| (interner.resolve(*k).to_string(), ty_to_desc(&interner, v)))
        .collect();

    let node_locals_str: FxHashMap<String, NodeLocalTypes> = env
        .node_locals
        .iter()
        .map(|(name, locals)| {
            (
                interner.resolve(*name).to_string(),
                NodeLocalTypes {
                    raw: ty_to_desc(&interner, &locals.raw_ty),
                    self_ty: ty_to_desc(&interner, &locals.self_ty),
                },
            )
        })
        .collect();

    // 2. Per-node, per-field typecheck
    let full_reg = env.registry.to_full();
    let mut node_errors: FxHashMap<String, NodeErrors> = FxHashMap::default();
    for spec in &specs {
        let Some(locals) = env.node_locals.get(&spec.name) else {
            continue;
        };
        let mut errors = NodeErrors::default();
        let locals_ref = Some(locals);

        // Body context: fn_params + @self (if initial_value exists)
        let node_reg = match spec.build_node_context(&interner, &full_reg, acvus_orchestration::ContextScope::Body, locals_ref) {
            Ok(r) => r,
            Err(e) => {
                errors.env = vec![EngineError::general(
                    ErrorCategory::Type,
                    format!("context type conflict: @{} in {} and {}", interner.resolve(e.key), e.tier_a, e.tier_b),
                )];
                node_errors.insert(interner.resolve(spec.name).to_string(), errors);
                continue;
            }
        };

        // initial_value: no @self, no @raw
        if let Some(init_src) = spec.strategy.initial_value {
            let hint = match &locals.self_ty {
                Ty::Error => None,
                ty => Some(ty),
            };
            let init_reg = spec.build_node_context(&interner, &full_reg, acvus_orchestration::ContextScope::InitialValue, locals_ref)
                .expect("InitialValue scope should not conflict");
            errors.initial_value = check_script(
                &interner,
                interner.resolve(init_src),
                &init_reg,
                hint,
            );
        }

        // if_modified key: no @self context
        match &spec.strategy.execution {
            acvus_orchestration::Execution::IfModified { key } => {
                let no_self_reg = spec.build_node_context(&interner, &full_reg, acvus_orchestration::ContextScope::InitialValue, locals_ref)
                    .expect("InitialValue scope should not conflict");
                errors.if_modified_key = check_script(
                    &interner,
                    interner.resolve(*key),
                    &no_self_reg,
                    None,
                );
            }
            _ => {}
        }

        // persistency: bind script (Deque/Diff) uses @self + @raw + all context
        match &spec.strategy.persistency {
            acvus_orchestration::Persistency::Deque { bind } | acvus_orchestration::Persistency::Diff { bind } => {
                let bind_reg = spec.build_node_context(&interner, &full_reg, acvus_orchestration::ContextScope::Bind, locals_ref)
                    .expect("Bind scope should not conflict with reserved @self/@raw");
                let hint = if locals.self_ty != Ty::Error {
                    Some(&locals.self_ty)
                } else {
                    None
                };
                errors.bind = check_script(
                    &interner,
                    interner.resolve(*bind),
                    &bind_reg,
                    hint,
                );
            }
            _ => {}
        }

        // assert: Bind scope (@self + @raw), expected tail = Bool
        if let Some(assert_src) = spec.strategy.assert {
            let assert_reg = spec.build_node_context(&interner, &full_reg, acvus_orchestration::ContextScope::Bind, locals_ref)
                .expect("Bind scope should not conflict with reserved @self/@raw");
            errors.assert = check_script(
                &interner,
                interner.resolve(assert_src),
                &assert_reg,
                Some(&Ty::Bool),
            );
        }

        // messages (LLM only)
        let messages: &[acvus_orchestration::MessageSpec] = match &spec.kind {
            acvus_orchestration::NodeKind::Llm(llm) => &llm.messages,
            _ => &[],
        };

        for (mi, msg) in messages.iter().enumerate() {
            let errs = match msg {
                acvus_orchestration::MessageSpec::Block { source, .. } => {
                    check_template(&interner, source, &node_reg)
                }
                acvus_orchestration::MessageSpec::Iterator { key, .. } => {
                    check_script(
                        &interner,
                        interner.resolve(*key),
                        &node_reg,
                        None,
                    )
                }
            };
            if !errs.is_empty() {
                errors.messages.insert(mi.to_string(), errs);
            }
        }

        // Expr source typecheck
        if let acvus_orchestration::NodeKind::Expr(expr_spec) = &spec.kind {
            errors.expr_source = check_script(
                &interner,
                &expr_spec.source,
                &node_reg,
                None,
            );
        }

        if !errors.is_empty() {
            node_errors.insert(interner.resolve(spec.name).to_string(), errors);
        }
    }

    Ok(TypecheckNodesResult {
        env_errors: vec![],
        context_types: context_types_str,
        node_locals: node_locals_str,
        node_errors,
    }.into_ts()?)
}

/// Helper: typecheck a script, returning errors if any.
fn check_script(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Vec<error::EngineError> {
    use error::EngineError;

    if source.trim().is_empty() {
        return vec![];
    }
    let script = match acvus_ast::parse_script(interner, source) {
        Ok(s) => s,
        Err(e) => return vec![EngineError::from_parse(&e)],
    };
    match acvus_mir::compile_script_with_hint(interner, &script, registry, expected_tail) {
        Ok(_) => vec![],
        Err(errs) => EngineError::from_mir_errors(&errs, interner),
    }
}

/// Helper: typecheck a template, returning errors if any.
fn check_template(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> Vec<error::EngineError> {
    use error::EngineError;

    if source.trim().is_empty() {
        return vec![];
    }
    let ast = match acvus_ast::parse(interner, source) {
        Ok(a) => a,
        Err(e) => return vec![EngineError::from_parse(&e)],
    };
    match acvus_mir::compile(interner, &ast, registry) {
        Ok(_) => vec![],
        Err(errs) => EngineError::from_mir_errors(&errs, interner),
    }
}

#[wasm_bindgen]
pub async fn evaluate(options: Ts<EvaluateOptions>) -> Result<JsValue, JsError> {
    let options = options.to_rust()?;
    use acvus_interpreter::{Interpreter, Value};
    use error::EngineError;

    let interner = Interner::new();

    // Infer context types from values
    let mut user_types: FxHashMap<Astr, Ty> = FxHashMap::default();
    for (k, v) in &options.context {
        user_types.insert(interner.intern(k), jcv_to_ty(&interner, v));
    }

    let registry = match build_registry(&interner, user_types) {
        Ok(r) => r,
        Err(e) => {
            let key_name = interner.resolve(e.key);
            return Ok(EvaluateResult {
                ok: false,
                errors: vec![EngineError::general(
                    error::ErrorCategory::Type,
                    format!("context type conflict: @{key_name} exists in both {} and {} tier", e.tier_a, e.tier_b),
                )],
                value: None,
            }.into_ts()?.js_value());
        }
    };
    let full_reg = registry.to_full();

    let module = match &options.mode {
        Mode::Template => {
            let ast = match acvus_ast::parse(&interner, &options.source) {
                Ok(ast) => ast,
                Err(e) => {
                    return Ok(EvaluateResult {
                        ok: false,
                        errors: vec![EngineError::from_parse(&e)],
                        value: None,
                    }.into_ts()?.js_value());
                }
            };
            match acvus_mir::compile_analysis(&interner, &ast, &full_reg) {
                Ok((module, _)) => module,
                Err(errs) => {
                    return Ok(EvaluateResult {
                        ok: false,
                        errors: EngineError::from_mir_errors(&errs, &interner),
                        value: None,
                    }.into_ts()?.js_value());
                }
            }
        }
        Mode::Script => {
            let script = match acvus_ast::parse_script(&interner, &options.source) {
                Ok(s) => s,
                Err(e) => {
                    return Ok(EvaluateResult {
                        ok: false,
                        errors: vec![EngineError::from_parse(&e)],
                        value: None,
                    }.into_ts()?.js_value());
                }
            };
            match acvus_mir::compile_script_analysis(&interner, &script, &full_reg) {
                Ok((module, _, _)) => module,
                Err(errs) => {
                    return Ok(EvaluateResult {
                        ok: false,
                        errors: EngineError::from_mir_errors(&errs, &interner),
                        value: None,
                    }.into_ts()?.js_value());
                }
            }
        }
    };

    // Build context values
    let ctx: FxHashMap<Astr, Value> = options.context
        .into_iter()
        .map(|(k, v)| {
            let cv: acvus_interpreter::ConcreteValue = v.into();
            (interner.intern(&k), Value::from_concrete(&cv, &interner))
        })
        .collect();

    let interp = Interpreter::new(&interner, module);
    let emits = interp.execute_with_context(ctx).await;

    let result_value = match &options.mode {
        Mode::Template => {
            let mut output = String::new();
            for v in &emits {
                match v {
                    Value::Pure(acvus_interpreter::PureValue::String(s)) => output.push_str(s),
                    other => panic!("template emit: expected String, got {other:?}"),
                }
            }
            JsConcreteValue::String { v: output }
        }
        Mode::Script => {
            assert!(emits.len() <= 1, "script emitted {} values, expected at most 1", emits.len());
            match emits.into_iter().next() {
                Some(v) => v.to_concrete(&interner).into(),
                None => JsConcreteValue::Unit,
            }
        }
    };

    Ok(EvaluateResult {
        ok: true,
        errors: vec![],
        value: Some(result_value),
    }.into_ts()?.js_value())
}

fn jcv_to_ty(interner: &Interner, v: &JsConcreteValue) -> Ty {
    match v {
        JsConcreteValue::Int { .. } => Ty::Int,
        JsConcreteValue::Float { .. } => Ty::Float,
        JsConcreteValue::String { .. } => Ty::String,
        JsConcreteValue::Bool { .. } => Ty::Bool,
        JsConcreteValue::Unit => Ty::Unit,
        JsConcreteValue::Range { .. } => Ty::Range,
        JsConcreteValue::Byte { .. } => Ty::Byte,
        JsConcreteValue::List { items } => {
            let elem_ty = items
                .first()
                .map(|i| jcv_to_ty(interner, i))
                .unwrap_or(Ty::Infer);
            Ty::List(Box::new(elem_ty))
        }
        JsConcreteValue::Object { fields } => {
            let ty_fields: FxHashMap<Astr, Ty> = fields
                .iter()
                .map(|(k, v)| (interner.intern(k), jcv_to_ty(interner, v)))
                .collect();
            Ty::Object(ty_fields)
        }
        JsConcreteValue::Tuple { items } => {
            Ty::Tuple(items.iter().map(|i| jcv_to_ty(interner, i)).collect())
        }
        JsConcreteValue::Variant { tag, payload } => {
            let mut variants = FxHashMap::default();
            variants.insert(
                interner.intern(tag),
                payload.as_ref().map(|p| Box::new(jcv_to_ty(interner, p))),
            );
            Ty::Enum { name: interner.intern(tag), variants }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_interner() -> Interner {
        Interner::new()
    }

    fn do_analyze_test(
        interner: &Interner,
        source: &str,
        mode: &Mode,
        ctx: &FxHashMap<Astr, Ty>,
    ) -> AnalyzeResult {
        let registry = PartialContextTypeRegistry::user_only(ctx.clone());
        let full_reg = registry.to_full();
        do_analyze(interner, source, mode, &full_reg, None, &FxHashMap::default(), &registry)
    }

    /// Helper: serialize a TypeDesc to a JSON string for comparison.
    fn desc_json(desc: &TypeDesc) -> String {
        serde_json::to_string(desc).unwrap()
    }

    #[test]
    fn test_ty_to_desc_primitives() {
        let interner = test_interner();
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Int)), r#"{"kind":"primitive","name":"int"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Float)), r#"{"kind":"primitive","name":"float"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::String)), r#"{"kind":"primitive","name":"string"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Bool)), r#"{"kind":"primitive","name":"bool"}"#);
    }

    #[test]
    fn test_ty_to_desc_unsupported() {
        let interner = test_interner();
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Var(acvus_mir::ty::TyVar(0)))), r#"{"kind":"unsupported","raw":"?"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Infer)), r#"{"kind":"unsupported","raw":"?"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Error)), r#"{"kind":"unsupported","raw":"?"}"#);
    }

    #[test]
    fn test_ty_to_desc_list() {
        let interner = test_interner();
        let desc = ty_to_desc(&interner, &Ty::List(Box::new(Ty::String)));
        assert_eq!(desc_json(&desc), r#"{"kind":"list","elem":{"kind":"primitive","name":"string"}}"#);
    }

    #[test]
    fn test_desc_to_ty_roundtrip() {
        let interner = test_interner();
        // Primitive roundtrip
        let desc = ty_to_desc(&interner, &Ty::Int);
        assert_eq!(desc_to_ty(&interner, &desc), Ty::Int);

        // List roundtrip
        let desc = ty_to_desc(&interner, &Ty::List(Box::new(Ty::String)));
        assert_eq!(desc_to_ty(&interner, &desc), Ty::List(Box::new(Ty::String)));

        // Option roundtrip
        let desc = ty_to_desc(&interner, &Ty::Option(Box::new(Ty::Bool)));
        assert_eq!(desc_to_ty(&interner, &desc), Ty::Option(Box::new(Ty::Bool)));

        // Object roundtrip
        let mut fields = FxHashMap::default();
        fields.insert(interner.intern("name"), Ty::String);
        fields.insert(interner.intern("age"), Ty::Int);
        let desc = ty_to_desc(&interner, &Ty::Object(fields.clone()));
        assert_eq!(desc_to_ty(&interner, &desc), Ty::Object(fields));

        // Enum roundtrip (with payload types)
        let mut variants = FxHashMap::default();
        variants.insert(interner.intern("Ok"), Some(Box::new(Ty::Int)));
        variants.insert(interner.intern("Err"), Some(Box::new(Ty::String)));
        variants.insert(interner.intern("None"), None);
        let enum_ty = Ty::Enum { name: interner.intern("Result"), variants };
        let desc = ty_to_desc(&interner, &enum_ty);
        let roundtripped = desc_to_ty(&interner, &desc);
        match &roundtripped {
            Ty::Enum { name, variants } => {
                assert_eq!(interner.resolve(*name), "Result");
                assert_eq!(variants.len(), 3);
                assert_eq!(*variants.get(&interner.intern("Ok")).unwrap(), Some(Box::new(Ty::Int)));
                assert_eq!(*variants.get(&interner.intern("Err")).unwrap(), Some(Box::new(Ty::String)));
                assert_eq!(*variants.get(&interner.intern("None")).unwrap(), None);
            }
            _ => panic!("expected Enum, got {roundtripped:?}"),
        }
    }

    #[test]
    fn test_analyze_script_context_types() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze_test(&interner, "@x + 1", &Mode::Script, &ctx);
        if !result.ok {
            eprintln!("errors: {:?}", result.errors);
        }
        assert!(result.ok);
        assert_eq!(result.context_keys.len(), 1);
        assert_eq!(result.context_keys[0].name, "x");
        assert_eq!(desc_json(&result.context_keys[0].ty), r#"{"kind":"primitive","name":"int"}"#);
        assert_eq!(desc_json(&result.tail_type), r#"{"kind":"primitive","name":"int"}"#);
    }

    #[test]
    fn test_analyze_with_provided_context_types() {
        let interner = test_interner();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("x"), Ty::Int);
        let result = do_analyze_test(&interner, "@x + 1", &Mode::Script, &ctx);
        assert!(result.ok);
        assert_eq!(desc_json(&result.context_keys[0].ty), r#"{"kind":"primitive","name":"int"}"#);
    }

    #[test]
    fn test_analyze_template() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze_test(&interner, "hello {{ @name }}", &Mode::Template, &ctx);
        assert!(result.ok);
        assert_eq!(result.context_keys.len(), 1);
        assert_eq!(result.context_keys[0].name, "name");
        assert_eq!(desc_json(&result.tail_type), r#"{"kind":"primitive","name":"string"}"#);
    }

    #[test]
    fn test_tail_type_mismatch() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let registry = PartialContextTypeRegistry::system_only(ctx);
        let full_reg = registry.to_full();
        let result = do_analyze(&interner, "\"hello\"", &Mode::Script, &full_reg, Some(&Ty::Int), &FxHashMap::default(), &registry);
        assert!(!result.ok, "should fail: String vs Int");
    }

    #[test]
    fn test_tail_type_match() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let registry = PartialContextTypeRegistry::system_only(ctx);
        let full_reg = registry.to_full();
        let result = do_analyze(&interner, "1 + 2", &Mode::Script, &full_reg, Some(&Ty::Int), &FxHashMap::default(), &registry);
        assert!(result.ok, "should succeed: Int vs Int");
    }

    #[test]
    fn test_analyze_unresolved_type() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze_test(&interner, "@x", &Mode::Script, &ctx);
        if !result.ok {
            eprintln!("errors: {:?}", result.errors);
        }
        assert!(result.ok);
        assert_eq!(desc_json(&result.context_keys[0].ty), r#"{"kind":"unsupported","raw":"?"}"#);
    }

    #[test]
    fn test_analyze_enum_from_match_arms() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        // Template with match block: two arms, one with payload.
        let template = "{{ Status::Active(msg) = @status }}{{ msg }}{{ Status::Inactive = }}inactive{{/}}";
        let result = do_analyze_test(&interner, template, &Mode::Template, &ctx);
        if !result.ok {
            eprintln!("errors: {:?}", result.errors);
        }
        assert!(result.ok, "template should compile");
        assert_eq!(result.context_keys.len(), 1, "should have one context key");
        assert_eq!(result.context_keys[0].name, "status");
        let ty_json = desc_json(&result.context_keys[0].ty);
        eprintln!("enum type JSON: {ty_json}");
        // Should be an Enum with two variants: Active(String) and Inactive
        assert!(ty_json.contains(r#""kind":"enum""#), "should be enum type, got: {ty_json}");
        assert!(ty_json.contains(r#""hasPayload":true"#), "Active should have payload, got: {ty_json}");
        assert!(ty_json.contains(r#""hasPayload":false"#), "Inactive should have no payload, got: {ty_json}");
    }

    #[test]
    fn test_analyze_discovers_keys_despite_type_error() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        // Template with a type error: `msg + 1` is String + Int mismatch.
        // Context key @status should still be discovered.
        let template = "{{ Status::Active(msg) = @status }}{{ msg + 1 }}{{ Status::Inactive = }}inactive{{/}}";
        let result = do_analyze_test(&interner, template, &Mode::Template, &ctx);
        assert!(!result.ok, "should have type errors");
        assert!(!result.errors.is_empty());
        // Despite errors, context_keys should still be discovered
        assert!(!result.context_keys.is_empty(), "should discover context keys despite type error");
        assert_eq!(result.context_keys[0].name, "status");
    }

    #[test]
    fn test_analyze_script_discovers_keys_despite_type_error() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        // Script with type error: @x is inferred as Int, but "hello" + @x is String + Int mismatch.
        let result = do_analyze_test(&interner, "@x + 1; \"hello\" + @x", &Mode::Script, &ctx);
        assert!(!result.ok, "should have type errors");
        // Despite errors, context key @x should still be discovered
        assert!(!result.context_keys.is_empty(), "should discover context keys despite type error");
        assert_eq!(result.context_keys[0].name, "x");
    }
}
