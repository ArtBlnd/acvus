mod convert;
mod session;

#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn init() {
    console_error_panic_hook::set_once();
}

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;
use acvus_orchestration::NodeSpec;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;

/// Serialize any Serialize type to JsValue via JSON.
fn to_js<T: Serialize>(value: &T) -> JsValue {
    let json_str = serde_json::to_string(value).expect("Serialize impl should not fail");
    js_sys::JSON::parse(&json_str).expect("serde_json output is always valid JSON")
}

fn default_registry(interner: &Interner) -> ExternRegistry {
    let mut registry = ExternRegistry::new();
    let mut fn_reg = acvus_interpreter::ExternFnRegistry::new(interner);
    let regex_mod = acvus_ext::regex_module(interner, &mut fn_reg);
    registry.register(&regex_mod);
    registry
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub(crate) enum TypeDesc {
    #[serde(rename = "primitive")]
    Primitive { name: String },
    #[serde(rename = "option")]
    Option { inner: Box<TypeDesc> },
    #[serde(rename = "object")]
    Object { fields: Vec<TypeDescField> },
    #[serde(rename = "list")]
    List { elem: Box<TypeDesc> },
    #[serde(rename = "enum")]
    Enum {
        name: String,
        variants: Vec<TypeDescVariant>,
    },
    #[serde(rename = "unsupported")]
    Unsupported { raw: String },
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct TypeDescField {
    name: String,
    #[serde(rename = "type")]
    ty: TypeDesc,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct TypeDescVariant {
    tag: String,
    #[serde(rename = "hasPayload")]
    has_payload: bool,
}

fn ty_to_desc(interner: &Interner, ty: &Ty) -> TypeDesc {
    match ty {
        Ty::Int => TypeDesc::Primitive { name: "Int".into() },
        Ty::Float => TypeDesc::Primitive { name: "Float".into() },
        Ty::String => TypeDesc::Primitive { name: "String".into() },
        Ty::Bool => TypeDesc::Primitive { name: "Bool".into() },
        Ty::Unit => TypeDesc::Unsupported { raw: "Unit".into() },
        Ty::Range => TypeDesc::Unsupported { raw: "Range".into() },
        Ty::Byte => TypeDesc::Unsupported { raw: "Byte".into() },
        Ty::Option(inner) => TypeDesc::Option {
            inner: Box::new(ty_to_desc(interner, inner)),
        },
        Ty::List(inner) => TypeDesc::List {
            elem: Box::new(ty_to_desc(interner, inner)),
        },
        Ty::Object(fields) => {
            let mut desc_fields: Vec<TypeDescField> = fields
                .iter()
                .map(|(k, v)| TypeDescField {
                    name: interner.resolve(*k).to_string(),
                    ty: ty_to_desc(interner, v),
                })
                .collect();
            desc_fields.sort_by(|a, b| a.name.cmp(&b.name));
            TypeDesc::Object { fields: desc_fields }
        }
        Ty::Enum { name, variants } => {
            let mut desc_variants: Vec<TypeDescVariant> = variants
                .iter()
                .map(|(tag, payload)| TypeDescVariant {
                    tag: interner.resolve(*tag).to_string(),
                    has_payload: payload.is_some(),
                })
                .collect();
            desc_variants.sort_by(|a, b| a.tag.cmp(&b.tag));
            TypeDesc::Enum {
                name: interner.resolve(*name).to_string(),
                variants: desc_variants,
            }
        }
        Ty::Var(_) | Ty::Infer | Ty::Error => TypeDesc::Unsupported { raw: "?".into() },
        Ty::Fn { .. } => TypeDesc::Unsupported { raw: "Fn".into() },
        Ty::Opaque(_) => TypeDesc::Unsupported { raw: "Opaque".into() },
        Ty::Tuple(_) => TypeDesc::Unsupported { raw: "Tuple".into() },
    }
}

pub(crate) fn desc_to_ty(interner: &Interner, desc: &TypeDesc) -> Ty {
    match desc {
        TypeDesc::Primitive { name } => match name.as_str() {
            "Int" => Ty::Int,
            "Float" => Ty::Float,
            "String" => Ty::String,
            "Bool" => Ty::Bool,
            _ => Ty::Infer,
        },
        TypeDesc::Option { inner } => Ty::Option(Box::new(desc_to_ty(interner, inner))),
        TypeDesc::List { elem } => Ty::List(Box::new(desc_to_ty(interner, elem))),
        TypeDesc::Object { fields } => {
            let ty_fields: FxHashMap<Astr, Ty> = fields
                .iter()
                .map(|f| (interner.intern(&f.name), desc_to_ty(interner, &f.ty)))
                .collect();
            Ty::Object(ty_fields)
        }
        TypeDesc::Enum { name, variants } => {
            let ty_variants: FxHashMap<Astr, Option<Box<Ty>>> = variants
                .iter()
                .map(|v| {
                    let tag = interner.intern(&v.tag);
                    // TypeDescVariant only tracks has_payload; without payload type info,
                    // we cannot reconstruct the payload Ty, so use None for no-payload
                    // and Infer for has-payload.
                    let payload = if v.has_payload {
                        Some(Box::new(Ty::Infer))
                    } else {
                        None
                    };
                    (tag, payload)
                })
                .collect();
            Ty::Enum {
                name: interner.intern(name),
                variants: ty_variants,
            }
        }
        TypeDesc::Unsupported { .. } => Ty::Infer,
    }
}

fn extract_context_keys_with_types(
    interner: &Interner,
    module: &MirModule,
) -> Vec<ContextKey> {
    let mut seen = FxHashMap::<Astr, Ty>::default();

    let mut collect = |insts: &[acvus_mir::ir::Inst],
                       val_types: &FxHashMap<acvus_mir::ir::ValueId, Ty>| {
        for inst in insts {
            if let InstKind::ContextLoad { dst, name, .. } = &inst.kind {
                if seen.contains_key(name) {
                    continue;
                }
                let ty = val_types.get(dst).cloned().unwrap_or(Ty::Infer);
                seen.insert(*name, ty);
            }
        }
    };

    collect(&module.main.insts, &module.main.val_types);
    for body in module.closures.values() {
        collect(&body.body.insts, &body.body.val_types);
    }

    let mut keys: Vec<_> = seen
        .into_iter()
        .map(|(name, ty)| ContextKey {
            name: interner.resolve(name).to_string(),
            ty: ty_to_desc(interner, &ty),
        })
        .collect();
    keys.sort_by(|a, b| a.name.cmp(&b.name));
    keys
}

/// Parse a JSON map of `name -> TypeDesc` into `HashMap<Astr, Ty>`.
fn parse_context_types(interner: &Interner, json: &str) -> Result<FxHashMap<Astr, Ty>, String> {
    let raw: FxHashMap<String, TypeDesc> = serde_json::from_str(json)
        .map_err(|e| format!("failed to parse context types JSON: {e}"))?;
    let mut result = FxHashMap::default();
    for (k, v) in raw {
        result.insert(interner.intern(&k), desc_to_ty(interner, &v));
    }
    Ok(result)
}

#[derive(Serialize)]
struct ContextKey {
    name: String,
    #[serde(rename = "type")]
    ty: TypeDesc,
}

#[derive(Serialize)]
struct AnalyzeResult {
    ok: bool,
    errors: Vec<String>,
    context_keys: Vec<ContextKey>,
    tail_type: TypeDesc,
}

fn do_analyze(
    interner: &Interner,
    source: &str,
    mode: &str,
    context_types: &FxHashMap<Astr, Ty>,
    expected_tail: Option<&Ty>,
) -> AnalyzeResult {
    let registry = default_registry(interner);

    match mode {
        "template" => {
            let ast = match acvus_ast::parse(interner, source) {
                Ok(ast) => ast,
                Err(e) => {
                    return AnalyzeResult {
                        ok: false,
                        errors: vec![format!("{e}")],
                        context_keys: vec![],
                        tail_type: TypeDesc::Unsupported { raw: String::new() },
                    };
                }
            };
            match acvus_mir::compile_analysis(interner, &ast, context_types, &registry)
            {
                Ok((module, _hints)) => {
                    let keys = extract_context_keys_with_types(interner, &module);
                    AnalyzeResult {
                        ok: true,
                        errors: vec![],
                        context_keys: keys,
                        tail_type: TypeDesc::Primitive { name: "String".into() },
                    }
                }
                Err(errs) => AnalyzeResult {
                    ok: false,
                    errors: errs
                        .into_iter()
                        .map(|e| e.display(interner).to_string())
                        .collect(),
                    context_keys: vec![],
                    tail_type: TypeDesc::Unsupported { raw: String::new() },
                },
            }
        }
        _ => {
            let script = match acvus_ast::parse_script(interner, source) {
                Ok(s) => s,
                Err(e) => {
                    return AnalyzeResult {
                        ok: false,
                        errors: vec![format!("{e}")],
                        context_keys: vec![],
                        tail_type: TypeDesc::Unsupported { raw: String::new() },
                    };
                }
            };
            match acvus_mir::compile_script_analysis_with_tail(
                interner,
                &script,
                context_types,
                &registry,
                expected_tail,
            ) {
                Ok((module, _hints, tail_ty)) => {
                    let keys = extract_context_keys_with_types(interner, &module);
                    AnalyzeResult {
                        ok: true,
                        errors: vec![],
                        context_keys: keys,
                        tail_type: ty_to_desc(interner, &tail_ty),
                    }
                }
                Err(errs) => AnalyzeResult {
                    ok: false,
                    errors: errs
                        .into_iter()
                        .map(|e| e.display(interner).to_string())
                        .collect(),
                    context_keys: vec![],
                    tail_type: TypeDesc::Unsupported { raw: String::new() },
                },
            }
        }
    }
}

/// Analyze a script or template. Returns JSON:
/// `{ ok, errors, context_keys: [{name, type}], tail_type }`
#[wasm_bindgen]
pub fn analyze(source: &str, mode: &str) -> JsValue {
    let interner = Interner::new();
    let context_types = FxHashMap::default();
    let result = do_analyze(&interner, source, mode, &context_types, None);
    to_js(&result)
}

/// Analyze with user-provided context types. `context_types_json` is a JSON
/// object mapping names to type strings, e.g. `{"name": "String", "age": "Int"}`.
#[wasm_bindgen]
pub fn analyze_with_types(source: &str, mode: &str, context_types_json: &str) -> JsValue {
    let interner = Interner::new();
    let context_types = match parse_context_types(&interner, context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let result = AnalyzeResult {
                ok: false,
                errors: vec![e],
                context_keys: vec![],
                tail_type: TypeDesc::Unsupported { raw: String::new() },
            };
            return to_js(&result);
        }
    };
    let result = do_analyze(&interner, source, mode, &context_types, None);
    to_js(&result)
}

/// Analyze with user-provided context types and expected tail type.
#[wasm_bindgen]
pub fn analyze_with_tail(
    source: &str,
    mode: &str,
    context_types_json: &str,
    expected_tail_type: &str,
) -> JsValue {
    let interner = Interner::new();
    let context_types = match parse_context_types(&interner, context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let result = AnalyzeResult {
                ok: false,
                errors: vec![e],
                context_keys: vec![],
                tail_type: TypeDesc::Unsupported { raw: String::new() },
            };
            return to_js(&result);
        }
    };
    let expected_tail: Option<Ty> = if expected_tail_type.is_empty() {
        None
    } else {
        match serde_json::from_str::<TypeDesc>(expected_tail_type) {
            Ok(desc) => Some(desc_to_ty(&interner, &desc)),
            Err(e) => {
                let result = AnalyzeResult {
                    ok: false,
                    errors: vec![format!("failed to parse tail type: {e}")],
                    context_keys: vec![],
                    tail_type: TypeDesc::Unsupported { raw: String::new() },
                };
                return to_js(&result);
            }
        }
    };
    let result = do_analyze(
        &interner,
        source,
        mode,
        &context_types,
        expected_tail.as_ref(),
    );
    to_js(&result)
}

#[derive(Serialize)]
struct CheckResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

fn do_typecheck(
    interner: &Interner,
    source: &str,
    mode: &str,
    context_types: &FxHashMap<Astr, Ty>,
    expected_tail: Option<&Ty>,
) -> CheckResult {
    let registry = default_registry(interner);

    let result = match mode {
        "template" => {
            let ast = match acvus_ast::parse(interner, source) {
                Ok(ast) => ast,
                Err(e) => {
                    return CheckResult {
                        ok: false,
                        message: Some(format!("{e}")),
                    };
                }
            };
            acvus_mir::compile(interner, &ast, context_types, &registry).map(|_| ())
        }
        _ => {
            let script = match acvus_ast::parse_script(interner, source) {
                Ok(s) => s,
                Err(e) => {
                    return CheckResult {
                        ok: false,
                        message: Some(format!("{e}")),
                    };
                }
            };
            acvus_mir::compile_script_with_hint(
                interner,
                &script,
                context_types,
                &registry,
                expected_tail,
            )
            .map(|_| ())
        }
    };

    match result {
        Ok(()) => CheckResult {
            ok: true,
            message: None,
        },
        Err(errs) => CheckResult {
            ok: false,
            message: Some(
                errs.into_iter()
                    .map(|e| e.display(interner).to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        },
    }
}

/// Quick type-check (hard mode). Returns JSON: `{ ok, message? }`
#[wasm_bindgen]
pub fn typecheck(source: &str, mode: &str) -> JsValue {
    let interner = Interner::new();
    let context_types = FxHashMap::default();
    let check = do_typecheck(&interner, source, mode, &context_types, None);
    to_js(&check)
}

/// Quick type-check with user-provided context types (hard mode).
#[wasm_bindgen]
pub fn typecheck_with_types(source: &str, mode: &str, context_types_json: &str) -> JsValue {
    let interner = Interner::new();
    let context_types = match parse_context_types(&interner, context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let check = CheckResult {
                ok: false,
                message: Some(e),
            };
            return to_js(&check);
        }
    };
    let check = do_typecheck(&interner, source, mode, &context_types, None);
    to_js(&check)
}

/// Quick type-check with context types and expected tail type (hard mode).
#[wasm_bindgen]
pub fn typecheck_with_tail(
    source: &str,
    mode: &str,
    context_types_json: &str,
    expected_tail_type: &str,
) -> JsValue {
    let interner = Interner::new();
    let context_types = match parse_context_types(&interner, context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let check = CheckResult {
                ok: false,
                message: Some(e),
            };
            return to_js(&check);
        }
    };
    let expected_tail: Ty = match serde_json::from_str::<TypeDesc>(expected_tail_type) {
        Ok(desc) => desc_to_ty(&interner, &desc),
        Err(e) => {
            let check = CheckResult {
                ok: false,
                message: Some(format!(
                    "failed to parse expected tail type: {e}"
                )),
            };
            return to_js(&check);
        }
    };
    let check = do_typecheck(
        &interner,
        source,
        mode,
        &context_types,
        Some(&expected_tail),
    );
    to_js(&check)
}

/// Typecheck all nodes at once using the full orchestration pipeline.
///
/// Returns JSON: `{ contextTypes, nodeLocals, nodeErrors }` where:
/// - `contextTypes`: Record<name, type> — all externally-visible context types
/// - `nodeLocals`: Record<nodeName, {raw, self}> — per-node local types
/// - `nodeErrors`: Record<nodeName, {selfBind?, initialValue?, historyBind?, ifModifiedKey?, assert?, messages?: Record<index, string>}>
///   — per-node, per-field error strings (absent = ok)
///
/// On env-level failure: `{ error: string }`
#[wasm_bindgen]
pub fn typecheck_nodes(nodes_json: &str, injected_types_json: &str) -> JsValue {
    let interner = Interner::new();

    let web_nodes: Vec<convert::WebNode> = match serde_json::from_str(nodes_json) {
        Ok(v) => v,
        Err(e) => {
            let result = FxHashMap::from_iter([("error".to_string(), format!("json parse: {e}"))]);
            let json =
                serde_json::to_string(&result).expect("internal serialization should not fail");
            return js_sys::JSON::parse(&json).expect("serde_json output is always valid JSON");
        }
    };

    let injected_types = match parse_context_types(&interner, injected_types_json) {
        Ok(t) => t,
        Err(e) => {
            let result = FxHashMap::from_iter([("error".to_string(), e)]);
            let json =
                serde_json::to_string(&result).expect("internal serialization should not fail");
            return js_sys::JSON::parse(&json).expect("serde_json output is always valid JSON");
        }
    };

    let specs: Vec<NodeSpec> = match web_nodes
        .iter()
        .map(|w| convert::convert_node(&interner, w))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(specs) => specs,
        Err(e) => {
            let result = FxHashMap::from_iter([("error".to_string(), e)]);
            let json =
                serde_json::to_string(&result).expect("internal serialization should not fail");
            return js_sys::JSON::parse(&json).expect("serde_json output is always valid JSON");
        }
    };
    let registry = default_registry(&interner);

    // 1. Compute external context env (this gives us context_types + per-node locals)
    let env = match acvus_orchestration::compute_external_context_env(
        &interner,
        &specs,
        &injected_types,
        &registry,
    ) {
        Ok(env) => env,
        Err(errs) => {
            let msg = errs
                .into_iter()
                .map(|e| e.display(&interner).to_string())
                .collect::<Vec<_>>()
                .join("\n");
            let result = FxHashMap::from_iter([("error".to_string(), msg)]);
            let json =
                serde_json::to_string(&result).expect("internal serialization should not fail");
            return js_sys::JSON::parse(&json).expect("serde_json output is always valid JSON");
        }
    };

    let context_types_str: FxHashMap<String, TypeDesc> = env
        .context_types
        .iter()
        .map(|(k, v)| (interner.resolve(*k).to_string(), ty_to_desc(&interner, v)))
        .collect();

    let node_locals_str: FxHashMap<String, FxHashMap<String, TypeDesc>> = env
        .node_locals
        .iter()
        .map(|(name, locals)| {
            let mut m = FxHashMap::default();
            m.insert("raw".into(), ty_to_desc(&interner, &locals.raw_ty));
            m.insert("self".into(), ty_to_desc(&interner, &locals.self_ty));
            (interner.resolve(*name).to_string(), m)
        })
        .collect();

    // 2. Per-node, per-field typecheck
    let mut node_errors: FxHashMap<String, FxHashMap<String, String>> = FxHashMap::default();

    for spec in &specs {
        let Some(locals) = env.node_locals.get(&spec.name) else {
            continue;
        };
        let mut field_errors: FxHashMap<String, String> = FxHashMap::default();

        // Context visible inside this node: external + @self (only when initial_value is Some)
        let mut node_ctx = env.context_types.clone();
        if spec.self_spec.initial_value.is_some() {
            node_ctx.insert(interner.intern("self"), locals.self_ty.clone());
        }

        // initial_value: expected tail = stored type (only when Some)
        if let Some(ref init_src) = spec.self_spec.initial_value {
            let hint = match &locals.self_ty {
                Ty::Error => None,
                ty => Some(ty),
            };
            if let Some(err) = check_script(
                &interner,
                interner.resolve(*init_src),
                &env.context_types,
                &registry,

                hint,
            ) {
                field_errors.insert("initialValue".into(), err);
            }
        }

        // strategy: history_bind uses @self (= stored type), no @raw
        let mut strategy_ctx = env.context_types.clone();
        strategy_ctx.insert(interner.intern("self"), locals.self_ty.clone());
        match &spec.strategy {
            acvus_orchestration::Strategy::History { history_bind } => {
                if let Some(err) = check_script(
                    &interner,
                    interner.resolve(*history_bind),
                    &strategy_ctx,
                    &registry,
    
                    None,
                ) {
                    field_errors.insert("historyBind".into(), err);
                }
            }
            acvus_orchestration::Strategy::IfModified { key } => {
                if let Some(err) = check_script(
                    &interner,
                    interner.resolve(*key),
                    &env.context_types,
                    &registry,
    
                    None,
                ) {
                    field_errors.insert("ifModifiedKey".into(), err);
                }
            }
            _ => {}
        }

        // assert: context = external + @self (= stored type), expected tail = Bool
        if let Some(ref assert_src) = spec.assert {
            let mut assert_ctx = env.context_types.clone();
            assert_ctx.insert(interner.intern("self"), locals.self_ty.clone());
            if let Some(err) = check_script(
                &interner,
                interner.resolve(*assert_src),
                &assert_ctx,
                &registry,

                Some(&Ty::Bool),
            ) {
                field_errors.insert("assert".into(), err);
            }
        }

        // messages (LLM only)
        let messages: &[acvus_orchestration::MessageSpec] = match &spec.kind {
            acvus_orchestration::NodeKind::Llm(llm) => &llm.messages,
            _ => &[],
        };

        let mut msg_errors: FxHashMap<String, String> = FxHashMap::default();
        for (mi, msg) in messages.iter().enumerate() {
            match msg {
                acvus_orchestration::MessageSpec::Block { source, .. } => {
                    if let Some(err) =
                        check_template(&interner, source, &node_ctx, &registry)
                    {
                        msg_errors.insert(mi.to_string(), err);
                    }
                }
                acvus_orchestration::MessageSpec::Iterator { key, .. } => {
                    if let Some(err) = check_script(
                        &interner,
                        interner.resolve(*key),
                        &node_ctx,
                        &registry,
        
                        None,
                    ) {
                        msg_errors.insert(mi.to_string(), err);
                    }
                }
            }
        }
        if !msg_errors.is_empty() {
            field_errors.insert(
                "messages".into(),
                serde_json::to_string(&msg_errors).expect("internal serialization should not fail"),
            );
        }

        if !field_errors.is_empty() {
            node_errors.insert(interner.resolve(spec.name).to_string(), field_errors);
        }
    }

    // 3. Serialize result
    let mut result = FxHashMap::default();
    result.insert(
        "contextTypes",
        serde_json::to_value(&context_types_str).expect("internal serialization should not fail"),
    );
    result.insert(
        "nodeLocals",
        serde_json::to_value(&node_locals_str).expect("internal serialization should not fail"),
    );
    result.insert(
        "nodeErrors",
        serde_json::to_value(&node_errors).expect("internal serialization should not fail"),
    );
    let json = serde_json::to_string(&result).expect("internal serialization should not fail");
    js_sys::JSON::parse(&json).expect("serde_json output is always valid JSON")
}

/// Helper: typecheck a script, returning error string if any.
fn check_script(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
    expected_tail: Option<&Ty>,
) -> Option<String> {
    if source.trim().is_empty() {
        return None;
    }
    let script = match acvus_ast::parse_script(interner, source) {
        Ok(s) => s,
        Err(e) => return Some(format!("{e}")),
    };
    match acvus_mir::compile_script_with_hint(
        interner,
        &script,
        context_types,
        registry,
        expected_tail,
    ) {
        Ok(_) => None,
        Err(errs) => Some(
            errs.into_iter()
                .map(|e| e.display(interner).to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    }
}

/// Helper: typecheck a template, returning error string if any.
fn check_template(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Option<String> {
    if source.trim().is_empty() {
        return None;
    }
    let ast = match acvus_ast::parse(interner, source) {
        Ok(a) => a,
        Err(e) => return Some(format!("{e}")),
    };
    match acvus_mir::compile(interner, &ast, context_types, registry) {
        Ok(_) => None,
        Err(errs) => Some(
            errs.into_iter()
                .map(|e| e.display(interner).to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    }
}

/// Evaluate a template with the given context values. Returns the rendered string.
/// `context_json`: JSON object mapping names to values, e.g. `{"role": "user", "content": "hi"}`
#[wasm_bindgen]
pub async fn evaluate(source: &str, mode: &str, context_json: &str) -> Result<JsValue, JsValue> {
    use acvus_interpreter::{ExternFnRegistry, Interpreter, Value};
    use acvus_utils::Stepped;
    use session::pure_to_json;
    use std::sync::Arc;

    let interner = Interner::new();

    let context_values: FxHashMap<String, serde_json::Value> =
        serde_json::from_str(context_json)
            .map_err(|e| JsValue::from_str(&format!("invalid context JSON: {e}")))?;

    // Infer context types from values
    let mut context_types: FxHashMap<Astr, Ty> = FxHashMap::default();
    for (k, v) in &context_values {
        context_types.insert(interner.intern(k), json_to_ty(&interner, v));
    }

    let registry = default_registry(&interner);

    let module = match mode {
        "template" => {
            let ast = acvus_ast::parse(&interner, source)
                .map_err(|e| JsValue::from_str(&format!("parse error: {e}")))?;
            acvus_mir::compile_analysis(&interner, &ast, &context_types, &registry)
                .map(|(module, _)| module)
                .map_err(|errs| {
                    let msg = errs
                        .into_iter()
                        .map(|e| e.display(&interner).to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    JsValue::from_str(&format!("compile error: {msg}"))
                })?
        }
        _ => {
            let script = acvus_ast::parse_script(&interner, source)
                .map_err(|e| JsValue::from_str(&format!("parse error: {e}")))?;
            acvus_mir::compile_script_analysis(
                &interner,
                &script,
                &context_types,
                &registry,

            )
            .map(|(module, _, _)| module)
            .map_err(|errs| {
                let msg = errs
                    .into_iter()
                    .map(|e| e.display(&interner).to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                JsValue::from_str(&format!("compile error: {msg}"))
            })?
        }
    };

    // Build context values
    let ctx: FxHashMap<Astr, Value> = context_values
        .iter()
        .map(|(k, v)| (interner.intern(k), json_to_value(&interner, v)))
        .collect();

    let mut extern_fns = ExternFnRegistry::new(&interner);
    let _regex_mod = acvus_ext::regex_module(&interner, &mut extern_fns);

    let interp = Interpreter::new(&interner, module, &extern_fns);

    if mode == "template" {
        // Template: concatenate all emits into a string
        let result = interp.execute_to_string(ctx).await;
        Ok(JsValue::from_str(&result))
    } else {
        // Script: return the first emitted value (tail expression)
        let mut coroutine = interp.execute();
        let mut result_value = Value::Unit;
        loop {
            match coroutine.resume().await {
                Stepped::Emit(value) => {
                    result_value = value;
                    break;
                }
                Stepped::NeedContext(request) => {
                    let name = request.name();
                    let v = ctx.get(&name).cloned().ok_or_else(|| {
                        JsValue::from_str(&format!(
                            "runtime error: context key '{}' not found",
                            interner.resolve(name)
                        ))
                    })?;
                    request.resolve(Arc::new(v));
                }
                Stepped::Done => break,
                Stepped::Error(e) => {
                    return Err(JsValue::from_str(&format!("runtime error: {e}")));
                }
            }
        }
        let json = pure_to_json(&interner, &result_value.into_pure());
        let json_str =
            serde_json::to_string(&json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::JSON::parse(&json_str).map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

fn json_to_ty(interner: &Interner, v: &serde_json::Value) -> Ty {
    match v {
        serde_json::Value::Null => Ty::Unit,
        serde_json::Value::Bool(_) => Ty::Bool,
        serde_json::Value::Number(n) => {
            if n.is_f64() && n.as_i64().is_none() {
                Ty::Float
            } else {
                Ty::Int
            }
        }
        serde_json::Value::String(_) => Ty::String,
        serde_json::Value::Array(items) => {
            let elem_ty = items
                .first()
                .map(|i| json_to_ty(interner, i))
                .unwrap_or(Ty::Infer);
            Ty::List(Box::new(elem_ty))
        }
        serde_json::Value::Object(map) => {
            let fields: FxHashMap<Astr, Ty> = map
                .iter()
                .map(|(k, v)| (interner.intern(k), json_to_ty(interner, v)))
                .collect();
            Ty::Object(fields)
        }
    }
}

fn json_to_value(interner: &Interner, v: &serde_json::Value) -> acvus_interpreter::Value {
    use acvus_interpreter::Value;
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(
                    n.as_f64()
                        .expect("JSON number should be representable as f64"),
                )
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(items) => {
            Value::List(items.iter().map(|i| json_to_value(interner, i)).collect())
        }
        serde_json::Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (interner.intern(k), json_to_value(interner, v)))
                .collect(),
        ),
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
        mode: &str,
        ctx: &FxHashMap<Astr, Ty>,
    ) -> AnalyzeResult {
        do_analyze(interner, source, mode, ctx, None)
    }

    /// Helper: serialize a TypeDesc to a JSON string for comparison.
    fn desc_json(desc: &TypeDesc) -> String {
        serde_json::to_string(desc).expect("TypeDesc serialization should not fail")
    }

    #[test]
    fn test_ty_to_desc_primitives() {
        let interner = test_interner();
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Int)), r#"{"kind":"primitive","name":"Int"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Float)), r#"{"kind":"primitive","name":"Float"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::String)), r#"{"kind":"primitive","name":"String"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::Bool)), r#"{"kind":"primitive","name":"Bool"}"#);
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
        assert_eq!(desc_json(&desc), r#"{"kind":"list","elem":{"kind":"primitive","name":"String"}}"#);
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
    }

    #[test]
    fn test_analyze_script_context_types() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze_test(&interner, "@x + 1", "script", &ctx);
        if !result.ok {
            eprintln!("errors: {:?}", result.errors);
        }
        assert!(result.ok);
        assert_eq!(result.context_keys.len(), 1);
        assert_eq!(result.context_keys[0].name, "x");
        assert_eq!(desc_json(&result.context_keys[0].ty), r#"{"kind":"primitive","name":"Int"}"#);
        assert_eq!(desc_json(&result.tail_type), r#"{"kind":"primitive","name":"Int"}"#);
    }

    #[test]
    fn test_analyze_with_provided_context_types() {
        let interner = test_interner();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("x"), Ty::Int);
        let result = do_analyze_test(&interner, "@x + 1", "script", &ctx);
        assert!(result.ok);
        assert_eq!(desc_json(&result.context_keys[0].ty), r#"{"kind":"primitive","name":"Int"}"#);
    }

    #[test]
    fn test_analyze_template() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze_test(&interner, "hello {{ @name }}", "template", &ctx);
        assert!(result.ok);
        assert_eq!(result.context_keys.len(), 1);
        assert_eq!(result.context_keys[0].name, "name");
        assert_eq!(desc_json(&result.tail_type), r#"{"kind":"primitive","name":"String"}"#);
    }

    #[test]
    fn test_tail_type_mismatch() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze(&interner, "\"hello\"", "script", &ctx, Some(&Ty::Int));
        assert!(!result.ok, "should fail: String vs Int");
    }

    #[test]
    fn test_tail_type_match() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze(&interner, "1 + 2", "script", &ctx, Some(&Ty::Int));
        assert!(result.ok, "should succeed: Int vs Int");
    }

    #[test]
    fn test_analyze_unresolved_type() {
        let interner = test_interner();
        let ctx = FxHashMap::default();
        let result = do_analyze_test(&interner, "@x", "script", &ctx);
        if !result.ok {
            eprintln!("errors: {:?}", result.errors);
        }
        assert!(result.ok);
        assert_eq!(desc_json(&result.context_keys[0].ty), r#"{"kind":"unsupported","raw":"?"}"#);
    }
}
