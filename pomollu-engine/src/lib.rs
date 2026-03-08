mod convert;
mod session;

use std::collections::{BTreeMap, HashMap};

#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn init() {
    console_error_panic_hook::set_once();
}

use acvus_orchestration::NodeSpec;
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;
use acvus_mir::user_type::UserTypeRegistry;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Serialize any Serialize type to JsValue via JSON.
fn to_js<T: Serialize>(value: &T) -> JsValue {
    let json_str = serde_json::to_string(value).unwrap_or_default();
    js_sys::JSON::parse(&json_str).unwrap_or(JsValue::NULL)
}

fn default_registry() -> ExternRegistry {
    let mut registry = ExternRegistry::new();
    let mut fn_reg = acvus_interpreter::ExternFnRegistry::default();
    let regex_mod = acvus_ext::regex_module(&mut fn_reg);
    registry.register(&regex_mod);
    registry
}

fn extract_context_keys_with_types(module: &MirModule) -> Vec<ContextKey> {
    let mut seen = HashMap::<String, Ty>::new();

    let mut collect = |insts: &[acvus_mir::ir::Inst],
                       val_types: &HashMap<acvus_mir::ir::ValueId, Ty>| {
        for inst in insts {
            if let InstKind::ContextLoad { dst, name, .. } = &inst.kind {
                if seen.contains_key(name) {
                    continue;
                }
                let ty = val_types.get(dst).cloned().unwrap_or(Ty::Infer);
                seen.insert(name.clone(), ty);
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
            name,
            ty: format_ty(&ty),
        })
        .collect();
    keys.sort_by(|a, b| a.name.cmp(&b.name));
    keys
}

fn format_ty(ty: &Ty) -> String {
    match ty {
        Ty::Var(_) | Ty::Infer => "?".to_string(),
        Ty::Error => "?".to_string(),
        other => format!("{other}"),
    }
}

/// Find the closing position for a generic bracket `<` at position `open`.
/// Returns the byte index of the matching `>`, respecting nested `<>`, `{}`, `()`.
fn find_matching_close(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s[open..].char_indices() {
        match c {
            '<' | '{' | '(' => depth += 1,
            '>' | '}' | ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_ty(s: &str) -> Option<Ty> {
    let s = s.trim();
    match s {
        "Int" => Some(Ty::Int),
        "Float" => Some(Ty::Float),
        "String" => Some(Ty::String),
        "Bool" => Some(Ty::Bool),
        "Unit" => Some(Ty::Unit),
        "Range" => Some(Ty::Range),
        "Byte" => Some(Ty::Byte),
        _ if s.starts_with("List<") => {
            let close = find_matching_close(s, 4)?;
            if close != s.len() - 1 {
                return None;
            }
            let inner = &s[5..close];
            Some(Ty::List(Box::new(parse_ty(inner)?)))
        }
        _ if s.starts_with("Option<") => {
            let close = find_matching_close(s, 6)?;
            if close != s.len() - 1 {
                return None;
            }
            let inner = &s[7..close];
            Some(Ty::Option(Box::new(parse_ty(inner)?)))
        }
        _ if s.starts_with('{') && s.ends_with('}') => {
            let inner = &s[1..s.len() - 1].trim();
            if inner.is_empty() {
                return Some(Ty::Object(BTreeMap::new()));
            }
            let mut fields = BTreeMap::new();
            for pair in split_top_level(inner, ',') {
                let pair = pair.trim();
                let colon = pair.find(':')?;
                let key = pair[..colon].trim().to_string();
                let val = parse_ty(pair[colon + 1..].trim())?;
                fields.insert(key, val);
            }
            Some(Ty::Object(fields))
        }
        _ => None,
    }
}

/// Parse a JSON map of `name → type_string` into `HashMap<String, Ty>`.
/// Returns an error string if any type string fails to parse.
fn parse_context_types(json: &str) -> Result<HashMap<String, Ty>, String> {
    let raw: HashMap<String, String> = serde_json::from_str(json).unwrap_or_default();
    let mut result = HashMap::new();
    for (k, v) in raw {
        let ty = parse_ty(&v).ok_or_else(|| format!("failed to parse type for @{k}: {v}"))?;
        result.insert(k, ty);
    }
    Ok(result)
}

/// Split a string by `sep` only at the top level (not inside `<>`, `{}`, `()`).
fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '{' | '(' => depth += 1,
            '>' | '}' | ')' => depth -= 1,
            c if c == sep && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

#[derive(Serialize)]
struct ContextKey {
    name: String,
    #[serde(rename = "type")]
    ty: String,
}

#[derive(Serialize)]
struct AnalyzeResult {
    ok: bool,
    errors: Vec<String>,
    context_keys: Vec<ContextKey>,
    tail_type: String,
}

fn do_analyze(
    source: &str,
    mode: &str,
    context_types: &HashMap<String, Ty>,
    expected_tail: Option<&Ty>,
) -> AnalyzeResult {
    let registry = default_registry();
    let user_types = UserTypeRegistry::new();

    match mode {
        "template" => {
            let ast = match acvus_ast::parse(source) {
                Ok(ast) => ast,
                Err(e) => {
                    return AnalyzeResult {
                        ok: false,
                        errors: vec![format!("{e}")],
                        context_keys: vec![],
                        tail_type: String::new(),
                    };
                }
            };
            match acvus_mir::compile_analysis(&ast, context_types, &registry, &user_types) {
                Ok((module, _hints)) => {
                    let keys = extract_context_keys_with_types(&module);
                    AnalyzeResult {
                        ok: true,
                        errors: vec![],
                        context_keys: keys,
                        tail_type: "String".to_string(),
                    }
                }
                Err(errs) => AnalyzeResult {
                    ok: false,
                    errors: errs.into_iter().map(|e| format!("{e}")).collect(),
                    context_keys: vec![],
                    tail_type: String::new(),
                },
            }
        }
        _ => {
            let script = match acvus_ast::parse_script(source) {
                Ok(s) => s,
                Err(e) => {
                    return AnalyzeResult {
                        ok: false,
                        errors: vec![format!("{e}")],
                        context_keys: vec![],
                        tail_type: String::new(),
                    };
                }
            };
            match acvus_mir::compile_script_analysis_with_tail(
                &script,
                context_types,
                &registry,
                &user_types,
                expected_tail,
            ) {
                Ok((module, _hints, tail_ty)) => {
                    let keys = extract_context_keys_with_types(&module);
                    AnalyzeResult {
                        ok: true,
                        errors: vec![],
                        context_keys: keys,
                        tail_type: format_ty(&tail_ty),
                    }
                }
                Err(errs) => AnalyzeResult {
                    ok: false,
                    errors: errs.into_iter().map(|e| format!("{e}")).collect(),
                    context_keys: vec![],
                    tail_type: String::new(),
                },
            }
        }
    }
}

/// Analyze a script or template. Returns JSON:
/// `{ ok, errors, context_keys: [{name, type}], tail_type }`
#[wasm_bindgen]
pub fn analyze(source: &str, mode: &str) -> JsValue {
    let context_types = HashMap::new();
    let result = do_analyze(source, mode, &context_types, None);
    to_js(&result)
}

/// Analyze with user-provided context types. `context_types_json` is a JSON
/// object mapping names to type strings, e.g. `{"name": "String", "age": "Int"}`.
#[wasm_bindgen]
pub fn analyze_with_types(source: &str, mode: &str, context_types_json: &str) -> JsValue {
    let context_types = match parse_context_types(context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let result = AnalyzeResult {
                ok: false,
                errors: vec![e],
                context_keys: vec![],
                tail_type: String::new(),
            };
            return to_js(&result);
        }
    };
    let result = do_analyze(source, mode, &context_types, None);
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
    let context_types = match parse_context_types(context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let result = AnalyzeResult {
                ok: false,
                errors: vec![e],
                context_keys: vec![],
                tail_type: String::new(),
            };
            return to_js(&result);
        }
    };
    let expected_tail = parse_ty(expected_tail_type);
    let result = do_analyze(source, mode, &context_types, expected_tail.as_ref());
    to_js(&result)
}

#[derive(Serialize)]
struct CheckResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

fn do_typecheck(
    source: &str,
    mode: &str,
    context_types: &HashMap<String, Ty>,
    expected_tail: Option<&Ty>,
) -> CheckResult {
    let registry = default_registry();
    let user_types = UserTypeRegistry::new();

    let result = match mode {
        "template" => {
            let ast = match acvus_ast::parse(source) {
                Ok(ast) => ast,
                Err(e) => return CheckResult { ok: false, message: Some(format!("{e}")) },
            };
            acvus_mir::compile(&ast, context_types, &registry, &user_types).map(|_| ())
        }
        _ => {
            let script = match acvus_ast::parse_script(source) {
                Ok(s) => s,
                Err(e) => return CheckResult { ok: false, message: Some(format!("{e}")) },
            };
            acvus_mir::compile_script_with_hint(&script, context_types, &registry, &user_types, expected_tail)
                .map(|_| ())
        }
    };

    match result {
        Ok(()) => CheckResult { ok: true, message: None },
        Err(errs) => CheckResult {
            ok: false,
            message: Some(errs.into_iter().map(|e| format!("{e}")).collect::<Vec<_>>().join("\n")),
        },
    }
}

/// Quick type-check (hard mode). Returns JSON: `{ ok, message? }`
#[wasm_bindgen]
pub fn typecheck(source: &str, mode: &str) -> JsValue {
    let context_types = HashMap::new();
    let check = do_typecheck(source, mode, &context_types, None);
    to_js(&check)
}

/// Quick type-check with user-provided context types (hard mode).
#[wasm_bindgen]
pub fn typecheck_with_types(source: &str, mode: &str, context_types_json: &str) -> JsValue {
    let context_types = match parse_context_types(context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let check = CheckResult { ok: false, message: Some(e) };
            return to_js(&check);
        }
    };
    let check = do_typecheck(source, mode, &context_types, None);
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
    let context_types = match parse_context_types(context_types_json) {
        Ok(t) => t,
        Err(e) => {
            let check = CheckResult { ok: false, message: Some(e) };
            return to_js(&check);
        }
    };
    let Some(expected_tail) = parse_ty(expected_tail_type) else {
        let check = CheckResult {
            ok: false,
            message: Some(format!("failed to parse expected tail type: {expected_tail_type}")),
        };
        return to_js(&check);
    };
    let check = do_typecheck(source, mode, &context_types, Some(&expected_tail));
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
    let web_nodes: Vec<convert::WebNode> = match serde_json::from_str(nodes_json) {
        Ok(v) => v,
        Err(e) => {
            let result = HashMap::from([("error".to_string(), format!("json parse: {e}"))]);
            let json = serde_json::to_string(&result).unwrap_or_default();
            return js_sys::JSON::parse(&json).unwrap_or(JsValue::NULL);
        }
    };

    let injected_types = match parse_context_types(injected_types_json) {
        Ok(t) => t,
        Err(e) => {
            let result = HashMap::from([("error".to_string(), e)]);
            let json = serde_json::to_string(&result).unwrap_or_default();
            return js_sys::JSON::parse(&json).unwrap_or(JsValue::NULL);
        }
    };

    let specs: Vec<NodeSpec> = match web_nodes.iter().map(convert::convert_node).collect::<Result<Vec<_>, _>>() {
        Ok(specs) => specs,
        Err(e) => {
            let result = HashMap::from([("error".to_string(), e)]);
            let json = serde_json::to_string(&result).unwrap_or_default();
            return js_sys::JSON::parse(&json).unwrap_or(JsValue::NULL);
        }
    };
    let registry = default_registry();

    // 1. Compute external context env (this gives us context_types + per-node locals)
    let env = match acvus_orchestration::compute_external_context_env(&specs, &injected_types, &registry) {
        Ok(env) => env,
        Err(errs) => {
            let msg = errs.into_iter().map(|e| format!("{e}")).collect::<Vec<_>>().join("\n");
            let result = HashMap::from([("error".to_string(), msg)]);
            let json = serde_json::to_string(&result).unwrap_or_default();
            return js_sys::JSON::parse(&json).unwrap_or(JsValue::NULL);
        }
    };

    let context_types_str: HashMap<String, String> = env
        .context_types
        .iter()
        .map(|(k, v)| (k.clone(), format_ty(v)))
        .collect();

    let node_locals_str: HashMap<String, HashMap<String, String>> = env
        .node_locals
        .iter()
        .map(|(name, locals)| {
            let mut m = HashMap::new();
            m.insert("raw".into(), format_ty(&locals.raw_ty));
            m.insert("self".into(), format_ty(&locals.self_ty));
            (name.clone(), m)
        })
        .collect();

    // 2. Per-node, per-field typecheck
    let user_types = UserTypeRegistry::new();
    let mut node_errors: HashMap<String, HashMap<String, String>> = HashMap::new();

    for spec in &specs {
        let Some(locals) = env.node_locals.get(&spec.name) else { continue };
        let mut field_errors: HashMap<String, String> = HashMap::new();

        // Context visible inside this node: external + @self (only when initial_value is Some)
        let mut node_ctx = env.context_types.clone();
        if spec.self_spec.initial_value.is_some() {
            node_ctx.insert("self".into(), locals.self_ty.clone());
        }

        // initial_value: expected tail = stored type (only when Some)
        if let Some(ref init_src) = spec.self_spec.initial_value {
            let hint = match &locals.self_ty {
                Ty::Error => None,
                ty => Some(ty),
            };
            if let Some(err) = check_script(init_src, &env.context_types, &registry, &user_types, hint) {
                field_errors.insert("initialValue".into(), err);
            }
        }

        // strategy: history_bind uses @self (= stored type), no @raw
        let mut strategy_ctx = env.context_types.clone();
        strategy_ctx.insert("self".into(), locals.self_ty.clone());
        match &spec.strategy {
            acvus_orchestration::Strategy::History { history_bind } => {
                if let Some(err) = check_script(history_bind, &strategy_ctx, &registry, &user_types, None) {
                    field_errors.insert("historyBind".into(), err);
                }
            }
            acvus_orchestration::Strategy::IfModified { key } => {
                if let Some(err) = check_script(key, &env.context_types, &registry, &user_types, None) {
                    field_errors.insert("ifModifiedKey".into(), err);
                }
            }
            _ => {}
        }

        // assert: context = external + @self (= stored type), expected tail = Bool
        if let Some(ref assert_src) = spec.assert {
            let mut assert_ctx = env.context_types.clone();
            assert_ctx.insert("self".into(), locals.self_ty.clone());
            if let Some(err) = check_script(assert_src, &assert_ctx, &registry, &user_types, Some(&Ty::Bool)) {
                field_errors.insert("assert".into(), err);
            }
        }

        // messages (LLM only)
        let messages: &[acvus_orchestration::MessageSpec] = match &spec.kind {
            acvus_orchestration::NodeKind::Llm(llm) => &llm.messages,
            _ => &[],
        };

        let mut msg_errors: HashMap<String, String> = HashMap::new();
        for (mi, msg) in messages.iter().enumerate() {
            match msg {
                acvus_orchestration::MessageSpec::Block { source, .. } => {
                    if let Some(err) = check_template(source, &node_ctx, &registry, &user_types) {
                        msg_errors.insert(mi.to_string(), err);
                    }
                }
                acvus_orchestration::MessageSpec::Iterator { key, .. } => {
                    if let Some(err) = check_script(key, &node_ctx, &registry, &user_types, None) {
                        msg_errors.insert(mi.to_string(), err);
                    }
                }
            }
        }
        if !msg_errors.is_empty() {
            field_errors.insert("messages".into(), serde_json::to_string(&msg_errors).unwrap_or_default());
        }

        if !field_errors.is_empty() {
            node_errors.insert(spec.name.clone(), field_errors);
        }
    }

    // 3. Serialize result
    let mut result = HashMap::new();
    result.insert("contextTypes", serde_json::to_value(&context_types_str).unwrap_or_default());
    result.insert("nodeLocals", serde_json::to_value(&node_locals_str).unwrap_or_default());
    result.insert("nodeErrors", serde_json::to_value(&node_errors).unwrap_or_default());
    let json = serde_json::to_string(&result).unwrap_or_default();
    js_sys::JSON::parse(&json).unwrap_or(JsValue::NULL)
}

/// Helper: typecheck a script, returning error string if any.
fn check_script(
    source: &str,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Option<String> {
    if source.trim().is_empty() { return None; }
    let script = match acvus_ast::parse_script(source) {
        Ok(s) => s,
        Err(e) => return Some(format!("{e}")),
    };
    match acvus_mir::compile_script_with_hint(&script, context_types, registry, user_types, expected_tail) {
        Ok(_) => None,
        Err(errs) => Some(errs.into_iter().map(|e| format!("{e}")).collect::<Vec<_>>().join("\n")),
    }
}

/// Helper: typecheck a template, returning error string if any.
fn check_template(
    source: &str,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
) -> Option<String> {
    if source.trim().is_empty() { return None; }
    let ast = match acvus_ast::parse(source) {
        Ok(a) => a,
        Err(e) => return Some(format!("{e}")),
    };
    match acvus_mir::compile(&ast, context_types, registry, user_types) {
        Ok(_) => None,
        Err(errs) => Some(errs.into_iter().map(|e| format!("{e}")).collect::<Vec<_>>().join("\n")),
    }
}

/// Evaluate a template with the given context values. Returns the rendered string.
/// `context_json`: JSON object mapping names to values, e.g. `{"role": "user", "content": "hi"}`
#[wasm_bindgen]
pub async fn evaluate(source: &str, mode: &str, context_json: &str) -> Result<JsValue, JsValue> {
    use acvus_coroutine::Stepped;
    use acvus_interpreter::{ExternFnRegistry, Interpreter, Value};
    use acvus_mir::user_type::UserTypeRegistry;
    use session::pure_to_json;
    use std::sync::Arc;

    let context_values: HashMap<String, serde_json::Value> =
        serde_json::from_str(context_json).unwrap_or_default();

    // Infer context types from values
    let mut context_types: HashMap<String, Ty> = HashMap::new();
    for (k, v) in &context_values {
        context_types.insert(k.clone(), json_to_ty(v));
    }

    let registry = default_registry();
    let user_types = UserTypeRegistry::new();

    let module = match mode {
        "template" => {
            let ast = acvus_ast::parse(source)
                .map_err(|e| JsValue::from_str(&format!("parse error: {e}")))?;
            acvus_mir::compile_analysis(&ast, &context_types, &registry, &user_types)
                .map(|(module, _)| module)
                .map_err(|errs| {
                    let msg = errs.into_iter().map(|e| format!("{e}")).collect::<Vec<_>>().join("\n");
                    JsValue::from_str(&format!("compile error: {msg}"))
                })?
        }
        _ => {
            let script = acvus_ast::parse_script(source)
                .map_err(|e| JsValue::from_str(&format!("parse error: {e}")))?;
            acvus_mir::compile_script_analysis(&script, &context_types, &registry, &user_types)
                .map(|(module, _, _)| module)
                .map_err(|errs| {
                    let msg = errs.into_iter().map(|e| format!("{e}")).collect::<Vec<_>>().join("\n");
                    JsValue::from_str(&format!("compile error: {msg}"))
                })?
        }
    };

    // Build context values
    let ctx: HashMap<String, Value> = context_values
        .iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect();

    let mut extern_fns = ExternFnRegistry::new();
    let _regex_mod = acvus_ext::regex_module(&mut extern_fns);

    let interp = Interpreter::new(module, &extern_fns);

    if mode == "template" {
        // Template: concatenate all emits into a string
        let result = interp.execute_to_string(ctx).await;
        Ok(JsValue::from_str(&result))
    } else {
        // Script: return the first emitted value (tail expression)
        let (mut coroutine, mut key) = interp.execute();
        let mut result_value = Value::Unit;
        loop {
            match coroutine.resume(key).await {
                Stepped::Emit(emit) => {
                    let (value, _next_key) = emit.into_parts();
                    result_value = value;
                    break;
                }
                Stepped::NeedContext(need) => {
                    let name = need.name().to_string();
                    let v = ctx.get(&name).cloned().unwrap_or(Value::Unit);
                    key = need.into_key(Arc::new(v));
                }
                Stepped::Done => break,
                Stepped::Error(e) => {
                    return Err(JsValue::from_str(&format!("runtime error: {e}")));
                }
            }
        }
        let json = pure_to_json(&result_value.into_pure());
        let json_str = serde_json::to_string(&json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::JSON::parse(&json_str).map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

fn json_to_ty(v: &serde_json::Value) -> Ty {
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
            let elem_ty = items.first().map(json_to_ty).unwrap_or(Ty::Infer);
            Ty::List(Box::new(elem_ty))
        }
        serde_json::Value::Object(map) => {
            let fields: std::collections::BTreeMap<String, Ty> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_ty(v)))
                .collect();
            Ty::Object(fields)
        }
    }
}

fn json_to_value(v: &serde_json::Value) -> acvus_interpreter::Value {
    use acvus_interpreter::Value;
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(items) => {
            Value::List(items.iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn do_analyze_test(source: &str, mode: &str, ctx: &HashMap<String, Ty>) -> AnalyzeResult {
        do_analyze(source, mode, ctx, None)
    }

    #[test]
    fn test_parse_ty_basic() {
        assert_eq!(parse_ty("Int"), Some(Ty::Int));
        assert_eq!(parse_ty("Float"), Some(Ty::Float));
        assert_eq!(parse_ty("String"), Some(Ty::String));
        assert_eq!(parse_ty("Bool"), Some(Ty::Bool));
        assert_eq!(parse_ty("Unit"), Some(Ty::Unit));
        assert_eq!(parse_ty("Range"), Some(Ty::Range));
        assert_eq!(parse_ty("Byte"), Some(Ty::Byte));
    }

    #[test]
    fn test_parse_ty_list() {
        assert_eq!(parse_ty("List<Int>"), Some(Ty::List(Box::new(Ty::Int))));
        assert_eq!(
            parse_ty("List<List<String>>"),
            Some(Ty::List(Box::new(Ty::List(Box::new(Ty::String)))))
        );
    }

    #[test]
    fn test_parse_ty_option() {
        assert_eq!(
            parse_ty("Option<Int>"),
            Some(Ty::Option(Box::new(Ty::Int)))
        );
    }

    #[test]
    fn test_parse_ty_object() {
        let ty = parse_ty("{name: String, age: Int}").unwrap();
        let mut expected = BTreeMap::new();
        expected.insert("name".to_string(), Ty::String);
        expected.insert("age".to_string(), Ty::Int);
        assert_eq!(ty, Ty::Object(expected));
    }

    #[test]
    fn test_parse_ty_unknown() {
        assert_eq!(parse_ty("Foo"), None);
        assert_eq!(parse_ty("?"), None);
    }

    #[test]
    fn test_format_ty_var() {
        assert_eq!(format_ty(&Ty::Var(acvus_mir::ty::TyVar(0))), "?");
        assert_eq!(format_ty(&Ty::Infer), "?");
        assert_eq!(format_ty(&Ty::Error), "?");
    }

    #[test]
    fn test_format_ty_concrete() {
        assert_eq!(format_ty(&Ty::Int), "Int");
        assert_eq!(
            format_ty(&Ty::List(Box::new(Ty::String))),
            "List<String>"
        );
    }

    #[test]
    fn test_analyze_script_context_types() {
        let ctx = HashMap::new();
        let result = do_analyze_test("@x + 1", "script", &ctx);
        if !result.ok {
            eprintln!("errors: {:?}", result.errors);
        }
        assert!(result.ok);
        assert_eq!(result.context_keys.len(), 1);
        assert_eq!(result.context_keys[0].name, "x");
        assert_eq!(result.context_keys[0].ty, "Int");
        assert_eq!(result.tail_type, "Int");
    }

    #[test]
    fn test_analyze_with_provided_context_types() {
        let mut ctx = HashMap::new();
        ctx.insert("x".to_string(), Ty::Int);
        let result = do_analyze_test("@x + 1", "script", &ctx);
        assert!(result.ok);
        assert_eq!(result.context_keys[0].ty, "Int");
    }

    #[test]
    fn test_analyze_template() {
        let ctx = HashMap::new();
        let result = do_analyze_test("hello {{ @name }}", "template", &ctx);
        assert!(result.ok);
        assert_eq!(result.context_keys.len(), 1);
        assert_eq!(result.context_keys[0].name, "name");
        assert_eq!(result.tail_type, "String");
    }

    #[test]
    fn test_parse_ty_list_object() {
        let ty = parse_ty("List<{content: String, content_type: String, role: String}>").unwrap();
        let mut fields = BTreeMap::new();
        fields.insert("content".to_string(), Ty::String);
        fields.insert("content_type".to_string(), Ty::String);
        fields.insert("role".to_string(), Ty::String);
        assert_eq!(ty, Ty::List(Box::new(Ty::Object(fields))));
    }

    #[test]
    fn test_tail_type_mismatch() {
        let ctx = HashMap::new();
        let expected = parse_ty("Int").unwrap();
        let result = do_analyze("\"hello\"", "script", &ctx, Some(&expected));
        assert!(!result.ok, "should fail: String vs Int");
    }

    #[test]
    fn test_tail_type_match() {
        let ctx = HashMap::new();
        let expected = parse_ty("Int").unwrap();
        let result = do_analyze("1 + 2", "script", &ctx, Some(&expected));
        assert!(result.ok, "should succeed: Int vs Int");
    }

    #[test]
    fn test_analyze_unresolved_type() {
        let ctx = HashMap::new();
        let result = do_analyze_test("@x", "script", &ctx);
        if !result.ok {
            eprintln!("errors: {:?}", result.errors);
        }
        assert!(result.ok);
        assert_eq!(result.context_keys[0].ty, "?");
    }
}
