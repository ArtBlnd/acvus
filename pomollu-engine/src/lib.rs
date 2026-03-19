pub mod builtin_types;
mod config;
mod convert;
pub mod error;
mod fetch;
mod history;
mod idb;
mod language_session;
pub mod schema;
mod session;

#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn init() {
    console_error_panic_hook::set_once();
}

use acvus_mir::ir::{InstKind};
use acvus_mir::analysis::reachable_context::KnownValue;
use acvus_mir::ty::{Effect, FnKind, Ty};
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap};
use tsify::Ts;
use tsify::Tsify;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;

use acvus_mir::context_registry::{ContextTypeRegistry, PartialContextTypeRegistry, RegistryConflictError};
use schema::*;

/// Compile-time context types for asset extern functions.
pub(crate) fn asset_context_types(interner: &Interner) -> FxHashMap<Astr, Ty> {
    let mut types = FxHashMap::default();
    types.insert(interner.intern("asset_url"), Ty::Fn {
        params: vec![Ty::String],
        ret: Box::new(Ty::Option(Box::new(Ty::String))),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types
}

/// Build a registry with extern fns, empty system, and user-provided types.
pub(crate) fn build_registry(
    interner: &Interner,
    user_types: FxHashMap<Astr, Ty>,
) -> Result<PartialContextTypeRegistry, RegistryConflictError> {
    let mut extern_fns = acvus_ext::regex_context_types(interner);
    extern_fns.extend(asset_context_types(interner));
    PartialContextTypeRegistry::new(extern_fns, FxHashMap::default(), user_types)
}


/// Try to compile a short script and extract a known value (literal or variant).
/// Returns None if the script is not a simple constant expression.
pub(crate) fn try_extract_known(
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

/// Convert a map of `name -> TypeDesc` into `HashMap<Astr, Ty>`.
pub(crate) fn convert_context_types(interner: &Interner, raw: &FxHashMap<String, TypeDesc>) -> FxHashMap<Astr, Ty> {
    raw.iter()
        .map(|(k, v)| (interner.intern(k), desc_to_ty(interner, v)))
        .collect()
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
    let ctx: FxHashMap<Astr, acvus_interpreter::TypedValue> = options.context
        .into_iter()
        .map(|(k, v)| {
            let ty = jcv_to_ty(&interner, &v);
            let cv: acvus_interpreter::ConcreteValue = v.into();
            (interner.intern(&k), acvus_interpreter::TypedValue::from_concrete(&cv, &interner, ty))
        })
        .collect();

    let interp = Interpreter::new(&interner, module);
    let emits = interp.execute_with_context(ctx).await;

    let result_value = match &options.mode {
        Mode::Template => {
            let mut output = String::new();
            for tv in &emits {
                match tv.value() {
                    Value::Pure(acvus_interpreter::PureValue::String(s)) => output.push_str(s),
                    other => panic!("template emit: expected String, got {other:?}"),
                }
            }
            JsConcreteValue::String { v: output }
        }
        Mode::Script => {
            assert!(emits.len() <= 1, "script emitted {} values, expected at most 1", emits.len());
            match emits.into_iter().next() {
                Some(tv) => tv.to_concrete(&interner).into(),
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
                .unwrap_or_else(Ty::error);
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
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::error())), r#"{"kind":"unsupported","raw":"?"}"#);
        assert_eq!(desc_json(&ty_to_desc(&interner, &Ty::error())), r#"{"kind":"unsupported","raw":"?"}"#);
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
}
