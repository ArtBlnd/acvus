use acvus_interpreter::{IntoValue, OpaqueValue, PureValue, RuntimeError, UnpureValue, Value, ValueKind};
#[cfg(test)]
use acvus_interpreter::{LazyValue, TypedValue};
#[cfg(test)]
use std::sync::Arc;
use acvus_mir::ty::{Effect, FnKind, Ty};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

const OPAQUE_NAME: &str = "Regex";

fn opaque_ty() -> Ty {
    Ty::Opaque(OPAQUE_NAME.into())
}

fn extract_regex(v: &Value) -> &regex::Regex {
    let Value::Unpure(UnpureValue::Opaque(o)) = v else {
        panic!("expected Opaque<Regex>, got {v:?}");
    };
    o.downcast_ref::<regex::Regex>()
        .expect("opaque value is not a Regex")
}

fn compile_regex(pattern: &str) -> regex::Regex {
    regex::Regex::new(pattern).unwrap_or_else(|e| panic!("regex: invalid pattern '{pattern}': {e}"))
}

/// Build the compile-time context types for regex functions.
pub fn regex_context_types(interner: &Interner) -> FxHashMap<Astr, Ty> {
    let mut types = FxHashMap::default();
    types.insert(interner.intern("regex"), Ty::Fn {
        params: vec![Ty::String],
        ret: Box::new(opaque_ty()),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types.insert(interner.intern("regex_match"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::Bool),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types.insert(interner.intern("regex_find"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::String),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types.insert(interner.intern("regex_find_all"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types.insert(interner.intern("regex_replace"), Ty::Fn {
        params: vec![Ty::String, opaque_ty(), Ty::String],
        ret: Box::new(Ty::String),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types.insert(interner.intern("regex_split"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types.insert(interner.intern("regex_extract"), Ty::Fn {
        params: vec![Ty::String, opaque_ty()],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    });
    types
}

/// Runtime dispatch for regex extern functions.
pub async fn regex_call(
    interner: &Interner,
    name: Astr,
    args: Vec<&Value>,
) -> Result<Value, RuntimeError> {
    let name_str = interner.resolve(name);
    match name_str {
        "regex" => {
            let Value::Pure(PureValue::String(pattern)) = &args[0] else {
                return Err(RuntimeError::unexpected_type(
                    "regex",
                    &[ValueKind::String],
                    args[0].kind(),
                ));
            };
            Ok(Value::opaque(OpaqueValue::new(
                OPAQUE_NAME,
                compile_regex(pattern),
            )))
        }
        "regex_match" => {
            let re = extract_regex(&args[0]);
            let Value::Pure(PureValue::String(s)) = &args[1] else {
                return Err(RuntimeError::unexpected_type(
                    "regex_match",
                    &[ValueKind::String],
                    args[1].kind(),
                ));
            };
            Ok(Value::bool_(re.is_match(s)))
        }
        "regex_find" => {
            acvus_interpreter::set_interner_ctx(interner);
            let re = extract_regex(&args[0]);
            let Value::Pure(PureValue::String(s)) = &args[1] else {
                return Err(RuntimeError::unexpected_type(
                    "regex_find",
                    &[ValueKind::String],
                    args[1].kind(),
                ));
            };
            let result: Option<String> = re.find(s).map(|m| m.as_str().to_string());
            Ok(result.into_value())
        }
        "regex_find_all" => {
            let re = extract_regex(&args[0]);
            let Value::Pure(PureValue::String(s)) = &args[1] else {
                return Err(RuntimeError::unexpected_type(
                    "regex_find_all",
                    &[ValueKind::String],
                    args[1].kind(),
                ));
            };
            let matches: Vec<Value> = re
                .find_iter(s)
                .map(|m| Value::string(m.as_str().to_string()))
                .collect();
            Ok(Value::list(matches))
        }
        "regex_replace" => {
            let Value::Pure(PureValue::String(s)) = &args[0] else {
                return Err(RuntimeError::unexpected_type(
                    "regex_replace",
                    &[ValueKind::String],
                    args[0].kind(),
                ));
            };
            let re = extract_regex(&args[1]);
            let Value::Pure(PureValue::String(rep)) = &args[2] else {
                return Err(RuntimeError::unexpected_type(
                    "regex_replace",
                    &[ValueKind::String],
                    args[2].kind(),
                ));
            };
            Ok(Value::string(re.replace_all(s, rep.as_str()).into_owned()))
        }
        "regex_split" => {
            let re = extract_regex(&args[0]);
            let Value::Pure(PureValue::String(s)) = &args[1] else {
                return Err(RuntimeError::unexpected_type(
                    "regex_split",
                    &[ValueKind::String],
                    args[1].kind(),
                ));
            };
            let parts: Vec<Value> = re.split(s).map(|p| Value::string(p.to_string())).collect();
            Ok(Value::list(parts))
        }
        "regex_extract" => {
            let Value::Pure(PureValue::String(s)) = &args[0] else {
                return Err(RuntimeError::unexpected_type(
                    "regex_extract",
                    &[ValueKind::String],
                    args[0].kind(),
                ));
            };
            let re = extract_regex(&args[1]);
            let parts: Vec<Value> = re
                .captures_iter(s)
                .filter_map(|c| c.get(1).map(|m| Value::string(m.as_str().to_string())))
                .collect();
            Ok(Value::list(parts))
        }
        _ => Err(RuntimeError::extern_call(name_str, "unknown regex function")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Interner {
        Interner::new()
    }

    async fn call(interner: &Interner, name: &str, args: Vec<TypedValue>) -> TypedValue {
        acvus_interpreter::set_interner_ctx(interner);
        let refs: Vec<&Value> = args.iter().map(|tv| tv.value()).collect();
        let value = regex_call(interner, interner.intern(name), refs)
            .await
            .unwrap();
        TypedValue::new(value, Ty::error())
    }

    #[tokio::test]
    async fn compile_and_match() {
        let interner = setup();
        let re = call(&interner, "regex", vec![TypedValue::string(r"\d+")]).await;
        let result = call(
            &interner,
            "regex_match",
            vec![re, TypedValue::string("abc123")],
        )
        .await;
        assert_eq!(*result.value(), Value::bool_(true));
    }

    #[tokio::test]
    async fn match_no_hit() {
        let interner = setup();
        let re = call(&interner, "regex", vec![TypedValue::string(r"\d+")]).await;
        let result = call(
            &interner,
            "regex_match",
            vec![re, TypedValue::string("abc")],
        )
        .await;
        assert_eq!(*result.value(), Value::bool_(false));
    }

    #[tokio::test]
    async fn find_first() {
        let interner = setup();
        let re = call(&interner, "regex", vec![TypedValue::string(r"\d+")]).await;
        let result = call(
            &interner,
            "regex_find",
            vec![re, TypedValue::string("abc123def456")],
        )
        .await;
        let some_tag = interner.intern("Some");
        let value = result.value();
        assert!(matches!(
            value,
            Value::Lazy(LazyValue::Variant { tag, payload: Some(inner) })
            if *tag == some_tag && **inner == Value::string("123")
        ));
    }

    #[tokio::test]
    async fn find_all_matches() {
        let interner = setup();
        let re = call(&interner, "regex", vec![TypedValue::string(r"\d+")]).await;
        let result = call(
            &interner,
            "regex_find_all",
            vec![re, TypedValue::string("a1b22c333")],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result.value() else {
            panic!("expected List");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Value::string("1"));
        assert_eq!(items[1], Value::string("22"));
        assert_eq!(items[2], Value::string("333"));
    }

    #[tokio::test]
    async fn replace_all() {
        let interner = setup();
        let re = call(&interner, "regex", vec![TypedValue::string(r"\s+")]).await;
        let result = call(
            &interner,
            "regex_replace",
            vec![
                TypedValue::string("hello   world  !"),
                re,
                TypedValue::string(" "),
            ],
        )
        .await;
        assert_eq!(*result.value(), Value::string("hello world !"));
    }

    #[tokio::test]
    async fn split_by_pattern() {
        let interner = setup();
        let re = call(
            &interner,
            "regex",
            vec![TypedValue::string(r"[,;]\s*")],
        )
        .await;
        let result = call(
            &interner,
            "regex_split",
            vec![re, TypedValue::string("a, b;c; d")],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result.value() else {
            panic!("expected List");
        };
        assert_eq!(
            *items,
            vec![
                Value::string("a"),
                Value::string("b"),
                Value::string("c"),
                Value::string("d"),
            ]
        );
    }

    #[tokio::test]
    async fn extract_capture_groups() {
        let interner = setup();
        let re = call(
            &interner,
            "regex",
            vec![TypedValue::string(r"(?s)<thinking>(.*?)</thinking>")],
        )
        .await;
        let result = call(
            &interner,
            "regex_extract",
            vec![
                TypedValue::string(
                    "hello <thinking>inner1</thinking> mid <thinking>inner2</thinking> end",
                ),
                re,
            ],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result.value() else {
            panic!("expected List");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::string("inner1"));
        assert_eq!(items[1], Value::string("inner2"));
    }

    #[tokio::test]
    async fn extract_no_capture_group() {
        let interner = setup();
        let re = call(&interner, "regex", vec![TypedValue::string(r"\d+")]).await;
        let result = call(
            &interner,
            "regex_extract",
            vec![TypedValue::string("abc123def"), re],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result.value() else {
            panic!("expected List");
        };
        assert!(items.is_empty());
    }
}
