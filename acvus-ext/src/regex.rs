use acvus_interpreter::{IntoValue, OpaqueValue, PureValue, RuntimeError, UnpureValue, Value};
#[cfg(test)]
use acvus_interpreter::LazyValue;
use acvus_mir::ty::Ty;
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
        is_extern: true,
        captures: vec![],
    });
    types.insert(interner.intern("regex_match"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::Bool),
        is_extern: true,
        captures: vec![],
    });
    types.insert(interner.intern("regex_find"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::String),
        is_extern: true,
        captures: vec![],
    });
    types.insert(interner.intern("regex_find_all"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        is_extern: true,
        captures: vec![],
    });
    types.insert(interner.intern("regex_replace"), Ty::Fn {
        params: vec![Ty::String, opaque_ty(), Ty::String],
        ret: Box::new(Ty::String),
        is_extern: true,
        captures: vec![],
    });
    types.insert(interner.intern("regex_split"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        is_extern: true,
        captures: vec![],
    });
    types.insert(interner.intern("regex_extract"), Ty::Fn {
        params: vec![Ty::String, opaque_ty()],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        is_extern: true,
        captures: vec![],
    });
    types
}

/// Runtime dispatch for regex extern functions.
pub async fn regex_call(
    interner: &Interner,
    name: Astr,
    args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    let name_str = interner.resolve(name);
    match name_str {
        "regex" => {
            let Value::Pure(PureValue::String(pattern)) = &args[0] else {
                return Err(RuntimeError::type_mismatch(
                    "regex",
                    "String",
                    &format!("{:?}", args[0]),
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
                return Err(RuntimeError::type_mismatch(
                    "regex_match",
                    "String",
                    &format!("{:?}", args[1]),
                ));
            };
            Ok(Value::bool_(re.is_match(s)))
        }
        "regex_find" => {
            acvus_interpreter::set_interner_ctx(interner);
            let re = extract_regex(&args[0]);
            let Value::Pure(PureValue::String(s)) = &args[1] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_find",
                    "String",
                    &format!("{:?}", args[1]),
                ));
            };
            let result: Option<String> = re.find(s).map(|m| m.as_str().to_string());
            Ok(result.into_value())
        }
        "regex_find_all" => {
            let re = extract_regex(&args[0]);
            let Value::Pure(PureValue::String(s)) = &args[1] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_find_all",
                    "String",
                    &format!("{:?}", args[1]),
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
                return Err(RuntimeError::type_mismatch(
                    "regex_replace",
                    "String",
                    &format!("{:?}", args[0]),
                ));
            };
            let re = extract_regex(&args[1]);
            let Value::Pure(PureValue::String(rep)) = &args[2] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_replace",
                    "String",
                    &format!("{:?}", args[2]),
                ));
            };
            Ok(Value::string(re.replace_all(s, rep.as_str()).into_owned()))
        }
        "regex_split" => {
            let re = extract_regex(&args[0]);
            let Value::Pure(PureValue::String(s)) = &args[1] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_split",
                    "String",
                    &format!("{:?}", args[1]),
                ));
            };
            let parts: Vec<Value> = re.split(s).map(|p| Value::string(p.to_string())).collect();
            Ok(Value::list(parts))
        }
        "regex_extract" => {
            let Value::Pure(PureValue::String(s)) = &args[0] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_extract",
                    "String",
                    &format!("{:?}", args[0]),
                ));
            };
            let re = extract_regex(&args[1]);
            let parts: Vec<Value> = re
                .captures_iter(s)
                .filter_map(|c| c.get(1).map(|m| Value::string(m.as_str().to_string())))
                .collect();
            Ok(Value::list(parts))
        }
        _ => Err(RuntimeError::other(&format!(
            "unknown regex function: {name_str}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Interner {
        Interner::new()
    }

    async fn call(interner: &Interner, name: &str, args: Vec<Value>) -> Value {
        acvus_interpreter::set_interner_ctx(interner);
        regex_call(interner, interner.intern(name), args)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn compile_and_match() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::string(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_match",
            vec![re, Value::string("abc123".into())],
        )
        .await;
        assert_eq!(result, Value::bool_(true));
    }

    #[tokio::test]
    async fn match_no_hit() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::string(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_match",
            vec![re, Value::string("abc".into())],
        )
        .await;
        assert_eq!(result, Value::bool_(false));
    }

    #[tokio::test]
    async fn find_first() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::string(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_find",
            vec![re, Value::string("abc123def456".into())],
        )
        .await;
        let some_tag = interner.intern("Some");
        assert!(matches!(
            result,
            Value::Lazy(LazyValue::Variant { ref tag, payload: Some(ref inner) })
            if *tag == some_tag && **inner == Value::string("123".into())
        ));
    }

    #[tokio::test]
    async fn find_all_matches() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::string(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_find_all",
            vec![re, Value::string("a1b22c333".into())],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result else {
            panic!("expected List");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Value::string("1".into()));
        assert_eq!(items[1], Value::string("22".into()));
        assert_eq!(items[2], Value::string("333".into()));
    }

    #[tokio::test]
    async fn replace_all() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::string(r"\s+".into())]).await;
        let result = call(
            &interner,
            "regex_replace",
            vec![
                Value::string("hello   world  !".into()),
                re,
                Value::string(" ".into()),
            ],
        )
        .await;
        assert_eq!(result, Value::string("hello world !".into()));
    }

    #[tokio::test]
    async fn split_by_pattern() {
        let interner = setup();
        let re = call(
            &interner,
            "regex",
            vec![Value::string(r"[,;]\s*".into())],
        )
        .await;
        let result = call(
            &interner,
            "regex_split",
            vec![re, Value::string("a, b;c; d".into())],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result else {
            panic!("expected List");
        };
        assert_eq!(
            items,
            vec![
                Value::string("a".into()),
                Value::string("b".into()),
                Value::string("c".into()),
                Value::string("d".into()),
            ]
        );
    }

    #[tokio::test]
    async fn extract_capture_groups() {
        let interner = setup();
        let re = call(
            &interner,
            "regex",
            vec![Value::string(r"(?s)<thinking>(.*?)</thinking>".into())],
        )
        .await;
        let result = call(
            &interner,
            "regex_extract",
            vec![
                Value::string(
                    "hello <thinking>inner1</thinking> mid <thinking>inner2</thinking> end".into(),
                ),
                re,
            ],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result else {
            panic!("expected List");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::string("inner1".into()));
        assert_eq!(items[1], Value::string("inner2".into()));
    }

    #[tokio::test]
    async fn extract_no_capture_group() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::string(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_extract",
            vec![Value::string("abc123def".into()), re],
        )
        .await;
        let Value::Lazy(LazyValue::List(items)) = result else {
            panic!("expected List");
        };
        assert!(items.is_empty());
    }
}
