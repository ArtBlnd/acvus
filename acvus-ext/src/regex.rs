use acvus_interpreter::{IntoValue, OpaqueValue, RuntimeError, Value};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

const OPAQUE_NAME: &str = "Regex";

fn opaque_ty() -> Ty {
    Ty::Opaque(OPAQUE_NAME.into())
}

fn extract_regex(v: &Value) -> &regex::Regex {
    let Value::Opaque(o) = v else {
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
    });
    types.insert(interner.intern("regex_match"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::Bool),
        is_extern: true,
    });
    types.insert(interner.intern("regex_find"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::String),
        is_extern: true,
    });
    types.insert(interner.intern("regex_find_all"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        is_extern: true,
    });
    types.insert(interner.intern("regex_replace"), Ty::Fn {
        params: vec![Ty::String, opaque_ty(), Ty::String],
        ret: Box::new(Ty::String),
        is_extern: true,
    });
    types.insert(interner.intern("regex_split"), Ty::Fn {
        params: vec![opaque_ty(), Ty::String],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        is_extern: true,
    });
    types.insert(interner.intern("regex_extract"), Ty::Fn {
        params: vec![Ty::String, opaque_ty()],
        ret: Box::new(Ty::List(Box::new(Ty::String))),
        is_extern: true,
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
            let Value::String(pattern) = &args[0] else {
                return Err(RuntimeError::type_mismatch(
                    "regex",
                    "String",
                    &format!("{:?}", args[0]),
                ));
            };
            Ok(Value::Opaque(OpaqueValue::new(
                OPAQUE_NAME,
                compile_regex(pattern),
            )))
        }
        "regex_match" => {
            let re = extract_regex(&args[0]);
            let Value::String(s) = &args[1] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_match",
                    "String",
                    &format!("{:?}", args[1]),
                ));
            };
            Ok(Value::Bool(re.is_match(s)))
        }
        "regex_find" => {
            acvus_interpreter::set_interner_ctx(interner);
            let re = extract_regex(&args[0]);
            let Value::String(s) = &args[1] else {
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
            let Value::String(s) = &args[1] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_find_all",
                    "String",
                    &format!("{:?}", args[1]),
                ));
            };
            let matches: Vec<Value> = re
                .find_iter(s)
                .map(|m| Value::String(m.as_str().to_string()))
                .collect();
            Ok(Value::List(matches))
        }
        "regex_replace" => {
            let Value::String(s) = &args[0] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_replace",
                    "String",
                    &format!("{:?}", args[0]),
                ));
            };
            let re = extract_regex(&args[1]);
            let Value::String(rep) = &args[2] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_replace",
                    "String",
                    &format!("{:?}", args[2]),
                ));
            };
            Ok(Value::String(re.replace_all(s, rep.as_str()).into_owned()))
        }
        "regex_split" => {
            let re = extract_regex(&args[0]);
            let Value::String(s) = &args[1] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_split",
                    "String",
                    &format!("{:?}", args[1]),
                ));
            };
            let parts: Vec<Value> = re.split(s).map(|p| Value::String(p.to_string())).collect();
            Ok(Value::List(parts))
        }
        "regex_extract" => {
            let Value::String(s) = &args[0] else {
                return Err(RuntimeError::type_mismatch(
                    "regex_extract",
                    "String",
                    &format!("{:?}", args[0]),
                ));
            };
            let re = extract_regex(&args[1]);
            let parts: Vec<Value> = re
                .captures_iter(s)
                .filter_map(|c| c.get(1).map(|m| Value::String(m.as_str().to_string())))
                .collect();
            Ok(Value::List(parts))
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
        let re = call(&interner, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_match",
            vec![re, Value::String("abc123".into())],
        )
        .await;
        assert_eq!(result, Value::Bool(true));
    }

    #[tokio::test]
    async fn match_no_hit() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_match",
            vec![re, Value::String("abc".into())],
        )
        .await;
        assert_eq!(result, Value::Bool(false));
    }

    #[tokio::test]
    async fn find_first() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_find",
            vec![re, Value::String("abc123def456".into())],
        )
        .await;
        let some_tag = interner.intern("Some");
        assert!(matches!(
            result,
            Value::Variant { ref tag, payload: Some(ref inner) }
            if *tag == some_tag && **inner == Value::String("123".into())
        ));
    }

    #[tokio::test]
    async fn find_all_matches() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_find_all",
            vec![re, Value::String("a1b22c333".into())],
        )
        .await;
        let Value::List(items) = result else {
            panic!("expected List");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Value::String("1".into()));
        assert_eq!(items[1], Value::String("22".into()));
        assert_eq!(items[2], Value::String("333".into()));
    }

    #[tokio::test]
    async fn replace_all() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::String(r"\s+".into())]).await;
        let result = call(
            &interner,
            "regex_replace",
            vec![
                Value::String("hello   world  !".into()),
                re,
                Value::String(" ".into()),
            ],
        )
        .await;
        assert_eq!(result, Value::String("hello world !".into()));
    }

    #[tokio::test]
    async fn split_by_pattern() {
        let interner = setup();
        let re = call(
            &interner,
            "regex",
            vec![Value::String(r"[,;]\s*".into())],
        )
        .await;
        let result = call(
            &interner,
            "regex_split",
            vec![re, Value::String("a, b;c; d".into())],
        )
        .await;
        let Value::List(items) = result else {
            panic!("expected List");
        };
        assert_eq!(
            items,
            vec![
                Value::String("a".into()),
                Value::String("b".into()),
                Value::String("c".into()),
                Value::String("d".into()),
            ]
        );
    }

    #[tokio::test]
    async fn extract_capture_groups() {
        let interner = setup();
        let re = call(
            &interner,
            "regex",
            vec![Value::String(r"(?s)<thinking>(.*?)</thinking>".into())],
        )
        .await;
        let result = call(
            &interner,
            "regex_extract",
            vec![
                Value::String(
                    "hello <thinking>inner1</thinking> mid <thinking>inner2</thinking> end".into(),
                ),
                re,
            ],
        )
        .await;
        let Value::List(items) = result else {
            panic!("expected List");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::String("inner1".into()));
        assert_eq!(items[1], Value::String("inner2".into()));
    }

    #[tokio::test]
    async fn extract_no_capture_group() {
        let interner = setup();
        let re = call(&interner, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &interner,
            "regex_extract",
            vec![Value::String("abc123def".into()), re],
        )
        .await;
        let Value::List(items) = result else {
            panic!("expected List");
        };
        assert!(items.is_empty());
    }
}
