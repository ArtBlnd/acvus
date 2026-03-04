use acvus_interpreter::{
    ExternFn, ExternFnBody, ExternFnRegistry, ExternFnSig, OpaqueValue, Value,
};
use acvus_mir::extern_module::ExternModule;
use acvus_mir::ty::Ty;

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

/// Build the compile-time `ExternModule` and register runtime functions.
pub fn regex_module(fn_reg: &mut ExternFnRegistry) -> ExternModule {
    let mut module = ExternModule::new("regex");
    module.add_opaque(OPAQUE_NAME);

    module.add_fn("regex", vec![Ty::String], opaque_ty(), false);
    module.add_fn(
        "regex_match",
        vec![opaque_ty(), Ty::String],
        Ty::Bool,
        false,
    );
    module.add_fn(
        "regex_find",
        vec![opaque_ty(), Ty::String],
        Ty::String,
        false,
    );
    module.add_fn(
        "regex_find_all",
        vec![opaque_ty(), Ty::String],
        Ty::List(Box::new(Ty::String)),
        false,
    );
    module.add_fn(
        "regex_replace",
        vec![opaque_ty(), Ty::String, Ty::String],
        Ty::String,
        false,
    );
    module.add_fn(
        "regex_split",
        vec![opaque_ty(), Ty::String],
        Ty::List(Box::new(Ty::String)),
        false,
    );

    fn_reg.register(RegexCompile);
    fn_reg.register(RegexMatch);
    fn_reg.register(RegexFind);
    fn_reg.register(RegexFindAll);
    fn_reg.register(RegexReplace);
    fn_reg.register(RegexSplit);

    module
}

// -- ExternFn implementations ------------------------------------------------

struct RegexCompile;
impl ExternFn for RegexCompile {
    fn name(&self) -> &str {
        "regex"
    }
    fn sig(&self) -> ExternFnSig {
        ExternFnSig {
            params: vec![Ty::String],
            ret: opaque_ty(),
            effectful: false,
        }
    }
    fn into_body(self) -> ExternFnBody {
        ExternFnBody::from_fn(|pattern: String| async move {
            Value::Opaque(OpaqueValue::new(OPAQUE_NAME, compile_regex(&pattern)))
        })
    }
}

struct RegexMatch;
impl ExternFn for RegexMatch {
    fn name(&self) -> &str {
        "regex_match"
    }
    fn sig(&self) -> ExternFnSig {
        ExternFnSig {
            params: vec![opaque_ty(), Ty::String],
            ret: Ty::Bool,
            effectful: false,
        }
    }
    fn into_body(self) -> ExternFnBody {
        ExternFnBody::from_fn(|re: Value, s: String| async move { extract_regex(&re).is_match(&s) })
    }
}

struct RegexFind;
impl ExternFn for RegexFind {
    fn name(&self) -> &str {
        "regex_find"
    }
    fn sig(&self) -> ExternFnSig {
        ExternFnSig {
            params: vec![opaque_ty(), Ty::String],
            ret: Ty::Option(Box::new(Ty::String)),
            effectful: false,
        }
    }
    fn into_body(self) -> ExternFnBody {
        ExternFnBody::from_fn(|re: Value, s: String| async move {
            extract_regex(&re).find(&s).map(|m| m.as_str().to_string())
        })
    }
}

struct RegexFindAll;
impl ExternFn for RegexFindAll {
    fn name(&self) -> &str {
        "regex_find_all"
    }
    fn sig(&self) -> ExternFnSig {
        ExternFnSig {
            params: vec![opaque_ty(), Ty::String],
            ret: Ty::List(Box::new(Ty::String)),
            effectful: false,
        }
    }
    fn into_body(self) -> ExternFnBody {
        ExternFnBody::from_fn(|re: Value, s: String| async move {
            let matches: Vec<Value> = extract_regex(&re)
                .find_iter(&s)
                .map(|m| Value::String(m.as_str().to_string()))
                .collect();
            Value::List(matches)
        })
    }
}

struct RegexReplace;
impl ExternFn for RegexReplace {
    fn name(&self) -> &str {
        "regex_replace"
    }
    fn sig(&self) -> ExternFnSig {
        ExternFnSig {
            params: vec![opaque_ty(), Ty::String, Ty::String],
            ret: Ty::String,
            effectful: false,
        }
    }
    fn into_body(self) -> ExternFnBody {
        ExternFnBody::from_fn(|re: Value, s: String, rep: String| async move {
            extract_regex(&re)
                .replace_all(&s, rep.as_str())
                .into_owned()
        })
    }
}

struct RegexSplit;
impl ExternFn for RegexSplit {
    fn name(&self) -> &str {
        "regex_split"
    }
    fn sig(&self) -> ExternFnSig {
        ExternFnSig {
            params: vec![opaque_ty(), Ty::String],
            ret: Ty::List(Box::new(Ty::String)),
            effectful: false,
        }
    }
    fn into_body(self) -> ExternFnBody {
        ExternFnBody::from_fn(|re: Value, s: String| async move {
            let parts: Vec<Value> = extract_regex(&re)
                .split(&s)
                .map(|p| Value::String(p.to_string()))
                .collect();
            Value::List(parts)
        })
    }
}

#[cfg(test)]
mod tests {
    use acvus_interpreter::ExternFnRegistry;
    use acvus_mir::extern_module::ExternRegistry;

    use super::*;

    fn setup() -> (ExternRegistry, ExternFnRegistry) {
        let mut fn_reg = ExternFnRegistry::new();
        let module = regex_module(&mut fn_reg);
        let mut mir_reg = ExternRegistry::new();
        mir_reg.register(&module);
        (mir_reg, fn_reg)
    }

    async fn call(fn_reg: &ExternFnRegistry, name: &str, args: Vec<Value>) -> Value {
        fn_reg.get(name).unwrap().call(args).await
    }

    #[tokio::test]
    async fn compile_and_match() {
        let (_mir, fns) = setup();
        let re = call(&fns, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &fns,
            "regex_match",
            vec![re, Value::String("abc123".into())],
        )
        .await;
        assert_eq!(result, Value::Bool(true));
    }

    #[tokio::test]
    async fn match_no_hit() {
        let (_mir, fns) = setup();
        let re = call(&fns, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(&fns, "regex_match", vec![re, Value::String("abc".into())]).await;
        assert_eq!(result, Value::Bool(false));
    }

    #[tokio::test]
    async fn find_first() {
        let (_mir, fns) = setup();
        let re = call(&fns, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &fns,
            "regex_find",
            vec![re, Value::String("abc123def456".into())],
        )
        .await;
        assert!(matches!(
            result,
            Value::Variant { ref tag, payload: Some(ref inner) }
            if tag == "Some" && **inner == Value::String("123".into())
        ));
    }

    #[tokio::test]
    async fn find_all_matches() {
        let (_mir, fns) = setup();
        let re = call(&fns, "regex", vec![Value::String(r"\d+".into())]).await;
        let result = call(
            &fns,
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
        let (_mir, fns) = setup();
        let re = call(&fns, "regex", vec![Value::String(r"\s+".into())]).await;
        let result = call(
            &fns,
            "regex_replace",
            vec![
                re,
                Value::String("hello   world  !".into()),
                Value::String(" ".into()),
            ],
        )
        .await;
        assert_eq!(result, Value::String("hello world !".into()));
    }

    #[tokio::test]
    async fn split_by_pattern() {
        let (_mir, fns) = setup();
        let re = call(&fns, "regex", vec![Value::String(r"[,;]\s*".into())]).await;
        let result = call(
            &fns,
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
}
