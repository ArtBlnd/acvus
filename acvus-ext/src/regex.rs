//! Regex extension functions via ExternRegistry.

use acvus_interpreter::iter::IterHandle;
use acvus_interpreter::{ExternFn, ExternRegistry, OpaqueValue, Value};
use acvus_mir::ty::{Effect, Ty};

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
    regex::Regex::new(pattern)
        .unwrap_or_else(|e| panic!("regex: invalid pattern '{pattern}': {e}"))
}

/// Build the regex ExternRegistry.
pub fn regex_registry() -> ExternRegistry {
    ExternRegistry::new(|_interner| vec![
        // regex(pattern) -> Opaque<Regex>
        ExternFn::build("regex")
            .params(vec![Ty::String])
            .ret(opaque_ty())
            .pure()
            .sync_handler(|args, _interner| {
                let pattern = args[0].as_str();
                Ok(Value::opaque(OpaqueValue::new(
                    OPAQUE_NAME,
                    compile_regex(pattern),
                )))
            }),

        // regex_match(re, text) -> Bool
        ExternFn::build("regex_match")
            .params(vec![opaque_ty(), Ty::String])
            .ret(Ty::Bool)
            .pure()
            .sync_handler(|args, _interner| {
                let re = extract_regex(&args[0]);
                let text = args[1].as_str();
                Ok(Value::Bool(re.is_match(text)))
            }),

        // regex_find(re, text) -> Option<String>
        ExternFn::build("regex_find")
            .params(vec![opaque_ty(), Ty::String])
            .ret(Ty::String) // TODO: proper Option<String> return type
            .pure()
            .sync_handler(|args, interner| {
                let re = extract_regex(&args[0]);
                let text = args[1].as_str();
                match re.find(text) {
                    Some(m) => Ok(Value::some(interner, Value::string(m.as_str()))),
                    None => Ok(Value::none(interner)),
                }
            }),

        // regex_find_all(re, text) -> Iterator<String, IO>
        ExternFn::build("regex_find_all")
            .params(vec![opaque_ty(), Ty::String])
            .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
            .pure()
            .sync_handler(|args, _interner| {
                let re = extract_regex(&args[0]).clone();
                let text = args[1].as_str().to_string();
                let mut start = 0;
                Ok(Value::iterator(IterHandle::from_fn(Effect::self_modifying(), move || {
                    let m = re.find_at(&text, start)?;
                    start = m.end();
                    Some(Value::string(m.as_str()))
                })))
            }),

        // regex_replace(text, re, replacement) -> String
        ExternFn::build("regex_replace")
            .params(vec![Ty::String, opaque_ty(), Ty::String])
            .ret(Ty::String)
            .pure()
            .sync_handler(|args, _interner| {
                let text = args[0].as_str();
                let re = extract_regex(&args[1]);
                let rep = args[2].as_str();
                Ok(Value::string(re.replace_all(text, rep).into_owned()))
            }),

        // regex_split(re, text) -> Iterator<String, IO>
        ExternFn::build("regex_split")
            .params(vec![opaque_ty(), Ty::String])
            .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
            .pure()
            .sync_handler(|args, _interner| {
                let re = extract_regex(&args[0]).clone();
                let text = args[1].as_str().to_string();
                // Split produces all segments — collect lazily via find boundaries.
                let mut last_end = 0;
                let mut done = false;
                Ok(Value::iterator(IterHandle::from_fn(Effect::self_modifying(), move || {
                    if done {
                        return None;
                    }
                    match re.find_at(&text, last_end) {
                        Some(m) => {
                            let segment = &text[last_end..m.start()];
                            last_end = m.end();
                            Some(Value::string(segment))
                        }
                        None => {
                            done = true;
                            // Emit the trailing segment.
                            Some(Value::string(&text[last_end..]))
                        }
                    }
                })))
            }),

        // regex_extract(text, re) -> Iterator<String, IO>  (capture group 1)
        ExternFn::build("regex_extract")
            .params(vec![Ty::String, opaque_ty()])
            .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
            .pure()
            .sync_handler(|args, _interner| {
                let text = args[0].as_str().to_string();
                let re = extract_regex(&args[1]).clone();
                let mut start = 0;
                Ok(Value::iterator(IterHandle::from_fn(Effect::self_modifying(), move || {
                    loop {
                        let caps = re.captures_at(&text, start)?;
                        let full = caps.get(0)?;
                        start = full.end();
                        // Return capture group 1 if present.
                        if let Some(group1) = caps.get(1) {
                            return Some(Value::string(group1.as_str()));
                        }
                        // No group 1 — skip this match.
                    }
                })))
            }),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = regex_registry();
        let registered = reg.register(&i);
        assert_eq!(registered.functions.len(), 7);
        assert_eq!(registered.executables.len(), 7);
    }
}
