//! Regex extension functions via ExternRegistry.

use acvus_interpreter::iter::IterHandle;
use acvus_interpreter::{
    Defs, ExternFn, ExternRegistry, FromValue, IntoValue, OpaqueValue, RuntimeError, Uses, Value,
    ValueKind,
};
use acvus_mir::ty::{Effect, Ty};
use acvus_utils::Interner;

const OPAQUE_NAME: &str = "Regex";

fn opaque_ty() -> Ty {
    Ty::Opaque(OPAQUE_NAME.into())
}

/// Newtype for `regex::Regex` — enables typed FromValue/IntoValue conversion.
struct Re(regex::Regex);

impl FromValue for Re {
    fn from_value(value: Value) -> Result<Self, RuntimeError> {
        match value {
            Value::Opaque(o) => {
                let r = o.downcast_ref::<regex::Regex>().ok_or_else(|| {
                    RuntimeError::unexpected_type(
                        "FromValue<Re>",
                        &[ValueKind::Opaque],
                        ValueKind::Opaque,
                    )
                })?;
                Ok(Re(r.clone()))
            }
            other => Err(RuntimeError::unexpected_type(
                "FromValue<Re>",
                &[ValueKind::Opaque],
                other.kind(),
            )),
        }
    }
}

impl IntoValue for Re {
    fn into_value(self) -> Value {
        Value::opaque(OpaqueValue::new(OPAQUE_NAME, self.0))
    }
}

/// Build the regex ExternRegistry.
pub fn regex_registry() -> ExternRegistry {
    ExternRegistry::new(|_interner| {
        vec![
            // regex(pattern) -> Opaque<Regex>
            ExternFn::build("regex")
                .params(vec![Ty::String])
                .ret(opaque_ty())
                .pure()
                .handler(
                    |_interner: &Interner, (pattern,): (String,), Uses(()): Uses<()>| {
                        let re = regex::Regex::new(&pattern)
                            .unwrap_or_else(|e| panic!("regex: invalid pattern '{pattern}': {e}"));
                        Ok((Re(re), Defs(())))
                    },
                ),
            // regex_match(re, text) -> Bool
            ExternFn::build("regex_match")
                .params(vec![opaque_ty(), Ty::String])
                .ret(Ty::Bool)
                .pure()
                .handler(
                    |_interner: &Interner, (Re(re), text): (Re, String), Uses(()): Uses<()>| {
                        Ok((re.is_match(&text), Defs(())))
                    },
                ),
            // regex_find(re, text) -> Option<String>
            ExternFn::build("regex_find")
                .params(vec![opaque_ty(), Ty::String])
                .ret(Ty::String) // TODO: proper Option<String> return type
                .pure()
                .handler(
                    |interner: &Interner, (Re(re), text): (Re, String), Uses(()): Uses<()>| {
                        let result = match re.find(&text) {
                            Some(m) => Value::some(interner, Value::string(m.as_str())),
                            None => Value::none(interner),
                        };
                        Ok((result, Defs(())))
                    },
                ),
            // regex_find_all(re, text) -> Iterator<String, SelfModifying>
            ExternFn::build("regex_find_all")
                .params(vec![opaque_ty(), Ty::String])
                .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
                .pure()
                .handler(
                    |_interner: &Interner, (Re(re), text): (Re, String), Uses(()): Uses<()>| {
                        let mut start = 0;
                        let iter = Value::iterator(IterHandle::from_fn(
                            Effect::self_modifying(),
                            move || {
                                let m = re.find_at(&text, start)?;
                                start = m.end();
                                Some(Value::string(m.as_str()))
                            },
                        ));
                        Ok((iter, Defs(())))
                    },
                ),
            // regex_replace(text, re, replacement) -> String
            ExternFn::build("regex_replace")
                .params(vec![Ty::String, opaque_ty(), Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner,
                     (text, Re(re), rep): (String, Re, String),
                     Uses(()): Uses<()>| {
                        Ok((re.replace_all(&text, rep.as_str()).into_owned(), Defs(())))
                    },
                ),
            // regex_split(re, text) -> Iterator<String, SelfModifying>
            ExternFn::build("regex_split")
                .params(vec![opaque_ty(), Ty::String])
                .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
                .pure()
                .handler(
                    |_interner: &Interner, (Re(re), text): (Re, String), Uses(()): Uses<()>| {
                        let mut last_end = 0;
                        let mut done = false;
                        let iter = Value::iterator(IterHandle::from_fn(
                            Effect::self_modifying(),
                            move || {
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
                                        Some(Value::string(&text[last_end..]))
                                    }
                                }
                            },
                        ));
                        Ok((iter, Defs(())))
                    },
                ),
            // regex_extract(text, re) -> Iterator<String, SelfModifying>  (capture group 1)
            ExternFn::build("regex_extract")
                .params(vec![Ty::String, opaque_ty()])
                .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
                .pure()
                .handler(
                    |_interner: &Interner, (text, Re(re)): (String, Re), Uses(()): Uses<()>| {
                        let mut start = 0;
                        let iter = Value::iterator(IterHandle::from_fn(
                            Effect::self_modifying(),
                            move || {
                                loop {
                                    let caps = re.captures_at(&text, start)?;
                                    let full = caps.get(0)?;
                                    start = full.end();
                                    if let Some(group1) = caps.get(1) {
                                        return Some(Value::string(group1.as_str()));
                                    }
                                }
                            },
                        ));
                        Ok((iter, Defs(())))
                    },
                ),
        ]
    })
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
