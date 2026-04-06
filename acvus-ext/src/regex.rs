//! Regex extension functions via ExternRegistry.

use acvus_interpreter::iter::IterHandle;
use acvus_interpreter::{
    Defs, ExternFnBuilder, ExternRegistry, FromValue, IntoValue, OpaqueValue, RuntimeError, Uses,
    Value, ValueKind,
};
use acvus_mir::graph::{Constraint, FnConstraint, QualifiedRef, Signature};
use acvus_mir::ty::{Effect, Param, Ty, TypeRegistry, UserDefinedDecl};
use acvus_utils::Interner;

fn user_defined_ty(id: QualifiedRef) -> Ty {
    Ty::UserDefined {
        id,
        type_args: vec![],
        effect_args: vec![],
    }
}

/// Newtype for `regex::Regex` — carries its `QualifiedRef` for type-safe conversion.
struct Re(regex::Regex, QualifiedRef);

impl FromValue for Re {
    fn from_value(value: Value) -> Result<Self, RuntimeError> {
        match value {
            Value::Opaque(o) => {
                let id = o.type_id;
                let r = o.downcast_ref::<regex::Regex>().ok_or_else(|| {
                    RuntimeError::unexpected_type(
                        "FromValue<Re>",
                        &[ValueKind::Opaque],
                        ValueKind::Opaque,
                    )
                })?;
                Ok(Re(r.clone(), id))
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
        Value::opaque(OpaqueValue::new(self.1, self.0))
    }
}

// ── Constraint builders ─────────────────────────────────────────────

fn sig(interner: &Interner, params: Vec<Ty>, ret: Ty) -> FnConstraint {
    let named: Vec<Param> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    FnConstraint {
        signature: Some(Signature {
            params: named.clone(),
        }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
        hint: None,
    }
}

/// Build the regex ExternRegistry.
/// Registers the `Regex` UserDefined type into `type_registry`.
pub fn regex_registry(interner: &Interner, type_registry: &mut TypeRegistry) -> ExternRegistry {
    let qref = QualifiedRef::root(interner.intern("Regex"));
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    type_registry.register(UserDefinedDecl {
        qref,
        type_params: vec![],
        effect_params: vec![],
    });

    let ty = user_defined_ty(qref);
    ExternRegistry::new(move |interner| {
        vec![
            // regex(pattern) -> Regex
            ExternFnBuilder::new("regex", sig(interner, vec![Ty::String], ty.clone())).handler(
                move |_interner: &Interner, (pattern,): (String,), Uses(()): Uses<()>| {
                    let re = regex::Regex::new(&pattern)
                        .unwrap_or_else(|e| panic!("regex: invalid pattern '{pattern}': {e}"));
                    Ok((Re(re, qref), Defs(())))
                },
            ),
            // regex_match(re, text) -> Bool
            ExternFnBuilder::new(
                "regex_match",
                sig(interner, vec![ty.clone(), Ty::String], Ty::Bool),
            )
            .handler(
                |_interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                    Ok((re.is_match(&text), Defs(())))
                },
            ),
            // regex_find(re, text) -> Option<String>
            ExternFnBuilder::new(
                "regex_find",
                sig(interner, vec![ty.clone(), Ty::String], Ty::String), // TODO: proper Option<String> return type
            )
            .handler(
                |interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                    let result = match re.find(&text) {
                        Some(m) => Value::some(interner, Value::string(m.as_str())),
                        None => Value::none(interner),
                    };
                    Ok((result, Defs(())))
                },
            ),
            // regex_find_all(re, text) -> Iterator<String, SelfModifying>
            ExternFnBuilder::new(
                "regex_find_all",
                sig(
                    interner,
                    vec![ty.clone(), Ty::String],
                    Ty::UserDefined {
                        id: iter_qref,
                        type_args: vec![Ty::String],
                        effect_args: vec![Effect::pure() /* TODO: Token(iter_qref) */],
                    },
                ),
            )
            .handler(
                |_interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                    let mut start = 0;
                    let iter =
                        Value::iterator(IterHandle::from_fn(Effect::pure() /* TODO: Token(iter_qref) */, move || {
                            let m = re.find_at(&text, start)?;
                            start = m.end();
                            Some(Value::string(m.as_str()))
                        }));
                    Ok((iter, Defs(())))
                },
            ),
            // regex_replace(text, re, replacement) -> String
            ExternFnBuilder::new(
                "regex_replace",
                sig(
                    interner,
                    vec![Ty::String, ty.clone(), Ty::String],
                    Ty::String,
                ),
            )
            .handler(
                |_interner: &Interner,
                 (text, Re(re, _), rep): (String, Re, String),
                 Uses(()): Uses<()>| {
                    Ok((re.replace_all(&text, rep.as_str()).into_owned(), Defs(())))
                },
            ),
            // regex_split(re, text) -> Iterator<String, SelfModifying>
            ExternFnBuilder::new(
                "regex_split",
                sig(
                    interner,
                    vec![ty.clone(), Ty::String],
                    Ty::UserDefined {
                        id: iter_qref,
                        type_args: vec![Ty::String],
                        effect_args: vec![Effect::pure() /* TODO: Token(iter_qref) */],
                    },
                ),
            )
            .handler(
                |_interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                    let mut last_end = 0;
                    let mut done = false;
                    let iter =
                        Value::iterator(IterHandle::from_fn(Effect::pure() /* TODO: Token(iter_qref) */, move || {
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
                        }));
                    Ok((iter, Defs(())))
                },
            ),
            // regex_extract(text, re) -> Iterator<String, SelfModifying>  (capture group 1)
            ExternFnBuilder::new(
                "regex_extract",
                sig(
                    interner,
                    vec![Ty::String, ty.clone()],
                    Ty::UserDefined {
                        id: iter_qref,
                        type_args: vec![Ty::String],
                        effect_args: vec![Effect::pure() /* TODO: Token(iter_qref) */],
                    },
                ),
            )
            .handler(
                |_interner: &Interner, (text, Re(re, _)): (String, Re), Uses(()): Uses<()>| {
                    let mut start = 0;
                    let iter =
                        Value::iterator(IterHandle::from_fn(Effect::pure() /* TODO: Token(iter_qref) */, move || {
                            loop {
                                let caps = re.captures_at(&text, start)?;
                                let full = caps.get(0)?;
                                start = full.end();
                                if let Some(group1) = caps.get(1) {
                                    return Some(Value::string(group1.as_str()));
                                }
                            }
                        }));
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
        let mut tr = TypeRegistry::new();
        let reg = regex_registry(&i, &mut tr);
        let registered = reg.register(&i);
        assert_eq!(registered.functions.len(), 7);
        assert_eq!(registered.executables.len(), 7);
    }
}
