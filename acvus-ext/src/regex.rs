//! Regex extension functions via ExternRegistry.

use crate::iter_pipeline::{IterHandle, iter_value};
use acvus_interpreter::{
    ExternFnBuilder, ExternRegistry, FromValue, IntoValue, ExternValue, RuntimeError,
    Value, ValueKind,
};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{ParamTerm, Poly, PolyTy, Ty, TyTerm, TypeRegistry, UserDefinedDecl, lift_to_poly};
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
            Value::Extern(o) => {
                let id = o.type_id;
                let r = o.downcast_ref::<regex::Regex>().ok_or_else(|| {
                    RuntimeError::unexpected_type(
                        "FromValue<Re>",
                        &[ValueKind::Extern],
                        ValueKind::Extern,
                    )
                })?;
                Ok(Re(r.clone(), id))
            }
            other => Err(RuntimeError::unexpected_type(
                "FromValue<Re>",
                &[ValueKind::Extern],
                other.kind(),
            )),
        }
    }
}

impl IntoValue for Re {
    fn into_value(self) -> Value {
        Value::extern_value(ExternValue::new(self.1, self.0))
    }
}

// ── Constraint builders ─────────────────────────────────────────────

fn sig(interner: &Interner, params: Vec<Ty>, ret: Ty) -> PolyTy {
    let named: Vec<ParamTerm<Poly>> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), lift_to_poly(ty)))
        .collect();
    TyTerm::Fn {
        params: named,
        ret: Box::new(lift_to_poly(&ret)),
        captures: vec![],
        effect: acvus_mir::ty::Effect::Pure.into(),
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
        effect_params: 0,
    });

    let ty = user_defined_ty(qref);
    ExternRegistry::new(move |interner| {
        vec![
            // regex(pattern) -> Regex
            ExternFnBuilder::new("regex", sig(interner, vec![Ty::String], ty.clone())).handler(
                move |_interner: &Interner, (pattern,): (String,)| {
                    let re = regex::Regex::new(&pattern)
                        .unwrap_or_else(|e| panic!("regex: invalid pattern '{pattern}': {e}"));
                    Ok(Re(re, qref))
                },
            ),
            // regex_match(re, text) -> Bool
            ExternFnBuilder::new(
                "regex_match",
                sig(interner, vec![ty.clone(), Ty::String], Ty::Bool),
            )
            .handler(
                |_interner: &Interner, (Re(re, _), text): (Re, String)| {
                    Ok(re.is_match(&text))
                },
            ),
            // regex_find(re, text) -> Option<String>
            ExternFnBuilder::new(
                "regex_find",
                sig(interner, vec![ty.clone(), Ty::String], Ty::String), // TODO: proper Option<String> return type
            )
            .handler(
                |interner: &Interner, (Re(re, _), text): (Re, String)| {
                    let result = match re.find(&text) {
                        Some(m) => Value::some(interner, Value::string(m.as_str())),
                        None => Value::none(interner),
                    };
                    Ok(result)
                },
            ),
            // regex_find_all(re, text) -> Iterator<String>
            ExternFnBuilder::new(
                "regex_find_all",
                sig(
                    interner,
                    vec![ty.clone(), Ty::String],
                    Ty::UserDefined {
                        id: iter_qref,
                        type_args: vec![Ty::String],
                        effect_args: vec![acvus_mir::ty::Effect::Pure.into()],
                    },
                ),
            )
            .handler(
                |_interner: &Interner, (Re(re, _), text): (Re, String)| {
                    let mut start = 0;
                    let iter =
                        iter_value(_interner, IterHandle::from_fn(move || {
                            let m = re.find_at(&text, start)?;
                            start = m.end();
                            Some(Value::string(m.as_str()))
                        }));
                    Ok(iter)
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
                 (text, Re(re, _), rep): (String, Re, String)| {
                    Ok(re.replace_all(&text, rep.as_str()).into_owned())
                },
            ),
            // regex_split(re, text) -> Iterator<String>
            ExternFnBuilder::new(
                "regex_split",
                sig(
                    interner,
                    vec![ty.clone(), Ty::String],
                    Ty::UserDefined {
                        id: iter_qref,
                        type_args: vec![Ty::String],
                        effect_args: vec![acvus_mir::ty::Effect::Pure.into()],
                    },
                ),
            )
            .handler(
                |_interner: &Interner, (Re(re, _), text): (Re, String)| {
                    let mut last_end = 0;
                    let mut done = false;
                    let iter =
                        iter_value(_interner, IterHandle::from_fn(move || {
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
                    Ok(iter)
                },
            ),
            // regex_extract(text, re) -> Iterator<String>  (capture group 1)
            ExternFnBuilder::new(
                "regex_extract",
                sig(
                    interner,
                    vec![Ty::String, ty.clone()],
                    Ty::UserDefined {
                        id: iter_qref,
                        type_args: vec![Ty::String],
                        effect_args: vec![acvus_mir::ty::Effect::Pure.into()],
                    },
                ),
            )
            .handler(
                |_interner: &Interner, (text, Re(re, _)): (String, Re)| {
                    let mut start = 0;
                    let iter =
                        iter_value(_interner, IterHandle::from_fn(move || {
                            loop {
                                let caps = re.captures_at(&text, start)?;
                                let full = caps.get(0)?;
                                start = full.end();
                                if let Some(group1) = caps.get(1) {
                                    return Some(Value::string(group1.as_str()));
                                }
                            }
                        }));
                    Ok(iter)
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
