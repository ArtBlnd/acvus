//! DateTime extension functions via ExternRegistry.
//!
//! Provides UserDefined<DateTime> with formatting and parsing.
//! All functions except `now()` are pure — DateTime is immutable.

use acvus_interpreter::{
    Defs, ExternFnBuilder, ExternRegistry, FromValue, IntoValue, OpaqueValue, RuntimeError, Uses,
    Value, ValueKind,
};
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{Effect, Param, Ty, TypeRegistry, UserDefinedDecl};
use acvus_utils::Interner;

fn user_defined_ty(id: QualifiedRef) -> Ty {
    Ty::UserDefined {
        id,
        type_args: vec![],
        effect_args: vec![],
    }
}

/// Newtype for `chrono::DateTime<Utc>` — carries its `QualifiedRef`.
struct Dt(chrono::DateTime<chrono::Utc>, QualifiedRef);

impl FromValue for Dt {
    fn from_value(value: Value) -> Result<Self, RuntimeError> {
        match value {
            Value::Opaque(o) => {
                let id = o.type_id;
                let dt = o
                    .downcast_ref::<chrono::DateTime<chrono::Utc>>()
                    .ok_or_else(|| {
                        RuntimeError::unexpected_type(
                            "FromValue<Dt>",
                            &[ValueKind::Opaque],
                            ValueKind::Opaque,
                        )
                    })?;
                Ok(Dt(*dt, id))
            }
            other => Err(RuntimeError::unexpected_type(
                "FromValue<Dt>",
                &[ValueKind::Opaque],
                other.kind(),
            )),
        }
    }
}

impl IntoValue for Dt {
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
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
    }
}

fn sig_io(interner: &Interner, params: Vec<Ty>, ret: Ty) -> FnConstraint {
    let named: Vec<Param> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    FnConstraint {
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::io(),
        }),
        effect: None,
    }
}

// ── Handlers ────────────────────────────────────────────────────────

fn h_format_date(
    _interner: &Interner,
    (Dt(dt, _), fmt): (Dt, String),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((dt.format(&fmt).to_string(), Defs(())))
}

fn h_timestamp(
    _interner: &Interner,
    (Dt(dt, _),): (Dt,),
    Uses(()): Uses<()>,
) -> Result<(i64, Defs<()>), RuntimeError> {
    Ok((dt.timestamp(), Defs(())))
}

/// Build the datetime ExternRegistry.
/// Registers the `DateTime` UserDefined type into `type_registry`.
pub fn datetime_registry(interner: &Interner, type_registry: &mut TypeRegistry) -> ExternRegistry {
    let qref = QualifiedRef::root(interner.intern("DateTime"));
    type_registry.register(UserDefinedDecl {
        qref,
        type_params: vec![],
        effect_params: vec![],
    });

    let ty = user_defined_ty(qref);
    ExternRegistry::new(move |interner| {
        let mut fns = Vec::new();

        // now() -> DateTime  (not available on wasm — requires system clock)
        #[cfg(not(target_arch = "wasm32"))]
        fns.push(
            ExternFnBuilder::new("now", sig_io(interner, vec![], ty.clone()))
                .handler(move |_interner: &Interner, (): (), Uses(()): Uses<()>| {
                    Ok((Dt(chrono::Utc::now(), qref), Defs(())))
                }),
        );

        fns.extend([
            // format_date(dt, fmt) -> String
            ExternFnBuilder::new(
                "format_date",
                sig(interner, vec![ty.clone(), Ty::String], Ty::String),
            )
            .handler(h_format_date),
            // parse_date(s, fmt) -> DateTime
            ExternFnBuilder::new(
                "parse_date",
                sig(interner, vec![Ty::String, Ty::String], ty.clone()),
            )
            .handler(
                move |_interner: &Interner,
                      (s, fmt): (String, String),
                      Uses(()): Uses<()>| {
                    let dt = chrono::NaiveDateTime::parse_from_str(&s, &fmt)
                        .map(|ndt| ndt.and_utc())
                        .unwrap_or_else(|e| {
                            panic!(
                                "parse_date: invalid input '{s}' with format '{fmt}': {e}"
                            )
                        });
                    Ok((Dt(dt, qref), Defs(())))
                },
            ),
            // timestamp(dt) -> Int  (Unix epoch seconds)
            ExternFnBuilder::new(
                "timestamp",
                sig(interner, vec![ty.clone()], Ty::Int),
            )
            .handler(h_timestamp),
            // from_timestamp(epoch) -> DateTime
            ExternFnBuilder::new(
                "from_timestamp",
                sig(interner, vec![Ty::Int], ty.clone()),
            )
            .handler(
                move |_interner: &Interner, (epoch,): (i64,), Uses(()): Uses<()>| {
                    let dt = chrono::DateTime::from_timestamp(epoch, 0)
                        .unwrap_or_else(|| panic!("from_timestamp: invalid epoch {epoch}"));
                    Ok((Dt(dt, qref), Defs(())))
                },
            ),
            // add_days(dt, n) -> DateTime
            ExternFnBuilder::new(
                "add_days",
                sig(interner, vec![ty.clone(), Ty::Int], ty.clone()),
            )
            .handler(
                move |_interner: &Interner,
                      (Dt(dt, _), n): (Dt, i64),
                      Uses(()): Uses<()>| {
                    Ok((Dt(dt + chrono::Duration::days(n), qref), Defs(())))
                },
            ),
            // add_hours(dt, n) -> DateTime
            ExternFnBuilder::new(
                "add_hours",
                sig(interner, vec![ty.clone(), Ty::Int], ty.clone()),
            )
            .handler(
                move |_interner: &Interner,
                      (Dt(dt, _), n): (Dt, i64),
                      Uses(()): Uses<()>| {
                    Ok((Dt(dt + chrono::Duration::hours(n), qref), Defs(())))
                },
            ),
        ]);

        fns
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
        let reg = datetime_registry(&i, &mut tr);
        let registered = reg.register(&i);
        // 6 pure functions + now() on non-wasm targets.
        assert!(registered.functions.len() >= 6);
        assert_eq!(registered.functions.len(), registered.executables.len());
    }
}
