//! DateTime extension functions via ExternRegistry.
//!
//! Provides Opaque<DateTime> with formatting and parsing.
//! All functions are pure — DateTime is immutable.

use acvus_interpreter::{
    Defs, ExternFn, ExternRegistry, FromValue, IntoValue, OpaqueValue, RuntimeError, Uses, Value,
    ValueKind,
};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

const OPAQUE_NAME: &str = "DateTime";

fn opaque_ty() -> Ty {
    Ty::Opaque(OPAQUE_NAME.into())
}

/// Newtype for `chrono::DateTime<Utc>` — enables typed FromValue/IntoValue conversion.
struct Dt(chrono::DateTime<chrono::Utc>);

impl FromValue for Dt {
    fn from_value(value: Value) -> Result<Self, RuntimeError> {
        match value {
            Value::Opaque(o) => {
                let dt = o
                    .downcast_ref::<chrono::DateTime<chrono::Utc>>()
                    .ok_or_else(|| {
                        RuntimeError::unexpected_type(
                            "FromValue<Dt>",
                            &[ValueKind::Opaque],
                            ValueKind::Opaque,
                        )
                    })?;
                Ok(Dt(*dt))
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
        Value::opaque(OpaqueValue::new(OPAQUE_NAME, self.0))
    }
}

pub fn datetime_registry() -> ExternRegistry {
    ExternRegistry::new(|_interner| {
        let mut fns = Vec::new();

        // now() -> DateTime  (not available on wasm — requires system clock)
        #[cfg(not(target_arch = "wasm32"))]
        fns.push(
            ExternFn::build("now")
                .params(vec![])
                .ret(opaque_ty())
                .io()
                .handler(|_interner: &Interner, (): (), Uses(()): Uses<()>| {
                    Ok((Dt(chrono::Utc::now()), Defs(())))
                }),
        );

        fns.extend([
            // format_date(dt, fmt) -> String
            ExternFn::build("format_date")
                .params(vec![opaque_ty(), Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner, (Dt(dt), fmt): (Dt, String), Uses(()): Uses<()>| {
                        Ok((dt.format(&fmt).to_string(), Defs(())))
                    },
                ),
            // parse_date(s, fmt) -> DateTime
            ExternFn::build("parse_date")
                .params(vec![Ty::String, Ty::String])
                .ret(opaque_ty())
                .pure()
                .handler(
                    |_interner: &Interner, (s, fmt): (String, String), Uses(()): Uses<()>| {
                        let dt = chrono::NaiveDateTime::parse_from_str(&s, &fmt)
                            .map(|ndt| ndt.and_utc())
                            .unwrap_or_else(|e| {
                                panic!("parse_date: invalid input '{s}' with format '{fmt}': {e}")
                            });
                        Ok((Dt(dt), Defs(())))
                    },
                ),
            // timestamp(dt) -> Int  (Unix epoch seconds)
            ExternFn::build("timestamp")
                .params(vec![opaque_ty()])
                .ret(Ty::Int)
                .pure()
                .handler(
                    |_interner: &Interner, (Dt(dt),): (Dt,), Uses(()): Uses<()>| {
                        Ok((dt.timestamp(), Defs(())))
                    },
                ),
            // from_timestamp(epoch) -> DateTime
            ExternFn::build("from_timestamp")
                .params(vec![Ty::Int])
                .ret(opaque_ty())
                .pure()
                .handler(
                    |_interner: &Interner, (epoch,): (i64,), Uses(()): Uses<()>| {
                        let dt = chrono::DateTime::from_timestamp(epoch, 0)
                            .unwrap_or_else(|| panic!("from_timestamp: invalid epoch {epoch}"));
                        Ok((Dt(dt), Defs(())))
                    },
                ),
            // add_days(dt, n) -> DateTime
            ExternFn::build("add_days")
                .params(vec![opaque_ty(), Ty::Int])
                .ret(opaque_ty())
                .pure()
                .handler(
                    |_interner: &Interner, (Dt(dt), n): (Dt, i64), Uses(()): Uses<()>| {
                        Ok((Dt(dt + chrono::Duration::days(n)), Defs(())))
                    },
                ),
            // add_hours(dt, n) -> DateTime
            ExternFn::build("add_hours")
                .params(vec![opaque_ty(), Ty::Int])
                .ret(opaque_ty())
                .pure()
                .handler(
                    |_interner: &Interner, (Dt(dt), n): (Dt, i64), Uses(()): Uses<()>| {
                        Ok((Dt(dt + chrono::Duration::hours(n)), Defs(())))
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
        let reg = datetime_registry();
        let registered = reg.register(&i);
        // 6 pure functions + now() on non-wasm targets.
        assert!(registered.functions.len() >= 6);
        assert_eq!(registered.functions.len(), registered.executables.len());
    }
}
