//! DateTime extension functions via ExternRegistry.
//!
//! Provides Opaque<DateTime> with formatting and parsing.
//! All functions are pure — DateTime is immutable.

use acvus_interpreter::{ExternFn, ExternRegistry, OpaqueValue, Value};
use acvus_mir::ty::Ty;

const OPAQUE_NAME: &str = "DateTime";

fn opaque_ty() -> Ty {
    Ty::Opaque(OPAQUE_NAME.into())
}

fn extract_dt(v: &Value) -> &chrono::DateTime<chrono::Utc> {
    let Value::Opaque(o) = v else {
        panic!("expected Opaque<DateTime>, got {v:?}");
    };
    o.downcast_ref::<chrono::DateTime<chrono::Utc>>()
        .expect("opaque value is not a DateTime")
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
                .io()  // reads system clock — genuine IO
                .sync_handler(|_args, _interner| {
                    Ok(Value::opaque(OpaqueValue::new(
                        OPAQUE_NAME,
                        chrono::Utc::now(),
                    )))
                }),
        );

        fns.extend([
            // format_date(dt, fmt) -> String
            ExternFn::build("format_date")
                .params(vec![opaque_ty(), Ty::String])
                .ret(Ty::String)
                .pure()
                .sync_handler(|args, _interner| {
                    let dt = extract_dt(&args[0]);
                    let fmt = args[1].as_str();
                    Ok(Value::string(dt.format(fmt).to_string()))
                }),

            // parse_date(s, fmt) -> DateTime
            ExternFn::build("parse_date")
                .params(vec![Ty::String, Ty::String])
                .ret(opaque_ty())
                .pure()
                .sync_handler(|args, _interner| {
                    let s = args[0].as_str();
                    let fmt = args[1].as_str();
                    let dt = chrono::NaiveDateTime::parse_from_str(s, fmt)
                        .map(|ndt| ndt.and_utc())
                        .unwrap_or_else(|e| panic!("parse_date: invalid input '{s}' with format '{fmt}': {e}"));
                    Ok(Value::opaque(OpaqueValue::new(OPAQUE_NAME, dt)))
                }),

            // timestamp(dt) -> Int  (Unix epoch seconds)
            ExternFn::build("timestamp")
                .params(vec![opaque_ty()])
                .ret(Ty::Int)
                .pure()
                .sync_handler(|args, _interner| {
                    let dt = extract_dt(&args[0]);
                    Ok(Value::Int(dt.timestamp()))
                }),

            // from_timestamp(epoch) -> DateTime
            ExternFn::build("from_timestamp")
                .params(vec![Ty::Int])
                .ret(opaque_ty())
                .pure()
                .sync_handler(|args, _interner| {
                    let epoch = args[0].as_int();
                    let dt = chrono::DateTime::from_timestamp(epoch, 0)
                        .unwrap_or_else(|| panic!("from_timestamp: invalid epoch {epoch}"));
                    Ok(Value::opaque(OpaqueValue::new(OPAQUE_NAME, dt)))
                }),

            // add_days(dt, n) -> DateTime
            ExternFn::build("add_days")
                .params(vec![opaque_ty(), Ty::Int])
                .ret(opaque_ty())
                .pure()
                .sync_handler(|args, _interner| {
                    let dt = *extract_dt(&args[0]);
                    let n = args[1].as_int();
                    let result = dt + chrono::Duration::days(n);
                    Ok(Value::opaque(OpaqueValue::new(OPAQUE_NAME, result)))
                }),

            // add_hours(dt, n) -> DateTime
            ExternFn::build("add_hours")
                .params(vec![opaque_ty(), Ty::Int])
                .ret(opaque_ty())
                .pure()
                .sync_handler(|args, _interner| {
                    let dt = *extract_dt(&args[0]);
                    let n = args[1].as_int();
                    let result = dt + chrono::Duration::hours(n);
                    Ok(Value::opaque(OpaqueValue::new(OPAQUE_NAME, result)))
                }),
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
