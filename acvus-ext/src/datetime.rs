//! The `DateTime` extension type. Every function but `now` is pure.

use acvus_extern::{ExternRegistry, ExternType, Interner, RuntimeError, extern_fn, extern_registry};

#[derive(ExternType)]
pub struct DateTime(chrono::DateTime<chrono::Utc>);

#[cfg(not(target_arch = "wasm32"))]
#[extern_fn]
fn now(_: &Interner) -> DateTime {
    DateTime(chrono::Utc::now())
}

#[extern_fn(effect = pure)]
fn format_date(_: &Interner, dt: DateTime, fmt: String) -> String {
    dt.0.format(&fmt).to_string()
}

#[extern_fn(effect = pure)]
fn parse_date(_: &Interner, s: String, fmt: String) -> Result<DateTime, RuntimeError> {
    chrono::NaiveDateTime::parse_from_str(&s, &fmt)
        .map(|ndt| DateTime(ndt.and_utc()))
        .map_err(|e| {
            RuntimeError::extern_call("parse_date", format!("invalid input '{s}' with format '{fmt}': {e}"))
        })
}

/// Unix epoch seconds.
#[extern_fn(effect = pure)]
fn timestamp(_: &Interner, dt: DateTime) -> i64 {
    dt.0.timestamp()
}

#[extern_fn(effect = pure)]
fn from_timestamp(_: &Interner, epoch: i64) -> Result<DateTime, RuntimeError> {
    chrono::DateTime::from_timestamp(epoch, 0)
        .map(DateTime)
        .ok_or_else(|| RuntimeError::extern_call("from_timestamp", format!("invalid epoch {epoch}")))
}

#[extern_fn(effect = pure)]
fn add_days(_: &Interner, dt: DateTime, n: i64) -> DateTime {
    DateTime(dt.0 + chrono::Duration::days(n))
}

#[extern_fn(effect = pure)]
fn add_hours(_: &Interner, dt: DateTime, n: i64) -> DateTime {
    DateTime(dt.0 + chrono::Duration::hours(n))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn datetime_registry() -> ExternRegistry {
    extern_registry! {
        types: [DateTime],
        fns: [now, format_date, parse_date, timestamp, from_timestamp, add_days, add_hours],
    }
}

#[cfg(target_arch = "wasm32")]
pub fn datetime_registry() -> ExternRegistry {
    extern_registry! {
        types: [DateTime],
        fns: [format_date, parse_date, timestamp, from_timestamp, add_days, add_hours],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::TypeRegistry;

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let registered = datetime_registry().register(&i, &mut TypeRegistry::new());
        assert_eq!(registered.functions.len(), 7);
        assert_eq!(registered.functions.len(), registered.executables.len());
    }
}
