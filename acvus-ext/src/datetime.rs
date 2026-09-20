//! The `DateTime` extension type. Every function but `now` is pure.

use acvus_extern::{ExternType, Registry, Runtime, TyArg, extern_fn, extern_registry};

#[derive(ExternType)]
#[repr(transparent)]
pub struct DateTime(chrono::DateTime<chrono::Utc>);

#[cfg(not(target_arch = "wasm32"))]
#[extern_fn]
fn now() -> DateTime {
    DateTime(chrono::Utc::now())
}

#[extern_fn(effect = pure)]
fn format_date(dt: DateTime, fmt: &str) -> String {
    dt.0.format(fmt).to_string()
}

/// Why a text is not a date, or an epoch not a moment.
#[derive(TyArg)]
pub enum DateError {
    Unparsable {
        input: String,
        format: String,
        message: String,
    },
    OutOfRange(i64),
}

#[extern_fn(effect = pure)]
fn parse_date(s: &str, fmt: &str) -> Result<DateTime, DateError> {
    chrono::NaiveDateTime::parse_from_str(s, fmt)
        .map(|ndt| DateTime(ndt.and_utc()))
        .map_err(|e| DateError::Unparsable {
            message: e.to_string(),
            input: s.to_owned(),
            format: fmt.to_owned(),
        })
}

/// Unix epoch seconds.
#[extern_fn(effect = pure)]
fn timestamp(dt: DateTime) -> i64 {
    dt.0.timestamp()
}

#[extern_fn(effect = pure)]
fn from_timestamp(epoch: i64) -> Result<DateTime, DateError> {
    chrono::DateTime::from_timestamp(epoch, 0)
        .map(DateTime)
        .ok_or(DateError::OutOfRange(epoch))
}

#[extern_fn(effect = pure)]
fn add_days(dt: DateTime, n: i64) -> DateTime {
    DateTime(dt.0 + chrono::Duration::days(n))
}

#[extern_fn(effect = pure)]
fn add_hours(dt: DateTime, n: i64) -> DateTime {
    DateTime(dt.0 + chrono::Duration::hours(n))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn datetime_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [DateTime],
        fns: [now, format_date, parse_date, timestamp, from_timestamp, add_days, add_hours],
    }
}

#[cfg(target_arch = "wasm32")]
pub fn datetime_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [DateTime],
        fns: [format_date, parse_date, timestamp, from_timestamp, add_days, add_hours],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let registered = Externs::combine(vec![datetime_registry::<TypesOnly>()], &i)
            .expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(registered.functions.len() - core.functions.len(), 7);
        assert_eq!(registered.functions.len(), registered.handlers.len());
    }
}
