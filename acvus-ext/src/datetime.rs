//! The `DateTime` extension type. Every function but `now` is pure.

use acvus_extern::{ExternError, ExternRegistry, ExternType, Runtime, extern_fn, extern_registry};

#[derive(ExternType)]
pub struct DateTime(chrono::DateTime<chrono::Utc>);

#[cfg(not(target_arch = "wasm32"))]
#[extern_fn]
fn now<R>(_: &R) -> DateTime
where
    R: Runtime,
{
    DateTime(chrono::Utc::now())
}

#[extern_fn(effect = pure)]
fn format_date<R>(_: &R, dt: DateTime, fmt: String) -> String
where
    R: Runtime,
{
    dt.0.format(&fmt).to_string()
}

#[extern_fn(effect = pure)]
fn parse_date<R>(_: &R, s: String, fmt: String) -> Result<DateTime, ExternError>
where
    R: Runtime,
{
    chrono::NaiveDateTime::parse_from_str(&s, &fmt)
        .map(|ndt| DateTime(ndt.and_utc()))
        .map_err(|e| {
            ExternError::call(
                "parse_date",
                format!("invalid input '{s}' with format '{fmt}': {e}"),
            )
        })
}

/// Unix epoch seconds.
#[extern_fn(effect = pure)]
fn timestamp<R>(_: &R, dt: DateTime) -> i64
where
    R: Runtime,
{
    dt.0.timestamp()
}

#[extern_fn(effect = pure)]
fn from_timestamp<R>(_: &R, epoch: i64) -> Result<DateTime, ExternError>
where
    R: Runtime,
{
    chrono::DateTime::from_timestamp(epoch, 0)
        .map(DateTime)
        .ok_or_else(|| ExternError::call("from_timestamp", format!("invalid epoch {epoch}")))
}

#[extern_fn(effect = pure)]
fn add_days<R>(_: &R, dt: DateTime, n: i64) -> DateTime
where
    R: Runtime,
{
    DateTime(dt.0 + chrono::Duration::days(n))
}

#[extern_fn(effect = pure)]
fn add_hours<R>(_: &R, dt: DateTime, n: i64) -> DateTime
where
    R: Runtime,
{
    DateTime(dt.0 + chrono::Duration::hours(n))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn datetime_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        types: [DateTime],
        fns: [now, format_date, parse_date, timestamp, from_timestamp, add_days, add_hours],
    }
}

#[cfg(target_arch = "wasm32")]
pub fn datetime_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        types: [DateTime],
        fns: [format_date, parse_date, timestamp, from_timestamp, add_days, add_hours],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Interner, TypeRegistry, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let registered = datetime_registry::<TypesOnly>().register(&i, &mut TypeRegistry::new());
        assert_eq!(registered.functions.len(), 7);
        assert_eq!(registered.functions.len(), registered.handlers.len());
    }
}
