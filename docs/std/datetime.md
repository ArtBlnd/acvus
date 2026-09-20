# `datetime`

The `DateTime` extension type over `chrono`, registered by
`datetime_registry` rather than by `std_registries`: a host adds it when it
wants it. Every function but `now` is pure, and every moment is UTC.

A format string and a text to parse are read and not kept, so both are
`&str`: `first.parse_date(stamp_format)`, not
`first.to_string().parse_date(stamp_format.to_string())`.

| name | signature | Rust `std` twin | difference |
| --- | --- | --- | --- |
| `now` | `() -> DateTime` | `SystemTime::now` | `chrono::Utc::now`; not registered on wasm32 |
| `format_date` | `(DateTime, &str) -> String` | none | `chrono`'s `strftime` format |
| `parse_date` | `(&str, &str) -> Result<DateTime, DateError>` | `str::parse` | `chrono::NaiveDateTime::parse_from_str`, read as UTC; the error names the input, the format and `chrono`'s message |
| `timestamp` | `(DateTime) -> i64` | `SystemTime::duration_since` | Unix epoch seconds |
| `from_timestamp` | `(i64) -> Result<DateTime, DateError>` | none | an epoch outside the representable range is `DateError::OutOfRange` |
| `add_days` | `(DateTime, i64) -> DateTime` | none | `chrono::Duration::days` |
| `add_hours` | `(DateTime, i64) -> DateTime` | none | `chrono::Duration::hours` |

`DateError` is `Unparsable { input, format, message }` or
`OutOfRange(i64)`.
