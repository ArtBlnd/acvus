//! Why a request failed, as a value the script reads: nothing here panics.

use reqwest::header::{HeaderName, HeaderValue};
use reqwest::{Method, Url};

/// A structural enum, not an extension type. An extension type is opaque to
/// the language — a script can hold one and hand it back to an extern, and
/// nothing more — while the whole content of an error is its variant and
/// that variant's payload, which `match Err(HttpError::Status(code))` reads
/// directly. `regex::RegexError` is the same decision.
#[derive(acvus_extern::TyArg)]
pub enum HttpError {
    Timeout,
    Connect(String),
    Status(u16),
    Body(String),
    InvalidUrl(String),
    Method(String),
    Header(String),
}

/// Every `reqwest::Error` reaches one variant. `Connect` is the last arm
/// rather than an eighth "other": reqwest reports name resolution, connect,
/// TLS and an unending redirect chain through predicates this enum does not
/// separate, and to a script all of them say the request reached no server.
/// The message it carries is the distinction.
pub(crate) fn of(e: &reqwest::Error) -> HttpError {
    let text = e.to_string();
    if e.is_timeout() {
        HttpError::Timeout
    } else if let Some(status) = e.status() {
        HttpError::Status(status.as_u16())
    } else if e.is_builder() {
        HttpError::InvalidUrl(text)
    } else if e.is_body() || e.is_decode() {
        HttpError::Body(text)
    } else {
        HttpError::Connect(text)
    }
}

pub(crate) fn resolve(base: Option<&Url>, url: &str) -> Result<Url, HttpError> {
    let joined = match base {
        Some(base) => base.join(url),
        None => Url::parse(url),
    };
    joined.map_err(|e| HttpError::InvalidUrl(format!("{url}: {e}")))
}

pub(crate) fn method(name: &str) -> Result<Method, HttpError> {
    Method::from_bytes(name.as_bytes()).map_err(|e| HttpError::Method(format!("{name}: {e}")))
}

pub(crate) fn header_pair(name: &str, value: &str) -> Result<(HeaderName, HeaderValue), HttpError> {
    let header = HeaderName::from_bytes(name.as_bytes())
        .map_err(|e| HttpError::Header(format!("{name}: {e}")))?;
    let value =
        HeaderValue::from_str(value).map_err(|e| HttpError::Header(format!("{name}: {e}")))?;
    Ok((header, value))
}
