//! What a server answered, read whole before it crosses.

use std::ops::Range;

use acvus_extern::{ExternType, extern_fn};

use crate::http::Header;
use crate::http::error::{self, HttpError};

/// RFC 9110 §15.3.
const SUCCESS: Range<u16> = 200..300;

/// RFC 9110 §15.5 and §15.6 together: the classes a request may not treat
/// as an answer.
const FAILED: Range<u16> = 400..600;

pub struct Answer {
    status: u16,
    url: String,
    headers: Vec<Header>,
    body: Vec<u8>,
}

#[derive(ExternType)]
#[repr(transparent)]
pub struct Response(Answer);

/// Streaming bodies are not offered, so the socket is finished with here and
/// a `Response` a script holds owns no connection: `text` and `bytes` take
/// from this buffer and cannot fail on the network.
pub(crate) async fn read(sent: reqwest::Response) -> Result<Response, HttpError> {
    let status = sent.status().as_u16();
    let url = sent.url().to_string();
    let headers = sent
        .headers()
        .iter()
        .map(|(name, value)| {
            let value = value
                .to_str()
                .map_err(|e| HttpError::Header(format!("{name}: {e}")))?;
            Ok(Header {
                name: name.as_str().to_owned(),
                value: value.to_owned(),
            })
        })
        .collect::<Result<Vec<Header>, HttpError>>()?;
    let body = sent.bytes().await.map_err(|e| error::of(&e))?.to_vec();
    Ok(Response(Answer {
        status,
        url,
        headers,
        body,
    }))
}

pub(crate) fn checked(r: Response) -> Result<Response, HttpError> {
    match FAILED.contains(&r.0.status) {
        true => Err(HttpError::Status(r.0.status)),
        false => Ok(r),
    }
}

pub(crate) fn into_text(r: Response) -> Result<String, HttpError> {
    String::from_utf8(r.0.body).map_err(|e| HttpError::Body(e.to_string()))
}

#[extern_fn(effect = pure)]
pub(crate) fn status(r: &Response) -> u16 {
    r.0.status
}

#[extern_fn(effect = pure)]
pub(crate) fn ok(r: &Response) -> bool {
    SUCCESS.contains(&r.0.status)
}

#[extern_fn(effect = pure)]
pub(crate) fn url(r: &Response) -> String {
    r.0.url.clone()
}

#[extern_fn(effect = pure)]
pub(crate) fn header(r: &Response, name: &str) -> Option<String> {
    r.0.headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case(name))
        .map(|h| h.value.clone())
}

#[extern_fn(effect = pure)]
pub(crate) fn headers(r: &Response) -> Vec<Header> {
    r.0.headers.clone()
}

#[extern_fn(effect = pure)]
pub(crate) fn text(r: Response) -> Result<String, HttpError> {
    into_text(r)
}

/// The body is already read, so there is nothing here left to fail and no
/// `Result` to return. `text` still has one: the bytes may not be UTF-8.
#[extern_fn(effect = pure)]
pub(crate) fn bytes(r: Response) -> Vec<u8> {
    r.0.body
}

#[extern_fn(effect = pure)]
pub(crate) fn error_for_status(r: Response) -> Result<Response, HttpError> {
    checked(r)
}
