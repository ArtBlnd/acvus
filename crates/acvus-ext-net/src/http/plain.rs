//! The operations that need no client: each sends on the host's own.

use acvus_extern::extern_fn;
use reqwest::Client as Inner;

use crate::http::error::HttpError;
use crate::http::request::{self, Call, IdempotentRequest, Request};
use crate::http::response::Response;
use crate::http::{Header, text_of};

fn call<'a>(inner: &'a Inner, method: &'a str, url: &'a str) -> Call<'a> {
    Call {
        client: inner,
        base: None,
        method,
        url,
    }
}

#[extern_fn(effect = pure)]
pub(crate) fn request_for(#[state] inner: &Inner, method: &str, url: &str) -> Request {
    request::of(request::parts(call(inner, method, url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn get_request(#[state] inner: &Inner, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(call(inner, "GET", url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn head_request(#[state] inner: &Inner, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(call(inner, "HEAD", url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn put_request(#[state] inner: &Inner, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(call(inner, "PUT", url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn delete_request(#[state] inner: &Inner, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(call(inner, "DELETE", url)))
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn get(#[state] inner: &Inner, url: String) -> Result<Response, HttpError> {
    request::once(call(inner, "GET", &url), None, &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_get(
    #[state] inner: &Inner,
    url: String,
) -> Result<Response, HttpError> {
    request::once(call(inner, "GET", &url), None, &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn head(#[state] inner: &Inner, url: String) -> Result<Response, HttpError> {
    request::once(call(inner, "HEAD", &url), None, &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_head(
    #[state] inner: &Inner,
    url: String,
) -> Result<Response, HttpError> {
    request::once(call(inner, "HEAD", &url), None, &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn put(
    #[state] inner: &Inner,
    url: String,
    body: String,
) -> Result<Response, HttpError> {
    request::once(call(inner, "PUT", &url), Some(&body), &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_put(
    #[state] inner: &Inner,
    url: String,
    body: String,
) -> Result<Response, HttpError> {
    request::once(call(inner, "PUT", &url), Some(&body), &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn delete(#[state] inner: &Inner, url: String) -> Result<Response, HttpError> {
    request::once(call(inner, "DELETE", &url), None, &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_delete(
    #[state] inner: &Inner,
    url: String,
) -> Result<Response, HttpError> {
    request::once(call(inner, "DELETE", &url), None, &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn get_text(#[state] inner: &Inner, url: String) -> Result<String, HttpError> {
    text_of(call(inner, "GET", &url)).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_get_text(
    #[state] inner: &Inner,
    url: String,
) -> Result<String, HttpError> {
    text_of(call(inner, "GET", &url)).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn post(
    #[state] inner: &Inner,
    url: String,
    body: String,
) -> Result<Response, HttpError> {
    request::once(call(inner, "POST", &url), Some(&body), &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn patch(
    #[state] inner: &Inner,
    url: String,
    body: String,
) -> Result<Response, HttpError> {
    request::once(call(inner, "PATCH", &url), Some(&body), &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn request(
    #[state] inner: &Inner,
    method: String,
    url: String,
    body: String,
    headers: Vec<Header>,
) -> Result<Response, HttpError> {
    request::once(call(inner, &method, &url), Some(&body), &headers).await
}
