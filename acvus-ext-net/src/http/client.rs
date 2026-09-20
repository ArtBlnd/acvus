//! A client the script builds with its own settings, and the operations it
//! carries.

use std::time::Duration;

use acvus_extern::{ExternType, TyArg, extern_fn};
use reqwest::header::HeaderMap;
use reqwest::redirect::Policy;
use reqwest::{Client as Inner, Url};

use crate::http::Header;
use crate::http::error::{self, HttpError};
use crate::http::request::{self, Call, IdempotentRequest, Request};
use crate::http::response::Response;

/// RFC 9110 §15.4 sets no limit; ten is `reqwest`'s own default and what
/// `follow_redirects: true` means here.
const REDIRECT_LIMIT: usize = 10;

/// Every field is written, as `RegexFlags` is: an object of five fields is
/// the type and an object of four is not (RFC-0042). `client_settings()` is
/// the value a script starts from, and `Option` is how a field says it sets
/// nothing rather than setting an empty string.
#[derive(TyArg, Clone)]
pub struct ClientSettings {
    pub timeout_ms: Option<u64>,
    pub base_url: Option<String>,
    pub headers: Vec<Header>,
    pub follow_redirects: bool,
    pub user_agent: Option<String>,
}

impl Default for ClientSettings {
    fn default() -> Self {
        ClientSettings {
            timeout_ms: None,
            base_url: None,
            headers: Vec::new(),
            follow_redirects: true,
            user_agent: None,
        }
    }
}

pub struct Settled {
    inner: Inner,
    base: Option<Url>,
}

#[derive(ExternType)]
#[repr(transparent)]
pub struct Client(Settled);

impl Client {
    fn call<'a>(&'a self, method: &'a str, url: &'a str) -> Call<'a> {
        Call {
            client: &self.0.inner,
            base: self.0.base.as_ref(),
            method,
            url,
        }
    }
}

#[extern_fn(effect = pure)]
pub(crate) fn client_settings() -> ClientSettings {
    ClientSettings::default()
}

/// The host's client, which is also what the plain functions send on, with
/// no base URL: a script that wants the defaults pays for no second
/// connection pool.
#[extern_fn(effect = pure)]
pub(crate) fn client(#[state] inner: &Inner) -> Client {
    Client(Settled {
        inner: inner.clone(),
        base: None,
    })
}

#[extern_fn(effect = pure)]
pub(crate) fn client_with(settings: ClientSettings) -> Result<Client, HttpError> {
    let mut defaults = HeaderMap::new();
    for header in &settings.headers {
        let (name, value) = error::header_pair(&header.name, &header.value)?;
        defaults.insert(name, value);
    }
    let mut building =
        Inner::builder()
            .default_headers(defaults)
            .redirect(match settings.follow_redirects {
                true => Policy::limited(REDIRECT_LIMIT),
                false => Policy::none(),
            });
    if let Some(ms) = settings.timeout_ms {
        building = building.timeout(Duration::from_millis(ms));
    }
    if let Some(agent) = &settings.user_agent {
        building = building.user_agent(agent);
    }
    let base = match &settings.base_url {
        Some(url) => Some(error::resolve(None, url)?),
        None => None,
    };
    let inner = building.build().map_err(|e| error::of(&e))?;
    Ok(Client(Settled { inner, base }))
}

#[extern_fn(effect = pure)]
pub(crate) fn request_for(c: &Client, method: &str, url: &str) -> Request {
    request::of(request::parts(c.call(method, url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn get_request(c: &Client, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(c.call("GET", url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn head_request(c: &Client, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(c.call("HEAD", url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn put_request(c: &Client, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(c.call("PUT", url)))
}

#[extern_fn(effect = pure)]
pub(crate) fn delete_request(c: &Client, url: &str) -> IdempotentRequest {
    request::idempotent(request::parts(c.call("DELETE", url)))
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn get(c: &Client, url: String) -> Result<Response, HttpError> {
    request::once(c.call("GET", &url), None, &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_get(c: &Client, url: String) -> Result<Response, HttpError> {
    request::once(c.call("GET", &url), None, &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn head(c: &Client, url: String) -> Result<Response, HttpError> {
    request::once(c.call("HEAD", &url), None, &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_head(c: &Client, url: String) -> Result<Response, HttpError> {
    request::once(c.call("HEAD", &url), None, &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn put(c: &Client, url: String, body: String) -> Result<Response, HttpError> {
    request::once(c.call("PUT", &url), Some(&body), &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_put(
    c: &Client,
    url: String,
    body: String,
) -> Result<Response, HttpError> {
    request::once(c.call("PUT", &url), Some(&body), &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn delete(c: &Client, url: String) -> Result<Response, HttpError> {
    request::once(c.call("DELETE", &url), None, &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_delete(c: &Client, url: String) -> Result<Response, HttpError> {
    request::once(c.call("DELETE", &url), None, &[]).await
}

#[extern_fn(effect = idempotent)]
pub(crate) async fn get_text(c: &Client, url: String) -> Result<String, HttpError> {
    crate::http::text_of(c.call("GET", &url)).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn effectful_get_text(c: &Client, url: String) -> Result<String, HttpError> {
    crate::http::text_of(c.call("GET", &url)).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn post(c: &Client, url: String, body: String) -> Result<Response, HttpError> {
    request::once(c.call("POST", &url), Some(&body), &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn patch(c: &Client, url: String, body: String) -> Result<Response, HttpError> {
    request::once(c.call("PATCH", &url), Some(&body), &[]).await
}

#[extern_fn(effect = opaque)]
pub(crate) async fn request(
    c: &Client,
    method: String,
    url: String,
    body: String,
    headers: Vec<Header>,
) -> Result<Response, HttpError> {
    request::once(c.call(&method, &url), Some(&body), &headers).await
}
