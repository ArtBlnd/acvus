//! A request the script builds a piece at a time, and the one place a
//! request is sent.
//!
//! Which of the two request types a constructor answers with is what the
//! method means to the order chain: `send` at an `IdempotentRequest` is
//! declared idempotent, `send` at a `Request` opaque, and one bare name
//! picks by its receiver (RFC-0043).

use std::time::Duration;

use acvus_extern::{ExternType, extern_fn};
use reqwest::Url;

use crate::http::error::{self, HttpError};
use crate::http::response::{self, Response};
use crate::http::{Field, Header};

/// What a call names before any piece is set: the client it goes out on,
/// the base its URL is resolved against, and what it addresses.
pub(crate) struct Call<'a> {
    pub client: &'a reqwest::Client,
    pub base: Option<&'a Url>,
    pub method: &'a str,
    pub url: &'a str,
}

enum Body {
    Text(String),
    Form(Vec<Field>),
}

enum Auth {
    Bearer(String),
    Basic { user: String, password: String },
}

pub struct Parts {
    client: reqwest::Client,
    base: Option<Url>,
    method: String,
    url: String,
    headers: Vec<Header>,
    query: Vec<Field>,
    body: Option<Body>,
    auth: Option<Auth>,
    timeout: Option<Duration>,
}

impl Parts {
    fn header(mut self, name: &str, value: &str) -> Parts {
        self.headers.push(Header {
            name: name.to_owned(),
            value: value.to_owned(),
        });
        self
    }

    fn query(mut self, name: &str, value: &str) -> Parts {
        self.query.push(Field {
            name: name.to_owned(),
            value: value.to_owned(),
        });
        self
    }

    fn bearer(mut self, token: &str) -> Parts {
        self.auth = Some(Auth::Bearer(token.to_owned()));
        self
    }

    fn basic(mut self, user: &str, password: &str) -> Parts {
        self.auth = Some(Auth::Basic {
            user: user.to_owned(),
            password: password.to_owned(),
        });
        self
    }

    fn body(mut self, text: &str) -> Parts {
        self.body = Some(Body::Text(text.to_owned()));
        self
    }

    fn form(mut self, fields: Vec<Field>) -> Parts {
        self.body = Some(Body::Form(fields));
        self
    }

    fn json(mut self, text: &str) -> Parts {
        self.body = Some(Body::Text(text.to_owned()));
        self.header("content-type", "application/json")
    }

    fn timeout_ms(mut self, ms: u64) -> Parts {
        self.timeout = Some(Duration::from_millis(ms));
        self
    }
}

/// A request whose method the declaration cannot read: `request_for` takes
/// it as text, so `send` here is opaque.
#[derive(ExternType)]
#[repr(transparent)]
pub struct Request(Parts);

/// A request a constructor named a method RFC 9110 calls safe or idempotent
/// for — `get_request`, `head_request`, `put_request`, `delete_request`.
/// `send` at this type is idempotent, and `effectful` is how a script that
/// knows its server mutates on such a method hands back a `Request`.
#[derive(ExternType)]
#[repr(transparent)]
pub struct IdempotentRequest(Parts);

pub(crate) fn parts(call: Call<'_>) -> Parts {
    Parts {
        client: call.client.clone(),
        base: call.base.cloned(),
        method: call.method.to_owned(),
        url: call.url.to_owned(),
        headers: Vec::new(),
        query: Vec::new(),
        body: None,
        auth: None,
        timeout: None,
    }
}

pub(crate) fn of(parts: Parts) -> Request {
    Request(parts)
}

pub(crate) fn idempotent(parts: Parts) -> IdempotentRequest {
    IdempotentRequest(parts)
}

/// One call with a body and headers and nothing else set: what every plain
/// function and every client method is.
pub(crate) async fn once(
    call: Call<'_>,
    body: Option<&str>,
    headers: &[Header],
) -> Result<Response, HttpError> {
    let mut p = parts(call);
    p.headers = headers.to_vec();
    p.body = body.map(|text| Body::Text(text.to_owned()));
    send_parts(p).await
}

pub(crate) async fn send_parts(p: Parts) -> Result<Response, HttpError> {
    let url = error::resolve(p.base.as_ref(), &p.url)?;
    let method = error::method(&p.method)?;
    let mut building = p.client.request(method, url);
    for header in &p.headers {
        let (name, value) = error::header_pair(&header.name, &header.value)?;
        building = building.header(name, value);
    }
    if !p.query.is_empty() {
        building = building.query(&pairs(&p.query));
    }
    building = match p.auth {
        None => building,
        Some(Auth::Bearer(token)) => building.bearer_auth(token),
        Some(Auth::Basic { user, password }) => building.basic_auth(user, Some(password)),
    };
    building = match p.body {
        None => building,
        Some(Body::Text(text)) => building.body(text),
        Some(Body::Form(fields)) => building.form(&pairs(&fields)),
    };
    if let Some(timeout) = p.timeout {
        building = building.timeout(timeout);
    }
    let sent = building.send().await.map_err(|e| error::of(&e))?;
    response::read(sent).await
}

fn pairs(fields: &[Field]) -> Vec<(&str, &str)> {
    fields
        .iter()
        .map(|f| (f.name.as_str(), f.value.as_str()))
        .collect()
}

/// A setter is one name over both request types, as `num::abs` is one name
/// over both widths (RFC-0019): the instances below fill `R`, and a
/// setter answers at the type it was given, so a chain cannot lose the
/// method's meaning halfway.
pub mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "http_request",
        fn header<R>(r: R, name: &str, value: &str) -> R
        where
            R: Var<kind::Type>;
    }

    extern_signature! {
        ns: "http_request",
        fn query<R>(r: R, name: &str, value: &str) -> R
        where
            R: Var<kind::Type>;
    }

    extern_signature! {
        ns: "http_request",
        fn bearer<R>(r: R, token: &str) -> R
        where
            R: Var<kind::Type>;
    }

    extern_signature! {
        ns: "http_request",
        fn basic<R>(r: R, user: &str, password: &str) -> R
        where
            R: Var<kind::Type>;
    }

    extern_signature! {
        ns: "http_request",
        fn body<R>(r: R, text: &str) -> R
        where
            R: Var<kind::Type>;
    }

    extern_signature! {
        ns: "http_request",
        fn form<R>(r: R, fields: Vec<crate::http::Field>) -> R
        where
            R: Var<kind::Type>;
    }

    extern_signature! {
        ns: "http_request",
        fn json<R>(r: R, text: &str) -> R
        where
            R: Var<kind::Type>;
    }

    extern_signature! {
        ns: "http_request",
        fn timeout_ms<R>(r: R, ms: u64) -> R
        where
            R: Var<kind::Type>;
    }
}

// -- header -------------------------------------------------------------

#[extern_fn(instance_of = sig::header, effect = pure)]
pub(crate) fn header_request(r: Request, name: &str, value: &str) -> Request {
    Request(r.0.header(name, value))
}

#[extern_fn(instance_of = sig::header, effect = pure)]
pub(crate) fn header_idempotent(
    r: IdempotentRequest,
    name: &str,
    value: &str,
) -> IdempotentRequest {
    IdempotentRequest(r.0.header(name, value))
}

// -- query --------------------------------------------------------------

#[extern_fn(instance_of = sig::query, effect = pure)]
pub(crate) fn query_request(r: Request, name: &str, value: &str) -> Request {
    Request(r.0.query(name, value))
}

#[extern_fn(instance_of = sig::query, effect = pure)]
pub(crate) fn query_idempotent(r: IdempotentRequest, name: &str, value: &str) -> IdempotentRequest {
    IdempotentRequest(r.0.query(name, value))
}

// -- bearer -------------------------------------------------------------

#[extern_fn(instance_of = sig::bearer, effect = pure)]
pub(crate) fn bearer_request(r: Request, token: &str) -> Request {
    Request(r.0.bearer(token))
}

#[extern_fn(instance_of = sig::bearer, effect = pure)]
pub(crate) fn bearer_idempotent(r: IdempotentRequest, token: &str) -> IdempotentRequest {
    IdempotentRequest(r.0.bearer(token))
}

// -- basic --------------------------------------------------------------

#[extern_fn(instance_of = sig::basic, effect = pure)]
pub(crate) fn basic_request(r: Request, user: &str, password: &str) -> Request {
    Request(r.0.basic(user, password))
}

#[extern_fn(instance_of = sig::basic, effect = pure)]
pub(crate) fn basic_idempotent(
    r: IdempotentRequest,
    user: &str,
    password: &str,
) -> IdempotentRequest {
    IdempotentRequest(r.0.basic(user, password))
}

// -- body ---------------------------------------------------------------

#[extern_fn(instance_of = sig::body, effect = pure)]
pub(crate) fn body_request(r: Request, text: &str) -> Request {
    Request(r.0.body(text))
}

#[extern_fn(instance_of = sig::body, effect = pure)]
pub(crate) fn body_idempotent(r: IdempotentRequest, text: &str) -> IdempotentRequest {
    IdempotentRequest(r.0.body(text))
}

// -- form ---------------------------------------------------------------

#[extern_fn(instance_of = sig::form, effect = pure)]
pub(crate) fn form_request(r: Request, fields: Vec<Field>) -> Request {
    Request(r.0.form(fields))
}

#[extern_fn(instance_of = sig::form, effect = pure)]
pub(crate) fn form_idempotent(r: IdempotentRequest, fields: Vec<Field>) -> IdempotentRequest {
    IdempotentRequest(r.0.form(fields))
}

// -- json ---------------------------------------------------------------

#[extern_fn(instance_of = sig::json, effect = pure)]
pub(crate) fn json_request(r: Request, text: &str) -> Request {
    Request(r.0.json(text))
}

#[extern_fn(instance_of = sig::json, effect = pure)]
pub(crate) fn json_idempotent(r: IdempotentRequest, text: &str) -> IdempotentRequest {
    IdempotentRequest(r.0.json(text))
}

// -- timeout_ms ---------------------------------------------------------

#[extern_fn(instance_of = sig::timeout_ms, effect = pure)]
pub(crate) fn timeout_ms_request(r: Request, ms: u64) -> Request {
    Request(r.0.timeout_ms(ms))
}

#[extern_fn(instance_of = sig::timeout_ms, effect = pure)]
pub(crate) fn timeout_ms_idempotent(r: IdempotentRequest, ms: u64) -> IdempotentRequest {
    IdempotentRequest(r.0.timeout_ms(ms))
}

// -- sending ------------------------------------------------------------

#[extern_fn(effect = opaque)]
pub(crate) async fn send(r: Request) -> Result<Response, HttpError> {
    send_parts(r.0).await
}

#[extern_fn(name = "send", effect = idempotent)]
pub(crate) async fn send_idempotent(r: IdempotentRequest) -> Result<Response, HttpError> {
    send_parts(r.0).await
}

#[extern_fn(effect = pure)]
pub(crate) fn effectful(r: IdempotentRequest) -> Request {
    Request(r.0)
}
