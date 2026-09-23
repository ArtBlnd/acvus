//! HTTP as `reqwest` does it, cut to what a script author writes in a day.
//!
//! Not offered: streaming bodies, multipart, proxies, cookie jars, HTTP/2
//! settings, TLS options, and a connection pool as a value.

mod client;
mod error;
mod plain;
mod request;
mod response;

use acvus_extern::{Contribution, Interner, Manifest, Registry, Runtime, SharedSignature, TyArg};

pub use client::{Client, ClientSettings};
pub use error::HttpError;
pub use request::{IdempotentRequest, Request};
pub use response::Response;

use request::Call;

/// One header, as the script writes it: `{ name: "accept", value: "text/plain" }`.
///
/// A `(String, String)` would say the same thing and does not cross the
/// boundary: `acvus-extern` implements `TyArg` for tuples and `OneValue`
/// for no tuple, so no `Vec<(K, V)>` crosses today.
#[derive(TyArg, Clone)]
pub struct Header {
    pub name: String,
    pub value: String,
}

/// One `application/x-www-form-urlencoded` field, or one query parameter.
#[derive(TyArg, Clone)]
pub struct Field {
    pub name: String,
    pub value: String,
}

async fn text_of(call: Call<'_>) -> Result<String, HttpError> {
    let answered = request::once(call, None, &[]).await?;
    response::into_text(response::checked(answered)?)
}

/// The namespaces this registry declares into. They exist because a bare
/// name is resolved by its receiver (RFC-0043) while `Externs::combine`
/// refuses two declarations at one qualified name: `get(url)` and
/// `c.get(url)` are one name and cannot both be `http::get`.
const PLAIN: Option<&str> = Some("http");
const ON_CLIENT: Option<&str> = Some("http_client");
const ON_REQUEST: Option<&str> = Some("http_request");
const ON_IDEMPOTENT_REQUEST: Option<&str> = Some("http_idempotent_request");
const ON_RESPONSE: Option<&str> = Some("http_response");

/// `extern_registry!` declares into the one namespace it is written with,
/// and this registry spans four, so it calls what that macro calls: the
/// `__extern_fn_*` each `#[extern_fn]` emits beside its function, which
/// takes the interner and the namespace to declare into.
macro_rules! declare_into {
    ($c:expr, $i:expr, $ns:expr, [$($decl:path $(=> $state:expr)?),* $(,)?]) => {
        $(
            for f in $decl($i, $ns $(, $state)?) {
                $c.declare(f);
            }
        )*
    };
}

fn contribution<R>(i: &Interner) -> Contribution<R>
where
    R: Runtime,
{
    let mut c: Contribution<R> = Contribution::of(Manifest {
        types: vec![
            acvus_extern::DeclaredType::of::<Client>(i),
            acvus_extern::DeclaredType::of::<Request>(i),
            acvus_extern::DeclaredType::of::<IdempotentRequest>(i),
            acvus_extern::DeclaredType::of::<Response>(i),
        ],
        signatures: vec![
            <request::sig::header as SharedSignature>::signature_decl(i),
            <request::sig::query as SharedSignature>::signature_decl(i),
            <request::sig::bearer as SharedSignature>::signature_decl(i),
            <request::sig::basic as SharedSignature>::signature_decl(i),
            <request::sig::body as SharedSignature>::signature_decl(i),
            <request::sig::form as SharedSignature>::signature_decl(i),
            <request::sig::json as SharedSignature>::signature_decl(i),
            <request::sig::timeout_ms as SharedSignature>::signature_decl(i),
        ],
        fns: Vec::new(),
    });
    let host = reqwest::Client::new();
    declare_into!(c, i, PLAIN, [
        client::__extern_fn_client_settings,
        client::__extern_fn_client => host.clone(),
        client::__extern_fn_client_with,
        plain::__extern_fn_request_for => host.clone(),
        plain::__extern_fn_get_request => host.clone(),
        plain::__extern_fn_head_request => host.clone(),
        plain::__extern_fn_put_request => host.clone(),
        plain::__extern_fn_delete_request => host.clone(),
        plain::__extern_fn_get => host.clone(),
        plain::__extern_fn_effectful_get => host.clone(),
        plain::__extern_fn_head => host.clone(),
        plain::__extern_fn_effectful_head => host.clone(),
        plain::__extern_fn_put => host.clone(),
        plain::__extern_fn_effectful_put => host.clone(),
        plain::__extern_fn_delete => host.clone(),
        plain::__extern_fn_effectful_delete => host.clone(),
        plain::__extern_fn_get_text => host.clone(),
        plain::__extern_fn_effectful_get_text => host.clone(),
        plain::__extern_fn_post => host.clone(),
        plain::__extern_fn_patch => host.clone(),
        plain::__extern_fn_request => host.clone(),
    ]);
    declare_into!(
        c,
        i,
        ON_CLIENT,
        [
            client::__extern_fn_request_for,
            client::__extern_fn_get_request,
            client::__extern_fn_head_request,
            client::__extern_fn_put_request,
            client::__extern_fn_delete_request,
            client::__extern_fn_get,
            client::__extern_fn_effectful_get,
            client::__extern_fn_head,
            client::__extern_fn_effectful_head,
            client::__extern_fn_put,
            client::__extern_fn_effectful_put,
            client::__extern_fn_delete,
            client::__extern_fn_effectful_delete,
            client::__extern_fn_get_text,
            client::__extern_fn_effectful_get_text,
            client::__extern_fn_post,
            client::__extern_fn_patch,
            client::__extern_fn_request,
        ]
    );
    declare_into!(
        c,
        i,
        ON_REQUEST,
        [
            request::__extern_fn_header_request,
            request::__extern_fn_header_idempotent,
            request::__extern_fn_query_request,
            request::__extern_fn_query_idempotent,
            request::__extern_fn_bearer_request,
            request::__extern_fn_bearer_idempotent,
            request::__extern_fn_basic_request,
            request::__extern_fn_basic_idempotent,
            request::__extern_fn_body_request,
            request::__extern_fn_body_idempotent,
            request::__extern_fn_form_request,
            request::__extern_fn_form_idempotent,
            request::__extern_fn_json_request,
            request::__extern_fn_json_idempotent,
            request::__extern_fn_timeout_ms_request,
            request::__extern_fn_timeout_ms_idempotent,
            request::__extern_fn_send,
        ]
    );
    declare_into!(
        c,
        i,
        ON_IDEMPOTENT_REQUEST,
        [
            request::__extern_fn_send_idempotent,
            request::__extern_fn_effectful,
        ]
    );
    declare_into!(
        c,
        i,
        ON_RESPONSE,
        [
            response::__extern_fn_status,
            response::__extern_fn_ok,
            response::__extern_fn_url,
            response::__extern_fn_header,
            response::__extern_fn_headers,
            response::__extern_fn_text,
            response::__extern_fn_bytes,
            response::__extern_fn_error_for_status,
        ]
    );
    c
}

pub fn http_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    Registry::new(contribution)
}
