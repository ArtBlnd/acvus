//! Network IO as extern functions.

mod http;

pub use http::{
    Client, ClientSettings, Field, Header, HttpError, IdempotentRequest, Request, Response,
    http_registry,
};
