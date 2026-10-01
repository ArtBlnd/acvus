//! Proc macros for acvus-extern (rfc_extern macros).
//!
//! - `#[extern_fn]`: a Rust function declares an extern fn. The function is
//!   kept as written; beside it, in the type namespace under its name, the
//!   attribute emits the function's marker, which a runtime registers: the
//!   captures it holds, its dispatch (`ExternFn`, or `AwaitedFn` for an
//!   `async fn`), and its declaration (`Declared`).

use proc_macro::TokenStream;
use syn::{ItemFn, parse_macro_input};

mod attr;
mod glue;
mod lifetimes;
mod sig;

#[proc_macro_attribute]
pub fn extern_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as attr::ExternFnAttr);
    let func = parse_macro_input!(item as ItemFn);
    match glue::extern_fn(attr, func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
