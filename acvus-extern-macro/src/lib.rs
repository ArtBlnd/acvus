//! Proc macros for acvus-extern (rfc_extern macros).
//!
//! - `#[extern_fn]`: a Rust function declares an extern fn. The function is
//!   kept as written; beside it, in the type namespace under its name, the
//!   attribute emits the function's marker, which a runtime registers: the
//!   captures it holds, its dispatch (`ExternFn`, or `AwaitedFn` for an
//!   `async fn`, and both where `sync` names its returning body), and its
//!   declaration (`Declared`).

use proc_macro::TokenStream;
use syn::{ItemFn, parse_macro_input};

mod attr;
mod facts;
mod generic;
mod glue;
mod lifetimes;
mod opaque;
mod sig;

/// An extern-defined opaque type (rfc_extern macros §3, types §2): its
/// crossing, its declared type and its structural facts.
#[proc_macro_derive(Opaque)]
pub fn derive_opaque(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::DeriveInput);
    match opaque::derive(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// A generic function's signature (rfc_extern macros, impls §1-§2): a name
/// with a signature and no body.
#[proc_macro_attribute]
pub fn generic_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as attr::ExternFnAttr);
    let func = parse_macro_input!(item as syn::ForeignItemFn);
    match generic::generic_fn(attr, func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn extern_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as attr::ExternFnAttr);
    let func = parse_macro_input!(item as ItemFn);
    match glue::extern_fn(attr, func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
