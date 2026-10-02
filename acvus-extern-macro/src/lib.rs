//! The macros of the extern side (`docs/extern-note.md`).

mod opaque;
mod structural;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Crosses a struct with named fields as an acvus object, or an enum as an
/// acvus enum, by value alone (`acvus_extern::Crossing`).
#[proc_macro_derive(Structural)]
pub fn structural(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    structural::derive(&input).unwrap_or_else(syn::Error::into_compile_error).into()
}

/// Marks a Rust type as an extern-defined type, held whole by the runtime's
/// Opaque kind.
#[proc_macro_derive(Opaque)]
pub fn opaque(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    opaque::derive(&input).unwrap_or_else(syn::Error::into_compile_error).into()
}
