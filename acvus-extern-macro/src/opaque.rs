//! `#[derive(Opaque)]`: the author's mark that a Rust type is an
//! extern-defined type, held whole by the runtime's Opaque kind, whatever its
//! shape. It crosses by value through `Crossing` and is lent as itself
//! through `Lend`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_quote};

pub fn derive(input: &DeriveInput) -> TokenStream {
    let name = &input.ident;
    let mut generics = input.generics.clone();
    generics.params.insert(0, parse_quote!(__R));
    let predicates = &mut generics.make_where_clause().predicates;
    predicates.push(parse_quote!(__R: ::acvus_extern::Runtime));
    let (_, ty_generics, _) = input.generics.split_for_impl();
    predicates.push(parse_quote!(#name #ty_generics: ::core::marker::Send + ::core::marker::Sync));
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let opaque = quote!(<<__R as ::acvus_extern::Runtime>::Opaque as ::acvus_extern::OpaqueKind<__R>>);
    let value = quote!(<__R as ::acvus_extern::Runtime>::Value);
    quote! {
        impl #impl_generics ::acvus_extern::Crossing<__R> for #name #ty_generics #where_clause {
            fn give(self, __rt: &__R) -> #value {
                #opaque::make(__rt, self)
            }

            fn take(__rt: &__R, __value: #value) -> Self {
                #opaque::take(__rt, __value)
            }
        }

        impl #impl_generics ::acvus_extern::Lend<__R> for #name #ty_generics #where_clause {
            type Viewed = Self;

            fn view<__O>(__rt: &__R, __value: &#value, __lent: impl ::core::ops::FnOnce(&Self) -> __O) -> __O {
                #opaque::view(__rt, __value, __lent)
            }

            fn view_mut<__O>(__rt: &__R, __value: &mut #value, __lent: impl ::core::ops::FnOnce(&mut Self) -> __O) -> __O {
                #opaque::view_mut(__rt, __value, __lent)
            }
        }
    }
}
