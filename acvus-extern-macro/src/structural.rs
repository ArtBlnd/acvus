//! `#[derive(Structural)]`: a struct's or an enum's `Crossing`, field by
//! field. Fields and variants take their places in their names' text order
//! (rfc_compiler types §2), which the derive reads off the names alone.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::{Data, DataEnum, DeriveInput, Fields, FieldsNamed, Generics, Ident, Type, WherePredicate, parse_quote};

pub fn derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    let (give, take, crossing) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => object(&input.ident, fields)?,
            Fields::Unnamed(_) | Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "`Structural` derives a struct with named fields: a struct crosses as an acvus object, which names its fields",
                ));
            }
        },
        Data::Enum(data) => sum(&input.ident, data)?,
        Data::Union(data) => {
            return Err(syn::Error::new_spanned(data.union_token, "`Structural` derives a struct or an enum; a union has no acvus counterpart"));
        }
    };
    let name = &input.ident;
    let generics = with_runtime(&input.generics, &crossing);
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::acvus_extern::Crossing<__R> for #name #ty_generics #where_clause {
            fn give(self, __rt: &__R) -> <__R as ::acvus_extern::Runtime>::Value {
                #give
            }

            fn take(__rt: &__R, __value: <__R as ::acvus_extern::Runtime>::Value) -> Self {
                #take
            }
        }
    })
}

/// The impl's generics: the type's own, a runtime `__R`, and a `Crossing<__R>`
/// bound on every field's type.
fn with_runtime(generics: &Generics, crossing: &[Type]) -> Generics {
    let mut generics = generics.clone();
    generics.params.insert(0, parse_quote!(__R));
    let predicates = &mut generics.make_where_clause().predicates;
    predicates.push(parse_quote!(__R: ::acvus_extern::Runtime));
    for ty in crossing {
        let predicate: WherePredicate = parse_quote!(#ty: ::acvus_extern::Crossing<__R>);
        predicates.push(predicate);
    }
    generics
}

/// A named field, with the binding the generated code gives its value.
struct Named<'f> {
    name: &'f Ident,
    ty: &'f Type,
    binding: Ident,
}

/// The fields of `fields` in their names' text order, each bound to a fresh
/// name so that no field can shadow the generated code's own.
fn in_text_order(fields: &FieldsNamed) -> syn::Result<Vec<Named<'_>>> {
    let mut named = fields
        .named
        .iter()
        .zip(0..)
        .map(|(field, at): (&syn::Field, usize)| {
            let name = field.ident.as_ref().ok_or_else(|| syn::Error::new_spanned(field, "a named field carries its name"))?;
            Ok(Named { name, ty: &field.ty, binding: format_ident!("__f{}", at) })
        })
        .collect::<syn::Result<Vec<Named<'_>>>>()?;
    named.sort_by_key(|field| field.name.unraw().to_string());
    Ok(named)
}

/// An object made of `fields`, given in text order; and the bindings taken
/// back from an object value in that order.
fn object_parts(fields: &[Named<'_>]) -> (TokenStream, TokenStream) {
    let bindings: Vec<&Ident> = fields.iter().map(|field| &field.binding).collect();
    let give = quote! {
        <<__R as ::acvus_extern::Runtime>::Object as ::acvus_extern::ObjectKind<__R>>::make(
            __rt,
            [#(::acvus_extern::Crossing::<__R>::give(#bindings, __rt)),*],
        )
    };
    let take = |value: TokenStream| {
        quote! {
            let [#(#bindings),*] = <<__R as ::acvus_extern::Runtime>::Object as ::acvus_extern::ObjectKind<__R>>::take(__rt, #value);
        }
    };
    (give, take(quote!(__payload)))
}

fn object(name: &Ident, fields: &FieldsNamed) -> syn::Result<(TokenStream, TokenStream, Vec<Type>)> {
    let fields = in_text_order(fields)?;
    let (give_object, _) = object_parts(&fields);
    let names: Vec<&Ident> = fields.iter().map(|field| field.name).collect();
    let tys: Vec<&Type> = fields.iter().map(|field| field.ty).collect();
    let bindings: Vec<&Ident> = fields.iter().map(|field| &field.binding).collect();
    let give = quote! {
        let #name { #(#names: #bindings),* } = self;
        #give_object
    };
    let take = quote! {
        let [#(#bindings),*] = <<__R as ::acvus_extern::Runtime>::Object as ::acvus_extern::ObjectKind<__R>>::take(__rt, __value);
        #name { #(#names: <#tys as ::acvus_extern::Crossing<__R>>::take(__rt, #bindings)),* }
    };
    let crossing = fields.iter().map(|field| field.ty.clone()).collect();
    Ok((give, take, crossing))
}

/// One variant's two halves: the arm that gives its payload, and the
/// constructor that takes it back.
struct Variant {
    give: TokenStream,
    take: TokenStream,
}

fn variant(name: &Ident, tag: usize, variant: &syn::Variant, crossing: &mut Vec<Type>) -> syn::Result<Variant> {
    let label = &variant.ident;
    let make = |payload: TokenStream| {
        quote! {
            <<__R as ::acvus_extern::Runtime>::Enum as ::acvus_extern::EnumKind<__R>>::make(__rt, #tag, #payload)
        }
    };
    let variant = match &variant.fields {
        Fields::Unit => {
            let made = make(quote!(::acvus_extern::Crossing::<__R>::give((), __rt)));
            Variant {
                give: quote!(#name::#label => #made),
                take: quote! {
                    |__rt, __payload| {
                        <() as ::acvus_extern::Crossing<__R>>::take(__rt, __payload);
                        #name::#label
                    }
                },
            }
        }
        Fields::Unnamed(fields) => {
            let tys: Vec<&Type> = fields.unnamed.iter().map(|field| &field.ty).collect();
            crossing.extend(tys.iter().map(|ty| (*ty).clone()));
            let bindings: Vec<Ident> = (0..tys.len()).map(|at| format_ident!("__f{}", at)).collect();
            let made = make(quote!(::acvus_extern::Crossing::<__R>::give((#(#bindings),*), __rt)));
            Variant {
                give: quote!(#name::#label(#(#bindings),*) => #made),
                take: quote! {
                    |__rt, __payload| {
                        let (#(#bindings),*) = <(#(#tys),*) as ::acvus_extern::Crossing<__R>>::take(__rt, __payload);
                        #name::#label(#(#bindings),*)
                    }
                },
            }
        }
        Fields::Named(fields) => {
            let fields = in_text_order(fields)?;
            crossing.extend(fields.iter().map(|field| field.ty.clone()));
            let (give_object, take_object) = object_parts(&fields);
            let names: Vec<&Ident> = fields.iter().map(|field| field.name).collect();
            let tys: Vec<&Type> = fields.iter().map(|field| field.ty).collect();
            let bindings: Vec<&Ident> = fields.iter().map(|field| &field.binding).collect();
            let made = make(give_object);
            Variant {
                give: quote!(#name::#label { #(#names: #bindings),* } => #made),
                take: quote! {
                    |__rt, __payload| {
                        #take_object
                        #name::#label { #(#names: <#tys as ::acvus_extern::Crossing<__R>>::take(__rt, #bindings)),* }
                    }
                },
            }
        }
    };
    Ok(variant)
}

fn sum(name: &Ident, data: &DataEnum) -> syn::Result<(TokenStream, TokenStream, Vec<Type>)> {
    let mut ordered: Vec<&syn::Variant> = data.variants.iter().collect();
    ordered.sort_by_key(|variant| variant.ident.unraw().to_string());
    let mut crossing = Vec::new();
    let mut variants = Vec::new();
    for (tag, each) in ordered.into_iter().enumerate() {
        variants.push(variant(name, tag, each, &mut crossing)?);
    }
    let gives = variants.iter().map(|variant| &variant.give);
    let takes = variants.iter().map(|variant| &variant.take);
    let count = variants.len();
    let give = quote! {
        match self {
            #(#gives,)*
        }
    };
    let take = quote! {
        let __variants: [fn(&__R, <__R as ::acvus_extern::Runtime>::Value) -> Self; #count] = [#(#takes),*];
        <<__R as ::acvus_extern::Runtime>::Enum as ::acvus_extern::EnumKind<__R>>::take(__rt, __value, &__variants)
    };
    Ok((give, take, crossing))
}
