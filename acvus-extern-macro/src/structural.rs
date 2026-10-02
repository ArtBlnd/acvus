//! `#[derive(Structural)]`: a struct's or an enum's `Crossing`, field by
//! field. Fields and variants take their places in their names' text order
//! (rfc_compiler types §2), which the derive reads off the names alone.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::{Data, DataEnum, DeriveInput, Fields, FieldsNamed, Generics, Ident, Type, WherePredicate, parse_quote};

pub fn derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    let ((give, take, crossing), declare) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => (object(&input.ident, fields)?, declared_object(&in_text_order(fields)?)),
            Fields::Unnamed(_) | Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "`Structural` derives a struct with named fields: a struct crosses as an acvus object, which names its fields",
                ));
            }
        },
        Data::Enum(data) => (sum(&input.ident, data)?, declared_enum(&input.ident, data)?),
        Data::Union(data) => {
            return Err(syn::Error::new_spanned(data.union_token, "`Structural` derives a struct or an enum; a union has no acvus counterpart"));
        }
    };
    let name = &input.ident;
    let generics = with_runtime(&input.generics, &crossing);
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();
    let declared = declared(input, &crossing, &declare);
    // Every `unsafe` block the derive writes takes back a part of a value the
    // caller of `take` owns and vouched was given as this type, so each part
    // was given as its field's type and is the caller's along with the whole.
    Ok(quote! {
        #[allow(unsafe_code)]
        impl #impl_generics ::acvus_extern::Crossing<__R> for #name #ty_generics #where_clause {
            fn give(self, __rt: &__R) -> <__R as ::acvus_extern::Runtime>::Value {
                #give
            }

            unsafe fn take(__rt: &__R, __value: <__R as ::acvus_extern::Runtime>::Value) -> Self {
                #take
            }
        }

        #declared
    })
}

/// The enum's declared type: its variants in their names' text order, each
/// carrying the payload its fields make (the extern note: `Structural`).
fn declared_enum(name: &Ident, data: &DataEnum) -> syn::Result<TokenStream> {
    let mut ordered: Vec<&syn::Variant> = data.variants.iter().collect();
    ordered.sort_by_key(|variant| variant.ident.unraw().to_string());
    let mut variants = Vec::new();
    for variant in ordered {
        let tag = variant.ident.unraw().to_string();
        let payload = match &variant.fields {
            Fields::Unit => quote!(::core::option::Option::None),
            Fields::Unnamed(unnamed) => {
                let tys = unnamed.unnamed.iter().map(|field| &field.ty);
                quote!(::core::option::Option::Some(<(#(#tys),*) as ::acvus_extern::Declared>::declare(__cx)))
            }
            Fields::Named(named) => {
                let object = declared_object(&in_text_order(named)?);
                quote!(::core::option::Option::Some({ #object }))
            }
        };
        variants.push(quote!((#tag, #payload)));
    }
    let label = name.unraw().to_string();
    Ok(quote! {
        let __variants = [#(#variants),*];
        __cx.enumeration(#label, &__variants)
    })
}

/// The type's `Declared`, whose body `declare` builds its type.
fn declared(input: &DeriveInput, fields: &[Type], declare: &TokenStream) -> TokenStream {
    let name = &input.ident;
    let mut generics = input.generics.clone();
    let predicates = &mut generics.make_where_clause().predicates;
    for ty in fields {
        let predicate: WherePredicate = parse_quote!(#ty: ::acvus_extern::Declared);
        predicates.push(predicate);
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics ::acvus_extern::Declared for #name #ty_generics #where_clause {
            const UNIFORM: bool = false #(|| <#fields as ::acvus_extern::Declared>::UNIFORM)*;

            fn declare(__cx: &mut ::acvus_extern::Declaring<'_>) -> ::acvus_extern::Ty {
                #declare
            }
        }
    }
}

/// An object of `fields`, given in text order.
fn declared_object(fields: &[Named<'_>]) -> TokenStream {
    let names = fields.iter().map(|field| field.name.unraw().to_string());
    let tys = fields.iter().map(|field| field.ty);
    quote! {
        let __fields = [#((#names, <#tys as ::acvus_extern::Declared>::declare(__cx))),*];
        __cx.object(&__fields)
    }
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
            let [#(#bindings),*] = unsafe { <<__R as ::acvus_extern::Runtime>::Object as ::acvus_extern::ObjectKind<__R>>::take(__rt, #value) };
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
        let [#(#bindings),*] = unsafe { <<__R as ::acvus_extern::Runtime>::Object as ::acvus_extern::ObjectKind<__R>>::take(__rt, __value) };
        #name { #(#names: unsafe { <#tys as ::acvus_extern::Crossing<__R>>::take(__rt, #bindings) }),* }
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
                        unsafe { <() as ::acvus_extern::Crossing<__R>>::take(__rt, __payload) };
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
                        let (#(#bindings),*) = unsafe { <(#(#tys),*) as ::acvus_extern::Crossing<__R>>::take(__rt, __payload) };
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
                        #name::#label { #(#names: unsafe { <#tys as ::acvus_extern::Crossing<__R>>::take(__rt, #bindings) }),* }
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
    // Each constructor is a closure coerced at once to an `unsafe fn`, so the
    // only handle to it states its premise: the payload is the one made for
    // its variant, which the runtime asserts by calling it for the tag.
    let take = quote! {
        let __variants: [unsafe fn(&__R, <__R as ::acvus_extern::Runtime>::Value) -> Self; #count] = [#(#takes),*];
        unsafe { <<__R as ::acvus_extern::Runtime>::Enum as ::acvus_extern::EnumKind<__R>>::take(__rt, __value, &__variants) }
    };
    Ok((give, take, crossing))
}
