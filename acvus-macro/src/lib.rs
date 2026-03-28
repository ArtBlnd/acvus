use proc_macro::TokenStream;
use quote::quote;
use syn::{LitStr, parse_macro_input};

/// Placeholder info extracted from acvus source.
struct Placeholder {
    name: String,
    /// Byte offset range in the original source.
    start: usize,
    end: usize,
    /// Whether this is a splice placeholder (`*ident` → `Vec<Expr>`).
    is_splice: bool,
}

/// Scan acvus source for `%ident` (single) and `*ident` (splice) placeholders.
/// Returns (substituted source with placeholders replaced by dummy idents, placeholder list).
fn extract_placeholders(source: &str) -> (String, Vec<Placeholder>) {
    let mut result = String::with_capacity(source.len());
    let mut placeholders = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let sigil = bytes[i];
        if (sigil == b'%' || sigil == b'*') && i + 1 < bytes.len() && is_ident_start(bytes[i + 1]) {
            let is_splice = sigil == b'*';
            let start = i;
            i += 1; // skip sigil
            let name_start = i;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let name = &source[name_start..i];
            let prefix = if is_splice { "splice" } else { "ph" };
            let dummy = format!("__acvus_{prefix}_{name}__");
            placeholders.push(Placeholder {
                name: name.to_string(),
                start,
                end: i,
                is_splice,
            });
            result.push_str(&dummy);
        } else {
            result.push(source[i..].chars().next().unwrap());
            i += source[i..].chars().next().unwrap().len_utf8();
        }
    }

    (result, placeholders)
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Validate acvus source (with placeholders substituted) as a script.
/// Also validates that splice placeholders are only in sequence contexts.
fn validate_script(source: &str, placeholders: &[Placeholder]) -> Result<(), String> {
    let interner = acvus_utils::Interner::new();
    let script = acvus_ast::parse_script(&interner, source).map_err(|e| format!("{e:?}"))?;

    let splice_names: Vec<_> = placeholders
        .iter()
        .filter(|p| p.is_splice)
        .map(|p| interner.intern(&format!("__acvus_splice_{}__", p.name)))
        .collect();

    if !splice_names.is_empty() {
        let errors =
            acvus_ast::substitute::validate_splice_positions_script(&script, &splice_names);
        if !errors.is_empty() {
            return Err("splice placeholder (*) in non-sequence context; \
                        splice is only valid inside list elements, function arguments, \
                        tuple elements, pipe chains, or binary operation chains"
                .to_string());
        }
    }

    Ok(())
}

/// Validate acvus source (with placeholders substituted) as a template.
/// Also validates that splice placeholders are only in sequence contexts.
fn validate_template(source: &str, placeholders: &[Placeholder]) -> Result<(), String> {
    let interner = acvus_utils::Interner::new();
    let template = acvus_ast::parse(&interner, source).map_err(|e| format!("{e:?}"))?;

    let splice_names: Vec<_> = placeholders
        .iter()
        .filter(|p| p.is_splice)
        .map(|p| interner.intern(&format!("__acvus_splice_{}__", p.name)))
        .collect();

    if !splice_names.is_empty() {
        let errors =
            acvus_ast::substitute::validate_splice_positions_template(&template, &splice_names);
        if !errors.is_empty() {
            return Err("splice placeholder (*) in non-sequence context; \
                        splice is only valid inside list elements, function arguments, \
                        tuple elements, pipe chains, or binary operation chains"
                .to_string());
        }
    }

    Ok(())
}

/// `acvus_script!("source with %placeholders and *splices")`
///
/// Validates acvus script syntax at compile time.
/// Returns a closure that takes `(&Interner, placeholder_exprs...)` and returns a `Script`.
///
/// - `%ident` — single placeholder, closure parameter is `Expr`
/// - `*ident` — splice placeholder, closure parameter is `Vec<Expr>`
///
/// Splice placeholders can only appear in sequence contexts:
/// list elements, function arguments, tuple elements, pipe chains, or binary op chains.
///
/// ```ignore
/// let result_ast: Expr = ...;
/// let make = acvus_script!("@history = append(@history, %result); @history");
/// let script: Script = make(&interner, result_ast);
///
/// // With splice:
/// let items: Vec<Expr> = vec![...];
/// let make = acvus_script!("[%first, *rest]");
/// let script: Script = make(&interner, first_expr, items);
/// ```
#[proc_macro]
pub fn acvus_script(input: TokenStream) -> TokenStream {
    let source_lit = parse_macro_input!(input as LitStr);
    let source = source_lit.value();

    let (substituted, placeholders) = extract_placeholders(&source);

    if let Err(e) = validate_script(&substituted, &placeholders) {
        return syn::Error::new(source_lit.span(), format!("acvus script error: {e}"))
            .to_compile_error()
            .into();
    }

    emit_script_closure(&substituted, &placeholders)
}

/// `acvus_template!("source with %placeholders and *splices")`
///
/// Validates acvus template syntax at compile time.
/// Returns a closure that takes `(&Interner, placeholder_exprs...)` and returns a `Template`.
///
/// See `acvus_script!` for placeholder syntax.
#[proc_macro]
pub fn acvus_template(input: TokenStream) -> TokenStream {
    let source_lit = parse_macro_input!(input as LitStr);
    let source = source_lit.value();

    let (substituted, placeholders) = extract_placeholders(&source);

    if let Err(e) = validate_template(&substituted, &placeholders) {
        return syn::Error::new(source_lit.span(), format!("acvus template error: {e}"))
            .to_compile_error()
            .into();
    }

    emit_template_closure(&substituted, &placeholders)
}

fn emit_script_closure(substituted: &str, placeholders: &[Placeholder]) -> TokenStream {
    if placeholders.is_empty() {
        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner| -> acvus_ast::Script {
                acvus_ast::parse_script(__acvus_interner__, #substituted)
                    .expect("acvus_script!: pre-validated source failed to parse")
            }
        };
        expanded.into()
    } else {
        let (ph_idents, ph_types, ph_dummy_names, ph_values) = placeholder_tokens(placeholders);

        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner, #( #ph_idents: #ph_types ),*|
                -> acvus_ast::Script
            {
                let __script__ = acvus_ast::parse_script(__acvus_interner__, #substituted)
                    .expect("acvus_script!: pre-validated source failed to parse");
                let mut __subs__ = rustc_hash::FxHashMap::default();
                #(
                    __subs__.insert(
                        __acvus_interner__.intern(#ph_dummy_names),
                        #ph_values,
                    );
                )*
                acvus_ast::substitute::substitute_script(__script__, &__subs__)
            }
        };
        expanded.into()
    }
}

fn emit_template_closure(substituted: &str, placeholders: &[Placeholder]) -> TokenStream {
    if placeholders.is_empty() {
        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner| -> acvus_ast::Template {
                acvus_ast::parse_template(__acvus_interner__, #substituted)
                    .expect("acvus_template!: pre-validated source failed to parse")
            }
        };
        expanded.into()
    } else {
        let (ph_idents, ph_types, ph_dummy_names, ph_values) = placeholder_tokens(placeholders);

        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner, #( #ph_idents: #ph_types ),*|
                -> acvus_ast::Template
            {
                let __template__ = acvus_ast::parse_template(__acvus_interner__, #substituted)
                    .expect("acvus_template!: pre-validated source failed to parse");
                let mut __subs__ = rustc_hash::FxHashMap::default();
                #(
                    __subs__.insert(
                        __acvus_interner__.intern(#ph_dummy_names),
                        #ph_values,
                    );
                )*
                acvus_ast::substitute::substitute_template(__template__, &__subs__)
            }
        };
        expanded.into()
    }
}

/// Generate the token fragments for placeholder closure parameters and map insertions.
fn placeholder_tokens(
    placeholders: &[Placeholder],
) -> (
    Vec<proc_macro2::Ident>,
    Vec<proc_macro2::TokenStream>,
    Vec<String>,
    Vec<proc_macro2::TokenStream>,
) {
    let mut ph_idents = Vec::new();
    let mut ph_types = Vec::new();
    let mut ph_dummy_names = Vec::new();
    let mut ph_values = Vec::new();

    for p in placeholders {
        let ident = proc_macro2::Ident::new(&p.name, proc_macro2::Span::call_site());

        if p.is_splice {
            ph_types.push(quote! { Vec<acvus_ast::Expr> });
            ph_dummy_names.push(format!("__acvus_splice_{}__", p.name));
            let ident_ref = ident.clone();
            ph_values.push(quote! {
                acvus_ast::substitute::SubstValue::Splice(#ident_ref)
            });
        } else {
            ph_types.push(quote! { acvus_ast::Expr });
            ph_dummy_names.push(format!("__acvus_ph_{}__", p.name));
            let ident_ref = ident.clone();
            ph_values.push(quote! {
                acvus_ast::substitute::SubstValue::Single(#ident_ref)
            });
        }

        ph_idents.push(ident);
    }

    (ph_idents, ph_types, ph_dummy_names, ph_values)
}
