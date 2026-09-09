//! Base64 and URL encoding. All pure.

use acvus_extern::{ExternRegistry, Interner, RuntimeError, extern_fn, extern_registry};
use base64::Engine;

#[extern_fn(effect = pure)]
fn base64_encode(_: &Interner, s: String) -> String {
    base64::engine::general_purpose::STANDARD.encode(&s)
}

#[extern_fn(effect = pure)]
fn base64_decode(_: &Interner, s: String) -> Result<String, RuntimeError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&s)
        .map_err(|e| RuntimeError::extern_call("base64_decode", format!("invalid input: {e}")))?;
    String::from_utf8(bytes)
        .map_err(|e| RuntimeError::extern_call("base64_decode", format!("invalid UTF-8: {e}")))
}

#[extern_fn(effect = pure)]
fn url_encode(_: &Interner, s: String) -> String {
    percent_encoding::utf8_percent_encode(&s, percent_encoding::NON_ALPHANUMERIC).to_string()
}

#[extern_fn(effect = pure)]
fn url_decode(_: &Interner, s: String) -> String {
    percent_encoding::percent_decode_str(&s)
        .decode_utf8_lossy()
        .into_owned()
}

pub fn encoding_registry() -> ExternRegistry {
    extern_registry! {
        fns: [base64_encode, base64_decode, url_encode, url_decode],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::TypeRegistry;

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = encoding_registry().register(&i, &mut TypeRegistry::new());
        assert_eq!(reg.functions.len(), 4);
        assert_eq!(reg.executables.len(), 4);
    }
}
