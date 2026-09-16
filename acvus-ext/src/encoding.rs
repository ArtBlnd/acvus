//! Base64 and URL encoding. All pure.

use acvus_extern::{Registry, Runtime, TyArg, extern_fn, extern_registry};
use base64::Engine;

#[derive(TyArg)]
pub enum Base64Error {
    InvalidBase64 { input: String, message: String },
    InvalidUtf8 { input: String, message: String },
}

#[extern_fn(effect = pure)]
fn base64_encode(s: String) -> String {
    base64::engine::general_purpose::STANDARD.encode(&s)
}

#[extern_fn(effect = pure)]
fn base64_decode(s: String) -> Result<String, Base64Error> {
    let bytes = match base64::engine::general_purpose::STANDARD.decode(&s) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Err(Base64Error::InvalidBase64 {
                message: e.to_string(),
                input: s,
            });
        }
    };
    String::from_utf8(bytes).map_err(|e| Base64Error::InvalidUtf8 {
        message: e.to_string(),
        input: s,
    })
}

#[extern_fn(effect = pure)]
fn url_encode(s: String) -> String {
    percent_encoding::utf8_percent_encode(&s, percent_encoding::NON_ALPHANUMERIC).to_string()
}

#[extern_fn(effect = pure)]
fn url_decode(s: String) -> String {
    percent_encoding::percent_decode_str(&s)
        .decode_utf8_lossy()
        .into_owned()
}

pub fn encoding_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        fns: [base64_encode, base64_decode, url_encode, url_decode],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = Externs::combine(vec![encoding_registry::<TypesOnly>()], &i)
            .expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 4);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 4);
    }
}
