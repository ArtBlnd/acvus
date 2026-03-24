//! Encoding extension functions via ExternRegistry.
//!
//! Provides base64 and URL encoding/decoding. All pure.

use acvus_interpreter::{ExternFn, ExternRegistry, Value};
use acvus_mir::ty::Ty;

use base64::Engine;

pub fn encoding_registry() -> ExternRegistry {
    ExternRegistry::new(|_interner| vec![
        // base64_encode(s) -> String
        ExternFn::build("base64_encode")
            .params(vec![Ty::String])
            .ret(Ty::String)
            .pure()
            .sync_handler(|args, _interner| {
                let s = args[0].as_str();
                Ok(Value::string(base64::engine::general_purpose::STANDARD.encode(s)))
            }),

        // base64_decode(s) -> String
        ExternFn::build("base64_decode")
            .params(vec![Ty::String])
            .ret(Ty::String)
            .pure()
            .sync_handler(|args, _interner| {
                let s = args[0].as_str();
                let bytes = base64::engine::general_purpose::STANDARD.decode(s)
                    .unwrap_or_else(|e| panic!("base64_decode: invalid input: {e}"));
                let decoded = String::from_utf8(bytes)
                    .unwrap_or_else(|e| panic!("base64_decode: invalid UTF-8: {e}"));
                Ok(Value::string(decoded))
            }),

        // url_encode(s) -> String
        ExternFn::build("url_encode")
            .params(vec![Ty::String])
            .ret(Ty::String)
            .pure()
            .sync_handler(|args, _interner| {
                let s = args[0].as_str();
                let encoded = percent_encoding::utf8_percent_encode(
                    s,
                    percent_encoding::NON_ALPHANUMERIC,
                ).to_string();
                Ok(Value::string(encoded))
            }),

        // url_decode(s) -> String
        ExternFn::build("url_decode")
            .params(vec![Ty::String])
            .ret(Ty::String)
            .pure()
            .sync_handler(|args, _interner| {
                let s = args[0].as_str();
                let decoded = percent_encoding::percent_decode_str(s)
                    .decode_utf8_lossy()
                    .into_owned();
                Ok(Value::string(decoded))
            }),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = encoding_registry();
        let registered = reg.register(&i);
        assert_eq!(registered.functions.len(), 4);
        assert_eq!(registered.executables.len(), 4);
    }
}
