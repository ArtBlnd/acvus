//! Encoding extension functions via ExternRegistry.
//!
//! Provides base64 and URL encoding/decoding. All pure.

use acvus_interpreter::{Defs, ExternFn, ExternRegistry, Uses};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

use base64::Engine;

pub fn encoding_registry() -> ExternRegistry {
    ExternRegistry::new(|_interner| {
        vec![
            // base64_encode(s) -> String
            ExternFn::build("base64_encode")
                .params(vec![Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner, (s,): (String,), Uses(()): Uses<()>| {
                        Ok((
                            base64::engine::general_purpose::STANDARD.encode(&s),
                            Defs(()),
                        ))
                    },
                ),
            // base64_decode(s) -> String
            ExternFn::build("base64_decode")
                .params(vec![Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner, (s,): (String,), Uses(()): Uses<()>| {
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(&s)
                            .unwrap_or_else(|e| panic!("base64_decode: invalid input: {e}"));
                        let decoded = String::from_utf8(bytes)
                            .unwrap_or_else(|e| panic!("base64_decode: invalid UTF-8: {e}"));
                        Ok((decoded, Defs(())))
                    },
                ),
            // url_encode(s) -> String
            ExternFn::build("url_encode")
                .params(vec![Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner, (s,): (String,), Uses(()): Uses<()>| {
                        let encoded = percent_encoding::utf8_percent_encode(
                            &s,
                            percent_encoding::NON_ALPHANUMERIC,
                        )
                        .to_string();
                        Ok((encoded, Defs(())))
                    },
                ),
            // url_decode(s) -> String
            ExternFn::build("url_decode")
                .params(vec![Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner, (s,): (String,), Uses(()): Uses<()>| {
                        let decoded = percent_encoding::percent_decode_str(&s)
                            .decode_utf8_lossy()
                            .into_owned();
                        Ok((decoded, Defs(())))
                    },
                ),
        ]
    })
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
