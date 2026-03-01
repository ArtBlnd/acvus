use acvus_ast::{BinOp, Literal, Span};
use acvus_mir::ir::{InstKind, ValueId};
use acvus_mir::ty::Ty;
use rand::rngs::StdRng;
use rand::Rng;

use super::rewriter::PassState;

/// Pre-computed encrypted text fragment.
pub struct EncryptedFragment {
    /// XOR-encrypted char codes.
    pub encrypted_codes: Vec<i64>,
    /// The XOR key for this fragment's first char.
    pub initial_key: i64,
}

/// A single text entry split into chained fragments.
pub struct EncryptedText {
    pub fragments: Vec<EncryptedFragment>,
    /// chain_keys[i] is the initial key for fragment[i].
    /// chain_keys[i+1] = last_plain_char_of_fragment[i] ^ salt.
    pub chain_keys: Vec<i64>,
}

/// Encrypt all text pool entries.
pub fn encrypt_texts(texts: &[String], rng: &mut StdRng) -> Vec<EncryptedText> {
    texts
        .iter()
        .map(|text| encrypt_single_text(text, rng))
        .collect()
}

fn encrypt_single_text(text: &str, rng: &mut StdRng) -> EncryptedText {
    if text.is_empty() {
        return EncryptedText {
            fragments: vec![],
            chain_keys: vec![],
        };
    }

    let chars: Vec<char> = text.chars().collect();
    let fragment_size = 3.max(chars.len() / 4).min(chars.len());
    let chunks: Vec<&[char]> = chars.chunks(fragment_size).collect();

    let mut fragments = Vec::new();
    let mut chain_keys = Vec::new();

    let mut next_key: i64 = rng.random_range(1..256);
    chain_keys.push(next_key);

    for chunk in &chunks {
        let mut encrypted_codes = Vec::new();
        let mut current_key = next_key;

        for &ch in *chunk {
            let code = ch as i64;
            encrypted_codes.push(code ^ current_key);
            current_key = code;
        }

        let last_code = chunk.last().map(|&c| c as i64).unwrap_or(0);
        let salt: i64 = rng.random_range(1..256);
        next_key = last_code ^ salt;

        fragments.push(EncryptedFragment {
            encrypted_codes,
            initial_key: chain_keys.last().copied().unwrap(),
        });
        chain_keys.push(next_key);
    }

    EncryptedText {
        fragments,
        chain_keys,
    }
}

/// Emit instructions that decrypt an EncryptedText and emit the result.
pub fn emit_encrypted_text(
    ctx: &mut PassState,
    rng: &mut StdRng,
    span: Span,
    enc: &EncryptedText,
) {
    if enc.fragments.is_empty() {
        return;
    }

    let mut prev_last_code: Option<ValueId> = None;

    for (frag_idx, frag) in enc.fragments.iter().enumerate() {
        let v_key = if frag_idx == 0 {
            let v = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::Const {
                dst: v,
                value: Literal::Int(frag.initial_key),
            });
            v
        } else {
            // runtime_key = prev_last_code XOR derivation_const
            // derivation_const = frag.initial_key XOR actual_prev_last_plaintext_code
            let prev_frag_chars_plain =
                decrypt_fragment_plain(&enc.fragments[frag_idx - 1], &enc.chain_keys[frag_idx - 1]);
            let prev_last_plain = *prev_frag_chars_plain.last().unwrap();
            let derivation_const = frag.initial_key ^ prev_last_plain;

            let v_derive = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::Const {
                dst: v_derive,
                value: Literal::Int(derivation_const),
            });

            let v = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::BinOp {
                dst: v,
                op: BinOp::Xor,
                left: prev_last_code.unwrap(),
                right: v_derive,
            });
            v
        };

        // Pick a decryption variant for this fragment.
        let variant = rng.random_range(0u32..2);

        // Variant 1: emit offset const.
        let (offset_val, v_offset) = if variant == 1 {
            let offset: i64 = rng.random_range(1..256);
            let vo = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::Const { dst: vo, value: Literal::Int(offset) });
            (offset, Some(vo))
        } else {
            (0, None)
        };

        let mut v_accum: Option<ValueId> = None;
        let mut v_current_key = v_key;
        let mut ct_key = frag.initial_key;

        for &enc_code_orig in &frag.encrypted_codes {
            let plain_code = enc_code_orig ^ ct_key;

            // Re-encrypt for the chosen variant.
            let enc_code = if variant == 1 {
                (plain_code - offset_val) ^ ct_key
            } else {
                enc_code_orig
            };

            let v_enc = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::Const {
                dst: v_enc,
                value: Literal::Int(enc_code),
            });

            let v_xored = ctx.alloc_val(Ty::Int);
            ctx.emit(span, InstKind::BinOp {
                dst: v_xored,
                op: BinOp::Xor,
                left: v_enc,
                right: v_current_key,
            });

            let v_dec = if variant == 1 {
                let v_added = ctx.alloc_val(Ty::Int);
                ctx.emit(span, InstKind::BinOp {
                    dst: v_added,
                    op: BinOp::Add,
                    left: v_xored,
                    right: v_offset.unwrap(),
                });
                v_added
            } else {
                v_xored
            };

            let v_char = ctx.alloc_val(Ty::String);
            ctx.emit(span, InstKind::Call {
                dst: v_char,
                func: "int_to_char".into(),
                args: vec![v_dec],
            });

            v_accum = Some(match v_accum {
                None => v_char,
                Some(prev) => {
                    let v_concat = ctx.alloc_val(Ty::String);
                    ctx.emit(span, InstKind::BinOp {
                        dst: v_concat,
                        op: BinOp::Add,
                        left: prev,
                        right: v_char,
                    });
                    v_concat
                }
            });

            v_current_key = v_dec;
            ct_key = plain_code;
        }

        prev_last_code = Some(v_current_key);

        if let Some(v) = v_accum {
            ctx.emit(span, InstKind::EmitValue(v));
        }
    }
}

/// Helper: decrypt a fragment at compile time to get plaintext char codes.
fn decrypt_fragment_plain(frag: &EncryptedFragment, initial_key: &i64) -> Vec<i64> {
    let mut result = Vec::new();
    let mut key = *initial_key;
    for &enc in &frag.encrypted_codes {
        let plain = enc ^ key;
        result.push(plain);
        key = plain;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let texts = vec!["hello world".to_string(), "foo".to_string()];
        let encrypted = encrypt_texts(&texts, &mut rng);

        for (i, text) in texts.iter().enumerate() {
            let enc = &encrypted[i];
            let mut decrypted = String::new();
            for (fi, frag) in enc.fragments.iter().enumerate() {
                let plain_codes = decrypt_fragment_plain(frag, &enc.chain_keys[fi]);
                for code in plain_codes {
                    decrypted.push(char::from_u32(code as u32).unwrap());
                }
            }
            assert_eq!(&decrypted, text);
        }
    }

    #[test]
    fn empty_text() {
        let mut rng = StdRng::seed_from_u64(0);
        let encrypted = encrypt_texts(&["".to_string()], &mut rng);
        assert!(encrypted[0].fragments.is_empty());
    }
}
