//! The text of a character, a byte and a byte-string literal, decoded.
//!
//! One escape table serves all three, and it is Rust's: `\n`, `\r`, `\t`,
//! `\\`, `\0`, `\'`, `\"`, `\xNN` and `\u{…}`. An escape outside it is an
//! error rather than two characters — what a `"…"` string does with an
//! unknown escape (`parse_string_literal`, `token.rs`) is the older rule
//! and is left where it is.

use std::fmt;

/// The width an integer literal's suffix names (RFC-0037's eight). The
/// same eight `acvus_mir::ty::IntTy` holds; this is the parser's copy,
/// which is below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntWidth {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntWidth {
    pub const ALL: [IntWidth; 8] = [
        IntWidth::I8,
        IntWidth::I16,
        IntWidth::I32,
        IntWidth::I64,
        IntWidth::U8,
        IntWidth::U16,
        IntWidth::U32,
        IntWidth::U64,
    ];

    pub fn name(self) -> &'static str {
        match self {
            IntWidth::I8 => "i8",
            IntWidth::I16 => "i16",
            IntWidth::I32 => "i32",
            IntWidth::I64 => "i64",
            IntWidth::U8 => "u8",
            IntWidth::U16 => "u16",
            IntWidth::U32 => "u32",
            IntWidth::U64 => "u64",
        }
    }

    pub fn of_name(name: &str) -> Option<IntWidth> {
        IntWidth::ALL.into_iter().find(|w| w.name() == name)
    }
}

/// `10u64`: an integer literal that names its own width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuffixedInt {
    pub value: i128,
    pub width: IntWidth,
}

/// What a literal's text was not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiteralErrorKind {
    /// A `'…'` holding no scalar value or more than one.
    NotOneChar,
    /// `\q`: no escape the table names.
    UnknownEscape(char),
    /// A `\x`, `\u` or `\'` that ran off the end of the literal.
    UnfinishedEscape,
    /// `\xZZ` or `\u{ZZ}`: not hexadecimal.
    BadHexEscape,
    /// `\u{110000}` or a surrogate: not a Unicode scalar value.
    NotAScalarValue(u32),
    /// A `\u{…}` inside a byte literal, which has no UTF-8 to encode into.
    UnicodeEscapeInByte,
    /// A character above `\x7F` inside a byte or byte-string literal.
    NotAscii(char),
    /// `\xNN` above `\x7F` in a `char` literal, which is Rust's rule.
    ByteEscapeOutOfRange(u32),
}

impl fmt::Display for LiteralErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiteralErrorKind::NotOneChar => {
                write!(f, "a character literal holds exactly one scalar value")
            }
            LiteralErrorKind::UnknownEscape(c) => write!(f, "unknown character escape `\\{c}`"),
            LiteralErrorKind::UnfinishedEscape => write!(f, "unfinished escape sequence"),
            LiteralErrorKind::BadHexEscape => {
                write!(f, "a numeric escape takes hexadecimal digits")
            }
            LiteralErrorKind::NotAScalarValue(n) => {
                write!(f, "{n:#x} is not a Unicode scalar value")
            }
            LiteralErrorKind::UnicodeEscapeInByte => write!(
                f,
                "`\\u{{…}}` is not a byte escape; a byte literal takes `\\xNN`"
            ),
            LiteralErrorKind::NotAscii(c) => write!(
                f,
                "a byte literal takes ASCII; `{c}` is not, and `\\xNN` writes its bytes"
            ),
            LiteralErrorKind::ByteEscapeOutOfRange(n) => write!(
                f,
                "`\\x{n:02X}` is above `\\x7F`; a character literal writes it `\\u{{{n:X}}}`"
            ),
        }
    }
}

/// Where an escape may reach.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// A `char`: `\u{…}` is admitted and `\xNN` stops at `\x7F`, as Rust's.
    Scalar,
    /// A byte: `\xNN` reaches `0xFF` and `\u{…}` is refused.
    Byte,
}

/// One unit of a literal's text: the value it stands for, and whether an
/// escape wrote it. A byte literal admits any value an escape writes and
/// only ASCII written directly, so the two cannot be one field.
#[derive(Clone, Copy)]
struct Unit {
    value: u32,
    escaped: bool,
}

struct Escapes<'a> {
    rest: std::str::Chars<'a>,
    reach: Reach,
}

impl Iterator for Escapes<'_> {
    type Item = Result<Unit, LiteralErrorKind>;

    fn next(&mut self) -> Option<Result<Unit, LiteralErrorKind>> {
        let c = self.rest.next()?;
        if c != '\\' {
            return Some(Ok(Unit {
                value: u32::from(c),
                escaped: false,
            }));
        }
        Some(self.escape().map(|value| Unit {
            value,
            escaped: true,
        }))
    }
}

impl Escapes<'_> {
    fn escape(&mut self) -> Result<u32, LiteralErrorKind> {
        let Some(c) = self.rest.next() else {
            return Err(LiteralErrorKind::UnfinishedEscape);
        };
        match c {
            'n' => Ok(u32::from('\n')),
            'r' => Ok(u32::from('\r')),
            't' => Ok(u32::from('\t')),
            '0' => Ok(0),
            '\\' => Ok(u32::from('\\')),
            '\'' => Ok(u32::from('\'')),
            '"' => Ok(u32::from('"')),
            'x' => self.hex_byte(),
            'u' if self.reach == Reach::Byte => Err(LiteralErrorKind::UnicodeEscapeInByte),
            'u' => self.hex_scalar(),
            other => Err(LiteralErrorKind::UnknownEscape(other)),
        }
    }

    fn hex_byte(&mut self) -> Result<u32, LiteralErrorKind> {
        let (Some(hi), Some(lo)) = (self.rest.next(), self.rest.next()) else {
            return Err(LiteralErrorKind::UnfinishedEscape);
        };
        let (Some(hi), Some(lo)) = (hi.to_digit(16), lo.to_digit(16)) else {
            return Err(LiteralErrorKind::BadHexEscape);
        };
        let value = hi * 16 + lo;
        match self.reach {
            Reach::Byte => Ok(value),
            Reach::Scalar if value <= 0x7F => Ok(value),
            Reach::Scalar => Err(LiteralErrorKind::ByteEscapeOutOfRange(value)),
        }
    }

    fn hex_scalar(&mut self) -> Result<u32, LiteralErrorKind> {
        if self.rest.next() != Some('{') {
            return Err(LiteralErrorKind::UnfinishedEscape);
        }
        let mut value: u32 = 0;
        let mut digits = 0;
        loop {
            let Some(c) = self.rest.next() else {
                return Err(LiteralErrorKind::UnfinishedEscape);
            };
            if c == '}' {
                break;
            }
            let Some(d) = c.to_digit(16) else {
                return Err(LiteralErrorKind::BadHexEscape);
            };
            digits += 1;
            value = value
                .checked_mul(16)
                .and_then(|v| v.checked_add(d))
                .ok_or(LiteralErrorKind::NotAScalarValue(u32::MAX))?;
        }
        match digits {
            0 => Err(LiteralErrorKind::BadHexEscape),
            _ => Ok(value),
        }
    }
}

fn scalars(text: &str) -> Result<Vec<char>, LiteralErrorKind> {
    Escapes {
        rest: text.chars(),
        reach: Reach::Scalar,
    }
    .map(|unit| {
        let Unit { value, .. } = unit?;
        char::from_u32(value).ok_or(LiteralErrorKind::NotAScalarValue(value))
    })
    .collect()
}

/// The one scalar value `'…'` holds, where its text is what stands between
/// the quotes.
pub fn decode_char(text: &str) -> Result<char, LiteralErrorKind> {
    match scalars(text)?.as_slice() {
        [one] => Ok(*one),
        _ => Err(LiteralErrorKind::NotOneChar),
    }
}

/// The bytes `b"…"` holds. A character of the text must be ASCII; a byte
/// above `\x7F` is written `\xNN`.
pub fn decode_bytes(text: &str) -> Result<Vec<u8>, LiteralErrorKind> {
    Escapes {
        rest: text.chars(),
        reach: Reach::Byte,
    }
    .map(|unit| {
        let Unit { value, escaped } = unit?;
        match u8::try_from(value) {
            Ok(byte) if escaped || byte.is_ascii() => Ok(byte),
            Ok(_) | Err(_) => Err(LiteralErrorKind::NotAscii(
                char::from_u32(value).expect("an unescaped unit came from a char"),
            )),
        }
    })
    .collect()
}

/// The one byte `b'…'` holds.
pub fn decode_byte(text: &str) -> Result<u8, LiteralErrorKind> {
    match decode_bytes(text)?.as_slice() {
        [one] => Ok(*one),
        _ => Err(LiteralErrorKind::NotOneChar),
    }
}

#[cfg(test)]
mod every_escape_is_rusts {
    use super::*;

    #[test]
    fn a_char_literal_is_one_scalar_value() {
        assert_eq!(decode_char("x"), Ok('x'));
        assert_eq!(decode_char("\\n"), Ok('\n'));
        assert_eq!(decode_char("\\r"), Ok('\r'));
        assert_eq!(decode_char("\\t"), Ok('\t'));
        assert_eq!(decode_char("\\0"), Ok('\0'));
        assert_eq!(decode_char("\\'"), Ok('\''));
        assert_eq!(decode_char("\\\\"), Ok('\\'));
        assert_eq!(decode_char("\\x41"), Ok('A'));
        assert_eq!(decode_char("\\u{1F600}"), Ok('\u{1F600}'));
        assert_eq!(decode_char("é"), Ok('é'));
    }

    #[test]
    fn a_char_literal_refuses_what_is_not_one_scalar_value() {
        assert_eq!(decode_char("ab"), Err(LiteralErrorKind::NotOneChar));
        assert_eq!(decode_char(""), Err(LiteralErrorKind::NotOneChar));
        assert_eq!(
            decode_char("\\u{D800}"),
            Err(LiteralErrorKind::NotAScalarValue(0xD800))
        );
        assert_eq!(
            decode_char("\\xFF"),
            Err(LiteralErrorKind::ByteEscapeOutOfRange(0xFF))
        );
        assert_eq!(
            decode_char("\\q"),
            Err(LiteralErrorKind::UnknownEscape('q'))
        );
    }

    #[test]
    fn a_byte_string_is_ascii_and_hex_escapes() {
        assert_eq!(decode_bytes("GET"), Ok(b"GET".to_vec()));
        assert_eq!(decode_bytes("\\xFF\\x00"), Ok(vec![0xFF, 0x00]));
        assert_eq!(decode_bytes("a\\nb"), Ok(b"a\nb".to_vec()));
        assert_eq!(decode_byte("G"), Ok(b'G'));
        assert_eq!(decode_byte("\\xFF"), Ok(0xFF));
    }

    #[test]
    fn a_byte_string_refuses_what_is_not_ascii() {
        assert_eq!(decode_bytes("é"), Err(LiteralErrorKind::NotAscii('é')));
        assert_eq!(
            decode_bytes("\\u{41}"),
            Err(LiteralErrorKind::UnicodeEscapeInByte)
        );
    }
}
