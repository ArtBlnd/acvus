use super::kind::TyKind;
use super::Phase;

/// The width and signedness of an integer type: `i8` to `i64`, `u8` to
/// `u64` (operators §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntTy {
    pub const ALL: [IntTy; 8] = [
        IntTy::I8,
        IntTy::I16,
        IntTy::I32,
        IntTy::I64,
        IntTy::U8,
        IntTy::U16,
        IntTy::U32,
        IntTy::U64,
    ];

    pub fn bits(self) -> u32 {
        match self {
            IntTy::I8 | IntTy::U8 => 8,
            IntTy::I16 | IntTy::U16 => 16,
            IntTy::I32 | IntTy::U32 => 32,
            IntTy::I64 | IntTy::U64 => 64,
        }
    }

    pub fn bytes(self) -> usize {
        self.bits() as usize / 8
    }

    pub fn signed(self) -> bool {
        matches!(self, IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64)
    }

    pub fn name(self) -> &'static str {
        match self {
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::U8 => "u8",
            IntTy::U16 => "u16",
            IntTy::U32 => "u32",
            IntTy::U64 => "u64",
        }
    }

    pub fn min(self) -> i128 {
        if self.signed() {
            -(1i128 << (self.bits() - 1))
        } else {
            0
        }
    }

    pub fn max(self) -> i128 {
        if self.signed() {
            (1i128 << (self.bits() - 1)) - 1
        } else {
            (1i128 << self.bits()) - 1
        }
    }

    /// Whether `value` is representable in this type.
    pub fn holds(self, value: i128) -> bool {
        (self.min()..=self.max()).contains(&value)
    }

    /// The value the low `bits()` of `bits` spell, sign- or zero-extended
    /// to `i128`.
    pub fn read(self, bits: u64) -> i128 {
        match self {
            IntTy::I8 => bits as u8 as i8 as i128,
            IntTy::I16 => bits as u16 as i16 as i128,
            IntTy::I32 => bits as u32 as i32 as i128,
            IntTy::I64 => bits as i64 as i128,
            IntTy::U8 => bits as u8 as i128,
            IntTy::U16 => bits as u16 as i128,
            IntTy::U32 => bits as u32 as i128,
            IntTy::U64 => bits as i128,
        }
    }
}

impl From<acvus_ast::IntWidth> for IntTy {
    fn from(width: acvus_ast::IntWidth) -> IntTy {
        match width {
            acvus_ast::IntWidth::I8 => IntTy::I8,
            acvus_ast::IntWidth::I16 => IntTy::I16,
            acvus_ast::IntWidth::I32 => IntTy::I32,
            acvus_ast::IntWidth::I64 => IntTy::I64,
            acvus_ast::IntWidth::U8 => IntTy::U8,
            acvus_ast::IntWidth::U16 => IntTy::U16,
            acvus_ast::IntWidth::U32 => IntTy::U32,
            acvus_ast::IntWidth::U64 => IntTy::U64,
        }
    }
}

/// The type an `as` cast names, at either end of it (operators §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CastTy {
    Int(IntTy),
    F64,
    Char,
}

impl CastTy {
    pub const NAMES: &'static str = "i8, i16, i32, i64, u8, u16, u32, u64, f64 or char";

    pub fn of_name(name: &str) -> Option<CastTy> {
        match name {
            "f64" => Some(CastTy::F64),
            "char" => Some(CastTy::Char),
            _ => IntTy::ALL
                .into_iter()
                .find(|k| k.name() == name)
                .map(CastTy::Int),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CastTy::Int(k) => k.name(),
            CastTy::F64 => "f64",
            CastTy::Char => "char",
        }
    }

    pub fn of_kind<V: Phase>(kind: &TyKind<V>) -> Option<CastTy> {
        match kind {
            TyKind::Int(k) => Some(CastTy::Int(*k)),
            TyKind::Float => Some(CastTy::F64),
            TyKind::Char => Some(CastTy::Char),
            _ => None,
        }
    }

    /// The numeric type whose word this type's word is. A `char` is a
    /// Unicode scalar value, which is its `u32`, and every cast Rust admits
    /// at either end of a `char` has the value of the same cast through
    /// `u32`.
    pub fn word(self) -> WordTy {
        match self {
            CastTy::Int(k) => WordTy::Int(k),
            CastTy::F64 => WordTy::F64,
            CastTy::Char => WordTy::Int(IntTy::U32),
        }
    }

    /// Whether `self as to` is a cast Rust admits: every pair of numbers,
    /// a `char` to any integer, and `u8` to `char` and nothing else.
    pub fn admits(self, to: CastTy) -> bool {
        match (self, to) {
            (_, CastTy::Char) => self == CastTy::Int(IntTy::U8),
            (CastTy::Char, CastTy::F64) => false,
            (CastTy::Char, _) | (CastTy::Int(_) | CastTy::F64, _) => true,
        }
    }
}

impl<V: Phase> From<CastTy> for TyKind<V> {
    fn from(cast: CastTy) -> TyKind<V> {
        match cast {
            CastTy::Int(k) => TyKind::Int(k),
            CastTy::F64 => TyKind::Float,
            CastTy::Char => TyKind::Char,
        }
    }
}

/// The type a word is read and written at: the eight integer widths and
/// `f64`. A `char` has no word of its own (`CastTy::word`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WordTy {
    Int(IntTy),
    F64,
}
