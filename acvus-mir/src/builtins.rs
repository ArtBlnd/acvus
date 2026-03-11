use acvus_utils::Interner;

use crate::ty::{Ty, TySubst};

/// Numeric identifier for a builtin function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BuiltinId {
    Filter,
    Map,
    Pmap,
    ToString,
    ToFloat,
    ToInt,
    Find,
    Reduce,
    Fold,
    Any,
    All,
    Len,
    Reverse,
    Flatten,
    Join,
    CharToInt,
    IntToChar,
    Contains,
    ContainsStr,
    Substring,
    LenStr,
    ToBytes,
    ToUtf8,
    ToUtf8Lossy,
    Trim,
    TrimStart,
    TrimEnd,
    Upper,
    Lower,
    ReplaceStr,
    SplitStr,
    StartsWithStr,
    EndsWithStr,
    RepeatStr,
    Unwrap,
    First,
    Last,
    UnwrapOr,
}

impl BuiltinId {
    pub fn name(self) -> &'static str {
        match self {
            Self::Filter => "filter",
            Self::Map => "map",
            Self::Pmap => "pmap",
            Self::ToString => "to_string",
            Self::ToFloat => "to_float",
            Self::ToInt => "to_int",
            Self::Find => "find",
            Self::Reduce => "reduce",
            Self::Fold => "fold",
            Self::Any => "any",
            Self::All => "all",
            Self::Len => "len",
            Self::Reverse => "reverse",
            Self::Flatten => "flatten",
            Self::Join => "join",
            Self::CharToInt => "char_to_int",
            Self::IntToChar => "int_to_char",
            Self::Contains => "contains",
            Self::ContainsStr => "contains_str",
            Self::Substring => "substring",
            Self::LenStr => "len_str",
            Self::ToBytes => "to_bytes",
            Self::ToUtf8 => "to_utf8",
            Self::ToUtf8Lossy => "to_utf8_lossy",
            Self::Trim => "trim",
            Self::TrimStart => "trim_start",
            Self::TrimEnd => "trim_end",
            Self::Upper => "upper",
            Self::Lower => "lower",
            Self::ReplaceStr => "replace_str",
            Self::SplitStr => "split_str",
            Self::StartsWithStr => "starts_with_str",
            Self::EndsWithStr => "ends_with_str",
            Self::RepeatStr => "repeat_str",
            Self::Unwrap => "unwrap",
            Self::First => "first",
            Self::Last => "last",
            Self::UnwrapOr => "unwrap_or",
        }
    }

    pub fn resolve(name: &str) -> Option<BuiltinId> {
        match name {
            "filter" => Some(Self::Filter),
            "map" => Some(Self::Map),
            "pmap" => Some(Self::Pmap),
            "to_string" => Some(Self::ToString),
            "to_float" => Some(Self::ToFloat),
            "to_int" => Some(Self::ToInt),
            "find" => Some(Self::Find),
            "reduce" => Some(Self::Reduce),
            "fold" => Some(Self::Fold),
            "any" => Some(Self::Any),
            "all" => Some(Self::All),
            "len" => Some(Self::Len),
            "reverse" => Some(Self::Reverse),
            "flatten" => Some(Self::Flatten),
            "join" => Some(Self::Join),
            "char_to_int" => Some(Self::CharToInt),
            "int_to_char" => Some(Self::IntToChar),
            "contains" => Some(Self::Contains),
            "contains_str" => Some(Self::ContainsStr),
            "substring" => Some(Self::Substring),
            "len_str" => Some(Self::LenStr),
            "to_bytes" => Some(Self::ToBytes),
            "to_utf8" => Some(Self::ToUtf8),
            "to_utf8_lossy" => Some(Self::ToUtf8Lossy),
            "trim" => Some(Self::Trim),
            "trim_start" => Some(Self::TrimStart),
            "trim_end" => Some(Self::TrimEnd),
            "upper" => Some(Self::Upper),
            "lower" => Some(Self::Lower),
            "replace_str" => Some(Self::ReplaceStr),
            "split_str" => Some(Self::SplitStr),
            "starts_with_str" => Some(Self::StartsWithStr),
            "ends_with_str" => Some(Self::EndsWithStr),
            "repeat_str" => Some(Self::RepeatStr),
            "unwrap" => Some(Self::Unwrap),
            "first" => Some(Self::First),
            "last" => Some(Self::Last),
            "unwrap_or" => Some(Self::UnwrapOr),
            _ => None,
        }
    }
}

/// Post-unification constraint on resolved arg types.
/// Returns `Some(error_message)` if the constraint is violated.
pub type BuiltinConstraint = fn(&[Ty], &Interner) -> Option<String>;

pub trait BuiltinSig {
    fn id(&self) -> BuiltinId;
    fn name(&self) -> &'static str;
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty);
    fn is_effectful(&self) -> bool {
        false
    }
    fn constraint(&self) -> Option<BuiltinConstraint> {
        None
    }
}

fn is_scalar(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Byte)
}

fn require_scalar(args: &[Ty], interner: &Interner) -> Option<String> {
    match &args[0] {
        ty if is_scalar(ty) => None,
        Ty::Var(_) | Ty::Error => None, // not yet resolved or error — skip
        ty => Some(format!(
            "`to_string` requires a scalar type (Int, Float, Bool, String, Byte), got {}",
            ty.display(interner),
        )),
    }
}

fn require_to_int(args: &[Ty], interner: &Interner) -> Option<String> {
    match &args[0] {
        Ty::Float | Ty::Byte => None,
        Ty::Var(_) | Ty::Error => None,
        ty => Some(format!(
            "`to_int` requires Float or Byte, got {}",
            ty.display(interner),
        )),
    }
}

// -- unit structs ---------------------------------------------------------

pub struct Filter;
impl BuiltinSig for Filter {
    fn id(&self) -> BuiltinId {
        BuiltinId::Filter
    }
    fn name(&self) -> &'static str {
        "filter"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                Ty::Fn {
                    params: vec![t.clone()],
                    ret: Box::new(Ty::Bool),
                    is_extern: false,
                },
            ],
            Ty::List(Box::new(t)),
        )
    }
}

pub struct Map;
impl BuiltinSig for Map {
    fn id(&self) -> BuiltinId {
        BuiltinId::Map
    }
    fn name(&self) -> &'static str {
        "map"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        let u = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                Ty::Fn {
                    params: vec![t],
                    ret: Box::new(u.clone()),
                    is_extern: false,
                },
            ],
            Ty::List(Box::new(u)),
        )
    }
}

pub struct Pmap;
impl BuiltinSig for Pmap {
    fn id(&self) -> BuiltinId {
        BuiltinId::Pmap
    }
    fn name(&self) -> &'static str {
        "pmap"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        let u = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                Ty::Fn {
                    params: vec![t],
                    ret: Box::new(u.clone()),
                    is_extern: false,
                },
            ],
            Ty::List(Box::new(u)),
        )
    }
}

pub struct ToString;
impl BuiltinSig for ToString {
    fn id(&self) -> BuiltinId {
        BuiltinId::ToString
    }
    fn name(&self) -> &'static str {
        "to_string"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![t], Ty::String)
    }
    fn constraint(&self) -> Option<BuiltinConstraint> {
        Some(require_scalar)
    }
}

pub struct ToFloat;
impl BuiltinSig for ToFloat {
    fn id(&self) -> BuiltinId {
        BuiltinId::ToFloat
    }
    fn name(&self) -> &'static str {
        "to_float"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::Int], Ty::Float)
    }
}

pub struct ToInt;
impl BuiltinSig for ToInt {
    fn id(&self) -> BuiltinId {
        BuiltinId::ToInt
    }
    fn name(&self) -> &'static str {
        "to_int"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![t], Ty::Int)
    }
    fn constraint(&self) -> Option<BuiltinConstraint> {
        Some(require_to_int)
    }
}

pub struct Find;
impl BuiltinSig for Find {
    fn id(&self) -> BuiltinId {
        BuiltinId::Find
    }
    fn name(&self) -> &'static str {
        "find"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                Ty::Fn {
                    params: vec![t.clone()],
                    ret: Box::new(Ty::Bool),
                    is_extern: false,
                },
            ],
            t,
        )
    }
}

pub struct Reduce;
impl BuiltinSig for Reduce {
    fn id(&self) -> BuiltinId {
        BuiltinId::Reduce
    }
    fn name(&self) -> &'static str {
        "reduce"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                Ty::Fn {
                    params: vec![t.clone(), t.clone()],
                    ret: Box::new(t.clone()),
                    is_extern: false,
                },
            ],
            t,
        )
    }
}

pub struct Fold;
impl BuiltinSig for Fold {
    fn id(&self) -> BuiltinId {
        BuiltinId::Fold
    }
    fn name(&self) -> &'static str {
        "fold"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        let u = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                u.clone(),
                Ty::Fn {
                    params: vec![u.clone(), t],
                    ret: Box::new(u.clone()),
                    is_extern: false,
                },
            ],
            u,
        )
    }
}

pub struct Any;
impl BuiltinSig for Any {
    fn id(&self) -> BuiltinId {
        BuiltinId::Any
    }
    fn name(&self) -> &'static str {
        "any"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                Ty::Fn {
                    params: vec![t],
                    ret: Box::new(Ty::Bool),
                    is_extern: false,
                },
            ],
            Ty::Bool,
        )
    }
}

pub struct All;
impl BuiltinSig for All {
    fn id(&self) -> BuiltinId {
        BuiltinId::All
    }
    fn name(&self) -> &'static str {
        "all"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (
            vec![
                Ty::List(Box::new(t.clone())),
                Ty::Fn {
                    params: vec![t],
                    ret: Box::new(Ty::Bool),
                    is_extern: false,
                },
            ],
            Ty::Bool,
        )
    }
}

pub struct Len;
impl BuiltinSig for Len {
    fn id(&self) -> BuiltinId {
        BuiltinId::Len
    }
    fn name(&self) -> &'static str {
        "len"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![Ty::List(Box::new(t))], Ty::Int)
    }
}

pub struct Reverse;
impl BuiltinSig for Reverse {
    fn id(&self) -> BuiltinId {
        BuiltinId::Reverse
    }
    fn name(&self) -> &'static str {
        "reverse"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![Ty::List(Box::new(t.clone()))], Ty::List(Box::new(t)))
    }
}

pub struct Flatten;
impl BuiltinSig for Flatten {
    fn id(&self) -> BuiltinId {
        BuiltinId::Flatten
    }
    fn name(&self) -> &'static str {
        "flatten"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (
            vec![Ty::List(Box::new(Ty::List(Box::new(t.clone()))))],
            Ty::List(Box::new(t)),
        )
    }
}

pub struct Join;
impl BuiltinSig for Join {
    fn id(&self) -> BuiltinId {
        BuiltinId::Join
    }
    fn name(&self) -> &'static str {
        "join"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::List(Box::new(Ty::String)), Ty::String], Ty::String)
    }
}

pub struct CharToInt;
impl BuiltinSig for CharToInt {
    fn id(&self) -> BuiltinId {
        BuiltinId::CharToInt
    }
    fn name(&self) -> &'static str {
        "char_to_int"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::Int)
    }
}

pub struct IntToChar;
impl BuiltinSig for IntToChar {
    fn id(&self) -> BuiltinId {
        BuiltinId::IntToChar
    }
    fn name(&self) -> &'static str {
        "int_to_char"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::Int], Ty::String)
    }
}

pub struct Contains;
impl BuiltinSig for Contains {
    fn id(&self) -> BuiltinId {
        BuiltinId::Contains
    }
    fn name(&self) -> &'static str {
        "contains"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![Ty::List(Box::new(t.clone())), t], Ty::Bool)
    }
}

pub struct ContainsStr;
impl BuiltinSig for ContainsStr {
    fn id(&self) -> BuiltinId {
        BuiltinId::ContainsStr
    }
    fn name(&self) -> &'static str {
        "contains_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::String], Ty::Bool)
    }
}

pub struct Substring;
impl BuiltinSig for Substring {
    fn id(&self) -> BuiltinId {
        BuiltinId::Substring
    }
    fn name(&self) -> &'static str {
        "substring"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::Int, Ty::Int], Ty::String)
    }
}

pub struct LenStr;
impl BuiltinSig for LenStr {
    fn id(&self) -> BuiltinId {
        BuiltinId::LenStr
    }
    fn name(&self) -> &'static str {
        "len_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::Int)
    }
}

pub struct ToBytes;
impl BuiltinSig for ToBytes {
    fn id(&self) -> BuiltinId {
        BuiltinId::ToBytes
    }
    fn name(&self) -> &'static str {
        "to_bytes"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::bytes())
    }
}

pub struct ToUtf8;
impl BuiltinSig for ToUtf8 {
    fn id(&self) -> BuiltinId {
        BuiltinId::ToUtf8
    }
    fn name(&self) -> &'static str {
        "to_utf8"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::bytes()], Ty::Option(Box::new(Ty::String)))
    }
}

pub struct ToUtf8Lossy;
impl BuiltinSig for ToUtf8Lossy {
    fn id(&self) -> BuiltinId {
        BuiltinId::ToUtf8Lossy
    }
    fn name(&self) -> &'static str {
        "to_utf8_lossy"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::bytes()], Ty::String)
    }
}

pub struct Trim;
impl BuiltinSig for Trim {
    fn id(&self) -> BuiltinId {
        BuiltinId::Trim
    }
    fn name(&self) -> &'static str {
        "trim"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::String)
    }
}

pub struct TrimStart;
impl BuiltinSig for TrimStart {
    fn id(&self) -> BuiltinId {
        BuiltinId::TrimStart
    }
    fn name(&self) -> &'static str {
        "trim_start"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::String)
    }
}

pub struct TrimEnd;
impl BuiltinSig for TrimEnd {
    fn id(&self) -> BuiltinId {
        BuiltinId::TrimEnd
    }
    fn name(&self) -> &'static str {
        "trim_end"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::String)
    }
}

pub struct Upper;
impl BuiltinSig for Upper {
    fn id(&self) -> BuiltinId {
        BuiltinId::Upper
    }
    fn name(&self) -> &'static str {
        "upper"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::String)
    }
}

pub struct Lower;
impl BuiltinSig for Lower {
    fn id(&self) -> BuiltinId {
        BuiltinId::Lower
    }
    fn name(&self) -> &'static str {
        "lower"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::String)
    }
}

pub struct ReplaceStr;
impl BuiltinSig for ReplaceStr {
    fn id(&self) -> BuiltinId {
        BuiltinId::ReplaceStr
    }
    fn name(&self) -> &'static str {
        "replace_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::String, Ty::String], Ty::String)
    }
}

pub struct SplitStr;
impl BuiltinSig for SplitStr {
    fn id(&self) -> BuiltinId {
        BuiltinId::SplitStr
    }
    fn name(&self) -> &'static str {
        "split_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::String], Ty::List(Box::new(Ty::String)))
    }
}

pub struct StartsWithStr;
impl BuiltinSig for StartsWithStr {
    fn id(&self) -> BuiltinId {
        BuiltinId::StartsWithStr
    }
    fn name(&self) -> &'static str {
        "starts_with_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::String], Ty::Bool)
    }
}

pub struct EndsWithStr;
impl BuiltinSig for EndsWithStr {
    fn id(&self) -> BuiltinId {
        BuiltinId::EndsWithStr
    }
    fn name(&self) -> &'static str {
        "ends_with_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::String], Ty::Bool)
    }
}

pub struct RepeatStr;
impl BuiltinSig for RepeatStr {
    fn id(&self) -> BuiltinId {
        BuiltinId::RepeatStr
    }
    fn name(&self) -> &'static str {
        "repeat_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::Int], Ty::String)
    }
}

pub struct Unwrap;
impl BuiltinSig for Unwrap {
    fn id(&self) -> BuiltinId {
        BuiltinId::Unwrap
    }
    fn name(&self) -> &'static str {
        "unwrap"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![Ty::Option(Box::new(t.clone()))], t)
    }
}

pub struct First;
impl BuiltinSig for First {
    fn id(&self) -> BuiltinId {
        BuiltinId::First
    }
    fn name(&self) -> &'static str {
        "first"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![Ty::List(Box::new(t.clone()))], Ty::Option(Box::new(t)))
    }
}

pub struct Last;
impl BuiltinSig for Last {
    fn id(&self) -> BuiltinId {
        BuiltinId::Last
    }
    fn name(&self) -> &'static str {
        "last"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![Ty::List(Box::new(t.clone()))], Ty::Option(Box::new(t)))
    }
}

pub struct UnwrapOr;
impl BuiltinSig for UnwrapOr {
    fn id(&self) -> BuiltinId {
        BuiltinId::UnwrapOr
    }
    fn name(&self) -> &'static str {
        "unwrap_or"
    }
    fn signature(&self, subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        let t = subst.fresh_var();
        (vec![Ty::Option(Box::new(t.clone())), t.clone()], t)
    }
}

pub fn builtins() -> Vec<(BuiltinId, &'static dyn BuiltinSig)> {
    vec![
        (BuiltinId::Filter, &Filter as &dyn BuiltinSig),
        (BuiltinId::Map, &Map),
        (BuiltinId::Pmap, &Pmap),
        (BuiltinId::ToString, &ToString),
        (BuiltinId::ToFloat, &ToFloat),
        (BuiltinId::ToInt, &ToInt),
        (BuiltinId::Find, &Find),
        (BuiltinId::Reduce, &Reduce),
        (BuiltinId::Fold, &Fold),
        (BuiltinId::Any, &Any),
        (BuiltinId::All, &All),
        (BuiltinId::Len, &Len),
        (BuiltinId::Reverse, &Reverse),
        (BuiltinId::Flatten, &Flatten),
        (BuiltinId::Join, &Join),
        (BuiltinId::CharToInt, &CharToInt),
        (BuiltinId::IntToChar, &IntToChar),
        (BuiltinId::Contains, &Contains),
        (BuiltinId::ContainsStr, &ContainsStr),
        (BuiltinId::Substring, &Substring),
        (BuiltinId::LenStr, &LenStr),
        (BuiltinId::ToBytes, &ToBytes),
        (BuiltinId::ToUtf8, &ToUtf8),
        (BuiltinId::ToUtf8Lossy, &ToUtf8Lossy),
        (BuiltinId::Trim, &Trim),
        (BuiltinId::TrimStart, &TrimStart),
        (BuiltinId::TrimEnd, &TrimEnd),
        (BuiltinId::Upper, &Upper),
        (BuiltinId::Lower, &Lower),
        (BuiltinId::ReplaceStr, &ReplaceStr),
        (BuiltinId::SplitStr, &SplitStr),
        (BuiltinId::StartsWithStr, &StartsWithStr),
        (BuiltinId::EndsWithStr, &EndsWithStr),
        (BuiltinId::RepeatStr, &RepeatStr),
        (BuiltinId::Unwrap, &Unwrap),
        (BuiltinId::First, &First),
        (BuiltinId::Last, &Last),
        (BuiltinId::UnwrapOr, &UnwrapOr),
    ]
}
