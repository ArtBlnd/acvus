use crate::ty::{Ty, TySubst};

/// Post-unification constraint on resolved arg types.
/// Returns `Some(error_message)` if the constraint is violated.
pub type BuiltinConstraint = fn(&[Ty]) -> Option<String>;

pub trait BuiltinSig {
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

fn require_scalar(args: &[Ty]) -> Option<String> {
    match &args[0] {
        ty if is_scalar(ty) => None,
        Ty::Var(_) | Ty::Error => None, // not yet resolved or error — skip
        ty => Some(format!(
            "`to_string` requires a scalar type (Int, Float, Bool, String, Byte), got {ty}",
        )),
    }
}

fn require_to_int(args: &[Ty]) -> Option<String> {
    match &args[0] {
        Ty::Float | Ty::Byte => None,
        Ty::Var(_) | Ty::Error => None,
        ty => Some(format!("`to_int` requires Float or Byte, got {ty}",)),
    }
}

// -- unit structs ---------------------------------------------------------

pub struct Filter;
impl BuiltinSig for Filter {
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
                },
            ],
            Ty::List(Box::new(t)),
        )
    }
}

pub struct Map;
impl BuiltinSig for Map {
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
                },
            ],
            Ty::List(Box::new(u)),
        )
    }
}

pub struct Pmap;
impl BuiltinSig for Pmap {
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
                },
            ],
            Ty::List(Box::new(u)),
        )
    }
}

pub struct ToString;
impl BuiltinSig for ToString {
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
    fn name(&self) -> &'static str {
        "to_float"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::Int], Ty::Float)
    }
}

pub struct ToInt;
impl BuiltinSig for ToInt {
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
                },
            ],
            t,
        )
    }
}

pub struct Reduce;
impl BuiltinSig for Reduce {
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
                },
            ],
            t,
        )
    }
}

pub struct Fold;
impl BuiltinSig for Fold {
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
                },
            ],
            u,
        )
    }
}

pub struct Any;
impl BuiltinSig for Any {
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
                },
            ],
            Ty::Bool,
        )
    }
}

pub struct All;
impl BuiltinSig for All {
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
                },
            ],
            Ty::Bool,
        )
    }
}

pub struct Len;
impl BuiltinSig for Len {
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
    fn name(&self) -> &'static str {
        "join"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::List(Box::new(Ty::String)), Ty::String], Ty::String)
    }
}

pub struct CharToInt;
impl BuiltinSig for CharToInt {
    fn name(&self) -> &'static str {
        "char_to_int"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::Int)
    }
}

pub struct IntToChar;
impl BuiltinSig for IntToChar {
    fn name(&self) -> &'static str {
        "int_to_char"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::Int], Ty::String)
    }
}

pub struct Contains;
impl BuiltinSig for Contains {
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
    fn name(&self) -> &'static str {
        "contains_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::String], Ty::Bool)
    }
}

pub struct Substring;
impl BuiltinSig for Substring {
    fn name(&self) -> &'static str {
        "substring"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String, Ty::Int, Ty::Int], Ty::String)
    }
}

pub struct LenStr;
impl BuiltinSig for LenStr {
    fn name(&self) -> &'static str {
        "len_str"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::Int)
    }
}

pub struct ToBytes;
impl BuiltinSig for ToBytes {
    fn name(&self) -> &'static str {
        "to_bytes"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::String], Ty::bytes())
    }
}

pub struct ToUtf8;
impl BuiltinSig for ToUtf8 {
    fn name(&self) -> &'static str {
        "to_utf8"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::bytes()], Ty::String)
    }
}

pub struct ToUtf8Lossy;
impl BuiltinSig for ToUtf8Lossy {
    fn name(&self) -> &'static str {
        "to_utf8_lossy"
    }
    fn signature(&self, _subst: &mut TySubst) -> (Vec<Ty>, Ty) {
        (vec![Ty::bytes()], Ty::String)
    }
}

pub fn builtins() -> Vec<&'static dyn BuiltinSig> {
    vec![
        &Filter,
        &Map,
        &Pmap,
        &ToString,
        &ToFloat,
        &ToInt,
        &Find,
        &Reduce,
        &Fold,
        &Any,
        &All,
        &Len,
        &Reverse,
        &Flatten,
        &Join,
        &CharToInt,
        &IntToChar,
        &Contains,
        &ContainsStr,
        &Substring,
        &LenStr,
        &ToBytes,
        &ToUtf8,
        &ToUtf8Lossy,
    ]
}
