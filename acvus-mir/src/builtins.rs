use std::sync::LazyLock;

use acvus_utils::Interner;
use rustc_hash::FxHashMap;

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
    Iter,
    RevIter,
    Collect,
    Take,
    Skip,
    Chain,
    // -- Iterator overloads --
    FlattenIter,
    FlatMapIter,
    JoinIter,
    ContainsIter,
    FirstIter,
    LastIter,
}

impl BuiltinId {
    pub fn name(self) -> &'static str {
        REGISTRY.get(self).name
    }
}

// ---------------------------------------------------------------------------
// BuiltinEntry — data-driven replacement for the old unit-struct trait impls
// ---------------------------------------------------------------------------

/// Post-unification constraint on resolved arg types.
/// Returns `Some(error_message)` if the constraint is violated.
pub type BuiltinConstraint = fn(&[Ty], &Interner) -> Option<String>;

pub struct BuiltinEntry {
    pub id: BuiltinId,
    pub name: &'static str,
    pub signature: fn(&mut TySubst) -> (Vec<Ty>, Ty),
    pub constraint: Option<BuiltinConstraint>,
}

// ---------------------------------------------------------------------------
// BuiltinRegistry — central lookup, supports overloaded names
// ---------------------------------------------------------------------------

pub struct BuiltinRegistry {
    entries: FxHashMap<BuiltinId, BuiltinEntry>,
    by_name: FxHashMap<&'static str, Vec<BuiltinId>>,
}

impl BuiltinRegistry {
    fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
            by_name: FxHashMap::default(),
        }
    }

    fn add(
        &mut self,
        name: &'static str,
        id: BuiltinId,
        signature: fn(&mut TySubst) -> (Vec<Ty>, Ty),
        constraint: Option<BuiltinConstraint>,
    ) {
        self.entries.insert(
            id,
            BuiltinEntry {
                id,
                name,
                signature,
                constraint,
            },
        );
        self.by_name.entry(name).or_default().push(id);
    }

    /// Look up a single entry by ID.
    pub fn get(&self, id: BuiltinId) -> &BuiltinEntry {
        &self.entries[&id]
    }

    /// Return all candidate IDs for a given user-facing name.
    /// Empty slice means "not a builtin".
    pub fn candidates(&self, name: &str) -> &[BuiltinId] {
        self.by_name.get(name).map_or(&[], |v| v.as_slice())
    }

    /// Check if a name is a known builtin (any overload).
    pub fn is_builtin(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Global registry (built once, read-only afterwards)
// ---------------------------------------------------------------------------

pub static REGISTRY: LazyLock<BuiltinRegistry> = LazyLock::new(build_registry);

/// Access the global registry.
pub fn registry() -> &'static BuiltinRegistry {
    &REGISTRY
}

// ---------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------

fn is_scalar(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Byte)
}

fn require_scalar(args: &[Ty], interner: &Interner) -> Option<String> {
    match &args[0] {
        ty if is_scalar(ty) => None,
        Ty::Var(_) | Ty::Error => None,
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

// ---------------------------------------------------------------------------
// Signature helpers (each is a `fn(&mut TySubst) -> (Vec<Ty>, Ty)`)
// ---------------------------------------------------------------------------

fn sig_filter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone())),
            Ty::Fn { params: vec![t.clone()], ret: Box::new(Ty::Bool), is_extern: false },
        ],
        Ty::Iterator(Box::new(t)),
    )
}

fn sig_map(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    let u = s.fresh_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone())),
            Ty::Fn { params: vec![t], ret: Box::new(u.clone()), is_extern: false },
        ],
        Ty::Iterator(Box::new(u)),
    )
}

fn sig_pmap(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_map(s)
}

fn sig_to_string(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![t], Ty::String)
}

fn sig_to_float(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::Int], Ty::Float)
}

fn sig_to_int(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![t], Ty::Int)
}

fn sig_find(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone())),
            Ty::Fn { params: vec![t.clone()], ret: Box::new(Ty::Bool), is_extern: false },
        ],
        t,
    )
}

fn sig_reduce(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone())),
            Ty::Fn { params: vec![t.clone(), t.clone()], ret: Box::new(t.clone()), is_extern: false },
        ],
        t,
    )
}

fn sig_fold(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    let u = s.fresh_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone())),
            u.clone(),
            Ty::Fn { params: vec![u.clone(), t], ret: Box::new(u.clone()), is_extern: false },
        ],
        u,
    )
}

fn sig_any(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone())),
            Ty::Fn { params: vec![t], ret: Box::new(Ty::Bool), is_extern: false },
        ],
        Ty::Bool,
    )
}

fn sig_all(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_any(s)
}

fn sig_len(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::List(Box::new(t))], Ty::Int)
}

fn sig_reverse(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::List(Box::new(t.clone()))], Ty::List(Box::new(t)))
}

fn sig_flatten(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (
        vec![Ty::List(Box::new(Ty::List(Box::new(t.clone()))))],
        Ty::List(Box::new(t)),
    )
}

fn sig_flatten_iter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (
        vec![Ty::Iterator(Box::new(Ty::List(Box::new(t.clone()))))],
        Ty::Iterator(Box::new(t)),
    )
}

fn sig_flat_map_iter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    let u = s.fresh_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone())),
            Ty::Fn {
                params: vec![t],
                ret: Box::new(Ty::List(Box::new(u.clone()))),
                is_extern: false,
            },
        ],
        Ty::Iterator(Box::new(u)),
    )
}

fn sig_join(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::List(Box::new(Ty::String)), Ty::String], Ty::String)
}

fn sig_join_iter(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::Iterator(Box::new(Ty::String)), Ty::String], Ty::String)
}

fn sig_char_to_int(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String], Ty::Int)
}

fn sig_int_to_char(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::Int], Ty::String)
}

fn sig_contains(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::List(Box::new(t.clone())), t], Ty::Bool)
}

fn sig_contains_iter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::Iterator(Box::new(t.clone())), t], Ty::Bool)
}

fn sig_contains_str(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String, Ty::String], Ty::Bool)
}

fn sig_substring(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String, Ty::Int, Ty::Int], Ty::String)
}

fn sig_len_str(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String], Ty::Int)
}

fn sig_to_bytes(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String], Ty::bytes())
}

fn sig_to_utf8(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::bytes()], Ty::Option(Box::new(Ty::String)))
}

fn sig_to_utf8_lossy(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::bytes()], Ty::String)
}

fn sig_str_to_str(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String], Ty::String)
}

fn sig_replace_str(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String, Ty::String, Ty::String], Ty::String)
}

fn sig_split_str(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String, Ty::String], Ty::List(Box::new(Ty::String)))
}

fn sig_str_str_to_bool(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String, Ty::String], Ty::Bool)
}

fn sig_repeat_str(_s: &mut TySubst) -> (Vec<Ty>, Ty) {
    (vec![Ty::String, Ty::Int], Ty::String)
}

fn sig_unwrap(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::Option(Box::new(t.clone()))], t)
}

fn sig_first(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::List(Box::new(t.clone()))], Ty::Option(Box::new(t)))
}

fn sig_first_iter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::Iterator(Box::new(t.clone()))], Ty::Option(Box::new(t)))
}

fn sig_last(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::List(Box::new(t.clone()))], Ty::Option(Box::new(t)))
}

fn sig_last_iter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::Iterator(Box::new(t.clone()))], Ty::Option(Box::new(t)))
}

fn sig_unwrap_or(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::Option(Box::new(t.clone())), t.clone()], t)
}

fn sig_iter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::List(Box::new(t.clone()))], Ty::Iterator(Box::new(t)))
}

fn sig_rev_iter(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_iter(s)
}

fn sig_collect(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::Iterator(Box::new(t.clone()))], Ty::List(Box::new(t)))
}

fn sig_take(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (vec![Ty::Iterator(Box::new(t.clone())), Ty::Int], Ty::Iterator(Box::new(t)))
}

fn sig_skip(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_take(s)
}

fn sig_chain(s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_var();
    (
        vec![Ty::Iterator(Box::new(t.clone())), Ty::Iterator(Box::new(t.clone()))],
        Ty::Iterator(Box::new(t)),
    )
}

// ---------------------------------------------------------------------------
// Registry construction
// ---------------------------------------------------------------------------

fn build_registry() -> BuiltinRegistry {
    let mut r = BuiltinRegistry::new();

    // -- Iterator HOFs --
    r.add("filter",      BuiltinId::Filter,      sig_filter,        None);
    r.add("map",         BuiltinId::Map,          sig_map,           None);
    r.add("pmap",        BuiltinId::Pmap,         sig_pmap,          None);
    r.add("find",        BuiltinId::Find,         sig_find,          None);
    r.add("reduce",      BuiltinId::Reduce,       sig_reduce,        None);
    r.add("fold",        BuiltinId::Fold,         sig_fold,          None);
    r.add("any",         BuiltinId::Any,          sig_any,           None);
    r.add("all",         BuiltinId::All,          sig_all,           None);

    // -- Conversions --
    r.add("to_string",   BuiltinId::ToString,     sig_to_string,     Some(require_scalar));
    r.add("to_float",    BuiltinId::ToFloat,      sig_to_float,      None);
    r.add("to_int",      BuiltinId::ToInt,        sig_to_int,        Some(require_to_int));
    r.add("char_to_int", BuiltinId::CharToInt,    sig_char_to_int,   None);
    r.add("int_to_char", BuiltinId::IntToChar,    sig_int_to_char,   None);

    // -- List ops --
    r.add("len",         BuiltinId::Len,          sig_len,           None);
    r.add("reverse",     BuiltinId::Reverse,      sig_reverse,       None);

    // -- flatten (overloaded: List + Iterator) --
    r.add("flatten",     BuiltinId::Flatten,      sig_flatten,       None);
    r.add("flatten",     BuiltinId::FlattenIter,  sig_flatten_iter,  None);

    // -- flat_map (Iterator only) --
    r.add("flat_map",    BuiltinId::FlatMapIter,  sig_flat_map_iter, None);

    // -- join (overloaded: List + Iterator) --
    r.add("join",        BuiltinId::Join,         sig_join,          None);
    r.add("join",        BuiltinId::JoinIter,     sig_join_iter,     None);

    // -- contains (overloaded: List + Iterator) --
    r.add("contains",    BuiltinId::Contains,     sig_contains,      None);
    r.add("contains",    BuiltinId::ContainsIter, sig_contains_iter, None);

    // -- first / last (overloaded: List + Iterator) --
    r.add("first",       BuiltinId::First,        sig_first,         None);
    r.add("first",       BuiltinId::FirstIter,    sig_first_iter,    None);
    r.add("last",        BuiltinId::Last,         sig_last,          None);
    r.add("last",        BuiltinId::LastIter,     sig_last_iter,     None);

    // -- String ops --
    r.add("contains_str",    BuiltinId::ContainsStr,    sig_contains_str,     None);
    r.add("substring",       BuiltinId::Substring,      sig_substring,        None);
    r.add("len_str",         BuiltinId::LenStr,         sig_len_str,          None);
    r.add("to_bytes",        BuiltinId::ToBytes,        sig_to_bytes,         None);
    r.add("to_utf8",         BuiltinId::ToUtf8,         sig_to_utf8,          None);
    r.add("to_utf8_lossy",   BuiltinId::ToUtf8Lossy,    sig_to_utf8_lossy,    None);
    r.add("trim",            BuiltinId::Trim,           sig_str_to_str,       None);
    r.add("trim_start",      BuiltinId::TrimStart,      sig_str_to_str,       None);
    r.add("trim_end",        BuiltinId::TrimEnd,        sig_str_to_str,       None);
    r.add("upper",           BuiltinId::Upper,          sig_str_to_str,       None);
    r.add("lower",           BuiltinId::Lower,          sig_str_to_str,       None);
    r.add("replace_str",     BuiltinId::ReplaceStr,     sig_replace_str,      None);
    r.add("split_str",       BuiltinId::SplitStr,       sig_split_str,        None);
    r.add("starts_with_str", BuiltinId::StartsWithStr,  sig_str_str_to_bool,  None);
    r.add("ends_with_str",   BuiltinId::EndsWithStr,    sig_str_str_to_bool,  None);
    r.add("repeat_str",      BuiltinId::RepeatStr,      sig_repeat_str,       None);

    // -- Option ops --
    r.add("unwrap",    BuiltinId::Unwrap,    sig_unwrap,    None);
    r.add("unwrap_or", BuiltinId::UnwrapOr,  sig_unwrap_or, None);

    // -- Iterator constructors --
    r.add("iter",      BuiltinId::Iter,      sig_iter,      None);
    r.add("rev_iter",  BuiltinId::RevIter,   sig_rev_iter,  None);
    r.add("collect",   BuiltinId::Collect,   sig_collect,   None);
    r.add("take",      BuiltinId::Take,      sig_take,      None);
    r.add("skip",      BuiltinId::Skip,      sig_skip,      None);
    r.add("chain",     BuiltinId::Chain,     sig_chain,     None);

    r
}
