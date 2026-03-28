use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::graph::types::{Constraint, FnConstraint, FnKind, Function, QualifiedRef, Signature};
use crate::ty::{Effect, Param, Ty, TySubst};

/// Shorthand: build a `Param` with a positional dummy name `_0`, `_1`, …
fn p(interner: &Interner, idx: usize, ty: Ty) -> Param {
    Param::new(interner.intern(&format!("_{idx}")), ty)
}

/// Type alias for builtin signature generators.
type SigFn = fn(&Interner, &mut TySubst) -> (Vec<Ty>, Ty);

// ---------------------------------------------------------------------------
// Signature helpers (each is a `fn(&Interner, &mut TySubst) -> (Vec<Ty>, Ty)`)
// ---------------------------------------------------------------------------

pub(crate) fn sig_filter(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            Ty::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(Ty::Bool),
                captures: vec![],
                effect: e.clone(),
            },
        ],
        Ty::Iterator(Box::new(t), e),
    )
}

fn sig_map(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let u = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            Ty::Fn {
                params: vec![p(interner, 0, t)],
                ret: Box::new(u.clone()),
                captures: vec![],
                effect: e.clone(),
            },
        ],
        Ty::Iterator(Box::new(u), e),
    )
}

fn sig_pmap(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_map(interner, s)
}

fn sig_find(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            Ty::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(Ty::Bool),
                captures: vec![],
                effect: e.clone(),
            },
        ],
        t,
    )
}

fn sig_reduce(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            Ty::Fn {
                params: vec![p(interner, 0, t.clone()), p(interner, 1, t.clone())],
                ret: Box::new(t.clone()),
                captures: vec![],
                effect: e.clone(),
            },
        ],
        t,
    )
}

fn sig_fold(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let u = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            u.clone(),
            Ty::Fn {
                params: vec![p(interner, 0, u.clone()), p(interner, 1, t)],
                ret: Box::new(u.clone()),
                captures: vec![],
                effect: e.clone(),
            },
        ],
        u,
    )
}

fn sig_any(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            Ty::Fn {
                params: vec![p(interner, 0, t)],
                ret: Box::new(Ty::Bool),
                captures: vec![],
                effect: e.clone(),
            },
        ],
        Ty::Bool,
    )
}

fn sig_all(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_any(interner, s)
}

fn sig_flatten_iter(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(
            Box::new(Ty::List(Box::new(t.clone()))),
            e.clone(),
        )],
        Ty::Iterator(Box::new(t), e),
    )
}

fn sig_join_iter(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(Box::new(Ty::String), e.clone()), Ty::String],
        Ty::String,
    )
}

fn sig_contains_iter(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(Box::new(t.clone()), e.clone()), t],
        Ty::Bool,
    )
}

fn sig_next(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(Box::new(t.clone()), e.clone())],
        Ty::Option(Box::new(Ty::Tuple(vec![
            t.clone(),
            Ty::Iterator(Box::new(t), e),
        ]))),
    )
}

fn sig_next_seq(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let o = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Sequence(Box::new(t.clone()), Box::new(o.clone()), e.clone())],
        Ty::Option(Box::new(Ty::Tuple(vec![
            t.clone(),
            Ty::Sequence(Box::new(t), Box::new(o), e),
        ]))),
    )
}

fn sig_first_iter(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(Box::new(t.clone()), e.clone())],
        Ty::Option(Box::new(t)),
    )
}

fn sig_last_iter(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(Box::new(t.clone()), e.clone())],
        Ty::Option(Box::new(t)),
    )
}

// ---------------------------------------------------------------------------
// Signature helpers — Deque ops (identity-preserving)
// ---------------------------------------------------------------------------

fn sig_append(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let o = s.fresh_param();
    (
        vec![Ty::Deque(Box::new(t.clone()), Box::new(o.clone())), t.clone()],
        Ty::Deque(Box::new(t), Box::new(o)),
    )
}

fn sig_extend(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let o = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Deque(Box::new(t.clone()), Box::new(o.clone())),
            Ty::Iterator(Box::new(t.clone()), e.clone()),
        ],
        Ty::Deque(Box::new(t), Box::new(o)),
    )
}

fn sig_consume(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let o = s.fresh_param();
    (
        vec![Ty::Deque(Box::new(t.clone()), Box::new(o.clone())), Ty::Int],
        Ty::Deque(Box::new(t), Box::new(o)),
    )
}

// (Iterator<T>, Fn(T) → Iterator<U>) → Iterator<U>
fn sig_flat_map(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let u = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            Ty::Fn {
                params: vec![p(interner, 0, t)],
                ret: Box::new(Ty::Iterator(Box::new(u.clone()), e.clone())),
                captures: vec![],
                effect: e.clone(),
            },
        ],
        Ty::Iterator(Box::new(u), e),
    )
}

fn sig_iter(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    (
        vec![Ty::List(Box::new(t.clone()))],
        Ty::Iterator(Box::new(t), Effect::pure()),
    )
}

fn sig_rev_iter(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_iter(interner, s)
}

fn sig_collect(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(Box::new(t.clone()), e.clone())],
        Ty::List(Box::new(t)),
    )
}

fn sig_take(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Iterator(Box::new(t.clone()), e.clone()), Ty::Int],
        Ty::Iterator(Box::new(t), e),
    )
}

fn sig_skip(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_take(interner, s)
}

fn sig_pchain(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::List(Box::new(Ty::Iterator(Box::new(t.clone()), e.clone())))],
        Ty::Iterator(Box::new(t), e),
    )
}

fn sig_chain(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Iterator(Box::new(t.clone()), e.clone()),
            Ty::Iterator(Box::new(t.clone()), e.clone()),
        ],
        Ty::Iterator(Box::new(t), e),
    )
}

// ---------------------------------------------------------------------------
// Signature helpers — Sequence ops (lazy Deque)
// ---------------------------------------------------------------------------

// Structural ops: same identity preserved

fn sig_take_seq(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let o = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![Ty::Sequence(Box::new(t.clone()), Box::new(o.clone()), e.clone()), Ty::Int],
        Ty::Sequence(Box::new(t), Box::new(o), e),
    )
}

fn sig_skip_seq(interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    sig_take_seq(interner, s)
}

// chain_seq: (Sequence<T, O, E>, Iterator<T, E>) → Sequence<T, O, E>
// Second argument is Iterator (not Sequence) — chain appends from any source.
fn sig_chain_seq(_interner: &Interner, s: &mut TySubst) -> (Vec<Ty>, Ty) {
    let t = s.fresh_param();
    let o = s.fresh_param();
    let e = s.fresh_effect_var();
    (
        vec![
            Ty::Sequence(Box::new(t.clone()), Box::new(o.clone()), e.clone()),
            Ty::Iterator(Box::new(t.clone()), e.clone()),
        ],
        Ty::Sequence(Box::new(t), Box::new(o), e),
    )
}

// ---------------------------------------------------------------------------
// Graph Function generation
// ---------------------------------------------------------------------------

/// Build a graph `Function` from a name and signature generator.
fn make_builtin(interner: &Interner, name: &str, sig_fn: SigFn) -> Function {
    let mut sig_subst = TySubst::new();
    let (params, ret) = sig_fn(interner, &mut sig_subst);
    let named_params: Vec<Param> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty.clone()))
        .collect();
    Function {
        qref: QualifiedRef::root(interner.intern(name)),
        kind: FnKind::Extern,
        constraint: FnConstraint {
            signature: Some(Signature {
                params: named_params.clone(),
            }),
            output: Constraint::Exact(Ty::Fn {
                params: named_params,
                ret: Box::new(ret),
                captures: vec![],
                effect: Effect::pure(),
            }),
            effect: None,
        },
    }
}

/// Generate all builtin functions as graph `Function` entries.
///
/// Each builtin gets a unique `QualifiedRef`. Polymorphic signatures contain
/// `Ty::Param` tokens from a throwaway `TySubst` — callers must
/// `instantiate()` before unification.
pub fn standard_builtins(interner: &Interner) -> Vec<Function> {
    vec![
        // Iterator HOFs
        make_builtin(interner, "filter", sig_filter),
        make_builtin(interner, "map", sig_map),
        make_builtin(interner, "pmap", sig_pmap),
        make_builtin(interner, "pchain", sig_pchain),
        make_builtin(interner, "find", sig_find),
        make_builtin(interner, "reduce", sig_reduce),
        make_builtin(interner, "fold", sig_fold),
        make_builtin(interner, "any", sig_any),
        make_builtin(interner, "all", sig_all),
        // flatten / flat_map
        make_builtin(interner, "flatten", sig_flatten_iter),
        make_builtin(interner, "flat_map", sig_flat_map),
        // Deque ops
        make_builtin(interner, "append", sig_append),
        make_builtin(interner, "extend", sig_extend),
        make_builtin(interner, "consume", sig_consume),
        // join / contains / first / last
        make_builtin(interner, "join", sig_join_iter),
        make_builtin(interner, "contains", sig_contains_iter),
        make_builtin(interner, "first", sig_first_iter),
        make_builtin(interner, "last", sig_last_iter),
        // Iterator/Sequence next
        make_builtin(interner, "next_seq", sig_next_seq),
        make_builtin(interner, "next", sig_next),
        // Iterator constructors
        make_builtin(interner, "iter", sig_iter),
        make_builtin(interner, "rev_iter", sig_rev_iter),
        make_builtin(interner, "collect", sig_collect),
        // take/skip/chain (Sequence + Iterator variants)
        make_builtin(interner, "take_seq", sig_take_seq),
        make_builtin(interner, "take", sig_take),
        make_builtin(interner, "skip_seq", sig_skip_seq),
        make_builtin(interner, "skip", sig_skip),
        make_builtin(interner, "chain_seq", sig_chain_seq),
        make_builtin(interner, "chain", sig_chain),
    ]
}

/// Build a name → Ty::Fn map from all builtins, for use in `TypeEnv.functions`.
pub fn builtin_fn_types(interner: &Interner) -> FxHashMap<Astr, Ty> {
    let mut map: FxHashMap<Astr, Ty> = FxHashMap::default();
    for func in standard_builtins(interner) {
        if let Constraint::Exact(ty) = func.constraint.output {
            assert!(
                !map.contains_key(&func.qref.name),
                "duplicate builtin name: {}",
                interner.resolve(func.qref.name)
            );
            map.insert(func.qref.name, ty);
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::ty::Polarity;

    /// Signature generator for a given builtin name.
    fn sig(name: &str) -> Option<SigFn> {
        match name {
            "filter" => Some(sig_filter),
            "map" => Some(sig_map),
            "pmap" => Some(sig_pmap),
            "find" => Some(sig_find),
            "reduce" => Some(sig_reduce),
            "fold" => Some(sig_fold),
            "any" => Some(sig_any),
            "all" => Some(sig_all),
            "flatten" => Some(sig_flatten_iter),
            "flat_map" => Some(sig_flat_map),
            "join" => Some(sig_join_iter),
            "contains" => Some(sig_contains_iter),
            "first" => Some(sig_first_iter),
            "last" => Some(sig_last_iter),
            "iter" => Some(sig_iter),
            "rev_iter" => Some(sig_rev_iter),
            "collect" => Some(sig_collect),
            "next_seq" => Some(sig_next_seq),
            "next" => Some(sig_next),
            "take_seq" => Some(sig_take_seq),
            "take" => Some(sig_take),
            "skip_seq" => Some(sig_skip_seq),
            "skip" => Some(sig_skip),
            "chain_seq" => Some(sig_chain_seq),
            "chain" => Some(sig_chain),
            "pchain" => Some(sig_pchain),
            "append" => Some(sig_append),
            "extend" => Some(sig_extend),
            "consume" => Some(sig_consume),
            _ => None,
        }
    }

    /// Try builtin resolution for a given name with the given arg types.
    pub fn try_builtin(
        interner: &Interner,
        s: &mut TySubst,
        name: &str,
        arg_types: &[Ty],
    ) -> Result<Ty, ()> {
        let sig_fn = sig(name).ok_or(())?;
        let (params, ret) = sig_fn(interner, s);
        if arg_types.len() != params.len() {
            return Err(());
        }
        for (a, param_ty) in arg_types.iter().zip(params.iter()) {
            if s.unify(a, param_ty, Polarity::Covariant).is_err() {
                return Err(());
            }
        }
        Ok(s.resolve(&ret))
    }
}
