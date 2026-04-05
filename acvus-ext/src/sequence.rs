//! Sequence operations as ExternFn.

use acvus_interpreter::{Args, ExternFnBuilder, ExternRegistry, RuntimeError, Value, exec_next};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::ty::{CastRule, Effect, Ownership, Param, Ty, TySubst, TypeRegistry, UserDefinedDecl};
use acvus_utils::Interner;
use futures::future::BoxFuture;

// ── Handlers ────────────────────────────────────────────────────────

fn h_take_seq(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let seq = args[0].take().into_sequence();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::sequence(seq.take(n)))
}

fn h_skip_seq(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let seq = args[0].take().into_sequence();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::sequence(seq.skip(n)))
}

fn h_chain_seq(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let seq = args[0].take().into_sequence();
    let iter = args[1].take().into_iterator();
    Ok(Value::sequence(seq.chain(*iter)))
}

fn h_deque_to_seq(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    use acvus_interpreter::iter::SequenceChain;
    use acvus_mir::ty::Effect;
    use std::sync::Arc;
    let d = match args[0].take() {
        Value::Deque(d) => Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone()),
        other => panic!("deque_to_seq: expected Deque, got {other:?}"),
    };
    Ok(Value::sequence(SequenceChain::from_stored(
        d,
        Effect::pure(),
    )))
}

fn h_seq_to_iter(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let seq = args[0].take().into_sequence();
    Ok(Value::iterator(seq.into_iter_handle()))
}

fn h_next_seq(
    mut args: Args,
    interner: Interner,
) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let seq = *args[0].take().into_sequence();
        let mut iter = seq.into_iter_handle();
        match exec_next(&mut iter).await? {
            Some(val) => {
                let pair = Value::tuple(vec![val, Value::iterator(iter)]);
                Ok(Value::variant(interner.intern("Some"), Some(pair)))
            }
            None => Ok(Value::variant(interner.intern("None"), None)),
        }
    })
}

// ── Signature helpers ───────────────────────────────────────────────

fn p(interner: &Interner, idx: usize, ty: Ty) -> Param {
    Param::new(interner.intern(&format!("_{idx}")), ty)
}

fn make_sig(params: &[Ty], ret: Ty, interner: &Interner) -> FnConstraint {
    let named: Vec<Param> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| p(interner, i, ty.clone()))
        .collect();
    FnConstraint {
        signature: Some(Signature {
            params: named.clone(),
        }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
        hint: None,
    }
}

// ── Registry ────────────────────────────────────────────────────────

pub fn sequence_registry(interner: &Interner, type_registry: &mut TypeRegistry) -> ExternRegistry {
    let seq_qref = QualifiedRef::root(interner.intern("Sequence"));
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    type_registry.register(UserDefinedDecl {
        qref: seq_qref,
        type_params: vec![None, None], // T, O
        effect_params: vec![None],     // E
        ownership: Ownership::MoveOnly,
    });

    // CastRule: Deque<T, O> → Sequence<T, O, Pure>
    {
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        type_registry.register_cast(CastRule {
            from: Ty::Deque(Box::new(t.clone()), Box::new(o.clone())),
            to: Ty::UserDefined {
                id: seq_qref,
                type_args: vec![t, o],
                effect_args: vec![Effect::pure()],
                ownership: Ownership::MoveOnly,
            },
            fn_ref: QualifiedRef::root(interner.intern("__cast_deque_to_seq")),
        });
    }
    // CastRule: Sequence<T, O, E> → Iterator<T, E>
    {
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        let e = s.fresh_effect_var();
        type_registry.register_cast(CastRule {
            from: Ty::UserDefined {
                id: seq_qref,
                type_args: vec![t.clone(), o],
                effect_args: vec![e.clone()],
                ownership: Ownership::MoveOnly,
            },
            to: Ty::UserDefined {
                id: iter_qref,
                type_args: vec![t],
                effect_args: vec![e],
                ownership: Ownership::MoveOnly,
            },
            fn_ref: QualifiedRef::root(interner.intern("__cast_seq_to_iter")),
        });
    }

    ExternRegistry::new(move |interner| {
        // Helper: Sequence<T, O, E>
        let sq = |t: Ty, o: Ty, e: Effect| -> Ty {
            Ty::UserDefined {
                id: seq_qref,
                type_args: vec![t, o],
                effect_args: vec![e],
                ownership: Ownership::MoveOnly,
            }
        };
        // Helper: Iterator<T, E>
        let it = |t: Ty, e: Effect| -> Ty {
            Ty::UserDefined {
                id: iter_qref,
                type_args: vec![t],
                effect_args: vec![e],
                ownership: Ownership::MoveOnly,
            }
        };

        // take_seq: (Sequence<T, O, E>, Int) → Sequence<T, O, E>
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        let e = s.fresh_effect_var();
        let seq = sq(t.clone(), o.clone(), e.clone());
        let take_seq = ExternFnBuilder::new(
            "take_seq",
            make_sig(&[seq.clone(), Ty::Int], seq.clone(), interner),
        )
        .sync_handler(h_take_seq);

        // skip_seq: same sig
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        let e = s.fresh_effect_var();
        let seq = sq(t.clone(), o.clone(), e.clone());
        let skip_seq = ExternFnBuilder::new(
            "skip_seq",
            make_sig(&[seq.clone(), Ty::Int], seq.clone(), interner),
        )
        .sync_handler(h_skip_seq);

        // chain_seq: (Sequence<T, O, E>, Iterator<T, E>) → Sequence<T, O, E>
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        let e = s.fresh_effect_var();
        let seq = sq(t.clone(), o.clone(), e.clone());
        let chain_seq = ExternFnBuilder::new(
            "chain_seq",
            make_sig(
                &[seq.clone(), it(t.clone(), e.clone())],
                seq.clone(),
                interner,
            ),
        )
        .sync_handler(h_chain_seq);

        // next_seq: (Sequence<T, O, E>) → Option<(T, Sequence<T, O, E>)>
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        let e = s.fresh_effect_var();
        let seq = sq(t.clone(), o.clone(), e.clone());
        let next_seq = ExternFnBuilder::new(
            "next_seq",
            make_sig(
                &[seq.clone()],
                Ty::Option(Box::new(Ty::Tuple(vec![t, seq]))),
                interner,
            ),
        )
        .async_handler(h_next_seq);

        // Cast helpers.
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        let deque_to_seq = ExternFnBuilder::new(
            "__cast_deque_to_seq",
            make_sig(
                &[Ty::Deque(Box::new(t.clone()), Box::new(o.clone()))],
                sq(t, o, Effect::pure()),
                interner,
            ),
        )
        .sync_handler(h_deque_to_seq);

        let mut s = TySubst::new();
        let t = s.fresh_param();
        let o = s.fresh_param();
        let e = s.fresh_effect_var();
        let seq_to_iter = ExternFnBuilder::new(
            "__cast_seq_to_iter",
            make_sig(&[sq(t.clone(), o, e.clone())], it(t, e), interner),
        )
        .sync_handler(h_seq_to_iter);

        vec![
            take_seq,
            skip_seq,
            chain_seq,
            next_seq,
            deque_to_seq,
            seq_to_iter,
        ]
    })
}
