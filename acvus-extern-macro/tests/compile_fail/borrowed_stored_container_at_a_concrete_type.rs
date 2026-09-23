//! A map, a set and a deque are each kept as one box of the Rust type at
//! their variables' run-time instantiation: every type variable is
//! `Owned<Rt>` and every effect variable `()` when the handler runs, so the
//! box a script holds is `HashMap<Owned<Rt>, Owned<Rt>, (), Rt>`. A borrow
//! at a concrete key, value, element or known effect read a box of another
//! type, and was accepted and failed at its first call; `Borrowable` for
//! each now asks `InPlaceElement` of its type variables and `InPlaceEffect`
//! of its effect, and the declaration is refused.
use acvus_ext::{Deque, HashMap, HashSet};
use acvus_extern::{Pure, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
fn map_at_ints<E, Rt>(_m: &HashMap<i64, i64, E, Rt>) -> u64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    0
}

#[extern_fn(effect = pure)]
fn map_at_pure<K, V, Rt>(_m: &HashMap<K, V, Pure, Rt>) -> u64
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    Rt: Runtime,
{
    0
}

#[extern_fn(effect = pure)]
fn set_at_strings<E, Rt>(_s: &mut HashSet<String, E, Rt>)
where
    E: Var<kind::Effect>,
    Rt: Runtime,
{
}

#[extern_fn(effect = pure)]
fn deque_at_ints(d: &Deque<i64>) -> u64 {
    d.len() as u64
}

fn main() {}
