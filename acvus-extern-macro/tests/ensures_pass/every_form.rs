//! Every form RFC-0082 rule 4 holds: `=`, `<=` and `<`; a constant, a
//! parameter, `ret`, `len` of a slice, a container or `ret`, `+`, `-`,
//! `*` and `max` of terms, and `old` of a term over `&mut` parameters.
use acvus_extern::{Runtime, TransparentOver, Var, extern_fn, kind};

#[extern_fn(effect = pure, ensures(ret = len(s)))]
fn count<T, Rt>(s: &[T]) -> u64
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s.len() as u64
}

#[extern_fn(effect = pure, ensures(ret = len(c)))]
fn held<T>(c: &Vec<T>) -> u64
where
    T: Var<kind::Type>,
{
    c.len() as u64
}

#[extern_fn(effect = pure, ensures(ret <= a, ret <= b, ret < max(a, b) + 1, 0 <= ret * 2 - ret))]
fn lesser(a: u64, b: u64) -> u64 {
    a.min(b)
}

#[extern_fn(effect = pure, ensures(len(ret) = len(s)))]
fn same<T, Rt>(s: &[T]) -> &[T]
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    s
}

#[extern_fn(effect = pure, ensures(len(c) = old(len(c)) + 1, len(c) <= old(len(c) * 2) + 1))]
fn grown<T>(c: &mut Vec<T>, item: T)
where
    T: Var<kind::Type>,
{
    c.push(item);
}

fn main() {}
