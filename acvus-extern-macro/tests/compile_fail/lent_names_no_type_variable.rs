//! `unsafe(lent(..))` names the declaration's own type variables; an effect
//! variable or a name the declaration does not have is refused.
use acvus_extern::{Var, extern_fn, kind};

#[extern_fn(effect = pure, unsafe(lent(U)))]
fn pass<T>(x: T) -> T
where
    T: Var<kind::Type>,
{
    x
}

fn main() {}
