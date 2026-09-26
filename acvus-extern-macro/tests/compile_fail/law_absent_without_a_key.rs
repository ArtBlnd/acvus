//! `absent` is stated over a call whose `reaches` names the entry `x[k]`
//! the reference it returns lends.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(absent = value))]
fn first(xs: &mut Vec<i64>, value: i64) -> &mut i64 {
    xs.push(value);
    &mut xs[0]
}

fn main() {}
