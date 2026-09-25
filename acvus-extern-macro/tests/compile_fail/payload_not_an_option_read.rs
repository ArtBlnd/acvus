//! `payload` is stated over `f(o: Option<T>) -> T` (RFC-0082 rule 3): a
//! function returning something other than the option's payload is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, payload(o))]
fn or_zero(o: Option<i64>) -> u64 {
    o.map_or(0, |x| x as u64)
}

fn main() {}
