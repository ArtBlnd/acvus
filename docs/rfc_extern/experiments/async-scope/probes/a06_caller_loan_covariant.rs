// expect: E0597
// The control: the caller's own value, lent through a covariant loan outside the
// scope. The closure must hold for every loan of the scope, so a capture must
// outlive them all, and a caller's loan is refused.
use model::*;
pub fn run(function: Uniform<Program>) {
    let mut own = Uniform::new(10i64);
    let loan = CovLentMut::of(&mut own);
    let text = Shared(Uniform::new(String::from("abc")));
    let _ = block_on(function.scope(text, async move |calling, text| calling.call_covariant(loan, text).await));
}
