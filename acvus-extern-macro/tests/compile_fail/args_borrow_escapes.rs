//! What `with` lends a closure is lent for the closure's call alone
//! (RFC-0097 rule 1, RFC-0079 rule 6).
use acvus_extern::{Args, Owned, Runtime};

fn kept<'c, R>(args: &'c Args<'_, (Owned<R>,), R>) -> &'c String
where
    R: Runtime,
{
    let mut kept = None;
    args.with(0, |s: &String| kept = Some(s));
    kept.expect("argument 0 is a `String`")
}

fn main() {}
