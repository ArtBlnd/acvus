//! Which of the standard declarations cross a run unboxed (RFC-0047 §6).

use acvus_extern::{ExternHandler, Externs, Interner, QualifiedRef, SyncAbi, SyncCall, TypesOnly};

fn name(i: &Interner, qref: &QualifiedRef) -> String {
    match qref.namespace {
        Some(ns) => format!("{}::{}", i.resolve(ns), i.resolve(qref.name)),
        None => i.resolve(qref.name).to_string(),
    }
}

#[test]
fn the_unboxed_entry_is_the_slice_returning_declarations_and_nothing_else() {
    let i = Interner::new();
    let externs =
        Externs::combine(acvus_ext::std_registries::<TypesOnly>(), &i).expect("registries combine");
    let mut unboxed: Vec<String> = externs
        .handlers
        .iter()
        .filter(|(_, handlers)| {
            handlers
                .iter()
                .any(|h| matches!(h, ExternHandler::Sync(SyncCall::Plain(SyncAbi::Slice(_)))))
        })
        .map(|(qref, _)| name(&i, qref))
        .collect();
    unboxed.sort();
    assert_eq!(
        unboxed,
        [
            "array::as_slice",
            "array::as_slice_mut",
            "vec::as_slice",
            "vec::as_slice_mut",
        ]
    );
}
