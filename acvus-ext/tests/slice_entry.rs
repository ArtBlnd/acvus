//! Which of the standard declarations hand a run back in two registers
//! (RFC-0047 amended, rule 2; RFC-0062 Decision 1 for the run of bytes).

use acvus_extern::{Externs, Interner, QualifiedRef, TypesOnly};

fn name(i: &Interner, qref: &QualifiedRef) -> String {
    match qref.namespace {
        Some(ns) => format!("{}::{}", i.resolve(ns), i.resolve(qref.name)),
        None => i.resolve(qref.name).to_string(),
    }
}

#[test]
fn the_slice_entry_is_the_slice_returning_declarations_and_nothing_else() {
    let i = Interner::new();
    let externs =
        Externs::combine(acvus_ext::std_registries::<TypesOnly>(), &i).expect("registries combine");
    let mut in_registers: Vec<String> = externs
        .handlers
        .iter()
        .filter(|(_, handlers)| handlers.iter().any(|h| h.width().ret == 2))
        .map(|(qref, _)| name(&i, qref))
        .collect();
    in_registers.sort();
    assert_eq!(
        in_registers,
        [
            "array::as_slice",
            "array::as_slice_mut",
            "string::as_str",
            "vec::as_slice",
            "vec::as_slice_mut",
        ]
    );
}
