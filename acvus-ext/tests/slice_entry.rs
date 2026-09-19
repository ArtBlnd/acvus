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
            "core::as_str",
            "string::substring",
            "string::trim",
            "string::trim_end",
            "string::trim_start",
            "vec::as_slice",
            "vec::as_slice_mut",
        ]
    );
}

/// RFC-0062 Decision 3 reaches a `&str` parameter through `as_str`, which
/// `slice_coercion` resolves out of the environment's machine set, so a
/// registry set that declares no `string` module still has it.
#[test]
fn the_view_of_a_string_is_declared_with_no_registry_at_all() {
    let i = Interner::new();
    let externs = Externs::<TypesOnly>::combine(vec![], &i).expect("core alone combines");
    let declared: Vec<String> = externs
        .handlers
        .iter()
        .filter(|(_, handlers)| handlers.iter().any(|h| h.width().ret == 2))
        .map(|(qref, _)| name(&i, qref))
        .collect();
    assert_eq!(declared, ["core::as_str"]);
}
