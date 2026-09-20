//! Which of the standard declarations hand a run back in two registers
//! (RFC-0047 amended, rule 2; RFC-0062 Decision 1 for the run of bytes).

use acvus_extern::{Externs, FormKind, Interner, QualifiedRef, TypesOnly};

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
        .filter(|(_, handlers)| handlers.iter().any(|h| h.width().result == FormKind::View))
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
            "string::trim_end_matches",
            "string::trim_matches",
            "string::trim_start",
            "string::trim_start_matches",
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
        .filter(|(_, handlers)| handlers.iter().any(|h| h.width().result == FormKind::View))
        .map(|(qref, _)| name(&i, qref))
        .collect();
    assert_eq!(declared, ["core::as_str"]);
}

/// Which of this crate's declarations write an aggregate's components where
/// the caller placed the result (RFC-0050 rules 5 and 6). A declaration whose
/// Rust result is `Option<S>` is not one of them: an option is one of the
/// runtime's values, so `regex::find` and its family stay off this list.
///
/// The set is every registry this crate builds, not `std_registries`, which
/// carries neither `regex` nor `datetime` nor `encoding` nor `io`.
#[test]
fn the_component_entry_is_the_struct_returning_declarations_and_nothing_else() {
    let i = Interner::new();
    let mut registries = acvus_ext::std_registries::<TypesOnly>();
    registries.push(acvus_ext::regex_registry());
    registries.push(acvus_ext::datetime_registry());
    registries.push(acvus_ext::encoding_registry());
    registries.push(acvus_ext::io_registry());
    let externs = Externs::combine(registries, &i).expect("registries combine");
    let mut in_components: Vec<String> = externs
        .handlers
        .iter()
        .filter(|(_, handlers)| {
            handlers
                .iter()
                .any(|h| h.width().result == FormKind::Components)
        })
        .map(|(qref, _)| name(&i, qref))
        .collect();
    in_components.sort();
    assert_eq!(in_components, ["std::regex_flags"]);
}
