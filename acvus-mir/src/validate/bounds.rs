//! An `Index` or `IndexSet` marked `Proven` runs without its bound check,
//! so the mark is refused unless `analysis::interval` derives the bound
//! again from the module as it reaches the machine (RFC-0047 rule 7).

use crate::analysis::interval::{InstAt, bounded_indices};
use crate::cfg::{self, BlockIdx};
use crate::ir::{IndexBound, InstKind, MirBody, MirModule};
use crate::laws::LawTable;
use crate::validate::type_check::{ValidationError, ValidationErrorKind};

pub fn check_bounds(module: &MirModule, laws: &LawTable) -> Vec<ValidationError> {
    let mut errors = check_body("main", &module.main, laws);
    for (label, closure) in &module.closures {
        errors.extend(check_body(&format!("closure({label:?})"), closure, laws));
    }
    errors
}

fn check_body(scope: &str, body: &MirBody, laws: &LawTable) -> Vec<ValidationError> {
    let marked = |kind: &InstKind| {
        matches!(
            kind,
            InstKind::Index {
                bound: IndexBound::Proven,
                ..
            } | InstKind::IndexSet {
                bound: IndexBound::Proven,
                ..
            }
        )
    };
    if !body.insts.iter().any(|inst| marked(&inst.kind)) {
        return Vec::new();
    }
    let cfg = cfg::promote(body.clone());
    let bounded = bounded_indices(&cfg, laws);
    let mut underived = Vec::new();
    for (b, block) in cfg.blocks.iter().enumerate() {
        for (at, inst) in block.insts.iter().enumerate() {
            let derived = bounded.contains(&InstAt {
                block: BlockIdx(b),
                at,
            });
            if marked(&inst.kind) && !derived {
                underived.push(&inst.kind);
            }
        }
    }
    // A marked instruction `promote` pruned stands in a block no path
    // reaches, and never runs.
    body.insts
        .iter()
        .enumerate()
        .filter(|(_, flat)| underived.iter().any(|kind| same_access(&flat.kind, kind)))
        .map(|(inst_index, flat)| ValidationError {
            scope: scope.to_string(),
            inst_index,
            span: flat.span,
            kind: ValidationErrorKind::UnprovenBound,
        })
        .collect()
}

/// The flat body holds the instruction a block holds: an `Index` is its
/// `dst`'s one definition, and an `IndexSet` is found by what it writes.
fn same_access(flat: &InstKind, placed: &InstKind) -> bool {
    match (flat, placed) {
        (InstKind::Index { dst: a, .. }, InstKind::Index { dst: b, .. }) => a == b,
        (
            InstKind::IndexSet {
                slice: s1,
                index: i1,
                value: v1,
                ..
            },
            InstKind::IndexSet {
                slice: s2,
                index: i2,
                value: v2,
                ..
            },
        ) => s1 == s2 && i1 == i2 && v1 == v2,
        _ => false,
    }
}
