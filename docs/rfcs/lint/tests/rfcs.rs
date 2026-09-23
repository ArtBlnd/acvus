//! `cargo test --manifest-path docs/rfcs/lint/Cargo.toml`: the tree keeps the
//! rules of `docs/rfcs/README.md`.

use std::path::Path;

#[test]
fn the_tree_keeps_the_rfc_rules() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("the crate sits at docs/rfcs/lint under the repository root");
    let violations = acvus_rfcs::check(root);
    let listed: Vec<String> = violations.iter().map(ToString::to_string).collect();
    assert!(listed.is_empty(), "{} violations:\n{}", listed.len(), listed.join("\n"));
}
