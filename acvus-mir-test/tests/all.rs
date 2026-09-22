//! One binary: `cargo test -p acvus-mir-test --test all <file>::<test>`.

mod array;
mod body_summary;
mod bound;
mod bound_input;
mod captures;
mod cast;
mod code_motion;
mod consumer_task;
mod context_move;
mod context_variable;
mod cyclic;
mod dead_store;
mod declared_struct;
mod diagnostic_labels;
mod diamond;
mod e2e;
mod effect;
mod error_types;
mod exclusion;
mod fold;
mod for_loop;
mod hash_types;
mod identity;
mod identity_sources;
mod index;
mod inline;
mod integers;
mod lambda_summary;
mod lend;
mod lent_iterator_liveness;
mod literal;
mod loans;
mod loop_invariant_phi;
mod lsr;
mod machine_coercion;
mod match_expr;
mod method;
mod moved_lend;
mod object_width;
mod operators;
mod optimized;
mod optimized_cross_fn;
mod option_drop;
mod order_tree;
mod overload;
mod overload_signature;
mod patterns_through;
mod projection_parameter;
mod reborrow;
mod regression_0041;
mod return_statement;
mod return_type;
mod script;
mod shadow;
mod short_circuit;
mod signature;
mod slice_param;
mod solver_lifecycle;
mod sroa;
mod str_param;
mod structural_union;
mod task;
mod temporary_scrutinee;
mod through;
mod try_op;
mod user_defined_ty;

/// The targets this crate's manifest declares under `[[test]]`, as the paths
/// of the files that carry them: `path` when it is written, and otherwise the
/// `tests/<name>.rs` Cargo derives from the target's name.
fn own_test_paths(manifest: &str) -> Vec<String> {
    format!("\n{manifest}")
        .split("\n[")
        .filter(|table| table.starts_with("[test]]"))
        .map(|table| {
            let field = |key: &str| {
                table.lines().find_map(|line| {
                    Some(line.trim().strip_prefix(key)?.split('"').nth(1)?.to_owned())
                })
            };
            field("path =").unwrap_or_else(|| {
                format!(
                    "tests/{}.rs",
                    field("name =").expect("a [[test]] has a name")
                )
            })
        })
        .collect()
}

/// `autotests = false` is what buys this crate one test binary, and it is also
/// what makes a file under `tests/` that nothing names invisible: it compiles
/// nowhere, so it passes everywhere. Every such file is a `mod` above, and
/// this is what says so out loud.
#[test]
fn every_test_file_is_linked() {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest =
        std::fs::read_to_string(crate_dir.join("Cargo.toml")).expect("the crate's manifest");
    let listed =
        std::fs::read_to_string(crate_dir.join("tests/all.rs")).expect("tests/all.rs is readable");
    let own = own_test_paths(&manifest);
    let is_mod = |stem: &str| {
        listed
            .lines()
            .any(|line| line.trim_end() == format!("mod {stem};"))
    };

    let mut unlisted: Vec<String> = std::fs::read_dir(crate_dir.join("tests"))
        .expect("the tests directory")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .map(|path| {
            let stem = path.file_stem().expect("a file name").to_string_lossy();
            (format!("tests/{stem}.rs"), stem.into_owned())
        })
        .filter(|(path, stem)| !own.contains(path) && !is_mod(stem))
        .map(|(path, _)| path)
        .collect();
    unlisted.sort();

    assert!(
        unlisted.is_empty(),
        "these files reach no build - add a `mod` for each to tests/all.rs: {unlisted:?}"
    );
}
