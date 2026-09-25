//! One binary: `cargo test -p acvus-ext --test all <file>::<test>`.

mod e2e;
mod erased;
mod hash_instances;
mod labelled_flows;
mod owned_in_scripts;
mod owned_stage_holders;
mod region_params;
mod regression_0041;
mod slice_entry;
mod vec_ops;

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
