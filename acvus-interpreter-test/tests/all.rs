//! One binary: `cargo test -p acvus-interpreter-test --test all <file>::<test>`.

mod arith_chain;
mod attention_loop_shape;
mod attention_shape;
mod attention_shape_overflow;
mod block_splitting;
mod body_result_reference;
mod body_returns_a_view;
mod captured_word;
mod cast;
mod classify;
mod closure_nesting;
mod container;
mod container_contains;
mod container_read;
mod conversion;
mod deque;
mod diamond;
mod differential;
mod e2e;
mod element_borrow_key;
mod enum_projection;
mod erased;
mod extern_aggregate_result;
mod extern_call_forms;
mod extern_fn;
mod fold_agreement;
mod for_loop;
mod fused_run;
mod hash_instances;
mod inlined_closure_capture;
mod instance_entry;
mod instance_rest;
mod integers;
mod iter_more;
mod iter_next;
mod laid_argument_drop;
mod lambda_capture;
mod lent_iterator;
mod literal;
mod loop_carried_value;
mod loop_escape;
mod loop_exit_moves;
mod main_return_declaration;
mod mandelbrot_loop_shape;
mod match_dispatch;
mod match_literal_dispatch;
mod match_source;
mod mono;
mod mono_result;
mod move_then_lend;
mod moved_lend;
mod never;
mod num;
mod num_std;
mod object_field_drop;
mod operators;
mod option_form;
mod option_methods;
mod option_payload_drop;
mod option_string_drop;
mod pattern_through;
mod prepare_contract;
mod recursive_summary;
mod regex;
mod register_reuse_across_kinds;
mod regression_0041;
mod result;
mod result_form;
mod result_methods;
mod result_run;
mod resume_in_a_loop;
mod return_statement;
mod reused_frame_drop;
mod row_slice_hoist;
mod run_shape;
mod script;
mod script_programs;
mod select;
mod shadow;
mod short_circuit;
mod signature_effect;
mod slice_hoist_by_write_kind;
mod slice_index;
mod slice_pair;
mod slice_param;
mod space;
mod spawn_eval;
mod strength_reduction;
mod string_std;
mod sync_call_is_an_operation;
mod task_instances;
mod temporary_borrow;
mod try_op;
mod typed_stage_list;
mod unread_store_drop;

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
