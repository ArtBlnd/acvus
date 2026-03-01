use std::path::PathBuf;

use libtest_mimic::{Arguments, Trial};

fn main() {
    let args = Arguments::from_args();
    let tests = discover_fixtures();
    libtest_mimic::run(&args, tests).exit();
}

fn discover_fixtures() -> Vec<Trial> {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let pattern = format!("{}/**/*.json", fixture_dir.display());

    glob::glob(&pattern)
        .unwrap()
        .filter_map(Result::ok)
        .map(|path| {
            let name = path
                .strip_prefix(&fixture_dir)
                .unwrap()
                .with_extension("")
                .display()
                .to_string()
                .replace(std::path::MAIN_SEPARATOR, "::");

            Trial::test(name, move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(acvus_interpreter_test::run_fixture(&path))
                    .map_err(|e| e.into())
            })
        })
        .collect()
}
