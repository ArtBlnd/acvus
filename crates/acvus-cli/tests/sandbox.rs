//! The binary under test reads its ctl config from `XDG_CONFIG_HOME`, and
//! every test hands it a directory under the target directory with `HOME`
//! unset, so no test reads or writes the config of the person running it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn target_dir() -> PathBuf {
    Path::new(env!("CARGO_BIN_EXE_acvus"))
        .parent()
        .and_then(Path::parent)
        .expect("cargo puts the binary at <target>/<profile>/acvus")
        .to_path_buf()
}

pub fn scratch() -> PathBuf {
    let dir = target_dir().join("acvus-cli-tests");
    std::fs::create_dir_all(&dir).expect("make the tests' scratch directory");
    dir
}

pub fn tempdir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .tempdir_in(scratch())
        .expect("a temp dir under the target directory")
}

pub fn acvus() -> Command {
    let config = scratch().join("no-config");
    std::fs::create_dir_all(&config).expect("make the empty config directory");
    with_config(&config)
}

pub fn with_config(config: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_acvus"));
    command.env("XDG_CONFIG_HOME", config).env_remove("HOME");
    command
}
