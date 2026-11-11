//! Runs programs written as text, and says of each what it came to.
//!
//! `check <path>...`: each path is a file, or a directory searched for files
//! named `*.prog`. One line is printed for each program:
//!
//! | Line | The program |
//! |---|---|
//! | `HOLDS` | is solved in every order to what it asserts |
//! | `DIFFERS` | is solved, and not to what it asserts: what it came to follows |
//! | `UNREAD` | is no text of a program, or cannot be recorded: why follows |

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use acvus_typeck_model::text::record::{Failed, run};

/// The orders of asking each program is solved in.
const ORDERS: u64 = 24;

fn programs(path: &Path, found: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !path.is_dir() {
        found.push(path.to_owned());
        return Ok(());
    }
    let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() || path.extension().is_some_and(|extension| extension == "prog") {
            programs(&path, found)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut found = Vec::new();
    for argument in std::env::args().skip(1) {
        if let Err(error) = programs(Path::new(&argument), &mut found) {
            eprintln!("{argument}: {error}");
            return ExitCode::FAILURE;
        }
    }
    let (mut holds, mut differs, mut unread) = (0, 0, 0);
    for path in &found {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("{}: {error}", path.display());
                return ExitCode::FAILURE;
            }
        };
        match run(&text, ORDERS) {
            Ok(()) => {
                holds += 1;
                println!("HOLDS    {}", path.display());
            }
            Err(Failed::Differs(found)) => {
                differs += 1;
                println!("DIFFERS  {}: {found}", path.display());
            }
            Err(Failed::Unread(why)) => {
                unread += 1;
                println!("UNREAD   {}: {}", path.display(), why.replace('\n', " "));
            }
        }
    }
    println!("{holds} hold, {differs} differ, {unread} unread, of {}", found.len());
    ExitCode::SUCCESS
}
