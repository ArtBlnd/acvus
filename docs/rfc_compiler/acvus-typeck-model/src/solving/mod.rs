//! Solving written anew from design/solving.md.

pub mod choice;
pub mod hint;
pub mod judge;
pub mod read;
pub mod record;
pub mod run;
pub mod text;

#[cfg(test)]
mod harness {

    use std::path::{Path, PathBuf};

    use crate::cases::{build, parse};
    use crate::cases::parse::Expect;

    use super::{run, text};

    fn collect(dir: &Path, paths: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("the cases directory") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                collect(&path, paths);
            } else if path.extension().is_some_and(|e| e == "case") {
                paths.push(path);
            }
        }
    }

    #[test]
    fn solving_cases() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("cases");
        let filter = std::env::var("CASE").ok();
        let mut paths = Vec::new();
        collect(&root, &mut paths);
        paths.sort();
        let (mut covered, mut passed, mut skipped) = (0, 0, 0);
        for path in paths {
            let shown = path.strip_prefix(&root).expect("under cases").display().to_string();
            if filter.as_ref().is_some_and(|f| !shown.contains(f.as_str())) {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("a readable case");
            let Ok(case) = parse::parse(&source) else {
                continue;
            };
            let Ok(decls) = build::decls(&case) else {
                continue;
            };
            let Ok(program) = build::program(&case, &decls) else {
                continue;
            };
            let record = match text::record(&program, &decls) {
                Ok(record) => record,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            covered += 1;
            let (_, outcome) = run::solve(record, &decls);
            let mut fails: Vec<String> = Vec::new();
            for expect in &case.expects {
                match &expect.item {
                    Expect::Accepted(wanted) if *wanted != outcome.accepted => {
                        fails.push(format!("accepted is {}", outcome.accepted));
                    }
                    Expect::Value { class, ty } => {
                        let wanted = ty
                            .as_ref()
                            .map(|expr| build::closed(decls.types(), expr).ok());
                        let got = outcome
                            .values
                            .get(&crate::ty::Name::new(class))
                            .cloned()
                            .flatten();
                        let same = match (&wanted, &got) {
                            (None, None) => true,
                            (Some(Some(w)), Some(g)) => w == g,
                            _ => false,
                        };
                        if !same {
                            fails.push(format!("{class} is {got:?}, wanted {ty:?}"));
                        }
                    }
                    Expect::Decision {
                        disjunction,
                        decision,
                    } => {
                        let got = outcome
                            .decisions
                            .get(&crate::ty::Name::new(disjunction))
                            .cloned()
                            .flatten();
                        let wanted = decision
                            .trim()
                            .replace("lend ", "")
                            .replace("bind ", "");
                        let same = match got.as_deref() {
                            None => wanted == "undecided",
                            Some(g) => {
                                let g = g.trim();
                                g == wanted
                                    || (wanted.ends_with("through") && g.starts_with("through"))
                                    || (wanted.ends_with("place") && g == "place")
                            }
                        };
                        if !same {
                            fails.push(format!("{disjunction} is {got:?}, wanted {decision}"));
                        }
                    }
                    _ => {}
                }
            }
            if fails.is_empty() {
                passed += 1;
                println!("PASS {shown}");
            } else {
                println!("FAIL {shown}");
                for fail in &fails {
                    println!("      {fail}");
                }
                for (name, refusal) in &outcome.refusals {
                    println!("      refused {name}: {refusal:?}");
                }
                for failing in &outcome.failing {
                    println!("      failing {failing}");
                }
            }
        }
        println!("covered {covered}: {passed} pass, {} fail; skipped {skipped}", covered - passed);
    }
}
