//! The rules `docs/rfcs/README.md` states, checked over the tree.
//!
//! A decision is a `## RFC-NNNN: <ruling>` section of a topic document under
//! `docs/rfcs/`. Its first line is its status, its rules are the numbered
//! items at the start of a line, and a citation anywhere in the tree names a
//! live decision and, where it names a rule, a rule that section has.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use regex::Regex;

/// A section longer than this is two decisions, or one written at length:
/// the p90 of the sections the 2026-09 consolidation left was 1159 words.
pub const SECTION_WORDS: usize = 1500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Accepted,
    Proposed,
}

#[derive(Debug)]
pub struct Section {
    pub id: u16,
    pub title: String,
    pub status: Status,
    pub file: String,
    rules: BTreeSet<u32>,
}

#[derive(Debug, Default)]
pub struct Corpus {
    pub sections: BTreeMap<u16, Section>,
    pub retired: BTreeSet<u16>,
}

/// A line of a file under the root.
#[derive(Debug, Clone)]
pub struct Location {
    pub file: String,
    pub line: usize,
}

#[derive(Debug)]
pub struct Violation {
    pub at: Location,
    pub what: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.at.file, self.at.line, self.what)
    }
}

/// Text a decision never carries, and what it is.
struct Forbidden {
    pattern: Regex,
    what: &'static str,
}

struct Patterns {
    heading: Regex,
    /// A run of hex digits as long as git's shortest abbreviation; it is a
    /// commit hash when it holds both a letter and a digit, which a word or a
    /// number alone does not.
    hex_run: Regex,
    rule: Regex,
    forbidden: Vec<Forbidden>,
    citation: Regex,
    cited_rules: Regex,
    rule_numbers: Regex,
    other_label: Regex,
}

impl Patterns {
    fn new() -> Patterns {
        let re = |pattern: &str| Regex::new(pattern).expect("the pattern is valid");
        let forbidden = |pattern: &str, what| Forbidden { pattern: re(pattern), what };
        Patterns {
            heading: re(r"^## RFC-(\d{4}): (\S.*)$"),
            hex_run: re(r"\b[0-9a-f]{7,40}\b"),
            rule: re(r"^(\d+)\. "),
            forbidden: vec![
                forbidden(r"\b\d{4}-\d{2}-\d{2}\b", "a date"),
                forbidden(r"\.(?:rs|md|toml|lalrpop):\d", "a file:line reference"),
                forbidden(r"^\s*\|", "a table"),
                forbidden(
                    r"\b(?:Extends|Supersedes|Superseded|Amends):|\b[Aa]mended\b|\b[Ss]uperseded\b",
                    "a revision note",
                ),
            ],
            citation: re(r"RFC-?(\d{4})"),
            cited_rules: re(r"^(?:'s)? rules? (\d+(?:(?:, | and |, and )\d+)*)"),
            rule_numbers: re(r"\d+"),
            other_label: re(
                r#"^(?:'s)?,? ?\(?(?:§|D\d|Decision \d|[Ss]tage \d|[Ss]tep \d|[Rr]ule [A-Za-z]|T\d|R\d|"|Rationale|Consequences|Problem|What it costs|Order of work|Not built)"#,
            ),
        }
    }
}

/// Every violation of the rules over the tree at `root`.
pub fn check(root: &Path) -> Vec<Violation> {
    let patterns = Patterns::new();
    let mut out = Vec::new();
    let rfcs = root.join("docs/rfcs");
    let readme_path = rfcs.join("README.md");
    let readme_at = |line| Location { file: "docs/rfcs/README.md".to_string(), line };
    let readme_text = std::fs::read_to_string(&readme_path).expect("docs/rfcs/README.md is text");
    let readme = Readme(&readme_text);

    let mut corpus = Corpus::default();
    match readme.section("## Retired") {
        Some(body) => retired(body, &mut corpus.retired, &mut out),
        None => out.push(Violation { at: readme_at(0), what: "no `## Retired` section".to_string() }),
    }

    let mut topics: Vec<PathBuf> = std::fs::read_dir(&rfcs)
        .expect("docs/rfcs is a directory")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md") && *path != readme_path)
        .collect();
    topics.sort();
    for path in &topics {
        let text = std::fs::read_to_string(path).expect("a topic document is text");
        let file = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("a topic document's name is UTF-8")
            .to_string();
        topic(&patterns, Document { file: &file, text: &text }, &mut corpus, &mut out);
    }

    let expected = index(&corpus);
    match readme.section("## Index") {
        Some(body) if body.trim() == expected.trim() => {}
        Some(_) => out.push(Violation {
            at: readme_at(0),
            what: format!("the `## Index` section is not the one the headings give; it is:\n{expected}"),
        }),
        None => out.push(Violation { at: readme_at(0), what: "no `## Index` section".to_string() }),
    }

    let mut files = Vec::new();
    walk(root, &mut files);
    for path in files.iter().filter(|path| **path != readme_path) {
        let file = path
            .strip_prefix(root)
            .expect("the walk stays under the root")
            .display()
            .to_string();
        match std::fs::read_to_string(path) {
            Ok(text) => citations(&patterns, Document { file: &file, text: &text }, &corpus, &mut out),
            Err(error) => out.push(Violation { at: Location { file, line: 0 }, what: format!("unreadable: {error}") }),
        }
    }
    out
}

/// The IDs `## Retired` lists, one `- RFC-NNNN` per line.
fn retired(body: &str, into: &mut BTreeSet<u16>, out: &mut Vec<Violation>) {
    for line in body.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let id = line
            .strip_prefix("- RFC-")
            .and_then(|rest| rest.get(..4))
            .and_then(|digits| digits.parse().ok());
        match id {
            Some(id) => {
                into.insert(id);
            }
            None => out.push(Violation {
                at: Location { file: "docs/rfcs/README.md".to_string(), line: 0 },
                what: format!("a `## Retired` line is `- RFC-NNNN`: {line}"),
            }),
        }
    }
}

/// `docs/rfcs/README.md`, read for its `## ` sections.
struct Readme<'a>(&'a str);

impl<'a> Readme<'a> {
    /// The text under the `## ` heading, up to the next one or the end.
    fn section(&self, heading: &str) -> Option<&'a str> {
        let start = self.0.find(&format!("\n{heading}\n"))? + heading.len() + 2;
        let rest = &self.0[start..];
        Some(match rest.find("\n## ") {
            Some(end) => &rest[..end],
            None => rest,
        })
    }
}

/// A file under the root, by its path from the root, and its text.
struct Document<'a> {
    file: &'a str,
    text: &'a str,
}

/// A `## ` heading read, whose status line has not come yet.
struct Heading {
    id: u16,
    title: String,
}

/// The section a topic document's walk is inside.
struct Open {
    id: u16,
    words: usize,
    last_rule: u32,
}

impl Open {
    fn close(self, at: Location, out: &mut Vec<Violation>) {
        if self.words > SECTION_WORDS {
            out.push(Violation {
                at,
                what: format!("RFC-{:04} is {} words, over {SECTION_WORDS}", self.id, self.words),
            });
        }
    }
}

/// Where a topic document's walk stands between two lines.
enum State {
    Intro,
    AwaitingStatus(Heading),
    /// Inside a section the corpus holds.
    In(Open),
    /// Inside a section whose status line was refused: its words count, and
    /// it holds no rules a citation could reach.
    Unrecorded(Open),
}

fn topic(patterns: &Patterns, document: Document<'_>, corpus: &mut Corpus, out: &mut Vec<Violation>) {
    let Document { file, text } = document;
    let at = |line| Location { file: format!("docs/rfcs/{file}"), line };
    if !text.starts_with("# ") {
        out.push(Violation { at: at(1), what: "a topic document opens with its `# ` title".to_string() });
    }
    let mut state = State::Intro;
    let mut fenced = false;
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        if let State::AwaitingStatus(heading) = state {
            if line.trim().is_empty() {
                state = State::AwaitingStatus(heading);
                continue;
            }
            let open = Open { id: heading.id, words: 0, last_rule: 0 };
            let status = match line {
                "Status: Accepted" => Status::Accepted,
                "Status: Proposed" => Status::Proposed,
                _ => {
                    out.push(Violation {
                        at: at(number),
                        what: format!("RFC-{:04}'s first line is `Status: Accepted` or `Status: Proposed`", heading.id),
                    });
                    state = State::Unrecorded(open);
                    continue;
                }
            };
            corpus.sections.insert(
                heading.id,
                Section { id: heading.id, title: heading.title, status, file: file.to_string(), rules: BTreeSet::new() },
            );
            state = State::In(open);
            continue;
        }
        if line.starts_with("```") {
            fenced = !fenced;
        }
        if !fenced && line.starts_with("## ") {
            if let State::In(open) | State::Unrecorded(open) = std::mem::replace(&mut state, State::Intro) {
                open.close(at(number), out);
            }
            let Some(captures) = patterns.heading.captures(line) else {
                out.push(Violation { at: at(number), what: "a `## ` heading is `## RFC-NNNN: <ruling>`".to_string() });
                continue;
            };
            let id: u16 = captures[1].parse().expect("the heading pattern matched four digits");
            if corpus.retired.contains(&id) {
                out.push(Violation { at: at(number), what: format!("RFC-{id:04} is retired") });
            }
            if let Some(other) = corpus.sections.get(&id) {
                out.push(Violation { at: at(number), what: format!("RFC-{id:04} is also a section of {}", other.file) });
            }
            state = State::AwaitingStatus(Heading { id, title: captures[2].to_string() });
            continue;
        }
        if let State::In(open) | State::Unrecorded(open) = &mut state {
            open.words += line.split_whitespace().count();
        }
        if fenced || line.starts_with("```") {
            continue;
        }
        if line.starts_with("Status:") {
            out.push(Violation { at: at(number), what: "a status line stands only first in its section".to_string() });
        }
        let is_hash = |run: regex::Match<'_>| {
            let run = run.as_str();
            run.bytes().any(|byte| byte.is_ascii_digit()) && run.bytes().any(|byte| byte.is_ascii_lowercase())
        };
        if patterns.hex_run.find_iter(line).any(is_hash) {
            out.push(Violation { at: at(number), what: format!("a commit hash: {line}") });
        }
        for forbidden in &patterns.forbidden {
            if forbidden.pattern.is_match(line) {
                out.push(Violation { at: at(number), what: format!("{}: {line}", forbidden.what) });
            }
        }
        let Some(captures) = patterns.rule.captures(line) else {
            continue;
        };
        let rule: u32 = captures[1].parse().expect("the rule pattern matched digits");
        match &mut state {
            State::In(open) => {
                if rule <= open.last_rule {
                    out.push(Violation { at: at(number), what: format!("rule {rule} follows rule {}", open.last_rule) });
                }
                open.last_rule = rule;
                corpus
                    .sections
                    .get_mut(&open.id)
                    .expect("an open section was inserted when its status was read")
                    .rules
                    .insert(rule);
            }
            State::Unrecorded(open) => open.last_rule = rule,
            State::Intro | State::AwaitingStatus(_) => out.push(Violation {
                at: at(number),
                what: "a numbered rule outside any section".to_string(),
            }),
        }
    }
    match state {
        State::In(open) | State::Unrecorded(open) => open.close(at(text.lines().count()), out),
        State::AwaitingStatus(heading) => out.push(Violation {
            at: at(text.lines().count()),
            what: format!("RFC-{:04} has no status line", heading.id),
        }),
        State::Intro => {}
    }
}

/// The `## Index` section the headings give: one `### <file>` per topic
/// document, one line per decision in it.
pub fn index(corpus: &Corpus) -> String {
    let mut by_file: BTreeMap<&str, Vec<&Section>> = BTreeMap::new();
    for section in corpus.sections.values() {
        by_file.entry(section.file.as_str()).or_default().push(section);
    }
    let mut text = String::new();
    for (file, sections) in by_file {
        text.push_str(&format!("\n### [{file}]({file})\n\n"));
        for section in sections {
            let status = match section.status {
                Status::Accepted => "",
                Status::Proposed => " (Proposed)",
            };
            text.push_str(&format!("- RFC-{:04}: {}{status}\n", section.id, section.title));
        }
    }
    text
}

/// Directories no citation lives in: build output and tool state.
const SKIPPED_DIRS: [&str; 4] = ["target", ".git", ".claude", "node_modules"];

/// The file kinds a citation can live in.
const CITING_EXTENSIONS: [&str; 6] = ["rs", "md", "lalrpop", "toml", "acvus", "acvt"];

fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            let skipped = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| SKIPPED_DIRS.contains(&name));
            if !skipped {
                walk(&path, files);
            }
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| CITING_EXTENSIONS.contains(&ext))
        {
            files.push(path);
        }
    }
}

fn citations(patterns: &Patterns, document: Document<'_>, corpus: &Corpus, out: &mut Vec<Violation>) {
    let Document { file, text } = document;
    for (index, line) in text.lines().enumerate() {
        let at = || Location { file: file.to_string(), line: index + 1 };
        for captures in patterns.citation.captures_iter(line) {
            let id: u16 = captures[1].parse().expect("the citation pattern matched four digits");
            let after = &line[captures.get(0).expect("the whole match").end()..];
            let Some(section) = corpus.sections.get(&id) else {
                let what = if corpus.retired.contains(&id) { "retired" } else { "no decision" };
                out.push(Violation { at: at(), what: format!("RFC-{id:04} is {what}") });
                continue;
            };
            if let Some(cited) = patterns.cited_rules.captures(after) {
                for rule in patterns.rule_numbers.find_iter(&cited[1]) {
                    let rule: u32 = rule.as_str().parse().expect("the number pattern matched digits");
                    if !section.rules.contains(&rule) {
                        out.push(Violation { at: at(), what: format!("RFC-{id:04} has no rule {rule}") });
                    }
                }
            } else if patterns.other_label.is_match(after) {
                out.push(Violation { at: at(), what: format!("cite a rule as `RFC-{id:04} rule N`: {}", line.trim()) });
            }
        }
    }
}
