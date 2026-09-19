//! Regular expressions: the `Regex` and `Captures` extension types and
//! their functions.
//!
//! Every offset here is a byte offset into the searched text, the unit
//! `string::substring` cuts with. `string::find` reports a character index
//! instead, and the two units do not mix.
//!
//! A search borrows its regex and its text, so one compiled regex serves
//! any number of searches. The iterator-returning searches finish the
//! search before they return: `Iter::generate`'s stage outlives the call
//! and a borrow does not (RFC-0018), so a lazy stage cannot hold either
//! borrow.

use acvus_extern::{
    ClosureFn, EffectVar, ExternType, Fn1, IdentityVar, Pure, Registry, Runtime, TyArg, extern_fn,
    extern_registry,
};

use crate::iter::Iter;

#[derive(ExternType)]
#[repr(transparent)]
pub struct Regex(regex::Regex);

#[derive(TyArg, Clone)]
pub struct Match {
    start: u64,
    end: u64,
    text: String,
}

pub struct Group {
    name: Option<String>,
    matched: Option<Match>,
}

/// The groups of one match, owned: `regex::Captures` borrows the searched
/// text, which no value crossing into the language may do.
#[derive(ExternType)]
#[repr(transparent)]
pub struct Captures(Vec<Group>);

/// The `RegexBuilder` switches, all six of them. A script writes every one
/// or starts from `regex_flags()`.
#[derive(TyArg)]
pub struct RegexFlags {
    case_insensitive: bool,
    multi_line: bool,
    dot_matches_new_line: bool,
    ignore_whitespace: bool,
    unicode: bool,
    swap_greed: bool,
}

impl Default for RegexFlags {
    fn default() -> Self {
        RegexFlags {
            case_insensitive: false,
            multi_line: false,
            dot_matches_new_line: false,
            ignore_whitespace: false,
            unicode: true,
            swap_greed: false,
        }
    }
}

/// Why a pattern is not a regular expression. The `regex` crate reports a
/// syntax error and a compiled-size overflow through one `Error` whose
/// distinction is in its text alone, so a script acts on the message.
#[derive(TyArg)]
pub enum RegexError {
    Invalid { pattern: String, message: String },
}

fn match_of(m: regex::Match) -> Match {
    Match {
        start: m.start() as u64,
        end: m.end() as u64,
        text: m.as_str().to_owned(),
    }
}

fn groups_of(re: &regex::Regex, caps: &regex::Captures) -> Captures {
    Captures(
        re.capture_names()
            .enumerate()
            .map(|(i, name)| Group {
                name: name.map(str::to_owned),
                matched: caps.get(i).map(match_of),
            })
            .collect(),
    )
}

fn invalid(pattern: String, e: &regex::Error) -> RegexError {
    RegexError::Invalid {
        message: e.to_string(),
        pattern,
    }
}

fn at(text: &str, start: u64) -> Option<usize> {
    let start = usize::try_from(start).ok()?;
    text.is_char_boundary(start).then_some(start)
}

// -- Compiling ----------------------------------------------------------

/// Compiles `pattern` under the flags `regex_flags()` names.
#[extern_fn(effect = pure)]
fn regex(pattern: String) -> Result<Regex, RegexError> {
    regex::Regex::new(&pattern)
        .map(Regex)
        .map_err(|e| invalid(pattern, &e))
}

/// Compiles `pattern` under `flags`.
#[extern_fn(effect = pure)]
fn regex_with(pattern: String, flags: RegexFlags) -> Result<Regex, RegexError> {
    regex::RegexBuilder::new(&pattern)
        .case_insensitive(flags.case_insensitive)
        .multi_line(flags.multi_line)
        .dot_matches_new_line(flags.dot_matches_new_line)
        .ignore_whitespace(flags.ignore_whitespace)
        .unicode(flags.unicode)
        .swap_greed(flags.swap_greed)
        .build()
        .map(Regex)
        .map_err(|e| invalid(pattern, &e))
}

/// The flags `regex` itself compiles with.
#[extern_fn(effect = pure)]
fn regex_flags() -> RegexFlags {
    RegexFlags::default()
}

/// A pattern matching `text` literally.
#[extern_fn(effect = pure)]
fn escape(text: String) -> String {
    regex::escape(&text)
}

// -- Searching ----------------------------------------------------------

#[extern_fn(effect = pure)]
fn is_match(re: &Regex, text: &String) -> bool {
    re.0.is_match(text)
}

/// Whether a match begins at or after byte `start`. A `start` past the end
/// of `text` or inside a character matches nothing. `^` still anchors to
/// byte 0 of `text`, not to `start`.
#[extern_fn(effect = pure)]
fn is_match_at(re: &Regex, text: &String, start: u64) -> bool {
    at(text, start).is_some_and(|start| re.0.is_match_at(text, start))
}

/// The leftmost match, or `None` where the pattern does not match.
#[extern_fn(effect = pure)]
fn find(re: &Regex, text: &String) -> Option<Match> {
    re.0.find(text).map(match_of)
}

/// The leftmost match beginning at or after byte `start`. A `start` past
/// the end of `text` or inside a character gives `None`. `^` still anchors
/// to byte 0 of `text`, not to `start`.
#[extern_fn(effect = pure)]
fn find_at(re: &Regex, text: &String, start: u64) -> Option<Match> {
    let start = at(text, start)?;
    re.0.find_at(text, start).map(match_of)
}

/// Every non-overlapping match, left to right.
#[extern_fn(effect = pure)]
fn find_all<I, Rt>(re: &Regex, text: &String) -> Iter<Match, Pure, I, Rt>
where
    I: IdentityVar,
    Rt: Runtime,
{
    Iter::from_items(re.0.find_iter(text).map(match_of).collect())
}

/// The end of the shortest match beginning at the leftmost position that
/// matches, as a byte offset; `None` where the pattern does not match.
#[extern_fn(effect = pure)]
fn shortest_match(re: &Regex, text: &String) -> Option<u64> {
    re.0.shortest_match(text).map(|end| end as u64)
}

// -- Capture groups -----------------------------------------------------

/// The groups of the leftmost match, or `None` where the pattern does not
/// match.
#[extern_fn(effect = pure)]
fn captures(re: &Regex, text: &String) -> Option<Captures> {
    re.0.captures(text).map(|caps| groups_of(&re.0, &caps))
}

/// The groups of every non-overlapping match, left to right.
#[extern_fn(effect = pure)]
fn captures_all<I, Rt>(re: &Regex, text: &String) -> Iter<Captures, Pure, I, Rt>
where
    I: IdentityVar,
    Rt: Runtime,
{
    Iter::from_items(
        re.0.captures_iter(text)
            .map(|caps| groups_of(&re.0, &caps))
            .collect(),
    )
}

/// Group `i`, group 0 being the whole match. `None` where the pattern has
/// no such group or the group did not participate in this match.
#[extern_fn(effect = pure)]
fn group(caps: &Captures, i: u64) -> Option<Match> {
    let i = usize::try_from(i).ok()?;
    caps.0.get(i)?.matched.clone()
}

/// The group `name` names. `None` where the pattern has no group of that
/// name or the group did not participate in this match.
#[extern_fn(effect = pure)]
fn named(caps: &Captures, name: String) -> Option<Match> {
    caps.0
        .iter()
        .find(|group| group.name.as_deref() == Some(name.as_str()))?
        .matched
        .clone()
}

/// How many capture groups the pattern declares, not counting group 0.
#[extern_fn(effect = pure)]
fn group_count(re: &Regex) -> u64 {
    (re.0.captures_len() - 1) as u64
}

/// Each group's name at its own index, group 0 first and always unnamed.
#[extern_fn(effect = pure)]
fn group_names(re: &Regex) -> Vec<Option<String>> {
    re.0.capture_names()
        .map(|name| name.map(str::to_owned))
        .collect()
}

// -- Replacing ----------------------------------------------------------

/// `text` with the leftmost match replaced by `with`, in which `$1` and
/// `${name}` expand to the group of that number or name and `$$` is one
/// dollar.
#[extern_fn(effect = pure)]
fn replace(re: &Regex, text: &String, with: String) -> String {
    re.0.replace(text, with.as_str()).into_owned()
}

/// `text` with every non-overlapping match replaced, expanding `with` as
/// `replace` does.
#[extern_fn(effect = pure)]
fn replace_all(re: &Regex, text: &String, with: String) -> String {
    re.0.replace_all(text, with.as_str()).into_owned()
}

/// `text` with the first `n` matches replaced, expanding `with` as
/// `replace` does; an `n` of 0 replaces nothing.
#[extern_fn(effect = pure)]
fn replace_n(re: &Regex, text: &String, n: u64, with: String) -> String {
    match usize::try_from(n).unwrap_or(usize::MAX) {
        0 => text.clone(),
        n => re.0.replacen(text, n, with.as_str()).into_owned(),
    }
}

fn replace_with_now<E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    re: &Regex,
    text: &String,
    f: Fn1<Match, String, E, Rt>,
) -> String
where
    E: EffectVar,
    Rt: Runtime,
{
    let mut out = String::new();
    let mut last = 0;
    for m in re.0.find_iter(text) {
        out.push_str(&text[last..m.start()]);
        last = m.end();
        out.push_str(&f.call_now(rt, frame, (match_of(m),)));
    }
    out.push_str(&text[last..]);
    out
}

/// `text` with every non-overlapping match replaced by what `f` returns
/// for it. No `$1` expansion: `f` returns the replacement itself.
#[extern_fn(effect = E, sync = replace_with_now)]
async fn replace_with<E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    re: &Regex,
    text: &String,
    f: Fn1<Match, String, E, Rt>,
) -> String
where
    E: EffectVar,
    Rt: Runtime,
{
    let mut out = String::new();
    let mut last = 0;
    for m in re.0.find_iter(text) {
        out.push_str(&text[last..m.start()]);
        last = m.end();
        out.push_str(&f.call(rt, frame, (match_of(m),)).await);
    }
    out.push_str(&text[last..]);
    out
}

// -- Splitting ----------------------------------------------------------

/// The pieces of `text` between matches. A match at either end gives an
/// empty piece there.
#[extern_fn(effect = pure)]
fn split<I, Rt>(re: &Regex, text: &String) -> Iter<String, Pure, I, Rt>
where
    I: IdentityVar,
    Rt: Runtime,
{
    Iter::from_items(re.0.split(text).map(str::to_owned).collect())
}

/// At most `n` pieces: the last one holds the rest of `text`, matches and
/// all. An `n` of 0 gives no piece.
#[extern_fn(effect = pure)]
fn split_n<I, Rt>(re: &Regex, text: &String, n: u64) -> Iter<String, Pure, I, Rt>
where
    I: IdentityVar,
    Rt: Runtime,
{
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    Iter::from_items(re.0.splitn(text, n).map(str::to_owned).collect())
}

pub fn regex_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [Regex, Captures],
        fns: [
            regex, regex_with, regex_flags, escape,
            is_match, is_match_at, find, find_at, find_all, shortest_match,
            captures, captures_all, group, named, group_count, group_names,
            replace, replace_all, replace_n, replace_with,
            split, split_n,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let registered =
            Externs::combine(vec![regex_registry::<TypesOnly>()], &i).expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(registered.functions.len() - core.functions.len(), 22);
        assert_eq!(registered.handlers.len() - core.handlers.len(), 22);
    }
}
