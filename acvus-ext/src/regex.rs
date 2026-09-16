//! Regular expressions: the `Regex` extension type and its functions.

use acvus_extern::{
    ExternType, IdentityVar, Pure, Registry, Runtime, TyArg, extern_fn, extern_registry,
};

use crate::iter_pipeline::Iter;

#[derive(ExternType)]
#[repr(transparent)]
pub struct Regex(regex::Regex);

/// Why a pattern is not a regular expression.
#[derive(TyArg)]
pub enum RegexError {
    Invalid { pattern: String, message: String },
}

#[extern_fn(effect = pure)]
fn regex(pattern: String) -> Result<Regex, RegexError> {
    regex::Regex::new(&pattern)
        .map(Regex)
        .map_err(|e| RegexError::Invalid {
            message: e.to_string(),
            pattern,
        })
}

#[extern_fn(effect = pure)]
fn regex_match(re: Regex, text: String) -> bool {
    re.0.is_match(&text)
}

#[extern_fn(effect = pure)]
fn regex_find(re: Regex, text: String) -> Option<String> {
    re.0.find(&text).map(|m| m.as_str().to_owned())
}

#[extern_fn(effect = pure)]
fn regex_find_all<I, Rt>(re: Regex, text: String) -> Iter<String, Pure, I, Rt>
where
    I: IdentityVar,
    Rt: Runtime,
{
    let mut start = 0;
    Iter::generate(move || {
        let m = re.0.find_at(&text, start)?;
        start = m.end();
        Some(m.as_str().to_owned())
    })
}

#[extern_fn(effect = pure)]
fn regex_replace(text: String, re: Regex, replacement: String) -> String {
    re.0.replace_all(&text, replacement.as_str()).into_owned()
}

#[extern_fn(effect = pure)]
fn regex_split<I, Rt>(re: Regex, text: String) -> Iter<String, Pure, I, Rt>
where
    I: IdentityVar,
    Rt: Runtime,
{
    let mut last_end = 0;
    let mut done = false;
    Iter::generate(move || {
        if done {
            return None;
        }
        match re.0.find_at(&text, last_end) {
            Some(m) => {
                let segment = text[last_end..m.start()].to_owned();
                last_end = m.end();
                Some(segment)
            }
            None => {
                done = true;
                Some(text[last_end..].to_owned())
            }
        }
    })
}

/// Capture group 1 of every match.
#[extern_fn(effect = pure)]
fn regex_extract<I, Rt>(text: String, re: Regex) -> Iter<String, Pure, I, Rt>
where
    I: IdentityVar,
    Rt: Runtime,
{
    let mut start = 0;
    Iter::generate(move || {
        loop {
            let caps = re.0.captures_at(&text, start)?;
            let full = caps.get(0)?;
            start = full.end();
            if let Some(group1) = caps.get(1) {
                return Some(group1.as_str().to_owned());
            }
        }
    })
}

pub fn regex_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [Regex],
        fns: [regex, regex_match, regex_find, regex_find_all, regex_replace, regex_split, regex_extract],
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
        assert_eq!(registered.functions.len() - core.functions.len(), 7);
        assert_eq!(registered.handlers.len() - core.handlers.len(), 7);
    }
}
