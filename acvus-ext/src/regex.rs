//! Regular expressions: the `Regex` extension type and its functions.

use acvus_extern::{
    ExternError, ExternRegistry, ExternType, IdentityVar, Interner, Pure, Runtime, extern_fn,
    extern_registry,
};

use crate::iter_pipeline::Iter;

#[derive(ExternType)]
pub struct Regex(regex::Regex);

#[extern_fn(effect = pure)]
fn regex(_: &Interner, pattern: String) -> Result<Regex, ExternError> {
    regex::Regex::new(&pattern)
        .map(Regex)
        .map_err(|e| ExternError::call("regex", format!("invalid pattern '{pattern}': {e}")))
}

#[extern_fn(effect = pure)]
fn regex_match(_: &Interner, re: Regex, text: String) -> bool {
    re.0.is_match(&text)
}

#[extern_fn(effect = pure)]
fn regex_find(_: &Interner, re: Regex, text: String) -> Option<String> {
    re.0.find(&text).map(|m| m.as_str().to_owned())
}

#[extern_fn(effect = pure)]
fn regex_find_all<I, Rt>(_: &Interner, re: Regex, text: String) -> Iter<String, Pure, I, Rt>
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
fn regex_replace(_: &Interner, text: String, re: Regex, replacement: String) -> String {
    re.0.replace_all(&text, replacement.as_str()).into_owned()
}

#[extern_fn(effect = pure)]
fn regex_split<I, Rt>(_: &Interner, re: Regex, text: String) -> Iter<String, Pure, I, Rt>
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
fn regex_extract<I, Rt>(_: &Interner, text: String, re: Regex) -> Iter<String, Pure, I, Rt>
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

pub fn regex_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        types: [Regex],
        fns: [regex, regex_match, regex_find, regex_find_all, regex_replace, regex_split, regex_extract],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{TypeRegistry, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let registered = regex_registry::<TypesOnly>().register(&i, &mut TypeRegistry::new());
        assert_eq!(registered.functions.len(), 7);
        assert_eq!(registered.handlers.len(), 7);
    }
}
