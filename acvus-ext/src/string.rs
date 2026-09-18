//! String operations. All pure. A string is a value, not a container of
//! characters (RFC-0028): there is no `get` returning a reference into it,
//! because a `char` has no acvus type to be referenced as; a character is
//! read out by value with `char_at`.

use acvus_extern::{
    Cross, EffectVar, Erased, IdentityVar, Registry, Runtime, extern_fn, extern_registry,
};

use crate::iter::Iter;

fn char_index(s: &str, byte: usize) -> i64 {
    s[..byte].chars().count() as i64
}

fn padding(fill: &str, count: usize) -> String {
    fill.chars().cycle().take(count).collect()
}

fn shortfall(s: &str, width: i64) -> usize {
    let Ok(width) = usize::try_from(width) else {
        return 0;
    };
    width.saturating_sub(s.chars().count())
}

/// The length in characters.
#[extern_fn(effect = pure)]
fn len(s: &String) -> u64 {
    s.chars().count() as u64
}

#[extern_fn(effect = pure)]
fn is_empty(s: &String) -> bool {
    s.is_empty()
}

#[extern_fn(effect = pure)]
fn concat(a: &String, b: &String) -> String {
    let mut s = String::with_capacity(a.len() + b.len());
    s.push_str(a);
    s.push_str(b);
    s
}

#[extern_fn(effect = pure)]
fn trim(s: String) -> String {
    s.trim().to_owned()
}

#[extern_fn(effect = pure)]
fn trim_start(s: String) -> String {
    s.trim_start().to_owned()
}

#[extern_fn(effect = pure)]
fn trim_end(s: String) -> String {
    s.trim_end().to_owned()
}

#[extern_fn(effect = pure)]
fn upper(s: String) -> String {
    s.to_uppercase()
}

#[extern_fn(effect = pure)]
fn lower(s: String) -> String {
    s.to_lowercase()
}

#[extern_fn(effect = pure)]
fn contains(s: &String, pat: String) -> bool {
    s.contains(&*pat)
}

#[extern_fn(effect = pure)]
fn starts_with_str(s: String, pat: String) -> bool {
    s.starts_with(&*pat)
}

#[extern_fn(effect = pure)]
fn ends_with_str(s: String, pat: String) -> bool {
    s.ends_with(&*pat)
}

#[extern_fn(effect = pure)]
fn replace_str(s: String, from: String, to: String) -> String {
    s.replace(&*from, &to)
}

#[extern_fn(effect = pure)]
fn split_str<Rt>(rt: &Rt, s: String, sep: String) -> Vec<Erased<Rt, String>>
where
    Rt: Runtime,
{
    s.split(&*sep)
        .map(|part| Erased::new(rt, part.to_owned()))
        .collect()
}

#[extern_fn(effect = pure)]
fn repeat_str(s: String, n: u64) -> String {
    let Ok(n) = usize::try_from(n) else {
        panic!("repeat_str: count {n} exceeds the address space")
    };
    s.repeat(n)
}

/// Byte range `[start, end)` clamped to the string; an inverted range is empty.
#[extern_fn(effect = pure)]
fn substring(s: String, start: i64, end: i64) -> String {
    let start = start.max(0) as usize;
    let end = (end.max(0) as usize).min(s.len());
    let start = start.min(end);
    s[start..end].to_owned()
}

#[extern_fn(effect = pure)]
fn to_bytes(s: String) -> Vec<u8> {
    s.into_bytes()
}

#[extern_fn(effect = pure)]
fn to_utf8(bytes: Vec<u8>) -> Option<String> {
    String::from_utf8(bytes).ok()
}

#[extern_fn(effect = pure)]
fn to_utf8_lossy(bytes: Vec<u8>) -> String {
    String::from_utf8_lossy(&bytes).into_owned()
}

#[extern_fn(effect = pure)]
fn char_at(s: &String, i: i64) -> String {
    let out_of_range = || -> ! {
        panic!(
            "char_at: index {i} is out of range for length {}",
            s.chars().count()
        )
    };
    let Ok(index) = usize::try_from(i) else {
        out_of_range()
    };
    let Some(c) = s.chars().nth(index) else {
        out_of_range()
    };
    c.to_string()
}

// -- Producers ----------------------------------------------------------

fn iter_of<T, E, I, Rt>(items: Vec<T>) -> Iter<T, E, I, Rt>
where
    T: Cross<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut items = items.into_iter();
    Iter::generate(move |_| items.next())
}

#[extern_fn(effect = pure)]
fn chars<E, I, Rt>(s: String) -> Iter<String, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.chars().map(|c| c.to_string()).collect())
}

#[extern_fn(effect = pure)]
fn lines<E, I, Rt>(s: String) -> Iter<String, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.lines().map(str::to_owned).collect())
}

#[extern_fn(effect = pure)]
fn bytes<E, I, Rt>(s: String) -> Iter<i64, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.bytes().map(i64::from).collect())
}

#[extern_fn(effect = pure)]
fn split_whitespace<E, I, Rt>(s: String) -> Iter<String, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.split_whitespace().map(str::to_owned).collect())
}

// -- Searching and shaping ---------------------------------------------

/// The character index of the first `pat` in `s`.
#[extern_fn(effect = pure)]
fn find(s: &String, pat: String) -> Option<i64> {
    s.find(&*pat).map(|byte| char_index(s, byte))
}

/// The character index of the last `pat` in `s`.
#[extern_fn(effect = pure)]
fn rfind(s: &String, pat: String) -> Option<i64> {
    s.rfind(&*pat).map(|byte| char_index(s, byte))
}

/// JS `padStart`: `fill` repeated and cut to the shortfall on the left; an
/// empty `fill` or a `width` at or below the length leaves `s` as it is.
#[extern_fn(effect = pure)]
fn pad_start(s: String, width: i64, fill: String) -> String {
    let mut padded = padding(&fill, shortfall(&s, width));
    padded.push_str(&s);
    padded
}

/// JS `padEnd`: as `pad_start`, on the right.
#[extern_fn(effect = pure)]
fn pad_end(s: String, width: i64, fill: String) -> String {
    let mut padded = s;
    padded.push_str(&padding(&fill, shortfall(&padded, width)));
    padded
}

#[extern_fn(effect = pure)]
fn strip_prefix(s: String, pat: String) -> Option<String> {
    s.strip_prefix(&*pat).map(str::to_owned)
}

#[extern_fn(effect = pure)]
fn strip_suffix(s: String, pat: String) -> Option<String> {
    s.strip_suffix(&*pat).map(str::to_owned)
}

/// The text before and after the first `pat`, as a two-element Vec: an
/// extern function returns no tuple (`acvus-extern` has no `Cross` for
/// one) and no array of a constant length (`Len<K>` is a length variable).
#[extern_fn(effect = pure)]
fn split_once<Rt>(rt: &Rt, s: String, pat: String) -> Option<Vec<Erased<Rt, String>>>
where
    Rt: Runtime,
{
    s.split_once(&*pat).map(|(head, tail)| {
        vec![
            Erased::new(rt, head.to_owned()),
            Erased::new(rt, tail.to_owned()),
        ]
    })
}

#[extern_fn(effect = pure)]
fn eq_ignore_case(a: &String, b: &String) -> bool {
    a.to_lowercase() == b.to_lowercase()
}

#[extern_fn(effect = pure)]
fn capitalize(s: String) -> String {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return s;
    };
    let mut out: String = first.to_uppercase().collect();
    out.push_str(chars.as_str());
    out
}

pub fn string_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "string",
        fns: [
            len, is_empty, concat, trim, trim_start, trim_end, upper, lower, contains,
            starts_with_str, ends_with_str, replace_str, split_str, repeat_str,
            substring, to_bytes, to_utf8, to_utf8_lossy,
            char_at, chars, lines, bytes, split_whitespace,
            find, rfind, pad_start, pad_end, strip_prefix, strip_suffix, split_once,
            eq_ignore_case, capitalize,
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
        let reg =
            Externs::combine(vec![string_registry::<TypesOnly>()], &i).expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 32);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 32);
    }
}
