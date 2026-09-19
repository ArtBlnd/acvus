//! String operations. All pure. A string is a value, not a container of
//! characters (RFC-0028): there is no `get` returning a reference into it,
//! because a `String` is UTF-8 and a scalar value is not a storage inside
//! it; a character is read out by value with `char_at`, as the `char` it
//! is (RFC-0058).
//!
//! Every function that reads takes a `&str`. A function whose result is a
//! run of its argument's own bytes returns `&str` and the caller holds the
//! argument's loan for as long as the result; a function that builds new
//! bytes returns `String` (RFC-0062 Decisions 2 and 3). Two units coexist
//! here and each function states its own: `len`, `find`, `rfind` and
//! `substring` are in bytes, `char_at`, `chars` and the `pad_*` width in
//! Unicode scalar values.

use acvus_extern::{
    EffectVar, Erased, IdentityVar, OneValue, Registry, Runtime, extern_fn, extern_registry,
};

use crate::iter::Iter;

fn padding(fill: &str, count: usize) -> String {
    fill.chars().cycle().take(count).collect()
}

fn shortfall(s: &str, width: i64) -> usize {
    let Ok(width) = usize::try_from(width) else {
        return 0;
    };
    width.saturating_sub(s.chars().count())
}

/// The length in bytes.
#[extern_fn(effect = pure)]
fn len(s: &str) -> u64 {
    s.len() as u64
}

#[extern_fn(effect = pure)]
fn is_empty(s: &str) -> bool {
    s.is_empty()
}

#[extern_fn(effect = pure)]
fn concat(a: &str, b: &str) -> String {
    let mut s = String::with_capacity(a.len() + b.len());
    s.push_str(a);
    s.push_str(b);
    s
}

/// A view of `s` with the outer whitespace cut away. The bytes are `s`'s
/// own, so the caller holds `s`'s loan for as long as the result.
#[extern_fn(effect = pure)]
fn trim(s: &str) -> &str {
    s.trim()
}

/// A view of `s` with the outer whitespace cut away. The bytes are `s`'s
/// own, so the caller holds `s`'s loan for as long as the result.
#[extern_fn(effect = pure)]
fn trim_start(s: &str) -> &str {
    s.trim_start()
}

/// A view of `s` with the outer whitespace cut away. The bytes are `s`'s
/// own, so the caller holds `s`'s loan for as long as the result.
#[extern_fn(effect = pure)]
fn trim_end(s: &str) -> &str {
    s.trim_end()
}

#[extern_fn(effect = pure)]
fn upper(s: &str) -> String {
    s.to_uppercase()
}

#[extern_fn(effect = pure)]
fn lower(s: &str) -> String {
    s.to_lowercase()
}

#[extern_fn(effect = pure)]
fn contains(s: &str, pat: &str) -> bool {
    s.contains(pat)
}

#[extern_fn(effect = pure)]
fn starts_with_str(s: &str, pat: &str) -> bool {
    s.starts_with(pat)
}

#[extern_fn(effect = pure)]
fn ends_with_str(s: &str, pat: &str) -> bool {
    s.ends_with(pat)
}

#[extern_fn(effect = pure)]
fn replace_str(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

#[extern_fn(effect = pure)]
fn split_str<Rt>(rt: &Rt, s: &str, sep: &str) -> Vec<Erased<Rt, String>>
where
    Rt: Runtime,
{
    s.split(sep)
        .map(|part| Erased::new(rt, part.to_owned()))
        .collect()
}

#[extern_fn(effect = pure)]
fn repeat_str(s: &str, n: u64) -> String {
    let Ok(n) = usize::try_from(n) else {
        panic!("repeat_str: count {n} exceeds the address space")
    };
    s.repeat(n)
}

/// A view of the bytes `[start, end)` of `s`, which are `s`'s own, so the
/// caller holds `s`'s loan for as long as the result. Both offsets are byte
/// offsets and both must be on a character boundary; `start` past `end`, an
/// offset past the length, or an offset inside a character is refused
/// (RFC-0062 Decision 2).
#[extern_fn(effect = pure)]
fn substring(s: &str, start: u64, end: u64) -> &str {
    let refuse = |what: &str| -> ! {
        panic!(
            "substring: {what} for the range {start}..{end} over {} bytes of {s:?}",
            s.len()
        )
    };
    let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
        refuse("an offset exceeds the address space")
    };
    if start > end {
        refuse("the range is inverted")
    }
    if !s.is_char_boundary(start) {
        refuse("start is not on a character boundary")
    }
    if !s.is_char_boundary(end) {
        refuse("end is not on a character boundary")
    }
    &s[start..end]
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

/// `i` counts Unicode scalar values.
#[extern_fn(effect = pure)]
fn char_at(s: &str, i: i64) -> char {
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
    c
}

// -- Producers ----------------------------------------------------------

fn iter_of<T, E, I, Rt>(items: Vec<T>) -> Iter<T, E, I, Rt>
where
    T: OneValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut items = items.into_iter();
    Iter::generate(move |_| items.next())
}

/// One Unicode scalar value per step.
#[extern_fn(effect = pure)]
fn chars<E, I, Rt>(s: &str) -> Iter<char, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.chars().collect())
}

#[extern_fn(effect = pure)]
fn lines<E, I, Rt>(s: &str) -> Iter<String, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.lines().map(str::to_owned).collect())
}

/// One byte per step.
#[extern_fn(effect = pure)]
fn bytes<E, I, Rt>(s: &str) -> Iter<i64, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.bytes().map(i64::from).collect())
}

#[extern_fn(effect = pure)]
fn split_whitespace<E, I, Rt>(s: &str) -> Iter<String, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    iter_of(s.split_whitespace().map(str::to_owned).collect())
}

// -- Searching and shaping ---------------------------------------------

/// The byte offset of the first `pat` in `s`.
#[extern_fn(effect = pure)]
fn find(s: &str, pat: &str) -> Option<i64> {
    s.find(pat).map(|byte| byte as i64)
}

/// The byte offset of the last `pat` in `s`.
#[extern_fn(effect = pure)]
fn rfind(s: &str, pat: &str) -> Option<i64> {
    s.rfind(pat).map(|byte| byte as i64)
}

/// JS `padStart`: `fill` repeated and cut to the shortfall on the left; an
/// empty `fill` or a `width` at or below the length leaves `s` as it is.
/// `width` counts Unicode scalar values.
#[extern_fn(effect = pure)]
fn pad_start(s: &str, width: i64, fill: &str) -> String {
    let mut padded = padding(fill, shortfall(s, width));
    padded.push_str(s);
    padded
}

/// JS `padEnd`: as `pad_start`, on the right.
#[extern_fn(effect = pure)]
fn pad_end(s: &str, width: i64, fill: &str) -> String {
    let mut padded = s.to_owned();
    padded.push_str(&padding(fill, shortfall(s, width)));
    padded
}

#[extern_fn(effect = pure)]
fn strip_prefix(s: &str, pat: &str) -> Option<String> {
    s.strip_prefix(pat).map(str::to_owned)
}

#[extern_fn(effect = pure)]
fn strip_suffix(s: &str, pat: &str) -> Option<String> {
    s.strip_suffix(pat).map(str::to_owned)
}

/// The text before and after the first `pat`, as a two-element Vec: an
/// extern function returns no tuple (`acvus-extern` has no `Cross` for
/// one) and no array of a constant length (`Len<K>` is a length variable).
#[extern_fn(effect = pure)]
fn split_once<Rt>(rt: &Rt, s: &str, pat: &str) -> Option<Vec<Erased<Rt, String>>>
where
    Rt: Runtime,
{
    s.split_once(pat).map(|(head, tail)| {
        vec![
            Erased::new(rt, head.to_owned()),
            Erased::new(rt, tail.to_owned()),
        ]
    })
}

#[extern_fn(effect = pure)]
fn eq_ignore_case(a: &str, b: &str) -> bool {
    a.to_lowercase() == b.to_lowercase()
}

#[extern_fn(effect = pure)]
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return String::new();
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
