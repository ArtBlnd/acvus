//! String operations. All pure.

use acvus_extern::{ExternError, ExternRegistry, Interner, Runtime, extern_fn, extern_registry};

use crate::list::List;

#[extern_fn(effect = pure)]
fn len_str(_: &Interner, s: String) -> i64 {
    s.len() as i64
}

#[extern_fn(effect = pure)]
fn trim(_: &Interner, s: String) -> String {
    s.trim().to_owned()
}

#[extern_fn(effect = pure)]
fn trim_start(_: &Interner, s: String) -> String {
    s.trim_start().to_owned()
}

#[extern_fn(effect = pure)]
fn trim_end(_: &Interner, s: String) -> String {
    s.trim_end().to_owned()
}

#[extern_fn(effect = pure)]
fn upper(_: &Interner, s: String) -> String {
    s.to_uppercase()
}

#[extern_fn(effect = pure)]
fn lower(_: &Interner, s: String) -> String {
    s.to_lowercase()
}

#[extern_fn(effect = pure)]
fn contains_str(_: &Interner, s: String, pat: String) -> bool {
    s.contains(&*pat)
}

#[extern_fn(effect = pure)]
fn starts_with_str(_: &Interner, s: String, pat: String) -> bool {
    s.starts_with(&*pat)
}

#[extern_fn(effect = pure)]
fn ends_with_str(_: &Interner, s: String, pat: String) -> bool {
    s.ends_with(&*pat)
}

#[extern_fn(effect = pure)]
fn replace_str(_: &Interner, s: String, from: String, to: String) -> String {
    s.replace(&*from, &to)
}

#[extern_fn(effect = pure)]
fn split_str(_: &Interner, s: String, sep: String) -> List<String> {
    List(s.split(&*sep).map(str::to_owned).collect())
}

#[extern_fn(effect = pure)]
fn repeat_str(_: &Interner, s: String, n: i64) -> Result<String, ExternError> {
    let n = usize::try_from(n)
        .map_err(|_| ExternError::call("repeat_str", format!("negative count {n}")))?;
    Ok(s.repeat(n))
}

/// Byte range `[start, end)` clamped to the string; an inverted range is empty.
#[extern_fn(effect = pure)]
fn substring(_: &Interner, s: String, start: i64, end: i64) -> String {
    let start = start.max(0) as usize;
    let end = (end.max(0) as usize).min(s.len());
    let start = start.min(end);
    s[start..end].to_owned()
}

#[extern_fn(effect = pure)]
fn to_bytes(_: &Interner, s: String) -> List<u8> {
    List(s.into_bytes())
}

#[extern_fn(effect = pure)]
fn to_utf8(_: &Interner, bytes: List<u8>) -> Option<String> {
    String::from_utf8(bytes.0).ok()
}

#[extern_fn(effect = pure)]
fn to_utf8_lossy(_: &Interner, bytes: List<u8>) -> String {
    String::from_utf8_lossy(&bytes.0).into_owned()
}

pub fn string_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        fns: [
            len_str, trim, trim_start, trim_end, upper, lower, contains_str,
            starts_with_str, ends_with_str, replace_str, split_str, repeat_str,
            substring, to_bytes, to_utf8, to_utf8_lossy,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{TypeRegistry, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let mut tr = TypeRegistry::new();
        crate::list::list_registry::<TypesOnly>().register(&i, &mut tr);
        let reg = string_registry::<TypesOnly>().register(&i, &mut tr);
        assert_eq!(reg.functions.len(), 16);
        assert_eq!(reg.handlers.len(), 16);
    }
}
