//! String operations. All pure.

use acvus_extern::{ExternError, ExternRegistry, Runtime, extern_fn, extern_registry};

use crate::list::List;

#[extern_fn(effect = pure)]
fn len_str<R>(_: &R, s: String) -> i64
where
    R: Runtime,
{
    s.len() as i64
}

#[extern_fn(effect = pure)]
fn trim<R>(_: &R, s: String) -> String
where
    R: Runtime,
{
    s.trim().to_owned()
}

#[extern_fn(effect = pure)]
fn trim_start<R>(_: &R, s: String) -> String
where
    R: Runtime,
{
    s.trim_start().to_owned()
}

#[extern_fn(effect = pure)]
fn trim_end<R>(_: &R, s: String) -> String
where
    R: Runtime,
{
    s.trim_end().to_owned()
}

#[extern_fn(effect = pure)]
fn upper<R>(_: &R, s: String) -> String
where
    R: Runtime,
{
    s.to_uppercase()
}

#[extern_fn(effect = pure)]
fn lower<R>(_: &R, s: String) -> String
where
    R: Runtime,
{
    s.to_lowercase()
}

#[extern_fn(effect = pure)]
fn contains_str<R>(_: &R, s: String, pat: String) -> bool
where
    R: Runtime,
{
    s.contains(&*pat)
}

#[extern_fn(effect = pure)]
fn starts_with_str<R>(_: &R, s: String, pat: String) -> bool
where
    R: Runtime,
{
    s.starts_with(&*pat)
}

#[extern_fn(effect = pure)]
fn ends_with_str<R>(_: &R, s: String, pat: String) -> bool
where
    R: Runtime,
{
    s.ends_with(&*pat)
}

#[extern_fn(effect = pure)]
fn replace_str<R>(_: &R, s: String, from: String, to: String) -> String
where
    R: Runtime,
{
    s.replace(&*from, &to)
}

#[extern_fn(effect = pure)]
fn split_str<R>(_: &R, s: String, sep: String) -> List<String>
where
    R: Runtime,
{
    List(s.split(&*sep).map(str::to_owned).collect())
}

#[extern_fn(effect = pure)]
fn repeat_str<R>(_: &R, s: String, n: i64) -> Result<String, ExternError>
where
    R: Runtime,
{
    let n = usize::try_from(n)
        .map_err(|_| ExternError::call("repeat_str", format!("negative count {n}")))?;
    Ok(s.repeat(n))
}

/// Byte range `[start, end)` clamped to the string; an inverted range is empty.
#[extern_fn(effect = pure)]
fn substring<R>(_: &R, s: String, start: i64, end: i64) -> String
where
    R: Runtime,
{
    let start = start.max(0) as usize;
    let end = (end.max(0) as usize).min(s.len());
    let start = start.min(end);
    s[start..end].to_owned()
}

#[extern_fn(effect = pure)]
fn to_bytes<R>(_: &R, s: String) -> List<u8>
where
    R: Runtime,
{
    List(s.into_bytes())
}

#[extern_fn(effect = pure)]
fn to_utf8<R>(_: &R, bytes: List<u8>) -> Option<String>
where
    R: Runtime,
{
    String::from_utf8(bytes.0).ok()
}

#[extern_fn(effect = pure)]
fn to_utf8_lossy<R>(_: &R, bytes: List<u8>) -> String
where
    R: Runtime,
{
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
    use acvus_extern::{Interner, TypeRegistry, TypesOnly};

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
