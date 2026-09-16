//! String operations. All pure.

use acvus_extern::{ExternError, Registry, Runtime, extern_fn, extern_registry};


#[extern_fn(effect = pure)]
fn len_str<R>(_: &R, s: String) -> i64
where
    R: Runtime,
{
    s.len() as i64
}

#[extern_fn(effect = pure)]
fn concat<R>(_: &R, a: &String, b: &String) -> String
where
    R: Runtime,
{
    let mut s = String::with_capacity(a.len() + b.len());
    s.push_str(a);
    s.push_str(b);
    s
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
fn split_str<R>(_: &R, s: String, sep: String) -> Vec<String>
where
    R: Runtime,
{
    s.split(&*sep).map(str::to_owned).collect()
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
fn to_bytes<R>(_: &R, s: String) -> Vec<u8>
where
    R: Runtime,
{
    s.into_bytes()
}

#[extern_fn(effect = pure)]
fn to_utf8<R>(_: &R, bytes: Vec<u8>) -> Option<String>
where
    R: Runtime,
{
    String::from_utf8(bytes).ok()
}

#[extern_fn(effect = pure)]
fn to_utf8_lossy<R>(_: &R, bytes: Vec<u8>) -> String
where
    R: Runtime,
{
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn string_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        fns: [
            len_str, concat, trim, trim_start, trim_end, upper, lower, contains_str,
            starts_with_str, ends_with_str, replace_str, split_str, repeat_str,
            substring, to_bytes, to_utf8, to_utf8_lossy,
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
        let reg = Externs::combine(vec![string_registry::<TypesOnly>()], &i)
            .expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 17);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 17);
    }
}
