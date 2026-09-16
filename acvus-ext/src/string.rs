//! String operations. All pure.

use acvus_extern::{Registry, Runtime, Trap, extern_fn, extern_registry};

#[extern_fn(effect = pure)]
fn len_str(s: String) -> i64 {
    s.len() as i64
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
fn contains_str(s: String, pat: String) -> bool {
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
fn split_str(s: String, sep: String) -> Vec<String> {
    s.split(&*sep).map(str::to_owned).collect()
}

#[extern_fn(effect = pure)]
fn repeat_str(s: String, n: u64) -> Result<String, Trap> {
    let Ok(n) = usize::try_from(n) else {
        return Err(Trap::call(
            "repeat_str",
            format!("count {n} exceeds the address space"),
        ));
    };
    Ok(s.repeat(n))
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
        let reg =
            Externs::combine(vec![string_registry::<TypesOnly>()], &i).expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 17);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 17);
    }
}
