//! Type conversions. All pure.

use acvus_extern::{
    ExternError, Registry, Monomorphize, Runtime, extern_fn, extern_registry,
};

#[extern_fn(effect = pure)]
fn to_string<A, R>(_: &R, val: A) -> String
where
    A: Monomorphize<(i64, f64, bool, u8, String)> + ToString,
    R: Runtime,
{
    val.to_string()
}

/// Whole-number reading of a scalar, as scripts see it.
trait ToInt {
    fn to_int(self) -> Result<i64, ExternError>;
}

impl ToInt for i64 {
    fn to_int(self) -> Result<i64, ExternError> {
        Ok(self)
    }
}
impl ToInt for f64 {
    fn to_int(self) -> Result<i64, ExternError> {
        Ok(self as i64)
    }
}
impl ToInt for bool {
    fn to_int(self) -> Result<i64, ExternError> {
        Ok(i64::from(self))
    }
}
impl ToInt for u8 {
    fn to_int(self) -> Result<i64, ExternError> {
        Ok(i64::from(self))
    }
}
impl ToInt for String {
    fn to_int(self) -> Result<i64, ExternError> {
        self.parse::<i64>()
            .map_err(|e| ExternError::call("to_int", format!("cannot parse string: {e}")))
    }
}

#[extern_fn(effect = pure)]
fn to_int<A, R>(_: &R, val: A) -> Result<i64, ExternError>
where
    A: Monomorphize<(i64, f64, bool, u8, String)> + ToInt,
    R: Runtime,
{
    val.to_int()
}

#[extern_fn(effect = pure)]
fn to_float<R>(_: &R, n: i64) -> f64
where
    R: Runtime,
{
    n as f64
}

#[extern_fn(effect = pure)]
fn char_to_int<R>(_: &R, s: String) -> Result<i64, ExternError>
where
    R: Runtime,
{
    match s.chars().next() {
        Some(c) => Ok(c as i64),
        None => Err(ExternError::call("char_to_int", "empty string")),
    }
}

#[extern_fn(effect = pure)]
fn int_to_char<R>(_: &R, n: i64) -> Result<String, ExternError>
where
    R: Runtime,
{
    let code = u32::try_from(n)
        .map_err(|_| ExternError::call("int_to_char", format!("{n} is not a code point")))?;
    match char::from_u32(code) {
        Some(c) => Ok(c.to_string()),
        None => Err(ExternError::call(
            "int_to_char",
            format!("{n} is not a code point"),
        )),
    }
}

pub fn conversion_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        fns: [to_string, to_int, to_float, char_to_int, int_to_char],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = Externs::combine(vec![conversion_registry::<TypesOnly>()], &i).expect("registry combines");
        assert_eq!(reg.functions.len(), 5);
        assert_eq!(reg.handlers.len(), 5);
    }
}
