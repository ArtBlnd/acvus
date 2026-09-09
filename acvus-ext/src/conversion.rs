//! Type conversions. All pure.

use acvus_extern::{
    ExternError, ExternRegistry, Interner, Monomorphize, Runtime, extern_fn, extern_registry,
};

/// The text of a scalar, as scripts see it.
trait ScalarText {
    fn text(self) -> String;
}

impl ScalarText for i64 {
    fn text(self) -> String {
        self.to_string()
    }
}
impl ScalarText for f64 {
    fn text(self) -> String {
        self.to_string()
    }
}
impl ScalarText for bool {
    fn text(self) -> String {
        self.to_string()
    }
}
impl ScalarText for u8 {
    fn text(self) -> String {
        format!("0x{self:02x}")
    }
}
impl ScalarText for String {
    fn text(self) -> String {
        self
    }
}

#[extern_fn(effect = pure)]
fn to_string<A>(_: &Interner, val: A) -> String
where
    A: Monomorphize<(i64, f64, bool, u8, String)> + ScalarText,
{
    val.text()
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
fn to_int<A>(_: &Interner, val: A) -> Result<i64, ExternError>
where
    A: Monomorphize<(i64, f64, bool, u8, String)> + ToInt,
{
    val.to_int()
}

#[extern_fn(effect = pure)]
fn to_float(_: &Interner, n: i64) -> f64 {
    n as f64
}

#[extern_fn(effect = pure)]
fn char_to_int(_: &Interner, s: String) -> Result<i64, ExternError> {
    match s.chars().next() {
        Some(c) => Ok(c as i64),
        None => Err(ExternError::call("char_to_int", "empty string")),
    }
}

#[extern_fn(effect = pure)]
fn int_to_char(_: &Interner, n: i64) -> Result<String, ExternError> {
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

pub fn conversion_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        fns: [to_string, to_int, to_float, char_to_int, int_to_char],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{TypeRegistry, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = conversion_registry::<TypesOnly>().register(&i, &mut TypeRegistry::new());
        assert_eq!(reg.functions.len(), 5);
        assert_eq!(reg.handlers.len(), 5);
    }
}
