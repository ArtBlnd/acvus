//! Type conversions. All pure.

use acvus_extern::{
    ExternError, ExternRegistry, Interner, IntoValue, Runtime, Scalar, TyVar, extern_fn,
    extern_registry,
};

#[extern_fn(effect = pure)]
fn to_string<T, Rt>(i: &Interner, val: T) -> Result<String, Rt::Error>
where
    T: TyVar + IntoValue<Rt>,
    Rt: Runtime,
{
    match Rt::scalar(val.into_value(i)) {
        Ok(Scalar::Int(n)) => Ok(n.to_string()),
        Ok(Scalar::Float(f)) => Ok(f.to_string()),
        Ok(Scalar::Bool(b)) => Ok(b.to_string()),
        Ok(Scalar::String(s)) => Ok(s),
        Ok(Scalar::Byte(b)) => Ok(format!("0x{b:02x}")),
        Ok(Scalar::Unit) => Ok("()".to_string()),
        Err(_) => Err(ExternError::call("to_string", "not a scalar").into()),
    }
}

#[extern_fn(effect = pure)]
fn to_int<T, Rt>(i: &Interner, val: T) -> Result<i64, Rt::Error>
where
    T: TyVar + IntoValue<Rt>,
    Rt: Runtime,
{
    match Rt::scalar(val.into_value(i)) {
        Ok(Scalar::Int(n)) => Ok(n),
        Ok(Scalar::Float(f)) => Ok(f as i64),
        Ok(Scalar::String(s)) => s
            .parse::<i64>()
            .map_err(|e| ExternError::call("to_int", format!("cannot parse string: {e}")).into()),
        Ok(Scalar::Bool(b)) => Ok(i64::from(b)),
        Ok(Scalar::Byte(b)) => Ok(i64::from(b)),
        Ok(Scalar::Unit) | Err(_) => {
            Err(ExternError::call("to_int", "not a number, string, or bool").into())
        }
    }
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
