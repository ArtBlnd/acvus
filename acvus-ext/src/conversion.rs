//! Type conversions. All pure.

use acvus_extern::{
    ExternRegistry, Interner, RuntimeError, TyVar, Value, ValueKind, extern_fn, extern_registry,
};

#[extern_fn(effect = pure)]
fn to_string<T>(i: &Interner, val: T) -> String
where
    T: TyVar,
{
    match val.into_value(i) {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => s.to_string(),
        Value::Byte(b) => format!("0x{b:02x}"),
        Value::Unit => "()".to_string(),
        other => format!("{other:?}"),
    }
}

#[extern_fn(effect = pure)]
fn to_int<T>(i: &Interner, val: T) -> Result<i64, RuntimeError>
where
    T: TyVar,
{
    let val = val.into_value(i);
    match &val {
        Value::Int(n) => Ok(*n),
        Value::Float(f) => Ok(*f as i64),
        Value::String(s) => s
            .parse::<i64>()
            .map_err(|e| RuntimeError::extern_call("to_int", format!("cannot parse string: {e}"))),
        Value::Bool(b) => Ok(i64::from(*b)),
        _ => Err(RuntimeError::unexpected_type(
            "to_int",
            &[
                ValueKind::Int,
                ValueKind::Float,
                ValueKind::String,
                ValueKind::Bool,
            ],
            val.kind(),
        )),
    }
}

#[extern_fn(effect = pure)]
fn to_float(_: &Interner, n: i64) -> f64 {
    n as f64
}

#[extern_fn(effect = pure)]
fn char_to_int(_: &Interner, s: String) -> Result<i64, RuntimeError> {
    match s.chars().next() {
        Some(c) => Ok(c as i64),
        None => Err(RuntimeError::extern_call("char_to_int", "empty string")),
    }
}

#[extern_fn(effect = pure)]
fn int_to_char(_: &Interner, n: i64) -> Result<String, RuntimeError> {
    let code = u32::try_from(n).map_err(|_| {
        RuntimeError::extern_call("int_to_char", format!("{n} is not a code point"))
    })?;
    match char::from_u32(code) {
        Some(c) => Ok(c.to_string()),
        None => Err(RuntimeError::extern_call(
            "int_to_char",
            format!("{n} is not a code point"),
        )),
    }
}

pub fn conversion_registry() -> ExternRegistry {
    extern_registry! {
        fns: [to_string, to_int, to_float, char_to_int, int_to_char],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::TypeRegistry;

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = conversion_registry().register(&i, &mut TypeRegistry::new());
        assert_eq!(reg.functions.len(), 5);
        assert_eq!(reg.executables.len(), 5);
    }
}
