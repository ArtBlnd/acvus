//! Type conversion operations as ExternFn. All pure.

use std::sync::Arc;

use acvus_interpreter::{
    Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value, ValueKind,
};
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::ty::{Effect, Param, ParamConstraint, Ty, TySubst};
use acvus_utils::Interner;

// ── Handlers ────────────────────────────────────────────────────────

fn h_to_string(
    _: &Interner,
    (val,): (Value,),
    Uses(()): Uses<()>,
) -> Result<(Value, Defs<()>), RuntimeError> {
    let s = match &val {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => return Ok((Value::String(Arc::clone(s)), Defs(()))),
        Value::Byte(b) => format!("0x{b:02x}"),
        Value::Unit => "()".to_string(),
        other => format!("{other:?}"),
    };
    Ok((Value::string(s), Defs(())))
}

fn h_to_int(
    _: &Interner,
    (val,): (Value,),
    Uses(()): Uses<()>,
) -> Result<(i64, Defs<()>), RuntimeError> {
    let n = match &val {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        Value::String(s) => s.parse::<i64>().map_err(|e| {
            RuntimeError::extern_call("to_int", format!("cannot parse string: {e}"))
        })?,
        Value::Bool(b) => {
            if *b {
                1
            } else {
                0
            }
        }
        _ => {
            return Err(RuntimeError::unexpected_type(
                "to_int",
                &[
                    ValueKind::Int,
                    ValueKind::Float,
                    ValueKind::String,
                    ValueKind::Bool,
                ],
                val.kind(),
            ));
        }
    };
    Ok((n, Defs(())))
}

fn h_to_float(
    _: &Interner,
    (n,): (i64,),
    Uses(()): Uses<()>,
) -> Result<(f64, Defs<()>), RuntimeError> {
    Ok((n as f64, Defs(())))
}

fn h_char_to_int(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(i64, Defs<()>), RuntimeError> {
    Ok((s.chars().next().unwrap_or('\0') as i64, Defs(())))
}

fn h_int_to_char(
    _: &Interner,
    (n,): (i64,),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    let ch = char::from_u32(n as u32).unwrap_or('\u{FFFD}');
    Ok((ch.to_string(), Defs(())))
}

// ── Constraint builders ─────────────────────────────────────────────

fn sig(interner: &Interner, params: Vec<Ty>, ret: Ty) -> FnConstraint {
    let named: Vec<Param> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    FnConstraint {
        signature: Some(Signature {
            params: named.clone(),
        }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
        hint: None,
    }
}

fn scalar_sig(interner: &Interner, ret: Ty) -> FnConstraint {
    let mut s = TySubst::new();
    let t = s.fresh_param_constrained(ParamConstraint::scalar());
    sig(interner, vec![t], ret)
}

// ── Registry ────────────────────────────────────────────────────────

pub fn conversion_registry() -> ExternRegistry {
    ExternRegistry::new(|interner| {
        vec![
            ExternFnBuilder::new("to_string", scalar_sig(interner, Ty::String))
                .handler(h_to_string),
            ExternFnBuilder::new("to_int", scalar_sig(interner, Ty::Int)).handler(h_to_int),
            ExternFnBuilder::new("to_float", sig(interner, vec![Ty::Int], Ty::Float))
                .handler(h_to_float),
            ExternFnBuilder::new("char_to_int", sig(interner, vec![Ty::String], Ty::Int))
                .handler(h_char_to_int),
            ExternFnBuilder::new("int_to_char", sig(interner, vec![Ty::Int], Ty::String))
                .handler(h_int_to_char),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_produces_functions() {
        let i = acvus_utils::Interner::new();
        let reg = conversion_registry().register(&i);
        assert_eq!(reg.functions.len(), 5);
        assert_eq!(reg.executables.len(), 5);
    }
}
