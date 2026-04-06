//! Type conversion operations as ExternFn. All pure.

use std::sync::Arc;

use acvus_interpreter::{
    Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value, ValueKind,
};
use acvus_mir::ty::{Effect, ParamTerm, Poly, PolyBuilder, PolyTy, Ty, TyTerm, lift_effect_to_poly, lift_to_poly};
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

fn sig(interner: &Interner, params: Vec<Ty>, ret: Ty) -> PolyTy {
    let named: Vec<ParamTerm<Poly>> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), lift_to_poly(ty)))
        .collect();
    TyTerm::Fn {
        params: named,
        ret: Box::new(lift_to_poly(&ret)),
        captures: vec![],
        effect: lift_effect_to_poly(&Effect::pure()),
        hint: None,
    }
}

fn sig_poly(interner: &Interner, params: Vec<PolyTy>, ret: PolyTy) -> PolyTy {
    let named: Vec<ParamTerm<Poly>> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    TyTerm::Fn {
        params: named,
        ret: Box::new(ret),
        captures: vec![],
        effect: lift_effect_to_poly(&Effect::pure()),
        hint: None,
    }
}

fn scalar_sig(interner: &Interner, ret: Ty) -> PolyTy {
    let mut b = PolyBuilder::new();
    let t = b.fresh_ty_var();
    sig_poly(interner, vec![t], lift_to_poly(&ret))
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
