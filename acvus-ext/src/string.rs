//! String operations as ExternFn. All pure.

use acvus_interpreter::{
    Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value, ValueKind,
};
use acvus_mir::ty::{Effect, ParamTerm, Poly, PolyTy, Ty, TyTerm, lift_effect_to_poly, lift_to_poly};
use acvus_utils::Interner;

// ── Handlers ────────────────────────────────────────────────────────

fn h_len_str(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(i64, Defs<()>), RuntimeError> {
    Ok((s.len() as i64, Defs(())))
}

fn h_trim(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((s.trim().to_owned(), Defs(())))
}

fn h_trim_start(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((s.trim_start().to_owned(), Defs(())))
}

fn h_trim_end(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((s.trim_end().to_owned(), Defs(())))
}

fn h_upper(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((s.to_uppercase(), Defs(())))
}

fn h_lower(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((s.to_lowercase(), Defs(())))
}

fn h_contains_str(
    _: &Interner,
    (s, pat): (String, String),
    Uses(()): Uses<()>,
) -> Result<(bool, Defs<()>), RuntimeError> {
    Ok((s.contains(&*pat), Defs(())))
}

fn h_starts_with(
    _: &Interner,
    (s, pat): (String, String),
    Uses(()): Uses<()>,
) -> Result<(bool, Defs<()>), RuntimeError> {
    Ok((s.starts_with(&*pat), Defs(())))
}

fn h_ends_with(
    _: &Interner,
    (s, pat): (String, String),
    Uses(()): Uses<()>,
) -> Result<(bool, Defs<()>), RuntimeError> {
    Ok((s.ends_with(&*pat), Defs(())))
}

fn h_replace(
    _: &Interner,
    (s, from, to): (String, String, String),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((s.replace(&*from, &to), Defs(())))
}

fn h_split(
    _: &Interner,
    (s, sep): (String, String),
    Uses(()): Uses<()>,
) -> Result<(Vec<Value>, Defs<()>), RuntimeError> {
    let parts: Vec<Value> = s.split(&*sep).map(Value::string).collect();
    Ok((parts, Defs(())))
}

fn h_repeat(
    _: &Interner,
    (s, n): (String, i64),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    Ok((s.repeat(n.max(0) as usize), Defs(())))
}

fn h_substring(
    _: &Interner,
    (s, start, end): (String, i64, i64),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    let start = start.max(0) as usize;
    let end = (end.max(0) as usize).min(s.len());
    let start = start.min(end);
    Ok((s[start..end].to_owned(), Defs(())))
}

fn h_to_bytes(
    _: &Interner,
    (s,): (String,),
    Uses(()): Uses<()>,
) -> Result<(Vec<Value>, Defs<()>), RuntimeError> {
    let bytes: Vec<Value> = s.bytes().map(Value::byte).collect();
    Ok((bytes, Defs(())))
}

fn h_to_utf8(
    _: &Interner,
    (bytes,): (Vec<Value>,),
    Uses(()): Uses<()>,
) -> Result<(Value, Defs<()>), RuntimeError> {
    let raw: Vec<u8> = bytes.iter().map(|v| v.as_byte()).collect();
    let s = String::from_utf8(raw).map_err(|_| {
        RuntimeError::unexpected_type("to_utf8", &[ValueKind::List], ValueKind::List)
    })?;
    Ok((Value::string(s), Defs(())))
}

fn h_to_utf8_lossy(
    _: &Interner,
    (bytes,): (Vec<Value>,),
    Uses(()): Uses<()>,
) -> Result<(String, Defs<()>), RuntimeError> {
    let raw: Vec<u8> = bytes.iter().map(|v| v.as_byte()).collect();
    Ok((String::from_utf8_lossy(&raw).into_owned(), Defs(())))
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

// ── Registry ────────────────────────────────────────────────────────

pub fn string_registry() -> ExternRegistry {
    ExternRegistry::new(|interner| {
        vec![
            ExternFnBuilder::new("len_str", sig(interner, vec![Ty::String], Ty::Int))
                .handler(h_len_str),
            ExternFnBuilder::new("trim", sig(interner, vec![Ty::String], Ty::String))
                .handler(h_trim),
            ExternFnBuilder::new("trim_start", sig(interner, vec![Ty::String], Ty::String))
                .handler(h_trim_start),
            ExternFnBuilder::new("trim_end", sig(interner, vec![Ty::String], Ty::String))
                .handler(h_trim_end),
            ExternFnBuilder::new("upper", sig(interner, vec![Ty::String], Ty::String))
                .handler(h_upper),
            ExternFnBuilder::new("lower", sig(interner, vec![Ty::String], Ty::String))
                .handler(h_lower),
            ExternFnBuilder::new(
                "contains_str",
                sig(interner, vec![Ty::String, Ty::String], Ty::Bool),
            )
            .handler(h_contains_str),
            ExternFnBuilder::new(
                "starts_with_str",
                sig(interner, vec![Ty::String, Ty::String], Ty::Bool),
            )
            .handler(h_starts_with),
            ExternFnBuilder::new(
                "ends_with_str",
                sig(interner, vec![Ty::String, Ty::String], Ty::Bool),
            )
            .handler(h_ends_with),
            ExternFnBuilder::new(
                "replace_str",
                sig(
                    interner,
                    vec![Ty::String, Ty::String, Ty::String],
                    Ty::String,
                ),
            )
            .handler(h_replace),
            ExternFnBuilder::new(
                "split_str",
                sig(
                    interner,
                    vec![Ty::String, Ty::String],
                    Ty::List(Box::new(Ty::String)),
                ),
            )
            .handler(h_split),
            ExternFnBuilder::new(
                "repeat_str",
                sig(interner, vec![Ty::String, Ty::Int], Ty::String),
            )
            .handler(h_repeat),
            ExternFnBuilder::new(
                "substring",
                sig(interner, vec![Ty::String, Ty::Int, Ty::Int], Ty::String),
            )
            .handler(h_substring),
            ExternFnBuilder::new("to_bytes", sig(interner, vec![Ty::String], Ty::bytes()))
                .handler(h_to_bytes),
            ExternFnBuilder::new(
                "to_utf8",
                sig(
                    interner,
                    vec![Ty::bytes()],
                    Ty::Option(Box::new(Ty::String)),
                ),
            )
            .handler(h_to_utf8),
            ExternFnBuilder::new(
                "to_utf8_lossy",
                sig(interner, vec![Ty::bytes()], Ty::String),
            )
            .handler(h_to_utf8_lossy),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_produces_functions() {
        let i = acvus_utils::Interner::new();
        let reg = string_registry().register(&i);
        assert_eq!(reg.functions.len(), 16);
        assert_eq!(reg.executables.len(), 16);
    }
}
