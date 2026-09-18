//! A context file is data: each key a context, its type the value's type
//! (RFC-0031).

use std::path::Path;

use acvus_extern::Owned;
use acvus_interpreter::{AcvusRuntime, ContextWrite, Value};
use acvus_mir::ty::{IntTy, LenTerm, Ty};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

#[derive(Default)]
pub struct Loaded {
    pub types: FxHashMap<Astr, Ty>,
    pub snapshot: std::collections::HashMap<String, Owned<AcvusRuntime>>,
    /// Each context as the file wrote it, to tell a write from a commit
    /// of the loaded value.
    pub raw: std::collections::HashMap<String, serde_json::Value>,
}

struct Typed {
    ty: Ty,
    value: Value,
}

fn typed(interner: &Interner, at: &str, v: &serde_json::Value) -> Result<Typed, String> {
    Ok(match v {
        serde_json::Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Typed {
                ty: Ty::I64,
                value: Value::int(i),
            },
            (None, Some(u)) => Typed {
                ty: Ty::U64,
                value: Value::from_bits(IntTy::U64, u),
            },
            (None, None) => Typed {
                ty: Ty::Float,
                value: Value::float(
                    n.as_f64()
                        .ok_or_else(|| format!("{at}: {n} is neither an Int nor a Float"))?,
                ),
            },
        },
        serde_json::Value::String(s) => Typed {
            ty: Ty::String,
            value: Value::string(s.as_str()),
        },
        serde_json::Value::Bool(b) => Typed {
            ty: Ty::Bool,
            value: Value::bool_(*b),
        },
        serde_json::Value::Null => return Err(format!("{at}: null has no type")),
        serde_json::Value::Array(items) => {
            let mut elem: Option<Ty> = None;
            let mut values = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let t = typed(interner, &format!("{at}[{i}]"), item)?;
                match &elem {
                    None => elem = Some(t.ty),
                    Some(first) if *first != t.ty => {
                        return Err(format!(
                            "{at}[{i}]: {} where the array holds {}",
                            t.ty.display(interner),
                            first.display(interner)
                        ));
                    }
                    Some(_) => {}
                }
                values.push(Owned::from_value(t.value));
            }
            let elem = elem.ok_or_else(|| format!("{at}: an empty array has no element type"))?;
            Typed {
                ty: Ty::Array(Box::new(elem), LenTerm::Known(values.len())),
                value: Value::array(values),
            }
        }
        serde_json::Value::Object(fields) => {
            let mut tys = FxHashMap::default();
            let mut values = FxHashMap::default();
            for (k, v) in fields {
                let t = typed(interner, &format!("{at}.{k}"), v)?;
                let key = interner.intern(k);
                tys.insert(key, t.ty);
                values.insert(key, Owned::from_value(t.value));
            }
            Typed {
                ty: Ty::Object(tys),
                value: Value::object(values),
            }
        }
    })
}

pub fn load(interner: &Interner, path: &Path) -> Result<Loaded, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let root: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let serde_json::Value::Object(fields) = root else {
        return Err("the context file is a JSON object".to_string());
    };
    let mut loaded = Loaded::default();
    for (k, v) in &fields {
        let t = typed(interner, &format!("@{k}"), v)?;
        loaded.types.insert(interner.intern(k), t.ty);
        loaded
            .snapshot
            .insert(k.clone(), Owned::from_value(t.value));
        loaded.raw.insert(k.clone(), v.clone());
    }
    Ok(loaded)
}

/// Rewrite the context file with the run's writes applied.
pub fn commit(
    interner: &Interner,
    path: &Path,
    types: &FxHashMap<Astr, Ty>,
    writes: &[ContextWrite],
) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut root: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let serde_json::Value::Object(fields) = &mut root else {
        return Err("the context file is a JSON object".to_string());
    };
    for w in writes {
        let ty = types
            .get(&interner.intern(&w.key))
            .ok_or_else(|| format!("@{}: written but not in the context file", w.key))?;
        fields.insert(w.key.clone(), crate::json::of(interner, ty, &w.value));
    }
    let out = serde_json::to_string_pretty(&root).map_err(|e| e.to_string())?;
    std::fs::write(path, out + "\n").map_err(|e| e.to_string())
}
